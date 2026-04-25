from __future__ import annotations

import argparse
import inspect
import json
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional

from ._paths import ensure_external_paths
from .dispatcher import TheoryDrivenDispatcher
from .sparse_plan import (
    BACKEND_DENSE,
    BACKEND_FLEX_PREFILL_FLEX,
    BACKEND_FLEX_PREFILL_TRITON,
    BACKEND_PBS_ATTENTION,
    BACKEND_STANDALONE_FLEX,
    SparsePlan,
)

ensure_external_paths()
from baseline_test import apply_patch_with_prefill_legacy_qwen2, apply_standalone_flexattention_qwen2
from flex_prefill import patch_model
from flex_prefill.modules.patch import disable_hf_flash_attention_check
from pbs_attn.patch.huggingface import apply_patch_with_prefill, get_permuted_block_sparse_attn_fwd


@dataclass(frozen=True)
class AppliedTheoryPatch:
    plan: SparsePlan
    applied_backend: str
    mutated_model: bool
    notes: tuple[str, ...] = tuple()

    def as_dict(self) -> dict[str, Any]:
        return {
            "plan": self.plan.as_dict(),
            "applied_backend": self.applied_backend,
            "mutated_model": self.mutated_model,
            "notes": list(self.notes),
        }


def _annotate_model(model: Any, plan: SparsePlan, notes: Iterable[str]) -> None:
    setattr(model, "_theory_sparse_plan", plan)
    setattr(model, "_theory_sparse_backend", plan.backend)
    setattr(model, "_theory_sparse_notes", tuple(notes))


def _apply_dense_passthrough(model: Any, plan: SparsePlan) -> AppliedTheoryPatch:
    notes = ("No patch applied; the original dense path satisfies the current theoretical plan.",)
    _annotate_model(model, plan, notes)
    return AppliedTheoryPatch(plan=plan, applied_backend=plan.backend, mutated_model=False, notes=notes)


def _apply_flex_prefill_patch(model: Any, plan: SparsePlan) -> AppliedTheoryPatch:
    disable_hf_flash_attention_check()
    patch_cfg = plan.backend_config.flex_prefill_kwargs(plan.backend)
    patch_model(model, "flex_prefill", patch_cfg)
    notes = (f"Applied FlexPrefill patch using backend={patch_cfg['flex_prefill_backend']}.",)
    _annotate_model(model, plan, notes)
    return AppliedTheoryPatch(plan=plan, applied_backend=plan.backend, mutated_model=True, notes=notes)


def _apply_pbs_patch(model: Any, plan: SparsePlan) -> AppliedTheoryPatch:
    pbs_fn = get_permuted_block_sparse_attn_fwd(**plan.backend_config.pbs_kwargs())
    qwen_sig = inspect.signature(model.model.layers[0].self_attn.forward)
    if "position_embeddings" in qwen_sig.parameters:
        apply_patch_with_prefill(model, pbs_fn)
        notes = ("Applied PBS patch using the modern attention signature.",)
    else:
        if plan.model_family != "qwen2":
            raise NotImplementedError(
                f"Legacy PBS compatibility shim is only implemented for qwen2, got {plan.model_family!r}."
            )
        apply_patch_with_prefill_legacy_qwen2(model, pbs_fn)
        notes = ("Applied PBS patch through the qwen2 legacy compatibility shim.",)
    _annotate_model(model, plan, notes)
    return AppliedTheoryPatch(plan=plan, applied_backend=plan.backend, mutated_model=True, notes=notes)


def _apply_standalone_flex_patch(model: Any, plan: SparsePlan) -> AppliedTheoryPatch:
    if plan.model_family != "qwen2":
        raise NotImplementedError(
            f"Standalone FlexAttention helper is only wired for qwen2 in /root/test right now, got {plan.model_family!r}."
        )
    apply_standalone_flexattention_qwen2(model)
    notes = ("Applied the qwen2-only standalone FlexAttention patch from baseline_test.py.",)
    _annotate_model(model, plan, notes)
    return AppliedTheoryPatch(plan=plan, applied_backend=plan.backend, mutated_model=True, notes=notes)


def apply_sparse_plan(model: Any, plan: SparsePlan) -> AppliedTheoryPatch:
    if plan.backend == BACKEND_DENSE:
        return _apply_dense_passthrough(model, plan)
    if plan.backend in {BACKEND_FLEX_PREFILL_TRITON, BACKEND_FLEX_PREFILL_FLEX}:
        return _apply_flex_prefill_patch(model, plan)
    if plan.backend == BACKEND_PBS_ATTENTION:
        return _apply_pbs_patch(model, plan)
    if plan.backend == BACKEND_STANDALONE_FLEX:
        return _apply_standalone_flex_patch(model, plan)
    raise ValueError(f"Unsupported backend {plan.backend!r}")


def apply_theory_patch(
    model: Any,
    prompt_tokens: int,
    dispatcher: Optional[TheoryDrivenDispatcher] = None,
    objective: str = "balanced",
    allow_experimental: bool = False,
    preferred_backend: Optional[str] = None,
) -> AppliedTheoryPatch:
    dispatcher = dispatcher or TheoryDrivenDispatcher()
    plan = dispatcher.build_plan(
        model_or_name=model,
        prompt_tokens=prompt_tokens,
        objective=objective,
        allow_experimental=allow_experimental,
        preferred_backend=preferred_backend,
    )
    return apply_sparse_plan(model, plan)


def patch_new_model_with_fallbacks(
    load_model_fn: Callable[[], Any],
    model_name: str,
    prompt_tokens: int,
    dispatcher: Optional[TheoryDrivenDispatcher] = None,
    objective: str = "balanced",
    allow_experimental: bool = False,
) -> tuple[Any, AppliedTheoryPatch, list[dict[str, str]]]:
    dispatcher = dispatcher or TheoryDrivenDispatcher()
    first_plan = dispatcher.build_plan(
        model_or_name=model_name,
        prompt_tokens=prompt_tokens,
        objective=objective,
        allow_experimental=allow_experimental,
    )
    attempts = [first_plan.backend, *first_plan.fallback_backends]
    errors: list[dict[str, str]] = []
    for backend in attempts:
        plan = first_plan if backend == first_plan.backend else dispatcher.build_plan(
            model_or_name=model_name,
            prompt_tokens=prompt_tokens,
            objective=objective,
            allow_experimental=allow_experimental,
            preferred_backend=backend,
        )
        model = load_model_fn()
        try:
            return model, apply_sparse_plan(model, plan), errors
        except Exception as exc:
            errors.append({"backend": backend, "error_type": type(exc).__name__, "error": str(exc)})
            try:
                del model
            except Exception:
                pass
    raise RuntimeError(json.dumps(errors, ensure_ascii=False))


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Theory-driven sparse dispatcher prototype")
    parser.add_argument("--model", default="/root/autodl-tmp/qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--prompt_tokens", type=int, nargs="+", required=True)
    parser.add_argument("--objective", choices=["balanced", "speed", "stability", "memory"], default="balanced")
    parser.add_argument("--allow-experimental", action="store_true")
    parser.add_argument("--memory-budget-gb", type=float, default=72.0)
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    dispatcher = TheoryDrivenDispatcher(model_memory_budget_gb=args.memory_budget_gb)
    plans = dispatcher.build_plan_matrix(
        model_or_name=args.model,
        prompt_tokens=args.prompt_tokens,
        objective=args.objective,
        allow_experimental=args.allow_experimental,
    )
    print(json.dumps([plan.as_dict() for plan in plans], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
