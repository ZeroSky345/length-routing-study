from __future__ import annotations

from typing import Any, Sequence

from .cost_model import estimate_backend, get_profile, infer_workload_features
from .sparse_plan import (
    BACKEND_DENSE,
    BACKEND_FLEX_PREFILL_FLEX,
    BACKEND_FLEX_PREFILL_TRITON,
    BACKEND_PBS_ATTENTION,
    BACKEND_STANDALONE_FLEX,
    BackendEstimate,
    SparseBackendConfig,
    SparsePlan,
)


DEFAULT_CANDIDATES = (
    BACKEND_DENSE,
    BACKEND_PBS_ATTENTION,
    BACKEND_FLEX_PREFILL_TRITON,
    BACKEND_FLEX_PREFILL_FLEX,
    BACKEND_STANDALONE_FLEX,
)


class TheoryDrivenDispatcher:
    def __init__(
        self,
        backend_config: SparseBackendConfig | None = None,
        model_memory_budget_gb: float = 72.0,
        safety_margin: float = 0.9,
        version: str = "theory-v0",
    ) -> None:
        self.backend_config = backend_config or SparseBackendConfig()
        self.model_memory_budget_gb = model_memory_budget_gb
        self.safety_margin = safety_margin
        self.version = version

    def detect_model_family(self, model_or_name: Any) -> str:
        if hasattr(model_or_name, "config") and hasattr(model_or_name.config, "model_type"):
            model_type = str(model_or_name.config.model_type).lower()
        else:
            model_type = str(model_or_name).lower()
        if "qwen" in model_type:
            return "qwen2"
        if "llama" in model_type or "mistral" in model_type:
            return "llama"
        if "glm" in model_type or "chatglm" in model_type:
            return "glm"
        return "generic"

    def _normalize_model_name(self, model_or_name: Any) -> str:
        if hasattr(model_or_name, "name_or_path"):
            return str(model_or_name.name_or_path)
        if hasattr(model_or_name, "config") and hasattr(model_or_name.config, "_name_or_path"):
            return str(model_or_name.config._name_or_path)
        return str(model_or_name)

    def _fallback_chain(self, winner: BackendEstimate, estimates: list[BackendEstimate]) -> tuple[str, ...]:
        ordered = [estimate.backend for estimate in sorted(estimates, key=lambda item: item.score) if estimate.backend != winner.backend and estimate.feasible]
        return tuple(ordered[:3])

    def build_plan(
        self,
        model_or_name: Any,
        prompt_tokens: int,
        objective: str = "balanced",
        allow_experimental: bool = False,
        preferred_backend: str | None = None,
        sparsity_ratio: float = 0.0,
        kv_norm_cv: float = 0.0,
        sparsity_source: str = "geometric",
    ) -> SparsePlan:
        """
        Build a routing plan for the given sequence length.

        Parameters
        ----------
        sparsity_ratio : float in [0, 1]
            Estimated fraction of attention blocks that can be pruned.
            Pass 0 (default) to use the geometric approximation.
            Obtain from ``sparsity_estimator.estimate_sparsity(q, k)``.
        kv_norm_cv : float
            K-norm coefficient of variation (from ``sparsity_estimator``).
        sparsity_source : str
            "geometric" | "estimated" | "measured"
        """
        model_family = self.detect_model_family(model_or_name)
        model_name = self._normalize_model_name(model_or_name)
        profile = get_profile(model_family)
        workload = infer_workload_features(
            prompt_tokens=prompt_tokens,
            cfg=self.backend_config,
            sparsity_ratio=sparsity_ratio,
            kv_norm_cv=kv_norm_cv,
            sparsity_source=sparsity_source,
        )

        candidates = list(DEFAULT_CANDIDATES)
        if not allow_experimental:
            candidates = [backend for backend in candidates if backend not in {BACKEND_FLEX_PREFILL_FLEX, BACKEND_STANDALONE_FLEX}]
        if preferred_backend is not None:
            candidates = [preferred_backend]

        estimates = [
            estimate_backend(
                profile=profile,
                features=workload,
                backend=backend,
                objective=objective,
                model_memory_budget_gb=self.model_memory_budget_gb,
                safety_margin=self.safety_margin,
            )
            for backend in candidates
        ]
        winner = min(estimates, key=lambda item: item.score)
        fallbacks = self._fallback_chain(winner, estimates)

        notes = [
            f"Theoretical dispatcher compared {len(estimates)} backend candidates using estimated latency, memory, stability, and migration score.",
            f"Selected {winner.backend_label} because it minimized the composite {objective} score ({winner.score:.2f}).",
            "The plan is derived from mechanism-level assumptions: PBS pays more in permutation and block selection, while FlexPrefill pays more in mask/budget setup but grows with the sparse budget rather than the full block graph.",
        ]
        if sparsity_source != "geometric":
            notes.append(
                f"Sparsity-aware routing: estimated active_fraction={workload.active_fraction:.2f} "
                f"(sparsity={workload.sparsity_ratio:.2f}, cv={workload.kv_norm_cv:.2f}, "
                f"source={sparsity_source}). "
                "active_blocks were scaled from geometric estimate by the measured sparsity ratio."
            )
        if winner.reject_reason:
            notes.append(f"Selected backend is above the memory budget in theory: {winner.reject_reason}.")
        notes.extend(winner.notes)

        return SparsePlan(
            backend=winner.backend,
            model_family=model_family,
            model_name=model_name,
            prompt_tokens=prompt_tokens,
            objective=objective,
            workload=workload,
            selected_estimate=winner,
            candidate_estimates=tuple(sorted(estimates, key=lambda item: item.score)),
            backend_config=self.backend_config,
            fallback_backends=fallbacks,
            constraints={
                "model_memory_budget_gb": self.model_memory_budget_gb,
                "safety_margin": self.safety_margin,
                "allow_experimental": allow_experimental,
            },
            notes=tuple(notes),
            metadata={
                "profile_family": profile.family,
                "candidate_backends": tuple(candidates),
            },
            dispatcher_version=self.version,
        )

    def build_plan_matrix(
        self,
        model_or_name: Any,
        prompt_tokens: Sequence[int],
        objective: str = "balanced",
        allow_experimental: bool = False,
        preferred_backend: str | None = None,
        sparsity_ratios: Sequence[float] | None = None,
    ) -> list[SparsePlan]:
        """
        Build plans for multiple lengths.

        ``sparsity_ratios`` (optional) must be the same length as ``prompt_tokens``.
        """
        sparsity_list = sparsity_ratios or [0.0] * len(prompt_tokens)
        return [
            self.build_plan(
                model_or_name=model_or_name,
                prompt_tokens=token_count,
                objective=objective,
                allow_experimental=allow_experimental,
                preferred_backend=preferred_backend,
                sparsity_ratio=sr,
                sparsity_source="estimated" if sr > 0.0 else "geometric",
            )
            for token_count, sr in zip(prompt_tokens, sparsity_list)
        ]
