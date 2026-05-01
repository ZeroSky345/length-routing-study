#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from length_routing_study.dispatcher import TheoryDrivenDispatcher
from length_routing_study.length_router import LengthAwareRouter
from length_routing_study.patch import apply_sparse_plan
from length_routing_study.sparsity_estimator import estimate_sparsity
from length_routing_study.sparse_plan import (
    BACKEND_DENSE,
    BACKEND_FLEX_PREFILL_TRITON,
    BACKEND_PBS_ATTENTION,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run lightweight MCQ backend validation on real prompts.")
    parser.add_argument(
        "--model",
        default="/root/autodl-tmp/qwen/Qwen2.5-3B-Instruct",
        help="Local HuggingFace model path.",
    )
    parser.add_argument("--manifest", type=Path, default=Path("data/cache_real/real_text_manifest.json"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/cache_real"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--objective", default="balanced", choices=["balanced", "speed", "stability", "memory"])
    parser.add_argument(
        "--attn-implementation",
        default="flash_attention_2",
        choices=["auto", "eager", "sdpa", "flash_attention_2"],
    )
    parser.add_argument("--output", type=Path, default=Path("results/real_text_mcq_backend_eval.json"))
    return parser.parse_args()


def _entry_tag(entry: dict[str, Any]) -> str:
    return str(entry.get("sample_tag") or entry.get("source_id") or "sample")


def _entry_key(entry: dict[str, Any]) -> str:
    cache_key = entry.get("cache_key")
    if cache_key:
        return str(cache_key)
    cache_file = entry.get("cache_file")
    if cache_file:
        return Path(str(cache_file)).stem
    return f"{entry['category']}_{entry['target_len']}_{_entry_tag(entry)}"


def _backend_key(sparse_backend: str) -> str:
    if sparse_backend == BACKEND_DENSE:
        return "dense"
    if sparse_backend == BACKEND_PBS_ATTENTION:
        return "pbs"
    if sparse_backend == BACKEND_FLEX_PREFILL_TRITON:
        return "flex"
    raise KeyError(f"unsupported sparse backend {sparse_backend!r}")


def _load_model(model_path: str, device: str, attn_implementation: str) -> AutoModelForCausalLM:
    use_cuda = device.startswith("cuda") and torch.cuda.is_available()
    kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": {"": 0} if use_cuda else None,
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }
    if attn_implementation != "auto":
        kwargs["attn_implementation"] = attn_implementation
    try:
        model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    except Exception:
        kwargs.pop("attn_implementation", None)
        model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    if not use_cuda:
        model.to(device)
    model.eval()
    return model


def _load_entry_qkv(
    cache_dir: Path,
    entry: dict[str, Any],
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    payload = torch.load(cache_dir / entry["cache_file"], map_location="cpu")
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    return (
        payload["q"].to(device=dev, dtype=torch.bfloat16),
        payload["k"].to(device=dev, dtype=torch.bfloat16),
        payload["v"].to(device=dev, dtype=torch.bfloat16),
    )


def _choice_scores(
    model: AutoModelForCausalLM,
    input_ids: list[int],
    token_map: dict[str, int],
    tokenizer: Any,
) -> dict[str, Any]:
    device = next(model.parameters()).device
    ids = torch.tensor([input_ids], device=device, dtype=torch.long)
    with torch.inference_mode():
        generated = model.generate(
            input_ids=ids,
            max_new_tokens=1,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
        )
    step_scores = generated.scores[0][0]
    letters = ("A", "B", "C", "D")
    raw = torch.tensor([float(step_scores[token_map[letter]].item()) for letter in letters], dtype=torch.float32)
    probs = torch.softmax(raw, dim=0)
    scores = {letter: round(float(raw[idx].item()), 6) for idx, letter in enumerate(letters)}
    probabilities = {letter: round(float(probs[idx].item()), 6) for idx, letter in enumerate(letters)}
    pred = max(scores, key=scores.get)
    generated_token = int(generated.sequences[0, -1].item())
    generated_text = tokenizer.decode([generated_token], skip_special_tokens=True)
    return {
        "prediction": pred,
        "generated_text": generated_text,
        "scores": scores,
        "probabilities": probabilities,
    }


def _mask_hits_evidence(mask_2d: torch.Tensor, evidence_block_indices: list[int]) -> dict[str, Any] | None:
    if mask_2d.dim() != 2:
        return None
    q_blocks, kv_blocks = mask_2d.shape
    valid = [idx for idx in evidence_block_indices if 0 <= idx < kv_blocks]
    if not valid:
        return None
    last_row = mask_2d[q_blocks - 1]
    hits = [bool(last_row[idx].item()) for idx in valid]
    return {
        "any": bool(any(hits)),
        "all": bool(all(hits)),
        "hit_fraction": round(sum(hits) / len(hits), 4),
        "hits": hits,
        "num_targets": len(valid),
    }


def _run_backend(
    backend_name: str,
    *,
    model_path: str,
    device: str,
    prompt_tokens: int,
    entries: list[dict[str, Any]],
    token_map: dict[str, int],
    tokenizer: Any,
    objective: str,
    attn_implementation: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    model = _load_model(model_path, device, attn_implementation)
    patch_info: dict[str, Any] = {"backend": backend_name, "patched": backend_name != "dense"}
    if backend_name != "dense":
        dispatcher = TheoryDrivenDispatcher(model_memory_budget_gb=72.0)
        preferred = BACKEND_PBS_ATTENTION if backend_name == "pbs" else BACKEND_FLEX_PREFILL_TRITON
        plan = dispatcher.build_plan(
            model_or_name=model,
            prompt_tokens=prompt_tokens,
            objective=objective,
            allow_experimental=False,
            preferred_backend=preferred,
        )
        applied = apply_sparse_plan(model, plan)
        patch_info["plan"] = applied.as_dict()

    results: dict[str, dict[str, Any]] = {}
    for entry in entries:
        key = _entry_key(entry)
        record = _choice_scores(model, entry["input_ids"], token_map, tokenizer)
        record["correct"] = bool(record["prediction"] == entry["answer_letter"])
        results[key] = record

    try:
        del model
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return results, patch_info


def _prob_vector(record: dict[str, Any]) -> torch.Tensor:
    return torch.tensor(
        [record["probabilities"][letter] for letter in ("A", "B", "C", "D")],
        dtype=torch.float32,
    )


def _prob_delta(record: dict[str, Any], dense: dict[str, Any]) -> dict[str, float]:
    record_probs = _prob_vector(record)
    dense_probs = _prob_vector(dense)
    l1 = torch.sum(torch.abs(record_probs - dense_probs)).item()
    kl = torch.sum(dense_probs * torch.log((dense_probs + 1e-8) / (record_probs + 1e-8))).item()
    return {"l1": round(float(l1), 6), "kl_from_dense": round(float(kl), 6)}


def _aggregate(cells: list[dict[str, Any]], field: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for cell in cells:
        grouped[str(cell[field])].append(cell)

    def _sort_key(value: str) -> tuple[int, str]:
        return (0, f"{int(value):09d}") if value.isdigit() else (1, value)

    out: list[dict[str, Any]] = []
    for key in sorted(grouped, key=_sort_key):
        rows = grouped[key]
        summary = {
            field: int(key) if key.isdigit() else key,
            "n": len(rows),
            "dense_acc": round(sum(r["dense"]["correct"] for r in rows) / len(rows), 4),
            "pbs_acc": round(sum(r["pbs"]["correct"] for r in rows) / len(rows), 4),
            "flex_acc": round(sum(r["flex"]["correct"] for r in rows) / len(rows), 4),
            "backend_route_acc": round(sum(r["backend_route_result"]["correct"] for r in rows) / len(rows), 4),
            "pbs_change_rate": round(sum(r["answer_changed_vs_dense"]["pbs"] for r in rows) / len(rows), 4),
            "flex_change_rate": round(sum(r["answer_changed_vs_dense"]["flex"] for r in rows) / len(rows), 4),
            "backend_route_change_rate": round(
                sum(r["answer_changed_vs_dense"]["backend_route"] for r in rows) / len(rows),
                4,
            ),
            "pbs_prob_l1_avg": round(sum(r["prob_shift_vs_dense"]["pbs"]["l1"] for r in rows) / len(rows), 6),
            "flex_prob_l1_avg": round(sum(r["prob_shift_vs_dense"]["flex"]["l1"] for r in rows) / len(rows), 6),
            "backend_route_prob_l1_avg": round(
                sum(r["prob_shift_vs_dense"]["backend_route"]["l1"] for r in rows) / len(rows),
                6,
            ),
        }
        code_rows = [r for r in rows if r["mask_route_evidence"] is not None]
        if code_rows:
            summary["mask_route_code_hit_any_rate"] = round(
                sum(r["mask_route_evidence"]["any"] for r in code_rows) / len(code_rows),
                4,
            )
            summary["mask_route_code_hit_all_rate"] = round(
                sum(r["mask_route_evidence"]["all"] for r in code_rows) / len(code_rows),
                4,
            )
        out.append(summary)
    return out


def main() -> None:
    args = _parse_args()
    with open(args.manifest, "r", encoding="utf-8") as fp:
        manifest = json.load(fp)
    entries = manifest["entries"]

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    token_map = {}
    for letter in ("A", "B", "C", "D"):
        token_ids = tokenizer.encode(letter, add_special_tokens=False)
        if len(token_ids) != 1:
            raise ValueError(f"{letter!r} is not a single token")
        token_map[letter] = token_ids[0]
    max_prompt_tokens = max(int(entry["actual_len"]) for entry in entries)

    backend_outputs: dict[str, dict[str, dict[str, Any]]] = {}
    patch_infos: dict[str, dict[str, Any]] = {}
    for backend in ("dense", "pbs", "flex"):
        print(f"[backend] {backend}")
        outputs, patch_info = _run_backend(
            backend,
            model_path=args.model,
            device=args.device,
            prompt_tokens=max_prompt_tokens,
            entries=entries,
            token_map=token_map,
            tokenizer=tokenizer,
            objective=args.objective,
            attn_implementation=args.attn_implementation,
        )
        backend_outputs[backend] = outputs
        patch_infos[backend] = patch_info

    dispatcher = TheoryDrivenDispatcher(model_memory_budget_gb=72.0)
    router = LengthAwareRouter()
    cells: list[dict[str, Any]] = []
    for entry in entries:
        key = _entry_key(entry)
        q, k, v = _load_entry_qkv(args.cache_dir, entry, args.device)
        sp = estimate_sparsity(q, k, block_size=128, sample_rows=64, seed=42)
        backend_plan = dispatcher.build_plan(
            model_or_name=args.model,
            prompt_tokens=int(entry["actual_len"]),
            objective=args.objective,
            allow_experimental=False,
            sparsity_ratio=sp.estimated_sparsity_ratio,
            kv_norm_cv=sp.kv_norm_cv,
            sparsity_source="estimated",
        )
        backend_route = _backend_key(backend_plan.backend)
        mask_route = router.route_full(
            int(entry["actual_len"]),
            sparsity=sp.estimated_sparsity_ratio,
            k_norm_cv=sp.kv_norm_cv,
        )

        mask_route_evidence = None
        if mask_route.strategy_instance is not None and entry["category"] == "code_repo_qa":
            work = mask_route.strategy_instance.select(q, k, v, block_size=128)
            mask_route_evidence = _mask_hits_evidence(
                work.block_mask,
                list(entry.get("evidence_block_indices", [entry["evidence_block_index"]])),
            )

        dense = backend_outputs["dense"][key]
        pbs = backend_outputs["pbs"][key]
        flex = backend_outputs["flex"][key]
        backend_route_result = backend_outputs[backend_route][key]
        cells.append(
            {
                "category": entry["category"],
                "sample_tag": _entry_tag(entry),
                "cache_key": key,
                "target_len": int(entry["target_len"]),
                "seq_len": int(entry["actual_len"]),
                "source_id": entry["source_id"],
                "answer_letter": entry["answer_letter"],
                "dense": dense,
                "pbs": pbs,
                "flex": flex,
                "backend_route": {
                    "backend": backend_route,
                    "sparse_backend": backend_plan.backend,
                    "sparsity_ratio": round(sp.estimated_sparsity_ratio, 4),
                    "kv_norm_cv": round(sp.kv_norm_cv, 4),
                },
                "mask_route": {
                    "backend": mask_route.backend,
                    "reason": mask_route.reason,
                    "params": mask_route.params,
                },
                "backend_route_result": backend_route_result,
                "answer_changed_vs_dense": {
                    "pbs": bool(pbs["prediction"] != dense["prediction"]),
                    "flex": bool(flex["prediction"] != dense["prediction"]),
                    "backend_route": bool(backend_route_result["prediction"] != dense["prediction"]),
                },
                "prob_shift_vs_dense": {
                    "pbs": _prob_delta(pbs, dense),
                    "flex": _prob_delta(flex, dense),
                    "backend_route": _prob_delta(backend_route_result, dense),
                },
                "mask_route_evidence": mask_route_evidence,
            }
        )
        del q, k, v
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = {
        "n": len(cells),
        "dense_acc": round(sum(1 for cell in cells if cell["dense"]["correct"]) / len(cells), 4),
        "pbs_acc": round(sum(1 for cell in cells if cell["pbs"]["correct"]) / len(cells), 4),
        "flex_acc": round(sum(1 for cell in cells if cell["flex"]["correct"]) / len(cells), 4),
        "backend_route_acc": round(sum(1 for cell in cells if cell["backend_route_result"]["correct"]) / len(cells), 4),
        "pbs_change_rate": round(sum(1 for cell in cells if cell["answer_changed_vs_dense"]["pbs"]) / len(cells), 4),
        "flex_change_rate": round(sum(1 for cell in cells if cell["answer_changed_vs_dense"]["flex"]) / len(cells), 4),
        "backend_route_change_rate": round(
            sum(1 for cell in cells if cell["answer_changed_vs_dense"]["backend_route"]) / len(cells),
            4,
        ),
        "pbs_prob_l1_avg": round(sum(cell["prob_shift_vs_dense"]["pbs"]["l1"] for cell in cells) / len(cells), 6),
        "flex_prob_l1_avg": round(sum(cell["prob_shift_vs_dense"]["flex"]["l1"] for cell in cells) / len(cells), 6),
        "backend_route_prob_l1_avg": round(
            sum(cell["prob_shift_vs_dense"]["backend_route"]["l1"] for cell in cells) / len(cells),
            6,
        ),
    }
    code_rows = [cell for cell in cells if cell["mask_route_evidence"] is not None]
    if code_rows:
        summary["mask_route_code_hit_any_rate"] = round(
            sum(1 for cell in code_rows if cell["mask_route_evidence"]["any"]) / len(code_rows),
            4,
        )
        summary["mask_route_code_hit_all_rate"] = round(
            sum(1 for cell in code_rows if cell["mask_route_evidence"]["all"]) / len(code_rows),
            4,
        )

    output = {
        "model": args.model,
        "objective": args.objective,
        "manifest": str(args.manifest),
        "token_map": token_map,
        "patches": patch_infos,
        "summary": summary,
        "aggregated_by_category": _aggregate(cells, "category"),
        "aggregated_by_target_length": _aggregate(cells, "target_len"),
        "cells": cells,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fp:
        json.dump(output, fp, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"saved -> {args.output}")


if __name__ == "__main__":
    main()
