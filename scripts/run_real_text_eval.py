#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from length_routing_study._paths import ensure_external_paths

ensure_external_paths(include_baseline=False)

from length_routing_study.empirical_sweep import (
    FlexConfig,
    PBSConfig,
    run_flash,
    run_flex,
    run_pbs_decomposed,
)
from length_routing_study.length_router import LengthAwareRouter
from length_routing_study.selection_strategies import run_block_sparse_attention_flex
from length_routing_study.sparsity_estimator import estimate_sparsity


PBS_FIXED = PBSConfig(threshold=0.9, segment_size=256, block_size=128, use_triton=True, force_first=True)
FLEX_FIXED = FlexConfig(gamma=0.95, tau=0.1, min_budget_frac=0.0, block_size=128)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run real-text-driven QKV evaluation.")
    parser.add_argument("--cache-dir", type=Path, default=Path("data/cache_real"))
    parser.add_argument("--manifest", type=Path, default=Path("data/cache_real/real_text_manifest.json"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--output", type=Path, default=Path("results/real_text_eval.json"))
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


def _load_entry_qkv(cache_dir: Path, entry: dict[str, Any], device: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    payload = torch.load(cache_dir / entry["cache_file"], map_location="cpu")
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    return (
        payload["q"].to(device=dev, dtype=torch.bfloat16),
        payload["k"].to(device=dev, dtype=torch.bfloat16),
        payload["v"].to(device=dev, dtype=torch.bfloat16),
    )


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


def _run_ours(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    sparsity: float,
    k_norm_cv: float,
    flash_ms: float,
    reference: torch.Tensor,
    warmup: int,
    repeats: int,
) -> tuple[dict[str, Any], Any]:
    router = LengthAwareRouter()
    decision = router.route_full(q.shape[2], sparsity=sparsity, k_norm_cv=k_norm_cv)

    if decision.backend == "pbs":
        cfg = PBSConfig(
            threshold=float(decision.params["threshold"]),
            segment_size=int(decision.params["segment_size"]),
            block_size=int(decision.params["block_size"]),
            use_triton=bool(decision.params["use_triton"]),
            force_first=bool(decision.params["force_first"]),
        )
        record = run_pbs_decomposed(
            q,
            k,
            v,
            cfg,
            flash_time_ms=flash_ms,
            reference=reference,
            warmup=warmup,
            repeats=repeats,
        )
        return (
            {
                "t_total_ms": round(record.t_mean_ms, 4),
                "t_select_ms": round(record.t_select_ms, 4),
                "t_kernel_ms": round(record.t_kernel_ms, 4),
                "mse": round(record.mse, 7),
                "kl": round(record.kl, 7),
                "passed": bool(record.passed),
                "backend": decision.backend,
                "params": decision.params,
                "reason": decision.reason,
                "active_block_fraction": record.active_block_fraction,
            },
            decision,
        )

    if decision.strategy_instance is not None:
        strategy = decision.strategy_instance
        for _ in range(warmup):
            wr = strategy.select(q, k, v, block_size=128)
            run_block_sparse_attention_flex(q, k, v, wr.block_mask, block_size=128)

        sel_times: list[float] = []
        sel_result = None
        for _ in range(repeats):
            sel_result = strategy.select(q, k, v, block_size=128)
            sel_times.append(float(sel_result.t_select_ms))
        assert sel_result is not None
        t_sel_ms = float(sum(sel_times) / len(sel_times))

        out_sparse, t_ker_ms = run_block_sparse_attention_flex(q, k, v, sel_result.block_mask, block_size=128)
        ker_times = [t_ker_ms]
        for _ in range(repeats - 1):
            _, t_next = run_block_sparse_attention_flex(q, k, v, sel_result.block_mask, block_size=128)
            ker_times.append(t_next)
        t_ker_ms = float(sum(ker_times) / len(ker_times))
        mse = float(torch.mean((out_sparse.float() - reference.float()) ** 2).item())
        kl = float(
            F.kl_div(
                torch.log_softmax(out_sparse.float().mean(-1), dim=-1),
                torch.softmax(reference.float().mean(-1), dim=-1),
                reduction="batchmean",
            ).item()
        )
        return (
            {
                "t_total_ms": round(t_sel_ms + t_ker_ms, 4),
                "t_select_ms": round(t_sel_ms, 4),
                "t_kernel_ms": round(t_ker_ms, 4),
                "mse": round(mse, 7),
                "kl": round(kl, 7),
                "passed": bool(mse <= 0.02 and kl <= 0.10),
                "backend": decision.backend,
                "params": decision.params,
                "reason": decision.reason,
                "active_block_fraction": round(float(sel_result.active_fraction), 4),
            },
            decision,
        )

    cfg = FlexConfig(
        gamma=float(decision.params["gamma"]),
        tau=float(decision.params["tau"]),
        min_budget_frac=float(decision.params["min_budget_frac"]),
        block_size=int(decision.params["block_size"]),
    )
    record = run_flex(
        q,
        k,
        v,
        cfg,
        reference=reference,
        warmup=warmup,
        repeats=repeats,
    )
    return (
        {
            "t_total_ms": round(record.t_mean_ms, 4),
            "t_select_ms": 0.0,
            "t_kernel_ms": round(record.t_mean_ms, 4),
            "mse": round(record.mse, 7),
            "kl": round(record.kl, 7),
            "passed": bool(record.passed),
            "backend": "flex",
            "params": decision.params,
            "reason": decision.reason,
            "active_block_fraction": record.active_block_fraction,
        },
        decision,
    )


def _aggregate(rows: list[dict[str, Any]], field: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[field])].append(row)

    def _sort_key(value: str) -> tuple[int, str]:
        return (0, f"{int(value):09d}") if value.isdigit() else (1, value)

    out: list[dict[str, Any]] = []
    for key in sorted(grouped, key=_sort_key):
        cells = grouped[key]
        pbs_avg = sum(c["pbs_fixed"]["t_total_ms"] for c in cells) / len(cells)
        flex_avg = sum(c["flex_fixed"]["t_total_ms"] for c in cells) / len(cells)
        ours_avg = sum(c["ours"]["t_total_ms"] for c in cells) / len(cells)
        ours_mse = sum(c["ours"]["mse"] for c in cells) / len(cells)
        summary = {
            field: int(key) if key.isdigit() else key,
            "n": len(cells),
            "pbs_ms_avg": round(pbs_avg, 4),
            "flex_ms_avg": round(flex_avg, 4),
            "ours_ms_avg": round(ours_avg, 4),
            "ours_mse_avg": round(ours_mse, 7),
            "gain_pct_vs_pbs": round((pbs_avg - ours_avg) / pbs_avg * 100, 2),
            "gain_pct_vs_flex": round((flex_avg - ours_avg) / flex_avg * 100, 2),
            "ours_pass_rate": round(sum(1 for c in cells if c["ours"]["passed"]) / len(cells), 4),
        }
        code_hits = [
            c["code_evidence_hit_any"]
            for c in cells
            if c["code_evidence_hit_any"] is not None
        ]
        code_hits_all = [
            c["code_evidence_hit_all"]
            for c in cells
            if c["code_evidence_hit_all"] is not None
        ]
        if code_hits:
            summary["code_evidence_hit_any_rate"] = round(sum(code_hits) / len(code_hits), 4)
        if code_hits_all:
            summary["code_evidence_hit_all_rate"] = round(sum(code_hits_all) / len(code_hits_all), 4)
        out.append(summary)
    return out


def main() -> None:
    args = _parse_args()
    with open(args.manifest, "r", encoding="utf-8") as fp:
        manifest = json.load(fp)
    entries = manifest["entries"]

    cells: list[dict[str, Any]] = []
    for idx, entry in enumerate(entries, 1):
        prompt_family = entry["prompt_family"]
        seq_len = int(entry["actual_len"])
        target_len = int(entry["target_len"])
        category = entry["category"]
        print(
            f"[{idx:>2}/{len(entries)}] {category:<16} sample={_entry_tag(entry)} "
            f"L={seq_len:<7} target={target_len:<7}"
        )
        q, k, v = _load_entry_qkv(args.cache_dir, entry, args.device)

        sp = estimate_sparsity(q, k, block_size=128, sample_rows=64, seed=42)
        flash_rec = run_flash(q, k, v, warmup=args.warmup, repeats=args.repeats)
        with torch.inference_mode():
            reference = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        pbs_rec = run_pbs_decomposed(
            q,
            k,
            v,
            PBS_FIXED,
            flash_time_ms=flash_rec.t_mean_ms,
            reference=reference,
            warmup=args.warmup,
            repeats=args.repeats,
        )
        flex_rec = run_flex(
            q,
            k,
            v,
            FLEX_FIXED,
            reference=reference,
            warmup=args.warmup,
            repeats=args.repeats,
        )
        ours, decision = _run_ours(
            q,
            k,
            v,
            sparsity=sp.estimated_sparsity_ratio,
            k_norm_cv=sp.kv_norm_cv,
            flash_ms=flash_rec.t_mean_ms,
            reference=reference,
            warmup=args.warmup,
            repeats=args.repeats,
        )

        evidence_stats = None
        if category == "code_repo_qa" and decision.strategy_instance is not None:
            try:
                sel_result = decision.strategy_instance.select(q, k, v, block_size=128)
                evidence_stats = _mask_hits_evidence(
                    sel_result.block_mask,
                    list(entry.get("evidence_block_indices", [entry["evidence_block_index"]])),
                )
            except Exception as exc:  # pragma: no cover
                warnings.warn(f"evidence retention failed for {_entry_key(entry)}: {exc}")

        cells.append(
            {
                "category": category,
                "sample_tag": _entry_tag(entry),
                "cache_key": _entry_key(entry),
                "prompt_family": prompt_family,
                "seq_len": seq_len,
                "target_len": target_len,
                "source_id": entry["source_id"],
                "domain": entry["domain"],
                "sub_domain": entry["sub_domain"],
                "difficulty": entry.get("difficulty", ""),
                "answer_letter": entry["answer_letter"],
                "question": entry["question"],
                "evidence_block_index": entry["evidence_block_index"],
                "evidence_block_indices": entry.get("evidence_block_indices", [entry["evidence_block_index"]]),
                "evidence_preview": entry["evidence_preview"],
                "sparsity_ratio": round(sp.estimated_sparsity_ratio, 4),
                "kv_norm_cv": round(sp.kv_norm_cv, 4),
                "flash_ms": round(flash_rec.t_mean_ms, 4),
                "pbs_fixed": {
                    "t_total_ms": round(pbs_rec.t_mean_ms, 4),
                    "t_select_ms": round(pbs_rec.t_select_ms, 4),
                    "t_kernel_ms": round(pbs_rec.t_kernel_ms, 4),
                    "mse": round(pbs_rec.mse, 7),
                    "kl": round(pbs_rec.kl, 7),
                    "passed": bool(pbs_rec.passed),
                },
                "flex_fixed": {
                    "t_total_ms": round(flex_rec.t_mean_ms, 4),
                    "mse": round(flex_rec.mse, 7),
                    "kl": round(flex_rec.kl, 7),
                    "passed": bool(flex_rec.passed),
                },
                "ours": ours,
                "gain_vs_pbs_pct": round((pbs_rec.t_mean_ms - ours["t_total_ms"]) / pbs_rec.t_mean_ms * 100, 2),
                "gain_vs_flex_pct": round((flex_rec.t_mean_ms - ours["t_total_ms"]) / flex_rec.t_mean_ms * 100, 2),
                "code_evidence_hit_any": None if evidence_stats is None else evidence_stats["any"],
                "code_evidence_hit_all": None if evidence_stats is None else evidence_stats["all"],
                "code_evidence_hit_fraction": None if evidence_stats is None else evidence_stats["hit_fraction"],
            }
        )

        del q, k, v, reference
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    output = {
        "manifest": str(args.manifest),
        "cache_dir": str(args.cache_dir),
        "cells": cells,
        "aggregated_by_category": _aggregate(cells, "category"),
        "aggregated_by_target_length": _aggregate(cells, "target_len"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fp:
        json.dump(output, fp, indent=2, ensure_ascii=False)
    print(f"saved -> {args.output}")


if __name__ == "__main__":
    main()
