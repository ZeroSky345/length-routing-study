#!/usr/bin/env python3
"""
Multi-text-scenario length routing study.

For each (text scenario × sequence length) combination:
  1. Estimate attention sparsity from the scenario's QKV data.
  2. Build a sparsity-aware theory routing plan.
  3. Run real PBS / FlexPrefill kernel sweep (optional; --theory-only to skip).
  4. Aggregate per-length averages across scenarios.
  5. Write a JSON result suitable for the canvas visualization.

Usage
-----
  # Theory + sparsity only (fast, no GPU kernels):
  python scripts/multi_text_study.py \
      --lengths 4096 8192 16384 32768 65536 \
      --cache-dir data/cache \
      --theory-only \
      --output results/multi_text_study.json

  # Full sweep (slow — runs real kernels):
  python scripts/multi_text_study.py \
      --lengths 4096 8192 16384 32768 65536 \
      --cache-dir data/cache \
      --output results/multi_text_study.json
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import torch

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from length_routing_study._paths import ensure_external_paths
ensure_external_paths(include_baseline=False)

from length_routing_study.dispatcher import TheoryDrivenDispatcher
from length_routing_study.empirical_sweep import (
    FlexConfig,
    PBSConfig,
    load_qkv,
    run_flash,
    run_pbs,
    run_flex,
)
from length_routing_study.sparse_plan import SparseBackendConfig
from length_routing_study.sparsity_estimator import estimate_sparsity

SCENARIOS = ["code", "technical", "narrative", "dialogue", "structured"]

SCENARIO_LABELS = {
    "code":       "Code",
    "technical":  "Technical",
    "narrative":  "Narrative",
    "dialogue":   "Dialogue",
    "structured": "Structured",
}


def run_scenario_length(
    scenario: str,
    L: int,
    cache_dir: Path,
    seed: int,
    dispatcher: TheoryDrivenDispatcher,
    model_family: str,
    objective: str,
    pbs_configs: list[PBSConfig],
    flex_configs: list[FlexConfig],
    warmup: int,
    repeats: int,
    theory_only: bool,
    device: str,
) -> dict:
    """Run one (scenario, length) cell and return a result dict."""
    # ── Load QKV ─────────────────────────────────────────────────────────────
    q, k, v = load_qkv(cache_dir, L, prompt_family=scenario, seed=seed, device=device)

    # ── Sparsity estimation ───────────────────────────────────────────────────
    sp = estimate_sparsity(q, k, block_size=128, sample_rows=64, seed=seed)

    # ── Theory plan (sparsity-aware) ──────────────────────────────────────────
    plan = dispatcher.build_plan(
        model_or_name=model_family,
        prompt_tokens=L,
        objective=objective,
        sparsity_ratio=sp.estimated_sparsity_ratio,
        kv_norm_cv=sp.kv_norm_cv,
        sparsity_source="estimated",
    )
    theory_scores = {
        e.backend: {
            "latency_ms": round(e.estimated_latency_ms, 2),
            "score": round(e.score, 2),
        }
        for e in plan.candidate_estimates
    }

    result = {
        "scenario": scenario,
        "seq_len": L,
        "sparsity_ratio": round(sp.estimated_sparsity_ratio, 4),
        "active_fraction": round(sp.estimated_active_fraction, 4),
        "kv_norm_cv": round(sp.kv_norm_cv, 4),
        "topk_coverage": round(sp.sample_topk_coverage, 4),
        "theory_backend": plan.backend,
        "theory_scores": theory_scores,
        # empirical filled below
        "flash_ms": None,
        "pbs_best_ms": None,
        "pbs_best_config": None,
        "flex_best_ms": None,
        "flex_best_config": None,
        "empirical_winner": None,
    }

    if theory_only:
        del q, k, v
        return result

    # ── Flash reference ───────────────────────────────────────────────────────
    flash_rec = run_flash(q, k, v, warmup=warmup, repeats=repeats)
    result["flash_ms"] = round(flash_rec.t_mean_ms, 3)

    # ── Reference for MSE ────────────────────────────────────────────────────
    import torch.nn.functional as F
    with torch.inference_mode():
        ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)

    # ── PBS sweep ─────────────────────────────────────────────────────────────
    best_pbs_ms, best_pbs_name = float("inf"), None
    for cfg in pbs_configs:
        try:
            rec = run_pbs(q, k, v, cfg, reference=ref,
                          warmup=warmup, repeats=repeats)
            if rec.passed and rec.t_mean_ms < best_pbs_ms:
                best_pbs_ms = rec.t_mean_ms
                best_pbs_name = cfg.name
        except Exception as e:
            pass

    if best_pbs_name:
        result["pbs_best_ms"]     = round(best_pbs_ms, 3)
        result["pbs_best_config"] = best_pbs_name

    # ── Flex sweep ────────────────────────────────────────────────────────────
    best_flex_ms, best_flex_name = float("inf"), None
    for cfg in flex_configs:
        try:
            rec = run_flex(q, k, v, cfg, reference=ref,
                           warmup=warmup, repeats=repeats)
            if rec.passed and rec.t_mean_ms < best_flex_ms:
                best_flex_ms = rec.t_mean_ms
                best_flex_name = cfg.name
        except Exception as e:
            pass

    if best_flex_name:
        result["flex_best_ms"]     = round(best_flex_ms, 3)
        result["flex_best_config"] = best_flex_name

    # ── Empirical winner ──────────────────────────────────────────────────────
    candidates = {
        "flash":        result["flash_ms"],
        "pbs":          result["pbs_best_ms"],
        "flexprefill":  result["flex_best_ms"],
    }
    valid = {k: v for k, v in candidates.items() if v is not None}
    if valid:
        result["empirical_winner"] = min(valid, key=valid.get)

    del q, k, v, ref
    torch.cuda.empty_cache()
    return result


def aggregate_per_length(cells: list[dict]) -> list[dict]:
    """Average numeric fields across scenarios for each length."""
    from collections import defaultdict
    import statistics

    by_len: dict[int, list[dict]] = defaultdict(list)
    for c in cells:
        by_len[c["seq_len"]].append(c)

    agg = []
    for L in sorted(by_len.keys()):
        group = by_len[L]

        def avg(field: str) -> float | None:
            vals = [c[field] for c in group if c.get(field) is not None]
            return round(sum(vals) / len(vals), 4) if vals else None

        def mode_str(field: str) -> str:
            vals = [c[field] for c in group if c.get(field)]
            if not vals:
                return "unknown"
            from collections import Counter
            return Counter(vals).most_common(1)[0][0]

        agg.append({
            "seq_len": L,
            "n_scenarios": len(group),
            "sparsity_ratio_avg": avg("sparsity_ratio"),
            "sparsity_ratio_by_scenario": {c["scenario"]: c["sparsity_ratio"] for c in group},
            "kv_norm_cv_avg": avg("kv_norm_cv"),
            "flash_ms_avg": avg("flash_ms"),
            "pbs_best_ms_avg": avg("pbs_best_ms"),
            "flex_best_ms_avg": avg("flex_best_ms"),
            "theory_backend_plurality": mode_str("theory_backend"),
            "theory_backend_by_scenario": {c["scenario"]: c["theory_backend"] for c in group},
            "empirical_winner_plurality": mode_str("empirical_winner"),
        })
    return agg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lengths",     nargs="+", type=int,
                    default=[4096, 8192, 16384, 32768, 65536])
    ap.add_argument("--scenarios",   nargs="+", default=SCENARIOS)
    ap.add_argument("--cache-dir",   type=Path, default=Path("data/cache"))
    ap.add_argument("--model",       default="qwen2")
    ap.add_argument("--objective",   default="balanced")
    ap.add_argument("--seed",        type=int, default=42)
    ap.add_argument("--device",      default="cuda")
    ap.add_argument("--warmup",      type=int, default=2)
    ap.add_argument("--repeats",     type=int, default=5)
    ap.add_argument("--theory-only", action="store_true",
                    help="Skip real kernel sweep; only compute sparsity + theory routing.")
    # PBS / Flex grids (compact for multi-scenario speed)
    ap.add_argument("--pbs-thresholds",    nargs="+", type=float, default=[0.8, 0.9])
    ap.add_argument("--pbs-segment-sizes", nargs="+", type=int,   default=[256])
    ap.add_argument("--flex-gammas",       nargs="+", type=float, default=[0.90, 0.95])
    ap.add_argument("--flex-taus",         nargs="+", type=float, default=[0.05, 0.10, 0.20])
    ap.add_argument("--flex-mbfs",         nargs="+", type=float, default=[0.0, 0.10])
    ap.add_argument("--output",  type=Path, default=Path("results/multi_text_study.json"))
    args = ap.parse_args()

    pbs_configs = [
        PBSConfig(threshold=t, segment_size=s)
        for t in args.pbs_thresholds for s in args.pbs_segment_sizes
    ]
    flex_configs = [
        FlexConfig(gamma=g, tau=t, min_budget_frac=f)
        for g in args.flex_gammas for t in args.flex_taus for f in args.flex_mbfs
    ]

    dispatcher = TheoryDrivenDispatcher(
        backend_config=SparseBackendConfig(),
        model_memory_budget_gb=72.0,
    )

    total = len(args.scenarios) * len(args.lengths)
    cells: list[dict] = []
    n = 0

    for scenario in args.scenarios:
        for L in args.lengths:
            n += 1
            print(f"[{n:>3}/{total}] {scenario:>12}  L={L:>6}", end="  ")
            try:
                cell = run_scenario_length(
                    scenario=scenario, L=L, cache_dir=args.cache_dir,
                    seed=args.seed, dispatcher=dispatcher,
                    model_family=args.model, objective=args.objective,
                    pbs_configs=pbs_configs, flex_configs=flex_configs,
                    warmup=args.warmup, repeats=args.repeats,
                    theory_only=args.theory_only, device=args.device,
                )
                tag = f"sparsity={cell['sparsity_ratio']:.3f}  theory={cell['theory_backend']}"
                if not args.theory_only and cell.get("flash_ms"):
                    tag += f"  flash={cell['flash_ms']:.1f}ms"
                print(tag)
                cells.append(cell)
            except Exception as exc:
                print(f"ERROR: {exc}")
                import traceback; traceback.print_exc()

    agg = aggregate_per_length(cells)

    output = {
        "model_family": args.model,
        "objective":    args.objective,
        "theory_only":  args.theory_only,
        "scenarios":    args.scenarios,
        "lengths":      args.lengths,
        "cells":        cells,
        "aggregated":   agg,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved → {args.output}")

    # Quick summary table
    print("\n── Per-length average sparsity ─────────────────────────────────")
    print(f"  {'L':>7}  " + "  ".join(f"{s:>12}" for s in args.scenarios)
          + f"  {'avg':>7}")
    for row in agg:
        L = row["seq_len"]
        by_s = row["sparsity_ratio_by_scenario"]
        vals = [f"{by_s.get(s, float('nan')):>12.3f}" for s in args.scenarios]
        print(f"  {L:>7}  " + "  ".join(vals)
              + f"  {row['sparsity_ratio_avg']:>7.3f}")


if __name__ == "__main__":
    main()
