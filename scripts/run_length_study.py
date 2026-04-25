#!/usr/bin/env python3
"""
Main entry point: joint theory + empirical length-routing study.

Runs ``TheoryDrivenDispatcher`` across the given sequence lengths to predict
which backend should win, then runs the real PBS / FlexPrefill kernels to
measure what actually wins, and writes a unified ``LengthStudyResult`` JSON.

Usage
-----
  export PYTHONPATH=/root/length-routing-study/src:/root/pbs-attn-src

  # Quick smoke test with defaults
  python scripts/run_length_study.py \
      --lengths 4096 8192 16384 32768 65536 \
      --cache-dir data/cache \
      --output-dir results/

  # Full sweep with custom grids
  python scripts/run_length_study.py \
      --lengths 4096 8192 16384 32768 65536 \
      --pbs-thresholds 0.7 0.8 0.9 \
      --pbs-segment-sizes 128 256 512 \
      --flex-gammas 0.80 0.90 0.95 \
      --flex-taus 0.05 0.10 0.20 \
      --flex-min-budget-fracs 0.0 0.10 0.25 \
      --model qwen2 \
      --objective balanced \
      --warmup 2 --repeats 5 \
      --mse-threshold 0.02 \
      --output-dir results/
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

# ── path bootstrap ─────────────────────────────────────────────────────────────
_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from length_routing_study._paths import ensure_external_paths
ensure_external_paths(include_baseline=False)

from length_routing_study.empirical_sweep import (
    FlexConfig,
    PBSConfig,
    default_flex_grid,
    default_pbs_grid,
)
from length_routing_study.length_study import LengthStudyRunner


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Run theory + empirical joint length-routing study"
    )
    ap.add_argument("--lengths", nargs="+", type=int,
                    default=[4096, 8192, 16384, 32768, 65536])
    # Theory
    ap.add_argument("--model",     default="qwen2",
                    choices=("qwen2", "llama", "glm", "generic"),
                    help="Model family for theory cost model.")
    ap.add_argument("--objective", default="balanced",
                    choices=("balanced", "speed", "stability", "memory"))
    ap.add_argument("--memory-budget-gb", type=float, default=72.0)
    # PBS grid
    ap.add_argument("--pbs-thresholds",    nargs="+", type=float, default=[0.7, 0.8, 0.9])
    ap.add_argument("--pbs-segment-sizes", nargs="+", type=int,   default=[128, 256, 512])
    ap.add_argument("--pbs-block-size",    type=int,  default=128)
    # Flex grid
    ap.add_argument("--flex-gammas",           nargs="+", type=float, default=[0.80, 0.90, 0.95])
    ap.add_argument("--flex-taus",             nargs="+", type=float, default=[0.05, 0.10, 0.20])
    ap.add_argument("--flex-min-budget-fracs", nargs="+", type=float, default=[0.0, 0.10, 0.25])
    ap.add_argument("--flex-block-size",       type=int,  default=128)
    # Data
    ap.add_argument("--cache-dir",     default="data/cache")
    ap.add_argument("--prompt-family", default="default")
    ap.add_argument("--seed",          type=int, default=42)
    ap.add_argument("--device",        default="cuda")
    # Measurement
    ap.add_argument("--warmup",        type=int,   default=2)
    ap.add_argument("--repeats",       type=int,   default=5)
    ap.add_argument("--mse-threshold", type=float, default=0.02)
    ap.add_argument("--kl-threshold",  type=float, default=0.10)
    # Output
    ap.add_argument("--output-dir", type=Path, default=Path("results"))
    ap.add_argument("--tag",        default="",
                    help="Optional tag appended to the output filename.")
    ap.add_argument("--quiet", action="store_true")
    # Sparsity estimation
    ap.add_argument("--no-sparsity-estimation", dest="sparsity_estimation",
                    action="store_false", default=True,
                    help="Disable real-QKV sparsity estimation (use geometric fallback).")
    ap.add_argument("--sparsity-sample-rows", type=int, default=64,
                    help="Number of query rows sampled for sparsity estimation.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    # ── Build parameter grids ─────────────────────────────────────────────────
    pbs_configs = [
        PBSConfig(threshold=thr, segment_size=seg, block_size=args.pbs_block_size)
        for thr in args.pbs_thresholds
        for seg in args.pbs_segment_sizes
    ]
    flex_configs = [
        FlexConfig(gamma=g, tau=t, min_budget_frac=f, block_size=args.flex_block_size)
        for g in args.flex_gammas
        for t in args.flex_taus
        for f in args.flex_min_budget_fracs
    ]

    print(f"Lengths   : {args.lengths}")
    print(f"PBS cfgs  : {len(pbs_configs)}")
    print(f"Flex cfgs : {len(flex_configs)}")
    print(f"Theory    : model={args.model}  objective={args.objective}")
    print(f"Cache dir : {args.cache_dir}")
    print()

    runner = LengthStudyRunner(
        model_family=args.model,
        objective=args.objective,
        memory_budget_gb=args.memory_budget_gb,
        pbs_configs=pbs_configs,
        flex_configs=flex_configs,
        cache_dir=args.cache_dir,
        prompt_family=args.prompt_family,
        seed=args.seed,
        device=args.device,
        warmup=args.warmup,
        repeats=args.repeats,
        mse_threshold=args.mse_threshold,
        kl_threshold=args.kl_threshold,
        estimate_sparsity=args.sparsity_estimation,
        sparsity_sample_rows=args.sparsity_sample_rows,
        verbose=not args.quiet,
    )

    result = runner.run(args.lengths)

    # ── Print summary table ───────────────────────────────────────────────────
    print("\n" + "="*72)
    print("SUMMARY")
    print("="*72)
    hdr = f"{'L':>7}  {'Theory':>22}  {'Empirical winner':>26}  {'Agree':>6}  {'Err%':>7}"
    print(hdr)
    print("-"*72)
    for c in result.cells:
        agree = "YES" if c.theory_agrees else "NO "
        err = f"{c.theory_latency_error_pct:+.1f}%" if c.theory_latency_error_pct == c.theory_latency_error_pct else "N/A"
        print(f"{c.seq_len:>7}  {c.theory_backend:>22}  {c.empirical_winner:>26}  {agree:>6}  {err:>7}")

    print()
    print("Flash vs PBS best vs Flex best:")
    hdr2 = f"{'L':>7}  {'Flash ms':>10}  {'PBS best ms':>12}  {'Flex best ms':>13}  {'PBS/Flash':>10}  {'Flex/Flash':>11}"
    print(hdr2)
    print("-"*72)
    for c in result.cells:
        pbs_ratio  = c.pbs_best_ms  / c.flash_ms if c.flash_ms > 0 and c.pbs_best_ms  == c.pbs_best_ms  else float("nan")
        flex_ratio = c.flex_best_ms / c.flash_ms if c.flash_ms > 0 and c.flex_best_ms == c.flex_best_ms else float("nan")
        print(f"{c.seq_len:>7}  {c.flash_ms:>10.3f}  {c.pbs_best_ms:>12.3f}  {c.flex_best_ms:>13.3f}"
              f"  {pbs_ratio:>10.3f}  {flex_ratio:>11.3f}")

    # ── Save ──────────────────────────────────────────────────────────────────
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    tag   = f"_{args.tag}" if args.tag else ""
    fname = args.output_dir / f"length_study_{args.model}_{args.objective}{tag}_{stamp}.json"
    result.save(fname)

    # Also write a convenience "latest" symlink
    latest = args.output_dir / f"length_study_latest.json"
    latest.unlink(missing_ok=True)
    latest.symlink_to(fname.name)
    print(f"Latest symlink → {latest}")


if __name__ == "__main__":
    main()
