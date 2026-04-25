#!/usr/bin/env python3
"""
Run custom mask-based selector sweep across text scenarios and lengths.

Measures for each (selector × scenario × length):
  - sparsity ratio of the generated mask
  - selection overhead (mask generation time, ms)
  - masked-attention kernel time (ms)
  - MSE vs Flash reference

Also loads the existing multi_text_study.json (PBS/Flex results) and
produces a unified comparison JSON.

Usage
-----
  export PYTHONPATH=/root/length-routing-study/src
  python scripts/run_selector_study.py \
      --lengths 4096 8192 16384 32768 65536 \
      --scenarios code technical narrative dialogue structured \
      --cache-dir data/cache \
      --warmup 2 --repeats 5 \
      --output results/selector_study.json
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from collections import defaultdict

import torch

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from length_routing_study.empirical_sweep import load_qkv
from length_routing_study.selector_sweep import (
    default_selector_suite,
    run_selector_sweep,
    SelectorRecord,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lengths",    nargs="+", type=int,
                    default=[4096, 8192, 16384, 32768, 65536])
    ap.add_argument("--scenarios",  nargs="+",
                    default=["code", "technical", "narrative", "dialogue", "structured"])
    ap.add_argument("--cache-dir",  type=Path, default=Path("data/cache"))
    ap.add_argument("--seed",       type=int,  default=42)
    ap.add_argument("--device",     default="cuda")
    ap.add_argument("--warmup",     type=int,  default=2)
    ap.add_argument("--repeats",    type=int,  default=5)
    ap.add_argument("--mse-threshold", type=float, default=0.05)
    ap.add_argument("--output",  type=Path, default=Path("results/selector_study.json"))
    args = ap.parse_args()

    # ── Load all QKV upfront ─────────────────────────────────────────────────
    print("Loading QKV caches...")
    scenarios_qkv: dict[str, dict[int, tuple]] = {}
    for scenario in args.scenarios:
        qkv_by_len = {}
        for L in args.lengths:
            try:
                q, k, v = load_qkv(
                    args.cache_dir, L,
                    prompt_family=scenario,
                    seed=args.seed,
                    device=args.device,
                )
                qkv_by_len[L] = (q, k, v)
            except Exception as e:
                warnings.warn(f"Could not load QKV for {scenario} L={L}: {e}")
        if qkv_by_len:
            scenarios_qkv[scenario] = qkv_by_len
    print(f"  Loaded: {sum(len(v) for v in scenarios_qkv.values())} (scenario,length) pairs\n")

    # ── Run selector sweep ───────────────────────────────────────────────────
    suite = default_selector_suite()
    print(f"Selector suite: {len(suite)} selectors")
    print(f"Scenarios × lengths: {len(scenarios_qkv)} × {len(args.lengths)}\n")

    records = run_selector_sweep(
        lengths=args.lengths,
        scenarios_qkv=scenarios_qkv,
        selectors=suite,
        warmup=args.warmup,
        repeats=args.repeats,
        mse_threshold=args.mse_threshold,
        verbose=True,
    )

    # ── Aggregate: per selector, average over scenarios ──────────────────────
    by_sel_len: dict[tuple, list[SelectorRecord]] = defaultdict(list)
    for r in records:
        by_sel_len[(r.selector_name, r.seq_len)].append(r)

    agg_rows = []
    for (sel_name, L), recs in sorted(by_sel_len.items(), key=lambda x: (x[0][0], x[0][1])):
        def avg(field):
            return round(sum(getattr(r, field) for r in recs) / len(recs), 4)
        agg_rows.append({
            "selector_name":  sel_name,
            "selector_type":  recs[0].selector_type,
            "seq_len":        L,
            "n_scenarios":    len(recs),
            "sparsity_avg":   avg("sparsity"),
            "t_select_ms_avg": avg("t_select_ms"),
            "t_kernel_ms_avg": avg("t_kernel_ms"),
            "t_total_ms_avg":  avg("t_total_ms"),
            "mse_avg":         round(sum(r.mse for r in recs) / len(recs), 6),
            "pass_rate":       round(sum(1 for r in recs if r.passed) / len(recs), 3),
            "sparsity_by_scenario": {r.scenario: r.sparsity for r in recs},
        })

    # ── Print summary ─────────────────────────────────────────────────────────
    print("\n" + "="*90)
    print("SUMMARY: avg over scenarios")
    print("="*90)
    print(f"{'Selector':<45} {'Type':<8} "
          + "  ".join(f"{L//1024}K" for L in args.lengths))
    print("-"*90)

    sel_names = sorted(set(r["selector_name"] for r in agg_rows))
    for sn in sel_names:
        rows_for_sel = {r["seq_len"]: r for r in agg_rows if r["selector_name"] == sn}
        stype = next(r["selector_type"] for r in agg_rows if r["selector_name"] == sn)
        sparsities = [rows_for_sel.get(L, {}).get("sparsity_avg", float("nan"))
                      for L in args.lengths]
        print(f"  {sn:<43} {stype:<8} "
              + "  ".join(f"{s:.3f}" for s in sparsities))

    # ── Save ──────────────────────────────────────────────────────────────────
    output = {
        "lengths":   args.lengths,
        "scenarios": args.scenarios,
        "selectors": [s.name for s in suite],
        "records":   [r.as_dict() for r in records],
        "aggregated": agg_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved → {args.output}  ({len(records)} records, {len(agg_rows)} aggregated)")


if __name__ == "__main__":
    main()
