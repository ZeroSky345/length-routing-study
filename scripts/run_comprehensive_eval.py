#!/usr/bin/env python3
"""
Comprehensive Evaluation: Ours vs PBS Fixed (4K?128K)
=======================================================
Measures PBS Fixed, Flex Fixed, and Ours (adaptive router) empirically on 8 scenarios.

For each (scenario ? length) cell, records:
  Speed : t_total_ms, t_select_ms, t_kernel_ms
  Accuracy: mse (vs Flash dense reference), kl divergence, passed (bool)
  Structure: kernel_time_ratio, active_block_fraction
  Gain : % speedup of Ours vs PBS Fixed

Usage
-----
  python scripts/run_comprehensive_eval.py       --lengths 4096 8192 16384 32768 65536 131072       --cache-dir data/cache       --warmup 3 --repeats 7
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from length_routing_study._paths import ensure_external_paths
ensure_external_paths(include_baseline=False)

from length_routing_study.empirical_sweep import (
    PBSConfig, FlexConfig, load_qkv, run_flash, run_pbs_decomposed, run_flex, _cuda_ms, _stats,
)
from length_routing_study.sparsity_estimator import estimate_sparsity
from length_routing_study.length_router import LengthAwareRouter
from length_routing_study.selection_strategies import (
    run_block_sparse_attention,
    run_block_sparse_attention_flex,
)

# ── Config ────────────────────────────────────────────────────────────────────

PBS_FIXED  = PBSConfig(threshold=0.9, segment_size=256, block_size=128,
                       use_triton=True, force_first=True)
FLEX_FIXED = FlexConfig(gamma=0.95, tau=0.1, min_budget_frac=0.0, block_size=128)
# 3 uniform-distribution scenarios  +  5 realistic-LLM-attention scenarios  (ratio ≈ 3:5)
# Realistic scenarios use block-context injection so attention is genuinely concentrated,
# enabling FlexPrefill and sparse strategies to achieve meaningful sparsity.
ALL_SCENARIOS = [
    # ── uniform / K-norm controlled (3) ──────────────────────────────────────
    "sparse_low", "sparse_med", "sparse_high",
    # ── realistic LLM attention patterns (5) ─────────────────────────────────
    "lm_sink_local",     # sink(0,1) + local blocks (B=64)
    "lm_sparse_global",  # 5% global anchor tokens
    "lm_hierarchical",   # block leaders every 128 tokens
    "lm_local_periodic", # small local blocks (B=32) + periodic sinks
    "lm_mixed",          # composite: sink + anchors + local
]

def run_cell(
    scenario: str, L: int,
    cache_dir: Path, router: LengthAwareRouter,
    warmup: int, repeats: int, seed: int, device: str,
) -> dict:
    q, k, v = load_qkv(cache_dir, L, prompt_family=scenario,
                        seed=seed, device=device)
    dtype = q.dtype
    k = k.to(dtype=dtype)
    v = v.to(dtype=dtype)

    sp       = estimate_sparsity(q, k, block_size=128, sample_rows=64, seed=seed)
    sparsity = sp.estimated_sparsity_ratio
    knorm_cv = sp.kv_norm_cv

    flash_rec = run_flash(q, k, v, warmup=warmup, repeats=repeats)
    flash_ms  = flash_rec.t_mean_ms

    with torch.inference_mode():
        ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)

    strats: dict[str, dict] = {}

    try:
        r = run_pbs_decomposed(q, k, v, PBS_FIXED,
                               flash_time_ms=flash_ms, reference=ref,
                               warmup=warmup, repeats=repeats)
        strats['pbs_fixed'] = {
            't_total_ms':  round(r.t_mean_ms, 4),
            't_select_ms': round(r.t_select_ms, 4),
            't_kernel_ms': round(r.t_kernel_ms, 4),
            'mse': round(r.mse, 7), 'kl': round(r.kl, 7),
            'passed': bool(r.passed),
            'kernel_time_ratio': r.kernel_time_ratio,
            'active_block_fraction': r.active_block_fraction,
            'is_theory': False,
        }
    except Exception as exc:
        warnings.warn(f'PBS @ {scenario} L={L}: {exc}')

    try:
        rf = run_flex(q, k, v, FLEX_FIXED, reference=ref,
                      warmup=warmup, repeats=repeats)
        strats['flex_fixed'] = {
            't_total_ms':  round(rf.t_mean_ms, 4),
            't_select_ms': 0.0,
            't_kernel_ms': round(rf.t_mean_ms, 4),
            'mse': round(rf.mse, 7), 'kl': round(rf.kl, 7),
            'passed': bool(rf.mse <= 0.02 and rf.kl <= 0.10),
            'kernel_time_ratio': rf.kernel_time_ratio,
            'active_block_fraction': rf.active_block_fraction,
            'is_theory': False,
        }
    except Exception as exc:
        warnings.warn(f'Flex @ {scenario} L={L}: {exc}')

    decision = router.route_full(L, sparsity=sparsity, k_norm_cv=knorm_cv)

    if decision.backend == 'pbs':
        pbs_cfg = PBSConfig(**{
            k_: v_ for k_, v_ in decision.params.items()
            if k_ in ('threshold', 'segment_size', 'block_size', 'use_triton', 'force_first')
        })
        try:
            r = run_pbs_decomposed(q, k, v, pbs_cfg,
                                   flash_time_ms=flash_ms, reference=ref,
                                   warmup=warmup, repeats=repeats)
            strats['ours'] = {
                't_total_ms':  round(r.t_mean_ms, 4),
                't_select_ms': round(r.t_select_ms, 4),
                't_kernel_ms': round(r.t_kernel_ms, 4),
                'mse': round(r.mse, 7), 'kl': round(r.kl, 7),
                'passed': bool(r.passed),
                'backend': decision.backend,
                'params': decision.params,
                'reason': decision.reason,
                'kernel_time_ratio': r.kernel_time_ratio,
                'active_block_fraction': r.active_block_fraction,
                'is_theory': False,
            }
        except Exception as exc:
            warnings.warn(f'Ours-PBS @ {scenario} L={L}: {exc}')

    elif decision.strategy_instance is not None:
        strategy = decision.strategy_instance
        if strategy is None:
            warnings.warn(f'Ours-mask @ {scenario} L={L}: no strategy_instance')
        else:
            try:
                for _ in range(warmup):
                    wr = strategy.select(q, k, v, block_size=128)
                    run_block_sparse_attention_flex(q, k, v, wr.block_mask, block_size=128)

                sel_times: list[float] = []
                sel_result = None
                for _ in range(repeats):
                    sel_result = strategy.select(q, k, v, block_size=128)
                    sel_times.append(sel_result.t_select_ms)
                assert sel_result is not None
                t_sel_ms = float(sum(sel_times) / len(sel_times))
                af = sel_result.active_fraction

                out_sparse, t_ker_ms = run_block_sparse_attention_flex(
                    q, k, v, sel_result.block_mask, block_size=128)
                ker_times = [t_ker_ms]
                for _ in range(repeats - 1):
                    _, t_ = run_block_sparse_attention_flex(
                        q, k, v, sel_result.block_mask, block_size=128)
                    ker_times.append(t_)
                t_ker_ms = float(sum(ker_times) / len(ker_times))
                t_total  = t_sel_ms + t_ker_ms

                out = out_sparse
                mse = float(torch.mean((out.float() - ref.float()) ** 2).item())
                kl  = float(
                    torch.nn.functional.kl_div(
                        torch.log_softmax(out.float().mean(-1), dim=-1),
                        torch.softmax(ref.float().mean(-1), dim=-1),
                        reduction='batchmean',
                    ).item()
                )
                strats['ours'] = {
                    't_total_ms':  round(t_total, 4),
                    't_select_ms': round(t_sel_ms, 4),
                    't_kernel_ms': round(t_ker_ms, 4),
                    'mse': round(mse, 7), 'kl': round(kl, 7),
                    'passed': bool(mse <= 0.02 and kl <= 0.10),
                    'backend': decision.backend,
                    'params':  decision.params,
                    'reason':  decision.reason,
                    'kernel_time_ratio': None,
                    'active_block_fraction': round(af, 4),
                    'is_theory': False,
                }
            except Exception as exc:
                warnings.warn(f'Ours-mask @ {scenario} L={L}: {exc}')
                import traceback; traceback.print_exc()

    else:
        flex_cfg = FlexConfig(**{
            k_: v_ for k_, v_ in decision.params.items()
            if k_ in ('gamma', 'tau', 'min_budget_frac', 'block_size')
        })
        try:
            rf = run_flex(q, k, v, flex_cfg, reference=ref,
                          warmup=warmup, repeats=repeats)
            strats['ours'] = {
                't_total_ms':  round(rf.t_mean_ms, 4),
                't_select_ms': 0.0,
                't_kernel_ms': round(rf.t_mean_ms, 4),
                'mse': round(rf.mse, 7), 'kl': round(rf.kl, 7),
                'passed': bool(rf.mse <= 0.02 and rf.kl <= 0.10),
                'backend': 'flex',
                'params': decision.params,
                'reason': decision.reason,
                'kernel_time_ratio': rf.kernel_time_ratio,
                'active_block_fraction': rf.active_block_fraction,
                'is_theory': False,
            }
        except Exception as exc:
            warnings.warn(f'Ours-Flex @ {scenario} L={L}: {exc}')

    del q, k, v, ref
    torch.cuda.empty_cache()

    gain_vs_pbs = None
    if 'pbs_fixed' in strats and 'ours' in strats:
        pb = strats['pbs_fixed']['t_total_ms']
        ou = strats['ours']['t_total_ms']
        if pb and ou and pb > 0:
            gain_vs_pbs = round((pb - ou) / pb * 100, 2)

    return {
        'scenario': scenario, 'seq_len': L,
        'sparsity': round(sparsity, 4),
        'flash_ms': round(flash_ms, 4),
        'strategies': strats,
        'gain_vs_pbs_pct': gain_vs_pbs,
        'ours_decision': decision.summary(),
    }

def aggregate(cells: list[dict]) -> list[dict]:
    by_len: dict[int, list[dict]] = defaultdict(list)
    for c in cells:
        by_len[c["seq_len"]].append(c)

    rows = []
    for L in sorted(by_len):
        grp = by_len[L]

        def avg(strat: str, field: str) -> float | None:
            vals = [c["strategies"][strat][field]
                    for c in grp
                    if strat in c["strategies"]
                    and c["strategies"][strat].get(field) is not None]
            return round(sum(vals) / len(vals), 4) if vals else None

        row = {
            "seq_len": L, "n": len(grp),
            "sparsity_avg": round(sum(c["sparsity"] for c in grp) / len(grp), 4),
            "flash_ms_avg": round(sum(c["flash_ms"] for c in grp) / len(grp), 4),
            "pbs_fixed_t_ms":  avg("pbs_fixed", "t_total_ms"),
            "pbs_fixed_sel_ms":avg("pbs_fixed", "t_select_ms"),
            "pbs_fixed_ker_ms":avg("pbs_fixed", "t_kernel_ms"),
            "pbs_fixed_mse":   avg("pbs_fixed", "mse"),
            "pbs_fixed_kl":    avg("pbs_fixed", "kl"),
            "ours_t_ms":       avg("ours",      "t_total_ms"),
            "ours_sel_ms":     avg("ours",      "t_select_ms"),
            "ours_ker_ms":     avg("ours",      "t_kernel_ms"),
            "ours_mse":        avg("ours",       "mse"),
            "ours_kl":         avg("ours",       "kl"),
            "flex_fixed_t_ms": avg("flex_fixed", "t_total_ms"),
            "flex_fixed_mse":  avg("flex_fixed", "mse"),
            "flex_fixed_kl":   avg("flex_fixed", "kl"),
        }
        pb = row["pbs_fixed_t_ms"]
        ou = row["ours_t_ms"]
        fx = row["flex_fixed_t_ms"]
        row["gain_pct_vs_pbs"]  = round((pb - ou) / pb * 100, 2) if pb and ou else None
        row["gain_pct_vs_flex"] = round((fx - ou) / fx * 100, 2) if fx and ou else None
        # backward compat
        row["gain_pct"] = row["gain_pct_vs_pbs"]

        # Per-scenario breakdown
        row["by_scenario"] = {
            c["scenario"]: {
                "sparsity":      c["sparsity"],
                "pbs_ms":        c["strategies"].get("pbs_fixed",  {}).get("t_total_ms"),
                "flex_ms":       c["strategies"].get("flex_fixed", {}).get("t_total_ms"),
                "ours_ms":       c["strategies"].get("ours",       {}).get("t_total_ms"),
                "pbs_mse":       c["strategies"].get("pbs_fixed",  {}).get("mse"),
                "flex_mse":      c["strategies"].get("flex_fixed", {}).get("mse"),
                "ours_mse":      c["strategies"].get("ours",       {}).get("mse"),
                "gain_vs_pbs":   c.get("gain_vs_pbs_pct"),
                "ours_decision": c.get("ours_decision", ""),
            }
            for c in grp
        }
        rows.append(row)
    return rows


# ── Printing ──────────────────────────────────────────────────────────────────

def print_results(agg: list[dict], cells: list[dict]) -> None:
    print("\n" + "═" * 100)
    print("COMPREHENSIVE EVALUATION: PBS Fixed  vs  Flex Fixed  vs  Ours")
    print("═" * 100)

    # ── Speed table ────────────────────────────────────────────────────────
    print(f"\n{'Speed (ms) — averaged over 8 scenarios':^95}")
    print(f"{'L':>8}  {'sparsity':>9}  {'PBS Fixed':>10}  {'Flex Fixed':>11}  {'Ours':>10}  "
          f"{'vs PBS':>8}  {'vs Flex':>8}")
    print("─" * 82)
    for r in agg:
        L   = r["seq_len"]
        sp  = r["sparsity_avg"]
        pb  = r["pbs_fixed_t_ms"]
        fx  = r["flex_fixed_t_ms"]
        ou  = r["ours_t_ms"]
        gp  = r.get("gain_pct_vs_pbs")
        gf  = r.get("gain_pct_vs_flex")
        fx_s = f"{fx:.2f}" if fx else "  N/A  "
        gp_s = f"+{gp:.1f}%" if gp else "N/A"
        gf_s = f"+{gf:.1f}%" if gf else "N/A"
        print(f"{L:>8,}  {sp:>9.3f}  {pb:>10.2f}  {fx_s:>11}  {ou:>10.2f}  "
              f"{gp_s:>8}  {gf_s:>8}")

    # ── Accuracy table ─────────────────────────────────────────────────────
    print(f"\n\n{'Accuracy (MSE × 10⁻⁵) — averaged over 8 scenarios':^95}")
    print(f"{'L':>8}  {'PBS MSE':>10}  {'Flex MSE':>10}  {'Ours MSE':>10}  "
          f"{'PBS KL':>9}  {'Flex KL':>9}  {'Ours KL':>9}  {'OK?':>5}")
    print("─" * 82)
    for r in agg:
        L   = r["seq_len"]
        pm  = r["pbs_fixed_mse"]
        fm  = r.get("flex_fixed_mse")
        om  = r["ours_mse"]
        pk  = r["pbs_fixed_kl"]
        fk  = r.get("flex_fixed_kl")
        ok  = r["ours_kl"]
        acc = "✓" if (om is not None and om <= 0.02) else "✗"
        def ms(v): return f"{v*1e5:.2f}" if v is not None else " N/A"
        def kl(v): return f"{v:.5f}"    if v is not None else "  N/A"
        print(f"{L:>8,}  {ms(pm):>10}  {ms(fm):>10}  {ms(om):>10}  "
              f"{kl(pk):>9}  {kl(fk):>9}  {kl(ok):>9}  {acc:>5}")

    # ── Per-scenario detail at 32K ─────────────────────────────────────────
    print(f"\n\nPer-scenario breakdown at L=32K:")
    print(f"  {'Scenario':>12}  {'sp':>6}  {'PBS ms':>8}  {'Flex ms':>8}  {'Ours ms':>8}  "
          f"{'PBS MSE':>9}  {'Flex MSE':>9}  {'Ours MSE':>9}  {'vs PBS':>7}")
    print("  " + "─" * 95)
    r32 = next((r for r in agg if r["seq_len"] == 32768), None)
    if r32:
        for sc, d in sorted(r32["by_scenario"].items()):
            pb  = d.get("pbs_ms");   fx = d.get("flex_ms");  ou = d.get("ours_ms")
            pm  = d.get("pbs_mse");  fm = d.get("flex_mse"); om = d.get("ours_mse")
            gn  = d.get("gain_vs_pbs")
            def v(x):   return f"{x:.2f}" if x is not None else " N/A"
            def vm(x):  return f"{x*1e5:.2f}" if x is not None else "  N/A"
            def vg(x):  return f"+{x:.1f}%" if x is not None else "  N/A"
            print(f"  {sc:>12}  {d['sparsity']:>6.3f}  {v(pb):>8}  {v(fx):>8}  {v(ou):>8}  "
                  f"{vm(pm):>9}  {vm(fm):>9}  {vm(om):>9}  {vg(gn):>7}")


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lengths", type=int, nargs="+",
                   default=[4096, 8192, 16384, 32768, 65536, 131072])
    p.add_argument("--scenarios", nargs="+", default=ALL_SCENARIOS)
    p.add_argument("--cache-dir", type=str, default="data/cache")
    p.add_argument("--seed",    type=int, default=42)
    p.add_argument("--device",  type=str, default="cuda")
    p.add_argument("--warmup",  type=int, default=3)
    p.add_argument("--repeats", type=int, default=7)
    p.add_argument("--output",  type=str, default="results/comprehensive_eval.json")
    return p.parse_args()


def main():
    args = parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"
    cache_dir = Path(args.cache_dir)
    out_path  = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    router = LengthAwareRouter()

    # ── Calibrate router from a quick PBS pass ─────────────────────────────
    print("=== Calibrating router (PBS overhead model) ===")
    flash_ms_list: list[tuple[int, float]] = []
    pbs_sel_list:  list[tuple[int, float]] = []
    cal_scenario = "lm_mixed"
    for L in args.lengths:
        try:
            q, k, v = load_qkv(cache_dir, L, prompt_family=cal_scenario,
                                seed=args.seed, device=device)
            dtype = q.dtype; k = k.to(dtype=dtype); v = v.to(dtype=dtype)
            fr = run_flash(q, k, v, warmup=2, repeats=5)
            pr = run_pbs_decomposed(q, k, v, PBS_FIXED, flash_time_ms=fr.t_mean_ms,
                                    warmup=2, repeats=5)
            flash_ms_list.append((L, fr.t_mean_ms))
            pbs_sel_list.append((L, pr.t_select_ms))
            print(f"  L={L:>7,}  flash={fr.t_mean_ms:.2f}ms  pbs_sel={pr.t_select_ms:.2f}ms")
            del q, k, v; torch.cuda.empty_cache()
        except Exception as exc:
            warnings.warn(f"Calibration @ L={L}: {exc}")

    router = router.calibrate_from_full_measurements(
        flash_ms=flash_ms_list, pbs_total_ms=[], pbs_select_ms=pbs_sel_list)
    print()

    # ── Main sweep ────────────────────────────────────────────────────────
    total = len(args.scenarios) * len(args.lengths)
    cells: list[dict] = []
    n = 0

    print(f"Running {len(args.scenarios)} scenarios × {len(args.lengths)} lengths = {total} cells")
    print(f"  Scenarios: {args.scenarios}")
    print(f"  Lengths  : {args.lengths}\n")

    for sc in args.scenarios:
        for L in args.lengths:
            n += 1
            t0 = time.time()
            print(f"  [{n:>3}/{total}] {sc:>12}  L={L:>7,}", end="  ", flush=True)
            try:
                cell = run_cell(
                    sc, L, cache_dir, router,
                    args.warmup, args.repeats, args.seed, device,
                )
                sp = cell["sparsity"]
                gn = cell.get("gain_vs_pbs_pct", "?")
                pb_ms = cell["strategies"].get("pbs_fixed", {}).get("t_total_ms", "N/A")
                fx_ms = cell["strategies"].get("flex_fixed",{}).get("t_total_ms", "N/A")
                ou_ms = cell["strategies"].get("ours",      {}).get("t_total_ms", "N/A")
                elapsed = time.time() - t0
                print(f"sp={sp:.3f}  PBS={pb_ms}ms  Flex={fx_ms}ms  Ours={ou_ms}ms  gain={gn}%  ({elapsed:.1f}s)")
                cells.append(cell)
            except Exception as exc:
                print(f"ERROR: {exc}")
                import traceback; traceback.print_exc()

    agg = aggregate(cells)
    print_results(agg, cells)

    # Save
    out = {
        "config": vars(args),
        "aggregated": agg,
        "cells": cells,
        "router_crossovers": router.overhead_crossover_lengths(),
    }
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\n✓ Saved → {out_path}")


if __name__ == "__main__":
    main()
