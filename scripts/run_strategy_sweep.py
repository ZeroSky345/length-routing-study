#!/usr/bin/env python3
"""
Selection Strategy Sweep
========================
Benchmarks all mask-based selection strategies (GlobalSinkWindow, SqrtWindow,
BlockKNormTopK, CoverageTarget, SampledAttentionTopK) across a range of
sequence lengths and compares them against:
  * PBS-Attn (fixed params: threshold=0.9, segment_size=256)
  * Flash reference (dense SDPA, used for accuracy baseline only)

For each (strategy × length) cell, we measure:
  - t_select_ms  : wall-clock selection time (mask generation)
  - active_frac  : fraction of key blocks selected (1 - sparsity)
  - t_kernel_est : estimated kernel time = t_flash × active_frac
  - t_total_est  : t_select + t_kernel_est  (theoretical end-to-end)
  - mse          : mean-square error vs dense Flash output
  - kl           : KL divergence vs dense Flash logits
  - passed       : mse ≤ threshold AND kl ≤ threshold

NOTE: The mask-based strategies here generate boolean masks and run
      attention via PyTorch SDPA (not a real block-sparse Triton kernel).
      t_kernel_est is therefore a THEORETICAL projection, not a measured one.
      The key research value is in comparing t_select_ms and sparsity across
      strategies and lengths.

Usage
-----
  python scripts/run_strategy_sweep.py \
    --lengths 4096 8192 16384 32768 \
    --cache-dir data/cache_128d \
    --repeats 5 --output results/strategy_sweep.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import torch

from length_routing_study.empirical_sweep import (
    EmpiricalRecord,
    PBSConfig,
    load_qkv,
    run_flash,
    run_pbs,
    _cuda_ms,
    _stats,
    _kl,
)
from length_routing_study.selection_strategies import (
    SelectionStrategy,
    SelectionResult,
    run_block_sparse_attention,
    default_strategies,
    overhead_class,
    GlobalSinkWindow,
    SqrtWindow,
    BlockKNormTopK,
    CoverageTarget,
    SampledAttentionTopK,
)
from length_routing_study.length_router import LengthAwareRouter


# ─── Result type ──────────────────────────────────────────────────────────────

@dataclass
class StrategyRecord:
    strategy:      str
    overhead_cls:  str
    seq_len:       int
    # Selection timing
    t_select_ms:   float
    t_select_std:  float
    # Sparsity
    active_frac:   float
    sparsity:      float
    # Kernel + total (theoretical projections)
    t_flash_ref_ms: float
    t_kernel_est_ms: float   # t_flash_ref × active_frac
    t_total_est_ms:  float   # t_select + t_kernel_est
    # Accuracy vs Flash reference
    mse:           float
    kl:            float
    passed_mse:    bool
    passed_kl:     bool
    # Router's routing decision at this L
    router_decision: str     # "structural" | "pbs" | "knorm" | "flex"
    router_strategy: str
    meta:          dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.passed_mse and self.passed_kl

    def as_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items()}
        d["passed"] = self.passed
        return d


# ─── Core measurement ─────────────────────────────────────────────────────────

def measure_strategy(
    strategy: SelectionStrategy,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_size: int,
    flash_ref: torch.Tensor,
    flash_time_ms: float,
    repeats: int = 5,
    mse_threshold: float = 0.02,
    kl_threshold:  float = 0.10,
) -> StrategyRecord:
    """
    Measure one (strategy × length) cell.

    Selection is timed ``repeats`` times (CPU-side, includes GPU sync).
    Attention is run once for accuracy (SDPA with mask bias).
    """
    B, H, L, D = q.shape
    ohcls = overhead_class(strategy)

    # ── Time the selection (repeats) ──────────────────────────────────────
    sel_times: list[float] = []
    last_result: SelectionResult | None = None
    for _ in range(repeats):
        torch.cuda.synchronize()
        r = strategy.select(q, k, v, block_size=block_size)
        torch.cuda.synchronize()
        sel_times.append(r.t_select_ms)
        last_result = r
    assert last_result is not None

    t_sel_mean = float(sum(sel_times) / len(sel_times))
    t_sel_std  = float(
        (sum((x - t_sel_mean) ** 2 for x in sel_times) / len(sel_times)) ** 0.5
    )

    # ── Compute accuracy (single forward pass with the mask) ──────────────
    with torch.inference_mode():
        out = run_block_sparse_attention(
            q, k, v,
            block_mask=last_result.block_mask,
            block_size=block_size,
            causal=True,
        )

    mse = float(torch.mean((out.float() - flash_ref.float()) ** 2).item())
    kl  = _kl(out, flash_ref)

    # ── Theoretical projections ───────────────────────────────────────────
    active_frac      = last_result.active_fraction
    t_kernel_est     = flash_time_ms * active_frac
    t_total_est      = t_sel_mean + t_kernel_est

    # ── Ours routing decision at this L ──────────────────────────────────
    router = LengthAwareRouter()
    rd     = router.route_full(L, sparsity=last_result.sparsity)

    return StrategyRecord(
        strategy=strategy.name,
        overhead_cls=ohcls,
        seq_len=L,
        t_select_ms=round(t_sel_mean, 4),
        t_select_std=round(t_sel_std, 4),
        active_frac=round(active_frac, 4),
        sparsity=round(last_result.sparsity, 4),
        t_flash_ref_ms=round(flash_time_ms, 4),
        t_kernel_est_ms=round(t_kernel_est, 4),
        t_total_est_ms=round(t_total_est, 4),
        mse=round(mse, 6),
        kl=round(kl, 6),
        passed_mse=(mse <= mse_threshold),
        passed_kl=(kl <= kl_threshold),
        router_decision=f"ours:{rd.backend}",   # prefixed with "ours:" for clarity
        router_strategy=rd.params.get("strategy", rd.backend),
        meta=last_result.meta,
    )


# ─── Summary helpers ──────────────────────────────────────────────────────────

def print_summary_table(records: list[StrategyRecord]) -> None:
    """Print a compact pivot table: rows = strategy, cols = seq_len."""
    lengths = sorted(set(r.seq_len for r in records))
    strategies = list(dict.fromkeys(r.strategy for r in records))

    # Headers
    col_w = 14
    hdr = f"{'Strategy':<28} {'Class':<12}" + "".join(
        f"{'L='+str(L):<{col_w}}" for L in lengths
    )
    print("\n" + "=" * len(hdr))
    print(hdr)
    print("─" * len(hdr))

    # t_select rows
    print("  [t_select_ms]")
    for s in strategies:
        row = f"  {s:<26} {overhead_class_of(s, records):<12}"
        for L in lengths:
            cell = next((r for r in records if r.strategy == s and r.seq_len == L), None)
            row += f"  {cell.t_select_ms:>6.3f} ms   " if cell else f"  {'—':>10}   "
        print(row)

    print("─" * len(hdr))
    # active_frac rows
    print("  [active_fraction]")
    for s in strategies:
        row = f"  {s:<26} {overhead_class_of(s, records):<12}"
        for L in lengths:
            cell = next((r for r in records if r.strategy == s and r.seq_len == L), None)
            row += f"  {cell.active_frac:>8.3f}   " if cell else f"  {'—':>10}   "
        print(row)

    print("─" * len(hdr))
    # t_total_est rows  (t_select + dense_ref × active_frac — theoretical projection)
    print("  [t_total_est_ms  (t_select + dense_ref_ms × active_frac)]")
    for s in strategies:
        row = f"  {s:<26} {overhead_class_of(s, records):<12}"
        for L in lengths:
            cell = next((r for r in records if r.strategy == s and r.seq_len == L), None)
            row += f"  {cell.t_total_est_ms:>8.3f}   " if cell else f"  {'—':>10}   "
        print(row)

    print("─" * len(hdr))
    # MSE rows  (vs dense SDPA reference — internal use only)
    print("  [MSE vs dense reference]")
    for s in strategies:
        row = f"  {s:<26} {overhead_class_of(s, records):<12}"
        for L in lengths:
            cell = next((r for r in records if r.strategy == s and r.seq_len == L), None)
            if cell:
                ok = "✓" if cell.passed else "✗"
                row += f"  {ok}{cell.mse:>7.5f}   "
            else:
                row += f"  {'—':>10}   "
        print(row)

    print("=" * len(hdr))


def overhead_class_of(strategy_name: str, records: list[StrategyRecord]) -> str:
    for r in records:
        if r.strategy == strategy_name:
            return r.overhead_cls
    return "?"


def print_crossover_analysis(router: LengthAwareRouter) -> None:
    """Print the overhead crossover lengths."""
    crossovers = router.overhead_crossover_lengths()
    print("\n── Overhead Crossover Analysis ──────────────────────────────────────")
    for key, info in crossovers.items():
        L = info.get("crossover_L")
        L_str = f"L={L:,}" if L else "never"
        print(f"  {info['strategy_a']:<35} vs {info['strategy_b']:<20} crossover at {L_str}")
        print(f"    Note: {info['note']}")
    print()


# ─── Also compare with PBS ────────────────────────────────────────────────────

def measure_pbs_baseline(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    flash_ref: torch.Tensor,
    warmup: int = 2,
    repeats: int = 5,
    mse_threshold: float = 0.02,
    kl_threshold:  float = 0.10,
) -> EmpiricalRecord:
    """Run PBS with default fixed params for comparison."""
    cfg = PBSConfig(threshold=0.9, segment_size=256, block_size=128,
                    use_triton=True, force_first=True)
    return run_pbs(q, k, v, cfg, reference=flash_ref,
                   warmup=warmup, repeats=repeats,
                   mse_threshold=mse_threshold, kl_threshold=kl_threshold)


# ─── Main ────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lengths",   type=int, nargs="+",
                   default=[4096, 8192, 16384, 32768])
    p.add_argument("--cache-dir", type=str, default="data/cache_128d")
    p.add_argument("--seed",      type=int, default=42)
    p.add_argument("--device",    type=str, default="cuda")
    p.add_argument("--warmup",    type=int, default=3)
    p.add_argument("--repeats",   type=int, default=5)
    p.add_argument("--block-size", type=int, default=128)
    p.add_argument("--mse-threshold", type=float, default=0.02)
    p.add_argument("--kl-threshold",  type=float, default=0.10)
    p.add_argument("--output",    type=str, default=None,
                   help="JSON output file path (default: results/strategy_sweep_<ts>.json)")
    p.add_argument("--no-pbs",    action="store_true",
                   help="Skip PBS baseline measurement")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"

    out_dir = ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.output:
        out_path = Path(args.output)
    else:
        ts = int(time.time())
        out_path = out_dir / f"strategy_sweep_{ts}.json"

    strategies = default_strategies(block_size=args.block_size)
    router     = LengthAwareRouter()

    print("Selection Strategy Sweep")
    print(f"  Lengths  : {args.lengths}")
    print(f"  Strategies: {[s.name for s in strategies]}")
    print(f"  Device   : {device}")
    print(f"  Cache dir: {args.cache_dir}")

    # Print theoretical crossover analysis
    print_crossover_analysis(router)

    all_records:     list[StrategyRecord]  = []
    pbs_records:     list[EmpiricalRecord] = []
    flash_times:     dict[int, float]      = {}

    for L in args.lengths:
        print(f"\n{'═'*70}")
        print(f"  L = {L:,}")
        print(f"{'─'*70}")

        q, k, v = load_qkv(
            args.cache_dir, L,
            seed=args.seed, device=device,
        )
        # Ensure consistent dtype (cached files may have mixed dtypes)
        dtype = q.dtype
        k = k.to(dtype=dtype)
        v = v.to(dtype=dtype)

        # ── Flash reference ────────────────────────────────────────────────
        flash_rec = run_flash(q, k, v, warmup=args.warmup, repeats=args.repeats)
        flash_time = flash_rec.t_mean_ms
        flash_times[L] = flash_time

        with torch.inference_mode():
            import torch.nn.functional as F
            flash_ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        # Flash is internal only (accuracy ref + PBS decomposition); not printed in speed output

        # ── PBS baseline ───────────────────────────────────────────────────
        if not args.no_pbs:
            try:
                pbs_rec = measure_pbs_baseline(
                    q, k, v, flash_ref,
                    warmup=args.warmup, repeats=args.repeats,
                    mse_threshold=args.mse_threshold, kl_threshold=args.kl_threshold,
                )
                pbs_records.append(pbs_rec)
                print(f"  PBS fixed: {pbs_rec.t_mean_ms:.3f} ms  "
                      f"mse={pbs_rec.mse:.5f}  {'OK' if pbs_rec.passed else 'FAIL'}")
            except Exception as exc:
                print(f"  PBS failed: {exc}")

        # ── Strategy sweep ─────────────────────────────────────────────────
        for strat in strategies:
            try:
                rec = measure_strategy(
                    strat, q, k, v,
                    block_size=args.block_size,
                    flash_ref=flash_ref,
                    flash_time_ms=flash_time,
                    repeats=args.repeats,
                    mse_threshold=args.mse_threshold,
                    kl_threshold=args.kl_threshold,
                )
                all_records.append(rec)
                ok  = "✓" if rec.passed else "✗"
                print(f"  {strat.name:<35} t_sel={rec.t_select_ms:>6.3f}ms  "
                      f"active={rec.active_frac:.2f}  t_est={rec.t_total_est_ms:>7.3f}ms  "
                      f"mse={rec.mse:.5f} {ok}")
            except Exception as exc:
                import traceback
                print(f"  {strat.name}: ERROR — {exc}")
                traceback.print_exc()

    # ── Print summary table ────────────────────────────────────────────────
    print_summary_table(all_records)

    # ── Print Ours routing table ───────────────────────────────────────────
    print("\n── Ours Routing Decisions (route_full) ────────────────────────────")
    print(f"  {'L':>8}  {'Sparsity':>10}  {'Ours Decision':>15}  {'t_est':>10}")
    print(f"  {'─'*8}  {'─'*10}  {'─'*15}  {'─'*10}")
    for L in args.lengths:
        for s in [0.0, 0.30, 0.60, 0.80]:
            rd = router.route_full(L, sparsity=s)
            t_pbs_est = router._t_pbs_total(L, s)
            t_struct_act = max(0.30, min(0.70, (1.0 - s) * 1.1))
            t_struct_est = router._t_struct_total(L, t_struct_act)
            print(f"  {L:>8,}  {s:>10.2f}  {rd.backend:>15}  "
                  f"t_pbs={t_pbs_est:.2f}ms  t_struct={t_struct_est:.2f}ms")

    # ── Save output ────────────────────────────────────────────────────────
    output = {
        "config": {
            "lengths": args.lengths,
            "device":  device,
            "repeats": args.repeats,
            "block_size": args.block_size,
            "mse_threshold": args.mse_threshold,
            "kl_threshold":  args.kl_threshold,
        },
        "flash_times":  {str(L): t for L, t in flash_times.items()},
        "pbs_baseline": [r.as_dict() for r in pbs_records] if pbs_records else [],
        "strategy_records": [r.as_dict() for r in all_records],
        "overhead_crossovers": router.overhead_crossover_lengths(),
        "routing_table": router.build_routing_table(args.lengths),
    }
    out_path.write_text(json.dumps(output, indent=2, default=str))
    print(f"\n✓ Results saved → {out_path}")


if __name__ == "__main__":
    main()
