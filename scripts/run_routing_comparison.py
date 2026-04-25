#!/usr/bin/env python3
"""
Routing comparison: fixed sparse strategies vs length-aware router (Ours).

Flash is NOT included — we assume a sparse method is required.
Compares three strategies on each (scenario, length) combination:
  1. pbs_fixed  — always use PBS(thr=0.9, seg=256)  [naive default]
  2. flex_fixed — always use FlexPrefill(γ=0.90, τ=0.05) [naive default]
  3. ours       — LengthAwareRouter: picks PBS or Flex with adaptive params

For each strategy, records:
  - t_total_ms  : end-to-end latency
  - t_select_ms : selection/permutation overhead
  - t_kernel_ms : sparse attention kernel time
  - mse         : accuracy vs dense reference

Usage
-----
  python scripts/run_routing_comparison.py \
      --lengths 4096 8192 16384 32768 65536 \
      --scenarios sparse_high sparse_med code structured \
      --output results/routing_comparison.json

  # Theory-only (no GPU):
  python scripts/run_routing_comparison.py --theory-only \
      --output results/routing_comparison_theory.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import torch

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from length_routing_study._paths import ensure_external_paths
ensure_external_paths(include_baseline=False)

from length_routing_study.empirical_sweep import (
    FlexConfig, PBSConfig,
    load_qkv, run_flash, run_pbs, run_flex, run_pbs_decomposed,
)
from length_routing_study.sparsity_estimator import estimate_sparsity
from length_routing_study.length_router import LengthAwareRouter


# ── Fixed strategy configs ────────────────────────────────────────────────────

PBS_FIXED  = PBSConfig(threshold=0.9, segment_size=256, block_size=128,
                        use_triton=True, force_first=True)
FLEX_FIXED = FlexConfig(gamma=0.90, tau=0.05, min_budget_frac=0.0, block_size=128)


# ── Theory-only cost model ────────────────────────────────────────────────────

def theory_estimate(
    strategy: str,
    L: int,
    sparsity: float,
    router: LengthAwareRouter,
) -> dict:
    """Estimate strategy latency from the analytical cost model (no GPU needed)."""
    t_flash = router._t_flash(L)

    if strategy == "pbs_fixed":
        t_sel = router._t_select_pbs(L)
        af    = max(0.05, 1.0 - sparsity)
        t_ker = t_flash * af
        return {
            "strategy": strategy, "backend": "pbs",
            "params": PBS_FIXED.as_dict(),
            "t_total_ms": round(t_sel + t_ker, 3),
            "t_select_ms": round(t_sel, 3),
            "t_kernel_ms": round(t_ker, 3),
            "mse": None, "kl": None, "passed": None,
        }

    if strategy == "flex_fixed":
        t_sel = router._t_select_flex(L)
        af    = max(0.05, 1.0 - sparsity)
        t_ker = t_flash * af
        return {
            "strategy": strategy, "backend": "flexprefill",
            "params": FLEX_FIXED.as_dict(),
            "t_total_ms": round(t_sel + t_ker, 3),
            "t_select_ms": round(t_sel, 3),
            "t_kernel_ms": round(t_ker, 3),
            "mse": None, "kl": None, "passed": None,
        }

    if strategy == "ours":
        decision = router.route(L, sparsity=sparsity)
        b = decision.backend
        if b == "pbs":
            t_sel = router._t_select_pbs(L)
            t_ker = t_flash * max(0.05, 1.0 - sparsity) * 0.95
        else:
            t_sel = router._t_select_flex(L)
            t_ker = t_flash * max(0.05, 1.0 - sparsity) * 0.90
        return {
            "strategy": strategy, "backend": b,
            "params": decision.params,
            "decision_reason": decision.reason,
            "t_total_ms": round(t_sel + t_ker, 3),
            "t_select_ms": round(t_sel, 3),
            "t_kernel_ms": round(t_ker, 3),
            "mse": None, "kl": None, "passed": None,
        }

    raise ValueError(f"Unknown strategy: {strategy}")


# ── One (scenario, length) cell ───────────────────────────────────────────────

def run_cell(
    scenario: str,
    L: int,
    cache_dir: Path,
    seed: int,
    router: LengthAwareRouter,
    warmup: int,
    repeats: int,
    device: str,
    theory_only: bool,
) -> dict:
    """Run all four strategies for one (scenario, L) and return results."""

    # ── Sparsity estimation from QKV ─────────────────────────────────────
    q, k, v = load_qkv(cache_dir, L, prompt_family=scenario, seed=seed, device=device)
    sp = estimate_sparsity(q, k, block_size=128, sample_rows=64, seed=seed)
    sparsity = sp.estimated_sparsity_ratio
    kv_cv    = sp.kv_norm_cv

    # Use the pre-calibrated router (calibration done once in main)
    router_calibrated = router
    flash_ms = None

    if theory_only:
        strategies = {}
        for strat in ("pbs_fixed", "flex_fixed", "ours"):
            strategies[strat] = theory_estimate(strat, L, sparsity, router_calibrated)
        del q, k, v
        return {
            "scenario": scenario, "seq_len": L,
            "sparsity_ratio": round(sparsity, 4),
            "kv_norm_cv": round(kv_cv, 4),
            "block_sparsity": round(sp.block_sparsity_ratio or sparsity, 4),
            "strategies": strategies,
            "router_decision": router_calibrated.route(L, sparsity).summary(),
            "t_pbs_sel_est":  round(router_calibrated._t_select_pbs(L), 3),
            "t_flex_sel_est": round(router_calibrated._t_select_flex(L), 3),
        }

    # ── Dense reference (only for MSE computation, not reported as a strategy) ──
    import torch.nn.functional as F
    with torch.inference_mode():
        ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)

    # Need flash time to decompose PBS timing
    flash_rec = run_flash(q, k, v, warmup=warmup, repeats=repeats)
    flash_ms  = flash_rec.t_mean_ms

    strategies: dict[str, dict] = {}

    # ── PBS fixed (with decomposed timing) ────────────────────────────────
    try:
        pbs_rec = run_pbs_decomposed(
            q, k, v, PBS_FIXED,
            flash_time_ms=flash_ms,
            reference=ref,
            warmup=warmup, repeats=repeats,
        )
        strategies["pbs_fixed"] = {
            "strategy": "pbs_fixed",
            "backend": "pbs", "params": PBS_FIXED.as_dict(),
            "t_total_ms":  round(pbs_rec.t_mean_ms, 3),
            "t_select_ms": round(pbs_rec.t_select_ms, 3),
            "t_kernel_ms": round(pbs_rec.t_kernel_ms, 3),
            "mse": round(pbs_rec.mse, 6), "kl": round(pbs_rec.kl, 6),
            "passed": pbs_rec.passed,
        }
    except Exception as exc:
        warnings.warn(f"PBS fixed @ L={L}: {exc}")

    # ── Flex fixed ────────────────────────────────────────────────────────
    try:
        flex_rec = run_flex(q, k, v, FLEX_FIXED, reference=ref,
                            warmup=warmup, repeats=repeats)
        strategies["flex_fixed"] = {
            "strategy": "flex_fixed",
            "backend": "flexprefill", "params": FLEX_FIXED.as_dict(),
            "t_total_ms":  round(flex_rec.t_mean_ms, 3),
            "t_select_ms": 0.0,
            "t_kernel_ms": round(flex_rec.t_mean_ms, 3),
            "mse": round(flex_rec.mse, 6), "kl": round(flex_rec.kl, 6),
            "passed": flex_rec.passed,
        }
    except Exception as exc:
        warnings.warn(f"Flex fixed @ L={L}: {exc}")

    # ── Ours: adaptive router (PBS or Flex with tuned params) ─────────────
    decision = router_calibrated.route(L, sparsity=sparsity)
    router_backend = decision.backend

    if router_backend == "pbs":
        pbs_cfg = PBSConfig(**{
            k_: v_ for k_, v_ in decision.params.items()
            if k_ in ("threshold", "segment_size", "block_size", "use_triton", "force_first")
        })
        try:
            r = run_pbs_decomposed(
                q, k, v, pbs_cfg,
                flash_time_ms=flash_ms,
                reference=ref, warmup=warmup, repeats=repeats,
            )
            strategies["ours"] = {
                "strategy": "ours",
                "backend": "pbs", "params": pbs_cfg.as_dict(),
                "decision_reason": decision.reason,
                "t_total_ms":  round(r.t_mean_ms, 3),
                "t_select_ms": round(r.t_select_ms, 3),
                "t_kernel_ms": round(r.t_kernel_ms, 3),
                "mse": round(r.mse, 6), "kl": round(r.kl, 6),
                "passed": r.passed,
            }
        except Exception as exc:
            warnings.warn(f"Ours-PBS @ L={L}: {exc}")
            strategies["ours"] = {**strategies.get("pbs_fixed", {}),
                                   "strategy": "ours",
                                   "decision_reason": decision.reason}
    else:   # flex
        flex_cfg = FlexConfig(**{
            k_: v_ for k_, v_ in decision.params.items()
            if k_ in ("gamma", "tau", "min_budget_frac", "block_size")
        })
        try:
            r = run_flex(q, k, v, flex_cfg, reference=ref,
                         warmup=warmup, repeats=repeats)
            strategies["ours"] = {
                "strategy": "ours",
                "backend": "flexprefill", "params": flex_cfg.as_dict(),
                "decision_reason": decision.reason,
                "t_total_ms":  round(r.t_mean_ms, 3),
                "t_select_ms": 0.0,
                "t_kernel_ms": round(r.t_mean_ms, 3),
                "mse": round(r.mse, 6), "kl": round(r.kl, 6),
                "passed": r.passed,
            }
        except Exception as exc:
            warnings.warn(f"Ours-Flex @ L={L}: {exc}")
            strategies["ours"] = {**strategies.get("flex_fixed", {}),
                                   "strategy": "ours",
                                   "decision_reason": decision.reason}

    del q, k, v, ref
    torch.cuda.empty_cache()

    return {
        "scenario": scenario, "seq_len": L,
        "sparsity_ratio": round(sparsity, 4),
        "kv_norm_cv": round(kv_cv, 4),
        "block_sparsity": round(sp.block_sparsity_ratio or sparsity, 4),
        "strategies": strategies,
        "router_decision": decision.summary(),
        "t_pbs_sel_est":  round(router_calibrated._t_select_pbs(L), 3),
        "t_flex_sel_est": round(router_calibrated._t_select_flex(L), 3),
    }


# ── Aggregation ───────────────────────────────────────────────────────────────

def aggregate(cells: list[dict]) -> list[dict]:
    """Average per-strategy latency across scenarios at each length."""
    from collections import defaultdict

    by_len: dict[int, list[dict]] = defaultdict(list)
    for c in cells:
        by_len[c["seq_len"]].append(c)

    rows = []
    for L in sorted(by_len.keys()):
        group = by_len[L]

        def avg_strat(strat: str, field: str) -> float | None:
            vals = [c["strategies"][strat][field]
                    for c in group
                    if strat in c["strategies"] and c["strategies"][strat].get(field) is not None]
            return round(sum(vals) / len(vals), 4) if vals else None

        def pct_gain(base_strat: str, comp_strat: str) -> float | None:
            b = avg_strat(base_strat, "t_total_ms")
            c = avg_strat(comp_strat, "t_total_ms")
            if b and c and b > 0:
                return round((b - c) / b * 100, 2)
            return None

        row = {
            "seq_len": L,
            "n_scenarios": len(group),
            "sparsity_avg": round(sum(c["sparsity_ratio"] for c in group) / len(group), 4),
            "sparsity_by_scenario": {c["scenario"]: c["sparsity_ratio"] for c in group},
        }
        for strat in ("pbs_fixed", "flex_fixed", "ours"):
            row[f"{strat}_t_ms"]      = avg_strat(strat, "t_total_ms")
            row[f"{strat}_t_sel_ms"]  = avg_strat(strat, "t_select_ms")
            row[f"{strat}_t_ker_ms"]  = avg_strat(strat, "t_kernel_ms")
            row[f"{strat}_mse"]       = avg_strat(strat, "mse")

        row["gain_vs_pbs_pct"]  = pct_gain("pbs_fixed",  "ours")
        row["gain_vs_flex_pct"] = pct_gain("flex_fixed", "ours")
        row["ours_decisions"]   = [c["router_decision"] for c in group]

        rows.append(row)
    return rows


# ── Main ─────────────────────────────────────────────────────────────────────

def calibrate_router(
    router: LengthAwareRouter,
    lengths: list[int],
    cache_dir: Path,
    seed: int,
    device: str,
    warmup: int,
    repeats: int,
) -> LengthAwareRouter:
    """
    Quick calibration pass: measure Flash, PBS_fixed, Flex_fixed at each length
    using ONE (averaged) scenario to fit the router's cost model coefficients.
    Uses 'structured' scenario if available, else first available.
    """
    print("=== Calibrating router from measurements ===")
    flash_ms_list: list[tuple[int, float]] = []
    pbs_ms_list:   list[tuple[int, float]] = []
    pbs_sel_list:  list[tuple[int, float]] = []
    flex_ms_list:  list[tuple[int, float]] = []

    cal_scenario = "structured"
    import torch, torch.nn.functional as F, warnings
    for L in lengths:
        try:
            q, k, v = load_qkv(cache_dir, L, prompt_family=cal_scenario,
                                seed=seed, device=device)
            flash_rec = run_flash(q, k, v, warmup=warmup, repeats=repeats)
            flash_t   = flash_rec.t_mean_ms
            flash_ms_list.append((L, flash_t))

            with torch.inference_mode():
                ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)

            pbs_rec = run_pbs_decomposed(q, k, v, PBS_FIXED,
                                          flash_time_ms=flash_t, reference=ref,
                                          warmup=warmup, repeats=repeats)
            pbs_ms_list.append((L, pbs_rec.t_mean_ms))
            pbs_sel_list.append((L, pbs_rec.t_select_ms))

            try:
                flex_rec = run_flex(q, k, v, FLEX_FIXED, reference=ref,
                                    warmup=warmup, repeats=repeats)
                flex_ms_list.append((L, flex_rec.t_mean_ms))
            except Exception:
                pass

            del q, k, v, ref
            torch.cuda.empty_cache()
            print(f"  L={L:>6}  [ref={flash_t:.2f}ms]  pbs={pbs_rec.t_mean_ms:.2f}ms"
                  f"(sel={pbs_rec.t_select_ms:.2f})  flex={flex_ms_list[-1][1]:.2f}ms"
                  if flex_ms_list and flex_ms_list[-1][0] == L else
                  f"  L={L:>6}  [ref={flash_t:.2f}ms]  pbs={pbs_rec.t_mean_ms:.2f}ms"
                  f"(sel={pbs_rec.t_select_ms:.2f})")
        except Exception as exc:
            warnings.warn(f"Calibration failed at L={L}: {exc}")

    cal_router = router.calibrate_from_full_measurements(
        flash_ms=flash_ms_list,
        pbs_total_ms=pbs_ms_list,
        pbs_select_ms=pbs_sel_list,
        flex_total_ms=flex_ms_list,
    )
    print(f"Calibrated (internal cost model): ref_alpha={cal_router.flash_alpha:.3e}, "
          f"pbs_sel_a={cal_router.pbs_select_a:.3e}, "
          f"pbs_sel_b={cal_router.pbs_select_b:.3f}, "
          f"flex_sel_a={cal_router.flex_select_a:.3e}, "
          f"flex_sel_b={cal_router.flex_select_b_val:.3f}\n")
    return cal_router


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lengths",   nargs="+", type=int,
                    default=[4096, 8192, 16384, 32768, 65536])
    ap.add_argument("--scenarios", nargs="+",
                    default=["sparse_low", "sparse_med", "sparse_high",
                             "code", "structured"])
    ap.add_argument("--cache-dir", type=Path, default=Path("data/cache"))
    ap.add_argument("--seed",      type=int, default=42)
    ap.add_argument("--device",    default="cuda")
    ap.add_argument("--warmup",    type=int, default=2)
    ap.add_argument("--repeats",   type=int, default=5)
    ap.add_argument("--theory-only", action="store_true",
                    help="Use cost model only, skip GPU kernel measurements.")
    ap.add_argument("--skip-calibration", action="store_true",
                    help="Skip the pre-calibration pass (use default model params).")
    ap.add_argument("--output",    type=Path,
                    default=Path("results/routing_comparison.json"))
    args = ap.parse_args()

    router = LengthAwareRouter()

    # ── Pre-calibration pass ──────────────────────────────────────────────
    if not args.theory_only and not args.skip_calibration:
        router = calibrate_router(
            router,
            lengths=args.lengths,
            cache_dir=args.cache_dir,
            seed=args.seed,
            device=args.device,
            warmup=args.warmup,
            repeats=args.repeats,
        )

    total = len(args.scenarios) * len(args.lengths)
    cells: list[dict] = []
    n = 0

    print(f"Running routing comparison: {len(args.scenarios)} scenarios × "
          f"{len(args.lengths)} lengths = {total} cells")
    print(f"Theory-only: {args.theory_only}\n")

    for scenario in args.scenarios:
        for L in args.lengths:
            n += 1
            t0 = time.time()
            print(f"[{n:>3}/{total}] {scenario:>12}  L={L:>6}", end="  ", flush=True)
            try:
                cell = run_cell(
                    scenario=scenario, L=L,
                    cache_dir=args.cache_dir,
                    seed=args.seed,
                    router=router,
                    warmup=args.warmup,
                    repeats=args.repeats,
                    device=args.device,
                    theory_only=args.theory_only,
                )
                s = cell["sparsity_ratio"]
                rd = cell.get("router_decision", "?")[:60]
                elapsed = time.time() - t0
                print(f"sparsity={s:.3f}  {rd}  ({elapsed:.1f}s)")
                cells.append(cell)
            except Exception as exc:
                print(f"ERROR: {exc}")
                import traceback; traceback.print_exc()

    agg = aggregate(cells)

    output = {
        "theory_only": args.theory_only,
        "scenarios": args.scenarios,
        "lengths": args.lengths,
        "cells": cells,
        "aggregated": agg,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved → {args.output}")

    # ── Summary table ─────────────────────────────────────────────────────
    print("\n── Sparse routing comparison summary ────────────────────────────────")
    print(f"{'L':>7}  {'sparsity':>9}  {'pbs_fix':>9}  {'flex_fix':>9}  "
          f"{'ours':>8}  {'gain%_pbs':>10}  {'gain%_flex':>11}")
    print("─" * 72)
    for row in agg:
        L   = row["seq_len"]
        sp  = row.get("sparsity_avg", 0)
        pb  = row.get("pbs_fixed_t_ms")
        fx  = row.get("flex_fixed_t_ms")
        rt  = row.get("ours_t_ms")
        gp  = row.get("gain_vs_pbs_pct")
        gf  = row.get("gain_vs_flex_pct")

        def fmt(v): return f"{v:>9.2f}" if v is not None else f"{'N/A':>9}"
        def fmtg(v): return f"{v:>+10.1f}%" if v is not None else f"{'N/A':>10}"

        print(f"{L:>7}  {sp:>9.3f}  {fmt(pb)}  {fmt(fx)}  {fmt(rt)}"
              f"  {fmtg(gp)}  {fmtg(gf)}")


if __name__ == "__main__":
    main()
