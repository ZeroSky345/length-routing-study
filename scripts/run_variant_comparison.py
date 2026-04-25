#!/usr/bin/env python3
"""
Variant comparison: fixed-parameter PBS/Flex vs. self-adaptive PBS/Flex vs. Ours.

Configurations tested
---------------------
PBS variants (3 fixed + 1 adaptive):
  PBS-Conservative  : threshold=0.95, segment_size=128  -- ultra-safe, slow
  PBS-Default       : threshold=0.90, segment_size=256  -- current baseline
  PBS-Aggressive    : threshold=0.80, segment_size=512  -- fast, trades accuracy
  PBS-Adaptive      : threshold=0.90, segment_size=auto(L)
                      seg=128 (L≤8K) / 256 (L≤32K) / 512 (L>32K)
                      Simulates PBS's "own" length-aware strategy.

Flex variants (3 fixed + 1 adaptive):
  Flex-Tight        : gamma=0.99, tau=0.01  -- near-lossless, slow
  Flex-Default      : gamma=0.95, tau=0.10  -- current baseline
  Flex-Loose        : gamma=0.85, tau=0.20  -- aggressive pruning
  Flex-Adaptive     : gamma=auto(sparsity), tau=0.05
                      gamma=0.90 (sp>0.6) / 0.95 (sp>0.4) / 0.99 (sp≤0.4)
                      Simulates FlexPrefill's own sparsity-sensing strategy.

Ours              : adaptive router from v8b (already computed, loaded from JSON).

Output: results/variant_comparison.json + printed tables.
"""
from __future__ import annotations

import argparse, json, sys, time, warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(_SRC))

from length_routing_study.empirical_sweep import (
    PBSConfig, FlexConfig, load_qkv, run_flash, run_pbs_decomposed, run_flex, _cuda_ms, _stats,
)
from length_routing_study.sparsity_estimator import estimate_sparsity

# ── Configuration ──────────────────────────────────────────────────────────────

LENGTHS   = [4096, 8192, 16384, 32768, 65536, 131072]
SCENARIOS = [
    "sparse_low", "sparse_med", "sparse_high",
    "lm_sink_local", "lm_sparse_global", "lm_hierarchical",
    "lm_local_periodic", "lm_mixed",
]

REPEATS = 3
WARMUP  = 2
SEED    = 42

# PBS fixed variants
PBS_CONSERVATIVE = PBSConfig(threshold=0.95, segment_size=128)
PBS_DEFAULT      = PBSConfig(threshold=0.90, segment_size=256)   # existing baseline
PBS_AGGRESSIVE   = PBSConfig(threshold=0.80, segment_size=512)

# Flex fixed variants
FLEX_TIGHT   = FlexConfig(gamma=0.99, tau=0.01,  min_budget_frac=0.0, block_size=128)
FLEX_DEFAULT = FlexConfig(gamma=0.95, tau=0.10,  min_budget_frac=0.0, block_size=128)  # existing
FLEX_LOOSE   = FlexConfig(gamma=0.85, tau=0.20,  min_budget_frac=0.0, block_size=128)


def pbs_adaptive_config(L: int) -> PBSConfig:
    """PBS length-aware: threshold fixed at 0.90, segment_size auto-scaled with L."""
    if L <= 8_192:
        seg = 128
    elif L <= 32_768:
        seg = 256
    else:
        seg = 512
    return PBSConfig(threshold=0.90, segment_size=seg)


def flex_adaptive_config(sparsity: float) -> FlexConfig:
    """Flex sparsity-aware: gamma auto-selected from estimated sparsity."""
    if sparsity > 0.60:
        gamma, tau = 0.90, 0.05   # data is sparse → looser coverage ok
    elif sparsity > 0.40:
        gamma, tau = 0.95, 0.05   # moderate sparsity → standard
    else:
        gamma, tau = 0.99, 0.02   # dense data → need tight coverage
    return FlexConfig(gamma=gamma, tau=tau, min_budget_frac=0.0, block_size=128)


# ── Helper ─────────────────────────────────────────────────────────────────────

def _flash_ref(q, k, v):
    """Compute flash reference output and timing."""
    import torch.nn.functional as F
    with torch.inference_mode():
        ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    r = run_flash(q, k, v)
    return ref, r.t_mean_ms


def run_pbs_timed(q, k, v, cfg: PBSConfig, flash_ms: float,
                  warmup: int, repeats: int, ref: "torch.Tensor") -> dict:
    try:
        r = run_pbs_decomposed(q, k, v, cfg, flash_time_ms=flash_ms,
                               reference=ref, warmup=warmup, repeats=repeats)
        return {"t_ms": r.t_mean_ms, "mse_e5": r.mse * 1e5, "kl": r.kl,
                "ok": r.mse * 1e5 < 2000}
    except Exception as exc:
        warnings.warn(f"PBS error: {exc}")
        return {"t_ms": float("nan"), "mse_e5": float("nan"), "kl": float("nan"), "ok": False}


def run_flex_timed(q, k, v, cfg: FlexConfig, flash_ms: float,
                   warmup: int, repeats: int, ref: "torch.Tensor") -> dict:
    try:
        r = run_flex(q, k, v, cfg, reference=ref, warmup=warmup, repeats=repeats)
        return {"t_ms": r.t_mean_ms, "mse_e5": r.mse * 1e5, "kl": r.kl,
                "ok": r.mse * 1e5 < 2000}
    except Exception as exc:
        warnings.warn(f"Flex error: {exc}")
        return {"t_ms": float("nan"), "mse_e5": float("nan"),
                "kl": float("nan"), "ok": False}


# ── Main evaluation loop ───────────────────────────────────────────────────────

def run_all(args) -> list[dict]:
    cache_dir = args.cache_dir
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.bfloat16

    cells = []
    total = len(SCENARIOS) * len(LENGTHS)
    idx   = 0

    for sc in SCENARIOS:
        for L in LENGTHS:
            idx += 1
            t0 = time.time()
            q, k, v = load_qkv(cache_dir, L, prompt_family=sc, seed=SEED, device=device)
            q = q.to(dtype=dtype); k = k.to(dtype=dtype); v = v.to(dtype=dtype)

            # Sparsity estimate (needed for adaptive configs)
            sp = estimate_sparsity(q, k, block_size=128, sample_rows=64, seed=SEED)
            sparsity = sp.estimated_sparsity_ratio

            # Flash reference (compute once for accuracy baseline + timing)
            ref, flash_ms = _flash_ref(q, k, v)

            # ── PBS variants ─────────────────────────────────────────────────
            pbs_cons  = run_pbs_timed(q, k, v, PBS_CONSERVATIVE, flash_ms, WARMUP, REPEATS, ref)
            pbs_def   = run_pbs_timed(q, k, v, PBS_DEFAULT,      flash_ms, WARMUP, REPEATS, ref)
            pbs_agg   = run_pbs_timed(q, k, v, PBS_AGGRESSIVE,   flash_ms, WARMUP, REPEATS, ref)
            pbs_adap  = run_pbs_timed(q, k, v, pbs_adaptive_config(L), flash_ms, WARMUP, REPEATS, ref)

            # ── Flex variants ─────────────────────────────────────────────────
            flex_tight = run_flex_timed(q, k, v, FLEX_TIGHT,   flash_ms, WARMUP, REPEATS, ref)
            flex_def   = run_flex_timed(q, k, v, FLEX_DEFAULT,  flash_ms, WARMUP, REPEATS, ref)
            flex_loose = run_flex_timed(q, k, v, FLEX_LOOSE,    flash_ms, WARMUP, REPEATS, ref)
            flex_adap  = run_flex_timed(q, k, v, flex_adaptive_config(sparsity),
                                        flash_ms, WARMUP, REPEATS, ref)

            elapsed = time.time() - t0
            print(f"  [{idx:3d}/{total}] {sc:>22}  L={L:>7,}  "
                  f"sp={sparsity:.3f}  "
                  f"PBS-def={pbs_def['t_ms']:.2f}ms  "
                  f"PBS-adap={pbs_adap['t_ms']:.2f}ms  "
                  f"Flex-adap={flex_adap['t_ms']:.2f}ms  "
                  f"({elapsed:.1f}s)", flush=True)

            cells.append({
                "scenario": sc, "seq_len": L, "sparsity": sparsity,
                "pbs_conservative": pbs_cons,
                "pbs_default":      pbs_def,
                "pbs_aggressive":   pbs_agg,
                "pbs_adaptive":     pbs_adap,
                "flex_tight":       flex_tight,
                "flex_default":     flex_def,
                "flex_loose":       flex_loose,
                "flex_adaptive":    flex_adap,
            })
            del q, k, v
            torch.cuda.empty_cache()

    return cells


# ── Merge with Ours from existing JSON ────────────────────────────────────────

def load_ours(json_path: str) -> dict[tuple, dict]:
    """Return {(scenario, L): ours_metrics} from v8b results."""
    with open(json_path) as f:
        data = json.load(f)
    result = {}
    for c in data["cells"]:
        ours = c["strategies"].get("ours", {})
        t   = ours.get("t_total_ms") or ours.get("t_ms")
        mse = ours.get("mse")
        kl  = ours.get("kl")
        if t is None:
            t = float("nan")
        mse_e5 = (mse * 1e5) if mse is not None else float("nan")
        result[(c["scenario"], c["seq_len"])] = {
            "t_ms":    t,
            "mse_e5":  mse_e5,
            "kl":      kl or float("nan"),
            "ok":      mse_e5 < 2000 if mse_e5 == mse_e5 else False,
            "backend": ours.get("backend", "?"),
        }
    return result


# ── Print tables ───────────────────────────────────────────────────────────────

CONFIGS = [
    ("PBS-Conservative",  "pbs_conservative",  "PBS θ=0.95 seg=128"),
    ("PBS-Default",       "pbs_default",        "PBS θ=0.90 seg=256  ← baseline"),
    ("PBS-Aggressive",    "pbs_aggressive",     "PBS θ=0.80 seg=512"),
    ("PBS-Adaptive",      "pbs_adaptive",       "PBS θ=0.90 seg=auto(L)  ← own adaptive"),
    ("Flex-Tight",        "flex_tight",         "Flex γ=0.99 τ=0.01"),
    ("Flex-Default",      "flex_default",       "Flex γ=0.95 τ=0.10  ← baseline"),
    ("Flex-Loose",        "flex_loose",         "Flex γ=0.85 τ=0.20"),
    ("Flex-Adaptive",     "flex_adaptive",      "Flex γ=auto(sp) τ=auto  ← own adaptive"),
    ("Ours",              "_ours_",             "Length-aware router (v8b)"),
]

def print_tables(cells: list[dict], ours_lkp: dict):
    """Print speed and accuracy tables averaged over all scenarios."""
    lkp = {(c["scenario"], c["seq_len"]): c for c in cells}

    print("\n" + "═" * 105)
    print("VARIANT COMPARISON  — Speed (ms) averaged over 8 scenarios")
    print("═" * 105)
    # Header
    hdr = f"  {'配置':>24}  " + "  ".join(f"L={L//1024}K" for L in LENGTHS)
    print(hdr)
    print("─" * 105)

    for label, key, desc in CONFIGS:
        vals = []
        for L in LENGTHS:
            ts = []
            for sc in SCENARIOS:
                if key == "_ours_":
                    d = ours_lkp.get((sc, L))
                    t = d["t_ms"] if d else float("nan")
                else:
                    c = lkp.get((sc, L), {})
                    t = c.get(key, {}).get("t_ms", float("nan"))
                if t == t:  # not nan
                    ts.append(t)
            avg = sum(ts) / len(ts) if ts else float("nan")
            vals.append(f"{avg:7.1f}")
        print(f"  {label:>24}  " + "  ".join(vals))

    print("\n" + "═" * 105)
    print("VARIANT COMPARISON  — Accuracy (MSE×10⁻⁵) averaged over 8 scenarios")
    print("═" * 105)
    print(hdr)
    print("─" * 105)

    for label, key, desc in CONFIGS:
        vals = []
        for L in LENGTHS:
            ms = []
            for sc in SCENARIOS:
                if key == "_ours_":
                    d = ours_lkp.get((sc, L))
                    m = d["mse_e5"] if d else float("nan")
                else:
                    c = lkp.get((sc, L), {})
                    m = c.get(key, {}).get("mse_e5", float("nan"))
                if m == m:
                    ms.append(m)
            avg = sum(ms) / len(ms) if ms else float("nan")
            flag = "✓" if avg < 2000 else "✗"
            vals.append(f"{avg:6.0f}{flag}")
        print(f"  {label:>24}  " + "  ".join(vals))

    # ── Per-scenario breakdown at L=32K ────────────────────────────────────
    L32 = 32768
    print(f"\n\n{'─'*105}")
    print(f"Per-scenario detail at L=32K  (speed ms | MSE×10⁻⁵)")
    print(f"{'─'*105}")
    sc_hdr = f"  {'场景':>22}  " + "  ".join(f"{cfg[0]:>14}" for cfg in CONFIGS)
    print(sc_hdr)
    print("─" * 105)
    for sc in SCENARIOS:
        vals = []
        for label, key, desc in CONFIGS:
            if key == "_ours_":
                d = ours_lkp.get((sc, L32))
                t = d["t_ms"] if d else float("nan")
                m = d["mse_e5"] if d else float("nan")
            else:
                c = lkp.get((sc, L32), {})
                t = c.get(key, {}).get("t_ms", float("nan"))
                m = c.get(key, {}).get("mse_e5", float("nan"))
            flag = "✓" if m < 2000 else "✗"
            vals.append(f"{t:5.1f}/{m:5.0f}{flag}")
        print(f"  {sc:>22}  " + "  ".join(vals))
    print("\n格式: 速度ms / MSE×10⁻⁵  (✓=精度合格)")

    # ── Config legend ────────────────────────────────────────────────────────
    print(f"\n\n{'─'*105}")
    print("配置说明")
    print("─" * 105)
    for label, key, desc in CONFIGS:
        print(f"  {label:>24}  {desc}")


# ── Entry point ────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cache-dir",  default="data/cache")
    p.add_argument("--ours-json",  default="results/comprehensive_eval.json")
    p.add_argument("--out-json",   default="results/variant_comparison.json")
    return p.parse_args()


def main():
    args = parse_args()
    print("=" * 70)
    print("Variant comparison: PBS & Flex fixed/adaptive vs Ours")
    print("=" * 70)
    print(f"Scenarios : {len(SCENARIOS)}  |  Lengths: {[L//1024 for L in LENGTHS]}K")
    print(f"Variants  : 4 PBS + 4 Flex + Ours  (warmup={WARMUP}, repeats={REPEATS})\n")

    cells = run_all(args)

    # Save immediately before any post-processing
    Path(args.out_json).parent.mkdir(exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump({"cells": cells}, f, indent=2, default=float)
    print(f"\n✓ Saved → {args.out_json}")

    ours_lkp = load_ours(args.ours_json)

    print_tables(cells, ours_lkp)


if __name__ == "__main__":
    main()
