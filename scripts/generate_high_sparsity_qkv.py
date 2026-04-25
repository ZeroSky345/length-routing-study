#!/usr/bin/env python3
"""
Generate synthetic QKV data with controlled attention sparsity.

The key insight: attention sparsity is driven by the distribution of
K-token norms. If a fraction signal_frac of tokens have K-norms that
are signal_scale× larger than background tokens, then softmax will
concentrate most attention mass on those "important" tokens.

Sparsity model:
--------------
  P(attn on important token) ∝ exp(signal_scale × base_norm)
  P(attn on background token) ∝ exp(base_norm)

  For signal_scale=4, signal_frac=0.20:
    mass on 20% important tokens ≈ e^4 / (0.2 e^4 + 0.8 e^1) ≈ 0.97
    → ~80% of blocks can be pruned by PBS

Three sparsity levels are generated:
  sparse_low  : signal_frac=0.35, signal_scale=2.0 → ~50-60% block sparsity
  sparse_med  : signal_frac=0.20, signal_scale=4.0 → ~70-80% block sparsity
  sparse_high : signal_frac=0.10, signal_scale=7.0 → ~85-90% block sparsity

These complement the existing diverse scenarios (code, technical, ...) which
have natural sparsity 0.27-0.53.

Usage
-----
  python scripts/generate_high_sparsity_qkv.py \
      --lengths 4096 8192 16384 32768 65536 \
      --out-dir data/cache
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# ── Scenario definitions ─────────────────────────────────────────────────────

SCENARIOS = {
    "sparse_low": {
        "signal_frac":  0.35,
        "signal_scale": 2.0,
        "desc": "Moderate sparsity (~55% block sparsity). "
                "Mimics dialogue or repetitive technical text.",
    },
    "sparse_med": {
        "signal_frac":  0.20,
        "signal_scale": 4.0,
        "desc": "High sparsity (~75% block sparsity). "
                "Mimics structured data with strong token dependencies.",
    },
    "sparse_high": {
        "signal_frac":  0.10,
        "signal_scale": 7.0,
        "desc": "Very high sparsity (~87% block sparsity). "
                "Mimics code with local variable references or tables.",
    },
}

NUM_HEADS = 16
HEAD_DIM  = 128


def generate_qkv(
    scenario: str,
    L: int,
    seed: int = 42,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cpu",
    block_size: int = 128,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate (Q, K, V) tensors with CLUSTERED K-norm distribution.

    Real attention sparsity comes from *spatially clustered* important tokens:
    key tokens in the same semantic chunk (a code function, a paragraph, a
    dialogue turn) form contiguous high-norm regions.  Scattering signal
    tokens uniformly across all positions does NOT create block sparsity
    because every block would contain at least some signal tokens.

    Strategy: divide the sequence into blocks of `block_size`, then mark
    a fraction (signal_frac) of *entire blocks* as "signal blocks".  Within
    a signal block, all tokens get amplified K-norms.  Background blocks
    keep their original (lower) norms.

    This creates true block-level sparsity that PBS and FlexPrefill can exploit.
    """
    params = SCENARIOS[scenario]
    signal_frac  = params["signal_frac"]
    signal_scale = params["signal_scale"]

    rng = torch.Generator()
    rng.manual_seed(seed + L)

    q = torch.randn(1, NUM_HEADS, L, HEAD_DIM, generator=rng, dtype=torch.float32)
    k = torch.randn(1, NUM_HEADS, L, HEAD_DIM, generator=rng, dtype=torch.float32)
    v = torch.randn(1, NUM_HEADS, L, HEAD_DIM, generator=rng, dtype=torch.float32)

    # ── Block-level K-norm modulation ────────────────────────────────────────
    num_blocks  = max(1, L // block_size)
    n_signal_blocks = max(1, round(num_blocks * signal_frac))

    for h in range(NUM_HEADS):
        rng_h = torch.Generator()
        rng_h.manual_seed(seed + L * 100 + h)
        # Pick which blocks are "signal blocks" for this head
        block_perm = torch.randperm(num_blocks, generator=rng_h)
        signal_blocks = block_perm[:n_signal_blocks]
        for b_idx in signal_blocks:
            start = int(b_idx) * block_size
            end   = min(start + block_size, L)
            k[0, h, start:end, :] *= signal_scale

    # Normalize Q (token-level) to standard scale; sparsity comes from K-norm
    q = q / q.norm(dim=-1, keepdim=True).clamp(min=1e-8) * (HEAD_DIM ** 0.25)

    return (
        q.to(dtype=dtype, device=device),
        k.to(dtype=dtype, device=device),
        v.to(dtype=dtype, device=device),
    )


def estimate_block_sparsity(
    q: torch.Tensor,
    k: torch.Tensor,
    block_size: int = 128,
    sample_rows: int = 64,
    mass_threshold: float = 0.01,
    seed: int = 0,
) -> float:
    """Quick block sparsity estimate (same as sparsity_estimator.block_sparsity_ratio)."""
    import math
    B, H, L, D = q.shape
    scale = math.sqrt(D)
    rng = torch.Generator(device=q.device)
    rng.manual_seed(seed)

    min_row = min(block_size, L - 1)
    row_idx = torch.randint(min_row, L, (min(sample_rows, L - min_row),),
                            generator=rng, device=q.device)

    pruned = total = 0
    with torch.inference_mode():
        for ri in row_idx:
            qi = q[:, :, ri:ri+1, :]
            ki = k[:, :, :ri+1, :]
            scores = (qi @ ki.transpose(-1, -2)) / scale
            attn = torch.softmax(scores.float(), dim=-1).squeeze(2)
            ctx_blocks = max(1, (ri + 1).item() // block_size)
            trim_len = ctx_blocks * block_size
            if trim_len > attn.shape[-1]:
                trim_len = attn.shape[-1]
                ctx_blocks = trim_len // block_size
            if ctx_blocks < 1:
                continue
            attn_trim = attn[..., :trim_len]
            block_mass = attn_trim.reshape(B, H, ctx_blocks, block_size).sum(-1)
            row_sum = block_mass.sum(-1, keepdim=True).clamp(min=1e-8)
            norm_mass = block_mass / row_sum
            below = (norm_mass < mass_threshold).sum().item()
            pruned += below
            total  += norm_mass.numel()

    return pruned / max(total, 1)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate high-sparsity QKV caches for routing comparison."
    )
    ap.add_argument(
        "--lengths", nargs="+", type=int,
        default=[4096, 8192, 16384, 32768, 65536],
    )
    ap.add_argument(
        "--scenarios", nargs="+",
        default=list(SCENARIOS.keys()),
    )
    ap.add_argument("--out-dir", type=Path, default=Path("data/cache"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--verify-sparsity", action="store_true",
                    help="Compute block sparsity estimate before saving (slow at L≥32K).")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    for scenario in args.scenarios:
        if scenario not in SCENARIOS:
            print(f"Unknown scenario '{scenario}'. Available: {list(SCENARIOS.keys())}")
            continue
        params = SCENARIOS[scenario]
        print(f"\n── {scenario} ─────────────────────────────────────────")
        print(f"   signal_frac={params['signal_frac']}, "
              f"signal_scale={params['signal_scale']}")
        print(f"   {params['desc']}")

        for L in args.lengths:
            q, k, v = generate_qkv(scenario, L, seed=args.seed)

            sparsity_str = ""
            if args.verify_sparsity and L <= 32768:
                sp = estimate_block_sparsity(q, k, block_size=128, sample_rows=64)
                sparsity_str = f"  block_sparsity≈{sp:.3f}"

            fname = args.out_dir / f"qkv_len_{L}_{scenario}_seed_{args.seed}.pt"
            torch.save({"q": q, "k": k, "v": v, "scenario": scenario,
                        "seq_len": L, "seed": args.seed,
                        "signal_frac": params["signal_frac"],
                        "signal_scale": params["signal_scale"]}, fname)
            print(f"  L={L:>6}  saved → {fname.name}{sparsity_str}")

    # Merge sparse_* entries into unified manifest.json (with lm_* from generate_lm_patterns_qkv.py)
    manifest_path = args.out_dir / "manifest.json"
    m: dict = {}
    if manifest_path.exists():
        m = json.loads(manifest_path.read_text())
    for scenario in args.scenarios:
        if scenario not in SCENARIOS:
            continue
        for L in args.lengths:
            fname = f"qkv_len_{L}_{scenario}_seed_{args.seed}.pt"
            m[f"L{L}_{scenario}"] = {
                "file": fname, "seq_len": L, "scenario": scenario,
            }
    manifest_path.write_text(json.dumps(m, indent=2))
    print(f"\nUpdated manifest → {manifest_path}  ({len(m)} total entries)")

    print("\nDone.")


if __name__ == "__main__":
    main()
