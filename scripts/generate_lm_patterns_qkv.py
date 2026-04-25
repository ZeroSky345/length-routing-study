#!/usr/bin/env python3
"""
Generate QKV data that mimics REAL LLM attention patterns.

Key construction principle
--------------------------
For attention to concentrate on a SET of positions S (|S| = W_eff) we need:

    q[i] · k[j] / sqrt(D)  ≈  signal_score  >>  background_score  ∀j ∈ S

The fundamental challenge is that with L background tokens, even a mild signal
gets swamped unless:
    signal_score  >>  log(L / W_eff)   (from softmax denominator analysis)

METHOD: **Block shared-context injection**
- Divide the sequence into blocks of size B.
- Sample one random unit vector u_b per block (per head).
- Add u_b * Q_SCALE  in-place to q[i] for all i in block b.
- Set k[j] += u_b * K_SCALE for all j in block b.
- Then q[i] · k[j] / sqrt(D) ≈ Q_SCALE * K_SCALE / sqrt(D)  when same block,
  ≈ 0                                                         when different blocks.

With Q_SCALE = K_SCALE = 10 and D = 128:
  signal_score  = 10 * 10 / sqrt(128) = 8.84
  background_max ≈ 0.2 * sqrt(2*ln(L)) ≈ 2.0  (noise k std=0.2)
  → signal >> background for all L ≤ 512K  ✓

Five scenarios
--------------
lm_sink_local     : Attention sinks (0,1) + local blocks (B=64).
lm_sparse_global  : 5% of positions are global anchors (all queries attend to them).
lm_hierarchical   : Block leaders every 128 tokens attract within-block attention.
lm_local_periodic : Small blocks (B=32) + periodic sinks every 256 tokens.
lm_mixed          : Sink + local blocks + global anchors combined.

Usage
-----
  python scripts/generate_lm_patterns_qkv.py \\
      --lengths 4096 8192 16384 32768 65536 131072 \\
      --output-dir data/cache --seed 42
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

NUM_HEADS = 16
HEAD_DIM  = 128

SCENARIO_DESCRIPTIONS = {
    "lm_sink_local":    "Sink tokens (0,1) + local block attention (block=64). Most common LLM pattern.",
    "lm_sparse_global": "5% random global anchor tokens attract all queries. Mimics retrieval layers.",
    "lm_hierarchical":  "Block leaders every 128 tokens collect within-block attention.",
    "lm_local_periodic":"Small local blocks (B=32) + periodic global sinks every 256 tokens.",
    "lm_mixed":         "Sink + local blocks + global anchors combined (composite realistic).",
}

# ── Primitive helpers ─────────────────────────────────────────────────────────

def _unit(H: int, D: int, g: torch.Generator) -> torch.Tensor:
    """Sample a unit vector [H, D] per head."""
    u = torch.randn(H, D, generator=g)
    return u / u.norm(dim=-1, keepdim=True)


def _inject_block_context(
    q: torch.Tensor, k: torch.Tensor,
    H: int, L: int, D: int,
    bsize: int, q_scale: float, k_scale: float,
    g: torch.Generator,
) -> None:
    """
    In-place block-context injection.
    q, k: [1, H, L, D]  float32 (modified in-place)
    Samples per-head, per-block unit vectors; adds to q and k slices.
    """
    num_b = (L + bsize - 1) // bsize
    u_all = torch.randn(H, num_b, D, generator=g)
    u_all = u_all / u_all.norm(dim=-1, keepdim=True)  # [H, num_b, D]
    for b in range(num_b):
        s = b * bsize
        e = min(s + bsize, L)
        u_b = u_all[:, b, :]   # [H, D]
        # broadcast over batch(1) and positions(e-s)
        dq = u_b.unsqueeze(0).unsqueeze(2) * q_scale   # [1, H, 1, D]
        dk = u_b.unsqueeze(0).unsqueeze(2) * k_scale
        q[:, :, s:e, :].add_(dq)
        k[:, :, s:e, :].add_(dk)


def _inject_global_sink(
    q: torch.Tensor, k: torch.Tensor,
    H: int, D: int, positions: list[int],
    q_scale: float, k_scales: list[float],
    g: torch.Generator,
) -> None:
    """
    Inject a shared sink direction so that ALL queries attend to given positions.
    q_scale: amplitude added to all Q positions.
    k_scales[i]: key amplitude for positions[i].
    """
    u = _unit(H, D, g)          # [H, D]
    q.add_(u.unsqueeze(0).unsqueeze(2) * q_scale)
    for pos, ks in zip(positions, k_scales):
        k[:, :, pos, :] = u * ks


def _inject_anchor_keys(
    q: torch.Tensor, k: torch.Tensor,
    H: int, D: int, anchor_positions: torch.Tensor,
    q_scale: float, k_scale: float,
    g: torch.Generator,
) -> None:
    """
    Shared-direction global anchors: all Q shifted by u * q_scale;
    k[anchor] = u * k_scale for each anchor.
    """
    u = _unit(H, D, g)
    q.add_(u.unsqueeze(0).unsqueeze(2) * q_scale)
    for pos in anchor_positions.tolist():
        k[:, :, pos, :] = u * k_scale


# ── Scenario generators ───────────────────────────────────────────────────────

def _gen_lm_sink_local(L: int, H: int, D: int, seed: int) -> tuple:
    """
    Attention sinks at positions 0,1 (all queries attend to them)
    + block-local attention (block size = 64).

    Signal analysis (D=128, block=64, L=131072):
      Sink score  = Q_SINK * K_SINK / sqrt(D) = 4 * 12 / 11.3 = 4.25
      Block score = Q_LOC * K_LOC / sqrt(D)   = 10 * 10 / 11.3 = 8.85
      Background std (noise=0.2, after q-scaling) ≈ 0.2 * sqrt((D + Q²)/D) ≈ 1.1
      Max background ≈ 1.1 * sqrt(2 ln(L)) ≈ 5.4  → block score (8.85) > max background ✓
    """
    g  = torch.Generator().manual_seed(seed)
    q  = torch.randn(1, H, L, D, generator=g)
    v  = torch.randn(1, H, L, D, generator=g)
    k  = torch.randn(1, H, L, D, generator=g) * 0.2

    # Attention sinks
    _inject_global_sink(q, k, H, D, [0, 1], q_scale=4.0, k_scales=[12.0, 6.0], g=g)

    # Local block context
    _inject_block_context(q, k, H, L, D, bsize=64, q_scale=10.0, k_scale=10.0, g=g)

    s = D ** -0.5
    return (q * s).bfloat16(), (k * s).bfloat16(), (v * s).bfloat16()


def _gen_lm_sparse_global(L: int, H: int, D: int, seed: int) -> tuple:
    """
    5% random positions are global anchors. All queries attend to them.
    Mimics copy/retrieval attention in middle Transformer layers.

    Signal analysis:
      anchor_score = Q_ANCH * K_ANCH / sqrt(D) = 8 * 12 / 11.3 = 8.49
      Background std ≈ 0.2 * sqrt(1 + 64/128) = 0.24
      Max background ≈ 5.2  → anchor score (8.49) > max background ✓
    """
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(1, H, L, D, generator=g)
    v = torch.randn(1, H, L, D, generator=g)
    k = torch.randn(1, H, L, D, generator=g) * 0.2

    n_anch = max(4, round(L * 0.05))
    perm   = torch.randperm(L, generator=g)[:n_anch]
    _inject_anchor_keys(q, k, H, D, perm, q_scale=8.0, k_scale=12.0, g=g)

    s = D ** -0.5
    return (q * s).bfloat16(), (k * s).bfloat16(), (v * s).bfloat16()


def _gen_lm_hierarchical(L: int, H: int, D: int, seed: int) -> tuple:
    """
    Block leaders (first token of each 128-token block) attract within-block attention.
    Plus a global sink at position 0.

    Each block's leader key is set to u_b * K_LEADER; each block's Q vectors have
    u_b * Q_LEADER added, so ALL queries in block b attend to block b's leader.
    The global sink captures cross-block context.
    """
    g = torch.Generator().manual_seed(seed)
    BSIZE = 128
    q = torch.randn(1, H, L, D, generator=g)
    v = torch.randn(1, H, L, D, generator=g)
    k = torch.randn(1, H, L, D, generator=g) * 0.2

    # Global sink
    _inject_global_sink(q, k, H, D, [0], q_scale=3.0, k_scales=[10.0], g=g)

    # Per-block leaders: block context shared ONLY within each block
    Q_LEAD = 10.0;  K_LEAD = 10.0
    num_b = (L + BSIZE - 1) // BSIZE
    u_all = torch.randn(H, num_b, D, generator=g)
    u_all = u_all / u_all.norm(dim=-1, keepdim=True)
    for b in range(num_b):
        s_b  = b * BSIZE
        e_b  = min(s_b + BSIZE, L)
        u_b  = u_all[:, b, :]          # [H, D]
        dq   = u_b.unsqueeze(0).unsqueeze(2) * Q_LEAD  # [1,H,1,D]
        q[:, :, s_b:e_b, :].add_(dq)
        # Key: only the LEADER position gets high key strength
        k[:, :, s_b, :] = u_b * K_LEAD

    sc = D ** -0.5
    return (q * sc).bfloat16(), (k * sc).bfloat16(), (v * sc).bfloat16()


def _gen_lm_local_periodic(L: int, H: int, D: int, seed: int) -> tuple:
    """
    Very strong local blocks (B=32) + periodic global sinks every 256 tokens.
    Typical of later Transformer layers with short context dependencies.
    """
    g = torch.Generator().manual_seed(seed)
    PERIOD = 256
    BSIZE  = 32
    q = torch.randn(1, H, L, D, generator=g)
    v = torch.randn(1, H, L, D, generator=g)
    k = torch.randn(1, H, L, D, generator=g) * 0.2

    # Periodic checkpoint sinks
    cp_positions = list(range(0, L, PERIOD))
    _inject_global_sink(q, k, H, D, cp_positions,
                        q_scale=3.5,
                        k_scales=[10.0] * len(cp_positions), g=g)

    # Strong local blocks
    _inject_block_context(q, k, H, L, D, bsize=BSIZE, q_scale=10.0, k_scale=10.0, g=g)

    s = D ** -0.5
    return (q * s).bfloat16(), (k * s).bfloat16(), (v * s).bfloat16()


def _gen_lm_mixed(L: int, H: int, D: int, seed: int) -> tuple:
    """
    Most realistic composite LLM pattern:
      Sinks (0,1) + 5% global anchors + local blocks (B=64).
    """
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(1, H, L, D, generator=g)
    v = torch.randn(1, H, L, D, generator=g)
    k = torch.randn(1, H, L, D, generator=g) * 0.2

    # Sinks
    _inject_global_sink(q, k, H, D, [0, 1], q_scale=3.5, k_scales=[10.0, 5.0], g=g)

    # 5% global anchors
    n_anch = max(4, round(L * 0.05))
    perm   = torch.randperm(L - 2, generator=g)[:n_anch] + 2
    _inject_anchor_keys(q, k, H, D, perm, q_scale=3.0, k_scale=8.0, g=g)

    # Local block context
    _inject_block_context(q, k, H, L, D, bsize=64, q_scale=10.0, k_scale=10.0, g=g)

    s = D ** -0.5
    return (q * s).bfloat16(), (k * s).bfloat16(), (v * s).bfloat16()


# ── Registry & main ───────────────────────────────────────────────────────────

_GENERATORS = {
    "lm_sink_local":    _gen_lm_sink_local,
    "lm_sparse_global": _gen_lm_sparse_global,
    "lm_hierarchical":  _gen_lm_hierarchical,
    "lm_local_periodic":_gen_lm_local_periodic,
    "lm_mixed":         _gen_lm_mixed,
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lengths", type=int, nargs="+",
                   default=[4096, 8192, 16384, 32768, 65536, 131072])
    p.add_argument("--output-dir", type=str, default="data/cache")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-heads", type=int, default=NUM_HEADS)
    p.add_argument("--head-dim",  type=int, default=HEAD_DIM)
    p.add_argument("--scenarios", nargs="+", default=list(_GENERATORS.keys()))
    return p.parse_args()


def main():
    args   = parse_args()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    H, D   = args.num_heads, args.head_dim

    manifest_path = outdir / "manifest.json"
    manifest: dict = {}
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)

    for sc in args.scenarios:
        if sc not in _GENERATORS:
            print(f"[WARN] unknown scenario {sc!r}, skipping")
            continue
        fn = _GENERATORS[sc]
        for L in args.lengths:
            fname = f"qkv_len_{L}_{sc}_seed_{args.seed}.pt"
            fpath = outdir / fname

            if fpath.exists():
                print(f"  skip (exists): {fname}")
                key = f"L{L}_{sc}"
                manifest[key] = {"file": fname, "seq_len": L, "scenario": sc}
                continue

            print(f"  generating {fname} ...", end=" ", flush=True)
            q, k, v = fn(L, H, D, args.seed)
            torch.save({"q": q, "k": k, "v": v,
                        "meta": {"scenario": sc, "seq_len": L,
                                 "num_heads": H, "head_dim": D,
                                 "description": SCENARIO_DESCRIPTIONS[sc]}},
                       fpath)
            key = f"L{L}_{sc}"
            manifest[key] = {"file": fname, "seq_len": L, "scenario": sc}
            print(f"saved ({q.shape})")

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest → {manifest_path}  ({len(manifest)} entries)")


if __name__ == "__main__":
    main()
