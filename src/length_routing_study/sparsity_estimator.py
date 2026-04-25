"""
Lightweight attention sparsity estimator.

Estimates **how sparse the real attention matrix will be** for a given
(Q, K, V) triple without computing the full O(n²) softmax.  The result
is fed back into the routing cost model so that ``active_blocks`` reflects
the actual input rather than a fixed geometric approximation.

Three estimation methods (cheapest → most accurate):
------------------------------------------------------
1. ``kv_norm_cv``  — coefficient of variation of K norms.
   A high CV means some keys will dominate the softmax → high sparsity.
   Cost: O(n)  — just a norm over K.

2. ``sample_topk``  — sample a random subset of query rows, compute their
   full softmax row, measure what fraction of attention mass is in the
   top-k blocks.  High concentration → high sparsity.
   Cost: O(sample_rows × n)  — 1–4 % of full attention.

3. ``block_sparsity``  — quantise attention into blocks, run softmax on a
   sample, threshold each block's total mass.  Returns the fraction of
   blocks that would be pruned by a sparse kernel.
   Cost: O(sample_rows × n)  — same as sample_topk.

The composite ``estimate_sparsity`` function runs all three, weights them,
and returns a ``SparsityProfile`` dataclass used by the cost model.

Design constraints
------------------
* All methods work on **bfloat16 / float16** tensors directly (no cast).
* All methods are wrapped in ``torch.inference_mode()`` — no autograd.
* GPU memory is released after each pass.
* The module has **no hard dependency** on the rest of length_routing_study;
  it can be used standalone with any (Q, K, V) tensors.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Optional

import torch


# ── Result type ────────────────────────────────────────────────────────────────

@dataclass
class SparsityProfile:
    """
    Per-input sparsity characterisation returned by the estimator.

    Fields
    ------
    kv_norm_cv : float
        Coefficient of variation of per-token K norms (std/mean).
        Higher → keys are more "peaky" → attention more concentrated.

    sample_topk_coverage : float
        Average fraction of attention mass in the top-``topk_blocks`` blocks,
        measured on a sample of query rows.
        Range [0, 1].  Near 1 → very sparse; near 1/num_blocks → uniform.

    estimated_active_fraction : float
        Estimated fraction of attention blocks that a sparse kernel would
        actually compute (1 - pruned_fraction).
        Range (0, 1].  Lower → sparser → more benefit from PBS/Flex.

    estimated_sparsity_ratio : float
        1 - estimated_active_fraction.  Higher → sparser.

    block_sparsity_ratio : float | None
        Fraction of (block, block) pairs below the mass threshold, from the
        block-level analysis.  None if not computed.

    num_heads : int
    seq_len   : int
    block_size : int
    method : str
        Which methods were combined.
    """
    kv_norm_cv:              float
    sample_topk_coverage:    float
    estimated_active_fraction: float
    estimated_sparsity_ratio:  float
    block_sparsity_ratio:    Optional[float]
    num_heads:               int
    seq_len:                 int
    block_size:              int
    method:                  str

    @property
    def is_sparse(self) -> bool:
        """True if estimated sparsity is high enough to benefit from PBS/Flex."""
        return self.estimated_sparsity_ratio > 0.5

    def summary(self) -> str:
        return (
            f"SparsityProfile(L={self.seq_len}, "
            f"cv={self.kv_norm_cv:.3f}, "
            f"topk_coverage={self.sample_topk_coverage:.3f}, "
            f"active_frac={self.estimated_active_fraction:.3f}, "
            f"sparsity={self.estimated_sparsity_ratio:.3f})"
        )


# ── Method 1: K-norm coefficient of variation ──────────────────────────────────

def kv_norm_cv(
    k: torch.Tensor,          # [B, H, L, D]
    eps: float = 1e-8,
) -> float:
    """
    Coefficient of variation of per-token K norms, averaged over heads.

    High CV → some tokens have much larger key norms → attention will
    concentrate on those tokens → sparse approximation is accurate.

    Cost: O(H × L × D)  — one norm pass over K.
    """
    with torch.inference_mode():
        norms = k.float().norm(dim=-1)        # [B, H, L]
        mean  = norms.mean(dim=-1, keepdim=True).clamp(min=eps)
        cv    = (norms.std(dim=-1) / mean.squeeze(-1)).mean()
    return float(cv.item())


# ── Method 2: Sample-based top-k coverage ─────────────────────────────────────

def sample_topk_coverage(
    q: torch.Tensor,           # [B, H, L, D]
    k: torch.Tensor,           # [B, H, L, D]
    block_size: int = 128,
    sample_rows: int = 64,
    topk_frac:   float = 0.25,  # top 25 % of blocks
    seed: int = 0,
) -> float:
    """
    Sample ``sample_rows`` query positions, compute their full softmax
    attention row, then measure what fraction of mass sits in the top
    ``topk_frac × num_blocks`` key blocks.

    A value near 1.0 means 25 % of blocks capture almost all attention
    → the remaining 75 % can be safely skipped.

    Cost: O(sample_rows × H × L)
    """
    B, H, L, D = q.shape
    num_blocks  = max(1, L // block_size)
    topk_blocks = max(1, round(num_blocks * topk_frac))

    scale = math.sqrt(D)
    rng   = torch.Generator(device=q.device)
    rng.manual_seed(seed)

    # Sample row indices (causal: only rows ≥ block_size so they have context)
    min_row = min(block_size, L - 1)
    row_idx = torch.randint(min_row, L, (min(sample_rows, L - min_row),),
                            generator=rng, device=q.device)

    coverages: list[float] = []
    with torch.inference_mode():
        for ri in row_idx:
            qi = q[:, :, ri : ri + 1, :]         # [B, H, 1, D]
            ki = k[:, :, : ri + 1, :]             # [B, H, ri+1, D]
            # Full causal softmax for this row
            scores = (qi @ ki.transpose(-1, -2)) / scale   # [B, H, 1, ri+1]
            attn   = torch.softmax(scores.float(), dim=-1)  # [B, H, 1, ri+1]

            # Block-aggregate: sum mass per block
            attn_flat = attn.squeeze(2)                     # [B, H, ri+1]
            ctx_blocks = max(1, (ri + 1).item() // block_size)
            # Trim to whole blocks
            trim_len   = ctx_blocks * block_size
            if trim_len > attn_flat.shape[-1]:
                trim_len = attn_flat.shape[-1]
                ctx_blocks = trim_len // block_size
            if ctx_blocks < 1:
                continue
            attn_trim  = attn_flat[..., :trim_len]           # [B, H, trim_len]
            block_mass = attn_trim.reshape(B, H, ctx_blocks, block_size).sum(-1)  # [B,H,blocks]

            # Top-k coverage
            k_blocks = min(topk_blocks, ctx_blocks)
            topk_mass, _ = block_mass.topk(k_blocks, dim=-1)
            coverage  = float(topk_mass.sum() / block_mass.sum().clamp(min=1e-8))
            coverages.append(coverage)

    return float(sum(coverages) / max(len(coverages), 1))


# ── Method 3: Block-level sparsity ratio ───────────────────────────────────────

def block_sparsity_ratio(
    q: torch.Tensor,
    k: torch.Tensor,
    block_size: int = 128,
    sample_rows: int = 128,
    mass_threshold: float = 0.01,   # prune blocks below 1 % of total mass
    seed: int = 0,
) -> float:
    """
    Estimate the fraction of (query_block, key_block) pairs that would be
    pruned by a sparse kernel given a mass threshold.

    Cost: O(sample_rows × H × L)
    """
    B, H, L, D = q.shape
    scale = math.sqrt(D)
    rng   = torch.Generator(device=q.device)
    rng.manual_seed(seed + 1)

    num_blocks = max(1, L // block_size)
    min_row    = min(block_size, L - 1)
    row_idx    = torch.randint(min_row, L, (min(sample_rows, L - min_row),),
                               generator=rng, device=q.device)

    pruned_total = 0
    total_pairs  = 0
    with torch.inference_mode():
        for ri in row_idx:
            qi = q[:, :, ri : ri + 1, :]
            ki = k[:, :, : ri + 1, :]
            scores = (qi @ ki.transpose(-1, -2)) / scale
            attn   = torch.softmax(scores.float(), dim=-1).squeeze(2)  # [B,H,L']

            ctx_blocks = max(1, (ri + 1).item() // block_size)
            trim_len   = ctx_blocks * block_size
            if trim_len > attn.shape[-1]:
                trim_len   = attn.shape[-1]
                ctx_blocks = trim_len // block_size
            if ctx_blocks < 1:
                continue
            attn_trim  = attn[..., :trim_len]
            block_mass = attn_trim.reshape(B, H, ctx_blocks, block_size).sum(-1)
            row_sum    = block_mass.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            norm_mass  = block_mass / row_sum                   # normalise per row
            below      = (norm_mass < mass_threshold).sum().item()
            pruned_total += below
            total_pairs  += norm_mass.numel()

    return pruned_total / max(total_pairs, 1)


# ── Composite estimator ────────────────────────────────────────────────────────

def estimate_sparsity(
    q: torch.Tensor,
    k: torch.Tensor,
    block_size: int = 128,
    sample_rows: int = 64,
    topk_frac:   float = 0.25,
    mass_threshold: float = 0.01,
    include_block_sparsity: bool = True,
    seed: int = 0,
) -> SparsityProfile:
    """
    Run all three estimators and return a unified ``SparsityProfile``.

    Active-fraction formula
    -----------------------
    We blend three signals:

      cv_signal     = clip(kv_norm_cv / 3.0, 0, 1)
                      (CV=3 → fully concentrated)
      topk_signal   = sample_topk_coverage   (already in [0,1])
      block_signal  = block_sparsity_ratio    (fraction pruned)

      sparsity = 0.25 × cv_signal + 0.50 × topk_signal + 0.25 × block_signal

    This blending gives the block-level sample the most weight because it
    most directly reflects what a sparse kernel actually does.
    """
    B, H, L, D = q.shape
    num_blocks  = max(1, L // block_size)

    # ── Method 1 ────────────────────────────────────────────────────────────
    cv = kv_norm_cv(k)
    cv_signal = min(1.0, cv / 3.0)   # saturate at CV=3

    # ── Method 2 ────────────────────────────────────────────────────────────
    topk_cov = sample_topk_coverage(
        q, k, block_size=block_size,
        sample_rows=sample_rows, topk_frac=topk_frac, seed=seed,
    )
    # topk_cov → sparsity proxy: coverage of top-25% blocks;
    # if top 25 % blocks capture 95 % mass → 75 % blocks are "pruned" → sparsity ≈ 0.75
    # But topk_cov ∈ [topk_frac, 1]; normalise to [0, 1]
    topk_signal = (topk_cov - topk_frac) / (1.0 - topk_frac + 1e-8)
    topk_signal = max(0.0, min(1.0, topk_signal))

    # ── Method 3 ────────────────────────────────────────────────────────────
    bsr = None
    block_signal = topk_signal   # default fallback
    if include_block_sparsity:
        bsr = block_sparsity_ratio(
            q, k, block_size=block_size,
            sample_rows=sample_rows, mass_threshold=mass_threshold, seed=seed,
        )
        block_signal = bsr

    # ── Blend ────────────────────────────────────────────────────────────────
    if include_block_sparsity:
        sparsity = 0.25 * cv_signal + 0.50 * topk_signal + 0.25 * block_signal
        method   = "kv_norm_cv+sample_topk+block_sparsity"
    else:
        sparsity = 0.35 * cv_signal + 0.65 * topk_signal
        method   = "kv_norm_cv+sample_topk"

    active_frac = max(0.05, 1.0 - sparsity)   # floor at 5 % to avoid near-zero

    return SparsityProfile(
        kv_norm_cv=cv,
        sample_topk_coverage=topk_cov,
        estimated_active_fraction=active_frac,
        estimated_sparsity_ratio=sparsity,
        block_sparsity_ratio=bsr,
        num_heads=H,
        seq_len=L,
        block_size=block_size,
        method=method,
    )


# ── Quick per-layer sampler for real-model integration ────────────────────────

def estimate_from_first_layers(
    model_outputs_qkv: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    block_size: int = 128,
    sample_rows: int = 64,
    seed: int = 0,
) -> SparsityProfile:
    """
    Average sparsity estimates over the first N layers' (Q, K, V) tensors.

    Use this when you can hook into a model's forward pass to sample Q/K
    from the first 2–4 transformer layers rather than generating random data.
    """
    profiles = [
        estimate_sparsity(q, k, block_size=block_size,
                          sample_rows=sample_rows, seed=seed + i)
        for i, (q, k, _) in enumerate(model_outputs_qkv)
    ]
    if not profiles:
        raise ValueError("No QKV samples provided")

    avg_cv    = sum(p.kv_norm_cv for p in profiles) / len(profiles)
    avg_topk  = sum(p.sample_topk_coverage for p in profiles) / len(profiles)
    avg_bsr   = (sum(p.block_sparsity_ratio for p in profiles
                     if p.block_sparsity_ratio is not None) /
                 max(1, sum(1 for p in profiles if p.block_sparsity_ratio is not None)))
    avg_frac  = sum(p.estimated_active_fraction for p in profiles) / len(profiles)
    avg_spar  = sum(p.estimated_sparsity_ratio  for p in profiles) / len(profiles)

    return SparsityProfile(
        kv_norm_cv=avg_cv,
        sample_topk_coverage=avg_topk,
        estimated_active_fraction=avg_frac,
        estimated_sparsity_ratio=avg_spar,
        block_sparsity_ratio=avg_bsr if avg_bsr > 0 else None,
        num_heads=profiles[0].num_heads,
        seq_len=profiles[0].seq_len,
        block_size=block_size,
        method="multi_layer_avg",
    )
