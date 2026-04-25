"""
Benchmark custom mask-based selectors across sequence lengths and text scenarios.

These selectors (FixedWindow, TopKScore, VerticalOnly, etc.) generate a boolean
attention mask that controls WHICH token pairs to compute.  Unlike PBS/Flex which
ship their own sparse CUDA kernels, here we apply the mask via PyTorch's standard
scaled_dot_product_attention so the comparison measures:

  t_select_ms  — time to compute the mask (selection overhead)
  t_kernel_ms  — time for masked-SDPA (proxy; mask applied but compute is dense)
  t_total_ms   — t_select + t_kernel
  sparsity     — fraction of zero entries in the mask
  mse          — mean squared error vs Flash reference

Note: masked-SDPA does not actually skip zero-attention pairs — it only zeroes
them after softmax.  So t_kernel_ms reflects the full O(n²) compute cost.
For a fair speed comparison, only t_select_ms is a true overhead measurement;
for a realistic comparison with PBS/Flex the two phases are shown separately.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from .selector_base import BaseSelector, SelectionResult
from .mask_selectors import (
    DenseSelector,
    FixedWindowSelector,
    TopKScoreSelector,
    FixedTopKSelector,
    VerticalOnlySelector,
    SlashOnlySelector,
    VerticalSlashSelector,
    QueryAwareFullBlockSelector,
    LengthBasedHybridSelector,
    AdaptiveFractionWindowSelector,
    SqrtWindowSelector,
    ProgressiveSqrtTopKSelector,
    TierRouterAlphaSelector,
    TierRouterBetaSelector,
    TierRouterGammaSelector,
)


# ── Timing helper ─────────────────────────────────────────────────────────────
def _cuda_ms(fn) -> tuple[Any, float]:
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    s.record()
    out = fn()
    e.record()
    torch.cuda.synchronize()
    return out, float(s.elapsed_time(e))


# ── Result dataclass ──────────────────────────────────────────────────────────
@dataclass
class SelectorRecord:
    selector_name: str
    selector_type: str       # "window" | "score" | "pattern" | "hybrid" | "dense"
    seq_len: int
    scenario: str

    sparsity: float          # fraction of zero entries in mask
    density: float           # 1 - sparsity

    t_select_ms: float       # mask generation time (selection overhead)
    t_kernel_ms: float       # masked-SDPA time
    t_total_ms: float        # t_select + t_kernel

    mse: float               # vs Flash reference
    passed_mse: bool

    layout: dict[str, Any]

    @property
    def passed(self) -> bool:
        return self.passed_mse

    def as_dict(self) -> dict[str, Any]:
        return {
            "selector_name":  self.selector_name,
            "selector_type":  self.selector_type,
            "seq_len":        self.seq_len,
            "scenario":       self.scenario,
            "sparsity":       round(self.sparsity, 4),
            "density":        round(self.density, 4),
            "t_select_ms":    round(self.t_select_ms, 4),
            "t_kernel_ms":    round(self.t_kernel_ms, 4),
            "t_total_ms":     round(self.t_total_ms, 4),
            "mse":            self.mse,
            "passed_mse":     self.passed_mse,
            "layout":         self.layout,
        }


# ── Selector type classification ──────────────────────────────────────────────
def _classify(sel: BaseSelector) -> str:
    if isinstance(sel, DenseSelector):
        return "dense"
    if isinstance(sel, (FixedWindowSelector, AdaptiveFractionWindowSelector,
                         SqrtWindowSelector)):
        return "window"
    if isinstance(sel, (TopKScoreSelector, FixedTopKSelector,
                         ProgressiveSqrtTopKSelector, QueryAwareFullBlockSelector)):
        return "score"
    if isinstance(sel, (VerticalOnlySelector, SlashOnlySelector, VerticalSlashSelector)):
        return "pattern"
    if isinstance(sel, (TierRouterAlphaSelector, TierRouterBetaSelector,
                         TierRouterGammaSelector, LengthBasedHybridSelector)):
        return "hybrid"
    return "other"


# ── Single selector benchmark ─────────────────────────────────────────────────
def run_selector(
    selector: BaseSelector,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scenario: str,
    reference: torch.Tensor | None = None,
    warmup: int = 2,
    repeats: int = 5,
    mse_threshold: float = 0.05,
) -> SelectorRecord | None:
    L = q.shape[2]
    sel_type = _classify(selector)

    # ── Warmup ───────────────────────────────────────────────────────────────
    try:
        for _ in range(warmup):
            with torch.inference_mode():
                result = selector.select(q, k)
            torch.cuda.synchronize()
    except Exception as exc:
        warnings.warn(f"{selector.name} warmup failed @ L={L}: {exc}")
        return None

    # ── Time selection ────────────────────────────────────────────────────────
    sel_lats, kernel_lats = [], []
    last_result: SelectionResult | None = None
    last_out: torch.Tensor | None = None

    for _ in range(repeats):
        with torch.inference_mode():
            result, t_sel = _cuda_ms(lambda: selector.select(q, k))
        sel_lats.append(t_sel)
        last_result = result

    # ── Time masked attention ─────────────────────────────────────────────────
    with torch.inference_mode():
        mask = last_result.mask  # bool [B,H,L,L] or None

    for _ in range(repeats):
        with torch.inference_mode():
            if mask is not None:
                # Convert bool mask → float additive mask (0 / -inf)
                additive = torch.zeros_like(mask, dtype=q.dtype)
                additive = additive.masked_fill(~mask, float("-inf"))
                out, t_k = _cuda_ms(lambda: F.scaled_dot_product_attention(
                    q, k, v, attn_mask=additive, is_causal=False))
            else:
                # Dense / window selectors that return mask=None
                out, t_k = _cuda_ms(lambda: F.scaled_dot_product_attention(
                    q, k, v, is_causal=True))
        kernel_lats.append(t_k)
        last_out = out

    t_select = sum(sel_lats) / len(sel_lats)
    t_kernel = sum(kernel_lats) / len(kernel_lats)
    t_total  = t_select + t_kernel

    # ── Accuracy ──────────────────────────────────────────────────────────────
    ref = reference if reference is not None else \
        F.scaled_dot_product_attention(q, k, v, is_causal=True)
    mse = float(torch.mean((last_out.float() - ref.float()) ** 2).item())

    sparsity = float(last_result.sparsity) if last_result else 0.0
    layout   = dict(last_result.layout) if last_result and last_result.layout else {}

    return SelectorRecord(
        selector_name=selector.name,
        selector_type=sel_type,
        seq_len=L,
        scenario=scenario,
        sparsity=sparsity,
        density=1.0 - sparsity,
        t_select_ms=t_select,
        t_kernel_ms=t_kernel,
        t_total_ms=t_total,
        mse=mse,
        passed_mse=(mse <= mse_threshold),
        layout=layout,
    )


# ── Default selector suite ────────────────────────────────────────────────────
def default_selector_suite() -> list[BaseSelector]:
    """
    A representative set of selectors for benchmarking.
    Skips TierRouter selectors (they internally branch on length — fine for
    standalone use, but noted as 'hybrid' in type classification).
    """
    return [
        DenseSelector(),
        # Window family
        FixedWindowSelector(window_size=256),
        FixedWindowSelector(window_size=512),
        FixedWindowSelector(window_size=1024),
        FixedWindowSelector(window_size=2048),
        AdaptiveFractionWindowSelector(fraction=0.10, min_w=128, max_w=4096),
        AdaptiveFractionWindowSelector(fraction=0.25, min_w=256, max_w=8192),
        SqrtWindowSelector(coeff=2.0, min_w=128, max_w=4096),
        SqrtWindowSelector(coeff=4.0, min_w=256, max_w=8192),
        # Score-based
        TopKScoreSelector(keep_ratio=0.10),
        TopKScoreSelector(keep_ratio=0.20),
        FixedTopKSelector(topk=256),
        FixedTopKSelector(topk=512),
        QueryAwareFullBlockSelector(block_size=128, blocks_per_query_block=4),
        QueryAwareFullBlockSelector(block_size=128, blocks_per_query_block=8),
        ProgressiveSqrtTopKSelector(coeff=6.0, lo=0.05, hi=0.40),
        # Pattern-based
        VerticalOnlySelector(keep_columns=256),
        VerticalOnlySelector(keep_columns=512),
        SlashOnlySelector(offsets=(0, 128, 256, 512)),
        VerticalSlashSelector(keep_columns=128, offsets=(0, 64, 128, 256)),
        VerticalSlashSelector(keep_columns=256, offsets=(0, 128, 256, 512)),
        # Hybrid / tier
        LengthBasedHybridSelector(),
        TierRouterAlphaSelector(),
        TierRouterBetaSelector(),
        TierRouterGammaSelector(),
    ]


# ── Sweep entry point ─────────────────────────────────────────────────────────
def run_selector_sweep(
    lengths: list[int],
    scenarios_qkv: dict[str, dict[int, tuple]],  # scenario → {L → (q,k,v)}
    selectors: list[BaseSelector] | None = None,
    warmup: int = 2,
    repeats: int = 5,
    mse_threshold: float = 0.05,
    verbose: bool = True,
) -> list[SelectorRecord]:
    """
    Sweep all selectors across lengths and scenarios.

    Parameters
    ----------
    scenarios_qkv : {scenario_name: {seq_len: (q, k, v)}}
        Pre-loaded QKV tensors.  Load once, reuse across selectors.
    """
    suite = selectors or default_selector_suite()
    records: list[SelectorRecord] = []

    for scenario, qkv_by_len in scenarios_qkv.items():
        for L, (q, k, v) in sorted(qkv_by_len.items()):
            if L not in lengths:
                continue
            if verbose:
                print(f"\n  [{scenario}] L={L}")

            # Flash reference (computed once per length × scenario)
            with torch.inference_mode():
                ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)

            for sel in suite:
                rec = run_selector(
                    sel, q, k, v,
                    scenario=scenario,
                    reference=ref,
                    warmup=warmup,
                    repeats=repeats,
                    mse_threshold=mse_threshold,
                )
                if rec is not None:
                    records.append(rec)
                    if verbose:
                        status = "OK" if rec.passed else "FAIL"
                        print(f"    {rec.selector_name:<45} "
                              f"sparsity={rec.sparsity:.3f}  "
                              f"t_sel={rec.t_select_ms:.2f}ms  "
                              f"t_ker={rec.t_kernel_ms:.2f}ms  "
                              f"mse={rec.mse:.5f}  {status}")

    return records
