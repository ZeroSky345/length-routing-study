"""
Length- and sparsity-aware routing algorithm for sparse attention methods.

Given that a sparse attention backend MUST be used (the caller has already
decided against dense FlashAttention), this router selects the optimal
sparse backend (PBS, FlexPrefill, or a lightweight mask-based strategy) and
its parameters based on sequence length and estimated input sparsity.

Problem statement
-----------------
All sparse backends share the same two-stage structure:

    T_total = T_select(L)   +   T_kernel(L, sparsity)
              ─────────────     ────────────────────────
              selection /        sparse attention kernel
              permutation        (only active blocks)

Selection overhead classes
--------------------------
  O(1)     : GlobalSinkWindow, SqrtWindow
               T_select ≈ constant (5 µs)
  O(L)     : PBS selection
               T_select ≈ 0.1 µs/token  (~0.4–6 ms for 4K–64K)
  O(L·D)   : BlockKNormTopK, CoverageTarget
               T_select ≈ 5 ns/token  (but with GPU overhead)
  O(r·L·D) : SampledAttentionTopK(r)
               T_select ≈ r × 0.01 ms × (L/4K)

Length-dependent routing logic
-------------------------------
At short L (4K–8K):
  - PBS T_select (0.8–1.2 ms) may EXCEED T_flash (0.5–1.6 ms)
  - O(1) structural strategies have negligible overhead → preferred
At medium L (8K–32K):
  - PBS T_select is 10–30% of T_flash → worth paying for good masks
  - O(L·D) strategies become competitive if GPU execution is fast
At long L (32K–128K+):
  - PBS T_select is ~10% of T_flash → amortised by kernel savings
  - Flex may eventually beat PBS (empirically: L ≫ 64K)

Routing decision
----------------
1. Estimate T_total for each candidate (structural → PBS → Flex)
2. Pick the backend with lowest estimated T_total
3. Within a backend: adapt parameters to (L, sparsity)

Usage
-----
  from length_routing_study.length_router import LengthAwareRouter

  router = LengthAwareRouter()
  decision = router.route(L=16384, sparsity=0.75)
  print(decision.backend, decision.params)

  # Full strategy-aware routing (includes O(1) strategies)
  decision = router.route_full(L=4096, sparsity=0.30)
"""
from __future__ import annotations

import bisect
import dataclasses
from dataclasses import dataclass, field
from typing import Any


# ─── Method identity ─────────────────────────────────────────────────────────

#: Short name used in result tables and Canvas charts to identify this method.
METHOD_NAME       = "Ours"
METHOD_FULL_NAME  = "Ours (Length-Aware Tier Router)"
METHOD_DESCRIPTION = (
    "Three-tier selection router: O(1) Structural at short L, "
    "O(L·D) KNorm at medium L, O(L) PBS at long L — "
    "with within-tier adaptive parameter tuning."
)

# ─── Routing decision ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class RoutingDecision:
    """
    Output of ``LengthAwareRouter.route()`` / ``route_full()``.

    Fields
    ------
    backend : "pbs" | "flex" | "structural" | "knorm" | "coverage" | "sampled"
    params  : dict with backend-specific parameters
    reason  : human-readable justification string
    strategy_instance : optional SelectionStrategy object
        For mask-based backends (structural / knorm / coverage / sampled), this holds
        the ready-to-call strategy instance so callers can do:
            result = decision.strategy_instance.select(q, k, v, block_size)
    critical_sparsity : float
    """
    backend: str
    params: dict[str, Any]
    reason: str
    strategy_instance: Any = None   # SelectionStrategy | None
    critical_sparsity: float = 0.0

    def is_sparse(self) -> bool:
        return self.backend != "flash"

    def summary(self) -> str:
        pstr = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return f"[{self.backend}]({pstr}) — {self.reason}"


# ─── Profiled overhead estimates (from empirical calibration) ─────────────────

# PBS selection overhead (ms): approximate linear fit from empirical data
# t_select_pbs ≈ T_SELECT_PBS_A × L + T_SELECT_PBS_B
T_SELECT_PBS_A  = 0.0001   # ms per token  (roughly 0.1 µs/token)
T_SELECT_PBS_B  = 0.40     # ms fixed cost (CUDA kernel launch, etc.)

# FlashAttention quadratic fit: T_flash ≈ FLASH_ALPHA × L²  (ms)
# Recalibrated from PBS kernel time at active_fraction=1.0 (A800 BF16):
#   L=4K →0.64ms, L=8K →1.82ms, L=16K→7.30ms,
#   L=32K→32.45ms, L=64K→140ms, L=128K→607ms
# α = T / L²: ranges from 2.7e-8 (8K) to 3.5e-8 (128K).
# Use 3.5e-8 as a conservative upper-bound (favours correct Flex routing at long L).
FLASH_ALPHA = 3.5e-8   # ms per token²

# PBS kernel effectiveness factor.
# PBS prunes blocks using a fixed threshold on K-norm scores.  In practice,
# this threshold is often too conservative — many workloads have near-uniform
# K-norms (especially synthetic or code-like text), so the PBS kernel runs at
# close to full density (active_fraction ≈ 1.0 in our benchmarks).
# This factor models the fraction of the estimated sparsity that PBS actually
# converts into block pruning:
#   effective_pbs_active = (1 - sparsity × PBS_KERNEL_EFF) × 0.95
# With PBS_KERNEL_EFF=0.30, PBS achieves ≈30% of the theoretical savings.
# Setting this low makes the routing more conservative (prefers Flex at long L).
PBS_KERNEL_EFF = 0.30

# FlexPrefill stats/budget overhead (roughly constant kernel)
T_SELECT_FLEX_B = 2.5   # ms (stats computation + Triton launch overhead)

# PBS block_size / segment_size defaults
PBS_BLOCK_SIZE  = 128
FLEX_BLOCK_SIZE = 128

# Flex selection overhead model (from A800 empirical calibration):
#   T_select_flex(L) = FLEX_A * L + FLEX_B
# Observed at warm Triton state: ~0.5–1.0 µs/token fixed, plus GPU launch cost.
T_SELECT_FLEX_A = 0.0010     # ms per token (1 µs/token) — measured ≈ 0.6-2.1µs
T_SELECT_FLEX_B = 2.0        # ms fixed cost

# Flex kernel efficiency model — length-dependent sparsity bonus.
#
# PBS uses a fixed threshold on block-level scores.  At long L, attention
# distributions become increasingly concentrated (power-law tails), so
# a coverage-based selector (Flex gamma) prunes far more blocks than a
# fixed-threshold selector.  Empirically on the A800:
#
#   L=4K   → Flex ≈ PBS active_frac  (no advantage)
#   L=32K  → Flex active_frac ≈ 0.85 × PBS
#   L=64K  → Flex active_frac ≈ 0.70 × PBS   (30 % fewer active blocks)
#   L=128K → Flex active_frac ≈ 0.55 × PBS   (45 % fewer active blocks)
#
# Model:  flex_active_frac = pbs_active_frac × FLEX_SPARSITY_SCALE(L)
# where FLEX_SPARSITY_SCALE(L) decays linearly from 1.0 at L=4K to
# FLEX_SPARSITY_SCALE_MIN at L=FLEX_SPARSITY_L_MAX and beyond.
FLEX_SPARSITY_SCALE_MIN = 0.45   # at L ≥ 128K, Flex keeps ~55% fewer blocks vs PBS
FLEX_SPARSITY_L_MIN     = 4_096  # below this: scale = 1.0 (no advantage)
FLEX_SPARSITY_L_MAX     = 131_072  # at this L scale hits the minimum

# ─── Mask-strategy overhead constants ─────────────────────────────────────────
#
# These are the O(1) and O(L·D) structural/data-driven strategies that can
# be used INSTEAD of PBS/Flex at short sequences where PBS overhead is high.
#
# O(1) strategies: negligible overhead (index arithmetic only)
T_SELECT_STRUCT_MS   = 0.005   # ~5 µs for GlobalSinkWindow / SqrtWindow

# O(L·D) strategies: GPU norm + topk — recalibrated from HierarchicalKNorm on A800 BF16
# Fit from: L=[8K→0.71ms, 16K→1.25ms, 32K→1.52ms, 64K→2.91ms, 128K→5.83ms]
# t_select ≈ 4.2e-5 × L + 0.37  (actual overhead is ~8x higher than raw FLOP estimate
# due to topk sorting, memory access fragmentation, and CUDA synchronize overhead)
T_SELECT_KNORM_A     = 4.2e-5  # ms per token  (~42 ns/token, A800 measured)
T_SELECT_KNORM_B     = 0.37    # ms fixed cost (CUDA kernel launch + topk overhead)

# O(r·L·D) sampled attention: r independent dot-products of length L
T_SELECT_SAMPLED_PER_SAMPLE = 0.01   # ms per sample for L=4096, D=128
T_SELECT_SAMPLED_L0         = 4096   # reference length for the above

# Kernel efficiency for mask-based strategies vs PBS (approximate):
#   These strategies produce boolean masks, not PBS permutations.
#   Actual kernel is SDPA with explicit attention_mask — less efficient than
#   PBS Triton kernel.  Use 1.0 × T_flash as kernel estimate (no sparsity gain
#   from the kernel itself; only the mask limits the effective computation).
#   For theoretical projections, use active_fraction × T_flash.
STRUCT_KERNEL_EFFICIENCY = 1.0   # placeholder: real block-sparse kernel TBD


# ─── Parameter selectors ─────────────────────────────────────────────────────

def pick_pbs_threshold(L: int, sparsity: float) -> float:
    """
    Adaptive PBS threshold based on (L, sparsity).

    The threshold controls how aggressively PBS prunes blocks.
    Higher threshold = more pruning = lower active_fraction = lower t_kernel,
    but potentially higher MSE.

    When the input is genuinely sparse (sparsity > 0.40), we can safely lower
    the threshold (more aggressive pruning) because there are already few
    important blocks — pruning more doesn't lose much accuracy.

    Formula:
      thr★ = max(0.70, 0.90 - clip(sparsity - 0.40, 0, 1) × 0.50)

    Examples:
      sparsity=0.30 → thr=0.90 (conservative)
      sparsity=0.50 → thr=0.85
      sparsity=0.70 → thr=0.80
      sparsity=0.90 → thr=0.75
      sparsity≥1.00 → thr=0.70 (floor)
    """
    reduction = max(0.0, sparsity - 0.40) * 0.50
    return round(max(0.70, 0.90 - reduction), 2)


def pick_flex_tau(L: int) -> float:
    """
    Length-adaptive FlexPrefill tau.

    Empirical finding: at L<32K, τ=0.05 minimises latency because the
    min_budget floor still forces a dense-enough computation.
    At L≥32K, τ=0.20 becomes faster — the longer context allows a sparser
    budget without sacrificing coverage (the "tau inversion" effect).
    """
    if L < 32_768:
        return 0.05
    return 0.20


def pick_flex_gamma(L: int, sparsity: float) -> float:
    """
    Length- and sparsity-adaptive FlexPrefill gamma.

    At short lengths, use high gamma (0.95) for better accuracy.
    At long lengths with high sparsity, lower gamma (0.80) for more pruning.
    """
    if L >= 32_768 and sparsity >= 0.60:
        return 0.80
    if L >= 16_384 and sparsity >= 0.75:
        return 0.85
    return 0.90


def pick_pbs_segment_size(L: int) -> int:
    """
    Length-adaptive PBS segment_size.

    Larger segment gives the PBS permutation kernel more flexibility in
    grouping similar-importance blocks, improving efficiency at long L.

    Empirical calibration (A800, BF16):
      L < 64K  : seg=256 is stable and matches the PBS_FIXED baseline.
                 Using seg=512 at short/medium L can trigger Triton
                 recompilation and shows inconsistent performance across runs.
      L >= 64K : seg=512 improves throughput by ~10-19% vs seg=256 because
                 the larger segment reduces memory access fragmentation in the
                 sparse Triton kernel.

    Note: PBS_FIXED baseline always uses seg=256.  Ours only deviates at
    L ≥ 64K (where seg=512 is consistently faster).
    """
    if L >= 65_536:
        return 512
    return 256


# ─── Critical sparsity calculation ───────────────────────────────────────────

def critical_sparsity_pbs(L: int) -> float:
    """
    Minimum sparsity for PBS to beat Flash.

    Derived from: T_select_pbs + T_flash × (1-s) < T_flash
      ⟺  s > T_select_pbs / T_flash

    Uses calibrated PBS selection overhead and FlashAttention model.
    """
    t_select = T_SELECT_PBS_A * L + T_SELECT_PBS_B   # ms
    t_flash  = FLASH_ALPHA * L * L                   # ms
    if t_flash < 1e-9:
        return 1.0
    s_star = t_select / t_flash
    return min(1.0, max(0.0, s_star))


def critical_sparsity_flex(L: int) -> float:
    """
    Minimum sparsity for FlexPrefill to beat Flash.

    Flex has higher T_select overhead than PBS (Triton stats kernel).
    """
    t_select = T_SELECT_FLEX_B
    t_flash  = FLASH_ALPHA * L * L
    if t_flash < 1e-9:
        return 1.0
    s_star = t_select / t_flash
    return min(1.0, max(0.0, s_star))


# ─── Main router ─────────────────────────────────────────────────────────────

@dataclass
class LengthAwareRouter:
    """
    Length- and sparsity-aware routing algorithm for PBS / FlexPrefill.

    Decision logic
    --------------
    1. Compute s★_pbs(L) and s★_flex(L) — the critical sparsity values.
    2. If estimated_sparsity < s★_pbs(L):
         → Route to Flash (sparse methods can't pay back their overhead).
    3. If s★_pbs(L) ≤ estimated_sparsity < s★_flex(L):
         → Route to PBS with adaptive threshold.
    4. If estimated_sparsity ≥ s★_flex(L):
         → Compare PBS vs Flex with adaptive params and pick lower cost.

    Within-backend adaptation:
    * PBS threshold: pick_pbs_threshold(L, sparsity)
    * Flex tau:      pick_flex_tau(L)
    * Flex gamma:    pick_flex_gamma(L, sparsity)
    """

    # Calibration overrides (optional — use to update from real measurements)
    flash_alpha: float = FLASH_ALPHA
    pbs_select_a: float = T_SELECT_PBS_A
    pbs_select_b: float = T_SELECT_PBS_B
    flex_select_b: float = T_SELECT_FLEX_B
    pbs_kernel_eff: float = PBS_KERNEL_EFF  # fraction of sparsity PBS actually prunes

    # PBS is always the fallback — never route to Flash.
    # Flex is considered when L is large enough for its selection overhead to be
    # offset by its superior content-adaptive sparsity.
    # Empirical crossover on A800: PBS vs Flex crossover is at L≈64K–128K.
    # At L=32K, Flex selection overhead (≈35ms) still exceeds kernel savings.
    # We set flex_min_length=65536 to be safe.
    flex_min_length: int = 0

    # Flex selection overhead model (empirically calibrated)
    flex_select_a: float = T_SELECT_FLEX_A
    flex_select_b_val: float = T_SELECT_FLEX_B

    def _t_flash(self, L: int) -> float:
        """Estimated dense FlashAttention time (used internally for kernel time estimation)."""
        return self.flash_alpha * L * L

    def _t_select_pbs(self, L: int) -> float:
        return self.pbs_select_a * L + self.pbs_select_b

    def _t_select_flex(self, L: int) -> float:
        return self.flex_select_a * L + self.flex_select_b_val

    def _t_kernel(self, L: int, active_fraction: float) -> float:
        """Estimated sparse-kernel time = flash_time × active_fraction."""
        return self._t_flash(L) * active_fraction

    def _t_pbs_total(self, L: int, sparsity: float) -> float:
        # PBS prunes only a fraction (pbs_kernel_eff) of the estimated sparsity.
        eff_sparsity = sparsity * self.pbs_kernel_eff
        active_frac  = max(0.05, 1.0 - eff_sparsity) * 0.95
        return self._t_select_pbs(L) + self._t_kernel(L, active_frac)

    def _flex_sparsity_scale(self, L: int) -> float:
        """
        Length-dependent multiplier that captures Flex's superior sparsity vs PBS.

        At short L, Flex and PBS achieve similar active fractions.
        At long L, attention concentrates → Flex's coverage-based gamma selector
        prunes far more blocks than PBS's fixed threshold.

        Returns a scale in [FLEX_SPARSITY_SCALE_MIN, 1.0]:
          scale=1.0  at L ≤ FLEX_SPARSITY_L_MIN  (no advantage)
          scale=MIN  at L ≥ FLEX_SPARSITY_L_MAX   (maximum advantage)
        """
        if L <= FLEX_SPARSITY_L_MIN:
            return 1.0
        if L >= FLEX_SPARSITY_L_MAX:
            return FLEX_SPARSITY_SCALE_MIN
        t = (L - FLEX_SPARSITY_L_MIN) / (FLEX_SPARSITY_L_MAX - FLEX_SPARSITY_L_MIN)
        return 1.0 - t * (1.0 - FLEX_SPARSITY_SCALE_MIN)

    def _t_flex_total(self, L: int, sparsity: float) -> float:
        # Flex achieves better sparsity than PBS at long L — model the advantage
        pbs_active = max(0.05, 1.0 - sparsity)
        flex_active = pbs_active * self._flex_sparsity_scale(L)
        flex_active = max(0.05, flex_active)
        return self._t_select_flex(L) + self._t_kernel(L, flex_active)

    def _crossover_sparsity_flex_over_pbs(self, L: int) -> float:
        """
        Minimum sparsity at which Flex's better compression offsets its
        higher selection overhead compared to PBS.

        Derived from T_flex < T_pbs:
          T_sel_flex + T_flash×(1-s)×0.90 < T_sel_pbs + T_flash×(1-s)×0.95
          (T_sel_flex - T_sel_pbs) < T_flash × (1-s) × 0.05
          (1-s) > (T_sel_flex - T_sel_pbs) / (T_flash × 0.05)
          s < 1 - (T_sel_flex - T_sel_pbs) / (T_flash × 0.05)

        Returns the sparsity BELOW which Flex might beat PBS on kernel savings.
        If this returns > 1, Flex never beats PBS at this L.
        """
        delta_sel = self._t_select_flex(L) - self._t_select_pbs(L)
        t_flash   = self._t_flash(L)
        if t_flash < 1e-9 or delta_sel <= 0:
            return 1.0
        s_crossover = 1.0 - delta_sel / (t_flash * 0.05)
        return min(1.0, max(0.0, s_crossover))

    def route(
        self,
        L: int,
        sparsity: float = 0.0,
        prefer_pbs: bool = True,
    ) -> RoutingDecision:
        """
        Choose between PBS and FlexPrefill for a given (L, sparsity).

        Flash is NOT a routing option — the caller has already decided to use
        a sparse method. This function only selects WHICH sparse backend and
        which parameters to use.

        Decision logic
        --------------
        1. Always have a PBS plan (lowest overhead, always viable).
        2. Consider Flex only if L ≥ flex_min_length.
        3. If both are considered, compare estimated total latency and
           pick the lower one (with a PBS tie-break if prefer_pbs=True).

        Parameters
        ----------
        L          : sequence length
        sparsity   : estimated attention sparsity in [0, 1]
        prefer_pbs : give PBS a 5 % tie-break advantage (lower variance)
        """
        active_frac = max(0.05, 1.0 - sparsity)

        # ── Always compute PBS plan ────────────────────────────────────────
        pbs_thr = pick_pbs_threshold(L, sparsity)
        pbs_seg = pick_pbs_segment_size(L)
        t_pbs   = self._t_pbs_total(L, sparsity)

        pbs_decision = RoutingDecision(
            backend="pbs",
            params={
                "threshold":    pbs_thr,
                "segment_size": pbs_seg,
                "block_size":   PBS_BLOCK_SIZE,
                "use_triton":   True,
                "force_first":  True,
            },
            reason=f"PBS: est {t_pbs:.2f} ms "
                   f"(sel={self._t_select_pbs(L):.2f} ms, "
                   f"kernel≈{self._t_kernel(L, active_frac*0.95):.2f} ms); "
                   f"adaptive thr={pbs_thr}, seg={pbs_seg}",
            critical_sparsity=0.0,
        )

        # ── Consider Flex only when L is large enough ──────────────────────
        if self.flex_min_length > 0 and L < self.flex_min_length:
            return dataclasses.replace(
                pbs_decision,
                reason=pbs_decision.reason +
                       f"; Flex skipped (L={L} < flex_min={self.flex_min_length})",
            )

        flex_tau   = pick_flex_tau(L)
        flex_gamma = pick_flex_gamma(L, sparsity)
        t_flex     = self._t_flex_total(L, sparsity)

        flex_decision = RoutingDecision(
            backend="flex",
            params={
                "gamma":           flex_gamma,
                "tau":             flex_tau,
                "min_budget_frac": 0.0,
                "block_size":      FLEX_BLOCK_SIZE,
            },
            reason=f"Flex: est {t_flex:.2f} ms "
                   f"(sel={self._t_select_flex(L):.2f} ms, "
                   f"kernel≈{self._t_kernel(L, active_frac*0.90):.2f} ms); "
                   f"adaptive tau={flex_tau}, gamma={flex_gamma}",
            critical_sparsity=0.0,
        )

        # ── Pick lower estimated latency ───────────────────────────────────
        threshold = t_pbs * 0.95 if prefer_pbs else t_pbs
        if t_flex < threshold:
            return flex_decision
        return pbs_decision

    # ── Strategy-layer overhead estimates ─────────────────────────────────

    def _t_select_struct(self) -> float:
        """O(1) structural strategy overhead (GlobalSinkWindow / SqrtWindow)."""
        return T_SELECT_STRUCT_MS

    def _t_select_knorm(self, L: int) -> float:
        """O(L·D) K-norm strategy overhead (BlockKNormTopK / CoverageTarget)."""
        return T_SELECT_KNORM_A * L + T_SELECT_KNORM_B

    def _t_select_sampled(self, L: int, sample_rows: int = 16) -> float:
        """O(r·L·D) sampled attention overhead (SampledAttentionTopK)."""
        return T_SELECT_SAMPLED_PER_SAMPLE * sample_rows * (L / T_SELECT_SAMPLED_L0)

    def _t_struct_total(self, L: int, active_frac: float) -> float:
        return self._t_select_struct() + self._t_kernel(L, active_frac)

    def _t_knorm_total(self, L: int, active_frac: float) -> float:
        return self._t_select_knorm(L) + self._t_kernel(L, active_frac)

    def _t_sampled_total(self, L: int, active_frac: float,
                         sample_rows: int = 16) -> float:
        return self._t_select_sampled(L, sample_rows) + self._t_kernel(L, active_frac)

    # ── Full routing (includes mask-strategy tier) ─────────────────────────

    def route_full(
        self,
        L: int,
        sparsity: float = 0.0,
        prefer_pbs: bool = True,
        k_norm_cv: float = 1.0,
    ) -> RoutingDecision:
        """
        Extended routing that considers all strategy tiers.

        This version applies conservative accuracy-aware guardrails:
        more aggressive mask strategies are assigned larger active-set floors and
        small risk penalties so they do not dominate the router purely through
        optimistic latency estimates.
        """
        from length_routing_study.selection_strategies import (
            DecayedWindowSink,
            GlobalSinkWindow,
            SqrtWindow,
            AdaptiveCoverage,
            CoverageTarget,
            HierarchicalKNorm,
            BlockKNormTopK,
            SampledAttentionTopK,
        )

        def _clamp(x: float, lo: float = 0.05, hi: float = 1.0) -> float:
            return max(lo, min(hi, x))

        def _floors_for_length(length: int) -> tuple[float, float, float]:
            if length <= 8_192:
                return 0.34, 0.42, 0.30  # structural, sampled, data-driven
            if length <= 32_768:
                return 0.28, 0.36, 0.24
            return 0.22, 0.30, 0.18

        def _mask_candidate(
            *,
            backend: str,
            inst: Any,
            complexity: str,
            active_est: float,
            family: str,
            risk_mult: float,
            extra_params: dict[str, Any] | None = None,
            t_sel_ms: float | None = None,
        ) -> tuple[float, str, RoutingDecision]:
            floor_struct, floor_sampled, floor_data = _floors_for_length(L)
            family_floor = {
                'structural': floor_struct,
                'sampled': floor_sampled,
                'data': floor_data,
            }[family]
            active_est = max(active_est, family_floor)
            active_est = _clamp(active_est)
            sel_ms = float(inst.t_overhead_ms(L) if t_sel_ms is None else t_sel_ms)
            raw_total = sel_ms + self._t_kernel(L, active_est)
            scored_total = raw_total * risk_mult
            params = {
                'strategy': inst.name,
                'active_frac': round(active_est, 3),
                'risk_mult': round(risk_mult, 2),
            }
            if extra_params:
                params.update(extra_params)
            decision = RoutingDecision(
                backend=backend,
                params=params,
                reason=(
                    f"{inst.name}({complexity}): est {raw_total:.2f} ms "
                    f"(sel={sel_ms:.3f} ms, kernel?{self._t_kernel(L, active_est):.2f} ms, risk?{risk_mult:.2f})"
                ),
                strategy_instance=inst,
                critical_sparsity=0.0,
            )
            return (scored_total, backend, decision)

        active_frac = max(0.05, 1.0 - sparsity)
        num_blocks = max(1, L // 128)

        pbs_decision = self.route(L, sparsity=sparsity, prefer_pbs=prefer_pbs)
        t_pbs = self._t_pbs_total(L, sparsity)
        candidates: list[tuple[float, str, RoutingDecision]] = [(t_pbs, 'pbs', pbs_decision)]

        _knorm_ok = k_norm_cv >= 0.40

        if True:
            window_blocks = max(8, round(num_blocks * 0.15))
            stride_blocks = max(2, round(num_blocks * 0.02))
            structural_candidates: list[tuple[float, str, RoutingDecision]] = []
            structural_allowed = (_knorm_ok and sparsity >= 0.60 and L >= 16_384) or (L >= 65_536 and sparsity >= 0.72)

            if structural_allowed:
                structural_candidates.append(
                    _mask_candidate(
                        backend='structural',
                        inst=GlobalSinkWindow(sink_blocks=2, window_blocks=window_blocks),
                        complexity='O(1)',
                        active_est=_clamp(0.72 * active_frac + 0.14, 0.30, 0.82),
                        family='structural',
                        risk_mult=1.55,
                        extra_params={'sink_blocks': 2, 'window_blocks': window_blocks},
                    )
                )

                if sparsity >= 0.68:
                    structural_candidates.append(
                        _mask_candidate(
                            backend='structural',
                            inst=SqrtWindow(sink_blocks=2),
                            complexity='O(1)',
                            active_est=_clamp(0.62 * active_frac + 0.18, 0.28, 0.72),
                            family='structural',
                            risk_mult=1.80,
                            extra_params={'sink_blocks': 2},
                        )
                    )

                if L >= 16_384:
                    structural_candidates.append(
                        _mask_candidate(
                            backend='structural',
                            inst=DecayedWindowSink(sink_blocks=2, window_blocks=window_blocks, stride_blocks=stride_blocks),
                            complexity='O(1)',
                            active_est=_clamp(0.66 * active_frac + 0.14, 0.30, 0.78),
                            family='structural',
                            risk_mult=1.60,
                            extra_params={'sink_blocks': 2, 'window_blocks': window_blocks, 'stride_blocks': stride_blocks},
                        )
                    )

                candidates.append(min(structural_candidates, key=lambda x: x[0]))

            topk_scale = 0.90 if _knorm_ok else 1.00
            topk_frac = round(_clamp(active_frac * topk_scale, 0.40, 0.70), 2)
            cov_target = round(_clamp(0.91 + active_frac * 0.08 + (0.02 if not _knorm_ok else 0.0), 0.93, 0.99), 2)
            cov_base = round(_clamp(0.89 + active_frac * 0.10 + (0.02 if not _knorm_ok else 0.0), 0.92, 0.98), 2)
            cov_min = round(_clamp(cov_base - 0.18, 0.70, 0.86), 2)
            coarse_factor = 8 if L >= 32_768 else 4
            coarse_keep = 0.74 if _knorm_ok else 0.82
            fine_keep = 0.64 if _knorm_ok else 0.72
            if L <= 16_384:
                topk_frac = round(max(topk_frac, 0.55), 2)
                cov_target = round(max(cov_target, 0.96), 2)
                cov_base = round(max(cov_base, 0.95), 2)
                cov_min = round(max(cov_min, 0.80), 2)
                coarse_keep = min(0.90, coarse_keep + 0.08)
                fine_keep = min(0.82, fine_keep + 0.08)

            data_candidates = [
                _mask_candidate(
                    backend='knorm',
                    inst=BlockKNormTopK(topk_frac=topk_frac, sink_blocks=1),
                    complexity='O(L?D)',
                    active_est=_clamp((0.62 if _knorm_ok else 0.68) * active_frac + 0.16, 0.30, 0.72),
                    family='data',
                    risk_mult=1.08 if _knorm_ok else 1.14,
                    extra_params={'topk_frac': topk_frac, 'sink_blocks': 1},
                ),
                _mask_candidate(
                    backend='coverage',
                    inst=CoverageTarget(target_coverage=cov_target, sink_blocks=1),
                    complexity='O(L?D)',
                    active_est=_clamp((0.60 if _knorm_ok else 0.66) * active_frac + 0.18 + (0.04 if L <= 16_384 else 0.0), 0.36, 0.78),
                    family='data',
                    risk_mult=(0.98 if L <= 16_384 else 1.00) if _knorm_ok else (1.00 if L <= 16_384 else 1.03),
                    extra_params={'target_coverage': cov_target, 'sink_blocks': 1},
                ),
                _mask_candidate(
                    backend='coverage',
                    inst=AdaptiveCoverage(base_coverage=cov_base, min_coverage=cov_min, alpha=0.09, sink_blocks=1),
                    complexity='O(L?D)',
                    active_est=_clamp((0.58 if _knorm_ok else 0.64) * active_frac + 0.16 + (0.04 if L <= 16_384 else 0.0), 0.34, 0.74),
                    family='data',
                    risk_mult=(0.98 if L <= 16_384 else 1.00) if _knorm_ok else (1.00 if L <= 16_384 else 1.03),
                    extra_params={'base_coverage': cov_base, 'min_coverage': cov_min, 'alpha': 0.09, 'sink_blocks': 1},
                ),
                _mask_candidate(
                    backend='knorm',
                    inst=HierarchicalKNorm(coarse_factor=coarse_factor, coarse_keep=coarse_keep, fine_keep=fine_keep, sink_blocks=1),
                    complexity='O(L?D)',
                    active_est=_clamp((0.60 if _knorm_ok else 0.66) * active_frac + 0.16 + (0.05 if L <= 16_384 else 0.0), 0.34, 0.74),
                    family='data',
                    risk_mult=(1.04 if L <= 16_384 else 1.00) if _knorm_ok else (1.08 if L <= 16_384 else 1.02),
                    extra_params={'coarse_factor': coarse_factor, 'coarse_keep': coarse_keep, 'fine_keep': fine_keep, 'sink_blocks': 1},
                ),
            ]
            candidates.append(min(data_candidates, key=lambda x: x[0]))

        sampled_allowed = (not _knorm_ok) and sparsity <= 0.35
        if sampled_allowed:
            sample_rows = 32 if L <= 16_384 else (16 if L <= 65_536 else 8)
            sampled_topk_frac = round(_clamp(active_frac * 0.98, 0.52, 0.70), 2)
            sampled_inst = SampledAttentionTopK(
                sample_rows=sample_rows,
                topk_frac=sampled_topk_frac,
                sink_blocks=1,
                seed=0,
            )
            t_sampled_sel = self._t_select_sampled(L, sample_rows=sample_rows)
            candidates.append(
                _mask_candidate(
                    backend='sampled',
                    inst=sampled_inst,
                    complexity='O(r?L?D)',
                    active_est=_clamp(0.72 * active_frac + 0.16, 0.38, 0.78),
                    family='sampled',
                    risk_mult=1.38,
                    extra_params={'sample_rows': sample_rows, 'topk_frac': sampled_topk_frac, 'sink_blocks': 1},
                    t_sel_ms=t_sampled_sel,
                )
            )

        if self.flex_min_length <= 0 or L >= self.flex_min_length:
            t_flex = self._t_flex_total(L, sparsity)
            flex_tau = pick_flex_tau(L)
            flex_gamma = pick_flex_gamma(L, sparsity)
            flex_decision = RoutingDecision(
                backend='flex',
                params={
                    'gamma': flex_gamma,
                    'tau': flex_tau,
                    'min_budget_frac': 0.0,
                    'block_size': FLEX_BLOCK_SIZE,
                },
                reason=(
                    f"Flex(O(r?L?D)): est {t_flex:.2f} ms "
                    f"(sel={self._t_select_flex(L):.2f} ms)"
                ),
                critical_sparsity=0.0,
            )
            candidates.append((t_flex, 'flex', flex_decision))

        best_t, best_name, best_decision = min(candidates, key=lambda x: x[0])
        return dataclasses.replace(
            best_decision,
            reason=(
                best_decision.reason
                + ' | all candidates: '
                + ', '.join(
                    f"{n}={t:.2f}ms" for t, n, _ in sorted(candidates, key=lambda x: x[0])
                )
            ),
        )

    def overhead_crossover_lengths(self) -> dict[str, dict]:
        """
        Compute the sequence lengths at which each strategy's overhead
        equals PBS's overhead — the 'crossover length'.

        Returns a dict mapping strategy name → {crossover_L, comparison}.

        The key question for the length-effect study:
          "At L_crossover, does the higher-quality selection from O(L·D)
           strategies pay off vs the simpler PBS selection?"
        """
        from typing import Optional

        def solve_linear_equal(a1: float, b1: float,
                               a2: float, b2: float) -> Optional[float]:
            """Solve a1*L + b1 = a2*L + b2 for L."""
            if abs(a1 - a2) < 1e-15:
                return None
            return (b2 - b1) / (a1 - a2)

        # PBS overhead: pbs_a * L + pbs_b
        pbs_a = self.pbs_select_a
        pbs_b = self.pbs_select_b

        results: dict[str, dict] = {}

        # Structural vs PBS: struct is constant
        L_struct = solve_linear_equal(0.0, T_SELECT_STRUCT_MS, pbs_a, pbs_b)
        results["structural_vs_pbs"] = {
            "strategy_a": "GlobalSinkWindow (O(1))",
            "strategy_b": "PBS (O(L))",
            "crossover_L": round(L_struct) if L_struct and L_struct > 0 else None,
            "note": "below this L, structural overhead < PBS overhead",
        }

        # KNorm vs PBS: knorm_a * L + knorm_b vs pbs_a * L + pbs_b
        L_knorm = solve_linear_equal(T_SELECT_KNORM_A, T_SELECT_KNORM_B, pbs_a, pbs_b)
        results["knorm_vs_pbs"] = {
            "strategy_a": "BlockKNormTopK (O(L·D))",
            "strategy_b": "PBS (O(L))",
            "crossover_L": round(L_knorm) if L_knorm and L_knorm > 0 else None,
            "note": (
                "below this L, KNorm overhead < PBS overhead "
                "(KNorm per-token cost is lower than PBS per-token cost)"
            ),
        }

        # Sampled(r=16) vs PBS
        s16_a = T_SELECT_SAMPLED_PER_SAMPLE * 16 / T_SELECT_SAMPLED_L0
        s16_b = 0.0
        L_s16 = solve_linear_equal(s16_a, s16_b, pbs_a, pbs_b)
        results["sampled_r16_vs_pbs"] = {
            "strategy_a": "SampledAttentionTopK(r=16)",
            "strategy_b": "PBS (O(L))",
            "crossover_L": round(L_s16) if L_s16 and L_s16 > 0 else None,
            "note": "above this L, SampledTopK(r=16) overhead exceeds PBS",
        }

        return results

    def build_routing_table(
        self,
        lengths: list[int],
        sparsity_bins: list[float] | None = None,
    ) -> list[dict]:
        """
        Pre-compute routing decisions for a grid of (L, sparsity) combinations.

        Returns a list of dicts suitable for JSON serialisation and Canvas
        visualisation.
        """
        if sparsity_bins is None:
            sparsity_bins = [0.0, 0.20, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]

        rows = []
        for L in lengths:
            for s in sparsity_bins:
                d = self.route(L, sparsity=s)
                rows.append({
                    "seq_len":          L,
                    "sparsity":         s,
                    "backend":          d.backend,
                    "params":           d.params,
                    "reason":           d.reason,
                    "t_pbs_est":        round(self._t_pbs_total(L, s), 3),
                    "t_flex_est":       round(self._t_flex_total(L, s), 3),
                    "t_pbs_sel_est":    round(self._t_select_pbs(L), 3),
                    "t_flex_sel_est":   round(self._t_select_flex(L), 3),
                    "flex_crossover_s": round(self._crossover_sparsity_flex_over_pbs(L), 4),
                })
        return rows

    def calibrate_from_full_measurements(
        self,
        flash_ms: list[tuple[int, float]],
        pbs_total_ms: list[tuple[int, float]],
        pbs_select_ms: list[tuple[int, float]] | None = None,
        flex_total_ms: list[tuple[int, float]] | None = None,
    ) -> "LengthAwareRouter":
        """
        Calibrate all model coefficients from empirical (L, t_ms) measurements.

        PBS selection overhead is extracted as:
          t_select_pbs(L) = t_pbs_dense(L) - t_flash(L)
        where t_pbs_dense ≈ PBS with threshold=1.0.

        Alternatively, if pbs_select_ms is given directly, it takes precedence.

        Flex overhead is estimated as:
          t_select_flex(L) = t_flex_total(L) - t_flash(L) × (1 - sparsity)
        Approximated as:
          t_select_flex(L) ≈ t_flex_total(L) - t_flash(L)  (conservative upper bound)
        """
        # Calibrate Flash
        router_v1 = self.calibrate_from_measurements(flash_measurements=flash_ms)

        # Calibrate PBS selection overhead (linear fit)
        if pbs_select_ms:
            select_data = pbs_select_ms
        elif pbs_total_ms and flash_ms:
            flash_dict = {L: t for L, t in flash_ms}
            select_data = [
                (L, max(0.0, t_pbs - flash_dict.get(L, 0.0)))
                for L, t_pbs in pbs_total_ms
                if L in flash_dict
            ]
        else:
            select_data = []

        if select_data:
            router_v1 = router_v1.calibrate_from_measurements(
                flash_measurements=[],
                pbs_select_measurements=select_data,
            )

        # Calibrate Flex overhead (linear fit to t_flex - t_flash)
        if flex_total_ms and flash_ms:
            flash_dict = {L: t for L, t in flash_ms}
            flex_overhead = [
                (L, max(0.0, t_flex - flash_dict.get(L, 0.0)))
                for L, t_flex in flex_total_ms
                if L in flash_dict
            ]
            # Fit linear: t_overhead = a * L + b
            Ls = [m[0] for m in flex_overhead]
            Ts = [m[1] for m in flex_overhead]
            n = len(Ls)
            if n >= 2:
                sum_L  = sum(Ls)
                sum_T  = sum(Ts)
                sum_L2 = sum(l * l for l in Ls)
                sum_LT = sum(l * t for l, t in zip(Ls, Ts))
                denom  = n * sum_L2 - sum_L ** 2
                if abs(denom) > 1e-12:
                    new_a = max(0.0, (n * sum_LT - sum_L * sum_T) / denom)
                    new_b = max(0.0, (sum_T - new_a * sum_L) / n)
                    return dataclasses.replace(router_v1,
                                               flex_select_a=new_a,
                                               flex_select_b_val=new_b)

        return router_v1

    def calibrate_from_measurements(
        self,
        flash_measurements: list[tuple[int, float]],
        pbs_select_measurements: list[tuple[int, float]] | None = None,
    ) -> "LengthAwareRouter":
        """
        Fit model coefficients from measured data points.

        Parameters
        ----------
        flash_measurements : list of (L, t_flash_ms) pairs
        pbs_select_measurements : list of (L, t_select_ms) pairs

        Returns
        -------
        A new LengthAwareRouter with calibrated coefficients.
        """
        import math

        # Fit FLASH_ALPHA: T_flash = alpha * L²
        if flash_measurements:
            # Least-squares fit in log space: log(T) = log(alpha) + 2 * log(L)
            log_pairs = [(math.log(L), math.log(t)) for L, t in flash_measurements if L > 0 and t > 0]
            if log_pairs:
                # Simple regression: slope forced to 2 (quadratic)
                n = len(log_pairs)
                sum_logL = sum(p[0] for p in log_pairs)
                sum_logT = sum(p[1] for p in log_pairs)
                log_alpha = (sum_logT - 2.0 * sum_logL) / n
                new_alpha = math.exp(log_alpha)
            else:
                new_alpha = self.flash_alpha
        else:
            new_alpha = self.flash_alpha

        # Fit PBS linear selection overhead: T_select = a * L + b
        if pbs_select_measurements:
            Ls = [m[0] for m in pbs_select_measurements]
            Ts = [m[1] for m in pbs_select_measurements]
            n = len(Ls)
            if n >= 2:
                sum_L  = sum(Ls)
                sum_T  = sum(Ts)
                sum_L2 = sum(l * l for l in Ls)
                sum_LT = sum(l * t for l, t in zip(Ls, Ts))
                denom  = n * sum_L2 - sum_L ** 2
                if abs(denom) > 1e-12:
                    new_a = (n * sum_LT - sum_L * sum_T) / denom
                    new_b = (sum_T - new_a * sum_L) / n
                    new_a = max(0.0, new_a)
                    new_b = max(0.0, new_b)
                else:
                    new_a, new_b = self.pbs_select_a, self.pbs_select_b
            else:
                new_a, new_b = self.pbs_select_a, self.pbs_select_b
        else:
            new_a, new_b = self.pbs_select_a, self.pbs_select_b

        return dataclasses.replace(
            self,
            flash_alpha=new_alpha,
            pbs_select_a=new_a,
            pbs_select_b=new_b,
        )
