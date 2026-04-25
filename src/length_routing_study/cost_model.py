from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Mapping

from .sparse_plan import (
    BACKEND_DENSE,
    BACKEND_FLEX_PREFILL_FLEX,
    BACKEND_FLEX_PREFILL_TRITON,
    BACKEND_PBS_ATTENTION,
    BACKEND_STANDALONE_FLEX,
    BackendEstimate,
    SparseBackendConfig,
    WorkloadFeatures,
)


@dataclass(frozen=True)
class TheoryModelProfile:
    family: str
    num_layers: int = 36
    linear_ms_per_token_per_layer: float = 1.0e-5
    dense_block_ms_per_layer: float = 0.0012
    dense_mem_per_token_gb: float = 7.0e-5
    pbs_reorder_ms_per_token_per_layer: float = 3.4e-6
    pbs_select_ms_per_block_pair_per_layer: float = 2.0e-5
    pbs_kernel_ms_per_active_block_per_layer: float = 1.8e-4
    pbs_extra_mem_per_block_gb: float = 8.0e-4
    flex_stats_ms_per_token_per_layer: float = 2.8e-6
    flex_mask_ms_per_block_log_per_layer: float = 0.003
    flex_kernel_ms_per_active_block_per_layer: float = 1.3e-4
    flex_extra_mem_per_block_gb: float = 9.0e-5
    standalone_flex_block_ms_per_layer: float = 0.0015
    standalone_flex_extra_mem_per_block_gb: float = 0.0025
    base_memory_gb: float = 5.5
    stability_dense: float = 0.99
    stability_pbs: float = 0.81
    stability_flexprefill: float = 0.9
    stability_flex_experimental: float = 0.68
    migration_dense: float = 0.95
    migration_pbs: float = 0.62
    migration_flexprefill: float = 0.88
    migration_flex_experimental: float = 0.72
    pbs_selection_pressure_scale: float = 0.0
    pbs_selection_pressure_power: float = 1.0
    flex_mask_long_context_discount: float = 0.0
    flex_mask_discount_floor: float = 1.0


DEFAULT_PROFILES = {
    "qwen2": TheoryModelProfile(
        family="qwen2",
        stability_pbs=0.86,
        migration_pbs=0.78,
        pbs_selection_pressure_scale=0.20,
        pbs_selection_pressure_power=1.2,
        flex_mask_long_context_discount=0.25,
        flex_mask_discount_floor=0.35,
    ),
    "llama": TheoryModelProfile(
        family="llama",
        pbs_select_ms_per_block_pair_per_layer=2.3e-5,
        pbs_extra_mem_per_block_gb=7.0e-4,
        flex_mask_ms_per_block_log_per_layer=0.0027,
        migration_flexprefill=0.9,
    ),
    "glm": TheoryModelProfile(
        family="glm",
        pbs_select_ms_per_block_pair_per_layer=2.2e-5,
        migration_pbs=0.58,
        migration_flexprefill=0.84,
    ),
    "generic": TheoryModelProfile(family="generic"),
}


def get_profile(model_family: str) -> TheoryModelProfile:
    return DEFAULT_PROFILES.get(model_family, DEFAULT_PROFILES["generic"])


def infer_workload_features(
    prompt_tokens: int,
    cfg: SparseBackendConfig,
    sparsity_ratio: float = 0.0,
    kv_norm_cv: float = 0.0,
    sparsity_source: str = "geometric",
) -> WorkloadFeatures:
    """
    Derive workload features from sequence length and (optionally) a sparsity hint.

    Parameters
    ----------
    sparsity_ratio : float in [0, 1]
        Estimated fraction of attention that is near-zero and can be pruned.
        0 = unknown (falls back to geometric estimate).
        Computed by ``sparsity_estimator.estimate_sparsity`` from real Q/K data.
    kv_norm_cv : float
        Coefficient of variation of K norms (proxy for attention concentration).
    sparsity_source : str
        How ``sparsity_ratio`` was obtained: "geometric" | "estimated" | "measured".
    """
    block_size = cfg.block_size
    num_blocks = max(1, math.ceil(prompt_tokens / block_size))
    recent_window_blocks = max(1, math.ceil(min(prompt_tokens, cfg.keep_recent_tokens) / block_size))
    local_window_blocks = max(1, math.ceil(min(prompt_tokens, 2048) / block_size))
    min_budget_blocks = max(1, math.ceil(cfg.flex_prefill_min_budget / block_size))

    # ── Active-block estimation ────────────────────────────────────────────────
    # When sparsity_ratio is known (> 0), use it directly to scale the maximum
    # possible active blocks (num_blocks²).  Otherwise fall back to the geometric
    # sqrt approximation that was used before.
    if sparsity_ratio > 0.0:
        active_fraction = max(0.05, 1.0 - sparsity_ratio)
        max_pairs = num_blocks * num_blocks
        # Flex retains a minimum window regardless of sparsity
        flex_min  = max(recent_window_blocks, min_budget_blocks)
        estimated_flex_active_blocks = max(
            num_blocks * flex_min,
            round(max_pairs * active_fraction),
        )
        # PBS selector must still examine most block pairs even if kernel skips many
        # → selection complexity stays quadratic, but kernel active set is sparsity-scaled
        estimated_pbs_active_blocks = max(
            num_blocks * recent_window_blocks,
            round(max_pairs * active_fraction),
        )
    else:
        active_fraction = 1.0   # unknown — geometric fallback
        flex_growth = max(recent_window_blocks, min_budget_blocks + math.ceil(math.sqrt(num_blocks) * 2.0))
        estimated_flex_active_blocks = min(num_blocks * num_blocks, num_blocks * flex_growth)
        pbs_growth = max(recent_window_blocks, math.ceil(math.sqrt(num_blocks) * 3.0) + 2)
        estimated_pbs_active_blocks = min(num_blocks * num_blocks, num_blocks * pbs_growth)

    flex_mask_complexity = num_blocks * math.log2(num_blocks + 1)
    pbs_selection_complexity = float(num_blocks * num_blocks)
    long_context_pressure = max(0.0, prompt_tokens - 8192) / 8192.0

    return WorkloadFeatures(
        prompt_tokens=prompt_tokens,
        block_size=block_size,
        num_blocks=num_blocks,
        recent_window_blocks=recent_window_blocks,
        local_window_blocks=local_window_blocks,
        estimated_flex_active_blocks=estimated_flex_active_blocks,
        estimated_pbs_active_blocks=estimated_pbs_active_blocks,
        flex_mask_complexity=flex_mask_complexity,
        pbs_selection_complexity=pbs_selection_complexity,
        long_context_pressure=long_context_pressure,
        sparsity_ratio=sparsity_ratio,
        active_fraction=active_fraction,
        kv_norm_cv=kv_norm_cv,
        sparsity_source=sparsity_source,
        metadata={"min_budget_blocks": min_budget_blocks},
    )


def _memory_limit_gb(model_memory_budget_gb: float, safety_margin: float) -> float:
    return model_memory_budget_gb * safety_margin


def _pbs_selection_pressure_multiplier(profile: TheoryModelProfile, features: WorkloadFeatures) -> float:
    if profile.pbs_selection_pressure_scale <= 0.0 or features.long_context_pressure <= 0.0:
        return 1.0
    return 1.0 + profile.pbs_selection_pressure_scale * (
        features.long_context_pressure ** profile.pbs_selection_pressure_power
    )


def _flex_mask_discount_multiplier(profile: TheoryModelProfile, features: WorkloadFeatures) -> float:
    if profile.flex_mask_long_context_discount <= 0.0 or features.long_context_pressure <= 0.0:
        return 1.0
    discounted = 1.0 / (1.0 + profile.flex_mask_long_context_discount * features.long_context_pressure)
    return max(profile.flex_mask_discount_floor, discounted)


def _sparsity_stability_boost(sparsity_ratio: float) -> float:
    """
    When input attention is genuinely sparse, PBS/Flex approximation errors
    are smaller (the skipped blocks carry little mass).  Boost stability score
    proportionally to measured sparsity.

    Returns an additive boost in [0, 0.12].
    """
    return min(0.12, sparsity_ratio * 0.15)


def estimate_backend(
    profile: TheoryModelProfile,
    features: WorkloadFeatures,
    backend: str,
    objective: str,
    model_memory_budget_gb: float,
    safety_margin: float,
) -> BackendEstimate:
    layers = profile.num_layers
    tokens = features.prompt_tokens
    num_blocks = features.num_blocks
    linear_ms = profile.linear_ms_per_token_per_layer * tokens * layers
    dense_mem = profile.base_memory_gb + profile.dense_mem_per_token_gb * tokens

    sparsity_boost = _sparsity_stability_boost(features.sparsity_ratio)

    if backend == BACKEND_DENSE:
        dense_attention_ms = profile.dense_block_ms_per_layer * (num_blocks ** 2) * layers
        latency = linear_ms + dense_attention_ms
        memory = dense_mem
        # Dense becomes LESS stable (higher penalty) when input IS sparse —
        # it wastes compute on near-zero blocks.  No boost for dense.
        stability = profile.stability_dense
        migration = profile.migration_dense
        components = {
            "linear": linear_ms,
            "dense_attention": dense_attention_ms,
        }
        notes = (
            "Dense keeps the most faithful execution path, but its attention cost grows quadratically with the block grid.",
        )
    elif backend == BACKEND_PBS_ATTENTION:
        reorder_ms = profile.pbs_reorder_ms_per_token_per_layer * tokens * layers
        selection_pressure = _pbs_selection_pressure_multiplier(profile, features)
        select_ms = (
            profile.pbs_select_ms_per_block_pair_per_layer
            * features.pbs_selection_complexity
            * layers
            * selection_pressure
        )
        kernel_ms = profile.pbs_kernel_ms_per_active_block_per_layer * features.estimated_pbs_active_blocks * layers
        latency = linear_ms + reorder_ms + select_ms + kernel_ms
        memory = dense_mem + profile.pbs_extra_mem_per_block_gb * num_blocks
        # Sparse input → PBS approximation is more accurate → higher stability
        stability = min(0.97, max(0.35,
            profile.stability_pbs
            - 0.03 * features.long_context_pressure
            + sparsity_boost
        ))
        migration = profile.migration_pbs
        components = {
            "linear": linear_ms,
            "reorder": reorder_ms,
            "selection": select_ms,
            "kernel": kernel_ms,
        }
        notes = [
            "PBS can win when sequence lengths are still moderate because its sparse kernel is efficient once the permutation overhead is amortized.",
            "Its theoretical weak point is the block-selection stage, which grows with the block graph rather than only with the final active set.",
        ]
        if selection_pressure > 1.0:
            notes.append(
                f"Applied a long-context selector pressure multiplier ({selection_pressure:.2f}x) to reflect the rapidly growing block-selection overhead observed on the current stack."
            )
        notes = tuple(notes)
    elif backend == BACKEND_FLEX_PREFILL_TRITON:
        stats_ms = profile.flex_stats_ms_per_token_per_layer * tokens * layers
        mask_discount = _flex_mask_discount_multiplier(profile, features)
        mask_ms = profile.flex_mask_ms_per_block_log_per_layer * features.flex_mask_complexity * layers * mask_discount
        kernel_ms = profile.flex_kernel_ms_per_active_block_per_layer * features.estimated_flex_active_blocks * layers
        latency = linear_ms + stats_ms + mask_ms + kernel_ms
        memory = dense_mem + profile.flex_extra_mem_per_block_gb * num_blocks
        # Flex budget-based selection benefits greatly from sparse input
        stability = min(0.97, max(0.45,
            profile.stability_flexprefill
            - 0.01 * max(0.0, features.long_context_pressure - 2.0)
            + sparsity_boost * 0.8   # Flex benefits somewhat less than PBS
        ))
        migration = profile.migration_flexprefill
        components = {
            "linear": linear_ms,
            "stats": stats_ms,
            "mask": mask_ms,
            "kernel": kernel_ms,
        }
        notes = [
            "FlexPrefill spends more on budgeting and mask construction up front, but its active-set growth is closer to budget-driven sparse coverage than to the full block graph.",
            "That makes it theoretically better suited to longer contexts where selection overhead dominates kernel time.",
        ]
        if mask_discount < 1.0:
            notes.append(
                f"Applied a long-context mask discount ({mask_discount:.2f}x) to reflect amortized planner and mask-setup reuse on the current stack."
            )
        notes = tuple(notes)
    elif backend == BACKEND_FLEX_PREFILL_FLEX:
        stats_ms = profile.flex_stats_ms_per_token_per_layer * tokens * layers
        mask_discount = _flex_mask_discount_multiplier(profile, features)
        mask_ms = (profile.flex_mask_ms_per_block_log_per_layer * 1.35) * features.flex_mask_complexity * layers * mask_discount
        kernel_ms = (profile.flex_kernel_ms_per_active_block_per_layer * 0.95) * features.estimated_flex_active_blocks * layers
        latency = linear_ms + stats_ms + mask_ms + kernel_ms
        memory = dense_mem + (profile.flex_extra_mem_per_block_gb * 6.0) * num_blocks
        stability = max(0.25, profile.stability_flex_experimental - 0.05 * features.long_context_pressure)
        migration = profile.migration_flex_experimental
        components = {
            "linear": linear_ms,
            "stats": stats_ms,
            "mask": mask_ms,
            "kernel": kernel_ms,
        }
        notes = [
            "FlexPrefill-FlexAttention keeps the same sparse planner as FlexPrefill-Triton, but pays extra for block-mask realization and has a higher long-context OOM risk on the current local stack.",
        ]
        if mask_discount < 1.0:
            notes.append(
                f"Applied a long-context mask discount ({mask_discount:.2f}x) to reflect amortized planner and mask-setup reuse on the current stack."
            )
        notes = tuple(notes)
    elif backend == BACKEND_STANDALONE_FLEX:
        dense_attention_ms = profile.standalone_flex_block_ms_per_layer * (num_blocks ** 2) * layers
        latency = linear_ms + dense_attention_ms
        memory = dense_mem + profile.standalone_flex_extra_mem_per_block_gb * num_blocks
        stability = max(0.2, profile.stability_flex_experimental - 0.06 * features.long_context_pressure)
        migration = 0.55
        components = {
            "linear": linear_ms,
            "dense_like_blockmask": dense_attention_ms,
        }
        notes = (
            "Standalone FlexAttention is included as an experimental upper path, but without a sparse planner its long-context cost still tracks the dense block grid.",
        )
    else:
        raise ValueError(f"Unsupported backend {backend!r}")

    memory_limit = _memory_limit_gb(model_memory_budget_gb=model_memory_budget_gb, safety_margin=safety_margin)
    feasible = memory <= memory_limit
    reject_reason = None if feasible else f"estimated_memory_gb={memory:.2f} exceeds budget={memory_limit:.2f}"

    objective = objective.lower()
    stability_penalty = (1.0 - stability) * 220.0
    migration_penalty = (1.0 - migration) * 60.0
    memory_penalty = max(0.0, memory - memory_limit * 0.7) * 30.0
    if objective == "speed":
        score = latency + memory_penalty + stability_penalty * 0.4
    elif objective == "stability":
        score = latency * 0.55 + stability_penalty * 1.7 + migration_penalty * 0.8 + memory_penalty
    elif objective == "memory":
        score = latency * 0.75 + memory * 45.0 + stability_penalty * 0.7
    else:
        score = latency + stability_penalty + migration_penalty + memory_penalty

    if not feasible:
        score += 10_000.0

    return BackendEstimate(
        backend=backend,
        estimated_latency_ms=latency,
        estimated_memory_gb=memory,
        stability_score=stability,
        migration_score=migration,
        score=score,
        feasible=feasible,
        reject_reason=reject_reason,
        components_ms=components,
        notes=notes,
    )
