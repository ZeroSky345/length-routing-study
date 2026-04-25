from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Optional, Tuple

BACKEND_DENSE = "dense"
BACKEND_FLEX_PREFILL_TRITON = "flex_prefill_triton"
BACKEND_FLEX_PREFILL_FLEX = "flex_prefill_flex_attention"
BACKEND_PBS_ATTENTION = "pbs_attention"
BACKEND_STANDALONE_FLEX = "standalone_flex_attention"

SUPPORTED_BACKENDS = (
    BACKEND_DENSE,
    BACKEND_FLEX_PREFILL_TRITON,
    BACKEND_FLEX_PREFILL_FLEX,
    BACKEND_PBS_ATTENTION,
    BACKEND_STANDALONE_FLEX,
)

BACKEND_LABELS = {
    BACKEND_DENSE: "Dense / Original Attention",
    BACKEND_FLEX_PREFILL_TRITON: "FlexPrefill-Triton",
    BACKEND_FLEX_PREFILL_FLEX: "FlexPrefill-FlexAttention",
    BACKEND_PBS_ATTENTION: "PBS-Attention",
    BACKEND_STANDALONE_FLEX: "Standalone FlexAttention",
}


@dataclass(frozen=True)
class SparseBackendConfig:
    block_size: int = 128
    flex_prefill_gamma: float = 0.9
    flex_prefill_tau: float = 0.1
    flex_prefill_min_budget: int = 512
    flex_prefill_max_budget: Optional[int] = None
    pbs_segment_size: int = 256
    pbs_threshold: float = 0.9
    pbs_force_select_first_block: bool = True
    pbs_use_triton: bool = True
    keep_recent_tokens: int = 4096
    keep_first_block: bool = True

    def flex_prefill_kwargs(self, backend: str) -> dict[str, Any]:
        if backend not in {BACKEND_FLEX_PREFILL_TRITON, BACKEND_FLEX_PREFILL_FLEX}:
            raise ValueError(f"Backend {backend!r} does not use FlexPrefill patch kwargs")
        return {
            "block_size": self.block_size,
            "flex_prefill_gamma": self.flex_prefill_gamma,
            "flex_prefill_tau": self.flex_prefill_tau,
            "flex_prefill_min_budget": self.flex_prefill_min_budget,
            "flex_prefill_max_budget": self.flex_prefill_max_budget,
            "flex_prefill_backend": "triton" if backend == BACKEND_FLEX_PREFILL_TRITON else "flex_attention",
        }

    def pbs_kwargs(self) -> dict[str, Any]:
        return {
            "block_size": self.block_size,
            "segment_size": self.pbs_segment_size,
            "threshold": self.pbs_threshold,
            "force_select_first_block": self.pbs_force_select_first_block,
            "use_triton": self.pbs_use_triton,
        }


@dataclass(frozen=True)
class WorkloadFeatures:
    prompt_tokens: int
    block_size: int
    num_blocks: int
    recent_window_blocks: int
    local_window_blocks: int
    estimated_flex_active_blocks: int
    estimated_pbs_active_blocks: int
    flex_mask_complexity: float
    pbs_selection_complexity: float
    long_context_pressure: float
    # ── Sparsity fields (optional; filled by sparsity_estimator) ─────────────
    sparsity_ratio: float = 0.0          # 0 = unknown / dense; 1 = fully sparse
    active_fraction: float = 1.0         # 1 - sparsity_ratio (clamped ≥ 0.05)
    kv_norm_cv: float = 0.0             # coefficient of variation of K norms
    sparsity_source: str = "geometric"  # "geometric" | "estimated" | "measured"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BackendEstimate:
    backend: str
    estimated_latency_ms: float
    estimated_memory_gb: float
    stability_score: float
    migration_score: float
    score: float
    feasible: bool = True
    reject_reason: Optional[str] = None
    components_ms: Mapping[str, float] = field(default_factory=dict)
    notes: Tuple[str, ...] = field(default_factory=tuple)

    @property
    def backend_label(self) -> str:
        return BACKEND_LABELS[self.backend]

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["backend_label"] = self.backend_label
        return payload


@dataclass(frozen=True)
class SparsePlan:
    backend: str
    model_family: str
    model_name: str
    prompt_tokens: int
    objective: str
    workload: WorkloadFeatures
    selected_estimate: BackendEstimate
    candidate_estimates: Tuple[BackendEstimate, ...]
    backend_config: SparseBackendConfig = field(default_factory=SparseBackendConfig)
    fallback_backends: Tuple[str, ...] = field(default_factory=tuple)
    constraints: Mapping[str, Any] = field(default_factory=dict)
    notes: Tuple[str, ...] = field(default_factory=tuple)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    dispatcher_version: str = "theory-v0"

    def __post_init__(self) -> None:
        if self.backend not in SUPPORTED_BACKENDS:
            raise ValueError(f"Unsupported backend {self.backend!r}")

    @property
    def backend_label(self) -> str:
        return BACKEND_LABELS[self.backend]

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["backend_label"] = self.backend_label
        return payload
