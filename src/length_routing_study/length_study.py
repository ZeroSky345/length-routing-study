"""
Joint theory + empirical length study.

``LengthStudyRunner`` ties together the ``TheoryDrivenDispatcher`` and the
``empirical_sweep.run_sweep``.  For each sequence length it produces a
``LengthCell`` that contains:

  * the theory ``SparsePlan`` (what the cost model predicts should win)
  * all ``EmpiricalRecord``s (what the real kernels actually measured)
  * derived comparison metrics (theory vs empirical agreement, error ratios)

``LengthStudyResult`` is the top-level artefact written to disk.
"""
from __future__ import annotations

import json
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .dispatcher import TheoryDrivenDispatcher
from .empirical_sweep import (
    EmpiricalRecord,
    FlexConfig,
    PBSConfig,
    default_flex_grid,
    default_pbs_grid,
    load_qkv,
    run_sweep,
)
from .sparsity_estimator import SparsityProfile, estimate_sparsity
from .sparse_plan import (
    BACKEND_DENSE,
    BACKEND_FLEX_PREFILL_TRITON,
    BACKEND_PBS_ATTENTION,
    SparseBackendConfig,
    SparsePlan,
)


# ── Mapping: theory backend → empirical backend string ───────────────────────
_THEORY_TO_EMPIRICAL_BACKEND: dict[str, str] = {
    BACKEND_DENSE:              "flash",
    BACKEND_FLEX_PREFILL_TRITON:"flexprefill",
    BACKEND_PBS_ATTENTION:      "pbs",
}


# ── Cell: one length point ────────────────────────────────────────────────────

@dataclass
class LengthCell:
    seq_len: int
    theory_plan: SparsePlan
    empirical_records: list[EmpiricalRecord]
    sparsity_profile: SparsityProfile | None = None

    # ── Derived fields (computed on init via __post_init__) ───────────────────
    theory_backend: str         = field(init=False)
    theory_latency_ms: float    = field(init=False)

    empirical_winner: str       = field(init=False)   # config_name of fastest passing record
    empirical_winner_backend: str = field(init=False) # "pbs" / "flexprefill" / "flash"
    empirical_winner_ms: float  = field(init=False)

    flash_ms: float             = field(init=False)   # Flash reference latency (NaN if missing)
    pbs_best_ms: float          = field(init=False)
    flex_best_ms: float         = field(init=False)

    theory_agrees: bool         = field(init=False)   # theory backend family == empirical winner family
    theory_latency_error_pct: float = field(init=False)  # (theory_pred - empirical) / empirical * 100

    def __post_init__(self) -> None:
        plan = self.theory_plan
        self.theory_backend    = plan.backend
        self.theory_latency_ms = plan.selected_estimate.estimated_latency_ms

        passing = [r for r in self.empirical_records if r.passed]
        if not passing:
            passing = self.empirical_records   # fall back: use all if none pass

        flash_recs = [r for r in self.empirical_records if r.backend == "flash"]
        pbs_recs   = [r for r in self.empirical_records if r.backend == "pbs"   and r.passed]
        flex_recs  = [r for r in self.empirical_records if r.backend == "flexprefill" and r.passed]

        self.flash_ms    = min((r.t_mean_ms for r in flash_recs), default=float("nan"))
        self.pbs_best_ms = min((r.t_mean_ms for r in pbs_recs),  default=float("nan"))
        self.flex_best_ms= min((r.t_mean_ms for r in flex_recs), default=float("nan"))

        winner_rec = min(passing, key=lambda r: r.t_mean_ms)
        self.empirical_winner         = winner_rec.config_name
        self.empirical_winner_backend = winner_rec.backend
        self.empirical_winner_ms      = winner_rec.t_mean_ms

        # Map theory backend to empirical family name for agreement check
        emp_family = _THEORY_TO_EMPIRICAL_BACKEND.get(plan.backend, plan.backend)
        self.theory_agrees = (emp_family == winner_rec.backend)

        # Latency error: compare theory estimate for its chosen backend
        # against the actual best measurement of that same backend family
        emp_actual = {
            BACKEND_DENSE:              self.flash_ms,
            BACKEND_FLEX_PREFILL_TRITON:self.flex_best_ms,
            BACKEND_PBS_ATTENTION:      self.pbs_best_ms,
        }.get(plan.backend, float("nan"))
        if not (emp_actual != emp_actual):  # nan check
            self.theory_latency_error_pct = (
                (self.theory_latency_ms - emp_actual) / emp_actual * 100.0
            )
        else:
            self.theory_latency_error_pct = float("nan")

    def summary(self) -> dict[str, Any]:
        sp = self.sparsity_profile
        return {
            "seq_len": self.seq_len,
            "theory_backend": self.theory_backend,
            "theory_latency_ms": round(self.theory_latency_ms, 3),
            "empirical_winner": self.empirical_winner,
            "empirical_winner_backend": self.empirical_winner_backend,
            "empirical_winner_ms": round(self.empirical_winner_ms, 3),
            "flash_ms": round(self.flash_ms, 3) if self.flash_ms == self.flash_ms else None,
            "pbs_best_ms":  round(self.pbs_best_ms,  3) if self.pbs_best_ms  == self.pbs_best_ms  else None,
            "flex_best_ms": round(self.flex_best_ms, 3) if self.flex_best_ms == self.flex_best_ms else None,
            "theory_agrees": self.theory_agrees,
            "theory_latency_error_pct": (
                round(self.theory_latency_error_pct, 2)
                if self.theory_latency_error_pct == self.theory_latency_error_pct else None
            ),
            "sparsity": {
                "ratio":           round(sp.estimated_sparsity_ratio, 4) if sp else None,
                "active_fraction": round(sp.estimated_active_fraction, 4) if sp else None,
                "kv_norm_cv":      round(sp.kv_norm_cv, 4) if sp else None,
                "topk_coverage":   round(sp.sample_topk_coverage, 4) if sp else None,
                "source":          sp.method if sp else "none",
            },
        }

    def top_empirical(self, n: int = 5) -> list[dict[str, Any]]:
        passing = sorted(
            [r for r in self.empirical_records if r.passed],
            key=lambda r: r.t_mean_ms,
        )
        return [
            {
                "rank": i + 1,
                "config": r.config_name,
                "backend": r.backend,
                "t_mean_ms": round(r.t_mean_ms, 3),
                "mse": r.mse,
                "kl": r.kl,
                "params": r.params,
            }
            for i, r in enumerate(passing[:n])
        ]


# ── Full study result ─────────────────────────────────────────────────────────

@dataclass
class LengthStudyResult:
    model_family: str
    objective:    str
    cells:        list[LengthCell]
    metadata:     dict[str, Any] = field(default_factory=dict)

    def summary_table(self) -> list[dict[str, Any]]:
        return [c.summary() for c in self.cells]

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_family": self.model_family,
            "objective":    self.objective,
            "metadata":     self.metadata,
            "summary_table": self.summary_table(),
            "cells": [
                {
                    **c.summary(),
                    "top_empirical": c.top_empirical(5),
                    "theory_plan": {
                        "backend": c.theory_plan.backend,
                        "estimated_latency_ms": c.theory_plan.selected_estimate.estimated_latency_ms,
                        "estimated_memory_gb":  c.theory_plan.selected_estimate.estimated_memory_gb,
                        "score":                c.theory_plan.selected_estimate.score,
                        "components_ms":        dict(c.theory_plan.selected_estimate.components_ms),
                        "notes":                list(c.theory_plan.notes),
                        "all_candidates": [
                            {
                                "backend": e.backend,
                                "estimated_latency_ms": round(e.estimated_latency_ms, 3),
                                "score": round(e.score, 3),
                                "feasible": e.feasible,
                            }
                            for e in c.theory_plan.candidate_estimates
                        ],
                    },
                }
                for c in self.cells
            ],
        }

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.as_dict(), f, indent=2, ensure_ascii=False)
        print(f"Saved study result → {path}")

    @classmethod
    def load(cls, path: str | Path) -> "LengthStudyResult":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        # Return a lightweight proxy for analysis (cells are summary-only)
        result = object.__new__(cls)
        result.model_family = data["model_family"]
        result.objective    = data["objective"]
        result.metadata     = data.get("metadata", {})
        result.cells        = []          # summary-only load; use data["cells"] for full info
        result._raw         = data        # attach raw data for analysis scripts
        return result


# ── Runner ────────────────────────────────────────────────────────────────────

class LengthStudyRunner:
    """
    Combines ``TheoryDrivenDispatcher`` with ``empirical_sweep.run_sweep``
    to produce a ``LengthStudyResult``.
    """

    def __init__(
        self,
        *,
        model_family:        str   = "qwen2",
        objective:           str   = "balanced",
        memory_budget_gb:    float = 72.0,
        safety_margin:       float = 0.9,
        backend_config:      SparseBackendConfig | None = None,
        pbs_configs:         list[PBSConfig]  | None = None,
        flex_configs:        list[FlexConfig] | None = None,
        cache_dir:           str   = "data/cache",
        prompt_family:       str   = "default",
        seed:                int   = 42,
        device:              str   = "cuda",
        warmup:              int   = 2,
        repeats:             int   = 5,
        mse_threshold:       float = 0.02,
        kl_threshold:        float = 0.10,
        estimate_sparsity:   bool  = True,   # use real QKV to estimate sparsity
        sparsity_sample_rows: int  = 64,     # rows sampled for sparsity estimation
        verbose:             bool  = True,
    ) -> None:
        self.model_family        = model_family
        self.objective           = objective
        self.cache_dir           = cache_dir
        self.prompt_family       = prompt_family
        self.seed                = seed
        self.device              = device
        self.warmup              = warmup
        self.repeats             = repeats
        self.mse_threshold       = mse_threshold
        self.kl_threshold        = kl_threshold
        self.do_estimate_sparsity = estimate_sparsity
        self.sparsity_sample_rows = sparsity_sample_rows
        self.verbose             = verbose
        self.pbs_configs         = pbs_configs  or default_pbs_grid()
        self.flex_configs        = flex_configs or default_flex_grid()

        self.dispatcher = TheoryDrivenDispatcher(
            backend_config=backend_config or SparseBackendConfig(),
            model_memory_budget_gb=memory_budget_gb,
            safety_margin=safety_margin,
        )

    def run(self, lengths: list[int]) -> LengthStudyResult:
        if self.verbose:
            print(f"Theory model family: {self.model_family}")
            print(f"Objective: {self.objective}")
            print(f"Lengths: {lengths}")
            print(f"PBS configs: {len(self.pbs_configs)}")
            print(f"Flex configs: {len(self.flex_configs)}")
            print()

        # ── Sparsity estimation (load QKV once, estimate, then reuse for sweep) ──
        sparsity_profiles: dict[int, SparsityProfile | None] = {}
        if self.do_estimate_sparsity:
            if self.verbose:
                print("Estimating attention sparsity from QKV data...")
            for L in lengths:
                try:
                    q, k, v = load_qkv(
                        self.cache_dir, L, self.prompt_family,
                        self.seed, self.device,
                    )
                    sp = estimate_sparsity(
                        q, k,
                        block_size=self.dispatcher.backend_config.block_size,
                        sample_rows=self.sparsity_sample_rows,
                        seed=self.seed,
                    )
                    sparsity_profiles[L] = sp
                    if self.verbose:
                        print(f"  L={L:>6}: sparsity={sp.estimated_sparsity_ratio:.3f}  "
                              f"active_frac={sp.estimated_active_fraction:.3f}  "
                              f"cv={sp.kv_norm_cv:.3f}  "
                              f"({'sparse' if sp.is_sparse else 'dense'})")
                    del q, k, v
                    import torch; torch.cuda.empty_cache()
                except Exception as exc:
                    warnings.warn(f"Sparsity estimation failed at L={L}: {exc}")
                    sparsity_profiles[L] = None
            if self.verbose:
                print()
        else:
            sparsity_profiles = {L: None for L in lengths}

        # ── Theory plans (sparsity-aware) ────────────────────────────────────
        plans: dict[int, SparsePlan] = {}
        for L in lengths:
            sp = sparsity_profiles.get(L)
            plan = self.dispatcher.build_plan(
                model_or_name=self.model_family,
                prompt_tokens=L,
                objective=self.objective,
                allow_experimental=False,
                sparsity_ratio=sp.estimated_sparsity_ratio if sp else 0.0,
                kv_norm_cv=sp.kv_norm_cv if sp else 0.0,
                sparsity_source="estimated" if sp else "geometric",
            )
            plans[L] = plan
            if self.verbose:
                src = f" [sparsity={sp.estimated_sparsity_ratio:.2f}]" if sp else " [geometric]"
                print(f"Theory @ L={L:>6}: {plan.backend:<28} "
                      f"(est. {plan.selected_estimate.estimated_latency_ms:.1f} ms){src}")

        if self.verbose:
            print()

        # ── Empirical sweep ──────────────────────────────────────────────────
        emp_by_len = run_sweep(
            lengths=lengths,
            pbs_configs=self.pbs_configs,
            flex_configs=self.flex_configs,
            cache_dir=self.cache_dir,
            prompt_family=self.prompt_family,
            seed=self.seed,
            device=self.device,
            warmup=self.warmup,
            repeats=self.repeats,
            mse_threshold=self.mse_threshold,
            kl_threshold=self.kl_threshold,
            include_flash=True,
            verbose=self.verbose,
        )

        # ── Join into cells ──────────────────────────────────────────────────
        cells = [
            LengthCell(
                seq_len=L,
                theory_plan=plans[L],
                empirical_records=emp_by_len.get(L, []),
                sparsity_profile=sparsity_profiles.get(L),
            )
            for L in lengths
        ]

        return LengthStudyResult(
            model_family=self.model_family,
            objective=self.objective,
            cells=cells,
            metadata={
                "lengths": lengths,
                "pbs_config_count":  len(self.pbs_configs),
                "flex_config_count": len(self.flex_configs),
                "warmup":  self.warmup,
                "repeats": self.repeats,
                "mse_threshold": self.mse_threshold,
                "cache_dir": str(self.cache_dir),
                "prompt_family": self.prompt_family,
                "seed": self.seed,
                "sparsity_estimation": self.do_estimate_sparsity,
                "sparsity_sample_rows": self.sparsity_sample_rows,
            },
        )
