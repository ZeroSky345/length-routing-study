"""
Calibrate ``TheoryModelProfile`` coefficients against empirical measurements.

Given a ``LengthStudyResult`` (or its JSON), this module fits the cost-model
coefficients for the PBS and FlexPrefill backends using ordinary least squares
(numpy, no scipy required).

Fitted coefficients (per backend)
-----------------------------------
PBS
  - ``pbs_kernel_ms_per_active_block_per_layer`` (slope in kernel_ms vs active_blocks)
  - ``pbs_select_ms_per_block_pair_per_layer``   (slope in select_ms vs block_pairs)

Flex
  - ``flex_kernel_ms_per_active_block_per_layer`` (slope)
  - ``flex_mask_ms_per_block_log_per_layer``       (slope)

The linear baseline (``linear_ms_per_token_per_layer``) and dense-attention
coefficient (``dense_block_ms_per_layer``) are NOT re-fitted because we measure
only the attention kernel time, not the full model forward pass.

Output: a JSON ``CalibratedProfile`` that can be passed back to
``TheoryModelProfile`` to update the relevant coefficients.
"""
from __future__ import annotations

import json
import math
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

try:
    import numpy as np
    _NUMPY = True
except ImportError:
    _NUMPY = False


@dataclass
class CalibratedProfile:
    """Fitted coefficients returned by the calibration routine."""
    model_family: str

    # PBS
    pbs_kernel_ms_per_active_block_per_layer: float
    pbs_select_ms_per_block_pair_per_layer:   float

    # Flex
    flex_kernel_ms_per_active_block_per_layer: float
    flex_mask_ms_per_block_log_per_layer:      float

    # Diagnostics
    pbs_r2:  float
    flex_r2: float
    n_pbs:   int
    n_flex:  int
    notes:   list[str]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.as_dict(), f, indent=2)
        print(f"Saved calibrated profile → {path}")

    def print_summary(self) -> None:
        print("── Calibrated Profile ──────────────────────────────────────────")
        print(f"  PBS kernel coeff   : {self.pbs_kernel_ms_per_active_block_per_layer:.4e} ms/block/layer")
        print(f"  PBS select coeff   : {self.pbs_select_ms_per_block_pair_per_layer:.4e} ms/pair/layer")
        print(f"  PBS R²             : {self.pbs_r2:.4f}  (n={self.n_pbs})")
        print(f"  Flex kernel coeff  : {self.flex_kernel_ms_per_active_block_per_layer:.4e} ms/block/layer")
        print(f"  Flex mask coeff    : {self.flex_mask_ms_per_block_log_per_layer:.4e} ms/block_log/layer")
        print(f"  Flex R²            : {self.flex_r2:.4f}  (n={self.n_flex})")
        for note in self.notes:
            print(f"  note: {note}")


# ── OLS helper ────────────────────────────────────────────────────────────────

def _ols_2d(X1: list[float], X2: list[float], Y: list[float]) -> tuple[float, float, float]:
    """
    Fit Y = a*X1 + b*X2 (no intercept) via OLS.
    Returns (a, b, R²).
    """
    if not _NUMPY:
        raise ImportError("numpy is required for calibration. Install it with: pip install numpy")
    import numpy as np  # type: ignore
    A = np.column_stack([X1, X2])
    y = np.array(Y)
    coeff, residuals, rank, sv = np.linalg.lstsq(A, y, rcond=None)
    y_hat = A @ coeff
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(coeff[0]), float(coeff[1]), r2


def _ols_1d(X: list[float], Y: list[float]) -> tuple[float, float]:
    """Fit Y = a*X (no intercept). Returns (a, R²)."""
    if not _NUMPY:
        raise ImportError("numpy is required for calibration.")
    import numpy as np  # type: ignore
    x = np.array(X).reshape(-1, 1)
    y = np.array(Y)
    coeff, *_ = np.linalg.lstsq(x, y, rcond=None)
    y_hat = x.flatten() * coeff[0]
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(coeff[0]), r2


# ── Main calibration function ─────────────────────────────────────────────────

def calibrate_from_study(
    study_data: dict[str, Any],
    num_layers: int = 36,
    block_size: int = 128,
) -> CalibratedProfile:
    """
    Fit PBS and Flex kernel coefficients from a ``LengthStudyResult.as_dict()`` payload.

    Strategy
    ---------
    For **PBS**: measure total_attention_time ≈ select_time + kernel_time.
    We approximate:
        select_time ≈ select_coeff * num_blocks² * num_layers
        kernel_time ≈ kernel_coeff * estimated_active_blocks * num_layers

    We don't have phase-decomposed timings for individual PBS configs, only total
    latency.  We therefore use the total PBS latency across multiple (L, config)
    points and regress against (active_blocks*layers, block_pairs*layers) to
    jointly estimate both coefficients.

    For **Flex**: same approach with (active_blocks*layers, block_log*layers).

    In practice we use the *best* PBS record at each length to avoid noisy outliers
    skewing the fit.  If you have full per-record timings (e.g. from raw JSONL),
    pass them via the ``cells[*].top_empirical`` field.
    """
    from .cost_model import infer_workload_features
    from .sparse_plan import SparseBackendConfig

    cells = study_data.get("cells", [])
    if not cells:
        raise ValueError("study_data contains no 'cells'")

    cfg = SparseBackendConfig(block_size=block_size)

    pbs_X1:  list[float] = []  # active_blocks * layers
    pbs_X2:  list[float] = []  # block_pairs  * layers  (selection complexity)
    pbs_Y:   list[float] = []  # total_ms (attention portion, approx = total - linear)

    flex_X1: list[float] = []  # active_blocks * layers
    flex_X2: list[float] = []  # block_log     * layers  (mask complexity)
    flex_Y:  list[float] = []

    notes: list[str] = []

    for cell in cells:
        L = cell["seq_len"]
        wf = infer_workload_features(prompt_tokens=L, cfg=cfg)

        # Subtract linear contribution (token-linear, not length-quadratic)
        # Using the default profile's linear_ms_per_token_per_layer
        # (This is a minor correction; for attention-only measurements it is ~0)
        linear_correction = 0.0  # set to non-zero if full model timings are available

        # Best PBS record at this length
        top = cell.get("top_empirical", [])
        pbs_top = [r for r in top if r["backend"] == "pbs"]
        if pbs_top:
            t = pbs_top[0]["t_mean_ms"] - linear_correction
            if t > 0:
                pbs_X1.append(wf.estimated_pbs_active_blocks * num_layers)
                pbs_X2.append(wf.pbs_selection_complexity     * num_layers)
                pbs_Y.append(t)

        # Best Flex record at this length
        flex_top = [r for r in top if r["backend"] == "flexprefill"]
        if flex_top:
            t = flex_top[0]["t_mean_ms"] - linear_correction
            if t > 0:
                flex_X1.append(wf.estimated_flex_active_blocks * num_layers)
                flex_X2.append(wf.flex_mask_complexity          * num_layers)
                flex_Y.append(t)

    # ── Fit PBS ──────────────────────────────────────────────────────────────
    pbs_kernel, pbs_select, pbs_r2 = float("nan"), float("nan"), float("nan")
    if len(pbs_Y) >= 2:
        try:
            pbs_kernel, pbs_select, pbs_r2 = _ols_2d(pbs_X1, pbs_X2, pbs_Y)
            if pbs_kernel < 0 or pbs_select < 0:
                notes.append(f"PBS fit produced negative coefficient(s): kernel={pbs_kernel:.3e}, select={pbs_select:.3e}. "
                             "Consider adding more length diversity or using a non-negative solver.")
        except Exception as e:
            notes.append(f"PBS OLS failed: {e}")
    else:
        notes.append(f"Too few PBS data points ({len(pbs_Y)}) for 2-variable fit.")

    # ── Fit Flex ─────────────────────────────────────────────────────────────
    flex_kernel, flex_mask, flex_r2 = float("nan"), float("nan"), float("nan")
    if len(flex_Y) >= 2:
        try:
            flex_kernel, flex_mask, flex_r2 = _ols_2d(flex_X1, flex_X2, flex_Y)
            if flex_kernel < 0 or flex_mask < 0:
                notes.append(f"Flex fit produced negative coefficient(s): kernel={flex_kernel:.3e}, mask={flex_mask:.3e}.")
        except Exception as e:
            notes.append(f"Flex OLS failed: {e}")
    else:
        notes.append(f"Too few Flex data points ({len(flex_Y)}) for 2-variable fit.")

    model_family = study_data.get("model_family", "generic")
    return CalibratedProfile(
        model_family=model_family,
        pbs_kernel_ms_per_active_block_per_layer=pbs_kernel,
        pbs_select_ms_per_block_pair_per_layer=pbs_select,
        flex_kernel_ms_per_active_block_per_layer=flex_kernel,
        flex_mask_ms_per_block_log_per_layer=flex_mask,
        pbs_r2=pbs_r2,
        flex_r2=flex_r2,
        n_pbs=len(pbs_Y),
        n_flex=len(flex_Y),
        notes=notes,
    )
