"""
Empirical PBS / FlexPrefill kernel sweep.

Runs the **real** PBS and FlexPrefill CUDA/Triton kernels across a
parameterised grid and a set of sequence lengths, returning
``EmpiricalRecord`` objects whose fields mirror ``BackendEstimate`` so
they can be placed side-by-side in the joint ``LengthStudyResult``.

Key design points
-----------------
* Uses ``FlexConfig.min_budget_frac`` instead of a hard-coded
  ``min_budget`` token count — avoids the "L=2048 spike" that plagued
  the original benchmark when ``min_budget=1024`` forced 50 % density.
* All timings are wall-clock CUDA-event latencies averaged over
  ``repeats`` iterations after ``warmup`` hot-start runs.
* Flash (SDPA) is always run as the reference; MSE / KL are computed
  against its output.
"""
from __future__ import annotations

import math
import os
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

# ── Path bootstrap ────────────────────────────────────────────────────────────
from ._paths import DEFAULT_PBS_ROOT, DEFAULT_FLEX_ROOT

for _p in (str(DEFAULT_PBS_ROOT), str(DEFAULT_FLEX_ROOT)):
    if _p and _p not in sys.path:
        sys.path.append(_p)


# ── Timing helpers ────────────────────────────────────────────────────────────

def _cuda_ms(fn) -> tuple[Any, float]:
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    s.record()
    out = fn()
    e.record()
    torch.cuda.synchronize()
    return out, float(s.elapsed_time(e))


def _stats(samples: list[float]) -> dict[str, float]:
    t = torch.tensor(samples, dtype=torch.float64)
    return {
        "mean": float(t.mean()),
        "std":  float(t.std(unbiased=False)),
        "p50":  float(torch.quantile(t, 0.50)),
        "p95":  float(torch.quantile(t, 0.95)),
    }


# ── Parameter configs ─────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PBSConfig:
    threshold:   float = 0.9
    segment_size: int  = 256
    block_size:  int   = 128
    use_triton:  bool  = True
    force_first: bool  = True

    @property
    def name(self) -> str:
        t = str(self.threshold).replace(".", "p")
        return f"pbs_seg{self.segment_size}_thr{t}"

    def as_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in
                ("threshold", "segment_size", "block_size", "use_triton", "force_first")}


@dataclass(frozen=True)
class FlexConfig:
    gamma:           float = 0.95
    tau:             float = 0.1
    min_budget_frac: float = 0.0   # fraction of seq_len; 0 = no floor
    block_size:      int   = 128

    @property
    def name(self) -> str:
        g = str(self.gamma).replace(".", "p")
        t = str(self.tau).replace(".", "p")
        f = str(self.min_budget_frac).replace(".", "p")
        return f"flex_g{g}_t{t}_mbf{f}"

    def min_budget_tokens(self, seq_len: int) -> int | None:
        if self.min_budget_frac <= 0.0:
            return None
        return max(self.block_size, round(seq_len * self.min_budget_frac))

    def as_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in
                ("gamma", "tau", "min_budget_frac", "block_size")}


# ── Result type ───────────────────────────────────────────────────────────────

@dataclass
class EmpiricalRecord:
    """Single (kernel-config ? sequence-length) measurement cell."""

    backend:     str          # "pbs" | "flexprefill" | "flash"
    config_name: str
    params:      dict[str, Any]
    seq_len:     int

    t_mean_ms:   float
    t_std_ms:    float
    t_p50_ms:    float
    t_p95_ms:    float

    t_select_ms: float = 0.0
    t_kernel_ms: float = 0.0

    mse:         float = 0.0
    kl:          float = 0.0
    kernel_time_ratio: float | None = None
    active_block_fraction: float | None = None

    passed_mse:  bool  = True
    passed_kl:   bool  = True

    @property
    def passed(self) -> bool:
        return self.passed_mse and self.passed_kl

    @property
    def active_fraction(self) -> float | None:
        if self.active_block_fraction is not None:
            return self.active_block_fraction
        return self.kernel_time_ratio

    def as_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "config_name": self.config_name,
            "params": self.params,
            "seq_len": self.seq_len,
            "t_mean_ms":   self.t_mean_ms,
            "t_std_ms":    self.t_std_ms,
            "t_p50_ms":    self.t_p50_ms,
            "t_p95_ms":    self.t_p95_ms,
            "t_select_ms": self.t_select_ms,
            "t_kernel_ms": self.t_kernel_ms,
            "kernel_time_ratio": self.kernel_time_ratio,
            "active_block_fraction": self.active_block_fraction,
            "mse": self.mse,
            "kl":  self.kl,
            "passed": self.passed,
        }


def _causal_block_fraction(mask: torch.Tensor) -> float:
    if mask.dim() == 4:
        _, _, q_blocks, kv_blocks = mask.shape
        valid = torch.tril(torch.ones((q_blocks, kv_blocks), dtype=torch.bool, device=mask.device), diagonal=(kv_blocks - q_blocks))
        denom = int(valid.sum().item()) * int(mask.shape[0]) * int(mask.shape[1])
        numer = int((mask.to(torch.bool) & valid[None, None, :, :]).sum().item())
        return numer / max(1, denom)
    if mask.dim() == 2:
        q_blocks, kv_blocks = mask.shape
        valid = torch.tril(torch.ones((q_blocks, kv_blocks), dtype=torch.bool, device=mask.device), diagonal=(kv_blocks - q_blocks))
        denom = int(valid.sum().item())
        numer = int((mask.to(torch.bool) & valid).sum().item())
        return numer / max(1, denom)
    raise ValueError(f"unsupported mask rank: {mask.dim()}")


def _measure_pbs_active_block_fraction(
    q: torch.Tensor,
    k: torch.Tensor,
    cfg: PBSConfig,
) -> float | None:
    try:
        from pbs_attn.src.pbs import permuted_block_selection
        from pbs_attn.src.permute_states import apply_permutation
    except ImportError:
        return None

    batch_size, num_q_heads, q_len, _ = q.shape
    batch_size, num_kv_heads, kv_len, _ = k.shape
    if num_q_heads != num_kv_heads or q_len != kv_len:
        return None

    with torch.inference_mode():
        perm_key_states, perm_key_indices = apply_permutation(
            query_states=q,
            key_states=k,
            block_size=cfg.block_size,
            segment_size=cfg.segment_size,
        )
        perm_query_states = q
        perm_query_indices = torch.arange(q_len, device=q.device).unsqueeze(0).unsqueeze(0).expand(batch_size, num_q_heads, -1)
        _, block_mask, segment_mask = permuted_block_selection(
            permuted_query_states=perm_query_states,
            permuted_key_states=perm_key_states,
            query_indices=perm_query_indices,
            key_indices=perm_key_indices,
            block_size=cfg.block_size,
            segment_size=cfg.segment_size,
            threshold=cfg.threshold,
            causal=True,
            force_select_first_block=cfg.force_first,
        )
        full_mask = block_mask | segment_mask[None, None, :, :]
    return _causal_block_fraction(full_mask)

def run_flash(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    warmup: int = 2,
    repeats: int = 5,
) -> EmpiricalRecord:
    import torch.nn.functional as F
    for _ in range(warmup):
        with torch.inference_mode():
            F.scaled_dot_product_attention(q, k, v, is_causal=True)
        torch.cuda.synchronize()
    lats: list[float] = []
    out = None
    for _ in range(repeats):
        with torch.inference_mode():
            out, ms = _cuda_ms(lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True))
        lats.append(ms)
    assert out is not None
    s = _stats(lats)
    return EmpiricalRecord(
        backend="flash", config_name="flash_ref", params={},
        seq_len=q.shape[2],
        t_mean_ms=s["mean"], t_std_ms=s["std"],
        t_p50_ms=s["p50"],   t_p95_ms=s["p95"],
    )


# ── PBS kernel ────────────────────────────────────────────────────────────────

def run_pbs(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    cfg: PBSConfig,
    reference: torch.Tensor | None = None,
    warmup: int = 2, repeats: int = 5,
    mse_threshold: float = 0.02, kl_threshold: float = 0.10,
) -> EmpiricalRecord:
    try:
        from pbs_attn.src.pbs import permuted_block_sparse_attn_fwd
    except ImportError as e:
        raise ImportError("pbs_attn not found. Set LRS_PBS_ROOT.") from e

    def _call():
        return permuted_block_sparse_attn_fwd(
            q, k, v,
            block_size=cfg.block_size,
            segment_size=cfg.segment_size,
            threshold=cfg.threshold,
            causal=True,
            force_select_first_block=cfg.force_first,
            use_triton=cfg.use_triton,
        )

    for _ in range(warmup):
        with torch.inference_mode():
            _call()
        torch.cuda.synchronize()

    lats: list[float] = []
    out = None
    for _ in range(repeats):
        with torch.inference_mode():
            out, ms = _cuda_ms(_call)
        lats.append(ms)
    assert out is not None
    s = _stats(lats)

    import torch.nn.functional as F
    ref = reference if reference is not None else \
        F.scaled_dot_product_attention(q, k, v, is_causal=True)
    mse = float(torch.mean((out.float() - ref.float()) ** 2).item())
    kl  = _kl(out, ref)
    return EmpiricalRecord(
        backend="pbs", config_name=cfg.name, params=cfg.as_dict(),
        seq_len=q.shape[2],
        t_mean_ms=s["mean"], t_std_ms=s["std"],
        t_p50_ms=s["p50"],   t_p95_ms=s["p95"],
        mse=mse, kl=kl,
        passed_mse=(mse <= mse_threshold),
        passed_kl=(kl  <= kl_threshold),
    )


# ── PBS with decomposed timing ────────────────────────────────────────────────

def run_pbs_decomposed(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    cfg: PBSConfig,
    flash_time_ms: float,
    reference: torch.Tensor | None = None,
    warmup: int = 2, repeats: int = 5,
    mse_threshold: float = 0.02, kl_threshold: float = 0.10,
) -> EmpiricalRecord:
    """
    Run PBS with timing decomposed into t_select and t_kernel.

    ``kernel_time_ratio`` is a latency proxy: t_kernel / t_flash.
    ``active_block_fraction`` is measured separately from the actual PBS block
    mask, so downstream analysis can distinguish true block density from timing.
    """
    try:
        from pbs_attn.src.pbs import permuted_block_sparse_attn_fwd
    except ImportError as e:
        raise ImportError("pbs_attn not found. Set LRS_PBS_ROOT.") from e

    dense_cfg = PBSConfig(
        threshold=1.0,
        segment_size=cfg.segment_size,
        block_size=cfg.block_size,
        use_triton=cfg.use_triton,
        force_first=cfg.force_first,
    )

    def _call_dense():
        return permuted_block_sparse_attn_fwd(
            q, k, v,
            block_size=dense_cfg.block_size,
            segment_size=dense_cfg.segment_size,
            threshold=dense_cfg.threshold,
            causal=True,
            force_select_first_block=dense_cfg.force_first,
            use_triton=dense_cfg.use_triton,
        )

    try:
        for _ in range(warmup):
            with torch.inference_mode():
                _call_dense()
            torch.cuda.synchronize()
        dense_lats: list[float] = []
        for _ in range(repeats):
            with torch.inference_mode():
                _, ms = _cuda_ms(_call_dense)
            dense_lats.append(ms)
        t_dense_pbs = float(torch.tensor(dense_lats).mean())
        t_select = max(0.0, t_dense_pbs - flash_time_ms)
    except Exception:
        from .length_router import T_SELECT_PBS_A, T_SELECT_PBS_B
        L = q.shape[2]
        t_select = T_SELECT_PBS_A * L + T_SELECT_PBS_B

    normal_rec = run_pbs(
        q, k, v, cfg, reference=reference,
        warmup=warmup, repeats=repeats,
        mse_threshold=mse_threshold, kl_threshold=kl_threshold,
    )

    t_kernel = max(0.0, normal_rec.t_mean_ms - t_select)
    kernel_time_ratio = t_kernel / max(1e-6, flash_time_ms)
    active_block_fraction = _measure_pbs_active_block_fraction(q, k, cfg)

    return EmpiricalRecord(
        backend=normal_rec.backend,
        config_name=normal_rec.config_name,
        params=normal_rec.params,
        seq_len=normal_rec.seq_len,
        t_mean_ms=normal_rec.t_mean_ms,
        t_std_ms=normal_rec.t_std_ms,
        t_p50_ms=normal_rec.t_p50_ms,
        t_p95_ms=normal_rec.t_p95_ms,
        t_select_ms=round(t_select, 4),
        t_kernel_ms=round(t_kernel, 4),
        kernel_time_ratio=round(kernel_time_ratio, 4),
        active_block_fraction=(None if active_block_fraction is None else round(active_block_fraction, 4)),
        mse=normal_rec.mse,
        kl=normal_rec.kl,
        passed_mse=normal_rec.passed_mse,
        passed_kl=normal_rec.passed_kl,
    )

def run_flex(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    cfg: FlexConfig,
    reference: torch.Tensor | None = None,
    warmup: int = 2, repeats: int = 5,
    mse_threshold: float = 0.02, kl_threshold: float = 0.10,
) -> EmpiricalRecord:
    try:
        from flex_prefill.ops.flex_prefill_attention import flex_prefill_attention
    except ImportError as e:
        raise ImportError("FlexPrefill not found. Set LRS_FLEX_ROOT.") from e

    seq_len    = q.shape[2]
    min_budget = cfg.min_budget_tokens(seq_len)
    q_bnhd     = q.transpose(1, 2).contiguous()
    k_bnhd     = k.transpose(1, 2).contiguous()
    v_bnhd     = v.transpose(1, 2).contiguous()

    def _call():
        out = flex_prefill_attention(
            q_bnhd, k_bnhd, v_bnhd,
            gamma=cfg.gamma, tau=cfg.tau,
            min_budget=min_budget, max_budget=None,
            block_size=cfg.block_size,
        )
        return out.transpose(1, 2).contiguous()

    for _ in range(warmup):
        with torch.inference_mode():
            _call()
        torch.cuda.synchronize()

    lats: list[float] = []
    out = None
    for _ in range(repeats):
        with torch.inference_mode():
            out, ms = _cuda_ms(_call)
        lats.append(ms)
    assert out is not None
    s = _stats(lats)

    import torch.nn.functional as F
    ref = reference if reference is not None else \
        F.scaled_dot_product_attention(q, k, v, is_causal=True)
    mse = float(torch.mean((out.float() - ref.float()) ** 2).item())
    kl  = _kl(out, ref)
    return EmpiricalRecord(
        backend="flexprefill", config_name=cfg.name, params=cfg.as_dict(),
        seq_len=seq_len,
        t_mean_ms=s["mean"], t_std_ms=s["std"],
        t_p50_ms=s["p50"],   t_p95_ms=s["p95"],
        mse=mse, kl=kl,
        passed_mse=(mse <= mse_threshold),
        passed_kl=(kl  <= kl_threshold),
    )


# ── Accuracy helpers ──────────────────────────────────────────────────────────

def _kl(pred: torch.Tensor, ref: torch.Tensor, eps: float = 1e-8) -> float:
    p = torch.clamp(torch.softmax(pred.float(), dim=-1), min=eps)
    q = torch.clamp(torch.softmax(ref.float(),  dim=-1), min=eps)
    v = float(torch.sum(p * (torch.log(p) - torch.log(q)), dim=-1).mean().item())
    return v if math.isfinite(v) else float("inf")


# ── QKV loader ────────────────────────────────────────────────────────────────

_WARNED_SYNTHETIC: set[int] = set()


def load_qkv(
    cache_dir: str | Path,
    seq_len: int,
    prompt_family: str = "default",
    seed: int = 42,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    synthetic_heads: int = 16,
    synthetic_dim: int = 128,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load QKV from model-derived cache; fall back to realistic synthetic data."""
    cache_dir = Path(cache_dir)
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    for candidate in [
        cache_dir / f"qkv_len_{seq_len}_{prompt_family}_seed_{seed}.pt",
        cache_dir / f"qkv_len_{seq_len}_{prompt_family}.pt",
        cache_dir / f"qkv_len_{seq_len}.pt",
    ]:
        if candidate.exists():
            payload = torch.load(candidate, map_location="cpu")
            q = payload["q"].to(device=dev, dtype=dtype)
            k = payload["k"].to(device=dev, dtype=dtype)
            v = payload["v"].to(device=dev, dtype=dtype)
            # Warn on dimension mismatch with target model (Qwen2.5-3B: 16h, 128d)
            if q.shape[3] <= 64 and seq_len not in _WARNED_SYNTHETIC:
                _WARNED_SYNTHETIC.add(seq_len)
                warnings.warn(
                    f"Cache {candidate.name} has head_dim={q.shape[3]} (≤64). "
                    "This is likely synthetic data not derived from a real model. "
                    "Run scripts/generate_qkv_cache.py for model-derived caches.",
                    stacklevel=2,
                )
            return q, k, v

    if seq_len not in _WARNED_SYNTHETIC:
        _WARNED_SYNTHETIC.add(seq_len)
        warnings.warn(
            f"No QKV cache found for L={seq_len}. Using synthetic [{synthetic_heads}h, {synthetic_dim}d]. "
            "Results will not reflect real model attention patterns.",
            stacklevel=2,
        )
    torch.manual_seed(seed + seq_len)
    decay = torch.linspace(0.5, 1.5, seq_len).view(1, 1, seq_len, 1)
    q = torch.randn(1, synthetic_heads, seq_len, synthetic_dim, dtype=dtype)
    k = torch.randn(1, synthetic_heads, seq_len, synthetic_dim, dtype=dtype) * decay
    v = torch.randn(1, synthetic_heads, seq_len, synthetic_dim, dtype=dtype)
    return q.to(dev), k.to(dev), v.to(dev)


# ── Grid builders ─────────────────────────────────────────────────────────────

def default_pbs_grid(block_size: int = 128) -> list[PBSConfig]:
    configs = []
    for thr in (0.7, 0.8, 0.9):
        for seg in (128, 256, 512):
            configs.append(PBSConfig(threshold=thr, segment_size=seg, block_size=block_size))
    return configs


def default_flex_grid(block_size: int = 128) -> list[FlexConfig]:
    configs = []
    for gamma in (0.80, 0.90, 0.95):
        for tau in (0.05, 0.10, 0.20):
            for mbf in (0.0, 0.10, 0.25):
                configs.append(FlexConfig(gamma=gamma, tau=tau,
                                          min_budget_frac=mbf, block_size=block_size))
    return configs


# ── Main sweep entry point ────────────────────────────────────────────────────

def run_sweep(
    lengths: list[int],
    pbs_configs: list[PBSConfig] | None = None,
    flex_configs: list[FlexConfig] | None = None,
    cache_dir: str | Path = "data/cache",
    prompt_family: str = "default",
    seed: int = 42,
    device: str = "cuda",
    warmup: int = 2,
    repeats: int = 5,
    mse_threshold: float = 0.02,
    kl_threshold: float = 0.10,
    include_flash: bool = True,
    verbose: bool = True,
) -> dict[int, list[EmpiricalRecord]]:
    """
    Run the full empirical sweep.

    Returns a dict mapping ``seq_len → list[EmpiricalRecord]`` (Flash + PBS + Flex).
    """
    pbs_cfgs  = pbs_configs  if pbs_configs  is not None else default_pbs_grid()
    flex_cfgs = flex_configs if flex_configs is not None else default_flex_grid()

    results: dict[int, list[EmpiricalRecord]] = {}

    for L in lengths:
        if verbose:
            print(f"\n── L={L} ──────────────────────────────────────────")
        q, k, v = load_qkv(cache_dir, L, prompt_family, seed, device)

        with torch.inference_mode():
            import torch.nn.functional as F
            ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        records: list[EmpiricalRecord] = []

        if include_flash:
            r = run_flash(q, k, v, warmup=warmup, repeats=repeats)
            records.append(r)
            if verbose:
                print(f"  flash_ref       {r.t_mean_ms:>9.3f} ms")

        for cfg in pbs_cfgs:
            try:
                r = run_pbs(q, k, v, cfg, reference=ref,
                            warmup=warmup, repeats=repeats,
                            mse_threshold=mse_threshold, kl_threshold=kl_threshold)
                records.append(r)
                if verbose:
                    status = "OK" if r.passed else "FAIL"
                    print(f"  {cfg.name:<35} {r.t_mean_ms:>9.3f} ms  mse={r.mse:.5f}  {status}")
            except Exception as exc:
                warnings.warn(f"PBS {cfg.name} @ L={L}: {exc}")

        for cfg in flex_cfgs:
            mb = cfg.min_budget_tokens(L)
            try:
                r = run_flex(q, k, v, cfg, reference=ref,
                             warmup=warmup, repeats=repeats,
                             mse_threshold=mse_threshold, kl_threshold=kl_threshold)
                records.append(r)
                if verbose:
                    status = "OK" if r.passed else "FAIL"
                    print(f"  {cfg.name:<35} {r.t_mean_ms:>9.3f} ms  mse={r.mse:.5f}  mb={mb}  {status}")
            except Exception as exc:
                warnings.warn(f"Flex {cfg.name} @ L={L}: {exc}")

        results[L] = records

    return results
