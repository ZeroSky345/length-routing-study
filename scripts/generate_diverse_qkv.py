#!/usr/bin/env python3
"""
Generate QKV caches for five distinct "text scenario" attention patterns.

Each scenario simulates the kind of attention distribution produced by a
Transformer processing a specific text genre, without needing a real model.
The key dimension is the **K-norm distribution** — it controls which positions
will dominate the softmax, and thus how sparse the resulting attention is.

Scenarios
---------
code       — Code / program text.
             Strong local window (operators, brackets) + sparse global peaks
             (function names, import targets).  High sparsity (~0.65-0.75).

technical  — Technical / scientific text.
             Cross-references to definitions, section headers, citations.
             Medium sparsity (~0.45-0.55).

narrative  — Fiction / narrative prose.
             Mostly uniform local, broad contextual co-reference.
             Low sparsity (~0.25-0.35).

dialogue   — Conversational / dialogue text.
             Turn-based periodic structure.  Each turn has a strong start token
             and moderate within-turn locality.
             Medium-high sparsity (~0.50-0.60).

structured — Lists, tables, repetitive/structured text.
             Very sharp periodic peaks (list markers, column headers).
             Very high sparsity (~0.72-0.82).

Usage
-----
  python scripts/generate_diverse_qkv.py \
      --lengths 4096 8192 16384 32768 65536 \
      --output-dir data/cache \
      --num-heads 16 --head-dim 128 --seed 42
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


# ── Pattern generators ────────────────────────────────────────────────────────

def _k_norm_code(L: int, seed: int) -> torch.Tensor:
    """
    Code pattern: strong diagonal (recent) + sparse global spikes.
    Function names / keywords appear every ~64-128 tokens with large norms.
    """
    g = torch.Generator(); g.manual_seed(seed)
    # Base: recent-heavy decay (last 256 tokens are ~2× the average)
    decay = torch.linspace(0.3, 2.5, L)
    # Sparse global spikes: identifiers / keywords every ~80 tokens
    spikes = torch.zeros(L)
    spike_period = max(1, L // (L // 80 + 1))
    for i in range(0, L, spike_period):
        offset = int(torch.randint(0, max(1, spike_period // 2), (1,), generator=g).item())
        pos = min(i + offset, L - 1)
        spikes[pos] = torch.empty(1).uniform_(2.5, 5.0, generator=g).item()
    norms = (decay + spikes).clamp(min=0.2)
    # Add head diversity (different heads focus on different spans)
    return norms


def _k_norm_technical(L: int, seed: int) -> torch.Tensor:
    """
    Technical / scientific: section headers every ~512 tokens, definitions
    referenced later have elevated norms.
    """
    g = torch.Generator(); g.manual_seed(seed)
    decay = torch.linspace(0.5, 1.8, L)
    spikes = torch.zeros(L)
    # Section headers
    for i in range(0, L, max(1, L // 10)):
        spikes[i] = torch.empty(1).uniform_(1.5, 3.0, generator=g).item()
    # Scattered definition anchors
    n_defs = L // 200
    for _ in range(n_defs):
        pos = int(torch.randint(0, L, (1,), generator=g).item())
        spikes[pos] += torch.empty(1).uniform_(0.8, 2.0, generator=g).item()
    return (decay + spikes).clamp(min=0.3)


def _k_norm_narrative(L: int, seed: int) -> torch.Tensor:
    """
    Narrative / fiction: broad, relatively uniform context.  Low peaks,
    moderate recent-token bias.
    """
    g = torch.Generator(); g.manual_seed(seed)
    # Slow increasing trend + gentle noise
    trend = torch.linspace(0.7, 1.4, L)
    noise = torch.empty(L).normal_(0, 0.25, generator=g)
    return (trend + noise).clamp(min=0.2)


def _k_norm_dialogue(L: int, seed: int) -> torch.Tensor:
    """
    Dialogue: periodic turn boundaries (~200 tokens) with very high norm at
    the start of each turn, moderate within-turn locality.
    """
    g = torch.Generator(); g.manual_seed(seed)
    turn_len = max(1, L // (L // 200 + 1))
    norms = torch.ones(L) * 0.6
    for i in range(0, L, turn_len):
        # Strong turn-start token
        norms[i] = torch.empty(1).uniform_(3.0, 6.0, generator=g).item()
        # Within-turn: moderate decay
        for j in range(1, min(turn_len, L - i)):
            norms[i + j] = (norms[i] * 0.5 * math.exp(-j / (turn_len * 0.4))
                            + 0.5 + torch.empty(1).normal_(0, 0.15, generator=g).item())
    noise = torch.empty(L).normal_(0, 0.1, generator=g)
    return (norms + noise).clamp(min=0.1)


def _k_norm_structured(L: int, seed: int) -> torch.Tensor:
    """
    Structured text (tables, lists): very sharp periodic peaks at item
    boundaries (~32-64 tokens), very low between.
    """
    g = torch.Generator(); g.manual_seed(seed)
    period = max(1, L // (L // 48 + 1))
    norms = torch.ones(L) * 0.2
    for i in range(0, L, period):
        norms[i] = torch.empty(1).uniform_(4.0, 8.0, generator=g).item()
        if i + 1 < L:
            norms[i + 1] = torch.empty(1).uniform_(1.5, 3.0, generator=g).item()
    noise = torch.empty(L).normal_(0, 0.05, generator=g)
    return (norms + noise).clamp(min=0.05)


_SCENARIOS = {
    "code":       (_k_norm_code,       "Code / program text — strong local + sparse global spikes"),
    "technical":  (_k_norm_technical,  "Technical / scientific — section headers + definition anchors"),
    "narrative":  (_k_norm_narrative,  "Fiction / narrative prose — broad uniform context"),
    "dialogue":   (_k_norm_dialogue,   "Conversational / dialogue — periodic turn-boundary peaks"),
    "structured": (_k_norm_structured, "Lists / tables / structured — very sharp periodic peaks"),
}


# ── QKV builder ───────────────────────────────────────────────────────────────

def generate_qkv(
    scenario: str,
    L: int,
    num_heads: int = 16,
    head_dim: int  = 128,
    seed: int = 42,
    dtype: torch.dtype = torch.bfloat16,
) -> dict:
    """
    Generate (Q, K, V) tensors for a given scenario and length.

    K is constructed so that its per-token norm follows the scenario pattern.
    Q and V are standard Gaussian with moderate norm variation.
    All tensors have shape [1, num_heads, L, head_dim].
    """
    norm_fn, description = _SCENARIOS[scenario]
    g = torch.Generator(); g.manual_seed(seed)

    # K: directional Gaussian, then scale rows by the scenario norm profile
    k_raw = torch.randn(1, num_heads, L, head_dim, generator=g)
    # Normalise each token's raw vector to unit length, then apply norm
    base_norms = k_raw.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    k_unit = k_raw / base_norms                         # [1, H, L, D] unit vectors

    # Per-head slight rotation of the norm profile (different heads attend differently)
    k_tensors = []
    for h in range(num_heads):
        h_seed = seed + h * 1000
        norm_profile = norm_fn(L, h_seed)               # [L]
        head_scale = norm_profile.view(1, 1, L, 1)      # broadcast
        k_tensors.append(k_unit[:, h:h+1, :, :] * head_scale)
    k = torch.cat(k_tensors, dim=1).to(dtype)

    # Q: Gaussian with mild norm variation
    q_raw = torch.randn(1, num_heads, L, head_dim, generator=g)
    q = (q_raw * 1.0 / math.sqrt(head_dim)).to(dtype)

    # V: uniform Gaussian
    v = torch.randn(1, num_heads, L, head_dim, generator=g).to(dtype)

    return {"q": q, "k": k, "v": v,
            "meta": {
                "scenario": scenario,
                "description": description,
                "seq_len": L,
                "num_heads": num_heads,
                "head_dim": head_dim,
                "seed": seed,
            }}


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Generate diverse QKV caches for multi-text study")
    ap.add_argument("--lengths",    nargs="+", type=int,
                    default=[4096, 8192, 16384, 32768, 65536])
    ap.add_argument("--scenarios",  nargs="+", default=list(_SCENARIOS.keys()))
    ap.add_argument("--output-dir", type=Path, default=Path("data/cache"))
    ap.add_argument("--num-heads",  type=int, default=16)
    ap.add_argument("--head-dim",   type=int, default=128)
    ap.add_argument("--seed",       type=int, default=42)
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = []

    for scenario in args.scenarios:
        if scenario not in _SCENARIOS:
            print(f"Unknown scenario: {scenario}")
            continue
        for L in args.lengths:
            fname = args.output_dir / f"qkv_len_{L}_{scenario}_seed_{args.seed}.pt"
            if fname.exists():
                print(f"  [skip] {fname.name} (exists)")
                manifest.append({"file": fname.name, "scenario": scenario, "seq_len": L})
                continue

            print(f"  Generating {scenario:>12} L={L:>6} ...", end=" ", flush=True)
            payload = generate_qkv(
                scenario=scenario,
                L=L,
                num_heads=args.num_heads,
                head_dim=args.head_dim,
                seed=args.seed,
            )
            torch.save({"q": payload["q"], "k": payload["k"], "v": payload["v"],
                        "meta": payload["meta"]}, fname)
            print(f"saved → {fname.name}")
            manifest.append({"file": fname.name, "scenario": scenario,
                              "seq_len": L, **payload["meta"]})

    # Save manifest
    # Legacy multi-scenario format; not used by current eval (see manifest.json + sparse/lm scripts).
    mf_path = args.output_dir / "manifest_diverse.json"
    with open(mf_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest → {mf_path}  ({len(manifest)} files)")


if __name__ == "__main__":
    main()
