#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
REMOTE_PROJECT_ROOT = Path("/root/length-routing-study")

if str(REMOTE_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REMOTE_PROJECT_ROOT / "src"))

from length_routing_study._paths import ensure_external_paths

ensure_external_paths(include_baseline=False)

from length_routing_study.empirical_sweep import (
    PBSConfig,
    FlexConfig,
    load_qkv,
    run_flash,
    run_flex,
    run_pbs_decomposed,
)
from length_routing_study.selection_strategies import (
    GlobalSinkWindow,
    SqrtWindow,
    DecayedWindowSink,
    BlockKNormTopK,
    CoverageTarget,
    AdaptiveCoverage,
    HierarchicalKNorm,
    SampledAttentionTopK,
    run_block_sparse_attention_flex,
)


PBS_FIXED = PBSConfig(
    threshold=0.9,
    segment_size=256,
    block_size=128,
    use_triton=True,
    force_first=True,
)
FLEX_FIXED = FlexConfig(
    gamma=0.95,
    tau=0.1,
    min_budget_frac=0.0,
    block_size=128,
)

ALL_SCENARIOS = [
    "sparse_low",
    "sparse_med",
    "sparse_high",
    "lm_sink_local",
    "lm_sparse_global",
    "lm_hierarchical",
    "lm_local_periodic",
    "lm_mixed",
]


@dataclass
class MethodSpec:
    category: str
    name: str
    kind: str
    config: dict[str, Any]
    strategy: Any = None


def build_methods() -> list[MethodSpec]:
    return [
        MethodSpec(
            category="结构先验",
            name="GlobalSinkWindow",
            kind="mask",
            config={"sink_blocks": 2, "window_blocks": 4},
            strategy=GlobalSinkWindow(),
        ),
        MethodSpec(
            category="结构先验",
            name="SqrtWindow",
            kind="mask",
            config={"sink_blocks": 1},
            strategy=SqrtWindow(),
        ),
        MethodSpec(
            category="结构先验",
            name="DecayedWindowSink",
            kind="mask",
            config={"sink_blocks": 2, "window_blocks": 8, "stride_blocks": 4},
            strategy=DecayedWindowSink(),
        ),
        MethodSpec(
            category="K 范数",
            name="BlockKNormTopK",
            kind="mask",
            config={"topk_frac": 0.25, "sink_blocks": 1},
            strategy=BlockKNormTopK(),
        ),
        MethodSpec(
            category="覆盖率",
            name="CoverageTarget",
            kind="mask",
            config={"target_coverage": 0.90, "sink_blocks": 1},
            strategy=CoverageTarget(),
        ),
        MethodSpec(
            category="覆盖率",
            name="AdaptiveCoverage",
            kind="mask",
            config={
                "base_coverage": 0.90,
                "min_coverage": 0.50,
                "alpha": 0.08,
                "l_ref": 4096,
                "sink_blocks": 1,
            },
            strategy=AdaptiveCoverage(),
        ),
        MethodSpec(
            category="分层 K 范数",
            name="HierarchicalKNorm",
            kind="mask",
            config={
                "coarse_factor": 8,
                "coarse_keep": 0.50,
                "fine_keep": 0.40,
                "sink_blocks": 1,
            },
            strategy=HierarchicalKNorm(),
        ),
        MethodSpec(
            category="采样型",
            name="SampledAttentionTopK",
            kind="mask",
            config={
                "sample_rows": 16,
                "topk_frac": 0.25,
                "sink_blocks": 1,
                "seed": 0,
            },
            strategy=SampledAttentionTopK(),
        ),
        MethodSpec(
            category="完整后端",
            name="PBS",
            kind="pbs",
            config={
                "threshold": 0.9,
                "segment_size": 256,
                "block_size": 128,
                "use_triton": True,
                "force_first": True,
            },
        ),
        MethodSpec(
            category="完整后端",
            name="Flex",
            kind="flex",
            config={
                "gamma": 0.95,
                "tau": 0.1,
                "min_budget_frac": 0.0,
                "block_size": 128,
            },
        ),
    ]


def _kl_div(out: torch.Tensor, ref: torch.Tensor) -> float:
    return float(
        F.kl_div(
            torch.log_softmax(out.float().mean(-1), dim=-1),
            torch.softmax(ref.float().mean(-1), dim=-1),
            reduction="batchmean",
        ).item()
    )


def measure_mask_method(
    method: MethodSpec,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    ref: torch.Tensor,
    block_size: int,
    warmup: int,
    repeats: int,
    mse_threshold: float,
    kl_threshold: float,
) -> dict[str, Any]:
    strategy = method.strategy

    for _ in range(warmup):
        wr = strategy.select(q, k, v, block_size=block_size)
        run_block_sparse_attention_flex(q, k, v, wr.block_mask, block_size=block_size)

    sel_times: list[float] = []
    ker_times: list[float] = []
    last_result = None
    last_out = None
    for _ in range(repeats):
        sr = strategy.select(q, k, v, block_size=block_size)
        out, t_ker = run_block_sparse_attention_flex(q, k, v, sr.block_mask, block_size=block_size)
        sel_times.append(sr.t_select_ms)
        ker_times.append(t_ker)
        last_result = sr
        last_out = out

    assert last_result is not None and last_out is not None

    t_sel = float(sum(sel_times) / len(sel_times))
    t_ker = float(sum(ker_times) / len(ker_times))
    mse = float(torch.mean((last_out.float() - ref.float()) ** 2).item())
    kl = _kl_div(last_out, ref)

    return {
        "category": method.category,
        "method": method.name,
        "config": method.config,
        "t_total_ms": round(t_sel + t_ker, 4),
        "t_select_ms": round(t_sel, 4),
        "t_kernel_ms": round(t_ker, 4),
        "mse": round(mse, 7),
        "kl": round(kl, 7),
        "passed": bool(mse <= mse_threshold and kl <= kl_threshold),
        "active_block_fraction": round(last_result.active_fraction, 4),
        "is_theory": False,
    }


def measure_pbs_method(
    method: MethodSpec,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    ref: torch.Tensor,
    flash_ms: float,
    warmup: int,
    repeats: int,
    mse_threshold: float,
    kl_threshold: float,
) -> dict[str, Any]:
    rec = run_pbs_decomposed(
        q,
        k,
        v,
        PBS_FIXED,
        flash_time_ms=flash_ms,
        reference=ref,
        warmup=warmup,
        repeats=repeats,
        mse_threshold=mse_threshold,
        kl_threshold=kl_threshold,
    )
    return {
        "category": method.category,
        "method": method.name,
        "config": method.config,
        "t_total_ms": round(rec.t_mean_ms, 4),
        "t_select_ms": round(rec.t_select_ms, 4),
        "t_kernel_ms": round(rec.t_kernel_ms, 4),
        "mse": round(rec.mse, 7),
        "kl": round(rec.kl, 7),
        "passed": bool(rec.passed),
        "active_block_fraction": rec.active_block_fraction,
        "kernel_time_ratio": rec.kernel_time_ratio,
        "is_theory": False,
    }


def measure_flex_method(
    method: MethodSpec,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    ref: torch.Tensor,
    warmup: int,
    repeats: int,
    mse_threshold: float,
    kl_threshold: float,
) -> dict[str, Any]:
    rec = run_flex(
        q,
        k,
        v,
        FLEX_FIXED,
        reference=ref,
        warmup=warmup,
        repeats=repeats,
    )
    return {
        "category": method.category,
        "method": method.name,
        "config": method.config,
        "t_total_ms": round(rec.t_mean_ms, 4),
        "t_select_ms": 0.0,
        "t_kernel_ms": round(rec.t_mean_ms, 4),
        "mse": round(rec.mse, 7),
        "kl": round(rec.kl, 7),
        "passed": bool(rec.mse <= mse_threshold and rec.kl <= kl_threshold),
        "active_block_fraction": rec.active_block_fraction,
        "kernel_time_ratio": rec.kernel_time_ratio,
        "is_theory": False,
    }


def aggregate(rows: list[dict[str, Any]], group_keys: list[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = tuple(row[k] for k in group_keys)
        grouped[key].append(row)

    out = []
    for key in sorted(grouped):
        grp = grouped[key]
        base = {k: v for k, v in zip(group_keys, key)}
        base["n_cells"] = len(grp)
        for metric in [
            "t_total_ms",
            "t_select_ms",
            "t_kernel_ms",
            "mse",
            "kl",
            "active_block_fraction",
        ]:
            vals = [r[metric] for r in grp if r.get(metric) is not None]
            base[f"avg_{metric}"] = round(sum(vals) / len(vals), 4 if metric.endswith("_ms") or metric == "active_block_fraction" else 7) if vals else None
        base["pass_rate"] = round(sum(1 for r in grp if r["passed"]) / len(grp), 4)
        out.append(base)
    return out


def render_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    header = "| " + " | ".join(title for _, title in columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in rows:
        vals = []
        for key, _ in columns:
            v = row.get(key)
            if isinstance(v, float):
                if abs(v) >= 100:
                    vals.append(f"{v:.4f}")
                elif key in {"avg_mse", "avg_kl"}:
                    vals.append(f"{v:.7f}")
                else:
                    vals.append(f"{v:.4f}")
            else:
                vals.append(str(v))
        body.append("| " + " | ".join(vals) + " |")
    return "\n".join([header, sep, *body])


def build_report(
    methods: list[MethodSpec],
    overall: list[dict[str, Any]],
    by_scenario: list[dict[str, Any]],
    lengths: list[int],
    scenarios: list[str],
    raw_output_path: Path,
) -> str:
    lines: list[str] = []
    lines.append("# Individual Strategy Benchmark")
    lines.append("")
    lines.append(f"- Lengths: {lengths}")
    lines.append(f"- Scenarios: {scenarios}")
    lines.append(f"- Pass criterion: `MSE <= 0.02` and `KL <= 0.10`")
    lines.append(f"- Raw JSON: `{raw_output_path.name}`")
    lines.append("")
    lines.append("## Method Configurations")
    lines.append("")
    config_rows = [
        {
            "category": m.category,
            "method": m.name,
            "config": json.dumps(m.config, ensure_ascii=False),
        }
        for m in methods
    ]
    lines.append(
        render_table(
            config_rows,
            [
                ("category", "Category"),
                ("method", "Method"),
                ("config", "Config"),
            ],
        )
    )
    lines.append("")
    lines.append("## Overall Average Across All Lengths and Scenarios")
    lines.append("")
    lines.append(
        render_table(
            overall,
            [
                ("category", "Category"),
                ("method", "Method"),
                ("n_cells", "Cells"),
                ("avg_t_total_ms", "Avg Total ms"),
                ("avg_t_select_ms", "Avg Select ms"),
                ("avg_t_kernel_ms", "Avg Kernel ms"),
                ("avg_mse", "Avg MSE"),
                ("avg_kl", "Avg KL"),
                ("avg_active_block_fraction", "Avg ActiveFrac"),
                ("pass_rate", "PassRate"),
            ],
        )
    )
    lines.append("")
    lines.append("## Per-Scenario Average Across Lengths")
    lines.append("")

    for scenario in scenarios:
        lines.append(f"### {scenario}")
        lines.append("")
        rows = [r for r in by_scenario if r["scenario"] == scenario]
        lines.append(
            render_table(
                rows,
                [
                    ("category", "Category"),
                    ("method", "Method"),
                    ("n_cells", "Lengths"),
                    ("avg_t_total_ms", "Avg Total ms"),
                    ("avg_t_select_ms", "Avg Select ms"),
                    ("avg_t_kernel_ms", "Avg Kernel ms"),
                    ("avg_mse", "Avg MSE"),
                    ("avg_kl", "Avg KL"),
                    ("avg_active_block_fraction", "Avg ActiveFrac"),
                    ("pass_rate", "PassRate"),
                ],
            )
        )
        lines.append("")

    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--lengths",
        nargs="+",
        type=int,
        default=[4096, 8192, 16384, 32768, 65536, 131072],
    )
    ap.add_argument(
        "--scenarios",
        nargs="+",
        default=ALL_SCENARIOS,
    )
    ap.add_argument("--cache-dir", type=Path, default=REMOTE_PROJECT_ROOT / "data" / "cache")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--block-size", type=int, default=128)
    ap.add_argument("--mse-threshold", type=float, default=0.02)
    ap.add_argument("--kl-threshold", type=float, default=0.10)
    ap.add_argument(
        "--output-json",
        type=Path,
        default=REMOTE_PROJECT_ROOT / "results" / "individual_strategy_benchmark.json",
    )
    ap.add_argument(
        "--output-md",
        type=Path,
        default=REMOTE_PROJECT_ROOT / "results" / "individual_strategy_benchmark.md",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    methods = build_methods()
    rows: list[dict[str, Any]] = []

    for scenario in args.scenarios:
        for L in args.lengths:
            print(f"\n=== {scenario} @ L={L} ===", flush=True)
            q, k, v = load_qkv(
                args.cache_dir,
                L,
                prompt_family=scenario,
                seed=args.seed,
                device=args.device,
            )
            dtype = q.dtype
            k = k.to(dtype=dtype)
            v = v.to(dtype=dtype)

            flash_rec = run_flash(q, k, v, warmup=args.warmup, repeats=args.repeats)
            with torch.inference_mode():
                ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)

            flash_ms = flash_rec.t_mean_ms
            print(f"Flash ref: {flash_ms:.4f} ms", flush=True)

            for method in methods:
                t0 = time.time()
                if method.kind == "mask":
                    result = measure_mask_method(
                        method,
                        q,
                        k,
                        v,
                        ref,
                        block_size=args.block_size,
                        warmup=args.warmup,
                        repeats=args.repeats,
                        mse_threshold=args.mse_threshold,
                        kl_threshold=args.kl_threshold,
                    )
                elif method.kind == "pbs":
                    result = measure_pbs_method(
                        method,
                        q,
                        k,
                        v,
                        ref,
                        flash_ms=flash_ms,
                        warmup=args.warmup,
                        repeats=args.repeats,
                        mse_threshold=args.mse_threshold,
                        kl_threshold=args.kl_threshold,
                    )
                elif method.kind == "flex":
                    result = measure_flex_method(
                        method,
                        q,
                        k,
                        v,
                        ref,
                        warmup=args.warmup,
                        repeats=args.repeats,
                        mse_threshold=args.mse_threshold,
                        kl_threshold=args.kl_threshold,
                    )
                else:
                    raise ValueError(f"Unknown method kind: {method.kind}")

                result["scenario"] = scenario
                result["seq_len"] = L
                rows.append(result)
                dt = time.time() - t0
                active = result.get("active_block_fraction")
                active_str = f"{active:.4f}" if isinstance(active, (int, float)) else "NA"
                print(
                    f"{method.name:<20} total={result['t_total_ms']:>8.4f} ms  "
                    f"select={result['t_select_ms']:>8.4f} ms  "
                    f"kernel={result['t_kernel_ms']:>8.4f} ms  "
                    f"mse={result['mse']:.7f}  kl={result['kl']:.7f}  "
                    f"pass={'Y' if result['passed'] else 'N'}  "
                    f"active={active_str}  "
                    f"wall={dt:.2f}s",
                    flush=True,
                )

            del q, k, v, ref
            torch.cuda.empty_cache()

    overall = aggregate(rows, ["category", "method"])
    by_scenario = aggregate(rows, ["scenario", "category", "method"])

    payload = {
        "config": {
            "lengths": args.lengths,
            "scenarios": args.scenarios,
            "cache_dir": str(args.cache_dir),
            "warmup": args.warmup,
            "repeats": args.repeats,
            "block_size": args.block_size,
            "mse_threshold": args.mse_threshold,
            "kl_threshold": args.kl_threshold,
        },
        "methods": [
            {
                "category": m.category,
                "method": m.name,
                "kind": m.kind,
                "config": m.config,
            }
            for m in methods
        ],
        "records": rows,
        "overall_average": overall,
        "per_scenario_average": by_scenario,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    report = build_report(methods, overall, by_scenario, args.lengths, args.scenarios, args.output_json)
    args.output_md.write_text(report, encoding="utf-8")

    print(f"\nSaved JSON -> {args.output_json}")
    print(f"Saved MD   -> {args.output_md}")


if __name__ == "__main__":
    main()
