#!/usr/bin/env python3
"""
Visualize a ``LengthStudyResult`` as a self-contained HTML dashboard.

Produces four chart sections:
  1. Flash / PBS-best / Flex-best latency vs length (line chart)
  2. Theory vs empirical backend agreement per length (bar/table)
  3. Theory latency prediction error % per length
  4. Top-N empirical configs at each length (table)

Usage
-----
  python scripts/visualize_length_study.py \
      --input results/length_study_latest.json \
      --output results/dashboard.html
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import math
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def render_dashboard(study_data: dict, output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    cells = study_data.get("cells", [])
    if not cells:
        print("No cells in study data; nothing to visualize.")
        return

    lengths     = [c["seq_len"] for c in cells]
    flash_ms    = [c.get("flash_ms")    or float("nan") for c in cells]
    pbs_ms      = [c.get("pbs_best_ms") or float("nan") for c in cells]
    flex_ms     = [c.get("flex_best_ms")or float("nan") for c in cells]
    theory_ms   = [c["theory_plan"]["estimated_latency_ms"] for c in cells]
    err_pct     = [c.get("theory_latency_error_pct") for c in cells]
    agrees      = [c["theory_agrees"] for c in cells]
    t_backend   = [c["theory_backend"] for c in cells]
    e_backend   = [c["empirical_winner_backend"] for c in cells]

    COLORS = {"flash": "#4C72B0", "pbs": "#DD8452", "flexprefill": "#55A868", "theory": "#C44E52"}

    # ── Chart 1: latency vs length ────────────────────────────────────────────
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(lengths, flash_ms, "o-",  color=COLORS["flash"],       label="Flash (reference)", lw=2)
    ax1.plot(lengths, pbs_ms,   "s--", color=COLORS["pbs"],         label="PBS best config",   lw=2)
    ax1.plot(lengths, flex_ms,  "^-.", color=COLORS["flexprefill"],  label="Flex best config",  lw=2)
    ax1.plot(lengths, theory_ms,"D:",  color=COLORS["theory"],       label="Theory prediction", lw=1.5, alpha=0.7)
    ax1.set_xscale("log", base=2)
    ax1.set_yscale("log")
    ax1.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: f"{int(x/1024)}K" if x >= 1024 else str(int(x))))
    ax1.set_xlabel("Sequence length (tokens)")
    ax1.set_ylabel("Attention latency (ms, log)")
    ax1.set_title("Latency vs Sequence Length")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    img1 = _fig_to_b64(fig1)
    plt.close(fig1)

    # ── Chart 2: PBS/Flash and Flex/Flash ratios ──────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    pbs_ratio  = [p / f if (f and p and f == f and p == p) else None for p, f in zip(pbs_ms, flash_ms)]
    flex_ratio = [x / f if (f and x and f == f and x == x) else None for x, f in zip(flex_ms, flash_ms)]
    ax2.plot(lengths, pbs_ratio,  "s--", color=COLORS["pbs"],         label="PBS / Flash", lw=2)
    ax2.plot(lengths, flex_ratio, "^-.", color=COLORS["flexprefill"],  label="Flex / Flash", lw=2)
    ax2.axhline(1.0, color="gray", lw=1, linestyle=":")
    ax2.set_xscale("log", base=2)
    ax2.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: f"{int(x/1024)}K" if x >= 1024 else str(int(x))))
    ax2.set_xlabel("Sequence length (tokens)")
    ax2.set_ylabel("Ratio vs Flash")
    ax2.set_title("Sparse / Flash Overhead Ratio (lower → sparse advantage)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    img2 = _fig_to_b64(fig2)
    plt.close(fig2)

    # ── Chart 3: theory prediction error % ───────────────────────────────────
    fig3, ax3 = plt.subplots(figsize=(10, 4))
    valid_err = [(L, e) for L, e in zip(lengths, err_pct) if e is not None and math.isfinite(e)]
    if valid_err:
        Lv, ev = zip(*valid_err)
        bars = ax3.bar([str(L) for L in Lv], ev,
                       color=[("#e74c3c" if e > 0 else "#2ecc71") for e in ev], alpha=0.8)
        ax3.axhline(0, color="black", lw=1)
        ax3.set_xlabel("Sequence length")
        ax3.set_ylabel("Theory error %  (pos = over-estimate)")
        ax3.set_title("Theory vs Empirical Latency Error")
        ax3.grid(True, alpha=0.3, axis="y")
    img3 = _fig_to_b64(fig3)
    plt.close(fig3)

    # ── Build HTML ─────────────────────────────────────────────────────────────
    agree_total = sum(1 for a in agrees if a)

    def _top_table(cell: dict) -> str:
        rows = ""
        for r in cell.get("top_empirical", []):
            mse = f"{r['mse']:.5f}"
            rows += (f"<tr><td>{r['rank']}</td><td><code>{r['config']}</code></td>"
                     f"<td>{r['backend']}</td><td>{r['t_mean_ms']:.3f}</td><td>{mse}</td></tr>")
        return (
            "<table class='top'><thead><tr>"
            "<th>#</th><th>Config</th><th>Backend</th><th>ms</th><th>MSE</th>"
            f"</tr></thead><tbody>{rows}</tbody></table>"
        )

    summary_rows = ""
    for c in cells:
        agree_cls = "agree" if c["theory_agrees"] else "disagree"
        err_s = f"{c.get('theory_latency_error_pct', 'N/A'):+.1f}%" \
            if isinstance(c.get('theory_latency_error_pct'), (int, float)) else "N/A"
        summary_rows += (
            f"<tr class='{agree_cls}'>"
            f"<td>{c['seq_len']}</td>"
            f"<td>{c['theory_backend']}</td>"
            f"<td>{c['empirical_winner_backend']} ({c['empirical_winner']})</td>"
            f"<td>{'✓' if c['theory_agrees'] else '✗'}</td>"
            f"<td>{c.get('theory_latency_ms') or '—':.3f}</td>"
            f"<td>{c.get('empirical_winner_ms') or '—':.3f}</td>"
            f"<td>{err_s}</td>"
            f"<td>{c.get('flash_ms') or '—':.3f}</td>"
            f"<td>{c.get('pbs_best_ms') or '—':.3f}</td>"
            f"<td>{c.get('flex_best_ms') or '—':.3f}</td>"
            "</tr>"
        )

    top_sections = ""
    for c in cells:
        top_sections += f"<h4>L = {c['seq_len']}</h4>{_top_table(c)}"

    html = f"""<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8">
<title>Length Routing Study – Theory vs Empirical</title>
<style>
  body{{font-family:Arial,sans-serif;margin:20px;background:#f8f9fa;color:#212529}}
  h1{{color:#343a40}} h2{{color:#495057;border-bottom:2px solid #dee2e6;padding-bottom:6px}}
  h3{{color:#6c757d}} h4{{color:#868e96}}
  .chart{{text-align:center;margin:20px 0}}
  table{{border-collapse:collapse;width:100%;font-size:13px;margin-bottom:20px}}
  th{{background:#495057;color:#fff;padding:7px 10px}}
  td{{padding:5px 10px;border-bottom:1px solid #dee2e6}}
  tr:hover td{{background:#e9ecef}}
  .agree td{{background:#d4edda}}
  .disagree td{{background:#f8d7da}}
  table.top td,table.top th{{font-size:12px;padding:3px 8px}}
  .pill{{display:inline-block;padding:2px 8px;border-radius:12px;font-size:11px}}
  .ok{{background:#d4edda;color:#155724}} .fail{{background:#f8d7da;color:#721c24}}
  .meta{{background:#fff;padding:12px;border-radius:6px;margin-bottom:20px;
          border:1px solid #dee2e6;font-size:13px}}
  .grid{{display:grid;grid-template-columns:1fr 1fr;gap:20px}}
</style>
</head><body>
<h1>Length-Routing Study: Theory vs Empirical</h1>
<div class="meta">
  <b>Model family:</b> {study_data.get("model_family","?")} &nbsp;&nbsp;
  <b>Objective:</b> {study_data.get("objective","?")} &nbsp;&nbsp;
  <b>Agreement:</b> {agree_total}/{len(agrees)} lengths
  <br>
  <b>Metadata:</b> {json.dumps(study_data.get("metadata",{}), ensure_ascii=False)}
</div>

<h2>1. Latency vs Length</h2>
<div class="chart"><img src="data:image/png;base64,{img1}" style="max-width:900px"></div>

<h2>2. Sparse / Flash Overhead Ratio</h2>
<div class="chart"><img src="data:image/png;base64,{img2}" style="max-width:900px"></div>

<h2>3. Theory Prediction Error</h2>
<div class="chart"><img src="data:image/png;base64,{img3}" style="max-width:900px"></div>

<h2>4. Theory vs Empirical Summary Table</h2>
<table>
<thead><tr>
  <th>Length</th><th>Theory backend</th><th>Empirical winner</th><th>Agree</th>
  <th>Theory pred. ms</th><th>Empirical ms</th><th>Error %</th>
  <th>Flash ms</th><th>PBS best ms</th><th>Flex best ms</th>
</tr></thead>
<tbody>{summary_rows}</tbody>
</table>

<h2>5. Top Empirical Configs per Length</h2>
{top_sections}

</body></html>
"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")
    print(f"Dashboard written → {output}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",  type=Path, required=True,
                    help="length_study_*.json from run_length_study.py")
    ap.add_argument("--output", type=Path, default=None,
                    help="Output HTML path (default: <input>.html)")
    args = ap.parse_args()

    with open(args.input, encoding="utf-8") as f:
        data = json.load(f)

    out = args.output or args.input.with_suffix(".html")
    render_dashboard(data, out)


if __name__ == "__main__":
    main()
