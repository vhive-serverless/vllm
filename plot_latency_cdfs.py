"""
plot_latency_cdfs.py
--------------------
Reads result_times.json (JSONL) produced by the instrumented vLLM API server
and plots CDF curves for:
  1. E2E Latency
  2. Time-to-First-Token (TTFT)
  3. Time-Between-Tokens (TBT) – mean per request

P50, P95, P99 are marked on every plot.

Usage:
    python plot_latency_cdfs.py [--input results/result_times.json] [--out images]
"""

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


# ── style ────────────────────────────────────────────────────────────────────
STYLE = {
    "figure.facecolor": "#0f1117",
    "axes.facecolor":   "#1a1d27",
    "axes.edgecolor":   "#3a3d4d",
    "axes.labelcolor":  "#e0e0e0",
    "axes.titlecolor":  "#ffffff",
    "axes.grid":        True,
    "grid.color":       "#2a2d3d",
    "grid.linewidth":   0.8,
    "xtick.color":      "#a0a0b0",
    "ytick.color":      "#a0a0b0",
    "text.color":       "#e0e0e0",
    "legend.facecolor": "#1a1d27",
    "legend.edgecolor": "#3a3d4d",
    "font.family":      "monospace",
}

PERCENTILE_LINES = [
    (50,  "#facc15", "--"),   # yellow
    (95,  "#f97316", "-."),   # orange
    (99,  "#ef4444", ":"),    # red
]


def load_records(path: str) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"  [warn] line {lineno}: {exc}", file=sys.stderr)
    return records


def extract_metrics(records: list[dict]) -> dict[str, np.ndarray]:
    e2e, ttft, tbt_mean = [], [], []

    for r in records:
        if r.get("error"):
            continue  # skip errored requests
        if "e2e_ms" in r:
            e2e.append(float(r["e2e_ms"]))
        if "ttft_ms" in r:
            ttft.append(float(r["ttft_ms"]))
        tbt = r.get("tbt", {})
        if tbt and tbt.get("count", 0) > 0:
            tbt_mean.append(float(tbt["mean_ms"]))

    return {
        "E2E Latency":          np.array(sorted(e2e)),
        "TTFT":                 np.array(sorted(ttft)),
        "TBT (mean/request)":   np.array(sorted(tbt_mean)),
    }


def percentile(arr: np.ndarray, p: float) -> float:
    return float(np.percentile(arr, p))


def plot_cdf(
    ax: plt.Axes,
    data: np.ndarray,
    label: str,
    color: str,
    unit: str = "ms",
) -> None:
    if len(data) == 0:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                ha="center", va="center", color="#a0a0b0", fontsize=13)
        return

    # CDF
    y = np.arange(1, len(data) + 1) / len(data)
    ax.plot(data, y, color=color, linewidth=2.2, label=label, zorder=3)
    ax.fill_betweenx(y, data, alpha=0.08, color=color)

    # Percentile markers
    for p, pc, ls in PERCENTILE_LINES:
        pv = percentile(data, p)
        cdf_at_p = p / 100.0
        ax.axvline(pv, color=pc, linestyle=ls, linewidth=1.4,
                   label=f"P{p} = {pv:.1f} {unit}", zorder=4)
        ax.axhline(cdf_at_p, color=pc, linestyle=ls, linewidth=0.7,
                   alpha=0.4, zorder=2)
        # Annotate on the curve
        ax.annotate(
            f"P{p}\n{pv:.1f}",
            xy=(pv, cdf_at_p),
            xytext=(pv + (data[-1] - data[0]) * 0.03, cdf_at_p - 0.07),
            fontsize=8,
            color=pc,
            arrowprops=dict(arrowstyle="-", color=pc, lw=0.8),
        )

    ax.set_ylim(0, 1.05)
    ax.set_xlim(left=0)
    ax.set_xlabel(f"Latency ({unit})", fontsize=11)
    ax.set_ylabel("CDF", fontsize=11)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
    ax.legend(fontsize=9, loc="lower right")

    # Stats box
    stats = (
        f"n={len(data)}\n"
        f"min={data[0]:.1f}  max={data[-1]:.1f}\n"
        f"mean={data.mean():.1f}  std={data.std():.1f}"
    )
    ax.text(
        0.02, 0.97, stats,
        transform=ax.transAxes,
        fontsize=8, va="top", ha="left",
        color="#a0a0b0",
        bbox=dict(facecolor="#0f1117", alpha=0.6, edgecolor="none", pad=4),
    )


def make_plot(
    data: np.ndarray,
    title: str,
    color: str,
    out_path: str,
    unit: str = "ms",
) -> None:
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(8, 5))
        plot_cdf(ax, data, title, color, unit)
        ax.set_title(title + " — CDF", fontsize=14, fontweight="bold", pad=12)
        fig.tight_layout()
        fig.savefig(out_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
    print(f"  saved → {out_path}")


def make_combined_plot(
    metrics: dict[str, np.ndarray],
    colors: list[str],
    out_path: str,
) -> None:
    keys  = list(metrics.keys())
    units = ["ms", "ms", "ms"]

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(21, 6))
        fig.suptitle(
            "vLLM Inference Latency — CDF Summary",
            fontsize=16, fontweight="bold", y=1.01,
        )
        for ax, key, color, unit in zip(axes, keys, colors, units):
            plot_cdf(ax, metrics[key], key, color, unit)
            ax.set_title(key, fontsize=13, fontweight="bold", pad=8)

        fig.tight_layout()
        fig.savefig(out_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
    print(f"  saved → {out_path}")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Plot latency CDFs from vLLM timing JSONL.")
    parser.add_argument(
        "--input", "-i",
        default="results/result_times.json",
        help="Path to the JSONL timing file (default: results/result_times.json)",
    )
    parser.add_argument(
        "--out", "-o",
        default="images",
        help="Output directory for PNG files (default: images/)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        sys.exit(f"[error] input file not found: {args.input}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading records from: {args.input}")
    records = load_records(args.input)
    print(f"  {len(records)} records loaded.")

    metrics = extract_metrics(records)
    for k, v in metrics.items():
        print(f"  {k}: {len(v)} valid data points")

    colors = ["#6366f1", "#22d3ee", "#a78bfa"]   # indigo, cyan, violet

    # Individual plots
    pairs = [
        ("E2E Latency",        "e2e_latency_cdf.png"),
        ("TTFT",               "ttft_cdf.png"),
        ("TBT (mean/request)", "tbt_mean_cdf.png"),
    ]
    for (key, fname), color in zip(pairs, colors):
        make_plot(
            data=metrics[key],
            title=key,
            color=color,
            out_path=str(out_dir / fname),
        )

    # Combined 1×3 summary
    make_combined_plot(
        metrics=metrics,
        colors=colors,
        out_path=str(out_dir / "latency_cdf_combined.png"),
    )

    print("\nDone. Plots saved to:", out_dir.resolve())


if __name__ == "__main__":
    main()
