"""
plot_latency_cdfs.py
--------------------
Reads vllm_results.jsonl and plots CDF curves for:
  1. E2E Latency   (e2e_s)
  2. TTFT          (ttft_s)
  3. Mean TBT      (mean_tbt_s)

P50, P95, P99 are marked on every plot.

Usage:
    python plot_latency_cdfs.py [--input vllm_results.jsonl] [--out images]
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

PERCENTILES = [
    (50,  "#facc15", "--"),   # yellow
    (95,  "#f97316", "-."),   # orange
    (99,  "#ef4444", ":"),    # red  ← P99 highlighted
]


# ── data loading ─────────────────────────────────────────────────────────────

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
    """
    Parse the schema produced by the instrumented api_server:
      {"endpoint": "...", "e2e_s": 0.621, "ttft_s": 0.044,
       "mean_tbt_s": 0.013, "tbt_count": 42}
    All values are in seconds — convert to ms for readability.
    """
    e2e, ttft, tbt = [], [], []

    for r in records:
        if "e2e_s" in r:
            e2e.append(r["e2e_s"] * 1000.0)
        if "ttft_s" in r:
            ttft.append(r["ttft_s"] * 1000.0)
        # only include TBT if there were actual inter-token intervals
        if r.get("tbt_count", 0) > 0 and "mean_tbt_s" in r:
            tbt.append(r["mean_tbt_s"] * 1000.0)

    return {
        "E2E Latency":        np.array(sorted(e2e)),
        "TTFT":               np.array(sorted(ttft)),
        "Mean TBT":           np.array(sorted(tbt)),
    }


# ── plotting ─────────────────────────────────────────────────────────────────

def plot_cdf(ax: plt.Axes, data: np.ndarray, color: str, unit: str = "ms") -> None:
    if len(data) == 0:
        ax.text(0.5, 0.5, "No data\n(need streaming or forced-stream)",
                transform=ax.transAxes, ha="center", va="center",
                color="#a0a0b0", fontsize=11)
        return

    # CDF line
    y = np.arange(1, len(data) + 1) / len(data)
    ax.plot(data, y, color=color, linewidth=2.2, zorder=3)
    ax.fill_betweenx(y, data, alpha=0.08, color=color)

    # Percentile markers
    x_range = data[-1] - data[0]
    for p, pc, ls in PERCENTILES:
        pv  = float(np.percentile(data, p))
        cdf = p / 100.0
        ax.axvline(pv, color=pc, linestyle=ls, linewidth=1.5,
                   label=f"P{p} = {pv:.2f} {unit}", zorder=4)
        ax.axhline(cdf, color=pc, linestyle=ls, linewidth=0.7,
                   alpha=0.35, zorder=2)
        # annotate value on the line
        ax.annotate(
            f"P{p}\n{pv:.2f}",
            xy=(pv, cdf),
            xytext=(pv + x_range * 0.04, cdf - 0.08),
            fontsize=8, color=pc,
            arrowprops=dict(arrowstyle="-", color=pc, lw=0.8),
        )

    ax.set_ylim(0, 1.05)
    ax.set_xlim(left=0)
    ax.set_xlabel(f"Latency ({unit})", fontsize=11)
    ax.set_ylabel("CDF", fontsize=11)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
    ax.legend(fontsize=9, loc="lower right")

    # stats box
    stats = (f"n = {len(data)}\n"
             f"min = {data[0]:.2f}  max = {data[-1]:.2f}\n"
             f"mean = {data.mean():.2f}  std = {data.std():.2f}")
    ax.text(0.02, 0.97, stats, transform=ax.transAxes,
            fontsize=8, va="top", ha="left", color="#a0a0b0",
            bbox=dict(facecolor="#0f1117", alpha=0.6, edgecolor="none", pad=4))


def save_individual(data: np.ndarray, title: str, color: str,
                    out_path: str, unit: str = "ms") -> None:
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(8, 5))
        plot_cdf(ax, data, color, unit)
        ax.set_title(f"{title} — CDF", fontsize=14, fontweight="bold", pad=12)
        fig.tight_layout()
        fig.savefig(out_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
    print(f"  saved → {out_path}")


def save_combined(metrics: dict[str, np.ndarray], colors: list[str],
                  out_path: str) -> None:
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(21, 6))
        fig.suptitle("vLLM Inference Latency — CDF Summary",
                     fontsize=16, fontweight="bold", y=1.01)
        for ax, (title, data), color in zip(axes, metrics.items(), colors):
            plot_cdf(ax, data, color)
            ax.set_title(title, fontsize=13, fontweight="bold", pad=8)
        fig.tight_layout()
        fig.savefig(out_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
    print(f"  saved → {out_path}")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Plot latency CDFs from vllm_results.jsonl")
    parser.add_argument("--input", "-i", default="vllm_results.jsonl",
                        help="Path to JSONL file (default: vllm_results.jsonl)")
    parser.add_argument("--out", "-o", default="images",
                        help="Output directory for PNGs (default: images/)")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        sys.exit(f"[error] input file not found: {args.input}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading: {args.input}")
    records = load_records(args.input)
    print(f"  {len(records)} records loaded")

    metrics = extract_metrics(records)
    for k, v in metrics.items():
        print(f"  {k}: {len(v)} data points")

    colors = ["#6366f1", "#22d3ee", "#a78bfa"]   # indigo, cyan, violet

    plots = [
        ("E2E Latency", "e2e_cdf.png"),
        ("TTFT",        "ttft_cdf.png"),
        ("Mean TBT",    "tbt_cdf.png"),
    ]
    for (title, fname), color in zip(plots, colors):
        save_individual(metrics[title], title, color, str(out_dir / fname))

    save_combined(metrics, colors, str(out_dir / "latency_cdf_combined.png"))

    print(f"\nDone. All plots saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
