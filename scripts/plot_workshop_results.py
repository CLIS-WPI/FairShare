#!/usr/bin/env python3
"""
Generate publication-quality plots for the workshop revision.

Reads JSON results produced by run_workshop_revision.py and creates:
  1. Time-series Δ_geo per snapshot (one line per policy)
  2. Constellation comparison bar chart
  3. SINR with/without interference comparison

Usage:
    python scripts/plot_workshop_results.py
    python scripts/plot_workshop_results.py --input-dir results/workshop_revision
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


POLICY_LABELS = {
    "equal_static": "Equal Static",
    "snr_priority": "SNR Priority",
    "demand_proportional": "Demand Prop.",
    "fairshare": "FairShare",
}
POLICY_COLORS = {
    "equal_static": "#4CAF50",
    "snr_priority": "#F44336",
    "demand_proportional": "#FF9800",
    "fairshare": "#2196F3",
}
POLICY_MARKERS = {
    "equal_static": "s",
    "snr_priority": "^",
    "demand_proportional": "D",
    "fairshare": "o",
}


def plot_timeseries(snapshots: list, constellation_name: str, out_dir: Path):
    """Δ_geo per snapshot for each policy."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    n = len(snapshots)
    x = np.arange(1, n + 1)

    for p in ["equal_static", "snr_priority", "demand_proportional", "fairshare"]:
        y = [s[p]["delta_geo"] for s in snapshots]
        ax.plot(x, y, marker=POLICY_MARKERS[p], color=POLICY_COLORS[p],
                label=POLICY_LABELS[p], linewidth=1.8, markersize=5)

    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, label="Perfect fairness")
    ax.set_xlabel("Snapshot Index", fontsize=12)
    ax.set_ylabel(r"$\Delta_{\mathrm{geo}}$", fontsize=13)
    ax.set_title(f"Geographic Disparity Over Orbital Snapshots — {constellation_name}",
                 fontsize=12)
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.5, n + 0.5)

    path = out_dir / f"fig_timeseries_{constellation_name.replace(' ', '_').lower()}.pdf"
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

    # Also save PNG for quick viewing
    png_path = path.with_suffix(".png")
    fig2, ax2 = plt.subplots(figsize=(8, 4.5))
    for p in ["equal_static", "snr_priority", "demand_proportional", "fairshare"]:
        y = [s[p]["delta_geo"] for s in snapshots]
        ax2.plot(x, y, marker=POLICY_MARKERS[p], color=POLICY_COLORS[p],
                 label=POLICY_LABELS[p], linewidth=1.8, markersize=5)
    ax2.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, label="Perfect fairness")
    ax2.set_xlabel("Snapshot Index", fontsize=12)
    ax2.set_ylabel(r"$\Delta_{\mathrm{geo}}$", fontsize=13)
    ax2.set_title(f"Geographic Disparity Over Orbital Snapshots — {constellation_name}",
                  fontsize=12)
    ax2.legend(fontsize=9, loc="best")
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0.5, n + 0.5)
    fig2.tight_layout()
    fig2.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig2)


def plot_constellation_comparison(summary: dict, out_dir: Path):
    """Bar chart: Δ_geo across constellations for Priority vs FairShare."""
    names = []
    pri_means, pri_stds = [], []
    fs_means, fs_stds = [], []

    for ckey, res in summary.items():
        cname = res.get("constellation", {}).get("name", ckey)
        names.append(cname)
        pri_means.append(res["snr_priority"]["delta_geo"]["mean"])
        pri_stds.append(res["snr_priority"]["delta_geo"]["std"])
        fs_means.append(res["fairshare"]["delta_geo"]["mean"])
        fs_stds.append(res["fairshare"]["delta_geo"]["std"])

    x = np.arange(len(names))
    w = 0.35

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - w / 2, pri_means, w, yerr=pri_stds, label="SNR Priority",
           color=POLICY_COLORS["snr_priority"], capsize=4, alpha=0.85)
    ax.bar(x + w / 2, fs_means, w, yerr=fs_stds, label="FairShare",
           color=POLICY_COLORS["fairshare"], capsize=4, alpha=0.85)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylabel(r"$\Delta_{\mathrm{geo}}$", fontsize=13)
    ax.set_title("Geographic Disparity Across Constellations", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    path = out_dir / "fig_constellation_comparison.pdf"
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_interference_comparison(with_intf: dict, no_intf: dict, ckey: str, out_dir: Path):
    """Grouped bar chart: Δ_geo with vs without interference."""
    policies = ["equal_static", "snr_priority", "demand_proportional", "fairshare"]
    labels = [POLICY_LABELS[p] for p in policies]

    dg_nointf = [no_intf[p]["delta_geo"]["mean"] for p in policies]
    dg_intf = [with_intf[p]["delta_geo"]["mean"] for p in policies]

    x = np.arange(len(policies))
    w = 0.35

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - w / 2, dg_nointf, w, label="SNR only (no interference)",
           color="#90CAF9", edgecolor="#1565C0", linewidth=0.8)
    ax.bar(x + w / 2, dg_intf, w, label="With interference",
           color="#EF9A9A", edgecolor="#C62828", linewidth=0.8)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel(r"$\Delta_{\mathrm{geo}}$", fontsize=13)
    cname = with_intf.get("constellation", {}).get("name", ckey)
    ax.set_title(f"Impact of Interference on Geographic Disparity — {cname}", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    path = out_dir / f"fig_interference_{ckey}.pdf"
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="results/workshop_revision")
    args = parser.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = in_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load summary
    summary_path = in_dir / "summary.json"
    if not summary_path.exists():
        print(f"ERROR: {summary_path} not found. Run run_workshop_revision.py first.")
        return

    with open(summary_path) as f:
        summary = json.load(f)

    # Time-series plots (per constellation)
    for ckey in summary:
        snap_path = in_dir / f"snapshots_{ckey}.json"
        if snap_path.exists():
            with open(snap_path) as f:
                snapshots = json.load(f)
            cname = summary[ckey].get("constellation", {}).get("name", ckey)
            plot_timeseries(snapshots, cname, out_dir)

    # Constellation comparison
    if len(summary) > 1:
        plot_constellation_comparison(summary, out_dir)

    # Interference comparison
    nointf_path = in_dir / "summary_no_interference.json"
    if nointf_path.exists():
        with open(nointf_path) as f:
            nointf = json.load(f)
        for ckey in summary:
            if ckey in nointf:
                plot_interference_comparison(summary[ckey], nointf[ckey], ckey, out_dir)

    print("\n  All figures saved to:", out_dir)


if __name__ == "__main__":
    main()
