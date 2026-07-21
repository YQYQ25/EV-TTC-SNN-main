#!/usr/bin/env python3
"""绘制固定 ROI 内 1k 事件时间尺度统计图表。"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_scan(scan_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """读取 scan 输出的 steps.csv 和 blocks.csv。"""

    steps = []
    blocks = []
    for seq_dir in sorted(p for p in scan_dir.iterdir() if p.is_dir()):
        if (seq_dir / "steps.csv").exists():
            steps.append(pd.read_csv(seq_dir / "steps.csv"))
        if (seq_dir / "blocks.csv").exists():
            blocks.append(pd.read_csv(seq_dir / "blocks.csv"))
    if not steps or not blocks:
        raise FileNotFoundError(f"{scan_dir} 下没有可绘图的 CSV")
    return pd.concat(steps, ignore_index=True), pd.concat(blocks, ignore_index=True)


def save_hist_cdf(values: np.ndarray, title: str, xlabel: str, hist_path: Path, cdf_path: Path) -> None:
    """保存直方图和 CDF。"""

    values = values[np.isfinite(values)]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(values, bins=80, color="#4c78a8", alpha=0.85)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(hist_path, dpi=150)
    plt.close(fig)

    xs = np.sort(values)
    ys = np.arange(1, len(xs) + 1) / max(len(xs), 1)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(xs, ys, color="#f58518", linewidth=2)
    ax.set_title(title + " CDF")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("cumulative probability")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(cdf_path, dpi=150)
    plt.close(fig)


def plot_sequence(sequence: str, seq_steps: pd.DataFrame, seq_blocks: pd.DataFrame, out_dir: Path) -> None:
    """为单个序列生成指导文件要求的 7 类图。"""

    out_dir.mkdir(parents=True, exist_ok=True)
    save_hist_cdf(
        seq_steps["dt_ms"].to_numpy(),
        f"{sequence} step dt",
        "dt (ms)",
        out_dir / "step_dt_hist.png",
        out_dir / "step_dt_cdf.png",
    )
    save_hist_cdf(
        seq_blocks["block_duration_ms"].to_numpy(),
        f"{sequence} 10-step block duration",
        "block duration (ms)",
        out_dir / "block_duration_hist.png",
        out_dir / "block_duration_cdf.png",
    )

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(seq_steps["step_index"], seq_steps["dt_ms"], linewidth=0.7)
    ax.set_title(f"{sequence} dt over steps")
    ax.set_xlabel("step index")
    ax.set_ylabel("dt (ms)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "dt_over_steps.png", dpi=150)
    plt.close(fig)

    roi_stats = seq_steps.groupby(["roi_x0", "roi_y0"], as_index=False)["dt_ms"].mean()
    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(roi_stats["roi_x0"], roi_stats["roi_y0"], c=roi_stats["dt_ms"], cmap="viridis", s=18)
    ax.set_title(f"{sequence} ROI position vs mean dt")
    ax.set_xlabel("roi_x0")
    ax.set_ylabel("roi_y0")
    ax.set_xlim(0, 232)
    ax.set_ylim(0, 232)
    ax.invert_yaxis()
    fig.colorbar(sc, ax=ax, label="mean dt (ms)")
    fig.tight_layout()
    fig.savefig(out_dir / "roi_position_dt_scatter.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    sample = seq_blocks.iloc[: min(200, len(seq_blocks))]
    for _, row in sample.iterrows():
        rect = plt.Rectangle((row.block_roi_x0, row.block_roi_y0), 128, 128, fill=False, alpha=0.15, linewidth=0.8)
        ax.add_patch(rect)
    ax.set_title(f"{sequence} ROI switching overview")
    ax.set_xlabel("x in 360x360")
    ax.set_ylabel("y in 360x360")
    ax.set_xlim(0, 360)
    ax.set_ylim(360, 0)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(out_dir / "roi_switching_overview.png", dpi=150)
    plt.close(fig)


def plot_summary(steps: pd.DataFrame, blocks: pd.DataFrame, out_dir: Path) -> None:
    """生成跨序列汇总图。"""

    out_dir.mkdir(parents=True, exist_ok=True)
    seqs = sorted(steps["sequence_name"].unique())
    dt_data = [steps.loc[steps["sequence_name"] == s, "dt_ms"].to_numpy() for s in seqs]
    block_data = [blocks.loc[blocks["sequence_name"] == s, "block_duration_ms"].to_numpy() for s in seqs]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.boxplot(dt_data, tick_labels=seqs, showfliers=False)
    ax.set_title("7train step dt boxplot")
    ax.set_ylabel("dt (ms)")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(out_dir / "7train_step_dt_boxplot.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.boxplot(block_data, tick_labels=seqs, showfliers=False)
    ax.set_title("7train 10-step block duration boxplot")
    ax.set_ylabel("block duration (ms)")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(out_dir / "7train_block_duration_boxplot.png", dpi=150)
    plt.close(fig)

    spot = steps.loc[steps["platform"] == "spot", "dt_ms"]
    car = steps.loc[steps["platform"] == "car", "dt_ms"]
    if len(spot) > 0 and len(car) > 0:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.boxplot([spot, car], tick_labels=["spot", "car"], showfliers=False)
        ax.set_title("Spot vs Car step dt")
        ax.set_ylabel("dt (ms)")
        fig.tight_layout()
        fig.savefig(out_dir / "spot_vs_car_step_dt.png", dpi=150)
        plt.close(fig)

    target_csv = out_dir.parent.parent / "target_window_event_counts.csv"
    if target_csv.exists():
        targets = pd.read_csv(target_csv)
        subset = targets[targets["group"].isin(["all", "spot", "car"])]
        fig, ax = plt.subplots(figsize=(8, 4))
        for group, gdf in subset.groupby("group"):
            ax.plot(gdf["target_window_ms"], gdf["median_events"], marker="o", label=group)
        ax.set_title("Estimated events for target time window")
        ax.set_xlabel("target window (ms)")
        ax.set_ylabel("median ROI events")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "target_window_event_counts.png", dpi=150)
        plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan-dir", type=Path, default=Path("EV-TTC-SNN-main/analysis/roi_1k_dt_distribution"))
    args = parser.parse_args()
    steps, blocks = load_scan(args.scan_dir)
    fig_dir = args.scan_dir / "figures"
    for sequence, seq_steps in steps.groupby("sequence_name"):
        seq_blocks = blocks[blocks["sequence_name"] == sequence]
        plot_sequence(sequence, seq_steps, seq_blocks, fig_dir / sequence)
    plot_summary(steps, blocks, fig_dir / "summary")
    print(f"figures: {fig_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
