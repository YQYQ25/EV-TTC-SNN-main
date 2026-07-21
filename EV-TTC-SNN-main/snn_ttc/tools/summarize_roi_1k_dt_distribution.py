#!/usr/bin/env python3
"""汇总固定 ROI 内 1k 事件时间尺度扫描结果，并生成中文报告。"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd


DT_THRESHOLDS_US = [100, 500, 1000, 3300, 7000, 10000]
BLOCK_THRESHOLDS_US = [1000, 3300, 7000, 10000]
TARGET_WINDOWS_US = [1000, 3300, 7000, 10000]


def _fmt(value: float, digits: int = 3) -> str:
    if value is None or not np.isfinite(value):
        return "nan"
    if value != 0 and abs(value) < 1e-3:
        return f"{value:.3e}"
    return f"{value:.{digits}f}"


def _series_stats(values: Iterable[float], percentiles=(1, 5, 25, 50, 75, 95, 99)) -> Dict[str, float]:
    """返回常用统计量，空序列返回 nan。"""

    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        out = {"count": 0, "min": np.nan, "max": np.nan, "mean": np.nan, "std": np.nan}
        out.update({f"p{p}": np.nan for p in percentiles})
        return out
    out = {
        "count": int(arr.size),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
    }
    out.update({f"p{p}": float(np.percentile(arr, p)) for p in percentiles})
    return out


def _stat_table_row(name: str, stats: Dict[str, float], unit: str) -> str:
    return (
        f"| {name} | {int(stats['count'])} | {_fmt(stats['min'])} | {_fmt(stats['max'])} | "
        f"{_fmt(stats['mean'])} | {_fmt(stats['std'])} | {_fmt(stats['p1'])} | "
        f"{_fmt(stats['p5'])} | {_fmt(stats['p25'])} | {_fmt(stats['p50'])} | "
        f"{_fmt(stats['p75'])} | {_fmt(stats['p95'])} | {_fmt(stats['p99'])} | {unit} |"
    )


def load_scan(scan_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """读取 scan 脚本输出的所有 per-sequence CSV。"""

    meta_path = scan_dir / "scan_run_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    step_frames = []
    block_frames = []
    for seq_dir in sorted(p for p in scan_dir.iterdir() if p.is_dir()):
        step_csv = seq_dir / "steps.csv"
        block_csv = seq_dir / "blocks.csv"
        if step_csv.exists():
            step_frames.append(pd.read_csv(step_csv))
        if block_csv.exists():
            block_frames.append(pd.read_csv(block_csv))
    if not step_frames or not block_frames:
        raise FileNotFoundError(f"{scan_dir} 下没有可汇总的 steps.csv / blocks.csv")
    return pd.concat(step_frames, ignore_index=True), pd.concat(block_frames, ignore_index=True), meta


def target_window_event_counts(steps: pd.DataFrame) -> pd.DataFrame:
    """用局部 1k 事件率反推目标时间窗内大约需要多少 ROI 源事件。"""

    rows = []
    rate_per_us = 1000.0 / steps["dt"].clip(lower=1)
    for target_us in TARGET_WINDOWS_US:
        est = rate_per_us * target_us
        stats = _series_stats(est, percentiles=(5, 50, 95))
        rows.append(
            {
                "target_window_us": target_us,
                "target_window_ms": target_us / 1000.0,
                "mean_events": stats["mean"],
                "median_events": stats["p50"],
                "p5_events": stats["p5"],
                "p95_events": stats["p95"],
            }
        )
    return pd.DataFrame(rows)


def write_sequence_report(sequence: str, seq_steps: pd.DataFrame, seq_blocks: pd.DataFrame, out_path: Path) -> None:
    """写单个序列的完整时间尺度报告。"""

    dt_stats_us = _series_stats(seq_steps["dt"])
    dt_stats_ms = _series_stats(seq_steps["dt_ms"])
    rate_stats = _series_stats(seq_steps["event_rate_ev_per_s"])
    block_stats_us = _series_stats(seq_blocks["block_duration"])
    raw_span_stats = _series_stats(seq_steps["raw_event_index_span"])

    lines = [
        f"# {sequence} 完整分片时间统计",
        "",
        "## 单步 1k ROI 事件时间长度",
        "| 序列 | 分片数 | min | max | mean | std | P1 | P5 | P25 | P50 | P75 | P95 | P99 | 单位 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        _stat_table_row(sequence, dt_stats_us, "us"),
        _stat_table_row(sequence, dt_stats_ms, "ms"),
        "",
        "## 事件率",
        f"- 平均事件率：`{_fmt(rate_stats['mean'])}` events/s",
        f"- 中位事件率：`{_fmt(rate_stats['p50'])}` events/s",
        "",
        "## dt 阈值占比",
    ]
    for threshold in DT_THRESHOLDS_US:
        ratio = float((seq_steps["dt"] < threshold).mean() * 100.0)
        lines.append(f"- dt < {threshold / 1000:g} ms：{_fmt(ratio)}%")

    lines.extend(
        [
            "",
            "## 10 步 BPTT 物理时间跨度",
            f"- block 数：{len(seq_blocks)}",
            f"- mean/std：{_fmt(block_stats_us['mean'])} / {_fmt(block_stats_us['std'])} us",
            f"- P5/P50/P95：{_fmt(block_stats_us['p5'])} / {_fmt(block_stats_us['p50'])} / {_fmt(block_stats_us['p95'])} us",
            f"- min/max：{_fmt(block_stats_us['min'])} / {_fmt(block_stats_us['max'])} us",
            "",
            "## block_duration 阈值占比",
        ]
    )
    for threshold in BLOCK_THRESHOLDS_US:
        ratio = float((seq_blocks["block_duration"] < threshold).mean() * 100.0)
        lines.append(f"- block_duration < {threshold / 1000:g} ms：{_fmt(ratio)}%")

    lines.extend(
        [
            "",
            "## 原始事件索引跨度",
            f"- mean/std：{_fmt(raw_span_stats['mean'])} / {_fmt(raw_span_stats['std'])} events",
            f"- P5/P50/P95：{_fmt(raw_span_stats['p5'])} / {_fmt(raw_span_stats['p50'])} / {_fmt(raw_span_stats['p95'])} events",
            "",
            "## ROI 位置敏感性摘要",
            f"- ROI 数量：{seq_blocks[['block_roi_x0', 'block_roi_y0']].drop_duplicates().shape[0]}",
            f"- 每 ROI dt 均值范围：{_fmt(seq_steps.groupby(['roi_x0', 'roi_y0'])['dt'].mean().min())} - {_fmt(seq_steps.groupby(['roi_x0', 'roi_y0'])['dt'].mean().max())} us",
            "",
        ]
    )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def write_summary_reports(steps: pd.DataFrame, blocks: pd.DataFrame, scan_meta: dict, out_dir: Path) -> None:
    """生成 7train 总报告、ROI 分析和目标时间窗事件数报告。"""

    report_dir = out_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    sequence_rows = []
    for sequence, seq_steps in steps.groupby("sequence_name"):
        seq_blocks = blocks[blocks["sequence_name"] == sequence]
        dt_stats = _series_stats(seq_steps["dt"])
        block_stats = _series_stats(seq_blocks["block_duration"])
        sequence_rows.append(
            {
                "sequence_name": sequence,
                "platform": seq_steps["platform"].iloc[0],
                "steps": len(seq_steps),
                "blocks": len(seq_blocks),
                "dt_mean_us": dt_stats["mean"],
                "dt_p50_us": dt_stats["p50"],
                "dt_p95_us": dt_stats["p95"],
                "block_mean_us": block_stats["mean"],
                "block_p50_us": block_stats["p50"],
                "block_p95_us": block_stats["p95"],
                "dt_lt_1ms_percent": float((seq_steps["dt"] < 1000).mean() * 100),
                "block_lt_10ms_percent": float((seq_blocks["block_duration"] < 10000).mean() * 100),
            }
        )
    seq_summary = pd.DataFrame(sequence_rows).sort_values("sequence_name")
    seq_summary.to_csv(out_dir / "sequence_summary.csv", index=False)

    lines = [
        "# M3ED 7train 固定 1k 事件时间尺度汇总",
        "",
        "## 数据来源",
        f"- train_h5：`{scan_meta.get('train_h5', '')}`",
        f"- 扫描序列：{', '.join(scan_meta.get('sequences', []))}",
        f"- 扫描耗时：{_fmt(scan_meta.get('elapsed_seconds', np.nan))} s",
        "",
        "## 序列表",
        "| 序列 | 平台 | steps | blocks | dt mean(us) | dt P50(us) | dt P95(us) | 10步 mean(us) | 10步 P50(us) | 10步 P95(us) | dt<1ms | block<10ms |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in seq_summary.iterrows():
        lines.append(
            f"| {row.sequence_name} | {row.platform} | {int(row.steps)} | {int(row.blocks)} | "
            f"{_fmt(row.dt_mean_us)} | {_fmt(row.dt_p50_us)} | {_fmt(row.dt_p95_us)} | "
            f"{_fmt(row.block_mean_us)} | {_fmt(row.block_p50_us)} | {_fmt(row.block_p95_us)} | "
            f"{_fmt(row.dt_lt_1ms_percent)}% | {_fmt(row.block_lt_10ms_percent)}% |"
        )

    for group_name, group_steps in [("全部", steps), ("Spot", steps[steps["platform"] == "spot"]), ("Car", steps[steps["platform"] == "car"])]:
        if group_steps.empty:
            continue
        group_blocks = blocks[blocks["sequence_name"].isin(group_steps["sequence_name"].unique())]
        dts = _series_stats(group_steps["dt"])
        bstats = _series_stats(group_blocks["block_duration"])
        lines.extend(
            [
                "",
                f"## {group_name} 汇总",
                f"- 单步 dt mean / P50 / P95：{_fmt(dts['mean'])} / {_fmt(dts['p50'])} / {_fmt(dts['p95'])} us",
                f"- 10步 block mean / P50 / P95：{_fmt(bstats['mean'])} / {_fmt(bstats['p50'])} / {_fmt(bstats['p95'])} us",
                f"- dt < 1ms 比例：{_fmt(float((group_steps['dt'] < 1000).mean() * 100))}%",
                f"- 10步 block < 10ms 比例：{_fmt(float((group_blocks['block_duration'] < 10000).mean() * 100))}%",
            ]
        )

    all_dt = _series_stats(steps["dt"])
    all_block = _series_stats(blocks["block_duration"])
    decision = (
        "固定 1k 事件不适合直接作为 M3ED 的通用 SNN 时间步；建议优先比较增大每步事件数 N 或改用固定时间窗。"
        if all_dt["p50"] < 1000 and all_block["p50"] < 10000
        else "固定 1k 事件在当前数据上不是明显过短，需要结合训练效果进一步验证。"
    )
    lines.extend(
        [
            "",
            "## 最终判断",
            f"- 全部数据单步 dt 中位数：{_fmt(all_dt['p50'])} us。",
            f"- 全部数据 10 步 BPTT 时长中位数：{_fmt(all_block['p50'])} us。",
            f"- 判断：{decision}",
            "- 后续建议同时比较：方案A 增大每步事件数 N；方案B 增大 BPTT 长度 K；方案C 固定时间窗。",
        ]
    )
    (report_dir / "M3ED_7train固定1k事件时间尺度汇总.md").write_text("\n".join(lines), encoding="utf-8")

    roi = (
        steps.groupby(["sequence_name", "platform", "roi_x0", "roi_y0"])
        .agg(dt_mean_us=("dt", "mean"), dt_median_us=("dt", "median"), event_rate_mean=("event_rate_ev_per_s", "mean"))
        .reset_index()
    )
    roi_block = (
        blocks.groupby(["sequence_name", "platform", "block_roi_x0", "block_roi_y0"])
        .agg(block_duration_mean_us=("block_duration", "mean"), block_count=("block_index", "count"))
        .reset_index()
        .rename(columns={"block_roi_x0": "roi_x0", "block_roi_y0": "roi_y0"})
    )
    roi = roi.merge(roi_block, on=["sequence_name", "platform", "roi_x0", "roi_y0"], how="left")
    roi.to_csv(out_dir / "roi_sensitivity_summary.csv", index=False)
    roi_lines = [
        "# ROI 位置敏感性分析",
        "",
        f"- ROI 总数：{len(roi)}",
        f"- dt 均值范围：{_fmt(roi['dt_mean_us'].min())} - {_fmt(roi['dt_mean_us'].max())} us",
        f"- dt 中位数范围：{_fmt(roi['dt_median_us'].min())} - {_fmt(roi['dt_median_us'].max())} us",
        f"- 事件率均值范围：{_fmt(roi['event_rate_mean'].min())} - {_fmt(roi['event_rate_mean'].max())} events/s",
        "",
        "## 结论",
        "- 不同 ROI 的事件率差异会直接改变 1k 事件对应的物理时间。",
        "- ROI 切换处后续训练应执行 `reset_states()`，避免空间状态跨 ROI 传递。",
    ]
    (report_dir / "ROI位置敏感性分析.md").write_text("\n".join(roi_lines), encoding="utf-8")

    target_rows = []
    for name, group in [("all", steps), ("spot", steps[steps["platform"] == "spot"]), ("car", steps[steps["platform"] == "car"])]:
        if group.empty:
            continue
        tdf = target_window_event_counts(group)
        tdf.insert(0, "group", name)
        target_rows.append(tdf)
    for sequence, seq_steps in steps.groupby("sequence_name"):
        tdf = target_window_event_counts(seq_steps)
        tdf.insert(0, "group", sequence)
        target_rows.append(tdf)
    target_df = pd.concat(target_rows, ignore_index=True)
    target_df.to_csv(out_dir / "target_window_event_counts.csv", index=False)
    target_lines = [
        "# 目标时间窗对应事件数量统计",
        "",
        "说明：这里使用每个 1k 分片的局部事件率反推目标时间窗内的 ROI 源事件数。",
        "",
        "| group | target(ms) | mean events | median events | P5 | P95 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in target_df.iterrows():
        target_lines.append(
            f"| {row['group']} | {_fmt(row.target_window_ms)} | {_fmt(row.mean_events)} | "
            f"{_fmt(row.median_events)} | {_fmt(row.p5_events)} | {_fmt(row.p95_events)} |"
        )
    (report_dir / "目标时间窗对应事件数量统计.md").write_text("\n".join(target_lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan-dir", type=Path, default=Path("EV-TTC-SNN-main/analysis/roi_1k_dt_distribution"))
    args = parser.parse_args()
    steps, blocks, meta = load_scan(args.scan_dir)
    report_dir = args.scan_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    for sequence, seq_steps in steps.groupby("sequence_name"):
        seq_blocks = blocks[blocks["sequence_name"] == sequence]
        name = "spot_outdoor_day_skatepark_1完整分片时间统计.md" if sequence == "spot_outdoor_day_skatepark_1" else f"{sequence}完整分片时间统计.md"
        write_sequence_report(sequence, seq_steps, seq_blocks, report_dir / name)
    write_summary_reports(steps, blocks, meta, args.scan_dir)
    run_doc = [
        "# 实际运行命令与耗时",
        "",
        "```bash",
        "EV-TTC-main/.venv/bin/python EV-TTC-SNN-main/snn_ttc/tools/scan_roi_1k_dt_distribution.py",
        "EV-TTC-main/.venv/bin/python EV-TTC-SNN-main/snn_ttc/tools/summarize_roi_1k_dt_distribution.py",
        "EV-TTC-main/.venv/bin/python EV-TTC-SNN-main/snn_ttc/tools/plot_roi_1k_dt_distribution.py",
        "```",
        "",
        f"- scan started_at：{meta.get('started_at', '')}",
        f"- scan finished_at：{meta.get('finished_at', '')}",
        f"- scan elapsed_seconds：{_fmt(meta.get('elapsed_seconds', np.nan))}",
        "- 硬件占用：扫描阶段只读 CPU/HDF5/磁盘，不使用 GPU。",
    ]
    (report_dir / "实际运行命令与耗时.md").write_text("\n".join(run_doc), encoding="utf-8")
    print(f"reports: {report_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

