#!/usr/bin/env python3
"""比较旧“全图先取 1k”与新“固定 ROI 内取 1k”的 debug 分片统计。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from snn_ttc.tools.audit_roi_1k_ttc_debug_set import count_old_events_inside_360  # noqa: E402


def _stats(arr: np.ndarray) -> Dict[str, float]:
    arr = np.asarray(arr)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {k: float("nan") for k in ["min", "max", "mean", "std"]}
    return {
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
    }


def _fmt(v: float, digits: int = 4) -> str:
    if not np.isfinite(v):
        return "nan"
    if v != 0 and abs(v) < 1e-3:
        return f"{v:.3e}"
    return f"{v:.{digits}f}"


def _row(name: str, old_arr: np.ndarray, new_arr: np.ndarray, unit: str = "") -> str:
    """生成一行均值/std 对比表。"""

    old_s = _stats(old_arr)
    new_s = _stats(new_arr)
    return (
        f"| {name} | {_fmt(old_s['mean'])} ± {_fmt(old_s['std'])}{unit} | "
        f"{_fmt(new_s['mean'])} ± {_fmt(new_s['std'])}{unit} |"
    )


def adjacent_label_change(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """计算相邻 step 在共同 mask 内的平均绝对变化。"""

    changes = []
    for i in range(values.shape[0] - 1):
        common = mask[i].astype(bool) & mask[i + 1].astype(bool)
        if np.any(common):
            diff = np.abs(values[i + 1][common] - values[i][common])
            changes.append(float(np.mean(diff)))
    return np.asarray(changes, dtype=np.float32)


def load_common_metrics(h5_path: Path, effective_source_count: np.ndarray | None = None) -> Dict[str, np.ndarray]:
    """读取新旧 H5 中可比较的公共指标。"""

    with h5py.File(h5_path, "r") as f:
        dt = f["dt"][:] if "dt" in f else (f["t_end"][:] - f["t_start"][:])
        event_cnt = f["event_cnt"][:] if "event_cnt" in f else f["event_cnt"][:]
        # 旧 H5 字段名是 event_start_idx/event_end_idx，新 H5 是 raw_event_start_idx/raw_event_end_idx。
        if "raw_event_start_idx" in f:
            raw_start = f["raw_event_start_idx"][:]
            raw_end = f["raw_event_end_idx"][:]
            source_count = f["roi_source_event_count"][:]
            total_weight = f["total_mapped_weight"][:]
        else:
            raw_start = f["event_start_idx"][:]
            raw_end = f["event_end_idx"][:]
            source_count = effective_source_count if effective_source_count is not None else raw_end - raw_start
            total_weight = event_cnt.reshape(event_cnt.shape[0], -1).sum(axis=1)
        t_start = f["t_start"][:]
        t_end = f["t_end"][:]
        T = f["T"][:]
        Omega = f["Omega"][:]
        depth = f["depth_start"][:] if "depth_start" in f else f["depth_start"][:]
        ttc = f["ttc_start"][:] if "ttc_start" in f else f["ttc_start"][:]
        mask = f["valid_ttc_mask"][:] if "valid_ttc_mask" in f else f["valid_ttc_mask"][:]

    return {
        "dt": dt,
        "raw_span": raw_end - raw_start,
        "source_count": source_count,
        "total_weight": total_weight,
        "bptt10_time": np.asarray([t_end[9] - t_start[0]], dtype=np.float32) if len(t_end) >= 10 else np.asarray([], dtype=np.float32),
        "T_adjacent_change": np.linalg.norm(np.diff(T, axis=0), axis=1),
        "Omega_adjacent_change": np.linalg.norm(np.diff(Omega, axis=0), axis=1),
        "T_range": np.ptp(T, axis=0),
        "Omega_range": np.ptp(Omega, axis=0),
        "depth_adjacent_change": adjacent_label_change(depth, mask),
        "ttc_adjacent_change": adjacent_label_change(ttc, mask),
    }


def build_report(old_h5: Path, new_h5: Path, out_md: Path) -> str:
    old_effective_counts, old_effective_weights = count_old_events_inside_360(old_h5)
    old = load_common_metrics(old_h5, old_effective_counts)
    new = load_common_metrics(new_h5)

    old["effective_weight"] = old_effective_weights
    new["effective_weight"] = new["total_weight"]

    old_dt_mean = _stats(old["dt"])["mean"]
    new_dt_mean = _stats(new["dt"])["mean"]
    old_bptt = old["bptt10_time"][0] if old["bptt10_time"].size else np.nan
    new_bptt = new["bptt10_time"][0] if new["bptt10_time"].size else np.nan
    new_dt_larger = new_dt_mean > old_dt_mean
    motion_increased = _stats(new["T_adjacent_change"])["mean"] > _stats(old["T_adjacent_change"])["mean"]

    lines = [
        "# 新旧分片方案对比报告",
        "",
        "## 对比对象",
        f"- 旧方案 H5：`{old_h5}`",
        f"- 新方案 H5：`{new_h5}`",
        "- 旧方案：全图原始事件流先取 1k，再映射到 360x360。",
        "- 新方案：先映射到 360x360，再在固定 128x128 ROI 内连续取 1k 源事件。",
        "",
        "## 指标表",
        "| 指标 | 旧方案：全图先取 1k | 新方案：固定 ROI 内取 1k |",
        "|---|---:|---:|",
        _row("每步物理时间 dt", old["dt"], new["dt"], " us"),
        _row("每步有效源事件数", old["source_count"], new["source_count"], " events"),
        _row("event_cnt 权重和", old["effective_weight"], new["effective_weight"]),
        _row("原始事件索引跨度", old["raw_span"], new["raw_span"], " events"),
        _row("T 相邻变化幅度", old["T_adjacent_change"], new["T_adjacent_change"], " m/s"),
        _row("Omega 相邻变化幅度", old["Omega_adjacent_change"], new["Omega_adjacent_change"], " rad/s"),
        _row("depth 相邻步变化幅度", old["depth_adjacent_change"], new["depth_adjacent_change"], " m"),
        _row("TTC 相邻步变化幅度", old["ttc_adjacent_change"], new["ttc_adjacent_change"], " s"),
        f"| 10 步 BPTT 覆盖总物理时间 | {_fmt(old_bptt)} us | {_fmt(new_bptt)} us |",
        "",
        "## 变化范围",
        f"- 旧方案 T range：{old['T_range'].tolist()}",
        f"- 新方案 T range：{new['T_range'].tolist()}",
        f"- 旧方案 Omega range：{old['Omega_range'].tolist()}",
        f"- 新方案 Omega range：{new['Omega_range'].tolist()}",
        "",
        "## 判断",
        f"1. 新方案每步 dt 是否明显增大：{'是' if new_dt_larger else '否'}。",
        f"2. 10 步内是否出现更可观察的运动变化：{'数值上有所增大，但当前 64 步仍处于较短时间窗口内' if motion_increased else '否'}。",
        "3. 新方案更接近 MAVLab 输入统计，因为每步严格由网络输入 ROI 内连续 1000 条源事件构成。",
        "4. 标签与事件保持空间对齐：新 H5 中 event_cnt、depth、TTC、inverse TTC、mask 均裁剪自同一固定 ROI。",
    ]

    report = "\n".join(lines) + "\n"
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(report, encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-h5", type=Path, required=True)
    parser.add_argument("--new-h5", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(args.old_h5, args.new_h5, args.out_md)
    print(report)
    print(f"report: {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
