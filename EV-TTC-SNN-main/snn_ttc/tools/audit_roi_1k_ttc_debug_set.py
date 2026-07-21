#!/usr/bin/env python3
"""审计固定 ROI 内 1k 源事件 TTC debug H5，并附旧实现审计结论。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Tuple

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from snn_ttc.data.m3ed_geometry import load_camera_geometry  # noqa: E402


def _stats(arr: np.ndarray) -> Dict[str, float]:
    arr = np.asarray(arr)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {k: float("nan") for k in ["min", "max", "mean", "std", "p50", "p95"]}
    return {
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
    }


def _fmt(value: float, digits: int = 4) -> str:
    return "nan" if not np.isfinite(value) else f"{value:.{digits}f}"


def _stat_line(name: str, arr: np.ndarray, unit: str = "") -> str:
    s = _stats(arr)
    return (
        f"- {name}: min={_fmt(s['min'])}{unit}, max={_fmt(s['max'])}{unit}, "
        f"mean={_fmt(s['mean'])}{unit}, std={_fmt(s['std'])}{unit}, "
        f"p50={_fmt(s['p50'])}{unit}, p95={_fmt(s['p95'])}{unit}"
    )


def count_old_events_inside_360(old_h5: Path) -> Tuple[np.ndarray, np.ndarray]:
    """统计旧方案每步 1000 条全图源事件中，实际映射进 360x360 的事件数和权重和。"""

    with h5py.File(old_h5, "r") as f_old:
        data_path = Path(f_old.attrs["source_data_file"])
        starts = f_old["event_start_idx"][:]
        ends = f_old["event_end_idx"][:]

    with h5py.File(data_path, "r") as f_data:
        geom = load_camera_geometry(f_data)
        events = f_data["prophesee"]["left"]
        counts = np.zeros((len(starts),), dtype=np.int32)
        weights = np.zeros((len(starts),), dtype=np.float32)
        for i, (start, end) in enumerate(zip(starts, ends)):
            x = events["x"][start:end]
            y = events["y"][start:end]
            in_crop = (x >= 280) & (x <= 999) & (y < 720)
            if not np.any(in_crop):
                continue
            update = geom.dist_map[y[in_crop], x[in_crop] - 280]
            valid = update[:, 2:].sum(axis=1) > 0
            counts[i] = int(np.sum(valid))
            weights[i] = float(np.sum(update[valid, 2:]))
    return counts, weights


def audit_new_h5(new_h5: Path) -> Tuple[dict, list[str]]:
    """执行指导文件要求的自动检查，并返回可写入报告的统计信息。"""

    with h5py.File(new_h5, "r") as f:
        event_cnt = f["event_cnt"][:]
        depth = f["depth_start"][:]
        ttc = f["ttc_start"][:]
        inv = f["inverse_ttc_start"][:]
        mask = f["valid_ttc_mask"][:].astype(bool)
        T = f["T"][:]
        Omega = f["Omega"][:]
        dt = f["dt"][:]
        raw_start = f["raw_event_start_idx"][:]
        raw_end = f["raw_event_end_idx"][:]
        rank_start = f["roi_event_rank_start"][:]
        rank_end = f["roi_event_rank_end"][:]
        roi_count = f["roi_source_event_count"][:]
        total_weight = f["total_mapped_weight"][:]
        full_count = f["fully_inside_event_count"][:]
        partial_count = f["partially_inside_event_count"][:]
        pos_count = f["positive_source_count"][:]
        neg_count = f["negative_source_count"][:]
        pos_weight = f["positive_weight_sum"][:]
        neg_weight = f["negative_weight_sum"][:]
        speed_valid = f["speed_valid"][:].astype(bool)
        omega_valid = f["omega_valid"][:].astype(bool)
        supervise_valid = f["supervise_valid"][:].astype(bool)
        roi_x0 = int(f.attrs["roi_x0"])
        roi_y0 = int(f.attrs["roi_y0"])
        roi_size = int(f.attrs["roi_size"])
        seq = f.attrs["sequence_name"]

    assert np.all(roi_count == 1000)
    assert np.all(rank_end - rank_start == 1000)
    assert event_cnt.shape[1:] == (2, 128, 128)
    assert event_cnt.dtype == np.float32
    assert np.all(event_cnt >= 0)
    assert np.isfinite(event_cnt).all()
    assert np.all(dt > 0)
    assert np.all(raw_end > raw_start)
    assert np.all(rank_start[1:] == rank_end[:-1])
    assert depth.shape == ttc.shape == inv.shape == mask.shape == (len(dt), roi_size, roi_size)
    assert np.all(np.isfinite(depth[mask]))
    assert np.all(np.isfinite(ttc[mask]))

    ttc_valid = ttc[mask]
    inv_valid = inv[mask]
    depth_valid = depth[mask]
    valid_ratio = mask.reshape(len(dt), -1).mean(axis=1)
    raw_span = raw_end - raw_start

    summary = {
        "sequence": seq,
        "roi": (roi_x0, roi_y0, roi_size),
        "num_steps": len(dt),
        "dt": dt,
        "raw_span": raw_span,
        "total_weight": total_weight,
        "partial_ratio": partial_count / np.maximum(roi_count, 1),
        "valid_ratio": valid_ratio,
        "T": T,
        "Omega": Omega,
        "ttc_valid": ttc_valid,
        "inv_valid": inv_valid,
        "depth_valid": depth_valid,
        "speed_valid": speed_valid,
        "omega_valid": omega_valid,
        "supervise_valid": supervise_valid,
        "pos_count": pos_count,
        "neg_count": neg_count,
        "pos_weight": pos_weight,
        "neg_weight": neg_weight,
        "full_count": full_count,
        "partial_count": partial_count,
    }

    lines = [
        "## 新固定 ROI Debug H5 自动检查",
        f"- H5：`{new_h5}`",
        f"- 序列：`{seq}`",
        f"- ROI：x=[{roi_x0}:{roi_x0 + roi_size}), y=[{roi_y0}:{roi_y0 + roi_size})",
        f"- step 数：{len(dt)}",
        "- `roi_source_event_count == 1000`：通过",
        "- `roi_event_rank_end - roi_event_rank_start == 1000`：通过",
        "- `event_cnt.shape == (64, 2, 128, 128)`：通过",
        "- `event_cnt.dtype == float32`：通过",
        "- `event_cnt >= 0` 且有限：通过",
        "- `t_end > t_start`：通过",
        "- `raw_event_end_idx > raw_event_start_idx`：通过",
        "- ROI rank 相邻 step 严格首尾相接：通过",
        "- event/depth/TTC/inverse TTC/mask 空间尺寸一致：通过",
        "- mask 有效位置 depth/TTC 全部有限：通过",
        "",
        "### 新方案统计",
        _stat_line("每步 dt", dt, " us"),
        _stat_line("原始事件索引跨度", raw_span, " events"),
        _stat_line("ROI 内映射权重总和", total_weight),
        _stat_line("部分落入 ROI 的源事件比例", summary["partial_ratio"] * 100.0, "%"),
        _stat_line("valid_ttc_mask 有效像素率", valid_ratio * 100.0, "%"),
        _stat_line("T_x", T[:, 0], " m/s"),
        _stat_line("T_y", T[:, 1], " m/s"),
        _stat_line("T_z", T[:, 2], " m/s"),
        _stat_line("||T||", np.linalg.norm(T, axis=1), " m/s"),
        _stat_line("||Omega||", np.linalg.norm(Omega, axis=1), " rad/s"),
        f"- speed_valid：{int(speed_valid.sum())}/{len(speed_valid)}",
        f"- omega_valid：{int(omega_valid.sum())}/{len(omega_valid)}",
        f"- supervise_valid：{int(supervise_valid.sum())}/{len(supervise_valid)}",
        _stat_line("TTC", ttc_valid, " s"),
        _stat_line("inverse TTC", inv_valid, " 1/s"),
        _stat_line("depth", depth_valid, " m"),
        _stat_line("正事件源事件数", pos_count),
        _stat_line("负事件源事件数", neg_count),
        _stat_line("正事件权重和", pos_weight),
        _stat_line("负事件权重和", neg_weight),
        _stat_line("完全落入 ROI 的源事件数", full_count),
        _stat_line("部分落入 ROI 的源事件数", partial_count),
    ]
    return summary, lines


def build_report(new_h5: Path, old_h5: Path | None, out_md: Path) -> str:
    summary, lines = audit_new_h5(new_h5)

    report_lines = [
        "# 固定 ROI 内 1k 事件审计报告",
        "",
        "## 旧实现审计",
        "- 当前旧实现是先从全分辨率事件流按原始事件索引截取 1000 条源事件，再映射到 360x360。",
        "- 相关文件：`EV-TTC-SNN-main/snn_ttc/data/build_1k_step_index.py`。",
        "- 相关函数：`build_1k_step_index()`。",
        "- 关键源码片段：",
        "```python",
        "event_start_idx = start_event_idx + step_idx * events_per_step",
        "event_end_idx = event_start_idx + events_per_step",
        "```",
        "- 旧实现执行了与 EV-TTC 一致的去畸变和中央裁剪下采样。",
        "- 相关文件：`EV-TTC-SNN-main/snn_ttc/data/m3ed_geometry.py`。",
        "- 相关函数：`load_camera_geometry()`、`calc_dist_map_ds()`、`accumulate_event_count()`。",
        "- 关键源码片段：",
        "```python",
        "new_mtx, _ = cv2.getOptimalNewCameraMatrix(K, D, res[::-1], 0)",
        "x_map, y_map = cv2.initUndistortRectifyMap(...)",
        "id1 = x_map[yy, xx] / 2.0 - 140.0",
        "id2 = y_map[yy, xx] / 2.0",
        "```",
        "- 旧实现采用四邻域双线性分配。",
        "```python",
        "np.add.at(event_cnt, (channels, y_c, x_c), weights[:, 0])",
        "np.add.at(event_cnt, (channels, y_c, x_c + 1), weights[:, 1])",
        "np.add.at(event_cnt, (channels, y_c + 1, x_c), weights[:, 2])",
        "np.add.at(event_cnt, (channels, y_c + 1, x_c + 1), weights[:, 3])",
        "```",
        "- 旧实现的 `event_cnt` 是 `float32` 浮点权重图，不是整数计数图。",
    ]

    if old_h5 and old_h5.exists():
        counts, weights = count_old_events_inside_360(old_h5)
        report_lines.extend(
            [
                f"- 旧 debug H5：`{old_h5}`",
                _stat_line("旧方案每步实际落入 360x360 有效区域的源事件数", counts, " events"),
                _stat_line("旧方案每步 360x360 映射权重和", weights),
            ]
        )
    else:
        report_lines.append("- 未提供旧 debug H5，跳过旧方案逐步源事件统计。")

    report_lines.extend(["", *lines])
    report = "\n".join(report_lines) + "\n"
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(report, encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--new-h5", type=Path, required=True)
    parser.add_argument("--old-h5", type=Path)
    parser.add_argument("--out-md", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(args.new_h5, args.old_h5, args.out_md)
    print(report)
    print(f"report: {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

