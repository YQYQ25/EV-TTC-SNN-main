#!/usr/bin/env python3
"""审计官方 EV-TTC Skatepark 筛选、TTC 正负号和 mask 逻辑。

本脚本只读取官方 create_exp.py / calc_gt.py 已生成的 H5，不改写官方数据。
统计口径严格复用 TTCEF/merge.py 中的样本级阈值和像素级 mask 条件。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import shutil
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import h5py
import hdf5plugin  # noqa: F401  # 读取官方 Blosc2 压缩 H5 需要注册插件。
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[3]
OFFICIAL_ROOT = PROJECT_ROOT / "EV-TTC-main"
DEFAULT_DATA_ROOT = Path("/data/evttc_storage/event+SNN+TTC/data_m3ed_official_7train_3test")
DEFAULT_ROI_DIR = PROJECT_ROOT / "EV-TTC-SNN-main/debug_sets/探索1k，5k...20k输入的数据质量如何"
DEFAULT_OUT_DIR = PROJECT_ROOT / "EV-TTC-SNN-main/analysis/官方EVTTC_Skatepark筛选与负TTC统计"
SEQ_NAME = "spot_outdoor_day_skatepark_1"
PIXELS_360 = 360 * 360
ROI_PIXELS = 128 * 128


def pct(num: float, den: float) -> float:
    """百分比工具函数，分母为 0 时返回 0。"""

    return 100.0 * float(num) / float(den) if den else 0.0


def fmt(value: Any, digits: int = 4) -> str:
    """报告中的数字格式化。"""

    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(v):
        return "NA"
    if abs(v) >= 1000:
        return f"{v:.1f}"
    if abs(v) > 0 and abs(v) < 1e-3:
        return f"{v:.3e}"
    return f"{v:.{digits}f}"


def to_jsonable(value: Any) -> Any:
    """把 numpy 标量递归转成 JSON 可写类型。"""

    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return value


def stat_summary(values: np.ndarray) -> dict[str, Any]:
    """返回 mean 和常用分位数。"""

    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"count": 0, "mean": None, "p5": None, "p25": None, "p50": None, "p75": None, "p95": None}
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "p5": float(np.percentile(arr, 5)),
        "p25": float(np.percentile(arr, 25)),
        "p50": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "p95": float(np.percentile(arr, 95)),
    }


def find_official_paths(data_root: Path, seq_name: str) -> dict[str, Path]:
    """定位官方流程所需的 data/depth/pose/exp/ttcef 文件。"""

    paths = {
        "data": data_root / f"events/m3ed/{seq_name}/{seq_name}_data.h5",
        "depth": data_root / f"depth/m3ed/{seq_name}/{seq_name}_depth_gt.h5",
        "pose": data_root / f"depth/m3ed/{seq_name}/{seq_name}_pose_gt.h5",
        "exp": data_root / f"exp_filts/m3ed/{seq_name}.h5",
        "gt": data_root / f"ttcef/m3ed/{seq_name}.h5",
    }
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError("缺少官方数据文件:\n" + "\n".join(missing))
    return paths


def official_pixel_masks(raw_mask: np.ndarray, exp_last: np.ndarray, ttc_raw: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """复刻 TTCEF/merge.py 的像素 mask，并额外构造正 TTC mask。

    官方逻辑：
    mask_data & abs(exp_last)>1e-3 & ~isnan(ttc)，随后 nan_to_num(ttc) < 100。
    注意这里没有 TTC>0，因此负 TTC 会被保留。
    """

    pre_mask = raw_mask.astype(bool) & (np.abs(exp_last) > 1e-3)
    ttc_clean = np.nan_to_num(ttc_raw)
    official_mask = pre_mask & (~np.isnan(ttc_raw)) & (ttc_clean < 100)
    with np.errstate(divide="ignore", invalid="ignore"):
        inv_ttc = np.divide(1.0, ttc_raw, out=np.full_like(ttc_raw, np.nan, dtype=np.float32), where=(ttc_raw != 0))
    positive_ttc_mask = official_mask & (ttc_raw > 0) & (inv_ttc > 0)
    return pre_mask, official_mask, positive_ttc_mask


def pixel_row_from_arrays(i: int, ttc_raw: np.ndarray, raw_mask: np.ndarray, exp_last: np.ndarray, official_valid: bool, tz: float) -> dict[str, Any]:
    """计算单帧 TTC 正负号、空 mask 和 inverse TTC 统计。"""

    pre_mask, official_mask, positive_ttc_mask = official_pixel_masks(raw_mask, exp_last, ttc_raw)
    official_pixels = int(np.count_nonzero(official_mask))
    positive_pixels = int(np.count_nonzero(positive_ttc_mask))
    ttc_in_mask = ttc_raw[official_mask]
    with np.errstate(divide="ignore", invalid="ignore"):
        inv_in_mask = 1.0 / ttc_in_mask

    pos_count = int(np.count_nonzero(ttc_in_mask > 0))
    neg_count = int(np.count_nonzero(ttc_in_mask < 0))
    zero_count = int(np.count_nonzero(ttc_in_mask == 0))
    inf_count = int(np.count_nonzero(np.isinf(ttc_in_mask)))
    nan_pre = int(np.count_nonzero(pre_mask & np.isnan(ttc_raw)))
    inf_pre = int(np.count_nonzero(pre_mask & np.isinf(ttc_raw)))
    inv_pos = int(np.count_nonzero(inv_in_mask > 0))
    inv_neg = int(np.count_nonzero(inv_in_mask < 0))
    inv_zero = int(np.count_nonzero(inv_in_mask == 0))

    if official_pixels == 0:
        sign_class = "official_mask_empty"
    elif pos_count == official_pixels:
        sign_class = "all_positive"
    elif neg_count == official_pixels:
        sign_class = "all_negative"
    elif pos_count > 0 and neg_count > 0:
        sign_class = "mixed"
    else:
        sign_class = "zero_or_inf_only"

    return {
        "sample_index": int(i),
        "official_sample_valid": bool(official_valid),
        "T_z": float(tz),
        "pre_mask_pixels": int(np.count_nonzero(pre_mask)),
        "official_mask_pixels": official_pixels,
        "positive_ttc_mask_pixels": positive_pixels,
        "official_mask_ratio": float(official_pixels / PIXELS_360),
        "positive_ttc_mask_ratio": float(positive_pixels / PIXELS_360),
        "ttc_positive_pixels": pos_count,
        "ttc_negative_pixels": neg_count,
        "ttc_zero_pixels": zero_count,
        "ttc_nan_pixels_before_nan_filter": nan_pre,
        "ttc_inf_pixels_before_final_filter": inf_pre,
        "ttc_inf_pixels_in_official_mask": inf_count,
        "inverse_ttc_positive_pixels": inv_pos,
        "inverse_ttc_negative_pixels": inv_neg,
        "inverse_ttc_zero_pixels": inv_zero,
        "sign_class": sign_class,
        "official_mask_empty": official_pixels == 0,
        "positive_ttc_mask_empty": positive_pixels == 0,
        "has_negative_ttc": neg_count > 0,
        "has_positive_ttc": pos_count > 0,
    }


def aggregate_pixel_rows(rows: list[dict[str, Any]], selector: np.ndarray) -> dict[str, Any]:
    """按样本集合聚合像素级和样本级正负 TTC 统计。"""

    selected = [r for r, keep in zip(rows, selector) if keep]
    total_official_pixels = sum(int(r["official_mask_pixels"]) for r in selected)
    total_positive_mask_pixels = sum(int(r["positive_ttc_mask_pixels"]) for r in selected)
    sign_counter = Counter(str(r["sign_class"]) for r in selected)
    negative_samples_any = sum(1 for r in selected if bool(r["has_negative_ttc"]))
    positive_samples_any = sum(1 for r in selected if bool(r["has_positive_ttc"]))
    positive_mask_nonempty = sum(1 for r in selected if not bool(r["positive_ttc_mask_empty"]))
    official_mask_nonempty = sum(1 for r in selected if not bool(r["official_mask_empty"]))
    return {
        "sample_count": len(selected),
        "official_mask_pixels": int(total_official_pixels),
        "positive_ttc_mask_pixels": int(total_positive_mask_pixels),
        "ttc_positive_pixels": int(sum(int(r["ttc_positive_pixels"]) for r in selected)),
        "ttc_negative_pixels": int(sum(int(r["ttc_negative_pixels"]) for r in selected)),
        "ttc_zero_pixels": int(sum(int(r["ttc_zero_pixels"]) for r in selected)),
        "ttc_nan_pixels_before_nan_filter": int(sum(int(r["ttc_nan_pixels_before_nan_filter"]) for r in selected)),
        "ttc_inf_pixels_before_final_filter": int(sum(int(r["ttc_inf_pixels_before_final_filter"]) for r in selected)),
        "ttc_inf_pixels_in_official_mask": int(sum(int(r["ttc_inf_pixels_in_official_mask"]) for r in selected)),
        "inverse_ttc_positive_pixels": int(sum(int(r["inverse_ttc_positive_pixels"]) for r in selected)),
        "inverse_ttc_negative_pixels": int(sum(int(r["inverse_ttc_negative_pixels"]) for r in selected)),
        "inverse_ttc_zero_pixels": int(sum(int(r["inverse_ttc_zero_pixels"]) for r in selected)),
        "official_mask_nonempty_samples": int(official_mask_nonempty),
        "official_mask_empty_samples": int(len(selected) - official_mask_nonempty),
        "positive_ttc_mask_nonempty_samples": int(positive_mask_nonempty),
        "positive_ttc_mask_empty_samples": int(len(selected) - positive_mask_nonempty),
        "negative_ttc_any_samples": int(negative_samples_any),
        "positive_ttc_any_samples": int(positive_samples_any),
        "all_positive_samples": int(sign_counter["all_positive"]),
        "all_negative_samples": int(sign_counter["all_negative"]),
        "mixed_samples": int(sign_counter["mixed"]),
        "zero_or_inf_only_samples": int(sign_counter["zero_or_inf_only"]),
        "negative_ttc_pixel_ratio_percent": pct(sum(int(r["ttc_negative_pixels"]) for r in selected), total_official_pixels),
        "positive_ttc_pixel_ratio_percent": pct(sum(int(r["ttc_positive_pixels"]) for r in selected), total_official_pixels),
        "positive_ttc_remaining_sample_ratio_percent": pct(positive_mask_nonempty, len(selected)),
        "official_mask_empty_ratio_percent": pct(len(selected) - official_mask_nonempty, len(selected)),
        "positive_ttc_mask_empty_ratio_percent": pct(len(selected) - positive_mask_nonempty, len(selected)),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """写 CSV；字段顺序取第一行。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def read_roi_stats(roi_dir: Path) -> dict[str, dict[str, Any]]:
    """读取当前固定 ROI 多事件数 H5 的标量统计，用于和官方口径对比。"""

    stats: dict[str, dict[str, Any]] = {}
    for n in [5000, 10000, 15000, 20000]:
        matches = sorted(
            set((roi_dir / "H5").glob(f"*N{n}.h5"))
            | set(roi_dir.glob(f"*N{n}.h5"))
            | set(roi_dir.rglob(f"*N{n}.h5"))
        )
        matches = [p for p in matches if not p.name.startswith("audit_subset_")]
        if not matches:
            stats[f"{n//1000}k"] = {"path": None, "missing": True}
            continue
        path = matches[0]
        with h5py.File(path, "r") as f:
            total = int(f["T"].shape[0])
            T = f["T"][:].astype(np.float64)
            speed_valid = f["speed_valid"][:].astype(bool)
            omega_valid = f["omega_valid"][:].astype(bool)
            pose_valid = f["pose_valid"][:].astype(bool) if "pose_valid" in f else np.ones(total, dtype=bool)
            supervise_valid = f["supervise_valid"][:].astype(bool)
            valid_mask_ratio = f["valid_mask_ratio"][:].astype(np.float64)
            tz = T[:, 2]

        # 当前 ROI H5 只保存正 inverse TTC 监督，Tz<=0 的负 TTC 不作为标签保留。
        label_empty = valid_mask_ratio <= 0
        stats[f"{n//1000}k"] = {
            "path": str(path),
            "missing": False,
            "candidate_count": total,
            "speed_pass_rate_percent": pct(int(speed_valid.sum()), total),
            "omega_pass_rate_percent": pct(int(omega_valid.sum()), total),
            "official_sample_keep_rate_percent": pct(int(supervise_valid.sum()), total),
            "Tz_positive_rate_percent": pct(int(np.count_nonzero(tz > 0)), total),
            "negative_ttc_sample_rate_percent": pct(int(np.count_nonzero(pose_valid & (tz <= 0))), int(np.count_nonzero(pose_valid))),
            "official_mask_empty_rate_percent": pct(int(np.count_nonzero(label_empty)), total),
            "positive_ttc_mask_empty_rate_percent": pct(int(np.count_nonzero(label_empty)), total),
            "note": "ROI H5 中 negative TTC 未按标签保留；负 TTC 比例用 pose_valid 且 Tz<=0 近似表示。",
        }
    return stats


def clipped_image(data: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """用于可视化的分位数裁剪，避免极端 TTC 值拉坏色彩范围。"""

    arr = np.asarray(data, dtype=np.float32)
    vals = arr[mask] if mask is not None else arr[np.isfinite(arr)]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return arr
    lo, hi = np.percentile(vals, [1, 99])
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        return arr
    return np.clip(arr, lo, hi)


def save_visualization(
    out_path: Path,
    sample_index: int,
    exp_img: np.ndarray,
    depth: np.ndarray,
    ttc: np.ndarray,
    raw_mask: np.ndarray,
    row: dict[str, Any],
) -> None:
    """保存官方样本抽查图：事件表示、depth、TTC、inverse TTC 和两种 mask。"""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    _, official_mask, positive_ttc_mask = official_pixel_masks(raw_mask, exp_img[-1], ttc)
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = np.divide(1.0, ttc, out=np.full_like(ttc, np.nan, dtype=np.float32), where=(ttc != 0))

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    event_vmax = np.percentile(np.abs(exp_img[-1][np.isfinite(exp_img[-1])]), 99) if np.any(np.isfinite(exp_img[-1])) else 1.0
    panels = [
        ("event exp ch5", exp_img[-1], "coolwarm", -event_vmax, event_vmax, None),
        ("depth", depth, "viridis", None, None, raw_mask.astype(bool)),
        ("TTC", ttc, "coolwarm", None, None, official_mask),
        ("inverse TTC", inv, "coolwarm", None, None, official_mask),
        ("official mask", official_mask.astype(np.float32), "gray", 0, 1, None),
        ("positive TTC mask", positive_ttc_mask.astype(np.float32), "gray", 0, 1, None),
    ]
    for ax, (title, img, cmap, vmin, vmax, mask) in zip(axes.flat, panels):
        show = clipped_image(img, mask) if mask is not None else img
        im = ax.imshow(show, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    official_pixels = max(int(row["official_mask_pixels"]), 1)
    title = (
        f"sample {sample_index} | t={fmt(row['timestamp_us'], 1)} us | "
        f"||T||={fmt(row['T_norm'])} | Tz={fmt(row['T_z'])} | "
        f"||Omega||={fmt(row['Omega_norm'])} | speed={row['speed_valid']} | "
        f"omega={row['omega_valid']} | official={row['official_sample_valid']} | "
        f"TTC+={pct(row['ttc_positive_pixels'], official_pixels):.2f}% | "
        f"TTC-={pct(row['ttc_negative_pixels'], official_pixels):.2f}%"
    )
    fig.suptitle(title, fontsize=11)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    """生成简单 Markdown 表格。"""

    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(lines)


def build_reports(
    out_dir: Path,
    args: argparse.Namespace,
    paths: dict[str, Path],
    metrics: dict[str, Any],
    roi_stats: dict[str, dict[str, Any]],
    runtime: dict[str, Any],
) -> None:
    """生成指导文件要求的中文报告。"""

    reports = out_dir / "reports"
    reports.mkdir(parents=True, exist_ok=True)

    official = metrics["official"]
    groups = metrics["pixel_groups"]
    file_sizes = {k: paths[k].stat().st_size for k in paths}
    disk = runtime["disk_usage"]

    sequence_name = args.seq_name
    seq_type = "car" if "car" in sequence_name else "非 car"
    sample_filter = f"`valid & ||T|| > {args.speed_threshold:g} & ||Omega|| < {args.omega_threshold:g}`"
    flow_audit = f"""# EV-TTC 官方 {sequence_name} 生成流程审计

## 数据与脚本
- 序列：`{sequence_name}`
- 官方仓库：`{OFFICIAL_ROOT}`
- 数据根目录：`{args.data_root}`
- data：`{paths['data']}`，{file_sizes['data'] / 1024**3:.2f} GiB
- depth_gt：`{paths['depth']}`，{file_sizes['depth'] / 1024**3:.2f} GiB
- pose_gt：`{paths['pose']}`，{file_sizes['pose'] / 1024**3:.2f} GiB
- exp_filts：`{paths['exp']}`，{file_sizes['exp'] / 1024**3:.2f} GiB
- ttcef：`{paths['gt']}`，{file_sizes['gt'] / 1024**3:.2f} GiB

## 实际调用链
1. `EV-TTC-main/TTCEF/create_exp.py`
   - `convert_file(...)` 读取 `events/m3ed/{{seq}}/{{seq}}_data.h5`，生成 `exp_filts/m3ed/{{seq}}.h5`。
   - `calc_exp(...)` 在事件到来时更新 6 个指数滤波通道，事件极性使用 `sign = 1 if p else -1`，因此官方输入是 signed IIR。
   - 默认输出分辨率为 360x360；原 1280x720 的 x 方向裁剪 `[280, 999]` 后下采样。
2. `EV-TTC-main/TTCEF/calc_gt.py`
   - `run(args)` 读取 data / pose_gt / depth_gt / exp_filts。
   - 默认 `--dt=0.010` 秒，运动窗口是 `start_times = exp_time - 10ms` 到 `end_times = exp_time`。
   - 深度帧通过 `np.searchsorted(depth_ts, start_times)` 取靠近运动起始时刻的 depth。
   - 相对位姿：`CN_T_CNP1 = CN_T_C0 @ inv(CNP1_T_C0)`，`T = CN_T_CNP1[:3,-1] / dt`，`Omega = log(R) / dt`，即在起始相机坐标系下的相对运动速度。
   - TTC 公式：`ttc = depth_img / (Tz + 1e-5)`。
   - `mask = ~np.isnan(depth_img)`，只表示重投影后有深度。
3. `EV-TTC-main/TTCEF/merge.py`
   - 本序列按 {seq_type} 规则审计，样本级阈值：{sample_filter}。
   - 像素级 mask：`mask_data & abs(exp_chunk[:,-1]) > 1e-3 & ~np.isnan(ttc_chunk)`，然后 `ttc_chunk=np.nan_to_num(ttc_chunk)`，最后 `mask_chunk & (ttc_chunk < 100)`。
   - 源码中没有 `TTC > 0`、没有 `Tz > 0`、没有 `inverse_TTC > 0` 的过滤，因此官方 mask 会保留负 TTC。

## 源码片段证明

`EV-TTC-main/TTCEF/create_exp.py:111-132`
```python
if time_since_last > frame_interval:
    last_active = t_ev[i]
    exp_img = exp_img * filt_constants[:, -1][:, None, None]
    exp_times[cnt] = last_active
    save_imgs[cnt] = exp_img
sign = 1 if p_ev[i] else -1
exp_img[j, y_c, x_c] += filt_constants[j][act_ind] * sign * update[2]
```

`EV-TTC-main/TTCEF/calc_gt.py:147-180,190,306-325,359,436-440`
```python
valid_exp_mask = (... ((all_exp_times - dt*S_TO_MS) >= min_pose_time))
start_times = end_times - dt*S_TO_MS
CN_T_CNP1 = CN_T_C0[i] @ np.linalg.inv(CNP1_T_C0[i])
T = CN_T_CNP1[:, :3, -1]/dt
Omega = ... / dt
start_depth = np.searchsorted(depth_ts, start_times)
pos_depth_idx = depth_splat > 0
ttc = depth_img / (T[idx, 2] + 1e-5)
mask = ~np.isnan(depth_img)
parser.add_argument("--dt", ..., default=0.010)
```

`EV-TTC-main/TTCEF/merge.py:100-112`
```python
if "car" in seq_name:
    valid_mask = valid_chunk & (np.linalg.norm(T_chunk, axis=1) > 1.3) & (np.linalg.norm(Omega_chunk, axis=1) < 0.18)
else:
    valid_mask = valid_chunk & (np.linalg.norm(T_chunk, axis=1) > 0.25) & (np.linalg.norm(Omega_chunk, axis=1) < 0.18)
mask_chunk = mask_chunk[valid_mask] & (np.abs(exp_chunk[:, -1]) > 1e-3) & ~np.isnan(ttc_chunk)
ttc_chunk = np.nan_to_num(ttc_chunk)
mask_chunk = mask_chunk & (ttc_chunk < 100)
```

## 本次审计命令
```bash
{runtime['command']}
```

## 运行与磁盘
- 本脚本只读现有官方 H5，没有重新生成 exp_filts / TTC。
- 审计耗时：{runtime['elapsed_sec']:.2f} 秒
- 输出目录占用：{runtime['out_dir_size_mb']:.2f} MiB
- `/` 剩余：{disk.get('/', 'unknown')}
- `/data` 剩余：{disk.get('/data', 'unknown')}
"""
    flow_name = "Car官方EVTTC生成流程审计.md" if "car" in sequence_name else "EVTTC官方Skatepark生成流程审计.md"
    (reports / flow_name).write_text(flow_audit, encoding="utf-8")

    category_rows = []
    for name in ["A_speed_pass_omega_pass", "B_speed_fail_omega_pass", "C_speed_pass_omega_fail", "D_speed_fail_omega_fail", "boundary_invalid"]:
        count = official["category_counts"].get(name, 0)
        category_rows.append([name, count, f"{pct(count, official['candidate_count']):.2f}%"])

    stat_rows = []
    for group_name, group_stats in metrics["motion_stats"].items():
        for key in ["T_norm", "T_z", "Omega_norm"]:
            s = group_stats[key]
            stat_rows.append([
                group_name,
                key,
                s["count"],
                fmt(s["mean"]),
                fmt(s["p5"]),
                fmt(s["p25"]),
                fmt(s["p50"]),
                fmt(s["p75"]),
                fmt(s["p95"]),
            ])

    filter_report = f"""# EV-TTC 官方 {sequence_name} 筛选前后统计

## 样本级数量
- 官方候选样本总数：{official['candidate_count']}
- boundary 有效样本数：{official['boundary_valid_count']} / {official['candidate_count']}（{official['boundary_valid_rate_percent']:.2f}%）
- 速度通过样本数：{official['speed_valid_count']} / {official['boundary_valid_count']}（按 boundary 有效计 {official['speed_valid_rate_in_boundary_percent']:.2f}%）
- 角速度通过样本数：{official['omega_valid_count']} / {official['boundary_valid_count']}（按 boundary 有效计 {official['omega_valid_rate_in_boundary_percent']:.2f}%）
- 官方样本级最终保留：{official['official_valid_count']} / {official['candidate_count']}（总保留率 {official['official_keep_rate_percent']:.2f}%，按 boundary 有效计 {official['official_keep_rate_in_boundary_percent']:.2f}%）

## A/B/C/D 分类
{markdown_table(['类别', '数量', '占候选比例'], category_rows)}

## 运动量统计
{markdown_table(['集合', '指标', 'count', 'mean', 'P5', 'P25', 'P50', 'P75', 'P95'], stat_rows)}

## 结论
- 主要淘汰来源按数量看：{official['main_elimination_reason']}。
- 本序列为 {seq_type}，本次官方 merge 阈值为 `||T|| > {args.speed_threshold:g} m/s` 与 `||Omega|| < {args.omega_threshold:g} rad/s`。
"""
    filter_name = "Car官方EVTTC筛选前后统计.md" if "car" in sequence_name else "EVTTC官方Skatepark筛选前后统计.md"
    (reports / filter_name).write_text(filter_report, encoding="utf-8")

    neg_rows = []
    for group_name, group in groups.items():
        den = group["official_mask_pixels"]
        neg_rows.append([
            group_name,
            group["sample_count"],
            group["official_mask_pixels"],
            f"{pct(group['ttc_positive_pixels'], den):.2f}%",
            f"{pct(group['ttc_negative_pixels'], den):.2f}%",
            f"{pct(group['ttc_zero_pixels'], den):.2f}%",
            group["negative_ttc_any_samples"],
            f"{pct(group['negative_ttc_any_samples'], group['sample_count']):.2f}%",
            group["all_negative_samples"],
            group["mixed_samples"],
        ])

    neg_report = f"""# EV-TTC 官方 {sequence_name} 负 TTC 比例统计

## 像素级与样本级
{markdown_table(['集合', '样本数', 'official mask 像素', 'TTC>0 像素占比', 'TTC<0 像素占比', 'TTC=0 像素占比', '含负 TTC 样本数', '含负 TTC 样本比例', '全负样本数', '正负混合样本数'], neg_rows)}

## 说明
- 这里的像素比例只在官方实际 mask 内统计。
- `TTC = depth / (Tz + 1e-5)`，深度为正时，TTC 符号基本由全局 `Tz` 决定；本次若出现正负混合，通常来自数值极端或无穷值边界。
- 官方 merge 没有正 TTC 过滤，因此 `Tz<=0` 的远离/后退片段会以负 TTC 形式进入监督 mask。
"""
    neg_name = "Car官方EVTTC负TTC比例统计.md" if "car" in sequence_name else "EVTTC官方Skatepark负TTC比例统计.md"
    (reports / neg_name).write_text(neg_report, encoding="utf-8")

    mask_rows = []
    for group_name, group in groups.items():
        mask_rows.append([
            group_name,
            group["sample_count"],
            f"{pct(group['official_mask_pixels'], group['sample_count'] * PIXELS_360):.4f}%",
            f"{pct(group['positive_ttc_mask_pixels'], group['sample_count'] * PIXELS_360):.4f}%",
            f"{group['official_mask_empty_samples']} ({group['official_mask_empty_ratio_percent']:.2f}%)",
            f"{group['positive_ttc_mask_empty_samples']} ({group['positive_ttc_mask_empty_ratio_percent']:.2f}%)",
            f"{group['positive_ttc_mask_nonempty_samples']} ({group['positive_ttc_remaining_sample_ratio_percent']:.2f}%)",
        ])

    mask_report = f"""# 官方 Mask 与正 TTC Mask 对比

## 对比表
{markdown_table(['集合', '样本数', 'official mask 有效像素率', 'positive TTC mask 有效像素率', 'official mask 空样本', 'positive TTC mask 空样本', '只保留正 TTC 后剩余样本'], mask_rows)}

## 结论
- 官方 mask 保留负 TTC：是。
- 离线添加 `TTC > 0 AND inverse_TTC > 0` 后，负 TTC 样本会变成空标签或被剔除；这与当前 ROI 正 TTC 方案更接近，但不等同于官方论文训练口径。
"""
    (reports / "官方Mask与正TTCMask对比.md").write_text(mask_report, encoding="utf-8")

    if "car" in sequence_name:
        combined = "\n\n".join(
            [
                "# Car官方EVTTC筛选与负TTC统计",
                "## 生成流程摘要",
                flow_audit.split("## 源码片段证明")[0],
                "## 筛选前后统计",
                filter_report,
                "## 负TTC比例统计",
                neg_report,
                "## 官方Mask与正TTC Mask对比",
                mask_report,
            ]
        )
        (reports / "Car官方EVTTC筛选与负TTC统计.md").write_text(combined, encoding="utf-8")

    official_roi = {
        "candidate_count": official["candidate_count"],
        "speed_pass_rate_percent": official["speed_valid_rate_in_boundary_percent"],
        "omega_pass_rate_percent": official["omega_valid_rate_in_boundary_percent"],
        "official_sample_keep_rate_percent": official["official_keep_rate_percent"],
        "Tz_positive_rate_percent": official["Tz_positive_rate_percent"],
        "negative_ttc_sample_rate_percent": groups["official_valid"]["negative_ttc_any_sample_ratio_percent"],
        "official_mask_empty_rate_percent": groups["official_valid"]["official_mask_empty_ratio_percent"],
        "positive_ttc_mask_empty_rate_percent": groups["official_valid"]["positive_ttc_mask_empty_ratio_percent"],
    }
    comparison_rows = []
    columns = {"官方EV-TTC": official_roi, **roi_stats}
    for label, item in columns.items():
        if item.get("missing"):
            comparison_rows.append([label, "缺失", "缺失", "缺失", "缺失", "缺失", "缺失", "缺失", "缺失"])
            continue
        comparison_rows.append([
            label,
            item["candidate_count"],
            f"{item['speed_pass_rate_percent']:.2f}%",
            f"{item['omega_pass_rate_percent']:.2f}%",
            f"{item['official_sample_keep_rate_percent']:.2f}%",
            f"{item['Tz_positive_rate_percent']:.2f}%",
            f"{item['negative_ttc_sample_rate_percent']:.2f}%",
            f"{item['official_mask_empty_rate_percent']:.2f}%",
            f"{item['positive_ttc_mask_empty_rate_percent']:.2f}%",
        ])

    roi_report = f"""# 官方 EV-TTC 与当前 ROI 方案对比

{markdown_table(['方案', '候选样本数', '速度通过率', '角速度通过率', '样本保留率', 'Tz>0 比例', '负 TTC 样本比例', 'mask 空样本比例', '正 TTC mask 空样本比例'], comparison_rows)}

## 差异来源
- 时间窗口：官方使用固定 10ms；当前 ROI 方案使用固定事件数分片，窗口随事件密度变化。
- 空间范围：官方为 360x360 全图；当前 ROI 是 128x128 固定/切换 ROI。
- mask 定义：官方 mask 不过滤负 TTC；当前 ROI 方案保存的是正 inverse TTC 监督，Tz<=0 通常不会形成有效正 TTC 标签。
- 样本筛选：官方先按 {sample_filter} 做样本级筛选，随后保存全图 mask；ROI 方案还受 ROI 内事件量、重投影覆盖和正 TTC 监督密度影响。

## 判断
- 当前 ROI 方案与官方结果在运动阈值方向上可对齐，但在时间窗口、空间裁剪和负 TTC 保留策略上并不完全一致。
- 若目标是严格复现 EV-TTC 论文，应沿用官方负 TTC 策略；若目标是前向避障/碰撞风险，建议单独构造正 TTC / inverse TTC 监督，并在报告里说明这不是论文同口径。
"""
    roi_name = "Car官方EVTTC与当前ROI方案对比.md" if "car" in sequence_name else "官方EVTTC与当前ROI方案对比.md"
    (reports / roi_name).write_text(roi_report, encoding="utf-8")

    command_report = f"""# 实际运行命令

## 阶段 A 示例
```bash
EV-TTC-main/.venv/bin/python EV-TTC-SNN-main/snn_ttc/tools/audit_official_evttc_skatepark.py \\
  --max-samples 5000 \\
  --output-dir {out_dir / 'stage_A_subset5000'}
```

## 完整序列
```bash
{runtime['command']}
```
"""
    (reports / "实际运行命令.md").write_text(command_report, encoding="utf-8")

    runtime_report = f"""# 运行耗时和磁盘占用

- 开始时间：{runtime['start_time']}
- 结束时间：{runtime['end_time']}
- 耗时：{runtime['elapsed_sec']:.2f} 秒
- 输出目录：`{out_dir}`
- 输出目录占用：{runtime['out_dir_size_mb']:.2f} MiB
- `/` 剩余：{disk.get('/', 'unknown')}
- `/data` 剩余：{disk.get('/data', 'unknown')}
"""
    (reports / "运行耗时和磁盘占用.md").write_text(runtime_report, encoding="utf-8")

    source_report = f"""# 源码改动或 Git Diff

当前工作目录不是 git 仓库，无法生成 `git diff`。

本次新增只读审计脚本：
- `EV-TTC-SNN-main/snn_ttc/tools/audit_official_evttc_skatepark.py`
- `EV-TTC-SNN-main/snn_ttc/tools/run_official_evttc_skatepark_subset.py`
- `EV-TTC-SNN-main/snn_ttc/tools/compare_official_vs_roi_ttc.py`

未修改官方 EV-TTC 源码：
- `EV-TTC-main/TTCEF/create_exp.py`
- `EV-TTC-main/TTCEF/calc_gt.py`
- `EV-TTC-main/TTCEF/merge.py`
"""
    (reports / "源码改动或git_diff.md").write_text(source_report, encoding="utf-8")


def choose_visual_samples(candidate_rows: list[dict[str, Any]], limit: int, seed: int) -> dict[str, list[int]]:
    """按指导文件要求抽取各类可视化样本。"""

    rng = random.Random(seed)
    buckets: dict[str, list[int]] = {
        "official_pass_Tz_positive": [],
        "official_pass_Tz_nonpositive": [],
        "only_omega_failed": [],
        "only_speed_failed": [],
        "official_mask_nonempty_positive_mask_empty": [],
    }
    for row in candidate_rows:
        if row["official_sample_valid"] and row["T_z"] > 0:
            buckets["official_pass_Tz_positive"].append(int(row["sample_index"]))
        if row["official_sample_valid"] and row["T_z"] <= 0:
            buckets["official_pass_Tz_nonpositive"].append(int(row["sample_index"]))
        if row["boundary_valid"] and row["speed_valid"] and not row["omega_valid"]:
            buckets["only_omega_failed"].append(int(row["sample_index"]))
        if row["boundary_valid"] and (not row["speed_valid"]) and row["omega_valid"]:
            buckets["only_speed_failed"].append(int(row["sample_index"]))
        if row["official_mask_pixels"] > 0 and row["positive_ttc_mask_pixels"] == 0:
            buckets["official_mask_nonempty_positive_mask_empty"].append(int(row["sample_index"]))

    selected = {}
    for name, idxs in buckets.items():
        idxs = sorted(set(idxs))
        if len(idxs) > limit:
            selected[name] = sorted(rng.sample(idxs, limit))
        else:
            selected[name] = idxs
    return selected


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    """主审计流程。"""

    start_wall = time.time()
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_dir = out_dir / "csv"
    vis_dir = out_dir / "visualizations"
    paths = find_official_paths(args.data_root, args.seq_name)

    with h5py.File(paths["exp"], "r") as f_exp, h5py.File(paths["gt"], "r") as f_gt:
        n_exp = int(f_exp["exp_times"].shape[0])
        n_gt = int(f_gt["T"].shape[0])
        total = min(n_exp, n_gt)
        start = int(args.start_index)
        end = total if args.full_sequence or args.max_samples is None else min(total, start + int(args.max_samples))
        if start < 0 or start >= total or end <= start:
            raise ValueError(f"无效扫描范围: start={start}, end={end}, total={total}")
        indices = np.arange(start, end, dtype=np.int64)

        timestamps = f_exp["exp_times"][start:end].astype(np.float64)
        T = f_gt["T"][start:end].astype(np.float64)
        Omega = f_gt["Omega"][start:end].astype(np.float64)
        boundary_valid = f_gt["valid"][start:end].astype(bool)

        speed_norm = np.linalg.norm(T, axis=1)
        omega_norm = np.linalg.norm(Omega, axis=1)
        speed_valid = speed_norm > args.speed_threshold
        omega_valid = omega_norm < args.omega_threshold
        official_sample_valid = boundary_valid & speed_valid & omega_valid

        candidate_rows: list[dict[str, Any]] = []
        for local_i, sample_i in enumerate(indices):
            if not boundary_valid[local_i]:
                category = "boundary_invalid"
            elif speed_valid[local_i] and omega_valid[local_i]:
                category = "A_speed_pass_omega_pass"
            elif (not speed_valid[local_i]) and omega_valid[local_i]:
                category = "B_speed_fail_omega_pass"
            elif speed_valid[local_i] and (not omega_valid[local_i]):
                category = "C_speed_pass_omega_fail"
            else:
                category = "D_speed_fail_omega_fail"
            candidate_rows.append(
                {
                    "sample_index": int(sample_i),
                    "timestamp_us": float(timestamps[local_i]),
                    "T_x": float(T[local_i, 0]),
                    "T_y": float(T[local_i, 1]),
                    "T_z": float(T[local_i, 2]),
                    "T_norm": float(speed_norm[local_i]),
                    "Omega_x": float(Omega[local_i, 0]),
                    "Omega_y": float(Omega[local_i, 1]),
                    "Omega_z": float(Omega[local_i, 2]),
                    "Omega_norm": float(omega_norm[local_i]),
                    "boundary_valid": bool(boundary_valid[local_i]),
                    "speed_valid": bool(speed_valid[local_i]),
                    "omega_valid": bool(omega_valid[local_i]),
                    "official_sample_valid": bool(official_sample_valid[local_i]),
                    "category": category,
                }
            )

        pixel_rows: list[dict[str, Any]] = []
        chunk_size = int(args.chunk_size)
        for chunk_start in range(start, end, chunk_size):
            chunk_end = min(end, chunk_start + chunk_size)
            exp_last = f_exp["exp_filts"][chunk_start:chunk_end, -1, :, :]
            ttc_chunk = f_gt["ttc"][chunk_start:chunk_end]
            raw_mask_chunk = f_gt["mask"][chunk_start:chunk_end]
            local0 = chunk_start - start
            for j in range(chunk_end - chunk_start):
                row = pixel_row_from_arrays(
                    chunk_start + j,
                    ttc_chunk[j],
                    raw_mask_chunk[j],
                    exp_last[j],
                    bool(official_sample_valid[local0 + j]),
                    float(T[local0 + j, 2]),
                )
                pixel_rows.append(row)
                candidate_rows[local0 + j].update(
                    {
                        "pre_mask_pixels": row["pre_mask_pixels"],
                        "official_mask_pixels": row["official_mask_pixels"],
                        "positive_ttc_mask_pixels": row["positive_ttc_mask_pixels"],
                        "ttc_positive_pixels": row["ttc_positive_pixels"],
                        "ttc_negative_pixels": row["ttc_negative_pixels"],
                        "ttc_zero_pixels": row["ttc_zero_pixels"],
                        "official_mask_empty": row["official_mask_empty"],
                        "positive_ttc_mask_empty": row["positive_ttc_mask_empty"],
                        "has_negative_ttc": row["has_negative_ttc"],
                        "has_positive_ttc": row["has_positive_ttc"],
                        "sign_class": row["sign_class"],
                    }
                )

        csv_prefix = "Car官方" if "car" in args.seq_name else "官方Skatepark"
        write_csv(csv_dir / f"{csv_prefix}候选样本级统计.csv", candidate_rows)
        write_csv(csv_dir / f"{csv_prefix}_TTC像素统计.csv", pixel_rows)

        visual_samples = choose_visual_samples(candidate_rows, args.visual_per_category, args.seed)
        row_by_idx = {int(r["sample_index"]): r for r in candidate_rows}
        for category, idxs in visual_samples.items():
            for sample_i in idxs:
                exp_img = f_exp["exp_filts"][sample_i]
                depth = f_gt["depth"][sample_i]
                ttc = f_gt["ttc"][sample_i]
                raw_mask = f_gt["mask"][sample_i]
                save_visualization(
                    vis_dir / category / f"sample_{sample_i:06d}.png",
                    sample_i,
                    exp_img,
                    depth,
                    ttc,
                    raw_mask,
                    row_by_idx[sample_i],
                )

    selection = np.ones(len(pixel_rows), dtype=bool)
    official_selection = official_sample_valid.astype(bool)
    pixel_groups = {
        "all_candidates": aggregate_pixel_rows(pixel_rows, selection),
        "official_valid": aggregate_pixel_rows(pixel_rows, official_selection),
    }
    for group in pixel_groups.values():
        group["negative_ttc_any_sample_ratio_percent"] = pct(group["negative_ttc_any_samples"], group["sample_count"])

    category_counts = Counter(r["category"] for r in candidate_rows)
    boundary_count = int(boundary_valid.sum())
    speed_fail_only = int(category_counts["B_speed_fail_omega_pass"])
    omega_fail_only = int(category_counts["C_speed_pass_omega_fail"])
    both_fail = int(category_counts["D_speed_fail_omega_fail"])
    if speed_fail_only > omega_fail_only and speed_fail_only >= both_fail:
        reason = "速度条件淘汰更多"
    elif omega_fail_only > speed_fail_only and omega_fail_only >= both_fail:
        reason = "角速度条件淘汰更多"
    elif both_fail > 0:
        reason = "速度和角速度同时失败最多"
    else:
        reason = "淘汰样本很少"

    def group_motion_stats(sel: np.ndarray) -> dict[str, Any]:
        return {
            "T_norm": stat_summary(speed_norm[sel]),
            "T_z": stat_summary(T[:, 2][sel]),
            "Omega_norm": stat_summary(omega_norm[sel]),
        }

    motion_stats = {
        "全部候选": group_motion_stats(np.ones(len(indices), dtype=bool)),
        "boundary有效": group_motion_stats(boundary_valid),
        "官方保留": group_motion_stats(official_sample_valid),
    }

    official_metrics = {
        "candidate_count": int(len(indices)),
        "scan_start_index": int(start),
        "scan_end_index_exclusive": int(end),
        "full_sequence_total": int(total),
        "boundary_valid_count": boundary_count,
        "boundary_valid_rate_percent": pct(boundary_count, len(indices)),
        "speed_valid_count": int(np.count_nonzero(boundary_valid & speed_valid)),
        "omega_valid_count": int(np.count_nonzero(boundary_valid & omega_valid)),
        "speed_valid_rate_in_boundary_percent": pct(np.count_nonzero(boundary_valid & speed_valid), boundary_count),
        "omega_valid_rate_in_boundary_percent": pct(np.count_nonzero(boundary_valid & omega_valid), boundary_count),
        "official_valid_count": int(np.count_nonzero(official_sample_valid)),
        "official_keep_rate_percent": pct(np.count_nonzero(official_sample_valid), len(indices)),
        "official_keep_rate_in_boundary_percent": pct(np.count_nonzero(official_sample_valid), boundary_count),
        "Tz_positive_count": int(np.count_nonzero(boundary_valid & (T[:, 2] > 0))),
        "Tz_nonpositive_count": int(np.count_nonzero(boundary_valid & (T[:, 2] <= 0))),
        "Tz_positive_rate_percent": pct(np.count_nonzero(boundary_valid & (T[:, 2] > 0)), boundary_count),
        "category_counts": dict(category_counts),
        "main_elimination_reason": reason,
        "speed_threshold_mps": float(args.speed_threshold),
        "omega_threshold_radps": float(args.omega_threshold),
    }

    roi_stats = read_roi_stats(args.roi_dir)

    elapsed = time.time() - start_wall
    disk = {
        "/": shutil.disk_usage("/").free,
        "/data": shutil.disk_usage("/data").free if Path("/data").exists() else None,
    }
    disk_human = {k: (f"{v / 1024**3:.2f} GiB" if v is not None else "unknown") for k, v in disk.items()}
    out_dir_size = sum(p.stat().st_size for p in out_dir.rglob("*") if p.is_file())
    runtime = {
        "command": " ".join([sys.executable, *sys.argv]),
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_wall)),
        "end_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_sec": elapsed,
        "out_dir_size_mb": out_dir_size / 1024**2,
        "disk_usage": disk_human,
    }

    metrics = {
        "official": official_metrics,
        "pixel_groups": pixel_groups,
        "motion_stats": motion_stats,
        "roi_stats": roi_stats,
        "visual_samples": visual_samples,
        "paths": {k: str(v) for k, v in paths.items()},
        "runtime": runtime,
    }
    (out_dir / "audit_summary.json").write_text(json.dumps(to_jsonable(metrics), ensure_ascii=False, indent=2), encoding="utf-8")
    build_reports(out_dir, args, paths, metrics, roi_stats, runtime)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq-name", default=SEQ_NAME, help="只建议使用 spot_outdoor_day_skatepark_1")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT, help="官方 EV-TTC 数据根目录")
    parser.add_argument("--roi-dir", type=Path, default=DEFAULT_ROI_DIR, help="当前 ROI 多事件数结果目录")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR, help="报告输出目录")
    parser.add_argument("--max-samples", type=int, default=None, help="阶段 A 使用的最大连续样本数")
    parser.add_argument("--start-index", type=int, default=0, help="扫描起始样本索引")
    parser.add_argument("--full-sequence", action="store_true", help="扫描完整序列")
    parser.add_argument("--resume", action="store_true", help="保留兼容参数；本脚本重算统计但不会覆盖官方数据")
    parser.add_argument("--chunk-size", type=int, default=32, help="读取 TTC/exp 的 chunk 大小")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--visual-per-category", type=int, default=10)
    parser.add_argument("--speed-threshold", type=float, default=0.25)
    parser.add_argument("--omega-threshold", type=float, default=0.18)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_audit(args)
    print(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
