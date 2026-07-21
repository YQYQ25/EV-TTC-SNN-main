#!/usr/bin/env python3
"""基于已有 car 固定事件数 H5 构建三种空间方案配对探索集。

本脚本不重新搜索事件分片边界，而是复用已有 H5 中的
`raw_event_start_idx/raw_event_end_idx/t_start/t_end/T/Omega`。对同一个
step 只改变空间处理方式，从而比较：

- S1：360x360 全图事件计数
- S2：360x360 下采样到 180x180 后中心裁剪 128x128
- S3：360x360 上的随机/九宫格 128x128 ROI

同时导出与官方 EV-TTC 360x360 signed IIR 的近时刻可视化，便于肉眼检查。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
import time
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Any

import h5py
import hdf5plugin  # noqa: F401
import matplotlib
import numba
import numpy as np
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402


ROOT = Path(__file__).resolve().parents[3]
SNN_ROOT = ROOT / "EV-TTC-SNN-main"
if str(SNN_ROOT) not in sys.path:
    sys.path.insert(0, str(SNN_ROOT))

from snn_ttc.data.m3ed_geometry import (  # noqa: E402
    CAR_SPEED_THRESHOLD,
    OMEGA_THRESHOLD,
    ROI_NEIGHBOR_DX,
    ROI_NEIGHBOR_DY,
    interpolate_poses,
    load_camera_geometry,
)
from snn_ttc.tools.build_skatepark_multi_n_ttc import compute_motion_from_pose  # noqa: E402
from snn_ttc.tools.make_1k_ttc_debug_set import resolve_sequence_paths  # noqa: E402


EVENT_COUNTS = [5000, 10000, 15000, 20000]
SCHEMES = {
    "S1_360": {"size": 360, "coverage": "360x360 全图"},
    "S2_180_center128": {"size": 128, "coverage": "360->180 后中心 128，约覆盖原 360 的 [52:308)"},
    "S3_random128": {"size": 128, "coverage": "原 360 上当前 block 的 128x128 ROI"},
}
S2_CROP0 = 26
S2_CROP1 = 154
S2_BOX_360 = (52, 52, 256)


def pct(num: float, den: float) -> float:
    return 100.0 * float(num) / float(den) if den else 0.0


def fmt(value: Any, digits: int = 3) -> str:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(v):
        return "NA"
    if abs(v) >= 1000:
        return f"{v:.1f}"
    if 0 < abs(v) < 1e-3:
        return f"{v:.3e}"
    return f"{v:.{digits}f}"


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(str(v) for v in row) + " |" for row in rows)
    return "\n".join(lines)


def to_jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return value


@numba.njit(cache=True)
def accumulate_full_360_numba(
    x_ev: np.ndarray,
    y_ev: np.ndarray,
    p_ev: np.ndarray,
    dist_map: np.ndarray,
) -> np.ndarray:
    """按 EV-TTC 去畸变/裁剪/下采样口径，把事件累积到 360x360 正负双通道。"""

    out = np.zeros((2, 360, 360), dtype=np.float32)
    for i in range(x_ev.shape[0]):
        x = int(x_ev[i])
        y = int(y_ev[i])
        if x < 280 or x > 999 or y < 0 or y >= 720:
            continue
        update = dist_map[y, x - 280]
        base_x = int(update[0])
        base_y = int(update[1])
        if base_x < 0 or base_x >= 359 or base_y < 0 or base_y >= 359:
            continue
        channel = 0 if p_ev[i] > 0 else 1
        w0 = update[2]
        w1 = update[3]
        w2 = update[4]
        w3 = update[5]
        if w0 > 0:
            out[channel, base_y, base_x] += w0
        if w1 > 0:
            out[channel, base_y, base_x + 1] += w1
        if w2 > 0:
            out[channel, base_y + 1, base_x] += w2
        if w3 > 0:
            out[channel, base_y + 1, base_x + 1] += w3
    return out


def pool2_sum_event(event_360: np.ndarray) -> np.ndarray:
    """事件图 2x2 sum pooling：事件计数应保持总量守恒。"""

    return event_360.reshape(2, 180, 2, 180, 2).sum(axis=(2, 4)).astype(np.float32)


def pool2_any(mask_360: np.ndarray) -> np.ndarray:
    """mask 2x2 内任一有效则有效。"""

    return mask_360.reshape(180, 2, 180, 2).any(axis=(1, 3))


def pool2_mean_valid(values_360: np.ndarray, mask_360: np.ndarray, fill: float) -> np.ndarray:
    """只在有效像素上做 2x2 均值，避免无效 0/NaN 污染标签。"""

    vals = np.where(mask_360, values_360, 0.0).reshape(180, 2, 180, 2)
    counts = mask_360.reshape(180, 2, 180, 2).sum(axis=(1, 3))
    sums = vals.sum(axis=(1, 3))
    out = np.full((180, 180), fill, dtype=np.float32)
    valid = counts > 0
    out[valid] = (sums[valid] / counts[valid]).astype(np.float32)
    return out


def center128_from_180(arr_180: np.ndarray) -> np.ndarray:
    return arr_180[..., S2_CROP0:S2_CROP1, S2_CROP0:S2_CROP1]


def crop128(arr_360: np.ndarray, x0: int, y0: int) -> np.ndarray:
    return arr_360[..., y0 : y0 + 128, x0 : x0 + 128]


def nearest_index(sorted_times: np.ndarray, target: int) -> int:
    pos = int(np.searchsorted(sorted_times, target, side="left"))
    candidates = [min(max(pos, 0), len(sorted_times) - 1)]
    if pos > 0:
        candidates.append(pos - 1)
    if pos + 1 < len(sorted_times):
        candidates.append(pos + 1)
    return min(candidates, key=lambda i: abs(int(sorted_times[i]) - int(target)))


def choose_complete_blocks(f: h5py.File, blocks_per_n: int, seed: int) -> list[int]:
    """优先覆盖时间、ROI、风险、正负 Tz 和角速度失败样本的完整 10 step block。"""

    rng = np.random.default_rng(seed)
    block_index = f["block_index"][:].astype(np.int64)
    step_in_block = f["step_in_block"][:].astype(np.int64)
    event_dt = f["event_dt"][:].astype(np.float64)
    tz = f["T"][:, 2].astype(np.float64)
    omega_valid = f["omega_valid"][:].astype(bool)
    min_ttc = f["min_ttc"][:].astype(np.float64) if "min_ttc" in f else np.full(len(block_index), np.nan)
    roi_x0 = f["roi_x0"][:].astype(np.int64)
    roi_y0 = f["roi_y0"][:].astype(np.int64)

    block_to_rows: dict[int, list[int]] = defaultdict(list)
    for i, b in enumerate(block_index):
        block_to_rows[int(b)].append(i)

    complete: list[tuple[int, np.ndarray]] = []
    for b, rows in block_to_rows.items():
        rows_arr = np.asarray(sorted(rows, key=lambda j: step_in_block[j]), dtype=np.int64)
        if len(rows_arr) == 10 and set(step_in_block[rows_arr].tolist()) == set(range(10)):
            complete.append((b, rows_arr))
    complete.sort(key=lambda item: int(f["t_start"][item[1][0]]))
    if len(complete) <= blocks_per_n:
        return [b for b, _ in complete]

    selected: set[int] = set()

    def add_blocks(candidates: list[int], limit: int) -> None:
        unique = sorted(set(candidates))
        if not unique:
            return
        if len(unique) > limit:
            pick = rng.choice(unique, size=limit, replace=False).tolist()
        else:
            pick = unique
        selected.update(int(x) for x in pick)

    # 时间均匀覆盖是主轴。
    even_ids = np.linspace(0, len(complete) - 1, max(1, int(blocks_per_n * 0.55))).round().astype(int)
    selected.update(int(complete[i][0]) for i in even_ids)

    # 再补足容易被均匀抽样遗漏的类别。
    categories: dict[str, list[int]] = defaultdict(list)
    q_dt = np.nanpercentile(event_dt, [20, 80])
    for b, rows in complete:
        if np.any(tz[rows] <= 0):
            categories["negative_tz"].append(b)
        if np.any(np.isfinite(min_ttc[rows]) & (min_ttc[rows] < 1.0)):
            categories["high_risk"].append(b)
        if np.any(~omega_valid[rows]):
            categories["omega_failed"].append(b)
        if np.nanmedian(event_dt[rows]) <= q_dt[0]:
            categories["short_dt"].append(b)
        if np.nanmedian(event_dt[rows]) >= q_dt[1]:
            categories["long_dt"].append(b)
        roi_key = f"roi_{int(roi_x0[rows[0]])}_{int(roi_y0[rows[0]])}"
        categories[roi_key].append(b)
    for name, blocks in categories.items():
        add_blocks(blocks, 25 if name.startswith("roi_") else 60)

    # 超过目标就保持已选集合的时间覆盖；不足则继续按时间均匀填充。
    if len(selected) > blocks_per_n:
        ordered = [b for b, _ in complete if b in selected]
        keep_idx = np.linspace(0, len(ordered) - 1, blocks_per_n).round().astype(int)
        selected = {ordered[i] for i in keep_idx}
    if len(selected) < blocks_per_n:
        for b, _ in complete:
            selected.add(b)
            if len(selected) >= blocks_per_n:
                break
    return [b for b, _ in complete if b in selected][:blocks_per_n]


def selected_rows_for_blocks(f: h5py.File, selected_blocks: list[int]) -> np.ndarray:
    block_index = f["block_index"][:].astype(np.int64)
    step_in_block = f["step_in_block"][:].astype(np.int64)
    selected_set = set(int(b) for b in selected_blocks)
    rows = np.asarray([i for i, b in enumerate(block_index) if int(b) in selected_set], dtype=np.int64)
    order = np.lexsort((step_in_block[rows], block_index[rows]))
    return rows[order]


def create_output_h5(path: Path, n_steps: int, n_events: int, sequence: str, compression: str | None) -> h5py.File:
    """创建固定大小的输出 H5，避免反复 resize 带来的额外开销。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    f = h5py.File(path, "w")
    f.attrs["sequence_name"] = sequence
    f.attrs["N"] = n_events
    f.attrs["num_steps"] = n_steps
    f.attrs["complete"] = False
    scalar_specs = {
        "N": np.int32,
        "source_row_index": np.int64,
        "step_index": np.int64,
        "block_index": np.int64,
        "step_in_block": np.int16,
        "raw_event_start_idx": np.int64,
        "raw_event_end_idx": np.int64,
        "raw_event_index_span": np.int64,
        "t_start": np.int64,
        "t_end": np.int64,
        "event_dt": np.int64,
        "roi_x0": np.int16,
        "roi_y0": np.int16,
        "reset_required": bool,
        "direct_supervision_valid_source": bool,
        "official_sample_index": np.int64,
        "official_exp_time": np.int64,
        "official_delta_t_end_us": np.int64,
        "valid_mask_ratio_source": np.float32,
    }
    for key, dtype in scalar_specs.items():
        f.create_dataset(key, shape=(n_steps,), dtype=dtype, chunks=(min(1024, max(1, n_steps)),))
    for key in ["T_event_window", "Omega_event_window", "T_fixed10ms", "Omega_fixed10ms"]:
        f.create_dataset(key, shape=(n_steps, 3), dtype=np.float32, chunks=(min(256, max(1, n_steps)), 3))

    full_chunks = (1, 2, 360, 360)
    map_chunks = (1, 360, 360)
    roi_event_chunks = (1, 2, 128, 128)
    roi_map_chunks = (1, 128, 128)
    f.create_dataset("event_cnt_360", shape=(n_steps, 2, 360, 360), chunks=full_chunks, dtype=np.float32, compression=compression)
    f.create_dataset("event_cnt_180_center128", shape=(n_steps, 2, 128, 128), chunks=roi_event_chunks, dtype=np.float32, compression=compression)
    f.create_dataset("event_cnt_random128", shape=(n_steps, 2, 128, 128), chunks=roi_event_chunks, dtype=np.float32, compression=compression)

    for name in [
        "depth_360",
        "inverse_ttc_360",
        "inverse_ttc_fixed10ms_360",
    ]:
        f.create_dataset(name, shape=(n_steps, 360, 360), chunks=map_chunks, dtype=np.float32, compression=compression)
    for name in [
        "depth_180_center128",
        "depth_random128",
        "inverse_ttc_180_center128",
        "inverse_ttc_random128",
        "inverse_ttc_fixed10ms_180_center128",
        "inverse_ttc_fixed10ms_random128",
    ]:
        f.create_dataset(name, shape=(n_steps, 128, 128), chunks=roi_map_chunks, dtype=np.float32, compression=compression)

    for name in [
        "mask_360",
        "positive_mask_360",
        "mask_fixed10ms_360",
        "positive_mask_fixed10ms_360",
    ]:
        f.create_dataset(name, shape=(n_steps, 360, 360), chunks=map_chunks, dtype=np.uint8, compression=compression)
    for name in [
        "mask_180_center128",
        "mask_random128",
        "positive_mask_180_center128",
        "positive_mask_random128",
        "mask_fixed10ms_180_center128",
        "mask_fixed10ms_random128",
        "positive_mask_fixed10ms_180_center128",
        "positive_mask_fixed10ms_random128",
    ]:
        f.create_dataset(name, shape=(n_steps, 128, 128), chunks=roi_map_chunks, dtype=np.uint8, compression=compression)
    return f


def collect_event_stats(event_cnt: np.ndarray) -> dict[str, float]:
    total_map = event_cnt.sum(axis=0)
    nonzero = total_map > 0
    values = total_map.reshape(-1)
    total = float(values.sum())
    pos = float(event_cnt[0].sum())
    neg = float(event_cnt[1].sum())
    return {
        "event_weight_sum": total,
        "nonzero_pixel_rate_percent": pct(np.count_nonzero(nonzero), total_map.size),
        "pos_event_ratio_percent": pct(pos, pos + neg),
        "neg_event_ratio_percent": pct(neg, pos + neg),
        "pixel_value_p50": float(np.percentile(values, 50)),
        "pixel_value_p95": float(np.percentile(values, 95)),
        "pixel_value_p99": float(np.percentile(values, 99)),
        "pixel_value_max": float(np.max(values)) if values.size else 0.0,
    }


def collect_label_stats(inv: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    valid = mask.astype(bool)
    vals = inv[valid]
    return {
        "mask_valid_pixel_rate_percent": pct(np.count_nonzero(valid), valid.size),
        "inv_positive_pixel_ratio_percent": pct(np.count_nonzero(vals > 0), vals.size),
        "inv_negative_pixel_ratio_percent": pct(np.count_nonzero(vals < 0), vals.size),
        "inv_near_zero_pixel_ratio_percent": pct(np.count_nonzero(np.abs(vals) <= 1e-6), vals.size),
        "high_risk_inv_gt_1_pixel_ratio_percent": pct(np.count_nonzero(vals > 1.0), vals.size),
        "inv_p50": float(np.percentile(vals, 50)) if vals.size else np.nan,
        "inv_p95": float(np.percentile(vals, 95)) if vals.size else np.nan,
        "inv_p99": float(np.percentile(vals, 99)) if vals.size else np.nan,
    }


def update_aggregate(agg: dict[str, list[float]], prefix: str, metrics: dict[str, float]) -> None:
    for key, value in metrics.items():
        agg[f"{prefix}.{key}"].append(float(value))


def aggregate_mean(agg: dict[str, list[float]]) -> dict[str, float]:
    return {key: float(np.nanmean(values)) if values else np.nan for key, values in agg.items()}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def find_nearest_depth_index(gt_ts: np.ndarray, t_start: int) -> int:
    idx = int(np.searchsorted(gt_ts, t_start, side="left"))
    idx = min(max(idx, 0), len(gt_ts) - 1)
    if idx > 0 and abs(int(gt_ts[idx - 1]) - t_start) < abs(int(gt_ts[idx]) - t_start):
        idx -= 1
    return idx


def clipped_image(arr: np.ndarray, mask: np.ndarray | None = None) -> tuple[np.ndarray, float | None, float | None]:
    vals = arr[mask] if mask is not None else arr[np.isfinite(arr)]
    vals = vals[np.isfinite(vals)]
    if vals.size < 10:
        return arr, None, None
    lo, hi = np.percentile(vals, [1, 99])
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        return arr, None, None
    return np.clip(arr, lo, hi), float(lo), float(hi)


def draw_panel(
    ax: plt.Axes,
    img: np.ndarray,
    title: str,
    cmap: str,
    boxes: list[tuple[int, int, int, str]] | None = None,
    mask: np.ndarray | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    show = img
    if vmin is None and vmax is None and cmap != "gray":
        show, vmin, vmax = clipped_image(img, mask)
    im = ax.imshow(show, cmap=cmap, vmin=vmin, vmax=vmax)
    for x0, y0, size, color in boxes or []:
        ax.add_patch(Rectangle((x0, y0), size, size, fill=False, edgecolor=color, linewidth=1.6))
    ax.set_title(title, fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)


def official_pixel_masks(raw_mask: np.ndarray, exp_ch: np.ndarray, ttc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ttc_clean = np.nan_to_num(ttc)
    official = raw_mask.astype(bool) & (np.abs(exp_ch) > 1e-3) & np.isfinite(ttc) & (ttc_clean < 100)
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = np.divide(1.0, ttc, out=np.full_like(ttc, np.nan, dtype=np.float32), where=(ttc != 0))
    positive = official & (ttc > 0) & np.isfinite(inv) & (inv > 0)
    return official, positive


def save_visualizations_for_n(
    h5_path: Path,
    out_dir: Path,
    official_exp_path: Path,
    official_ttcef_path: Path,
    max_groups: int,
) -> int:
    """每个 N 导出配对图：S1/S2/S3 与官方 EV-TTC 同屏展示。"""

    out_n = out_dir / f"N{h5_path.stem.split('N')[-1].split('_')[0]}"
    out_n.mkdir(parents=True, exist_ok=True)
    saved = 0
    with h5py.File(h5_path, "r") as f, h5py.File(official_exp_path, "r") as fe, h5py.File(official_ttcef_path, "r") as fg:
        n_steps = int(f.attrs["num_steps"])
        indices = np.linspace(0, n_steps - 1, min(max_groups, n_steps)).round().astype(int)
        for local_id, i in enumerate(indices):
            roi = (int(f["roi_x0"][i]), int(f["roi_y0"][i]), 128)
            boxes = [(S2_BOX_360[0], S2_BOX_360[1], S2_BOX_360[2], "cyan"), (roi[0], roi[1], roi[2], "yellow")]
            official_idx = int(f["official_sample_index"][i])
            exp = fe["exp_filts"][official_idx]
            official_ttc = fg["ttc"][official_idx]
            official_mask, _ = official_pixel_masks(fg["mask"][official_idx], exp[-1], official_ttc)
            event_vmax = np.percentile(np.abs(exp[-1][np.isfinite(exp[-1])]), 99) if np.any(np.isfinite(exp[-1])) else 1.0

            fig, axes = plt.subplots(3, 4, figsize=(18, 12), constrained_layout=True)
            draw_panel(axes[0, 0], f["event_cnt_360"][i].sum(axis=0), "S1 event sum 360", "magma", boxes=boxes)
            draw_panel(axes[0, 1], f["event_cnt_180_center128"][i].sum(axis=0), "S2 event sum 128", "magma")
            draw_panel(axes[0, 2], f["event_cnt_random128"][i].sum(axis=0), "S3 event sum 128", "magma")
            draw_panel(axes[0, 3], exp[-1], "Official signed IIR ch5", "coolwarm", boxes=boxes, vmin=-event_vmax, vmax=event_vmax)

            draw_panel(axes[1, 0], f["inverse_ttc_360"][i], "S1 signed inverse TTC", "coolwarm", boxes=boxes, mask=f["mask_360"][i].astype(bool))
            draw_panel(axes[1, 1], f["inverse_ttc_180_center128"][i], "S2 signed inverse TTC", "coolwarm", mask=f["mask_180_center128"][i].astype(bool))
            draw_panel(axes[1, 2], f["inverse_ttc_random128"][i], "S3 signed inverse TTC", "coolwarm", mask=f["mask_random128"][i].astype(bool))
            draw_panel(axes[1, 3], official_ttc, "Official TTC", "coolwarm", boxes=boxes, mask=official_mask)

            draw_panel(axes[2, 0], f["mask_360"][i].astype(np.float32), "S1 mask", "gray", boxes=boxes, vmin=0, vmax=1)
            draw_panel(axes[2, 1], f["mask_180_center128"][i].astype(np.float32), "S2 mask", "gray", vmin=0, vmax=1)
            draw_panel(axes[2, 2], f["mask_random128"][i].astype(np.float32), "S3 mask", "gray", vmin=0, vmax=1)
            draw_panel(axes[2, 3], official_mask.astype(np.float32), "Official mask", "gray", boxes=boxes, vmin=0, vmax=1)

            inv360 = f["inverse_ttc_360"][i]
            mask360 = f["mask_360"][i].astype(bool)
            vals = inv360[mask360]
            pos_ratio = pct(np.count_nonzero(vals > 0), vals.size)
            neg_ratio = pct(np.count_nonzero(vals < 0), vals.size)
            fig.suptitle(
                f"N={int(f.attrs['N'])} row={i} src_step={int(f['step_index'][i])} "
                f"t=[{int(f['t_start'][i])},{int(f['t_end'][i])}] dt={int(f['event_dt'][i])}us "
                f"Tz={float(f['T_event_window'][i,2]):.3f} |T|={np.linalg.norm(f['T_event_window'][i]):.3f} "
                f"|Omega|={np.linalg.norm(f['Omega_event_window'][i]):.3f} ROI=({roi[0]},{roi[1]}) "
                f"inv+={pos_ratio:.1f}% inv-={neg_ratio:.1f}% official_delta={int(f['official_delta_t_end_us'][i])}us",
                fontsize=10,
            )
            fig.savefig(out_n / f"{h5_path.stem}_pair_{local_id:03d}_row_{i:05d}.png", dpi=130)
            plt.close(fig)
            saved += 1
    return saved


@numba.njit(cache=True)
def scan_independent_fixed_n_chunk(
    x_ev: np.ndarray,
    y_ev: np.ndarray,
    t_ev: np.ndarray,
    raw_offset: int,
    dist_map: np.ndarray,
    event_counts: np.ndarray,
    counts: np.ndarray,
    start_raws: np.ndarray,
    start_ts: np.ndarray,
    step_indices: np.ndarray,
    next_resets: np.ndarray,
    timeout_counts: np.ndarray,
    max_step_us: int,
    max_records: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """一次chunk内同时更新 S1/S2 的多个固定N计数器。"""

    schemes = 2
    n_n = event_counts.shape[0]
    rec_scheme = np.empty(max_records, dtype=np.int8)
    rec_n = np.empty(max_records, dtype=np.int32)
    rec_step = np.empty(max_records, dtype=np.int64)
    rec_start_raw = np.empty(max_records, dtype=np.int64)
    rec_end_raw = np.empty(max_records, dtype=np.int64)
    rec_t_start = np.empty(max_records, dtype=np.int64)
    rec_t_end = np.empty(max_records, dtype=np.int64)
    rec_raw_span = np.empty(max_records, dtype=np.int64)
    rec_reset = np.empty(max_records, dtype=np.uint8)
    rec_count = 0

    for i in range(x_ev.shape[0]):
        x = int(x_ev[i])
        y = int(y_ev[i])
        if x < 280 or x > 999 or y < 0 or y >= 720:
            continue
        update = dist_map[y, x - 280]
        base_x = int(update[0])
        base_y = int(update[1])
        if base_x < 0 or base_x >= 359 or base_y < 0 or base_y >= 359:
            continue
        w0 = update[2]
        w1 = update[3]
        w2 = update[4]
        w3 = update[5]
        s1_valid = (w0 > 0.0) or (w1 > 0.0) or (w2 > 0.0) or (w3 > 0.0)
        if not s1_valid:
            continue

        # S2以360坐标中的中心[52:308)为源事件计数区域；一个源事件只计1次。
        s2_valid = False
        nx = base_x
        ny = base_y
        if w0 > 0.0 and nx >= 52 and nx < 308 and ny >= 52 and ny < 308:
            s2_valid = True
        nx = base_x + 1
        ny = base_y
        if w1 > 0.0 and nx >= 52 and nx < 308 and ny >= 52 and ny < 308:
            s2_valid = True
        nx = base_x
        ny = base_y + 1
        if w2 > 0.0 and nx >= 52 and nx < 308 and ny >= 52 and ny < 308:
            s2_valid = True
        nx = base_x + 1
        ny = base_y + 1
        if w3 > 0.0 and nx >= 52 and nx < 308 and ny >= 52 and ny < 308:
            s2_valid = True

        raw_idx = raw_offset + i
        t_now = int(t_ev[i])
        for scheme in range(schemes):
            if scheme == 1 and not s2_valid:
                continue
            for n_i in range(n_n):
                state_i = scheme * n_n + n_i
                if counts[state_i] == 0:
                    start_raws[state_i] = raw_idx
                    start_ts[state_i] = t_now
                elif t_now - start_ts[state_i] > max_step_us:
                    timeout_counts[state_i] += 1
                    counts[state_i] = 0
                    start_raws[state_i] = raw_idx
                    start_ts[state_i] = t_now
                    next_resets[state_i] = 1

                counts[state_i] += 1
                target_n = int(event_counts[n_i])
                if counts[state_i] == target_n:
                    if rec_count < max_records:
                        rec_scheme[rec_count] = scheme
                        rec_n[rec_count] = target_n
                        rec_step[rec_count] = step_indices[state_i]
                        rec_start_raw[rec_count] = start_raws[state_i]
                        rec_end_raw[rec_count] = raw_idx + 1
                        rec_t_start[rec_count] = start_ts[state_i]
                        rec_t_end[rec_count] = t_now
                        rec_raw_span[rec_count] = raw_idx + 1 - start_raws[state_i]
                        rec_reset[rec_count] = next_resets[state_i]
                        rec_count += 1
                    step_indices[state_i] += 1
                    counts[state_i] = 0
                    next_resets[state_i] = 0

    return (
        rec_scheme[:rec_count],
        rec_n[:rec_count],
        rec_step[:rec_count],
        rec_start_raw[:rec_count],
        rec_end_raw[:rec_count],
        rec_t_start[:rec_count],
        rec_t_end[:rec_count],
        rec_raw_span[:rec_count],
        rec_reset[:rec_count],
        rec_count,
    )


def append_1d_dataset(dset: h5py.Dataset, values: np.ndarray) -> None:
    old = dset.shape[0]
    dset.resize((old + len(values),))
    dset[old : old + len(values)] = values


def create_index_h5(path: Path, scheme: str, n_events: int, sequence: str, compression: str | None) -> h5py.File:
    """创建轻量索引H5，只保存分片边界，不物化事件图。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    f = h5py.File(path, "w")
    f.attrs["sequence_name"] = sequence
    f.attrs["experiment_mode"] = "independent_fixed_n_spatial_schemes"
    f.attrs["spatial_scheme"] = scheme
    f.attrs["N"] = int(n_events)
    f.attrs["event_count_definition"] = (
        "fixed_N_in_360_full_view" if scheme == "S1_360" else "fixed_N_in_center256_then_downsample_to_128"
    )
    f.attrs["window_source"] = "independent_scan"
    f.attrs["label_geometry"] = "not_materialized_in_index"
    f.attrs["label_version"] = "index_only"
    f.attrs["max_step_duration_us"] = 10_000
    f.attrs["complete"] = False
    specs = {
        "step_index": np.int64,
        "raw_event_start_idx": np.int64,
        "raw_event_end_idx": np.int64,
        "t_start": np.int64,
        "t_end": np.int64,
        "event_dt": np.int64,
        "source_event_count": np.int32,
        "raw_event_index_span": np.int64,
        "block_index": np.int64,
        "step_in_block": np.int16,
        "reset_required": bool,
        "timeout": bool,
    }
    for name, dtype in specs.items():
        f.create_dataset(name, shape=(0,), maxshape=(None,), chunks=(4096,), dtype=dtype, compression=compression)
    return f


def append_index_records(f: h5py.File, rows: dict[str, np.ndarray]) -> None:
    n = len(rows["step_index"])
    if n == 0:
        return
    append_1d_dataset(f["step_index"], rows["step_index"].astype(np.int64))
    append_1d_dataset(f["raw_event_start_idx"], rows["raw_event_start_idx"].astype(np.int64))
    append_1d_dataset(f["raw_event_end_idx"], rows["raw_event_end_idx"].astype(np.int64))
    append_1d_dataset(f["t_start"], rows["t_start"].astype(np.int64))
    append_1d_dataset(f["t_end"], rows["t_end"].astype(np.int64))
    append_1d_dataset(f["event_dt"], (rows["t_end"] - rows["t_start"]).astype(np.int64))
    append_1d_dataset(f["source_event_count"], rows["source_event_count"].astype(np.int32))
    append_1d_dataset(f["raw_event_index_span"], rows["raw_event_index_span"].astype(np.int64))
    append_1d_dataset(f["block_index"], (rows["step_index"] // 10).astype(np.int64))
    append_1d_dataset(f["step_in_block"], (rows["step_index"] % 10).astype(np.int16))
    append_1d_dataset(f["reset_required"], rows["reset_required"].astype(bool))
    append_1d_dataset(f["timeout"], np.zeros(n, dtype=bool))


def index_h5_complete(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        with h5py.File(path, "r") as f:
            return bool(f.attrs.get("complete", False))
    except OSError:
        return False


def mark_paired_window_h5s(out_dir: Path) -> list[dict[str, Any]]:
    """给旧paired-window H5补充口径attrs，避免后续误解为独立fixed-N。"""

    rows: list[dict[str, Any]] = []
    for h5_path in sorted(out_dir.glob("N*_spatial_compare.h5")):
        with h5py.File(h5_path, "a") as f:
            f.attrs["experiment_mode"] = "paired_window_spatial_ablation"
            f.attrs["window_source"] = "S3_random128"
            f.attrs["event_count_definition_S1"] = "variable_events_in_S3_window"
            f.attrs["event_count_definition_S2"] = "variable_events_in_S3_window"
            f.attrs["event_count_definition_S3"] = "fixed_N_in_random128"
            f.attrs["label_version"] = "old_or_v2"
            f.attrs["invalid_fixed_n_claims"] = (
                "S1不是360全图固定N; S2不是中央大视野固定N; S1/S2/S3事件数不一致"
            )
            rows.append({"path": str(h5_path), "N": int(f.attrs.get("N", -1)), "num_steps": int(f.attrs.get("num_steps", 0))})
    return rows


def min_splat_depth(x: np.ndarray, y: np.ndarray, depth: np.ndarray, height: int, width: int, splat: int = 3) -> np.ndarray:
    """用min聚合执行简化z-buffer splat，口径与EV-TTC深度渲染保持同向。"""

    depth_img = np.full((height, width), np.inf, dtype=np.float32)
    xi = np.floor(x).astype(np.int32)
    yi = np.floor(y).astype(np.int32)
    valid = (xi >= 0) & (xi < width) & (yi >= 0) & (yi < height) & np.isfinite(depth) & (depth > 0)
    xi = xi[valid]
    yi = yi[valid]
    d = depth[valid].astype(np.float32)
    if d.size == 0:
        depth_img[:] = np.nan
        return depth_img
    for dy in range(-splat, splat + 1):
        yy = yi + dy
        y_ok = (yy >= 0) & (yy < height)
        if not np.any(y_ok):
            continue
        for dx in range(-splat, splat + 1):
            xx = xi + dx
            ok = y_ok & (xx >= 0) & (xx < width)
            if np.any(ok):
                np.minimum.at(depth_img, (yy[ok], xx[ok]), d[ok])
    depth_img[~np.isfinite(depth_img)] = np.nan
    return depth_img


def project_depth_to_spatial_scheme(
    depth_raw: np.ndarray,
    depth_pose: np.ndarray,
    start_pose: np.ndarray,
    geom: Any,
    scheme: str,
    splat: int = 3,
) -> np.ndarray:
    """把原始depth_gt按目标空间方案直接投影，S2不再从S1标签池化。"""

    cn_t_dn = start_pose @ np.linalg.inv(depth_pose)
    flat_depth = depth_raw.reshape(-1)
    valid_pts = np.isfinite(flat_depth) & (flat_depth > 0)
    height, width = (360, 360) if scheme == "S1_360" else (128, 128)
    if not np.any(valid_pts):
        return np.full((height, width), np.nan, dtype=np.float32)

    valid_depth = flat_depth[valid_pts].astype(np.float32)
    depth_cord = geom.proj_pts[:, valid_pts] * valid_depth[None, :]
    depth_cord_aug = np.vstack([depth_cord, np.ones_like(depth_cord[0])])
    event_cord = cn_t_dn @ depth_cord_aug
    in_front = event_cord[2] > 0
    if not np.any(in_front):
        return np.full((height, width), np.nan, dtype=np.float32)

    event_cord = event_cord[:, in_front]
    valid_depth = valid_depth[in_front]
    img_cord = geom.K @ event_cord[:3]
    img_cord /= img_cord[2]
    crop = (img_cord[0] > 280) & (img_cord[0] < 1000) & (img_cord[1] > 0) & (img_cord[1] < 720)
    if not np.any(crop):
        return np.full((height, width), np.nan, dtype=np.float32)

    if scheme == "S1_360":
        x = (img_cord[0, crop] - 280.0) / 2.0
        y = img_cord[1, crop] / 2.0
    else:
        # 等效于先得到360坐标，再360->180并中心裁剪[26:154)。
        x = (img_cord[0, crop] - 280.0) / 4.0 - S2_CROP0
        y = img_cord[1, crop] / 4.0 - S2_CROP0
    return min_splat_depth(x, y, valid_depth[crop], height=height, width=width, splat=splat)


class DepthFrameCache:
    """小型LRU缓存，避免相邻step反复解压同一depth帧。"""

    def __init__(self, dset: h5py.Dataset, max_items: int = 8):
        self.dset = dset
        self.max_items = max_items
        self.cache: OrderedDict[int, np.ndarray] = OrderedDict()

    def get(self, idx: int) -> np.ndarray:
        if idx in self.cache:
            value = self.cache.pop(idx)
            self.cache[idx] = value
            return value
        value = self.dset[idx].astype(np.float32)
        self.cache[idx] = value
        if len(self.cache) > self.max_items:
            self.cache.popitem(last=False)
        return value


def count_isolated_pixels(mask: np.ndarray) -> int:
    """统计8邻域内没有同类邻居的孤立有效像素。"""

    m = mask.astype(bool)
    if not np.any(m):
        return 0
    p = np.pad(m.astype(np.uint8), 1)
    neigh = (
        p[:-2, :-2]
        + p[:-2, 1:-1]
        + p[:-2, 2:]
        + p[1:-1, :-2]
        + p[1:-1, 1:-1]
        + p[1:-1, 2:]
        + p[2:, :-2]
        + p[2:, 1:-1]
        + p[2:, 2:]
    )
    return int(np.count_nonzero(m & (neigh <= 1)))


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    aa = a.astype(bool)
    bb = b.astype(bool)
    union = np.logical_or(aa, bb)
    return float(np.count_nonzero(aa & bb) / max(1, np.count_nonzero(union)))


def write_fixed_size_scheme_h5(
    path: Path,
    scheme: str,
    n_events: int,
    n_steps: int,
    sequence: str,
    compression: str | None,
) -> h5py.File:
    """创建N=10000独立fixed-N物化H5。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    size = 360 if scheme == "S1_360" else 128
    f = h5py.File(path, "w")
    f.attrs["sequence_name"] = sequence
    f.attrs["experiment_mode"] = "independent_fixed_n_spatial_schemes"
    f.attrs["spatial_scheme"] = scheme
    f.attrs["N"] = int(n_events)
    f.attrs["num_steps"] = int(n_steps)
    f.attrs["event_count_definition"] = (
        "fixed_N_in_360_full_view" if scheme == "S1_360" else "fixed_N_in_center256_then_downsample_to_128"
    )
    f.attrs["window_source"] = "independent_scan"
    f.attrs["label_geometry"] = "direct_raw_depth_projection_to_target_geometry"
    f.attrs["label_version"] = "signed_inverse_ttc_v2"
    f.attrs["complete"] = False
    str_dtype = h5py.string_dtype(encoding="utf-8")
    f.create_dataset("scheme", data=np.asarray([scheme] * n_steps, dtype=object), dtype=str_dtype)
    scalar_specs = {
        "N": np.int32,
        "step_index": np.int64,
        "block_index": np.int64,
        "step_in_block": np.int16,
        "raw_event_start_idx": np.int64,
        "raw_event_end_idx": np.int64,
        "t_start": np.int64,
        "t_end": np.int64,
        "event_dt": np.int64,
        "source_event_count": np.int32,
        "raw_event_index_span": np.int64,
        "pose_valid": bool,
        "speed_valid": bool,
        "omega_valid": bool,
        "supervise_valid": bool,
        "direct_supervision_valid": bool,
        "timeout": bool,
        "reset_required": bool,
        "depth_source_index": np.int64,
    }
    for key, dtype in scalar_specs.items():
        f.create_dataset(key, shape=(n_steps,), dtype=dtype, chunks=(min(1024, max(1, n_steps)),))
    for key in ["T", "Omega", "T_fixed10ms", "Omega_fixed10ms"]:
        f.create_dataset(key, shape=(n_steps, 3), dtype=np.float32, chunks=(min(256, max(1, n_steps)), 3))
    f.create_dataset("event_cnt", shape=(n_steps, 2, size, size), chunks=(1, 2, size, size), dtype=np.float32, compression=compression)
    for key in ["depth_start", "signed_inverse_ttc", "fixed10_signed_inverse_ttc"]:
        f.create_dataset(key, shape=(n_steps, size, size), chunks=(1, size, size), dtype=np.float32, compression=compression)
    for key in ["valid_ttc_mask", "positive_ttc_mask", "fixed10_valid_ttc_mask", "fixed10_positive_ttc_mask"]:
        f.create_dataset(key, shape=(n_steps, size, size), chunks=(1, size, size), dtype=np.uint8, compression=compression)
    return f


def scan_independent_indices(args: argparse.Namespace, paths: dict[str, Path], geom: Any) -> dict[str, Any]:
    """一次raw pass生成S1/S2在多个N下的独立固定事件数索引。"""

    index_dir = args.out_dir / "independent_fixed_n" / "indices"
    index_dir.mkdir(parents=True, exist_ok=True)
    event_counts = np.asarray(args.event_counts, dtype=np.int32)
    schemes = ["S1_360", "S2_center256_to128"]
    index_paths = {(scheme, int(n)): index_dir / f"{scheme}_N{int(n)}_index.h5" for scheme in schemes for n in event_counts}
    if args.resume and all(index_h5_complete(p) for p in index_paths.values()):
        return summarize_index_files(index_paths, reused=True)

    handles: dict[tuple[str, int], h5py.File] = {}
    for (scheme, n_events), path in index_paths.items():
        if path.exists():
            path.unlink()
        handles[(scheme, n_events)] = create_index_h5(path, scheme, n_events, args.sequence, args.compression)

    states = 2 * len(event_counts)
    counts = np.zeros(states, dtype=np.int32)
    start_raws = np.zeros(states, dtype=np.int64)
    start_ts = np.zeros(states, dtype=np.int64)
    step_indices = np.zeros(states, dtype=np.int64)
    next_resets = np.ones(states, dtype=np.uint8)
    timeout_counts = np.zeros(states, dtype=np.int64)
    started = time.time()

    with h5py.File(paths["data"], "r") as f_data:
        events = f_data["prophesee"]["left"]
        total_raw = int(events["t"].shape[0])
        if args.max_raw_events is not None:
            total_raw = min(total_raw, int(args.max_raw_events))
        pbar = tqdm(total=total_raw, desc="independent fixed-N index scan", unit="ev")
        cursor = 0
        min_n = int(np.min(event_counts))
        while cursor < total_raw:
            end = min(cursor + args.chunk_size, total_raw)
            x = events["x"][cursor:end]
            y = events["y"][cursor:end]
            t = events["t"][cursor:end]
            max_records = max(1024, int((end - cursor) / max(1, min_n) * len(event_counts) * 2 + 2048))
            result = scan_independent_fixed_n_chunk(
                x,
                y,
                t,
                cursor,
                geom.dist_map,
                event_counts,
                counts,
                start_raws,
                start_ts,
                step_indices,
                next_resets,
                timeout_counts,
                int(args.max_step_duration_ms * 1000),
                max_records,
            )
            rec_scheme, rec_n, rec_step, rec_start_raw, rec_end_raw, rec_t_start, rec_t_end, rec_raw_span, rec_reset, rec_count = result
            for scheme_id, scheme in enumerate(schemes):
                for n_events in event_counts:
                    keep = (rec_scheme[:rec_count] == scheme_id) & (rec_n[:rec_count] == int(n_events))
                    if not np.any(keep):
                        continue
                    rows = {
                        "step_index": rec_step[:rec_count][keep],
                        "raw_event_start_idx": rec_start_raw[:rec_count][keep],
                        "raw_event_end_idx": rec_end_raw[:rec_count][keep],
                        "t_start": rec_t_start[:rec_count][keep],
                        "t_end": rec_t_end[:rec_count][keep],
                        "source_event_count": np.full(np.count_nonzero(keep), int(n_events), dtype=np.int32),
                        "raw_event_index_span": rec_raw_span[:rec_count][keep],
                        "reset_required": rec_reset[:rec_count][keep].astype(bool),
                    }
                    append_index_records(handles[(scheme, int(n_events))], rows)
            pbar.update(end - cursor)
            cursor = end
            pbar.refresh()
        pbar.close()

    for scheme_id, scheme in enumerate(schemes):
        for n_i, n_events in enumerate(event_counts):
            state_i = scheme_id * len(event_counts) + n_i
            f = handles[(scheme, int(n_events))]
            f.attrs["timeout_count"] = int(timeout_counts[state_i])
            f.attrs["unfinished_event_count_at_eof"] = int(counts[state_i])
            f.attrs["num_steps"] = int(f["step_index"].shape[0])
            f.attrs["complete"] = True
            f.close()

    summary = summarize_index_files(index_paths, reused=False)
    summary["elapsed_seconds"] = time.time() - started
    (index_dir / "index_summary.json").write_text(json.dumps(to_jsonable(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def summarize_index_files(index_paths: dict[tuple[str, int], Path], reused: bool) -> dict[str, Any]:
    rows = []
    for (scheme, n_events), path in sorted(index_paths.items(), key=lambda item: (item[0][0], item[0][1])):
        with h5py.File(path, "r") as f:
            event_dt = f["event_dt"][:].astype(np.float64)
            steps = int(f["step_index"].shape[0])
            timeouts = int(f.attrs.get("timeout_count", 0))
            rows.append(
                {
                    "scheme": scheme,
                    "N": int(n_events),
                    "path": str(path),
                    "steps": steps,
                    "timeout_count": timeouts,
                    "timeout_ratio_percent": pct(timeouts, timeouts + steps),
                    "event_dt_p50_us": float(np.percentile(event_dt, 50)) if steps else np.nan,
                    "event_dt_p95_us": float(np.percentile(event_dt, 95)) if steps else np.nan,
                    "event_dt_mean_us": float(np.mean(event_dt)) if steps else np.nan,
                    "complete_10step_sequences": steps // 10,
                    "strict_source_event_count": bool(np.all(f["source_event_count"][:] == int(n_events))) if steps else True,
                    "h5_size_bytes": path.stat().st_size,
                    "reused": reused,
                }
            )
    return {"rows": rows, "reused": reused}


def fix_paired_window_s2_label_v2(args: argparse.Namespace, paths: dict[str, Path], geom: Any, pose_mats: np.ndarray, pose_ts: np.ndarray) -> dict[str, Any]:
    """重算旧N=10000 paired-window中S2的depth/inv/mask，并导出对比。"""

    old_h5 = args.out_dir / "N10000_spatial_compare.h5"
    out_dir = args.out_dir / "paired_window_v2"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_h5 = out_dir / "N10000_S2_label_v2.h5"
    if args.resume and out_h5.exists():
        with h5py.File(out_h5, "r") as f:
            if bool(f.attrs.get("complete", False)):
                summary = json.loads((out_dir / "N10000_S2_label_v2_summary.json").read_text(encoding="utf-8"))
                return summary

    tmp_h5 = out_h5.with_suffix(".tmp.h5")
    if tmp_h5.exists():
        tmp_h5.unlink()
    started = time.time()
    with h5py.File(old_h5, "r") as f_old:
        n_steps = int(f_old["t_start"].shape[0])
        t_start = f_old["t_start"][:].astype(np.int64)
        t_end = f_old["t_end"][:].astype(np.int64)
        t_fixed_end = t_start + 10_000
        T_fixed10, Omega_fixed10, fixed_pose_valid = compute_motion_from_pose(pose_mats, pose_ts, t_start, t_fixed_end)
        start_pose, start_pose_valid = interpolate_poses(pose_mats, pose_ts, t_start)
        with h5py.File(paths["depth"], "r") as f_depth, h5py.File(tmp_h5, "w") as f_out:
            f_out.attrs["sequence_name"] = args.sequence
            f_out.attrs["experiment_mode"] = "paired_window_spatial_ablation"
            f_out.attrs["window_source"] = "S3_random128"
            f_out.attrs["spatial_scheme"] = "S2_180_center128"
            f_out.attrs["event_count_definition"] = "variable_events_in_S3_window"
            f_out.attrs["label_geometry"] = "direct_raw_depth_projection_to_S2_target_geometry"
            f_out.attrs["label_version"] = "signed_inverse_ttc_v2"
            f_out.attrs["N"] = 10000
            f_out.attrs["num_steps"] = n_steps
            f_out.attrs["complete"] = False
            scalar_names = [
                "source_row_index",
                "step_index",
                "block_index",
                "step_in_block",
                "raw_event_start_idx",
                "raw_event_end_idx",
                "raw_event_index_span",
                "t_start",
                "t_end",
                "event_dt",
                "roi_x0",
                "roi_y0",
                "reset_required",
            ]
            for name in scalar_names:
                f_out.create_dataset(name, data=f_old[name][:])
            f_out.create_dataset("T_event_window", data=f_old["T_event_window"][:].astype(np.float32))
            f_out.create_dataset("Omega_event_window", data=f_old["Omega_event_window"][:].astype(np.float32))
            f_out.create_dataset("T_fixed10ms", data=T_fixed10.astype(np.float32))
            f_out.create_dataset("Omega_fixed10ms", data=Omega_fixed10.astype(np.float32))
            f_out.create_dataset("event_cnt_180_center128", data=f_old["event_cnt_180_center128"][:].astype(np.float32), chunks=(1, 2, 128, 128), compression=args.compression)

            for name in [
                "old_depth_180_center128",
                "depth_180_center128",
                "old_inverse_ttc_180_center128",
                "inverse_ttc_180_center128",
                "inverse_ttc_fixed10ms_180_center128",
            ]:
                f_out.create_dataset(name, shape=(n_steps, 128, 128), chunks=(1, 128, 128), dtype=np.float32, compression=args.compression)
            for name in [
                "old_mask_180_center128",
                "mask_180_center128",
                "old_positive_mask_180_center128",
                "positive_mask_180_center128",
                "mask_fixed10ms_180_center128",
                "positive_mask_fixed10ms_180_center128",
            ]:
                f_out.create_dataset(name, shape=(n_steps, 128, 128), chunks=(1, 128, 128), dtype=np.uint8, compression=args.compression)
            f_out["old_depth_180_center128"][:] = f_old["depth_180_center128"][:]
            f_out["old_inverse_ttc_180_center128"][:] = f_old["inverse_ttc_180_center128"][:]
            f_out["old_mask_180_center128"][:] = f_old["mask_180_center128"][:]
            f_out["old_positive_mask_180_center128"][:] = f_old["positive_mask_180_center128"][:]

            depth_ts = f_depth["ts"][:].astype(np.int64)
            depth_poses = f_depth["Cn_T_C0"][:].astype(np.float64)
            depth_dset = f_depth["depth"]["prophesee"]["left"]
            depth_cache = DepthFrameCache(depth_dset)
            agg = defaultdict(list)
            vis_candidates: list[dict[str, Any]] = []
            for i in tqdm(range(n_steps), desc="paired N10000 S2 label v2", unit="step"):
                old_inv = f_old["inverse_ttc_180_center128"][i].astype(np.float32)
                old_mask = f_old["mask_180_center128"][i].astype(bool)
                old_pos = f_old["positive_mask_180_center128"][i].astype(bool)
                event_active = f_old["event_cnt_180_center128"][i].sum(axis=0) > 0
                depth_idx = int(np.searchsorted(depth_ts, int(t_start[i]), side="left"))
                valid_depth_idx = 0 <= depth_idx < len(depth_ts) and bool(start_pose_valid[i])
                if valid_depth_idx:
                    depth_s2 = project_depth_to_spatial_scheme(depth_cache.get(depth_idx), depth_poses[depth_idx], start_pose[i], geom, "S2_180_center128")
                else:
                    depth_s2 = np.full((128, 128), np.nan, dtype=np.float32)
                depth_valid = np.isfinite(depth_s2) & (depth_s2 > 0)
                inv = np.zeros((128, 128), dtype=np.float32)
                inv[depth_valid] = (float(f_old["T_event_window"][i, 2]) / np.maximum(depth_s2[depth_valid], 1e-6)).astype(np.float32)
                mask = event_active & depth_valid & np.isfinite(inv)
                pos = mask & (inv > 0)
                inv_fixed = np.zeros((128, 128), dtype=np.float32)
                if fixed_pose_valid[i]:
                    inv_fixed[depth_valid] = (float(T_fixed10[i, 2]) / np.maximum(depth_s2[depth_valid], 1e-6)).astype(np.float32)
                mask_fixed = event_active & depth_valid & np.isfinite(inv_fixed)
                pos_fixed = mask_fixed & (inv_fixed > 0)

                f_out["depth_180_center128"][i] = depth_s2
                f_out["inverse_ttc_180_center128"][i] = inv
                f_out["mask_180_center128"][i] = mask.astype(np.uint8)
                f_out["positive_mask_180_center128"][i] = pos.astype(np.uint8)
                f_out["inverse_ttc_fixed10ms_180_center128"][i] = inv_fixed
                f_out["mask_fixed10ms_180_center128"][i] = mask_fixed.astype(np.uint8)
                f_out["positive_mask_fixed10ms_180_center128"][i] = pos_fixed.astype(np.uint8)

                union_mask = old_mask | mask
                mae = float(np.mean(np.abs(old_inv[union_mask] - inv[union_mask]))) if np.any(union_mask) else np.nan
                old_iso = count_isolated_pixels(old_mask)
                new_iso = count_isolated_pixels(mask)
                old_valid = int(np.count_nonzero(old_mask))
                new_valid = int(np.count_nonzero(mask))
                stats = {
                    "inv_mae": mae,
                    "mask_iou": mask_iou(old_mask, mask),
                    "old_mask_rate": pct(old_valid, old_mask.size),
                    "new_mask_rate": pct(new_valid, mask.size),
                    "old_isolated_ratio": pct(old_iso, old_valid),
                    "new_isolated_ratio": pct(new_iso, new_valid),
                    "old_high_risk_ratio": pct(np.count_nonzero(old_inv[old_mask] > 1.0), old_valid),
                    "new_high_risk_ratio": pct(np.count_nonzero(inv[mask] > 1.0), new_valid),
                }
                for key, value in stats.items():
                    agg[key].append(value)
                if len(vis_candidates) < 200:
                    vis_candidates.append({"row": i, **stats})
            f_out.attrs["complete"] = True

    tmp_h5.replace(out_h5)
    summary = {
        "out_h5": str(out_h5),
        "elapsed_seconds": time.time() - started,
        "num_steps": n_steps,
        "metrics": aggregate_mean(agg),
        "visualization_dir": str(out_dir / "visualizations" / "N10000_S2_label_v2_compare"),
    }
    (out_dir / "N10000_S2_label_v2_summary.json").write_text(json.dumps(to_jsonable(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    save_s2_label_v2_visualizations(old_h5, out_h5, out_dir / "visualizations" / "N10000_S2_label_v2_compare")
    return summary


def save_s2_label_v2_visualizations(old_h5: Path, new_h5: Path, out_dir: Path, max_images: int = 20) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    with h5py.File(old_h5, "r") as fo, h5py.File(new_h5, "r") as fn:
        n = int(fn["t_start"].shape[0])
        candidates = []
        for i in np.linspace(0, n - 1, min(200, n)).round().astype(int):
            old_mask = fo["mask_180_center128"][i].astype(bool)
            new_mask = fn["mask_180_center128"][i].astype(bool)
            score = count_isolated_pixels(old_mask) - count_isolated_pixels(new_mask)
            candidates.append((score, int(i)))
        indices = [i for _, i in sorted(candidates, reverse=True)[:max_images]]
        for local_id, i in enumerate(indices):
            old_inv = fo["inverse_ttc_180_center128"][i]
            new_inv = fn["inverse_ttc_180_center128"][i]
            old_mask = fo["mask_180_center128"][i].astype(bool)
            new_mask = fn["mask_180_center128"][i].astype(bool)
            diff = np.zeros_like(new_inv)
            union = old_mask | new_mask
            diff[union] = new_inv[union] - old_inv[union]
            fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
            draw_panel(axes[0, 0], fo["event_cnt_180_center128"][i].sum(axis=0), "S2 event", "magma")
            draw_panel(axes[0, 1], old_inv, "旧S2 inverse TTC", "coolwarm", mask=old_mask)
            draw_panel(axes[0, 2], new_inv, "新版S2 inverse TTC", "coolwarm", mask=new_mask)
            draw_panel(axes[1, 0], old_mask.astype(np.float32), "旧mask", "gray", vmin=0, vmax=1)
            draw_panel(axes[1, 1], new_mask.astype(np.float32), "新版mask", "gray", vmin=0, vmax=1)
            draw_panel(axes[1, 2], diff, "新版-旧版", "coolwarm", mask=union)
            fig.suptitle(
                f"row={i} t=[{int(fn['t_start'][i])},{int(fn['t_end'][i])}] "
                f"old_iso={count_isolated_pixels(old_mask)} new_iso={count_isolated_pixels(new_mask)} "
                f"IoU={mask_iou(old_mask, new_mask):.3f}",
                fontsize=10,
            )
            fig.savefig(out_dir / f"s2_label_v2_compare_{local_id:03d}_row_{i:05d}.png", dpi=140)
            plt.close(fig)
            saved += 1
    return saved


def find_s3_reference_motion_start_time(
    source_dir: Path | None,
    sequence: str,
    n_events: int,
) -> tuple[int | None, dict[str, Any]]:
    """从已有S3固定N数据中找一段车速/角速度都合格的参考时间。

    car序列前段可能处于低速或停车状态。S1/S2只物化一个小片段时，若直接从
    第一个有depth的位置开始，会得到几乎全是`supervise_valid=False`的样本。
    这里用已有S3文件定位真实运动片段，再让S1/S2从相近时间开始连续取样。
    """

    if source_dir is None:
        return None, {"selection_reference": "none"}
    s3_path = source_dir / "H5" / f"{sequence}_N{n_events}.h5"
    if not s3_path.exists():
        return None, {"selection_reference": "missing_s3", "s3_path": str(s3_path)}

    with h5py.File(s3_path, "r") as f:
        t_start = f["t_start"][:].astype(np.int64)
        speed = f["speed_valid"][:].astype(bool) if "speed_valid" in f else np.ones(len(t_start), dtype=bool)
        omega = f["omega_valid"][:].astype(bool) if "omega_valid" in f else np.ones(len(t_start), dtype=bool)
        pose = f["pose_valid"][:].astype(bool) if "pose_valid" in f else np.ones(len(t_start), dtype=bool)

    eligible = speed & omega & pose
    best_len = 0
    best_start = -1
    cur_len = 0
    cur_start = 0
    for i, ok in enumerate(eligible):
        if ok:
            if cur_len == 0:
                cur_start = i
            cur_len += 1
            if cur_len > best_len:
                best_len = cur_len
                best_start = cur_start
        else:
            cur_len = 0

    if best_start < 0:
        return None, {
            "selection_reference": "s3_no_valid_motion_run",
            "s3_path": str(s3_path),
            "s3_valid_motion_rate_percent": pct(np.count_nonzero(eligible), len(eligible)),
        }
    return int(t_start[best_start]), {
        "selection_reference": "s3_longest_speed_omega_valid_run",
        "s3_path": str(s3_path),
        "s3_reference_row": int(best_start),
        "s3_reference_t_start": int(t_start[best_start]),
        "s3_reference_run_length": int(best_len),
        "s3_valid_motion_rate_percent": pct(np.count_nonzero(eligible), len(eligible)),
    }


def select_materialization_rows(
    index_h5: Path,
    pose_ts: np.ndarray,
    depth_ts: np.ndarray,
    max_steps: int,
    preferred_start_time: int | None = None,
) -> tuple[np.ndarray, int, dict[str, Any]]:
    """选择一段连续索引；优先从已有S3运动片段的相近时刻开始。

    仍然保持S1/S2各自独立fixed-N的连续step，不把不相邻样本拼在一起。
    """

    with h5py.File(index_h5, "r") as f:
        t_start = f["t_start"][:].astype(np.int64)
    min_label_time = int(max(pose_ts[1], depth_ts[0]))
    requested_start = max(min_label_time, int(preferred_start_time)) if preferred_start_time is not None else min_label_time
    start = int(np.searchsorted(t_start, requested_start, side="left"))
    end = len(t_start) if max_steps <= 0 else min(len(t_start), start + max_steps)
    meta = {
        "min_label_time": min_label_time,
        "preferred_start_time": int(preferred_start_time) if preferred_start_time is not None else None,
        "selected_index_start": start,
        "selected_index_end": end,
        "selected_t_start": int(t_start[start]) if start < len(t_start) else None,
        "selection_strategy": "continuous_from_s3_motion_anchor" if preferred_start_time is not None else "continuous_from_first_label_time",
    }
    return np.arange(start, end, dtype=np.int64), start, meta


def materialize_one_scheme_n10000(
    args: argparse.Namespace,
    paths: dict[str, Path],
    geom: Any,
    pose_mats: np.ndarray,
    pose_ts: np.ndarray,
    scheme: str,
    index_h5: Path,
) -> dict[str, Any]:
    """按独立fixed-N索引物化N=10000的事件图与signed inverse TTC标签。"""

    out_name = "N10000_S1_360.h5" if scheme == "S1_360" else "N10000_S2_center256_to128.h5"
    out_h5 = args.out_dir / "independent_fixed_n" / out_name
    if args.resume and out_h5.exists():
        with h5py.File(out_h5, "r") as f:
            if bool(f.attrs.get("complete", False)):
                summary_path = out_h5.with_suffix(".summary.json")
                if summary_path.exists():
                    return json.loads(summary_path.read_text(encoding="utf-8"))
                return summarize_materialized_h5(out_h5)

    tmp_h5 = out_h5.with_suffix(".tmp.h5")
    if tmp_h5.exists():
        tmp_h5.unlink()
    started = time.time()
    source_dir = args.source_dir
    if source_dir is None:
        source_dir = next((ROOT / "EV-TTC-SNN-main/debug_sets").glob("[[]4[]]*"), None)
    preferred_start_time, selection_ref = find_s3_reference_motion_start_time(source_dir, args.sequence, 10000)

    with h5py.File(paths["depth"], "r") as f_depth:
        depth_ts = f_depth["ts"][:].astype(np.int64)
    selected_rows, skipped, selection_meta = select_materialization_rows(
        index_h5,
        pose_ts,
        depth_ts,
        args.materialize_max_steps,
        preferred_start_time=preferred_start_time,
    )
    n_steps = len(selected_rows)

    with h5py.File(index_h5, "r") as f_idx:
        idx_data = {key: f_idx[key][selected_rows] for key in [
            "step_index",
            "raw_event_start_idx",
            "raw_event_end_idx",
            "t_start",
            "t_end",
            "event_dt",
            "source_event_count",
            "raw_event_index_span",
            "reset_required",
        ]}
    if n_steps > 0:
        idx_data["reset_required"] = idx_data["reset_required"].astype(bool)
        idx_data["reset_required"][0] = True
    T, Omega, pose_valid = compute_motion_from_pose(pose_mats, pose_ts, idx_data["t_start"], idx_data["t_end"])
    T_fixed, Omega_fixed, fixed_pose_valid = compute_motion_from_pose(pose_mats, pose_ts, idx_data["t_start"], idx_data["t_start"] + 10_000)
    start_pose, start_pose_valid = interpolate_poses(pose_mats, pose_ts, idx_data["t_start"])

    f_out = write_fixed_size_scheme_h5(out_h5.with_suffix(".tmp.h5"), scheme, 10000, n_steps, args.sequence, args.compression)
    agg = defaultdict(list)
    with h5py.File(paths["data"], "r") as f_data, h5py.File(paths["depth"], "r") as f_depth:
        events = f_data["prophesee"]["left"]
        depth_ts = f_depth["ts"][:].astype(np.int64)
        depth_poses = f_depth["Cn_T_C0"][:].astype(np.float64)
        depth_cache = DepthFrameCache(f_depth["depth"]["prophesee"]["left"])
        for out_i in tqdm(range(n_steps), desc=f"materialize {scheme} N10000", unit="step"):
            raw_start = int(idx_data["raw_event_start_idx"][out_i])
            raw_end = int(idx_data["raw_event_end_idx"][out_i])
            x = events["x"][raw_start:raw_end]
            y = events["y"][raw_start:raw_end]
            p = events["p"][raw_start:raw_end]
            event_360 = accumulate_full_360_numba(x, y, p, geom.dist_map)
            if scheme == "S1_360":
                event_cnt = event_360
            else:
                event_cnt = center128_from_180(pool2_sum_event(event_360))
            event_active = event_cnt.sum(axis=0) > 0

            depth_idx = int(np.searchsorted(depth_ts, int(idx_data["t_start"][out_i]), side="left"))
            valid_depth_idx = 0 <= depth_idx < len(depth_ts) and bool(start_pose_valid[out_i])
            if valid_depth_idx:
                depth_img = project_depth_to_spatial_scheme(depth_cache.get(depth_idx), depth_poses[depth_idx], start_pose[out_i], geom, scheme)
            else:
                size = 360 if scheme == "S1_360" else 128
                depth_img = np.full((size, size), np.nan, dtype=np.float32)
                depth_idx = -1
            depth_valid = np.isfinite(depth_img) & (depth_img > 0)
            inv = np.zeros_like(depth_img, dtype=np.float32)
            inv[depth_valid] = (float(T[out_i, 2]) / np.maximum(depth_img[depth_valid], 1e-6)).astype(np.float32)
            valid_mask = event_active & depth_valid & np.isfinite(inv)
            pos_mask = valid_mask & (inv > 0)
            inv_fixed = np.zeros_like(depth_img, dtype=np.float32)
            if fixed_pose_valid[out_i]:
                inv_fixed[depth_valid] = (float(T_fixed[out_i, 2]) / np.maximum(depth_img[depth_valid], 1e-6)).astype(np.float32)
            fixed_mask = event_active & depth_valid & np.isfinite(inv_fixed)
            fixed_pos = fixed_mask & (inv_fixed > 0)
            speed_valid = bool(np.linalg.norm(T[out_i]) > CAR_SPEED_THRESHOLD)
            omega_valid = bool(np.linalg.norm(Omega[out_i]) < OMEGA_THRESHOLD)
            # 样本级监督有效性只包含运动/位姿条件；像素是否有标签由valid_ttc_mask独立表达。
            supervise_valid = bool(pose_valid[out_i] and speed_valid and omega_valid)
            direct_supervision_valid = bool(supervise_valid and np.count_nonzero(valid_mask) > 0)

            f_out["N"][out_i] = 10000
            f_out["step_index"][out_i] = int(idx_data["step_index"][out_i])
            f_out["block_index"][out_i] = out_i // 10
            f_out["step_in_block"][out_i] = out_i % 10
            for key in ["raw_event_start_idx", "raw_event_end_idx", "t_start", "t_end", "event_dt", "source_event_count", "raw_event_index_span"]:
                f_out[key][out_i] = idx_data[key][out_i]
            f_out["pose_valid"][out_i] = bool(pose_valid[out_i])
            f_out["speed_valid"][out_i] = speed_valid
            f_out["omega_valid"][out_i] = omega_valid
            f_out["supervise_valid"][out_i] = supervise_valid
            f_out["direct_supervision_valid"][out_i] = direct_supervision_valid
            f_out["timeout"][out_i] = False
            f_out["reset_required"][out_i] = bool(idx_data["reset_required"][out_i])
            f_out["depth_source_index"][out_i] = depth_idx
            f_out["T"][out_i] = T[out_i]
            f_out["Omega"][out_i] = Omega[out_i]
            f_out["T_fixed10ms"][out_i] = T_fixed[out_i]
            f_out["Omega_fixed10ms"][out_i] = Omega_fixed[out_i]
            f_out["event_cnt"][out_i] = event_cnt.astype(np.float32)
            f_out["depth_start"][out_i] = depth_img.astype(np.float32)
            f_out["signed_inverse_ttc"][out_i] = inv.astype(np.float32)
            f_out["fixed10_signed_inverse_ttc"][out_i] = inv_fixed.astype(np.float32)
            f_out["valid_ttc_mask"][out_i] = valid_mask.astype(np.uint8)
            f_out["positive_ttc_mask"][out_i] = pos_mask.astype(np.uint8)
            f_out["fixed10_valid_ttc_mask"][out_i] = fixed_mask.astype(np.uint8)
            f_out["fixed10_positive_ttc_mask"][out_i] = fixed_pos.astype(np.uint8)

            update_aggregate(agg, "event", collect_event_stats(event_cnt))
            update_aggregate(agg, "label", collect_label_stats(inv, valid_mask))
            if np.any(valid_mask):
                agg["event_vs_fixed10_inv_mae"].append(float(np.mean(np.abs(inv[valid_mask] - inv_fixed[valid_mask]))))
    f_out.attrs["skipped_initial_index_rows"] = int(skipped)
    f_out.attrs["materialize_max_steps"] = int(args.materialize_max_steps)
    for key, value in {**selection_ref, **selection_meta}.items():
        if value is not None:
            f_out.attrs[key] = value
    f_out.attrs["complete"] = True
    f_out.close()
    tmp_h5.replace(out_h5)
    summary = summarize_materialized_h5(out_h5)
    summary["elapsed_seconds"] = time.time() - started
    summary["aggregate_means"] = aggregate_mean(agg)
    (out_h5.with_suffix(".summary.json")).write_text(json.dumps(to_jsonable(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    save_materialized_visualizations(out_h5, args.out_dir / "independent_fixed_n" / "visualizations" / out_h5.stem)
    return summary


def summarize_materialized_h5(path: Path) -> dict[str, Any]:
    with h5py.File(path, "r") as f:
        n = int(f["event_dt"].shape[0])
        event_dt = f["event_dt"][:].astype(np.float64)
        speed = f["speed_valid"][:].astype(bool)
        omega = f["omega_valid"][:].astype(bool)
        sup = f["supervise_valid"][:].astype(bool)
        resets = f["reset_required"][:].astype(bool) if "reset_required" in f else np.zeros(n, dtype=bool)
        mask_rates = []
        nonzero_rates = []
        inv_pos = []
        inv_neg = []
        high = []
        chunk = 256
        for start in range(0, n, chunk):
            end = min(n, start + chunk)
            event_cnt = f["event_cnt"][start:end]
            masks = f["valid_ttc_mask"][start:end].astype(bool)
            inv = f["signed_inverse_ttc"][start:end]
            nonzero_rates.extend(np.mean(event_cnt.sum(axis=1) > 0, axis=(1, 2)).tolist())
            mask_rates.extend(np.mean(masks, axis=(1, 2)).tolist())
            for i in range(end - start):
                vals = inv[i][masks[i]]
                inv_pos.append(pct(np.count_nonzero(vals > 0), vals.size))
                inv_neg.append(pct(np.count_nonzero(vals < 0), vals.size))
                high.append(pct(np.count_nonzero(vals > 1.0), vals.size))
        complete_blocks = 0
        for block_start in range(0, n - 9, 10):
            if not np.any(resets[block_start + 1 : block_start + 10]):
                complete_blocks += 1
        return {
            "path": str(path),
            "scheme": str(f.attrs.get("spatial_scheme", "")),
            "N": int(f.attrs.get("N", 0)),
            "num_steps": n,
            "selection_strategy": str(f.attrs.get("selection_strategy", "")),
            "selected_t_start": int(f.attrs.get("selected_t_start", -1)),
            "selected_index_start": int(f.attrs.get("selected_index_start", -1)),
            "s3_reference_run_length": int(f.attrs.get("s3_reference_run_length", -1)),
            "event_dt_p50_us": float(np.percentile(event_dt, 50)) if n else np.nan,
            "event_dt_p95_us": float(np.percentile(event_dt, 95)) if n else np.nan,
            "speed_valid_rate_percent": pct(np.count_nonzero(speed), n),
            "omega_valid_rate_percent": pct(np.count_nonzero(omega), n),
            "supervise_valid_rate_percent": pct(np.count_nonzero(sup), n),
            "event_nonzero_rate_percent": 100.0 * float(np.mean(nonzero_rates)) if nonzero_rates else np.nan,
            "mask_valid_rate_percent": 100.0 * float(np.mean(mask_rates)) if mask_rates else np.nan,
            "inv_positive_rate_percent": float(np.nanmean(inv_pos)) if inv_pos else np.nan,
            "inv_negative_rate_percent": float(np.nanmean(inv_neg)) if inv_neg else np.nan,
            "high_risk_inv_gt_1_rate_percent": float(np.nanmean(high)) if high else np.nan,
            "complete_10step_sequences": complete_blocks,
            "h5_size_bytes": path.stat().st_size,
        }


def save_materialized_visualizations(h5_path: Path, out_dir: Path, max_images: int = 20) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    with h5py.File(h5_path, "r") as f:
        n = int(f["event_dt"].shape[0])
        if n == 0:
            return 0
        indices = np.linspace(0, n - 1, min(max_images, n)).round().astype(int)
        for local_id, i in enumerate(indices):
            event = f["event_cnt"][i].sum(axis=0)
            depth = f["depth_start"][i]
            inv = f["signed_inverse_ttc"][i]
            mask = f["valid_ttc_mask"][i].astype(bool)
            fixed_inv = f["fixed10_signed_inverse_ttc"][i]
            diff = np.zeros_like(inv)
            diff[mask] = inv[mask] - fixed_inv[mask]
            fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
            draw_panel(axes[0, 0], event, "event sum", "magma")
            draw_panel(axes[0, 1], depth, "depth", "viridis", mask=np.isfinite(depth))
            draw_panel(axes[0, 2], inv, "signed inverse TTC", "coolwarm", mask=mask)
            draw_panel(axes[1, 0], mask.astype(np.float32), "mask", "gray", vmin=0, vmax=1)
            draw_panel(axes[1, 1], fixed_inv, "fixed10 inverse TTC", "coolwarm", mask=mask)
            draw_panel(axes[1, 2], diff, "event - fixed10", "coolwarm", mask=mask)
            fig.suptitle(
                f"{f.attrs['spatial_scheme']} row={i} step={int(f['step_index'][i])} "
                f"dt={int(f['event_dt'][i])}us |T|={np.linalg.norm(f['T'][i]):.3f} "
                f"|Omega|={np.linalg.norm(f['Omega'][i]):.3f}",
                fontsize=10,
            )
            fig.savefig(out_dir / f"{h5_path.stem}_{local_id:03d}_row_{i:05d}.png", dpi=140)
            plt.close(fig)
            saved += 1
    return saved


def summarize_s3_source_h5(path: Path) -> dict[str, Any]:
    with h5py.File(path, "r") as f:
        n = int(f["event_dt"].shape[0])
        event_dt = f["event_dt"][:].astype(np.float64)
        speed = f["speed_valid"][:].astype(bool)
        omega = f["omega_valid"][:].astype(bool)
        sup = f["supervise_valid"][:].astype(bool)
        mask_rate = float(np.mean(f["valid_mask_ratio"][:])) * 100.0
        inv_pos = []
        inv_neg = []
        high = []
        nonzero = []
        for start in range(0, n, 512):
            end = min(n, start + 512)
            event_cnt = f["event_cnt"][start:end]
            masks = f["valid_ttc_mask"][start:end].astype(bool)
            inv = f["inverse_ttc_start"][start:end]
            nonzero.extend(np.mean(event_cnt.sum(axis=1) > 0, axis=(1, 2)).tolist())
            for i in range(end - start):
                vals = inv[i][masks[i]]
                inv_pos.append(pct(np.count_nonzero(vals > 0), vals.size))
                inv_neg.append(pct(np.count_nonzero(vals < 0), vals.size))
                high.append(pct(np.count_nonzero(vals > 1.0), vals.size))
        timeouts = int(f.attrs.get("timeout_count", 0))
        return {
            "path": str(path),
            "scheme": "S3_random128",
            "N": int(f.attrs.get("N", 10000)),
            "num_steps": n,
            "event_dt_p50_us": float(np.percentile(event_dt, 50)) if n else np.nan,
            "event_dt_p95_us": float(np.percentile(event_dt, 95)) if n else np.nan,
            "timeout_count": timeouts,
            "timeout_ratio_percent": pct(timeouts, timeouts + n),
            "event_nonzero_rate_percent": 100.0 * float(np.mean(nonzero)) if nonzero else np.nan,
            "mask_valid_rate_percent": mask_rate,
            "inv_positive_rate_percent": float(np.nanmean(inv_pos)) if inv_pos else np.nan,
            "inv_negative_rate_percent": float(np.nanmean(inv_neg)) if inv_neg else np.nan,
            "high_risk_inv_gt_1_rate_percent": float(np.nanmean(high)) if high else np.nan,
            "speed_valid_rate_percent": pct(np.count_nonzero(speed), n),
            "omega_valid_rate_percent": pct(np.count_nonzero(omega), n),
            "supervise_valid_rate_percent": pct(np.count_nonzero(sup), n),
            "complete_10step_sequences": int(f.attrs.get("complete_blocks", n // 10)),
            "h5_size_bytes": path.stat().st_size,
            "strict_source_event_count": True,
        }


def write_minimal_fix_reports(
    args: argparse.Namespace,
    mark_rows: list[dict[str, Any]],
    s2_v2_summary: dict[str, Any],
    index_summary: dict[str, Any],
    materialized: dict[str, dict[str, Any]],
    s3_summary: dict[str, Any],
    official_summary: dict[str, Any] | None,
    started: float,
) -> None:
    """写本次最小修正任务要求的中文报告。"""

    report_dir = args.out_dir / "independent_fixed_n" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    root_report_dir = args.out_dir / "reports" / "01_当前审计结论"
    root_report_dir.mkdir(parents=True, exist_ok=True)

    mark_table = markdown_table(
        ["H5", "N", "样本数", "新experiment_mode", "window_source"],
        [[Path(r["path"]).name, r["N"], r["num_steps"], "paired_window_spatial_ablation", "S3_random128"] for r in mark_rows],
    )
    text = (
        "# 旧paired-window数据口径修正报告\n\n"
        "本次没有覆盖旧H5内容，只给旧H5补充attributes，明确其真实含义为同时间窗空间消融。\n\n"
        + mark_table
        + "\n\n已明确作废的旧表述：S1是360全图固定N、S2是中央大视野固定N、S1/S2/S3事件数一致、旧结果足以决定正式训练排序。\n"
    )
    (report_dir / "旧paired-window数据口径修正报告.md").write_text(text, encoding="utf-8")
    (root_report_dir / "旧paired-window数据口径修正报告.md").write_text(text, encoding="utf-8")

    m = s2_v2_summary["metrics"]
    s2_text = (
        "# N10000旧S2与新S2标签对比\n\n"
        f"- 新H5：`{s2_v2_summary['out_h5']}`\n"
        f"- 样本数：{s2_v2_summary['num_steps']}\n"
        f"- 生成耗时：{fmt(s2_v2_summary['elapsed_seconds'], 1)} 秒\n"
        f"- 可视化目录：`{s2_v2_summary['visualization_dir']}`\n\n"
        + markdown_table(
            ["指标", "数值"],
            [
                ["inverse TTC MAE", fmt(m.get("inv_mae"))],
                ["mask IoU", fmt(m.get("mask_iou"))],
                ["旧mask有效率", f"{m.get('old_mask_rate', float('nan')):.2f}%"],
                ["新版mask有效率", f"{m.get('new_mask_rate', float('nan')):.2f}%"],
                ["旧孤立像素比例", f"{m.get('old_isolated_ratio', float('nan')):.2f}%"],
                ["新版孤立像素比例", f"{m.get('new_isolated_ratio', float('nan')):.2f}%"],
                ["旧高风险inv>1比例", f"{m.get('old_high_risk_ratio', float('nan')):.2f}%"],
                ["新版高风险inv>1比例", f"{m.get('new_high_risk_ratio', float('nan')):.2f}%"],
            ],
        )
        + "\n\n结论：新版S2不再对旧inverse TTC和最终mask做池化，而是在S2目标几何上重投影depth并重新生成mask。\n"
    )
    (report_dir / "N10000旧S2与新S2标签对比.md").write_text(s2_text, encoding="utf-8")

    index_rows = []
    for r in index_summary["rows"]:
        index_rows.append(
            [
                r["scheme"],
                r["N"],
                r["steps"],
                f"{r['event_dt_p50_us']:.1f}/{r['event_dt_p95_us']:.1f}",
                r["timeout_count"],
                f"{r['timeout_ratio_percent']:.2f}%",
                r["complete_10step_sequences"],
                r["strict_source_event_count"],
                f"{r['h5_size_bytes'] / 1024**2:.2f} MiB",
            ]
        )
    index_text = (
        "# S1S2全部N独立分片索引统计\n\n"
        "索引为一次raw pass同时维护S1/S2和5k/10k/15k/20k计数器生成。S1/S2没有ROI切换；timeout后丢弃未完成step，并从下一条有效事件继续累计，下一条完成step标记reset。\n\n"
        + markdown_table(["方案", "N", "完成step", "event_dt P50/P95(us)", "timeout数", "timeout比例", "完整10步序列", "source_event_count严格等于N", "索引H5大小"], index_rows)
        + "\n"
    )
    (report_dir / "S1S2全部N独立分片索引统计.md").write_text(index_text, encoding="utf-8")

    compare_rows = []
    for name, s in [("S1_360", materialized["S1_360"]), ("S2_center256_to128", materialized["S2_center256_to128"]), ("S3_random128", s3_summary)]:
        compare_rows.append(
            [
                name,
                f"{s['event_dt_p50_us']:.1f}/{s['event_dt_p95_us']:.1f}",
                f"{s.get('timeout_ratio_percent', 0.0):.2f}%",
                f"{s['event_nonzero_rate_percent']:.2f}%",
                f"{s['mask_valid_rate_percent']:.2f}%",
                f"{s['inv_positive_rate_percent']:.2f}%",
                f"{s['inv_negative_rate_percent']:.2f}%",
                f"{s['high_risk_inv_gt_1_rate_percent']:.2f}%",
                f"{s['speed_valid_rate_percent']:.2f}%",
                f"{s['omega_valid_rate_percent']:.2f}%",
                f"{s['supervise_valid_rate_percent']:.2f}%",
                s["complete_10step_sequences"],
                f"{s['h5_size_bytes'] / 1024**3:.2f} GiB",
            ]
        )
    compare_text = (
        "# N10000三种独立固定N方案对比\n\n"
        f"说明：S1/S2物化H5本次最多物化 `{args.materialize_max_steps}` 个连续step；"
        "为避免car序列起步/停车阶段污染监督质量统计，本次从已有S3最长车速与角速度合格片段的相近时刻开始取连续样本。"
        "S3统计来自现有完整源H5。\n\n"
        + markdown_table(
            [
                "方案",
                "event_dt P50/P95(us)",
                "timeout比例",
                "事件非零率",
                "mask有效率",
                "inv正值比例",
                "inv负值比例",
                "高风险像素比例",
                "速度通过率",
                "角速度通过率",
                "supervise_valid比例",
                "完整10步序列",
                "H5大小",
            ],
            compare_rows,
        )
        + "\n\n三者的真实事件数均按各自定义严格等于10000条源事件；S1/S2由索引H5断言，S3由源H5 `roi_source_event_count=N` 和构建逻辑保证。\n"
    )
    (report_dir / "N10000三种独立固定N方案对比.md").write_text(compare_text, encoding="utf-8")

    official_lines = [
        "# N10000三方案与EVTTC官方对比\n",
        "官方EV-TTC保持360x360 signed IIR、固定10ms窗口、car速度阈值1.3、角速度阈值0.18。本报告不拿最近邻官方样本直接算标签误差；S1/S2在各自H5中另外保存了由`t_start -> t_start+10ms`重算的fixed10 signed inverse TTC，供后续同起点比较。",
        "",
        markdown_table(
            ["方案", "时间窗定义", "event_dt P50/P95(us)", "输入尺寸", "标签说明"],
            [
                ["S1 fixed10k", "360全图累计10000源事件", f"{materialized['S1_360']['event_dt_p50_us']:.1f}/{materialized['S1_360']['event_dt_p95_us']:.1f}", "360x360", "事件窗口Tz/depth；另存fixed10"],
                ["S2 fixed10k", "中央256视野累计10000源事件", f"{materialized['S2_center256_to128']['event_dt_p50_us']:.1f}/{materialized['S2_center256_to128']['event_dt_p95_us']:.1f}", "128x128", "S2目标几何重投影depth；另存fixed10"],
                ["S3 fixed10k", "随机/九宫格128 ROI累计10000源事件", f"{s3_summary['event_dt_p50_us']:.1f}/{s3_summary['event_dt_p95_us']:.1f}", "128x128", "复用源S3事件窗口标签"],
                ["EV-TTC官方", "固定10ms", "10000/10000", "360x360x6", "官方ttcef TTC/mask/flow"],
            ],
        ),
    ]
    if official_summary:
        official_core = official_summary.get("official", official_summary)
        keep_rate = official_core.get("official_keep_rate_percent", "NA")
        keep_rate_text = f"{float(keep_rate):.2f}%" if isinstance(keep_rate, (int, float)) else f"{keep_rate}%"
        official_lines.extend(
            [
                "",
                "## 官方筛选摘要",
                f"- 官方候选样本数：{official_core.get('candidate_count', 'NA')}",
                f"- 官方保留率：{keep_rate_text}",
            ]
        )
    (report_dir / "N10000三方案与EVTTC官方对比.md").write_text("\n".join(official_lines) + "\n", encoding="utf-8")

    expand_text = (
        "# 后续是否扩展其余N的建议\n\n"
        "建议先不要立刻物化5k/15k/20k完整H5。理由：\n\n"
        "1. N=10000已经能验证S1/S2/S3三种独立fixed-N定义是否正确。\n"
        "2. S1全图360的单样本面积是128的7.91倍，完整物化全序列会快速消耗磁盘。\n"
        "3. 若N=10000显示S2相对S3监督质量没有明显优势，则扩展其它N的收益有限。\n\n"
        "下一步正式训练优先级：`S2_center256_to128_N10000` 小规模训练 > `S3_random128_N10000` 对照 > 视显存决定是否做 `S1_360_N10000` 上限验证。\n"
    )
    (report_dir / "后续是否扩展其余N的建议.md").write_text(expand_text, encoding="utf-8")

    run_summary = {
        "elapsed_seconds": time.time() - started,
        "paired_s2_v2": s2_v2_summary,
        "index_summary": index_summary,
        "materialized": materialized,
        "s3_summary": s3_summary,
    }
    (args.out_dir / "independent_fixed_n" / "minimal_fix_summary.json").write_text(
        json.dumps(to_jsonable(run_summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def run_minimal_fix(args: argparse.Namespace) -> int:
    started = time.time()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    paths = resolve_sequence_paths(args.root, args.sequence)
    with h5py.File(paths["data"], "r") as f_data:
        geom = load_camera_geometry(f_data)
        accumulate_full_360_numba(
            np.asarray([280], dtype=np.int16),
            np.asarray([0], dtype=np.int16),
            np.asarray([1], dtype=np.int8),
            geom.dist_map,
        )
        scan_independent_fixed_n_chunk(
            np.asarray([280], dtype=np.uint16),
            np.asarray([0], dtype=np.uint16),
            np.asarray([0], dtype=np.int64),
            0,
            geom.dist_map,
            np.asarray([5000], dtype=np.int32),
            np.zeros(2, dtype=np.int32),
            np.zeros(2, dtype=np.int64),
            np.zeros(2, dtype=np.int64),
            np.zeros(2, dtype=np.int64),
            np.ones(2, dtype=np.uint8),
            np.zeros(2, dtype=np.int64),
            10_000,
            16,
        )
    with h5py.File(paths["pose"], "r") as f_pose:
        pose_ts = f_pose["ts"][:].astype(np.int64)
        pose_mats = f_pose["Cn_T_C0"][:].astype(np.float64)

    mark_rows = mark_paired_window_h5s(args.out_dir)
    s2_v2_summary = fix_paired_window_s2_label_v2(args, paths, geom, pose_mats, pose_ts)
    index_summary = scan_independent_indices(args, paths, geom)

    index_dir = args.out_dir / "independent_fixed_n" / "indices"
    materialized = {
        "S1_360": materialize_one_scheme_n10000(args, paths, geom, pose_mats, pose_ts, "S1_360", index_dir / "S1_360_N10000_index.h5"),
        "S2_center256_to128": materialize_one_scheme_n10000(
            args,
            paths,
            geom,
            pose_mats,
            pose_ts,
            "S2_center256_to128",
            index_dir / "S2_center256_to128_N10000_index.h5",
        ),
    }
    source_dir = args.source_dir
    if source_dir is None:
        source_dir = next((ROOT / "EV-TTC-SNN-main/debug_sets").glob("[[]4[]]*"))
    s3_h5 = source_dir / "H5" / f"{args.sequence}_N10000.h5"
    s3_summary = summarize_s3_source_h5(s3_h5)
    official_summary = load_official_summary(args.official_summary) if args.official_summary and args.official_summary.exists() else None
    write_minimal_fix_reports(args, mark_rows, s2_v2_summary, index_summary, materialized, s3_summary, official_summary, started)
    print(json.dumps(to_jsonable({
        "mode": "minimal_fix",
        "out_dir": str(args.out_dir),
        "s2_v2": s2_v2_summary,
        "index_rows": index_summary["rows"],
        "materialized": materialized,
        "s3": s3_summary,
    }), ensure_ascii=False, indent=2)[:4000])
    return 0


def build_one_n(
    n_events: int,
    args: argparse.Namespace,
    paths: dict[str, Path],
    geom: Any,
    pose_mats: np.ndarray,
    pose_ts: np.ndarray,
    official_exp_times: np.ndarray,
) -> dict[str, Any]:
    source_h5 = args.source_dir / "H5" / f"{args.sequence}_N{n_events}.h5"
    out_h5 = args.out_dir / f"N{n_events}_spatial_compare.h5"
    done_path = args.out_dir / f"N{n_events}_spatial_compare.done.json"
    if args.resume and done_path.exists() and out_h5.exists():
        with h5py.File(out_h5, "r") as f:
            if bool(f.attrs.get("complete", False)):
                return json.loads(done_path.read_text(encoding="utf-8"))

    started = time.time()
    tmp_h5 = out_h5.with_suffix(".tmp.h5")
    if tmp_h5.exists():
        tmp_h5.unlink()

    with h5py.File(source_h5, "r") as fs:
        selected_blocks = choose_complete_blocks(fs, args.blocks_per_n, args.seed + n_events)
        selected_rows = selected_rows_for_blocks(fs, selected_blocks)
        scalar_cache = {
            "step_index": fs["step_index"][selected_rows],
            "block_index": fs["block_index"][selected_rows],
            "step_in_block": fs["step_in_block"][selected_rows],
            "raw_event_start_idx": fs["raw_event_start_idx"][selected_rows],
            "raw_event_end_idx": fs["raw_event_end_idx"][selected_rows],
            "raw_event_index_span": fs["raw_event_index_span"][selected_rows],
            "t_start": fs["t_start"][selected_rows],
            "t_end": fs["t_end"][selected_rows],
            "event_dt": fs["event_dt"][selected_rows],
            "roi_x0": fs["roi_x0"][selected_rows],
            "roi_y0": fs["roi_y0"][selected_rows],
            "reset_required": fs["reset_required"][selected_rows],
            "direct_supervision_valid": fs["direct_supervision_valid"][selected_rows],
            "valid_mask_ratio": fs["valid_mask_ratio"][selected_rows],
            "T": fs["T"][selected_rows],
            "Omega": fs["Omega"][selected_rows],
            "source_event_cnt": fs["event_cnt"][selected_rows],
            "source_inv": fs["inverse_ttc_start"][selected_rows],
            "source_mask": fs["valid_ttc_mask"][selected_rows].astype(bool),
        }

    n_steps = len(selected_rows)
    t_fixed_end = scalar_cache["t_start"].astype(np.int64) + 10_000
    T_fixed10, Omega_fixed10, fixed_pose_valid = compute_motion_from_pose(
        pose_mats, pose_ts, scalar_cache["t_start"].astype(np.int64), t_fixed_end
    )

    f_out = create_output_h5(tmp_h5, n_steps, n_events, args.sequence, args.compression)
    step_rows: list[dict[str, Any]] = []
    agg: dict[str, list[float]] = defaultdict(list)
    align_values: dict[str, list[float]] = defaultdict(list)

    with h5py.File(paths["data"], "r") as f_data, h5py.File(args.official_ttcef, "r") as f_gt:
        events = f_data["prophesee"]["left"]
        gt_ts = f_gt["ts"][:].astype(np.float64).astype(np.int64)
        for out_i in tqdm(range(n_steps), desc=f"N={n_events} spatial", unit="step"):
            raw_start = int(scalar_cache["raw_event_start_idx"][out_i])
            raw_end = int(scalar_cache["raw_event_end_idx"][out_i])
            x = events["x"][raw_start:raw_end]
            y = events["y"][raw_start:raw_end]
            p = events["p"][raw_start:raw_end]
            event_360 = accumulate_full_360_numba(x, y, p, geom.dist_map)

            t_start = int(scalar_cache["t_start"][out_i])
            depth_idx = find_nearest_depth_index(gt_ts, t_start)
            depth_360 = f_gt["depth"][depth_idx].astype(np.float32)
            depth_valid = np.isfinite(depth_360) & (depth_360 > 0)

            T_event = scalar_cache["T"][out_i].astype(np.float32)
            Omega_event = scalar_cache["Omega"][out_i].astype(np.float32)
            inv_360 = np.zeros((360, 360), dtype=np.float32)
            inv_360[depth_valid] = (float(T_event[2]) / np.maximum(depth_360[depth_valid], 1e-6)).astype(np.float32)
            event_activity_360 = event_360.sum(axis=0) > 0
            mask_360 = event_activity_360 & depth_valid & np.isfinite(inv_360)
            positive_360 = mask_360 & (inv_360 > 0)

            inv_fixed_360 = np.zeros((360, 360), dtype=np.float32)
            if fixed_pose_valid[out_i]:
                inv_fixed_360[depth_valid] = (float(T_fixed10[out_i, 2]) / np.maximum(depth_360[depth_valid], 1e-6)).astype(np.float32)
            mask_fixed_360 = event_activity_360 & depth_valid & np.isfinite(inv_fixed_360)
            positive_fixed_360 = mask_fixed_360 & (inv_fixed_360 > 0)

            event_180 = pool2_sum_event(event_360)
            mask_180 = pool2_any(mask_360)
            positive_180 = pool2_any(positive_360)
            mask_fixed_180 = pool2_any(mask_fixed_360)
            positive_fixed_180 = pool2_any(positive_fixed_360)
            depth_180 = pool2_mean_valid(depth_360, depth_valid, np.nan)
            inv_180 = pool2_mean_valid(inv_360, mask_360, 0.0)
            inv_fixed_180 = pool2_mean_valid(inv_fixed_360, mask_fixed_360, 0.0)

            event_s2 = center128_from_180(event_180)
            depth_s2 = center128_from_180(depth_180)
            inv_s2 = center128_from_180(inv_180)
            inv_fixed_s2 = center128_from_180(inv_fixed_180)
            mask_s2 = center128_from_180(mask_180)
            pos_s2 = center128_from_180(positive_180)
            mask_fixed_s2 = center128_from_180(mask_fixed_180)
            pos_fixed_s2 = center128_from_180(positive_fixed_180)

            x0 = int(scalar_cache["roi_x0"][out_i])
            y0 = int(scalar_cache["roi_y0"][out_i])
            event_s3 = crop128(event_360, x0, y0)
            depth_s3 = crop128(depth_360, x0, y0)
            inv_s3 = crop128(inv_360, x0, y0)
            inv_fixed_s3 = crop128(inv_fixed_360, x0, y0)
            mask_s3 = crop128(mask_360, x0, y0)
            pos_s3 = crop128(positive_360, x0, y0)
            mask_fixed_s3 = crop128(mask_fixed_360, x0, y0)
            pos_fixed_s3 = crop128(positive_fixed_360, x0, y0)

            target_t_end = int(scalar_cache["t_end"][out_i])
            official_idx = nearest_index(official_exp_times, target_t_end)
            official_time = int(official_exp_times[official_idx])

            # 标量写入。
            f_out["N"][out_i] = n_events
            f_out["source_row_index"][out_i] = int(selected_rows[out_i])
            for key in [
                "step_index",
                "block_index",
                "step_in_block",
                "raw_event_start_idx",
                "raw_event_end_idx",
                "raw_event_index_span",
                "t_start",
                "t_end",
                "event_dt",
                "roi_x0",
                "roi_y0",
                "reset_required",
            ]:
                f_out[key][out_i] = scalar_cache[key][out_i]
            f_out["direct_supervision_valid_source"][out_i] = bool(scalar_cache["direct_supervision_valid"][out_i])
            f_out["valid_mask_ratio_source"][out_i] = float(scalar_cache["valid_mask_ratio"][out_i])
            f_out["T_event_window"][out_i] = T_event
            f_out["Omega_event_window"][out_i] = Omega_event
            f_out["T_fixed10ms"][out_i] = T_fixed10[out_i].astype(np.float32)
            f_out["Omega_fixed10ms"][out_i] = Omega_fixed10[out_i].astype(np.float32)
            f_out["official_sample_index"][out_i] = official_idx
            f_out["official_exp_time"][out_i] = official_time
            f_out["official_delta_t_end_us"][out_i] = official_time - target_t_end

            # 空间数据写入。
            f_out["event_cnt_360"][out_i] = event_360
            f_out["event_cnt_180_center128"][out_i] = event_s2
            f_out["event_cnt_random128"][out_i] = event_s3
            f_out["depth_360"][out_i] = depth_360
            f_out["depth_180_center128"][out_i] = depth_s2
            f_out["depth_random128"][out_i] = depth_s3
            f_out["inverse_ttc_360"][out_i] = inv_360
            f_out["inverse_ttc_180_center128"][out_i] = inv_s2
            f_out["inverse_ttc_random128"][out_i] = inv_s3
            f_out["inverse_ttc_fixed10ms_360"][out_i] = inv_fixed_360
            f_out["inverse_ttc_fixed10ms_180_center128"][out_i] = inv_fixed_s2
            f_out["inverse_ttc_fixed10ms_random128"][out_i] = inv_fixed_s3
            for key, value in {
                "mask_360": mask_360,
                "positive_mask_360": positive_360,
                "mask_fixed10ms_360": mask_fixed_360,
                "positive_mask_fixed10ms_360": positive_fixed_360,
                "mask_180_center128": mask_s2,
                "positive_mask_180_center128": pos_s2,
                "mask_fixed10ms_180_center128": mask_fixed_s2,
                "positive_mask_fixed10ms_180_center128": pos_fixed_s2,
                "mask_random128": mask_s3,
                "positive_mask_random128": pos_s3,
                "mask_fixed10ms_random128": mask_fixed_s3,
                "positive_mask_fixed10ms_random128": pos_fixed_s3,
            }.items():
                f_out[key][out_i] = value.astype(np.uint8)

            update_aggregate(agg, "S1_360.event", collect_event_stats(event_360))
            update_aggregate(agg, "S2_180_center128.event", collect_event_stats(event_s2))
            update_aggregate(agg, "S3_random128.event", collect_event_stats(event_s3))
            update_aggregate(agg, "S1_360.label", collect_label_stats(inv_360, mask_360))
            update_aggregate(agg, "S2_180_center128.label", collect_label_stats(inv_s2, mask_s2))
            update_aggregate(agg, "S3_random128.label", collect_label_stats(inv_s3, mask_s3))

            src_event = scalar_cache["source_event_cnt"][out_i]
            src_inv = scalar_cache["source_inv"][out_i]
            src_mask = scalar_cache["source_mask"][out_i]
            align_values["S3_event_mae_vs_existing"].append(float(np.mean(np.abs(event_s3 - src_event))))
            src_union = np.logical_or(pos_s3, src_mask)
            align_values["S3_positive_mask_iou_vs_existing"].append(
                float(np.count_nonzero(pos_s3 & src_mask) / max(1, np.count_nonzero(src_union)))
            )
            if np.any(src_mask):
                align_values["S3_inv_mae_on_existing_mask"].append(float(np.mean(np.abs(inv_s3[src_mask] - src_inv[src_mask]))))
            if np.any(mask_s3):
                align_values["event_window_vs_fixed10_inv_mae_S3"].append(float(np.mean(np.abs(inv_s3[mask_s3] - inv_fixed_s3[mask_s3]))))

            step_row = {
                "N": n_events,
                "out_row": out_i,
                "source_row_index": int(selected_rows[out_i]),
                "block_index": int(scalar_cache["block_index"][out_i]),
                "step_in_block": int(scalar_cache["step_in_block"][out_i]),
                "t_start": int(scalar_cache["t_start"][out_i]),
                "t_end": int(scalar_cache["t_end"][out_i]),
                "event_dt": int(scalar_cache["event_dt"][out_i]),
                "roi_x0": x0,
                "roi_y0": y0,
                "Tz_event": float(T_event[2]),
                "T_norm_event": float(np.linalg.norm(T_event)),
                "Omega_norm_event": float(np.linalg.norm(Omega_event)),
                "Tz_fixed10": float(T_fixed10[out_i, 2]),
                "official_sample_index": official_idx,
                "official_delta_t_end_us": official_time - target_t_end,
                "S1_mask_rate": pct(np.count_nonzero(mask_360), mask_360.size),
                "S2_mask_rate": pct(np.count_nonzero(mask_s2), mask_s2.size),
                "S3_mask_rate": pct(np.count_nonzero(mask_s3), mask_s3.size),
                "S1_event_nonzero_rate": pct(np.count_nonzero(event_360.sum(axis=0) > 0), 360 * 360),
                "S2_event_nonzero_rate": pct(np.count_nonzero(event_s2.sum(axis=0) > 0), 128 * 128),
                "S3_event_nonzero_rate": pct(np.count_nonzero(event_s3.sum(axis=0) > 0), 128 * 128),
                "source_direct": bool(scalar_cache["direct_supervision_valid"][out_i]),
            }
            step_rows.append(step_row)

    f_out.attrs["complete"] = True
    f_out.close()
    tmp_h5.replace(out_h5)

    summary = {
        "N": n_events,
        "source_h5": str(source_h5),
        "out_h5": str(out_h5),
        "selected_blocks": len(selected_blocks),
        "selected_steps": n_steps,
        "elapsed_seconds": time.time() - started,
        "h5_size_bytes": out_h5.stat().st_size,
        "aggregate_means": aggregate_mean(agg),
        "alignment_means": aggregate_mean(align_values),
    }
    done_path.write_text(json.dumps(to_jsonable(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(args.out_dir / "csv" / f"N{n_events}_step_stats.csv", step_rows)
    return summary


def load_official_summary(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def estimate_compute_table() -> list[list[Any]]:
    rows = []
    base_area = 128 * 128
    for name, info in SCHEMES.items():
        size = int(info["size"])
        area = size * size
        ratio = area / base_area
        input_elements = 2 * area
        rows.append([name, f"{size}x{size}", input_elements, f"{ratio:.2f}x", f"{ratio:.2f}x", f"{ratio:.2f}x"])
    rows.append(["官方EV-TTC", "360x360", 6 * 360 * 360, "7.91x area / 3x channels vs 2ch", "约 7.91x 空间面积", "不适用于 SNN BPTT估计"])
    return rows


def write_reports(
    args: argparse.Namespace,
    summaries: dict[int, dict[str, Any]],
    official_summary: dict[str, Any] | None,
    runtime: dict[str, Any],
) -> None:
    reports = args.out_dir / "reports"
    reports.mkdir(parents=True, exist_ok=True)

    gen_rows = []
    for n, s in summaries.items():
        gen_rows.append([n, s["selected_blocks"], s["selected_steps"], f"{s['h5_size_bytes'] / 1024**3:.2f} GiB", fmt(s["elapsed_seconds"], 1), s["out_h5"]])
    (reports / "Car三种空间方案数据生成说明.md").write_text(
        "# Car三种空间方案数据生成说明\n\n"
        f"- 源目录：`{args.source_dir}`\n"
        f"- 输出目录：`{args.out_dir}`\n"
        f"- 每种 N 抽取完整 block 数：`{args.blocks_per_n}`\n"
        f"- 三种空间方案共享同一 `raw_event_start_idx/raw_event_end_idx/t_start/t_end/T/Omega`。\n\n"
        + markdown_table(["N", "完整block", "step数", "H5大小", "生成耗时(s)", "H5路径"], gen_rows)
        + "\n",
        encoding="utf-8",
    )

    align_rows = []
    for n, s in summaries.items():
        a = s["alignment_means"]
        align_rows.append(
            [
                n,
                fmt(a.get("S3_event_mae_vs_existing")),
                fmt(a.get("S3_positive_mask_iou_vs_existing")),
                fmt(a.get("S3_inv_mae_on_existing_mask")),
                fmt(a.get("event_window_vs_fixed10_inv_mae_S3")),
            ]
        )
    (reports / "Car三种空间方案空间对齐审计.md").write_text(
        "# Car三种空间方案空间对齐审计\n\n"
        "- S2 由 S1 的 2x2 pooling 与中心裁剪直接得到，因此空间坐标按构造对齐。\n"
        "- S3 由 S1 按已有 ROI 坐标裁剪，并与旧 ROI H5 的事件/正 TTC mask 做一致性检查。\n\n"
        + markdown_table(["N", "S3事件MAE vs旧H5", "S3正mask IoU vs旧H5", "S3 inv MAE(旧mask)", "事件窗 vs 固定10ms inv MAE(S3)"], align_rows)
        + "\n\n黄色框表示 S3 ROI，青色框表示 S2 在原 360 图中的覆盖范围 `[52:308) x [52:308)`。\n",
        encoding="utf-8",
    )

    event_rows = []
    label_rows = []
    compare_rows = []
    for n, s in summaries.items():
        means = s["aggregate_means"]
        for scheme in SCHEMES:
            event_rows.append(
                [
                    n,
                    scheme,
                    fmt(means.get(f"{scheme}.event.event_weight_sum")),
                    f"{means.get(f'{scheme}.event.nonzero_pixel_rate_percent', float('nan')):.2f}%",
                    f"{means.get(f'{scheme}.event.pos_event_ratio_percent', float('nan')):.2f}%",
                    fmt(means.get(f"{scheme}.event.pixel_value_p95")),
                    fmt(means.get(f"{scheme}.event.pixel_value_p99")),
                    fmt(means.get(f"{scheme}.event.pixel_value_max")),
                ]
            )
            label_rows.append(
                [
                    n,
                    scheme,
                    f"{means.get(f'{scheme}.label.mask_valid_pixel_rate_percent', float('nan')):.2f}%",
                    f"{means.get(f'{scheme}.label.inv_positive_pixel_ratio_percent', float('nan')):.2f}%",
                    f"{means.get(f'{scheme}.label.inv_negative_pixel_ratio_percent', float('nan')):.2f}%",
                    f"{means.get(f'{scheme}.label.high_risk_inv_gt_1_pixel_ratio_percent', float('nan')):.2f}%",
                    fmt(means.get(f"{scheme}.label.inv_p50")),
                    fmt(means.get(f"{scheme}.label.inv_p95")),
                ]
            )
        event_dt_p50 = "见源H5"
        for scheme, info in SCHEMES.items():
            compare_rows.append(
                [
                    n,
                    scheme,
                    f"{info['size']}x{info['size']}",
                    info["coverage"],
                    event_dt_p50,
                    f"{means.get(f'{scheme}.event.nonzero_pixel_rate_percent', float('nan')):.2f}%",
                    f"{means.get(f'{scheme}.label.mask_valid_pixel_rate_percent', float('nan')):.2f}%",
                    f"{means.get(f'{scheme}.label.inv_positive_pixel_ratio_percent', float('nan')):.2f}%",
                    f"{means.get(f'{scheme}.label.inv_negative_pixel_ratio_percent', float('nan')):.2f}%",
                    f"{means.get(f'{scheme}.label.high_risk_inv_gt_1_pixel_ratio_percent', float('nan')):.2f}%",
                    "全图" if scheme == "S1_360" else ("中等" if scheme.startswith("S2") else "局部"),
                ]
            )
    if official_summary:
        off = official_summary["pixel_groups"]["official_valid"]
        compare_rows.append(
            [
                "官方",
                "官方EV-TTC",
                "360x360",
                "360x360 全图",
                "10 ms",
                "signed IIR 非零率见官方图",
                f"{100.0 * off['official_mask_pixels'] / (off['sample_count'] * 360 * 360):.2f}%",
                f"{off['positive_ttc_pixel_ratio_percent']:.2f}%",
                f"{off['negative_ttc_pixel_ratio_percent']:.2f}%",
                "需按TTC阈值另算",
                "全图",
            ]
        )

    (reports / "Car三种空间方案事件统计.md").write_text(
        "# Car三种空间方案事件统计\n\n"
        + markdown_table(["N", "方案", "事件权重总和均值", "非零像素率", "正事件比例", "像素值P95", "像素值P99", "最大像素值"], event_rows)
        + "\n",
        encoding="utf-8",
    )
    (reports / "Car三种空间方案标签统计.md").write_text(
        "# Car三种空间方案标签统计\n\n"
        + markdown_table(["N", "方案", "mask有效率", "inv正值比例", "inv负值比例", "高风险inv>1比例", "inv P50", "inv P95"], label_rows)
        + "\n",
        encoding="utf-8",
    )
    (reports / "Car三种空间方案计算量对比.md").write_text(
        "# Car三种空间方案计算量对比\n\n"
        + markdown_table(["方案", "输入尺寸", "输入元素(2通道)", "第一层相对面积", "10步BPTT显存估计", "10步BPTT耗时估计"], estimate_compute_table())
        + "\n\n360x360 面积是 128x128 的约 7.91 倍；若网络通道数和隐藏状态不变，显存与耗时通常也会接近该比例增加。\n",
        encoding="utf-8",
    )
    (reports / "Car多事件数与EVTTC官方综合对比.md").write_text(
        "# Car多事件数与EVTTC官方综合对比\n\n"
        + markdown_table(["N", "方案", "输入尺寸", "覆盖视野", "event_dt P50", "事件非零率", "mask有效率", "inv正值比例", "inv负值比例", "高风险像素比例", "上下文完整性"], compare_rows)
        + "\n\n## 初步判断\n"
        "- S1 提供最高上下文和标签覆盖，但输入面积约为 128x128 的 7.91 倍。\n"
        "- S2 在 128x128 输入尺寸下覆盖中央约 256x256 原始视野，通常比随机局部 ROI 更保留道路上下文。\n"
        "- S3 事件局部细节最清楚，但上下文依赖 ROI，容易缺失全局道路结构。\n",
        encoding="utf-8",
    )
    (reports / "运行时间和磁盘占用.md").write_text(
        "# 运行时间和磁盘占用\n\n"
        f"- 开始时间：{runtime['started_at']}\n"
        f"- 结束时间：{runtime['finished_at']}\n"
        f"- 总耗时：{runtime['elapsed_seconds']:.2f} 秒\n"
        f"- 输出目录：`{args.out_dir}`\n"
        f"- 输出目录占用：{runtime['out_dir_size_gib']:.2f} GiB\n"
        f"- `/` 剩余：{runtime['root_free_gib']:.2f} GiB\n"
        f"- `/data` 剩余：{runtime['data_free_gib']:.2f} GiB\n"
        f"- 可视化数量：{runtime['visualization_count']}\n",
        encoding="utf-8",
    )

    # 给下一步训练一个可直接引用的推荐。
    (reports / "下一步训练推荐方案.md").write_text(
        "# 下一步训练推荐方案\n\n"
        "优先建议：`N=10000 + S2_180_center128`。\n\n"
        "理由：S2 与 S3 同为 128x128，训练成本相同，但 S2 保留更大的中央视野；"
        "`N=10000` 相比 `N=15000/20000` timeout 更低、样本更多，同时事件窗口仍接近 EV-TTC 的 7-10ms 时间尺度。"
        "若要验证上限，可额外训练 `N=5000 + S1_360` 的小规模版本，但不建议一开始就承担完整 360 BPTT 成本。\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        default="paired_window_spatial_ablation",
        choices=["paired_window_spatial_ablation", "independent_fixed_n_spatial_schemes", "minimal_fix"],
        help="paired_window保留旧行为；minimal_fix执行本次最小修正全流程。",
    )
    parser.add_argument("--sequence", default="car_urban_night_rittenhouse")
    parser.add_argument("--root", type=Path, default=Path("/data/M3ED"))
    parser.add_argument("--source-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "EV-TTC-SNN-main/debug_sets/car_spatial_ablation")
    parser.add_argument("--official-exp", type=Path, default=Path("/data/evttc_storage/event+SNN+TTC/data_m3ed_official_7train_3test/exp_filts/m3ed/car_urban_night_rittenhouse.h5"))
    parser.add_argument("--official-ttcef", type=Path, default=Path("/data/evttc_storage/event+SNN+TTC/data_m3ed_official_7train_3test/ttcef/m3ed/car_urban_night_rittenhouse.h5"))
    parser.add_argument("--official-summary", type=Path, default=None)
    parser.add_argument("--event-counts", type=int, nargs="+", default=EVENT_COUNTS)
    parser.add_argument("--schemes", nargs="+", default=["S1", "S2"], choices=["S1", "S2"])
    parser.add_argument("--index-only", action="store_true")
    parser.add_argument("--materialize-n", type=int, default=10000)
    parser.add_argument("--materialize-max-steps", type=int, default=5000)
    parser.add_argument("--chunk-size", type=int, default=2_000_000)
    parser.add_argument("--max-step-duration-ms", type=float, default=10.0)
    parser.add_argument("--max-raw-events", type=int, default=None)
    parser.add_argument("--min-valid-pixels", type=int, default=500)
    parser.add_argument("--blocks-per-n", type=int, default=500)
    parser.add_argument("--visual-groups-per-n", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--compression", default="lzf", choices=["lzf", "gzip", "none"])
    args = parser.parse_args()
    if args.compression == "none":
        args.compression = None
    if args.mode == "minimal_fix":
        return run_minimal_fix(args)
    if args.mode == "independent_fixed_n_spatial_schemes":
        args.out_dir.mkdir(parents=True, exist_ok=True)
        paths = resolve_sequence_paths(args.root, args.sequence)
        with h5py.File(paths["data"], "r") as f_data:
            geom = load_camera_geometry(f_data)
            scan_independent_fixed_n_chunk(
                np.asarray([280], dtype=np.uint16),
                np.asarray([0], dtype=np.uint16),
                np.asarray([0], dtype=np.int64),
                0,
                geom.dist_map,
                np.asarray([5000], dtype=np.int32),
                np.zeros(2, dtype=np.int32),
                np.zeros(2, dtype=np.int64),
                np.zeros(2, dtype=np.int64),
                np.zeros(2, dtype=np.int64),
                np.ones(2, dtype=np.uint8),
                np.zeros(2, dtype=np.int64),
                10_000,
                16,
            )
        summary = scan_independent_indices(args, paths, geom)
        print(json.dumps(to_jsonable(summary), ensure_ascii=False, indent=2)[:4000])
        return 0
    if args.source_dir is None:
        raise SystemExit("--source-dir is required for paired_window_spatial_ablation")

    started = time.time()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "logs").mkdir(exist_ok=True)
    paths = resolve_sequence_paths(args.root, args.sequence)
    with h5py.File(paths["data"], "r") as f_data:
        geom = load_camera_geometry(f_data)
        # 触发一次 numba 编译，避免正式进度条第一步看起来卡住。
        accumulate_full_360_numba(
            np.asarray([280], dtype=np.int16),
            np.asarray([0], dtype=np.int16),
            np.asarray([1], dtype=np.int8),
            geom.dist_map,
        )
    with h5py.File(paths["pose"], "r") as f_pose:
        pose_ts = f_pose["ts"][:].astype(np.int64)
        pose_mats = f_pose["Cn_T_C0"][:].astype(np.float64)
    with h5py.File(args.official_exp, "r") as f_exp:
        official_exp_times = np.rint(f_exp["exp_times"][:]).astype(np.int64)

    summaries: dict[int, dict[str, Any]] = {}
    for n_events in args.event_counts:
        summaries[n_events] = build_one_n(n_events, args, paths, geom, pose_mats, pose_ts, official_exp_times)

    vis_dir = args.out_dir / "visualizations"
    vis_count = 0
    for n_events in args.event_counts:
        vis_count += save_visualizations_for_n(
            args.out_dir / f"N{n_events}_spatial_compare.h5",
            vis_dir,
            args.official_exp,
            args.official_ttcef,
            args.visual_groups_per_n,
        )

    official_summary = None
    if args.official_summary and args.official_summary.exists():
        official_summary = load_official_summary(args.official_summary)
    runtime = {
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(started)),
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_seconds": time.time() - started,
        "out_dir_size_gib": sum(p.stat().st_size for p in args.out_dir.rglob("*") if p.is_file()) / 1024**3,
        "root_free_gib": shutil.disk_usage("/").free / 1024**3,
        "data_free_gib": shutil.disk_usage("/data").free / 1024**3,
        "visualization_count": vis_count,
    }
    output = {
        "source_dir": str(args.source_dir),
        "out_dir": str(args.out_dir),
        "summaries": summaries,
        "runtime": runtime,
    }
    (args.out_dir / "spatial_ablation_summary.json").write_text(json.dumps(to_jsonable(output), ensure_ascii=False, indent=2), encoding="utf-8")
    write_reports(args, summaries, official_summary, runtime)
    print(json.dumps(to_jsonable(output), ensure_ascii=False, indent=2)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
