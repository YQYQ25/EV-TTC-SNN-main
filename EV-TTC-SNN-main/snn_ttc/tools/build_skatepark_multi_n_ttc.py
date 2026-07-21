#!/usr/bin/env python3
"""构建 Skatepark 多事件数 ROI-SNN TTC 验证 H5。

本脚本只做数据与标签验证，不修改网络、不启动训练。主标签的速度窗口使用
当前事件分片 `[t_start, t_end]`；深度图为了避免重复重投影完整点云，复用
EV-TTC 已生成的 `ttcef/m3ed/spot_outdoor_day_skatepark_1.h5` 中最邻近
`t_start` 的 360x360 depth，并裁剪到当前 ROI。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import hdf5plugin  # noqa: F401
import numba
import numpy as np
from scipy.linalg import logm
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from snn_ttc.data.m3ed_geometry import (  # noqa: E402
    CAR_SPEED_THRESHOLD,
    OMEGA_THRESHOLD,
    ROI_NEIGHBOR_DX,
    ROI_NEIGHBOR_DY,
    SPOT_SPEED_THRESHOLD,
    accumulate_roi_event_count,
    interpolate_poses,
    load_camera_geometry,
)
from snn_ttc.tools.make_1k_ttc_debug_set import resolve_sequence_paths  # noqa: E402


ROI_CANDIDATES = [(116, 116)] + [
    (x, y) for x in (0, 116, 232) for y in (0, 116, 232) if (x, y) != (116, 116)
]
RESET_NONE = 0
RESET_NORMAL = 1
RESET_TIMEOUT = 2
RESET_SEQUENCE_START = 3
RESET_REASON_NAMES = {
    RESET_NONE: "none",
    RESET_NORMAL: "normal_roi_change",
    RESET_TIMEOUT: "timeout_roi_change",
    RESET_SEQUENCE_START: "sequence_start",
}


def infer_speed_threshold(sequence: str, override: float | None) -> float:
    """根据序列类型选择官方速度阈值；car 使用 1.3，Spot/Falcon 使用 0.25。"""

    if override is not None:
        return float(override)
    return CAR_SPEED_THRESHOLD if "car" in sequence else SPOT_SPEED_THRESHOLD


def infer_out_prefix(sequence: str, override: str | None) -> str:
    """保持旧 Skatepark 默认文件名，同时让 car/falcon 序列自动使用可读前缀。"""

    if override:
        return override
    if sequence == "spot_outdoor_day_skatepark_1":
        return "skatepark"
    return sequence


@numba.njit(cache=True)
def map_chunk_to_roi_details(
    x: np.ndarray,
    y: np.ndarray,
    p: np.ndarray,
    t: np.ndarray,
    raw_offset: int,
    dist_map: np.ndarray,
    roi_x0: int,
    roi_y0: int,
    roi_size: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """把 chunk 中进入 ROI 的源事件映射成后续 step 需要的紧凑数组。"""

    n = len(x)
    raw_out = np.empty(n, dtype=np.int64)
    t_out = np.empty(n, dtype=np.int64)
    p_out = np.empty(n, dtype=np.int8)
    bx_out = np.empty(n, dtype=np.int32)
    by_out = np.empty(n, dtype=np.int32)
    w_out = np.empty((n, 4), dtype=np.float32)
    tw_out = np.empty(n, dtype=np.float32)
    count = 0
    roi_x1 = roi_x0 + roi_size
    roi_y1 = roi_y0 + roi_size
    for i in range(n):
        xi = int(x[i])
        yi = int(y[i])
        if xi < 280 or xi > 999 or yi < 0 or yi >= 720:
            continue
        update = dist_map[yi, xi - 280]
        base_x = int(update[0])
        base_y = int(update[1])
        weights = np.zeros(4, dtype=np.float32)
        total = 0.0

        nx = base_x
        ny = base_y
        if nx >= roi_x0 and nx < roi_x1 and ny >= roi_y0 and ny < roi_y1 and update[2] > 0:
            weights[0] = update[2]
            total += update[2]
        nx = base_x + 1
        ny = base_y
        if nx >= roi_x0 and nx < roi_x1 and ny >= roi_y0 and ny < roi_y1 and update[3] > 0:
            weights[1] = update[3]
            total += update[3]
        nx = base_x
        ny = base_y + 1
        if nx >= roi_x0 and nx < roi_x1 and ny >= roi_y0 and ny < roi_y1 and update[4] > 0:
            weights[2] = update[4]
            total += update[4]
        nx = base_x + 1
        ny = base_y + 1
        if nx >= roi_x0 and nx < roi_x1 and ny >= roi_y0 and ny < roi_y1 and update[5] > 0:
            weights[3] = update[5]
            total += update[5]

        if total > 0:
            raw_out[count] = raw_offset + i
            t_out[count] = t[i]
            p_out[count] = p[i]
            bx_out[count] = base_x
            by_out[count] = base_y
            w_out[count, :] = weights
            tw_out[count] = total
            count += 1
    return (
        raw_out[:count],
        t_out[:count],
        p_out[:count],
        bx_out[:count],
        by_out[:count],
        w_out[:count],
        tw_out[:count],
    )


def next_roi_sequence(seed: int) -> List[Tuple[int, int]]:
    """生成固定 9 个 ROI 的循环顺序，第一项一定是中心 ROI。"""

    rest = ROI_CANDIDATES[1:]
    rng = random.Random(seed)
    rng.shuffle(rest)
    return [(116, 116)] + rest


class DepthCache:
    """缓存 EV-TTC 预生成 depth，避免每个 step 反复随机读大 H5。"""

    def __init__(self, depth_dset: h5py.Dataset, max_items: int = 32):
        self.depth_dset = depth_dset
        self.max_items = max_items
        self.cache: OrderedDict[int, np.ndarray] = OrderedDict()

    def get(self, idx: int) -> np.ndarray:
        idx = int(idx)
        if idx in self.cache:
            value = self.cache.pop(idx)
            self.cache[idx] = value
            return value
        value = self.depth_dset[idx].astype(np.float32)
        self.cache[idx] = value
        if len(self.cache) > self.max_items:
            self.cache.popitem(last=False)
        return value


def compute_motion_from_pose(pose_mats: np.ndarray, pose_ts: np.ndarray, t_start: np.ndarray, t_end: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """批量计算事件窗口 `[t_start,t_end]` 的 T/Omega。"""

    start_pose, v0 = interpolate_poses(pose_mats, pose_ts, t_start)
    end_pose, v1 = interpolate_poses(pose_mats, pose_ts, t_end)
    valid = v0 & v1 & (t_end > t_start)
    T = np.zeros((len(t_start), 3), dtype=np.float32)
    Omega = np.zeros((len(t_start), 3), dtype=np.float32)
    for i, ok in enumerate(valid):
        if not ok:
            continue
        dt_sec = float((t_end[i] - t_start[i]) * 1e-6)
        rel = start_pose[i] @ np.linalg.inv(end_pose[i])
        T[i] = (rel[:3, 3] / dt_sec).astype(np.float32)
        w_hat = logm(rel[:3, :3], disp=False)[0]
        Omega[i] = (np.array([w_hat[2, 1], w_hat[0, 2], w_hat[1, 0]]) / dt_sec).real.astype(np.float32)
    return T, Omega, valid


def create_output_h5(path: Path, sequence: str, n_events: int, roi_size: int, compression: str) -> h5py.File:
    """创建可追加写入的主 H5。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    f = h5py.File(path, "w")
    f.attrs["sequence_name"] = sequence
    f.attrs["N"] = n_events
    f.attrs["roi_size"] = roi_size
    f.attrs["complete"] = False
    f.attrs["reset_reason_names"] = json.dumps(RESET_REASON_NAMES, ensure_ascii=False)
    scalar_specs = {
        "N": np.int32,
        "step_index": np.int64,
        "block_index": np.int64,
        "step_in_block": np.int16,
        "roi_x0": np.int16,
        "roi_y0": np.int16,
        "raw_event_start_idx": np.int64,
        "raw_event_end_idx": np.int64,
        "t_start": np.int64,
        "t_end": np.int64,
        "event_dt": np.int64,
        "raw_event_index_span": np.int64,
        "roi_source_event_count": np.int32,
        "total_mapped_weight": np.float32,
        "T": np.float32,
        "Omega": np.float32,
        "speed_valid": bool,
        "omega_valid": bool,
        "supervise_valid": bool,
        "direct_supervision_valid": bool,
        "pose_valid": bool,
        "reset_required": bool,
        "reset_reason": np.int8,
        "valid_mask_ratio": np.float32,
        "min_ttc": np.float32,
        "pos_event_count": np.int32,
        "neg_event_count": np.int32,
    }
    for key, dtype in scalar_specs.items():
        shape = (0, 3) if key in {"T", "Omega"} else (0,)
        maxshape = (None, 3) if key in {"T", "Omega"} else (None,)
        chunks = (256, 3) if key in {"T", "Omega"} else (1024,)
        f.create_dataset(key, shape=shape, maxshape=maxshape, chunks=chunks, dtype=dtype)
    f.create_dataset("event_cnt", shape=(0, 2, roi_size, roi_size), maxshape=(None, 2, roi_size, roi_size), chunks=(1, 2, roi_size, roi_size), dtype=np.float32, compression=compression)
    f.create_dataset("inverse_ttc_start", shape=(0, roi_size, roi_size), maxshape=(None, roi_size, roi_size), chunks=(1, roi_size, roi_size), dtype=np.float32, compression=compression)
    f.create_dataset("valid_ttc_mask", shape=(0, roi_size, roi_size), maxshape=(None, roi_size, roi_size), chunks=(1, roi_size, roi_size), dtype=np.uint8, compression=compression)
    f.create_dataset("positive_ttc_mask", shape=(0, roi_size, roi_size), maxshape=(None, roi_size, roi_size), chunks=(1, roi_size, roi_size), dtype=np.uint8, compression=compression)
    f.create_dataset("official_style_mask", shape=(0, roi_size, roi_size), maxshape=(None, roi_size, roi_size), chunks=(1, roi_size, roi_size), dtype=np.uint8, compression=compression)
    f.create_dataset("sequence_name", data=np.bytes_(sequence))
    timeout_group = f.create_group("timeouts")
    for key, dtype in {
        "block_index": np.int64,
        "roi_x0": np.int16,
        "roi_y0": np.int16,
        "t_start": np.int64,
        "timeout_t": np.int64,
        "raw_event_start_idx": np.int64,
        "raw_event_end_idx": np.int64,
        "collected_event_count": np.int32,
        "completion_ratio": np.float32,
    }.items():
        timeout_group.create_dataset(key, shape=(0,), maxshape=(None,), chunks=(1024,), dtype=dtype)
    return f


def append_dataset(dset: h5py.Dataset, values: np.ndarray) -> None:
    """向 H5 dataset 追加一批数据。"""

    if len(values) == 0:
        return
    old = dset.shape[0]
    dset.resize((old + len(values), *dset.shape[1:]))
    dset[old : old + len(values)] = values


def append_batch(
    f: h5py.File,
    batch: Dict[str, List[np.ndarray | int | float | bool]],
    pose_mats: np.ndarray,
    pose_ts: np.ndarray,
    gt_ts: np.ndarray,
    depth_cache: DepthCache,
    roi_size: int,
    speed_threshold: float,
) -> None:
    """补齐 T/Omega 和标签后，将 step batch 写入 H5。"""

    if not batch["step_index"]:
        return
    t_start = np.asarray(batch["t_start"], dtype=np.int64)
    t_end = np.asarray(batch["t_end"], dtype=np.int64)
    T, Omega, pose_valid = compute_motion_from_pose(pose_mats, pose_ts, t_start, t_end)
    speed_valid = np.linalg.norm(T, axis=1) > speed_threshold
    omega_valid = np.linalg.norm(Omega, axis=1) < OMEGA_THRESHOLD
    supervise_valid = speed_valid & omega_valid & pose_valid

    n = len(t_start)
    inv = np.zeros((n, roi_size, roi_size), dtype=np.float32)
    mask = np.zeros((n, roi_size, roi_size), dtype=np.uint8)
    positive_mask = np.zeros((n, roi_size, roi_size), dtype=np.uint8)
    official_style_mask = np.zeros((n, roi_size, roi_size), dtype=np.uint8)
    valid_mask_ratio = np.zeros((n,), dtype=np.float32)
    min_ttc = np.full((n,), np.nan, dtype=np.float32)
    for i in range(n):
        if not pose_valid[i]:
            continue
        idx = int(np.searchsorted(gt_ts, t_start[i], side="left"))
        idx = min(max(idx, 0), len(gt_ts) - 1)
        if idx > 0 and abs(gt_ts[idx - 1] - t_start[i]) < abs(gt_ts[idx] - t_start[i]):
            idx -= 1
        x0 = int(batch["roi_x0"][i])
        y0 = int(batch["roi_y0"][i])
        depth = depth_cache.get(idx)[y0 : y0 + roi_size, x0 : x0 + roi_size]
        depth_valid = np.isfinite(depth) & (depth > 0)
        ttc_i = depth / (T[i, 2] + 1e-5)
        event_activity = np.asarray(batch["event_cnt"][i]).sum(axis=0) > 0
        # official_style_mask仅用于和EV-TTC旧口径对比；训练mask按signed inverse TTC口径单独计算。
        official_valid = event_activity & depth_valid & np.isfinite(ttc_i) & (ttc_i < 100)
        official_style_mask[i] = official_valid.astype(np.uint8)
        inv_i = T[i, 2] / np.maximum(depth, 1e-6)
        # signed inverse TTC 训练的像素mask不筛掉 Tz<=0，也不按inverse_ttc正负筛选。
        valid = event_activity & depth_valid & np.isfinite(inv_i)
        inv[i] = np.where(valid, inv_i, 0.0).astype(np.float32)
        mask[i] = valid.astype(np.uint8)
        positive_mask[i] = (valid & (inv_i > 0)).astype(np.uint8)
        valid_mask_ratio[i] = float(valid.mean())
        # min_ttc只用于高风险正TTC统计，避免负TTC把风险指标含义混淆。
        positive_ttc = valid & (ttc_i > 0)
        if np.any(positive_ttc):
            min_ttc[i] = float(np.min(ttc_i[positive_ttc]))
    direct_supervision_valid = supervise_valid & (valid_mask_ratio > 0)

    scalar_arrays = {
        "N": np.full((n,), f.attrs["N"], dtype=np.int32),
        "step_index": np.asarray(batch["step_index"], dtype=np.int64),
        "block_index": np.asarray(batch["block_index"], dtype=np.int64),
        "step_in_block": np.asarray(batch["step_in_block"], dtype=np.int16),
        "roi_x0": np.asarray(batch["roi_x0"], dtype=np.int16),
        "roi_y0": np.asarray(batch["roi_y0"], dtype=np.int16),
        "raw_event_start_idx": np.asarray(batch["raw_event_start_idx"], dtype=np.int64),
        "raw_event_end_idx": np.asarray(batch["raw_event_end_idx"], dtype=np.int64),
        "t_start": t_start,
        "t_end": t_end,
        "event_dt": t_end - t_start,
        "raw_event_index_span": np.asarray(batch["raw_event_index_span"], dtype=np.int64),
        "roi_source_event_count": np.asarray(batch["roi_source_event_count"], dtype=np.int32),
        "total_mapped_weight": np.asarray(batch["total_mapped_weight"], dtype=np.float32),
        "T": T,
        "Omega": Omega,
        "speed_valid": speed_valid.astype(bool),
        "omega_valid": omega_valid.astype(bool),
        "supervise_valid": supervise_valid.astype(bool),
        "direct_supervision_valid": direct_supervision_valid.astype(bool),
        "pose_valid": pose_valid.astype(bool),
        "reset_required": np.asarray(batch["reset_required"], dtype=bool),
        "reset_reason": np.asarray(batch["reset_reason"], dtype=np.int8),
        "valid_mask_ratio": valid_mask_ratio,
        "min_ttc": min_ttc,
        "pos_event_count": np.asarray(batch["pos_event_count"], dtype=np.int32),
        "neg_event_count": np.asarray(batch["neg_event_count"], dtype=np.int32),
    }
    for key, values in scalar_arrays.items():
        append_dataset(f[key], values)
    append_dataset(f["event_cnt"], np.asarray(batch["event_cnt"], dtype=np.float32))
    append_dataset(f["inverse_ttc_start"], inv)
    append_dataset(f["valid_ttc_mask"], mask)
    append_dataset(f["positive_ttc_mask"], positive_mask)
    append_dataset(f["official_style_mask"], official_style_mask)
    f.flush()
    for key in batch:
        batch[key].clear()


def append_timeout(f: h5py.File, row: dict) -> None:
    """记录一次 timeout 候选 step。"""

    g = f["timeouts"]
    for key, value in row.items():
        dset = g[key]
        old = dset.shape[0]
        dset.resize((old + 1,))
        dset[old] = value


def new_batch() -> Dict[str, list]:
    keys = [
        "step_index",
        "block_index",
        "step_in_block",
        "roi_x0",
        "roi_y0",
        "raw_event_start_idx",
        "raw_event_end_idx",
        "t_start",
        "t_end",
        "raw_event_index_span",
        "roi_source_event_count",
        "total_mapped_weight",
        "event_cnt",
        "reset_required",
        "reset_reason",
        "pos_event_count",
        "neg_event_count",
    ]
    return {k: [] for k in keys}


def build_one_n(args: argparse.Namespace, n_events: int, paths: dict, gt_path: Path, log_dir: Path) -> dict:
    """为一个 N 完整扫描序列并写主 H5。"""

    out_prefix = infer_out_prefix(args.sequence, args.out_prefix)
    speed_threshold = infer_speed_threshold(args.sequence, args.speed_threshold)
    out_path = args.out_dir / f"{out_prefix}_N{n_events}.h5"
    done_path = args.out_dir / f"{out_prefix}_N{n_events}.done.json"
    log_path = log_dir / f"N{n_events}.log"
    if args.resume and done_path.exists() and out_path.exists():
        try:
            with h5py.File(out_path, "r") as f:
                if bool(f.attrs.get("complete", False)):
                    return json.loads(done_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    tmp_path = out_path.with_suffix(".tmp.h5")
    if tmp_path.exists():
        tmp_path.unlink()

    started = time.time()
    roi_order = next_roi_sequence(args.seed)
    max_us = int(args.max_step_duration_ms * 1000)
    roi_size = args.roi_size
    step_need = n_events
    block_steps = args.steps_per_roi
    timeout_count = 0
    normal_roi_changes = 0
    timeout_roi_changes = 0
    total_blocks = 0
    complete_blocks = 0
    incomplete_blocks = 0
    last_reset_reason = RESET_SEQUENCE_START

    with log_path.open("w", encoding="utf-8") as log, h5py.File(paths["data"], "r") as f_data, h5py.File(paths["pose"], "r") as f_pose, h5py.File(gt_path, "r") as f_gt:
        log.write(f"N={n_events} speed_threshold={speed_threshold} start {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        geom = load_camera_geometry(f_data)
        pose_ts = f_pose["ts"][:].astype(np.int64)
        pose_mats = f_pose["Cn_T_C0"][:].astype(np.float64)
        gt_ts = f_gt["ts"][:].astype(np.float64).astype(np.int64)
        depth_cache = DepthCache(f_gt["depth"])
        events = f_data["prophesee"]["left"]
        total_raw = int(events["t"].shape[0])
        if args.max_raw_events is not None:
            total_raw = min(total_raw, int(args.max_raw_events))
        f_out = create_output_h5(tmp_path, args.sequence, n_events, roi_size, args.compression)

        batch = new_batch()
        roi_id = 0
        roi_x0, roi_y0 = roi_order[roi_id % len(roi_order)]
        block_index = 0
        step_index = 0
        step_in_block = 0
        cur = {k: [] for k in ["raw", "t", "p", "base_x", "base_y", "weights", "total_weight"]}
        cur_count = 0
        cursor = 0

        pbar = tqdm(total=total_raw, desc=f"N={n_events}", unit="ev")
        while cursor < total_raw:
            end = min(cursor + args.chunk_size, total_raw)
            x_chunk = events["x"][cursor:end]
            y_chunk = events["y"][cursor:end]
            p_chunk = events["p"][cursor:end]
            t_chunk = events["t"][cursor:end]
            local_start = 0
            while local_start < len(x_chunk):
                raw, ts, pol, bx, by, weights, total_weight = map_chunk_to_roi_details(
                    x_chunk[local_start:],
                    y_chunk[local_start:],
                    p_chunk[local_start:],
                    t_chunk[local_start:],
                    cursor + local_start,
                    geom.dist_map,
                    roi_x0,
                    roi_y0,
                    roi_size,
                )
                if raw.size == 0:
                    if cur_count > 0 and int(t_chunk[-1]) - int(cur["t"][0][0]) > max_us:
                        deadline = int(cur["t"][0][0]) + max_us
                        resume_local = int(np.searchsorted(t_chunk, deadline, side="right"))
                    else:
                        break
                else:
                    resume_local = None

                while raw.size > 0:
                    if cur_count == 0:
                        deadline = int(ts[0]) + max_us
                    else:
                        deadline = int(cur["t"][0][0]) + max_us

                    need = step_need - cur_count
                    if raw.size >= need:
                        candidate_t_end = int(ts[need - 1])
                        if candidate_t_end - (int(ts[0]) if cur_count == 0 else int(cur["t"][0][0])) <= max_us:
                            take = need
                            pieces = {
                                "raw": raw[:take],
                                "t": ts[:take],
                                "p": pol[:take],
                                "base_x": bx[:take],
                                "base_y": by[:take],
                                "weights": weights[:take],
                                "total_weight": total_weight[:take],
                            }
                            for k, v in pieces.items():
                                cur[k].append(v)
                            all_raw = np.concatenate(cur["raw"])
                            all_t = np.concatenate(cur["t"])
                            all_p = np.concatenate(cur["p"])
                            all_bx = np.concatenate(cur["base_x"])
                            all_by = np.concatenate(cur["base_y"])
                            all_w = np.concatenate(cur["weights"])
                            all_tw = np.concatenate(cur["total_weight"])
                            event_cnt = accumulate_roi_event_count(all_bx, all_by, all_p, all_w, roi_x0, roi_y0, roi_size)

                            batch["step_index"].append(step_index)
                            batch["block_index"].append(block_index)
                            batch["step_in_block"].append(step_in_block)
                            batch["roi_x0"].append(roi_x0)
                            batch["roi_y0"].append(roi_y0)
                            batch["raw_event_start_idx"].append(int(all_raw[0]))
                            batch["raw_event_end_idx"].append(int(all_raw[-1]) + 1)
                            batch["t_start"].append(int(all_t[0]))
                            batch["t_end"].append(int(all_t[-1]))
                            batch["raw_event_index_span"].append(int(all_raw[-1] + 1 - all_raw[0]))
                            batch["roi_source_event_count"].append(n_events)
                            batch["total_mapped_weight"].append(float(all_tw.sum()))
                            batch["event_cnt"].append(event_cnt)
                            batch["reset_required"].append(step_in_block == 0)
                            batch["reset_reason"].append(last_reset_reason if step_in_block == 0 else RESET_NONE)
                            batch["pos_event_count"].append(int(np.sum(all_p > 0)))
                            batch["neg_event_count"].append(int(np.sum(all_p <= 0)))
                            step_index += 1
                            step_in_block += 1
                            cur = {k: [] for k in cur}
                            cur_count = 0
                            if len(batch["step_index"]) >= args.write_batch_steps:
                                append_batch(f_out, batch, pose_mats, pose_ts, gt_ts, depth_cache, roi_size, speed_threshold)

                            raw = raw[take:]
                            ts = ts[take:]
                            pol = pol[take:]
                            bx = bx[take:]
                            by = by[take:]
                            weights = weights[take:]
                            total_weight = total_weight[take:]

                            if step_in_block >= block_steps:
                                complete_blocks += 1
                                total_blocks += 1
                                normal_roi_changes += 1
                                block_index += 1
                                step_in_block = 0
                                last_reset_reason = RESET_NORMAL
                                prev_roi = (roi_x0, roi_y0)
                                roi_id += 1
                                roi_x0, roi_y0 = roi_order[roi_id % len(roi_order)]
                                if (roi_x0, roi_y0) == prev_roi:
                                    roi_id += 1
                                    roi_x0, roi_y0 = roi_order[roi_id % len(roi_order)]
                                if raw.size > 0:
                                    # ROI 已切换，剩余 chunk 必须按新 ROI 重新映射。
                                    local_start = int(raw[0] - cursor)
                                    break
                            continue

                        # N 条事件无法在 10ms 内集齐，timeout。
                        over = int(np.searchsorted(ts, deadline, side="right"))
                        append_timeout(
                            f_out,
                            {
                                "block_index": block_index,
                                "roi_x0": roi_x0,
                                "roi_y0": roi_y0,
                                "t_start": int(ts[0]) if cur_count == 0 else int(cur["t"][0][0]),
                                "timeout_t": deadline,
                                "raw_event_start_idx": int(raw[0]) if cur_count == 0 else int(cur["raw"][0][0]),
                                "raw_event_end_idx": int(raw[max(over - 1, 0)]) + 1 if raw.size else cursor + local_start,
                                "collected_event_count": cur_count + over,
                                "completion_ratio": float((cur_count + over) / step_need),
                            },
                        )
                        timeout_count += 1
                        timeout_roi_changes += 1
                        if step_in_block > 0:
                            incomplete_blocks += 1
                            total_blocks += 1
                        block_index += 1
                        step_in_block = 0
                        cur = {k: [] for k in cur}
                        cur_count = 0
                        last_reset_reason = RESET_TIMEOUT
                        prev_roi = (roi_x0, roi_y0)
                        roi_id += 1
                        roi_x0, roi_y0 = roi_order[roi_id % len(roi_order)]
                        if (roi_x0, roi_y0) == prev_roi:
                            roi_id += 1
                            roi_x0, roi_y0 = roi_order[roi_id % len(roi_order)]
                        if over < raw.size:
                            local_start = int(raw[over] - cursor)
                        else:
                            local_start = len(x_chunk)
                        break

                    # 当前 chunk 内还不够 N，先累积；若已超过 10ms，则 timeout。
                    for k, v in {
                        "raw": raw,
                        "t": ts,
                        "p": pol,
                        "base_x": bx,
                        "base_y": by,
                        "weights": weights,
                        "total_weight": total_weight,
                    }.items():
                        cur[k].append(v)
                    cur_count += raw.size
                    raw = raw[:0]

                else:
                    break

                if raw.size == 0:
                    if resume_local is not None:
                        # 无 ROI 事件但已经跨过 timeout deadline。
                        append_timeout(
                            f_out,
                            {
                                "block_index": block_index,
                                "roi_x0": roi_x0,
                                "roi_y0": roi_y0,
                                "t_start": int(cur["t"][0][0]),
                                "timeout_t": int(cur["t"][0][0]) + max_us,
                                "raw_event_start_idx": int(cur["raw"][0][0]),
                                "raw_event_end_idx": cursor + resume_local,
                                "collected_event_count": cur_count,
                                "completion_ratio": float(cur_count / step_need),
                            },
                        )
                        timeout_count += 1
                        timeout_roi_changes += 1
                        if step_in_block > 0:
                            incomplete_blocks += 1
                            total_blocks += 1
                        block_index += 1
                        step_in_block = 0
                        cur = {k: [] for k in cur}
                        cur_count = 0
                        last_reset_reason = RESET_TIMEOUT
                        roi_id += 1
                        roi_x0, roi_y0 = roi_order[roi_id % len(roi_order)]
                        local_start = resume_local
                        continue
                    break
                if local_start >= len(x_chunk):
                    break
            pbar.update(end - cursor)
            cursor = end

        pbar.close()
        if step_in_block > 0:
            incomplete_blocks += 1
            total_blocks += 1
        append_batch(f_out, batch, pose_mats, pose_ts, gt_ts, depth_cache, roi_size, speed_threshold)
        f_out.attrs["complete"] = True
        f_out.attrs["timeout_count"] = timeout_count
        f_out.attrs["normal_roi_changes"] = normal_roi_changes
        f_out.attrs["timeout_roi_changes"] = timeout_roi_changes
        f_out.attrs["complete_blocks"] = complete_blocks
        f_out.attrs["incomplete_blocks"] = incomplete_blocks
        f_out.attrs["total_blocks"] = total_blocks
        f_out.attrs["elapsed_seconds"] = time.time() - started
        f_out.close()

    tmp_path.replace(out_path)
    summary = {
        "N": n_events,
        "path": str(out_path),
        "size_bytes": out_path.stat().st_size,
        "elapsed_seconds": time.time() - started,
        "speed_threshold": speed_threshold,
        "timeout_count": timeout_count,
        "normal_roi_changes": normal_roi_changes,
        "timeout_roi_changes": timeout_roi_changes,
        "complete_blocks": complete_blocks,
        "incomplete_blocks": incomplete_blocks,
        "total_blocks": total_blocks,
    }
    done_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", default="spot_outdoor_day_skatepark_1")
    parser.add_argument("--event-counts", type=int, nargs="+", default=[5000, 10000, 15000, 20000])
    parser.add_argument("--root", type=Path, default=Path("/data/M3ED"))
    parser.add_argument("--gt-path", type=Path, default=Path("/data/evttc_storage/event+SNN+TTC/data_m3ed_official_7train_3test/ttcef/m3ed/spot_outdoor_day_skatepark_1.h5"))
    parser.add_argument("--out-dir", type=Path, default=ROOT / "debug_sets" / "skatepark_multi_n_ttc")
    parser.add_argument("--out-prefix", default=None, help="输出 H5 前缀；默认 Skatepark 保持旧名，其他序列用 sequence 名")
    parser.add_argument("--speed-threshold", type=float, default=None, help="样本级速度阈值；默认按 car/非 car 自动选择")
    parser.add_argument("--roi-size", type=int, default=128)
    parser.add_argument("--steps-per-roi", type=int, default=10)
    parser.add_argument("--max-step-duration-ms", type=float, default=10.0)
    parser.add_argument("--audit-fixed-window-ms", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunk-size", type=int, default=5_000_000)
    parser.add_argument("--write-batch-steps", type=int, default=64)
    parser.add_argument("--compression", default="lzf", choices=["lzf", "gzip", "none"])
    parser.add_argument("--max-raw-events", type=int, default=None, help="调试用；正式运行不要设置")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()
    if args.compression == "none":
        args.compression = None
    return args


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = args.out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    paths = resolve_sequence_paths(args.root, args.sequence)
    free = shutil.disk_usage(args.out_dir.parent).free
    run_summary = {
        "sequence": args.sequence,
        "event_counts": args.event_counts,
        "out_dir": str(args.out_dir),
        "free_bytes_before": free,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "results": [],
        "failures": [],
    }
    for n_events in args.event_counts:
        try:
            result = build_one_n(args, n_events, paths, args.gt_path, log_dir)
            run_summary["results"].append(result)
        except Exception as exc:
            failure = {"N": n_events, "error": repr(exc)}
            run_summary["failures"].append(failure)
            (log_dir / f"N{n_events}_error.txt").write_text(repr(exc), encoding="utf-8")
            if not args.continue_on_error:
                raise
    run_summary["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    run_summary["free_bytes_after"] = shutil.disk_usage(args.out_dir.parent).free
    (args.out_dir / "build_summary.json").write_text(json.dumps(run_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(run_summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
