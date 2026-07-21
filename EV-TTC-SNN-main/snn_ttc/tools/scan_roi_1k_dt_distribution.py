#!/usr/bin/env python3
"""扫描固定 ROI 内 1k 源事件对应的真实物理时间跨度。"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import h5py
import numba
import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from snn_ttc.data.m3ed_geometry import (  # noqa: E402
    ROI_NEIGHBOR_DX,
    ROI_NEIGHBOR_DY,
    load_camera_geometry,
)


ROI_SIZE = 128
MAX_ROI_X0 = 360 - ROI_SIZE
MAX_ROI_Y0 = 360 - ROI_SIZE


def read_train_sequences(train_h5: Path) -> List[str]:
    """从当前 7train 合并 H5 自动读取训练序列名，避免手写维护列表。"""

    with h5py.File(train_h5, "r") as f:
        return [x.decode() if hasattr(x, "decode") else str(x) for x in f["file_names"][:]]


def resolve_data_file(sequence: str, roots: Iterable[Path]) -> Path:
    """在 /data/M3ED 和 evttc_storage 的常见布局中查找原始 data.h5。"""

    candidates: List[Path] = []
    for root in roots:
        candidates.extend(
            [
                root / "official_train" / sequence / f"{sequence}_data.h5",
                root / "paper_alignment" / sequence / f"{sequence}_data.h5",
                root / "task_oriented" / sequence / f"{sequence}_data.h5",
                root / sequence / f"{sequence}_data.h5",
                root / "events" / "m3ed" / sequence / f"{sequence}_data.h5",
            ]
        )
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"找不到 {sequence}_data.h5，搜索根目录：{[str(r) for r in roots]}")


def roi_for_block(block_index: int, rng: random.Random) -> Tuple[int, int]:
    """第 0 个 block 固定中心 ROI，之后使用固定随机种子均匀采样合法 ROI。"""

    if block_index == 0:
        return 116, 116
    return rng.randint(0, MAX_ROI_X0), rng.randint(0, MAX_ROI_Y0)


def map_chunk_to_roi(
    x: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    raw_offset: int,
    dist_map: np.ndarray,
    roi_x0: int,
    roi_y0: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """筛出当前 chunk 中贡献进入 ROI 的源事件，并返回 raw index、时间和 ROI 内权重。"""

    in_crop = (x >= 280) & (x <= 999) & (y < 720)
    if not np.any(in_crop):
        return (
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.float32),
        )

    local_indices = np.nonzero(in_crop)[0]
    update = dist_map[y[in_crop], x[in_crop] - 280]
    active = update[:, 2:] > 0
    base_x = update[:, 0].astype(np.int32)
    base_y = update[:, 1].astype(np.int32)
    nbr_x = base_x[:, None] + ROI_NEIGHBOR_DX[None, :]
    nbr_y = base_y[:, None] + ROI_NEIGHBOR_DY[None, :]
    inside = (
        active
        & (nbr_x >= roi_x0)
        & (nbr_x < roi_x0 + ROI_SIZE)
        & (nbr_y >= roi_y0)
        & (nbr_y < roi_y0 + ROI_SIZE)
    )
    total_weight = np.where(inside, update[:, 2:], 0.0).sum(axis=1).astype(np.float32)
    keep = total_weight > 0
    return (
        raw_offset + local_indices[keep].astype(np.int64),
        t[local_indices[keep]].astype(np.int64),
        total_weight[keep],
    )


@numba.njit(cache=True)
def map_chunk_to_roi_numba(
    x: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    raw_offset: int,
    dist_map: np.ndarray,
    roi_x0: int,
    roi_y0: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """numba 加速版 ROI 筛选；避免完整扫描大序列时 Python/NumPy 反复分配。"""

    n = len(x)
    raw_out = np.empty(n, dtype=np.int64)
    t_out = np.empty(n, dtype=np.int64)
    w_out = np.empty(n, dtype=np.float32)
    count = 0
    roi_x1 = roi_x0 + ROI_SIZE
    roi_y1 = roi_y0 + ROI_SIZE
    for i in range(n):
        xi = int(x[i])
        yi = int(y[i])
        if xi < 280 or xi > 999 or yi < 0 or yi >= 720:
            continue
        update = dist_map[yi, xi - 280]
        base_x = int(update[0])
        base_y = int(update[1])
        total = 0.0

        nx = base_x
        ny = base_y
        if nx >= roi_x0 and nx < roi_x1 and ny >= roi_y0 and ny < roi_y1 and update[2] > 0:
            total += update[2]
        nx = base_x + 1
        ny = base_y
        if nx >= roi_x0 and nx < roi_x1 and ny >= roi_y0 and ny < roi_y1 and update[3] > 0:
            total += update[3]
        nx = base_x
        ny = base_y + 1
        if nx >= roi_x0 and nx < roi_x1 and ny >= roi_y0 and ny < roi_y1 and update[4] > 0:
            total += update[4]
        nx = base_x + 1
        ny = base_y + 1
        if nx >= roi_x0 and nx < roi_x1 and ny >= roi_y0 and ny < roi_y1 and update[5] > 0:
            total += update[5]

        if total > 0:
            raw_out[count] = raw_offset + i
            t_out[count] = t[i]
            w_out[count] = total
            count += 1
    return raw_out[:count], t_out[:count], w_out[:count]


def finalize_block(
    sequence: str,
    block_index: int,
    step_start_index: int,
    roi_x0: int,
    roi_y0: int,
    raw_indices: np.ndarray,
    timestamps: np.ndarray,
    weights: np.ndarray,
    events_per_step: int,
    steps_per_block: int,
) -> Tuple[List[dict], dict]:
    """把一个 10k ROI 事件 block 切成 10 个 step 记录和一个 block 记录。"""

    rows: List[dict] = []
    assert raw_indices.shape[0] == events_per_step * steps_per_block
    for step_in_block in range(steps_per_block):
        start = step_in_block * events_per_step
        end = start + events_per_step
        step_raw = raw_indices[start:end]
        step_t = timestamps[start:end]
        step_w = weights[start:end]
        t_start = int(step_t[0])
        t_end = int(step_t[-1])
        raw_start = int(step_raw[0])
        raw_end = int(step_raw[-1]) + 1
        rows.append(
            {
                "sequence_name": sequence,
                "platform": "car" if sequence.startswith("car_") else "spot",
                "block_index": block_index,
                "step_index": step_start_index + step_in_block,
                "step_in_block": step_in_block,
                "roi_x0": roi_x0,
                "roi_y0": roi_y0,
                "roi_source_event_count": events_per_step,
                "raw_event_start_idx": raw_start,
                "raw_event_end_idx": raw_end,
                "t_start": t_start,
                "t_end": t_end,
                "dt": t_end - t_start,
                "dt_ms": (t_end - t_start) / 1000.0,
                "event_rate_ev_per_s": events_per_step / max((t_end - t_start) * 1e-6, 1e-12),
                "raw_event_index_span": raw_end - raw_start,
                "total_mapped_weight": float(np.sum(step_w)),
            }
        )

    block = {
        "sequence_name": sequence,
        "platform": "car" if sequence.startswith("car_") else "spot",
        "block_index": block_index,
        "block_t_start": int(timestamps[0]),
        "block_t_end": int(timestamps[-1]),
        "block_duration": int(timestamps[-1] - timestamps[0]),
        "block_duration_ms": float((timestamps[-1] - timestamps[0]) / 1000.0),
        "block_roi_x0": roi_x0,
        "block_roi_y0": roi_y0,
        "step_start_index": step_start_index,
        "step_end_index": step_start_index + steps_per_block,
        "roi_source_event_count": events_per_step * steps_per_block,
        "raw_event_start_idx": int(raw_indices[0]),
        "raw_event_end_idx": int(raw_indices[-1]) + 1,
        "raw_event_index_span": int(raw_indices[-1] + 1 - raw_indices[0]),
    }
    return rows, block


def scan_sequence(
    sequence: str,
    data_file: Path,
    out_dir: Path,
    events_per_step: int,
    steps_per_block: int,
    chunk_size: int,
    seed: int,
    max_blocks: int | None = None,
) -> Tuple[Path, Path, dict]:
    """完整扫描一个序列，只统计事件时间尺度，不读取 depth/pose/TTC。"""

    seq_dir = out_dir / sequence
    seq_dir.mkdir(parents=True, exist_ok=True)
    step_csv = seq_dir / "steps.csv"
    block_csv = seq_dir / "blocks.csv"
    meta_json = seq_dir / "meta.json"

    rng = random.Random(seed)
    start_time = time.time()
    step_fields = [
        "sequence_name",
        "platform",
        "block_index",
        "step_index",
        "step_in_block",
        "roi_x0",
        "roi_y0",
        "roi_source_event_count",
        "raw_event_start_idx",
        "raw_event_end_idx",
        "t_start",
        "t_end",
        "dt",
        "dt_ms",
        "event_rate_ev_per_s",
        "raw_event_index_span",
        "total_mapped_weight",
    ]
    block_fields = [
        "sequence_name",
        "platform",
        "block_index",
        "block_t_start",
        "block_t_end",
        "block_duration",
        "block_duration_ms",
        "block_roi_x0",
        "block_roi_y0",
        "step_start_index",
        "step_end_index",
        "roi_source_event_count",
        "raw_event_start_idx",
        "raw_event_end_idx",
        "raw_event_index_span",
    ]

    with h5py.File(data_file, "r") as f_data, step_csv.open("w", newline="", encoding="utf-8") as sf, block_csv.open("w", newline="", encoding="utf-8") as bf:
        geom = load_camera_geometry(f_data)
        events = f_data["prophesee"]["left"]
        n_events = int(events["t"].shape[0])
        step_writer = csv.DictWriter(sf, fieldnames=step_fields)
        block_writer = csv.DictWriter(bf, fieldnames=block_fields)
        step_writer.writeheader()
        block_writer.writeheader()

        block_index = 0
        step_index = 0
        cursor = 0
        roi_x0, roi_y0 = roi_for_block(block_index, rng)
        cur_raw: List[np.ndarray] = []
        cur_t: List[np.ndarray] = []
        cur_w: List[np.ndarray] = []
        cur_count = 0
        block_need = events_per_step * steps_per_block
        last_raw_end = 0

        pbar = tqdm(total=n_events, desc=f"scan {sequence}", unit="ev")
        while cursor < n_events:
            if max_blocks is not None and block_index >= max_blocks:
                break
            end = min(cursor + chunk_size, n_events)
            x_chunk = events["x"][cursor:end]
            y_chunk = events["y"][cursor:end]
            t_chunk = events["t"][cursor:end]
            local_start = 0

            while local_start < len(x_chunk):
                raw, ts, weight = map_chunk_to_roi_numba(
                    x_chunk[local_start:],
                    y_chunk[local_start:],
                    t_chunk[local_start:],
                    cursor + local_start,
                    geom.dist_map,
                    roi_x0,
                    roi_y0,
                )
                if raw.size == 0:
                    break

                need = block_need - cur_count
                take = min(need, raw.size)
                cur_raw.append(raw[:take])
                cur_t.append(ts[:take])
                cur_w.append(weight[:take])
                cur_count += take

                if cur_count == block_need:
                    raw_block = np.concatenate(cur_raw)
                    t_block = np.concatenate(cur_t)
                    w_block = np.concatenate(cur_w)
                    rows, block = finalize_block(
                        sequence,
                        block_index,
                        step_index,
                        roi_x0,
                        roi_y0,
                        raw_block,
                        t_block,
                        w_block,
                        events_per_step,
                        steps_per_block,
                    )
                    step_writer.writerows(rows)
                    block_writer.writerow(block)
                    last_raw_end = int(raw_block[-1]) + 1

                    block_index += 1
                    step_index += steps_per_block
                    roi_x0, roi_y0 = roi_for_block(block_index, rng)
                    cur_raw, cur_t, cur_w = [], [], []
                    cur_count = 0

                    # ROI 切换后从刚刚使用的最后一个原始事件之后继续扫描，避免重叠。
                    local_start = last_raw_end - cursor
                    if max_blocks is not None and block_index >= max_blocks:
                        break
                else:
                    break

            pbar.update(end - cursor)
            cursor = end
        pbar.close()

    meta = {
        "sequence_name": sequence,
        "data_file": str(data_file),
        "events_per_step": events_per_step,
        "steps_per_block": steps_per_block,
        "roi_size": ROI_SIZE,
        "seed": seed,
        "chunk_size": chunk_size,
        "blocks": block_index,
        "steps": step_index,
        "incomplete_roi_events_discarded": cur_count,
        "elapsed_seconds": time.time() - start_time,
    }
    meta_json.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return step_csv, block_csv, meta


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-h5", type=Path, default=Path("/data/evttc_storage/event+SNN+TTC/ttcef_official_7train_3test/train.h5"))
    parser.add_argument("--data-root", type=Path, action="append", default=[])
    parser.add_argument("--sequence", action="append", help="只扫描指定序列；不传则读取 train-h5 的 file_names")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "analysis" / "roi_1k_dt_distribution")
    parser.add_argument("--events-per-step", type=int, default=1000)
    parser.add_argument("--steps-per-block", type=int, default=10)
    parser.add_argument("--chunk-size", type=int, default=2_000_000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-blocks", type=int, default=None, help="调试用；默认完整扫描")
    args = parser.parse_args()

    roots = args.data_root or [
        Path("/data/M3ED"),
        Path("/data/evttc_storage/event+SNN+TTC/data_m3ed_official_7train_3test"),
        Path("/data/evttc_storage/event+SNN+TTC/data_m3ed_min"),
    ]
    sequences = args.sequence if args.sequence else read_train_sequences(args.train_h5)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    run_meta = {
        "train_h5": str(args.train_h5),
        "sequences": sequences,
        "roots": [str(r) for r in roots],
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "sequence_results": [],
    }
    total_start = time.time()
    for sequence in sequences:
        data_file = resolve_data_file(sequence, roots)
        step_csv, block_csv, meta = scan_sequence(
            sequence,
            data_file,
            args.out_dir,
            args.events_per_step,
            args.steps_per_block,
            args.chunk_size,
            args.seed,
            max_blocks=args.max_blocks,
        )
        meta["step_csv"] = str(step_csv)
        meta["block_csv"] = str(block_csv)
        run_meta["sequence_results"].append(meta)

    run_meta["elapsed_seconds"] = time.time() - total_start
    run_meta["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    (args.out_dir / "scan_run_meta.json").write_text(json.dumps(run_meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(run_meta, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
