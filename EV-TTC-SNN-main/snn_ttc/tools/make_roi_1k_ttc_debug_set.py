#!/usr/bin/env python3
"""生成固定 ROI 内连续 1k 源事件的 M3ED TTC debug H5；不启动训练。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import h5py
import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from snn_ttc.data.build_1k_step_index import find_event_index_at_or_after  # noqa: E402
from snn_ttc.data.m3ed_geometry import (  # noqa: E402
    accumulate_roi_event_count,
    compute_labels_for_steps,
    load_camera_geometry,
    map_raw_events_to_roi,
)
from snn_ttc.tools.make_1k_ttc_debug_set import resolve_sequence_paths  # noqa: E402


def collect_roi_events(
    f_data: h5py.File,
    geom,
    start_event_idx: int,
    required_roi_events: int,
    roi_x0: int,
    roi_y0: int,
    roi_size: int,
    chunk_size: int,
) -> Dict[str, np.ndarray]:
    """从原始事件流中连续扫描，收集贡献进入固定 ROI 的源事件。"""

    events = f_data["prophesee"]["left"]
    total_events = int(events["t"].shape[0])
    cursor = int(start_event_idx)
    chunks: Dict[str, List[np.ndarray]] = {
        "raw_indices": [],
        "t": [],
        "p": [],
        "base_x": [],
        "base_y": [],
        "weights_in_roi": [],
        "inside_mask": [],
        "fully_inside": [],
        "total_weight": [],
    }
    collected = 0

    with tqdm(total=required_roi_events, desc="collecting ROI source events") as pbar:
        while cursor < total_events and collected < required_roi_events:
            end = min(cursor + chunk_size, total_events)
            raw_indices = np.arange(cursor, end, dtype=np.int64)
            roi = map_raw_events_to_roi(
                events["x"][cursor:end],
                events["y"][cursor:end],
                events["p"][cursor:end],
                events["t"][cursor:end],
                raw_indices,
                geom.dist_map,
                roi_x0,
                roi_y0,
                roi_size,
            )
            if roi["raw_indices"].size:
                remain = required_roi_events - collected
                take = min(remain, roi["raw_indices"].size)
                for key in chunks:
                    chunks[key].append(roi[key][:take])
                collected += take
                pbar.update(take)
                if collected >= required_roi_events:
                    break
            cursor = end

    if collected < required_roi_events:
        raise RuntimeError(f"ROI events not enough: collected {collected}, need {required_roi_events}")

    return {key: np.concatenate(value, axis=0) for key, value in chunks.items()}


def build_roi_steps(
    roi_events: Dict[str, np.ndarray],
    num_steps: int,
    events_per_step: int,
    roi_x0: int,
    roi_y0: int,
    roi_size: int,
) -> Dict[str, np.ndarray]:
    """把连续 ROI 事件流按每 1000 条源事件切成 step，并生成事件计数图。"""

    event_cnt = np.zeros((num_steps, 2, roi_size, roi_size), dtype=np.float32)
    raw_start = np.zeros((num_steps,), dtype=np.int64)
    raw_end = np.zeros((num_steps,), dtype=np.int64)
    rank_start = np.zeros((num_steps,), dtype=np.int64)
    rank_end = np.zeros((num_steps,), dtype=np.int64)
    t_start = np.zeros((num_steps,), dtype=np.int64)
    t_end = np.zeros((num_steps,), dtype=np.int64)
    roi_count = np.zeros((num_steps,), dtype=np.int32)
    fully_inside = np.zeros((num_steps,), dtype=np.int32)
    partially_inside = np.zeros((num_steps,), dtype=np.int32)
    total_weight = np.zeros((num_steps,), dtype=np.float32)
    pos_count = np.zeros((num_steps,), dtype=np.int32)
    neg_count = np.zeros((num_steps,), dtype=np.int32)
    pos_weight = np.zeros((num_steps,), dtype=np.float32)
    neg_weight = np.zeros((num_steps,), dtype=np.float32)

    for step in range(num_steps):
        start = step * events_per_step
        end = start + events_per_step
        sl = slice(start, end)
        cnt = accumulate_roi_event_count(
            roi_events["base_x"][sl],
            roi_events["base_y"][sl],
            roi_events["p"][sl],
            roi_events["weights_in_roi"][sl],
            roi_x0,
            roi_y0,
            roi_size,
        )

        event_cnt[step] = cnt
        raw_start[step] = int(roi_events["raw_indices"][start])
        raw_end[step] = int(roi_events["raw_indices"][end - 1]) + 1
        rank_start[step] = start
        rank_end[step] = end
        t_start[step] = int(roi_events["t"][start])
        t_end[step] = int(roi_events["t"][end - 1])
        roi_count[step] = events_per_step
        fully_inside[step] = int(np.sum(roi_events["fully_inside"][sl]))
        partially_inside[step] = int(events_per_step - fully_inside[step])
        total_weight[step] = float(np.sum(roi_events["total_weight"][sl]))
        pos = roi_events["p"][sl] > 0
        pos_count[step] = int(np.sum(pos))
        neg_count[step] = int(events_per_step - pos_count[step])
        pos_weight[step] = float(cnt[0].sum())
        neg_weight[step] = float(cnt[1].sum())

        # 指导文件要求的核心断言：1k 统计的是 ROI 源事件条数，不是像素权重和。
        assert roi_count[step] == events_per_step
        assert rank_end[step] - rank_start[step] == events_per_step
        assert event_cnt[step].shape == (2, roi_size, roi_size)
        assert event_cnt[step].dtype == np.float32
        assert np.all(event_cnt[step] >= 0)
        assert np.isfinite(event_cnt[step]).all()
        assert t_end[step] > t_start[step]
        assert raw_end[step] > raw_start[step]
        if step > 0:
            assert rank_start[step] == rank_end[step - 1]
            assert raw_start[step] >= raw_end[step - 1]

    return {
        "event_cnt": event_cnt,
        "raw_event_start_idx": raw_start,
        "raw_event_end_idx": raw_end,
        "roi_event_rank_start": rank_start,
        "roi_event_rank_end": rank_end,
        "t_start": t_start,
        "t_end": t_end,
        "dt": t_end - t_start,
        "roi_source_event_count": roi_count,
        "fully_inside_event_count": fully_inside,
        "partially_inside_event_count": partially_inside,
        "total_mapped_weight": total_weight,
        "positive_source_count": pos_count,
        "negative_source_count": neg_count,
        "positive_weight_sum": pos_weight,
        "negative_weight_sum": neg_weight,
    }


def write_notes(out_dir: Path, h5_path: Path, sequence: str, args: argparse.Namespace, stdout_summary: str) -> None:
    """写入中文实现说明和实际运行命令，便于后续复现实验。"""

    notes = out_dir / "实现说明.md"
    notes.write_text(
        "\n".join(
            [
                "# 固定 ROI 内 1k 事件 TTC Debug Set 实现说明",
                "",
                "## 修正内容",
                "- 旧方案是先从全分辨率原始事件流连续取 1000 条，再映射到 360x360。",
                "- 新方案先完成 EV-TTC 同口径去畸变、中央裁剪和 2 倍下采样，再筛选固定 128x128 ROI。",
                "- 每个 step 由 ROI 内连续出现的 1000 条源事件组成。",
                "- 一个源事件双线性分配到多个像素时，仍只计为 1 条 ROI 源事件。",
                "- `event_cnt` 是 float32 非负权重图，不是整数计数图。",
                "- depth、TTC、inverse TTC、mask 使用与事件完全相同的 ROI 裁剪。",
                "",
                "## 参数",
                f"- 序列：`{sequence}`",
                f"- ROI：`x=[{args.roi_x0}:{args.roi_x0 + args.roi_size}), y=[{args.roi_y0}:{args.roi_y0 + args.roi_size})`",
                f"- 每步 ROI 源事件数：{args.events_per_step}",
                f"- 步数：{args.num_steps}",
                f"- 起始原始事件索引：{args.start_event_idx}",
                f"- 输出 H5：`{h5_path}`",
                "",
                "## 实际运行命令",
                "```bash",
                "EV-TTC-main/.venv/bin/python EV-TTC-SNN-main/snn_ttc/tools/make_roi_1k_ttc_debug_set.py \\",
                f"  --root {args.root} \\",
                f"  --sequence {sequence} \\",
                f"  --start-event-idx {args.start_event_idx} \\",
                f"  --roi-x0 {args.roi_x0} --roi-y0 {args.roi_y0} --roi-size {args.roi_size} \\",
                f"  --out {h5_path}",
                "```",
                "",
                "## 实际运行结果",
                stdout_summary,
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/data/M3ED"))
    parser.add_argument("--sequence", default="spot_outdoor_day_skatepark_1")
    parser.add_argument("--num-steps", type=int, default=64)
    parser.add_argument("--events-per-step", type=int, default=1000)
    parser.add_argument("--roi-x0", type=int, default=116)
    parser.add_argument("--roi-y0", type=int, default=116)
    parser.add_argument("--roi-size", type=int, default=128)
    parser.add_argument("--start-event-idx", type=int, default=615599491)
    parser.add_argument("--start-time-us", type=int, default=None)
    parser.add_argument("--chunk-size", type=int, default=2_000_000)
    parser.add_argument("--splat", type=int, default=3)
    parser.add_argument("--min-valid-pixels", type=int, default=500)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "debug_sets" / "roi_1k" / "spot_outdoor_day_skatepark_1_roi128_1k64_debug.h5",
    )
    args = parser.parse_args()
    paths = resolve_sequence_paths(args.root, args.sequence)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(paths["data"], "r") as f_data, h5py.File(paths["pose"], "r") as f_pose, h5py.File(paths["depth"], "r") as f_depth:
        geom = load_camera_geometry(f_data)
        if args.start_time_us is not None:
            args.start_event_idx = find_event_index_at_or_after(f_data, args.start_time_us)

        required = args.num_steps * args.events_per_step
        roi_events = collect_roi_events(
            f_data,
            geom,
            args.start_event_idx,
            required,
            args.roi_x0,
            args.roi_y0,
            args.roi_size,
            args.chunk_size,
        )
        step_data = build_roi_steps(
            roi_events,
            args.num_steps,
            args.events_per_step,
            args.roi_x0,
            args.roi_y0,
            args.roi_size,
        )

        labels = compute_labels_for_steps(
            f_depth,
            f_pose["Cn_T_C0"][:].astype(np.float64),
            f_pose["ts"][:].astype(np.int64),
            geom,
            step_data["t_start"],
            step_data["t_end"],
            args.sequence,
            splat=args.splat,
            min_valid_pixels=args.min_valid_pixels,
        )

    y_slice = slice(args.roi_y0, args.roi_y0 + args.roi_size)
    x_slice = slice(args.roi_x0, args.roi_x0 + args.roi_size)
    labels["depth_start"] = labels["depth_start"][:, y_slice, x_slice]
    labels["ttc_start"] = labels["ttc_start"][:, y_slice, x_slice]
    labels["inverse_ttc_start"] = labels["inverse_ttc_start"][:, y_slice, x_slice]
    labels["valid_ttc_mask"] = labels["valid_ttc_mask"][:, y_slice, x_slice]

    with h5py.File(args.out, "w") as f_out:
        f_out.attrs["sequence_name"] = args.sequence
        f_out.attrs["events_per_step"] = args.events_per_step
        f_out.attrs["roi_x0"] = args.roi_x0
        f_out.attrs["roi_y0"] = args.roi_y0
        f_out.attrs["roi_size"] = args.roi_size
        f_out.attrs["start_event_idx"] = args.start_event_idx
        f_out.attrs["source_data_file"] = str(paths["data"])
        f_out.attrs["source_depth_file"] = str(paths["depth"])
        f_out.attrs["source_pose_file"] = str(paths["pose"])
        f_out.create_dataset("sequence_name", data=np.bytes_(args.sequence))
        f_out.create_dataset("step_index", data=np.arange(args.num_steps, dtype=np.int32))
        f_out.create_dataset("roi_x0", data=np.full((args.num_steps,), args.roi_x0, dtype=np.int32))
        f_out.create_dataset("roi_y0", data=np.full((args.num_steps,), args.roi_y0, dtype=np.int32))

        for key, value in step_data.items():
            chunks = (1, *value.shape[1:]) if value.ndim >= 3 else None
            f_out.create_dataset(key, data=value, compression="lzf" if chunks else None, chunks=chunks)
        for key, value in labels.items():
            chunks = (1, *value.shape[1:]) if value.ndim >= 3 else None
            f_out.create_dataset(key, data=value, compression="lzf" if chunks else None, chunks=chunks)
        f_out.create_dataset(
            "roi_event_source_preview_json",
            data=np.bytes_(
                json.dumps(
                    {
                        "first_raw_indices": roi_events["raw_indices"][:10].astype(int).tolist(),
                        "last_raw_indices": roi_events["raw_indices"][-10:].astype(int).tolist(),
                    },
                    indent=2,
                )
            ),
        )

    summary = "\n".join(
        [
            f"- Debug H5：`{args.out}`",
            f"- ROI 源事件：{required}",
            f"- raw index 范围：{int(step_data['raw_event_start_idx'][0])} - {int(step_data['raw_event_end_idx'][-1])}",
            f"- dt 范围：{int(step_data['dt'].min())} - {int(step_data['dt'].max())} us",
            f"- supervise_valid：{int(labels['supervise_valid'].sum())}/{args.num_steps}",
            f"- event_cnt shape：{step_data['event_cnt'].shape}",
            f"- event_cnt dtype：{step_data['event_cnt'].dtype}",
        ]
    )
    write_notes(args.out.parent, args.out, args.sequence, args, summary)
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

