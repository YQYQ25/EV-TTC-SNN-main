#!/usr/bin/env python3
"""生成 64 步 M3ED 1k-event TTC debug H5；本脚本不启动训练。"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Tuple

import h5py
import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from snn_ttc.data.build_1k_step_index import (  # noqa: E402
    EVENTS_PER_STEP,
    build_1k_step_index,
    find_event_index_at_or_after,
    records_to_arrays,
)
from snn_ttc.data.m3ed_geometry import (  # noqa: E402
    accumulate_event_count,
    compute_labels_for_steps,
    load_camera_geometry,
)


def resolve_sequence_paths(root: Path, sequence: str) -> Dict[str, Path]:
    """在常见数据目录布局中查找 data/depth_gt/pose_gt 三个必要文件。"""

    candidates = [
        root / "official_train" / sequence,
        root / "official_test" / sequence,
        root / "paper_alignment" / sequence,
        root / "task_oriented" / sequence,
        root / sequence,
    ]
    for seq_dir in candidates:
        data_file = seq_dir / f"{sequence}_data.h5"
        depth_file = seq_dir / f"{sequence}_depth_gt.h5"
        pose_file = seq_dir / f"{sequence}_pose_gt.h5"
        if data_file.exists() and depth_file.exists() and pose_file.exists():
            return {"sequence_dir": seq_dir, "data": data_file, "depth": depth_file, "pose": pose_file}

    hits = list(root.rglob(f"{sequence}_data.h5"))
    for data_file in hits:
        seq_dir = data_file.parent
        depth_file = seq_dir / f"{sequence}_depth_gt.h5"
        pose_file = seq_dir / f"{sequence}_pose_gt.h5"
        if depth_file.exists() and pose_file.exists():
            return {"sequence_dir": seq_dir, "data": data_file, "depth": depth_file, "pose": pose_file}
    raise FileNotFoundError(f"cannot find complete data/depth_gt/pose_gt for {sequence} under {root}")


def _write_string_dataset(f_out: h5py.File, name: str, value: str) -> None:
    f_out.create_dataset(name, data=np.bytes_(value))


def write_implementation_notes(
    out_dir: Path,
    h5_path: Path,
    sequence: str,
    paths: Dict[str, Path],
    start_event_idx: int,
    start_policy: str,
) -> Path:
    """记录本次 debug set 的来源、复用逻辑和实际运行结果。"""

    notes = out_dir / "implementation_notes.md"
    notes.write_text(
        "\n".join(
            [
                "# M3ED 1k 事件 TTC 调试集实现说明",
                "",
                "## 本次生成",
                f"- 序列：`{sequence}`",
                f"- Debug H5：`{h5_path}`",
                f"- 原始 data：`{paths['data']}`",
                f"- depth_gt：`{paths['depth']}`",
                f"- pose_gt：`{paths['pose']}`",
                f"- 起始策略：`{start_policy}`",
                f"- 起始事件索引：`{start_event_idx}`",
                "- 步数：64",
                "- 每步事件数：1000",
                "- 空间尺寸：360x360",
                "",
                "## 复用 EV-TTC 的部分",
                "- 相机内参、畸变参数读取逻辑来自 `EV-TTC-main/TTCEF/calc_gt.py`。",
                "- 去畸变后中心裁剪 `[x=280:1000, y=0:720]` 并下采样到 `360x360`，与 `create_exp.py/calc_gt.py` 对齐。",
                "- 位姿插值、相对位姿、`T`、`Omega` 的计算口径与 `calc_gt.py` 对齐。",
                "- 深度重投影到事件相机起始时刻，TTC 使用 `Z / (Tz + 1e-5)`。",
                "",
                "## 本阶段新增或调整",
                "- 输入不再使用 7ms exponential filter，而是严格连续的 1k 原始事件分片。",
                "- `event_cnt` 保存为正负极性两通道计数图，正负不相减。",
                "- `valid_ttc_mask` 只表示像素级深度/TTC 标签是否有限可靠。",
                "- `speed_valid`、`omega_valid`、`supervise_valid` 是时间步级筛选标志，不会修改 `valid_ttc_mask`。",
                "- M3ED 事件时间从 0 开始，但 pose/depth 第一帧通常晚于 0；默认 `first_valid_label` 会跳到第一段可生成标签的事件，仍保持后续 64 步严格连续。",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return notes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/data/M3ED"))
    parser.add_argument("--sequence", default="spot_outdoor_day_skatepark_1")
    parser.add_argument("--num-steps", type=int, default=64)
    parser.add_argument("--events-per-step", type=int, default=EVENTS_PER_STEP)
    parser.add_argument("--start-event-idx", type=int, default=None)
    parser.add_argument(
        "--start-policy",
        choices=["first_valid_label", "zero"],
        default="first_valid_label",
        help="first_valid_label skips early events before pose/depth labels exist; zero starts at event 0.",
    )
    parser.add_argument("--splat", type=int, default=3)
    parser.add_argument("--min-valid-pixels", type=int, default=500)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "debug_sets" / "spot_outdoor_day_skatepark_1_1k64_debug.h5",
    )
    args = parser.parse_args()

    paths = resolve_sequence_paths(args.root, args.sequence)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(paths["data"], "r") as f_data, h5py.File(paths["pose"], "r") as f_pose, h5py.File(paths["depth"], "r") as f_depth:
        pose_ts = f_pose["ts"][:].astype(np.int64)
        pose_mats = f_pose["Cn_T_C0"][:].astype(np.float64)
        depth_ts = f_depth["ts"][:].astype(np.int64)

        if args.start_event_idx is not None:
            start_event_idx = int(args.start_event_idx)
            start_policy = "manual"
        elif args.start_policy == "zero":
            start_event_idx = 0
            start_policy = "zero"
        else:
            # 原始事件通常从 0 开始，而 depth/pose 标签更晚才可用；默认跳到首个可标注时间。
            first_label_time = int(max(pose_ts[1], depth_ts[0]))
            start_event_idx = find_event_index_at_or_after(f_data, first_label_time)
            start_policy = "first_valid_label"

        records = build_1k_step_index(
            f_data,
            args.sequence,
            num_steps=args.num_steps,
            events_per_step=args.events_per_step,
            start_event_idx=start_event_idx,
        )
        step_arrays = records_to_arrays(records)

        geom = load_camera_geometry(f_data)
        event_cnt = np.zeros((args.num_steps, 2, 360, 360), dtype=np.float32)
        events = f_data["prophesee"]["left"]
        for i, rec in enumerate(tqdm(records, desc="building event_cnt")):
            # 每个 step 只读对应的 1000 个事件，降低大 H5 随机访问开销。
            sl = slice(rec.event_start_idx, rec.event_end_idx)
            event_cnt[i] = accumulate_event_count(
                events["x"][sl], events["y"][sl], events["p"][sl], geom.dist_map
            )

        # 标签使用 step 起点深度和 step 起止位姿，符合当前指导文件的“分片起点标签”定义。
        labels = compute_labels_for_steps(
            f_depth,
            pose_mats,
            pose_ts,
            geom,
            step_arrays["t_start"],
            step_arrays["t_end"],
            args.sequence,
            splat=args.splat,
            min_valid_pixels=args.min_valid_pixels,
        )

    with h5py.File(args.out, "w") as f_out:
        f_out.attrs["sequence_name"] = args.sequence
        f_out.attrs["events_per_step"] = args.events_per_step
        f_out.attrs["start_event_idx"] = start_event_idx
        f_out.attrs["start_policy"] = start_policy
        f_out.attrs["source_data_file"] = str(paths["data"])
        f_out.attrs["source_depth_file"] = str(paths["depth"])
        f_out.attrs["source_pose_file"] = str(paths["pose"])
        _write_string_dataset(f_out, "sequence_name", args.sequence)

        for key, value in step_arrays.items():
            f_out.create_dataset(key, data=value)
        f_out.create_dataset("event_cnt", data=event_cnt, compression="lzf", chunks=(1, 2, 360, 360))
        for key, value in labels.items():
            if value.ndim >= 3:
                chunks = (1, *value.shape[1:])
                f_out.create_dataset(key, data=value, compression="lzf", chunks=chunks)
            else:
                f_out.create_dataset(key, data=value)

        f_out.create_dataset("source_records_json", data=np.bytes_(json.dumps([asdict(r) for r in records], indent=2)))

    notes_path = write_implementation_notes(args.out.parent, args.out, args.sequence, paths, start_event_idx, start_policy)
    print(f"debug_h5: {args.out}")
    print(f"implementation_notes: {notes_path}")
    print(f"supervise_valid: {int(labels['supervise_valid'].sum())}/{len(labels['supervise_valid'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
