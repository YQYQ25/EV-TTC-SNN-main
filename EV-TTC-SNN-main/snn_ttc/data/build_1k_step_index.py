#!/usr/bin/env python3
"""从 M3ED 原始事件文件中构建严格连续的 1k-event step 索引。"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import h5py
import numpy as np


EVENTS_PER_STEP = 1000


@dataclass(frozen=True)
class StepRecord:
    sequence_name: str
    step_index: int
    event_start_idx: int
    event_end_idx: int
    t_start: int
    t_end: int
    dt: int
    is_sequence_start: bool


def _event_group(f_data: h5py.File) -> h5py.Group:
    return f_data["prophesee"]["left"]


def find_event_index_at_or_after(f_data: h5py.File, timestamp_us: int) -> int:
    """找到第一个时间戳不小于 timestamp_us 的事件索引。

    M3ED 的事件量很大，这里优先使用 ms_map_idx 做毫秒级粗定位，
    避免把数十亿级时间戳完整读入内存。
    """

    events = _event_group(f_data)
    ts = events["t"]
    if timestamp_us <= int(ts[0]):
        return 0
    if timestamp_us > int(ts[-1]):
        raise ValueError(f"timestamp {timestamp_us} us is after last event {int(ts[-1])} us")

    ms_map = events.get("ms_map_idx")
    if ms_map is None:
        lo, hi = 0, int(ts.shape[0])
        while lo < hi:
            mid = (lo + hi) // 2
            if int(ts[mid]) < timestamp_us:
                lo = mid + 1
            else:
                hi = mid
        return lo

    ms = int(timestamp_us // 1000)
    lo_ms = max(0, ms - 2)
    hi_ms = min(int(ms_map.shape[0]) - 1, ms + 3)
    lo = int(ms_map[lo_ms])
    hi = int(ms_map[hi_ms])
    if hi <= lo:
        hi = min(int(ts.shape[0]), lo + 1_000_000)

    local_ts = ts[lo:hi]
    offset = int(np.searchsorted(local_ts, timestamp_us, side="left"))
    idx = lo + offset

    # ms_map_idx 通常已经足够准确；这里保留小范围修正，防止边界误差。
    while idx > 0 and int(ts[idx - 1]) >= timestamp_us:
        idx -= 1
    n_events = int(ts.shape[0])
    while idx < n_events and int(ts[idx]) < timestamp_us:
        idx += 1
    if idx >= n_events:
        raise ValueError(f"failed to locate event at or after {timestamp_us} us")
    return idx


def build_1k_step_index(
    f_data: h5py.File,
    sequence_name: str,
    num_steps: int = 64,
    events_per_step: int = EVENTS_PER_STEP,
    start_event_idx: int = 0,
) -> List[StepRecord]:
    """按固定事件数切分连续 step，并在生成时检查索引和时间连续性。"""

    if events_per_step <= 0:
        raise ValueError("events_per_step must be positive")
    if num_steps <= 0:
        raise ValueError("num_steps must be positive")

    ts = _event_group(f_data)["t"]
    total_needed = start_event_idx + num_steps * events_per_step
    if total_needed > int(ts.shape[0]):
        raise ValueError(
            f"not enough events: need end index {total_needed}, file has {int(ts.shape[0])}"
        )

    records: List[StepRecord] = []
    last_end: Optional[int] = None
    last_t_end: Optional[int] = None

    for step_idx in range(num_steps):
        # 按事件索引硬切片：第 n 步就是连续的 [start, start+1000)。
        event_start_idx = start_event_idx + step_idx * events_per_step
        event_end_idx = event_start_idx + events_per_step
        assert event_end_idx - event_start_idx == events_per_step
        if last_end is not None:
            assert event_start_idx == last_end

        t_start = int(ts[event_start_idx])
        t_end = int(ts[event_end_idx - 1])
        if last_t_end is not None:
            assert t_start >= last_t_end
        assert t_end >= t_start

        records.append(
            StepRecord(
                sequence_name=sequence_name,
                step_index=step_idx,
                event_start_idx=event_start_idx,
                event_end_idx=event_end_idx,
                t_start=t_start,
                t_end=t_end,
                dt=t_end - t_start,
                is_sequence_start=step_idx == 0,
            )
        )
        last_end = event_end_idx
        last_t_end = t_end

    return records


def records_to_arrays(records: Iterable[StepRecord]) -> dict:
    """把 StepRecord 列表转成便于写入 H5 的 numpy 数组。"""

    records = list(records)
    return {
        "event_start_idx": np.array([r.event_start_idx for r in records], dtype=np.int64),
        "event_end_idx": np.array([r.event_end_idx for r in records], dtype=np.int64),
        "t_start": np.array([r.t_start for r in records], dtype=np.int64),
        "t_end": np.array([r.t_end for r in records], dtype=np.int64),
        "dt": np.array([r.dt for r in records], dtype=np.int64),
        "is_sequence_start": np.array([r.is_sequence_start for r in records], dtype=bool),
        "step_index": np.array([r.step_index for r in records], dtype=np.int32),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-file", required=True, type=Path)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--num-steps", type=int, default=64)
    parser.add_argument("--events-per-step", type=int, default=EVENTS_PER_STEP)
    parser.add_argument("--start-event-idx", type=int, default=0)
    parser.add_argument("--out-json", type=Path)
    args = parser.parse_args()

    with h5py.File(args.data_file, "r") as f_data:
        records = build_1k_step_index(
            f_data,
            args.sequence,
            num_steps=args.num_steps,
            events_per_step=args.events_per_step,
            start_event_idx=args.start_event_idx,
        )

    payload = [asdict(r) for r in records]
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    else:
        print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
