#!/usr/bin/env python3
"""连续 7 ms 时序窗口 Dataset 的合成数据单元测试与真实 H5 烟雾测试。"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_MODEL_ROOT = PROJECT_ROOT / "EV-TTC-main/model/data"
if str(DATA_MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(DATA_MODEL_ROOT))

from ttc_temporal_dm import TTCEFTemporalDataset, make_temporal_dataloader  # noqa: E402


DEFAULT_DATA_ROOT = (
    PROJECT_ROOT
    / "EV-TTC-SNN-main/debug_sets/[13]official_car_single_sequence_evttc_reproduction/data"
)
DEFAULT_OUT_DIR = PROJECT_ROOT / "EV-TTC-SNN-main/reports/05_hybrid_snn_evslim"


def _rotate90(tensor: torch.Tensor) -> torch.Tensor:
    """确定性增强，用于证明 Block 内所有图和标签共享同一个空间变换。"""

    return torch.rot90(tensor, k=1, dims=(-2, -1))


def _rss_mib(include_children: bool = True) -> float:
    """读取当前进程及可见子进程 RSS，不额外依赖 psutil。"""

    page_size = os.sysconf("SC_PAGE_SIZE")

    def process_rss(pid: int) -> int:
        try:
            fields = Path(f"/proc/{pid}/statm").read_text(encoding="ascii").split()
            return int(fields[1]) * page_size
        except (FileNotFoundError, ProcessLookupError, PermissionError, IndexError, ValueError):
            return 0

    total = process_rss(os.getpid())
    if include_children:
        try:
            child_pids = [
                int(pid)
                for children_file in Path("/proc/self/task").glob("*/children")
                for pid in children_file.read_text(encoding="ascii").split()
            ]
        except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError):
            child_pids = []
        total += sum(process_rss(pid) for pid in set(child_pids))
    return total / (1024.0 * 1024.0)


def _write_synthetic_h5(path: Path) -> dict[str, Any]:
    """构造包含三个连续区间和两个明确断点的小型 H5。"""

    source = np.asarray([0, 1, 2, 3, 10, 11, 12, 13, 14, 20, 21, 22, 23, 24], dtype=np.int64)
    exp_time = np.asarray(
        [0, 7000, 14000, 21000, 70000, 77000, 84000, 91000, 98000,
         140000, 147000, 154000, 161000, 168000],
        dtype=np.float64,
    )
    count, channels, height, width = len(source), 6, 8, 8
    base = np.arange(height * width, dtype=np.float32).reshape(height, width)
    exp = np.stack(
        [np.stack([base + row * 1000 + channel * 100 for channel in range(channels)]) for row in range(count)]
    )
    ttc = np.stack([base + row * 10 for row in range(count)]).astype(np.float32)
    mask = np.stack([((base.astype(np.int64) + row) % 3) != 0 for row in range(count)])
    with h5py.File(path, "w") as handle:
        handle.create_dataset("exp_filts", data=exp)
        handle.create_dataset("ttc", data=ttc)
        handle.create_dataset("mask", data=mask)
        handle.create_dataset("source_index", data=source)
        handle.create_dataset("exp_time", data=exp_time)
        handle.attrs["sequence_name"] = "synthetic_sequence"
    return {
        "source_index": source,
        "exp_time": exp_time,
        "exp": exp,
        "ttc": ttc,
        "mask": mask,
    }


def _shutdown_loader(loader: torch.utils.data.DataLoader, iterator: Any) -> None:
    """测试结束后主动关闭 persistent worker，避免残留 HDF5 句柄。"""

    shutdown = getattr(iterator, "_shutdown_workers", None)
    if callable(shutdown):
        shutdown()
    del iterator, loader
    gc.collect()


def _smoke_loader(
    h5_path: Path,
    *,
    split: str,
    num_workers: int,
    batch_count: int,
    shuffle: bool,
) -> dict[str, Any]:
    dataset = TTCEFTemporalDataset(h5_path, window_length=3, window_stride=3, augment=False)
    loader = make_temporal_dataloader(
        dataset,
        batch_size=4,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )
    iterator = iter(loader)
    durations: list[float] = []
    rss_samples: list[float] = []
    mask_counts: list[int] = []
    first_shapes: dict[str, list[int]] | None = None
    finite = True
    continuous = True
    started = time.perf_counter()
    for batch_index in range(batch_count):
        before = time.perf_counter()
        batch = next(iterator)
        durations.append(time.perf_counter() - before)
        rss_samples.append(_rss_mib())
        if first_shapes is None:
            first_shapes = {
                "exp_filts": list(batch["exp_filts"].shape),
                "ttc": list(batch["ttc"].shape),
                "mask": list(batch["mask"].shape),
                "source_indices": list(batch["source_indices"].shape),
                "exp_times": list(batch["exp_times"].shape),
            }
        finite = finite and bool(torch.isfinite(batch["exp_filts"]).all())
        finite = finite and bool(torch.isfinite(batch["ttc"]).all())
        source_delta = torch.diff(batch["source_indices"], dim=1)
        time_delta = torch.diff(batch["exp_times"], dim=1)
        continuous = continuous and bool((source_delta == 1).all())
        continuous = continuous and bool((time_delta > 0).all())
        continuous = continuous and bool(((time_delta - 7000.0).abs() <= 500.0).all())
        mask_counts.extend(int(value) for value in batch["mask"].sum(dim=(1, 2, 3)).tolist())
        print(
            f"[{split}/workers={num_workers}] batch {batch_index + 1}/{batch_count} "
            f"read={durations[-1]:.3f}s rss={rss_samples[-1]:.1f}MiB",
            flush=True,
        )
    total_seconds = time.perf_counter() - started
    # 预取稳定后比较中段和末段 RSS；允许缓存与最后一个预取 batch 带来少量波动。
    half = max(1, len(rss_samples) // 2)
    early_stable_rss = float(np.mean(rss_samples[half // 2 : half]))
    late_stable_rss = float(np.mean(rss_samples[-half:]))
    result = {
        "split": split,
        "num_workers": num_workers,
        "requested_batches": batch_count,
        "completed_batches": len(durations),
        "dataset": dataset.summary(),
        "first_batch_shapes": first_shapes,
        "total_seconds": total_seconds,
        "average_batch_read_seconds": float(np.mean(durations)),
        "median_batch_read_seconds": float(np.median(durations)),
        "max_batch_read_seconds": float(np.max(durations)),
        "finite": finite,
        "strictly_continuous": continuous,
        "mask_valid_pixels_min": min(mask_counts),
        "mask_valid_pixels_max": max(mask_counts),
        "rss_mib_min": min(rss_samples),
        "rss_mib_max": max(rss_samples),
        "stable_rss_growth_mib": late_stable_rss - early_stable_rss,
        "memory_stable": (late_stable_rss - early_stable_rss) < 256.0,
    }
    _shutdown_loader(loader, iterator)
    dataset.close()
    return result


def _render_report(payload: dict[str, Any]) -> str:
    checks = payload["checks"]
    lines = [
        "# 连续时序窗口 Dataset 测试报告",
        "",
        f"- 总体结果：{'通过' if payload['passed'] else '失败'}",
        "- 默认设置：`T=3`、`stride=3`、`batch_size=4`、`pin_memory=True`。",
        "- 本测试只读取数据，不执行网络前向、loss、反向传播或参数更新。",
        "- 执行命令：`EV-TTC-main/.venv/bin/python EV-TTC-SNN-main/snn_ttc/tests/test_ttc_temporal_dataset.py`。",
        "",
        "## 连续区间与窗口",
        "",
        "| Split | H5样本 | 最大连续区间 | 断点 | stride=3窗口 | stride=1窗口 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for split in ("train", "val"):
        row = payload["real_dataset_summary"][split]
        lines.append(
            f"| {split} | {row['sample_count']} | {row['contiguous_run_count']} | "
            f"{row['break_count']} | {row['stride3_window_count']} | {row['stride1_window_count']} |"
        )
    lines.extend(["", "构造规则：先按 `source_index + 1`、正向 `exp_time`、`7000±500 us` 和同序列切成最大连续区间，再在每个区间内部按 stride 生成窗口；标签和 mask 只读取窗口最后一行。", "", "## 单元测试", ""])
    for check in checks:
        lines.append(f"- [{'通过' if check['passed'] else '失败'}] {check['name']}：{check['details']}")
    lines.extend(
        [
            "",
            "## DataLoader 烟雾测试",
            "",
            "| Split | workers | Batch数 | 首个输入shape | 总耗时(s) | 平均读取(s/batch) | mask有效像素范围 | 连续/有限 | RSS稳定增长(MiB) |",
            "|---|---:|---:|---|---:|---:|---|---|---:|",
        ]
    )
    for row in payload["smoke_tests"]:
        lines.append(
            f"| {row['split']} | {row['num_workers']} | {row['completed_batches']} | "
            f"`{row['first_batch_shapes']['exp_filts']}` | {row['total_seconds']:.3f} | "
            f"{row['average_batch_read_seconds']:.3f} | {row['mask_valid_pixels_min']}–{row['mask_valid_pixels_max']} | "
            f"{'是' if row['strictly_continuous'] and row['finite'] else '否'} | "
            f"{row['stable_rss_growth_mib']:.1f} |"
        )
    lines.extend(
        [
            "",
            "## 结论",
            "",
            f"- `num_workers=0/2`：{'均正常' if payload['workers_0_2_ok'] else '存在异常'}。",
            f"- `num_workers=4`：{'正常' if payload['workers_4_ok'] else '存在异常'}。",
            f"- 是否发现窗口跨断点：{'否' if payload['no_window_crosses_break'] else '是'}。",
            f"- 是否发现标签或 mask 错位：{'否' if payload['last_label_aligned'] else '是'}。",
            f"- 实际 batch：`exp_filts={payload['actual_batch_shapes']['exp_filts']}`，`ttc={payload['actual_batch_shapes']['ttc']}`，`mask={payload['actual_batch_shapes']['mask']}`。",
            "- 该 Dataset 可直接向 Hybrid SNN-EV-Slim 提供严格连续的 `[B,T,6,H,W]` 输入；本阶段未启动正式训练。",
            "",
        ]
    )
    return "\n".join(lines)


def run_tests(train_file: Path, val_file: Path, out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, details: str) -> None:
        row = {"name": name, "passed": bool(passed), "details": details}
        checks.append(row)
        print(f"[{'PASS' if passed else 'FAIL'}] {name}: {details}", flush=True)

    with tempfile.TemporaryDirectory(prefix="ttcef_temporal_dataset_") as temp_dir:
        synthetic_path = Path(temp_dir) / "synthetic.h5"
        raw = _write_synthetic_h5(synthetic_path)
        stride3 = TTCEFTemporalDataset(synthetic_path, window_length=3, window_stride=3)
        stride1 = TTCEFTemporalDataset(synthetic_path, window_length=3, window_stride=1)
        check("stride=3窗口起点", stride3.window_starts.tolist() == [0, 4, 9], str(stride3.window_starts.tolist()))
        check("stride=1窗口起点", stride1.window_starts.tolist() == [0, 1, 4, 5, 6, 9, 10, 11], str(stride1.window_starts.tolist()))
        sample = stride3[1]
        check("合成单样本shape", tuple(sample["exp_filts"].shape) == (3, 6, 8, 8) and tuple(sample["ttc"].shape) == (1, 8, 8) and tuple(sample["mask"].shape) == (1, 8, 8), f"exp={tuple(sample['exp_filts'].shape)}, ttc={tuple(sample['ttc'].shape)}, mask={tuple(sample['mask'].shape)}")
        check("合成窗口严格连续", np.array_equal(np.diff(sample["source_indices"].numpy()), np.ones(2)) and np.all(np.abs(np.diff(sample["exp_times"].numpy()) - 7000) <= 500), f"source={sample['source_indices'].tolist()}, time={sample['exp_times'].tolist()}")
        check("合成标签取最后一步", np.array_equal(sample["ttc"].numpy()[0], raw["ttc"][6]) and np.array_equal(sample["mask"].numpy()[0], raw["mask"][6]), "window [4,5,6] 对应 H5 第6行标签")

        augmented = TTCEFTemporalDataset(
            synthetic_path,
            window_length=3,
            window_stride=3,
            augment=True,
            augmentation_transform=_rotate90,
        )[0]
        expected_exp = torch.rot90(torch.from_numpy(raw["exp"][0:3]), 1, (-2, -1))
        expected_ttc = torch.rot90(torch.from_numpy(raw["ttc"][2])[None], 1, (-2, -1))
        expected_mask = torch.rot90(torch.from_numpy(raw["mask"][2])[None], 1, (-2, -1))
        check("Block增强参数一致", torch.equal(augmented["exp_filts"], expected_exp) and torch.equal(augmented["ttc"], expected_ttc) and torch.equal(augmented["mask"], expected_mask), "T步输入、最后一步TTC与mask均执行同一次90度旋转")

        synthetic_loader = make_temporal_dataloader(stride1, batch_size=4, num_workers=0)
        synthetic_batch = next(iter(synthetic_loader))
        check("合成batch shape", tuple(synthetic_batch["exp_filts"].shape) == (4, 3, 6, 8, 8) and tuple(synthetic_batch["ttc"].shape) == (4, 1, 8, 8), f"exp={tuple(synthetic_batch['exp_filts'].shape)}, ttc={tuple(synthetic_batch['ttc'].shape)}")
        stride3.close()
        stride1.close()

    real_summary: dict[str, Any] = {}
    no_window_crosses_break = True
    last_label_aligned = True
    for split, path in (("train", train_file), ("val", val_file)):
        dataset3 = TTCEFTemporalDataset(path, window_length=3, window_stride=3)
        dataset1 = TTCEFTemporalDataset(path, window_length=3, window_stride=1)
        real_summary[split] = {
            **dataset3.summary(),
            "stride3_window_count": len(dataset3),
            "stride1_window_count": len(dataset1),
        }
        starts = dataset1.window_starts
        # 任一窗口若覆盖断点边，则起点 <= break < 窗口末行。
        crosses = any(bool(np.any((starts <= point) & (starts + 2 > point))) for point in dataset1.break_after_rows)
        no_window_crosses_break = no_window_crosses_break and not crosses
        random_generator = np.random.default_rng(20260722)
        chosen = random_generator.choice(len(dataset1), size=min(1000, len(dataset1)), replace=False)
        metadata_ok = True
        for window_index in chosen:
            start = int(dataset1.window_starts[window_index])
            source_delta = np.diff(dataset1.source_indices[start : start + 3])
            time_delta = np.diff(dataset1.exp_times[start : start + 3])
            metadata_ok = metadata_ok and bool(np.all(source_delta == 1))
            metadata_ok = metadata_ok and bool(np.all(np.abs(time_delta - 7000) <= 500))
        check(f"{split}随机1000窗口严格连续", metadata_ok, f"windows={len(dataset1)}, breaks={len(dataset1.break_after_rows)}")
        check(f"{split}窗口不跨断点", not crosses, f"break_rows={dataset1.break_after_rows.tolist()}")

        actual = dataset3[len(dataset3) // 2]
        end_row = int(actual["end_row"])
        with h5py.File(path, "r") as handle:
            expected_ttc = np.asarray(handle["ttc"][end_row], dtype=np.float32)
            expected_mask = np.asarray(handle["mask"][end_row], dtype=bool)
        aligned = np.array_equal(actual["ttc"].numpy()[0], expected_ttc) and np.array_equal(actual["mask"].numpy()[0], expected_mask)
        last_label_aligned = last_label_aligned and aligned
        check(f"{split}标签对应窗口最后一步", aligned, f"end_row={end_row}")
        expected_shape = (3, 6, 360, 360)
        check(f"{split}真实单样本shape", tuple(actual["exp_filts"].shape) == expected_shape and tuple(actual["ttc"].shape) == (1, 360, 360) and tuple(actual["mask"].shape) == (1, 360, 360), f"exp={tuple(actual['exp_filts'].shape)}, ttc={tuple(actual['ttc'].shape)}, mask={tuple(actual['mask'].shape)}")
        dataset3.close()
        dataset1.close()

    smoke_tests = [
        _smoke_loader(train_file, split="train", num_workers=0, batch_count=20, shuffle=True),
        _smoke_loader(train_file, split="train", num_workers=2, batch_count=20, shuffle=True),
        _smoke_loader(val_file, split="val", num_workers=2, batch_count=10, shuffle=False),
        _smoke_loader(val_file, split="val", num_workers=4, batch_count=4, shuffle=False),
    ]
    for row in smoke_tests:
        ok = (
            row["completed_batches"] == row["requested_batches"]
            and row["finite"]
            and row["strictly_continuous"]
            and row["memory_stable"]
        )
        check(f"{row['split']} workers={row['num_workers']}烟雾测试", ok, f"batches={row['completed_batches']}, finite={row['finite']}, continuous={row['strictly_continuous']}, memory_stable={row['memory_stable']}")

    workers_0_2_ok = all(
        row["finite"] and row["strictly_continuous"] and row["memory_stable"]
        for row in smoke_tests
        if row["num_workers"] in (0, 2)
    )
    workers_4_ok = all(
        row["finite"] and row["strictly_continuous"] and row["memory_stable"]
        for row in smoke_tests
        if row["num_workers"] == 4
    )
    actual_batch_shapes = smoke_tests[1]["first_batch_shapes"]
    passed = all(item["passed"] for item in checks)
    payload: dict[str, Any] = {
        "passed": passed,
        "train_file": str(train_file),
        "val_file": str(val_file),
        "configuration": {
            "window_length": 3,
            "window_stride": 3,
            "batch_size": 4,
            "delta_t_us": 7000,
            "delta_t_tolerance_us": 500,
            "pin_memory": True,
        },
        "real_dataset_summary": real_summary,
        "checks": checks,
        "smoke_tests": smoke_tests,
        "workers_0_2_ok": workers_0_2_ok,
        "workers_4_ok": workers_4_ok,
        "no_window_crosses_break": no_window_crosses_break,
        "last_label_aligned": last_label_aligned,
        "actual_batch_shapes": actual_batch_shapes,
    }
    (out_dir / "temporal_dataset_test.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (out_dir / "TEMPORAL_DATASET_TEST.md").write_text(
        _render_report(payload), encoding="utf-8"
    )
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-file", type=Path, default=DEFAULT_DATA_ROOT / "train.h5")
    parser.add_argument("--val-file", type=Path, default=DEFAULT_DATA_ROOT / "val.h5")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = run_tests(args.train_file.resolve(), args.val_file.resolve(), args.out_dir.resolve())
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
