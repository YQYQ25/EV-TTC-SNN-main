#!/usr/bin/env python3
"""Audit TTCEF temporal continuity and build metadata-complete ordered H5 views.

The legacy merged H5 files intentionally store samples only.  This tool never
uses their row number as temporal metadata: it reconstructs the merge predicate
from the source TTCEF file, verifies selected sample content, and writes a small
HDF5 virtual dataset with sequence, source-frame, timestamp, and segment IDs.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import h5py
import hdf5plugin  # noqa: F401 - registers Blosc compression used by source H5 files.
import numpy as np


NOMINAL_READOUT_US = 7000.0
TIME_TOLERANCE_US = 500.0
CLIP_LENGTHS = (8, 16, 32)
DATASETS_TO_VIRTUALIZE = ("exp_filts", "ttc", "flow", "mask")


@dataclass
class SplitAudit:
    label: str
    merged_path: Path
    sequence: str
    source_indices: np.ndarray
    timestamps: np.ndarray
    continuous_from_previous: np.ndarray
    segment_ids: np.ndarray
    segments: list[tuple[int, int]]
    metadata_present: dict[str, bool]
    metadata_reconstructed: bool
    legacy_file_names: list[str]
    legacy_indices: list[list[int]]
    index_jump_count: int
    timestamp_backtrack_count: int
    timestamp_large_gap_count: int
    invalid_source_count: int
    empty_mask_sample_count: int
    valid_mask_pixel_summary: dict[str, float]
    mapping_checks: list[dict[str, Any]]


def decode_name(value: Any) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def source_keep_mask(gt: h5py.File, sequence: str) -> np.ndarray:
    valid = gt["valid"][:].astype(bool)
    translation = gt["T"][:].astype(np.float32)
    rotation = gt["Omega"][:].astype(np.float32)
    translation_threshold = 1.3 if "car" in sequence else 0.25
    return valid & (np.linalg.norm(translation, axis=1) > translation_threshold) & (
        np.linalg.norm(rotation, axis=1) < 0.18
    )


def source_paths(data_root: Path, sequence: str) -> tuple[Path, Path]:
    return (
        data_root / "ttcef" / "m3ed" / f"{sequence}.h5",
        data_root / "exp_filts" / "m3ed" / f"{sequence}.h5",
    )


def validate_reconstructed_mapping(
    merged: h5py.File,
    gt: h5py.File,
    source_indices: np.ndarray,
) -> list[dict[str, Any]]:
    """Sample the content mapping so source-order reconstruction is evidence based."""
    sample_rows = np.unique(np.linspace(0, len(source_indices) - 1, num=min(9, len(source_indices)), dtype=int))
    checks: list[dict[str, Any]] = []
    for row in sample_rows:
        expected = np.nan_to_num(gt["ttc"][int(source_indices[row])], nan=0.0)
        actual = merged["ttc"][int(row)]
        checks.append(
            {
                "h5_sample_index": int(row),
                "source_exp_index": int(source_indices[row]),
                "ttc_exact_match": bool(np.array_equal(actual, expected)),
                "ttc_max_abs_difference": float(np.max(np.abs(actual - expected))),
            }
        )
    if not all(check["ttc_exact_match"] for check in checks):
        raise RuntimeError("source-index reconstruction failed content validation; refusing temporal inference")
    return checks


def audit_split(label: str, merged_path: Path, data_root: Path) -> SplitAudit:
    with h5py.File(merged_path, "r") as merged:
        required = {"exp_filts", "ttc", "flow", "mask", "file_names", "indices"}
        missing = required - set(merged.keys())
        if missing:
            raise KeyError(f"{merged_path} is missing datasets: {sorted(missing)}")
        names = [decode_name(value) for value in merged["file_names"][:]]
        if len(names) != 1:
            raise RuntimeError(f"{merged_path} has {len(names)} sequences; this audit requires explicit per-sample mapping")
        sequence = names[0]
        n_samples = len(merged["exp_filts"])
        metadata_present = {
            "sequence_name": "sequence_name" in merged,
            "source_exp_index": "source_exp_index" in merged or "source_indices" in merged,
            "timestamp": "timestamp" in merged or "readout_time" in merged,
        }
        legacy_indices = merged["indices"][:].astype(int).tolist()
        mask_pixel_counts = []
        for start in range(0, n_samples, 64):
            batch = merged["mask"][start : min(start + 64, n_samples)]
            mask_pixel_counts.append(np.count_nonzero(batch, axis=(1, 2)))
        mask_pixel_counts = np.concatenate(mask_pixel_counts)

        gt_path, exp_path = source_paths(data_root, sequence)
        if not gt_path.is_file() or not exp_path.is_file():
            raise FileNotFoundError(f"cannot reconstruct {sequence}: {gt_path} / {exp_path}")
        with h5py.File(gt_path, "r") as gt, h5py.File(exp_path, "r") as exp:
            keep = source_keep_mask(gt, sequence)
            source_indices = np.flatnonzero(keep).astype(np.int64)
            if len(source_indices) != n_samples:
                raise RuntimeError(
                    f"{label}: reconstructed {len(source_indices)} source rows but merged H5 has {n_samples} samples"
                )
            timestamps = gt["te"][source_indices].astype(np.float64)
            exp_times = exp["exp_times"][source_indices].astype(np.float64)
            if not np.array_equal(timestamps, exp_times):
                raise RuntimeError(f"{label}: TTCEF te and exp_filts exp_times disagree")
            mapping_checks = validate_reconstructed_mapping(merged, gt, source_indices)
            source_valid = gt["valid"][source_indices].astype(bool)

    index_deltas = np.diff(source_indices)
    time_deltas = np.diff(timestamps)
    index_jump = index_deltas != 1
    timestamp_backtrack = time_deltas <= 0
    timestamp_large_gap = np.abs(time_deltas - NOMINAL_READOUT_US) > TIME_TOLERANCE_US
    invalid_source = ~source_valid
    empty_mask = mask_pixel_counts == 0
    unusable_sample = invalid_source | empty_mask | ~np.isfinite(timestamps)
    discontinuity = np.zeros(n_samples, dtype=bool)
    discontinuity[0] = True
    if n_samples > 1:
        discontinuity[1:] = index_jump | timestamp_backtrack | timestamp_large_gap | unusable_sample[1:]

    starts = np.flatnonzero(discontinuity)
    ends = np.r_[starts[1:], n_samples]
    segments = [(int(start), int(end)) for start, end in zip(starts, ends)]
    segment_ids = np.empty(n_samples, dtype=np.int32)
    for segment_id, (start, end) in enumerate(segments):
        segment_ids[start:end] = segment_id
    continuous_from_previous = ~discontinuity
    continuous_from_previous[0] = False

    return SplitAudit(
        label=label,
        merged_path=merged_path,
        sequence=sequence,
        source_indices=source_indices,
        timestamps=timestamps,
        continuous_from_previous=continuous_from_previous,
        segment_ids=segment_ids,
        segments=segments,
        metadata_present=metadata_present,
        metadata_reconstructed=True,
        legacy_file_names=names,
        legacy_indices=legacy_indices,
        index_jump_count=int(index_jump.sum()),
        timestamp_backtrack_count=int(timestamp_backtrack.sum()),
        timestamp_large_gap_count=int(timestamp_large_gap.sum()),
        invalid_source_count=int(invalid_source.sum()),
        empty_mask_sample_count=int(empty_mask.sum()),
        valid_mask_pixel_summary={
            "min": float(mask_pixel_counts.min()),
            "median": float(np.median(mask_pixel_counts)),
            "max": float(mask_pixel_counts.max()),
        },
        mapping_checks=mapping_checks,
    )


def clip_record(audit: SplitAudit, start: int, length: int) -> dict[str, Any]:
    end = start + length
    return {
        "sequence_name": audit.sequence,
        "h5_sample_indices": list(range(start, end)),
        "source_frame_indices": audit.source_indices[start:end].astype(int).tolist(),
        "timestamps": audit.timestamps[start:end].astype(float).tolist(),
        "clip_length": length,
        "segment_id": int(audit.segment_ids[start]),
    }


def segment_record(audit: SplitAudit, start: int, end: int) -> dict[str, Any]:
    return {
        "sequence_name": audit.sequence,
        "h5_sample_indices": list(range(start, end)),
        "source_frame_indices": audit.source_indices[start:end].astype(int).tolist(),
        "timestamps": audit.timestamps[start:end].astype(float).tolist(),
        "clip_length": end - start,
        "segment_id": int(audit.segment_ids[start]),
    }


def build_clip_payload(audit: SplitAudit) -> dict[str, Any]:
    clips_by_length: dict[str, list[dict[str, Any]]] = {}
    clip_counts: dict[str, dict[str, int]] = {}
    for length in CLIP_LENGTHS:
        non_overlapping: list[dict[str, Any]] = []
        sliding_count = 0
        for start, end in audit.segments:
            segment_length = end - start
            sliding_count += max(0, segment_length - length + 1)
            for clip_start in range(start, end - length + 1, length):
                non_overlapping.append(clip_record(audit, clip_start, length))
        clips_by_length[str(length)] = non_overlapping
        clip_counts[str(length)] = {
            "non_overlapping": len(non_overlapping),
            "sliding_windows": sliding_count,
        }
    return {
        "schema_version": 1,
        "split": audit.label,
        "source_merged_h5": str(audit.merged_path),
        "sequence_name": audit.sequence,
        "timestamp_unit": "microseconds",
        "clip_policy": "non-overlapping clips are stored; segment boundaries reset prior-risk state",
        "maximal_continuous_segments": [segment_record(audit, start, end) for start, end in audit.segments],
        "clips_by_length": clips_by_length,
        "clip_counts": clip_counts,
    }


def write_ordered_virtual_h5(audit: SplitAudit, output_path: Path) -> None:
    """Create a metadata-complete VDS without duplicating compressed model inputs."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(output_path.name + ".tmp")
    if temporary_path.exists():
        temporary_path.unlink()
    n_samples = len(audit.source_indices)
    with h5py.File(audit.merged_path, "r") as source, h5py.File(temporary_path, "w", libver="latest") as target:
        target.attrs["format"] = "TTCEF ordered temporal view"
        target.attrs["source_merged_h5"] = str(audit.merged_path)
        target.attrs["sequence_name"] = audit.sequence
        target.attrs["timestamp_unit"] = "microseconds"
        target.attrs["source_index_reconstruction"] = "TTCEF valid + translation/rotation merge predicate"
        target.attrs["prior_risk_reset_required_at_segment_boundary"] = True
        for name in DATASETS_TO_VIRTUALIZE:
            dataset = source[name]
            layout = h5py.VirtualLayout(shape=dataset.shape, dtype=dataset.dtype)
            virtual_source = h5py.VirtualSource(str(audit.merged_path), name, shape=dataset.shape)
            layout[...] = virtual_source
            target.create_virtual_dataset(name, layout)
        target.create_dataset("file_names", data=np.asarray([audit.sequence], dtype=h5py.string_dtype()))
        target.create_dataset("indices", data=np.asarray([[0, n_samples]], dtype=np.int64))
        target.create_dataset("sequence_name", data=np.full(n_samples, audit.sequence, dtype=h5py.string_dtype()))
        target.create_dataset("source_exp_index", data=audit.source_indices, dtype=np.int64)
        target.create_dataset("timestamp", data=audit.timestamps, dtype=np.float64)
        target.create_dataset("readout_time", data=audit.timestamps, dtype=np.float64)
        target.create_dataset("source_h5_sample_index", data=np.arange(n_samples, dtype=np.int64))
        target.create_dataset("continuous_from_previous", data=audit.continuous_from_previous, dtype=bool)
        target.create_dataset("segment_id", data=audit.segment_ids, dtype=np.int32)
    os.replace(temporary_path, output_path)
    with h5py.File(output_path, "r") as check:
        if len(check["source_exp_index"]) != n_samples or check["exp_filts"].shape[0] != n_samples:
            raise RuntimeError(f"ordered H5 validation failed: {output_path}")
        # Ensure the virtual source is readable, not merely syntactically declared.
        check["exp_filts"][0]
        check["ttc"][n_samples - 1]


def delta_summary(values: np.ndarray) -> dict[str, float | None]:
    if len(values) == 0:
        return {"mean": None, "std": None, "max": None, "min": None}
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "max": float(values.max()),
        "min": float(values.min()),
    }


def audit_summary(audit: SplitAudit, payload: dict[str, Any], ordered_h5: Path) -> dict[str, Any]:
    lengths = np.asarray([end - start for start, end in audit.segments], dtype=np.int64)
    time_deltas = np.diff(audit.timestamps)
    continuous_deltas = time_deltas[audit.continuous_from_previous[1:]]
    return {
        "split": audit.label,
        "merged_h5": str(audit.merged_path),
        "ordered_h5": str(ordered_h5),
        "num_samples": int(len(audit.source_indices)),
        "sequence_count": 1,
        "sequence_name": audit.sequence,
        "metadata_present_in_original_h5": audit.metadata_present,
        "metadata_reconstructed_from_source": audit.metadata_reconstructed,
        "legacy_file_names": audit.legacy_file_names,
        "legacy_indices": audit.legacy_indices,
        "continuous_segment_count": int(len(audit.segments)),
        "segment_lengths": lengths.astype(int).tolist(),
        "segment_length_summary": delta_summary(lengths.astype(float)),
        "continuous_adjacent_ratio": float(audit.continuous_from_previous[1:].mean()) if len(audit.source_indices) > 1 else 1.0,
        "time_delta_us_all_adjacent": delta_summary(time_deltas),
        "time_delta_us_continuous_only": delta_summary(continuous_deltas),
        "frame_index_jump_count": audit.index_jump_count,
        "timestamp_backtrack_count": audit.timestamp_backtrack_count,
        "timestamp_large_gap_count": audit.timestamp_large_gap_count,
        "invalid_source_count": audit.invalid_source_count,
        "empty_mask_sample_count": audit.empty_mask_sample_count,
        "valid_mask_pixel_summary": audit.valid_mask_pixel_summary,
        "clip_counts": payload["clip_counts"],
        "mapping_checks": audit.mapping_checks,
    }


def leakage_summary(train: SplitAudit, test: SplitAudit) -> dict[str, Any]:
    same_sequence = train.sequence == test.sequence
    if not same_sequence:
        return {
            "same_sequence": False,
            "temporal_overlap": False,
            "conclusion": "不存在 train/test 时序泄漏：两个数据划分来自不同源序列。",
        }
    train_times = set(train.timestamps.tolist())
    test_times = set(test.timestamps.tolist())
    overlap = len(train_times & test_times)
    return {
        "same_sequence": True,
        "temporal_overlap": bool(overlap),
        "overlapping_timestamp_count": int(overlap),
        "conclusion": "检测到时序泄漏。" if overlap else "来自同一序列，但没有完全相同的 readout 时间戳。",
    }


def write_report(path: Path, train: dict[str, Any], test: dict[str, Any], leakage: dict[str, Any]) -> None:
    def line(value: float | None, digits: int = 3) -> str:
        return "N/A" if value is None else f"{value:.{digits}f}"

    rows = [
        "# TTCEF 时序连续性审计报告",
        "",
        f"- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "- 时钟单位：微秒；名义 readout 间隔：7.000 ms；连续判定容差：±0.500 ms。",
        "- 原始 merged H5 不含逐样本 `sequence_name`、`source_exp_index`、`timestamp/readout_time`；本报告按 TTCEF 合并筛选规则从源文件恢复映射，并以 TTC 内容抽样逐项验证。",
        "- `ordered_*.h5` 是 HDF5 virtual dataset 视图，不复制特征图；原始 merged H5 必须保留在当前绝对路径。",
        "",
        "## 总览",
        "",
        "| 数据划分 | 样本数 | 序列 | 连续片段数 | 连续相邻比例 | 帧索引跳变 | 时间回退 | 时间大跳变 |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for summary in (train, test):
        rows.append(
            f"| {summary['split']} | {summary['num_samples']} | {summary['sequence_name']} | "
            f"{summary['continuous_segment_count']} | {summary['continuous_adjacent_ratio']:.6f} | "
            f"{summary['frame_index_jump_count']} | {summary['timestamp_backtrack_count']} | "
            f"{summary['timestamp_large_gap_count']} |"
        )
    for summary in (train, test):
        all_dt = summary["time_delta_us_all_adjacent"]
        continuous_dt = summary["time_delta_us_continuous_only"]
        rows.extend(
            [
                "",
                f"## {summary['split']} 片段与时间统计",
                "",
                f"- 原始 H5：`{summary['merged_h5']}`",
                f"- 有序 H5：`{summary['ordered_h5']}`",
                f"- 原 H5 `file_names`：{summary['legacy_file_names']}；`indices`：{summary['legacy_indices']}（不含逐样本源索引）。",
                f"- 片段长度：{summary['segment_lengths']}",
                f"- 空 mask 样本：{summary['empty_mask_sample_count']}；有效 mask 像素数：最小 {summary['valid_mask_pixel_summary']['min']:.0f}，中位数 {summary['valid_mask_pixel_summary']['median']:.0f}，最大 {summary['valid_mask_pixel_summary']['max']:.0f}。",
                f"- 相邻时间差（全部）：均值 {line(all_dt['mean'])} us，标准差 {line(all_dt['std'])} us，最大 {line(all_dt['max'])} us。",
                f"- 相邻时间差（仅连续）：均值 {line(continuous_dt['mean'])} us，标准差 {line(continuous_dt['std'])} us，最大 {line(continuous_dt['max'])} us。",
                "",
                "| clip 长度 | 不重叠 clip 数 | 可选滑动窗口数 |",
                "|---:|---:|---:|",
            ]
        )
        for length in map(str, CLIP_LENGTHS):
            counts = summary["clip_counts"][length]
            rows.append(f"| {length} | {counts['non_overlapping']} | {counts['sliding_windows']} |")
    rows.extend(
        [
            "",
            "## Train/Test 泄漏检查",
            "",
            f"- {leakage['conclusion']}",
            "",
            "## B3 结论",
            "",
            "- 当前 `train.h5` / `test.h5` 不能直接用于 B3 连续帧训练：虽然它们各自只有一个序列且内容顺序可回溯，但逐样本时序元数据缺失，不能把 H5 行号当作物理连续性依据。",
            "- 使用本次生成的 `ordered_train.h5` / `ordered_test.h5`；训练时只在同一 `segment_id` 内传递上一帧风险，片段开始处必须重置状态。",
            "- 推荐默认 clip 长度为 16 帧（约 112 ms）：相比 8 帧拥有更稳定的短时风险上下文，较 32 帧更易训练且边界损耗更小；8 帧可作为消融，32 帧可作为长上下文消融。",
        ]
    )
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-file", type=Path, default=Path("ttcef_merged_min/train.h5"))
    parser.add_argument("--test-file", type=Path, default=Path("ttcef_merged_min/test.h5"))
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/data/evttc_storage/event+SNN+TTC/data_m3ed_min"),
    )
    parser.add_argument("--out-dir", type=Path, default=Path("logs/ttc_TTCEF/temporal_audit"))
    parser.add_argument("--ordered-dir", type=Path, default=None)
    args = parser.parse_args()

    train_file = args.train_file.resolve()
    test_file = args.test_file.resolve()
    out_dir = args.out_dir.resolve()
    ordered_dir = args.ordered_dir.resolve() if args.ordered_dir else train_file.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    train_audit = audit_split("train", train_file, args.data_root.resolve())
    test_audit = audit_split("test", test_file, args.data_root.resolve())
    train_payload = build_clip_payload(train_audit)
    test_payload = build_clip_payload(test_audit)
    ordered_train = ordered_dir / "ordered_train.h5"
    ordered_test = ordered_dir / "ordered_test.h5"
    write_ordered_virtual_h5(train_audit, ordered_train)
    write_ordered_virtual_h5(test_audit, ordered_test)
    (out_dir / "train_clips.json").write_text(json.dumps(train_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out_dir / "test_clips.json").write_text(json.dumps(test_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    train_summary = audit_summary(train_audit, train_payload, ordered_train)
    test_summary = audit_summary(test_audit, test_payload, ordered_test)
    leakage = leakage_summary(train_audit, test_audit)
    write_report(out_dir / "temporal_continuity_report.md", train_summary, test_summary, leakage)
    summary = {"train": train_summary, "test": test_summary, "leakage": leakage}
    (out_dir / "temporal_continuity_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
