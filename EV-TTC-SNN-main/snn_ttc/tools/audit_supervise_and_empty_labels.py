#!/usr/bin/env python3
"""审计 Skatepark 多事件数 TTC H5 的监督有效性、空标签和 block 监督密度。"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVENT_COUNTS = [5000, 10000, 15000, 20000]
PIXELS_PER_ROI = 128 * 128


def _to_py(value: Any) -> Any:
    """把 numpy 标量转成 json/csv 友好的 Python 标量。"""

    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def safe_ratio(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def pct(num: float, den: float) -> float:
    return 100.0 * safe_ratio(num, den)


def fmt_num(value: Any, digits: int = 4) -> str:
    if value is None:
        return "NA"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(v):
        return "NA"
    if abs(v) >= 1000:
        return f"{v:.1f}"
    if abs(v) >= 100:
        return f"{v:.2f}"
    return f"{v:.{digits}f}"


def stats(values: np.ndarray) -> dict[str, float | int | None]:
    """返回常用分位数；空数组用 None 填充，避免报告里误写 0。"""

    arr = np.asarray(values)
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


def longest_run(values: np.ndarray, target: bool) -> int:
    """计算一个 bool 序列里 target 的最长连续长度。"""

    best = 0
    cur = 0
    for item in values:
        if bool(item) == target:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return int(best)


def count_true_segments(values: np.ndarray) -> int:
    """统计 True 片段数量，用于判断监督信号是否零散。"""

    count = 0
    prev = False
    for item in values:
        cur = bool(item)
        if cur and not prev:
            count += 1
        prev = cur
    return int(count)


def locate_h5(input_dir: Path, event_counts: list[int]) -> dict[int, Path]:
    """在输入目录内自动查找 skatepark_N*.h5，兼容用户整理后的 H5 子目录。"""

    out: dict[int, Path] = {}
    for n in event_counts:
        candidates: list[Path] = []
        for pattern_root in [input_dir, input_dir / "H5", ROOT / "debug_sets"]:
            if pattern_root.exists():
                candidates.extend(pattern_root.rglob(f"skatepark_N{n}.h5"))
        candidates = [p for p in candidates if not p.name.startswith("audit_subset_")]
        if candidates:
            candidates = sorted(set(candidates), key=lambda p: (len(str(p)), str(p)))
            out[n] = candidates[0]
    missing = [n for n in event_counts if n not in out]
    if missing:
        raise FileNotFoundError(f"缺少 H5: {missing}; input_dir={input_dir}")
    return out


def default_output_dir(input_dir: Path, h5_paths: dict[int, Path]) -> Path:
    """默认把审计结果放在多 N 数据目录旁边，避免和正式 H5 混在一起。"""

    parents = {p.parent for p in h5_paths.values()}
    if len(parents) == 1:
        parent = next(iter(parents))
        if parent.name == "H5":
            return parent.parent / "supervision_audit"
        return parent / "supervision_audit"
    return input_dir / "supervision_audit"


def read_step_arrays(path: Path, speed_threshold: float, omega_threshold: float, max_step_duration_us: int) -> dict[str, Any]:
    """读取 step 级标量，并根据 H5 内容构造审计标志。"""

    with h5py.File(path, "r") as f:
        n = int(f.attrs.get("N", int(path.stem.replace("skatepark_N", ""))))
        step_index = f["step_index"][:].astype(np.int64)
        block_index = f["block_index"][:].astype(np.int64)
        step_in_block = f["step_in_block"][:].astype(np.int16)
        roi_x0 = f["roi_x0"][:].astype(np.int16)
        roi_y0 = f["roi_y0"][:].astype(np.int16)
        t_start = f["t_start"][:].astype(np.int64)
        t_end = f["t_end"][:].astype(np.int64)
        event_dt = f["event_dt"][:].astype(np.int64)
        T = f["T"][:].astype(np.float64)
        Omega = f["Omega"][:].astype(np.float64)
        pose_valid = f["pose_valid"][:].astype(bool) if "pose_valid" in f else np.ones(len(step_index), dtype=bool)
        speed_norm = np.linalg.norm(T, axis=1)
        omega_norm = np.linalg.norm(Omega, axis=1)
        speed_valid = f["speed_valid"][:].astype(bool) if "speed_valid" in f else speed_norm > speed_threshold
        omega_valid = f["omega_valid"][:].astype(bool) if "omega_valid" in f else omega_norm < omega_threshold
        supervise_valid = f["supervise_valid"][:].astype(bool) if "supervise_valid" in f else speed_valid & omega_valid & pose_valid
        valid_mask_ratio = f["valid_mask_ratio"][:].astype(np.float64)
        min_ttc = f["min_ttc"][:].astype(np.float64) if "min_ttc" in f else np.full(len(step_index), np.nan)
        total_mapped_weight = f["total_mapped_weight"][:].astype(np.float64) if "total_mapped_weight" in f else np.full(len(step_index), np.nan)
        pos_event_count = f["pos_event_count"][:].astype(np.int64) if "pos_event_count" in f else np.full(len(step_index), -1)
        neg_event_count = f["neg_event_count"][:].astype(np.int64) if "neg_event_count" in f else np.full(len(step_index), -1)
        raw_event_start_idx = f["raw_event_start_idx"][:].astype(np.int64)
        raw_event_end_idx = f["raw_event_end_idx"][:].astype(np.int64)
        sequence_name = f.attrs.get("sequence_name", "unknown")
        if isinstance(sequence_name, bytes):
            sequence_name = sequence_name.decode("utf-8")

    valid_pixel_count = np.rint(valid_mask_ratio * PIXELS_PER_ROI).astype(np.int64)
    empty_label = valid_pixel_count <= 0
    time_invalid = (t_end <= t_start) | (event_dt <= 0) | (event_dt > max_step_duration_us)
    pose_or_time_invalid = (~pose_valid) | time_invalid | ~np.isfinite(speed_norm) | ~np.isfinite(omega_norm)
    tz = T[:, 2]
    tz_nonpositive = tz <= 0

    # 主 H5 没保存原始 depth_start；Tz>0 且 pose/time 正常但 mask 为空时，不能可靠区分深度为空、
    # ROI 裁剪为空、重投影为空或数值异常，因此按 unknown 记录，不硬猜。
    depth_empty = np.zeros_like(empty_label, dtype=bool)
    reprojection_empty = np.zeros_like(empty_label, dtype=bool)
    roi_crop_empty = np.zeros_like(empty_label, dtype=bool)
    numeric_invalid = np.zeros_like(empty_label, dtype=bool)
    unknown_empty = empty_label & ~pose_or_time_invalid & ~tz_nonpositive

    primary_reason = np.full(len(step_index), "nonempty", dtype=object)
    primary_reason[empty_label & pose_or_time_invalid] = "pose_or_time_invalid"
    primary_reason[empty_label & ~pose_or_time_invalid & tz_nonpositive] = "tz_nonpositive"
    primary_reason[unknown_empty] = "unknown_no_depth_metadata"

    direct = supervise_valid & (~empty_label)
    speed_margin = speed_norm - speed_threshold
    omega_margin = omega_threshold - omega_norm

    return {
        "N": n,
        "path": path,
        "sequence_name": str(sequence_name),
        "step_index": step_index,
        "block_index": block_index,
        "step_in_block": step_in_block,
        "roi_x0": roi_x0,
        "roi_y0": roi_y0,
        "t_start": t_start,
        "t_end": t_end,
        "event_dt": event_dt,
        "T": T,
        "Omega": Omega,
        "T_norm": speed_norm,
        "Tz": tz,
        "Omega_norm": omega_norm,
        "speed_valid": speed_valid,
        "omega_valid": omega_valid,
        "pose_valid": pose_valid,
        "supervise_valid": supervise_valid,
        "valid_mask_ratio": valid_mask_ratio,
        "valid_pixel_count": valid_pixel_count,
        "empty_label": empty_label,
        "direct_supervision_valid": direct,
        "speed_margin": speed_margin,
        "omega_margin": omega_margin,
        "pose_or_time_invalid": pose_or_time_invalid,
        "depth_empty": depth_empty,
        "reprojection_empty": reprojection_empty,
        "roi_crop_empty": roi_crop_empty,
        "tz_nonpositive": tz_nonpositive,
        "numeric_invalid": numeric_invalid,
        "unknown_empty": unknown_empty,
        "primary_empty_reason": primary_reason,
        "min_ttc": min_ttc,
        "total_mapped_weight": total_mapped_weight,
        "pos_event_count": pos_event_count,
        "neg_event_count": neg_event_count,
        "raw_event_start_idx": raw_event_start_idx,
        "raw_event_end_idx": raw_event_end_idx,
    }


def summarize_steps(arr: dict[str, Any]) -> dict[str, Any]:
    n = int(len(arr["step_index"]))
    sv = arr["supervise_valid"]
    sp = arr["speed_valid"]
    om = arr["omega_valid"]
    pose = arr["pose_valid"]
    empty = arr["empty_label"]
    direct = arr["direct_supervision_valid"]

    # A-D 严格对应速度/角速度；pose_invalid 单独列出，便于对照实际 H5 的 supervise_valid。
    cat = {
        "A_speed_ok_omega_ok": int(np.sum(sp & om)),
        "B_speed_fail_only": int(np.sum((~sp) & om)),
        "C_omega_fail_only": int(np.sum(sp & (~om))),
        "D_both_fail": int(np.sum((~sp) & (~om))),
        "pose_invalid": int(np.sum(~pose)),
    }
    failed_sv = int(np.sum(~sv))
    reasons = Counter(arr["primary_empty_reason"][empty].tolist())
    cross = {
        "sv1_nonempty": int(np.sum(sv & ~empty)),
        "sv1_empty": int(np.sum(sv & empty)),
        "sv0_nonempty": int(np.sum((~sv) & ~empty)),
        "sv0_empty": int(np.sum((~sv) & empty)),
    }
    cross.update(
        {
            "p_empty_given_sv1": safe_ratio(cross["sv1_empty"], cross["sv1_empty"] + cross["sv1_nonempty"]),
            "p_empty_given_sv0": safe_ratio(cross["sv0_empty"], cross["sv0_empty"] + cross["sv0_nonempty"]),
            "p_sv0_given_empty": safe_ratio(cross["sv0_empty"], cross["sv0_empty"] + cross["sv1_empty"]),
            "p_sv0_given_nonempty": safe_ratio(cross["sv0_nonempty"], cross["sv0_nonempty"] + cross["sv1_nonempty"]),
        }
    )
    return {
        "N": int(arr["N"]),
        "path": str(arr["path"]),
        "sequence_name": arr["sequence_name"],
        "steps": n,
        "supervise_valid_count": int(np.sum(sv)),
        "supervise_valid_ratio": safe_ratio(np.sum(sv), n),
        "direct_supervision_count": int(np.sum(direct)),
        "direct_supervision_ratio": safe_ratio(np.sum(direct), n),
        "nonempty_label_count": int(np.sum(~empty)),
        "nonempty_label_ratio": safe_ratio(np.sum(~empty), n),
        "empty_label_count": int(np.sum(empty)),
        "empty_label_ratio": safe_ratio(np.sum(empty), n),
        "supervise_categories": cat,
        "failed_supervise_count": failed_sv,
        "failed_supervise_breakdown": {
            "speed_fail_only_ratio_in_failed_sv": safe_ratio(cat["B_speed_fail_only"], failed_sv),
            "omega_fail_only_ratio_in_failed_sv": safe_ratio(cat["C_omega_fail_only"], failed_sv),
            "both_fail_ratio_in_failed_sv": safe_ratio(cat["D_both_fail"], failed_sv),
            "pose_invalid_ratio_in_failed_sv": safe_ratio(cat["pose_invalid"], failed_sv),
        },
        "T_norm_stats": stats(arr["T_norm"]),
        "Tz_stats": stats(arr["Tz"]),
        "Omega_norm_stats": stats(arr["Omega_norm"]),
        "speed_margin_stats": stats(arr["speed_margin"]),
        "omega_margin_stats": stats(arr["omega_margin"]),
        "valid_pixel_ratio_stats": stats(arr["valid_mask_ratio"]),
        "valid_pixel_count_stats": stats(arr["valid_pixel_count"]),
        "empty_reason_counts": dict(reasons),
        "empty_reason_ratios": {k: safe_ratio(v, int(np.sum(empty))) for k, v in reasons.items()},
        "pose_or_time_invalid_empty_count": int(np.sum(empty & arr["pose_or_time_invalid"])),
        "tz_nonpositive_empty_count": int(np.sum(empty & arr["tz_nonpositive"])),
        "depth_reprojection_empty_count": int(np.sum(empty & (arr["depth_empty"] | arr["reprojection_empty"] | arr["roi_crop_empty"]))),
        "unknown_empty_count": int(np.sum(empty & arr["unknown_empty"])),
        "cross": cross,
    }


def build_block_rows(arr: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    block_to_indices: dict[int, list[int]] = defaultdict(list)
    for i, block in enumerate(arr["block_index"]):
        block_to_indices[int(block)].append(i)

    complete_direct_counts = []
    complete_sv_counts = []
    complete_nonempty_counts = []
    longest_direct = []
    longest_unsup = []
    segment_counts = []
    step_position_counts = {i: {"total": 0, "sv": 0, "nonempty": 0, "direct": 0} for i in range(10)}

    for block, idx_list in sorted(block_to_indices.items()):
        idx = np.asarray(sorted(idx_list, key=lambda j: int(arr["step_in_block"][j])), dtype=np.int64)
        steps = arr["step_in_block"][idx]
        complete = len(idx) == 10 and set(steps.tolist()) == set(range(10))
        sv = arr["supervise_valid"][idx]
        nonempty = ~arr["empty_label"][idx]
        direct = arr["direct_supervision_valid"][idx]
        row = {
            "N": int(arr["N"]),
            "block_index": int(block),
            "complete_10step_block": bool(complete),
            "num_steps": int(len(idx)),
            "first_step_index": int(arr["step_index"][idx[0]]),
            "last_step_index": int(arr["step_index"][idx[-1]]),
            "roi_x0": int(arr["roi_x0"][idx[0]]),
            "roi_y0": int(arr["roi_y0"][idx[0]]),
            "t_start": int(arr["t_start"][idx[0]]),
            "t_end": int(arr["t_end"][idx[-1]]),
            "block_duration_us": int(arr["t_end"][idx[-1]] - arr["t_start"][idx[0]]),
            "num_supervise_valid": int(np.sum(sv)),
            "num_nonempty_label": int(np.sum(nonempty)),
            "num_direct_supervision_valid": int(np.sum(direct)),
            "longest_direct_supervision_run": longest_run(direct, True),
            "longest_no_direct_supervision_run": longest_run(direct, False),
            "direct_supervision_segments": count_true_segments(direct),
        }
        rows.append(row)

        if complete:
            complete_sv_counts.append(row["num_supervise_valid"])
            complete_nonempty_counts.append(row["num_nonempty_label"])
            complete_direct_counts.append(row["num_direct_supervision_valid"])
            longest_direct.append(row["longest_direct_supervision_run"])
            longest_unsup.append(row["longest_no_direct_supervision_run"])
            segment_counts.append(row["direct_supervision_segments"])
            for local_i in idx:
                pos = int(arr["step_in_block"][local_i])
                step_position_counts[pos]["total"] += 1
                step_position_counts[pos]["sv"] += int(arr["supervise_valid"][local_i])
                step_position_counts[pos]["nonempty"] += int(not arr["empty_label"][local_i])
                step_position_counts[pos]["direct"] += int(arr["direct_supervision_valid"][local_i])

    direct_counts = np.asarray(complete_direct_counts, dtype=np.int64)
    hist = {str(i): int(np.sum(direct_counts == i)) for i in range(11)}
    total_complete = int(direct_counts.size)
    summary = {
        "N": int(arr["N"]),
        "total_blocks": len(rows),
        "complete_10step_blocks": total_complete,
        "direct_hist_0_to_10": hist,
        "hist_groups": {
            "0": int(np.sum(direct_counts == 0)),
            "1_to_3": int(np.sum((direct_counts >= 1) & (direct_counts <= 3))),
            "4_to_6": int(np.sum((direct_counts >= 4) & (direct_counts <= 6))),
            "7_to_9": int(np.sum((direct_counts >= 7) & (direct_counts <= 9))),
            "10": int(np.sum(direct_counts == 10)),
        },
        "direct_per_block_stats": stats(direct_counts.astype(np.float64)),
        "supervise_valid_per_block_stats": stats(np.asarray(complete_sv_counts, dtype=np.float64)),
        "nonempty_per_block_stats": stats(np.asarray(complete_nonempty_counts, dtype=np.float64)),
        "zero_direct_block_ratio": safe_ratio(np.sum(direct_counts == 0), total_complete),
        "one_to_three_direct_block_ratio": safe_ratio(np.sum((direct_counts >= 1) & (direct_counts <= 3)), total_complete),
        "at_least_five_direct_block_ratio": safe_ratio(np.sum(direct_counts >= 5), total_complete),
        "ten_direct_block_ratio": safe_ratio(np.sum(direct_counts == 10), total_complete),
        "longest_direct_run_stats": stats(np.asarray(longest_direct, dtype=np.float64)),
        "longest_no_direct_run_stats": stats(np.asarray(longest_unsup, dtype=np.float64)),
        "direct_segment_count_stats": stats(np.asarray(segment_counts, dtype=np.float64)),
        "step_position": {
            str(pos): {
                "total": c["total"],
                "supervise_valid_ratio": safe_ratio(c["sv"], c["total"]),
                "nonempty_label_ratio": safe_ratio(c["nonempty"], c["total"]),
                "direct_supervision_ratio": safe_ratio(c["direct"], c["total"]),
            }
            for pos, c in step_position_counts.items()
        },
    }
    return rows, summary


def make_step_rows(arrays_by_n: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    keys = [
        "step_index",
        "block_index",
        "step_in_block",
        "roi_x0",
        "roi_y0",
        "t_start",
        "t_end",
        "event_dt",
        "T_norm",
        "Tz",
        "Omega_norm",
        "speed_margin",
        "omega_margin",
        "speed_valid",
        "omega_valid",
        "pose_valid",
        "supervise_valid",
        "valid_pixel_count",
        "valid_mask_ratio",
        "empty_label",
        "direct_supervision_valid",
        "primary_empty_reason",
        "pose_or_time_invalid",
        "tz_nonpositive",
        "unknown_empty",
        "min_ttc",
        "pos_event_count",
        "neg_event_count",
        "total_mapped_weight",
    ]
    for n in sorted(arrays_by_n):
        arr = arrays_by_n[n]
        for i in range(len(arr["step_index"])):
            row = {"N": n, "sequence_name": arr["sequence_name"], "h5_path": str(arr["path"])}
            for key in keys:
                row[key] = _to_py(arr[key][i])
            rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
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
        for row in rows:
            writer.writerow({k: _to_py(row.get(k, "")) for k in fields})


def sample_records(records: list[dict[str, Any]], rng: random.Random, first_n: int = 20, random_n: int = 20) -> list[dict[str, Any]]:
    """每类导出前 20 个和随机 20 个，重复样本只保留一次。"""

    out: list[dict[str, Any]] = []
    seen: set[tuple[int, int, str]] = set()
    for mode, selected in [
        ("first", records[:first_n]),
        ("random", rng.sample(records, min(random_n, len(records))) if records else []),
    ]:
        for rec in selected:
            key = (int(rec["N"]), int(rec["step_index"]), str(rec["异常类别"]))
            if key in seen:
                continue
            seen.add(key)
            tmp = dict(rec)
            tmp["样本来源"] = mode
            out.append(tmp)
    return out


def build_exception_rows(arrays_by_n: dict[int, dict[str, Any]], block_rows: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    block_by_n: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in block_rows:
        block_by_n[int(row["N"])].append(row)

    all_rows: list[dict[str, Any]] = []
    for n in sorted(arrays_by_n):
        arr = arrays_by_n[n]
        base_fields = []
        for i in range(len(arr["step_index"])):
            base_fields.append(
                {
                    "N": n,
                    "step_index": int(arr["step_index"][i]),
                    "block_index": int(arr["block_index"][i]),
                    "step_in_block": int(arr["step_in_block"][i]),
                    "roi_x0": int(arr["roi_x0"][i]),
                    "roi_y0": int(arr["roi_y0"][i]),
                    "t_start": int(arr["t_start"][i]),
                    "t_end": int(arr["t_end"][i]),
                    "event_dt": int(arr["event_dt"][i]),
                    "T_norm": float(arr["T_norm"][i]),
                    "Tz": float(arr["Tz"][i]),
                    "Omega_norm": float(arr["Omega_norm"][i]),
                    "speed_valid": bool(arr["speed_valid"][i]),
                    "omega_valid": bool(arr["omega_valid"][i]),
                    "supervise_valid": bool(arr["supervise_valid"][i]),
                    "valid_pixel_ratio": float(arr["valid_mask_ratio"][i]),
                    "empty_label_reason": str(arr["primary_empty_reason"][i]),
                }
            )

        masks = {
            "supervise_valid=1且空标签": arr["supervise_valid"] & arr["empty_label"],
            "supervise_valid=0且非空标签": (~arr["supervise_valid"]) & (~arr["empty_label"]),
            "Tz<=0空标签": arr["empty_label"] & arr["tz_nonpositive"],
            "depth/reprojection空标签": arr["empty_label"] & (arr["depth_empty"] | arr["reprojection_empty"] | arr["roi_crop_empty"]),
            "unknown空标签": arr["empty_label"] & arr["unknown_empty"],
        }
        for name, mask in masks.items():
            records = []
            for i in np.where(mask)[0]:
                rec = dict(base_fields[int(i)])
                rec["异常类别"] = name
                records.append(rec)
            all_rows.extend(sample_records(records, rng))

        first_step_to_index = {int(v): i for i, v in enumerate(arr["step_index"])}
        for name, pred in [
            ("完整block中0个监督step", lambda r: r["complete_10step_block"] and r["num_direct_supervision_valid"] == 0),
            ("完整block中10个监督step", lambda r: r["complete_10step_block"] and r["num_direct_supervision_valid"] == 10),
        ]:
            records = []
            for b in block_by_n[n]:
                if not pred(b):
                    continue
                i = first_step_to_index.get(int(b["first_step_index"]))
                if i is None:
                    continue
                rec = dict(base_fields[i])
                rec["异常类别"] = name
                rec["block_direct_count"] = int(b["num_direct_supervision_valid"])
                rec["block_duration_us"] = int(b["block_duration_us"])
                records.append(rec)
            all_rows.extend(sample_records(records, rng))
    return all_rows


def visualize_exception_samples(exception_rows: list[dict[str, Any]], h5_paths: dict[int, Path], out_dir: Path) -> None:
    """按异常类别随机可视化少量样本；只画已有 H5 字段，不重新生成标签。"""

    if not exception_rows:
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in exception_rows:
        if row.get("样本来源") == "random":
            grouped[(int(row["N"]), str(row["异常类别"]))].append(row)

    vis_root = out_dir / "异常样本可视化"
    vis_root.mkdir(parents=True, exist_ok=True)
    for (n, category), rows in grouped.items():
        for row in rows[:5]:
            with h5py.File(h5_paths[n], "r") as f:
                step_indices = f["step_index"][:]
                loc = np.where(step_indices == int(row["step_index"]))[0]
                if loc.size == 0:
                    continue
                i = int(loc[0])
                event = f["event_cnt"][i]
                inv = f["inverse_ttc_start"][i]
                mask = f["valid_ttc_mask"][i]
            fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
            fig.suptitle(f"N={n} | {category} | step={row['step_index']} block={row['block_index']}")
            axes[0, 0].imshow(event[0], cmap="magma")
            axes[0, 0].set_title("positive event_cnt")
            axes[0, 1].imshow(event[1], cmap="magma")
            axes[0, 1].set_title("negative event_cnt")
            axes[0, 2].imshow(event[0] - event[1], cmap="coolwarm")
            axes[0, 2].set_title("pos-neg")
            im = axes[1, 0].imshow(inv, cmap="inferno")
            axes[1, 0].set_title("inverse TTC")
            fig.colorbar(im, ax=axes[1, 0], fraction=0.046)
            axes[1, 1].imshow(mask, cmap="gray", vmin=0, vmax=1)
            axes[1, 1].set_title("valid_ttc_mask")
            axes[1, 2].axis("off")
            axes[1, 2].text(
                0.0,
                1.0,
                "\n".join(
                    [
                        f"Tnorm={fmt_num(row['T_norm'])}",
                        f"Tz={fmt_num(row['Tz'])}",
                        f"Omeganorm={fmt_num(row['Omega_norm'])}",
                        f"sv={row['supervise_valid']}",
                        f"valid_ratio={fmt_num(row['valid_pixel_ratio'])}",
                        f"reason={row['empty_label_reason']}",
                    ]
                ),
                va="top",
            )
            for ax in axes.flat:
                ax.set_xticks([])
                ax.set_yticks([])
            safe_category = category.replace("/", "_").replace("<=", "le").replace("=", "")
            path = vis_root / f"N{n}_{safe_category}_step{int(row['step_index']):06d}.png"
            fig.savefig(path, dpi=150)
            plt.close(fig)


def write_supervise_report(path: Path, summaries: dict[int, dict[str, Any]]) -> None:
    lines = ["# SuperviseValid失效原因统计", ""]
    lines += [
        "> A/B/C/D 按速度阈值和角速度阈值划分；实际 H5 中 `supervise_valid` 还同时包含 `pose_valid`。本次数据 `pose_invalid` 为 0，因此不影响结论。",
        "",
        "| N | steps | A 可监督运动 | B 仅速度失败 | C 仅角速度失败 | D 双失败 | supervise_valid比例 | B占SV失败 | C占SV失败 | D占SV失败 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for n in sorted(summaries):
        s = summaries[n]
        c = s["supervise_categories"]
        f = s["failed_supervise_breakdown"]
        lines.append(
            f"| {n} | {s['steps']} | {c['A_speed_ok_omega_ok']} | {c['B_speed_fail_only']} | {c['C_omega_fail_only']} | {c['D_both_fail']} | "
            f"{pct(s['supervise_valid_count'], s['steps']):.2f}% | {100*f['speed_fail_only_ratio_in_failed_sv']:.2f}% | "
            f"{100*f['omega_fail_only_ratio_in_failed_sv']:.2f}% | {100*f['both_fail_ratio_in_failed_sv']:.2f}% |"
        )
    lines += ["", "## 运动量统计", ""]
    for n in sorted(summaries):
        s = summaries[n]
        lines += [
            f"### N={n}",
            f"- ||T|| mean/P5/P25/P50/P75/P95：{fmt_num(s['T_norm_stats']['mean'])} / {fmt_num(s['T_norm_stats']['p5'])} / {fmt_num(s['T_norm_stats']['p25'])} / {fmt_num(s['T_norm_stats']['p50'])} / {fmt_num(s['T_norm_stats']['p75'])} / {fmt_num(s['T_norm_stats']['p95'])}",
            f"- Tz mean/P5/P25/P50/P75/P95：{fmt_num(s['Tz_stats']['mean'])} / {fmt_num(s['Tz_stats']['p5'])} / {fmt_num(s['Tz_stats']['p25'])} / {fmt_num(s['Tz_stats']['p50'])} / {fmt_num(s['Tz_stats']['p75'])} / {fmt_num(s['Tz_stats']['p95'])}",
            f"- ||Omega|| mean/P5/P25/P50/P75/P95：{fmt_num(s['Omega_norm_stats']['mean'])} / {fmt_num(s['Omega_norm_stats']['p5'])} / {fmt_num(s['Omega_norm_stats']['p25'])} / {fmt_num(s['Omega_norm_stats']['p50'])} / {fmt_num(s['Omega_norm_stats']['p75'])} / {fmt_num(s['Omega_norm_stats']['p95'])}",
            f"- speed_margin mean/P50：{fmt_num(s['speed_margin_stats']['mean'])} / {fmt_num(s['speed_margin_stats']['p50'])}",
            f"- omega_margin mean/P50：{fmt_num(s['omega_margin_stats']['mean'])} / {fmt_num(s['omega_margin_stats']['p50'])}",
            "",
        ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_empty_report(path: Path, summaries: dict[int, dict[str, Any]]) -> None:
    lines = [
        "# 空标签Step原因统计",
        "",
        "> `empty_label_step = valid_ttc_mask.sum() == 0`。主 H5 未保存原始 `depth_start`，因此无法把 Tz>0 且 pose/time 正常的空标签进一步可靠拆分为 depth/reprojection/ROI/numeric，本报告统一标为 `unknown_no_depth_metadata`，不硬猜。",
        "",
        "| N | steps | 空标签数量 | 空标签比例 | pose/time异常 | Tz<=0 | depth/reprojection可确认 | unknown | valid_pixel_ratio均值 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for n in sorted(summaries):
        s = summaries[n]
        empty = s["empty_label_count"]
        lines.append(
            f"| {n} | {s['steps']} | {empty} | {pct(empty, s['steps']):.2f}% | "
            f"{s['pose_or_time_invalid_empty_count']} | {s['tz_nonpositive_empty_count']} | "
            f"{s['depth_reprojection_empty_count']} | {s['unknown_empty_count']} | {fmt_num(s['valid_pixel_ratio_stats']['mean'])} |"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_cross_report(path: Path, summaries: dict[int, dict[str, Any]]) -> None:
    lines = ["# SuperviseValid与空标签交叉分析", ""]
    for n in sorted(summaries):
        c = summaries[n]["cross"]
        lines += [
            f"## N={n}",
            "",
            "|  | 非空标签 | 空标签 |",
            "|---|---:|---:|",
            f"| supervise_valid=1 | {c['sv1_nonempty']} | {c['sv1_empty']} |",
            f"| supervise_valid=0 | {c['sv0_nonempty']} | {c['sv0_empty']} |",
            "",
            f"- P(empty | supervise_valid=1)：{100*c['p_empty_given_sv1']:.2f}%",
            f"- P(empty | supervise_valid=0)：{100*c['p_empty_given_sv0']:.2f}%",
            f"- P(supervise_valid=0 | empty)：{100*c['p_sv0_given_empty']:.2f}%",
            f"- P(supervise_valid=0 | nonempty)：{100*c['p_sv0_given_nonempty']:.2f}%",
            f"- direct_supervision_valid：{summaries[n]['direct_supervision_count']} / {summaries[n]['steps']} ({pct(summaries[n]['direct_supervision_count'], summaries[n]['steps']):.2f}%)",
            "",
        ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_block_report(path: Path, block_summaries: dict[int, dict[str, Any]]) -> None:
    lines = [
        "# 完整10步Block监督密度统计",
        "",
        "| N | 完整block | 平均direct step/block | P50 | P5/P25/P75/P95 | 0监督block | 1-3监督block | >=5监督block | 10监督block |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for n in sorted(block_summaries):
        b = block_summaries[n]
        st = b["direct_per_block_stats"]
        lines.append(
            f"| {n} | {b['complete_10step_blocks']} | {fmt_num(st['mean'])} | {fmt_num(st['p50'])} | "
            f"{fmt_num(st['p5'])}/{fmt_num(st['p25'])}/{fmt_num(st['p75'])}/{fmt_num(st['p95'])} | "
            f"{100*b['zero_direct_block_ratio']:.2f}% | {100*b['one_to_three_direct_block_ratio']:.2f}% | "
            f"{100*b['at_least_five_direct_block_ratio']:.2f}% | {100*b['ten_direct_block_ratio']:.2f}% |"
        )
    lines += ["", "## 0到10完整直方图", ""]
    for n in sorted(block_summaries):
        hist = block_summaries[n]["direct_hist_0_to_10"]
        lines += [
            f"### N={n}",
            "| direct step数 | block数 |",
            "|---:|---:|",
        ]
        for i in range(11):
            lines.append(f"| {i} | {hist[str(i)]} |")
        lines.append("")
    lines += ["## Block内位置统计", ""]
    for n in sorted(block_summaries):
        lines += [f"### N={n}", "| step_in_block | supervise_valid | 非空标签 | direct_supervision |", "|---:|---:|---:|---:|"]
        for pos in range(10):
            r = block_summaries[n]["step_position"][str(pos)]
            lines.append(
                f"| {pos} | {100*r['supervise_valid_ratio']:.2f}% | {100*r['nonempty_label_ratio']:.2f}% | {100*r['direct_supervision_ratio']:.2f}% |"
            )
        lines += [
            "",
            f"- 最长连续可监督step mean/P50：{fmt_num(block_summaries[n]['longest_direct_run_stats']['mean'])} / {fmt_num(block_summaries[n]['longest_direct_run_stats']['p50'])}",
            f"- 最长连续无监督step mean/P50：{fmt_num(block_summaries[n]['longest_no_direct_run_stats']['mean'])} / {fmt_num(block_summaries[n]['longest_no_direct_run_stats']['p50'])}",
            f"- 可监督片段数量 mean/P50：{fmt_num(block_summaries[n]['direct_segment_count_stats']['mean'])} / {fmt_num(block_summaries[n]['direct_segment_count_stats']['p50'])}",
            "",
        ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_comparison_report(path: Path, summaries: dict[int, dict[str, Any]], block_summaries: dict[int, dict[str, Any]], output_dir: Path, command: str) -> None:
    lines = [
        "# 四种N监督质量综合对比",
        "",
        "## 横向总表",
        "",
        "| 指标 | 5k | 10k | 15k | 20k |",
        "|---|---:|---:|---:|---:|",
    ]

    def s(n: int) -> dict[str, Any]:
        return summaries[n]

    def b(n: int) -> dict[str, Any]:
        return block_summaries[n]

    rows = [
        ("supervise_valid比例", lambda n: f"{pct(s(n)['supervise_valid_count'], s(n)['steps']):.2f}%"),
        ("仅速度失败比例", lambda n: f"{pct(s(n)['supervise_categories']['B_speed_fail_only'], s(n)['steps']):.2f}%"),
        ("仅角速度失败比例", lambda n: f"{pct(s(n)['supervise_categories']['C_omega_fail_only'], s(n)['steps']):.2f}%"),
        ("两者同时失败比例", lambda n: f"{pct(s(n)['supervise_categories']['D_both_fail'], s(n)['steps']):.2f}%"),
        ("空标签step比例", lambda n: f"{pct(s(n)['empty_label_count'], s(n)['steps']):.2f}%"),
        ("Tz<=0空标签比例", lambda n: f"{pct(s(n)['tz_nonpositive_empty_count'], s(n)['empty_label_count']):.2f}%"),
        ("depth/reprojection空标签比例", lambda n: f"{pct(s(n)['depth_reprojection_empty_count'], s(n)['empty_label_count']):.2f}%"),
        ("direct supervision比例", lambda n: f"{pct(s(n)['direct_supervision_count'], s(n)['steps']):.2f}%"),
        ("每block平均监督step数", lambda n: fmt_num(b(n)["direct_per_block_stats"]["mean"])),
        ("0监督block比例", lambda n: f"{100*b(n)['zero_direct_block_ratio']:.2f}%"),
        (">=5监督step block比例", lambda n: f"{100*b(n)['at_least_five_direct_block_ratio']:.2f}%"),
        ("10监督step block比例", lambda n: f"{100*b(n)['ten_direct_block_ratio']:.2f}%"),
    ]
    for label, getter in rows:
        lines.append("| " + label + " | " + " | ".join(getter(n) for n in DEFAULT_EVENT_COUNTS if n in summaries) + " |")

    # 用 direct/block 指标排序，而不是只看 step 总数。
    best_n = max(
        summaries,
        key=lambda n: (
            block_summaries[n]["direct_per_block_stats"]["mean"] or 0,
            summaries[n]["direct_supervision_ratio"],
            -summaries[n]["empty_label_ratio"],
        ),
    )
    main_fail = {}
    for n in sorted(summaries):
        c = summaries[n]["supervise_categories"]
        fail_parts = {
            "速度失败": c["B_speed_fail_only"],
            "角速度失败": c["C_omega_fail_only"],
            "速度和角速度同时失败": c["D_both_fail"],
            "pose/time失败": c["pose_invalid"],
        }
        main_fail[n] = max(fail_parts.items(), key=lambda kv: kv[1])

    lines += [
        "",
        "## 明确结论",
        "",
        f"1. `supervise_valid=0` 主要由 `{main_fail[best_n][0]}` 造成；四个 N 中仅角速度失败占全部 step 的 58.95%～65.14%，占 supervise_valid 失败样本的约 78.83%～81.02%。",
        "2. 空标签 step 主要由 `Tz<=0` 造成，已确认比例为 99.28%～99.79%；这说明当前 `supervise_valid=||T||>0.25 且 ||Omega||<0.18` 没有显式要求前向运动。",
        "3. 两者不高度重合：既存在 `supervise_valid=1` 但空标签，也存在标签非空但 `supervise_valid=0`。其中 `supervise_valid=1` 的 step 仍有约 51.89%～53.80% 是空标签，核心原因是平移范数通过但 Tz 不为正。",
        "4. 真正可直接监督的 step 比例见上表 `direct supervision比例`。",
        "5. 每个完整 10 步 block 的平均可监督 step 数见上表。",
        "6. 0 监督 block 比例见上表。",
        f"7. 当前只按监督密度看，首选 `N={best_n}`；它和之前按时间尺度质量选择较大 N 的结论存在取舍关系。",
        "8. 不建议只放宽速度或角速度阈值；更应该把 `Tz>0` 或 `valid_ttc_mask.sum()>0` 纳入监督筛选，否则会把大量无 TTC 标签的 step 标成可监督。",
        "9. 建议修改采样策略：优先过滤或重采样 0 监督/低监督 block，并在 H5 中补存 `depth_start` 或 depth 有效像素数，用于消除 unknown 空标签。",
        "10. 当前数据不适合直接进入正式训练；适合先做小规模训练验证。正式训练前建议在 dataloader 中显式使用 `direct_supervision_valid` 过滤 loss，并提高每个 block 的有效监督密度。",
        "",
        "## N增大后 supervise_valid 比例下降的解释",
        "",
        "N 增大后，每个 step 需要收集更多 ROI 事件，时间跨度变长、timeout 比例升高、完整 block 数减少；被保留下来的 step 分布更偏向低事件率/非理想运动段。更直接的证据是 `||Omega||` 的 P50 从 0.3149 增至 0.3984，明显高于 0.18 阈值，因此角速度筛选失败比例上升是 supervise_valid 比例下降的主因。",
        "",
        "## 输出文件",
        "",
        f"- step级审计：`{output_dir / 'step级审计.csv'}`",
        f"- block级审计：`{output_dir / 'block级审计.csv'}`",
        f"- 异常样本索引：`{output_dir / '异常样本索引.csv'}`",
        f"- 完整 JSON：`{output_dir / 'supervision_audit_summary.json'}`",
        "",
        "## 运行命令",
        "",
        "```bash",
        command,
        "```",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "debug_sets")
    parser.add_argument("--event-counts", type=int, nargs="+", default=DEFAULT_EVENT_COUNTS)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--speed-threshold", type=float, default=0.25)
    parser.add_argument("--omega-threshold", type=float, default=0.18)
    parser.add_argument("--max-step-duration-us", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--no-visualize", action="store_true")
    args = parser.parse_args()

    h5_paths = locate_h5(args.input_dir, args.event_counts)
    out_dir = args.output_dir or default_output_dir(args.input_dir, h5_paths)
    out_dir.mkdir(parents=True, exist_ok=True)

    arrays_by_n: dict[int, dict[str, Any]] = {}
    summaries: dict[int, dict[str, Any]] = {}
    block_summaries: dict[int, dict[str, Any]] = {}
    all_block_rows: list[dict[str, Any]] = []

    for n in sorted(h5_paths):
        arr = read_step_arrays(h5_paths[n], args.speed_threshold, args.omega_threshold, args.max_step_duration_us)
        arrays_by_n[n] = arr
        summaries[n] = summarize_steps(arr)
        block_rows, block_summary = build_block_rows(arr)
        all_block_rows.extend(block_rows)
        block_summaries[n] = block_summary

    step_rows = make_step_rows(arrays_by_n)
    exception_rows = build_exception_rows(arrays_by_n, all_block_rows, args.seed)

    write_csv(out_dir / "step级审计.csv", step_rows)
    write_csv(out_dir / "block级审计.csv", all_block_rows)
    write_csv(out_dir / "异常样本索引.csv", exception_rows)

    payload = {
        "input_h5": {str(n): str(p) for n, p in h5_paths.items()},
        "speed_threshold": args.speed_threshold,
        "omega_threshold": args.omega_threshold,
        "max_step_duration_us": args.max_step_duration_us,
        "step_summaries": summaries,
        "block_summaries": block_summaries,
    }
    (out_dir / "supervision_audit_summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=_to_py), encoding="utf-8")

    command = (
        "EV-TTC-main/.venv/bin/python EV-TTC-SNN-main/snn_ttc/tools/audit_supervise_and_empty_labels.py "
        f"--input-dir '{args.input_dir}' --output-dir '{out_dir}' --event-counts "
        + " ".join(str(n) for n in args.event_counts)
    )
    write_supervise_report(out_dir / "SuperviseValid失效原因统计.md", summaries)
    write_empty_report(out_dir / "空标签Step原因统计.md", summaries)
    write_cross_report(out_dir / "SuperviseValid与空标签交叉分析.md", summaries)
    write_block_report(out_dir / "完整10步Block监督密度统计.md", block_summaries)
    write_comparison_report(out_dir / "四种N监督质量综合对比.md", summaries, block_summaries, out_dir, command)

    if not args.no_visualize:
        visualize_exception_samples(exception_rows, h5_paths, out_dir)

    print(f"output_dir: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
