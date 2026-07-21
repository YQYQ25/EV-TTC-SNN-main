#!/usr/bin/env python3
"""审计 car 固定事件数 ROI-TTC 数据，并生成与官方 EV-TTC 的中文对比报告。

本脚本只读取已经生成的 H5，不启动训练；固定 10ms 对齐比较在同一 ROI
和同一 t_start 上离线计算，用来隔离“事件窗口”和“官方 10ms 窗口”的差异。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import h5py
import hdf5plugin  # noqa: F401
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ROOT = Path(__file__).resolve().parents[3]
SNN_ROOT = ROOT / "EV-TTC-SNN-main"
if str(SNN_ROOT) not in sys.path:
    sys.path.insert(0, str(SNN_ROOT))

from snn_ttc.tools.build_skatepark_multi_n_ttc import compute_motion_from_pose  # noqa: E402
from snn_ttc.tools.make_1k_ttc_debug_set import resolve_sequence_paths  # noqa: E402


EVENT_COUNTS = [5000, 10000, 15000, 20000]
PIXELS = 128 * 128


def pct(num: float, den: float) -> float:
    return 100.0 * float(num) / float(den) if den else 0.0


def fmt(value: Any, digits: int = 4) -> str:
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


def stats(values: np.ndarray) -> dict[str, Any]:
    """返回常用分位数统计，空数组用 None 表示。"""

    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None, "p1": None, "p5": None, "p25": None, "p50": None, "p75": None, "p95": None, "p99": None}
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "p1": float(np.percentile(arr, 1)),
        "p5": float(np.percentile(arr, 5)),
        "p25": float(np.percentile(arr, 25)),
        "p50": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
    }


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
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(str(v) for v in row) + " |" for row in rows)
    return "\n".join(lines)


def locate_h5(input_dir: Path, sequence: str, event_counts: list[int]) -> dict[int, Path]:
    """在 H5 目录内查找 car_..._N*.h5。"""

    out: dict[int, Path] = {}
    for n in event_counts:
        candidates = sorted(input_dir.rglob(f"*N{n}.h5"))
        candidates = [p for p in candidates if not p.name.startswith("audit_subset_") and ".tmp" not in p.name]
        if candidates:
            seq_candidates = [p for p in candidates if sequence in p.name]
            out[n] = (seq_candidates or candidates)[0]
    return out


def read_arrays(path: Path) -> dict[str, Any]:
    """读取单个 N 的标量和 mask，用于统计。"""

    with h5py.File(path, "r") as f:
        n_steps = int(f["step_index"].shape[0])
        T = f["T"][:].astype(np.float64)
        Omega = f["Omega"][:].astype(np.float64)
        valid_mask_ratio = f["valid_mask_ratio"][:].astype(np.float64)
        valid_count = np.rint(valid_mask_ratio * PIXELS).astype(np.int64)
        pose_valid = f["pose_valid"][:].astype(bool) if "pose_valid" in f else np.ones(n_steps, dtype=bool)
        speed_valid = f["speed_valid"][:].astype(bool)
        omega_valid = f["omega_valid"][:].astype(bool)
        supervise_valid = f["supervise_valid"][:].astype(bool)
        direct = f["direct_supervision_valid"][:].astype(bool) if "direct_supervision_valid" in f else supervise_valid & (valid_count > 0)
        official_style_empty = None
        if "official_style_mask" in f:
            official_style_empty = []
            for s in range(0, n_steps, 256):
                m = f["official_style_mask"][s : s + 256].astype(bool)
                official_style_empty.append(np.sum(m.reshape(m.shape[0], -1), axis=1) == 0)
            official_style_empty = np.concatenate(official_style_empty)
        timeouts = int(f.attrs.get("timeout_count", f["timeouts/block_index"].shape[0] if "timeouts/block_index" in f else 0))
        attrs = dict(f.attrs)
        arr = {
            "path": path,
            "N": int(attrs.get("N", int(path.stem.split("N")[-1]))),
            "sequence_name": attrs.get("sequence_name", "unknown").decode("utf-8") if isinstance(attrs.get("sequence_name", ""), bytes) else attrs.get("sequence_name", "unknown"),
            "steps": n_steps,
            "step_index": f["step_index"][:].astype(np.int64),
            "block_index": f["block_index"][:].astype(np.int64),
            "step_in_block": f["step_in_block"][:].astype(np.int16),
            "roi_x0": f["roi_x0"][:].astype(np.int16),
            "roi_y0": f["roi_y0"][:].astype(np.int16),
            "t_start": f["t_start"][:].astype(np.int64),
            "t_end": f["t_end"][:].astype(np.int64),
            "event_dt": f["event_dt"][:].astype(np.int64),
            "T": T,
            "Omega": Omega,
            "T_norm": np.linalg.norm(T, axis=1),
            "Tz": T[:, 2],
            "Omega_norm": np.linalg.norm(Omega, axis=1),
            "speed_valid": speed_valid,
            "omega_valid": omega_valid,
            "pose_valid": pose_valid,
            "supervise_valid": supervise_valid,
            "direct_supervision_valid": direct,
            "valid_mask_ratio": valid_mask_ratio,
            "valid_pixel_count": valid_count,
            "min_ttc": f["min_ttc"][:].astype(np.float64) if "min_ttc" in f else np.full(n_steps, np.nan),
            "timeout_count": timeouts,
            "complete_blocks_attr": int(attrs.get("complete_blocks", 0)),
            "incomplete_blocks_attr": int(attrs.get("incomplete_blocks", 0)),
            "elapsed_seconds": float(attrs.get("elapsed_seconds", np.nan)),
            "size_bytes": path.stat().st_size,
            "official_style_empty": official_style_empty,
        }
    return arr


def block_summary(arr: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """统计完整 10 步 block 的 direct supervision 密度。"""

    rows: list[dict[str, Any]] = []
    block_to_idx: dict[int, list[int]] = defaultdict(list)
    for i, b in enumerate(arr["block_index"]):
        block_to_idx[int(b)].append(i)

    direct_counts = []
    durations = []
    for block, idx_list in sorted(block_to_idx.items()):
        idx = np.asarray(sorted(idx_list, key=lambda j: int(arr["step_in_block"][j])), dtype=np.int64)
        complete = len(idx) == 10 and set(arr["step_in_block"][idx].tolist()) == set(range(10))
        direct_count = int(np.sum(arr["direct_supervision_valid"][idx]))
        row = {
            "N": arr["N"],
            "block_index": block,
            "complete_10step_block": complete,
            "num_steps": int(len(idx)),
            "first_step_index": int(arr["step_index"][idx[0]]),
            "last_step_index": int(arr["step_index"][idx[-1]]),
            "roi_x0": int(arr["roi_x0"][idx[0]]),
            "roi_y0": int(arr["roi_y0"][idx[0]]),
            "block_duration_us": int(arr["t_end"][idx[-1]] - arr["t_start"][idx[0]]),
            "num_direct_supervision_valid": direct_count,
            "num_supervise_valid": int(np.sum(arr["supervise_valid"][idx])),
            "num_nonempty_label": int(np.sum(arr["valid_pixel_count"][idx] > 0)),
        }
        rows.append(row)
        if complete:
            direct_counts.append(direct_count)
            durations.append(row["block_duration_us"])

    direct_counts_arr = np.asarray(direct_counts, dtype=np.int64)
    hist = {str(i): int(np.sum(direct_counts_arr == i)) for i in range(11)}
    total_complete = int(len(direct_counts_arr))
    summary = {
        "total_blocks": len(rows),
        "complete_10step_blocks": total_complete,
        "incomplete_blocks": int(len(rows) - total_complete),
        "complete_block_ratio_percent": pct(total_complete, len(rows)),
        "direct_hist_0_to_10": hist,
        "zero_direct_block_ratio_percent": pct(np.sum(direct_counts_arr == 0), total_complete),
        "one_to_three_direct_block_ratio_percent": pct(np.sum((direct_counts_arr >= 1) & (direct_counts_arr <= 3)), total_complete),
        "four_to_six_direct_block_ratio_percent": pct(np.sum((direct_counts_arr >= 4) & (direct_counts_arr <= 6)), total_complete),
        "seven_to_nine_direct_block_ratio_percent": pct(np.sum((direct_counts_arr >= 7) & (direct_counts_arr <= 9)), total_complete),
        "ten_direct_block_ratio_percent": pct(np.sum(direct_counts_arr == 10), total_complete),
        "direct_per_block_stats": stats(direct_counts_arr.astype(np.float64)),
        "block_duration_stats": stats(np.asarray(durations, dtype=np.float64)),
    }
    return rows, summary


def summarize_one(arr: dict[str, Any], block_stats: dict[str, Any]) -> dict[str, Any]:
    """汇总单个 N 的核心指标。"""

    n = arr["steps"]
    positive_nonempty = arr["valid_pixel_count"] > 0
    min_ttc = arr["min_ttc"]
    official_empty = arr["official_style_empty"]
    official_empty_rate = pct(np.count_nonzero(official_empty), n) if official_empty is not None else None
    return {
        "N": arr["N"],
        "path": str(arr["path"]),
        "candidate_steps": n,
        "event_dt_stats_us": stats(arr["event_dt"]),
        "event_dt_lt_0_5ms_percent": pct(np.count_nonzero(arr["event_dt"] < 500), n),
        "event_dt_lt_1ms_percent": pct(np.count_nonzero(arr["event_dt"] < 1000), n),
        "event_dt_lt_3_3ms_percent": pct(np.count_nonzero(arr["event_dt"] < 3300), n),
        "event_dt_lt_7ms_percent": pct(np.count_nonzero(arr["event_dt"] < 7000), n),
        "event_dt_le_10ms_percent": pct(np.count_nonzero(arr["event_dt"] <= 10000), n),
        "speed_pass_rate_percent": pct(np.count_nonzero(arr["speed_valid"]), n),
        "omega_pass_rate_percent": pct(np.count_nonzero(arr["omega_valid"]), n),
        "Tz_positive_rate_percent": pct(np.count_nonzero(arr["pose_valid"] & (arr["Tz"] > 0)), np.count_nonzero(arr["pose_valid"])),
        "negative_ttc_sample_rate_percent": pct(np.count_nonzero(arr["pose_valid"] & (arr["Tz"] <= 0)), np.count_nonzero(arr["pose_valid"])),
        "positive_ttc_valid_step_rate_percent": pct(np.count_nonzero(positive_nonempty), n),
        "direct_supervision_rate_percent": pct(np.count_nonzero(arr["direct_supervision_valid"]), n),
        "timeout_rate_percent": pct(arr["timeout_count"], arr["timeout_count"] + n),
        "complete_block_ratio_percent": block_stats["complete_block_ratio_percent"],
        "zero_direct_block_ratio_percent": block_stats["zero_direct_block_ratio_percent"],
        "ten_direct_block_ratio_percent": block_stats["ten_direct_block_ratio_percent"],
        "mask_valid_pixel_rate_percent": 100.0 * float(np.mean(arr["valid_mask_ratio"])) if n else 0.0,
        "official_style_mask_empty_rate_percent": official_empty_rate,
        "TTC_lt_0_5s_step_rate_percent": pct(np.count_nonzero(np.isfinite(min_ttc) & (min_ttc < 0.5)), n),
        "TTC_lt_1s_step_rate_percent": pct(np.count_nonzero(np.isfinite(min_ttc) & (min_ttc < 1.0)), n),
        "TTC_lt_2s_step_rate_percent": pct(np.count_nonzero(np.isfinite(min_ttc) & (min_ttc < 2.0)), n),
        "TTC_lt_3s_step_rate_percent": pct(np.count_nonzero(np.isfinite(min_ttc) & (min_ttc < 3.0)), n),
        "TTC_lt_5s_step_rate_percent": pct(np.count_nonzero(np.isfinite(min_ttc) & (min_ttc < 5.0)), n),
        "inverse_TTC_gt_0_2_step_rate_percent": pct(np.count_nonzero(np.isfinite(min_ttc) & (min_ttc > 0) & ((1.0 / min_ttc) > 0.2)), n),
        "inverse_TTC_gt_0_5_step_rate_percent": pct(np.count_nonzero(np.isfinite(min_ttc) & (min_ttc > 0) & ((1.0 / min_ttc) > 0.5)), n),
        "inverse_TTC_gt_1_0_step_rate_percent": pct(np.count_nonzero(np.isfinite(min_ttc) & (min_ttc > 0) & ((1.0 / min_ttc) > 1.0)), n),
        "T_norm_stats": stats(arr["T_norm"]),
        "Tz_stats": stats(arr["Tz"]),
        "Omega_norm_stats": stats(arr["Omega_norm"]),
        "h5_size_bytes": arr["size_bytes"],
        "elapsed_seconds": arr["elapsed_seconds"],
    }


def fixed10_compare(arr: dict[str, Any], pose_mats: np.ndarray, pose_ts: np.ndarray) -> dict[str, Any]:
    """同一 ROI 和 t_start 下比较事件窗口 T/Omega 与固定 10ms T/Omega/标签。"""

    rows = []
    pixel_inv_abs = []
    pixel_inv_rel = []
    pixel_ttc_abs = []
    t_norm_diff = []
    omega_norm_diff = []
    mask_iou = []
    with h5py.File(arr["path"], "r") as f:
        for s in range(0, arr["steps"], 256):
            e = min(arr["steps"], s + 256)
            t_start = arr["t_start"][s:e]
            t_end_fixed = t_start + 10_000
            T10, Omega10, valid10 = compute_motion_from_pose(pose_mats, pose_ts, t_start, t_end_fixed)
            T = arr["T"][s:e]
            Omega = arr["Omega"][s:e]
            t_norm_diff.extend(np.linalg.norm(T10 - T, axis=1)[valid10].tolist())
            omega_norm_diff.extend(np.linalg.norm(Omega10 - Omega, axis=1)[valid10].tolist())
            inv = f["inverse_ttc_start"][s:e]
            mask = f["valid_ttc_mask"][s:e].astype(bool)
            for i in range(e - s):
                cur_mask = mask[i] & valid10[i] & (T10[i, 2] > 0) & (T[i, 2] > 0) & (inv[i] > 0)
                if not np.any(cur_mask):
                    continue
                # depth = Tz_event / inverse_TTC_event，只在事件窗口正 TTC 有效像素上比较。
                depth = T[i, 2] / np.maximum(inv[i][cur_mask], 1e-6)
                inv10 = T10[i, 2] / np.maximum(depth, 1e-6)
                inv_event = inv[i][cur_mask]
                pixel_inv_abs.append(np.abs(inv10 - inv_event))
                pixel_inv_rel.append(np.abs(inv10 - inv_event) / np.maximum(np.abs(inv_event), 1e-6))
                ttc_event = 1.0 / np.maximum(inv_event, 1e-6)
                ttc10 = depth / (T10[i, 2] + 1e-5)
                pixel_ttc_abs.append(np.abs(ttc10 - ttc_event))
                fixed_mask = mask[i] & valid10[i] & (T10[i, 2] > 0)
                union = mask[i] | fixed_mask
                if np.any(union):
                    mask_iou.append(float(np.count_nonzero(mask[i] & fixed_mask) / np.count_nonzero(union)))

    inv_abs = np.concatenate(pixel_inv_abs) if pixel_inv_abs else np.asarray([], dtype=np.float32)
    inv_rel = np.concatenate(pixel_inv_rel) if pixel_inv_rel else np.asarray([], dtype=np.float32)
    ttc_abs = np.concatenate(pixel_ttc_abs) if pixel_ttc_abs else np.asarray([], dtype=np.float32)
    return {
        "N": arr["N"],
        "T_diff_norm_stats": stats(np.asarray(t_norm_diff)),
        "Omega_diff_norm_stats": stats(np.asarray(omega_norm_diff)),
        "inverse_TTC_MAE": float(np.mean(inv_abs)) if inv_abs.size else None,
        "inverse_TTC_MRE_percent": 100.0 * float(np.mean(inv_rel)) if inv_rel.size else None,
        "TTC_MAE": float(np.mean(ttc_abs)) if ttc_abs.size else None,
        "mask_iou_mean": float(np.mean(mask_iou)) if mask_iou else None,
        "common_positive_pixels": int(inv_abs.size),
        "note": "像素标签比较只在事件窗口 positive_ttc_mask 有效像素上重构 depth 后进行。",
    }


def save_visuals(arr: dict[str, Any], out_dir: Path, max_blocks: int, seed: int) -> int:
    """为每种 N 抽取若干完整 block 保存可视化。"""

    rng = random.Random(seed + int(arr["N"]))
    out_dir.mkdir(parents=True, exist_ok=True)
    block_to_idx: dict[int, list[int]] = defaultdict(list)
    for i, b in enumerate(arr["block_index"]):
        block_to_idx[int(b)].append(i)
    complete = []
    for b, idxs in block_to_idx.items():
        idx = sorted(idxs, key=lambda j: int(arr["step_in_block"][j]))
        if len(idx) == 10 and set(arr["step_in_block"][idx].tolist()) == set(range(10)):
            complete.append((b, idx))
    if len(complete) > max_blocks:
        complete = rng.sample(complete, max_blocks)
    saved = 0
    with h5py.File(arr["path"], "r") as f:
        for block, idxs in complete:
            for local, i in enumerate(idxs):
                event_cnt = f["event_cnt"][i]
                inv = f["inverse_ttc_start"][i]
                pos_mask = f["valid_ttc_mask"][i].astype(bool)
                official_mask = f["official_style_mask"][i].astype(bool) if "official_style_mask" in f else pos_mask
                ttc = np.full_like(inv, np.nan, dtype=np.float32)
                ttc[pos_mask] = 1.0 / np.maximum(inv[pos_mask], 1e-6)
                depth = np.full_like(inv, np.nan, dtype=np.float32)
                if arr["Tz"][i] > 0:
                    depth[pos_mask] = arr["Tz"][i] / np.maximum(inv[pos_mask], 1e-6)
                fig, axes = plt.subplots(2, 4, figsize=(15, 7), constrained_layout=True)
                panels = [
                    ("正事件", event_cnt[0], "magma"),
                    ("负事件", event_cnt[1], "magma"),
                    ("正-负", event_cnt[0] - event_cnt[1], "coolwarm"),
                    ("depth(正TTC像素重构)", depth, "viridis"),
                    ("TTC", ttc, "viridis"),
                    ("inverse TTC", inv, "inferno"),
                    ("positive mask", pos_mask.astype(float), "gray"),
                    ("official-style mask", official_mask.astype(float), "gray"),
                ]
                for ax, (title, img, cmap) in zip(axes.flat, panels):
                    vals = img[np.isfinite(img)]
                    if vals.size and cmap != "gray":
                        vmin, vmax = np.percentile(vals, [2, 98])
                        if np.isclose(vmin, vmax):
                            vmax = vmin + 1.0
                    else:
                        vmin, vmax = (0, 1) if cmap == "gray" else (None, None)
                    im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
                    ax.set_title(title, fontsize=8)
                    ax.axis("off")
                    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
                fig.suptitle(
                    f"N={arr['N']} block={block} step={local} dt={int(arr['event_dt'][i])}us "
                    f"Tz={arr['Tz'][i]:.3f} ||T||={arr['T_norm'][i]:.3f} ||Omega||={arr['Omega_norm'][i]:.3f} "
                    f"direct={bool(arr['direct_supervision_valid'][i])}",
                    fontsize=9,
                )
                path = out_dir / f"N{arr['N']}_block{block:06d}_step{local:02d}.png"
                fig.savefig(path, dpi=130)
                plt.close(fig)
                saved += 1
    return saved


def load_official_summary(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def rank_configs(summaries: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    """根据监督密度、timeout、block连续性和标签覆盖率给 N 排序。"""

    rows = []
    for n, s in summaries.items():
        score = (
            s["direct_supervision_rate_percent"]
            + 0.2 * s["mask_valid_pixel_rate_percent"]
            + 0.2 * s["complete_block_ratio_percent"]
            - s["zero_direct_block_ratio_percent"]
            - s["timeout_rate_percent"]
        )
        rows.append({"N": n, "score": score, "reason": "综合 direct supervision、mask、完整 block、timeout 与 0监督 block。"})
    return sorted(rows, key=lambda r: r["score"], reverse=True)


def write_reports(
    out_dir: Path,
    summaries: dict[int, dict[str, Any]],
    block_summaries: dict[int, dict[str, Any]],
    fixed10: dict[int, dict[str, Any]],
    official: dict[str, Any] | None,
    skatepark: dict[str, Any] | None,
    skatepark_roi15: dict[str, Any] | None,
    ranking: list[dict[str, Any]],
    runtime: dict[str, Any],
) -> None:
    reports = out_dir / "reports"
    reports.mkdir(parents=True, exist_ok=True)

    gen_rows = []
    for n, s in summaries.items():
        gen_rows.append([n, s["candidate_steps"], fmt(s["event_dt_stats_us"]["p50"]), fmt(s["event_dt_stats_us"]["p95"]), f"{s['timeout_rate_percent']:.2f}%", f"{s['direct_supervision_rate_percent']:.2f}%", f"{s['h5_size_bytes'] / 1024**3:.2f} GiB", fmt(s["elapsed_seconds"], 2)])
    (reports / "Car多事件数TTC数据生成报告.md").write_text(
        "# Car多事件数TTC数据生成报告\n\n"
        + markdown_table(["N", "候选step数", "event_dt P50(us)", "event_dt P95(us)", "timeout比例", "direct supervision比例", "H5大小", "生成耗时(s)"], gen_rows)
        + f"\n\n输出目录：`{out_dir}`\n",
        encoding="utf-8",
    )

    fixed_rows = []
    for n, s in fixed10.items():
        fixed_rows.append([n, fmt(s["T_diff_norm_stats"]["p50"]), fmt(s["Omega_diff_norm_stats"]["p50"]), fmt(s["inverse_TTC_MAE"]), fmt(s["inverse_TTC_MRE_percent"]), fmt(s["TTC_MAE"]), fmt(s["mask_iou_mean"]), s["common_positive_pixels"]])
    (reports / "Car事件窗口与固定10ms标签对比.md").write_text(
        "# Car事件窗口与固定10ms标签对比\n\n"
        + markdown_table(["N", "T差异P50", "Omega差异P50", "inverse TTC MAE", "inverse TTC MRE(%)", "TTC MAE", "mask IoU均值", "共同正TTC像素"], fixed_rows)
        + "\n\n注：像素级比较只在事件窗口正TTC有效像素上重构深度后进行，用于观察时间窗口差异，不替代完整重投影标签。\n",
        encoding="utf-8",
    )

    block_rows = []
    for n, b in block_summaries.items():
        block_rows.append([n, b["complete_10step_blocks"], b["incomplete_blocks"], f"{b['complete_block_ratio_percent']:.2f}%", f"{b['zero_direct_block_ratio_percent']:.2f}%", f"{b['one_to_three_direct_block_ratio_percent']:.2f}%", f"{b['four_to_six_direct_block_ratio_percent']:.2f}%", f"{b['seven_to_nine_direct_block_ratio_percent']:.2f}%", f"{b['ten_direct_block_ratio_percent']:.2f}%", fmt(b["direct_per_block_stats"]["mean"])])
    (reports / "Car多事件数监督密度统计.md").write_text(
        "# Car多事件数监督密度统计\n\n"
        + markdown_table(["N", "完整10步block", "不完整block", "完整block比例", "0监督block", "1-3监督block", "4-6监督block", "7-9监督block", "10监督block", "平均监督step/block"], block_rows),
        encoding="utf-8",
    )

    official_row = None
    if official:
        off = official["official"]
        pix = official["pixel_groups"]["official_valid"]
        official_row = {
            "candidate_steps": off["candidate_count"],
            "event_dt_p50": "10 ms",
            "event_dt_p95": "10 ms",
            "speed_pass": off["speed_valid_rate_in_boundary_percent"],
            "omega_pass": off["omega_valid_rate_in_boundary_percent"],
            "Tz_pos": off["Tz_positive_rate_percent"],
            "positive_valid": pix["positive_ttc_remaining_sample_ratio_percent"],
            "direct": off["official_keep_rate_percent"],
            "timeout": "—",
            "complete": "—",
            "zero_block": "—",
            "ten_block": "—",
            "mask": pct(pix["positive_ttc_mask_pixels"], pix["sample_count"] * 360 * 360),
            "ttc_lt1": "NA",
            "neg": pix["negative_ttc_any_sample_ratio_percent"],
            "size": "见官方exp/ttcef",
            "elapsed": "见官方日志",
        }
    compare_rows = []
    for n, s in summaries.items():
        compare_rows.append([n, s["candidate_steps"], fmt(s["event_dt_stats_us"]["p50"]), fmt(s["event_dt_stats_us"]["p95"]), f"{s['speed_pass_rate_percent']:.2f}%", f"{s['omega_pass_rate_percent']:.2f}%", f"{s['Tz_positive_rate_percent']:.2f}%", f"{s['positive_ttc_valid_step_rate_percent']:.2f}%", f"{s['direct_supervision_rate_percent']:.2f}%", f"{s['timeout_rate_percent']:.2f}%", f"{s['complete_block_ratio_percent']:.2f}%", f"{s['zero_direct_block_ratio_percent']:.2f}%", f"{s['ten_direct_block_ratio_percent']:.2f}%", f"{s['mask_valid_pixel_rate_percent']:.2f}%", f"{s['TTC_lt_1s_step_rate_percent']:.2f}%", f"{s['negative_ttc_sample_rate_percent']:.2f}%", f"{s['h5_size_bytes'] / 1024**3:.2f} GiB", fmt(s["elapsed_seconds"], 2)])
    if official_row:
        compare_rows.append(["官方EV-TTC", official_row["candidate_steps"], official_row["event_dt_p50"], official_row["event_dt_p95"], f"{official_row['speed_pass']:.2f}%", f"{official_row['omega_pass']:.2f}%", f"{official_row['Tz_pos']:.2f}%", f"{official_row['positive_valid']:.2f}%", f"{official_row['direct']:.2f}%", official_row["timeout"], official_row["complete"], official_row["zero_block"], official_row["ten_block"], f"{official_row['mask']:.2f}%", official_row["ttc_lt1"], f"{official_row['neg']:.2f}%", official_row["size"], official_row["elapsed"]])
    (reports / "Car固定事件数与官方EVTTC综合对比.md").write_text(
        "# Car固定事件数与官方EVTTC综合对比\n\n"
        + markdown_table(["指标/方案", "候选step数", "event_dt P50", "event_dt P95", "速度通过率", "角速度通过率", "Tz>0比例", "正TTC有效step比例", "direct supervision比例", "timeout比例", "完整block比例", "0监督block比例", "10监督block比例", "mask有效像素率", "TTC<1s step比例", "负TTC样本比例", "H5大小", "运行时间"], compare_rows)
        + "\n\n## 排序\n\n"
        + markdown_table(["排名", "N", "综合分", "说明"], [[i + 1, r["N"], fmt(r["score"]), r["reason"]] for i, r in enumerate(ranking)])
        + f"\n\n正式训练推荐配置：优先使用 `N={ranking[0]['N']}`，若显存或训练速度受限，再考虑排名第二的 N。\n",
        encoding="utf-8",
    )

    skate_rows = []
    if skatepark_roi15:
        skate_rows.append(
            [
                "Skatepark 15k",
                f"{skatepark_roi15['speed_pass_rate_percent']:.2f}%",
                f"{skatepark_roi15['omega_pass_rate_percent']:.2f}%",
                f"{skatepark_roi15['Tz_positive_rate_percent']:.2f}%",
                f"{skatepark_roi15['positive_ttc_valid_step_rate_percent']:.2f}%",
                f"{skatepark_roi15['direct_supervision_rate_percent']:.2f}%",
                f"{skatepark_roi15['zero_direct_block_ratio_percent']:.2f}%",
                f"{skatepark_roi15['TTC_lt_1s_step_rate_percent']:.2f}%",
            ]
        )
    car15 = summaries.get(15000)
    if car15:
        skate_rows.append(["Car 15k", f"{car15['speed_pass_rate_percent']:.2f}%", f"{car15['omega_pass_rate_percent']:.2f}%", f"{car15['Tz_positive_rate_percent']:.2f}%", f"{car15['positive_ttc_valid_step_rate_percent']:.2f}%", f"{car15['direct_supervision_rate_percent']:.2f}%", f"{car15['zero_direct_block_ratio_percent']:.2f}%", f"{car15['TTC_lt_1s_step_rate_percent']:.2f}%"])
    if skatepark:
        try:
            sp_off = skatepark["official"]
            sp_pix = skatepark["pixel_groups"]["official_valid"]
            skate_rows.append(["Skatepark官方EV-TTC", f"{sp_off['speed_valid_rate_in_boundary_percent']:.2f}%", f"{sp_off['omega_valid_rate_in_boundary_percent']:.2f}%", f"{sp_off['Tz_positive_rate_percent']:.2f}%", f"{sp_pix['positive_ttc_remaining_sample_ratio_percent']:.2f}%", f"{sp_off['official_keep_rate_percent']:.2f}%", "—", "NA"])
        except Exception:
            pass
    if official:
        off = official["official"]
        pix = official["pixel_groups"]["official_valid"]
        skate_rows.append(["Car官方EV-TTC", f"{off['speed_valid_rate_in_boundary_percent']:.2f}%", f"{off['omega_valid_rate_in_boundary_percent']:.2f}%", f"{off['Tz_positive_rate_percent']:.2f}%", f"{pix['positive_ttc_remaining_sample_ratio_percent']:.2f}%", f"{off['official_keep_rate_percent']:.2f}%", "—", "NA"])
    (reports / "Car与Skatepark对比报告.md").write_text(
        "# Car与Skatepark对比报告\n\n"
        + markdown_table(["方案", "速度通过率", "角速度通过率", "Tz>0比例", "正TTC有效率", "direct/官方保留比例", "0监督block比例", "TTC<1s比例"], skate_rows)
        + "\n\n判断：若 Car 的角速度通过率、Tz>0 和 direct supervision 明显高于 Skatepark，则 Skatepark 监督稀疏更可能来自轨迹运动特性，而不是实现错误。\n",
        encoding="utf-8",
    )

    (reports / "运行时间和磁盘占用.md").write_text(
        "# 运行时间和磁盘占用\n\n"
        + f"- 开始时间：{runtime['started_at']}\n"
        + f"- 结束时间：{runtime['finished_at']}\n"
        + f"- 审计耗时：{runtime['elapsed_seconds']:.2f} 秒\n"
        + f"- 输出目录：`{out_dir}`\n"
        + f"- 输出目录占用：{runtime['out_dir_size_mb']:.2f} MiB\n"
        + f"- `/data` 剩余：{runtime['data_free_gib']:.2f} GiB\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", default="car_urban_night_rittenhouse")
    parser.add_argument("--root", type=Path, default=Path("/data/M3ED"))
    parser.add_argument("--input-dir", type=Path, default=ROOT / "EV-TTC-SNN-main/debug_sets/car_urban_night_rittenhouse_multi_n_ttc/H5")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "EV-TTC-SNN-main/debug_sets/car_urban_night_rittenhouse_multi_n_ttc")
    parser.add_argument("--official-summary", type=Path, default=ROOT / "EV-TTC-SNN-main/debug_sets/car_urban_night_rittenhouse_multi_n_ttc/official_evttc_audit/audit_summary.json")
    parser.add_argument("--skatepark-summary", type=Path, default=ROOT / "EV-TTC-SNN-main/analysis/官方EVTTC_Skatepark筛选与负TTC统计/audit_summary.json")
    parser.add_argument("--skatepark-roi15-h5", type=Path, default=ROOT / "EV-TTC-SNN-main/debug_sets/探索1k，5k...20k输入的数据质量如何/H5/skatepark_N15000.h5")
    parser.add_argument("--event-counts", type=int, nargs="+", default=EVENT_COUNTS)
    parser.add_argument("--visual-blocks", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    started = time.time()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_dir = args.out_dir / "csv"
    h5_paths = locate_h5(args.input_dir, args.sequence, args.event_counts)
    missing = [n for n in args.event_counts if n not in h5_paths]
    if missing:
        raise FileNotFoundError(f"缺少 H5: {missing}; input_dir={args.input_dir}")

    arrays = {n: read_arrays(path) for n, path in sorted(h5_paths.items())}
    block_rows_all: list[dict[str, Any]] = []
    step_rows_all: list[dict[str, Any]] = []
    summaries: dict[int, dict[str, Any]] = {}
    block_summaries: dict[int, dict[str, Any]] = {}
    for n, arr in arrays.items():
        block_rows, bsum = block_summary(arr)
        block_rows_all.extend(block_rows)
        block_summaries[n] = bsum
        summaries[n] = summarize_one(arr, bsum)
        for i in range(arr["steps"]):
            step_rows_all.append(
                {
                    "N": n,
                    "step_index": int(arr["step_index"][i]),
                    "block_index": int(arr["block_index"][i]),
                    "step_in_block": int(arr["step_in_block"][i]),
                    "event_dt": int(arr["event_dt"][i]),
                    "T_norm": float(arr["T_norm"][i]),
                    "Tz": float(arr["Tz"][i]),
                    "Omega_norm": float(arr["Omega_norm"][i]),
                    "speed_valid": bool(arr["speed_valid"][i]),
                    "omega_valid": bool(arr["omega_valid"][i]),
                    "pose_valid": bool(arr["pose_valid"][i]),
                    "supervise_valid": bool(arr["supervise_valid"][i]),
                    "direct_supervision_valid": bool(arr["direct_supervision_valid"][i]),
                    "valid_pixel_count": int(arr["valid_pixel_count"][i]),
                    "valid_mask_ratio": float(arr["valid_mask_ratio"][i]),
                    "min_ttc": float(arr["min_ttc"][i]) if np.isfinite(arr["min_ttc"][i]) else "",
                }
            )
    write_csv(csv_dir / "Car_step级统计.csv", step_rows_all)
    write_csv(csv_dir / "Car_block级统计.csv", block_rows_all)

    paths = resolve_sequence_paths(args.root, args.sequence)
    with h5py.File(paths["pose"], "r") as f_pose:
        pose_ts = f_pose["ts"][:].astype(np.int64)
        pose_mats = f_pose["Cn_T_C0"][:].astype(np.float64)
    fixed10 = {n: fixed10_compare(arr, pose_mats, pose_ts) for n, arr in arrays.items()}
    write_csv(csv_dir / "Car事件窗口与固定10ms对比.csv", [fixed10[n] for n in sorted(fixed10)])

    vis_count = 0
    for n, arr in arrays.items():
        vis_count += save_visuals(arr, args.out_dir / "visualizations" / f"N{n}", args.visual_blocks, args.seed)

    ranking = rank_configs(summaries)
    official = load_official_summary(args.official_summary)
    skatepark = load_official_summary(args.skatepark_summary)
    skatepark_roi15 = None
    if args.skatepark_roi15_h5.exists():
        sp_arr = read_arrays(args.skatepark_roi15_h5)
        _, sp_bsum = block_summary(sp_arr)
        skatepark_roi15 = summarize_one(sp_arr, sp_bsum)
    runtime = {
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(started)),
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_seconds": time.time() - started,
        "out_dir_size_mb": sum(p.stat().st_size for p in args.out_dir.rglob("*") if p.is_file()) / 1024**2,
        "data_free_gib": shutil.disk_usage("/data").free / 1024**3,
        "visualization_count": vis_count,
    }
    output = {
        "summaries": summaries,
        "block_summaries": block_summaries,
        "fixed10_comparison": fixed10,
        "skatepark_roi15_summary": skatepark_roi15,
        "ranking": ranking,
        "runtime": runtime,
    }
    (args.out_dir / "car_audit_summary.json").write_text(json.dumps(to_jsonable(output), ensure_ascii=False, indent=2), encoding="utf-8")
    write_reports(args.out_dir, summaries, block_summaries, fixed10, official, skatepark, skatepark_roi15, ranking, runtime)
    print(args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
