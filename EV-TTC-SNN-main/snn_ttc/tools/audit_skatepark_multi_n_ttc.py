#!/usr/bin/env python3
"""审计 Skatepark 多事件数 TTC H5，并生成固定 10ms 标签对比子集。"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from snn_ttc.tools.build_skatepark_multi_n_ttc import compute_motion_from_pose  # noqa: E402
from snn_ttc.tools.make_1k_ttc_debug_set import resolve_sequence_paths  # noqa: E402


EVENT_DT_BINS = [(0, 1000), (1000, 3300), (3300, 7000), (7000, 10000)]


def fmt(v: float, digits: int = 4) -> str:
    if v is None or not np.isfinite(v):
        return "nan"
    if v != 0 and abs(v) < 1e-3:
        return f"{v:.3e}"
    return f"{v:.{digits}f}"


def stats(arr: np.ndarray, percentiles=(1, 5, 25, 50, 75, 95, 99)) -> dict:
    arr = np.asarray(arr)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        out = {"count": 0, "min": np.nan, "max": np.nan, "mean": np.nan, "std": np.nan}
        out.update({f"p{p}": np.nan for p in percentiles})
        return out
    out = {
        "count": int(arr.size),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
    }
    out.update({f"p{p}": float(np.percentile(arr, p)) for p in percentiles})
    return out


def sample_complete_blocks(block_ids: np.ndarray, max_blocks: int, seed: int) -> np.ndarray:
    """从完整 block 中固定随机抽样，覆盖不同 ROI/时间段。"""

    rng = np.random.default_rng(seed)
    unique = np.unique(block_ids)
    if unique.size <= max_blocks:
        return unique
    # 均匀抽一部分首尾，再随机补足，避免只看序列开头。
    anchors = np.linspace(0, unique.size - 1, min(max_blocks // 2, unique.size), dtype=int)
    selected = set(unique[anchors].tolist())
    remaining = np.array([x for x in unique if x not in selected])
    need = max_blocks - len(selected)
    if need > 0 and remaining.size > 0:
        selected.update(rng.choice(remaining, size=min(need, remaining.size), replace=False).tolist())
    return np.asarray(sorted(selected), dtype=unique.dtype)


def complete_block_mask(block_index: np.ndarray, step_in_block: np.ndarray) -> np.ndarray:
    """返回属于完整 10-step block 的 step 掩码。"""

    out = np.zeros_like(block_index, dtype=bool)
    for b in np.unique(block_index):
        idx = np.where(block_index == b)[0]
        if len(idx) == 10 and set(step_in_block[idx].tolist()) == set(range(10)):
            out[idx] = True
    return out


def append_or_create(f: h5py.File, name: str, data: np.ndarray, compression: str = "lzf") -> None:
    chunks = (1, *data.shape[1:]) if data.ndim >= 3 else None
    f.create_dataset(name, data=data, chunks=chunks, compression=compression if chunks else None)


def audit_one_h5(path: Path, paths: dict, fixed_window_us: int, out_dir: Path, seed: int, audit_blocks: int) -> dict:
    """审计单个 N 的 H5，返回汇总指标，并保存审计子集。"""

    with h5py.File(path, "r") as f, h5py.File(paths["pose"], "r") as f_pose:
        N = int(f.attrs["N"])
        n_steps = int(f["step_index"].shape[0])
        block_index = f["block_index"][:]
        step_in_block = f["step_in_block"][:]
        event_dt = f["event_dt"][:]
        t_start = f["t_start"][:]
        t_end = f["t_end"][:]
        T = f["T"][:]
        Omega = f["Omega"][:]
        speed_valid = f["speed_valid"][:].astype(bool)
        omega_valid = f["omega_valid"][:].astype(bool)
        supervise_valid = f["supervise_valid"][:].astype(bool)
        valid_mask_ratio = f["valid_mask_ratio"][:]
        min_ttc = f["min_ttc"][:]
        roi_x0 = f["roi_x0"][:]
        roi_y0 = f["roi_y0"][:]
        reset_required = f["reset_required"][:].astype(bool)
        timeout_count = int(f["timeouts/collected_event_count"].shape[0])
        completion_ratio = f["timeouts/completion_ratio"][:] if timeout_count else np.asarray([], dtype=np.float32)
        complete_mask = complete_block_mask(block_index, step_in_block)
        complete_blocks = np.unique(block_index[complete_mask])
        incomplete_blocks = len(np.unique(block_index)) - len(complete_blocks)

        inv_samples = []
        ttc_samples = []
        risk_step_counts = {"ttc_lt_0_5": 0, "ttc_lt_1": 0, "ttc_lt_2": 0, "ttc_lt_3": 0, "ttc_lt_5": 0}
        continuity_inv = []
        continuity_mask_iou = []
        for start in range(0, n_steps, 128):
            end = min(start + 128, n_steps)
            inv = f["inverse_ttc_start"][start:end]
            mask = f["valid_ttc_mask"][start:end].astype(bool)
            vals = inv[mask]
            vals = vals[np.isfinite(vals) & (vals > 0)]
            if vals.size:
                if vals.size > 20000:
                    vals = vals[:: max(1, vals.size // 20000)]
                inv_samples.append(vals.astype(np.float32))
                ttc_samples.append((1.0 / vals).astype(np.float32))
            for i in range(end - start):
                if not np.any(mask[i]):
                    continue
                inv_i = inv[i][mask[i]]
                ttc_i = 1.0 / np.maximum(inv_i, 1e-6)
                risk_step_counts["ttc_lt_0_5"] += int(np.any(ttc_i < 0.5))
                risk_step_counts["ttc_lt_1"] += int(np.any(ttc_i < 1.0))
                risk_step_counts["ttc_lt_2"] += int(np.any(ttc_i < 2.0))
                risk_step_counts["ttc_lt_3"] += int(np.any(ttc_i < 3.0))
                risk_step_counts["ttc_lt_5"] += int(np.any(ttc_i < 5.0))

        for b in complete_blocks[: min(len(complete_blocks), 2000)]:
            idx = np.where(block_index == b)[0]
            inv = f["inverse_ttc_start"][idx]
            mask = f["valid_ttc_mask"][idx].astype(bool)
            for j in range(len(idx) - 1):
                common = mask[j] & mask[j + 1]
                union = mask[j] | mask[j + 1]
                if np.any(common):
                    continuity_inv.append(float(np.mean(np.abs(inv[j + 1][common] - inv[j][common]))))
                if np.any(union):
                    continuity_mask_iou.append(float(common.sum() / union.sum()))

        inv_all = np.concatenate(inv_samples) if inv_samples else np.asarray([], dtype=np.float32)
        ttc_all = np.concatenate(ttc_samples) if ttc_samples else np.asarray([], dtype=np.float32)

        pose_ts = f_pose["ts"][:].astype(np.int64)
        pose_mats = f_pose["Cn_T_C0"][:].astype(np.float64)
        sampled_blocks = sample_complete_blocks(complete_blocks, audit_blocks, seed)
        sample_idx = np.concatenate([np.where(block_index == b)[0] for b in sampled_blocks]) if sampled_blocks.size else np.asarray([], dtype=np.int64)
        sample_idx = np.sort(sample_idx)

        audit_path = out_dir / f"audit_subset_N{N}.h5"
        with h5py.File(audit_path, "w") as fa:
            fa.attrs["N"] = N
            fa.attrs["source_h5"] = str(path)
            for key in ["step_index", "block_index", "step_in_block", "roi_x0", "roi_y0", "t_start", "t_end", "event_dt", "T", "Omega", "supervise_valid"]:
                append_or_create(fa, key, f[key][sample_idx], compression="lzf")
            event_cnt = f["event_cnt"][sample_idx]
            inv = f["inverse_ttc_start"][sample_idx]
            mask = f["valid_ttc_mask"][sample_idx].astype(bool)
            T_s = f["T"][sample_idx]
            t_s = f["t_start"][sample_idx]
            depth = np.zeros_like(inv, dtype=np.float32)
            ttc = np.zeros_like(inv, dtype=np.float32)
            valid = mask & (inv > 0) & (T_s[:, 2, None, None] > 0)
            depth[valid] = (T_s[:, 2, None, None] / np.maximum(inv, 1e-6))[valid]
            ttc[valid] = 1.0 / np.maximum(inv, 1e-6)[valid]
            T10, Omega10, fixed_valid = compute_motion_from_pose(pose_mats, pose_ts, t_s, t_s + fixed_window_us)
            inv10 = np.zeros_like(inv, dtype=np.float32)
            ttc10 = np.zeros_like(inv, dtype=np.float32)
            v10 = valid & fixed_valid[:, None, None] & (T10[:, 2, None, None] > 0) & (depth > 0)
            inv10[v10] = (T10[:, 2, None, None] / np.maximum(depth, 1e-6))[v10]
            ttc10[v10] = 1.0 / np.maximum(inv10, 1e-6)[v10]
            append_or_create(fa, "event_cnt", event_cnt)
            append_or_create(fa, "depth_start", depth)
            append_or_create(fa, "ttc_start", ttc)
            append_or_create(fa, "inverse_ttc_start", inv)
            append_or_create(fa, "valid_ttc_mask", mask.astype(np.uint8))
            append_or_create(fa, "T_fixed10ms", T10.astype(np.float32), compression="lzf")
            append_or_create(fa, "Omega_fixed10ms", Omega10.astype(np.float32), compression="lzf")
            append_or_create(fa, "TTC_fixed10ms", ttc10)
            append_or_create(fa, "inverse_TTC_fixed10ms", inv10)

        # 固定 10ms 对比统计。
        with h5py.File(audit_path, "r") as fa:
            inv = fa["inverse_ttc_start"][:]
            inv10 = fa["inverse_TTC_fixed10ms"][:]
            ttc = fa["ttc_start"][:]
            ttc10 = fa["TTC_fixed10ms"][:]
            mask = fa["valid_ttc_mask"][:].astype(bool)
            both = mask & (inv > 0) & (inv10 > 0)
            inv_diff = np.abs(inv10[both] - inv[both])
            inv_mre = inv_diff / np.maximum(inv[both], 1e-6) if inv_diff.size else np.asarray([])
            ttc_diff = np.abs(ttc10[both] - ttc[both])
            T_diff = np.linalg.norm(fa["T_fixed10ms"][:] - fa["T"][:], axis=1)
            Omega_diff = np.linalg.norm(fa["Omega_fixed10ms"][:] - fa["Omega"][:], axis=1)
            fixed_by_bin = []
            edt = fa["event_dt"][:]
            for lo, hi in EVENT_DT_BINS:
                sel = (edt >= lo) & (edt < hi)
                if not np.any(sel):
                    continue
                m = both[sel]
                idiff = np.abs(inv10[sel][m] - inv[sel][m])
                imre = idiff / np.maximum(inv[sel][m], 1e-6) if idiff.size else np.asarray([])
                fixed_by_bin.append(
                    {
                        "N": N,
                        "bin_us": f"{lo}-{hi}",
                        "steps": int(sel.sum()),
                        "T_diff_mean": float(np.mean(T_diff[sel])),
                        "Omega_diff_mean": float(np.mean(Omega_diff[sel])),
                        "inverse_ttc_mae": float(np.mean(idiff)) if idiff.size else np.nan,
                        "inverse_ttc_mre": float(np.mean(imre)) if imre.size else np.nan,
                    }
                )

        event_dt_stats = stats(event_dt)
        block_durations = []
        for b in complete_blocks:
            idx = np.where(block_index == b)[0]
            block_durations.append(int(t_end[idx[-1]] - t_start[idx[0]]))
        block_durations = np.asarray(block_durations, dtype=np.int64)

        row = {
            "N": N,
            "h5_path": str(path),
            "h5_size_bytes": path.stat().st_size,
            "valid_steps": n_steps,
            "timeout_count": timeout_count,
            "timeout_ratio": timeout_count / max(timeout_count + n_steps, 1),
            "complete_10step_blocks": int(len(complete_blocks)),
            "incomplete_blocks": int(incomplete_blocks),
            "complete_block_ratio": float(len(complete_blocks) / max(len(complete_blocks) + incomplete_blocks, 1)),
            "supervised_steps": int(supervise_valid.sum()),
            "supervise_valid_ratio": float(supervise_valid.mean()) if n_steps else 0.0,
            "event_dt_p50": event_dt_stats["p50"],
            "event_dt_p95": event_dt_stats["p95"],
            "event_dt_mean": event_dt_stats["mean"],
            "event_dt_std": event_dt_stats["std"],
            "block_duration_p50": float(np.percentile(block_durations, 50)) if block_durations.size else np.nan,
            "block_duration_p95": float(np.percentile(block_durations, 95)) if block_durations.size else np.nan,
            "mask_valid_ratio_mean": float(np.mean(valid_mask_ratio)),
            "empty_label_steps": int(np.sum(valid_mask_ratio == 0)),
            "ttc_lt_1_step_ratio": risk_step_counts["ttc_lt_1"] / max(n_steps, 1),
            "ttc_lt_2_step_ratio": risk_step_counts["ttc_lt_2"] / max(n_steps, 1),
            "inverse_ttc_adjacent_mae": float(np.mean(continuity_inv)) if continuity_inv else np.nan,
            "mask_iou_adjacent_mean": float(np.mean(continuity_mask_iou)) if continuity_mask_iou else np.nan,
            "inverse_ttc_stats": stats(inv_all),
            "ttc_stats": stats(ttc_all),
            "T_norm_stats": stats(np.linalg.norm(T, axis=1)),
            "Tz_stats": stats(T[:, 2]),
            "Omega_norm_stats": stats(np.linalg.norm(Omega, axis=1)),
            "completion_ratio_stats": stats(completion_ratio),
            "audit_subset_h5": str(audit_path),
            "fixed10_inverse_mae": float(np.mean(inv_diff)) if inv_diff.size else np.nan,
            "fixed10_inverse_mre": float(np.mean(inv_mre)) if inv_mre.size else np.nan,
            "fixed10_ttc_mae": float(np.mean(ttc_diff)) if ttc_diff.size else np.nan,
            "fixed10_T_diff_mean": float(np.mean(T_diff)) if T_diff.size else np.nan,
            "fixed10_Omega_diff_mean": float(np.mean(Omega_diff)) if Omega_diff.size else np.nan,
            "fixed10_by_bin": fixed_by_bin,
        }
        return row


def write_reports(rows: List[dict], out_dir: Path) -> None:
    """写综合报告、质量审计、timeout 分析、固定 10ms 对比报告。"""

    report_dir = out_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    flat_rows = []
    for row in rows:
        flat = {k: v for k, v in row.items() if not isinstance(v, (dict, list))}
        flat_rows.append(flat)
    pd.DataFrame(flat_rows).to_csv(out_dir / "multi_n_summary.csv", index=False)

    lines = [
        "# Skatepark 多事件数 TTC 标签综合报告",
        "",
        "| 指标 | 5k | 10k | 15k | 20k |",
        "|---|---:|---:|---:|---:|",
    ]
    by_n = {int(r["N"]): r for r in rows}
    metrics = [
        ("有效step数", "valid_steps", "{:.0f}"),
        ("event_dt P50(us)", "event_dt_p50", "{:.1f}"),
        ("event_dt P95(us)", "event_dt_p95", "{:.1f}"),
        ("timeout比例", "timeout_ratio", "{:.4f}"),
        ("完整block比例", "complete_block_ratio", "{:.4f}"),
        ("block duration P50(us)", "block_duration_p50", "{:.1f}"),
        ("block duration P95(us)", "block_duration_p95", "{:.1f}"),
        ("supervise_valid比例", "supervise_valid_ratio", "{:.4f}"),
        ("mask有效像素率", "mask_valid_ratio_mean", "{:.4f}"),
        ("TTC<1s step比例", "ttc_lt_1_step_ratio", "{:.4f}"),
        ("inverse TTC相邻变化", "inverse_ttc_adjacent_mae", "{:.6f}"),
        ("固定10 ms标签差异", "fixed10_inverse_mae", "{:.6f}"),
        ("H5大小(MB)", "h5_size_bytes", "{:.1f}"),
    ]
    for name, key, form in metrics:
        vals = []
        for n in [5000, 10000, 15000, 20000]:
            v = by_n.get(n, {}).get(key, np.nan)
            if key == "h5_size_bytes" and np.isfinite(v):
                v = v / 1024 / 1024
            vals.append(form.format(v) if np.isfinite(v) else "nan")
        lines.append(f"| {name} | " + " | ".join(vals) + " |")

    valid_candidates = [r for r in rows if r["timeout_ratio"] < 0.5 and r["complete_block_ratio"] > 0.5]
    if valid_candidates:
        best = sorted(valid_candidates, key=lambda r: (abs(r["event_dt_p50"] - 3300), r["fixed10_inverse_mae"]))[0]
    else:
        best = sorted(rows, key=lambda r: abs(r["event_dt_p50"] - 3300))[0]
    ranking = sorted(rows, key=lambda r: (abs(r["event_dt_p50"] - 3300), r["timeout_ratio"]))
    lines.extend(
        [
            "",
            "## 最终排序",
            "",
            ", ".join([f"N={int(r['N'])}" for r in ranking]),
            "",
            f"## 正式训练首选 N",
            "",
            f"首选 `N={int(best['N'])}`。判断依据是 event_dt 中位数、timeout 率、完整 block 比例和固定 10ms 标签差异的综合折中。",
            "",
            "备选方案：如果后续训练显存允许，可比较更大的 N 或固定时间窗。当前阶段不修改网络、不训练。",
        ]
    )
    (report_dir / "Skatepark多事件数TTC标签综合报告.md").write_text("\n".join(lines), encoding="utf-8")

    timeout_lines = ["# 超时换ROI机制分析", ""]
    for r in rows:
        timeout_lines.extend(
            [
                f"## N={int(r['N'])}",
                f"- timeout 数量：{r['timeout_count']}",
                f"- timeout 比例：{fmt(r['timeout_ratio'])}",
                f"- 完整 block：{r['complete_10step_blocks']}",
                f"- 不完整 block：{r['incomplete_blocks']}",
                f"- timeout 完成比例 mean/P50/P95：{fmt(r['completion_ratio_stats']['mean'])} / {fmt(r['completion_ratio_stats']['p50'])} / {fmt(r['completion_ratio_stats']['p95'])}",
                "",
            ]
        )
    (report_dir / "超时换ROI机制分析.md").write_text("\n".join(timeout_lines), encoding="utf-8")

    quality_lines = ["# TTC标签质量审计", ""]
    for r in rows:
        quality_lines.extend(
            [
                f"## N={int(r['N'])}",
                f"- mask 有效像素率均值：{fmt(r['mask_valid_ratio_mean'])}",
                f"- 空标签 step：{r['empty_label_steps']}",
                f"- inverse TTC mean/P50/P95：{fmt(r['inverse_ttc_stats']['mean'])} / {fmt(r['inverse_ttc_stats']['p50'])} / {fmt(r['inverse_ttc_stats']['p95'])}",
                f"- TTC mean/P50/P95：{fmt(r['ttc_stats']['mean'])} / {fmt(r['ttc_stats']['p50'])} / {fmt(r['ttc_stats']['p95'])}",
                f"- ||T|| mean/P50/P95：{fmt(r['T_norm_stats']['mean'])} / {fmt(r['T_norm_stats']['p50'])} / {fmt(r['T_norm_stats']['p95'])}",
                f"- ||Omega|| mean/P50/P95：{fmt(r['Omega_norm_stats']['mean'])} / {fmt(r['Omega_norm_stats']['p50'])} / {fmt(r['Omega_norm_stats']['p95'])}",
                f"- TTC<1s step 比例：{fmt(r['ttc_lt_1_step_ratio'])}",
                f"- TTC<2s step 比例：{fmt(r['ttc_lt_2_step_ratio'])}",
                "",
            ]
        )
    (report_dir / "TTC标签质量审计.md").write_text("\n".join(quality_lines), encoding="utf-8")

    fixed_rows = []
    fixed_lines = ["# 事件窗口与固定10ms标签对比", ""]
    for r in rows:
        fixed_lines.extend(
            [
                f"## N={int(r['N'])}",
                f"- T差异均值：{fmt(r['fixed10_T_diff_mean'])}",
                f"- Omega差异均值：{fmt(r['fixed10_Omega_diff_mean'])}",
                f"- inverse TTC MAE：{fmt(r['fixed10_inverse_mae'])}",
                f"- inverse TTC MRE：{fmt(r['fixed10_inverse_mre'])}",
                f"- TTC MAE：{fmt(r['fixed10_ttc_mae'])}",
                "",
                "| event_dt区间(us) | steps | T差异 | Omega差异 | inverse TTC MAE | inverse TTC MRE |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for b in r["fixed10_by_bin"]:
            fixed_rows.append(b)
            fixed_lines.append(
                f"| {b['bin_us']} | {b['steps']} | {fmt(b['T_diff_mean'])} | {fmt(b['Omega_diff_mean'])} | {fmt(b['inverse_ttc_mae'])} | {fmt(b['inverse_ttc_mre'])} |"
            )
        fixed_lines.append("")
    pd.DataFrame(fixed_rows).to_csv(out_dir / "fixed10_comparison_by_bin.csv", index=False)
    (report_dir / "事件窗口与固定10ms标签对比.md").write_text("\n".join(fixed_lines), encoding="utf-8")

    align_lines = [
        "# 事件与TTC空间对齐审计",
        "",
        "本报告依赖 `visualize_skatepark_multi_n_ttc.py` 生成的图像人工检查。",
        "自动检查已确认 event_cnt、inverse_ttc_start、valid_ttc_mask 使用同一 ROI 尺寸 `[128,128]`。",
        "",
    ]
    for r in rows:
        align_lines.append(f"- N={int(r['N'])} 审计子集：`{r['audit_subset_h5']}`")
    (report_dir / "事件与TTC空间对齐审计.md").write_text("\n".join(align_lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", default="spot_outdoor_day_skatepark_1")
    parser.add_argument("--root", type=Path, default=Path("/data/M3ED"))
    parser.add_argument("--out-dir", type=Path, default=ROOT / "debug_sets" / "skatepark_multi_n_ttc")
    parser.add_argument("--fixed-window-ms", type=float, default=10.0)
    parser.add_argument("--audit-blocks", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = resolve_sequence_paths(args.root, args.sequence)
    rows = []
    for path in sorted(args.out_dir.glob("skatepark_N*.h5")):
        rows.append(audit_one_h5(path, paths, int(args.fixed_window_ms * 1000), args.out_dir, args.seed, args.audit_blocks))
    if not rows:
        raise FileNotFoundError(f"未找到 {args.out_dir}/skatepark_N*.h5")
    (args.out_dir / "audit_summary.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    write_reports(rows, args.out_dir)
    print(f"reports: {args.out_dir / 'reports'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

