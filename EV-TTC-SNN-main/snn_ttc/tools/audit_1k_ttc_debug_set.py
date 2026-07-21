#!/usr/bin/env python3
"""审计 1k-event TTC debug H5，并输出中文 Markdown 报告。"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


def _fmt(value: float, digits: int = 6) -> str:
    if value is None or not np.isfinite(value):
        return "nan"
    return f"{value:.{digits}f}"


def _safe_stats(arr: np.ndarray) -> dict:
    """对有限值计算常用统计量；空数组返回 nan，避免审计脚本中断。"""

    arr = np.asarray(arr)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"min": np.nan, "max": np.nan, "mean": np.nan, "std": np.nan, "p50": np.nan, "p95": np.nan}
    return {
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
    }


def _line_stats(name: str, arr: np.ndarray, unit: str = "") -> str:
    s = _safe_stats(arr)
    return (
        f"- {name}: min={_fmt(s['min'])}{unit}, max={_fmt(s['max'])}{unit}, "
        f"mean={_fmt(s['mean'])}{unit}, std={_fmt(s['std'])}{unit}, "
        f"p50={_fmt(s['p50'])}{unit}, p95={_fmt(s['p95'])}{unit}"
    )


def audit_h5(h5_path: Path, out_md: Path) -> str:
    """检查连续性、标签有效性、运动筛选标志和 TTC 分布。"""

    with h5py.File(h5_path, "r") as f:
        event_start = f["event_start_idx"][:]
        event_end = f["event_end_idx"][:]
        t_start = f["t_start"][:]
        t_end = f["t_end"][:]
        dt = f["dt"][:]
        event_cnt = f["event_cnt"][:]
        mask = f["valid_ttc_mask"][:].astype(bool)
        ttc = f["ttc_start"][:]
        inv_ttc = f["inverse_ttc_start"][:]
        depth = f["depth_start"][:]
        T = f["T"][:]
        Omega = f["Omega"][:]
        speed_valid = f["speed_valid"][:].astype(bool)
        omega_valid = f["omega_valid"][:].astype(bool)
        supervise_valid = f["supervise_valid"][:].astype(bool)
        seq = f.attrs.get("sequence_name", "unknown")
        events_per_step = int(f.attrs.get("events_per_step", 1000))
        start_policy = f.attrs.get("start_policy", "unknown")

    n = len(event_start)
    step_event_counts = event_end - event_start
    strict_event_count = bool(np.all(step_event_counts == events_per_step))
    strict_index_cont = bool(np.all(event_start[1:] == event_end[:-1])) if n > 1 else True
    strict_time_mono = bool(np.all(t_start[1:] >= t_end[:-1]) and np.all(t_end >= t_start))

    # mask 是像素级可靠性；speed/omega/supervise 是 step 级监督标志。
    valid_ratio = mask.reshape(n, -1).mean(axis=1)
    ttc_valid = ttc[mask]
    inv_valid = inv_ttc[mask]
    depth_valid = depth[mask]
    finite_on_mask = bool(np.all(np.isfinite(ttc_valid)) and np.all(np.isfinite(depth_valid)))
    empty_steps = np.where(mask.reshape(n, -1).sum(axis=1) == 0)[0]
    zero_event_steps = np.where(event_cnt.reshape(n, -1).sum(axis=1) == 0)[0]
    extreme_ttc = int(np.sum(np.abs(ttc_valid) > 100)) if ttc_valid.size else 0
    failed_first_10 = np.where(~supervise_valid[: min(10, n)])[0]

    lines = [
        "# M3ED 1k 事件 TTC Debug Set 审计报告",
        "",
        "## 基本信息",
        f"- H5 文件：`{h5_path}`",
        f"- 序列：`{seq}`",
        f"- 步数：{n}",
        f"- 每步事件数配置：{events_per_step}",
        f"- 起始策略：`{start_policy}`",
        "",
        "## 连续性检查",
        f"- 每步恰好 1000 个事件：{'通过' if strict_event_count else '失败'}",
        f"- 相邻 step 事件索引首尾相接：{'通过' if strict_index_cont else '失败'}",
        f"- t_start/t_end 单调：{'通过' if strict_time_mono else '失败'}",
        _line_stats("dt", dt, " us"),
        "",
        "## 运动量与筛选标志",
        _line_stats("T_x", T[:, 0], " m/s"),
        _line_stats("T_y", T[:, 1], " m/s"),
        _line_stats("T_z", T[:, 2], " m/s"),
        _line_stats("||T||", np.linalg.norm(T, axis=1), " m/s"),
        _line_stats("Omega_x", Omega[:, 0], " rad/s"),
        _line_stats("Omega_y", Omega[:, 1], " rad/s"),
        _line_stats("Omega_z", Omega[:, 2], " rad/s"),
        _line_stats("||Omega||", np.linalg.norm(Omega, axis=1), " rad/s"),
        f"- speed_valid：{int(speed_valid.sum())}/{n}",
        f"- omega_valid：{int(omega_valid.sum())}/{n}",
        f"- supervise_valid：{int(supervise_valid.sum())}/{n}",
        f"- 前 10 步筛选失败但仍保留的 step：{failed_first_10.tolist()}",
        "",
        "## Mask 与标签检查",
        _line_stats("valid_ttc_mask 有效像素率", valid_ratio * 100.0, "%"),
        f"- mask=1 位置 TTC/depth 全部有限：{'通过' if finite_on_mask else '失败'}",
        f"- 空标签 step：{empty_steps.tolist()}",
        f"- 全零事件图 step：{zero_event_steps.tolist()}",
        f"- |TTC| > 100 的 mask 内像素数：{extreme_ttc}",
        "",
        "## TTC / inverse TTC 分布",
        f"- mask 内有效像素总数：{int(mask.sum())}",
        f"- TTC 正值像素数：{int(np.sum(ttc_valid > 0))}",
        f"- TTC 负值像素数：{int(np.sum(ttc_valid < 0))}",
        f"- TTC 零值像素数：{int(np.sum(ttc_valid == 0))}",
        _line_stats("TTC", ttc_valid, " s"),
        _line_stats("inverse TTC", inv_valid, " 1/s"),
        _line_stats("depth", depth_valid, " m"),
        "",
        "## 结论",
        "- 当前 H5 已保留筛选失败时间步；训练阶段应使用 `supervise_valid` 决定是否参与监督，而不是删除 step。",
        "- `valid_ttc_mask` 是像素级标签可靠性，不包含速度/角速度筛选。",
    ]

    report = "\n".join(lines) + "\n"
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(report, encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5", type=Path, required=True)
    parser.add_argument("--out-md", type=Path)
    args = parser.parse_args()
    out_md = args.out_md or (args.h5.parent / "audit_report.md")
    report = audit_h5(args.h5, out_md)
    print(report)
    print(f"audit_report: {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
