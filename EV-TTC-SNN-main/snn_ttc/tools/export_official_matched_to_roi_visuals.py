#!/usr/bin/env python3
"""按 ROI 方法样本时间，在官方 EV-TTC 数据中导出对应的 360x360 可视化。

用途：把我们固定事件数 ROI 图中的每个 step，匹配到官方 exp_filts/ttcef
中最接近的全图样本，方便肉眼比较两种数据生成方式在同一时刻看到的内容。
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

import h5py
import hdf5plugin  # noqa: F401  # 让 h5py 能读取官方压缩 H5
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402


DEFAULT_ROOT = Path("/home/hello/research_project/event+SNN+TTC")
DEFAULT_SEQ = "car_urban_night_rittenhouse"
FILENAME_RE = re.compile(r"N(?P<N>\d+)_block(?P<block>\d+)_step(?P<step>\d+)\.png$")


def pct(num: float, den: float) -> float:
    return 100.0 * float(num) / float(den) if den else 0.0


def clipped_image(arr: np.ndarray, mask: np.ndarray | None = None) -> tuple[np.ndarray, float | None, float | None]:
    """按 1/99 分位裁剪显示范围，避免少数异常值压暗主体结构。"""

    values = arr[mask] if mask is not None else arr[np.isfinite(arr)]
    values = values[np.isfinite(values)]
    if values.size < 10:
        return arr, None, None
    lo, hi = np.percentile(values, [1, 99])
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        return arr, None, None
    return np.clip(arr, lo, hi), float(lo), float(hi)


def official_pixel_masks(raw_mask: np.ndarray, exp_ch: np.ndarray, ttc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """复刻官方像素 mask，并额外构造只保留正 TTC 的 mask。"""

    ttc_clean = np.nan_to_num(ttc)
    official = raw_mask.astype(bool) & (np.abs(exp_ch) > 1e-3) & np.isfinite(ttc) & (ttc_clean < 100)
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = np.divide(1.0, ttc, out=np.full_like(ttc, np.nan, dtype=np.float32), where=(ttc != 0))
    positive = official & (ttc > 0) & np.isfinite(inv) & (inv > 0)
    return official, positive


def draw_panel(
    ax: plt.Axes,
    img: np.ndarray,
    title: str,
    cmap: str,
    roi: tuple[int, int, int] | None,
    mask: np.ndarray | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    """绘制一个子图，并在全图上用黄色框标出 ROI。"""

    show = img
    if vmin is None and vmax is None and cmap != "gray":
        show, vmin, vmax = clipped_image(img, mask)
    im = ax.imshow(show, cmap=cmap, vmin=vmin, vmax=vmax)
    if roi is not None:
        x0, y0, size = roi
        ax.add_patch(Rectangle((x0, y0), size, size, fill=False, edgecolor="yellow", linewidth=1.8))
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)


def nearest_index(sorted_times: np.ndarray, target: int) -> int:
    """在官方 exp_times 中找最接近目标时间的样本下标。"""

    pos = int(np.searchsorted(sorted_times, target, side="left"))
    candidates = [min(max(pos, 0), len(sorted_times) - 1)]
    if pos > 0:
        candidates.append(pos - 1)
    if pos + 1 < len(sorted_times):
        candidates.append(pos + 1)
    return min(candidates, key=lambda i: abs(int(sorted_times[i]) - int(target)))


def parse_visual_name(path: Path) -> tuple[int, int, int]:
    """从 `N15000_block011268_step04.png` 中解析 N、block、step。"""

    match = FILENAME_RE.match(path.name)
    if not match:
        raise ValueError(f"无法解析文件名: {path}")
    return int(match.group("N")), int(match.group("block")), int(match.group("step"))


def locate_roi_rows(roi_h5: Path) -> dict[tuple[int, int], int]:
    """建立 `(block_index, step_in_block) -> H5 行号` 的查找表。"""

    with h5py.File(roi_h5, "r") as f:
        blocks = f["block_index"][:].astype(np.int64)
        steps = f["step_in_block"][:].astype(np.int64)
    return {(int(b), int(s)): i for i, (b, s) in enumerate(zip(blocks, steps))}


def save_official_visual(
    out_path: Path,
    exp_img: np.ndarray,
    depth: np.ndarray,
    ttc: np.ndarray,
    raw_mask: np.ndarray,
    official_index: int,
    official_time: int,
    roi_info: dict[str, Any],
) -> dict[str, Any]:
    """保存一张官方 360x360 图，并返回像素统计。"""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    official_mask, positive_mask = official_pixel_masks(raw_mask, exp_img[-1], ttc)
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = np.divide(1.0, ttc, out=np.full_like(ttc, np.nan, dtype=np.float32), where=(ttc != 0))

    event_vmax = np.percentile(np.abs(exp_img[-1][np.isfinite(exp_img[-1])]), 99) if np.any(np.isfinite(exp_img[-1])) else 1.0
    roi = tuple(roi_info["roi"])
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    panels = [
        ("official event exp ch5", exp_img[-1], "coolwarm", None, -event_vmax, event_vmax),
        ("official depth", depth, "viridis", raw_mask.astype(bool), None, None),
        ("official TTC", ttc, "coolwarm", official_mask, None, None),
        ("official inverse TTC", inv, "coolwarm", official_mask, None, None),
        ("official mask", official_mask.astype(np.float32), "gray", None, 0, 1),
        ("positive TTC mask", positive_mask.astype(np.float32), "gray", None, 0, 1),
    ]
    for ax, (title, img, cmap, mask, vmin, vmax) in zip(axes.flat, panels):
        draw_panel(ax, img, title, cmap, roi, mask=mask, vmin=vmin, vmax=vmax)

    official_pixels = max(int(np.count_nonzero(official_mask)), 1)
    fig.suptitle(
        f"official sample {official_index} t={official_time} us | "
        f"matched ours N={roi_info['N']} block={roi_info['block_index']} step={roi_info['step_in_block']} "
        f"target={roi_info['target_time_us']} us delta={official_time - roi_info['target_time_us']} us | "
        f"ROI=({roi[0]},{roi[1]},{roi[2]}) TTC+={pct(np.count_nonzero(positive_mask), official_pixels):.2f}%",
        fontsize=10,
    )
    fig.savefig(out_path, dpi=140)
    plt.close(fig)

    return {
        "official_mask_pixels": int(np.count_nonzero(official_mask)),
        "positive_ttc_mask_pixels": int(np.count_nonzero(positive_mask)),
        "positive_ttc_pixel_ratio_in_official_mask_percent": pct(np.count_nonzero(positive_mask), official_pixels),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=15000)
    parser.add_argument("--sequence", default=DEFAULT_SEQ)
    parser.add_argument("--match-time", choices=["t_start", "t_end", "center"], default="t_end")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--roi-vis-dir", type=Path, default=None)
    parser.add_argument("--roi-h5", type=Path, default=None)
    parser.add_argument("--official-exp", type=Path, default=None)
    parser.add_argument("--official-ttcef", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    base = args.root / "EV-TTC-SNN-main/debug_sets/car_urban_night_rittenhouse_multi_n_ttc"
    roi_vis_dir = args.roi_vis_dir or base / "visualizations" / f"N{args.n}"
    roi_h5 = args.roi_h5 or base / "H5" / f"{args.sequence}_N{args.n}.h5"
    official_root = Path("/data/evttc_storage/event+SNN+TTC/data_m3ed_official_7train_3test")
    official_exp = args.official_exp or official_root / "exp_filts/m3ed" / f"{args.sequence}.h5"
    official_ttcef = args.official_ttcef or official_root / "ttcef/m3ed" / f"{args.sequence}.h5"
    out_dir = args.out_dir or base / "official_evttc_audit/visualizations" / f"matched_to_ours_N{args.n}"
    out_dir.mkdir(parents=True, exist_ok=True)

    roi_images = sorted(roi_vis_dir.glob(f"N{args.n}_block*_step*.png"))[: args.limit]
    if len(roi_images) < args.limit:
        raise FileNotFoundError(f"{roi_vis_dir} 里只找到 {len(roi_images)} 张图，少于 {args.limit}")

    row_lookup = locate_roi_rows(roi_h5)
    rows: list[dict[str, Any]] = []
    with h5py.File(roi_h5, "r") as fr, h5py.File(official_exp, "r") as fe, h5py.File(official_ttcef, "r") as fg:
        exp_times = np.rint(fe["exp_times"][:]).astype(np.int64)
        for roi_img in roi_images:
            n, block, step = parse_visual_name(roi_img)
            row_index = row_lookup[(block, step)]
            t_start = int(fr["t_start"][row_index])
            t_end = int(fr["t_end"][row_index])
            center = int((t_start + t_end) // 2)
            target = {"t_start": t_start, "t_end": t_end, "center": center}[args.match_time]
            official_index = nearest_index(exp_times, target)
            official_time = int(exp_times[official_index])
            x0 = int(fr["roi_x0"][row_index])
            y0 = int(fr["roi_y0"][row_index])
            roi_info = {
                "N": n,
                "row_index": row_index,
                "block_index": block,
                "step_in_block": step,
                "roi": [x0, y0, int(fr.attrs.get("roi_size", 128))],
                "t_start": t_start,
                "t_end": t_end,
                "center": center,
                "target_time_us": target,
                "event_dt_us": int(fr["event_dt"][row_index]),
                "direct_supervision_valid": bool(fr["direct_supervision_valid"][row_index]),
                "valid_mask_ratio": float(fr["valid_mask_ratio"][row_index]),
            }
            out_name = f"{roi_img.stem}_official_sample_{official_index:06d}_delta_{official_time - target:+d}us.png"
            stats = save_official_visual(
                out_dir / out_name,
                fe["exp_filts"][official_index],
                fg["depth"][official_index],
                fg["ttc"][official_index],
                fg["mask"][official_index],
                official_index,
                official_time,
                roi_info,
            )
            rows.append(
                {
                    "roi_image": str(roi_img),
                    "official_image": str(out_dir / out_name),
                    "N": n,
                    "row_index": row_index,
                    "block_index": block,
                    "step_in_block": step,
                    "roi_x0": x0,
                    "roi_y0": y0,
                    "t_start": t_start,
                    "t_end": t_end,
                    "center": center,
                    "match_time_field": args.match_time,
                    "target_time_us": target,
                    "official_sample_index": official_index,
                    "official_exp_time_us": official_time,
                    "delta_us": official_time - target,
                    "event_dt_us": roi_info["event_dt_us"],
                    "direct_supervision_valid": roi_info["direct_supervision_valid"],
                    "valid_mask_ratio": roi_info["valid_mask_ratio"],
                    **stats,
                }
            )

    csv_path = out_dir / f"matched_to_ours_N{args.n}_index.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    deltas = np.asarray([r["delta_us"] for r in rows], dtype=np.float64)
    summary = {
        "N": args.n,
        "match_time_field": args.match_time,
        "matched_count": len(rows),
        "roi_visualization_dir": str(roi_vis_dir),
        "official_output_dir": str(out_dir),
        "csv": str(csv_path),
        "delta_abs_us_mean": float(np.mean(np.abs(deltas))),
        "delta_abs_us_median": float(np.median(np.abs(deltas))),
        "delta_abs_us_max": float(np.max(np.abs(deltas))),
    }
    (out_dir / f"matched_to_ours_N{args.n}_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    md_rows = [
        "| # | ours block/step | ROI | target us | official sample | official us | delta us | official image |",
        "|---:|---|---|---:|---:|---:|---:|---|",
    ]
    for idx, row in enumerate(rows, start=1):
        md_rows.append(
            f"| {idx} | {row['block_index']}/{row['step_in_block']} | "
            f"({row['roi_x0']},{row['roi_y0']},128) | {row['target_time_us']} | "
            f"{row['official_sample_index']} | {row['official_exp_time_us']} | {row['delta_us']} | "
            f"`{row['official_image']}` |"
        )
    md = (
        f"# 官方 EV-TTC 与我们 N={args.n} ROI 图同时间匹配\n\n"
        f"- 匹配基准：我们 ROI H5 的 `{args.match_time}`，对应官方 `exp_times/readout time`。\n"
        f"- 输入 ROI 图片目录：`{roi_vis_dir}`\n"
        f"- 输出官方图片目录：`{out_dir}`\n"
        f"- 匹配数量：{len(rows)}\n"
        f"- |delta| 平均：{summary['delta_abs_us_mean']:.1f} us\n"
        f"- |delta| 中位数：{summary['delta_abs_us_median']:.1f} us\n"
        f"- |delta| 最大：{summary['delta_abs_us_max']:.1f} us\n"
        f"- 索引 CSV：`{csv_path}`\n\n"
        + "\n".join(md_rows)
        + "\n"
    )
    (out_dir / f"matched_to_ours_N{args.n}_report.md").write_text(md, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
