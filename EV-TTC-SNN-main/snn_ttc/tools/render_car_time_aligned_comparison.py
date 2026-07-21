#!/usr/bin/env python3
"""生成 car 三种空间方案与官方 EV-TTC 的时间对齐可视化。

本脚本只读取已经生成好的 H5，不重新扫描 raw events，也不重新计算标签。
对齐方式是：在四组数据共同覆盖的时间范围内均匀选取参考时间，然后分别找
S1/S2/S3/official 中时间戳最近的样本。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import h5py
import hdf5plugin  # noqa: F401
import matplotlib
import numpy as np
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402


ROOT = Path(__file__).resolve().parents[3]
SNN_ROOT = ROOT / "EV-TTC-SNN-main"
if str(SNN_ROOT) not in sys.path:
    sys.path.insert(0, str(SNN_ROOT))

from snn_ttc.tools.build_car_spatial_ablation_from_existing_h5 import (  # noqa: E402
    clipped_image,
    official_pixel_masks,
)


S2_BOX_360 = (52, 52, 256)


def pct(num: float, den: float) -> float:
    return 100.0 * float(num) / float(den) if den else 0.0


def fmt(value: Any, digits: int = 3) -> str:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(v):
        return "NA"
    return f"{v:.{digits}f}"


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(lines)


def nearest_index(sorted_times: np.ndarray, target: int) -> int:
    pos = int(np.searchsorted(sorted_times, target, side="left"))
    candidates = [min(max(pos, 0), len(sorted_times) - 1)]
    if pos > 0:
        candidates.append(pos - 1)
    if pos + 1 < len(sorted_times):
        candidates.append(pos + 1)
    return min(candidates, key=lambda i: abs(int(sorted_times[i]) - int(target)))


class TimeLookup:
    """支持非严格排序时间戳的最近邻查询。"""

    def __init__(self, times: np.ndarray):
        self.order = np.argsort(times)
        self.sorted_times = times[self.order].astype(np.int64)

    def nearest_row(self, target: int) -> tuple[int, int]:
        j = nearest_index(self.sorted_times, target)
        row = int(self.order[j])
        return row, int(self.sorted_times[j])


def longest_true_run(mask: np.ndarray) -> tuple[int, int]:
    """返回最长连续True片段的[start,end)。"""

    best_start = 0
    best_len = 0
    cur_start = 0
    cur_len = 0
    for i, ok in enumerate(mask.astype(bool)):
        if ok:
            if cur_len == 0:
                cur_start = i
            cur_len += 1
            if cur_len > best_len:
                best_len = cur_len
                best_start = cur_start
        else:
            cur_len = 0
    return best_start, best_start + best_len


def draw_panel(
    ax: plt.Axes,
    img: np.ndarray,
    title: str,
    cmap: str,
    boxes: list[tuple[int, int, int, str]] | None = None,
    mask: np.ndarray | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    show = img
    if vmin is None and vmax is None and cmap != "gray":
        show, vmin, vmax = clipped_image(img, mask)
    im = ax.imshow(show, cmap=cmap, vmin=vmin, vmax=vmax)
    for x0, y0, size, color in boxes or []:
        ax.add_patch(Rectangle((x0, y0), size, size, fill=False, edgecolor=color, linewidth=1.6))
    ax.set_title(title, fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)


def signed_event_vlim(img: np.ndarray) -> float:
    vals = np.abs(img[np.isfinite(img)])
    if vals.size < 10:
        return 1.0
    vmax = float(np.percentile(vals, 99))
    return max(vmax, 1e-6)


def read_times(path: Path, key: str) -> np.ndarray:
    with h5py.File(path, "r") as f:
        return f[key][:].astype(np.int64)


def select_reference_times(
    s1_h5: Path,
    s2_h5: Path,
    s3_h5: Path,
    official_exp: Path,
    align_key: str,
    num_images: int,
    reference_source: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    """选择四组数据共同覆盖的参考时间。

    S3使用最长`supervise_valid`连续片段约束，避免随机ROI的无监督片段进入主对比。
    """

    s1_t = read_times(s1_h5, align_key)
    s2_t = read_times(s2_h5, align_key)
    s3_t = read_times(s3_h5, align_key)
    with h5py.File(official_exp, "r") as f:
        off_t = np.rint(f["exp_times"][:]).astype(np.int64)

    with h5py.File(s3_h5, "r") as f:
        if "supervise_valid" in f:
            s3_valid = f["supervise_valid"][:].astype(bool)
        else:
            s3_valid = np.ones_like(s3_t, dtype=bool)
    s3_run_start, s3_run_end = longest_true_run(s3_valid)
    if s3_run_end <= s3_run_start:
        s3_run_start, s3_run_end = 0, len(s3_t)

    low = max(int(s1_t.min()), int(s2_t.min()), int(s3_t[s3_run_start]), int(off_t.min()))
    high = min(int(s1_t.max()), int(s2_t.max()), int(s3_t[s3_run_end - 1]), int(off_t.max()))
    strategy = "S1/S2物化片段 + S3最长supervise_valid片段 + official共同时间范围"
    if high <= low:
        # 极端情况下退回到全局共同范围，仍然保证脚本可以产出审计图。
        low = max(int(s1_t.min()), int(s2_t.min()), int(s3_t.min()), int(off_t.min()))
        high = min(int(s1_t.max()), int(s2_t.max()), int(s3_t.max()), int(off_t.max()))
        strategy = "S1/S2/S3/official全局共同时间范围"
    if high <= low:
        raise RuntimeError("四组数据没有共同时间范围，无法生成时间对齐图。")

    if reference_source == "uniform":
        ref_times = np.linspace(low, high, num_images).round().astype(np.int64)
    else:
        source_times = {
            "S1": s1_t,
            "S2": s2_t,
            "S3": s3_t[s3_run_start:s3_run_end],
            "Official": off_t,
        }[reference_source]
        source_times = source_times[(source_times >= low) & (source_times <= high)]
        if source_times.size == 0:
            raise RuntimeError(f"{reference_source}在共同时间范围内没有可用时间戳。")
        pick = np.linspace(0, source_times.size - 1, min(num_images, source_times.size)).round().astype(np.int64)
        ref_times = source_times[pick].astype(np.int64)
    meta = {
        "align_key": align_key,
        "num_images": int(num_images),
        "reference_source": reference_source,
        "reference_time_start": int(low),
        "reference_time_end": int(high),
        "selection_strategy": strategy,
        "s3_valid_run_start_row": int(s3_run_start),
        "s3_valid_run_end_row_exclusive": int(s3_run_end),
        "s3_valid_run_length": int(s3_run_end - s3_run_start),
    }
    return ref_times, meta


def collect_rows(ref_times: np.ndarray, paths: dict[str, Path], align_key: str) -> tuple[list[dict[str, Any]], dict[str, TimeLookup]]:
    """为每个参考时间找到四组数据的最近邻行号。"""

    lookups = {
        "S1": TimeLookup(read_times(paths["s1"], align_key)),
        "S2": TimeLookup(read_times(paths["s2"], align_key)),
        "S3": TimeLookup(read_times(paths["s3"], align_key)),
    }
    with h5py.File(paths["official_exp"], "r") as f:
        official_times = np.rint(f["exp_times"][:]).astype(np.int64)
    lookups["Official"] = TimeLookup(official_times)

    rows: list[dict[str, Any]] = []
    for i, ref_t in enumerate(ref_times):
        item: dict[str, Any] = {"image_index": i, "reference_time": int(ref_t)}
        for name in ["S1", "S2", "S3", "Official"]:
            row, t = lookups[name].nearest_row(int(ref_t))
            item[f"{name}_row"] = row
            item[f"{name}_time"] = t
            item[f"{name}_delta_us"] = t - int(ref_t)
        rows.append(item)
    return rows, lookups


def finite_masked_values(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    vals = arr[mask.astype(bool)]
    return vals[np.isfinite(vals)]


def update_stats(acc: dict[str, list[float]], prefix: str, event: np.ndarray, inv: np.ndarray, mask: np.ndarray) -> None:
    active = event.sum(axis=0) > 0 if event.ndim == 3 else event > 0
    acc[f"{prefix}_event_nonzero_rate"].append(float(np.mean(active)))
    acc[f"{prefix}_mask_rate"].append(float(np.mean(mask.astype(bool))))
    vals = finite_masked_values(inv, mask)
    if vals.size:
        acc[f"{prefix}_inv_positive_rate"].append(pct(np.count_nonzero(vals > 0), vals.size))
        acc[f"{prefix}_inv_negative_rate"].append(pct(np.count_nonzero(vals < 0), vals.size))
        acc[f"{prefix}_high_risk_inv_gt1_rate"].append(pct(np.count_nonzero(vals > 1.0), vals.size))


def render_one(
    out_png: Path,
    row: dict[str, Any],
    f1: h5py.File,
    f2: h5py.File,
    f3: h5py.File,
    fe: h5py.File,
    fg: h5py.File,
) -> dict[str, Any]:
    s1_i = int(row["S1_row"])
    s2_i = int(row["S2_row"])
    s3_i = int(row["S3_row"])
    off_i = int(row["Official_row"])

    s1_event = f1["event_cnt"][s1_i]
    s1_inv = f1["signed_inverse_ttc"][s1_i]
    s1_mask = f1["valid_ttc_mask"][s1_i].astype(bool)

    s2_event = f2["event_cnt"][s2_i]
    s2_inv = f2["signed_inverse_ttc"][s2_i]
    s2_mask = f2["valid_ttc_mask"][s2_i].astype(bool)

    s3_event = f3["event_cnt"][s3_i]
    s3_inv = f3["inverse_ttc_start"][s3_i]
    s3_mask = f3["valid_ttc_mask"][s3_i].astype(bool)
    s3_x0 = int(f3["roi_x0"][s3_i]) if "roi_x0" in f3 else 0
    s3_y0 = int(f3["roi_y0"][s3_i]) if "roi_y0" in f3 else 0

    exp = fe["exp_filts"][off_i]
    exp_ch = exp[-1]
    off_ttc = fg["ttc"][off_i]
    off_mask, _ = official_pixel_masks(fg["mask"][off_i], exp_ch, off_ttc)

    boxes = [S2_BOX_360 + ("cyan",), (s3_x0, s3_y0, 128, "yellow")]
    event_vmax = signed_event_vlim(exp_ch)

    fig, axes = plt.subplots(3, 4, figsize=(22, 15), constrained_layout=True)
    draw_panel(axes[0, 0], s1_event.sum(axis=0), "S1 event sum 360", "magma", boxes=boxes)
    draw_panel(axes[0, 1], s2_event.sum(axis=0), "S2 event sum 128", "magma")
    draw_panel(axes[0, 2], s3_event.sum(axis=0), "S3 event sum 128", "magma")
    draw_panel(axes[0, 3], exp_ch, "Official signed IIR ch5", "coolwarm", boxes=boxes, vmin=-event_vmax, vmax=event_vmax)

    draw_panel(axes[1, 0], s1_inv, "S1 signed inverse TTC", "coolwarm", boxes=boxes, mask=s1_mask)
    draw_panel(axes[1, 1], s2_inv, "S2 signed inverse TTC", "coolwarm", mask=s2_mask)
    draw_panel(axes[1, 2], s3_inv, "S3 signed inverse TTC", "coolwarm", mask=s3_mask)
    draw_panel(axes[1, 3], off_ttc, "Official TTC", "coolwarm", boxes=boxes, mask=off_mask)

    draw_panel(axes[2, 0], s1_mask.astype(np.float32), "S1 mask", "gray", boxes=boxes, vmin=0, vmax=1)
    draw_panel(axes[2, 1], s2_mask.astype(np.float32), "S2 mask", "gray", vmin=0, vmax=1)
    draw_panel(axes[2, 2], s3_mask.astype(np.float32), "S3 mask", "gray", vmin=0, vmax=1)
    draw_panel(axes[2, 3], off_mask.astype(np.float32), "Official mask", "gray", boxes=boxes, vmin=0, vmax=1)

    fig.suptitle(
        "time-aligned fixed10k comparison | "
        f"ref={row['reference_time']}us | "
        f"S1 row={s1_i} d={row['S1_delta_us']}us dt={int(f1['event_dt'][s1_i])}us | "
        f"S2 row={s2_i} d={row['S2_delta_us']}us dt={int(f2['event_dt'][s2_i])}us | "
        f"S3 row={s3_i} d={row['S3_delta_us']}us dt={int(f3['event_dt'][s3_i])}us | "
        f"Off row={off_i} d={row['Official_delta_us']}us",
        fontsize=11,
    )
    fig.savefig(out_png, dpi=120)
    plt.close(fig)

    return {
        "S1_event_nonzero_rate": float(np.mean(s1_event.sum(axis=0) > 0)),
        "S2_event_nonzero_rate": float(np.mean(s2_event.sum(axis=0) > 0)),
        "S3_event_nonzero_rate": float(np.mean(s3_event.sum(axis=0) > 0)),
        "S1_mask_rate": float(np.mean(s1_mask)),
        "S2_mask_rate": float(np.mean(s2_mask)),
        "S3_mask_rate": float(np.mean(s3_mask)),
        "Official_mask_rate": float(np.mean(off_mask)),
        "S1_source_event_count": int(f1["source_event_count"][s1_i]),
        "S2_source_event_count": int(f2["source_event_count"][s2_i]),
        "S3_source_event_count": int(f3["roi_source_event_count"][s3_i]),
        "S1_speed_valid": bool(f1["speed_valid"][s1_i]),
        "S2_speed_valid": bool(f2["speed_valid"][s2_i]),
        "S3_speed_valid": bool(f3["speed_valid"][s3_i]),
        "S1_omega_valid": bool(f1["omega_valid"][s1_i]),
        "S2_omega_valid": bool(f2["omega_valid"][s2_i]),
        "S3_omega_valid": bool(f3["omega_valid"][s3_i]),
        "S1_supervise_valid": bool(f1["supervise_valid"][s1_i]),
        "S2_supervise_valid": bool(f2["supervise_valid"][s2_i]),
        "S3_supervise_valid": bool(f3["supervise_valid"][s3_i]),
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"count": len(rows)}
    numeric_keys = sorted({k for r in rows for k, v in r.items() if isinstance(v, (int, float, np.integer, np.floating, bool))})
    for key in numeric_keys:
        vals = np.asarray([float(r[key]) for r in rows if key in r], dtype=np.float64)
        if vals.size == 0:
            continue
        out[key] = {
            "mean": float(np.mean(vals)),
            "p50": float(np.percentile(vals, 50)),
            "p95": float(np.percentile(vals, 95)),
            "max": float(np.max(vals)),
            "min": float(np.min(vals)),
        }
    return out


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
        writer.writerows(rows)


def write_report(out_dir: Path, paths: dict[str, Path], meta: dict[str, Any], rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    def delta_row(name: str) -> list[Any]:
        vals = np.asarray([abs(r[f"{name}_delta_us"]) for r in rows], dtype=np.float64)
        return [name, fmt(np.percentile(vals, 50), 1), fmt(np.percentile(vals, 95), 1), fmt(np.max(vals), 1)]

    def selected_metric_row(name: str) -> list[Any]:
        return [
            name,
            f"{100.0 * summary[f'{name}_event_nonzero_rate']['mean']:.2f}%" if f"{name}_event_nonzero_rate" in summary else "NA",
            f"{100.0 * summary[f'{name}_mask_rate']['mean']:.2f}%" if f"{name}_mask_rate" in summary else "NA",
            f"{100.0 * summary[f'{name}_speed_valid']['mean']:.2f}%" if f"{name}_speed_valid" in summary else "NA",
            f"{100.0 * summary[f'{name}_omega_valid']['mean']:.2f}%" if f"{name}_omega_valid" in summary else "NA",
            f"{100.0 * summary[f'{name}_supervise_valid']['mean']:.2f}%" if f"{name}_supervise_valid" in summary else "NA",
        ]

    source_rows = []
    for name in ["S1", "S2", "S3"]:
        vals = np.asarray([r[f"{name}_source_event_count"] for r in rows], dtype=np.int64)
        source_rows.append([name, int(vals.min()), int(vals.max()), bool(np.all(vals == 10000))])

    report = [
        "# S1/S2/S3 与 Official 时间对齐可视化审计",
        "",
        "## 数据来源",
        f"- S1：`{paths['s1']}`",
        f"- S2：`{paths['s2']}`",
        f"- S3：`{paths['s3']}`",
        f"- official exp：`{paths['official_exp']}`",
        f"- official ttcef：`{paths['official_ttcef']}`",
        "",
        "## 对齐方法",
        f"- 对齐时间戳：`{meta['align_key']}`",
        f"- 参考时间范围：`{meta['reference_time_start']} - {meta['reference_time_end']} us`",
        f"- 参考时间来源：`{meta['reference_source']}`",
        f"- 参考时间选择：{meta['selection_strategy']}",
        f"- 图片数量：{meta['num_images']}",
        f"- 图片目录：`{out_dir / 'images'}`",
        "",
        "说明：这是时间戳最近邻对齐，不是同一事件窗口对齐。S1/S2/S3 都是各自独立 fixed-10k，所以 event_dt 可以不同。",
        "",
        "## 时间对齐误差",
        markdown_table(["方案", "|delta| P50(us)", "|delta| P95(us)", "|delta| max(us)"], [delta_row(x) for x in ["S1", "S2", "S3", "Official"]]),
        "",
        "## 选中样本质量",
        markdown_table(["方案", "事件非零率", "mask有效率", "速度通过率", "角速度通过率", "supervise_valid"], [selected_metric_row(x) for x in ["S1", "S2", "S3"]]),
        "",
        "## Official mask",
        f"- official mask 平均有效率：{100.0 * summary['Official_mask_rate']['mean']:.2f}%",
        "",
        "## 固定事件数检查",
        markdown_table(["方案", "source_event_count最小值", "最大值", "是否全为10000"], source_rows),
        "",
        "## 审计结论",
        "- 这些图适合肉眼比较同一时间附近的空间内容、事件密度、mask覆盖和TTC结构。",
        "- 这些图不能直接证明三种方案训练性能优劣，因为三者不是同一个事件窗口，只是同一时刻附近的独立fixed-N样本。",
        "- S2的优势应重点看：是否保留更多场景结构、mask是否比S3更连续、同时成本是否仍为128x128。",
        "- S1的价值应重点看：全图信息是否明显增加，但它的计算和存储成本显著更高。",
    ]
    (out_dir / "time_aligned_comparison_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    base = ROOT / "EV-TTC-SNN-main/debug_sets/[5]car_spatial_ablation"
    source4 = next((ROOT / "EV-TTC-SNN-main/debug_sets").glob("[[]4[]]*"), None)
    default_s3 = source4 / "H5/car_urban_night_rittenhouse_N10000.h5" if source4 else Path("")
    parser.add_argument("--s1-h5", type=Path, default=base / "independent_fixed_n/N10000_S1_360.h5")
    parser.add_argument("--s2-h5", type=Path, default=base / "independent_fixed_n/N10000_S2_center256_to128.h5")
    parser.add_argument("--s3-h5", type=Path, default=default_s3)
    parser.add_argument("--official-exp", type=Path, default=Path("/data/evttc_storage/event+SNN+TTC/data_m3ed_official_7train_3test/exp_filts/m3ed/car_urban_night_rittenhouse.h5"))
    parser.add_argument("--official-ttcef", type=Path, default=Path("/data/evttc_storage/event+SNN+TTC/data_m3ed_official_7train_3test/ttcef/m3ed/car_urban_night_rittenhouse.h5"))
    parser.add_argument("--out-dir", type=Path, default=base / "independent_fixed_n/time_aligned_s1_s2_s3_official_N10000")
    parser.add_argument("--num-images", type=int, default=200)
    parser.add_argument("--align-key", choices=["t_start", "t_end"], default="t_end")
    parser.add_argument("--reference-source", choices=["uniform", "S1", "S2", "S3", "Official"], default="S3")
    args = parser.parse_args()

    paths = {
        "s1": args.s1_h5,
        "s2": args.s2_h5,
        "s3": args.s3_h5,
        "official_exp": args.official_exp,
        "official_ttcef": args.official_ttcef,
    }
    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"{name} not found: {path}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    image_dir = args.out_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    ref_times, meta = select_reference_times(
        args.s1_h5,
        args.s2_h5,
        args.s3_h5,
        args.official_exp,
        args.align_key,
        args.num_images,
        args.reference_source,
    )
    rows, _ = collect_rows(ref_times, paths, args.align_key)

    rendered_rows: list[dict[str, Any]] = []
    with h5py.File(args.s1_h5, "r") as f1, h5py.File(args.s2_h5, "r") as f2, h5py.File(args.s3_h5, "r") as f3, h5py.File(args.official_exp, "r") as fe, h5py.File(args.official_ttcef, "r") as fg:
        for row in tqdm(rows, desc="render time-aligned comparisons", unit="img"):
            out_png = image_dir / f"time_aligned_{row['image_index']:03d}_ref_{row['reference_time']}.png"
            extra = render_one(out_png, row, f1, f2, f3, fe, fg)
            row = {**row, **extra, "image_path": str(out_png)}
            rendered_rows.append(row)

    summary = aggregate(rendered_rows)
    result = {"meta": meta, "paths": {k: str(v) for k, v in paths.items()}, "summary": summary, "rows": rendered_rows}
    (args.out_dir / "time_aligned_comparison_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(args.out_dir / "time_aligned_selection.csv", rendered_rows)
    write_report(args.out_dir, paths, meta, rendered_rows, summary)
    print(json.dumps({"out_dir": str(args.out_dir), "num_images": len(rendered_rows)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
