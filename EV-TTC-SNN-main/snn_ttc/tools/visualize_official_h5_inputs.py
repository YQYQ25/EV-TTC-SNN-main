#!/usr/bin/env python3
"""均匀抽样可视化官方 EV-TTC H5 输入。

用途：
  只读 train/val H5，不加载模型、不占 GPU。每张图展示 6 个 IIR 输入通道、
  GT TTC、inverse TTC、训练 mask 和基础元数据，便于在训练时检查输入是否正常。
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import h5py
import hdf5plugin  # noqa: F401  # 注册压缩插件，兼容带 hdf5plugin 的 H5
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# 图内文字使用英文，避免 matplotlib 中文字体匹配失败造成标题方框。
plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="可视化官方 EV-TTC H5 输入样本")
    parser.add_argument("--h5", required=True, type=Path, help="输入 H5，例如 train.h5")
    parser.add_argument("--out-dir", required=True, type=Path, help="图片和报告保存目录")
    parser.add_argument("--count", type=int, default=200, help="均匀抽样图片数量")
    parser.add_argument("--dpi", type=int, default=110, help="输出图片 DPI")
    return parser.parse_args()


def finite_quantile(arr: np.ndarray, qs: tuple[float, float], default: tuple[float, float]) -> tuple[float, float]:
    """对有限值计算分位数，避免 NaN/Inf 导致色条异常。"""
    vals = arr[np.isfinite(arr)]
    if vals.size == 0:
        return default
    lo, hi = np.quantile(vals, qs)
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        return default
    return float(lo), float(hi)


def symmetric_limits(arr: np.ndarray) -> tuple[float, float]:
    """IIR 通道有正负号，用对称色条更容易看极性结构。"""
    vals = arr[np.isfinite(arr)]
    if vals.size == 0:
        return -1.0, 1.0
    hi = float(np.quantile(np.abs(vals), 0.995))
    hi = max(hi, 1e-6)
    return -hi, hi


def dataset_value(handle: h5py.File, key: str, idx: int, default: Any = None) -> Any:
    if key not in handle:
        return default
    value = handle[key][idx]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    return value


def short_vec(value: Any) -> str:
    """把 T/Omega 等向量压短，避免元数据文字挤出图片。"""
    if value is None:
        return "None"
    arr = np.asarray(value, dtype=np.float64)
    return np.array2string(arr, precision=4, separator=", ")


def render_sample(handle: h5py.File, idx: int, out_path: Path, dpi: int) -> dict[str, Any]:
    exp = handle["exp_filts"][idx].astype(np.float32)
    ttc = handle["ttc"][idx].astype(np.float32)
    mask = handle["mask"][idx].astype(bool)
    source_index = dataset_value(handle, "source_index", idx, idx)
    ts = dataset_value(handle, "ts", idx, None)
    te = dataset_value(handle, "te", idx, None)
    exp_time = dataset_value(handle, "exp_time", idx, None)
    t_vec = dataset_value(handle, "T", idx, None)
    omega_vec = dataset_value(handle, "Omega", idx, None)

    valid_ttc = ttc[mask & np.isfinite(ttc)]
    ttc_lo, ttc_hi = finite_quantile(valid_ttc, (0.02, 0.98), (0.0, 10.0))
    inv_ttc = np.zeros_like(ttc, dtype=np.float32)
    valid_inv = mask & np.isfinite(ttc) & (np.abs(ttc) > 1e-6)
    inv_ttc[valid_inv] = 1.0 / ttc[valid_inv]
    inv_lo, inv_hi = finite_quantile(inv_ttc[valid_inv], (0.02, 0.98), (-1.0, 1.0))

    fig, axes = plt.subplots(3, 4, figsize=(18, 12), constrained_layout=True)
    axes = axes.reshape(-1)
    for ax in axes:
        ax.axis("off")

    for ch in range(6):
        lo, hi = symmetric_limits(exp[ch])
        im = axes[ch].imshow(exp[ch], cmap="coolwarm", vmin=lo, vmax=hi)
        axes[ch].set_title(f"IIR input ch{ch}")
        fig.colorbar(im, ax=axes[ch], fraction=0.046, pad=0.03)

    event_support = np.abs(exp[-1]) > 1e-3
    extra = [
        ("ch5 event support", event_support.astype(np.float32), "gray", 0.0, 1.0),
        ("GT TTC(s)", np.where(mask, ttc, np.nan), "viridis", ttc_lo, ttc_hi),
        ("inverse TTC", np.where(valid_inv, inv_ttc, np.nan), "coolwarm", inv_lo, inv_hi),
        ("training mask", mask.astype(np.float32), "gray", 0.0, 1.0),
    ]
    for pos, (title, arr, cmap, lo, hi) in enumerate(extra, start=6):
        im = axes[pos].imshow(arr, cmap=cmap, vmin=lo, vmax=hi)
        axes[pos].set_title(title)
        fig.colorbar(im, ax=axes[pos], fraction=0.046, pad=0.03)

    axes[10].axis("on")
    if valid_ttc.size:
        axes[10].hist(valid_ttc.ravel(), bins=60, color="#365c9a", alpha=0.85)
    axes[10].set_title("valid TTC histogram")

    axes[11].axis("on")
    text_lines = [
        f"H5 index: {idx}",
        f"source index: {source_index}",
        f"ts/te: {ts} / {te}",
        f"exp_time: {exp_time}",
        f"mask ratio: {float(mask.mean()):.4f}",
        f"valid pixels: {int(mask.sum())}",
        f"T: {short_vec(t_vec)}",
        f"Omega: {short_vec(omega_vec)}",
        f"IIR range: [{float(np.nanmin(exp)):.4g}, {float(np.nanmax(exp)):.4g}]",
        f"TTC range(valid): [{float(np.nanmin(valid_ttc)):.4g}, {float(np.nanmax(valid_ttc)):.4g}]" if valid_ttc.size else "TTC range(valid): empty",
    ]
    axes[11].text(0.02, 0.98, "\n".join(text_lines), va="top", ha="left", fontsize=10)
    axes[11].set_xticks([])
    axes[11].set_yticks([])
    axes[11].set_title("metadata")

    fig.suptitle(f"Official EV-TTC train input | row={idx} | source={source_index}", fontsize=15)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)

    return {
        "h5_index": int(idx),
        "source_index": int(source_index) if source_index is not None else "",
        "ts": ts,
        "te": te,
        "exp_time": exp_time,
        "mask_ratio": float(mask.mean()),
        "valid_pixels": int(mask.sum()),
        "iir_min": float(np.nanmin(exp)),
        "iir_max": float(np.nanmax(exp)),
        "ttc_valid_min": float(np.nanmin(valid_ttc)) if valid_ttc.size else "",
        "ttc_valid_max": float(np.nanmax(valid_ttc)) if valid_ttc.size else "",
        "image": str(out_path),
    }


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


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.h5, "r") as handle:
        total = int(handle["exp_filts"].shape[0])
        count = min(args.count, total)
        indices = np.linspace(0, total - 1, count).round().astype(np.int64)

        rows: list[dict[str, Any]] = []
        for n, idx in enumerate(indices, start=1):
            out_path = args.out_dir / "images" / f"train_input_{n:03d}_row_{int(idx):06d}.png"
            rows.append(render_sample(handle, int(idx), out_path, args.dpi))
            if n == 1 or n % 20 == 0 or n == count:
                print(f"[visualize] {n}/{count} row={int(idx)} -> {out_path}", flush=True)

    write_csv(args.out_dir / "sample_index.csv", rows)
    summary = {
        "h5": str(args.h5),
        "total_samples": total,
        "requested_count": args.count,
        "rendered_count": len(rows),
        "first_index": rows[0]["h5_index"] if rows else None,
        "last_index": rows[-1]["h5_index"] if rows else None,
        "mean_mask_ratio": float(np.mean([r["mask_ratio"] for r in rows])) if rows else None,
        "image_dir": str(args.out_dir / "images"),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.out_dir / "README.md").write_text(
        "\n".join(
            [
                "# 官方 EV-TTC 训练输入可视化",
                "",
                f"- H5：`{args.h5}`",
                f"- 总样本数：{total}",
                f"- 均匀抽样数量：{len(rows)}",
                f"- 图片目录：`{args.out_dir / 'images'}`",
                f"- 索引表：`{args.out_dir / 'sample_index.csv'}`",
                "",
                "每张图包含 6 个 IIR 输入通道、ch5 有事件区域、GT TTC、inverse TTC、训练 mask、TTC 分布和样本元数据。",
            ]
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
