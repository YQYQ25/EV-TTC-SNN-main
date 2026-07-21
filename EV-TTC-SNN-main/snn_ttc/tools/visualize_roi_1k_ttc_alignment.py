#!/usr/bin/env python3
"""可视化固定 ROI 内 1k 源事件与 TTC 标签的空间、时间对齐情况。"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np


def _range(img: np.ndarray, lo: float = 2.0, hi: float = 98.0) -> tuple[float, float]:
    """用有限值分位数做显示范围，减少极端值对可视化的影响。"""

    vals = img[np.isfinite(img)]
    if vals.size == 0:
        return 0.0, 1.0
    vmin, vmax = float(np.percentile(vals, lo)), float(np.percentile(vals, hi))
    return (vmin, vmax + 1.0) if np.isclose(vmin, vmax) else (vmin, vmax)


def _imshow(
    ax,
    img: np.ndarray,
    title: str,
    cmap: str = "viridis",
    symmetric: bool = False,
    fixed_range: tuple[float, float] | None = None,
) -> None:
    """统一显示单个诊断子图。"""

    if fixed_range is not None:
        vmin, vmax = fixed_range
    elif symmetric:
        vals = img[np.isfinite(img)]
        vmax = float(np.percentile(np.abs(vals), 98)) if vals.size else 1.0
        vmin = -vmax
    else:
        vmin, vmax = _range(img)
    im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=9)
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)


def visualize(h5_path: Path, out_dir: Path, start: int = 0, count: int = 10) -> list[Path]:
    """保存单步图和 10 步横向总览图。"""

    out_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(h5_path, "r") as f:
        n = f["event_cnt"].shape[0]
        ids = list(range(start, min(start + count, n)))
        event_cnt = f["event_cnt"][ids]
        depth = f["depth_start"][ids]
        ttc = f["ttc_start"][ids]
        inv = f["inverse_ttc_start"][ids]
        mask = f["valid_ttc_mask"][ids]
        raw_start = f["raw_event_start_idx"][ids]
        raw_end = f["raw_event_end_idx"][ids]
        t_start = f["t_start"][ids]
        t_end = f["t_end"][ids]
        dt = f["dt"][ids]
        T = f["T"][ids]
        roi_count = f["roi_source_event_count"][ids]
        total_weight = f["total_mapped_weight"][ids]
        supervise_valid = f["supervise_valid"][ids]
        roi_x0 = int(f.attrs["roi_x0"])
        roi_y0 = int(f.attrs["roi_y0"])
        roi_size = int(f.attrs["roi_size"])

    saved: list[Path] = []
    for local_i, step_idx in enumerate(ids):
        overlay = event_cnt[local_i, 0] - event_cnt[local_i, 1]
        fig, axes = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)
        fig.suptitle(
            f"step {step_idx} | raw=[{int(raw_start[local_i])},{int(raw_end[local_i])}) | "
            f"t={int(t_start[local_i])}-{int(t_end[local_i])} us | dt={int(dt[local_i])} us | "
            f"Tz={T[local_i, 2]:.4f} | ROI=({roi_x0},{roi_y0},{roi_size}) | "
            f"src={int(roi_count[local_i])} | weight={total_weight[local_i]:.2f} | "
            f"supervise={bool(supervise_valid[local_i])}",
            fontsize=10,
        )
        _imshow(axes[0, 0], event_cnt[local_i, 0], "positive channel", "magma")
        _imshow(axes[0, 1], event_cnt[local_i, 1], "negative channel", "magma")
        _imshow(axes[0, 2], overlay, "positive - negative", "coolwarm", symmetric=True)
        _imshow(axes[0, 3], depth[local_i], "depth", "viridis")
        _imshow(axes[1, 0], ttc[local_i], "TTC", "coolwarm", symmetric=False)
        _imshow(axes[1, 1], inv[local_i], "inverse TTC", "inferno")
        # mask 是二值图，固定 0/1 显示范围，避免“几乎全有效”时自动缩放成全黑。
        _imshow(axes[1, 2], mask[local_i].astype(float), "valid_ttc_mask", "gray", fixed_range=(0.0, 1.0))
        axes[1, 3].axis("off")
        axes[1, 3].text(
            0.0,
            0.95,
            "\n".join(
                [
                    f"step index: {step_idx}",
                    f"raw range: [{int(raw_start[local_i])}, {int(raw_end[local_i])})",
                    f"t_start: {int(t_start[local_i])}",
                    f"t_end: {int(t_end[local_i])}",
                    f"dt: {int(dt[local_i])} us",
                    f"Tz: {T[local_i, 2]:.6f}",
                    f"roi_source_event_count: {int(roi_count[local_i])}",
                    f"total_mapped_weight: {total_weight[local_i]:.3f}",
                    f"supervise_valid: {bool(supervise_valid[local_i])}",
                    f"ROI: x=[{roi_x0}:{roi_x0+roi_size}), y=[{roi_y0}:{roi_y0+roi_size})",
                ]
            ),
            va="top",
            fontsize=10,
        )
        path = out_dir / f"roi_step_{step_idx:03d}_alignment.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        saved.append(path)

    # 横向总览用于快速检查 10 步内事件与标签的连续变化。
    fig, axes = plt.subplots(5, len(ids), figsize=(2.4 * len(ids), 10), constrained_layout=True)
    if len(ids) == 1:
        axes = axes.reshape(5, 1)
    for col, step_idx in enumerate(ids):
        overlay = event_cnt[col, 0] - event_cnt[col, 1]
        axes[0, col].imshow(event_cnt[col].sum(axis=0), cmap="magma")
        axes[0, col].set_title(f"{step_idx}\nevent", fontsize=8)
        vmax = np.percentile(np.abs(overlay), 98) if np.any(np.isfinite(overlay)) else 1.0
        axes[1, col].imshow(overlay, cmap="coolwarm", vmin=-vmax, vmax=vmax)
        axes[1, col].set_title("pos-neg", fontsize=8)
        axes[2, col].imshow(depth[col], cmap="viridis", vmin=_range(depth[col])[0], vmax=_range(depth[col])[1])
        axes[2, col].set_title("depth", fontsize=8)
        axes[3, col].imshow(ttc[col], cmap="viridis", vmin=_range(ttc[col])[0], vmax=_range(ttc[col])[1])
        axes[3, col].set_title(f"TTC\nTz={T[col, 2]:.3f}", fontsize=8)
        axes[4, col].imshow(mask[col], cmap="gray")
        axes[4, col].set_title(f"mask\n{bool(supervise_valid[col])}", fontsize=8)
        for row in range(5):
            axes[row, col].axis("off")
    overview = out_dir / "roi_overview_10_steps.png"
    fig.savefig(overview, dpi=150)
    plt.close(fig)
    saved.append(overview)
    return saved


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=10)
    args = parser.parse_args()
    saved = visualize(args.h5, args.out_dir, start=args.start, count=args.count)
    print(f"visualization_dir: {args.out_dir}")
    for path in saved:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
