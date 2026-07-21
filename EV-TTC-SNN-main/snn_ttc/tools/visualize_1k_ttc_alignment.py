#!/usr/bin/env python3
"""可视化 1k-event step 的事件、depth、TTC、inverse TTC 和 mask 对齐情况。"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np


def _finite_percentile(img: np.ndarray, lo: float = 2.0, hi: float = 98.0) -> tuple:
    """用分位数确定显示范围，避免少量异常值把图像动态范围拉爆。"""

    vals = img[np.isfinite(img)]
    if vals.size == 0:
        return 0.0, 1.0
    return float(np.percentile(vals, lo)), float(np.percentile(vals, hi))


def _imshow(ax, img: np.ndarray, title: str, cmap: str = "viridis", symmetric: bool = False) -> None:
    """统一单张子图的显示风格。"""

    img = np.asarray(img)
    if symmetric:
        vals = img[np.isfinite(img)]
        vmax = float(np.percentile(np.abs(vals), 98)) if vals.size else 1.0
        vmin = -vmax
    else:
        vmin, vmax = _finite_percentile(img)
        if np.isclose(vmin, vmax):
            vmax = vmin + 1.0
    im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=9)
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)


def visualize(h5_path: Path, out_dir: Path, start: int = 0, count: int = 10) -> list[Path]:
    """保存逐步对齐图，并额外生成 10 步横向总览图。"""

    out_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(h5_path, "r") as f:
        n = f["event_cnt"].shape[0]
        indices = list(range(start, min(start + count, n)))
        event_cnt = f["event_cnt"][indices]
        depth = f["depth_start"][indices]
        ttc = f["ttc_start"][indices]
        inv = f["inverse_ttc_start"][indices]
        mask = f["valid_ttc_mask"][indices]
        t_start = f["t_start"][indices]
        t_end = f["t_end"][indices]
        dt = f["dt"][indices]
        T = f["T"][indices]
        supervise_valid = f["supervise_valid"][indices]

    saved: list[Path] = []
    for local_i, step_idx in enumerate(indices):
        # 单步图保留完整诊断信息，方便逐帧检查事件和标签是否同向变化。
        fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
        fig.suptitle(
            f"step {step_idx} | t={int(t_start[local_i])}-{int(t_end[local_i])} us | "
            f"dt={int(dt[local_i])} us | Tz={T[local_i, 2]:.4f} | "
            f"supervise_valid={bool(supervise_valid[local_i])}",
            fontsize=11,
        )
        _imshow(axes[0, 0], event_cnt[local_i, 0], "positive event count", "magma")
        _imshow(axes[0, 1], event_cnt[local_i, 1], "negative event count", "magma")
        _imshow(axes[0, 2], depth[local_i], "depth", "viridis")
        _imshow(axes[1, 0], ttc[local_i], "TTC", "coolwarm", symmetric=True)
        _imshow(axes[1, 1], inv[local_i], "inverse TTC", "inferno")
        _imshow(axes[1, 2], mask[local_i].astype(float), "valid TTC mask", "gray")
        path = out_dir / f"step_{step_idx:03d}_alignment.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        saved.append(path)

    # 总览图牺牲部分细节，重点看 10 步连续变化是否平滑。
    fig, axes = plt.subplots(4, len(indices), figsize=(2.4 * len(indices), 9), constrained_layout=True)
    if len(indices) == 1:
        axes = axes.reshape(4, 1)
    for col, step_idx in enumerate(indices):
        axes[0, col].imshow(event_cnt[col].sum(axis=0), cmap="magma")
        axes[0, col].set_title(f"step {step_idx}\nevents", fontsize=8)
        axes[1, col].imshow(depth[col], cmap="viridis", vmin=_finite_percentile(depth[col])[0], vmax=_finite_percentile(depth[col])[1])
        axes[1, col].set_title("depth", fontsize=8)
        v = np.percentile(np.abs(ttc[col][np.isfinite(ttc[col])]), 98) if np.isfinite(ttc[col]).any() else 1.0
        axes[2, col].imshow(ttc[col], cmap="coolwarm", vmin=-v, vmax=v)
        axes[2, col].set_title(f"TTC\nTz={T[col, 2]:.3f}", fontsize=8)
        axes[3, col].imshow(mask[col], cmap="gray")
        axes[3, col].set_title(f"mask\nvalid={bool(supervise_valid[col])}", fontsize=8)
        for row in range(4):
            axes[row, col].axis("off")
    overview = out_dir / "overview_10_steps.png"
    fig.savefig(overview, dpi=150)
    plt.close(fig)
    saved.append(overview)
    return saved


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=10)
    args = parser.parse_args()
    out_dir = args.out_dir or (args.h5.parent / "visualizations")
    saved = visualize(args.h5, out_dir, start=args.start, count=args.count)
    print(f"visualization_dir: {out_dir}")
    for path in saved:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
