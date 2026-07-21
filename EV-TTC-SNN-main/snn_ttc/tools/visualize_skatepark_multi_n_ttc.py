#!/usr/bin/env python3
"""可视化 Skatepark 多事件数 TTC 审计子集。"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np


def _range(img: np.ndarray, lo=2, hi=98):
    vals = img[np.isfinite(img)]
    if vals.size == 0:
        return 0.0, 1.0
    v0, v1 = float(np.percentile(vals, lo)), float(np.percentile(vals, hi))
    return (v0, v1 + 1.0) if np.isclose(v0, v1) else (v0, v1)


def _imshow(ax, img, title, cmap="viridis", symmetric=False, fixed=None):
    if fixed is not None:
        vmin, vmax = fixed
    elif symmetric:
        vals = img[np.isfinite(img)]
        vmax = float(np.percentile(np.abs(vals), 98)) if vals.size else 1.0
        vmin = -vmax
    else:
        vmin, vmax = _range(img)
    im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=8)
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)


def visualize_one(audit_h5: Path, out_dir: Path, max_blocks: int = 10) -> list[Path]:
    """每个 N 保存 10 个完整 block 的逐步图和总览图。"""

    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    with h5py.File(audit_h5, "r") as f:
        N = int(f.attrs["N"])
        block_ids = f["block_index"][:]
        unique_blocks = np.unique(block_ids)[:max_blocks]
        for block_id in unique_blocks:
            idx = np.where(block_ids == block_id)[0]
            for local_j, i in enumerate(idx):
                event_cnt = f["event_cnt"][i]
                depth = f["depth_start"][i]
                ttc = f["ttc_start"][i]
                inv = f["inverse_ttc_start"][i]
                mask = f["valid_ttc_mask"][i].astype(bool)
                overlay = event_cnt[0] - event_cnt[1]
                T = f["T"][i]
                Omega = f["Omega"][i]
                valid_ratio = float(mask.mean())
                min_ttc = float(np.min(ttc[mask])) if np.any(mask) else np.nan
                event_dt = int(f["event_dt"][i])
                roi_x0 = int(f["roi_x0"][i])
                roi_y0 = int(f["roi_y0"][i])
                supervise = bool(f["supervise_valid"][i])
                fig, axes = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)
                fig.suptitle(
                    f"N={N} | block={int(block_id)} step={local_j} | ROI=({roi_x0},{roi_y0}) | "
                    f"dt={event_dt}us | Tz={T[2]:.4f} | ||T||={np.linalg.norm(T):.4f} | "
                    f"||Omega||={np.linalg.norm(Omega):.4f} | supervise={supervise} | "
                    f"mask={valid_ratio:.3f} | minTTC={min_ttc:.3f}",
                    fontsize=9,
                )
                _imshow(axes[0, 0], event_cnt[0], "positive", "magma")
                _imshow(axes[0, 1], event_cnt[1], "negative", "magma")
                _imshow(axes[0, 2], overlay, "pos-neg", "coolwarm", symmetric=True)
                _imshow(axes[0, 3], depth, "depth", "viridis")
                _imshow(axes[1, 0], ttc, "TTC", "viridis")
                _imshow(axes[1, 1], inv, "inverse TTC", "inferno")
                _imshow(axes[1, 2], mask.astype(float), "valid mask", "gray", fixed=(0, 1))
                axes[1, 3].axis("off")
                axes[1, 3].text(0, 0.95, f"t_start={int(f['t_start'][i])}\nt_end={int(f['t_end'][i])}\nfixed audit subset", va="top")
                path = out_dir / f"N{N}_block{int(block_id):05d}_step{local_j:02d}.png"
                fig.savefig(path, dpi=130)
                plt.close(fig)
                saved.append(path)

            # 10步总览。
            fig, axes = plt.subplots(5, len(idx), figsize=(2.2 * len(idx), 10), constrained_layout=True)
            if len(idx) == 1:
                axes = axes.reshape(5, 1)
            for col, i in enumerate(idx):
                event_cnt = f["event_cnt"][i]
                axes[0, col].imshow(event_cnt.sum(axis=0), cmap="magma")
                axes[0, col].set_title(f"s{col}\nevent", fontsize=7)
                axes[1, col].imshow(f["depth_start"][i], cmap="viridis")
                axes[1, col].set_title("depth", fontsize=7)
                axes[2, col].imshow(f["ttc_start"][i], cmap="viridis")
                axes[2, col].set_title("TTC", fontsize=7)
                axes[3, col].imshow(f["inverse_ttc_start"][i], cmap="inferno")
                axes[3, col].set_title("inv", fontsize=7)
                axes[4, col].imshow(f["valid_ttc_mask"][i], cmap="gray", vmin=0, vmax=1)
                axes[4, col].set_title("mask", fontsize=7)
                for row in range(5):
                    axes[row, col].axis("off")
            overview = out_dir / f"N{N}_block{int(block_id):05d}_overview.png"
            fig.savefig(overview, dpi=130)
            plt.close(fig)
            saved.append(overview)
    return saved


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("EV-TTC-SNN-main/debug_sets/skatepark_multi_n_ttc"))
    parser.add_argument("--max-blocks", type=int, default=10)
    args = parser.parse_args()
    vis_root = args.out_dir / "visualizations"
    for h5 in sorted(args.out_dir.glob("audit_subset_N*.h5")):
        visualize_one(h5, vis_root / h5.stem, args.max_blocks)
    print(f"visualizations: {vis_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

