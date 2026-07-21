#!/usr/bin/env python3
"""为官方 EV-TTC 单序列复现实验生成密集预测可视化。

特点：
  1. 训练集和验证集都按索引均匀抽样；
  2. 只读 raw H5，不做训练增强，便于和原始输入一一对应；
  3. 同时展示 raw IIR 与 event-only IIR。event-only 会把近零无事件区域置白，
     避免把 0 像素映射颜色后误判成真实事件纹理；
  4. 使用同一个 checkpoint 做推理，输出 GT / Prediction / Error / Mask。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import h5py
import hdf5plugin  # noqa: F401  # 注册 HDF5 压缩插件
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[3]
EVTTC_MODEL_ROOT = PROJECT_ROOT / "EV-TTC-main" / "model"
if str(EVTTC_MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(EVTTC_MODEL_ROOT))

from evslim import EVSlim  # noqa: E402


def create_evslim_cfg() -> SimpleNamespace:
    """与当前官方 car 单序列复现实验一致的 EV-Slim 配置。"""

    return SimpleNamespace(
        enc_k_size=[3, 3, 3],
        enc_channels=[16, 16, 16],
        aspp_channels=[32, 16],
        rate=[12, 24, 36],
        dec_k_size=[3, 3, 3],
        dec_channels=[8, 8, 1],
        norm=True,
        in_channels=6,
        input=6,
        ds="TTCEF",
        task="ttc",
        act="ReLU",
        alpha=0.45,
        flip_prob=0.3,
        res=[360, 360],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成官方 EV-TTC train/val 密集预测可视化")
    parser.add_argument("--checkpoint", required=True, type=Path, help="best/latest checkpoint")
    parser.add_argument("--train-h5", required=True, type=Path)
    parser.add_argument("--val-h5", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--train-count", type=int, default=100)
    parser.add_argument("--val-count", type=int, default=100)
    parser.add_argument("--event-eps", type=float, default=1e-3, help="abs(IIR)<=eps 视为无事件区域")
    parser.add_argument("--dpi", type=int, default=120)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def uniform_indices(length: int, count: int) -> np.ndarray:
    count = max(0, min(int(count), int(length)))
    if count == 0:
        return np.asarray([], dtype=np.int64)
    return np.linspace(0, length - 1, count).round().astype(np.int64)


def compute_channel_limits(files_and_indices: list[tuple[Path, np.ndarray]], quantile: float = 0.995) -> list[float]:
    """在抽样样本上估计每个 IIR 通道的统一色条范围，便于跨 train/val 对比。"""

    values: list[list[np.ndarray]] = [[] for _ in range(6)]
    for h5_path, indices in files_and_indices:
        with h5py.File(h5_path, "r") as handle:
            for idx in indices:
                exp = handle["exp_filts"][int(idx)].astype(np.float32)
                for ch in range(6):
                    vals = np.abs(exp[ch].ravel())
                    # 每帧最多采样一部分像素，避免为了定色条占用太多内存。
                    if vals.size > 8192:
                        take = np.linspace(0, vals.size - 1, 8192, dtype=np.int64)
                        vals = vals[take]
                    values[ch].append(vals)
    limits: list[float] = []
    for ch in range(6):
        if not values[ch]:
            limits.append(1.0)
            continue
        pooled = np.concatenate(values[ch])
        hi = float(np.quantile(pooled[np.isfinite(pooled)], quantile))
        limits.append(max(hi, 1e-6))
    return limits


def finite_quantile(arr: np.ndarray, qs: tuple[float, float], default: tuple[float, float]) -> tuple[float, float]:
    vals = arr[np.isfinite(arr)]
    if vals.size == 0:
        return default
    lo, hi = np.quantile(vals, qs)
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        return default
    return float(lo), float(hi)


def compact_vec(value: Any) -> str:
    arr = np.asarray(value, dtype=np.float64)
    return np.array2string(arr, precision=4, separator=", ")


def load_model(checkpoint: Path, device: torch.device) -> EVSlim:
    model = EVSlim(create_evslim_cfg()).to(device)
    ckpt = torch.load(checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def render_one(
    *,
    split: str,
    row_id: int,
    h5_index: int,
    handle: h5py.File,
    model: EVSlim,
    device: torch.device,
    channel_limits: list[float],
    event_eps: float,
    out_path: Path,
    dpi: int,
) -> dict[str, Any]:
    exp = handle["exp_filts"][h5_index].astype(np.float32)
    ttc = handle["ttc"][h5_index].astype(np.float32)
    mask = handle["mask"][h5_index].astype(bool)
    source_index = int(handle["source_index"][h5_index]) if "source_index" in handle else int(h5_index)
    ts = float(handle["ts"][h5_index]) if "ts" in handle else math.nan
    te = float(handle["te"][h5_index]) if "te" in handle else math.nan
    t_vec = handle["T"][h5_index] if "T" in handle else np.asarray([])
    omega_vec = handle["Omega"][h5_index] if "Omega" in handle else np.asarray([])

    x = torch.from_numpy(exp)[None].to(device)
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=device.type == "cuda"):
        pred = model(x)[0, 0].detach().float().cpu().numpy()

    valid = mask & np.isfinite(ttc) & np.isfinite(pred)
    err = np.abs(pred - ttc)
    valid_ttc = ttc[valid]
    valid_pred = pred[valid]
    valid_err = err[valid]
    valid_rel = valid_err / np.maximum(np.abs(valid_ttc), 1e-6) if valid_ttc.size else np.asarray([], dtype=np.float32)
    ttc_lo, ttc_hi = finite_quantile(np.concatenate([valid_ttc, valid_pred]) if valid_ttc.size else np.asarray([]), (0.02, 0.98), (0.0, 10.0))
    err_hi = float(np.quantile(valid_err, 0.98)) if valid_err.size else 1.0
    err_hi = max(err_hi, 1e-6)

    cmap_iir = plt.get_cmap("coolwarm").copy()
    cmap_iir.set_bad("white")

    fig, axes = plt.subplots(3, 6, figsize=(24, 12), constrained_layout=True)
    for ax in axes.reshape(-1):
        ax.axis("off")

    for ch in range(6):
        hi = channel_limits[ch]
        im = axes[0, ch].imshow(exp[ch], cmap="coolwarm", vmin=-hi, vmax=hi)
        axes[0, ch].set_title(f"raw IIR ch{ch}")
        fig.colorbar(im, ax=axes[0, ch], fraction=0.046, pad=0.03)

    for ch in range(6):
        hi = channel_limits[ch]
        event_only = np.ma.masked_where(np.abs(exp[ch]) <= event_eps, exp[ch])
        im = axes[1, ch].imshow(event_only, cmap=cmap_iir, vmin=-hi, vmax=hi)
        axes[1, ch].set_title(f"event-only IIR ch{ch}")
        fig.colorbar(im, ax=axes[1, ch], fraction=0.046, pad=0.03)

    panels = [
        ("GT TTC(s)", np.where(valid, ttc, np.nan), "viridis", ttc_lo, ttc_hi),
        ("Prediction TTC(s)", np.where(valid, pred, np.nan), "viridis", ttc_lo, ttc_hi),
        ("Abs Error(s)", np.where(valid, err, np.nan), "inferno", 0.0, err_hi),
        ("Mask", mask.astype(np.float32), "gray", 0.0, 1.0),
    ]
    for col, (title, arr, cmap, lo, hi) in enumerate(panels):
        im = axes[2, col].imshow(arr, cmap=cmap, vmin=lo, vmax=hi)
        axes[2, col].set_title(title)
        fig.colorbar(im, ax=axes[2, col], fraction=0.046, pad=0.03)

    axes[2, 4].axis("on")
    if valid_ttc.size:
        axes[2, 4].hist(valid_ttc.ravel(), bins=50, alpha=0.55, label="GT")
        axes[2, 4].hist(valid_pred.ravel(), bins=50, alpha=0.55, label="Pred")
        axes[2, 4].legend(fontsize=8)
    axes[2, 4].set_title("GT/Pred histogram")

    axes[2, 5].axis("on")
    axes[2, 5].set_xticks([])
    axes[2, 5].set_yticks([])
    mae = float(np.mean(valid_err)) if valid_err.size else math.nan
    medae = float(np.median(valid_err)) if valid_err.size else math.nan
    mre = float(np.mean(valid_rel)) if valid_rel.size else math.nan
    text = "\n".join(
        [
            f"split: {split}",
            f"row: {h5_index}",
            f"source: {source_index}",
            f"ts/te: {ts:.0f} / {te:.0f}",
            f"mask ratio: {float(mask.mean()):.4f}",
            f"valid pixels: {int(valid.sum())}",
            f"MAE: {mae:.4f}s",
            f"Median AE: {medae:.4f}s",
            f"MRE: {mre:.4f}",
            f"T: {compact_vec(t_vec)}",
            f"Omega: {compact_vec(omega_vec)}",
        ]
    )
    axes[2, 5].text(0.02, 0.98, text, va="top", ha="left", fontsize=9)
    axes[2, 5].set_title("metadata")

    fig.suptitle(
        f"{split} uniform prediction | sample={row_id:03d} | row={h5_index} | source={source_index}",
        fontsize=15,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)

    return {
        "split": split,
        "sample_id": row_id,
        "h5_index": int(h5_index),
        "source_index": source_index,
        "ts": ts,
        "te": te,
        "mask_ratio": float(mask.mean()),
        "valid_pixels": int(valid.sum()),
        "mae": mae,
        "median_ae": medae,
        "mre": mre,
        "pred_min": float(np.nanmin(pred)),
        "pred_max": float(np.nanmax(pred)),
        "gt_valid_min": float(np.nanmin(valid_ttc)) if valid_ttc.size else "",
        "gt_valid_max": float(np.nanmax(valid_ttc)) if valid_ttc.size else "",
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
    device = torch.device("cuda:0" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = load_model(args.checkpoint, device)

    with h5py.File(args.train_h5, "r") as train_h5, h5py.File(args.val_h5, "r") as val_h5:
        train_indices = uniform_indices(int(train_h5["exp_filts"].shape[0]), args.train_count)
        val_indices = uniform_indices(int(val_h5["exp_filts"].shape[0]), args.val_count)

    channel_limits = compute_channel_limits([(args.train_h5, train_indices), (args.val_h5, val_indices)])
    rows: list[dict[str, Any]] = []
    for split, h5_path, indices in [("train", args.train_h5, train_indices), ("val", args.val_h5, val_indices)]:
        with h5py.File(h5_path, "r") as handle:
            for row_id, h5_index in enumerate(indices, start=1):
                out_path = args.out_dir / split / f"{split}_{row_id:03d}_row_{int(h5_index):06d}_source_{int(handle['source_index'][int(h5_index)]):06d}.png"
                rows.append(
                    render_one(
                        split=split,
                        row_id=row_id,
                        h5_index=int(h5_index),
                        handle=handle,
                        model=model,
                        device=device,
                        channel_limits=channel_limits,
                        event_eps=args.event_eps,
                        out_path=out_path,
                        dpi=args.dpi,
                    )
                )
                if row_id == 1 or row_id % 10 == 0 or row_id == len(indices):
                    print(f"[{split}] {row_id}/{len(indices)} row={int(h5_index)} -> {out_path}", flush=True)

    write_csv(args.out_dir / "visualization_index.csv", rows)
    summary = {
        "checkpoint": str(args.checkpoint),
        "train_h5": str(args.train_h5),
        "val_h5": str(args.val_h5),
        "train_count": int(len(train_indices)),
        "val_count": int(len(val_indices)),
        "event_eps": args.event_eps,
        "channel_limits_abs_995": channel_limits,
        "note": "raw IIR 显示真实输入数值；event-only IIR 将 abs(IIR)<=event_eps 的无事件区域置白。",
    }
    (args.out_dir / "visualization_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.out_dir / "README.md").write_text(
        "\n".join(
            [
                "# 官方 EV-TTC 20epoch 密集可视化",
                "",
                f"- checkpoint：`{args.checkpoint}`",
                f"- train H5：`{args.train_h5}`",
                f"- val H5：`{args.val_h5}`",
                f"- train 抽样：{len(train_indices)} 张",
                f"- val 抽样：{len(val_indices)} 张",
                f"- event-only 阈值：`abs(IIR) <= {args.event_eps}` 显示为白色",
                "",
                "每张图第一行是 raw IIR，表示模型真实输入；第二行是 event-only IIR，用于排除无事件 0 背景颜色的视觉干扰；第三行是 GT、预测、误差、mask、分布和元数据。",
            ]
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
