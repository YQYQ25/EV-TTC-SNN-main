#!/usr/bin/env python3
"""按固定epoch间隔生成 LIF-EV-FlowNet-TTC 输出时间序列可视化。

图像版式对齐 32Block 实验中的“05_output_预测GT误差”：
Prediction、GT、Mask、Masked Prediction、Abs Error 和有效区域直方图。
本脚本只做推理与画图，不训练、不更新参数。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import hdf5plugin  # noqa: F401  # H5压缩插件
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SNN_ROOT = PROJECT_ROOT / "EV-TTC-SNN-main"
if str(SNN_ROOT) not in sys.path:
    sys.path.insert(0, str(SNN_ROOT))

from snn_ttc.models import LIFEVFlowNetTTC  # noqa: E402
from snn_ttc.tools.train_lif_evflownet_ttc_500block_pipeline import (  # noqa: E402
    DEFAULT_H5,
    DEFAULT_OUT_DIR,
    S2BlockDataset,
    build_split_rows,
    load_h5_meta,
    make_loader,
)


@dataclass
class OutputCapture:
    epoch_label: str
    split: str
    anchor_name: str
    block_index: int
    step_id: int
    prediction: np.ndarray
    gt: np.ndarray
    mask: np.ndarray
    abs_error: np.ndarray


def parse_epoch_list(text: str) -> list[int]:
    return [int(part) for part in text.split(",") if part.strip()]


def read_best_epoch(run_dir: Path) -> int | None:
    metrics = run_dir / "val_epoch_metrics.csv"
    if not metrics.exists():
        return None
    with metrics.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return None
    best = min(rows, key=lambda row: float(row["masked_MAE"]))
    return int(best["epoch"])


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def checkpoint_for_epoch(run_dir: Path, epoch: int, *, best_epoch: int | None = None) -> Path:
    if epoch == 0:
        return run_dir / "checkpoints" / "epoch_000.pt"
    regular = run_dir / "checkpoints" / f"epoch_{epoch:03d}.pt"
    if regular.exists():
        return regular
    # Best epoch 不一定按固定间隔保存；此时用 best_val_mae.pt 补齐同一时刻输出图。
    if best_epoch is not None and epoch == best_epoch:
        return run_dir / "checkpoints" / "best_val_mae.pt"
    return regular


def load_model_from_checkpoint(path: Path, device: torch.device) -> LIFEVFlowNetTTC:
    model = LIFEVFlowNetTTC().to(device)
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    model.reset_states()
    return model


def collect_outputs(
    model: LIFEVFlowNetTTC,
    loader: torch.utils.data.DataLoader,
    *,
    epoch_label: str,
    split: str,
    anchors: dict[str, tuple[int, int]],
    device: torch.device,
    input_scale: float,
) -> list[OutputCapture]:
    """从split起点按时间重放，到固定锚点时保存输出图所需数组。"""

    captures: list[OutputCapture] = []
    remaining = dict(anchors)
    max_block = max(block for block, _ in anchors.values())
    model.reset_states()
    with torch.no_grad():
        for batch in loader:
            local_block = int(batch["local_block_index"][0].item())
            if bool(batch["block_reset_required"][0].item()):
                model.reset_states()
            for step_id in range(batch["events"].shape[1]):
                event_raw = batch["events"][:, step_id].to(device, non_blocking=True).float()
                target_t = batch["target"][:, step_id].to(device, non_blocking=True).float()
                mask_t = batch["mask"][:, step_id].to(device, non_blocking=True).bool()
                pred_t = model(event_raw * input_scale)

                for name, (anchor_block, anchor_step) in list(remaining.items()):
                    if local_block == anchor_block and step_id == anchor_step:
                        pred = pred_t[0, 0].detach().cpu().float().numpy()
                        gt = target_t[0].detach().cpu().float().numpy()
                        mask = mask_t[0].detach().cpu().bool().numpy()
                        captures.append(
                            OutputCapture(
                                epoch_label=epoch_label,
                                split=split,
                                anchor_name=name,
                                block_index=local_block,
                                step_id=step_id,
                                prediction=pred,
                                gt=gt,
                                mask=mask,
                                abs_error=np.abs(pred - gt) * mask.astype(np.float32),
                            )
                        )
                        del remaining[name]
                if not remaining and local_block >= max_block:
                    break
            model.detach_states()
            if not remaining and local_block >= max_block:
                break
    model.reset_states()
    return captures


def finite_percentile(values: list[np.ndarray], q: tuple[float, float], fallback: tuple[float, float]) -> tuple[float, float]:
    flat = []
    for arr in values:
        finite = np.isfinite(arr)
        if finite.any():
            flat.append(arr[finite].reshape(-1))
    if not flat:
        return fallback
    merged = np.concatenate(flat)
    lo, hi = np.percentile(merged, q)
    if abs(float(hi) - float(lo)) < 1e-12:
        hi = lo + 1e-6
    return float(lo), float(hi)


def compute_ranges(captures: list[OutputCapture]) -> dict[str, tuple[float, float]]:
    """所有epoch共用颜色范围，方便横向比较训练变化。"""

    valid_pred_gt: list[np.ndarray] = []
    valid_err: list[np.ndarray] = []
    for cap in captures:
        valid_pred_gt.extend([cap.prediction[cap.mask], cap.gt[cap.mask]])
        valid_err.append(cap.abs_error[cap.mask])
    value_range = finite_percentile(valid_pred_gt, (1, 99), (-1.0, 1.0))
    error_range = finite_percentile(valid_err, (1, 99), (0.0, 1.0))
    return {"value": value_range, "error": (0.0, error_range[1])}


def render_output_panel(cap: OutputCapture, out_path: Path, ranges: dict[str, tuple[float, float]]) -> None:
    """渲染单个epoch/锚点的05_output风格图。"""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    masked_pred = cap.prediction * cap.mask.astype(np.float32)
    panels = [
        (cap.prediction, "Prediction signed inverse TTC", "coolwarm", ranges["value"]),
        (cap.gt, "GT signed inverse TTC", "coolwarm", ranges["value"]),
        (cap.abs_error, "Abs Error", "magma", ranges["error"]),
        (cap.mask.astype(np.float32), "Mask", "gray", (0.0, 1.0)),
        (masked_pred, "Masked Prediction", "coolwarm", ranges["value"]),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    for ax, (arr, title, cmap, (vmin, vmax)) in zip(axes.flat[:5], panels):
        im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    ax = axes.flat[5]
    if cap.mask.any():
        ax.hist(cap.prediction[cap.mask].reshape(-1), bins=40, alpha=0.6, label="pred")
        ax.hist(cap.gt[cap.mask].reshape(-1), bins=40, alpha=0.6, label="gt")
    ax.set_title("Masked histogram")
    ax.legend()
    fig.suptitle(
        f"{cap.epoch_label} | {cap.split} {cap.anchor_name} | block={cap.block_index} step={cap.step_id}"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def render_anchor_timeline(captures: list[OutputCapture], out_path: Path, ranges: dict[str, tuple[float, float]]) -> None:
    """把同一个锚点的不同时刻拼成纵向时间序列，便于一眼看训练演化。"""

    if not captures:
        return
    captures = sorted(captures, key=lambda cap: cap.epoch_label)
    rows = len(captures)
    fig, axes = plt.subplots(rows, 4, figsize=(14, 3.2 * rows))
    if rows == 1:
        axes = axes[None, :]
    for row, cap in enumerate(captures):
        panel_data = [
            (cap.prediction, "Prediction", "coolwarm", ranges["value"]),
            (cap.gt, "GT", "coolwarm", ranges["value"]),
            (cap.abs_error, "Abs Error", "magma", ranges["error"]),
            (cap.mask.astype(np.float32), "Mask", "gray", (0.0, 1.0)),
        ]
        for col, (arr, title, cmap, (vmin, vmax)) in enumerate(panel_data):
            ax = axes[row, col]
            im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
            if row == 0:
                ax.set_title(title)
            if col == 0:
                ax.set_ylabel(cap.epoch_label, rotation=0, labelpad=38, va="center")
            ax.set_xticks([])
            ax.set_yticks([])
            if row == rows - 1:
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    first = captures[0]
    fig.suptitle(f"{first.split} {first.anchor_name} output timeline")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def write_readme(out_dir: Path, captures: list[OutputCapture], ranges: dict[str, tuple[float, float]]) -> None:
    epochs = sorted({cap.epoch_label for cap in captures})
    anchors = sorted({f"{cap.split}/{cap.anchor_name}" for cap in captures})
    text = "# 05_output预测GT误差时间序列说明\n\n"
    text += "本目录按固定训练间隔重新生成输出可视化，只包含预测、GT、误差和mask相关内容，不包含输入事件图。\n\n"
    text += "## Epoch\n\n"
    text += "- " + "\n- ".join(epochs) + "\n\n"
    text += "## 锚点\n\n"
    text += "- " + "\n- ".join(anchors) + "\n\n"
    text += "## 目录\n\n"
    text += "- `by_epoch/`：按epoch查看每个锚点的05_output图。\n"
    text += "- `timeline_by_anchor/`：按锚点查看从训练开始到结束的纵向时间序列拼图。\n\n"
    text += "## 颜色范围\n\n"
    text += f"- Prediction/GT/Masked Prediction 统一范围：`{ranges['value'][0]:.6f}` 到 `{ranges['value'][1]:.6f}`。\n"
    text += f"- Abs Error 统一范围：`{ranges['error'][0]:.6f}` 到 `{ranges['error'][1]:.6f}`。\n"
    text += "- Mask 固定范围：`0` 到 `1`。\n"
    (out_dir / "README_05_output时间序列说明.md").write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", type=Path, default=DEFAULT_H5)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--out-subdir", default="visualizations/output_timeline_05")
    parser.add_argument("--epochs", default="0,10,20,30,40,50")
    parser.add_argument("--include-best", action="store_true")
    parser.add_argument("--input-scale", type=float, default=0.3)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    epochs = parse_epoch_list(args.epochs)
    best_epoch = read_best_epoch(args.run_dir) if args.include_best else None
    if best_epoch is not None and best_epoch not in epochs:
        epochs.append(best_epoch)
    epochs = sorted(set(epochs))

    meta = load_h5_meta(args.h5, steps_per_block=10, num_blocks=500)
    split_rows = build_split_rows(meta, steps_per_block=10, num_blocks=500)
    train_loader = make_loader(S2BlockDataset(args.h5, [r for r in split_rows if r["split"] == "train"], 10), 1)
    val_loader = make_loader(S2BlockDataset(args.h5, [r for r in split_rows if r["split"] == "validation"], 10), 1)

    train_anchors = {"train_start": (0, 0), "train_end": (389, 9)}
    val_anchors = {"val_start": (400, 0), "val_middle": (450, 5), "val_end": (499, 9)}
    captures: list[OutputCapture] = []
    for epoch in epochs:
        ckpt = checkpoint_for_epoch(args.run_dir, epoch, best_epoch=best_epoch)
        if not ckpt.exists():
            continue
        model = load_model_from_checkpoint(ckpt, device)
        epoch_label = f"epoch_{epoch:03d}_best" if best_epoch is not None and epoch == best_epoch and ckpt.name == "best_val_mae.pt" else f"epoch_{epoch:03d}"
        captures.extend(
            collect_outputs(
                model,
                train_loader,
                epoch_label=epoch_label,
                split="train",
                anchors=train_anchors,
                device=device,
                input_scale=args.input_scale,
            )
        )
        captures.extend(
            collect_outputs(
                model,
                val_loader,
                epoch_label=epoch_label,
                split="validation",
                anchors=val_anchors,
                device=device,
                input_scale=args.input_scale,
            )
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    out_dir = args.run_dir / args.out_subdir
    ranges = compute_ranges(captures)
    write_json(out_dir / "display_ranges.json", ranges)

    for cap in captures:
        path = out_dir / "by_epoch" / cap.epoch_label / cap.split / f"{cap.anchor_name}_05_output_预测GT误差.png"
        render_output_panel(cap, path, ranges)

    groups: dict[tuple[str, str], list[OutputCapture]] = {}
    for cap in captures:
        groups.setdefault((cap.split, cap.anchor_name), []).append(cap)
    for (split, anchor), caps in groups.items():
        render_anchor_timeline(caps, out_dir / "timeline_by_anchor" / f"{split}_{anchor}_timeline.png", ranges)

    write_readme(out_dir, captures, ranges)
    print(f"saved_captures={len(captures)} out_dir={out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
