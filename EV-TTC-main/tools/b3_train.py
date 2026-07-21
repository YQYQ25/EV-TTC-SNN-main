#!/usr/bin/env python3
"""Train C0, B3-Oracle, or B3-Closed without changing B0/B1 code paths."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import h5py
import hdf5plugin  # noqa: F401
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model"))
sys.path.insert(0, str(ROOT / "tools"))

from b3_risk_adapter import B3RiskAdaptiveEVSlim  # noqa: E402
from b3_temporal_data import TemporalClipDataset  # noqa: E402
from evslim import EVSlim  # noqa: E402


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def charbonnier_loss(error: torch.Tensor, mask: torch.Tensor, alpha: float = 0.45) -> torch.Tensor:
    loss = (error.square() + 1e-5**2.0).pow(alpha)
    valid = mask.float()
    denominator = valid.sum(dim=(1, 2, 3)).clamp_min(1.0)
    return ((loss * valid).sum(dim=(1, 2, 3)) / denominator).mean()


def load_b1_full(checkpoint: Path, device: torch.device, enabled: bool) -> tuple[B3RiskAdaptiveEVSlim, dict, SimpleNamespace]:
    payload = torch.load(checkpoint, map_location="cpu")
    cfg = SimpleNamespace(**payload["cfg"])
    if int(cfg.in_channels) != 12:
        raise ValueError(f"B3 requires a 12-channel B1-Full checkpoint, got {cfg.in_channels}")
    core = EVSlim(cfg)
    core.load_state_dict(payload["model_state_dict"], strict=True)
    return B3RiskAdaptiveEVSlim(core, enabled=enabled).to(device), payload, cfg


def closed_probability(epoch: int, total_epochs: int) -> float:
    if epoch <= 5:
        return 0.0
    if total_epochs <= 6:
        return 1.0
    return float(np.clip((epoch - 6) / max(total_epochs - 6, 1), 0.0, 1.0))


def select_previous(
    mode: str,
    previous_gt: torch.Tensor | None,
    previous_prediction: torch.Tensor | None,
    epoch: int,
    total_epochs: int,
) -> torch.Tensor | None:
    if previous_gt is None or mode == "c0":
        return None
    if mode == "oracle":
        return previous_gt
    probability = closed_probability(epoch, total_epochs)
    if probability <= 0:
        return previous_gt
    if probability >= 1:
        return previous_prediction
    use_prediction = torch.rand((previous_gt.shape[0], 1, 1, 1), device=previous_gt.device) < probability
    return torch.where(use_prediction, previous_prediction, previous_gt)


def make_loader(dataset, batch_size: int, shuffle: bool, workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
    )


@torch.no_grad()
def quick_validation(model, loader, device, mode: str, alpha: float) -> float:
    model.eval()
    losses = []
    for x, ttc, mask, _ in loader:
        x = x.to(device, non_blocking=True)
        ttc = ttc.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        previous_prediction = None
        previous_x = None
        frame_losses = []
        for time_index in range(x.shape[1]):
            previous = select_previous(mode, ttc[:, time_index - 1] if time_index else None, previous_prediction, 20, 20)
            prediction = model(x[:, time_index], previous, previous_x)
            frame_losses.append(charbonnier_loss(ttc[:, time_index] - prediction, mask[:, time_index], alpha))
            previous_prediction = prediction.detach()
            previous_x = x[:, time_index].detach()
        losses.append(float(torch.stack(frame_losses).mean().cpu()))
    return float(np.mean(losses)) if losses else float("nan")


def run_epoch(model, loader, optimizer, scheduler, device, mode: str, epoch: int, total_epochs: int, alpha: float):
    model.train()
    batch_losses = []
    for x, ttc, mask, _ in loader:
        x = x.to(device, non_blocking=True)
        ttc = ttc.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        previous_prediction = None
        previous_x = None
        accumulated = 0.0
        for time_index in range(x.shape[1]):
            previous_gt = ttc[:, time_index - 1] if time_index else None
            previous = select_previous(mode, previous_gt, previous_prediction, epoch, total_epochs)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                prediction = model(x[:, time_index], previous, previous_x)
                loss = charbonnier_loss(ttc[:, time_index] - prediction, mask[:, time_index], alpha)
            (loss / x.shape[1]).backward()
            accumulated += float(loss.detach().cpu())
            # This is the only recurrent state; no graph crosses the next frame.
            previous_prediction = prediction.detach()
            previous_x = x[:, time_index].detach()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        batch_losses.append(accumulated / x.shape[1])
    return float(np.mean(batch_losses))


@torch.no_grad()
def save_overfit_visualizations(model, dataset, device, mode: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    selected_frames = [0, 5, 10, 15]
    for clip_index in range(min(4, len(dataset))):
        x, ttc, mask, _ = dataset[clip_index]
        x = x.to(device)
        ttc = ttc.to(device)
        mask = mask.to(device)
        model.eval()
        predictions = []
        previous_prediction = None
        previous_x = None
        for time_index in range(x.shape[0]):
            previous = select_previous(
                mode,
                ttc[time_index - 1][None] if time_index else None,
                previous_prediction,
                20,
                20,
            )
            prediction = model(x[time_index][None], previous, previous_x)
            predictions.append(prediction[0, 0].cpu().numpy())
            previous_prediction = prediction.detach()
            previous_x = x[time_index][None].detach()
        fig, axes = plt.subplots(4, len(selected_frames), figsize=(3 * len(selected_frames), 10), constrained_layout=True)
        for col, time_index in enumerate(selected_frames):
            gt = ttc[time_index, 0].cpu().numpy()
            pred = predictions[time_index]
            valid = mask[time_index, 0].cpu().numpy().astype(bool)
            arrays = [gt, pred, np.abs(pred - gt), valid.astype(float)]
            titles = ["GT TTC", "Prediction", "Abs Error", "Mask"]
            for row, (array, title) in enumerate(zip(arrays, titles)):
                axis = axes[row, col]
                shown = np.where(valid, array, np.nan) if title != "Mask" else array
                axis.imshow(shown, cmap="viridis" if row < 2 else "magma" if row == 2 else "gray")
                axis.set_title(f"t={time_index}: {title}")
                axis.axis("off")
        fig.savefig(out_dir / f"clip_{clip_index:02d}.png", dpi=140)
        plt.close(fig)


def write_curve(rows: list[dict], out_dir: Path) -> None:
    with (out_dir / "train_curve.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    values = np.asarray([[row["epoch"], row["train_loss"], row["quick_val_loss"]] for row in rows])
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.plot(values[:, 0], values[:, 1], label="train")
    axis.plot(values[:, 0], values[:, 2], label="quick val")
    axis.set_xlabel("epoch")
    axis.set_ylabel("masked Charbonnier loss")
    axis.legend()
    figure.tight_layout()
    figure.savefig(out_dir / "train_curve.png", dpi=160)
    plt.close(figure)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("c0", "oracle", "closed"), required=True)
    parser.add_argument("--train-h5", default="/home/hello/research_project/event+SNN+TTC/logs/ttc_TTCEF/b1_polarity_full_20ep/b1_full_train.h5")
    parser.add_argument("--test-h5", default="/home/hello/research_project/event+SNN+TTC/logs/ttc_TTCEF/b1_polarity_full_20ep/b1_full_test.h5")
    parser.add_argument("--train-clips", default="/home/hello/research_project/event+SNN+TTC/logs/ttc_TTCEF/temporal_audit/train_clips.json")
    parser.add_argument("--test-clips", default="/home/hello/research_project/event+SNN+TTC/logs/ttc_TTCEF/temporal_audit/test_clips.json")
    parser.add_argument("--checkpoint", default="/home/hello/research_project/event+SNN+TTC/logs/ttc_TTCEF/b1_polarity_full_20ep/b1_polarity_full_best.pt")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--clip-length", type=int, default=16)
    parser.add_argument("--train-stride", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--pct-start", type=float, default=0.1)
    parser.add_argument("--quick-val-clips", type=int, default=64)
    parser.add_argument("--max-train-clips", type=int, default=None)
    parser.add_argument("--overfit", action="store_true")
    parser.add_argument("--cache-in-memory", action="store_true")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    seed_all(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run_config.json").write_text(json.dumps(vars(args), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    train_dataset = TemporalClipDataset(
        args.train_h5, args.train_clips, args.clip_length, args.train_stride, args.max_train_clips, args.cache_in_memory
    )
    if args.overfit:
        val_dataset = TemporalClipDataset(
            args.train_h5, args.train_clips, args.clip_length, args.clip_length, args.max_train_clips, args.cache_in_memory
        )
    else:
        val_dataset = TemporalClipDataset(
            args.test_h5, args.test_clips, args.clip_length, args.clip_length, args.quick_val_clips, args.cache_in_memory
        )
    train_loader = make_loader(train_dataset, args.batch_size, True, args.workers)
    val_loader = make_loader(val_dataset, args.batch_size, False, args.workers)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model, source_payload, cfg = load_b1_full(Path(args.checkpoint), device, enabled=args.mode != "c0")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        total_steps=max(1, args.epochs * len(train_loader)),
        pct_start=args.pct_start,
        cycle_momentum=False,
    )
    best = float("inf")
    rows = []
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer, scheduler, device, args.mode, epoch, args.epochs, cfg.alpha)
        val_loss = quick_validation(model, val_loader, device, args.mode, cfg.alpha)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "quick_val_loss": val_loss,
            "lr": scheduler.get_last_lr()[0],
            "closed_prediction_probability": closed_probability(epoch, args.epochs) if args.mode == "closed" else 0.0,
        }
        rows.append(row)
        payload = {
            "model_state_dict": model.state_dict(),
            "base_cfg": vars(cfg),
            "mode": args.mode,
            "epoch": epoch,
            "row": row,
            "source_b1_full_checkpoint": str(Path(args.checkpoint).resolve()),
            "clip_length": args.clip_length,
        }
        torch.save(payload, out_dir / "last.pt")
        if val_loss < best:
            best = val_loss
            torch.save(payload, out_dir / "best.pt")
    write_curve(rows, out_dir)
    if args.overfit:
        save_overfit_visualizations(model, train_dataset, device, args.mode, out_dir / "overfit_visualizations")
    summary = {
        "mode": args.mode,
        "train_clips": len(train_dataset),
        "quick_val_clips": len(val_dataset),
        "final": rows[-1],
        "best_quick_val_loss": best,
        "best_checkpoint": str(out_dir / "best.pt"),
        "last_checkpoint": str(out_dir / "last.pt"),
        "source_checkpoint": str(Path(args.checkpoint).resolve()),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
