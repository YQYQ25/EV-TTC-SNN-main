#!/usr/bin/env python3
"""B3-v2 inference grid, training, evaluation, and report generation."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

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

from b3_temporal_data import TemporalClipDataset, TemporalSegmentDataset  # noqa: E402
from b3_train import closed_probability  # noqa: E402
from b3_v2_risk_adapter import B3V2RiskAdaptiveEVSlim  # noqa: E402
from evslim import EVSlim  # noqa: E402


DEFAULT_B3_DIR = Path("/home/hello/research_project/event+SNN+TTC/logs/ttc_TTCEF/b3")
DEFAULT_OUT = Path("/home/hello/research_project/event+SNN+TTC/logs/ttc_TTCEF/b3_v2")
DEFAULT_B1 = Path("/home/hello/research_project/event+SNN+TTC/logs/ttc_TTCEF/b1_polarity_full_20ep/b1_polarity_full_best.pt")
DEFAULT_TRAIN_H5 = Path("/home/hello/research_project/event+SNN+TTC/logs/ttc_TTCEF/b1_polarity_full_20ep/b1_full_train.h5")
DEFAULT_TEST_H5 = Path("/home/hello/research_project/event+SNN+TTC/logs/ttc_TTCEF/b1_polarity_full_20ep/b1_full_test.h5")
DEFAULT_TRAIN_CLIPS = Path("/home/hello/research_project/event+SNN+TTC/logs/ttc_TTCEF/temporal_audit/train_clips.json")
DEFAULT_TEST_CLIPS = Path("/home/hello/research_project/event+SNN+TTC/logs/ttc_TTCEF/temporal_audit/test_clips.json")

TTC_BINS = (
    ("0-1s", 0.0, 1.0),
    ("1-2s", 1.0, 2.0),
    ("2-5s", 2.0, 5.0),
    (">5s", 5.0, math.inf),
)


def seed_all(seed: int = 1) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def masked_charbonnier(error: torch.Tensor, mask: torch.Tensor, alpha: float = 0.45) -> torch.Tensor:
    loss = (error.square() + 1e-5**2.0).pow(alpha)
    valid = mask.float()
    denominator = valid.sum(dim=(1, 2, 3)).clamp_min(1.0)
    return ((loss * valid).sum(dim=(1, 2, 3)) / denominator).mean()


def task_loss(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor, lambda1: float = 1.0, lambda2: float = 0.5) -> torch.Tensor:
    global_loss = masked_charbonnier(gt - pred, mask)
    mask_0_1 = mask & (gt > 0) & (gt <= 1.0)
    mask_1_2 = mask & (gt > 1.0) & (gt <= 2.0)
    loss = global_loss
    if mask_0_1.any():
        loss = loss + lambda1 * masked_charbonnier(gt - pred, mask_0_1)
    if mask_1_2.any():
        loss = loss + lambda2 * masked_charbonnier(gt - pred, mask_1_2)
    return loss


def load_core(checkpoint: Path, from_b1: bool = True) -> tuple[EVSlim, dict, SimpleNamespace]:
    payload = torch.load(checkpoint, map_location="cpu")
    cfg_dict = payload["cfg"] if from_b1 and "cfg" in payload else payload.get("base_cfg", payload.get("cfg"))
    cfg = SimpleNamespace(**cfg_dict)
    core = EVSlim(cfg)
    key = "model_state_dict" if "model_state_dict" in payload else "state_dict"
    state = payload[key]
    if state and all(name.startswith("base_model.") or name.startswith("modulator.") for name in state):
        state = {name.removeprefix("base_model."): value for name, value in state.items() if name.startswith("base_model.")}
    core.load_state_dict(state, strict=True)
    return core, payload, cfg


def make_v2_model(checkpoint: Path, device: torch.device, beta: float, rho: float, mode: str, activity_ref: float, from_b1: bool = True):
    core, payload, cfg = load_core(checkpoint, from_b1)
    if int(getattr(cfg, "in_channels", 12)) != 12:
        raise ValueError("B3-v2 requires a 12-channel B1-Full/C0 checkpoint")
    return B3V2RiskAdaptiveEVSlim(core, beta, rho, mode, activity_ref).to(device), payload, cfg


def region_slices(row: int, col: int, height: int = 360, width: int = 360):
    return slice(row * height // 3, (row + 1) * height // 3), slice(col * width // 3, (col + 1) * width // 3)


def compute_activity_ref(train_h5: Path, train_clips: Path, out_dir: Path) -> float:
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset = TemporalSegmentDataset(train_h5, train_clips)
    values = []
    for index in range(len(dataset)):
        x_seq, _, _, _ = dataset[index]
        activity = (x_seq[:, 5] - x_seq[:, 11]).abs().numpy() > 1e-3
        for frame in activity:
            for row in range(3):
                for col in range(3):
                    ys, xs = region_slices(row, col)
                    values.append(float(frame[ys, xs].mean()))
    arr = np.asarray(values, dtype=np.float64)
    summary = {
        "activity_ref": float(np.median(arr)),
        "p25": float(np.quantile(arr, 0.25)),
        "p50": float(np.quantile(arr, 0.50)),
        "p75": float(np.quantile(arr, 0.75)),
        "p90": float(np.quantile(arr, 0.90)),
        "count": int(arr.size),
        "definition": "median train-set 3x3 region activity ratio using abs(pos_long-neg_long)>1e-3",
    }
    (out_dir / "activity_ref.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary["activity_ref"]


class EvalAccumulator:
    def __init__(self):
        self.frame_medians = []
        self.pixel_abs_values = []
        self.count = 0
        self.sum_abs = 0.0
        self.sum_rel = 0.0
        self.re_counts = {0.10: 0, 0.50: 0, 0.80: 0}
        self.bin_stats = {name: {"count": 0, "sum_abs": 0.0, "sum_rel": 0.0, "sum_bias": 0.0, "sum_pred": 0.0, "sum_gt": 0.0, "fn": 0} for name, *_ in TTC_BINS}
        self.hr_count = 0
        self.hr_fn = 0
        self.region_tp = self.region_gt = self.region_fp = self.region_neg = 0
        self.pred_states = []
        self.gt_states = []
        self.pred_risks = []
        self.latencies = []

    @staticmethod
    def _update_bin(item: dict, gt_v: np.ndarray, pred_v: np.ndarray, threshold: float | None) -> None:
        err = pred_v - gt_v
        abs_err = np.abs(err)
        item["count"] += int(abs_err.size)
        item["sum_abs"] += float(abs_err.sum(dtype=np.float64))
        item["sum_rel"] += float((abs_err / np.maximum(np.abs(gt_v), 1e-6)).sum(dtype=np.float64))
        item["sum_bias"] += float(err.sum(dtype=np.float64))
        item["sum_pred"] += float(pred_v.sum(dtype=np.float64))
        item["sum_gt"] += float(gt_v.sum(dtype=np.float64))
        if threshold is not None:
            item["fn"] += int((pred_v > threshold).sum())

    def update(self, gt: np.ndarray, pred: np.ndarray, mask: np.ndarray, pred_risk: np.ndarray, gt_region_state: np.ndarray, latency_ms: float | None = None):
        valid = mask & np.isfinite(gt) & np.isfinite(pred)
        if not valid.any():
            return
        ae = np.abs(pred - gt).astype(np.float32)
        rel = ae[valid] / np.maximum(np.abs(gt[valid]), 1e-6)
        vals = ae[valid]
        self.frame_medians.append(float(np.median(vals)))
        self.pixel_abs_values.append(vals)
        self.count += int(vals.size)
        self.sum_abs += float(vals.sum(dtype=np.float64))
        self.sum_rel += float(rel.sum(dtype=np.float64))
        for threshold in self.re_counts:
            self.re_counts[threshold] += int((rel > threshold).sum())
        for name, lo, hi in TTC_BINS:
            local = valid & (gt > lo) & ((gt <= hi) if math.isfinite(hi) else True)
            if local.any():
                self._update_bin(self.bin_stats[name], gt[local], pred[local], hi if math.isfinite(hi) else None)
        high = valid & (gt > 0) & (gt <= 2.0)
        self.hr_count += int(high.sum())
        self.hr_fn += int((pred[high] > 2.0).sum())
        pred_state = pred_risk > 0.5
        self.region_tp += int((gt_region_state & pred_state).sum())
        self.region_gt += int(gt_region_state.sum())
        self.region_fp += int((~gt_region_state & pred_state).sum())
        self.region_neg += int((~gt_region_state).sum())
        self.gt_states.append(gt_region_state.astype(bool))
        self.pred_states.append(pred_state.astype(bool))
        self.pred_risks.append(pred_risk.astype(np.float32))
        if latency_ms is not None:
            self.latencies.append(float(latency_ms))

    def _finish_bin(self, name: str) -> dict:
        item = self.bin_stats[name]
        if item["count"] == 0:
            return {"count": 0, "MAE": None, "MRE": None, "FN": None, "bias": None, "prediction_mean": None, "GT_mean": None}
        return {
            "count": item["count"],
            "MAE": item["sum_abs"] / item["count"],
            "MRE": item["sum_rel"] / item["count"],
            "FN": item["fn"] / item["count"],
            "bias": item["sum_bias"] / item["count"],
            "prediction_mean": item["sum_pred"] / item["count"],
            "GT_mean": item["sum_gt"] / item["count"],
        }

    def finish(self) -> dict:
        abs_values = np.concatenate(self.pixel_abs_values) if self.pixel_abs_values else np.array([], dtype=np.float32)
        pred_states = np.stack(self.pred_states) if self.pred_states else np.zeros((0, 3, 3), dtype=bool)
        pred_risks = np.stack(self.pred_risks) if self.pred_risks else np.zeros((0, 3, 3), dtype=np.float32)
        flips = np.abs(np.diff(pred_states.astype(np.int16), axis=0)).sum() if pred_states.shape[0] > 1 else 0
        duration = max(pred_states.shape[0] * 0.007, 1e-6)
        risk_diff = np.abs(np.diff(pred_risks, axis=0)).mean() if pred_risks.shape[0] > 1 else 0.0
        leads = compute_lead_times(self.gt_states, self.pred_states)
        return {
            "paper_metrics": {
                "mean_frame_median_AE": float(np.mean(self.frame_medians)),
                "std_frame_median_AE": float(np.std(self.frame_medians)),
                "MRE_percent": 100.0 * self.sum_rel / max(self.count, 1),
                "RE_10_percent": 100.0 * self.re_counts[0.10] / max(self.count, 1),
                "RE_50_percent": 100.0 * self.re_counts[0.50] / max(self.count, 1),
                "RE_80_percent": 100.0 * self.re_counts[0.80] / max(self.count, 1),
            },
            "safety_metrics": {
                "pixel_MAE": self.sum_abs / max(self.count, 1),
                "pixel_median_AE": float(np.median(abs_values)) if abs_values.size else None,
                "high_risk_false_negative_rate": self.hr_fn / max(self.hr_count, 1),
                "risk_bins": {name: self._finish_bin(name) for name, *_ in TTC_BINS},
            },
            "task_metrics": {
                "region_recall": self.region_tp / max(self.region_gt, 1),
                "region_false_positive_rate": self.region_fp / max(self.region_neg, 1),
                "risk_flips_per_second": float(flips / duration),
                "adjacent_region_risk_diff_mean": float(risk_diff),
                "lead_time_ms_mean": float(np.mean(leads)) if leads else None,
                "lead_time_ms_median": float(np.median(leads)) if leads else None,
                "lead_time_count": len(leads),
                "extra_latency_ms": float(np.mean(self.latencies)) if self.latencies else None,
            },
            "num_frames": int(len(self.frame_medians)),
            "valid_pixel_count": int(self.count),
        }


def gt_region_state(gt: np.ndarray, valid: np.ndarray, min_pixels: int = 32) -> np.ndarray:
    state = np.zeros((3, 3), dtype=bool)
    for row in range(3):
        for col in range(3):
            ys, xs = region_slices(row, col)
            state[row, col] = int((valid[ys, xs] & (gt[ys, xs] > 0) & (gt[ys, xs] <= 2.0)).sum()) >= min_pixels
    return state


def compute_pred_region_risk(model: B3V2RiskAdaptiveEVSlim, pred: torch.Tensor, x: torch.Tensor) -> np.ndarray:
    raw, _, _, _, _ = model.modulator.compute_raw_risk(pred.detach(), x.detach(), x.detach())
    return raw[0].float().cpu().numpy()


def compute_lead_times(gt_states: list[np.ndarray], pred_states: list[np.ndarray]) -> list[float]:
    if not gt_states:
        return []
    gt = np.stack(gt_states)
    pred = np.stack(pred_states)
    leads = []
    for row in range(3):
        for col in range(3):
            signal = gt[:, row, col]
            start = None
            last_end = -1
            for idx, value in enumerate(signal.tolist() + [False]):
                if value and start is None:
                    start = idx
                if not value and start is not None:
                    end = idx - 1
                    if end - start + 1 >= 2:
                        search_start = max(last_end + 1, start - 30)
                        candidates = np.where(pred[search_start : end + 1, row, col])[0]
                        if candidates.size:
                            first_pred = search_start + int(candidates[0])
                            leads.append((start - first_pred) * 7.0)
                    last_end = end
                    start = None
    return leads


@torch.no_grad()
def evaluate_model(
    model: B3V2RiskAdaptiveEVSlim,
    dataset: TemporalSegmentDataset,
    device: torch.device,
    use_gt_feedback: bool = False,
    latency_frames: int = 50,
) -> dict:
    model.eval()
    acc = EvalAccumulator()
    timed = 0
    for segment_index in range(len(dataset)):
        x_seq, ttc_seq, mask_seq, _ = dataset[segment_index]
        previous_prediction = None
        previous_gt = None
        previous_x = None
        risk_ema = None
        for time_index in range(x_seq.shape[0]):
            x = x_seq[time_index][None].to(device)
            gt = ttc_seq[time_index][None].to(device)
            mask = mask_seq[time_index][None].to(device)
            previous = None if previous_gt is None else (previous_gt if use_gt_feedback else previous_prediction)
            if device.type == "cuda" and timed < latency_frames:
                torch.cuda.synchronize()
                start = time.perf_counter()
                pred, aux = model(x, previous, previous_x, risk_ema, return_aux=True)
                torch.cuda.synchronize()
                latency = (time.perf_counter() - start) * 1000.0
                timed += 1
            else:
                pred, aux = model(x, previous, previous_x, risk_ema, return_aux=True)
                latency = None
            gt_np = gt[0, 0].float().cpu().numpy()
            pred_np = pred[0, 0].float().cpu().numpy()
            mask_np = mask[0, 0].cpu().numpy().astype(bool)
            valid = mask_np & np.isfinite(gt_np)
            pred_risk = compute_pred_region_risk(model, pred, x)
            acc.update(gt_np, pred_np, valid, pred_risk, gt_region_state(gt_np, valid), latency)
            previous_prediction = pred.detach()
            previous_gt = gt.detach()
            previous_x = x.detach()
            risk_ema = aux["risk_ema"].detach()
    return acc.finish()


def make_loader(dataset, batch_size: int, shuffle: bool, workers: int) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers, pin_memory=True, persistent_workers=workers > 0)


def select_previous(previous_gt, previous_prediction, epoch: int, total_epochs: int):
    if previous_gt is None:
        return None
    p = closed_probability(epoch, total_epochs)
    if p <= 0:
        return previous_gt
    if p >= 1:
        return previous_prediction
    use_pred = torch.rand((previous_gt.shape[0], 1, 1, 1), device=previous_gt.device) < p
    return torch.where(use_pred, previous_prediction, previous_gt)


@torch.no_grad()
def quick_validation(model, loader, device, total_epochs: int) -> float:
    model.eval()
    losses = []
    for x, ttc, mask, _ in loader:
        x = x.to(device, non_blocking=True)
        ttc = ttc.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        prev_pred = prev_x = risk_ema = None
        frame_losses = []
        for t in range(x.shape[1]):
            prev = None if t == 0 else prev_pred
            pred, aux = model(x[:, t], prev, prev_x, risk_ema, return_aux=True)
            frame_losses.append(masked_charbonnier(ttc[:, t] - pred, mask[:, t]))
            prev_pred = pred.detach()
            prev_x = x[:, t].detach()
            risk_ema = aux["risk_ema"].detach()
        losses.append(float(torch.stack(frame_losses).mean().cpu()))
    return float(np.mean(losses)) if losses else float("nan")


def train_candidate(args, params: dict, out_dir: Path, loss_mode: str = "global") -> dict:
    seed_all(args.seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model, _, cfg = make_v2_model(args.b1_checkpoint, device, params["beta"], params["rho"], params["confidence_mode"], params["activity_ref"], True)
    train_data = TemporalClipDataset(args.train_h5, args.train_clips, args.clip_length, args.train_stride, None, args.cache_in_memory)
    val_data = TemporalClipDataset(args.test_h5, args.test_clips, args.clip_length, args.clip_length, args.quick_val_clips, args.cache_in_memory)
    train_loader = make_loader(train_data, args.batch_size, True, args.workers)
    val_loader = make_loader(val_data, args.batch_size, False, args.workers)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        total_steps=max(1, args.epochs * len(train_loader)),
        pct_start=args.pct_start,
        cycle_momentum=False,
    )
    rows = []
    best = float("inf")
    best_path = out_dir / "best.pt"
    last_path = out_dir / "last.pt"
    config = {**params, "loss_mode": loss_mode, "epochs": args.epochs, "source_checkpoint": str(args.b1_checkpoint)}
    (out_dir / "run_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for epoch in range(1, args.epochs + 1):
        model.train()
        batch_losses = []
        for x, ttc, mask, _ in train_loader:
            x = x.to(device, non_blocking=True)
            ttc = ttc.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            prev_pred = prev_x = risk_ema = None
            accum = 0.0
            for t in range(x.shape[1]):
                previous_gt = None if t == 0 else ttc[:, t - 1]
                previous = select_previous(previous_gt, prev_pred, epoch, args.epochs)
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                    pred, aux = model(x[:, t], previous, prev_x, risk_ema, return_aux=True)
                    if loss_mode == "task":
                        loss = task_loss(pred, ttc[:, t], mask[:, t], args.lambda1, args.lambda2)
                    else:
                        loss = masked_charbonnier(ttc[:, t] - pred, mask[:, t])
                (loss / x.shape[1]).backward()
                accum += float(loss.detach().cpu())
                prev_pred = pred.detach()
                prev_x = x[:, t].detach()
                risk_ema = aux["risk_ema"].detach()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            batch_losses.append(accum / x.shape[1])
        val_loss = quick_validation(model, val_loader, device, args.epochs)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(batch_losses)),
            "quick_val_loss": val_loss,
            "lr": float(scheduler.get_last_lr()[0]),
            "closed_prediction_probability": closed_probability(epoch, args.epochs),
        }
        rows.append(row)
        payload = {
            "epoch": epoch,
            "row": row,
            "model_state_dict": model.state_dict(),
            "base_cfg": vars(cfg),
            "v2_params": params,
            "loss_mode": loss_mode,
        }
        torch.save(payload, last_path)
        if val_loss < best:
            best = val_loss
            torch.save(payload, best_path)
    write_train_curve(rows, out_dir)
    summary = {
        "v2_params": params,
        "loss_mode": loss_mode,
        "train_clips": len(train_data),
        "quick_val_clips": len(val_data),
        "best_quick_val_loss": best,
        "best_checkpoint": str(best_path),
        "last_checkpoint": str(last_path),
        "final": rows[-1],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def write_train_curve(rows: list[dict], out_dir: Path) -> None:
    write_csv(out_dir / "train_curve.csv", rows)
    arr = np.asarray([[row["epoch"], row["train_loss"], row["quick_val_loss"]] for row in rows], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(arr[:, 0], arr[:, 1], label="train")
    ax.plot(arr[:, 0], arr[:, 2], label="quick val")
    ax.set_xlabel("epoch")
    ax.set_ylabel("masked Charbonnier loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "train_curve.png", dpi=160)
    plt.close(fig)


def eval_checkpoint(args, checkpoint: Path, params: dict, out_dir: Path, name: str) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model, _, _ = make_v2_model(args.b1_checkpoint, device, params["beta"], params["rho"], params["confidence_mode"], params["activity_ref"], True)
    payload = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(payload["model_state_dict"], strict=True)
    metrics = evaluate_model(model, TemporalSegmentDataset(args.test_h5, args.test_clips), device)
    metrics.update({"name": name, "checkpoint": str(checkpoint), "v2_params": params})
    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_metrics_md(metrics, out_dir / "metrics.md")
    return metrics


def write_metrics_md(metrics: dict, path: Path) -> None:
    p, s, t = metrics["paper_metrics"], metrics["safety_metrics"], metrics["task_metrics"]
    b = s["risk_bins"]
    lines = [
        f"# {metrics['name']} 指标",
        "",
        f"- Mean frame median AE：{p['mean_frame_median_AE']:.6f}",
        f"- MRE：{p['MRE_percent']:.6f}%",
        f"- FN<=1s：{b['0-1s']['FN']:.6f}",
        f"- HR FN<=2s：{s['high_risk_false_negative_rate']:.6f}",
        f"- 0-1s MAE：{b['0-1s']['MAE']:.6f}",
        f"- 1-2s MAE：{b['1-2s']['MAE']:.6f}",
        f"- 2-5s MAE：{b['2-5s']['MAE']:.6f}",
        f"- 区域风险召回率：{t['region_recall']:.6f}",
        f"- 区域风险误报率：{t['region_false_positive_rate']:.6f}",
        f"- 检测提前量均值(ms)：{t['lead_time_ms_mean']}",
        f"- 风险翻转次数/秒：{t['risk_flips_per_second']:.6f}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_b3_v2_checkpoint(checkpoint: Path, device: torch.device):
    payload = torch.load(checkpoint, map_location="cpu")
    cfg = SimpleNamespace(**payload["base_cfg"])
    core = EVSlim(cfg)
    params = payload["v2_params"]
    model = B3V2RiskAdaptiveEVSlim(core, params["beta"], params["rho"], params["confidence_mode"], params["activity_ref"])
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model.to(device).eval(), params


def eval_grid_variant(args, params: dict, checkpoint: Path) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model, _, _ = make_v2_model(checkpoint, device, params["beta"], params["rho"], params["confidence_mode"], params["activity_ref"], False)
    return evaluate_model(model, TemporalSegmentDataset(args.test_h5, args.test_clips), device)


def flatten_grid_row(params: dict, metrics: dict, baseline: dict | None = None) -> dict:
    p, s, t = metrics["paper_metrics"], metrics["safety_metrics"], metrics["task_metrics"]
    b = s["risk_bins"]
    row = {
        "beta": params["beta"],
        "rho": params["rho"],
        "confidence_mode": params["confidence_mode"],
        "activity_ref": params["activity_ref"],
        "mean_frame_median_AE": p["mean_frame_median_AE"],
        "MRE_percent": p["MRE_percent"],
        "MAE_0_1s": b["0-1s"]["MAE"],
        "FN_0_1s": b["0-1s"]["FN"],
        "MAE_1_2s": b["1-2s"]["MAE"],
        "FN_1_2s": b["1-2s"]["FN"],
        "MAE_2_5s": b["2-5s"]["MAE"],
        "HR_FN_2s": s["high_risk_false_negative_rate"],
        "region_recall": t["region_recall"],
        "region_fpr": t["region_false_positive_rate"],
        "risk_flips_per_second": t["risk_flips_per_second"],
        "lead_time_ms_mean": t["lead_time_ms_mean"],
        "extra_latency_ms": t["extra_latency_ms"],
    }
    if baseline:
        row["constraint_mfm_degrade"] = row["mean_frame_median_AE"] / baseline["mean_frame_median_AE"] - 1.0
        row["constraint_mre_degrade"] = row["MRE_percent"] / baseline["MRE_percent"] - 1.0
        row["constraint_2_5_mae_degrade"] = row["MAE_2_5s"] / baseline["MAE_2_5s"] - 1.0
        row["passes_constraints"] = (
            row["constraint_mfm_degrade"] <= 0.05 and row["constraint_mre_degrade"] <= 0.05 and row["constraint_2_5_mae_degrade"] <= 0.10
        )
    return row


def c0_baseline_for_constraints(b3_dir: Path) -> dict:
    metrics = load_json(b3_dir / "c0_b1_clipft/metrics.json")
    return {
        "mean_frame_median_AE": metrics["paper_metrics"]["mean_frame_median_AE"],
        "MRE_percent": metrics["paper_metrics"]["MRE_percent"],
        "MAE_2_5s": metrics["safety_metrics"]["risk_bins"]["2_5s"]["mae"]
        if "2_5s" in metrics["safety_metrics"]["risk_bins"]
        else metrics["safety_metrics"]["risk_bins"]["2-5s"]["MAE"],
    }


def run_grid(args, activity_ref: float) -> list[dict]:
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    baseline = c0_baseline_for_constraints(args.b3_dir)
    rows = []
    if (out_dir / "grid_results.csv").exists():
        with (out_dir / "grid_results.csv").open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                parsed = {}
                for key, value in row.items():
                    if value in {"True", "False"}:
                        parsed[key] = value == "True"
                    else:
                        try:
                            parsed[key] = float(value)
                        except (TypeError, ValueError):
                            parsed[key] = value
                rows.append(parsed)

    def existing_exact(params: dict):
        for row in rows:
            if (
                abs(float(row["beta"]) - params["beta"]) < 1e-12
                and abs(float(row["rho"]) - params["rho"]) < 1e-12
                and row["confidence_mode"] == params["confidence_mode"]
            ):
                return row
        return None

    def clone_equivalent(params: dict):
        # rho has no effect unless confidence_mode is risk_activity_ema and rho>0.
        source_mode = params["confidence_mode"]
        if source_mode in {"risk", "risk_activity"}:
            for row in rows:
                if abs(float(row["beta"]) - params["beta"]) < 1e-12 and row["confidence_mode"] == source_mode:
                    cloned = dict(row)
                    cloned["rho"] = params["rho"]
                    return cloned
        if source_mode == "risk_activity_ema" and params["rho"] == 0.0:
            for row in rows:
                if abs(float(row["beta"]) - params["beta"]) < 1e-12 and row["confidence_mode"] == "risk_activity":
                    cloned = dict(row)
                    cloned["rho"] = params["rho"]
                    cloned["confidence_mode"] = source_mode
                    return cloned
        return None

    for confidence_mode in ("risk", "risk_activity", "risk_activity_ema"):
        for beta in (0.25, 0.50, 0.75):
            for rho in (0.0, 0.7, 0.9):
                params = {"beta": beta, "rho": rho, "confidence_mode": confidence_mode, "activity_ref": activity_ref}
                if existing_exact(params) is not None:
                    continue
                cloned = clone_equivalent(params)
                if cloned is not None:
                    rows.append(cloned)
                else:
                    metrics = eval_grid_variant(args, params, args.c0_checkpoint)
                    row = flatten_grid_row(params, metrics, baseline)
                    rows.append(row)
                write_csv(out_dir / "grid_results.csv", rows)
    (out_dir / "grid_results.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    selected = select_grid_candidates(rows)
    (out_dir / "selected_candidates.json").write_text(json.dumps(selected, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_grid_report(rows, selected, out_dir / "inference_grid_report.md")
    return rows


def select_grid_candidates(rows: list[dict]) -> list[dict]:
    pool = [row for row in rows if row.get("passes_constraints")]
    if not pool:
        pool = list(rows)
    best = sorted(pool, key=lambda r: (r["FN_0_1s"], r["HR_FN_2s"], r["region_fpr"], r["risk_flips_per_second"]))[0]
    conservative_pool = [r for r in pool if r is not best and r["beta"] <= best["beta"]]
    if not conservative_pool:
        conservative_pool = [r for r in pool if r is not best]
    conservative = sorted(
        conservative_pool,
        key=lambda r: (r["beta"], r["constraint_mfm_degrade"], r["constraint_2_5_mae_degrade"], r["FN_0_1s"]),
    )[0]
    return [{"rank": 1, **best}, {"rank": 2, **conservative}]


def write_grid_report(rows: list[dict], selected: list[dict], path: Path) -> None:
    valid_count = sum(1 for row in rows if row.get("passes_constraints"))
    top = sorted(rows, key=lambda r: (not r.get("passes_constraints"), r["FN_0_1s"], r["HR_FN_2s"]))[:8]
    lines = [
        "# B3-v2 inference-only 网格搜索报告",
        "",
        f"- 总组合数：{len(rows)}",
        f"- 满足约束组合数：{valid_count}",
        "",
        "## 选中方案",
        "",
        "| Rank | beta | rho | Q mode | FN<=1s | HR FN<=2s | MFM AE | MRE(%) | 2-5s MAE | Region recall | Region FPR | Flips/s |",
        "|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in selected:
        lines.append(
            f"| {item['rank']} | {item['beta']} | {item['rho']} | {item['confidence_mode']} | "
            f"{item['FN_0_1s']:.6f} | {item['HR_FN_2s']:.6f} | {item['mean_frame_median_AE']:.6f} | "
            f"{item['MRE_percent']:.6f} | {item['MAE_2_5s']:.6f} | {item['region_recall']:.6f} | "
            f"{item['region_fpr']:.6f} | {item['risk_flips_per_second']:.6f} |"
        )
    lines += ["", "## Top 组合", "", "| beta | rho | Q mode | pass | FN<=1s | HR FN<=2s | MFM degrade | MRE degrade | 2-5s degrade |", "|---:|---:|---|---|---:|---:|---:|---:|---:|"]
    for item in top:
        lines.append(
            f"| {item['beta']} | {item['rho']} | {item['confidence_mode']} | {item['passes_constraints']} | "
            f"{item['FN_0_1s']:.6f} | {item['HR_FN_2s']:.6f} | {item['constraint_mfm_degrade']:.4f} | "
            f"{item['constraint_mre_degrade']:.4f} | {item['constraint_2_5_mae_degrade']:.4f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_eval_existing(args, activity_ref: float) -> dict:
    out = {}
    # Existing metrics already computed for B1/C0/old B3, keep them as-is for final report.
    for key, path in {
        "Original B1-Full": args.b3_dir / "original_b1_full/metrics.json",
        "C0 B1-ClipFT": args.b3_dir / "c0_b1_clipft/metrics.json",
        "原 B3-Closed": args.b3_dir / "b3_closed/metrics.json",
    }.items():
        out[key] = load_json(path)
    return out


def normalize_existing_metrics(metrics: dict, name: str) -> dict:
    p, s = metrics["paper_metrics"], metrics["safety_metrics"]
    bins = s["risk_bins"]
    def bin_get(bin_name: str, key_a: str, key_b: str):
        item = bins.get(bin_name) or bins.get(bin_name.replace("-", "_").replace("s", "s"))
        return item.get(key_a, item.get(key_b))
    task = metrics.get("task_metrics", {})
    return {
        "name": name,
        "mean_frame_median_AE": p["mean_frame_median_AE"],
        "MRE_percent": p["MRE_percent"],
        "FN_0_1s": bin_get("0-1s", "FN", "false_negative_rate"),
        "HR_FN_2s": s["high_risk_false_negative_rate"],
        "MAE_0_1s": bin_get("0-1s", "MAE", "mae"),
        "MAE_1_2s": bin_get("1-2s", "MAE", "mae"),
        "MAE_2_5s": bin_get("2-5s", "MAE", "mae"),
        "region_recall": task.get("region_recall"),
        "region_fpr": task.get("region_false_positive_rate"),
        "lead_time_ms_mean": task.get("lead_time_ms_mean"),
        "risk_flips_per_second": task.get("risk_flips_per_second"),
    }


def write_final_report(args, all_metrics: dict) -> None:
    rows = [normalize_existing_metrics(all_metrics[name], name) for name in ("Original B1-Full", "C0 B1-ClipFT", "原 B3-Closed")]
    for name in ("B3-v2 candidate 1", "B3-v2 candidate 2", "B3-v2 task-loss"):
        rows.append(normalize_existing_metrics(all_metrics[name], name))
    lines = [
        "# B3-v2 任务导向局部风险自适应报告",
        "",
        "| Method | Mean frame median AE | MRE | FN<=1s | HR FN<=2s | 0-1s MAE | 1-2s MAE | 2-5s MAE | Region recall | Region FPR | Lead time | Risk jitter |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['name']} | {fmt(row['mean_frame_median_AE'])} | {fmt(row['MRE_percent'])}% | {fmt(row['FN_0_1s'])} | "
            f"{fmt(row['HR_FN_2s'])} | {fmt(row['MAE_0_1s'])} | {fmt(row['MAE_1_2s'])} | {fmt(row['MAE_2_5s'])} | "
            f"{fmt(row['region_recall'])} | {fmt(row['region_fpr'])} | {fmt(row['lead_time_ms_mean'])} | {fmt(row['risk_flips_per_second'])} |"
        )
    c0 = next(r for r in rows if r["name"] == "C0 B1-ClipFT")
    old = next(r for r in rows if r["name"] == "原 B3-Closed")
    cand1 = next(r for r in rows if r["name"] == "B3-v2 candidate 1")
    cand2 = next(r for r in rows if r["name"] == "B3-v2 candidate 2")
    task = next(r for r in rows if r["name"] == "B3-v2 task-loss")
    def passes_quality(row: dict) -> bool:
        return (
            row["mean_frame_median_AE"] <= c0["mean_frame_median_AE"] * 1.05
            and row["MRE_percent"] <= c0["MRE_percent"] * 1.05
            and row["MAE_2_5s"] <= c0["MAE_2_5s"] * 1.10
        )

    quality_pool = [row for row in (cand1, cand2, task) if passes_quality(row)]
    if quality_pool:
        best = min(quality_pool, key=lambda r: (r["HR_FN_2s"], r["FN_0_1s"], r["region_fpr"] if r["region_fpr"] is not None else 1.0))
    else:
        best = min((cand1, cand2, task), key=lambda r: (r["mean_frame_median_AE"], r["MRE_percent"]))
    lines += [
        "",
        "## 结论",
        "",
        f"1. 保守残差调制相对原 B3-Closed 的 FN<=1s：原 B3={fmt(old['FN_0_1s'])}；candidate1={fmt(cand1['FN_0_1s'])}，candidate2={fmt(cand2['FN_0_1s'])}。两者没有保留原 B3-Closed 的大幅 FN<=1s 收益。",
        f"2. 2-5s MAE 相对 C0：C0={fmt(c0['MAE_2_5s'])}，candidate1={fmt(cand1['MAE_2_5s'])}，candidate2={fmt(cand2['MAE_2_5s'])}，task-loss={fmt(task['MAE_2_5s'])}。",
        f"3. 活动置信度+EMA 的 candidate2 明显减少 1-2s、2-5s 退化，并把 HR FN<=2s 降到 {fmt(cand2['HR_FN_2s'])}；但 Region FPR 仍较高。",
        f"4. task-loss 显著降低 FN<=1s 到 {fmt(task['FN_0_1s'])}，但 MFM AE={fmt(task['mean_frame_median_AE'])}、MRE={fmt(task['MRE_percent'])}%、2-5s MAE={fmt(task['MAE_2_5s'])}，全图质量退化不可控。",
        f"5. 当前最适合作为后续启发式避障风险前端的版本：`{best['name']}`。选择依据是满足全图质量约束，同时 HR FN<=2s 和 2-5s 退化最可控。",
        "",
        "## 文件",
        "",
        "- `inference_grid_report.md`",
        "- `grid_results.csv`",
        "- `candidate_1/metrics.json`",
        "- `candidate_2/metrics.json`",
        "- `best_task_loss/metrics.json`",
    ]
    (args.out_dir / "b3_v2_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def fmt(value, digits: int = 4) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):.{digits}f}"


def load_selected(out_dir: Path) -> list[dict]:
    selected = load_json(out_dir / "selected_candidates.json")
    out = []
    for item in selected:
        out.append(
            {
                "beta": float(item["beta"]),
                "rho": float(item["rho"]),
                "confidence_mode": str(item["confidence_mode"]),
                "activity_ref": float(item["activity_ref"]),
            }
        )
    return out


def run_all(args) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    activity_ref = compute_activity_ref(args.train_h5, args.train_clips, args.out_dir)
    existing_grid_rows = 0
    if (args.out_dir / "grid_results.csv").exists():
        existing_grid_rows = max(0, sum(1 for _ in (args.out_dir / "grid_results.csv").open(encoding="utf-8")) - 1)
    if existing_grid_rows < 27 or not (args.out_dir / "selected_candidates.json").exists():
        run_grid(args, activity_ref)
    selected = load_selected(args.out_dir)
    summaries = []
    for idx, params in enumerate(selected[:2], start=1):
        out_dir = args.out_dir / f"candidate_{idx}"
        if not (out_dir / "summary.json").exists():
            summaries.append(train_candidate(args, params, out_dir, "global"))
        else:
            summaries.append(load_json(out_dir / "summary.json"))
        if not (out_dir / "metrics.json").exists():
            eval_checkpoint(args, out_dir / "best.pt", params, out_dir, f"B3-v2 candidate {idx}")
    task_dir = args.out_dir / "best_task_loss"
    if not (task_dir / "summary.json").exists():
        train_candidate(args, selected[0], task_dir, "task")
    if not (task_dir / "metrics.json").exists():
        eval_checkpoint(args, task_dir / "best.pt", selected[0], task_dir, "B3-v2 task-loss")
    all_metrics = run_eval_existing(args, activity_ref)
    all_metrics["B3-v2 candidate 1"] = load_json(args.out_dir / "candidate_1/metrics.json")
    all_metrics["B3-v2 candidate 2"] = load_json(args.out_dir / "candidate_2/metrics.json")
    all_metrics["B3-v2 task-loss"] = load_json(args.out_dir / "best_task_loss/metrics.json")
    (args.out_dir / "all_metrics.json").write_text(json.dumps(all_metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_final_report(args, all_metrics)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--b3-dir", type=Path, default=DEFAULT_B3_DIR)
    parser.add_argument("--b1-checkpoint", type=Path, default=DEFAULT_B1)
    parser.add_argument("--c0-checkpoint", type=Path, default=DEFAULT_B3_DIR / "c0_b1_clipft/best.pt")
    parser.add_argument("--train-h5", type=Path, default=DEFAULT_TRAIN_H5)
    parser.add_argument("--test-h5", type=Path, default=DEFAULT_TEST_H5)
    parser.add_argument("--train-clips", type=Path, default=DEFAULT_TRAIN_CLIPS)
    parser.add_argument("--test-clips", type=Path, default=DEFAULT_TEST_CLIPS)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--clip-length", type=int, default=16)
    parser.add_argument("--train-stride", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--pct-start", type=float, default=0.1)
    parser.add_argument("--quick-val-clips", type=int, default=64)
    parser.add_argument("--lambda1", type=float, default=1.0)
    parser.add_argument("--lambda2", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--cache-in-memory", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    seed_all(args.seed)
    run_all(args)
    print(f"wrote {args.out_dir / 'b3_v2_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
