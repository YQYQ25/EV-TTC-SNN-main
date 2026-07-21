#!/usr/bin/env python3
"""Full-segment B3 evaluation with paper, safety, regional, and latency metrics."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import hdf5plugin  # noqa: F401
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model"))
sys.path.insert(0, str(ROOT / "tools"))

from b3_risk_adapter import B3RiskAdaptiveEVSlim  # noqa: E402
from b3_temporal_data import TemporalSegmentDataset  # noqa: E402
from b3_train import charbonnier_loss  # noqa: E402
from evslim import EVSlim  # noqa: E402


def load_model(path: Path, device: torch.device, mode: str, b1_checkpoint: bool) -> B3RiskAdaptiveEVSlim:
    payload = torch.load(path, map_location="cpu")
    if b1_checkpoint:
        cfg = SimpleNamespace(**payload["cfg"])
        core = EVSlim(cfg)
        core.load_state_dict(payload["model_state_dict"], strict=True)
        return B3RiskAdaptiveEVSlim(core, enabled=False).to(device).eval()
    cfg = SimpleNamespace(**payload["base_cfg"])
    core = EVSlim(cfg)
    model = B3RiskAdaptiveEVSlim(core, enabled=mode != "c0")
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model.to(device).eval()


def new_bin(lo: float, hi: float) -> dict:
    return {"lo": lo, "hi": hi, "count": 0, "sum_abs": 0.0, "sum_gt": 0.0, "sum_pred": 0.0, "fn": 0, "values": []}


def update_bin(item: dict, gt: np.ndarray, pred: np.ndarray, ae: np.ndarray, valid: np.ndarray) -> None:
    if not valid.any():
        return
    values = ae[valid].astype(np.float32)
    item["values"].append(values)
    item["count"] += int(values.size)
    item["sum_abs"] += float(values.sum(dtype=np.float64))
    item["sum_gt"] += float(gt[valid].sum(dtype=np.float64))
    item["sum_pred"] += float(pred[valid].sum(dtype=np.float64))
    item["fn"] += int((pred[valid] > item["hi"]).sum())


def finish_bin(item: dict) -> dict:
    if not item["count"]:
        return {"count": 0, "mae": None, "median_ae": None, "false_negative_rate": None, "mean_gt": None, "mean_prediction": None}
    values = np.concatenate(item["values"])
    return {
        "count": item["count"],
        "mae": item["sum_abs"] / item["count"],
        "median_ae": float(np.median(values)),
        "false_negative_rate": item["fn"] / item["count"],
        "mean_gt": item["sum_gt"] / item["count"],
        "mean_prediction": item["sum_pred"] / item["count"],
    }


def choose_previous(mode: str, previous_gt, previous_prediction):
    if mode == "c0" or previous_gt is None:
        return None
    return previous_gt if mode == "oracle" else previous_prediction


def region_risk(model: B3RiskAdaptiveEVSlim, ttc: torch.Tensor, x: torch.Tensor):
    _, risk, counts = model.modulator.compute_gains(ttc, x, reference_x=x)
    return risk[0].detach().float().cpu().numpy(), counts[0].detach().cpu().numpy()


@torch.no_grad()
def evaluate(model: B3RiskAdaptiveEVSlim, dataset: TemporalSegmentDataset, device: torch.device, mode: str, latency_frames: int = 50):
    frame_medians, all_abs = [], []
    count = 0
    sum_abs = sum_rel = 0.0
    re_counts = {0.10: 0, 0.50: 0, 0.80: 0}
    bands_sum, bands_count = [0.0] * 3, [0] * 3
    high = new_bin(0.0, 2.0)
    bins = {"0_1s": new_bin(0.0, 1.0), "1_2s": new_bin(1.0, 2.0), "2_5s": new_bin(2.0, 5.0)}
    region_high = [[new_bin(0.0, 2.0) for _ in range(3)] for _ in range(3)]
    most_dangerous_total = most_dangerous_correct = 0
    center_total = center_triggered = 0
    direction_total = direction_correct = 0
    loss_values = []
    risk_times, modulation_times, forward_times = [], [], []
    height = width = 360

    for segment_index in range(len(dataset)):
        x_seq, ttc_seq, mask_seq, _ = dataset[segment_index]
        previous_prediction = previous_gt = previous_x = None
        for time_index in range(x_seq.shape[0]):
            x = x_seq[time_index][None].to(device)
            gt = ttc_seq[time_index][None].to(device)
            mask = mask_seq[time_index][None].to(device)
            previous = choose_previous(mode, previous_gt, previous_prediction)
            timed = len(forward_times) < latency_frames
            if timed and device.type == "cuda":
                torch.cuda.synchronize()
                start = time.perf_counter()
                gains, _, _ = model.modulator.compute_gains(previous, previous_x, reference_x=x)
                torch.cuda.synchronize()
                risk_times.append((time.perf_counter() - start) * 1000)
                if not model.enabled:
                    gains = torch.ones_like(gains)
                torch.cuda.synchronize()
                start = time.perf_counter()
                modulated_x = model.modulator.apply_gains(x, gains)
                torch.cuda.synchronize()
                modulation_times.append((time.perf_counter() - start) * 1000)
                torch.cuda.synchronize()
                start = time.perf_counter()
                prediction = model.base_model(modulated_x)
                torch.cuda.synchronize()
                forward_times.append((time.perf_counter() - start) * 1000)
            else:
                prediction = model(x, previous, previous_x)
            loss_values.append(float(charbonnier_loss(gt - prediction, mask).cpu()))
            gt_np, pred_np = gt[0, 0].cpu().numpy(), prediction[0, 0].float().cpu().numpy()
            mask_np = mask[0, 0].cpu().numpy().astype(bool)
            valid = mask_np & np.isfinite(gt_np) & np.isfinite(pred_np)
            if valid.any():
                ae = np.abs(pred_np - gt_np).astype(np.float32)
                values = ae[valid]
                rel = values / np.maximum(np.abs(gt_np[valid]), 1e-6)
                frame_medians.append(float(np.median(values)))
                all_abs.append(values)
                count += int(values.size)
                sum_abs += float(values.sum(dtype=np.float64))
                sum_rel += float(rel.sum(dtype=np.float64))
                for threshold in re_counts:
                    re_counts[threshold] += int((rel > threshold).sum())
                for band, (lo, hi) in enumerate(((0, 120), (120, 240), (240, 360))):
                    band_valid = valid[lo:hi]
                    if band_valid.any():
                        band_values = ae[lo:hi][band_valid]
                        bands_sum[band] += float(band_values.sum(dtype=np.float64))
                        bands_count[band] += int(band_values.size)
                update_bin(high, gt_np, pred_np, ae, valid & (gt_np > 0) & (gt_np <= 2.0))
                for item in bins.values():
                    update_bin(item, gt_np, pred_np, ae, valid & (gt_np > item["lo"]) & (gt_np <= item["hi"]))
                for row in range(3):
                    for col in range(3):
                        lo_y, hi_y = row * 120, (row + 1) * 120
                        lo_x, hi_x = col * 120, (col + 1) * 120
                        local = valid[lo_y:hi_y, lo_x:hi_x] & (gt_np[lo_y:hi_y, lo_x:hi_x] > 0) & (gt_np[lo_y:hi_y, lo_x:hi_x] <= 2.0)
                        update_bin(region_high[row][col], gt_np[lo_y:hi_y, lo_x:hi_x], pred_np[lo_y:hi_y, lo_x:hi_x], ae[lo_y:hi_y, lo_x:hi_x], local)

            gt_risk, gt_counts = region_risk(model, gt, x)
            pred_risk, pred_counts = region_risk(model, prediction.detach(), x)
            usable_regions = gt_counts >= model.modulator.min_activity_pixels
            if usable_regions.any():
                most_dangerous_total += 1
                most_dangerous_correct += int(np.argmax(gt_risk) == np.argmax(pred_risk))
            gt_columns = gt_risk.max(axis=0)
            pred_columns = pred_risk.max(axis=0)
            if gt_columns[1] > 0.5:
                center_total += 1
                center_triggered += int(pred_columns[1] > 0.5)
                if abs(gt_columns[0] - gt_columns[2]) > 1e-6 and abs(pred_columns[0] - pred_columns[2]) > 1e-6:
                    direction_total += 1
                    direction_correct += int(np.argmin([gt_columns[0], gt_columns[2]]) == np.argmin([pred_columns[0], pred_columns[2]]))
            previous_prediction = prediction.detach()
            previous_gt = gt.detach()
            previous_x = x.detach()

    abs_values = np.concatenate(all_abs) if all_abs else np.array([], dtype=np.float32)
    return {
        "num_segments": len(dataset),
        "num_frames": len(frame_medians),
        "quick_loss": float(np.mean(loss_values)),
        "paper_metrics": {
            "mean_frame_median_AE": float(np.mean(frame_medians)),
            "std_frame_median_AE": float(np.std(frame_medians)),
            "MRE_percent": 100 * sum_rel / max(count, 1),
            "RE_10_percent": 100 * re_counts[0.10] / max(count, 1),
            "RE_50_percent": 100 * re_counts[0.50] / max(count, 1),
            "RE_80_percent": 100 * re_counts[0.80] / max(count, 1),
            "MAE_B1_B3": [bands_sum[i] / max(bands_count[i], 1) for i in range(3)],
        },
        "safety_metrics": {
            "pixel_MAE": sum_abs / max(count, 1),
            "pixel_median_AE": float(np.median(abs_values)),
            "high_risk_MAE": finish_bin(high)["mae"],
            "high_risk_false_negative_rate": finish_bin(high)["false_negative_rate"],
            "risk_bins": {name: finish_bin(item) for name, item in bins.items()},
        },
        "regional_safety": {
            "high_risk_3x3": [[finish_bin(region_high[row][col]) for col in range(3)] for row in range(3)],
            "center_danger_recall": center_triggered / center_total if center_total else None,
            "center_danger_total": center_total,
            "most_dangerous_region_accuracy": most_dangerous_correct / most_dangerous_total if most_dangerous_total else None,
            "most_dangerous_region_total": most_dangerous_total,
            "left_right_safe_direction_agreement": direction_correct / direction_total if direction_total else None,
            "left_right_direction_total": direction_total,
        },
        "latency_ms": {
            "risk_calculation_mean": float(np.mean(risk_times)) if risk_times else None,
            "region_modulation_mean": float(np.mean(modulation_times)) if modulation_times else None,
            "evslim_forward_mean": float(np.mean(forward_times)) if forward_times else None,
            "complete_b3_mean": float(np.mean(np.asarray(risk_times) + np.asarray(modulation_times) + np.asarray(forward_times))) if forward_times else None,
        },
    }


def write_markdown(metrics: dict, path: Path, name: str) -> None:
    paper, safety, regional, latency = metrics["paper_metrics"], metrics["safety_metrics"], metrics["regional_safety"], metrics["latency_ms"]
    lines = [
        f"# {name} 连续 segment 评估",
        "",
        f"- 连续片段数：{metrics['num_segments']}",
        f"- 评估帧数：{metrics['num_frames']}",
        f"- Mean frame median AE：{paper['mean_frame_median_AE']:.6f}",
        f"- MRE：{paper['MRE_percent']:.6f}%",
        f"- RE[10/50/80]：{paper['RE_10_percent']:.4f} / {paper['RE_50_percent']:.4f} / {paper['RE_80_percent']:.4f}%",
        f"- MAE(B1-B3)：{' / '.join(f'{x:.6f}' for x in paper['MAE_B1_B3'])}",
        f"- Pixel MAE：{safety['pixel_MAE']:.6f}",
        f"- HR MAE <=2s：{safety['high_risk_MAE']}",
        f"- HR FN <=2s：{safety['high_risk_false_negative_rate']}",
        f"- 中心危险召回率：{regional['center_danger_recall']}",
        f"- 最危险区域识别准确率：{regional['most_dangerous_region_accuracy']}",
        f"- 左右安全方向一致率：{regional['left_right_safe_direction_agreement']}",
        f"- 延迟：风险 {latency['risk_calculation_mean']} ms；调制 {latency['region_modulation_mean']} ms；EV-Slim {latency['evslim_forward_mean']} ms；完整 {latency['complete_b3_mean']} ms。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--mode", choices=("c0", "oracle", "closed"), required=True)
    parser.add_argument("--b1-checkpoint", action="store_true")
    parser.add_argument("--test-h5", default="/home/hello/research_project/event+SNN+TTC/logs/ttc_TTCEF/b1_polarity_full_20ep/b1_full_test.h5")
    parser.add_argument("--test-clips", default="/home/hello/research_project/event+SNN+TTC/logs/ttc_TTCEF/temporal_audit/test_clips.json")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--name", default="B3")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = load_model(Path(args.checkpoint), device, args.mode, args.b1_checkpoint)
    metrics = evaluate(model, TemporalSegmentDataset(args.test_h5, args.test_clips), device, args.mode)
    metrics.update({"name": args.name, "mode": args.mode, "checkpoint": str(Path(args.checkpoint).resolve())})
    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(metrics, out_dir / "metrics.md", args.name)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

