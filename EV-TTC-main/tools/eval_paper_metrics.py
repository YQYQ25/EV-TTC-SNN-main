#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import h5py
import hdf5plugin  # noqa: F401
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model"))

from evslim import EVSlim  # noqa: E402


class TTCEFH5(Dataset):
    def __init__(self, path):
        self.path = str(path)
        self._h5 = None
        with h5py.File(self.path, "r") as f:
            self.length = len(f["exp_filts"])
            self.height = int(f["ttc"].shape[1])
            self.width = int(f["ttc"].shape[2])

    def __len__(self):
        return self.length

    def _open(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.path, "r", libver="latest")
        return self._h5

    def __getitem__(self, idx):
        f = self._open()
        exp = torch.from_numpy(f["exp_filts"][idx].astype(np.float32))
        ttc = torch.from_numpy(f["ttc"][idx].astype(np.float32))[None]
        mask = torch.from_numpy(f["mask"][idx].astype(np.bool_))[None]
        return exp, ttc, mask


def namespace_from_cfg(cfg):
    if cfg is None:
        cfg = {
            "in_channels": 6,
            "input": 6,
            "enc_k_size": [3, 3, 3],
            "enc_channels": [16, 16, 16],
            "aspp_channels": [32, 16],
            "rate": [12, 24, 36],
            "dec_k_size": [3, 3, 3],
            "dec_channels": [8, 8, 1],
            "norm": True,
            "act": "ReLU",
            "alpha": 0.45,
        }
    return SimpleNamespace(**cfg)


def load_model(checkpoint, device):
    ckpt = torch.load(checkpoint, map_location="cpu")
    if "model_state_dict" in ckpt:
        cfg = namespace_from_cfg(ckpt.get("cfg"))
        model = EVSlim(cfg)
        state = ckpt["model_state_dict"]
    elif "state_dict" in ckpt:
        cfg = namespace_from_cfg(None)
        model = EVSlim(cfg)
        state = {
            k.replace("model.", "", 1): v
            for k, v in ckpt["state_dict"].items()
            if k.startswith("model.")
        }
    else:
        raise ValueError(f"Unsupported checkpoint format: {checkpoint}")
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def band_ranges(height):
    if height == 260:
        return [(0, 87), (87, 174), (174, 260)]
    b1 = height // 3
    b2 = (2 * height) // 3
    return [(0, b1), (b1, b2), (b2, height)]


def init_bin(lo, hi):
    return {
        "lo": lo,
        "hi": hi,
        "count": 0,
        "sum_abs": 0.0,
        "sum_gt": 0.0,
        "sum_pred": 0.0,
        "fn": 0,
        "abs_errors": [],
    }


def finalize_bin(item):
    count = item["count"]
    abs_values = np.concatenate(item["abs_errors"]) if item["abs_errors"] else np.array([], dtype=np.float32)
    return {
        "range": f"{item['lo']} < TTC <= {item['hi']} seconds",
        "valid_pixel_count": count,
        "mae": item["sum_abs"] / max(count, 1),
        "median_ae": float(np.median(abs_values)) if abs_values.size else None,
        "false_negative_rate": item["fn"] / max(count, 1),
        "false_negative_definition": f"GT in bin and prediction > {item['hi']} seconds",
        "mean_gt": item["sum_gt"] / max(count, 1),
        "mean_prediction": item["sum_pred"] / max(count, 1),
    }


@torch.no_grad()
def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    dataset = TTCEFH5(args.test_file)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    model = load_model(args.checkpoint, device)

    frame_median_ae = []
    all_abs_errors = []
    pixel_count = 0
    sum_abs = 0.0
    sum_rel = 0.0
    re_outliers = {0.10: 0, 0.50: 0, 0.80: 0}

    bands = band_ranges(dataset.height)
    band_sum_abs = [0.0, 0.0, 0.0]
    band_count = [0, 0, 0]

    high = init_bin(0.0, args.high_risk_ttc)
    risk_bins = {
        "0_1s": init_bin(0.0, 1.0),
        "1_2s": init_bin(1.0, 2.0),
        "2_5s": init_bin(2.0, 5.0),
    }

    for exp, gt, mask in loader:
        exp = exp.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            pred = model(exp).float().cpu().numpy()[:, 0]

        gt = gt.numpy()[:, 0]
        mask = mask.numpy()[:, 0].astype(bool)

        for i in range(gt.shape[0]):
            valid = mask[i] & np.isfinite(gt[i]) & np.isfinite(pred[i])
            if not valid.any():
                continue

            abs_error_map = np.abs(pred[i] - gt[i]).astype(np.float32)
            abs_values = abs_error_map[valid]
            rel_values = abs_values / np.maximum(np.abs(gt[i][valid]), args.rel_eps)

            frame_median_ae.append(float(np.median(abs_values)))
            all_abs_errors.append(abs_values)
            pixel_count += int(abs_values.size)
            sum_abs += float(abs_values.sum(dtype=np.float64))
            sum_rel += float(rel_values.sum(dtype=np.float64))
            for threshold in re_outliers:
                re_outliers[threshold] += int((rel_values > threshold).sum())

            for bi, (r0, r1) in enumerate(bands):
                band_valid = valid[r0:r1]
                if band_valid.any():
                    band_abs = abs_error_map[r0:r1][band_valid]
                    band_sum_abs[bi] += float(band_abs.sum(dtype=np.float64))
                    band_count[bi] += int(band_abs.size)

            high_mask = valid & (gt[i] > high["lo"]) & (gt[i] <= high["hi"])
            update_bin(high, gt[i], pred[i], abs_error_map, high_mask)

            for item in risk_bins.values():
                bin_mask = valid & (gt[i] > item["lo"]) & (gt[i] <= item["hi"])
                update_bin(item, gt[i], pred[i], abs_error_map, bin_mask)

    all_abs = np.concatenate(all_abs_errors) if all_abs_errors else np.array([], dtype=np.float32)
    frame_medians = np.asarray(frame_median_ae, dtype=np.float64)
    high_metrics = finalize_bin(high)
    risk_metrics = {name: finalize_bin(item) for name, item in risk_bins.items()}

    return {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "test_file": str(Path(args.test_file).resolve()),
        "num_test_samples": len(dataset),
        "image_height": dataset.height,
        "image_width": dataset.width,
        "valid_pixel_count": pixel_count,
        "paper_metrics": {
            "mean_frame_median_AE": float(frame_medians.mean()) if frame_medians.size else None,
            "std_frame_median_AE": float(frame_medians.std()) if frame_medians.size else None,
            "MRE_percent": 100.0 * sum_rel / max(pixel_count, 1),
            "RE_10_percent": 100.0 * re_outliers[0.10] / max(pixel_count, 1),
            "RE_50_percent": 100.0 * re_outliers[0.50] / max(pixel_count, 1),
            "RE_80_percent": 100.0 * re_outliers[0.80] / max(pixel_count, 1),
            "band_ranges": bands,
            "MAE_B1_B3": [
                band_sum_abs[i] / max(band_count[i], 1)
                for i in range(3)
            ],
            "band_valid_pixel_count": band_count,
        },
        "safety_metrics": {
            "pixel_MAE": sum_abs / max(pixel_count, 1),
            "pixel_median_AE": float(np.median(all_abs)) if all_abs.size else None,
            "high_risk_definition": f"0 < GT_TTC <= {args.high_risk_ttc} seconds",
            "high_risk_MAE": high_metrics["mae"],
            "high_risk_false_negative_rate": high_metrics["false_negative_rate"],
            "high_risk_false_negative_definition": (
                f"0 < GT_TTC <= {args.high_risk_ttc} seconds and prediction > {args.high_risk_ttc} seconds"
            ),
            "risk_bins": risk_metrics,
        },
    }


def update_bin(item, gt, pred, abs_error_map, bin_mask):
    if not bin_mask.any():
        return
    abs_values = abs_error_map[bin_mask]
    item["abs_errors"].append(abs_values)
    item["count"] += int(abs_values.size)
    item["sum_abs"] += float(abs_values.sum(dtype=np.float64))
    item["sum_gt"] += float(gt[bin_mask].sum(dtype=np.float64))
    item["sum_pred"] += float(pred[bin_mask].sum(dtype=np.float64))
    item["fn"] += int((pred[bin_mask] > item["hi"]).sum())


def write_markdown(metrics, out_path):
    paper = metrics["paper_metrics"]
    safety = metrics["safety_metrics"]
    bands = paper["MAE_B1_B3"]
    lines = [
        "# B0 Paper-style Metrics",
        "",
        "## Dataset",
        f"- test file: `{metrics['test_file']}`",
        f"- checkpoint: `{metrics['checkpoint']}`",
        f"- num test samples: `{metrics['num_test_samples']}`",
        f"- valid pixel count: `{metrics['valid_pixel_count']}`",
        f"- TTC map size: `{metrics['image_height']} x {metrics['image_width']}`",
        "",
        "## EV-TTC Paper-style Metrics",
        (
            "- Mean frame median AE: "
            f"`{paper['mean_frame_median_AE']:.6f} ± {paper['std_frame_median_AE']:.6f}`"
        ),
        f"- MRE(%): `{paper['MRE_percent']:.6f}`",
        (
            "- RE[10%,50%,80%]: "
            f"`{paper['RE_10_percent']:.6f} / {paper['RE_50_percent']:.6f} / {paper['RE_80_percent']:.6f}`"
        ),
        f"- MAE(B1-B3): `{bands[0]:.6f} / {bands[1]:.6f} / {bands[2]:.6f}`",
        f"- band ranges: `{paper['band_ranges']}`",
        "",
        "## Safety-oriented Metrics",
        f"- Pixel MAE: `{safety['pixel_MAE']:.6f}`",
        f"- Pixel median AE: `{safety['pixel_median_AE']:.6f}`",
        f"- High-risk MAE <=2s: `{safety['high_risk_MAE']:.6f}`",
        f"- High-risk FN <=2s: `{safety['high_risk_false_negative_rate']:.6f}`",
        "",
        "Note: high-risk false negative rate is our obstacle-avoidance safety metric, not the EV-TTC paper's main metric.",
        "",
        "### Risk Stratification",
        "",
        "| range | valid pixel count | MAE | median AE | false negative rate | mean GT | mean prediction |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key in ["0_1s", "1_2s", "2_5s"]:
        item = safety["risk_bins"][key]
        lines.append(
            f"| {item['range']} | {item['valid_pixel_count']} | {item['mae']:.6f} | "
            f"{item['median_ae']:.6f} | {item['false_negative_rate']:.6f} | "
            f"{item['mean_gt']:.6f} | {item['mean_prediction']:.6f} |"
        )
    out_path.write_text("\n".join(lines) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate EV-TTC paper-style metrics for a B0 checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--test_file", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--rel_eps", type=float, default=1e-6)
    parser.add_argument("--high_risk_ttc", type=float, default=2.0)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = evaluate(args)
    json_path = out_dir / "paper_metrics.json"
    md_path = out_dir / "paper_metrics.md"
    json_path.write_text(json.dumps(metrics, indent=2))
    write_markdown(metrics, md_path)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
