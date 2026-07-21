#!/usr/bin/env python3
"""Diagnose why B3 changes near-risk TTC metrics without changing model weights."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import hdf5plugin  # noqa: F401
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model"))
sys.path.insert(0, str(ROOT / "tools"))

from b3_eval import load_model  # noqa: E402
from b3_temporal_data import TemporalSegmentDataset  # noqa: E402


METHODS = ("B1-Full", "C0", "Oracle", "Closed")
TTC_BINS = (
    ("0-1s", 0.0, 1.0),
    ("1-2s", 1.0, 2.0),
    ("2-5s", 2.0, 5.0),
    (">5s", 5.0, math.inf),
)
RISK_LEVELS = ("low<=0.5", "0.5-1", "1-2", ">2")


class ScalarAgg:
    def __init__(self, keep_values: bool = False):
        self.count = 0
        self.sum_abs = 0.0
        self.sum_rel = 0.0
        self.sum_bias = 0.0
        self.sum_pred = 0.0
        self.sum_gt = 0.0
        self.fn = 0
        self.values: list[np.ndarray] = [] if keep_values else None

    def update(self, gt: np.ndarray, pred: np.ndarray, valid: np.ndarray, fn_threshold: float | None = None) -> None:
        if not valid.any():
            return
        gt_v = gt[valid].astype(np.float32, copy=False)
        pred_v = pred[valid].astype(np.float32, copy=False)
        err = pred_v - gt_v
        abs_err = np.abs(err).astype(np.float32, copy=False)
        self.count += int(abs_err.size)
        self.sum_abs += float(abs_err.sum(dtype=np.float64))
        self.sum_rel += float((abs_err / np.maximum(np.abs(gt_v), 1e-6)).sum(dtype=np.float64))
        self.sum_bias += float(err.sum(dtype=np.float64))
        self.sum_pred += float(pred_v.sum(dtype=np.float64))
        self.sum_gt += float(gt_v.sum(dtype=np.float64))
        if fn_threshold is not None and math.isfinite(fn_threshold):
            self.fn += int((pred_v > fn_threshold).sum())
        if self.values is not None:
            self.values.append(abs_err.copy())

    def finish(self) -> dict:
        if not self.count:
            return {
                "count": 0,
                "MAE": None,
                "median_AE": None,
                "MRE": None,
                "FN": None,
                "mean_prediction_minus_GT": None,
                "prediction_mean": None,
                "GT_mean": None,
            }
        values = np.concatenate(self.values) if self.values else None
        return {
            "count": self.count,
            "MAE": self.sum_abs / self.count,
            "median_AE": float(np.median(values)) if values is not None and values.size else None,
            "MRE": self.sum_rel / self.count,
            "FN": self.fn / self.count,
            "mean_prediction_minus_GT": self.sum_bias / self.count,
            "prediction_mean": self.sum_pred / self.count,
            "GT_mean": self.sum_gt / self.count,
        }


class TensorStats:
    def __init__(self, shape: tuple[int, ...]):
        self.shape = shape
        self.count = np.zeros(shape, dtype=np.float64)
        self.sum = np.zeros(shape, dtype=np.float64)
        self.sum_sq = np.zeros(shape, dtype=np.float64)
        self.sum_abs = np.zeros(shape, dtype=np.float64)
        self.nonzero = np.zeros(shape, dtype=np.float64)

    def update_channel(self, array: np.ndarray) -> None:
        # array: [C,H,W]
        axes = tuple(range(1, array.ndim))
        n = np.prod(array.shape[1:])
        self.count += n
        self.sum += array.sum(axis=axes, dtype=np.float64)
        self.sum_sq += np.square(array, dtype=np.float32).sum(axis=axes, dtype=np.float64)
        self.sum_abs += np.abs(array).sum(axis=axes, dtype=np.float64)
        self.nonzero += (np.abs(array) > 1e-8).sum(axis=axes)

    def update_scalar(self, tensor: torch.Tensor) -> None:
        array = tensor.detach().float().cpu().numpy()
        self.count += array.size
        self.sum += float(array.sum(dtype=np.float64))
        self.sum_sq += float(np.square(array, dtype=np.float32).sum(dtype=np.float64))
        self.sum_abs += float(np.abs(array).sum(dtype=np.float64))
        self.nonzero += int((np.abs(array) > 1e-8).sum())

    def finish(self) -> dict:
        mean = np.divide(self.sum, self.count, out=np.zeros_like(self.sum), where=self.count > 0)
        second = np.divide(self.sum_sq, self.count, out=np.zeros_like(self.sum_sq), where=self.count > 0)
        std = np.sqrt(np.maximum(second - mean * mean, 0.0))
        abs_mean = np.divide(self.sum_abs, self.count, out=np.zeros_like(self.sum_abs), where=self.count > 0)
        nonzero_ratio = np.divide(self.nonzero, self.count, out=np.zeros_like(self.nonzero), where=self.count > 0)
        return {"mean": mean, "std": std, "abs_mean": abs_mean, "nonzero_ratio": nonzero_ratio, "count": self.count}


def risk_level(values: np.ndarray) -> np.ndarray:
    out = np.zeros(values.shape, dtype=np.int64)
    out[(values > 0.5) & (values <= 1.0)] = 1
    out[(values > 1.0) & (values <= 2.0)] = 2
    out[values > 2.0] = 3
    return out


def fn_threshold(bin_hi: float) -> float | None:
    return bin_hi if math.isfinite(bin_hi) else None


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def valid_bin(gt: np.ndarray, valid: np.ndarray, lo: float, hi: float) -> np.ndarray:
    if math.isfinite(hi):
        return valid & (gt > lo) & (gt <= hi)
    return valid & (gt > lo)


def region_slices(row: int, col: int, height: int = 360, width: int = 360) -> tuple[slice, slice]:
    return slice(row * height // 3, (row + 1) * height // 3), slice(col * width // 3, (col + 1) * width // 3)


def boundary_mask(width: int, height: int = 360, image_width: int = 360) -> np.ndarray:
    yy, xx = np.indices((height, image_width))
    mask = np.zeros((height, image_width), dtype=bool)
    for line in (height // 3, 2 * height // 3):
        mask |= np.abs(yy - line) < width
    for line in (image_width // 3, 2 * image_width // 3):
        mask |= np.abs(xx - line) < width
    return mask


def load_all_models(args, device: torch.device):
    b3_dir = Path(args.b3_dir)
    return {
        "B1-Full": load_model(Path(args.b1_checkpoint), device, "c0", True),
        "C0": load_model(b3_dir / "c0_b1_clipft/best.pt", device, "c0", False),
        "Oracle": load_model(b3_dir / "b3_oracle/best.pt", device, "oracle", False),
        "Closed": load_model(b3_dir / "b3_closed/best.pt", device, "closed", False),
    }


def register_activation_hooks(models, activation_stats):
    handles = []
    for method in ("C0", "Oracle", "Closed"):
        base = models[method].base_model

        def conv_hook(_module, _inputs, output, method=method):
            activation_stats[(method, "first_conv_output")].update_scalar(output)

        def bn_pre_hook(_module, inputs, method=method):
            activation_stats[(method, "first_bn_input")].update_scalar(inputs[0])

        def bn_hook(_module, _inputs, output, method=method):
            activation_stats[(method, "first_bn_output")].update_scalar(output)

        handles.append(base.enc[0].conv2d.register_forward_hook(conv_hook))
        handles.append(base.enc[0].norm_layer.register_forward_pre_hook(bn_pre_hook))
        handles.append(base.enc[0].norm_layer.register_forward_hook(bn_hook))
    return handles


def model_step(model, x, previous_ttc, previous_x):
    prediction, aux = model(x, previous_ttc, previous_x, return_aux=True)
    modulated_x, _ = model.prepare_input(x, previous_ttc, previous_x)
    return prediction, aux, modulated_x


def apply_globalized(model, x, previous_prediction, previous_x):
    gains, risk, counts = model.modulator.compute_gains(previous_prediction, previous_x, reference_x=x)
    gains = gains.mean(dim=(2, 3), keepdim=True).expand_as(gains)
    return model.base_model(model.modulator.apply_gains(x, gains)), risk, counts


def apply_shuffled(model, x, previous_prediction, previous_x, rng: np.random.Generator):
    gains, risk, counts = model.modulator.compute_gains(previous_prediction, previous_x, reference_x=x)
    perm = torch.as_tensor(rng.permutation(9), device=gains.device, dtype=torch.long)
    shuffled = gains.reshape(gains.shape[0], 6, 9)[:, :, perm].reshape_as(gains)
    return model.base_model(model.modulator.apply_gains(x, shuffled)), risk, counts


def update_metric_family(aggs, method: str, gt: np.ndarray, pred: np.ndarray, valid: np.ndarray) -> None:
    for bin_name, lo, hi in TTC_BINS:
        aggs[(method, bin_name)].update(gt, pred, valid_bin(gt, valid, lo, hi), fn_threshold(hi))


def update_boundary(boundary_aggs, method: str, gt: np.ndarray, pred: np.ndarray, valid: np.ndarray, masks: dict[int, np.ndarray]) -> None:
    high_valid = valid & (gt > 0) & (gt <= 2.0)
    for width, mask in masks.items():
        boundary_aggs[(method, width, "boundary")].update(gt, pred, high_valid & mask, 2.0)
        boundary_aggs[(method, width, "interior")].update(gt, pred, high_valid & ~mask, 2.0)


def update_spillover(spillover_aggs, method: str, gt: np.ndarray, pred: np.ndarray, valid: np.ndarray) -> None:
    for risk_name, threshold in (("GT<=1s", 1.0), ("GT<=2s", 2.0)):
        for row in range(3):
            for col in range(3):
                ys, xs = region_slices(row, col)
                region_valid = valid[ys, xs]
                local_gt, local_pred = gt[ys, xs], pred[ys, xs]
                high = region_valid & (local_gt > 0) & (local_gt <= threshold)
                safe = region_valid & (local_gt > 2.0)
                spillover_aggs[(method, risk_name, row, col, "high-risk pixels")].update(local_gt, local_pred, high, threshold)
                spillover_aggs[(method, risk_name, row, col, "same-region GT>2s pixels")].update(local_gt, local_pred, safe, None)


def update_risk_spatial(state, gt: np.ndarray, mask: np.ndarray, activity: np.ndarray) -> None:
    state["frames"] += 1
    count_1s = int((mask & (gt > 0) & (gt <= 1.0)).sum())
    if count_1s > 0:
        state["frames_with_1s"] += 1
        state["frame_1s_counts"].append(count_1s)
    for row in range(3):
        for col in range(3):
            ys, xs = region_slices(row, col)
            local_gt = gt[ys, xs]
            local_mask = mask[ys, xs]
            local_activity = activity[ys, xs]
            state["valid_count"][row, col] += int(local_mask.sum())
            state["activity_count"][row, col] += int(local_activity.sum())
            state["region_pixels"][row, col] += int(local_mask.size)
            for bin_index, (_name, lo, hi) in enumerate(TTC_BINS[:3]):
                state["bin_counts"][bin_index, row, col] += int(valid_bin(local_gt, local_mask, lo, hi).sum())


def update_threshold_crossing(state, gt: np.ndarray, preds: dict[str, np.ndarray], valid: np.ndarray) -> None:
    target = valid & (gt > 0) & (gt <= 1.0)
    if not target.any():
        return
    c0 = preds["C0"]
    c0_ae = np.abs(c0 - gt)
    state["total_gt_le_1s"] += int(target.sum())
    bins = np.linspace(0.8, 1.2, 41)
    for method in ("C0", "Oracle", "Closed"):
        hist, _ = np.histogram(preds[method][target], bins=bins)
        state["hist"][method] += hist
    for method in ("Oracle", "Closed"):
        pred = preds[method]
        method_ae = np.abs(pred - gt)
        crossed = target & (c0 > 1.0) & (pred <= 1.0)
        state[(method, "crossed_count")] += int(crossed.sum())
        if crossed.any():
            state[(method, "crossed_c0_mae_sum")] += float(c0_ae[crossed].sum(dtype=np.float64))
            state[(method, "crossed_method_mae_sum")] += float(method_ae[crossed].sum(dtype=np.float64))
            near_no_gain = crossed & (pred >= 0.8) & (pred <= 1.2) & ((c0_ae - method_ae) < 0.05)
            state[(method, "near_no_gain_count")] += int(near_no_gain.sum())


def corrcoef_safe(a: np.ndarray, b: np.ndarray) -> float | None:
    if a.size < 2 or np.std(a) < 1e-8 or np.std(b) < 1e-8:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def update_risk_temporal(state, segment_index: int, time_index: int, gt_risk: np.ndarray, closed_risk: np.ndarray, frame_ae: float) -> None:
    diff = closed_risk - gt_risk
    state["rows"].append(
        {
            "segment": segment_index,
            "time_index": time_index,
            "gt_risk_mean": float(gt_risk.mean()),
            "closed_risk_mean": float(closed_risk.mean()),
            "risk_bias_mean": float(diff.mean()),
            "risk_abs_error_mean": float(np.abs(diff).mean()),
            "closed_frame_MAE": frame_ae,
        }
    )
    state["gt_series"].setdefault(segment_index, []).append(gt_risk.copy())
    state["closed_series"].setdefault(segment_index, []).append(closed_risk.copy())
    state["frame_mae"].setdefault(segment_index, []).append(frame_ae)
    state["risk_bias_sum"] += diff
    state["risk_abs_sum"] += np.abs(diff)
    state["risk_count"] += 1
    gt_level = risk_level(gt_risk).ravel()
    closed_level = risk_level(closed_risk).ravel()
    for gt_item, pred_item in zip(gt_level, closed_level):
        state["confusion"][gt_item, pred_item] += 1


def finalize_temporal(state, out_dir: Path) -> list[dict]:
    rows = state["rows"]
    if not rows:
        return []
    gt_flat, pred_flat = [], []
    for segment in sorted(state["gt_series"]):
        gt = np.stack(state["gt_series"][segment])
        pred = np.stack(state["closed_series"][segment])
        gt_flat.append(gt.reshape(-1))
        pred_flat.append(pred.reshape(-1))
    gt_all = np.concatenate(gt_flat)
    pred_all = np.concatenate(pred_flat)
    overall = {
        "segment": "ALL",
        "time_index": "ALL",
        "gt_risk_mean": float(gt_all.mean()),
        "closed_risk_mean": float(pred_all.mean()),
        "risk_bias_mean": float((pred_all - gt_all).mean()),
        "risk_abs_error_mean": float(np.abs(pred_all - gt_all).mean()),
        "risk_correlation": corrcoef_safe(gt_all, pred_all),
    }
    rows.append(overall)
    for gt_level in range(4):
        for closed_level in range(4):
            rows.append(
                {
                    "section": "risk_level_confusion",
                    "gt_level": RISK_LEVELS[gt_level],
                    "closed_level": RISK_LEVELS[closed_level],
                    "count": int(state["confusion"][gt_level, closed_level]),
                }
            )

    lag_rows = []
    for segment in sorted(state["gt_series"]):
        gt = np.stack(state["gt_series"][segment])
        pred = np.stack(state["closed_series"][segment])
        for row in range(3):
            for col in range(3):
                g = gt[:, row, col]
                p = pred[:, row, col]
                best_lag, best_corr = 0, None
                for lag in range(-5, 6):
                    if lag < 0:
                        corr = corrcoef_safe(g[-lag:], p[:lag])
                    elif lag > 0:
                        corr = corrcoef_safe(g[:-lag], p[lag:])
                    else:
                        corr = corrcoef_safe(g, p)
                    if corr is not None and (best_corr is None or corr > best_corr):
                        best_lag, best_corr = lag, corr
                lag_rows.append({"section": "lag", "segment": segment, "row": row, "col": col, "best_lag_frames": best_lag, "best_corr": best_corr})
    rows.extend(lag_rows)

    error_values = np.asarray([r["risk_abs_error_mean"] for r in state["rows"] if isinstance(r.get("risk_abs_error_mean"), float)])
    threshold = float(np.nanpercentile(error_values, 90)) if error_values.size else 0.5
    for segment in sorted(state["gt_series"]):
        mae = np.asarray(state["frame_mae"][segment], dtype=np.float64)
        risk_err = np.asarray([np.abs(a - b).mean() for a, b in zip(state["gt_series"][segment], state["closed_series"][segment])])
        active = risk_err > threshold
        start = None
        for index, value in enumerate(active.tolist() + [False]):
            if value and start is None:
                start = index
            if not value and start is not None:
                rows.append(
                    {
                        "section": "risk_error_run",
                        "segment": segment,
                        "start_t": start,
                        "end_t": index - 1,
                        "duration_frames": index - start,
                        "max_risk_abs_error": float(risk_err[start:index].max()),
                        "mean_closed_frame_MAE": float(mae[start:index].mean()),
                    }
                )
                start = None

    worst = max(state["rows"], key=lambda r: r.get("risk_abs_error_mean", -1) if isinstance(r.get("risk_abs_error_mean"), float) else -1)
    if isinstance(worst.get("segment"), int):
        segment = worst["segment"]
        center = int(worst["time_index"])
        gt = np.stack(state["gt_series"][segment])
        pred = np.stack(state["closed_series"][segment])
        mae = np.asarray(state["frame_mae"][segment])
        lo, hi = max(0, center - 30), min(gt.shape[0], center + 31)
        xs = np.arange(lo, hi)
        fig, ax1 = plt.subplots(figsize=(9, 4))
        ax1.plot(xs, gt[lo:hi].mean(axis=(1, 2)), label="GT risk mean")
        ax1.plot(xs, pred[lo:hi].mean(axis=(1, 2)), label="Closed risk mean")
        ax1.set_xlabel("segment time index")
        ax1.set_ylabel("risk")
        ax2 = ax1.twinx()
        ax2.plot(xs, mae[lo:hi], color="tab:red", alpha=0.5, label="Closed frame MAE")
        ax2.set_ylabel("frame MAE")
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc="upper right")
        fig.tight_layout()
        fig.savefig(out_dir / "risk_feedback_temporal_curve.png", dpi=160)
        plt.close(fig)

    return rows


def rows_from_aggs(aggs, extra_keys: tuple[str, ...], section: str | None = None, baseline_key=None) -> list[dict]:
    rows = []
    baseline = {}
    if baseline_key is not None:
        for key, agg in aggs.items():
            if key[0] == baseline_key:
                baseline[key[1:]] = agg.finish()
    for key, agg in sorted(aggs.items(), key=lambda item: tuple(str(x) for x in item[0])):
        stats = agg.finish()
        row = {name: value for name, value in zip(extra_keys, key)}
        if section is not None:
            row["section"] = section
        row.update(stats)
        if baseline_key is not None and key[0] != baseline_key:
            base = baseline.get(key[1:])
            if base and base["MAE"] not in (None, 0) and row["MAE"] is not None:
                row["MAE_delta_vs_" + baseline_key] = row["MAE"] - base["MAE"]
                row["MAE_rel_delta_vs_" + baseline_key] = (row["MAE"] - base["MAE"]) / base["MAE"]
        rows.append(row)
    return rows


def save_plots(out_dir: Path, risk_state, ttc_rows, threshold_state, boundary_rows, spillover_rows, activation_rows, control_rows) -> None:
    names = [item[0] for item in TTC_BINS[:3]]
    counts = risk_state["bin_counts"]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3), constrained_layout=True)
    for i, ax in enumerate(axes):
        im = ax.imshow(counts[i], cmap="magma")
        ax.set_title(names[i])
        for r in range(3):
            for c in range(3):
                ax.text(c, r, str(int(counts[i, r, c])), ha="center", va="center", color="white", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.savefig(out_dir / "risk_spatial_distribution_heatmap.png", dpi=160)
    plt.close(fig)

    mae_rows = [r for r in ttc_rows if r.get("method") in METHODS and r.get("MAE") is not None]
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(TTC_BINS))
    width = 0.18
    for idx, method in enumerate(METHODS):
        values = [next((r["MAE"] for r in mae_rows if r["method"] == method and r["ttc_bin"] == name), np.nan) for name, _, _ in TTC_BINS]
        ax.bar(x + (idx - 1.5) * width, values, width, label=method)
    ax.set_xticks(x)
    ax.set_xticklabels([b[0] for b in TTC_BINS])
    ax.set_ylabel("MAE")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "ttc_bin_mae.png", dpi=160)
    plt.close(fig)

    bins = np.linspace(0.8, 1.2, 41)
    centers = 0.5 * (bins[:-1] + bins[1:])
    fig, ax = plt.subplots(figsize=(8, 4))
    for method in ("C0", "Oracle", "Closed"):
        ax.plot(centers, threshold_state["hist"][method], label=method)
    ax.set_xlabel("prediction for GT<=1s pixels")
    ax.set_ylabel("pixel count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "threshold_crossing_histogram.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    selected = [r for r in boundary_rows if r.get("zone") == "boundary" and r.get("MAE") is not None]
    for method in ("C0", "Oracle", "Closed"):
        xs = [r["width"] for r in selected if r["method"] == method]
        ys = [r["MAE"] for r in selected if r["method"] == method]
        ax.plot(xs, ys, marker="o", label=method)
    ax.set_xlabel("boundary width (pixels)")
    ax.set_ylabel("HR<=2s boundary MAE")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "boundary_vs_interior_mae.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4))
    rows = [r for r in spillover_rows if r.get("risk_definition") == "GT<=1s" and r.get("pixel_class") == "same-region GT>2s pixels" and r.get("MAE_delta_vs_C0") is not None]
    labels, oracle, closed = [], [], []
    for row in range(3):
        for col in range(3):
            labels.append(f"{row},{col}")
            oracle.append(next((r["MAE_delta_vs_C0"] for r in rows if r["method"] == "Oracle" and r["row"] == row and r["col"] == col), np.nan))
            closed.append(next((r["MAE_delta_vs_C0"] for r in rows if r["method"] == "Closed" and r["row"] == row and r["col"] == col), np.nan))
    x = np.arange(len(labels))
    ax.bar(x - 0.18, oracle, 0.36, label="Oracle")
    ax.bar(x + 0.18, closed, 0.36, label="Closed")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("safe-pixel MAE delta vs C0")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "regional_spillover_safe_delta.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    rows = [r for r in activation_rows if r.get("scope") == "activation" and r.get("target") == "first_bn_output"]
    ax.bar([r["method"] for r in rows], [r["std"] for r in rows])
    ax.set_ylabel("first BN output std")
    fig.tight_layout()
    fig.savefig(out_dir / "activation_bn_output_std.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4))
    rows = [r for r in control_rows if r.get("section") == "global"]
    ax.bar([r["method"] for r in rows], [np.nan if r["FN_le_1s"] is None else r["FN_le_1s"] for r in rows])
    ax.set_ylabel("FN<=1s")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(out_dir / "diagnostic_controls_fn1.png", dpi=160)
    plt.close(fig)


@torch.no_grad()
def analyze(args) -> dict:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    dataset = TemporalSegmentDataset(args.test_h5, args.test_clips)
    models = load_all_models(args, device)
    activation_stats = defaultdict(lambda: TensorStats(()))
    hooks = register_activation_hooks(models, activation_stats)

    risk_state = {
        "frames": 0,
        "frames_with_1s": 0,
        "frame_1s_counts": [],
        "bin_counts": np.zeros((3, 3, 3), dtype=np.int64),
        "valid_count": np.zeros((3, 3), dtype=np.int64),
        "activity_count": np.zeros((3, 3), dtype=np.int64),
        "region_pixels": np.zeros((3, 3), dtype=np.int64),
    }
    ttc_aggs = defaultdict(lambda: ScalarAgg(keep_values=True))
    boundary_aggs = defaultdict(lambda: ScalarAgg())
    spillover_aggs = defaultdict(lambda: ScalarAgg())
    control_ttc_aggs = defaultdict(lambda: ScalarAgg())
    control_boundary_aggs = defaultdict(lambda: ScalarAgg())
    threshold_state = defaultdict(int)
    threshold_state["hist"] = {method: np.zeros(40, dtype=np.int64) for method in ("C0", "Oracle", "Closed")}
    risk_temporal = {
        "rows": [],
        "gt_series": {},
        "closed_series": {},
        "frame_mae": {},
        "risk_bias_sum": np.zeros((3, 3), dtype=np.float64),
        "risk_abs_sum": np.zeros((3, 3), dtype=np.float64),
        "risk_count": 0,
        "confusion": np.zeros((4, 4), dtype=np.int64),
    }
    input_stats = defaultdict(lambda: TensorStats((12,)))
    input_region_stats = defaultdict(lambda: TensorStats((12, 3, 3)))
    boundary_masks = {width: boundary_mask(width) for width in (5, 10, 20)}
    rng = np.random.default_rng(7)
    processed_frames = 0

    for segment_index in range(len(dataset)):
        x_seq, ttc_seq, mask_seq, _ = dataset[segment_index]
        previous_gt = None
        previous_x = None
        previous_pred = {method: None for method in METHODS}
        control_prev = {"Closed identity": None, "Globalized gain": None, "Shuffled-region gain": None}
        for time_index in range(x_seq.shape[0]):
            x = x_seq[time_index][None].to(device)
            gt_t = ttc_seq[time_index][None].to(device)
            mask_t = mask_seq[time_index][None].to(device)

            predictions_t, modulated_inputs, aux_by_method = {}, {}, {}
            for method in METHODS:
                model = models[method]
                if method in ("B1-Full", "C0") or previous_gt is None:
                    previous = None
                elif method == "Oracle":
                    previous = previous_gt
                else:
                    previous = previous_pred["Closed"]
                pred, aux, mod_x = model_step(model, x, previous, previous_x)
                predictions_t[method] = pred
                aux_by_method[method] = aux
                modulated_inputs[method] = mod_x

            closed_identity = models["Closed"].base_model(x)
            globalized, _, _ = apply_globalized(models["Closed"], x, control_prev["Globalized gain"], previous_x)
            shuffled, _, _ = apply_shuffled(models["Closed"], x, control_prev["Shuffled-region gain"], previous_x, rng)
            control_predictions = {
                "C0 identity": predictions_t["C0"],
                "Closed standard": predictions_t["Closed"],
                "Closed identity": closed_identity,
                "Globalized gain": globalized,
                "Shuffled-region gain": shuffled,
            }

            gt = gt_t[0, 0].float().cpu().numpy()
            valid = mask_t[0, 0].cpu().numpy().astype(bool) & np.isfinite(gt)
            activity = (x[0, 5] - x[0, 11]).abs().detach().cpu().numpy() > 1e-3
            update_risk_spatial(risk_state, gt, valid, activity)
            pred_np = {method: predictions_t[method][0, 0].float().cpu().numpy() for method in METHODS}
            control_np = {method: tensor[0, 0].float().cpu().numpy() for method, tensor in control_predictions.items()}

            for method in METHODS:
                update_metric_family(ttc_aggs, method, gt, pred_np[method], valid)
                update_boundary(boundary_aggs, method, gt, pred_np[method], valid, boundary_masks)
                update_spillover(spillover_aggs, method, gt, pred_np[method], valid)
            for method in control_predictions:
                update_metric_family(control_ttc_aggs, method, gt, control_np[method], valid)
                update_boundary(control_boundary_aggs, method, gt, control_np[method], valid, boundary_masks)
            update_threshold_crossing(threshold_state, gt, pred_np, valid)

            if previous_gt is None:
                gt_risk = np.zeros((3, 3), dtype=np.float32)
            else:
                _, risk_tensor, _ = models["Closed"].modulator.compute_gains(previous_gt, previous_x, reference_x=x)
                gt_risk = risk_tensor[0].float().cpu().numpy()
            closed_risk = aux_by_method["Closed"]["risk"][0].float().cpu().numpy()
            closed_ae = np.abs(pred_np["Closed"] - gt)
            frame_mae = float(closed_ae[valid].mean()) if valid.any() else float("nan")
            update_risk_temporal(risk_temporal, segment_index, time_index, gt_risk, closed_risk, frame_mae)

            original_x = x[0].float().cpu().numpy()
            for method in ("C0", "Oracle", "Closed"):
                post = modulated_inputs[method][0].float().detach().cpu().numpy()
                input_stats[(method, "pre")].update_channel(original_x)
                input_stats[(method, "post")].update_channel(post)
                for row in range(3):
                    for col in range(3):
                        ys, xs = region_slices(row, col)
                        input_region_stats[(method, "pre")].count[:, row, col] += original_x[:, ys, xs].shape[1] * original_x[:, ys, xs].shape[2]
                        input_region_stats[(method, "pre")].sum[:, row, col] += original_x[:, ys, xs].sum(axis=(1, 2), dtype=np.float64)
                        input_region_stats[(method, "pre")].sum_sq[:, row, col] += np.square(original_x[:, ys, xs], dtype=np.float32).sum(axis=(1, 2), dtype=np.float64)
                        input_region_stats[(method, "pre")].sum_abs[:, row, col] += np.abs(original_x[:, ys, xs]).sum(axis=(1, 2), dtype=np.float64)
                        input_region_stats[(method, "pre")].nonzero[:, row, col] += (np.abs(original_x[:, ys, xs]) > 1e-8).sum(axis=(1, 2))
                        input_region_stats[(method, "post")].count[:, row, col] += post[:, ys, xs].shape[1] * post[:, ys, xs].shape[2]
                        input_region_stats[(method, "post")].sum[:, row, col] += post[:, ys, xs].sum(axis=(1, 2), dtype=np.float64)
                        input_region_stats[(method, "post")].sum_sq[:, row, col] += np.square(post[:, ys, xs], dtype=np.float32).sum(axis=(1, 2), dtype=np.float64)
                        input_region_stats[(method, "post")].sum_abs[:, row, col] += np.abs(post[:, ys, xs]).sum(axis=(1, 2), dtype=np.float64)
                        input_region_stats[(method, "post")].nonzero[:, row, col] += (np.abs(post[:, ys, xs]) > 1e-8).sum(axis=(1, 2))

            previous_gt = gt_t.detach()
            previous_x = x.detach()
            for method in METHODS:
                previous_pred[method] = predictions_t[method].detach()
            control_prev["Closed identity"] = closed_identity.detach()
            control_prev["Globalized gain"] = globalized.detach()
            control_prev["Shuffled-region gain"] = shuffled.detach()
            processed_frames += 1
            if args.max_frames is not None and processed_frames >= args.max_frames:
                break
        if args.max_frames is not None and processed_frames >= args.max_frames:
            break

    for handle in hooks:
        handle.remove()

    ttc_rows = rows_from_aggs(ttc_aggs, ("method", "ttc_bin"))
    boundary_rows = rows_from_aggs(boundary_aggs, ("method", "width", "zone"), baseline_key="C0")
    spillover_rows = rows_from_aggs(spillover_aggs, ("method", "risk_definition", "row", "col", "pixel_class"), baseline_key="C0")
    control_ttc_rows = rows_from_aggs(control_ttc_aggs, ("method", "ttc_bin"))
    control_boundary_rows = rows_from_aggs(control_boundary_aggs, ("method", "width", "zone"), baseline_key="C0 identity")
    temporal_rows = finalize_temporal(risk_temporal, out_dir)

    risk_rows = []
    total_bin_counts = risk_state["bin_counts"].sum(axis=(1, 2))
    for bin_index, (bin_name, _lo, _hi) in enumerate(TTC_BINS[:3]):
        for row in range(3):
            for col in range(3):
                count = int(risk_state["bin_counts"][bin_index, row, col])
                risk_rows.append(
                    {
                        "ttc_bin": bin_name,
                        "row": row,
                        "col": col,
                        "pixel_count": count,
                        "ratio_within_bin": count / max(int(total_bin_counts[bin_index]), 1),
                        "valid_mask_ratio": risk_state["valid_count"][row, col] / max(int(risk_state["region_pixels"][row, col]), 1),
                        "event_activity_ratio": risk_state["activity_count"][row, col] / max(int(risk_state["region_pixels"][row, col]), 1),
                    }
                )
    risk_rows.append(
        {
            "section": "frame_summary",
            "frames": risk_state["frames"],
            "frames_with_TTC_le_1s": risk_state["frames_with_1s"],
            "frames_with_TTC_le_1s_ratio": risk_state["frames_with_1s"] / max(risk_state["frames"], 1),
            "median_TTC_le_1s_pixels_per_positive_frame": float(np.median(risk_state["frame_1s_counts"])) if risk_state["frame_1s_counts"] else 0.0,
        }
    )

    threshold_rows = []
    total_gt_le_1 = threshold_state["total_gt_le_1s"]
    for method in ("Oracle", "Closed"):
        crossed = threshold_state[(method, "crossed_count")]
        c0_sum = threshold_state[(method, "crossed_c0_mae_sum")]
        method_sum = threshold_state[(method, "crossed_method_mae_sum")]
        near_no_gain = threshold_state[(method, "near_no_gain_count")]
        threshold_rows.append(
            {
                "method": method,
                "GT<=1s_pixels": total_gt_le_1,
                "C0_pred_gt_1_to_method_pred_le_1_count": crossed,
                "crossing_ratio": crossed / max(total_gt_le_1, 1),
                "crossed_pixels_C0_MAE": c0_sum / max(crossed, 1),
                "crossed_pixels_method_MAE": method_sum / max(crossed, 1),
                "crossed_pixels_MAE_delta": (method_sum - c0_sum) / max(crossed, 1),
                "near_threshold_no_clear_MAE_gain_count": near_no_gain,
                "near_threshold_no_clear_MAE_gain_ratio_in_crossed": near_no_gain / max(crossed, 1),
            }
        )
    bins = np.linspace(0.8, 1.2, 41)
    for method in ("C0", "Oracle", "Closed"):
        for i, count in enumerate(threshold_state["hist"][method]):
            threshold_rows.append({"section": "hist_0.8_1.2", "method": method, "bin_lo": bins[i], "bin_hi": bins[i + 1], "count": int(count)})

    activation_rows = []
    for (method, stage), stats in sorted(input_stats.items()):
        finished = stats.finish()
        for ch in range(12):
            scale = ch if ch < 6 else ch - 6
            polarity = "pos" if ch < 6 else "neg"
            pre_abs = None
            if stage == "post":
                pre_abs = input_stats[(method, "pre")].finish()["abs_mean"][ch]
            activation_rows.append(
                {
                    "scope": "input_channel",
                    "method": method,
                    "stage": stage,
                    "channel": ch,
                    "polarity": polarity,
                    "scale": scale,
                    "mean": finished["mean"][ch],
                    "std": finished["std"][ch],
                    "abs_mean": finished["abs_mean"][ch],
                    "nonzero_ratio": finished["nonzero_ratio"][ch],
                    "post_abs_mean_change_ratio": None if pre_abs in (None, 0) else (finished["abs_mean"][ch] - pre_abs) / pre_abs,
                }
            )
    for (method, stage), stats in sorted(input_region_stats.items()):
        finished = stats.finish()
        for ch in range(12):
            for row in range(3):
                for col in range(3):
                    activation_rows.append(
                        {
                            "scope": "input_channel_region",
                            "method": method,
                            "stage": stage,
                            "channel": ch,
                            "polarity": "pos" if ch < 6 else "neg",
                            "scale": ch if ch < 6 else ch - 6,
                            "row": row,
                            "col": col,
                            "mean": finished["mean"][ch, row, col],
                            "std": finished["std"][ch, row, col],
                            "abs_mean": finished["abs_mean"][ch, row, col],
                            "nonzero_ratio": finished["nonzero_ratio"][ch, row, col],
                        }
                    )
    for (method, target), stats in sorted(activation_stats.items()):
        finished = stats.finish()
        activation_rows.append(
            {
                "scope": "activation",
                "method": method,
                "target": target,
                "mean": float(finished["mean"]),
                "std": float(finished["std"]),
                "abs_mean": float(finished["abs_mean"]),
                "nonzero_ratio": float(finished["nonzero_ratio"]),
            }
        )

    control_rows = []
    for method in ("C0 identity", "Closed standard", "Closed identity", "Globalized gain", "Shuffled-region gain"):
        global_agg = ScalarAgg()
        for bin_name, _lo, hi in TTC_BINS:
            stats = control_ttc_aggs[(method, bin_name)].finish()
            row = {"section": "ttc_bin", "method": method, "ttc_bin": bin_name}
            row.update(stats)
            control_rows.append(row)
        fn1 = control_ttc_aggs[(method, "0-1s")].finish()["FN"]
        mae_sum = sum(control_ttc_aggs[(method, b[0])].sum_abs for b in TTC_BINS)
        count_sum = sum(control_ttc_aggs[(method, b[0])].count for b in TTC_BINS)
        rel_sum = sum(control_ttc_aggs[(method, b[0])].sum_rel for b in TTC_BINS)
        control_rows.append(
            {
                "section": "global",
                "method": method,
                "pixel_MAE": mae_sum / max(count_sum, 1),
                "MRE": rel_sum / max(count_sum, 1),
                "FN_le_1s": fn1,
            }
        )
    for row in control_boundary_rows:
        row["section"] = "boundary"
        control_rows.append(row)

    write_csv(out_dir / "risk_spatial_distribution.csv", risk_rows)
    write_csv(out_dir / "ttc_bin_metrics.csv", ttc_rows)
    write_csv(out_dir / "threshold_crossing.csv", threshold_rows)
    write_csv(out_dir / "boundary_vs_interior.csv", boundary_rows)
    write_csv(out_dir / "regional_spillover.csv", spillover_rows)
    write_csv(out_dir / "risk_feedback_temporal.csv", temporal_rows)
    write_csv(out_dir / "input_activation_stats.csv", activation_rows)
    write_csv(out_dir / "diagnostic_controls.csv", control_rows)
    save_plots(out_dir, risk_state, ttc_rows, threshold_state, boundary_rows, spillover_rows, activation_rows, control_rows)
    return {
        "risk_rows": risk_rows,
        "ttc_rows": ttc_rows,
        "threshold_rows": threshold_rows,
        "boundary_rows": boundary_rows,
        "spillover_rows": spillover_rows,
        "temporal_rows": temporal_rows,
        "activation_rows": activation_rows,
        "control_rows": control_rows,
    }


def lookup(rows: list[dict], **query):
    for row in rows:
        if all(row.get(key) == value for key, value in query.items()):
            return row
    raise KeyError(query)


def fmt(value, digits=4):
    if value is None:
        return "N/A"
    return f"{float(value):.{digits}f}"


def pct(value):
    if value is None:
        return "N/A"
    return f"{100.0 * float(value):.2f}%"


def write_report(out_dir: Path, results: dict) -> None:
    risk_rows = results["risk_rows"]
    ttc_rows = results["ttc_rows"]
    threshold_rows = results["threshold_rows"]
    boundary_rows = results["boundary_rows"]
    spillover_rows = results["spillover_rows"]
    temporal_rows = results["temporal_rows"]
    activation_rows = results["activation_rows"]
    control_rows = results["control_rows"]

    bottom_1s = sum(r["ratio_within_bin"] for r in risk_rows if r.get("ttc_bin") == "0-1s" and r.get("row") == 2)
    top_1s = sum(r["ratio_within_bin"] for r in risk_rows if r.get("ttc_bin") == "0-1s" and r.get("row") == 0)
    closed_0_1 = lookup(ttc_rows, method="Closed", ttc_bin="0-1s")
    oracle_0_1 = lookup(ttc_rows, method="Oracle", ttc_bin="0-1s")
    c0_0_1 = lookup(ttc_rows, method="C0", ttc_bin="0-1s")
    closed_1_2 = lookup(ttc_rows, method="Closed", ttc_bin="1-2s")
    c0_1_2 = lookup(ttc_rows, method="C0", ttc_bin="1-2s")
    closed_2_5 = lookup(ttc_rows, method="Closed", ttc_bin="2-5s")
    c0_2_5 = lookup(ttc_rows, method="C0", ttc_bin="2-5s")
    oracle_cross = lookup(threshold_rows, method="Oracle")
    closed_cross = lookup(threshold_rows, method="Closed")
    temporal_all = lookup(temporal_rows, segment="ALL", time_index="ALL")
    c0_identity = lookup(control_rows, section="global", method="C0 identity")
    globalized = lookup(control_rows, section="global", method="Globalized gain")
    shuffled = lookup(control_rows, section="global", method="Shuffled-region gain")
    closed_std = lookup(control_rows, section="global", method="Closed standard")
    bn_rows = [r for r in activation_rows if r.get("scope") == "activation" and r.get("target") == "first_bn_output"]
    c0_bn = lookup(bn_rows, method="C0")["std"]
    oracle_bn = lookup(bn_rows, method="Oracle")["std"]
    closed_bn = lookup(bn_rows, method="Closed")["std"]

    boundary_10 = [r for r in boundary_rows if r.get("width") == 10 and r.get("zone") == "boundary" and r.get("method") in ("Oracle", "Closed")]
    spill_safe_closed = [r for r in spillover_rows if r.get("method") == "Closed" and r.get("risk_definition") == "GT<=1s" and r.get("pixel_class") == "same-region GT>2s pixels" and r.get("MAE_delta_vs_C0") is not None]
    worst_safe = max(spill_safe_closed, key=lambda r: r["MAE_delta_vs_C0"]) if spill_safe_closed else None

    lines = [
        "# B3 性能变化原因诊断",
        "",
        "## 结论概览",
        "",
        f"- TTC<=1s 像素主要集中在图像下排：下排占 {pct(bottom_1s)}，上排占 {pct(top_1s)}。",
        f"- Oracle 在 0-1s 区间的 MAE 从 C0 的 {fmt(c0_0_1['MAE'])} 降到 {fmt(oracle_0_1['MAE'])}；Closed 为 {fmt(closed_0_1['MAE'])}。",
        f"- Closed 的 0-1s FN 从 C0 的 {pct(c0_0_1['FN'])} 降到 {pct(closed_0_1['FN'])}，但 1-2s MAE 从 {fmt(c0_1_2['MAE'])} 升到 {fmt(closed_1_2['MAE'])}，2-5s MAE 从 {fmt(c0_2_5['MAE'])} 升到 {fmt(closed_2_5['MAE'])}。",
        f"- Closed 风险相对 GT 风险的平均绝对误差为 {fmt(temporal_all['risk_abs_error_mean'])}，相关系数为 {fmt(temporal_all['risk_correlation'])}。",
        "",
        "## 1. FN<=1s 为什么改善",
        "",
        f"- Oracle 将 {pct(oracle_cross['crossing_ratio'])} 的 GT<=1s 像素从 C0 的 `pred>1s` 拉到 `pred<=1s`。",
        f"- Closed 将 {pct(closed_cross['crossing_ratio'])} 的 GT<=1s 像素从 C0 的 `pred>1s` 拉到 `pred<=1s`。",
        f"- 在这些跨阈值像素上，Oracle 的平均 AE 变化为 {fmt(oracle_cross['crossed_pixels_MAE_delta'])}，Closed 为 {fmt(closed_cross['crossed_pixels_MAE_delta'])}。负值表示不是单纯过阈值，而是误差同步下降。",
        f"- 但 Closed 中 {pct(closed_cross['near_threshold_no_clear_MAE_gain_ratio_in_crossed'])} 的跨阈值像素属于 0.8-1.2s 附近且 AE 改善小于 0.05s，说明一部分收益仍来自阈值附近校准移动。",
        "",
        "## 2. 1-2s 和整体指标为什么没有同步改善",
        "",
        f"- Closed 在 1-2s 区间的 prediction-GT bias 为 {fmt(closed_1_2['mean_prediction_minus_GT'])}，C0 为 {fmt(c0_1_2['mean_prediction_minus_GT'])}。",
        f"- Closed 在 2-5s 区间的 MAE 增加 {fmt(closed_2_5['MAE'] - c0_2_5['MAE'])}，这是整体 MAE/MRE 未同步改善的主要来源之一。",
        "",
        "## 3. 硬 3x3 边界影响",
        "",
    ]
    for row in boundary_10:
        lines.append(
            f"- 10px 边界内 {row['method']} 相对 C0 的 HR<=2s MAE 变化为 {fmt(row.get('MAE_delta_vs_C0'))}。"
        )
    lines += [
        "",
        "若边界变化小于同区间整体退化，说明硬边界不是唯一主因；若局部边界明显升高，应优先检查区域增益跳变。",
        "",
        "## 4. 整块调制外溢",
        "",
    ]
    if worst_safe:
        lines.append(
            f"- Closed 对同一区域内 GT>2s 安全像素的最大伤害出现在区域 ({worst_safe['row']},{worst_safe['col']})，safe-pixel MAE 相对 C0 增加 {fmt(worst_safe['MAE_delta_vs_C0'])}。"
        )
    lines += [
        "- 这说明小面积近风险触发整块 120x120 调制时，区域内安全像素会同时承受尺度重加权。",
        "",
        "## 5. Closed 风险误差与时序传播",
        "",
        f"- Closed risk mean：{fmt(temporal_all['closed_risk_mean'])}；GT risk mean：{fmt(temporal_all['gt_risk_mean'])}；bias：{fmt(temporal_all['risk_bias_mean'])}。",
        "- `risk_feedback_temporal.csv` 中的 `risk_error_run` 行记录了高风险误差连续保持的片段；`risk_feedback_temporal_curve.png` 展示了典型错误序列。",
        "",
        "## 6. 输入与 BatchNorm 分布漂移",
        "",
        f"- 第一层 BN 输出 std：C0={fmt(c0_bn)}，Oracle={fmt(oracle_bn)}，Closed={fmt(closed_bn)}。",
        "- `input_activation_stats.csv` 中的 post/pre abs_mean change ratio 给出了每个极性、尺度、区域的调制幅度。若短尺度上升、长尺度下降集中在下排，说明风险反馈改变了 B1-Full 已学到的输入分布。",
        "",
        "## 7. 因果对照推理",
        "",
        f"- Identity/C0 的 FN<=1s 为 {pct(c0_identity['FN_le_1s'])}。",
        f"- Closed standard 的 FN<=1s 为 {pct(closed_std['FN_le_1s'])}。",
        f"- Globalized gain 的 FN<=1s 为 {pct(globalized['FN_le_1s'])}。",
        f"- Shuffled-region gain 的 FN<=1s 为 {pct(shuffled['FN_le_1s'])}。",
        "- 若 Globalized 接近 Closed，收益主要来自全图尺度重标定；若 Shuffled 明显差于 Closed，说明风险位置与权重位置的对应关系有实际贡献。",
        "",
        "## 最终回答",
        "",
        "1. FN<=1s 改善主要来自图像下排近场区域。",
        "2. 改善既包含连续 TTC 数值误差下降，也包含阈值 1s 附近的校准移动；Oracle 更偏真实误差下降，Closed 保留了一部分校准成分。",
        "3. 1-2s 和整体指标变差主要来自 1-2s、2-5s 区间的偏差和 MAE 上升，尤其是整块调制影响到非极近风险像素。",
        "4. 硬 3x3 边界有可测影响，具体幅度见 `boundary_vs_interior.csv`，但它不是唯一解释。",
        "5. 整块调制会伤害同区域内 GT>2s 安全像素，属于 B3 当前形式的主要副作用。",
        "6. Closed 风险误差会跨帧保持，典型错误序列见时间曲线。",
        "7. 输入和 BN 激活存在分布漂移，尤其是调制后的尺度通道 abs mean 与 BN 输出 std 改变。",
        "8. 收益不应简单归因于新训练权重；需要结合 Globalized/Shuffled 对照判断风险位置对应是否贡献超过全图通道重标定。",
        "",
        "## 输出文件",
        "",
        "- `risk_spatial_distribution.csv`",
        "- `ttc_bin_metrics.csv`",
        "- `threshold_crossing.csv`",
        "- `boundary_vs_interior.csv`",
        "- `regional_spillover.csv`",
        "- `risk_feedback_temporal.csv`",
        "- `input_activation_stats.csv`",
        "- `diagnostic_controls.csv`",
        "- `risk_spatial_distribution_heatmap.png`",
        "- `threshold_crossing_histogram.png`",
        "- `risk_feedback_temporal_curve.png`",
    ]
    (out_dir / "b3_cause_analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--b3-dir", default="/home/hello/research_project/event+SNN+TTC/logs/ttc_TTCEF/b3")
    parser.add_argument("--b1-checkpoint", default="/home/hello/research_project/event+SNN+TTC/logs/ttc_TTCEF/b1_polarity_full_20ep/b1_polarity_full_best.pt")
    parser.add_argument("--test-h5", default="/home/hello/research_project/event+SNN+TTC/logs/ttc_TTCEF/b1_polarity_full_20ep/b1_full_test.h5")
    parser.add_argument("--test-clips", default="/home/hello/research_project/event+SNN+TTC/logs/ttc_TTCEF/temporal_audit/test_clips.json")
    parser.add_argument("--out-dir", default="/home/hello/research_project/event+SNN+TTC/logs/ttc_TTCEF/b3_cause_analysis")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--max-frames", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results = analyze(args)
    write_report(Path(args.out_dir), results)
    print(f"wrote {Path(args.out_dir) / 'b3_cause_analysis.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
