#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import h5py
import hdf5plugin  # noqa: F401
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model"))

from evslim import EVSlim  # noqa: E402


DEFAULT_MODEL_CFG = {
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


PAPER_TF2TTC_REFERENCE = {
    "case_1_mixed_indistribution": {
        "description": "论文 Table III Case 1，mixed day/night quadruped+car，TF2TTC",
        "mean_frame_median_AE": "1.87 ± 2.71",
        "MRE_percent": 44.07,
        "RE_10_50_80_percent": "77.09 / 24.20 / 11.76",
        "MAE_B1_B3": "4.81 / 5.24 / 2.31",
    },
    "case_2_car_day_to_car_night": {
        "description": "论文 Table III Case 2，car day -> car night，TF2TTC",
        "mean_frame_median_AE": "1.27 ± 0.85",
        "MRE_percent": 55.53,
        "RE_10_50_80_percent": "80.19 / 30.34 / 16.09",
        "MAE_B1_B3": "1.59 / 2.47 / 2.03",
    },
    "case_3_quadruped_day_to_night_indoor": {
        "description": "论文 Table III Case 3，quadruped day/stairs -> night+indoors，TF2TTC",
        "mean_frame_median_AE": "3.37 ± 2.48",
        "MRE_percent": 106.02,
        "RE_10_50_80_percent": "87.82 / 50.92 / 36.86",
        "MAE_B1_B3": "6.18 / 6.24 / 4.00",
    },
}


class TTCEFH5(Dataset):
    def __init__(self, path: str | Path):
        self.path = str(path)
        self._h5 = None
        with h5py.File(self.path, "r") as handle:
            self.length = int(handle["exp_filts"].shape[0])
            self.height = int(handle["ttc"].shape[1])
            self.width = int(handle["ttc"].shape[2])

    def __len__(self):
        return self.length

    def _open(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.path, "r", libver="latest")
        return self._h5

    def __getitem__(self, index):
        handle = self._open()
        exp = torch.from_numpy(handle["exp_filts"][index].astype(np.float32))
        ttc = torch.from_numpy(handle["ttc"][index].astype(np.float32))
        mask = torch.from_numpy(handle["mask"][index].astype(np.bool_))
        return exp, ttc, mask


@dataclass
class MetricAccumulator:
    name: str
    sample_count: int = 0
    valid_pixel_count: int = 0
    sum_abs: float = 0.0
    sum_rel: float = 0.0
    re_10: int = 0
    re_50: int = 0
    re_80: int = 0
    frame_median_ae: list[float] = field(default_factory=list)
    band_sum_abs: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    band_count: list[int] = field(default_factory=lambda: [0, 0, 0])
    gt_min: float | None = None
    gt_max: float | None = None
    pred_min: float | None = None
    pred_max: float | None = None

    def update(self, gt, pred, mask, bands, rel_eps):
        self.sample_count += 1
        valid = mask & np.isfinite(gt) & np.isfinite(pred)
        if not valid.any():
            return

        gt_valid = gt[valid]
        pred_valid = pred[valid]
        abs_values = np.abs(pred_valid - gt_valid).astype(np.float32)
        rel_values = abs_values / np.maximum(np.abs(gt_valid), rel_eps)

        self.valid_pixel_count += int(abs_values.size)
        self.sum_abs += float(abs_values.sum(dtype=np.float64))
        self.sum_rel += float(rel_values.sum(dtype=np.float64))
        self.re_10 += int((rel_values > 0.10).sum())
        self.re_50 += int((rel_values > 0.50).sum())
        self.re_80 += int((rel_values > 0.80).sum())
        self.frame_median_ae.append(float(np.median(abs_values)))

        for band_index, (row0, row1) in enumerate(bands):
            band_valid = valid[row0:row1]
            if not band_valid.any():
                continue
            band_abs = np.abs(pred[row0:row1][band_valid] - gt[row0:row1][band_valid])
            self.band_sum_abs[band_index] += float(band_abs.sum(dtype=np.float64))
            self.band_count[band_index] += int(band_abs.size)

        self.gt_min = float(gt_valid.min()) if self.gt_min is None else min(self.gt_min, float(gt_valid.min()))
        self.gt_max = float(gt_valid.max()) if self.gt_max is None else max(self.gt_max, float(gt_valid.max()))
        self.pred_min = float(pred_valid.min()) if self.pred_min is None else min(self.pred_min, float(pred_valid.min()))
        self.pred_max = float(pred_valid.max()) if self.pred_max is None else max(self.pred_max, float(pred_valid.max()))

    def finalize(self):
        frame = np.asarray(self.frame_median_ae, dtype=np.float64)
        denom = max(self.valid_pixel_count, 1)
        return {
            "name": self.name,
            "sample_count": self.sample_count,
            "valid_pixel_count": self.valid_pixel_count,
            "mean_frame_median_AE": float(frame.mean()) if frame.size else None,
            "std_frame_median_AE": float(frame.std()) if frame.size else None,
            "MRE_percent": 100.0 * self.sum_rel / denom,
            "RE_10_percent": 100.0 * self.re_10 / denom,
            "RE_50_percent": 100.0 * self.re_50 / denom,
            "RE_80_percent": 100.0 * self.re_80 / denom,
            "pixel_MAE": self.sum_abs / denom,
            "MAE_B1_B3": [
                self.band_sum_abs[i] / max(self.band_count[i], 1)
                for i in range(3)
            ],
            "band_valid_pixel_count": self.band_count,
            "gt_range": [self.gt_min, self.gt_max],
            "prediction_range": [self.pred_min, self.pred_max],
        }


def namespace_from_default():
    return SimpleNamespace(**DEFAULT_MODEL_CFG)


def load_evslim(checkpoint: str | Path, device):
    # The checkpoint is locally generated and trusted. weights_only=False is needed
    # because Lightning stores an OmegaConf object in hyper_parameters.
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    evslim_state = {
        key.replace("model.", "", 1): value
        for key, value in state.items()
        if key.startswith("model.")
    }
    if not evslim_state:
        evslim_state = state
    model = EVSlim(namespace_from_default())
    model.load_state_dict(evslim_state, strict=True)
    return model.to(device).eval(), {
        "epoch": ckpt.get("epoch"),
        "global_step": ckpt.get("global_step"),
    }


def band_ranges(height):
    if height == 260:
        return [(0, 87), (87, 174), (174, 260)]
    one_third = height // 3
    two_third = (2 * height) // 3
    return [(0, one_third), (one_third, two_third), (two_third, height)]


def sequence_ranges(h5_path: str | Path):
    ranges = []
    with h5py.File(h5_path, "r") as handle:
        if "file_names" not in handle or "indices" not in handle:
            return ranges
        names = [
            value.decode("utf-8") if isinstance(value, bytes) else str(value)
            for value in handle["file_names"][:]
        ]
        ends = [int(row[0]) for row in handle["indices"][:]]
        start = 0
        for name, end in zip(names, ends):
            if end > start:
                ranges.append({"name": name, "start": start, "end": end, "sample_count": end - start})
            start = end
    return ranges


def accumulator_for_index(index, seq_accumulators, ranges):
    for item in ranges:
        if item["start"] <= index < item["end"]:
            return seq_accumulators[item["name"]]
    return None


@torch.no_grad()
def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    dataset = TTCEFH5(args.test_file)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )
    model, ckpt_info = load_evslim(args.checkpoint, device)
    bands = band_ranges(dataset.height)
    ranges = sequence_ranges(args.test_file)
    overall = MetricAccumulator("overall")
    seq_accumulators = {item["name"]: MetricAccumulator(item["name"]) for item in ranges}

    iterable = enumerate(loader)
    if tqdm is not None:
        iterable = tqdm(iterable, total=math.ceil(len(dataset) / args.batch_size), desc="Evaluating")

    sample_offset = 0
    for _, (exp, gt, mask) in iterable:
        exp = exp.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            pred = model(exp).float().cpu().numpy()[:, 0]
        gt = gt.numpy()
        mask = mask.numpy().astype(bool)

        batch_size = gt.shape[0]
        for batch_index in range(batch_size):
            global_index = sample_offset + batch_index
            overall.update(gt[batch_index], pred[batch_index], mask[batch_index], bands, args.rel_eps)
            seq_acc = accumulator_for_index(global_index, seq_accumulators, ranges)
            if seq_acc is not None:
                seq_acc.update(gt[batch_index], pred[batch_index], mask[batch_index], bands, args.rel_eps)
        sample_offset += batch_size

    return {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_info": ckpt_info,
        "test_file": str(Path(args.test_file).resolve()),
        "num_test_samples": len(dataset),
        "image_height": dataset.height,
        "image_width": dataset.width,
        "band_ranges": bands,
        "sequence_ranges": ranges,
        "paper_reference_TF2TTC": PAPER_TF2TTC_REFERENCE,
        "metrics": {
            "overall": overall.finalize(),
            "per_sequence": {name: acc.finalize() for name, acc in seq_accumulators.items()},
        },
    }


def fmt(value, digits=3):
    if value is None:
        return "N/A"
    return f"{value:.{digits}f}"


def fmt_triplet(values, digits=3):
    return " / ".join(fmt(value, digits) for value in values)


def metric_table(metrics_by_name):
    lines = [
        "| 数据范围 | 样本数 | 有效像素 | Mean frame median AE | MRE(%) | RE[10/50/80](%) | MAE(B1-B3) | Pixel MAE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, metric in metrics_by_name:
        re = [metric["RE_10_percent"], metric["RE_50_percent"], metric["RE_80_percent"]]
        lines.append(
            f"| {name} | {metric['sample_count']} | {metric['valid_pixel_count']} | "
            f"{fmt(metric['mean_frame_median_AE'])} ± {fmt(metric['std_frame_median_AE'])} | "
            f"{fmt(metric['MRE_percent'])} | {fmt_triplet(re)} | "
            f"{fmt_triplet(metric['MAE_B1_B3'])} | {fmt(metric['pixel_MAE'])} |"
        )
    return lines


def write_markdown(payload, out_path):
    overall = payload["metrics"]["overall"]
    per_sequence = payload["metrics"]["per_sequence"]
    table_rows = [("总体测试集", overall)] + list(per_sequence.items())
    lines = [
        "# 原论文 EV-Slim 论文指标评估",
        "",
        "## 数据与模型",
        f"- checkpoint：`{payload['checkpoint']}`",
        f"- checkpoint epoch：`{payload['checkpoint_info'].get('epoch')}`",
        f"- checkpoint global_step：`{payload['checkpoint_info'].get('global_step')}`",
        f"- test.h5：`{payload['test_file']}`",
        f"- 测试样本数：`{payload['num_test_samples']}`",
        f"- 图像尺寸：`{payload['image_height']} x {payload['image_width']}`",
        f"- band 划分：`{payload['band_ranges']}`",
        "",
        "当前测试集包含：",
    ]
    for item in payload["sequence_ranges"]:
        lines.append(f"- `{item['name']}`：样本 `{item['sample_count']}`，H5 rows [{item['start']}, {item['end']})")

    lines += [
        "",
        "## 论文同口径指标",
        "",
        *metric_table(table_rows),
        "",
        "## 与论文结果的关系",
        "- 论文 Table III 不报告 train loss，因此当前训练 loss 只能用于判断训练过程是否稳定，不能直接与论文主结果表比较。",
        "- 本次使用的是当前已准备好的 7 个训练序列 + 3 个测试序列；官方 `merge.py` 中完整划分是 21 个训练序列 + 11 个测试序列，所以这里是部分复现，不是完整 Table III 复现。",
        "- 当前 H=360，因此 MAE(B1-B3) 使用高度三等分；论文文字中 T²CEF 的 band 是基于 260 高度的 `(0,87)/(87,174)/(174,260)`，两者不完全等价。",
        "",
        "论文 Table III 中 TF2TTC 参考值：",
        "",
        "| 论文设置 | Mean frame median AE | MRE(%) | RE[10/50/80](%) | MAE(B1-B3) |",
        "|---|---:|---:|---:|---:|",
    ]
    for ref in PAPER_TF2TTC_REFERENCE.values():
        lines.append(
            f"| {ref['description']} | {ref['mean_frame_median_AE']} | "
            f"{ref['MRE_percent']:.2f} | {ref['RE_10_50_80_percent']} | {ref['MAE_B1_B3']} |"
        )

    lines += [
        "",
        "## 初步判断",
    ]
    mean_ae = overall["mean_frame_median_AE"]
    mre = overall["MRE_percent"]
    if mean_ae is not None:
        if mean_ae <= 2.5:
            lines.append("- Mean frame median AE 与论文 TF2TTC 的 Case 1/2 量级接近。")
        elif mean_ae <= 4.0:
            lines.append("- Mean frame median AE 落在论文 Case 1 到 Case 3 的大致范围内。")
        else:
            lines.append("- Mean frame median AE 高于论文 TF2TTC 主要结果，需要进一步检查数据划分、训练集规模或评估裁剪。")
    if mre is not None:
        if mre <= 70:
            lines.append("- MRE 与论文 TF2TTC 的 Case 1/2 量级接近。")
        else:
            lines.append("- MRE 偏高；相对误差容易受小 TTC / 接近零 TTC 像素影响，建议后续按官方完整 test split 和 crop 设置复算。")
    lines.append("- 更严格复现需要补齐官方 21 train / 11 test 序列，并确认是否按论文 T²CEF-med 的 260 高度 crop 评估。")
    out_path.write_text("\n".join(lines) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate original EV-Slim using EV-TTC paper-style metrics.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--test-file", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--rel-eps", type=float, default=1e-6)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = evaluate(args)
    json_path = out_dir / "paper_metrics.json"
    md_path = out_dir / "paper_metrics.md"
    report_path = out_dir / "paper_comparison_report.md"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    write_markdown(payload, md_path)
    write_markdown(payload, report_path)
    print(json.dumps(payload["metrics"]["overall"], indent=2, ensure_ascii=False))
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
