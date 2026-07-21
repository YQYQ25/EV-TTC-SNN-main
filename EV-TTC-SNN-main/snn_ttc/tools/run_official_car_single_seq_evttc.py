#!/usr/bin/env python3
"""car 单序列官方 EV-TTC 链路复现。

本脚本只针对 `car_urban_night_rittenhouse` 做单序列 debug 复现：
官方 IIR + 官方 TTC 秒值 + 官方 mask + 官方 EV-Slim。由于指导文件要求
记录更多指标和 20/40 epoch 对照，这里没有直接调用 Lightning `train.py`，
而是复用同一网络、loss、优化器、增强与数据口径，实现一个可审计的训练循环。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import h5py
import hdf5plugin  # noqa: F401  # 注册 Blosc2/LZ4 HDF5 插件
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402
from torchvision.transforms import v2  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[3]
EVTTC_MODEL_ROOT = PROJECT_ROOT / "EV-TTC-main" / "model"
SNN_ROOT = PROJECT_ROOT / "EV-TTC-SNN-main"
if str(EVTTC_MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(EVTTC_MODEL_ROOT))

from evslim import EVSlim  # noqa: E402


SEQ_NAME = "car_urban_night_rittenhouse"
DEFAULT_OFFICIAL_ROOT = Path("/data/evttc_storage/event+SNN+TTC/data_m3ed_official_7train_3test")
DEFAULT_OUT_DIR = SNN_ROOT / "debug_sets" / "[13]official_car_single_sequence_evttc_reproduction"
CAR_SPEED_THRESHOLD = 1.3
OMEGA_THRESHOLD = 0.18
OFFICIAL_ALPHAS = [0.12, 0.06, 0.03, 0.015, 0.0095, 0.0045]


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
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


def fmt(value: Any, digits: int = 4) -> str:
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not math.isfinite(value_f):
        return "NA"
    return f"{value_f:.{digits}f}"


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(str(v) for v in row) + " |" for row in rows)
    return "\n".join(lines)


def compression_kwargs() -> dict[str, Any]:
    return hdf5plugin.Blosc2(cname="lz4", clevel=5, filters=hdf5plugin.Blosc.SHUFFLE)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def create_evslim_cfg() -> SimpleNamespace:
    """官方 EV-Slim TTC YAML 的最小等价配置。"""

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


def official_charbonnier_loss(error: torch.Tensor, mask: torch.Tensor, *, alpha: float = 0.45) -> torch.Tensor:
    """复刻 `EV-TTC-main/model/util.py` 的 masked Charbonnier。"""

    valid = mask.bool() & torch.isfinite(error)
    safe_error = torch.where(valid, error, torch.zeros_like(error))
    charbonnier = (safe_error**2.0 + 1e-5**2.0) ** alpha
    valid_float = valid.float()
    denom = torch.sum(valid_float, dim=(1, 2, 3))
    per_sample = torch.sum(charbonnier * valid_float, dim=(1, 2, 3)) / denom.clamp_min(1.0)
    keep = denom > 0
    return per_sample[keep].mean() if torch.any(keep) else safe_error.sum() * 0.0


def official_sequence_paths(official_root: Path) -> dict[str, Path]:
    return {
        "data": official_root / "events" / "m3ed" / SEQ_NAME / f"{SEQ_NAME}_data.h5",
        "exp": official_root / "exp_filts" / "m3ed" / f"{SEQ_NAME}.h5",
        "gt": official_root / "ttcef" / "m3ed" / f"{SEQ_NAME}.h5",
        "pose": official_root / "depth" / "m3ed" / SEQ_NAME / f"{SEQ_NAME}_pose_gt.h5",
        "depth": official_root / "depth" / "m3ed" / SEQ_NAME / f"{SEQ_NAME}_depth_gt.h5",
    }


def create_merged_h5(path: Path, sample_count: int, include_flow: bool) -> h5py.File:
    """创建与官方 DataModule 兼容的单序列 split H5。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    f = h5py.File(path, "w")
    f.attrs["complete"] = False
    f.attrs["sequence_name"] = SEQ_NAME
    f.create_dataset(
        "exp_filts",
        shape=(sample_count, 6, 360, 360),
        dtype=np.float16,
        chunks=(1, 6, 360, 360),
        **compression_kwargs(),
    )
    f.create_dataset(
        "ttc",
        shape=(sample_count, 360, 360),
        dtype=np.float32,
        chunks=(1, 360, 360),
        **compression_kwargs(),
    )
    f.create_dataset(
        "mask",
        shape=(sample_count, 360, 360),
        dtype=bool,
        chunks=(1, 360, 360),
        **compression_kwargs(),
    )
    if include_flow:
        f.create_dataset(
            "flow",
            shape=(sample_count, 2, 360, 360),
            dtype=np.float16,
            chunks=(1, 2, 360, 360),
            **compression_kwargs(),
        )
    for key, dtype in [
        ("source_index", np.int64),
        ("exp_time", np.float32),
        ("ts", np.float32),
        ("te", np.float32),
        ("valid_pixel_count", np.int64),
    ]:
        f.create_dataset(key, shape=(sample_count,), dtype=dtype, chunks=(min(2048, max(1, sample_count)),))
    f.create_dataset("T", shape=(sample_count, 3), dtype=np.float32, chunks=(min(512, max(1, sample_count)), 3))
    f.create_dataset("Omega", shape=(sample_count, 3), dtype=np.float32, chunks=(min(512, max(1, sample_count)), 3))
    return f


def contiguous_read(dataset: h5py.Dataset, indices: np.ndarray) -> np.ndarray:
    """HDF5 对连续切片更友好；这里读小段连续范围后再按 offset 取有效帧。"""

    start = int(indices[0])
    end = int(indices[-1]) + 1
    arr = dataset[start:end]
    return arr[indices - start]


def copy_split(
    split_name: str,
    indices: np.ndarray,
    output_path: Path,
    f_exp: h5py.File,
    f_gt: h5py.File,
    *,
    chunk_size: int,
    include_flow: bool,
) -> dict[str, Any]:
    if output_path.exists():
        output_path.unlink()
    tmp_path = output_path.with_suffix(".tmp.h5")
    if tmp_path.exists():
        tmp_path.unlink()
    f_out = create_merged_h5(tmp_path, len(indices), include_flow)
    start_time = time.time()
    valid_pixel_total = 0
    sample_with_pixels = 0
    try:
        out_pos = 0
        for chunk_start in range(0, len(indices), chunk_size):
            chunk_idx = indices[chunk_start : chunk_start + chunk_size]
            exp = contiguous_read(f_exp["exp_filts"], chunk_idx).astype(np.float32)
            ttc = contiguous_read(f_gt["ttc"], chunk_idx).astype(np.float32)
            raw_mask = contiguous_read(f_gt["mask"], chunk_idx).astype(bool)
            finite = np.isfinite(ttc)
            ttc_clean = np.nan_to_num(ttc)
            official_mask = raw_mask & (np.abs(exp[:, -1]) > 1e-3) & finite & (ttc_clean < 100)
            pix = official_mask.reshape(official_mask.shape[0], -1).sum(axis=1).astype(np.int64)
            n = len(chunk_idx)
            f_out["exp_filts"][out_pos : out_pos + n] = exp.astype(np.float16)
            f_out["ttc"][out_pos : out_pos + n] = ttc_clean.astype(np.float32)
            f_out["mask"][out_pos : out_pos + n] = official_mask
            if include_flow:
                flow = contiguous_read(f_gt["flow"], chunk_idx).astype(np.float16)
                f_out["flow"][out_pos : out_pos + n] = flow
            f_out["source_index"][out_pos : out_pos + n] = chunk_idx
            f_out["exp_time"][out_pos : out_pos + n] = f_exp["exp_times"][chunk_idx]
            f_out["ts"][out_pos : out_pos + n] = f_gt["ts"][chunk_idx]
            f_out["te"][out_pos : out_pos + n] = f_gt["te"][chunk_idx]
            f_out["valid_pixel_count"][out_pos : out_pos + n] = pix
            f_out["T"][out_pos : out_pos + n] = f_gt["T"][chunk_idx]
            f_out["Omega"][out_pos : out_pos + n] = f_gt["Omega"][chunk_idx]
            valid_pixel_total += int(pix.sum())
            sample_with_pixels += int(np.count_nonzero(pix > 0))
            out_pos += n
            if chunk_start % max(chunk_size * 10, 1) == 0:
                log(f"{split_name} merge {out_pos}/{len(indices)}")
    finally:
        f_out.attrs["complete"] = True
        f_out.close()
    tmp_path.rename(output_path)
    return {
        "split": split_name,
        "path": str(output_path),
        "samples": int(len(indices)),
        "sample_with_pixels": int(sample_with_pixels),
        "valid_pixel_total": int(valid_pixel_total),
        "avg_valid_pixels": float(valid_pixel_total / max(1, sample_with_pixels)),
        "seconds": float(time.time() - start_time),
    }


def prepare_data(args: argparse.Namespace) -> dict[str, Any]:
    """检查官方 create_exp/calc_gt 输出，并按官方 merge mask 生成单序列 split。"""

    out_dir = Path(args.out_dir)
    data_dir = out_dir / "data"
    meta_path = data_dir / "official_car_single_seq_data_summary.json"
    train_path = data_dir / "train.h5"
    val_path = data_dir / "val.h5"
    if meta_path.exists() and train_path.exists() and val_path.exists() and not args.force_prepare:
        summary = read_json(meta_path)
        if summary.get("complete"):
            return summary

    paths = official_sequence_paths(Path(args.official_root))
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError("官方链路输出缺失：" + "; ".join(missing))

    data_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(paths["exp"], "r") as f_exp, h5py.File(paths["gt"], "r") as f_gt:
        exp_times = f_exp["exp_times"][:].astype(np.float32)
        te = f_gt["te"][:].astype(np.float32)
        ts = f_gt["ts"][:].astype(np.float32)
        if exp_times.shape != te.shape or not np.array_equal(exp_times, te):
            raise RuntimeError("IIR exp_times 与 GT te 不完全一致，停止训练")
        valid = f_gt["valid"][:].astype(bool)
        t_norm = np.linalg.norm(f_gt["T"][:], axis=1)
        omega_norm = np.linalg.norm(f_gt["Omega"][:], axis=1)
        official_sample_mask = valid & (t_norm > CAR_SPEED_THRESHOLD) & (omega_norm < OMEGA_THRESHOLD)
        official_indices = np.where(official_sample_mask)[0].astype(np.int64)
        # 单序列 debug split：先保持官方过滤后的时间顺序，再按 80/5/15 划分。
        n = len(official_indices)
        train_n = int(n * 0.80)
        buffer_n = int(n * 0.05)
        train_idx = official_indices[:train_n]
        buffer_idx = official_indices[train_n : train_n + buffer_n]
        val_idx = official_indices[train_n + buffer_n :]
        split_summary = {
            "total_exp_frames": int(len(exp_times)),
            "official_valid_frames": int(valid.sum()),
            "official_motion_filtered_samples": int(n),
            "train_samples": int(len(train_idx)),
            "buffer_samples": int(len(buffer_idx)),
            "val_samples": int(len(val_idx)),
            "exp_te_exact_equal": True,
            "dt_us_mean": float(np.mean(te - ts)),
            "dt_us_min": float(np.min(te - ts)),
            "dt_us_max": float(np.max(te - ts)),
            "train_time_range": [float(exp_times[train_idx[0]]), float(exp_times[train_idx[-1]])] if len(train_idx) else [],
            "buffer_time_range": [float(exp_times[buffer_idx[0]]), float(exp_times[buffer_idx[-1]])] if len(buffer_idx) else [],
            "val_time_range": [float(exp_times[val_idx[0]]), float(exp_times[val_idx[-1]])] if len(val_idx) else [],
        }
        write_json(data_dir / "split_indices.json", {
            "train_indices": train_idx.tolist(),
            "buffer_indices": buffer_idx.tolist(),
            "val_indices": val_idx.tolist(),
        })
        stats = []
        stats.append(copy_split("train", train_idx, train_path, f_exp, f_gt, chunk_size=args.prepare_chunk_size, include_flow=args.include_flow))
        stats.append(copy_split("validation", val_idx, val_path, f_exp, f_gt, chunk_size=args.prepare_chunk_size, include_flow=args.include_flow))

    summary = {
        "complete": True,
        "sequence_name": SEQ_NAME,
        "official_root": str(args.official_root),
        "paths": {k: str(v) for k, v in paths.items()},
        "train_h5": str(train_path),
        "val_h5": str(val_path),
        "include_flow_in_split_h5": bool(args.include_flow),
        "official_create_exp": {
            "script": str(PROJECT_ROOT / "EV-TTC-main" / "TTCEF" / "create_exp.py"),
            "alphas": OFFICIAL_ALPHAS,
            "time_bin_ms": 0.2,
            "output_interval_ms": 7,
            "exp_file_exists": paths["exp"].exists(),
        },
        "official_calc_gt": {
            "script": str(PROJECT_ROOT / "EV-TTC-main" / "TTCEF" / "calc_gt.py"),
            "gt_file_exists": paths["gt"].exists(),
            "target": "TTC_seconds_Z_over_Tz",
        },
        "official_merge_logic": {
            "sample_filter": "valid & ||T||>1.3 & ||Omega||<0.18",
            "pixel_mask": "gt_mask & abs(exp_last_channel)>1e-3 & finite(TTC) & TTC<100",
            "single_sequence_debug_split": "train 80%, buffer 5%, validation 15% after official sample filtering",
        },
        "split_summary": split_summary,
        "copy_stats": stats,
    }
    write_json(meta_path, summary)
    return summary


class OfficialTTCDataset(Dataset):
    """官方 TTCEF_DL 的等价版本：读取 exp_filts / ttc / mask 并做官方增强。"""

    def __init__(self, h5_path: Path, *, augment: bool, flip_prob: float) -> None:
        self.h5_path = h5_path
        self.augment = augment
        self.flip_prob = flip_prob
        self._file: h5py.File | None = None
        with h5py.File(self.h5_path, "r") as f:
            self.length = int(f["exp_filts"].shape[0])
            self.source_indices = f["source_index"][:]
        self.transforms = v2.Compose(
            [
                v2.RandomHorizontalFlip(p=flip_prob),
                v2.RandomVerticalFlip(p=flip_prob),
                v2.RandomRotation(degrees=(0, 180)),
            ]
        )

    def _h5(self) -> h5py.File:
        if self._file is None:
            self._file = h5py.File(self.h5_path, "r", libver="latest")
        return self._file

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        f = self._h5()
        exp = f["exp_filts"][idx].astype(np.float32)
        ttc = f["ttc"][idx].astype(np.float32)
        mask = f["mask"][idx].astype(bool)
        input_ten = torch.cat(
            [
                torch.from_numpy(exp),
                torch.from_numpy(ttc)[None],
                torch.from_numpy(mask.astype(np.float32))[None],
            ]
        ).float()
        if self.augment:
            input_ten = self.transforms(input_ten)
        return (
            input_ten[:6].float(),
            input_ten[6][None].float(),
            input_ten[7][None].bool(),
            int(self.source_indices[idx]),
        )


@dataclass
class MetricAccumulator:
    losses: list[float]
    abs_sum: float
    median_samples: list[np.ndarray]
    median_sample_pixels: int
    median_sample_limit: int
    median_per_update_limit: int
    rel_sum: float
    valid_pixels: int
    pred_sum: float
    pred_sq_sum: float
    pred_min: float
    pred_max: float
    nan_count: int
    inf_count: int
    risk_abs_sum: dict[str, float]
    risk_count: dict[str, int]
    edge_abs_sum: float
    edge_count: int

    @classmethod
    def create(cls) -> "MetricAccumulator":
        # 不缓存整轮所有像素误差。360x360 全量训练时像素级误差会达到数十亿个，
        # 如果全部 append 后再 concatenate，validation 结束处容易触发 systemd-oomd。
        return cls(
            [],
            0.0,
            [],
            0,
            2_000_000,
            512,
            0.0,
            0,
            0.0,
            0.0,
            float("inf"),
            float("-inf"),
            0,
            0,
            {"lt1": 0.0, "lt2": 0.0, "lt5": 0.0},
            {"lt1": 0, "lt2": 0, "lt5": 0},
            0.0,
            0,
        )

    def append_median_sample(self, vals: np.ndarray) -> None:
        remaining = self.median_sample_limit - self.median_sample_pixels
        if remaining <= 0 or vals.size == 0:
            return
        take = min(int(vals.size), remaining, self.median_per_update_limit)
        if take <= 0:
            return
        if vals.size <= take:
            sample = vals
        else:
            # 每个 batch 内均匀抽样，用固定规则保证重复运行可复现。
            indices = np.linspace(0, vals.size - 1, take, dtype=np.int64)
            sample = vals[indices]
        self.median_samples.append(sample.astype(np.float32, copy=False))
        self.median_sample_pixels += int(sample.size)

    def update(self, loss: float, pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> None:
        self.losses.append(float(loss))
        self.nan_count += int(np.isnan(pred).sum())
        self.inf_count += int(np.isinf(pred).sum())
        valid = mask & np.isfinite(pred) & np.isfinite(target)
        if not np.any(valid):
            return
        err = np.abs(pred - target)
        vals = err[valid].astype(np.float32)
        self.abs_sum += float(np.sum(vals, dtype=np.float64))
        self.append_median_sample(vals)
        pred_vals = pred[valid].astype(np.float64)
        target_vals = target[valid].astype(np.float64)
        self.valid_pixels += int(vals.size)
        self.rel_sum += float(np.sum(vals.astype(np.float64) / np.maximum(np.abs(target_vals), 1e-6)))
        self.pred_sum += float(np.sum(pred_vals))
        self.pred_sq_sum += float(np.sum(pred_vals * pred_vals))
        self.pred_min = min(self.pred_min, float(np.min(pred_vals)))
        self.pred_max = max(self.pred_max, float(np.max(pred_vals)))
        for name, threshold in [("lt1", 1.0), ("lt2", 2.0), ("lt5", 5.0)]:
            risk = valid & (target > 0) & (target < threshold)
            if np.any(risk):
                self.risk_abs_sum[name] += float(np.sum(err[risk]))
                self.risk_count[name] += int(np.sum(risk))
        gy, gx = np.gradient(target.astype(np.float32))
        grad = np.sqrt(gx * gx + gy * gy)
        if np.sum(valid) > 64:
            thr = float(np.quantile(grad[valid], 0.8))
            edge = valid & (grad >= thr)
            if np.any(edge):
                self.edge_abs_sum += float(np.sum(err[edge]))
                self.edge_count += int(np.sum(edge))

    def result(self) -> dict[str, float | int]:
        pred_mean = self.pred_sum / max(1, self.valid_pixels)
        pred_var = max(self.pred_sq_sum / max(1, self.valid_pixels) - pred_mean * pred_mean, 0.0)
        median_pool = np.concatenate(self.median_samples) if self.median_samples else np.asarray([], dtype=np.float32)
        return {
            "loss": float(np.mean(self.losses)) if self.losses else float("nan"),
            "mae": float(self.abs_sum / max(1, self.valid_pixels)) if self.valid_pixels else float("nan"),
            "median_ae": float(np.median(median_pool)) if median_pool.size else float("nan"),
            "median_ae_sample_pixels": int(self.median_sample_pixels),
            "median_ae_is_sampled": int(self.valid_pixels > self.median_sample_pixels),
            "mre": float(self.rel_sum / max(1, self.valid_pixels)),
            "valid_pixels": int(self.valid_pixels),
            "prediction_mean": float(pred_mean),
            "prediction_std": float(math.sqrt(pred_var)),
            "prediction_min": float(self.pred_min),
            "prediction_max": float(self.pred_max),
            "nan_count": int(self.nan_count),
            "inf_count": int(self.inf_count),
            "ttc_lt1_mae": self.risk_abs_sum["lt1"] / self.risk_count["lt1"] if self.risk_count["lt1"] else float("nan"),
            "ttc_lt2_mae": self.risk_abs_sum["lt2"] / self.risk_count["lt2"] if self.risk_count["lt2"] else float("nan"),
            "ttc_lt5_mae": self.risk_abs_sum["lt5"] / self.risk_count["lt5"] if self.risk_count["lt5"] else float("nan"),
            "ttc_lt1_pixels": int(self.risk_count["lt1"]),
            "ttc_lt2_pixels": int(self.risk_count["lt2"]),
            "ttc_lt5_pixels": int(self.risk_count["lt5"]),
            "edge_mae": self.edge_abs_sum / self.edge_count if self.edge_count else float("nan"),
            "edge_pixels": int(self.edge_count),
        }


def make_loader(dataset: Dataset, batch_size: int, workers: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        persistent_workers=workers > 0,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def run_eval(model: nn.Module, loader: DataLoader, device: torch.device, *, log_prefix: str | None = None) -> dict[str, Any]:
    model.eval()
    acc = MetricAccumulator.create()
    log_every = max(1, len(loader) // 10)
    with torch.no_grad():
        for step, (exp, target, mask, _) in enumerate(loader, start=1):
            exp = exp.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                pred = model(exp)
                loss = official_charbonnier_loss(target - pred, mask)
            pred_np = pred[:, 0].detach().float().cpu().numpy()
            tgt_np = target[:, 0].detach().float().cpu().numpy()
            mask_np = mask[:, 0].detach().cpu().numpy().astype(bool)
            for i in range(pred_np.shape[0]):
                acc.update(float(loss.detach().cpu().item()), pred_np[i], tgt_np[i], mask_np[i])
            # 验证集也按分段写进度，避免大验证阶段看起来像卡死。
            if log_prefix and (step == 1 or step % log_every == 0 or step == len(loader)):
                log(f"{log_prefix} validation step {step}/{len(loader)}: loss={float(loss.detach().cpu().item()):.5f}")
    return acc.result()


def save_checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer, scheduler: Any, epoch: int, global_step: int, cfg: dict[str, Any], data_summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "official_config": cfg,
            "data_split": data_summary.get("split_summary", {}),
            "seed": cfg["seed"],
            "gpu_id": cfg["gpu_id"],
            "batch_size": cfg["global_batch_size"],
            "micro_batch_size": cfg["micro_batch_size"],
            "accumulate_grad_batches": cfg["accumulate_grad_batches"],
        },
        path,
    )


def select_anchor_indices(dataset: OfficialTTCDataset, count: int) -> list[int]:
    if len(dataset) == 0:
        return []
    return np.linspace(0, len(dataset) - 1, count).round().astype(int).tolist()


def render_epoch_visuals(
    model: nn.Module,
    train_ds: OfficialTTCDataset,
    val_ds: OfficialTTCDataset,
    out_dir: Path,
    device: torch.device,
    epoch_label: str,
    *,
    train_anchors: list[int],
    val_anchors: list[int],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    samples = [("train", train_ds, i) for i in train_anchors] + [("val", val_ds, i) for i in val_anchors]
    with torch.no_grad():
        for split, ds, idx in samples:
            exp, target, mask, source_idx = ds[idx]
            x = exp[None].to(device)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                pred = model(x)[0, 0].detach().float().cpu().numpy()
            tgt = target[0].numpy()
            m = mask[0].numpy().astype(bool)
            err = np.abs(pred - tgt)
            panels = [(f"IIR ch{i}", exp[i].numpy(), "coolwarm") for i in range(6)]
            valid_vals = tgt[m & np.isfinite(tgt)]
            if valid_vals.size:
                vmin, vmax = np.quantile(valid_vals, [0.02, 0.98])
            else:
                vmin, vmax = 0.0, 10.0
            vmax = max(float(vmax), float(np.nanquantile(pred, 0.98)))
            vmin = min(float(vmin), 0.0)
            fig, axes = plt.subplots(3, 4, figsize=(18, 12), constrained_layout=True)
            axes = axes.reshape(-1)
            for ax in axes:
                ax.axis("off")
            for j, (title, arr, cmap) in enumerate(panels):
                im = axes[j].imshow(arr, cmap=cmap)
                axes[j].set_title(title)
                fig.colorbar(im, ax=axes[j], fraction=0.046, pad=0.03)
            extra = [
                ("GT TTC(s)", tgt, "viridis", vmin, vmax),
                ("Prediction TTC(s)", pred, "viridis", vmin, vmax),
                ("Official train mask", m.astype(np.float32), "gray", 0.0, 1.0),
                ("Abs Error", np.where(m, err, np.nan), "inferno", 0.0, np.nanquantile(np.where(m, err, np.nan), 0.98) if np.any(m) else 1.0),
            ]
            for k, item in enumerate(extra, start=6):
                title, arr, cmap, lo, hi = item
                im = axes[k].imshow(arr, cmap=cmap, vmin=lo, vmax=hi)
                axes[k].set_title(title)
                fig.colorbar(im, ax=axes[k], fraction=0.046, pad=0.03)
            ax = axes[10]
            ax.axis("on")
            if np.any(m):
                ax.hist(tgt[m].ravel(), bins=60, alpha=0.5, label="GT")
                ax.hist(pred[m].ravel(), bins=60, alpha=0.5, label="Pred")
                ax.legend()
            ax.set_title("GT/Pred histogram")
            fig.suptitle(f"{epoch_label} | {split} idx={idx} source={source_idx} | GT range=({np.nanmin(tgt):.3f},{np.nanmax(tgt):.3f})")
            fig.savefig(out_dir / f"{epoch_label}_{split}_{idx:05d}_source_{source_idx:06d}.png", dpi=130)
            plt.close(fig)


def plot_curves(path: Path, rows: list[dict[str, Any]], title: str) -> None:
    if not rows:
        return
    epochs = [r["epoch"] for r in rows]
    plt.figure(figsize=(9, 5))
    plt.plot(epochs, [r["train_loss"] for r in rows], label="train loss")
    plt.plot(epochs, [r["val_loss"] for r in rows], label="val loss")
    plt.xlabel("epoch")
    plt.ylabel("masked Charbonnier")
    plt.title(title)
    plt.grid(alpha=0.25)
    plt.legend()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def train_run(args: argparse.Namespace, data_summary: dict[str, Any], *, epochs: int, run_name: str, visual_epochs: set[int]) -> dict[str, Any]:
    run_dir = Path(args.out_dir) / run_name
    summary_path = run_dir / "summary.json"
    if summary_path.exists() and not args.force_train:
        summary = read_json(summary_path)
        if summary.get("complete"):
            return summary

    set_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() and not args.cpu else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
    model = EVSlim(create_evslim_cfg()).to(device)
    train_ds = OfficialTTCDataset(Path(data_summary["train_h5"]), augment=True, flip_prob=0.3)
    val_ds = OfficialTTCDataset(Path(data_summary["val_h5"]), augment=False, flip_prob=0.3)

    micro_batch = args.micro_batch_size
    accumulate = max(1, math.ceil(args.global_batch_size / micro_batch))
    effective_batch = micro_batch * accumulate
    train_loader = make_loader(train_ds, micro_batch, args.workers, shuffle=True)
    val_loader = make_loader(val_ds, micro_batch, args.workers, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        total_steps=max(1, epochs * math.ceil(len(train_loader) / accumulate)),
        pct_start=args.pct_start,
        cycle_momentum=False,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    cfg = {
        "run_name": run_name,
        "epochs": epochs,
        "seed": args.seed,
        "gpu_id": 0 if device.type == "cuda" else -1,
        "gpu_name": gpu_name,
        "global_batch_size": args.global_batch_size,
        "micro_batch_size": micro_batch,
        "accumulate_grad_batches": accumulate,
        "effective_batch_size": effective_batch,
        "lr": args.lr,
        "pct_start": args.pct_start,
        "precision": "16-mixed" if device.type == "cuda" else "32",
        "official_augmentation": "RandomHorizontalFlip/RandomVerticalFlip p=0.3, RandomRotation(0,180)",
        "loss": "official masked Charbonnier alpha=0.45",
    }
    write_json(run_dir / "official_config.json", cfg)

    train_anchors = select_anchor_indices(train_ds, 2)
    val_anchors = select_anchor_indices(val_ds, 3)
    epoch_rows: list[dict[str, Any]] = []
    best_val = float("inf")
    best_epoch = -1
    global_step = 0
    run_start = time.time()

    render_epoch_visuals(model, train_ds, val_ds, run_dir / "visualizations", device, "epoch_000", train_anchors=train_anchors, val_anchors=val_anchors)
    for epoch in range(1, epochs + 1):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        epoch_start = time.time()
        model.train()
        train_acc = MetricAccumulator.create()
        optimizer.zero_grad(set_to_none=True)
        # 单个 360x360 epoch 可能较慢；分段写日志，方便判断训练是否仍在正常推进。
        log_every = max(1, len(train_loader) // 10)
        for step, (exp, target, mask, _) in enumerate(train_loader, start=1):
            exp = exp.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                pred = model(exp)
                loss = official_charbonnier_loss(target - pred, mask) / accumulate
            scaler.scale(loss).backward()
            pred_np = pred[:, 0].detach().float().cpu().numpy()
            tgt_np = target[:, 0].detach().float().cpu().numpy()
            mask_np = mask[:, 0].detach().cpu().numpy().astype(bool)
            for i in range(pred_np.shape[0]):
                train_acc.update(float((loss.detach().cpu().item()) * accumulate), pred_np[i], tgt_np[i], mask_np[i])
            if step % accumulate == 0 or step == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1
            if step == 1 or step % log_every == 0 or step == len(train_loader):
                log(
                    f"{run_name} epoch {epoch}/{epochs} step {step}/{len(train_loader)}: "
                    f"loss={float((loss.detach().cpu().item()) * accumulate):.5f}"
                )
        log(f"{run_name} epoch {epoch}/{epochs}: validation start")
        val_metrics = run_eval(model, val_loader, device, log_prefix=f"{run_name} epoch {epoch}/{epochs}")
        train_metrics = train_acc.result()
        epoch_seconds = time.time() - epoch_start
        peak_mem_mb = torch.cuda.max_memory_allocated(device) / (1024**2) if device.type == "cuda" else 0.0
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"],
            "val_mae": val_metrics["mae"],
            "val_mre": val_metrics["mre"],
            "val_median_ae": val_metrics["median_ae"],
            "valid_pixels": val_metrics["valid_pixels"],
            "pred_mean": val_metrics["prediction_mean"],
            "pred_std": val_metrics["prediction_std"],
            "pred_min": val_metrics["prediction_min"],
            "pred_max": val_metrics["prediction_max"],
            "nan_count": val_metrics["nan_count"],
            "inf_count": val_metrics["inf_count"],
            "ttc_lt1_mae": val_metrics["ttc_lt1_mae"],
            "ttc_lt2_mae": val_metrics["ttc_lt2_mae"],
            "ttc_lt5_mae": val_metrics["ttc_lt5_mae"],
            "edge_mae": val_metrics["edge_mae"],
            "lr": scheduler.get_last_lr()[0],
            "epoch_seconds": epoch_seconds,
            "peak_memory_mb": peak_mem_mb,
        }
        epoch_rows.append(row)
        save_checkpoint(run_dir / f"latest_{epochs}ep.ckpt", model, optimizer, scheduler, epoch, global_step, cfg, data_summary)
        if row["val_loss"] < best_val:
            best_val = float(row["val_loss"])
            best_epoch = epoch
            save_checkpoint(run_dir / f"best_val_loss_{epochs}ep.ckpt", model, optimizer, scheduler, epoch, global_step, cfg, data_summary)
        if epoch in visual_epochs:
            render_epoch_visuals(model, train_ds, val_ds, run_dir / "visualizations", device, f"epoch_{epoch:03d}", train_anchors=train_anchors, val_anchors=val_anchors)
        log(f"{run_name} epoch {epoch}/{epochs}: train={row['train_loss']:.5f}, val={row['val_loss']:.5f}, mae={row['val_mae']:.5f}, {epoch_seconds:.1f}s")

    if best_epoch not in visual_epochs:
        best_ckpt = torch.load(run_dir / f"best_val_loss_{epochs}ep.ckpt", map_location=device)
        model.load_state_dict(best_ckpt["model_state_dict"])
        render_epoch_visuals(model, train_ds, val_ds, run_dir / "visualizations", device, f"best_epoch_{best_epoch:03d}", train_anchors=train_anchors, val_anchors=val_anchors)

    write_csv(run_dir / "epoch_metrics.csv", epoch_rows)
    plot_curves(run_dir / "train_val_curves.png", epoch_rows, run_name)
    summary = {
        "complete": True,
        "run_name": run_name,
        "epochs": epochs,
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "last_epoch": epoch_rows[-1],
        "total_train_seconds": time.time() - run_start,
        "checkpoints": {
            "latest": str(run_dir / f"latest_{epochs}ep.ckpt"),
            "best": str(run_dir / f"best_val_loss_{epochs}ep.ckpt"),
        },
        "metrics_csv": str(run_dir / "epoch_metrics.csv"),
        "visualization_dir": str(run_dir / "visualizations"),
        "config": cfg,
    }
    write_json(summary_path, summary)
    return summary


def should_run_40(summary20: dict[str, Any]) -> tuple[bool, str]:
    rows = []
    with open(summary20["metrics_csv"], newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append({k: float(v) if k not in {"epoch"} else int(v) for k, v in row.items()})
    if len(rows) < 20:
        return False, "20 epoch 指标不完整，不启动40 epoch。"
    vals = [float(r["val_loss"]) for r in rows]
    best_epoch = int(summary20["best_epoch"])
    recent = vals[-5:]
    recent_down = recent[-1] < recent[0]
    best_near_end = best_epoch >= 18
    no_rebound = recent[-1] <= min(recent[:-1]) * 1.05
    if best_near_end or (recent_down and no_rebound):
        return True, f"满足扩展条件：best_epoch={best_epoch}, 最近5轮 {recent[0]:.5f}->{recent[-1]:.5f}。"
    return False, f"不满足扩展条件：best_epoch={best_epoch}, 最近5轮 {recent[0]:.5f}->{recent[-1]:.5f}。"


def create_reports(args: argparse.Namespace, data_summary: dict[str, Any], summary20: dict[str, Any], summary40: dict[str, Any] | None, decision_text: str) -> None:
    out_dir = Path(args.out_dir)
    report_dir = out_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    split = data_summary["split_summary"]
    (report_dir / "car单序列官方EV-TTC数据生成报告.md").write_text(
        f"""# car单序列官方EV-TTC数据生成报告

## 数据链路

- 序列：`{SEQ_NAME}`
- 官方原始数据：`{data_summary['paths']['data']}`
- 官方 IIR：`{data_summary['paths']['exp']}`
- 官方 GT：`{data_summary['paths']['gt']}`
- IIR 与 GT 时间戳：`exp_times == te`，完全一致。
- 预测目标：官方 TTC 秒值，`TTC = Z / Tz`。
- 训练 H5：`{data_summary['train_h5']}`
- 验证 H5：`{data_summary['val_h5']}`

## 官方参数

- IIR alpha：`{OFFICIAL_ALPHAS}`
- time bin：`0.2 ms`
- 输出间隔：`7 ms`
- 运动筛选：`valid & ||T|| > 1.3 & ||Omega|| < 0.18`
- 官方训练 mask：`gt_mask & abs(exp_last_channel)>1e-3 & finite(TTC) & TTC<100`

## 单序列Debug划分

{markdown_table(['项目','数量/范围'], [
    ['总IIR帧', split['total_exp_frames']],
    ['官方valid帧', split['official_valid_frames']],
    ['运动筛选后样本', split['official_motion_filtered_samples']],
    ['Train样本', split['train_samples']],
    ['Buffer样本', split['buffer_samples']],
    ['Validation样本', split['val_samples']],
    ['GT时间间隔均值(us)', fmt(split['dt_us_mean'], 2)],
])}
""",
        encoding="utf-8",
    )

    cfg = summary20["config"]
    (report_dir / "car单序列官方EV-TTC训练配置核对.md").write_text(
        f"""# car单序列官方EV-TTC训练配置核对

| 项目 | 本次设置 |
|---|---|
| 网络 | 官方 EV-Slim，6通道输入，1通道TTC输出 |
| Encoder | `[16,16,16]` |
| ASPP | `[32,16]`, rates `[12,24,36]` |
| Decoder | `[8,8,1]` |
| Loss | masked Charbonnier, alpha=0.45 |
| Optimizer | AdamW |
| LR | {cfg['lr']} |
| Scheduler | OneCycleLR, pct_start={cfg['pct_start']} |
| Epoch | 20 baseline |
| GPU | {cfg['gpu_name']} |
| global batch size | {cfg['global_batch_size']} |
| micro batch | {cfg['micro_batch_size']} |
| accumulate_grad_batches | {cfg['accumulate_grad_batches']} |
| precision | {cfg['precision']} |
| 数据增强 | {cfg['official_augmentation']} |
| 预训练权重 | 不加载，从头训练 |
""",
        encoding="utf-8",
    )

    def run_result_md(summary: dict[str, Any], epochs: int) -> str:
        last = summary["last_epoch"]
        return f"""# car单序列官方EV-TTC_{epochs}Epoch训练结果

| 指标 | 数值 |
|---|---:|
| Best Epoch | {summary['best_epoch']} |
| Best Val Loss | {fmt(summary['best_val_loss'], 6)} |
| Last Train Loss | {fmt(last['train_loss'], 6)} |
| Last Val Loss | {fmt(last['val_loss'], 6)} |
| Val MAE | {fmt(last['val_mae'], 6)} |
| Val MRE | {fmt(last['val_mre'], 6)} |
| Val Median AE | {fmt(last['val_median_ae'], 6)} |
| TTC<1s MAE | {fmt(last['ttc_lt1_mae'], 6)} |
| TTC<2s MAE | {fmt(last['ttc_lt2_mae'], 6)} |
| TTC<5s MAE | {fmt(last['ttc_lt5_mae'], 6)} |
| 预测std | {fmt(last['pred_std'], 6)} |
| 边缘清晰度代理指标 | {fmt(last['edge_mae'], 6)} |
| 总训练时间(s) | {fmt(summary['total_train_seconds'], 2)} |
| Best checkpoint | `{summary['checkpoints']['best']}` |
| Latest checkpoint | `{summary['checkpoints']['latest']}` |
| 曲线 | `{Path(summary['metrics_csv']).with_name('train_val_curves.png')}` |
| 可视化 | `{summary['visualization_dir']}` |
"""

    (report_dir / "car单序列官方EV-TTC_20Epoch训练结果.md").write_text(run_result_md(summary20, 20), encoding="utf-8")
    if summary40:
        (report_dir / "car单序列官方EV-TTC_40Epoch扩展结果.md").write_text(run_result_md(summary40, 40), encoding="utf-8")
    else:
        (report_dir / "car单序列官方EV-TTC_40Epoch扩展必要性判定.md").write_text(
            f"# car单序列官方EV-TTC_40Epoch扩展必要性判定\n\n{decision_text}\n",
            encoding="utf-8",
        )

    if summary40:
        rows = []
        for metric, key in [
            ("Best Epoch", "best_epoch"),
            ("Best Val Loss", "best_val_loss"),
            ("Val MAE", "val_mae"),
            ("Val MRE", "val_mre"),
            ("TTC<1s MAE", "ttc_lt1_mae"),
            ("TTC<2s MAE", "ttc_lt2_mae"),
            ("预测std", "pred_std"),
            ("边缘清晰度代理指标", "edge_mae"),
            ("总训练时间", "total_train_seconds"),
        ]:
            v20 = summary20[key] if key in summary20 else summary20["last_epoch"].get(key)
            v40 = summary40[key] if key in summary40 else summary40["last_epoch"].get(key)
            rows.append({"metric": metric, "20_epoch": v20, "40_epoch": v40})
        write_csv(out_dir / "official_20ep_vs_40ep.csv", rows)
        table = markdown_table(["指标", "20 Epoch", "40 Epoch"], [[r["metric"], fmt(r["20_epoch"], 6), fmt(r["40_epoch"], 6)] for r in rows])
        text = f"# car单序列官方EV-TTC_20与40Epoch对比\n\n{table}\n\n{decision_text}\n"
    else:
        text = f"# car单序列官方EV-TTC_20与40Epoch对比\n\n未运行40 Epoch。\n\n{decision_text}\n"
    (report_dir / "car单序列官方EV-TTC_20与40Epoch对比.md").write_text(text, encoding="utf-8")

    (report_dir / "car单序列官方EV-TTC可视化分析.md").write_text(
        f"""# car单序列官方EV-TTC可视化分析

- 20 Epoch 可视化目录：`{summary20['visualization_dir']}`
{f"- 40 Epoch 可视化目录：`{summary40['visualization_dir']}`" if summary40 else "- 40 Epoch 未运行。"}

观察重点：
- 道路纵向TTC梯度；
- 车辆和树干边界；
- 近场 TTC<1s / TTC<2s 区域；
- 预测是否过度平滑；
- 20到40 Epoch是否改善局部结构。
""",
        encoding="utf-8",
    )

    comparison = markdown_table(
        ["项目", "当前EVSlim_IIR360", "本次官方链路"],
        [
            ["IIR输入", "官方IIR", "官方IIR"],
            ["输入时间戳", "最近匹配", "与GT完全同exp_times"],
            ["标签", "signed inverse TTC", "官方TTC秒值"],
            ["Mask", "当前统一Mask", "官方Mask"],
            ["Batch size", "32", str(cfg["global_batch_size"])],
            ["数据增强", "关闭", "官方开启"],
            ["Epoch", "20", "官方20，必要时扩展40"],
            ["数据划分", "500-Block", "单序列官方链路debug划分"],
            ["网络", "EV-Slim", "官方EV-Slim"],
        ],
    )
    pass_text = f"""# car单序列官方EV-TTC复现通过或失败判定

## 逐项判定

- 官方 create_exp.py 输出已存在并被使用：通过。
- 官方 calc_gt.py 输出已存在并被使用：通过。
- IIR 与 GT 时间戳完全一致：通过。
- 目标为官方 TTC 秒值：通过。
- Mask 为官方定义：通过。
- 网络为官方 EV-Slim：通过。
- 单张 GPU：通过，`{cfg['gpu_name']}`。
- global batch size=128：{'通过' if cfg['global_batch_size'] == 128 else '失败'}。
- 20 Epoch 官方训练：{'通过' if summary20.get('complete') else '失败'}。
- NaN/Inf：最后一轮 nan={summary20['last_epoch']['nan_count']}，inf={summary20['last_epoch']['inf_count']}。

## 与之前EVSlim_IIR360对照

{comparison}

## 结论

本次结果可称为“car 单序列官方链路 debug 复现”，但不能称为论文完整复现；原因是只使用一条 car 序列，且 Train/Buffer/Validation 是单序列时间划分，不是论文完整多序列训练/测试划分。
"""
    (report_dir / "car单序列官方EV-TTC复现通过或失败判定.md").write_text(pass_text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行 car 单序列官方 EV-TTC 复现")
    parser.add_argument("--official-root", default=str(DEFAULT_OFFICIAL_ROOT))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--force-prepare", action="store_true")
    parser.add_argument("--force-train", action="store_true")
    parser.add_argument("--include-flow", action="store_true", help="在split H5中额外保存官方flow；训练不会读取")
    parser.add_argument("--prepare-chunk-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--global-batch-size", type=int, default=128)
    parser.add_argument("--micro-batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--pct-start", type=float, default=0.1)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--skip-40", action="store_true")
    parser.add_argument("--force-40", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_summary = prepare_data(args)
    summary20 = train_run(args, data_summary, epochs=20, run_name="EVTTC_Official_Car_20ep", visual_epochs={5, 10, 15, 20})
    run40, decision = should_run_40(summary20)
    summary40 = None
    if args.skip_40:
        decision = "用户/参数指定跳过40 Epoch。" + decision
    elif args.force_40 or run40:
        decision = ("强制运行40 Epoch。" if args.force_40 else decision)
        summary40 = train_run(args, data_summary, epochs=40, run_name="EVTTC_Official_Car_40ep_Extended", visual_epochs={10, 20, 30, 40})
    create_reports(args, data_summary, summary20, summary40, decision)
    log(f"报告已生成：{out_dir / 'reports'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
