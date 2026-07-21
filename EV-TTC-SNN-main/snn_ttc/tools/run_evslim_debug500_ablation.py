#!/usr/bin/env python3
"""EV-Slim 四组 500Block 快速对照实验。

本脚本对应指导文件中的阶段A，只做同一条 car 序列、同一 500-Block
时间范围内的输入表示/分辨率快速诊断：

1. 生成四组可共用的 debug H5：
   - 官方 6 通道 IIR 360x360
   - S2 中心 256 视场原生 6 通道 IIR 128x128
   - 固定 10k 事件计数 2 通道 360x360
   - 既有 S2 中心 256 视场固定 10k 事件计数 128x128
2. 使用同一个 EV-Slim 主体训练四组模型，仅第一层输入通道随输入变化。
3. 输出统一指标、可视化和后续 ANN / ANN-SNN 可复用的数据索引。

注意：这里的输出目标是 signed inverse TTC，用于和当前 SNN-TTC 迁移工作对齐；
不是 EV-TTC 论文原始的 TTC 秒值正式复现实验。
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
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

import cv2
import h5py
import hdf5plugin  # noqa: F401  # 读取/写入 Blosc 压缩 HDF5 需要注册插件
import matplotlib
import numba

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SNN_ROOT = PROJECT_ROOT / "EV-TTC-SNN-main"
EVTTC_MODEL_ROOT = PROJECT_ROOT / "EV-TTC-main" / "model"

for path in (SNN_ROOT, EVTTC_MODEL_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from evslim import EVSlim  # noqa: E402
from snn_ttc.data.m3ed_geometry import load_camera_geometry  # noqa: E402


DEFAULT_SOURCE_COUNT_H5 = (
    SNN_ROOT
    / "debug_sets"
    / "[5]car_spatial_ablation"
    / "independent_fixed_n"
    / "N10000_S1_360.h5"
)
DEFAULT_SOURCE_S2_H5 = (
    SNN_ROOT
    / "debug_sets"
    / "[5]car_spatial_ablation"
    / "independent_fixed_n"
    / "N10000_S2_center256_to128.h5"
)
DEFAULT_RAW_DATA_H5 = Path("/data/evttc_storage/event+SNN+TTC/data_m3ed_official_7train_3test/events/m3ed/car_urban_night_rittenhouse/car_urban_night_rittenhouse_data.h5")
DEFAULT_OFFICIAL_EXP_H5 = (
    Path("/data/evttc_storage/event+SNN+TTC/data_m3ed_official_7train_3test")
    / "exp_filts"
    / "m3ed"
    / "car_urban_night_rittenhouse.h5"
)
DEFAULT_OUT_ROOT = SNN_ROOT / "debug_sets" / "evslim_debug500_ablation"
DEFAULT_DATASET_ROOT = SNN_ROOT / "datasets" / "m3ed_evttc_debug500"

SEQUENCE_NAME = "car_urban_night_rittenhouse"
STEPS_PER_BLOCK = 10
NUM_BLOCKS = 500
NUM_STEPS = STEPS_PER_BLOCK * NUM_BLOCKS
TRAIN_BLOCK_MAX = 389
BUFFER_BLOCK_MAX = 399
VAL_BLOCK_MIN = 400

TARGET_KEY = "fixed10_signed_inverse_ttc"
MASK_KEY = "fixed10_valid_ttc_mask"
S2_BOX_360 = (52, 52, 256)
IIR_ALPHAS = np.asarray([0.12, 0.06, 0.03, 0.015, 0.0095, 0.0045], dtype=np.float32)
IIR_TIME_BIN_US = 200.0


@dataclass(frozen=True)
class MethodSpec:
    """四组输入的最小定义。"""

    name: str
    input_kind: str
    resolution: int
    in_channels: int
    display_name: str


METHODS: dict[str, MethodSpec] = {
    "EVSlim_IIR360": MethodSpec("EVSlim_IIR360", "iir", 360, 6, "官方IIR 360"),
    "EVSlim_IIR128": MethodSpec("EVSlim_IIR128", "iir", 128, 6, "S2原生IIR 128"),
    "EVSlim_Count10k360": MethodSpec("EVSlim_Count10k360", "count", 360, 2, "10k计数 360"),
    "EVSlim_Count10k128": MethodSpec("EVSlim_Count10k128", "count", 128, 2, "S2既有10k计数 128"),
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def format_float(value: float | int | None, digits: int = 4) -> str:
    if value is None:
        return "NA"
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not math.isfinite(value_f):
        return "NA"
    return f"{value_f:.{digits}f}"


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def split_name(local_block_index: int) -> str:
    if local_block_index <= TRAIN_BLOCK_MAX:
        return "train"
    if local_block_index <= BUFFER_BLOCK_MAX:
        return "buffer"
    return "validation"


def compression_kwargs() -> dict[str, Any]:
    return hdf5plugin.Blosc2(cname="lz4", clevel=5, filters=hdf5plugin.Blosc.SHUFFLE)


def resize_area(image: np.ndarray, out_hw: tuple[int, int], *, mass_preserve: bool = False) -> np.ndarray:
    """用面积插值做同视场缩放；事件计数可选择保持总事件量近似不变。"""

    out_h, out_w = out_hw
    in_h, in_w = image.shape[-2:]
    if (in_h, in_w) == (out_h, out_w):
        return image.astype(np.float32, copy=False)

    if image.ndim == 2:
        resized = cv2.resize(image.astype(np.float32), (out_w, out_h), interpolation=cv2.INTER_AREA)
    elif image.ndim == 3:
        channels = [
            cv2.resize(ch.astype(np.float32), (out_w, out_h), interpolation=cv2.INTER_AREA)
            for ch in image
        ]
        resized = np.stack(channels, axis=0)
    else:
        raise ValueError(f"resize_area 只支持 2D/3D，实际 shape={image.shape}")

    if mass_preserve:
        resized = resized * ((in_h * in_w) / float(out_h * out_w))
    return resized.astype(np.float32, copy=False)


def downsample_target_and_mask(target: np.ndarray, mask: np.ndarray, out_hw: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """先对 valid 区域做加权面积平均，再从有效面积比例恢复 128x128 mask。"""

    out_h, out_w = out_hw
    if target.shape == (out_h, out_w):
        return target.astype(np.float32, copy=False), mask.astype(bool, copy=False)

    mask_f = mask.astype(np.float32)
    tgt_sum = cv2.resize((target * mask_f).astype(np.float32), (out_w, out_h), interpolation=cv2.INTER_AREA)
    mask_area = cv2.resize(mask_f, (out_w, out_h), interpolation=cv2.INTER_AREA)
    out_target = np.zeros((out_h, out_w), dtype=np.float32)
    valid = mask_area > 0.25
    out_target[valid] = tgt_sum[valid] / np.maximum(mask_area[valid], 1e-6)
    return out_target, valid.astype(bool)


def event_active_mask(input_map: np.ndarray, kind: str) -> np.ndarray:
    """按照输入表示得到训练用事件活跃 mask。IIR 保留官方 merge 的末通道阈值口径。"""

    if kind == "iir":
        return np.abs(input_map[-1]) > 1e-3
    return np.sum(np.abs(input_map), axis=0) > 0


@numba.njit(cache=True)
def accumulate_s2_native_iir128_numba(
    x_ev: np.ndarray,
    y_ev: np.ndarray,
    p_ev: np.ndarray,
    t_ev: np.ndarray,
    dist_map: np.ndarray,
    alphas: np.ndarray,
    exp_img: np.ndarray,
    t_out: int,
    time_bin_us: float,
) -> None:
    """把原始事件直接写入 S2 128x128 IIR 状态，不从 360 图像读数。"""

    roi_x0 = 52
    roi_y0 = 52
    roi_size_360 = 256
    for i in range(x_ev.shape[0]):
        x = int(x_ev[i])
        y = int(y_ev[i])
        if x < 280 or x > 999 or y < 0 or y >= 720:
            continue
        age_us = float(t_out - int(t_ev[i]))
        if age_us < 0.0:
            continue
        update = dist_map[y, x - 280]
        base_x = int(update[0])
        base_y = int(update[1])
        if base_x < 0 or base_x >= 359 or base_y < 0 or base_y >= 359:
            continue
        sign = 1.0 if p_ev[i] > 0 else -1.0
        age_bins = age_us / time_bin_us

        # EV-TTC 的 360 坐标四邻域权重先保留，再直接聚合到 S2 的 2x2 sum-pooling 像素。
        for nbr in range(4):
            weight = update[2 + nbr]
            if weight <= 0.0:
                continue
            nx = base_x + (1 if nbr == 1 or nbr == 3 else 0)
            ny = base_y + (1 if nbr == 2 or nbr == 3 else 0)
            if nx < roi_x0 or nx >= roi_x0 + roi_size_360 or ny < roi_y0 or ny >= roi_y0 + roi_size_360:
                continue
            ox = (nx - roi_x0) // 2
            oy = (ny - roi_y0) // 2
            if ox < 0 or ox >= 128 or oy < 0 or oy >= 128:
                continue
            for c in range(alphas.shape[0]):
                alpha = alphas[c]
                exp_img[c, oy, ox] += sign * weight * alpha * ((1.0 - alpha) ** age_bins)


class NativeS2IIR128Generator:
    """按 S2 时间轴从原始事件流连续生成 128x128 IIR。

    这里的状态是连续携带的：每个 step 输出前先按当前时间间隔衰减上一帧状态，
    再把上一输出时刻到当前输出时刻之间的原始事件直接映射进 S2 128 网格。
    """

    def __init__(self, raw_h5: h5py.File, source_s2: h5py.File, dist_map: np.ndarray) -> None:
        self.events = raw_h5["prophesee"]["left"]
        self.raw_end = source_s2["raw_event_end_idx"][:NUM_STEPS].astype(np.int64)
        self.raw_start = source_s2["raw_event_start_idx"][:NUM_STEPS].astype(np.int64)
        self.t_start = source_s2["t_start"][:NUM_STEPS].astype(np.int64)
        self.t_end = source_s2["t_end"][:NUM_STEPS].astype(np.int64)
        self.dist_map = dist_map
        self.exp_img = np.zeros((6, 128, 128), dtype=np.float32)
        self.last_raw = int(self.raw_start[0])
        self.last_output_time = int(self.t_start[0])

    def get(self, idx: int) -> np.ndarray:
        t_out = int(self.t_end[idx])
        delta_bins = max(0.0, (t_out - self.last_output_time) / IIR_TIME_BIN_US)
        self.exp_img *= ((1.0 - IIR_ALPHAS) ** delta_bins)[:, None, None]

        raw_end = int(self.raw_end[idx])
        if raw_end > self.last_raw:
            x = self.events["x"][self.last_raw:raw_end]
            y = self.events["y"][self.last_raw:raw_end]
            p = self.events["p"][self.last_raw:raw_end]
            t = self.events["t"][self.last_raw:raw_end]
            accumulate_s2_native_iir128_numba(
                x,
                y,
                p,
                t,
                self.dist_map,
                IIR_ALPHAS,
                self.exp_img,
                t_out,
                IIR_TIME_BIN_US,
            )
            self.last_raw = raw_end
        self.last_output_time = t_out
        return self.exp_img.copy()


def create_evslim_cfg(in_channels: int) -> SimpleNamespace:
    """构造与 EV-TTC 官方 YAML 一致的 EV-Slim 主体配置。"""

    return SimpleNamespace(
        enc_k_size=[3, 3, 3],
        enc_channels=[16, 16, 16],
        aspp_channels=[32, 16],
        rate=[12, 24, 36],
        dec_k_size=[3, 3, 3],
        dec_channels=[8, 8, 1],
        norm=True,
        in_channels=in_channels,
        input=in_channels,
        act="ReLU",
        alpha=0.45,
    )


def masked_charbonnier_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    alpha: float = 0.45,
    epsilon: float = 1e-5,
) -> torch.Tensor:
    """复刻 EV-TTC 官方 mask 分支：逐样本有效像素平均，再对有效样本平均。"""

    error = target - prediction
    valid = mask.bool() & torch.isfinite(error)
    safe_error = torch.where(valid, error, torch.zeros_like(error))
    loss_map = (safe_error.square() + epsilon**2.0).pow(alpha)
    valid_float = valid.to(loss_map.dtype)
    denom = valid_float.flatten(1).sum(dim=1)
    per_sample = (loss_map * valid_float).flatten(1).sum(dim=1) / denom.clamp_min(1.0)
    keep = denom > 0
    if not bool(keep.any().detach().cpu().item()):
        return safe_error.sum() * 0.0
    return per_sample[keep].mean()


class EVSlimH5Dataset(Dataset):
    """每个样本是单帧输入、signed inverse TTC 标签和两个评价 mask。"""

    def __init__(self, h5_path: Path, preload: bool = True) -> None:
        self.h5_path = h5_path
        self.preload = preload
        self._handle: h5py.File | None = None
        with h5py.File(self.h5_path, "r") as f:
            self.length = int(f["exp_filts"].shape[0])
            self.in_channels = int(f["exp_filts"].shape[1])
            self.resolution = int(f["exp_filts"].shape[2])
            self.global_indices = f["global_index"][:].astype(np.int64)
        if preload:
            log(f"预加载 {self.h5_path.name} 到内存：{self.length} samples")
            with h5py.File(self.h5_path, "r") as f:
                self.exp = f["exp_filts"][:].astype(np.float32)
                self.target = f["ttc"][:].astype(np.float32)
                self.mask = f["mask"][:].astype(bool)
                self.dense_mask = f["dense_valid_mask"][:].astype(bool)
                self.event_mask = f["event_active_mask"][:].astype(bool)
        else:
            self.exp = None
            self.target = None
            self.mask = None
            self.dense_mask = None
            self.event_mask = None

    def _file(self) -> h5py.File:
        if self._handle is None:
            self._handle = h5py.File(self.h5_path, "r")
        return self._handle

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if self.preload:
            exp = self.exp[idx]
            target = self.target[idx]
            mask = self.mask[idx]
            dense = self.dense_mask[idx]
            event = self.event_mask[idx]
        else:
            f = self._file()
            exp = f["exp_filts"][idx].astype(np.float32)
            target = f["ttc"][idx].astype(np.float32)
            mask = f["mask"][idx].astype(bool)
            dense = f["dense_valid_mask"][idx].astype(bool)
            event = f["event_active_mask"][idx].astype(bool)

        return {
            "exp": torch.from_numpy(exp.astype(np.float32, copy=False)),
            "target": torch.from_numpy(target[None].astype(np.float32, copy=False)),
            "mask": torch.from_numpy(mask[None]),
            "dense_mask": torch.from_numpy(dense[None]),
            "event_mask": torch.from_numpy(event[None]),
            "global_index": int(self.global_indices[idx]),
        }


def make_loader(dataset: EVSlimH5Dataset, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


@dataclass
class MaskAccumulator:
    """一个 mask 口径下的流式指标累加器。"""

    name: str
    abs_errors: list[np.ndarray] = field(default_factory=list)
    rel_error_sum: float = 0.0
    pred_sum: float = 0.0
    target_sum: float = 0.0
    pred_sq_sum: float = 0.0
    valid_count: int = 0
    high_abs_sum: float = 0.0
    high_count: int = 0
    positive_abs_sum: float = 0.0
    positive_count: int = 0
    negative_abs_sum: float = 0.0
    negative_count: int = 0
    sign_correct: int = 0
    sign_count: int = 0
    edge_abs_sum: float = 0.0
    edge_count: int = 0
    interior_abs_sum: float = 0.0
    interior_count: int = 0
    grad_abs_sum: float = 0.0
    grad_count: int = 0

    def update_basic(self, pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> None:
        valid = mask & np.isfinite(pred) & np.isfinite(target)
        if not np.any(valid):
            return
        err = np.abs(pred - target)
        vals = err[valid].astype(np.float32)
        self.abs_errors.append(vals)
        pred_vals = pred[valid].astype(np.float64)
        target_vals = target[valid].astype(np.float64)
        self.valid_count += int(vals.size)
        self.rel_error_sum += float(np.sum(vals.astype(np.float64) / np.maximum(np.abs(target_vals), 1e-6)))
        self.pred_sum += float(np.sum(pred_vals))
        self.target_sum += float(np.sum(target_vals))
        self.pred_sq_sum += float(np.sum(pred_vals * pred_vals))

        high = valid & (target > 0.5)
        if np.any(high):
            self.high_abs_sum += float(np.sum(err[high]))
            self.high_count += int(np.sum(high))

        positive = valid & (target > 1e-6)
        negative = valid & (target < -1e-6)
        if np.any(positive):
            self.positive_abs_sum += float(np.sum(err[positive]))
            self.positive_count += int(np.sum(positive))
        if np.any(negative):
            self.negative_abs_sum += float(np.sum(err[negative]))
            self.negative_count += int(np.sum(negative))

        sign_valid = valid & (np.abs(target) > 1e-6)
        if np.any(sign_valid):
            self.sign_count += int(np.sum(sign_valid))
            self.sign_correct += int(np.sum(np.sign(pred[sign_valid]) == np.sign(target[sign_valid])))

    def update_structure(self, pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> None:
        """用 GT 梯度的高分位近似边缘区域，统计结构恢复误差。"""

        valid = mask & np.isfinite(pred) & np.isfinite(target)
        if not np.any(valid):
            return
        gy, gx = np.gradient(target.astype(np.float32))
        pgy, pgx = np.gradient(pred.astype(np.float32))
        gt_grad = np.sqrt(gx * gx + gy * gy)
        pred_grad = np.sqrt(pgx * pgx + pgy * pgy)
        grad_err = np.abs(pred_grad - gt_grad)
        self.grad_abs_sum += float(np.sum(grad_err[valid]))
        self.grad_count += int(np.sum(valid))

        if np.sum(valid) >= 64:
            thr = float(np.quantile(gt_grad[valid], 0.8))
            edge = valid & (gt_grad >= thr)
            interior = valid & ~edge
            err = np.abs(pred - target)
            if np.any(edge):
                self.edge_abs_sum += float(np.sum(err[edge]))
                self.edge_count += int(np.sum(edge))
            if np.any(interior):
                self.interior_abs_sum += float(np.sum(err[interior]))
                self.interior_count += int(np.sum(interior))

    def as_dict(self) -> dict[str, float | int]:
        if self.valid_count == 0:
            return {"valid_pixels": 0}
        abs_err = np.concatenate(self.abs_errors) if self.abs_errors else np.asarray([], dtype=np.float32)
        pred_mean = self.pred_sum / self.valid_count
        pred_var = max(self.pred_sq_sum / self.valid_count - pred_mean * pred_mean, 0.0)
        return {
            "valid_pixels": int(self.valid_count),
            "mae": float(abs_err.mean()) if abs_err.size else float("nan"),
            "median_ae": float(np.median(abs_err)) if abs_err.size else float("nan"),
            "mre": float(self.rel_error_sum / self.valid_count),
            "high_risk_mae_gt_inv_gt_0p5": self.high_abs_sum / self.high_count if self.high_count else float("nan"),
            "high_risk_valid_pixels": int(self.high_count),
            "positive_inv_ttc_mae": self.positive_abs_sum / self.positive_count if self.positive_count else float("nan"),
            "positive_valid_pixels": int(self.positive_count),
            "negative_inv_ttc_mae": self.negative_abs_sum / self.negative_count if self.negative_count else float("nan"),
            "negative_valid_pixels": int(self.negative_count),
            "sign_accuracy": self.sign_correct / self.sign_count if self.sign_count else float("nan"),
            "prediction_mean": pred_mean,
            "prediction_std": math.sqrt(pred_var),
            "target_mean": self.target_sum / self.valid_count,
            "edge_mae": self.edge_abs_sum / self.edge_count if self.edge_count else float("nan"),
            "edge_valid_pixels": int(self.edge_count),
            "interior_mae": self.interior_abs_sum / self.interior_count if self.interior_count else float("nan"),
            "interior_valid_pixels": int(self.interior_count),
            "gradient_mae": self.grad_abs_sum / self.grad_count if self.grad_count else float("nan"),
        }


def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, Any]:
    model.eval()
    event_acc = MaskAccumulator("event_active")
    dense_acc = MaskAccumulator("dense_valid")
    losses: list[float] = []
    with torch.no_grad():
        for batch in loader:
            x = batch["exp"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            train_mask = batch["mask"].to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                pred = model(x)
                loss = masked_charbonnier_loss(pred, target, train_mask)
            losses.append(float(loss.detach().cpu().item()))
            pred_np = pred[:, 0].detach().float().cpu().numpy()
            target_np = target[:, 0].detach().float().cpu().numpy()
            event_np = batch["mask"][:, 0].numpy().astype(bool)
            dense_np = batch["dense_mask"][:, 0].numpy().astype(bool)
            for i in range(pred_np.shape[0]):
                event_acc.update_basic(pred_np[i], target_np[i], event_np[i])
                dense_acc.update_basic(pred_np[i], target_np[i], dense_np[i])
                event_acc.update_structure(pred_np[i], target_np[i], event_np[i])
                dense_acc.update_structure(pred_np[i], target_np[i], dense_np[i])

    return {
        "masked_charbonnier": float(np.mean(losses)) if losses else float("nan"),
        "event_active": event_acc.as_dict(),
        "dense_valid": dense_acc.as_dict(),
    }


def count_conv_macs(model: nn.Module, in_channels: int, resolution: int, device: torch.device) -> int:
    """通过 hook 统计 Conv2d MACs，适合 EV-Slim 这类全卷积网络。"""

    macs = 0
    hooks = []

    def hook(module: nn.Conv2d, inputs: tuple[torch.Tensor], output: torch.Tensor) -> None:
        nonlocal macs
        out = output
        kh, kw = module.kernel_size
        in_ch = module.in_channels
        groups = module.groups
        macs += int(out.shape[2] * out.shape[3] * module.out_channels * (in_ch // groups) * kh * kw)

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(hook))
    model.eval()
    with torch.no_grad():
        dummy = torch.zeros(1, in_channels, resolution, resolution, device=device)
        model(dummy)
    for h in hooks:
        h.remove()
    return macs


def benchmark_inference(model: nn.Module, in_channels: int, resolution: int, device: torch.device) -> dict[str, float]:
    if device.type != "cuda":
        return {"single_sample_ms": float("nan"), "cuda_peak_mb": float("nan")}
    model.eval()
    dummy = torch.zeros(1, in_channels, resolution, resolution, device=device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=True):
        for _ in range(5):
            model(dummy)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(30):
            model(dummy)
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000.0 / 30.0
    peak_mb = torch.cuda.max_memory_allocated(device) / (1024.0**2)
    return {"single_sample_ms": elapsed_ms, "cuda_peak_mb": peak_mb}


def build_h5_file(path: Path, count: int, channels: int, resolution: int) -> h5py.File:
    path.parent.mkdir(parents=True, exist_ok=True)
    f = h5py.File(path, "w")
    f.attrs["complete"] = False
    f.create_dataset(
        "exp_filts",
        shape=(count, channels, resolution, resolution),
        dtype=np.float16,
        chunks=(1, channels, resolution, resolution),
        **compression_kwargs(),
    )
    f.create_dataset(
        "ttc",
        shape=(count, resolution, resolution),
        dtype=np.float32,
        chunks=(1, resolution, resolution),
        **compression_kwargs(),
    )
    for key in ("mask", "dense_valid_mask", "event_active_mask"):
        f.create_dataset(
            key,
            shape=(count, resolution, resolution),
            dtype=bool,
            chunks=(1, resolution, resolution),
            **compression_kwargs(),
        )
    for key, dtype in (
        ("global_index", np.int64),
        ("block_index", np.int64),
        ("step_in_block", np.int16),
        ("t_start", np.int64),
        ("t_end", np.int64),
        ("raw_event_start_idx", np.int64),
        ("raw_event_end_idx", np.int64),
    ):
        f.create_dataset(key, shape=(count,), dtype=dtype)
    return f


def write_h5_row(
    f: h5py.File,
    local_row: int,
    input_map: np.ndarray,
    target: np.ndarray,
    dense_mask: np.ndarray,
    active_mask: np.ndarray,
    meta: dict[str, int],
) -> None:
    train_mask = dense_mask.astype(bool) & active_mask.astype(bool)
    f["exp_filts"][local_row] = input_map.astype(np.float16)
    f["ttc"][local_row] = target.astype(np.float32)
    f["dense_valid_mask"][local_row] = dense_mask.astype(bool)
    f["event_active_mask"][local_row] = active_mask.astype(bool)
    f["mask"][local_row] = train_mask.astype(bool)
    for key, value in meta.items():
        f[key][local_row] = value


def make_debug_data(args: argparse.Namespace) -> dict[str, Any]:
    """物化四组输入 H5 和索引；128 组改用 S2 原生/既有数据。"""

    dataset_root = Path(args.dataset_root)
    meta_dir = dataset_root / "metadata"
    generation_summary_path = meta_dir / "generation_summary.json"
    required = [
        dataset_root / "iir360" / "train.h5",
        dataset_root / "iir360" / "val.h5",
        dataset_root / "iir128" / "train.h5",
        dataset_root / "iir128" / "val.h5",
        dataset_root / "count10k360" / "train.h5",
        dataset_root / "count10k360" / "val.h5",
        dataset_root / "count10k128" / "train.h5",
        dataset_root / "count10k128" / "val.h5",
    ]
    if not args.force_data and generation_summary_path.exists() and all(p.exists() for p in required):
        try:
            summary = read_json(generation_summary_path)
            if summary.get("complete") and summary.get("generation_version") == "s2_native_128_v2":
                log("检测到完整 debug 数据，跳过重新生成")
                return summary
        except Exception:
            pass

    dataset_root.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)
    for sub in ("iir360", "iir128", "count10k360", "count10k128", "targets360", "targets128", "masks"):
        (dataset_root / sub).mkdir(parents=True, exist_ok=True)

    source_count_h5 = Path(args.source_count_h5)
    source_s2_h5 = Path(args.source_s2_h5)
    raw_data_h5 = Path(args.raw_data_h5)
    official_exp_h5 = Path(args.official_exp_h5)
    if not source_count_h5.exists():
        raise FileNotFoundError(source_count_h5)
    if not source_s2_h5.exists():
        raise FileNotFoundError(source_s2_h5)
    if not raw_data_h5.exists():
        raise FileNotFoundError(raw_data_h5)
    if not official_exp_h5.exists():
        raise FileNotFoundError(official_exp_h5)

    log("读取 500Block 元数据、S2 128数据和官方 IIR 时间戳")
    with h5py.File(source_count_h5, "r") as f_src, h5py.File(source_s2_h5, "r") as f_s2, h5py.File(raw_data_h5, "r") as f_raw, h5py.File(official_exp_h5, "r") as f_exp:
        if f_src["event_cnt"].shape[0] < NUM_STEPS or f_s2["event_cnt"].shape[0] < NUM_STEPS:
            raise RuntimeError("S1/S2 H5 step 数不足，无法构建 500Block")
        exp_times = f_exp["exp_times"][:].astype(np.float64)
        s1_t_end = f_src["t_end"][:NUM_STEPS].astype(np.float64)
        search_idx = np.searchsorted(exp_times, s1_t_end)
        right = np.clip(search_idx, 0, len(exp_times) - 1)
        left = np.clip(search_idx - 1, 0, len(exp_times) - 1)
        choose_right = np.abs(exp_times[right] - s1_t_end) <= np.abs(exp_times[left] - s1_t_end)
        nearest_exp_idx = np.where(choose_right, right, left).astype(np.int64)
        nearest_delta_us = (exp_times[nearest_exp_idx] - s1_t_end).astype(np.float64)

        block_index = f_src["block_index"][:NUM_STEPS].astype(np.int64)
        step_in_block = f_src["step_in_block"][:NUM_STEPS].astype(np.int16)
        t_start = f_src["t_start"][:NUM_STEPS].astype(np.int64)
        t_end_i64 = f_src["t_end"][:NUM_STEPS].astype(np.int64)
        raw_start = f_src["raw_event_start_idx"][:NUM_STEPS].astype(np.int64)
        raw_end = f_src["raw_event_end_idx"][:NUM_STEPS].astype(np.int64)
        supervise_valid = f_src["supervise_valid"][:NUM_STEPS].astype(bool)
        source_event_count = f_src["source_event_count"][:NUM_STEPS].astype(np.int32)
        event_dt = f_src["event_dt"][:NUM_STEPS].astype(np.int64)
        s2_t_start = f_s2["t_start"][:NUM_STEPS].astype(np.int64)
        s2_t_end = f_s2["t_end"][:NUM_STEPS].astype(np.int64)
        s2_raw_start = f_s2["raw_event_start_idx"][:NUM_STEPS].astype(np.int64)
        s2_raw_end = f_s2["raw_event_end_idx"][:NUM_STEPS].astype(np.int64)
        s2_event_dt = f_s2["event_dt"][:NUM_STEPS].astype(np.int64)
        s2_time_offset = (s2_t_end - t_end_i64).astype(np.float64)

        split_rows: dict[str, list[int]] = {"train": [], "validation": []}
        index_rows: list[dict[str, Any]] = []
        split_row_counter = {"train": 0, "validation": 0}
        for global_idx in range(NUM_STEPS):
            local_block = global_idx // STEPS_PER_BLOCK
            split = split_name(local_block)
            path_fields: dict[str, str] = {}
            if split in split_rows:
                split_rows[split].append(global_idx)
                local_split_row = split_row_counter[split]
                split_row_counter[split] += 1
                path_fields = {
                    "iir360_sample_path": f"{dataset_root / 'iir360' / (split + '.h5')}::row={local_split_row}",
                    "iir128_sample_path": f"{dataset_root / 'iir128' / (split + '.h5')}::row={local_split_row}",
                    "count10k360_sample_path": f"{dataset_root / 'count10k360' / (split + '.h5')}::row={local_split_row}",
                    "count10k128_sample_path": f"{dataset_root / 'count10k128' / (split + '.h5')}::row={local_split_row}",
                }
            else:
                path_fields = {
                    "iir360_sample_path": "",
                    "iir128_sample_path": "",
                    "count10k360_sample_path": "",
                    "count10k128_sample_path": "",
                }
            index_rows.append(
                {
                    "sequence_name": SEQUENCE_NAME,
                    "split": split,
                    "block_index": int(block_index[global_idx]),
                    "sample_id": global_idx,
                    "output_timestamp": int(t_end_i64[global_idx]),
                    "s1_output_timestamp": int(t_end_i64[global_idx]),
                    "s2_output_timestamp": int(s2_t_end[global_idx]),
                    "s2_minus_s1_t_end_us": int(s2_t_end[global_idx] - t_end_i64[global_idx]),
                    "raw_event_start_idx": int(raw_start[global_idx]),
                    "raw_event_end_idx": int(raw_end[global_idx]),
                    "s2_raw_event_start_idx": int(s2_raw_start[global_idx]),
                    "s2_raw_event_end_idx": int(s2_raw_end[global_idx]),
                    "event_count": int(source_event_count[global_idx]),
                    "t_start": int(t_start[global_idx]),
                    "t_end": int(t_end_i64[global_idx]),
                    "depth_timestamp": int(t_start[global_idx]),
                    "pose_timestamp": int(t_end_i64[global_idx]),
                    "reset_required": int(global_idx == 0),
                    "supervise_valid": int(supervise_valid[global_idx]),
                    "nearest_official_exp_index": int(nearest_exp_idx[global_idx]),
                    "nearest_official_exp_time": float(exp_times[nearest_exp_idx[global_idx]]),
                    "nearest_official_delta_us": float(nearest_delta_us[global_idx]),
                    "target360_path": f"{dataset_root / 'targets360' / 'all.h5'}::row={global_idx}",
                    "target128_path": f"{dataset_root / 'targets128' / 'all.h5'}::row={global_idx}",
                    "mask360_path": f"{dataset_root / 'masks' / 'common_masks.h5'}::mask360_row={global_idx}",
                    "mask128_path": f"{dataset_root / 'masks' / 'common_masks.h5'}::mask128_row={global_idx}",
                    **path_fields,
                }
            )

        write_csv(meta_dir / "common_debug_index.csv", index_rows)
        write_csv(
            meta_dir / "debug_split.csv",
            [
                {
                    "split": "train",
                    "block_range": "0-389",
                    "sample_count": len(split_rows["train"]),
                },
                {
                    "split": "buffer",
                    "block_range": "390-399",
                    "sample_count": STEPS_PER_BLOCK * (BUFFER_BLOCK_MAX - TRAIN_BLOCK_MAX),
                },
                {
                    "split": "validation",
                    "block_range": "400-499",
                    "sample_count": len(split_rows["validation"]),
                },
            ],
            fieldnames=["split", "block_range", "sample_count"],
        )

        generation_config = {
            "sequence_name": SEQUENCE_NAME,
            "source_count_h5": str(source_count_h5),
            "source_s2_h5": str(source_s2_h5),
            "raw_data_h5": str(raw_data_h5),
            "official_exp_h5": str(official_exp_h5),
            "num_blocks": NUM_BLOCKS,
            "steps_per_block": STEPS_PER_BLOCK,
            "target_key": TARGET_KEY,
            "mask_key": MASK_KEY,
            "target_definition": "signed inverse TTC, fixed 10ms pose window, value = Tz / depth",
            "iir128_policy": "S2中心256视场原始事件直接映射到128x128后连续生成6通道IIR；不读取360图像下采样",
            "count10k128_policy": "复用已有 N10000_S2_center256_to128.h5 的2通道事件计数、标签和mask",
            "training_mask": "dense_valid_mask & event_active_mask",
        }
        (meta_dir / "generation_config.yaml").write_text(
            "\n".join(f"{k}: {v}" for k, v in generation_config.items()) + "\n",
            encoding="utf-8",
        )

        train_count = len(split_rows["train"])
        val_count = len(split_rows["validation"])
        h5_handles = {
            ("EVSlim_IIR360", "train"): build_h5_file(dataset_root / "iir360" / "train.h5", train_count, 6, 360),
            ("EVSlim_IIR360", "validation"): build_h5_file(dataset_root / "iir360" / "val.h5", val_count, 6, 360),
            ("EVSlim_IIR128", "train"): build_h5_file(dataset_root / "iir128" / "train.h5", train_count, 6, 128),
            ("EVSlim_IIR128", "validation"): build_h5_file(dataset_root / "iir128" / "val.h5", val_count, 6, 128),
            ("EVSlim_Count10k360", "train"): build_h5_file(dataset_root / "count10k360" / "train.h5", train_count, 2, 360),
            ("EVSlim_Count10k360", "validation"): build_h5_file(dataset_root / "count10k360" / "val.h5", val_count, 2, 360),
            ("EVSlim_Count10k128", "train"): build_h5_file(dataset_root / "count10k128" / "train.h5", train_count, 2, 128),
            ("EVSlim_Count10k128", "validation"): build_h5_file(dataset_root / "count10k128" / "val.h5", val_count, 2, 128),
        }
        local_rows = {"train": 0, "validation": 0}

        targets360 = h5py.File(dataset_root / "targets360" / "all.h5", "w")
        targets128 = h5py.File(dataset_root / "targets128" / "all.h5", "w")
        masks = h5py.File(dataset_root / "masks" / "common_masks.h5", "w")
        for f_t, res in ((targets360, 360), (targets128, 128)):
            f_t.attrs["complete"] = False
            f_t.create_dataset("signed_inverse_ttc", shape=(NUM_STEPS, res, res), dtype=np.float32, chunks=(1, res, res), **compression_kwargs())
            f_t.create_dataset("dense_valid_mask", shape=(NUM_STEPS, res, res), dtype=bool, chunks=(1, res, res), **compression_kwargs())
        masks.attrs["complete"] = False
        for key, res in (("dense_valid_mask360", 360), ("dense_valid_mask128", 128)):
            masks.create_dataset(key, shape=(NUM_STEPS, res, res), dtype=bool, chunks=(1, res, res), **compression_kwargs())

        geom = load_camera_geometry(f_raw)
        s2_iir_generator = NativeS2IIR128Generator(f_raw, f_s2, geom.dist_map)

        log("开始物化四组 EV-Slim debug H5，其中128组使用S2中心视场")
        try:
            for global_idx in range(NUM_STEPS):
                if global_idx % 100 == 0:
                    log(f"数据生成进度 {global_idx}/{NUM_STEPS}")
                local_block = global_idx // STEPS_PER_BLOCK
                split = split_name(local_block)

                count360 = f_src["event_cnt"][global_idx].astype(np.float32)
                target360 = f_src[TARGET_KEY][global_idx].astype(np.float32)
                dense360 = f_src[MASK_KEY][global_idx].astype(bool)
                count128 = f_s2["event_cnt"][global_idx].astype(np.float32)
                target128 = f_s2[TARGET_KEY][global_idx].astype(np.float32)
                dense128 = f_s2[MASK_KEY][global_idx].astype(bool)
                iir128 = s2_iir_generator.get(global_idx)

                targets360["signed_inverse_ttc"][global_idx] = target360
                targets360["dense_valid_mask"][global_idx] = dense360
                targets128["signed_inverse_ttc"][global_idx] = target128
                targets128["dense_valid_mask"][global_idx] = dense128
                masks["dense_valid_mask360"][global_idx] = dense360
                masks["dense_valid_mask128"][global_idx] = dense128

                if split not in ("train", "validation"):
                    continue

                meta360 = {
                    "global_index": global_idx,
                    "block_index": int(block_index[global_idx]),
                    "step_in_block": int(step_in_block[global_idx]),
                    "t_start": int(t_start[global_idx]),
                    "t_end": int(t_end_i64[global_idx]),
                    "raw_event_start_idx": int(raw_start[global_idx]),
                    "raw_event_end_idx": int(raw_end[global_idx]),
                }
                meta128 = {
                    "global_index": global_idx,
                    "block_index": int(block_index[global_idx]),
                    "step_in_block": int(step_in_block[global_idx]),
                    "t_start": int(s2_t_start[global_idx]),
                    "t_end": int(s2_t_end[global_idx]),
                    "raw_event_start_idx": int(s2_raw_start[global_idx]),
                    "raw_event_end_idx": int(s2_raw_end[global_idx]),
                }
                local_row = local_rows[split]

                iir360 = f_exp["exp_filts"][int(nearest_exp_idx[global_idx])].astype(np.float32)

                write_h5_row(
                    h5_handles[("EVSlim_IIR360", split)],
                    local_row,
                    iir360,
                    target360,
                    dense360,
                    event_active_mask(iir360, "iir"),
                    meta360,
                )
                write_h5_row(
                    h5_handles[("EVSlim_IIR128", split)],
                    local_row,
                    iir128,
                    target128,
                    dense128,
                    event_active_mask(iir128, "iir"),
                    meta128,
                )
                write_h5_row(
                    h5_handles[("EVSlim_Count10k360", split)],
                    local_row,
                    count360,
                    target360,
                    dense360,
                    event_active_mask(count360, "count"),
                    meta360,
                )
                write_h5_row(
                    h5_handles[("EVSlim_Count10k128", split)],
                    local_row,
                    count128,
                    target128,
                    dense128,
                    event_active_mask(count128, "count"),
                    meta128,
                )
                local_rows[split] += 1
        finally:
            for f in h5_handles.values():
                f.attrs["complete"] = True
                f.attrs["target_definition"] = "signed_inverse_ttc_fixed10ms"
                f.attrs["sequence_name"] = SEQUENCE_NAME
                f.close()
            for f in (targets360, targets128, masks):
                f.attrs["complete"] = True
                f.close()

    method_stats = {}
    for method, subdir in (
        ("EVSlim_IIR360", "iir360"),
        ("EVSlim_IIR128", "iir128"),
        ("EVSlim_Count10k360", "count10k360"),
        ("EVSlim_Count10k128", "count10k128"),
    ):
        stat_rows = []
        for split, filename in (("train", "train.h5"), ("validation", "val.h5")):
            h5_path = dataset_root / subdir / filename
            with h5py.File(h5_path, "r") as f:
                stat_rows.append(
                    {
                        "split": split,
                        "samples": int(f["exp_filts"].shape[0]),
                        "input_shape": list(f["exp_filts"].shape[1:]),
                        "input_min": float(np.min(f["exp_filts"][: min(32, f["exp_filts"].shape[0])])),
                        "input_max": float(np.max(f["exp_filts"][: min(32, f["exp_filts"].shape[0])])),
                        "target_min": float(np.min(f["ttc"][: min(32, f["ttc"].shape[0])])),
                        "target_max": float(np.max(f["ttc"][: min(32, f["ttc"].shape[0])])),
                        "train_mask_ratio": float(np.mean(f["mask"][: min(64, f["mask"].shape[0])])),
                        "dense_mask_ratio": float(np.mean(f["dense_valid_mask"][: min(64, f["dense_valid_mask"].shape[0])])),
                    }
                )
        method_stats[method] = stat_rows

    summary = {
        "complete": True,
        "generation_version": "s2_native_128_v2",
        "dataset_root": str(dataset_root),
        "source_count_h5": str(source_count_h5),
        "source_s2_h5": str(source_s2_h5),
        "raw_data_h5": str(raw_data_h5),
        "official_exp_h5": str(official_exp_h5),
        "sequence_name": SEQUENCE_NAME,
        "num_steps": NUM_STEPS,
        "splits": {"train": train_count, "buffer": 100, "validation": val_count},
        "nearest_official_delta_us": {
            "mean": float(np.mean(nearest_delta_us)),
            "std": float(np.std(nearest_delta_us)),
            "max_abs": float(np.max(np.abs(nearest_delta_us))),
            "p50_abs": float(np.quantile(np.abs(nearest_delta_us), 0.5)),
            "p95_abs": float(np.quantile(np.abs(nearest_delta_us), 0.95)),
        },
        "event_dt_us": {
            "mean": float(np.mean(event_dt)),
            "std": float(np.std(event_dt)),
            "min": int(np.min(event_dt)),
            "max": int(np.max(event_dt)),
        },
        "s2_event_dt_us": {
            "mean": float(np.mean(s2_event_dt)),
            "std": float(np.std(s2_event_dt)),
            "min": int(np.min(s2_event_dt)),
            "max": int(np.max(s2_event_dt)),
        },
        "s2_vs_s1_t_end_offset_us": {
            "mean": float(np.mean(s2_time_offset)),
            "std": float(np.std(s2_time_offset)),
            "min": float(np.min(s2_time_offset)),
            "max": float(np.max(s2_time_offset)),
            "p50_abs": float(np.quantile(np.abs(s2_time_offset), 0.5)),
            "p95_abs": float(np.quantile(np.abs(s2_time_offset), 0.95)),
        },
        "iir128_native": {
            "source": "raw M3ED events",
            "roi_in_360": list(S2_BOX_360),
            "alphas": IIR_ALPHAS.tolist(),
            "time_bin_us": IIR_TIME_BIN_US,
            "state": "continuous within selected 5000 S2 steps",
        },
        "count10k128_source": "N10000_S2_center256_to128.h5",
        "method_stats": method_stats,
    }
    write_json(generation_summary_path, summary)
    log("debug 数据生成完成")
    return summary


def refresh_existing_visuals(args: argparse.Namespace, method: MethodSpec, summary: dict[str, Any]) -> dict[str, Any]:
    """已有 checkpoint 不重训时，只按新的数量重画验证集可视化。"""

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = EVSlim(create_evslim_cfg(method.in_channels)).to(device)
    ckpt_path = Path(summary["checkpoints"]["best"])
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    val_ds = EVSlimH5Dataset(Path(summary["val_h5"]), preload=not args.no_preload)
    out_dir = Path(args.out_root) / method.name
    render_visualizations(model, method, val_ds, out_dir / "visualizations", device, max_images=args.num_visuals)
    summary["visualization_dir"] = str(out_dir / "visualizations")
    summary["num_visualizations"] = int(args.num_visuals)
    write_json(out_dir / "summary.json", summary)
    log(f"{method.name} 已刷新 {args.num_visuals} 张验证可视化")
    return summary


def train_one_method(args: argparse.Namespace, method: MethodSpec) -> dict[str, Any]:
    out_dir = Path(args.out_root) / method.name
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.json"
    if not args.force_train and summary_path.exists():
        try:
            summary = read_json(summary_path)
            if summary.get("complete"):
                if args.refresh_visuals:
                    return refresh_existing_visuals(args, method, summary)
                log(f"{method.name} 已有完整训练结果，跳过")
                return summary
        except Exception:
            pass

    dataset_subdir = {
        "EVSlim_IIR360": "iir360",
        "EVSlim_IIR128": "iir128",
        "EVSlim_Count10k360": "count10k360",
        "EVSlim_Count10k128": "count10k128",
    }[method.name]
    dataset_root = Path(args.dataset_root)
    train_h5 = dataset_root / dataset_subdir / "train.h5"
    val_h5 = dataset_root / dataset_subdir / "val.h5"

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = EVSlim(create_evslim_cfg(method.in_channels)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    train_ds = EVSlimH5Dataset(train_h5, preload=not args.no_preload)
    val_ds = EVSlimH5Dataset(val_h5, preload=not args.no_preload)
    train_loader = make_loader(train_ds, args.batch_size, shuffle=True)
    val_loader = make_loader(val_ds, args.batch_size, shuffle=False)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        total_steps=max(1, args.epochs * len(train_loader)),
        pct_start=args.pct_start,
        cycle_momentum=False,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    train_rows: list[dict[str, Any]] = []
    val_rows: list[dict[str, Any]] = []
    best_val_loss = float("inf")
    best_epoch = -1
    start_time = time.perf_counter()

    log(f"开始训练 {method.name}: epochs={args.epochs}, batch={args.batch_size}, device={device}")
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses: list[float] = []
        for batch in train_loader:
            x = batch["exp"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                pred = model(x)
                loss = masked_charbonnier_loss(pred, target, mask)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            epoch_losses.append(float(loss.detach().cpu().item()))

        val_metrics = evaluate_model(model, val_loader, device)
        train_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        val_loss = float(val_metrics["masked_charbonnier"])
        train_rows.append({"epoch": epoch, "train_loss": train_loss, "lr": float(scheduler.get_last_lr()[0])})
        val_row = {"epoch": epoch, "val_loss": val_loss, **{f"event_{k}": v for k, v in val_metrics["event_active"].items()}}
        val_row.update({f"dense_{k}": v for k, v in val_metrics["dense_valid"].items()})
        val_rows.append(val_row)
        log(f"{method.name} epoch {epoch:03d}/{args.epochs}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")

        ckpt = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "method": method.name,
            "config": vars(create_evslim_cfg(method.in_channels)),
        }
        torch.save(ckpt, out_dir / "last.pt")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(ckpt, out_dir / "best.pt")

    elapsed = time.perf_counter() - start_time
    write_csv(out_dir / "train_epoch_metrics.csv", train_rows)
    write_csv(out_dir / "val_epoch_metrics.csv", val_rows)

    best_ckpt = torch.load(out_dir / "best.pt", map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])
    final_metrics = evaluate_model(model, val_loader, device)
    efficiency = {
        "params": int(sum(p.numel() for p in model.parameters())),
        "macs_single_sample": int(count_conv_macs(model, method.in_channels, method.resolution, device)),
        **benchmark_inference(model, method.in_channels, method.resolution, device),
    }
    efficiency["train_seconds"] = elapsed

    plot_curves(out_dir / "train_val_curves.png", train_rows, val_rows, method.name)
    render_visualizations(model, method, val_ds, out_dir / "visualizations", device, max_images=args.num_visuals)

    summary = {
        "complete": True,
        "method": method.name,
        "display_name": method.display_name,
        "input_kind": method.input_kind,
        "in_channels": method.in_channels,
        "resolution": method.resolution,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "optimizer": "AdamW",
        "lr": args.lr,
        "scheduler": "OneCycleLR",
        "pct_start": args.pct_start,
        "loss": "masked Charbonnier alpha=0.45",
        "train_h5": str(train_h5),
        "val_h5": str(val_h5),
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "last_train_loss": train_rows[-1]["train_loss"] if train_rows else float("nan"),
        "last_val_loss": val_rows[-1]["val_loss"] if val_rows else float("nan"),
        "metrics": final_metrics,
        "efficiency": efficiency,
        "checkpoints": {"best": str(out_dir / "best.pt"), "last": str(out_dir / "last.pt")},
        "visualization_dir": str(out_dir / "visualizations"),
        "num_visualizations": int(args.num_visuals),
    }
    write_json(summary_path, summary)
    return summary


def plot_curves(path: Path, train_rows: list[dict[str, Any]], val_rows: list[dict[str, Any]], title: str) -> None:
    epochs = [int(row["epoch"]) for row in train_rows]
    train_loss = [float(row["train_loss"]) for row in train_rows]
    val_loss = [float(row["val_loss"]) for row in val_rows]
    plt.figure(figsize=(8, 4.8))
    plt.plot(epochs, train_loss, label="train masked Charbonnier")
    plt.plot(epochs, val_loss, label="val masked Charbonnier")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title(title)
    plt.grid(alpha=0.25)
    plt.legend()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def render_visualizations(
    model: nn.Module,
    method: MethodSpec,
    dataset: EVSlimH5Dataset,
    out_dir: Path,
    device: torch.device,
    *,
    max_images: int,
) -> None:
    """固定验证集锚点输出输入/GT/预测/误差/mask/直方图。"""

    out_dir.mkdir(parents=True, exist_ok=True)
    if len(dataset) == 0:
        return
    anchor_indices = np.linspace(0, len(dataset) - 1, num=min(max_images, len(dataset)), dtype=int)
    model.eval()
    with torch.no_grad():
        for local_idx in anchor_indices:
            item = dataset[int(local_idx)]
            x = item["exp"][None].to(device)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                pred = model(x)[0, 0].detach().float().cpu().numpy()
            inp = item["exp"].numpy()
            gt = item["target"][0].numpy()
            mask = item["mask"][0].numpy().astype(bool)
            dense = item["dense_mask"][0].numpy().astype(bool)
            err = np.abs(pred - gt)
            global_index = int(item["global_index"])
            save_visual_panel(out_dir / f"val_global_{global_index:05d}.png", method, inp, gt, pred, err, mask, dense)


def save_visual_panel(
    path: Path,
    method: MethodSpec,
    inp: np.ndarray,
    gt: np.ndarray,
    pred: np.ndarray,
    err: np.ndarray,
    mask: np.ndarray,
    dense: np.ndarray,
) -> None:
    if method.input_kind == "iir":
        input_panels = [(f"IIR ch{i}", inp[i]) for i in range(inp.shape[0])]
    else:
        diff = inp[0] - inp[1]
        input_panels = [("pos count", inp[0]), ("neg count", inp[1]), ("pos-neg", diff)]

    finite_gt = gt[np.isfinite(gt)]
    if finite_gt.size:
        vmin = float(np.quantile(finite_gt, 0.02))
        vmax = float(np.quantile(finite_gt, 0.98))
        if abs(vmax - vmin) < 1e-8:
            vmax = vmin + 1e-3
    else:
        vmin, vmax = -1.0, 1.0
    # 避免负 inverse TTC 被视觉范围截断。
    vmin = min(vmin, float(np.nanmin(pred)), float(np.nanmin(gt)))
    vmax = max(vmax, float(np.nanmax(pred)), float(np.nanmax(gt)))

    cols = 4
    rows = 3 if method.input_kind == "iir" else 2
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.2, rows * 3.7))
    axes = np.asarray(axes).reshape(rows, cols)
    for ax in axes.flat:
        ax.axis("off")

    for i, (title, arr) in enumerate(input_panels[: min(len(input_panels), rows * cols)]):
        ax = axes.flat[i]
        im = ax.imshow(arr, cmap="magma")
        ax.set_title(title)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    base = len(input_panels)
    panels = [
        ("GT signed inverse TTC", gt, "coolwarm", vmin, vmax),
        ("Prediction", pred, "coolwarm", vmin, vmax),
        ("Abs Error", np.where(mask, err, np.nan), "inferno", 0.0, float(np.nanquantile(np.where(mask, err, np.nan), 0.98)) if np.any(mask) else 1.0),
        ("event-active mask", mask.astype(np.float32), "gray", 0.0, 1.0),
        ("dense-valid mask", dense.astype(np.float32), "gray", 0.0, 1.0),
    ]
    for j, (title, arr, cmap, lo, hi) in enumerate(panels):
        idx = base + j
        if idx >= rows * cols:
            break
        ax = axes.flat[idx]
        im = ax.imshow(arr, cmap=cmap, vmin=lo, vmax=hi)
        ax.set_title(title)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    hist_ax = axes.flat[-1]
    hist_ax.axis("on")
    if np.any(mask):
        hist_ax.hist(gt[mask].ravel(), bins=60, alpha=0.55, label="GT")
        hist_ax.hist(pred[mask].ravel(), bins=60, alpha=0.55, label="Pred")
        hist_ax.legend(fontsize=8)
    hist_ax.set_title("masked histogram")
    fig.suptitle(f"{method.name} | GT range=({np.nanmin(gt):.4f},{np.nanmax(gt):.4f})")
    plt.tight_layout()
    plt.savefig(path, dpi=145)
    plt.close()


def load_method_summaries(out_root: Path) -> dict[str, dict[str, Any]]:
    summaries = {}
    for name in METHODS:
        path = out_root / name / "summary.json"
        if path.exists():
            summaries[name] = read_json(path)
    return summaries


def report_table_row(name: str, summary: dict[str, Any]) -> list[str]:
    event = summary["metrics"]["event_active"]
    dense = summary["metrics"]["dense_valid"]
    eff = summary["efficiency"]
    return [
        name,
        str(summary["epochs"]),
        str(summary["in_channels"]),
        f'{summary["resolution"]}x{summary["resolution"]}',
        format_float(summary["best_val_loss"], 5),
        format_float(event.get("mae"), 5),
        format_float(event.get("median_ae"), 5),
        format_float(event.get("mre"), 5),
        format_float(event.get("high_risk_mae_gt_inv_gt_0p5"), 5),
        format_float(event.get("positive_inv_ttc_mae"), 5),
        format_float(event.get("negative_inv_ttc_mae"), 5),
        format_float(event.get("sign_accuracy"), 4),
        format_float(event.get("prediction_std"), 5),
        format_float(event.get("edge_mae"), 5),
        format_float(event.get("gradient_mae"), 5),
        format_float(dense.get("mae"), 5),
        f'{eff.get("params", 0):,}',
        f'{eff.get("macs_single_sample", 0) / 1e9:.3f}G',
        format_float(eff.get("single_sample_ms"), 3),
        format_float(eff.get("cuda_peak_mb"), 1),
    ]


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    line1 = "| " + " | ".join(headers) + " |"
    line2 = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([line1, line2, *body])


def create_reports(args: argparse.Namespace, data_summary: dict[str, Any], summaries: dict[str, dict[str, Any]]) -> None:
    out_root = Path(args.out_root)
    report_dir = out_root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    headers = [
        "方法",
        "Epoch",
        "输入通道",
        "分辨率",
        "best val loss",
        "event MAE",
        "event median AE",
        "event MRE",
        "高风险MAE(inv>0.5)",
        "正inv MAE",
        "负inv MAE",
        "符号准确率",
        "预测std",
        "边缘MAE",
        "梯度MAE",
        "dense MAE",
        "参数量",
        "MACs",
        "单样本ms",
        "峰值显存MB",
    ]
    result_rows = [report_table_row(name, summaries[name]) for name in METHODS if name in summaries]
    result_table = markdown_table(headers, result_rows)

    best_by_event_mae = min(summaries.items(), key=lambda kv: kv[1]["metrics"]["event_active"].get("mae", float("inf")))[0]
    best_by_high = min(
        summaries.items(),
        key=lambda kv: kv[1]["metrics"]["event_active"].get("high_risk_mae_gt_inv_gt_0p5", float("inf")),
    )[0]
    best_by_edge = min(summaries.items(), key=lambda kv: kv[1]["metrics"]["event_active"].get("edge_mae", float("inf")))[0]

    iir360_macs = summaries["EVSlim_IIR360"]["efficiency"]["macs_single_sample"]
    iir128_macs = summaries["EVSlim_IIR128"]["efficiency"]["macs_single_sample"]
    count360_macs = summaries["EVSlim_Count10k360"]["efficiency"]["macs_single_sample"]
    count128_macs = summaries["EVSlim_Count10k128"]["efficiency"]["macs_single_sample"]
    iir_save = 100.0 * (1.0 - iir128_macs / max(iir360_macs, 1))
    count_save = 100.0 * (1.0 - count128_macs / max(count360_macs, 1))

    audit_md = f"""# EV-TTC官方代码与输入生成审计

## 结论

- 本阶段复用官方 `EV-Slim` 网络主体，第一层输入通道由参数控制；IIR 组使用 6 通道，Count10k 组使用 2 通道。
- 官方 IIR 输入直接读取已由 `EV-TTC-main/TTCEF/create_exp.py` 生成的 `exp_filts`，没有重新实现滤波公式。
- 128 组按最新要求修正：不再从 360 图像下采样输入。`EVSlim_IIR128` 从原始事件直接映射到 S2 中心 256 视场的 128x128 网格并连续生成 IIR；`EVSlim_Count10k128` 复用既有 `N10000_S2_center256_to128.h5`。
- 标签统一改为本项目当前 TTC-SNN 方向使用的 `signed inverse TTC = Tz / depth`，并统一使用 `{TARGET_KEY}`，因此本阶段不是论文 TTC 秒值正式复现。
- 官方 IIR 的输出时刻以 Count10k 500Block 的 `t_end` 为锚，匹配最近官方 `exp_times`。最近时间差绝对值 p50={format_float(data_summary['nearest_official_delta_us']['p50_abs'], 2)} us，p95={format_float(data_summary['nearest_official_delta_us']['p95_abs'], 2)} us，max={format_float(data_summary['nearest_official_delta_us']['max_abs'], 2)} us。

## 官方链路记录

- IIR 衰减常数：`calc_filt_constants(alphas, time_bins)`，每个时间 bin 使用 `alpha * (1-alpha)^(-j)`，保存帧间用 `(1-alpha)^time_bins` 衰减。
- 事件极性：正事件为 `+1`，负事件为 `-1`，写入同一 signed IIR 通道。
- 空间处理：官方先去畸变，再截取原始 x=[280,999]，坐标除以 2 得到 360x360。
- 默认输出间隔：官方 `create_exp.py` 使用 `frame_interval=0.007s` 保存 IIR 帧。
- 官方 GT：`calc_gt.py` 在 IIR 输出时刻上用 10ms 位姿间隔计算深度、TTC、flow、mask。
- 官方 merge 训练 mask：`gt_mask & abs(exp_last_channel)>1e-3 & finite(ttc)`，并过滤运动幅度和角速度。
- 官方 loss：masked Charbonnier，`alpha=0.45`，逐样本有效像素平均。
- 官方优化：`AdamW` + `OneCycleLR`，学习率 `2e-3`，`pct_start=0.1`，官方 YAML 的 `max_epochs=20`、`batch_size=128`。

## 本阶段适配

- EV-Slim 主体、ASPP、输出层不变；Count10k 只把第一层输入通道改为 2。
- 为保证四组公平对比，训练统一用 `dense_valid_mask & event_active_mask`。
- 128 版本使用 S2 中央 256 视场，因此不再与 360 版本构成“仅分辨率变化”的严格消融。
"""
    (report_dir / "EV-TTC官方代码与输入生成审计.md").write_text(audit_md, encoding="utf-8")

    stats_rows = []
    for method, rows in data_summary["method_stats"].items():
        for row in rows:
            stats_rows.append(
                [
                    method,
                    row["split"],
                    str(row["samples"]),
                    str(row["input_shape"]),
                    format_float(row["input_min"], 5),
                    format_float(row["input_max"], 5),
                    format_float(row["target_min"], 5),
                    format_float(row["target_max"], 5),
                    format_float(row["train_mask_ratio"], 5),
                    format_float(row["dense_mask_ratio"], 5),
                ]
            )
    data_md = f"""# EV-Slim四组Debug数据生成报告

## 数据来源

- 固定10k事件源：`{data_summary['source_count_h5']}`
- 官方 IIR 源：`{data_summary['official_exp_h5']}`
- 共用数据目录：`{data_summary['dataset_root']}`
- 序列：`{data_summary['sequence_name']}`

## 时间范围与划分

- 总样本：{data_summary['num_steps']}
- Train：{data_summary['splits']['train']} samples，Block 0-389
- Buffer：{data_summary['splits']['buffer']} samples，Block 390-399
- Validation：{data_summary['splits']['validation']} samples，Block 400-499
- 10k事件窗口 dt：均值 {format_float(data_summary['event_dt_us']['mean'], 2)} us，标准差 {format_float(data_summary['event_dt_us']['std'], 2)} us，范围 {data_summary['event_dt_us']['min']}~{data_summary['event_dt_us']['max']} us
- IIR 最近官方帧时间差：均值 {format_float(data_summary['nearest_official_delta_us']['mean'], 2)} us，标准差 {format_float(data_summary['nearest_official_delta_us']['std'], 2)} us，最大绝对值 {format_float(data_summary['nearest_official_delta_us']['max_abs'], 2)} us
- S2 128 与 S1 360 的输出时刻偏移：绝对值 p50={format_float(data_summary['s2_vs_s1_t_end_offset_us']['p50_abs'], 2)} us，p95={format_float(data_summary['s2_vs_s1_t_end_offset_us']['p95_abs'], 2)} us；说明本次 128 组与 360 组不是完全同一输出时刻。

## 物化数据统计

{markdown_table(['方法','split','样本数','input shape','input min','input max','target min','target max','train mask ratio','dense mask ratio'], stats_rows)}

## 输出文件

- `metadata/common_debug_index.csv`
- `metadata/debug_split.csv`
- `metadata/generation_config.yaml`
- `iir360/train.h5`, `iir360/val.h5`
- `iir128/train.h5`, `iir128/val.h5`
- `count10k360/train.h5`, `count10k360/val.h5`
- `count10k128/train.h5`, `count10k128/val.h5`
- `targets360/all.h5`, `targets128/all.h5`, `masks/common_masks.h5`
"""
    (report_dir / "EV-Slim四组Debug数据生成报告.md").write_text(data_md, encoding="utf-8")

    train_md = f"""# EV-Slim四组训练配置对照

## 共同设置

- 随机种子：{args.seed}
- Epoch：{args.epochs}
- Batch size：{args.batch_size}
- Optimizer：AdamW
- LR：{args.lr}
- Scheduler：OneCycleLR，pct_start={args.pct_start}
- Loss：masked Charbonnier，alpha=0.45
- 训练 mask：`dense_valid_mask & event_active_mask`
- Best 模型选择：Validation masked Charbonnier 最低
- 数据增强：本阶段关闭随机增强，以便四组可复现对照。
- AMP：CUDA 可用时启用 mixed precision。

## 四组差异

{markdown_table(['方法','输入表示','输入通道','分辨率','网络变化'], [
    ['EVSlim_IIR360','官方 signed IIR','6','360x360','官方 EV-Slim 主体不变'],
    ['EVSlim_IIR128','S2中心视场原生 signed IIR','6','128x128','输入从原始事件直接生成，不读取360图像下采样'],
    ['EVSlim_Count10k360','固定10k正负计数','2','360x360','仅第一层输入通道改为2'],
    ['EVSlim_Count10k128','既有S2中心视场固定10k正负计数','2','128x128','第一层输入通道改为2，复用S2 H5'],
])}

## 官方配置记录

- 官方 YAML：`EV-TTC-main/model/conf/config.yaml` 与 `EV-TTC-main/model/conf/models/evslim_ttc.yaml`
- 官方默认训练：20 epochs、batch size 128、learning rate 2e-3、precision 16-mixed。
- 本阶段为了快速诊断，四组使用同一套 fair comparison 配置；如果需要论文级正式复现，应单独运行官方 Lightning 训练链路。
"""
    (report_dir / "EV-Slim四组训练配置对照.md").write_text(train_md, encoding="utf-8")

    result_md = f"""# EV-Slim四组500Block结果对比

## 统一验证集指标

{result_table}

## 最优项

- event-active MAE 最低：`{best_by_event_mae}`
- 高风险区域 MAE 最低：`{best_by_high}`
- 边缘区域 MAE 最低：`{best_by_edge}`
- IIR 128 相对 IIR 360 的 MACs 节省：{format_float(iir_save, 2)}%
- Count10k 128 相对 Count10k 360 的 MACs 节省：{format_float(count_save, 2)}%

## 文件

- 每组曲线：`{out_root}/<method>/train_val_curves.png`
- 每组 checkpoint：`{out_root}/<method>/best.pt` 与 `last.pt`
- 每组可视化：`{out_root}/<method>/visualizations/`
"""
    (report_dir / "EV-Slim四组500Block结果对比.md").write_text(result_md, encoding="utf-8")

    conclusion_md = f"""# EV-Slim输入表示与分辨率消融结论

## 阶段A结论

- 局部 TTC 结构恢复最好的候选：`{best_by_edge}`。这里用 event-active 边缘 MAE 和梯度 MAE 作为结构清晰度代理指标。
- 高风险区域误差最低的候选：`{best_by_high}`。高风险在本阶段定义为 `signed inverse TTC > 0.5`，约等价于正向接近且 TTC 小于 2s。
- 整体 event-active MAE 最低的候选：`{best_by_event_mae}`。
- 128 输入的理论卷积 MACs 约为 360 输入的 {(iir128_macs / max(iir360_macs, 1)):.3f} 倍，计算量节省约 {format_float(iir_save, 2)}%。

## 对阶段B的含义

- 如果 IIR360 明显领先，后续应优先复现官方 EV-TTC，并把 SNN 迁移的目标设为学习官方 IIR 表示下的 signed inverse TTC。
- 如果 Count10k128 接近或优于 IIR128，说明固定事件数计数输入可作为 ANN/SNN 共用轻量数据接口。
- 如果 128 明显损失边缘结构，则后续 128 模型需要额外设计多尺度/局部增强，而不是只缩小输入。
- 阶段A只覆盖单条 car night 片段，不足以声称跨序列泛化；进入正式多序列阶段前仍需独立序列验证。

## 已知不一致

- IIR 输入来自官方 7ms 帧，Count10k 输入来自固定事件数窗口；本脚本用最近官方 `exp_times` 对齐 Count10k `t_end`，仍存在微小时间差。
- 128 组使用 S2 中心视场和 S2 自身固定10k时间轴，360组使用 S1 全视场时间轴；二者不再是严格同FOV、同输出时刻对照。
- 标签统一为 fixed10 signed inverse TTC，不是 EV-TTC 官方 merge 后的 TTC 秒值。
- 训练关闭随机增强，以优先验证输入表示和分辨率差异。
"""
    (report_dir / "EV-Slim输入表示与分辨率消融结论.md").write_text(conclusion_md, encoding="utf-8")

    mavlab_md = f"""# MAVLab_ANN与ANN-SNN接口审计

## event_flow-main 接口概况

- ANN 配置：`event_flow-main/configs/train_ANN.yml`
- SNN 配置：`event_flow-main/configs/train_SNN.yml`
- 默认输入：2 通道 `cnt`，正/负极性事件计数。
- 默认尺寸：128x128。
- 默认训练窗口：每次 forward 1000 events，loss 每 10000 events 汇总。
- 默认输出：光流网络原本输出二维 optical flow；迁移到 TTC 时需要替换为 1 通道 signed inverse TTC 输出头。
- 状态管理：ANN 无跨 step LIF 状态；SNN / ANN-SNN 需要显式 reset/detach 状态。

## 与本阶段 debug 数据的兼容性

- `count10k128` 与 MAVLab 的 2通道 128x128 count 输入最接近，可直接作为 ANN/SNN 共享入口。
- `count10k360` 需要确认 MAVLab U-Net/FireNet 是否允许任意分辨率；全卷积层通常可前向，但训练显存和可视化代码可能假定 128。
- `iir128` / `iir360` 是 6通道，MAVLab 原配置 `num_bins=2`，需要把第一层输入通道改为6或新增适配层。
- Loss 接口需从自监督 flow loss 改为 masked Charbonnier；本阶段 H5 已提供 `ttc`、`mask`、`dense_valid_mask`。
- 参数量会随第一层输入通道略有变化，但主体通道数不变。

## 后续建议

- ANN/SNN 共用训练优先从 `count10k128` 开始，因为它同时满足 2通道和128x128。
- 若要公平对比官方 IIR 表示，再扩展 `iir128` 的 6通道输入适配。
- 全 SNN 若继续沿用 10-step block 状态管理，可使用 `common_debug_index.csv` 中的 block 和 reset 字段。
"""
    (report_dir / "MAVLab_ANN与ANN-SNN接口审计.md").write_text(mavlab_md, encoding="utf-8")

    data_list_md = f"""# 后续模型共用Debug数据清单

## 根目录

`{data_summary['dataset_root']}`

## 共用索引

- `metadata/common_debug_index.csv`：四组输入、target、mask 的统一行索引。
- `metadata/debug_split.csv`：固定 Train / Buffer / Validation 划分。
- `metadata/generation_config.yaml`：生成口径记录。

## 可训练 H5

{markdown_table(['数据名','用途','路径'], [
    ['iir360 train','EVSlim_IIR360 训练', str(Path(data_summary['dataset_root']) / 'iir360' / 'train.h5')],
    ['iir360 val','EVSlim_IIR360 验证', str(Path(data_summary['dataset_root']) / 'iir360' / 'val.h5')],
    ['iir128 train','EVSlim_IIR128 训练', str(Path(data_summary['dataset_root']) / 'iir128' / 'train.h5')],
    ['iir128 val','EVSlim_IIR128 验证', str(Path(data_summary['dataset_root']) / 'iir128' / 'val.h5')],
    ['count10k360 train','EVSlim_Count10k360 / ANN 训练', str(Path(data_summary['dataset_root']) / 'count10k360' / 'train.h5')],
    ['count10k360 val','EVSlim_Count10k360 / ANN 验证', str(Path(data_summary['dataset_root']) / 'count10k360' / 'val.h5')],
    ['count10k128 train','EVSlim_Count10k128 / ANN-SNN 训练', str(Path(data_summary['dataset_root']) / 'count10k128' / 'train.h5')],
    ['count10k128 val','EVSlim_Count10k128 / ANN-SNN 验证', str(Path(data_summary['dataset_root']) / 'count10k128' / 'val.h5')],
])}

## 共用目标与Mask

- `targets360/all.h5`：5000 个 360x360 signed inverse TTC 标签。
- `targets128/all.h5`：5000 个 S2 中心视场 128x128 signed inverse TTC 标签。
- `masks/common_masks.h5`：dense-valid mask。

## 使用注意

- Buffer Block 390-399 只用于隔离 Train 与 Validation，不参与训练或验证。
- 跨 block 训练 SNN 时，每个连续片段起点必须按索引重置状态。
- IIR 组的事件表示与 Count10k 组的时间窗口不是完全同一种生成机制，正式结论需在阶段B继续验证。
"""
    (report_dir / "后续模型共用Debug数据清单.md").write_text(data_list_md, encoding="utf-8")

    combined = {
        "data_summary": data_summary,
        "method_summaries": summaries,
        "best_by_event_mae": best_by_event_mae,
        "best_by_high_risk_mae": best_by_high,
        "best_by_edge_mae": best_by_edge,
        "iir128_macs_saving_percent": iir_save,
        "count128_macs_saving_percent": count_save,
        "report_dir": str(report_dir),
    }
    write_json(out_root / "evslim_debug500_ablation_summary.json", combined)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行 EV-Slim 四组 500Block 快速对照实验")
    parser.add_argument("--source-count-h5", default=str(DEFAULT_SOURCE_COUNT_H5))
    parser.add_argument("--source-s2-h5", default=str(DEFAULT_SOURCE_S2_H5))
    parser.add_argument("--raw-data-h5", default=str(DEFAULT_RAW_DATA_H5))
    parser.add_argument("--official-exp-h5", default=str(DEFAULT_OFFICIAL_EXP_H5))
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--pct-start", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num-visuals", type=int, default=10)
    parser.add_argument("--force-data", action="store_true")
    parser.add_argument("--force-train", action="store_true")
    parser.add_argument("--refresh-visuals", action="store_true", help="不重训已有模型，仅按 num-visuals 重画验证可视化")
    parser.add_argument("--no-preload", action="store_true", help="关闭 H5 预加载，内存不足时使用")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--methods", nargs="*", default=list(METHODS.keys()), choices=list(METHODS.keys()))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    Path(args.out_root).mkdir(parents=True, exist_ok=True)
    data_summary = make_debug_data(args)
    summaries: dict[str, dict[str, Any]] = {}
    for name in args.methods:
        summaries[name] = train_one_method(args, METHODS[name])
    # 如果这次只指定了部分方法，报告仍尽量加载磁盘上已有的其他方法。
    summaries = {**load_method_summaries(Path(args.out_root)), **summaries}
    missing = [name for name in METHODS if name not in summaries]
    if missing:
        log(f"尚缺少方法结果，暂不生成总报告：{missing}")
    else:
        create_reports(args, data_summary, summaries)
        log(f"全部报告已生成：{Path(args.out_root) / 'reports'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
