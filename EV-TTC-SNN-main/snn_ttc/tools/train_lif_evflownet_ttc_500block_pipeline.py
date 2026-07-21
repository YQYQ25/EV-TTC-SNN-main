#!/usr/bin/env python3
"""LIF-EV-FlowNet-TTC 的 S2 N10k 500-Block 训练管线验证。

本脚本只验证训练管线，不做完整正式训练：
1. 使用同一条连续 500 Block 片段；
2. 按时间顺序划分 Train / Buffer / Validation；
3. 验证连续 SNN 状态管理、checkpoint 恢复、best 模型选择和基础可视化。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import h5py
import hdf5plugin  # noqa: F401  # 读取带压缩插件的 HDF5 时需要导入
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SNN_ROOT = PROJECT_ROOT / "EV-TTC-SNN-main"
if str(SNN_ROOT) not in sys.path:
    sys.path.insert(0, str(SNN_ROOT))

from snn_ttc.losses import (  # noqa: E402
    EVTTC_CHARBONNIER_ALPHA,
    EVTTC_CHARBONNIER_EPS,
    masked_charbonnier_per_sample,
)
from snn_ttc.models import LIFEVFlowNetTTC, LIFEVFlowNetTTCConfig  # noqa: E402
from snn_ttc.tools.audit_real_s2_lif_activity import LAYER_NAMES, fmt, tensor_stats  # noqa: E402


DEFAULT_H5 = (
    SNN_ROOT
    / "debug_sets"
    / "[5]car_spatial_ablation"
    / "independent_fixed_n"
    / "N10000_S2_center256_to128.h5"
)
DEFAULT_OUT_DIR = SNN_ROOT / "debug_sets" / "lif_evflownet_ttc_s2_n10k_500block_pipeline"
VIS_EPOCHS_BASE = {0, 10, 20, 25, 30, 40, 50}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """写入 CSV；字段取所有行字段并集，便于增量记录不同类型指标。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv_dicts(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def finite_mean(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size else float("nan")


def finite_min(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(arr.min()) if arr.size else float("nan")


def finite_max(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(arr.max()) if arr.size else float("nan")


def load_h5_meta(h5_path: Path, steps_per_block: int, num_blocks: int) -> dict[str, Any]:
    """读取用于切分和审计的轻量元数据。"""

    with h5py.File(h5_path, "r") as f:
        total_steps = int(f["event_cnt"].shape[0])
        if total_steps < num_blocks * steps_per_block:
            raise ValueError(f"H5 step 数不足：需要 {num_blocks * steps_per_block}，实际 {total_steps}")
        rows = np.arange(num_blocks * steps_per_block)
        meta = {
            "rows": rows,
            "block_index": f["block_index"][rows].astype(np.int64),
            "step_in_block": f["step_in_block"][rows].astype(np.int64),
            "step_index": f["step_index"][rows].astype(np.int64),
            "t_start": f["t_start"][rows].astype(np.int64),
            "t_end": f["t_end"][rows].astype(np.int64),
            "event_dt": f["event_dt"][rows].astype(np.int64),
            "timeout": f["timeout"][rows].astype(bool),
            "reset_required": f["reset_required"][rows].astype(bool),
            "supervise_valid": f["supervise_valid"][rows].astype(bool),
            "attrs": {key: str(value) for key, value in f.attrs.items()},
            "event_shape": tuple(f["event_cnt"].shape[1:]),
            "target_shape": tuple(f["signed_inverse_ttc"].shape[1:]),
        }
    return meta


def validate_500block_continuity(meta: dict[str, Any], steps_per_block: int, num_blocks: int) -> dict[str, Any]:
    """检查 500 个 block 是否严格连续；不通过时直接报错。"""

    block = meta["block_index"].reshape(num_blocks, steps_per_block)
    step = meta["step_in_block"].reshape(num_blocks, steps_per_block)
    expected_steps = np.arange(steps_per_block)
    complete_blocks = bool(np.all(step == expected_steps[None, :]))
    block_consistent = bool(np.all(block == block[:, :1]))
    block_consecutive = bool(np.all(np.diff(block[:, 0]) == 1))
    time_monotonic = bool(np.all(np.diff(meta["t_start"]) >= 0))
    no_timeout = not bool(meta["timeout"].any())
    # 第0个step允许作为片段起点；之后出现 reset 说明中间断裂。
    no_inner_reset = not bool(meta["reset_required"][1:].any())
    if not all([complete_blocks, block_consistent, block_consecutive, time_monotonic, no_timeout, no_inner_reset]):
        raise RuntimeError(
            "500Block连续性检查失败："
            f"complete={complete_blocks}, block_consistent={block_consistent}, "
            f"block_consecutive={block_consecutive}, time_monotonic={time_monotonic}, "
            f"no_timeout={no_timeout}, no_inner_reset={no_inner_reset}"
        )
    return {
        "complete_blocks": complete_blocks,
        "block_consistent": block_consistent,
        "block_consecutive": block_consecutive,
        "time_monotonic": time_monotonic,
        "no_timeout": no_timeout,
        "no_inner_reset": no_inner_reset,
        "mean_event_dt_us": float(np.mean(meta["event_dt"])),
        "std_event_dt_us": float(np.std(meta["event_dt"])),
        "max_event_dt_us": int(np.max(meta["event_dt"])),
    }


def build_split_rows(meta: dict[str, Any], steps_per_block: int, num_blocks: int) -> list[dict[str, Any]]:
    """按指导文件固定划分：0-389 训练，390-399 buffer，400-499 验证。"""

    rows: list[dict[str, Any]] = []
    block_indices = meta["block_index"].reshape(num_blocks, steps_per_block)
    step_indices = meta["step_index"].reshape(num_blocks, steps_per_block)
    t_start = meta["t_start"].reshape(num_blocks, steps_per_block)
    t_end = meta["t_end"].reshape(num_blocks, steps_per_block)
    reset = meta["reset_required"].reshape(num_blocks, steps_per_block)
    timeout = meta["timeout"].reshape(num_blocks, steps_per_block)

    for local_block in range(num_blocks):
        if local_block <= 389:
            split = "train"
        elif local_block <= 399:
            split = "buffer"
        else:
            split = "validation"
        rows.append(
            {
                "split": split,
                "local_block_index": local_block,
                "block_index": int(block_indices[local_block, 0]),
                "h5_row_start": local_block * steps_per_block,
                "h5_row_end": (local_block + 1) * steps_per_block - 1,
                "step_start": int(step_indices[local_block, 0]),
                "step_end": int(step_indices[local_block, -1]),
                "t_start": int(t_start[local_block, 0]),
                "t_end": int(t_end[local_block, -1]),
                "reset_required": int(bool(reset[local_block].any()) or bool(timeout[local_block].any())),
            }
        )
    return rows


class S2BlockDataset(Dataset):
    """每次返回一个完整 10-step Block，保持 `[T,2,128,128]` 输入结构。"""

    def __init__(self, h5_path: Path, block_rows: list[dict[str, Any]], steps_per_block: int) -> None:
        self.h5_path = h5_path
        self.block_rows = block_rows
        self.steps_per_block = steps_per_block
        row_ids: list[int] = []
        for item in block_rows:
            row_ids.extend(range(int(item["h5_row_start"]), int(item["h5_row_end"]) + 1))
        self.row_ids = np.asarray(row_ids, dtype=np.int64)
        self._load_arrays()

    def _load_arrays(self) -> None:
        with h5py.File(self.h5_path, "r") as f:
            rows = self.row_ids
            n_blocks = len(self.block_rows)
            t = self.steps_per_block
            self.events = f["event_cnt"][rows].astype(np.float32).reshape(n_blocks, t, 2, 128, 128)
            self.target = f["signed_inverse_ttc"][rows].astype(np.float32).reshape(n_blocks, t, 128, 128)
            self.mask = f["valid_ttc_mask"][rows].astype(bool).reshape(n_blocks, t, 128, 128)
            self.supervise_valid = f["supervise_valid"][rows].astype(bool).reshape(n_blocks, t)
            self.reset_required = f["reset_required"][rows].astype(bool).reshape(n_blocks, t)
            self.timeout = f["timeout"][rows].astype(bool).reshape(n_blocks, t)
            self.step_index = f["step_index"][rows].astype(np.int64).reshape(n_blocks, t)
            self.t_start = f["t_start"][rows].astype(np.int64).reshape(n_blocks, t)
            self.t_end = f["t_end"][rows].astype(np.int64).reshape(n_blocks, t)

    def __len__(self) -> int:
        return len(self.block_rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.block_rows[idx]
        # reset_required 是 block 级边界标记；step 级 reset 仍完整保留，供审计使用。
        block_reset = bool(row["reset_required"])
        return {
            "events": self.events[idx],
            "target": self.target[idx],
            "mask": self.mask[idx],
            "supervise_valid": self.supervise_valid[idx],
            "reset_required": self.reset_required[idx],
            "timeout": self.timeout[idx],
            "block_reset_required": np.asarray(block_reset, dtype=np.bool_),
            "block_index": np.asarray(row["block_index"], dtype=np.int64),
            "local_block_index": np.asarray(row["local_block_index"], dtype=np.int64),
            "step_indices": self.step_index[idx],
            "t_start": self.t_start[idx],
            "t_end": self.t_end[idx],
        }


def make_loader(dataset: S2BlockDataset, batch_size: int) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())


def state_to_mem_spike(state: Any) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """兼容 MAVLab encoder/residual 与 decoder 两类 LIF 状态布局。"""

    if state is None or not torch.is_tensor(state):
        return None, None
    if state.ndim == 6 and state.shape[0] >= 2 and state.shape[1] >= 2:
        stage_state = state[1]
        return stage_state[0].detach(), stage_state[1].detach()
    if state.ndim == 5 and state.shape[0] >= 2:
        return state[0].detach(), state[1].detach()
    return state.detach(), None


def layer_activity_rows(model: LIFEVFlowNetTTC, epoch: int, split: str, local_block_index: int) -> list[dict[str, Any]]:
    """记录 block 末尾状态，用于观察 scale=0.3 是否造成沉默或饱和。"""

    rows: list[dict[str, Any]] = []
    for idx, state in enumerate(model.multires_unetrec.states):
        layer = LAYER_NAMES[idx] if idx < len(LAYER_NAMES) else f"lif_{idx}"
        mem, spike = state_to_mem_spike(state)
        if mem is None:
            continue
        mem_s = tensor_stats(mem)
        if spike is None:
            spike_rate = silent = high = float("nan")
        else:
            spike_bool = spike != 0
            spike_rate = float(spike_bool.float().mean().detach().cpu())
            per_channel = spike_bool.float().flatten(2).mean(dim=2)
            silent = float((per_channel == 0).float().mean().detach().cpu())
            high = float((per_channel > 0.8).float().mean().detach().cpu())
        rows.append(
            {
                "epoch": epoch,
                "split": split,
                "local_block_index": local_block_index,
                "layer": layer,
                "spike_rate": spike_rate,
                "silent_ratio": silent,
                "high_firing_ratio": high,
                "mem_mean": mem_s["mean"],
                "mem_std": mem_s["std"],
                "mem_abs_p95": mem_s["abs_p95"],
                "mem_abs_p99": mem_s["abs_p99"],
                "finite_ratio": mem_s["finite_ratio"],
            }
        )
    return rows


def aggregate_activity(epoch: int, split: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_layer: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_layer.setdefault(str(row["layer"]), []).append(row)
    out: list[dict[str, Any]] = []
    for layer, vals in sorted(by_layer.items(), key=lambda item: LAYER_NAMES.index(item[0]) if item[0] in LAYER_NAMES else 99):
        out.append(
            {
                "epoch": epoch,
                "split": split,
                "layer": layer,
                "mean_spike_rate": finite_mean([to_float(v["spike_rate"]) for v in vals]),
                "mean_silent_ratio": finite_mean([to_float(v["silent_ratio"]) for v in vals]),
                "mean_high_firing_ratio": finite_mean([to_float(v["high_firing_ratio"]) for v in vals]),
                "mean_mem_mean": finite_mean([to_float(v["mem_mean"]) for v in vals]),
                "mean_mem_std": finite_mean([to_float(v["mem_std"]) for v in vals]),
                "max_mem_abs_p99": finite_max([to_float(v["mem_abs_p99"]) for v in vals]),
                "mean_finite_ratio": finite_mean([to_float(v["finite_ratio"]) for v in vals]),
            }
        )
    return out


class MetricAccumulator:
    """把一个 epoch 内的 loss、误差、符号和监督统计聚合到同一行。"""

    def __init__(self) -> None:
        self.block_loss_values: list[float] = []
        self.grad_norms: list[float] = []
        self.abs_errors: list[torch.Tensor] = []
        self.pos_abs_errors: list[torch.Tensor] = []
        self.neg_abs_errors: list[torch.Tensor] = []
        self.high_risk_abs_errors: list[torch.Tensor] = []
        self.pred_step_means: list[float] = []
        self.pred_step_stds: list[float] = []
        self.pred_mins: list[float] = []
        self.pred_maxs: list[float] = []
        self.total_blocks = 0
        self.total_steps = 0
        self.forward_steps = 0
        self.valid_step_samples = 0
        self.valid_pixels = 0
        self.zero_supervision_blocks = 0
        self.optimizer_steps = 0
        self.sign_correct = 0
        self.sign_total = 0
        self.nan_inf_count = 0

    def add_prediction_stats(self, pred: torch.Tensor) -> None:
        self.forward_steps += int(pred.shape[0])
        finite = torch.isfinite(pred)
        self.nan_inf_count += int((~finite).sum().detach().cpu())
        if bool(finite.any().detach().cpu()):
            vals = pred[finite].detach().float()
            self.pred_step_means.append(float(vals.mean().cpu()))
            self.pred_step_stds.append(float(vals.std(unbiased=False).cpu()))
            self.pred_mins.append(float(vals.min().cpu()))
            self.pred_maxs.append(float(vals.max().cpu()))

    def add_error_regions(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, supervise: torch.Tensor) -> tuple[int, int]:
        valid = mask.bool() & supervise.bool()[:, None, None] & torch.isfinite(pred[:, 0]) & torch.isfinite(target)
        count = int(valid.sum().detach().cpu())
        if count == 0:
            return 0, 0
        err = (pred[:, 0] - target).abs()
        valid_err = err[valid].detach().cpu()
        self.abs_errors.append(valid_err)
        self.valid_pixels += count

        pos = valid & (target > 0)
        neg = valid & (target < 0)
        high = valid & (target > 1.0)
        if bool(pos.any().detach().cpu()):
            self.pos_abs_errors.append(err[pos].detach().cpu())
        if bool(neg.any().detach().cpu()):
            self.neg_abs_errors.append(err[neg].detach().cpu())
        if bool(high.any().detach().cpu()):
            self.high_risk_abs_errors.append(err[high].detach().cpu())

        sign_mask = valid & (target != 0)
        sign_count = int(sign_mask.sum().detach().cpu())
        if sign_count:
            self.sign_total += sign_count
            self.sign_correct += int(((pred[:, 0][sign_mask] > 0) == (target[sign_mask] > 0)).sum().detach().cpu())
        return count, sign_count

    def tensor_mean(self, chunks: list[torch.Tensor]) -> float:
        if not chunks:
            return float("nan")
        return float(torch.cat(chunks).float().mean())

    def tensor_median(self, chunks: list[torch.Tensor]) -> float:
        if not chunks:
            return float("nan")
        return float(torch.cat(chunks).float().median())

    def epoch_row(self, *, epoch: int, split: str) -> dict[str, Any]:
        return {
            "epoch": epoch,
            "split": split,
            "charbonnier": finite_mean(self.block_loss_values),
            "masked_MAE": self.tensor_mean(self.abs_errors),
            "median_AE": self.tensor_median(self.abs_errors),
            "positive_inverse_ttc_MAE": self.tensor_mean(self.pos_abs_errors),
            "negative_inverse_ttc_MAE": self.tensor_mean(self.neg_abs_errors),
            "sign_accuracy": (self.sign_correct / self.sign_total) if self.sign_total else float("nan"),
            "high_risk_inverse_gt_gt1_MAE": self.tensor_mean(self.high_risk_abs_errors),
            "total_blocks": self.total_blocks,
            "total_steps": self.total_steps,
            "forward_steps": self.forward_steps,
            "valid_step_sample_count": self.valid_step_samples,
            "valid_pixel_count": self.valid_pixels,
            "zero_supervision_blocks": self.zero_supervision_blocks,
            "optimizer_update_count": self.optimizer_steps,
            "mean_grad_norm": finite_mean(self.grad_norms),
            "pred_mean": finite_mean(self.pred_step_means),
            "pred_std": finite_mean(self.pred_step_stds),
            "pred_min": finite_min(self.pred_mins),
            "pred_max": finite_max(self.pred_maxs),
            "nan_inf_count": self.nan_inf_count,
            "finite_ok": int(self.nan_inf_count == 0),
        }


def run_epoch(
    model: LIFEVFlowNetTTC,
    loader: DataLoader,
    *,
    epoch: int,
    split: str,
    device: torch.device,
    input_scale: float,
    optimizer: torch.optim.Optimizer | None,
    gradient_clip: float,
    global_step: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], int]:
    """训练或验证一个 epoch；验证时 optimizer=None。"""

    is_train = optimizer is not None
    model.train(is_train)
    model.reset_states()
    metrics = MetricAccumulator()
    block_rows: list[dict[str, Any]] = []
    activity_raw_rows: list[dict[str, Any]] = []

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for batch_id, batch in enumerate(loader):
            local_block = int(batch["local_block_index"][0].item())
            if batch_id == 0 or bool(batch["block_reset_required"][0].item()):
                model.reset_states()
            if is_train:
                optimizer.zero_grad(set_to_none=True)

            metrics.total_blocks += 1
            metrics.total_steps += int(batch["events"].shape[1])
            loss_sum: torch.Tensor | None = None
            valid_step_samples = 0
            block_abs_errors: list[torch.Tensor] = []
            block_valid_pixels = 0
            block_pred_stats: list[dict[str, float]] = []

            for step_id in range(batch["events"].shape[1]):
                events_t = batch["events"][:, step_id].to(device, non_blocking=True).float() * input_scale
                target_t = batch["target"][:, step_id].to(device, non_blocking=True).float()
                mask_t = batch["mask"][:, step_id].to(device, non_blocking=True).bool()
                supervise_t = batch["supervise_valid"][:, step_id].to(device, non_blocking=True).bool()

                # 无效监督 step 仍然 forward，确保 SNN 状态按真实时间推进。
                pred_t = model(events_t)
                metrics.add_prediction_stats(pred_t)
                per_sample_loss, valid_samples, stats = masked_charbonnier_per_sample(pred_t, target_t, mask_t, supervise_t)
                if bool(valid_samples.any().detach().cpu()):
                    current_sum = per_sample_loss[valid_samples].sum()
                    loss_sum = current_sum if loss_sum is None else loss_sum + current_sum
                    valid_step_samples += int(valid_samples.sum().detach().cpu())
                    valid_mask = mask_t & supervise_t[:, None, None]
                    err = (pred_t[:, 0] - target_t).abs()[valid_mask]
                    block_abs_errors.append(err.detach().cpu())
                    block_valid_pixels += int(stats.total_valid_pixels)
                    metrics.add_error_regions(pred_t.detach(), target_t.detach(), mask_t.detach(), supervise_t.detach())
                block_pred_stats.append(
                    {
                        "mean": finite_mean([float(pred_t.detach().mean().cpu())]),
                        "std": finite_mean([float(pred_t.detach().std(unbiased=False).cpu())]),
                        "min": float(pred_t.detach().min().cpu()),
                        "max": float(pred_t.detach().max().cpu()),
                    }
                )

            block_loss_value = float("nan")
            grad_norm = float("nan")
            backward_called = False
            optimizer_step_called = False
            if valid_step_samples > 0 and loss_sum is not None:
                block_loss = loss_sum / valid_step_samples
                block_loss_value = float(block_loss.detach().cpu())
                metrics.block_loss_values.append(block_loss_value)
                if is_train:
                    block_loss.backward()
                    backward_called = True
                    grad_norm_tensor = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip)
                    grad_norm = float(grad_norm_tensor.detach().cpu())
                    metrics.grad_norms.append(grad_norm)
                    if math.isfinite(grad_norm):
                        optimizer.step()
                        optimizer_step_called = True
                        metrics.optimizer_steps += 1
                        global_step += 1
            else:
                metrics.zero_supervision_blocks += 1

            metrics.valid_step_samples += valid_step_samples
            activity_raw_rows.extend(layer_activity_rows(model, epoch, split, local_block))
            model.detach_states()

            if block_abs_errors:
                block_err = torch.cat(block_abs_errors)
                block_mae = float(block_err.mean())
                block_median = float(block_err.median())
            else:
                block_mae = float("nan")
                block_median = float("nan")

            block_rows.append(
                {
                    "epoch": epoch,
                    "split": split,
                    "local_block_index": local_block,
                    "source_block_index": int(batch["block_index"][0].item()),
                    "block_loss": block_loss_value,
                    "masked_MAE": block_mae,
                    "median_AE": block_median,
                    "valid_step_sample_count": valid_step_samples,
                    "valid_pixel_count": block_valid_pixels,
                    "grad_norm": grad_norm,
                    "backward_called": int(backward_called),
                    "optimizer_step_called": int(optimizer_step_called),
                    "pred_mean": finite_mean([x["mean"] for x in block_pred_stats]),
                    "pred_std": finite_mean([x["std"] for x in block_pred_stats]),
                    "pred_min": finite_min([x["min"] for x in block_pred_stats]),
                    "pred_max": finite_max([x["max"] for x in block_pred_stats]),
                }
            )

    model.reset_states()
    return metrics.epoch_row(epoch=epoch, split=split), block_rows, aggregate_activity(epoch, split, activity_raw_rows), global_step


def save_checkpoint(
    path: Path,
    model: LIFEVFlowNetTTC,
    optimizer: torch.optim.Optimizer,
    *,
    epoch: int,
    global_step: int,
    best_val_mae: float,
    best_val_loss: float,
    args: argparse.Namespace,
    split_rows: list[dict[str, Any]],
) -> None:
    """保存恢复训练所需状态；不保存 SNN hidden state。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "best_val_mae": best_val_mae,
            "best_val_loss": best_val_loss,
            "model_config": asdict(LIFEVFlowNetTTCConfig()),
            "data_split": split_rows,
            "input_scale": args.input_scale,
            "random_seed": args.seed,
            "optimizer_config": {"name": "AdamW", "lr": args.lr, "weight_decay": args.weight_decay},
            "loss_config": {
                "type": "masked_charbonnier_per_sample",
                "alpha": EVTTC_CHARBONNIER_ALPHA,
                "epsilon": EVTTC_CHARBONNIER_EPS,
                "smoothness_weight": 0.0,
            },
            "gradient_clip": args.gradient_clip,
        },
        path,
    )


def load_checkpoint(
    checkpoint: Path,
    model: LIFEVFlowNetTTC,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, Any]:
    ckpt = torch.load(checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt


def save_visualization(
    out_path: Path,
    *,
    epoch: int,
    split: str,
    anchor_name: str,
    block_index: int,
    step_id: int,
    event: torch.Tensor,
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    event_np = event.detach().cpu().float().numpy()
    pred_np = pred.detach().cpu().float().numpy()[0, 0]
    target_np = target.detach().cpu().float().numpy()[0]
    mask_np = mask.detach().cpu().bool().numpy()[0]
    err_np = np.abs(pred_np - target_np) * mask_np.astype(np.float32)

    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    panels = [
        (event_np[0, 0], "positive events", "magma"),
        (event_np[0, 1], "negative events", "magma"),
        (pred_np, "prediction signed inv TTC", "coolwarm"),
        (target_np, "GT signed inv TTC", "coolwarm"),
        (mask_np.astype(np.float32), "valid mask", "gray"),
        (err_np, "abs error", "magma"),
    ]
    for ax, (arr, title, cmap) in zip(axes.flat, panels):
        if title == "valid mask":
            vmin, vmax = 0.0, 1.0
        else:
            finite = np.isfinite(arr)
            if finite.any():
                vmin, vmax = np.percentile(arr[finite], [1, 99])
                if abs(float(vmax) - float(vmin)) < 1e-12:
                    vmax = vmin + 1e-6
            else:
                vmin, vmax = 0.0, 1.0
        im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    fig.suptitle(f"epoch={epoch} | {split} {anchor_name} | block={block_index} step={step_id}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def capture_anchor_visuals(
    model: LIFEVFlowNetTTC,
    loader: DataLoader,
    *,
    epoch: int,
    split: str,
    anchors: dict[str, tuple[int, int]],
    out_dir: Path,
    device: torch.device,
    input_scale: float,
) -> None:
    """从 split 起点 reset 后按时间重放，保存固定锚点预测图。"""

    model.eval()
    model.reset_states()
    remaining = dict(anchors)
    max_block = max(block for block, _ in anchors.values())
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
                        save_visualization(
                            out_dir / "visualizations" / f"epoch_{epoch:03d}" / split / f"{name}_block{local_block:03d}_step{step_id:02d}.png",
                            epoch=epoch,
                            split=split,
                            anchor_name=name,
                            block_index=local_block,
                            step_id=step_id,
                            event=event_raw,
                            pred=pred_t,
                            target=target_t,
                            mask=mask_t,
                        )
                        del remaining[name]
                if not remaining and local_block >= max_block:
                    break
            model.detach_states()
            if not remaining and local_block >= max_block:
                break
    model.reset_states()


def plot_curves(out_dir: Path, train_rows: list[dict[str, Any]], val_rows: list[dict[str, Any]]) -> None:
    epochs = [int(r["epoch"]) for r in train_rows]
    val_by_epoch = {int(r["epoch"]): r for r in val_rows}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(epochs, [to_float(r["charbonnier"]) for r in train_rows], label="train")
    axes[0].plot(epochs, [to_float(val_by_epoch[e]["charbonnier"]) for e in epochs if e in val_by_epoch], label="val")
    axes[0].set_title("Charbonnier loss")
    axes[1].plot(epochs, [to_float(r["masked_MAE"]) for r in train_rows], label="train")
    axes[1].plot(epochs, [to_float(val_by_epoch[e]["masked_MAE"]) for e in epochs if e in val_by_epoch], label="val")
    axes[1].set_title("Masked MAE")
    axes[2].plot(epochs, [to_float(r["pred_std"]) for r in train_rows], label="train pred std")
    axes[2].plot(epochs, [to_float(val_by_epoch[e]["pred_std"]) for e in epochs if e in val_by_epoch], label="val pred std")
    axes[2].set_title("Prediction std")
    for ax in axes:
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.25)
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "train_val_curves.png", dpi=160)
    plt.close(fig)


def write_config(out_dir: Path, args: argparse.Namespace, continuity: dict[str, Any], shape_info: dict[str, Any]) -> None:
    text = f"""experiment: lif_evflownet_ttc_s2_n10k_500block_pipeline
h5: {args.h5}
spatial_scheme: S2_center256_to128
events_per_step: 10000
input_size: [128, 128]
input_channels: 2
input_scale: {args.input_scale}
steps_per_block: {args.steps_per_block}
num_blocks: {args.num_blocks}
batch_size: {args.batch_size}
shuffle: false
optimizer: AdamW
learning_rate: {args.lr}
weight_decay: {args.weight_decay}
gradient_clip: {args.gradient_clip}
max_epochs: {args.epochs}
loss:
  type: masked_charbonnier_per_sample
  alpha: {EVTTC_CHARBONNIER_ALPHA}
  epsilon: {EVTTC_CHARBONNIER_EPS}
  smoothness_weight: 0
split:
  train_blocks: [0, 389]
  buffer_blocks: [390, 399]
  validation_blocks: [400, 499]
continuity:
  complete_blocks: {continuity['complete_blocks']}
  block_consecutive: {continuity['block_consecutive']}
  time_monotonic: {continuity['time_monotonic']}
  no_timeout: {continuity['no_timeout']}
dataloader_shape:
  events: {shape_info['events']}
  target: {shape_info['target']}
  mask: {shape_info['mask']}
  supervise_valid: {shape_info['supervise_valid']}
"""
    (out_dir / "config.yaml").write_text(text, encoding="utf-8")


def summarize_final(train_rows: list[dict[str, Any]], val_rows: list[dict[str, Any]]) -> dict[str, Any]:
    train_by_epoch = {int(r["epoch"]): r for r in train_rows}
    val_by_epoch = {int(r["epoch"]): r for r in val_rows}
    initial_train = train_by_epoch[min(train_by_epoch)]
    final_train = train_by_epoch[max(train_by_epoch)]
    initial_val = val_by_epoch[min(val_by_epoch)]
    final_val = val_by_epoch[max(val_by_epoch)]
    best_mae_row = min(val_rows, key=lambda r: to_float(r["masked_MAE"]))
    best_loss_row = min(val_rows, key=lambda r: to_float(r["charbonnier"]))
    return {
        "initial_train": initial_train,
        "final_train": final_train,
        "initial_val": initial_val,
        "final_val": final_val,
        "best_val_mae": best_mae_row,
        "best_val_loss": best_loss_row,
        "train_loss_drop": 1.0 - to_float(final_train["charbonnier"]) / max(to_float(initial_train["charbonnier"]), 1e-12),
        "val_mae_drop": 1.0 - to_float(final_val["masked_MAE"]) / max(to_float(initial_val["masked_MAE"]), 1e-12),
    }


def write_reports(
    out_dir: Path,
    args: argparse.Namespace,
    split_rows: list[dict[str, Any]],
    continuity: dict[str, Any],
    shape_info: dict[str, Any],
    train_rows: list[dict[str, Any]],
    val_rows: list[dict[str, Any]],
    activity_rows: list[dict[str, Any]],
    recovery_info: dict[str, Any] | None,
) -> None:
    summary = summarize_final(train_rows, val_rows)
    init_train = summary["initial_train"]
    final_train = summary["final_train"]
    init_val = summary["initial_val"]
    final_val = summary["final_val"]
    best_mae = summary["best_val_mae"]
    best_loss = summary["best_val_loss"]
    final_activity = [r for r in activity_rows if int(r["epoch"]) == int(final_train["epoch"]) and r["split"] == "validation"]
    high_layers = [r["layer"] for r in final_activity if to_float(r["mean_high_firing_ratio"]) > 0.5 or to_float(r["mean_spike_rate"]) > 0.8]
    silent_decoders = [r["layer"] for r in final_activity if str(r["layer"]).startswith("dec") and to_float(r["mean_spike_rate"]) < 1e-4]
    checkpoint_ok = bool((out_dir / "checkpoints" / "best_val_mae.pt").exists() and (out_dir / "checkpoints" / "best_val_loss.pt").exists() and (out_dir / "checkpoints" / "latest.pt").exists())
    recovery_ok = bool(recovery_info and recovery_info.get("resume_success") and recovery_info.get("continued_epoch") in {11, 12})
    pipeline_pass = (
        to_float(summary["train_loss_drop"]) > 0
        and int(final_train["finite_ok"]) == 1
        and int(final_val["finite_ok"]) == 1
        and checkpoint_ok
        and recovery_ok
        and to_float(final_val["pred_std"]) > 1e-4
    )

    split_counts = {
        "train": sum(1 for r in split_rows if r["split"] == "train"),
        "buffer": sum(1 for r in split_rows if r["split"] == "buffer"),
        "validation": sum(1 for r in split_rows if r["split"] == "validation"),
    }

    main_report = f"""# 500Block训练管线验证报告

## 数据与划分
- H5：`{args.h5}`
- Train：Block 0-389，共 `{split_counts['train']}` 个 Block。
- Buffer：Block 390-399，共 `{split_counts['buffer']}` 个 Block，不参与训练和验证。
- Validation：Block 400-499，共 `{split_counts['validation']}` 个 Block。
- DataLoader events shape：`{shape_info['events']}`
- DataLoader target shape：`{shape_info['target']}`
- DataLoader mask shape：`{shape_info['mask']}`
- shuffle：`false`，训练和验证均按时间顺序。

## 连续性检查
- block 连续：`{continuity['block_consecutive']}`
- step 完整：`{continuity['complete_blocks']}`
- timestamp 单调：`{continuity['time_monotonic']}`
- timeout：`{not continuity['no_timeout']}`
- event dt 均值/标准差/最大值：`{fmt(continuity['mean_event_dt_us'])}` / `{fmt(continuity['std_event_dt_us'])}` / `{continuity['max_event_dt_us']}` us

## 训练配置
- 模型：LIF-EV-FlowNet-TTC，输入2通道，输出1通道 signed inverse TTC。
- 初始化：随机初始化，不加载32-Block过拟合权重。
- input_scale：`{args.input_scale}`，仅执行 `event_cnt.float() * {args.input_scale}`。
- optimizer：AdamW，lr=`{args.lr}`，weight_decay=`{args.weight_decay}`。
- gradient clip：`{args.gradient_clip}`。
- loss：masked Charbonnier per-sample，alpha=`{EVTTC_CHARBONNIER_ALPHA}`，epsilon=`{EVTTC_CHARBONNIER_EPS}`。

## 结果摘要
- 初始 train loss / MAE：`{fmt(init_train['charbonnier'])}` / `{fmt(init_train['masked_MAE'])}`
- 最终 train loss / MAE：`{fmt(final_train['charbonnier'])}` / `{fmt(final_train['masked_MAE'])}`
- 初始 val loss / MAE：`{fmt(init_val['charbonnier'])}` / `{fmt(init_val['masked_MAE'])}`
- 最终 val loss / MAE：`{fmt(final_val['charbonnier'])}` / `{fmt(final_val['masked_MAE'])}`
- Best Val MAE：epoch `{best_mae['epoch']}`，MAE=`{fmt(best_mae['masked_MAE'])}`
- Best Val Loss：epoch `{best_loss['epoch']}`，loss=`{fmt(best_loss['charbonnier'])}`
- Validation pred std：初始 `{fmt(init_val['pred_std'])}`，最终 `{fmt(final_val['pred_std'])}`

## 输出文件
- `config.yaml`
- `500block_split.csv`
- `train_epoch_metrics.csv`
- `train_block_metrics.csv`
- `val_epoch_metrics.csv`
- `val_block_metrics.csv`
- `lif_activity_by_epoch.csv`
- `checkpoints/`
- `visualizations/`
- `train_val_curves.png`
"""
    (out_dir / "500Block训练管线验证报告.md").write_text(main_report, encoding="utf-8")

    metric_report = f"""# 500Block训练验证指标分析

## Epoch 指标对比

| 指标 | 初始Train | 最终Train | 初始Val | 最终Val |
| --- | ---: | ---: | ---: | ---: |
| Charbonnier | {fmt(init_train['charbonnier'])} | {fmt(final_train['charbonnier'])} | {fmt(init_val['charbonnier'])} | {fmt(final_val['charbonnier'])} |
| Masked MAE | {fmt(init_train['masked_MAE'])} | {fmt(final_train['masked_MAE'])} | {fmt(init_val['masked_MAE'])} | {fmt(final_val['masked_MAE'])} |
| Median AE | {fmt(init_train['median_AE'])} | {fmt(final_train['median_AE'])} | {fmt(init_val['median_AE'])} | {fmt(final_val['median_AE'])} |
| Sign Accuracy | {fmt(init_train['sign_accuracy'])} | {fmt(final_train['sign_accuracy'])} | {fmt(init_val['sign_accuracy'])} | {fmt(final_val['sign_accuracy'])} |
| Pred std | {fmt(init_train['pred_std'])} | {fmt(final_train['pred_std'])} | {fmt(init_val['pred_std'])} | {fmt(final_val['pred_std'])} |

## Signed inverse TTC 分区
- 正 inverse TTC 区域 MAE：Train `{fmt(final_train['positive_inverse_ttc_MAE'])}`，Val `{fmt(final_val['positive_inverse_ttc_MAE'])}`。
- 负 inverse TTC 区域 MAE：Train `{fmt(final_train['negative_inverse_ttc_MAE'])}`，Val `{fmt(final_val['negative_inverse_ttc_MAE'])}`。
- 高风险 inverse TTC > 1 区域 MAE：Train `{fmt(final_train['high_risk_inverse_gt_gt1_MAE'])}`，Val `{fmt(final_val['high_risk_inverse_gt_gt1_MAE'])}`。

## 监督与稳定性
- 最终 Train forward step：`{final_train['forward_steps']}`，有效 step-sample：`{final_train['valid_step_sample_count']}`，零监督 Block：`{final_train['zero_supervision_blocks']}`。
- 最终 Val forward step：`{final_val['forward_steps']}`，有效 step-sample：`{final_val['valid_step_sample_count']}`，零监督 Block：`{final_val['zero_supervision_blocks']}`。
- Train NaN/Inf：`{final_train['nan_inf_count']}`；Val NaN/Inf：`{final_val['nan_inf_count']}`。
- 高发放层：`{', '.join(high_layers) if high_layers else '无'}`。
- 沉默 decoder 层：`{', '.join(silent_decoders) if silent_decoders else '无'}`。

## 判断
- Train loss 变化：`{fmt(summary['train_loss_drop'] * 100)}`%。
- Val MAE 相对随机初始化变化：`{fmt(summary['val_mae_drop'] * 100)}`%。
- Validation 预测是否接近常数：`{'否' if to_float(final_val['pred_std']) > 1e-4 else '是，需复查'}`。
"""
    (out_dir / "500Block训练验证指标分析.md").write_text(metric_report, encoding="utf-8")

    if recovery_info is None:
        recovery_info = {}
    recovery_report = f"""# 500BlockCheckpoint恢复审计

## 恢复测试
- 是否执行两阶段运行：`{bool(recovery_info)}`
- 恢复 checkpoint：`{recovery_info.get('resume_checkpoint', 'NA')}`
- 恢复前 epoch：`{recovery_info.get('loaded_epoch', 'NA')}`
- 恢复后继续 epoch：`{recovery_info.get('continued_epoch', 'NA')}`
- optimizer state 是否加载：`{recovery_info.get('optimizer_state_loaded', 'NA')}`
- global_step 恢复值：`{recovery_info.get('loaded_global_step', 'NA')}`
- 恢复后 global_step：`{recovery_info.get('continued_global_step', 'NA')}`
- epoch10 train loss：`{fmt(recovery_info.get('epoch10_train_loss', float('nan')))}`
- epoch11 train loss：`{fmt(recovery_info.get('epoch11_train_loss', float('nan')))}`
- 恢复是否通过：`{recovery_ok}`

## Checkpoint内容
保存字段包含：`model_state_dict`、`optimizer_state_dict`、`epoch`、`global_step`、`best_val_mae`、`best_val_loss`、模型配置、数据划分、`input_scale`、随机种子。

不保存 SNN hidden state；恢复后从新 epoch 起点执行 `model.reset_states()`。
"""
    (out_dir / "500BlockCheckpoint恢复审计.md").write_text(recovery_report, encoding="utf-8")

    decision_report = f"""# 500Block训练管线通过或失败判定

## 总判定
- 结果：`{'通过' if pipeline_pass else '未完全通过'}`

## 最终必须回答
1. 500个Block如何划分Train、Buffer和Validation：Train 0-389，Buffer 390-399，Validation 400-499。
2. Dataset/DataLoader输出shape是否正确：正确，events `{shape_info['events']}`，target `{shape_info['target']}`，mask `{shape_info['mask']}`。
3. 训练和验证是否按时间顺序：是，DataLoader `shuffle=false`。
4. Train与Validation之间是否正确reset状态：是，Validation 前后均 `model.reset_states()`，不继承 Train 状态。
5. 无效监督step是否仍执行forward：是，所有 step forward；无效监督只是不参与 loss/backward。
6. Loss和MAE是否稳定下降：Train 下降 `{fmt(summary['train_loss_drop'] * 100)}`%；Val MAE 相对初始化变化 `{fmt(summary['val_mae_drop'] * 100)}`%。
7. Validation预测是否明显优于随机初始化：`{'是' if to_float(final_val['masked_MAE']) < to_float(init_val['masked_MAE']) and to_float(final_val['pred_std']) > 1e-4 else '未明显优于，需结合可视化复查'}`。
8. Best模型是否按Validation MAE正确保存：`{'是' if checkpoint_ok else '否'}`，`best_val_mae.pt` 对应 epoch `{best_mae['epoch']}`。
9. Checkpoint恢复后训练是否连续：`{'是' if recovery_ok else '否'}`。
10. scale=0.3是否仍无明显饱和或沉默：`{'是' if not high_layers and not silent_decoders else '存在迹象，需复查'}`。
11. 当前训练管线是否满足完整S2正式训练要求：`{'满足' if pipeline_pass else '暂未完全满足'}`。
12. 下一步是否应物化完整S2并进行正式多序列训练：若继续推进，应先物化完整S2并做多序列训练/验证；本脚本已按要求停止，没有自动启动完整S2训练。
"""
    (out_dir / "500Block训练管线通过或失败判定.md").write_text(decision_report, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", type=Path, default=DEFAULT_H5)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--input-scale", type=float, default=0.3)
    parser.add_argument("--steps-per-block", type=int, default=10)
    parser.add_argument("--num-blocks", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--gradient-clip", type=float, default=100.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", type=Path, default=None)
    args = parser.parse_args()

    if args.batch_size != 1:
        raise ValueError("500Block管线验证固定 batch_size=1，避免跨样本状态混合。")
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    out_dir = args.out_dir
    for sub in ["checkpoints", "visualizations", "logs"]:
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    meta = load_h5_meta(args.h5, args.steps_per_block, args.num_blocks)
    continuity = validate_500block_continuity(meta, args.steps_per_block, args.num_blocks)
    split_rows = build_split_rows(meta, args.steps_per_block, args.num_blocks)
    write_csv(out_dir / "500block_split.csv", split_rows)

    train_blocks = [r for r in split_rows if r["split"] == "train"]
    val_blocks = [r for r in split_rows if r["split"] == "validation"]
    train_loader = make_loader(S2BlockDataset(args.h5, train_blocks, args.steps_per_block), args.batch_size)
    val_loader = make_loader(S2BlockDataset(args.h5, val_blocks, args.steps_per_block), args.batch_size)
    first_batch = next(iter(train_loader))
    shape_info = {
        "events": list(first_batch["events"].shape),
        "target": list(first_batch["target"].shape),
        "mask": list(first_batch["mask"].shape),
        "supervise_valid": list(first_batch["supervise_valid"].shape),
    }
    write_config(out_dir, args, continuity, shape_info)

    model = LIFEVFlowNetTTC().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_epoch_rows = read_csv_dicts(out_dir / "train_epoch_metrics.csv") if args.resume else []
    train_block_rows = read_csv_dicts(out_dir / "train_block_metrics.csv") if args.resume else []
    val_epoch_rows = read_csv_dicts(out_dir / "val_epoch_metrics.csv") if args.resume else []
    val_block_rows = read_csv_dicts(out_dir / "val_block_metrics.csv") if args.resume else []
    activity_rows = read_csv_dicts(out_dir / "lif_activity_by_epoch.csv") if args.resume else []

    start_epoch = 1
    global_step = 0
    best_val_mae = float("inf")
    best_val_loss = float("inf")
    recovery_info: dict[str, Any] | None = None

    if args.resume:
        ckpt = load_checkpoint(args.resume, model, optimizer, device)
        start_epoch = int(ckpt["epoch"]) + 1
        global_step = int(ckpt["global_step"])
        best_val_mae = float(ckpt.get("best_val_mae", best_val_mae))
        best_val_loss = float(ckpt.get("best_val_loss", best_val_loss))
        model.reset_states()
        recovery_info = {
            "resume_checkpoint": str(args.resume),
            "loaded_epoch": int(ckpt["epoch"]),
            "loaded_global_step": global_step,
            "optimizer_state_loaded": bool(ckpt.get("optimizer_state_dict")),
            "resume_success": True,
        }
    else:
        save_checkpoint(
            out_dir / "checkpoints" / "epoch_000.pt",
            model,
            optimizer,
            epoch=0,
            global_step=0,
            best_val_mae=best_val_mae,
            best_val_loss=best_val_loss,
            args=args,
            split_rows=split_rows,
        )
        # Epoch 0 只做随机初始化基线评估，不更新参数。
        tr0, tb0, ta0, global_step = run_epoch(
            model,
            train_loader,
            epoch=0,
            split="train",
            device=device,
            input_scale=args.input_scale,
            optimizer=None,
            gradient_clip=args.gradient_clip,
            global_step=global_step,
        )
        va0, vb0, aa0, global_step = run_epoch(
            model,
            val_loader,
            epoch=0,
            split="validation",
            device=device,
            input_scale=args.input_scale,
            optimizer=None,
            gradient_clip=args.gradient_clip,
            global_step=global_step,
        )
        train_epoch_rows.append(tr0)
        train_block_rows.extend(tb0)
        val_epoch_rows.append(va0)
        val_block_rows.extend(vb0)
        activity_rows.extend(ta0)
        activity_rows.extend(aa0)
        capture_anchor_visuals(
            model,
            train_loader,
            epoch=0,
            split="train",
            anchors={"train_start": (0, 0), "train_end": (389, 9)},
            out_dir=out_dir,
            device=device,
            input_scale=args.input_scale,
        )
        capture_anchor_visuals(
            model,
            val_loader,
            epoch=0,
            split="validation",
            anchors={"val_start": (400, 0), "val_middle": (450, 5), "val_end": (499, 9)},
            out_dir=out_dir,
            device=device,
            input_scale=args.input_scale,
        )

    visual_epochs = {e for e in VIS_EPOCHS_BASE if start_epoch <= e <= args.epochs}
    visual_epochs.add(args.epochs)

    for epoch in range(start_epoch, args.epochs + 1):
        train_row, train_blocks_rows, train_activity, global_step = run_epoch(
            model,
            train_loader,
            epoch=epoch,
            split="train",
            device=device,
            input_scale=args.input_scale,
            optimizer=optimizer,
            gradient_clip=args.gradient_clip,
            global_step=global_step,
        )
        val_row, val_blocks_rows, val_activity, global_step = run_epoch(
            model,
            val_loader,
            epoch=epoch,
            split="validation",
            device=device,
            input_scale=args.input_scale,
            optimizer=None,
            gradient_clip=args.gradient_clip,
            global_step=global_step,
        )
        train_epoch_rows.append(train_row)
        train_block_rows.extend(train_blocks_rows)
        val_epoch_rows.append(val_row)
        val_block_rows.extend(val_blocks_rows)
        activity_rows.extend(train_activity)
        activity_rows.extend(val_activity)

        val_mae = to_float(val_row["masked_MAE"])
        val_loss = to_float(val_row["charbonnier"])
        # 先更新两个best标量，再保存checkpoint，避免best_val_mae.pt中的best_val_loss仍是旧值。
        improved_mae = math.isfinite(val_mae) and val_mae < best_val_mae
        improved_loss = math.isfinite(val_loss) and val_loss < best_val_loss
        if improved_mae:
            best_val_mae = val_mae
        if improved_loss:
            best_val_loss = val_loss
        if improved_mae:
            save_checkpoint(
                out_dir / "checkpoints" / "best_val_mae.pt",
                model,
                optimizer,
                epoch=epoch,
                global_step=global_step,
                best_val_mae=best_val_mae,
                best_val_loss=best_val_loss,
                args=args,
                split_rows=split_rows,
            )
        if improved_loss:
            save_checkpoint(
                out_dir / "checkpoints" / "best_val_loss.pt",
                model,
                optimizer,
                epoch=epoch,
                global_step=global_step,
                best_val_mae=best_val_mae,
                best_val_loss=best_val_loss,
                args=args,
                split_rows=split_rows,
            )
        if epoch % 10 == 0 or epoch == args.epochs:
            save_checkpoint(
                out_dir / "checkpoints" / f"epoch_{epoch:03d}.pt",
                model,
                optimizer,
                epoch=epoch,
                global_step=global_step,
                best_val_mae=best_val_mae,
                best_val_loss=best_val_loss,
                args=args,
                split_rows=split_rows,
            )
        save_checkpoint(
            out_dir / "checkpoints" / "latest.pt",
            model,
            optimizer,
            epoch=epoch,
            global_step=global_step,
            best_val_mae=best_val_mae,
            best_val_loss=best_val_loss,
            args=args,
            split_rows=split_rows,
        )

        if epoch in visual_epochs:
            capture_anchor_visuals(
                model,
                train_loader,
                epoch=epoch,
                split="train",
                anchors={"train_start": (0, 0), "train_end": (389, 9)},
                out_dir=out_dir,
                device=device,
                input_scale=args.input_scale,
            )
            capture_anchor_visuals(
                model,
                val_loader,
                epoch=epoch,
                split="validation",
                anchors={"val_start": (400, 0), "val_middle": (450, 5), "val_end": (499, 9)},
                out_dir=out_dir,
                device=device,
                input_scale=args.input_scale,
            )

        if recovery_info is not None and epoch == start_epoch:
            recovery_info.update(
                {
                    "continued_epoch": epoch,
                    "continued_global_step": global_step,
                    "epoch10_train_loss": next((to_float(r["charbonnier"]) for r in train_epoch_rows if int(r["epoch"]) == 10), float("nan")),
                    "epoch11_train_loss": to_float(train_row["charbonnier"]) if epoch == 11 else float("nan"),
                }
            )
            write_json(out_dir / "logs" / "checkpoint_resume_audit.json", recovery_info)

        write_csv(out_dir / "train_epoch_metrics.csv", train_epoch_rows)
        write_csv(out_dir / "train_block_metrics.csv", train_block_rows)
        write_csv(out_dir / "val_epoch_metrics.csv", val_epoch_rows)
        write_csv(out_dir / "val_block_metrics.csv", val_block_rows)
        write_csv(out_dir / "lif_activity_by_epoch.csv", activity_rows)
        plot_curves(out_dir, train_epoch_rows, val_epoch_rows)

    # 额外保存 Best Val MAE 对应 epoch 的固定锚点可视化；若它已经是0/10/20等常规epoch，则跳过。
    if val_epoch_rows:
        best_vis_epoch = int(min(val_epoch_rows, key=lambda r: to_float(r["masked_MAE"]))["epoch"])
        best_vis_dir = out_dir / "visualizations" / f"epoch_{best_vis_epoch:03d}"
        if not best_vis_dir.exists() and (out_dir / "checkpoints" / "best_val_mae.pt").exists():
            best_ckpt = torch.load(out_dir / "checkpoints" / "best_val_mae.pt", map_location=device)
            model.load_state_dict(best_ckpt["model_state_dict"])
            model.reset_states()
            capture_anchor_visuals(
                model,
                train_loader,
                epoch=best_vis_epoch,
                split="train",
                anchors={"train_start": (0, 0), "train_end": (389, 9)},
                out_dir=out_dir,
                device=device,
                input_scale=args.input_scale,
            )
            capture_anchor_visuals(
                model,
                val_loader,
                epoch=best_vis_epoch,
                split="validation",
                anchors={"val_start": (400, 0), "val_middle": (450, 5), "val_end": (499, 9)},
                out_dir=out_dir,
                device=device,
                input_scale=args.input_scale,
            )

    if recovery_info is None and (out_dir / "logs" / "checkpoint_resume_audit.json").exists():
        with (out_dir / "logs" / "checkpoint_resume_audit.json").open("r", encoding="utf-8") as handle:
            recovery_info = json.load(handle)
    write_reports(out_dir, args, split_rows, continuity, shape_info, train_epoch_rows, val_epoch_rows, activity_rows, recovery_info)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
