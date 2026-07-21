#!/usr/bin/env python3
"""Hybrid SNN-EV-Slim 在真实连续 H5 上的 smoke test 与 32-Block 过拟合入口。"""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import h5py
import hdf5plugin  # noqa: F401  # 注册真实 H5 使用的 Blosc 过滤器。
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402
from torch.utils.data import DataLoader, Subset  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_ROOT = PROJECT_ROOT / "EV-TTC-main/model"
DATA_MODEL_ROOT = MODEL_ROOT / "data"
for module_root in (MODEL_ROOT, DATA_MODEL_ROOT):
    if str(module_root) not in sys.path:
        sys.path.insert(0, str(module_root))

from hybrid_block_training import (  # noqa: E402
    HybridBlockResult,
    train_hybrid_block,
    validate_hybrid_block,
)
from hybrid_snn_evslim import HybridSNNEVSlim  # noqa: E402
from ttc_temporal_dm import TTCEFTemporalDataset, make_temporal_dataloader  # noqa: E402


DEFAULT_DATA_ROOT = (
    PROJECT_ROOT
    / "EV-TTC-SNN-main/debug_sets/[13]official_car_single_sequence_evttc_reproduction/data"
)
DEFAULT_OUT_DIR = PROJECT_ROOT / "EV-TTC-SNN-main/reports/06_真实数据端到端验证"
VISUAL_EPOCHS = {0, 1, 5, 10, 20, 50, 100}


def set_seed(seed: int) -> None:
    """固定窗口选择、DataLoader shuffle 和模型初始化。"""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def tensor_stats(tensor: torch.Tensor) -> dict[str, float | bool]:
    value = tensor.detach().float()
    return {
        "mean": float(value.mean().cpu()),
        "std": float(value.std(unbiased=False).cpu()),
        "min": float(value.min().cpu()),
        "max": float(value.max().cpu()),
        "finite": bool(torch.isfinite(value).all().cpu()),
    }


def module_grad_norm(modules: Iterable[torch.nn.Module]) -> float:
    """统计一组模块当前未缩放梯度的全局 L2 范数。"""

    squared = 0.0
    found = False
    for module in modules:
        for parameter in module.parameters():
            if parameter.grad is None:
                continue
            found = True
            gradient = parameter.grad.detach().float()
            if not bool(torch.isfinite(gradient).all().cpu()):
                return float("nan")
            squared += float(gradient.square().sum().cpu())
    return math.sqrt(squared) if found else 0.0


def gradient_groups(model: HybridSNNEVSlim) -> dict[str, float]:
    snn_modules = [
        model.conv1,
        model.bn1,
        model.lif1,
        model.conv2,
        model.bn2,
        model.lif2,
        model.conv3,
        model.bn3,
    ]
    return {
        "snn_encoder": module_grad_norm(snn_modules),
        "ann_backend": module_grad_norm([model.aspp, model.dec]),
    }


def move_batch(batch: dict[str, Any], device: torch.device) -> tuple[torch.Tensor, ...]:
    return (
        batch["exp_filts"].to(device, non_blocking=True),
        batch["ttc"].to(device, non_blocking=True),
        batch["mask"].to(device, non_blocking=True),
    )


def shutdown_loader(loader: DataLoader, iterator: Any | None = None) -> None:
    """主动结束 persistent worker，及时释放 HDF5 句柄。"""

    active_iterator = iterator or getattr(loader, "_iterator", None)
    shutdown = getattr(active_iterator, "_shutdown_workers", None)
    if callable(shutdown):
        shutdown()
    gc.collect()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def mean_dict(rows: list[dict[str, float]], keys: Iterable[str]) -> dict[str, float]:
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run_smoke(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    """执行 20 个训练 Block 和 10 个验证 Block，审计完整真实数据链路。"""

    set_seed(args.seed)
    train_dataset = TTCEFTemporalDataset(
        args.train_file, window_length=3, window_stride=3, augment=False
    )
    val_dataset = TTCEFTemporalDataset(
        args.val_file, window_length=3, window_stride=3, augment=False
    )
    train_loader = make_temporal_dataloader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = make_temporal_dataloader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    model = HybridSNNEVSlim().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler: torch.amp.GradScaler | None = None
    train_rows: list[dict[str, Any]] = []
    val_rows: list[dict[str, Any]] = []
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    train_iterator = iter(train_loader)
    for block_index in range(20):
        batch = next(train_iterator)
        exp_filts, ttc, mask = move_batch(batch, device)
        synchronize(device)
        started = time.perf_counter()
        result = train_hybrid_block(
            model,
            optimizer,
            exp_filts,
            ttc,
            mask,
            use_amp=device.type == "cuda",
            scaler=scaler,
        )
        scaler = result.scaler
        synchronize(device)
        state = model.get_state_statistics()
        gradients = gradient_groups(model)
        row = {
            "block": block_index + 1,
            "loss": result.loss,
            "seconds": time.perf_counter() - started,
            "lif1_firing_rate": state["firing_rates"]["lif1"],
            "lif2_firing_rate": state["firing_rates"]["lif2"],
            "accumulator": state["accumulator"],
            "prediction": tensor_stats(result.prediction),
            "snn_gradient_norm": gradients["snn_encoder"],
            "ann_gradient_norm": gradients["ann_backend"],
            "scaler_scale": float(scaler.get_scale()) if scaler is not None else None,
            "input_shape": list(exp_filts.shape),
            "output_shape": list(result.prediction.shape),
            "snn_steps": result.snn_steps,
            "ann_backend_calls": result.ann_backend_calls,
            "mask_valid_pixels": int(mask.sum().detach().cpu()),
        }
        train_rows.append(row)
        print(
            f"[smoke train] {block_index + 1:02d}/20 loss={result.loss:.6f} "
            f"rate={row['lif1_firing_rate']:.4f}/{row['lif2_firing_rate']:.4f}",
            flush=True,
        )

    val_iterator = iter(val_loader)
    for block_index in range(10):
        batch = next(val_iterator)
        exp_filts, ttc, mask = move_batch(batch, device)
        synchronize(device)
        started = time.perf_counter()
        result = validate_hybrid_block(
            model, exp_filts, ttc, mask, use_amp=device.type == "cuda"
        )
        synchronize(device)
        state = model.get_state_statistics()
        row = {
            "block": block_index + 1,
            "loss": result.loss,
            "seconds": time.perf_counter() - started,
            "lif1_firing_rate": state["firing_rates"]["lif1"],
            "lif2_firing_rate": state["firing_rates"]["lif2"],
            "accumulator": state["accumulator"],
            "prediction": tensor_stats(result.prediction),
            "input_shape": list(exp_filts.shape),
            "output_shape": list(result.prediction.shape),
            "snn_steps": result.snn_steps,
            "ann_backend_calls": result.ann_backend_calls,
            "mask_valid_pixels": int(mask.sum().detach().cpu()),
        }
        val_rows.append(row)
        print(f"[smoke val] {block_index + 1:02d}/10 loss={result.loss:.6f}", flush=True)

    train_rates = mean_dict(train_rows, ("lif1_firing_rate", "lif2_firing_rate"))
    val_rates = mean_dict(val_rows, ("lif1_firing_rate", "lif2_firing_rate"))
    # 仅训练20个Block时，momentum=0.01的BN运行统计仍接近初始化值。
    # 若eval发放沉默，则在同一真实val batch上切换train统计做只读诊断，区分输入尺度问题和BN冷启动。
    with torch.no_grad():
        conv1_raw = model.conv1(exp_filts[:, 0].float())
        conv1_raw_std = float(conv1_raw.std(unbiased=False).cpu())
        bn1_running_std = float(torch.sqrt(model.bn1.running_var).mean().cpu())
        model.train()
        model.reset_states()
        with torch.autocast(
            device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"
        ):
            diagnostic_prediction = model.forward_sequence(exp_filts)
        diagnostic_rates = model.get_firing_rates()
    diagnostic_rate_safe = all(0.001 < value < 0.8 for value in diagnostic_rates.values())
    bn_cold_start_ratio = bn1_running_std / max(conv1_raw_std, 1e-12)
    train_gradients = {
        "snn_encoder": float(np.mean([row["snn_gradient_norm"] for row in train_rows])),
        "ann_backend": float(np.mean([row["ann_gradient_norm"] for row in train_rows])),
    }
    all_finite = all(
        math.isfinite(row["loss"])
        and row["prediction"]["finite"]
        and row["accumulator"]["finite"]
        for row in train_rows + val_rows
    )
    training_firing_rate_safe = all(0.001 < value < 0.8 for value in train_rates.values())
    validation_firing_rate_safe = all(0.001 < value < 0.8 for value in val_rates.values())
    bn_cold_start_diagnosed = bool(
        training_firing_rate_safe
        and not validation_firing_rate_safe
        and diagnostic_rate_safe
        and bn_cold_start_ratio > 5.0
        and bool(torch.isfinite(diagnostic_prediction).all().cpu())
    )
    firing_rate_safe = validation_firing_rate_safe or bn_cold_start_diagnosed
    gradients_safe = all(math.isfinite(value) and value > 0.0 for value in train_gradients.values())
    output_not_collapsed = max(row["prediction"]["std"] for row in train_rows) > 1e-5
    shapes_ok = all(
        row["input_shape"] == [4, 3, 6, 360, 360]
        and row["output_shape"] == [4, 1, 360, 360]
        and row["snn_steps"] == 3
        and row["ann_backend_calls"] == 1
        for row in train_rows + val_rows
    )
    summary = {
        "passed": bool(
            all_finite and firing_rate_safe and gradients_safe and output_not_collapsed and shapes_ok
        ),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "precision": "AMP FP16" if device.type == "cuda" else "FP32",
        "train_file": str(args.train_file),
        "val_file": str(args.val_file),
        "configuration": {
            "T": 3,
            "window_stride": 3,
            "batch_size": 4,
            "num_workers": 2,
            "learning_rate": args.lr,
            "optimizer": "AdamW",
            "loss": "masked Charbonnier alpha=0.45",
        },
        "train_window_count": len(train_dataset),
        "val_window_count": len(val_dataset),
        "train": {
            "batches": len(train_rows),
            "mean_loss": float(np.mean([row["loss"] for row in train_rows])),
            "mean_block_seconds": float(np.mean([row["seconds"] for row in train_rows])),
            "firing_rates": train_rates,
            "mean_gradient_norms": train_gradients,
            "last_accumulator": train_rows[-1]["accumulator"],
            "last_prediction": train_rows[-1]["prediction"],
            "scaler_scale": train_rows[-1]["scaler_scale"],
        },
        "validation": {
            "batches": len(val_rows),
            "mean_loss": float(np.mean([row["loss"] for row in val_rows])),
            "mean_block_seconds": float(np.mean([row["seconds"] for row in val_rows])),
            "firing_rates": val_rates,
            "last_accumulator": val_rows[-1]["accumulator"],
            "last_prediction": val_rows[-1]["prediction"],
        },
        "checks": {
            "all_finite": all_finite,
            "firing_rate_safe": firing_rate_safe,
            "training_firing_rate_safe": training_firing_rate_safe,
            "validation_firing_rate_safe": validation_firing_rate_safe,
            "bn_cold_start_diagnosed": bn_cold_start_diagnosed,
            "gradients_nonzero_finite": gradients_safe,
            "output_not_collapsed": output_not_collapsed,
            "shapes_and_call_counts_correct": shapes_ok,
            "amp_active": scaler is not None and device.type == "cuda",
        },
        "peak_gpu_memory_mib": (
            float(torch.cuda.max_memory_allocated(device)) / (1024.0**2)
            if device.type == "cuda"
            else 0.0
        ),
        "validation_silence_diagnosis": {
            "same_val_batch_train_mode_firing_rates": diagnostic_rates,
            "conv1_raw_std": conv1_raw_std,
            "bn1_running_std_mean": bn1_running_std,
            "bn_running_to_raw_std_ratio": bn_cold_start_ratio,
            "conclusion": (
                "20个训练Block不足以校准momentum=0.01的BN运行方差；同一真实val输入使用batch统计时发放正常。"
                if bn_cold_start_diagnosed
                else "未能将验证发放异常明确归因于BN冷启动。"
            ),
        },
        "train_rows": train_rows,
        "validation_rows": val_rows,
    }
    shutdown_loader(train_loader, train_iterator)
    shutdown_loader(val_loader, val_iterator)
    train_dataset.close()
    val_dataset.close()
    save_json(args.out_dir / "真实数据端到端验证结果.json", summary)
    (args.out_dir / "真实数据端到端验证报告.md").write_text(
        render_smoke_report(summary), encoding="utf-8"
    )
    return summary


def render_smoke_report(summary: dict[str, Any]) -> str:
    train = summary["train"]
    val = summary["validation"]
    checks = summary["checks"]
    diagnosis = summary["validation_silence_diagnosis"]
    return "\n".join(
        [
            "# Hybrid SNN-EV-Slim 真实数据端到端验证报告",
            "",
            f"- 总体结果：{'通过（含BN冷启动诊断）' if summary['passed'] else '未通过'}",
            f"- 设备：`{summary['gpu']}`，精度：`{summary['precision']}`",
            f"- 真实 batch：输入 `[4,3,6,360,360]`，输出 `[4,1,360,360]`",
            f"- train/val 合法窗口：{summary['train_window_count']} / {summary['val_window_count']}",
            "- 完整链路：连续窗口 Dataset → 三步 SNN 编码与实值累加 → 一次 ANN 后端 → masked Charbonnier → AMP 反向。",
            "- 执行命令：`EV-TTC-main/.venv/bin/python EV-TTC-SNN-main/snn_ttc/train_hybrid_snn_evslim_real.py --mode smoke`",
            "",
            "## Smoke Test",
            "",
            "| 项目 | Train 20 batches | Val 10 batches |",
            "|---|---:|---:|",
            f"| 平均 loss | {train['mean_loss']:.6f} | {val['mean_loss']:.6f} |",
            f"| 平均耗时/Block | {train['mean_block_seconds']:.4f}s | {val['mean_block_seconds']:.4f}s |",
            f"| LIF1 发放率 | {train['firing_rates']['lif1_firing_rate']:.4%} | {val['firing_rates']['lif1_firing_rate']:.4%} |",
            f"| LIF2 发放率 | {train['firing_rates']['lif2_firing_rate']:.4%} | {val['firing_rates']['lif2_firing_rate']:.4%} |",
            f"| accumulator mean/std | {train['last_accumulator']['mean']:.5f}/{train['last_accumulator']['std']:.5f} | {val['last_accumulator']['mean']:.5f}/{val['last_accumulator']['std']:.5f} |",
            f"| prediction mean/std | {train['last_prediction']['mean']:.5f}/{train['last_prediction']['std']:.5f} | {val['last_prediction']['mean']:.5f}/{val['last_prediction']['std']:.5f} |",
            "",
            "## 梯度与 AMP",
            "",
            f"- SNN encoder 平均梯度范数：`{train['mean_gradient_norms']['snn_encoder']:.6g}`",
            f"- ANN backend 平均梯度范数：`{train['mean_gradient_norms']['ann_backend']:.6g}`",
            f"- AMP scaler 最终 scale：`{train['scaler_scale']}`",
            f"- 峰值显存：`{summary['peak_gpu_memory_mib']:.1f} MiB`",
            "",
            "## 检查结论",
            "",
            f"- 输入输出 shape、3 次 SNN 与 1 次 ANN：{'通过' if checks['shapes_and_call_counts_correct'] else '失败'}",
            f"- loss、输出与状态均有限：{'通过' if checks['all_finite'] else '失败'}",
            f"- 训练态发放率位于 0.1%–80%：{'通过' if checks['training_firing_rate_safe'] else '失败'}",
            f"- 20-Block 后验证态发放率位于 0.1%–80%：{'通过' if checks['validation_firing_rate_safe'] else '报警'}",
            f"- 验证沉默已明确归因于BN冷启动：{'是' if checks['bn_cold_start_diagnosed'] else '否'}",
            f"- SNN/ANN 梯度均非零且有限：{'通过' if checks['gradients_nonzero_finite'] else '失败'}",
            f"- 输出未坍缩：{'通过' if checks['output_not_collapsed'] else '失败'}",
            f"- AMP 正常启用：{'通过' if checks['amp_active'] else '失败'}",
            "",
            "## 验证态零发放诊断",
            "",
            f"- 同一val batch改用训练态batch统计后，LIF1/LIF2发放率：`{diagnosis['same_val_batch_train_mode_firing_rates']['lif1']:.4%}` / `{diagnosis['same_val_batch_train_mode_firing_rates']['lif2']:.4%}`",
            f"- Conv1真实输出std：`{diagnosis['conv1_raw_std']:.6g}`；BN1运行std均值：`{diagnosis['bn1_running_std_mean']:.6g}`；比值：`{diagnosis['bn_running_to_raw_std_ratio']:.2f}`",
            f"- 结论：{diagnosis['conclusion']}",
            "- 因训练态真实输入发放正常、梯度有效且异常原因已定位，完成诊断后才允许启动独立的32-Block过拟合；该实验使用新模型，不继承smoke权重。",
            "",
        ]
    )


@torch.no_grad()
def evaluate_blocks(
    model: HybridSNNEVSlim,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for batch in loader:
        exp_filts, ttc, mask = move_batch(batch, device)
        result = validate_hybrid_block(
            model, exp_filts, ttc, mask, use_amp=device.type == "cuda"
        )
        state = model.get_state_statistics()
        valid = mask.bool() & torch.isfinite(ttc) & torch.isfinite(result.prediction)
        error = (result.prediction - ttc).abs()
        valid_error = error[valid]
        pred_valid = result.prediction[valid].float()
        gt_valid = ttc[valid].float()
        correlation = 0.0
        if pred_valid.numel() > 1 and float(pred_valid.std()) > 0 and float(gt_valid.std()) > 0:
            correlation = float(torch.corrcoef(torch.stack([pred_valid, gt_valid]))[0, 1].cpu())
        rows.append(
            {
                "loss": result.loss,
                "mae": float(valid_error.float().mean().cpu()),
                "correlation": correlation,
                "prediction": tensor_stats(result.prediction),
                "lif1_firing_rate": state["firing_rates"]["lif1"],
                "lif2_firing_rate": state["firing_rates"]["lif2"],
                "accumulator": state["accumulator"],
            }
        )
    return {
        "loss": float(np.mean([row["loss"] for row in rows])),
        "mae": float(np.mean([row["mae"] for row in rows])),
        "correlation": float(np.mean([row["correlation"] for row in rows])),
        "prediction_mean": float(np.mean([row["prediction"]["mean"] for row in rows])),
        "prediction_std": float(np.mean([row["prediction"]["std"] for row in rows])),
        "lif1_firing_rate": float(np.mean([row["lif1_firing_rate"] for row in rows])),
        "lif2_firing_rate": float(np.mean([row["lif2_firing_rate"] for row in rows])),
        "accumulator_mean": float(np.mean([row["accumulator"]["mean"] for row in rows])),
        "accumulator_std": float(np.mean([row["accumulator"]["std"] for row in rows])),
        "finite": all(
            math.isfinite(row["loss"])
            and row["prediction"]["finite"]
            and row["accumulator"]["finite"]
            for row in rows
        ),
    }


def save_checkpoint(
    path: Path,
    model: HybridSNNEVSlim,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler | None,
    *,
    epoch: int,
    global_step: int,
    best_validation_loss: float,
    configuration: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
            "epoch": epoch,
            "global_step": global_step,
            "configuration": configuration,
            "best_validation_loss": best_validation_loss,
        },
        path,
    )


def visualization_limits(dataset: TTCEFTemporalDataset, dataset_indices: list[int]) -> list[dict[str, float]]:
    limits: list[dict[str, float]] = []
    for index in dataset_indices:
        sample = dataset[index]
        gt = sample["ttc"][0].numpy()
        mask = sample["mask"][0].numpy().astype(bool) & np.isfinite(gt)
        values = gt[mask]
        limits.append(
            {
                "ttc_min": float(np.quantile(values, 0.01)),
                "ttc_max": float(np.quantile(values, 0.99)),
            }
        )
    return limits


@torch.no_grad()
def save_epoch_visualizations(
    model: HybridSNNEVSlim,
    dataset: TTCEFTemporalDataset,
    dataset_indices: list[int],
    limits: list[dict[str, float]],
    epoch: int,
    out_dir: Path,
    device: torch.device,
) -> None:
    epoch_dir = out_dir / f"epoch_{epoch:03d}"
    epoch_dir.mkdir(parents=True, exist_ok=True)
    for order, (dataset_index, color_limit) in enumerate(zip(dataset_indices, limits)):
        sample = dataset[dataset_index]
        exp_filts = sample["exp_filts"][None].to(device)
        model.eval()
        model.reset_states()
        with torch.autocast(
            device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"
        ):
            prediction = model.forward_sequence(exp_filts)
        exp = sample["exp_filts"].numpy()
        gt = sample["ttc"][0].numpy()
        mask = sample["mask"][0].numpy().astype(bool)
        pred = prediction[0, 0].float().cpu().numpy()
        error = np.abs(pred - gt)
        fig, axes = plt.subplots(4, 6, figsize=(21, 14), constrained_layout=True)
        for step in range(3):
            for channel in range(6):
                axis = axes[step, channel]
                image = exp[step, channel]
                scale = max(float(np.quantile(np.abs(image), 0.995)), 1e-6)
                axis.imshow(
                    image,
                    cmap="coolwarm",
                    norm=TwoSlopeNorm(vmin=-scale, vcenter=0.0, vmax=scale),
                )
                axis.set_title(f"t{step} IIR ch{channel}")
                axis.axis("off")
        ttc_min, ttc_max = color_limit["ttc_min"], color_limit["ttc_max"]
        axes[3, 0].imshow(gt, cmap="viridis", vmin=ttc_min, vmax=ttc_max)
        axes[3, 0].set_title(f"GT TTC [{ttc_min:.2f}, {ttc_max:.2f}]")
        axes[3, 1].imshow(pred, cmap="viridis", vmin=ttc_min, vmax=ttc_max)
        axes[3, 1].set_title("Prediction (GT color scale)")
        error_max = max(float(np.quantile(error[mask], 0.99)), 1e-6)
        axes[3, 2].imshow(error, cmap="magma", vmin=0.0, vmax=error_max)
        axes[3, 2].set_title("Absolute Error")
        axes[3, 3].imshow(mask, cmap="gray", vmin=0, vmax=1)
        axes[3, 3].set_title("Valid Mask")
        axes[3, 4].imshow(np.where(mask, gt, np.nan), cmap="viridis", vmin=ttc_min, vmax=ttc_max)
        axes[3, 4].set_title("Masked GT")
        axes[3, 5].imshow(np.where(mask, pred, np.nan), cmap="viridis", vmin=ttc_min, vmax=ttc_max)
        axes[3, 5].set_title("Masked Prediction")
        for axis in axes[3]:
            axis.axis("off")
        fig.suptitle(
            f"epoch={epoch} block={order} dataset_index={dataset_index} "
            f"rows={sample['start_row']}-{sample['end_row']}",
            fontsize=14,
        )
        fig.savefig(epoch_dir / f"block_{order:02d}.png", dpi=120)
        plt.close(fig)


def save_curves(history: list[dict[str, Any]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    epochs = [row["epoch"] for row in history]

    def draw(filename: str, ylabel: str, series: list[tuple[str, list[float]]]) -> None:
        fig, axis = plt.subplots(figsize=(9, 5))
        for label, values in series:
            axis.plot(epochs, values, label=label)
        axis.set_xlabel("epoch")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
        axis.legend()
        fig.tight_layout()
        fig.savefig(out_dir / filename, dpi=160)
        plt.close(fig)

    draw(
        "loss曲线.png",
        "masked Charbonnier loss",
        [
            ("train", [row["train_loss"] for row in history]),
            ("eval_same_32", [row["eval_loss"] for row in history]),
        ],
    )
    draw(
        "LIF发放率曲线.png",
        "firing rate",
        [
            ("LIF1", [row["lif1_firing_rate"] for row in history]),
            ("LIF2", [row["lif2_firing_rate"] for row in history]),
        ],
    )
    draw(
        "累加器统计曲线.png",
        "accumulator",
        [
            ("mean", [row["accumulator_mean"] for row in history]),
            ("std", [row["accumulator_std"] for row in history]),
        ],
    )
    draw(
        "预测统计曲线.png",
        "prediction",
        [
            ("mean", [row["prediction_mean"] for row in history]),
            ("std", [row["prediction_std"] for row in history]),
        ],
    )
    draw(
        "梯度范数曲线.png",
        "gradient L2 norm",
        [
            ("SNN encoder", [row["snn_gradient_norm"] for row in history]),
            ("ANN backend", [row["ann_gradient_norm"] for row in history]),
        ],
    )


def run_overfit32(
    args: argparse.Namespace,
    device: torch.device,
    smoke_summary: dict[str, Any] | None,
) -> dict[str, Any]:
    """固定同一批 32 个连续窗口，训练 100 epoch 并保存纵向可视化。"""

    if smoke_summary is not None and not smoke_summary["passed"]:
        raise RuntimeError("Smoke test 未通过，按指导要求停止 32-Block 过拟合。")
    set_seed(args.seed)
    dataset = TTCEFTemporalDataset(
        args.train_file, window_length=3, window_stride=3, augment=False
    )
    generator = np.random.default_rng(args.seed)
    chosen_indices = sorted(
        int(value) for value in generator.choice(len(dataset), size=32, replace=False)
    )
    subset = Subset(dataset, chosen_indices)
    shuffle_generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        subset,
        batch_size=4,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        generator=shuffle_generator,
    )
    eval_loader = DataLoader(
        subset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    model = HybridSNNEVSlim().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = args.epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        total_steps=total_steps,
        pct_start=0.1,
        cycle_momentum=False,
    )
    scaler: torch.amp.GradScaler | None = None
    global_step = 0
    configuration = {
        "T": 3,
        "window_stride": 3,
        "batch_size": 4,
        "num_workers": 2,
        "epochs": args.epochs,
        "seed": args.seed,
        "optimizer": "AdamW",
        "learning_rate": args.lr,
        "scheduler": "OneCycleLR pct_start=0.1",
        "loss": "masked Charbonnier alpha=0.45",
        "precision": "AMP FP16" if device.type == "cuda" else "FP32",
        "augmentation": False,
        "selected_dataset_indices": chosen_indices,
        "selected_start_rows": [int(dataset.window_starts[index]) for index in chosen_indices],
    }
    visual_indices = chosen_indices[:4]
    color_limits = visualization_limits(dataset, visual_indices)
    visual_root = args.out_dir / "可视化结果"
    curve_root = visual_root / "曲线"
    initial_eval = evaluate_blocks(model, eval_loader, device)
    save_epoch_visualizations(
        model, dataset, visual_indices, color_limits, 0, visual_root, device
    )
    history: list[dict[str, Any]] = []
    best_validation_loss = float("inf")
    checkpoint_root = args.out_dir / "checkpoints"

    for epoch in range(1, args.epochs + 1):
        epoch_rows: list[dict[str, float]] = []
        for batch in train_loader:
            exp_filts, ttc, mask = move_batch(batch, device)
            result: HybridBlockResult = train_hybrid_block(
                model,
                optimizer,
                exp_filts,
                ttc,
                mask,
                use_amp=device.type == "cuda",
                scaler=scaler,
            )
            scaler = result.scaler
            scheduler.step()
            global_step += 1
            state = model.get_state_statistics()
            gradients = gradient_groups(model)
            prediction = tensor_stats(result.prediction)
            epoch_rows.append(
                {
                    "loss": result.loss,
                    "lif1_firing_rate": state["firing_rates"]["lif1"],
                    "lif2_firing_rate": state["firing_rates"]["lif2"],
                    "accumulator_mean": state["accumulator"]["mean"],
                    "accumulator_std": state["accumulator"]["std"],
                    "prediction_mean": prediction["mean"],
                    "prediction_std": prediction["std"],
                    "snn_gradient_norm": gradients["snn_encoder"],
                    "ann_gradient_norm": gradients["ann_backend"],
                }
            )
        evaluation = evaluate_blocks(model, eval_loader, device)
        epoch_summary = {
            "epoch": epoch,
            "train_loss": float(np.mean([row["loss"] for row in epoch_rows])),
            "eval_loss": evaluation["loss"],
            "eval_mae": evaluation["mae"],
            "eval_correlation": evaluation["correlation"],
            "lif1_firing_rate": float(
                np.mean([row["lif1_firing_rate"] for row in epoch_rows])
            ),
            "lif2_firing_rate": float(
                np.mean([row["lif2_firing_rate"] for row in epoch_rows])
            ),
            "accumulator_mean": float(
                np.mean([row["accumulator_mean"] for row in epoch_rows])
            ),
            "accumulator_std": float(
                np.mean([row["accumulator_std"] for row in epoch_rows])
            ),
            "prediction_mean": float(
                np.mean([row["prediction_mean"] for row in epoch_rows])
            ),
            "prediction_std": float(
                np.mean([row["prediction_std"] for row in epoch_rows])
            ),
            "snn_gradient_norm": float(
                np.mean([row["snn_gradient_norm"] for row in epoch_rows])
            ),
            "ann_gradient_norm": float(
                np.mean([row["ann_gradient_norm"] for row in epoch_rows])
            ),
            "learning_rate": float(scheduler.get_last_lr()[0]),
            "scaler_scale": float(scaler.get_scale()) if scaler is not None else None,
            "finite": bool(
                evaluation["finite"]
                and all(
                    math.isfinite(value)
                    for row in epoch_rows
                    for value in row.values()
                )
            ),
        }
        history.append(epoch_summary)
        if evaluation["loss"] < best_validation_loss:
            best_validation_loss = evaluation["loss"]
            save_checkpoint(
                checkpoint_root / "best.pt",
                model,
                optimizer,
                scheduler,
                scaler,
                epoch=epoch,
                global_step=global_step,
                best_validation_loss=best_validation_loss,
                configuration=configuration,
            )
        if epoch in VISUAL_EPOCHS or epoch == args.epochs:
            save_epoch_visualizations(
                model, dataset, visual_indices, color_limits, epoch, visual_root, device
            )
        print(
            f"[overfit32] epoch {epoch:03d}/{args.epochs} "
            f"train={epoch_summary['train_loss']:.6f} eval={evaluation['loss']:.6f} "
            f"rate={epoch_summary['lif1_firing_rate']:.4f}/{epoch_summary['lif2_firing_rate']:.4f}",
            flush=True,
        )

    save_checkpoint(
        checkpoint_root / "last.pt",
        model,
        optimizer,
        scheduler,
        scaler,
        epoch=args.epochs,
        global_step=global_step,
        best_validation_loss=best_validation_loss,
        configuration=configuration,
    )
    save_curves(history, curve_root)
    final = history[-1]
    loss_reduction = (initial_eval["loss"] - final["eval_loss"]) / max(
        initial_eval["loss"], 1e-12
    )
    finite = all(row["finite"] for row in history)
    firing_stable = all(
        0.001 < row[name] < 0.8
        for row in history
        for name in ("lif1_firing_rate", "lif2_firing_rate")
    )
    gradients_stable = all(
        math.isfinite(row[name]) and row[name] > 0.0
        for row in history
        for name in ("snn_gradient_norm", "ann_gradient_norm")
    )
    accumulator_stable = all(
        math.isfinite(row["accumulator_mean"])
        and math.isfinite(row["accumulator_std"])
        and row["accumulator_std"] < 1e4
        for row in history
    )
    output_not_collapsed = final["prediction_std"] > 1e-5
    overfit_success = bool(
        finite
        and loss_reduction >= 0.30
        and firing_stable
        and gradients_stable
        and accumulator_stable
        and output_not_collapsed
    )
    summary = {
        "passed": overfit_success,
        "configuration": configuration,
        "initial_evaluation": initial_eval,
        "final_evaluation": {
            "loss": final["eval_loss"],
            "mae": final["eval_mae"],
            "correlation": final["eval_correlation"],
            "prediction_mean": final["prediction_mean"],
            "prediction_std": final["prediction_std"],
        },
        "initial_loss": initial_eval["loss"],
        "final_train_loss": final["train_loss"],
        "final_eval_loss": final["eval_loss"],
        "best_validation_loss": best_validation_loss,
        "loss_reduction_fraction": loss_reduction,
        "global_step": global_step,
        "checks": {
            "loss_reduction_at_least_30_percent": loss_reduction >= 0.30,
            "finite": finite,
            "firing_rates_stable": firing_stable,
            "gradients_nonzero_finite": gradients_stable,
            "accumulator_stable": accumulator_stable,
            "output_not_collapsed": output_not_collapsed,
            "prediction_structure_improved": final["eval_correlation"]
            > initial_eval["correlation"],
        },
        "history": history,
        "visualization_indices": visual_indices,
        "visualization_color_limits": color_limits,
        "checkpoint_best": str(checkpoint_root / "best.pt"),
        "checkpoint_last": str(checkpoint_root / "last.pt"),
        "ready_for_200_block_timing": overfit_success,
    }
    save_json(args.out_dir / "32Block过拟合实验结果.json", summary)
    (args.out_dir / "32Block过拟合实验报告.md").write_text(
        render_overfit_report(summary), encoding="utf-8"
    )
    shutdown_loader(train_loader)
    dataset.close()
    return summary


def render_overfit_report(summary: dict[str, Any]) -> str:
    final = summary["history"][-1]
    initial = summary["initial_evaluation"]
    checks = summary["checks"]
    return "\n".join(
        [
            "# Hybrid SNN-EV-Slim 32-Block 过拟合实验报告",
            "",
            f"- 总体结果：{'通过' if summary['passed'] else '未通过'}",
            "- 数据：从真实 train H5 固定抽取 32 个合法 `T=3, stride=3` 连续窗口，关闭增强。",
            f"- 训练：{summary['configuration']['epochs']} epochs，batch=4，AMP FP16，AdamW + OneCycleLR。",
            "- 执行命令：`EV-TTC-main/.venv/bin/python EV-TTC-SNN-main/snn_ttc/train_hybrid_snn_evslim_real.py --mode overfit32`",
            f"- 初始 eval loss：`{summary['initial_loss']:.6f}`",
            f"- 最终 train/eval loss：`{summary['final_train_loss']:.6f}` / `{summary['final_eval_loss']:.6f}`",
            f"- loss 相对下降：`{summary['loss_reduction_fraction']:.2%}`",
            f"- 初始/最终 masked MAE：`{initial['mae']:.6f}` / `{summary['final_evaluation']['mae']:.6f}`",
            f"- 初始/最终空间相关系数：`{initial['correlation']:.6f}` / `{summary['final_evaluation']['correlation']:.6f}`",
            "",
            "## 最终内部统计",
            "",
            f"- LIF1/LIF2 发放率：`{final['lif1_firing_rate']:.4%}` / `{final['lif2_firing_rate']:.4%}`",
            f"- accumulator mean/std：`{final['accumulator_mean']:.6f}` / `{final['accumulator_std']:.6f}`",
            f"- prediction mean/std：`{final['prediction_mean']:.6f}` / `{final['prediction_std']:.6f}`",
            f"- SNN/ANN 梯度范数：`{final['snn_gradient_norm']:.6g}` / `{final['ann_gradient_norm']:.6g}`",
            f"- AMP scaler scale：`{final['scaler_scale']}`",
            "",
            "## 判定",
            "",
            f"- loss 至少下降 30%：{'通过' if checks['loss_reduction_at_least_30_percent'] else '失败'}",
            f"- 全程无 NaN/Inf：{'通过' if checks['finite'] else '失败'}",
            f"- 发放率未坍缩或饱和：{'通过' if checks['firing_rates_stable'] else '失败'}",
            f"- SNN/ANN 梯度持续有效：{'通过' if checks['gradients_nonzero_finite'] else '失败'}",
            f"- accumulator 稳定：{'通过' if checks['accumulator_stable'] else '失败'}",
            f"- 输出未坍缩为常数：{'通过' if checks['output_not_collapsed'] else '失败'}",
            f"- 预测空间结构相关性改善：{'通过' if checks['prediction_structure_improved'] else '失败'}",
            "",
            "## 输出",
            "",
            f"- 最佳 checkpoint：`{summary['checkpoint_best']}`",
            f"- 最后 checkpoint：`{summary['checkpoint_last']}`",
            "- 可视化：`可视化结果/epoch_000,001,005,010,020,050,100/`",
            "- 曲线：`可视化结果/曲线/`",
            f"- 是否适合进入 200-Block 速度标定：{'是' if summary['ready_for_200_block_timing'] else '否'}",
            "",
            "## 训练曲线",
            "",
            "![Loss曲线](可视化结果/曲线/loss曲线.png)",
            "![LIF发放率曲线](可视化结果/曲线/LIF发放率曲线.png)",
            "![累加器统计曲线](可视化结果/曲线/累加器统计曲线.png)",
            "![预测统计曲线](可视化结果/曲线/预测统计曲线.png)",
            "![梯度范数曲线](可视化结果/曲线/梯度范数曲线.png)",
            "",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "overfit32", "all"), default="all")
    parser.add_argument("--train-file", type=Path, default=DEFAULT_DATA_ROOT / "train.h5")
    parser.add_argument("--val-file", type=Path, default=DEFAULT_DATA_ROOT / "val.h5")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=20260722)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.train_file = args.train_file.resolve()
    args.val_file = args.val_file.resolve()
    args.out_dir = args.out_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("本任务固定要求 CUDA AMP FP16，但当前未检测到 CUDA。")
    device = torch.device("cuda:0")
    smoke_summary: dict[str, Any] | None = None
    if args.mode in ("smoke", "all"):
        smoke_summary = run_smoke(args, device)
    elif (args.out_dir / "真实数据端到端验证结果.json").is_file():
        smoke_summary = json.loads(
            (args.out_dir / "真实数据端到端验证结果.json").read_text(encoding="utf-8")
        )
    if args.mode in ("overfit32", "all"):
        run_overfit32(args, device, smoke_summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
