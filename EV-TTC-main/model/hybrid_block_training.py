"""Hybrid SNN-EV-Slim 的 Block 级训练与验证单元。"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from hybrid_snn_evslim import HybridSNNEVSlim
from util import charbonnier_loss


@dataclass(frozen=True)
class HybridBlockResult:
    """保存一个 Block 的标量结果和无计算图预测。"""

    loss: float
    prediction: torch.Tensor
    snn_steps: int
    ann_backend_calls: int
    scaler: torch.amp.GradScaler | None


def _validate_batch(exp_filts: torch.Tensor, ttc: torch.Tensor, mask: torch.Tensor) -> None:
    if exp_filts.ndim != 5:
        raise ValueError(f"exp_filts 应为 [B,T,6,H,W]，实际为 {tuple(exp_filts.shape)}")
    batch, _, _, height, width = exp_filts.shape
    expected = (batch, 1, height, width)
    if tuple(ttc.shape) != expected:
        raise ValueError(f"ttc 应为 {expected}，实际为 {tuple(ttc.shape)}")
    if tuple(mask.shape) != expected:
        raise ValueError(f"mask 应为 {expected}，实际为 {tuple(mask.shape)}")


def get_amp_scaler(
    device: torch.device,
    use_amp: bool,
    scaler: torch.amp.GradScaler | None = None,
) -> torch.amp.GradScaler | None:
    """为 CUDA AMP 统一创建并复用 GradScaler，FP32 路径保持 None。"""

    if not use_amp or device.type != "cuda":
        return None
    return scaler if scaler is not None else torch.amp.GradScaler("cuda", enabled=True)


def train_hybrid_block(
    model: HybridSNNEVSlim,
    optimizer: torch.optim.Optimizer,
    exp_filts: torch.Tensor,
    ttc: torch.Tensor,
    mask: torch.Tensor,
    *,
    use_amp: bool = False,
    scaler: torch.amp.GradScaler | None = None,
) -> HybridBlockResult:
    """训练一个 Block：一次 reset、一次 ANN、一次 backward 和一次 step。"""

    _validate_batch(exp_filts, ttc, mask)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    model.reset_states()
    amp_enabled = bool(use_amp and exp_filts.device.type == "cuda")
    active_scaler = get_amp_scaler(exp_filts.device, use_amp, scaler)
    autocast_context = (
        torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True)
        if amp_enabled
        else nullcontext()
    )
    with autocast_context:
        prediction = model.forward_sequence(exp_filts)
        loss = charbonnier_loss(ttc - prediction, alpha=0.45, mask=mask)

    if active_scaler is not None:
        active_scaler.scale(loss).backward()
        active_scaler.step(optimizer)
        active_scaler.update()
    else:
        loss.backward()
        optimizer.step()
    return HybridBlockResult(
        loss=float(loss.detach().cpu()),
        prediction=prediction.detach(),
        snn_steps=model.snn_step_count,
        ann_backend_calls=model.ann_backend_call_count,
        scaler=active_scaler,
    )


@torch.no_grad()
def validate_hybrid_block(
    model: HybridSNNEVSlim,
    exp_filts: torch.Tensor,
    ttc: torch.Tensor,
    mask: torch.Tensor,
    *,
    use_amp: bool = False,
) -> HybridBlockResult:
    """验证单个 Block；同样禁止跨 Block 继承状态。"""

    _validate_batch(exp_filts, ttc, mask)
    model.eval()
    model.reset_states()
    amp_enabled = bool(use_amp and exp_filts.device.type == "cuda")
    autocast_context = (
        torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True)
        if amp_enabled
        else nullcontext()
    )
    with autocast_context:
        prediction = model.forward_sequence(exp_filts)
        loss = charbonnier_loss(ttc - prediction, alpha=0.45, mask=mask)
    return HybridBlockResult(
        loss=float(loss.detach().cpu()),
        prediction=prediction.detach(),
        snn_steps=model.snn_step_count,
        ann_backend_calls=model.ann_backend_call_count,
        scaler=None,
    )


def save_hybrid_checkpoint(
    path: str | Path,
    model: HybridSNNEVSlim,
    optimizer: torch.optim.Optimizer,
    *,
    epoch: int,
    extra: dict[str, Any] | None = None,
    scaler: torch.amp.GradScaler | None = None,
) -> None:
    """保存完整 SNN、累加编码器、ANN 后端和优化器参数。"""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": int(epoch),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "model_config": model.cfg.__dict__,
            "extra": extra or {},
        },
        output,
    )


def load_hybrid_checkpoint(
    path: str | Path,
    model: HybridSNNEVSlim,
    optimizer: torch.optim.Optimizer | None = None,
    *,
    scaler: torch.amp.GradScaler | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """恢复模型、优化器和 AMP scaler；状态变量在下一个 Block 重新 reset。"""

    checkpoint = torch.load(Path(path), map_location=map_location, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scaler is not None and checkpoint.get("scaler") is not None:
        scaler.load_state_dict(checkpoint["scaler"])
    model.reset_states()
    return checkpoint
