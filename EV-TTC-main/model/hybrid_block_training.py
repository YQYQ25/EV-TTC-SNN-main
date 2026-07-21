"""Hybrid SNN-EV-Slim 的 Block 级训练与验证单元。"""

from __future__ import annotations

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


def _validate_batch(exp_filts: torch.Tensor, ttc: torch.Tensor, mask: torch.Tensor) -> None:
    if exp_filts.ndim != 5:
        raise ValueError(f"exp_filts 应为 [B,T,6,H,W]，实际为 {tuple(exp_filts.shape)}")
    batch, _, _, height, width = exp_filts.shape
    expected = (batch, 1, height, width)
    if tuple(ttc.shape) != expected:
        raise ValueError(f"ttc 应为 {expected}，实际为 {tuple(ttc.shape)}")
    if tuple(mask.shape) != expected:
        raise ValueError(f"mask 应为 {expected}，实际为 {tuple(mask.shape)}")


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
    with torch.autocast(device_type=exp_filts.device.type, enabled=amp_enabled):
        prediction = model.forward_sequence(exp_filts)
        loss = charbonnier_loss(ttc - prediction, alpha=0.45, mask=mask)

    if scaler is not None and amp_enabled:
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        optimizer.step()
    return HybridBlockResult(
        loss=float(loss.detach().cpu()),
        prediction=prediction.detach(),
        snn_steps=model.snn_step_count,
        ann_backend_calls=model.ann_backend_call_count,
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
    with torch.autocast(device_type=exp_filts.device.type, enabled=amp_enabled):
        prediction = model.forward_sequence(exp_filts)
        loss = charbonnier_loss(ttc - prediction, alpha=0.45, mask=mask)
    return HybridBlockResult(
        loss=float(loss.detach().cpu()),
        prediction=prediction.detach(),
        snn_steps=model.snn_step_count,
        ann_backend_calls=model.ann_backend_call_count,
    )


def save_hybrid_checkpoint(
    path: str | Path,
    model: HybridSNNEVSlim,
    optimizer: torch.optim.Optimizer,
    *,
    epoch: int,
    extra: dict[str, Any] | None = None,
) -> None:
    """保存完整 SNN、累加编码器、ANN 后端和优化器参数。"""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": int(epoch),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "model_config": model.cfg.__dict__,
            "extra": extra or {},
        },
        output,
    )
