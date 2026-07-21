"""EV-TTC 口径的逐样本 masked Charbonnier loss。

原 EV-TTC 实现见 `EV-TTC-main/model/util.py`：
`(error**2 + 1e-5**2) ** alpha`，默认 `alpha=0.45`，
mask 情况下先按每个样本的有效像素平均，再对有效样本平均。
这里额外加入 `supervise_valid`，用于 SNN 连续事件块中的无监督 step。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


EVTTC_CHARBONNIER_ALPHA = 0.45
EVTTC_CHARBONNIER_EPS = 1e-5


@dataclass(frozen=True)
class MaskedCharbonnierStats:
    """loss 调试统计，便于写入阶段报告。"""

    pixel_count: torch.Tensor
    valid_sample_count: int
    total_valid_pixels: int
    has_supervision: bool
    alpha: float
    epsilon: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "pixel_count": self.pixel_count.detach().cpu().tolist(),
            "valid_sample_count": self.valid_sample_count,
            "total_valid_pixels": self.total_valid_pixels,
            "has_supervision": self.has_supervision,
            "alpha": self.alpha,
            "epsilon": self.epsilon,
        }


def _squeeze_prediction_like(x: torch.Tensor, name: str) -> torch.Tensor:
    """兼容 `[B,1,H,W]` 和 `[B,H,W]` 两种 TTC map 表示。"""

    if x.ndim == 4:
        if x.shape[1] != 1:
            raise ValueError(f"{name} 的4维输入必须是单通道，实际 shape={tuple(x.shape)}")
        return x[:, 0]
    if x.ndim == 3:
        return x
    raise ValueError(f"{name} 期望 [B,1,H,W] 或 [B,H,W]，实际 shape={tuple(x.shape)}")


def charbonnier(error: torch.Tensor, alpha: float = EVTTC_CHARBONNIER_ALPHA, epsilon: float = EVTTC_CHARBONNIER_EPS) -> torch.Tensor:
    """EV-TTC 原始 Charbonnier 数据项。"""

    return (error.square() + epsilon**2.0).pow(alpha)


def masked_charbonnier_per_sample(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid_ttc_mask: torch.Tensor,
    supervise_valid: torch.Tensor,
    *,
    alpha: float = EVTTC_CHARBONNIER_ALPHA,
    epsilon: float = EVTTC_CHARBONNIER_EPS,
) -> tuple[torch.Tensor, torch.Tensor, MaskedCharbonnierStats]:
    """计算逐样本 loss，不让有效像素更多的样本获得更大权重。

    返回的 `per_sample_loss` 长度为 B；无效样本位置填 0，并由
    `valid_samples` 指明哪些样本可用于 block loss 累计。
    """

    pred = _squeeze_prediction_like(prediction, "prediction")
    tgt = _squeeze_prediction_like(target, "target")
    if pred.shape != tgt.shape:
        raise ValueError(f"prediction 和 target shape 不一致：{tuple(pred.shape)} vs {tuple(tgt.shape)}")
    if valid_ttc_mask.shape != pred.shape:
        raise ValueError(f"valid_ttc_mask shape={tuple(valid_ttc_mask.shape)}，应为 {tuple(pred.shape)}")
    if supervise_valid.ndim != 1 or supervise_valid.shape[0] != pred.shape[0]:
        raise ValueError("supervise_valid 必须是长度为 B 的一维张量")

    finite = torch.isfinite(pred) & torch.isfinite(tgt)
    loss_mask = valid_ttc_mask.bool() & supervise_valid.bool()[:, None, None] & finite
    safe_error = torch.where(finite, pred - tgt, torch.zeros_like(pred))
    loss_map = charbonnier(safe_error, alpha=alpha, epsilon=epsilon)

    valid_float = loss_mask.to(loss_map.dtype)
    pixel_count = valid_float.flatten(1).sum(dim=1)
    valid_samples = pixel_count > 0
    loss_sum = (loss_map * valid_float).flatten(1).sum(dim=1)

    per_sample_loss = torch.zeros(pred.shape[0], dtype=loss_map.dtype, device=loss_map.device)
    per_sample_loss[valid_samples] = loss_sum[valid_samples] / pixel_count[valid_samples].to(loss_map.dtype)

    stats = MaskedCharbonnierStats(
        pixel_count=pixel_count,
        valid_sample_count=int(valid_samples.sum().detach().cpu().item()),
        total_valid_pixels=int(pixel_count.sum().detach().cpu().item()),
        has_supervision=bool(valid_samples.any().detach().cpu().item()),
        alpha=float(alpha),
        epsilon=float(epsilon),
    )
    return per_sample_loss, valid_samples, stats


def reduce_valid_sample_losses(per_sample_loss: torch.Tensor, valid_samples: torch.Tensor) -> torch.Tensor | None:
    """把单步 per-sample loss 合成为标量；无有效样本时返回 None。"""

    if not bool(valid_samples.any().detach().cpu().item()):
        return None
    return per_sample_loss[valid_samples].mean()


def evttc_reference_per_sample(
    error: torch.Tensor,
    mask: torch.Tensor,
    *,
    alpha: float = EVTTC_CHARBONNIER_ALPHA,
    epsilon: float = EVTTC_CHARBONNIER_EPS,
) -> tuple[torch.Tensor, torch.Tensor]:
    """逐样本复刻 EV-TTC 原 loss 的 mask 分支，用于数值对齐测试。"""

    err = _squeeze_prediction_like(error, "error")
    if mask.ndim == 4:
        if mask.shape[1] != 1:
            raise ValueError("mask 的4维输入必须是单通道")
        mask = mask[:, 0]
    valid = mask.bool() & torch.isfinite(err)
    safe_error = torch.where(valid, err, torch.zeros_like(err))
    valid_float = valid.to(err.dtype)
    denom = valid_float.flatten(1).sum(dim=1)
    loss_sum = (charbonnier(safe_error, alpha=alpha, epsilon=epsilon) * valid_float).flatten(1).sum(dim=1)
    per_sample = torch.zeros(err.shape[0], dtype=err.dtype, device=err.device)
    keep = denom > 0
    per_sample[keep] = loss_sum[keep] / denom[keep].clamp_min(1.0)
    return per_sample, keep
