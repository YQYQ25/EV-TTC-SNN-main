"""LIF-EV-FlowNet-TTC 模型封装。

模型主体复用 MAVLab event_flow 的 `SpikingMultiResUNetRecurrent`，仅将
EV-FlowNet 原本的 2 通道光流输出头改成 1 通道 signed inverse TTC 输出头。
输出层保持线性，允许预测值为正或负。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .lif_modules import MAVLAB_LIF_NEURON, SpikingMultiResUNetRecurrent, copy_states


@dataclass(frozen=True)
class LIFEVFlowNetTTCConfig:
    """保存从 MAVLab 配置读取并最小改动后的网络参数。"""

    input_channels: int = 2
    output_channels: int = 1
    base_num_channels: int = 32
    num_encoders: int = 4
    num_residual_blocks: int = 2
    skip_type: str = "concat"
    norm: str | None = None
    use_upsample_conv: bool = True
    kernel_size: int = 3
    channel_multiplier: int = 2
    activations: list[str] = field(default_factory=lambda: ["arctanspike", "arctanspike"])
    recurrent_block_type: str = "lif"
    spiking_feedforward_block_type: str = "lif"
    spiking_neuron: dict[str, Any] = field(default_factory=lambda: dict(MAVLAB_LIF_NEURON))
    output_activation: None = None


def _detach_state(state: Any) -> Any:
    """递归截断状态计算图，同时保留状态数值。"""

    if state is None:
        return None
    if torch.is_tensor(state):
        return state.detach()
    if isinstance(state, tuple):
        return tuple(_detach_state(item) for item in state)
    if isinstance(state, list):
        return [_detach_state(item) for item in state]
    return state


def _state_tensors(state: Any) -> list[torch.Tensor]:
    """把嵌套状态展开成张量列表，便于状态测试和脉冲统计。"""

    if state is None:
        return []
    if torch.is_tensor(state):
        return [state]
    if isinstance(state, (tuple, list)):
        out: list[torch.Tensor] = []
        for item in state:
            out.extend(_state_tensors(item))
        return out
    return []


class LIFEVFlowNetTTC(nn.Module):
    """面向 TTC 回归的 LIF-EV-FlowNet。

    输入为单个时间步的 `[B, 2, H, W]` 正/负事件计数图，输出为
    `[B, 1, H, W]` signed inverse TTC。状态由 MAVLab recurrent U-Net 保存。
    """

    def __init__(self, cfg: LIFEVFlowNetTTCConfig | None = None, **kwargs: Any) -> None:
        super().__init__()
        if cfg is None:
            cfg = LIFEVFlowNetTTCConfig(**kwargs)
        elif kwargs:
            raise ValueError("传入 cfg 时不要再额外传 kwargs，避免配置来源混乱。")
        self.cfg = cfg
        self.forward_step_count = 0

        unet_kwargs = {
            "base_num_channels": cfg.base_num_channels,
            "num_encoders": cfg.num_encoders,
            "num_residual_blocks": cfg.num_residual_blocks,
            "num_output_channels": cfg.output_channels,
            "skip_type": cfg.skip_type,
            "norm": cfg.norm,
            "use_upsample_conv": cfg.use_upsample_conv,
            "num_bins": cfg.input_channels,
            "recurrent_block_type": cfg.recurrent_block_type,
            "kernel_size": cfg.kernel_size,
            "channel_multiplier": cfg.channel_multiplier,
            "activations": cfg.activations,
            "final_activation": cfg.output_activation,
            "spiking_feedforward_block_type": cfg.spiking_feedforward_block_type,
            "spiking_neuron": cfg.spiking_neuron,
        }
        self.multires_unetrec = SpikingMultiResUNetRecurrent(unet_kwargs)

    @property
    def states(self) -> list[Any]:
        """返回状态副本，避免调用侧意外改写内部 recurrent state。"""

        return copy_states(self.multires_unetrec.states)

    @states.setter
    def states(self, states: list[Any]) -> None:
        self.multires_unetrec.states = states

    def reset_states(self) -> None:
        """在序列开始、跳变或 ROI 切换时重置历史状态。"""

        self.multires_unetrec.states = [None] * self.multires_unetrec.num_states

    def detach_states(self) -> None:
        """在 block 边界截断 BPTT 图，但不清空物理时间状态。"""

        self.multires_unetrec.states = [_detach_state(state) for state in self.multires_unetrec.states]

    @property
    def total_stride(self) -> int:
        return 2 ** int(self.cfg.num_encoders)

    def _pad_to_legal_size(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int, int, int]]:
        """padding 到总下采样倍率可整除的尺寸；不做 resize。"""

        _, _, h, w = x.shape
        stride = self.total_stride
        pad_h = (stride - h % stride) % stride
        pad_w = (stride - w % stride) % stride
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left
        if pad_h or pad_w:
            x = F.pad(x, (left, right, top, bottom))
        return x, (top, bottom, left, right)

    @staticmethod
    def _crop_back(y: torch.Tensor, padding: tuple[int, int, int, int]) -> torch.Tensor:
        top, bottom, left, right = padding
        h_end = y.shape[-2] - bottom if bottom else y.shape[-2]
        w_end = y.shape[-1] - right if right else y.shape[-1]
        return y[..., top:h_end, left:w_end].contiguous()

    def forward(self, events_t: torch.Tensor, return_activity: bool = False) -> torch.Tensor | tuple[torch.Tensor, list[dict[str, float]]]:
        """执行一个时间步前向传播。

        `events_t` 必须是 `[B, 2, H, W]`。若 H/W 不能整除网络总 stride，
        先 padding，网络输出后再裁回原尺寸。
        """

        if events_t.ndim != 4 or events_t.shape[1] != self.cfg.input_channels:
            raise ValueError(f"期望输入 [B,{self.cfg.input_channels},H,W]，实际为 {tuple(events_t.shape)}")
        x, padding = self._pad_to_legal_size(events_t)
        multires_pred = self.multires_unetrec.forward(x)
        pred = multires_pred[-1]
        pred = self._crop_back(pred, padding)
        self.forward_step_count += 1
        if return_activity:
            return pred, self.activity_stats()
        return pred

    def activity_stats(self) -> list[dict[str, float]]:
        """统计每层膜电位和 spike 活动，供后续输入幅值审计使用。"""

        rows: list[dict[str, float]] = []
        for layer_index, state in enumerate(self.multires_unetrec.states):
            tensors = _state_tensors(state)
            if not tensors:
                continue
            # ConvLIF 状态通常为 [2,B,C,H,W]，第0项是膜电位，第1项是 spike。
            state_tensor = tensors[0]
            if state_tensor.ndim >= 5 and state_tensor.shape[0] >= 2:
                membrane = state_tensor[0].detach()
                spikes = state_tensor[1].detach()
            else:
                membrane = state_tensor.detach()
                spikes = torch.zeros_like(membrane)
            per_neuron_spikes = spikes.flatten(2).mean(dim=2)
            rows.append(
                {
                    "layer": float(layer_index),
                    "membrane_mean": float(membrane.mean().detach().cpu()),
                    "membrane_std": float(membrane.std(unbiased=False).detach().cpu()),
                    "membrane_max": float(membrane.abs().max().detach().cpu()),
                    "spike_rate": float(spikes.mean().detach().cpu()),
                    "all_zero_neuron_ratio": float((per_neuron_spikes == 0).float().mean().detach().cpu()),
                    "high_frequency_ratio": float((per_neuron_spikes > 0.5).float().mean().detach().cpu()),
                }
            )
        return rows

    def state_value_signature(self) -> list[torch.Tensor]:
        """返回状态张量副本，用于单元测试比较 reset/detach 前后的数值。"""

        out: list[torch.Tensor] = []
        for state in self.multires_unetrec.states:
            for tensor in _state_tensors(state):
                out.append(tensor.detach().clone())
        return out
