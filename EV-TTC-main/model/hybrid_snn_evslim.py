"""Spike-FlowNet 式混合 SNN-EV-Slim。

前两层编码器使用仓库已有 ConvLIF 的状态更新与 arctan 代理梯度，第三层
保留实值并跨 T 步求和。ASPP 和 decoder/head 直接复用 EV-Slim 的实现。
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SNN_ROOT = PROJECT_ROOT / "EV-TTC-SNN-main"
if str(SNN_ROOT) not in sys.path:
    sys.path.insert(0, str(SNN_ROOT))

from snn_ttc.models.lif_modules import ConvLIF, MAVLAB_LIF_NEURON  # noqa: E402

from layers.aspp import ASPP  # noqa: E402
from layers.conv_lay import ConvLayer  # noqa: E402


@dataclass(frozen=True)
class HybridSNNEVSlimConfig:
    """第一版 Hybrid 网络配置，默认值与当前 EV-Slim 和 MAVLab LIF 对齐。"""

    in_channels: int = 6
    enc_channels: tuple[int, int, int] = (16, 16, 16)
    enc_k_size: tuple[int, int, int] = (3, 3, 3)
    aspp_channels: tuple[int, int] = (32, 16)
    rate: tuple[int, int, int] = (12, 24, 36)
    dec_channels: tuple[int, int, int] = (8, 8, 1)
    dec_k_size: tuple[int, int, int] = (3, 3, 3)
    norm: bool = True
    act: str = "ReLU"
    lif_activation: str = "arctanspike"
    lif_act_width: float = 10.0
    lif_leak: tuple[float, float] = tuple(MAVLAB_LIF_NEURON["leak"])
    lif_thresh: tuple[float, float] = tuple(MAVLAB_LIF_NEURON["thresh"])
    lif_learn_leak: bool = bool(MAVLAB_LIF_NEURON["learn_leak"])
    lif_learn_thresh: bool = bool(MAVLAB_LIF_NEURON["learn_thresh"])
    lif_hard_reset: bool = bool(MAVLAB_LIF_NEURON["hard_reset"])


def _cfg_value(cfg: Any, name: str, default: Any) -> Any:
    """同时兼容 dataclass、SimpleNamespace 和 OmegaConf 风格配置。"""

    return getattr(cfg, name, default)


class _ActivationOnlyConvLIF(ConvLIF):
    """复用 ConvLIF 动力学，但让外部官方 Conv+BN 提供输入电流。

    ConvLIF 原本把卷积和 LIF 合在同一个单元中。Hybrid 结构要求严格保持
    EV-Slim 的 ``Conv -> BN -> LIF`` 顺序，因此将其内部卷积替换为恒等映射；
    膜电位、泄漏、阈值、hard reset 和 arctan 代理梯度仍全部执行原实现。
    """

    def __init__(self, channels: int, cfg: HybridSNNEVSlimConfig) -> None:
        super().__init__(
            input_size=channels,
            hidden_size=channels,
            kernel_size=1,
            activation=cfg.lif_activation,
            act_width=cfg.lif_act_width,
            leak=cfg.lif_leak,
            thresh=cfg.lif_thresh,
            learn_leak=cfg.lif_learn_leak,
            learn_thresh=cfg.lif_learn_thresh,
            hard_reset=cfg.lif_hard_reset,
            detach=True,
            norm=None,
        )
        self.ff = nn.Identity()


def _tensor_statistics(tensor: torch.Tensor | None) -> dict[str, float | bool]:
    if tensor is None:
        return {
            "initialized": False,
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "finite": True,
        }
    detached = tensor.detach()
    return {
        "initialized": True,
        "mean": float(detached.mean().cpu()),
        "std": float(detached.std(unbiased=False).cpu()),
        "min": float(detached.min().cpu()),
        "max": float(detached.max().cpu()),
        "finite": bool(torch.isfinite(detached).all().cpu()),
    }


class HybridSNNEVSlim(nn.Module):
    """两层 LIF 编码器、实值累加器和 EV-Slim ANN 后端。"""

    def __init__(self, cfg: Any | None = None) -> None:
        super().__init__()
        defaults = HybridSNNEVSlimConfig()
        source = cfg if cfg is not None else defaults
        self.cfg = HybridSNNEVSlimConfig(
            in_channels=int(_cfg_value(source, "in_channels", _cfg_value(source, "input", defaults.in_channels))),
            enc_channels=tuple(_cfg_value(source, "enc_channels", defaults.enc_channels)),
            enc_k_size=tuple(_cfg_value(source, "enc_k_size", defaults.enc_k_size)),
            aspp_channels=tuple(_cfg_value(source, "aspp_channels", defaults.aspp_channels)),
            rate=tuple(_cfg_value(source, "rate", defaults.rate)),
            dec_channels=tuple(_cfg_value(source, "dec_channels", defaults.dec_channels)),
            dec_k_size=tuple(_cfg_value(source, "dec_k_size", defaults.dec_k_size)),
            norm=bool(_cfg_value(source, "norm", defaults.norm)),
            act=str(_cfg_value(source, "act", defaults.act)),
            lif_activation=str(_cfg_value(source, "lif_activation", defaults.lif_activation)),
            lif_act_width=float(_cfg_value(source, "lif_act_width", defaults.lif_act_width)),
            lif_leak=tuple(_cfg_value(source, "lif_leak", defaults.lif_leak)),
            lif_thresh=tuple(_cfg_value(source, "lif_thresh", defaults.lif_thresh)),
            lif_learn_leak=bool(_cfg_value(source, "lif_learn_leak", defaults.lif_learn_leak)),
            lif_learn_thresh=bool(_cfg_value(source, "lif_learn_thresh", defaults.lif_learn_thresh)),
            lif_hard_reset=bool(_cfg_value(source, "lif_hard_reset", defaults.lif_hard_reset)),
        )
        if len(self.cfg.enc_channels) != 3 or len(self.cfg.enc_k_size) != 3:
            raise ValueError("Hybrid SNN encoder 固定需要三层 encoder 配置。")
        if not self.cfg.norm:
            raise ValueError("当前官方 EV-Slim 使用 BatchNorm，Hybrid 第一版要求 norm=True。")

        c1, c2, c3 = self.cfg.enc_channels
        k1, k2, k3 = self.cfg.enc_k_size

        # 卷积与 BatchNorm 参数逐项对齐官方 ConvLayer；激活改为 LIF/LIF/Accumulator。
        self.conv1 = nn.Conv2d(self.cfg.in_channels, c1, k1, padding="same", bias=False)
        self.bn1 = nn.BatchNorm2d(c1, momentum=0.01)
        self.lif1 = _ActivationOnlyConvLIF(c1, self.cfg)
        self.conv2 = nn.Conv2d(c1, c2, k2, padding="same", bias=False)
        self.bn2 = nn.BatchNorm2d(c2, momentum=0.01)
        self.lif2 = _ActivationOnlyConvLIF(c2, self.cfg)
        self.conv3 = nn.Conv2d(c2, c3, k3, padding="same", bias=False)
        self.bn3 = nn.BatchNorm2d(c3, momentum=0.01)

        self.aspp = nn.Sequential(
            *[
                ASPP(
                    in_channels=c3 if index == 0 else self.cfg.aspp_channels[index - 1],
                    out_channels=out_channels,
                    atrous_rates=self.cfg.rate,
                )
                for index, out_channels in enumerate(self.cfg.aspp_channels)
            ]
        )
        self.dec = nn.Sequential(
            *[
                ConvLayer(
                    in_channels=self.cfg.aspp_channels[-1] if index == 0 else self.cfg.dec_channels[index - 1],
                    out_channels=out_channels,
                    kernel_size=(self.cfg.dec_k_size[index], self.cfg.dec_k_size[index]),
                    activation=self.cfg.act if index < len(self.cfg.dec_channels) - 1 else None,
                    norm=self.cfg.norm if index < len(self.cfg.dec_channels) - 1 else None,
                )
                for index, out_channels in enumerate(self.cfg.dec_channels)
            ]
        )

        self._lif1_state: torch.Tensor | None = None
        self._lif2_state: torch.Tensor | None = None
        self._accumulator: torch.Tensor | None = None
        self._spike_sum = {"lif1": 0.0, "lif2": 0.0}
        self._spike_elements = {"lif1": 0, "lif2": 0}
        self.snn_step_count = 0
        self.ann_backend_call_count = 0
        self.last_step_shapes: dict[str, tuple[int, ...]] = {}

    @property
    def accumulator(self) -> torch.Tensor | None:
        """只读暴露当前累加特征，便于审计与单元测试。"""

        return self._accumulator

    def reset_states(self) -> None:
        """在 Block 开始处清空两层 LIF 状态和实值累加器。"""

        self._lif1_state = None
        self._lif2_state = None
        self._accumulator = None
        self._spike_sum = {"lif1": 0.0, "lif2": 0.0}
        self._spike_elements = {"lif1": 0, "lif2": 0}
        self.snn_step_count = 0
        self.ann_backend_call_count = 0
        self.last_step_shapes = {}

    def _record_spikes(self, name: str, spikes: torch.Tensor) -> None:
        detached = spikes.detach()
        self._spike_sum[name] += float(detached.sum().cpu())
        self._spike_elements[name] += detached.numel()

    def forward_snn_step(self, x_t: torch.Tensor) -> torch.Tensor:
        """处理一个时间步并返回截至当前步的实值累加特征。"""

        if x_t.ndim != 4 or x_t.shape[1] != self.cfg.in_channels:
            raise ValueError(
                f"期望单步输入 [B,{self.cfg.in_channels},H,W]，实际为 {tuple(x_t.shape)}"
            )
        current1 = self.bn1(self.conv1(x_t))
        spikes1, self._lif1_state = self.lif1(current1, self._lif1_state)
        current2 = self.bn2(self.conv2(spikes1))
        spikes2, self._lif2_state = self.lif2(current2, self._lif2_state)
        analog = self.bn3(self.conv3(spikes2))
        self._accumulator = analog if self._accumulator is None else self._accumulator + analog

        self._record_spikes("lif1", spikes1)
        self._record_spikes("lif2", spikes2)
        self.snn_step_count += 1
        self.last_step_shapes = {
            "input": tuple(x_t.shape),
            "conv1_bn": tuple(current1.shape),
            "lif1_spikes": tuple(spikes1.shape),
            "conv2_bn": tuple(current2.shape),
            "lif2_spikes": tuple(spikes2.shape),
            "conv3_bn": tuple(analog.shape),
            "accumulator": tuple(self._accumulator.shape),
        }
        return self._accumulator

    def forward_ann_backend(self, feature: torch.Tensor) -> torch.Tensor:
        """让原 EV-Slim ASPP 与 decoder/head 对累加特征只执行一次。"""

        if feature.ndim != 4 or feature.shape[1] != self.cfg.enc_channels[-1]:
            raise ValueError(
                f"ANN backend 期望 [B,{self.cfg.enc_channels[-1]},H,W]，实际为 {tuple(feature.shape)}"
            )
        self.ann_backend_call_count += 1
        return self.dec(self.aspp(feature))

    def forward_sequence(self, x_seq: torch.Tensor) -> torch.Tensor:
        """顺序编码 T 步，随后只运行一次 ANN 后端；本方法不会自动 reset。"""

        if x_seq.ndim != 5 or x_seq.shape[2] != self.cfg.in_channels:
            raise ValueError(
                f"期望序列输入 [B,T,{self.cfg.in_channels},H,W]，实际为 {tuple(x_seq.shape)}"
            )
        if x_seq.shape[1] < 1:
            raise ValueError("T 必须至少为 1。")
        feature: torch.Tensor | None = None
        for step in range(x_seq.shape[1]):
            feature = self.forward_snn_step(x_seq[:, step])
        assert feature is not None
        return self.forward_ann_backend(feature)

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        return self.forward_sequence(x_seq)

    def get_firing_rates(self) -> dict[str, float]:
        """返回当前 Block 内所有已执行时间步的平均发放率。"""

        return {
            name: self._spike_sum[name] / max(1, self._spike_elements[name])
            for name in ("lif1", "lif2")
        }

    def get_state_statistics(self) -> dict[str, Any]:
        """汇总膜电位、脉冲、累加器和执行次数。"""

        lif1_mem = self._lif1_state[0] if self._lif1_state is not None else None
        lif1_spikes = self._lif1_state[1] if self._lif1_state is not None else None
        lif2_mem = self._lif2_state[0] if self._lif2_state is not None else None
        lif2_spikes = self._lif2_state[1] if self._lif2_state is not None else None
        return {
            "lif1_membrane": _tensor_statistics(lif1_mem),
            "lif1_spikes": _tensor_statistics(lif1_spikes),
            "lif2_membrane": _tensor_statistics(lif2_mem),
            "lif2_spikes": _tensor_statistics(lif2_spikes),
            "accumulator": _tensor_statistics(self._accumulator),
            "firing_rates": self.get_firing_rates(),
            "snn_step_count": self.snn_step_count,
            "ann_backend_call_count": self.ann_backend_call_count,
        }

