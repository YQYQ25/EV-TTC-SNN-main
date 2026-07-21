"""复用 MAVLab event_flow 中的 LIF/EV-FlowNet 基础模块。

这里不重新实现脉冲神经元和 U-Net 主体，只把 `event_flow-main` 加到
Python 路径后做显式导入，便于 TTC 任务的封装代码复用同一份实现。
"""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
EVENT_FLOW_ROOT = PROJECT_ROOT / "event_flow-main"

if str(EVENT_FLOW_ROOT) not in sys.path:
    sys.path.insert(0, str(EVENT_FLOW_ROOT))

from models.model_util import CropParameters, copy_states  # noqa: E402
from models.spiking_submodules import (  # noqa: E402
    ConvLIF,
    SpikingRecurrentConvLayer,
    SpikingResidualBlock,
    SpikingTransposedConvLayer,
    SpikingUpsampleConvLayer,
)
from models.unet import SpikingMultiResUNetRecurrent  # noqa: E402


# 与 event_flow-main/configs/train_SNN.yml 保持一致的 MAVLab LIF 参数。
MAVLAB_LIF_NEURON = {
    "leak": [-4.0, 0.1],
    "thresh": [0.8, 0.1],
    "learn_leak": True,
    "learn_thresh": True,
    "hard_reset": True,
}


__all__ = [
    "CropParameters",
    "copy_states",
    "ConvLIF",
    "SpikingRecurrentConvLayer",
    "SpikingResidualBlock",
    "SpikingTransposedConvLayer",
    "SpikingUpsampleConvLayer",
    "SpikingMultiResUNetRecurrent",
    "MAVLAB_LIF_NEURON",
]
