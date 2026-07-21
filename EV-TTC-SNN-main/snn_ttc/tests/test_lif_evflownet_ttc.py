#!/usr/bin/env python3
"""阶段1：LIF-EV-FlowNet-TTC 模型迁移单元测试。"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SNN_ROOT = PROJECT_ROOT / "EV-TTC-SNN-main"
if str(SNN_ROOT) not in sys.path:
    sys.path.insert(0, str(SNN_ROOT))

from snn_ttc.models import LIFEVFlowNetTTC  # noqa: E402
from snn_ttc.tests.report_utils import ensure_out_dir, markdown_table, pass_text, write_csv, write_json  # noqa: E402


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _all_finite_grads(model: torch.nn.Module) -> tuple[bool, bool]:
    """检查梯度是否存在、有限且不是全零。"""

    has_grad = False
    nonzero = False
    for param in model.parameters():
        if param.grad is None:
            continue
        has_grad = True
        if not bool(torch.isfinite(param.grad).all().detach().cpu()):
            return has_grad, False
        nonzero = nonzero or bool((param.grad.abs().max() > 0).detach().cpu())
    return has_grad, nonzero


def _signature_allclose(a: list[torch.Tensor], b: list[torch.Tensor], atol: float = 1e-6) -> bool:
    if len(a) != len(b):
        return False
    return all(torch.allclose(x, y, atol=atol, rtol=1e-5) for x, y in zip(a, b))


def _state_requires_grad_flags(model: LIFEVFlowNetTTC) -> list[bool]:
    flags: list[bool] = []
    for state in model.multires_unetrec.states:
        if state is None:
            continue
        if torch.is_tensor(state):
            flags.append(bool(state.requires_grad))
        elif isinstance(state, (tuple, list)):
            for item in state:
                if torch.is_tensor(item):
                    flags.append(bool(item.requires_grad))
    return flags


def run_tests(out_dir: str | Path | None = None) -> dict[str, Any]:
    torch.manual_seed(1)
    device = _device()
    out = ensure_out_dir(out_dir)
    rows: list[dict[str, Any]] = []

    model = LIFEVFlowNetTTC().to(device)
    model.train()

    # 128输入：连续10步只检查模型迁移，不接正式H5和loss。
    x128 = torch.randn(1, 2, 128, 128, device=device)
    model.reset_states()
    preds_128 = []
    for _ in range(10):
        preds_128.append(model(x128))
    pred128 = preds_128[-1]
    output_has_pos_neg = bool((pred128 > 0).any().detach().cpu()) and bool((pred128 < 0).any().detach().cpu())
    rows.append(
        {
            "test": "128连续10步forward",
            "passed": pred128.shape == (1, 1, 128, 128) and bool(torch.isfinite(pred128).all().detach().cpu()),
            "details": f"shape={tuple(pred128.shape)}, pos_neg={output_has_pos_neg}",
        }
    )

    # 状态保留、reset与detach测试。
    model.reset_states()
    y1 = model(x128)
    y2 = model(x128)
    state_before_reset = model.state_value_signature()
    model.reset_states()
    y3 = model(x128)
    reset_close = torch.allclose(y1, y3, atol=1e-6, rtol=1e-5)
    y12_diff = float((y2 - y1).abs().mean().detach().cpu())
    rows.append(
        {
            "test": "状态保留与reset",
            "passed": bool(reset_close),
            "details": f"mean_abs(y2-y1)={y12_diff:.6g}, reset_y3_close_y1={bool(reset_close)}",
        }
    )

    detach_before = model.state_value_signature()
    flags_before = _state_requires_grad_flags(model)
    model.detach_states()
    detach_after = model.state_value_signature()
    flags_after = _state_requires_grad_flags(model)
    detach_ok = _signature_allclose(detach_before, detach_after) and all(not flag for flag in flags_after)
    rows.append(
        {
            "test": "detach_states",
            "passed": detach_ok,
            "details": f"before_requires_grad={any(flags_before)}, after_requires_grad={any(flags_after)}",
        }
    )

    # 10步BPTT：确认反传、有限梯度和非零梯度。
    model.zero_grad(set_to_none=True)
    model.reset_states()
    inputs = torch.randn(1, 10, 2, 128, 128, device=device)
    loss = None
    for t in range(10):
        pred = model(inputs[:, t])
        current = pred.mean()
        loss = current if loss is None else loss + current
    assert loss is not None
    loss.backward()
    has_grad, nonzero_grad = _all_finite_grads(model)
    model.detach_states()
    rows.append(
        {
            "test": "10步BPTT",
            "passed": has_grad and nonzero_grad and math.isfinite(float(loss.detach().cpu())),
            "details": f"loss={float(loss.detach().cpu()):.6g}, has_grad={has_grad}, nonzero_grad={nonzero_grad}",
        }
    )
    del inputs, loss, preds_128
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # 360输入只做无梯度forward；内部pad到368再crop回360。
    model.eval()
    model.reset_states()
    with torch.no_grad():
        x360 = torch.randn(1, 2, 360, 360, device=device)
        pred360, activity = model(x360, return_activity=True)
    rows.append(
        {
            "test": "360 padding/crop forward",
            "passed": pred360.shape == (1, 1, 360, 360) and bool(torch.isfinite(pred360).all().detach().cpu()),
            "details": f"shape={tuple(pred360.shape)}, stride={model.total_stride}",
        }
    )
    activity_ok = len(activity) > 0 and all(
        {"membrane_mean", "membrane_std", "membrane_max", "spike_rate", "all_zero_neuron_ratio", "high_frequency_ratio"}.issubset(row.keys())
        for row in activity
    )
    rows.append(
        {
            "test": "脉冲活动监控",
            "passed": activity_ok,
            "details": f"layers={len(activity)}",
        }
    )
    rows.append(
        {
            "test": "输出正负支持",
            "passed": output_has_pos_neg,
            "details": f"min={float(pred128.min().detach().cpu()):.6g}, max={float(pred128.max().detach().cpu()):.6g}",
        }
    )

    passed = all(bool(row["passed"]) for row in rows)
    write_csv(out / "阶段1_模型单元测试结果.csv", rows)
    write_json(out / "阶段1_模型单元测试结果.json", {"passed": passed, "device": str(device), "rows": rows, "activity_example": activity[:3]})

    table = [[r["test"], pass_text(bool(r["passed"])), r["details"]] for r in rows]
    report = f"""# 阶段1 LIF-EV-FlowNet-TTC模型迁移报告

## 结论
- 总体结果：{pass_text(passed)}
- 测试设备：`{device}`
- 模型主体：复用 MAVLab `SpikingMultiResUNetRecurrent`
- 输出修改：多尺度预测头 `num_output_channels=1`，输出激活为 `None`

## 单元测试
{markdown_table(["测试项", "结果", "说明"], table)}

## 关键确认
- 128输入输出为 `[B,1,128,128]`。
- 360输入不resize，先按总stride={model.total_stride} padding到合法尺寸，再crop回360x360。
- `reset_states()` 后同一输入的输出回到初始状态。
- `detach_states()` 不改变状态数值，但会截断上一段计算图。
- 10步BPTT能够反传，梯度有限且非零。
- 已实现膜电位和spike活动统计接口。
"""
    (out / "阶段1_LIF-EV-FlowNet-TTC模型迁移报告.md").write_text(report, encoding="utf-8")
    return {"passed": passed, "rows": rows, "out_dir": str(out)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()
    result = run_tests(args.out_dir)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
