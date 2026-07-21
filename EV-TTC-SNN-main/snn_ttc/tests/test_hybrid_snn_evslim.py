#!/usr/bin/env python3
"""Hybrid SNN-EV-Slim 网络与 Block 训练循环的随机张量单元测试。"""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
from pathlib import Path
from typing import Any

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_ROOT = PROJECT_ROOT / "EV-TTC-main/model"
if str(MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_ROOT))

from hybrid_block_training import (  # noqa: E402
    load_hybrid_checkpoint,
    save_hybrid_checkpoint,
    train_hybrid_block,
    validate_hybrid_block,
)
from hybrid_snn_evslim import HybridSNNEVSlim  # noqa: E402
from util import charbonnier_loss  # noqa: E402


DEFAULT_OUT_DIR = PROJECT_ROOT / "EV-TTC-SNN-main/reports/05_hybrid_snn_evslim"


def _finite(tensor: torch.Tensor | None) -> bool:
    return tensor is None or bool(torch.isfinite(tensor).all().detach().cpu())


def _module_grad_norm(module: torch.nn.Module) -> float:
    squared = 0.0
    seen = False
    for parameter in module.parameters():
        if parameter.grad is None:
            continue
        seen = True
        if not bool(torch.isfinite(parameter.grad).all().detach().cpu()):
            return float("nan")
        squared += float(parameter.grad.detach().float().square().sum().cpu())
    return math.sqrt(squared) if seen else 0.0


def _parameter_snapshot(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def _changed_parameters(before: dict[str, torch.Tensor], model: torch.nn.Module) -> list[str]:
    return [
        name
        for name, parameter in model.named_parameters()
        if name in before and not torch.equal(before[name], parameter.detach())
    ]


def _stats(tensor: torch.Tensor) -> dict[str, float]:
    value = tensor.detach().float()
    return {
        "mean": float(value.mean().cpu()),
        "std": float(value.std(unbiased=False).cpu()),
        "min": float(value.min().cpu()),
        "max": float(value.max().cpu()),
    }


def run_tests(out_dir: str | Path = DEFAULT_OUT_DIR) -> dict[str, Any]:
    torch.manual_seed(20260722)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = HybridSNNEVSlim().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    batch, steps, channels, height, width = 2, 3, 6, 64, 64
    x_seq = torch.randn(batch, steps, channels, height, width, device=device)
    target = torch.randn(batch, 1, height, width, device=device)
    mask = torch.rand(batch, 1, height, width, device=device) > 0.2
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, details: str) -> None:
        checks.append({"name": name, "passed": bool(passed), "details": details})
        print(f"[{'PASS' if passed else 'FAIL'}] {name}: {details}", flush=True)

    # 逐步前向用于核对尺寸、sum accumulator 和 reset 行为。
    model.train()
    model.reset_states()
    for step in range(steps):
        accumulator = model.forward_snn_step(x_seq[:, step])
        expected_shapes = {
            "input": (2, 6, 64, 64),
            "conv1_bn": (2, 16, 64, 64),
            "lif1_spikes": (2, 16, 64, 64),
            "conv2_bn": (2, 16, 64, 64),
            "lif2_spikes": (2, 16, 64, 64),
            "conv3_bn": (2, 16, 64, 64),
            "accumulator": (2, 16, 64, 64),
        }
        check(
            f"SNN step {step + 1} 中间尺寸",
            model.last_step_shapes == expected_shapes,
            str(model.last_step_shapes),
        )
    check(
        "三步累加器非零",
        bool(accumulator.abs().max().detach().cpu() > 0),
        json.dumps(_stats(accumulator), ensure_ascii=False),
    )
    three_step_accumulator = accumulator.detach().clone()

    model.reset_states()
    one_step_accumulator = model.forward_snn_step(x_seq[:, -1]).detach().clone()
    check(
        "三步与最后一步累加结果不同",
        not torch.allclose(three_step_accumulator, one_step_accumulator),
        f"mean_abs_diff={float((three_step_accumulator - one_step_accumulator).abs().mean().cpu()):.6g}",
    )
    model.reset_states()
    reset_stats = model.get_state_statistics()
    reset_ok = (
        model.accumulator is None
        and not reset_stats["lif1_membrane"]["initialized"]
        and not reset_stats["lif2_membrane"]["initialized"]
        and reset_stats["snn_step_count"] == 0
        and reset_stats["ann_backend_call_count"] == 0
    )
    check("reset清空膜电位和累加器", reset_ok, json.dumps(reset_stats, ensure_ascii=False))

    # 完整序列反传：每个时间步都应透过递归 LIF 状态和 sum accumulator 获得梯度。
    gradient_model = HybridSNNEVSlim().to(device).train()
    gradient_input = torch.randn(
        batch, steps, channels, height, width, device=device, requires_grad=True
    )
    gradient_model.reset_states()
    gradient_prediction = gradient_model.forward_sequence(gradient_input)
    gradient_loss = charbonnier_loss(target - gradient_prediction, alpha=0.45, mask=mask)
    gradient_loss.backward()
    time_step_gradients: list[dict[str, float]] = []
    assert gradient_input.grad is not None
    for step in range(steps):
        current_gradient = gradient_input.grad[:, step].detach()
        row = {
            "gradient_sum": float(current_gradient.abs().sum().cpu()),
            "gradient_mean": float(current_gradient.abs().mean().cpu()),
            "gradient_max": float(current_gradient.abs().max().cpu()),
            "finite": bool(torch.isfinite(current_gradient).all().cpu()),
        }
        time_step_gradients.append(row)
        check(
            f"时间步t={step}输入梯度非零有限",
            row["finite"] and row["gradient_sum"] > 0.0,
            json.dumps(row, ensure_ascii=False),
        )

    # 输入消融：每一时刻被置零都必须改变最终预测，防止只使用最后一个 step。
    ablation_model = HybridSNNEVSlim().to(device).eval()
    ablation_input = torch.randn(batch, steps, channels, height, width, device=device)
    with torch.no_grad():
        ablation_model.reset_states()
        reference_prediction = ablation_model.forward_sequence(ablation_input)
        ablation_l1_differences: dict[str, float] = {}
        for step in range(steps):
            altered = ablation_input.clone()
            altered[:, step] = 0.0
            ablation_model.reset_states()
            altered_prediction = ablation_model.forward_sequence(altered)
            difference = float((reference_prediction - altered_prediction).abs().sum().cpu())
            ablation_l1_differences[f"zero_t{step}"] = difference
            check(
                f"置零时间步t={step}会改变输出",
                difference > 0.0,
                f"L1_difference={difference:.8f}",
            )

    # 调用独立 Block 训练模块，验证一次监督、梯度和优化器更新。
    before = _parameter_snapshot(model)
    train_result = train_hybrid_block(model, optimizer, x_seq, target, mask, use_amp=False)
    state_stats = model.get_state_statistics()
    firing_rates = model.get_firing_rates()
    grad_norms = {
        "conv1": _module_grad_norm(model.conv1),
        "conv2": _module_grad_norm(model.conv2),
        "conv3": _module_grad_norm(model.conv3),
        "aspp": _module_grad_norm(model.aspp),
        "ttc_head": _module_grad_norm(model.dec[-1]),
        "snn_encoder": math.sqrt(
            _module_grad_norm(model.conv1) ** 2
            + _module_grad_norm(model.lif1) ** 2
            + _module_grad_norm(model.conv2) ** 2
            + _module_grad_norm(model.lif2) ** 2
            + _module_grad_norm(model.conv3) ** 2
        ),
        "ann_backend": math.sqrt(_module_grad_norm(model.aspp) ** 2 + _module_grad_norm(model.dec) ** 2),
    }
    changed = _changed_parameters(before, model)

    check("输出shape", tuple(train_result.prediction.shape) == (2, 1, 64, 64), str(tuple(train_result.prediction.shape)))
    check("Loss正常", math.isfinite(train_result.loss), f"loss={train_result.loss:.8f}")
    check(
        "一个Block执行3个SNN step和1次ANN",
        train_result.snn_steps == 3 and train_result.ann_backend_calls == 1,
        f"snn_steps={train_result.snn_steps}, ann_calls={train_result.ann_backend_calls}",
    )
    for name in ("conv1", "conv2", "conv3", "aspp", "ttc_head"):
        value = grad_norms[name]
        check(f"{name}梯度非零有限", math.isfinite(value) and value > 0, f"grad_norm={value:.8g}")
    check("optimizer.step更新参数", len(changed) > 0, f"changed_parameter_count={len(changed)}")

    # Checkpoint 必须同时包含 SNN encoder、ANN backend 和优化器状态。
    with tempfile.TemporaryDirectory(prefix="hybrid_snn_evslim_test_") as temp_dir:
        checkpoint_path = Path(temp_dir) / "checkpoint.pt"
        save_hybrid_checkpoint(checkpoint_path, model, optimizer, epoch=1)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_keys = set(checkpoint["model_state_dict"])
    checkpoint_ok = (
        any(key.startswith("conv1.") for key in state_keys)
        and any(key.startswith("lif1.") for key in state_keys)
        and any(key.startswith("conv3.") for key in state_keys)
        and any(key.startswith("aspp.") for key in state_keys)
        and any(key.startswith("dec.") for key in state_keys)
        and bool(checkpoint["optimizer_state_dict"]["state"])
    )
    check("Checkpoint包含完整SNN/ANN参数", checkpoint_ok, f"model_state_keys={len(state_keys)}")

    # CUDA 环境才执行真实 AMP；CPU 环境将明确标记跳过，不能把 FP32 当 AMP 成功。
    amp_result: dict[str, Any]
    if device.type == "cuda":
        amp_model = HybridSNNEVSlim().to(device)
        amp_optimizer = torch.optim.AdamW(amp_model.parameters(), lr=1e-3)
        amp_before = _parameter_snapshot(amp_model)
        amp_train = train_hybrid_block(
            amp_model, amp_optimizer, x_seq, target, mask, use_amp=True
        )
        amp_gradients_finite = all(
            parameter.grad is None or bool(torch.isfinite(parameter.grad).all().cpu())
            for parameter in amp_model.parameters()
        )
        assert amp_train.scaler is not None
        amp_scale = float(amp_train.scaler.get_scale())
        amp_changed = _changed_parameters(amp_before, amp_model)
        with tempfile.TemporaryDirectory(prefix="hybrid_snn_evslim_amp_") as temp_dir:
            amp_checkpoint_path = Path(temp_dir) / "amp_checkpoint.pt"
            save_hybrid_checkpoint(
                amp_checkpoint_path,
                amp_model,
                amp_optimizer,
                epoch=1,
                scaler=amp_train.scaler,
            )
            restored_model = HybridSNNEVSlim().to(device)
            restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=1e-3)
            restored_scaler = torch.amp.GradScaler("cuda", enabled=True)
            restored_checkpoint = load_hybrid_checkpoint(
                amp_checkpoint_path,
                restored_model,
                restored_optimizer,
                scaler=restored_scaler,
                map_location=device,
            )
        scaler_restored = restored_checkpoint["scaler"] == restored_scaler.state_dict()
        amp_ok = (
            math.isfinite(amp_train.loss)
            and len(amp_changed) > 0
            and amp_gradients_finite
            and math.isfinite(amp_scale)
            and amp_scale > 0.0
            and scaler_restored
        )
        amp_result = {
            "status": "passed" if amp_ok else "failed",
            "loss": amp_train.loss,
            "parameter_changed_count": len(amp_changed),
            "gradients_finite": amp_gradients_finite,
            "scaler_scale": amp_scale,
            "checkpoint_scaler_restored": scaler_restored,
        }
        check("CUDA AMP训练与scaler恢复", amp_ok, json.dumps(amp_result, ensure_ascii=False))
    else:
        amp_result = {"status": "skipped", "reason": "当前环境没有CUDA，未伪造AMP测试结果。"}
        print(f"[SKIP] CUDA AMP训练与scaler恢复: {amp_result['reason']}", flush=True)

    states_finite = all(
        bool(item["finite"])
        for key, item in state_stats.items()
        if isinstance(item, dict) and key != "firing_rates" and "finite" in item
    )
    gradients_finite = all(math.isfinite(value) for value in grad_norms.values())
    check(
        "输出/膜电位/累加器/梯度无NaN或Inf",
        _finite(train_result.prediction) and states_finite and gradients_finite,
        f"states_finite={states_finite}, gradients_finite={gradients_finite}",
    )

    # Validation 必须自行 reset，且仍只调用一次 ANN backend。
    model.eval()
    validation = validate_hybrid_block(model, x_seq, target, mask)
    check(
        "Validation按Block reset",
        validation.snn_steps == 3 and validation.ann_backend_calls == 1,
        f"snn_steps={validation.snn_steps}, ann_calls={validation.ann_backend_calls}",
    )

    summary = {
        "passed": all(item["passed"] for item in checks),
        "device": str(device),
        "input_shape": list(x_seq.shape),
        "output_shape": list(train_result.prediction.shape),
        "checks": checks,
        "firing_rates": firing_rates,
        "accumulator": state_stats["accumulator"],
        "prediction": _stats(train_result.prediction),
        "gradient_norms": grad_norms,
        "time_step_input_gradients": time_step_gradients,
        "time_step_ablation_l1_differences": ablation_l1_differences,
        "amp": amp_result,
        "changed_parameter_count": len(changed),
    }
    print("\n关键统计：", flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)

    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "hybrid_snn_evslim_unit_test.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report = f"""# Hybrid SNN-EV-Slim 随机张量单元测试

- 总体结果：{'通过' if summary['passed'] else '失败'}
- 设备：`{device}`
- 输入：`{list(x_seq.shape)}`
- 输出：`{list(train_result.prediction.shape)}`
- LIF1/LIF2发放率：`{firing_rates}`
- accumulator：`{state_stats['accumulator']}`
- prediction：`{summary['prediction']}`
- 梯度范数：`{grad_norms}`
- 时间步输入梯度：`{time_step_gradients}`
- 时间步置零输出 L1 差异：`{ablation_l1_differences}`
- AMP：`{amp_result}`
- 更新参数数量：`{len(changed)}`

详细检查见同目录 `hybrid_snn_evslim_unit_test.json`。
"""
    (output / "HYBRID_SNN_EVSLIM_UNIT_TEST.md").write_text(report, encoding="utf-8")
    return summary


def test_hybrid_snn_evslim_required_behavior(tmp_path: Path) -> None:
    """供 pytest 调用的完整必要行为测试。"""

    assert run_tests(tmp_path)["passed"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    return 0 if run_tests(args.out_dir)["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
