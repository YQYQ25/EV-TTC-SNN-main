#!/usr/bin/env python3
"""阶段3：LIF模型与masked Charbonnier loss最小集成测试。"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import hdf5plugin  # noqa: F401
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SNN_ROOT = PROJECT_ROOT / "EV-TTC-SNN-main"
if str(SNN_ROOT) not in sys.path:
    sys.path.insert(0, str(SNN_ROOT))

from snn_ttc.losses import masked_charbonnier_per_sample  # noqa: E402
from snn_ttc.models import LIFEVFlowNetTTC  # noqa: E402
from snn_ttc.tests.report_utils import ensure_out_dir, markdown_table, pass_text, write_csv, write_json  # noqa: E402


DEFAULT_S2_H5 = (
    PROJECT_ROOT
    / "EV-TTC-SNN-main"
    / "debug_sets"
    / "[5]car_spatial_ablation"
    / "independent_fixed_n"
    / "N10000_S2_center256_to128.h5"
)


@dataclass
class BlockResult:
    block_type: str
    forward_step_count: int
    valid_sample_step_count: int
    backward_called: bool
    optimizer_step_called: bool
    detach_called: bool
    reset_called: bool
    grad_norm_before_or_returned: float | str
    parameter_changed: bool
    block_loss: float | str
    output_finite: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "block_type": self.block_type,
            "forward_step_count": self.forward_step_count,
            "valid_sample_step_count": self.valid_sample_step_count,
            "backward_called": int(self.backward_called),
            "optimizer_step_called": int(self.optimizer_step_called),
            "detach_called": int(self.detach_called),
            "reset_called": int(self.reset_called),
            "grad_norm_before_or_returned": self.grad_norm_before_or_returned,
            "parameter_changed": int(self.parameter_changed),
            "block_loss": self.block_loss,
            "output_finite": int(self.output_finite),
        }


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _snapshot_params(model: torch.nn.Module) -> list[torch.Tensor]:
    return [p.detach().clone() for p in model.parameters() if p.requires_grad]


def _params_changed(before: list[torch.Tensor], model: torch.nn.Module) -> bool:
    after = [p.detach() for p in model.parameters() if p.requires_grad]
    return any(not torch.allclose(a, b, atol=1e-8, rtol=1e-6) for a, b in zip(before, after))


def _grad_finite(model: torch.nn.Module) -> bool:
    seen = False
    for p in model.parameters():
        if p.grad is None:
            continue
        seen = True
        if not bool(torch.isfinite(p.grad).all().detach().cpu()):
            return False
    return seen


def run_training_block(
    *,
    block_type: str,
    events: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    supervise_valid: torch.Tensor,
    reset_at_start: bool,
    gradient_clip: float = 100.0,
) -> BlockResult:
    """执行一个10步block：所有step前向，只有有效step-sample累计loss。"""

    model = LIFEVFlowNetTTC().to(events.device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    if reset_at_start:
        model.reset_states()
    before = _snapshot_params(model)
    optimizer.zero_grad(set_to_none=True)

    loss_sum: torch.Tensor | None = None
    valid_sample_step_count = 0
    output_finite = True
    forward_step_count = 0
    for t in range(events.shape[1]):
        pred_t = model(events[:, t])
        forward_step_count += 1
        output_finite = output_finite and bool(torch.isfinite(pred_t).all().detach().cpu())
        per_sample_loss, valid_samples, _ = masked_charbonnier_per_sample(
            prediction=pred_t,
            target=target[:, t],
            valid_ttc_mask=mask[:, t],
            supervise_valid=supervise_valid[:, t],
        )
        if bool(valid_samples.any().detach().cpu()):
            current_sum = per_sample_loss[valid_samples].sum()
            loss_sum = current_sum if loss_sum is None else loss_sum + current_sum
            valid_sample_step_count += int(valid_samples.sum().detach().cpu().item())

    backward_called = False
    optimizer_step_called = False
    grad_norm: float | str = "NA"
    block_loss_value: float | str = "NA"
    if valid_sample_step_count > 0 and loss_sum is not None:
        block_loss = loss_sum / valid_sample_step_count
        block_loss_value = float(block_loss.detach().cpu())
        block_loss.backward()
        backward_called = True
        grad_ok = _grad_finite(model)
        grad_norm_tensor = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip)
        grad_norm = float(grad_norm_tensor.detach().cpu())
        if grad_ok and math.isfinite(grad_norm):
            optimizer.step()
            optimizer_step_called = True

    # block末尾必须detach；零监督block也需要截断历史图。
    model.detach_states()
    detach_called = True
    changed = _params_changed(before, model)
    return BlockResult(
        block_type=block_type,
        forward_step_count=forward_step_count,
        valid_sample_step_count=valid_sample_step_count,
        backward_called=backward_called,
        optimizer_step_called=optimizer_step_called,
        detach_called=detach_called,
        reset_called=reset_at_start,
        grad_norm_before_or_returned=grad_norm,
        parameter_changed=changed,
        block_loss=block_loss_value,
        output_finite=output_finite,
    )


def _make_synthetic_block(block_type: str, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """构造A/B/C三类人工block，尺寸保持128x128。"""

    torch.manual_seed({"A_all_valid": 11, "B_partial_valid": 12, "C_no_supervision": 13}[block_type])
    batch, steps, height, width = 2, 10, 128, 128
    events = torch.randn(batch, steps, 2, height, width, device=device) * 0.1
    target = torch.randn(batch, steps, height, width, device=device) * 0.05
    mask = torch.ones(batch, steps, height, width, dtype=torch.bool, device=device)
    supervise = torch.ones(batch, steps, dtype=torch.bool, device=device)
    expected = batch * steps

    if block_type == "B_partial_valid":
        supervise[:] = False
        supervise[0, [0, 1, 3, 5, 9]] = True
        supervise[1, [2, 4, 6, 8]] = True
        mask[1, 8] = False
        expected = int((supervise & (mask.flatten(2).sum(dim=2) > 0)).sum().item())
    elif block_type == "C_no_supervision":
        supervise[:] = False
        mask[:] = False
        expected = 0
    return events, target, mask, supervise, expected


def _load_real_s2_block(path: Path, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """读取一个真实S2连续10步block，只用于一次最小集成测试。"""

    with h5py.File(path, "r") as f:
        block_index = f["block_index"][:]
        step_in_block = f["step_in_block"][:]
        start_row = 0
        for start in range(0, len(block_index) - 9):
            rows = np.arange(start, start + 10)
            if len(set(block_index[rows].tolist())) == 1 and set(step_in_block[rows].tolist()) == set(range(10)):
                start_row = int(start)
                break
        sl = slice(start_row, start_row + 10)
        events_np = f["event_cnt"][sl].astype("float32")
        target_np = f["signed_inverse_ttc"][sl].astype("float32")
        mask_np = f["valid_ttc_mask"][sl].astype(bool)
        supervise_np = f["supervise_valid"][sl].astype(bool)
    events = torch.from_numpy(events_np).unsqueeze(0).to(device)
    target = torch.from_numpy(target_np).unsqueeze(0).to(device)
    mask = torch.from_numpy(mask_np).unsqueeze(0).to(device)
    supervise = torch.from_numpy(supervise_np).unsqueeze(0).to(device)
    expected = int((supervise & (mask.flatten(2).sum(dim=2) > 0)).sum().item())
    return events, target, mask, supervise, expected


def run_tests(out_dir: str | Path | None = None, s2_h5: str | Path = DEFAULT_S2_H5) -> dict[str, Any]:
    out = ensure_out_dir(out_dir)
    device = _device()
    audit_rows: list[dict[str, Any]] = []
    checks: list[dict[str, Any]] = []

    for block_type in ["A_all_valid", "B_partial_valid", "C_no_supervision"]:
        events, target, mask, supervise, expected = _make_synthetic_block(block_type, device)
        result = run_training_block(
            block_type=block_type,
            events=events,
            target=target,
            mask=mask,
            supervise_valid=supervise,
            reset_at_start=True,
        )
        audit_rows.append(result.to_dict())
        if block_type == "A_all_valid":
            passed = (
                result.forward_step_count == 10
                and result.valid_sample_step_count == expected
                and result.backward_called
                and result.optimizer_step_called
                and result.detach_called
                and result.parameter_changed
                and result.output_finite
            )
        elif block_type == "B_partial_valid":
            passed = (
                result.forward_step_count == 10
                and result.valid_sample_step_count == expected
                and result.backward_called
                and result.optimizer_step_called
                and result.detach_called
                and result.parameter_changed
                and result.output_finite
            )
        else:
            passed = (
                result.forward_step_count == 10
                and result.valid_sample_step_count == 0
                and not result.backward_called
                and not result.optimizer_step_called
                and result.detach_called
                and not result.parameter_changed
                and result.output_finite
            )
        checks.append({"test": block_type, "passed": passed, "details": f"expected_valid={expected}, got={result.valid_sample_step_count}"})
        del events, target, mask, supervise
        if device.type == "cuda":
            torch.cuda.empty_cache()

    real_events, real_target, real_mask, real_supervise, real_expected = _load_real_s2_block(Path(s2_h5), device)
    real_result = run_training_block(
        block_type="real_S2_single_block",
        events=real_events,
        target=real_target,
        mask=real_mask,
        supervise_valid=real_supervise,
        reset_at_start=True,
    )
    audit_rows.append(real_result.to_dict())
    real_passed = (
        real_result.forward_step_count == 10
        and real_result.valid_sample_step_count == real_expected
        and real_result.valid_sample_step_count > 0
        and real_result.backward_called
        and real_result.optimizer_step_called
        and real_result.detach_called
        and real_result.parameter_changed
        and real_result.output_finite
        and isinstance(real_result.block_loss, float)
        and math.isfinite(real_result.block_loss)
    )
    checks.append({"test": "真实S2单block", "passed": real_passed, "details": f"expected_valid={real_expected}, loss={real_result.block_loss}"})

    passed = all(bool(item["passed"]) for item in checks)
    write_csv(out / "阶段3_训练循环运行时审计.csv", audit_rows)
    write_json(out / "阶段3_模型与Loss最小集成测试结果.json", {"passed": passed, "device": str(device), "checks": checks, "s2_h5": str(s2_h5), "audit": audit_rows})

    check_table = [[row["test"], pass_text(bool(row["passed"])), row["details"]] for row in checks]
    audit_table = [
        [
            row["block_type"],
            row["forward_step_count"],
            row["valid_sample_step_count"],
            row["backward_called"],
            row["optimizer_step_called"],
            row["detach_called"],
            row["reset_called"],
            row["grad_norm_before_or_returned"],
            row["parameter_changed"],
        ]
        for row in audit_rows
    ]
    report = f"""# 阶段3 模型与Loss最小集成测试报告

## 结论
- 总体结果：{pass_text(passed)}
- 测试设备：`{device}`
- 真实S2 H5：`{s2_h5}`

## 行为检查
{markdown_table(["测试项", "结果", "说明"], check_table)}

## 运行时审计
{markdown_table(["block_type", "forward", "valid_step_sample", "backward", "optimizer", "detach", "reset", "grad_norm", "param_changed"], audit_table)}

## 关键确认
- 每个block的10个step全部执行forward。
- loss只累计 `supervise_valid & valid_ttc_mask.any()` 的 step-sample。
- block loss按有效 step-sample 平均，不固定除以10。
- 零监督block不执行backward和optimizer.step，但仍执行detach_states。
- 有效block完成一次BPTT、梯度裁剪和optimizer.step。
- 真实S2单block只做一次最小更新，没有进入32-block过拟合或正式训练。
"""
    (out / "阶段3_模型与Loss最小集成测试报告.md").write_text(report, encoding="utf-8")
    return {"passed": passed, "checks": checks, "audit": audit_rows, "out_dir": str(out)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--s2-h5", type=Path, default=DEFAULT_S2_H5)
    args = parser.parse_args()
    result = run_tests(args.out_dir, args.s2_h5)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
