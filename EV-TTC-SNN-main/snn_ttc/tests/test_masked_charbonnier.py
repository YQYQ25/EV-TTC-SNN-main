#!/usr/bin/env python3
"""阶段2：EV-TTC masked Charbonnier per-sample loss 单元测试。"""

from __future__ import annotations

import argparse
import importlib.util
import math
import sys
from pathlib import Path
from typing import Any

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SNN_ROOT = PROJECT_ROOT / "EV-TTC-SNN-main"
if str(SNN_ROOT) not in sys.path:
    sys.path.insert(0, str(SNN_ROOT))

from snn_ttc.losses import (  # noqa: E402
    EVTTC_CHARBONNIER_ALPHA,
    EVTTC_CHARBONNIER_EPS,
    charbonnier,
    evttc_reference_per_sample,
    masked_charbonnier_per_sample,
)
from snn_ttc.tests.report_utils import ensure_out_dir, markdown_table, pass_text, write_csv, write_json  # noqa: E402


def _load_evttc_original_loss():
    """直接加载 EV-TTC-main/model/util.py 中的原始 loss 函数。"""

    util_path = PROJECT_ROOT / "EV-TTC-main" / "model" / "util.py"
    spec = importlib.util.spec_from_file_location("evttc_original_util", util_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载 EV-TTC 原始loss文件：{util_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.charbonnier_loss


def _mean_valid(per_sample: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    return per_sample[valid].mean() if bool(valid.any()) else torch.tensor(float("nan"))


def _record(rows: list[dict[str, Any]], name: str, passed: bool, details: str) -> None:
    rows.append({"test": name, "passed": bool(passed), "details": details})


def run_tests(out_dir: str | Path | None = None) -> dict[str, Any]:
    torch.manual_seed(2)
    out = ensure_out_dir(out_dir)
    rows: list[dict[str, Any]] = []

    # A：预测等于标签，loss为Charbonnier最小值。
    pred = torch.zeros(2, 1, 4, 4)
    target = torch.zeros(2, 1, 4, 4)
    mask = torch.ones(2, 4, 4, dtype=torch.bool)
    supervise = torch.ones(2, dtype=torch.bool)
    per_sample, valid, stats = masked_charbonnier_per_sample(pred, target, mask, supervise)
    expected_min = float(charbonnier(torch.tensor(0.0)))
    _record(
        rows,
        "A 预测等于标签",
        bool(torch.allclose(per_sample[valid], torch.full_like(per_sample[valid], expected_min), atol=1e-8)),
        f"loss={per_sample.tolist()}, min={expected_min:.8g}",
    )

    # B：只修改mask外像素，loss必须不变。
    mask_b = torch.zeros(1, 4, 4, dtype=torch.bool)
    mask_b[:, :2, :2] = True
    pred_b1 = torch.zeros(1, 1, 4, 4)
    pred_b2 = pred_b1.clone()
    pred_b2[:, :, 2:, 2:] = 100.0
    loss_b1, valid_b1, _ = masked_charbonnier_per_sample(pred_b1, torch.zeros_like(pred_b1), mask_b, torch.ones(1, dtype=torch.bool))
    loss_b2, valid_b2, _ = masked_charbonnier_per_sample(pred_b2, torch.zeros_like(pred_b2), mask_b, torch.ones(1, dtype=torch.bool))
    _record(rows, "B mask外像素不影响loss", bool(valid_b1[0] and valid_b2[0] and torch.allclose(loss_b1, loss_b2)), f"{float(loss_b1[0]):.8g} vs {float(loss_b2[0]):.8g}")

    # C：修改mask内像素，loss应增大。
    pred_c = pred_b1.clone()
    pred_c[:, :, 0, 0] = 1.0
    loss_c, _, _ = masked_charbonnier_per_sample(pred_c, torch.zeros_like(pred_c), mask_b, torch.ones(1, dtype=torch.bool))
    _record(rows, "C mask内像素增大loss", float(loss_c[0]) > float(loss_b1[0]), f"base={float(loss_b1[0]):.8g}, changed={float(loss_c[0]):.8g}")

    # D：supervise_valid=0 的样本不参与有效样本集合。
    pred_d = torch.zeros(2, 1, 4, 4)
    pred_d[1] = 50.0
    loss_d, valid_d, _ = masked_charbonnier_per_sample(pred_d, torch.zeros_like(pred_d), torch.ones(2, 4, 4, dtype=torch.bool), torch.tensor([True, False]))
    _record(rows, "D supervise_valid=0排除样本", valid_d.tolist() == [True, False] and float(loss_d[1]) == 0.0, f"valid={valid_d.tolist()}, loss={loss_d.tolist()}")

    # E：负signed inverse TTC正常回归，不做截断。
    pred_e = torch.full((1, 1, 2, 2), -0.4)
    target_e = torch.full((1, 1, 2, 2), -0.5)
    loss_e, valid_e, _ = masked_charbonnier_per_sample(pred_e, target_e, torch.ones(1, 2, 2, dtype=torch.bool), torch.ones(1, dtype=torch.bool))
    expected_e = float(charbonnier(torch.tensor(0.1)))
    _record(rows, "E 负标签正常参与", bool(valid_e[0]) and math.isclose(float(loss_e[0]), expected_e, rel_tol=1e-6), f"loss={float(loss_e[0]):.8g}, expected={expected_e:.8g}")

    # F：空mask不产生NaN，且valid_samples=False。
    loss_f, valid_f, stats_f = masked_charbonnier_per_sample(torch.ones(1, 1, 2, 2), torch.zeros(1, 1, 2, 2), torch.zeros(1, 2, 2), torch.ones(1, dtype=torch.bool))
    _record(rows, "F 空mask安全处理", not bool(valid_f[0]) and torch.isfinite(loss_f).all().item() and not stats_f.has_supervision, f"valid={valid_f.tolist()}, loss={loss_f.tolist()}")

    # G：相同误差、不同mask面积，每个样本loss权重一致。
    pred_g = torch.full((2, 1, 4, 4), 0.2)
    target_g = torch.zeros_like(pred_g)
    mask_g = torch.zeros(2, 4, 4, dtype=torch.bool)
    mask_g[0, 0, 0] = True
    mask_g[1, :, :] = True
    loss_g, valid_g, _ = masked_charbonnier_per_sample(pred_g, target_g, mask_g, torch.ones(2, dtype=torch.bool))
    _record(rows, "G 不同mask面积样本权重一致", valid_g.tolist() == [True, True] and torch.allclose(loss_g[0], loss_g[1], atol=1e-7), f"loss={loss_g.tolist()}")

    # H：混合batch只保留有效监督样本。
    pred_h = torch.zeros(4, 1, 3, 3)
    target_h = torch.zeros_like(pred_h)
    target_h[3] = -0.5
    pred_h[3] = -0.4
    mask_h = torch.ones(4, 3, 3, dtype=torch.bool)
    mask_h[2] = False
    supervise_h = torch.tensor([True, False, True, True])
    loss_h, valid_h, _ = masked_charbonnier_per_sample(pred_h, target_h, mask_h, supervise_h)
    _record(rows, "H 混合batch筛选", valid_h.tolist() == [True, False, False, True] and torch.isfinite(loss_h).all().item(), f"valid={valid_h.tolist()}, loss={loss_h.tolist()}")

    # 与EV-TTC原实现数值对齐：原函数返回标量，新函数逐样本后再mean应一致。
    evttc_loss = _load_evttc_original_loss()
    torch.manual_seed(3)
    pred_ref = torch.randn(3, 1, 5, 5)
    target_ref = torch.randn(3, 1, 5, 5)
    mask_ref = (torch.rand(3, 1, 5, 5) > 0.3)
    supervise_ref = torch.ones(3, dtype=torch.bool)
    per_ref, valid_ref, _ = masked_charbonnier_per_sample(pred_ref, target_ref, mask_ref[:, 0], supervise_ref)
    scalar_new = _mean_valid(per_ref, valid_ref)
    scalar_old = evttc_loss(target_ref - pred_ref, alpha=EVTTC_CHARBONNIER_ALPHA, mask=mask_ref)
    expanded_ref, expanded_valid = evttc_reference_per_sample(target_ref - pred_ref, mask_ref, alpha=EVTTC_CHARBONNIER_ALPHA, epsilon=EVTTC_CHARBONNIER_EPS)
    scalar_abs = float((scalar_new - scalar_old).abs())
    sample_abs = float((per_ref[valid_ref] - expanded_ref[expanded_valid]).abs().max())
    _record(
        rows,
        "EV-TTC原实现数值对齐",
        scalar_abs < 1e-7 and sample_abs < 1e-7,
        f"scalar_abs={scalar_abs:.3e}, per_sample_abs={sample_abs:.3e}",
    )

    passed = all(bool(row["passed"]) for row in rows)
    write_csv(out / "阶段2_Loss单元测试结果.csv", rows)
    write_json(out / "阶段2_Loss单元测试结果.json", {"passed": passed, "stats_example": stats.to_dict(), "rows": rows})

    table = [[r["test"], pass_text(bool(r["passed"])), r["details"]] for r in rows]
    report = f"""# 阶段2 EV-TTC Masked Charbonnier实现报告

## 结论
- 总体结果：{pass_text(passed)}
- Charbonnier来源：`EV-TTC-main/model/util.py`
- `alpha={EVTTC_CHARBONNIER_ALPHA}`
- `epsilon={EVTTC_CHARBONNIER_EPS}`

## 单元测试
{markdown_table(["测试项", "结果", "说明"], table)}

## 关键确认
- signed inverse TTC 的负值没有被裁剪。
- 最终loss mask为 `valid_ttc_mask & supervise_valid & finite(pred,target)`。
- 每个样本先按自身有效像素平均，再由训练循环按有效 step-sample 平均。
- 空mask和 `supervise_valid=0` 样本不会进入 `valid_samples`，不会影响最终loss。
- EV-TTC原函数只返回batch标量；本测试同时比较原函数标量，以及按其源码展开的逐样本值，误差仅为浮点误差。
"""
    (out / "阶段2_EVTTC_Masked_Charbonnier实现报告.md").write_text(report, encoding="utf-8")
    return {"passed": passed, "rows": rows, "out_dir": str(out)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()
    result = run_tests(args.out_dir)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
