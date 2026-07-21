#!/usr/bin/env python3
"""顺序执行 LIF-EV-FlowNet-TTC 阶段1至阶段3核验。

脚本只跑单元测试和最小集成测试，不启动32-block过拟合或正式训练。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SNN_ROOT = PROJECT_ROOT / "EV-TTC-SNN-main"
if str(SNN_ROOT) not in sys.path:
    sys.path.insert(0, str(SNN_ROOT))

from snn_ttc.tests.report_utils import ensure_out_dir, markdown_table, pass_text, write_json  # noqa: E402
from snn_ttc.tests.test_lif_evflownet_ttc import run_tests as run_stage1  # noqa: E402
from snn_ttc.tests.test_masked_charbonnier import run_tests as run_stage2  # noqa: E402
from snn_ttc.tests.test_model_loss_integration import DEFAULT_S2_H5, run_tests as run_stage3  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--s2-h5", type=Path, default=DEFAULT_S2_H5)
    args = parser.parse_args()

    out = ensure_out_dir(args.out_dir)
    results = []

    stage1 = run_stage1(out)
    results.append({"stage": "阶段1 模型迁移", "passed": stage1["passed"], "out_dir": stage1["out_dir"]})
    if not stage1["passed"]:
        write_json(out / "阶段1至阶段3总结果.json", results)
        return 1

    stage2 = run_stage2(out)
    results.append({"stage": "阶段2 Loss", "passed": stage2["passed"], "out_dir": stage2["out_dir"]})
    if not stage2["passed"]:
        write_json(out / "阶段1至阶段3总结果.json", results)
        return 1

    stage3 = run_stage3(out, args.s2_h5)
    results.append({"stage": "阶段3 集成测试", "passed": stage3["passed"], "out_dir": stage3["out_dir"]})
    write_json(out / "阶段1至阶段3总结果.json", results)

    rows = [[r["stage"], pass_text(bool(r["passed"])), r["out_dir"]] for r in results]
    report = f"""# LIF-EV-FlowNet-TTC 阶段1至阶段3总览

{markdown_table(["阶段", "结果", "输出目录"], rows)}

说明：本脚本只执行模型、loss和单block最小集成核验，没有启动32-block过拟合、500-block训练或完整训练。
"""
    (out / "阶段1至阶段3总览.md").write_text(report, encoding="utf-8")
    return 0 if stage3["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
