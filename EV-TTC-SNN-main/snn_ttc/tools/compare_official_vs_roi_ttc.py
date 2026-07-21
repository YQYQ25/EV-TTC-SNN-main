#!/usr/bin/env python3
"""生成官方 EV-TTC 与当前 ROI 多事件数方案的对比报告。"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT_DIR = PROJECT_ROOT / "EV-TTC-SNN-main/analysis/官方EVTTC_Skatepark筛选与负TTC统计"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    # 对比表依赖官方全序列统计；如果没有汇总文件，就触发一次完整只读审计。
    summary = args.output_dir / "audit_summary.json"
    if not summary.exists():
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "EV-TTC-SNN-main/snn_ttc/tools/audit_official_evttc_skatepark.py"),
            "--full-sequence",
            "--output-dir",
            str(args.output_dir),
        ]
        return subprocess.call(cmd)
    print(args.output_dir / "reports/官方EVTTC与当前ROI方案对比.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
