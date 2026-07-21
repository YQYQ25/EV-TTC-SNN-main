#!/usr/bin/env python3
"""运行官方 Skatepark 审计的阶段 A 子集检查。"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def main() -> int:
    # 该包装脚本只负责给出干净入口，核心统计逻辑仍在 audit_official_evttc_skatepark.py。
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "EV-TTC-SNN-main/snn_ttc/tools/audit_official_evttc_skatepark.py"),
        "--max-samples",
        "5000",
        "--output-dir",
        str(PROJECT_ROOT / "EV-TTC-SNN-main/analysis/官方EVTTC_Skatepark筛选与负TTC统计/stage_A_subset5000"),
    ]
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
