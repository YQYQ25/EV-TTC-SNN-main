#!/usr/bin/env python3
"""汇总 Skatepark 多事件数 TTC 验证结果。"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("EV-TTC-SNN-main/debug_sets/skatepark_multi_n_ttc"))
    args = parser.parse_args()
    summary = args.out_dir / "reports" / "Skatepark多事件数TTC标签综合报告.md"
    if not summary.exists():
        subprocess.check_call([sys.executable, "EV-TTC-SNN-main/snn_ttc/tools/audit_skatepark_multi_n_ttc.py", "--out-dir", str(args.out_dir)])
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

