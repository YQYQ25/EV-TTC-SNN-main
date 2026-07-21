#!/usr/bin/env python3
"""输出事件窗口标签与固定 10ms 标签对比报告路径。"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("EV-TTC-SNN-main/debug_sets/skatepark_multi_n_ttc"))
    args = parser.parse_args()
    report = args.out_dir / "reports" / "事件窗口与固定10ms标签对比.md"
    if not report.exists():
        subprocess.check_call([sys.executable, "EV-TTC-SNN-main/snn_ttc/tools/audit_skatepark_multi_n_ttc.py", "--out-dir", str(args.out_dir)])
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

