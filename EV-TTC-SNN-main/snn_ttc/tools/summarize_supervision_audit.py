#!/usr/bin/env python3
"""兼容入口：读取多 N H5 并生成监督质量综合报告。"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from snn_ttc.tools.audit_supervise_and_empty_labels import main


if __name__ == "__main__":
    raise SystemExit(main())
