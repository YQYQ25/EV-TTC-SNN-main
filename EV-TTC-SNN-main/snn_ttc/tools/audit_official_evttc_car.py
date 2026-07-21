#!/usr/bin/env python3
"""审计 car_urban_night_rittenhouse 的官方 EV-TTC 样本筛选与负 TTC。"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def main() -> int:
    out_dir = ROOT / "EV-TTC-SNN-main/debug_sets/car_urban_night_rittenhouse_multi_n_ttc"
    h5_dir = out_dir / "H5"
    cmd = [
        sys.executable,
        str(ROOT / "EV-TTC-SNN-main/snn_ttc/tools/audit_official_evttc_skatepark.py"),
        "--seq-name",
        "car_urban_night_rittenhouse",
        "--speed-threshold",
        "1.3",
        "--roi-dir",
        str(h5_dir),
        "--full-sequence",
        "--output-dir",
        str(out_dir / "official_evttc_audit"),
    ]
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
