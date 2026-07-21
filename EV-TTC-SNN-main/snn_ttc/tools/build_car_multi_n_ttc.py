#!/usr/bin/env python3
"""构建 car_urban_night_rittenhouse 的固定事件数 ROI-SNN TTC 数据。

该脚本只是一个干净入口，核心实现复用已参数化的多事件数构建器。
car 序列按 EV-TTC 官方阈值使用 `||T|| > 1.3 m/s`。
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def main() -> int:
    out_dir = ROOT / "EV-TTC-SNN-main/debug_sets/car_urban_night_rittenhouse_multi_n_ttc/H5"
    gt_path = Path("/data/evttc_storage/event+SNN+TTC/data_m3ed_official_7train_3test/ttcef/m3ed/car_urban_night_rittenhouse.h5")
    cmd = [
        sys.executable,
        str(ROOT / "EV-TTC-SNN-main/snn_ttc/tools/build_skatepark_multi_n_ttc.py"),
        "--sequence",
        "car_urban_night_rittenhouse",
        "--root",
        "/data/M3ED",
        "--gt-path",
        str(gt_path),
        "--out-dir",
        str(out_dir),
        "--out-prefix",
        "car_urban_night_rittenhouse",
        "--event-counts",
        "5000",
        "10000",
        "15000",
        "20000",
        "--roi-size",
        "128",
        "--steps-per-roi",
        "10",
        "--max-step-duration-ms",
        "10",
        "--seed",
        "42",
        "--resume",
        "--continue-on-error",
    ]
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
