#!/usr/bin/env python3
"""Resumable sequential M3ED downloader using aria2c public HTTPS URLs only.

It consumes the selected-object manifest under /data/M3ED/_download_meta.  The
manifest is bootstrapped from the preceding s5cmd preflight JSON when needed.
No preprocessing, training, or evaluation is invoked by this utility.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from urllib.request import Request, urlopen

import h5py


S3_PREFIX = "s3://m3ed-dist/processed/"
HTTPS_PREFIX = "https://m3ed-dist.s3.us-west-2.amazonaws.com/processed/"
FILE_TYPES = ("data", "depth_gt", "pose_gt")

OFFICIAL_REMAINING_TRAIN = (
    "spot_indoor_stairs",
    "spot_forest_road_3",
    "spot_outdoor_day_skatepark_1",
    "spot_outdoor_day_srt_under_bridge_2",
    "spot_indoor_stairwell",
    "spot_outdoor_day_art_plaza_loop",
    "spot_outdoor_day_penno_short_loop",
    "car_forest_into_ponds_long",
    "car_urban_day_rittenhouse",
    "car_forest_into_ponds_short",
    "car_urban_night_rittenhouse",
    "car_urban_night_penno_big_loop",
    "car_urban_night_city_hall",
    "car_urban_day_penno_big_loop",
    "car_urban_day_city_hall",
    "car_forest_tree_tunnel",
    "car_urban_day_ucity_small_loop",
)

OFFICIAL_REMAINING_TEST = (
    "car_urban_night_ucity_small_loop",
    "car_forest_sand_1",
    "spot_forest_easy_1",
    "spot_outdoor_day_srt_under_bridge_1",
    "spot_outdoor_day_rocky_steps",
    "spot_forest_road_1",
    "spot_indoor_obstacles",
    "spot_indoor_building_loop",
)


@dataclass
class ObjectRecord:
    source: str
    expected_size: int
    local_path: Path
    state: str = "pending"
    local_size: int = 0
    hdf5_check: str = "not checked"
    detail: str = ""

    @property
    def url(self) -> str:
        if not self.source.startswith(S3_PREFIX):
            raise ValueError(f"unsupported source URI: {self.source}")
        return HTTPS_PREFIX + self.source.removeprefix(S3_PREFIX)

    @property
    def part_path(self) -> Path:
        return self.local_path.with_name(self.local_path.name + ".part")


def format_size(value: float) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    size = float(value)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size:.2f} TiB"


def write_log(path: Path, text: str) -> None:
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {text}"
    print(line, flush=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def hdf5_status(path: Path) -> str:
    try:
        with h5py.File(path, "r") as handle:
            list(handle.keys())
        return "ok"
    except Exception as exc:
        return f"failed: {type(exc).__name__}: {exc}".replace("\n", " ")


def bootstrap_manifest(manifest: Path, preflight: Path) -> None:
    if manifest.exists():
        return
    if not preflight.is_file():
        raise FileNotFoundError(f"missing both {manifest} and {preflight}")
    payload = json.loads(preflight.read_text(encoding="utf-8"))
    rows = [
        "# Generated from preflight.json; source_uri<TAB>expected_bytes<TAB>local_path",
    ]
    for entry in payload["records"]:
        rows.append(f"{entry['source']}\t{entry['expected_size']}\t{entry['local_path']}")
    manifest.write_text("\n".join(rows) + "\n", encoding="utf-8")


def read_manifest(path: Path) -> list[ObjectRecord]:
    records = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split("\t")
        if len(fields) != 3:
            raise ValueError(f"invalid manifest line {line_number}: {raw}")
        source, expected_size, local_path = fields
        records.append(ObjectRecord(source, int(expected_size), Path(local_path)))
    return records


def https_content_length(url: str) -> int:
    request = Request(url, method="HEAD", headers={"User-Agent": "ev-ttc-m3ed-downloader/1.0"})
    last_error: Exception | None = None
    for attempt in range(1, 4):
        try:
            with urlopen(request, timeout=60) as response:
                value = response.headers.get("Content-Length")
            if value is None:
                raise RuntimeError(f"missing Content-Length for {url}")
            return int(value)
        except Exception as exc:
            last_error = exc
            if attempt < 3:
                time.sleep(2 * attempt)
    raise RuntimeError(f"failed to query Content-Length for {url}: {last_error}")


def build_official_remaining_records(root: Path) -> list[ObjectRecord]:
    records: list[ObjectRecord] = []
    for split, sequences in (
        ("official_train", OFFICIAL_REMAINING_TRAIN),
        ("official_test", OFFICIAL_REMAINING_TEST),
    ):
        for sequence in sequences:
            for file_type in FILE_TYPES:
                name = f"{sequence}_{file_type}.h5"
                source = f"{S3_PREFIX}{sequence}/{name}"
                url = f"{HTTPS_PREFIX}{sequence}/{name}"
                records.append(
                    ObjectRecord(
                        source=source,
                        expected_size=https_content_length(url),
                        local_path=root / split / sequence / name,
                    )
                )
    return records


def pending_bytes(records: list[ObjectRecord]) -> int:
    total = 0
    for record in records:
        local_size = record.local_path.stat().st_size if record.local_path.is_file() else 0
        if local_size != record.expected_size:
            total += record.expected_size
    return total


def check_final(record: ObjectRecord) -> bool:
    path = record.local_path
    record.local_size = path.stat().st_size if path.is_file() else 0
    if record.local_size != record.expected_size:
        record.hdf5_check = "not checked"
        record.detail = "missing file" if not path.exists() else "local size differs from manifest"
        return False
    record.hdf5_check = hdf5_status(path)
    if record.hdf5_check == "ok":
        return True
    record.detail = "HDF5 read-only open failed"
    return False


def run_aria2(record: ObjectRecord, aria2: str, connections: int, log_path: Path, stall_seconds: int) -> tuple[bool, str]:
    record.local_path.parent.mkdir(parents=True, exist_ok=True)
    if record.part_path.exists() and record.part_path.stat().st_size > record.expected_size:
        bad_path = record.part_path.with_name(record.part_path.name + f".oversize_{int(time.time())}")
        os.replace(record.part_path, bad_path)
        write_log(log_path, f"moved oversized partial to {bad_path}")
    command = [
        aria2,
        "-c",
        "-x", str(connections),
        "-s", str(connections),
        "-k", "4M",
        "--file-allocation=none",
        "--timeout=60",
        "--connect-timeout=30",
        "--max-tries=0",
        "--retry-wait=10",
        "--console-log-level=notice",
        "--summary-interval=5",
        f"--log={log_path}",
        "--log-level=notice",
        "-d", str(record.local_path.parent),
        "-o", record.part_path.name,
        record.url,
    ]
    write_log(log_path, f"starting x={connections}: {' '.join(command)}")
    process = subprocess.Popen(command)
    previous_size = record.part_path.stat().st_size if record.part_path.exists() else 0
    last_growth = time.monotonic()
    while process.poll() is None:
        time.sleep(5)
        current_size = record.part_path.stat().st_size if record.part_path.exists() else 0
        if current_size > previous_size:
            previous_size = current_size
            last_growth = time.monotonic()
        elif time.monotonic() - last_growth >= stall_seconds:
            process.terminate()
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            return False, f"no .part growth for {stall_seconds}s"
    if process.returncode != 0:
        return False, f"aria2c exited with status {process.returncode}"
    actual_size = record.part_path.stat().st_size if record.part_path.is_file() else 0
    if actual_size != record.expected_size:
        return False, f".part size {actual_size} differs from expected {record.expected_size}"
    os.replace(record.part_path, record.local_path)
    return True, "downloaded and atomically renamed"


def report(path: Path, records: list[ObjectRecord], root: Path) -> None:
    succeeded = [record for record in records if record.state == "ok"]
    failed = [record for record in records if record.state == "failed"]
    total = sum(record.local_size for record in succeeded)
    lines = [
        "# M3ED aria2 下载报告",
        "",
        f"- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 数据根目录：`{root}`",
        f"- 传输方式：aria2c 单文件顺序下载，公开 HTTPS，无签名认证",
        f"- 成功文件：{len(succeeded)}/{len(records)}",
        f"- 已验证本地总量：{format_size(total)}",
        "",
        "| 序列 | 文件 | 期望大小 | 本地大小 | 状态 | HDF5 检查 | 本地路径 |",
        "|---|---|---:|---:|---|---|---|",
    ]
    for record in records:
        sequence = record.local_path.parent.name
        lines.append(
            f"| {sequence} | {record.local_path.name} | {format_size(record.expected_size)} | "
            f"{format_size(record.local_size)} | {record.state} | {record.hdf5_check} | `{record.local_path}` |"
        )
    lines.extend(["", "## 失败 URL", ""])
    if failed:
        lines.extend(["| 文件 | 原因 | HTTPS URL |", "|---|---|---|"])
        for record in failed:
            lines.append(f"| {record.local_path.name} | {record.detail} | `{record.url}` |")
    else:
        lines.append("无。")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/data/M3ED"))
    parser.add_argument(
        "--preset",
        choices=("selected_manifest", "official_remaining"),
        default="selected_manifest",
        help="selected_manifest reuses selected_objects.txt; official_remaining downloads the remaining EV-TTC official split sequences.",
    )
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--log", type=Path, default=None)
    parser.add_argument("--aria2", default="aria2c")
    parser.add_argument("--stall-seconds", type=int, default=180)
    parser.add_argument("--skip-space-check", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="resolve objects and check disk space, but do not download")
    args = parser.parse_args()

    root = args.root.resolve()
    meta = root / "_download_meta"
    meta.mkdir(parents=True, exist_ok=True)
    if shutil.which(args.aria2) is None and not Path(args.aria2).is_file():
        raise FileNotFoundError(f"aria2c not found: {args.aria2}")

    if args.preset == "selected_manifest":
        manifest = args.manifest or (meta / "selected_objects.txt")
        bootstrap_manifest(manifest, meta / "preflight.json")
        records = read_manifest(manifest)
        default_log = meta / "aria2_download.log"
        default_report = meta / "download_report.md"
        source_label = f"manifest {manifest}"
    else:
        records = build_official_remaining_records(root)
        default_log = meta / "aria2_official_remaining.log"
        default_report = meta / "official_remaining_download_report.md"
        source_label = "official_remaining preset"

    log_path = args.log or default_log
    report_path = args.report or default_report
    write_log(log_path, f"loaded {len(records)} entries from {source_label}")

    required_bytes = pending_bytes(records)
    free_bytes = shutil.disk_usage(root).free
    write_log(
        log_path,
        f"pending download size={format_size(required_bytes)}, free={format_size(free_bytes)}, required 1.2x={format_size(required_bytes * 1.2)}",
    )
    if not args.skip_space_check and free_bytes < required_bytes * 1.2:
        write_log(log_path, "STOP: free space is below the required 1.2x threshold")
        return 2
    if args.dry_run:
        complete = sum(
            record.local_path.is_file() and record.local_path.stat().st_size == record.expected_size
            for record in records
        )
        write_log(log_path, f"dry run complete: size-matched local files={complete}/{len(records)}")
        return 0

    for index, record in enumerate(records, start=1):
        if check_final(record):
            record.state = "ok"
            record.detail = "existing size verified; skipped"
            write_log(log_path, f"[{index}/{len(records)}] skip verified {record.local_path}")
            continue
        write_log(log_path, f"[{index}/{len(records)}] pending {record.url}")
        success = False
        reasons = []
        for connections in (8, 4, 1):
            completed, detail = run_aria2(record, args.aria2, connections, log_path, args.stall_seconds)
            reasons.append(f"x={connections}: {detail}")
            if completed:
                success = check_final(record)
                if success:
                    record.state = "ok"
                    record.detail = detail
                    write_log(log_path, f"completed {record.local_path}")
                    break
                reasons.append("post-download verification failed: " + record.detail)
        if not success:
            record.state = "failed"
            record.local_size = record.local_path.stat().st_size if record.local_path.exists() else 0
            record.detail = "; ".join(reasons)
            write_log(log_path, f"failed {record.url}: {record.detail}")

    # Re-open every completed file, including pre-existing pose ground truth files.
    for record in records:
        if record.state == "ok" and not check_final(record):
            record.state = "failed"
    report(report_path, records, root)
    failed = sum(record.state == "failed" for record in records)
    write_log(log_path, f"session complete: failed={failed}, report={report_path}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
