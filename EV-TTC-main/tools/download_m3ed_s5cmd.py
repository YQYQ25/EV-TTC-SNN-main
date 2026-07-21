#!/usr/bin/env python3
"""Download selected public M3ED HDF5 files with s5cmd and verify them.

This tool intentionally downloads only the event data plus depth/pose ground truth.
It never invokes the EV-TTC preprocessing or training pipeline.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import h5py


BUCKET_PREFIX = "s3://m3ed-dist/processed"
PAPER_ALIGNMENT = (
    "car_urban_day_penno_small_loop",
    "spot_outdoor_day_skatepark_2",
    "spot_outdoor_day_srt_green_loop",
    "spot_outdoor_night_penno_plaza_lights",
    "spot_outdoor_night_penno_short_loop",
)
TASK_ORIENTED = (
    "falcon_indoor_flight_1",
    "falcon_indoor_flight_2",
    "falcon_indoor_flight_3",
    "falcon_outdoor_night_high_beams",
    "falcon_outdoor_day_fast_flight_1",
)
FILE_TYPES = ("data", "depth_gt", "pose_gt")
S5CMD_LIST_LINE = re.compile(r"^\S+\s+\S+\s+(\d+)\s+(\S+)$")


@dataclass
class ObjectRecord:
    group: str
    sequence: str
    file_type: str
    source: str
    expected_size: int
    local_path: str
    action: str = "download"
    local_size: int = 0
    hdf5_check: str = "not checked"
    reason: str = ""


def format_size(size: float) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    value = float(size)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024
    return f"{value:.2f} TiB"


def log_line(handle, message: str) -> None:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{stamp}] {message}"
    print(line, flush=True)
    handle.write(line + "\n")
    handle.flush()


def s5cmd_list(s5cmd: Path, sequence: str) -> dict[str, int]:
    env = os.environ.copy()
    env["AWS_REGION"] = "us-west-2"
    env["AWS_DEFAULT_REGION"] = "us-west-2"
    result = subprocess.run(
        [str(s5cmd), "--no-sign-request", "ls", f"{BUCKET_PREFIX}/{sequence}/"],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "s5cmd ls failed")

    objects: dict[str, int] = {}
    for line in result.stdout.splitlines():
        match = S5CMD_LIST_LINE.match(line.strip())
        if match:
            objects[match.group(2)] = int(match.group(1))
    return objects


def hdf5_status(path: Path) -> str:
    try:
        with h5py.File(path, "r") as handle:
            keys = list(handle.keys())
        return "ok" if keys else "opened (empty root)"
    except Exception as exc:  # The text is captured in the final report.
        return f"failed: {type(exc).__name__}: {exc}".replace("\n", " ")


def build_records(root: Path, s5cmd: Path) -> list[ObjectRecord]:
    records: list[ObjectRecord] = []
    for group, sequences in (("paper_alignment", PAPER_ALIGNMENT), ("task_oriented", TASK_ORIENTED)):
        for sequence in sequences:
            objects = s5cmd_list(s5cmd, sequence)
            required_names = {file_type: f"{sequence}_{file_type}.h5" for file_type in FILE_TYPES}
            missing = [name for name in required_names.values() if name not in objects]
            if missing:
                raise RuntimeError(f"{sequence} is missing required S3 objects: {', '.join(missing)}")
            for file_type, name in required_names.items():
                local = root / group / sequence / name
                record = ObjectRecord(
                    group=group,
                    sequence=sequence,
                    file_type=file_type,
                    source=f"{BUCKET_PREFIX}/{sequence}/{name}",
                    expected_size=objects[name],
                    local_path=str(local),
                )
                if local.exists():
                    record.local_size = local.stat().st_size
                    if record.local_size == record.expected_size:
                        record.hdf5_check = hdf5_status(local)
                        if record.hdf5_check.startswith("ok") or record.hdf5_check.startswith("opened"):
                            record.action = "skip_verified"
                        else:
                            record.reason = "same size but HDF5 verification failed"
                    else:
                        record.reason = "local size differs from S3 object"
                records.append(record)
    return records


def pending_records(records: Iterable[ObjectRecord]) -> list[ObjectRecord]:
    return [record for record in records if record.action != "skip_verified"]


def write_commands(path: Path, records: Iterable[ObjectRecord]) -> None:
    lines = []
    for record in records:
        destination = Path(record.local_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        lines.append(f'cp "{record.source}" "{destination}"')
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def write_preflight(path: Path, records: list[ObjectRecord], free_bytes: int, required_bytes: int) -> None:
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "bucket_prefix": BUCKET_PREFIX,
        "free_bytes": free_bytes,
        "required_download_bytes": required_bytes,
        "required_free_bytes": int(required_bytes * 1.2),
        "records": [asdict(record) for record in records],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def stream_s5cmd(
    s5cmd: Path,
    command_file: Path,
    records: list[ObjectRecord],
    workers: int,
    log_handle,
) -> int:
    env = os.environ.copy()
    env["AWS_REGION"] = "us-west-2"
    env["AWS_DEFAULT_REGION"] = "us-west-2"
    command = [
        str(s5cmd),
        "--no-sign-request",
        "--numworkers",
        str(workers),
        "--retry-count",
        "0",
        "--stat",
        "run",
        str(command_file),
    ]
    log_line(log_handle, "running: " + " ".join(command))
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )
    output_queue: queue.Queue[str | None] = queue.Queue()

    def forward_output() -> None:
        assert process.stdout is not None
        for line in process.stdout:
            output_queue.put(line.rstrip())
        output_queue.put(None)

    reader = threading.Thread(target=forward_output, daemon=True)
    reader.start()

    expected_total = sum(record.expected_size for record in records)
    previous_bytes = sum(min(Path(record.local_path).stat().st_size, record.expected_size)
                         for record in records if Path(record.local_path).exists())
    previous_time = time.monotonic()
    next_status = previous_time
    stream_closed = False
    while process.poll() is None or not stream_closed:
        try:
            while True:
                line = output_queue.get_nowait()
                if line is None:
                    stream_closed = True
                    break
                log_line(log_handle, "s5cmd: " + line)
        except queue.Empty:
            pass
        now = time.monotonic()
        if now >= next_status:
            current_bytes = sum(min(Path(record.local_path).stat().st_size, record.expected_size)
                                for record in records if Path(record.local_path).exists())
            elapsed = max(now - previous_time, 1e-6)
            speed = max(current_bytes - previous_bytes, 0) / elapsed
            percent = 100 * current_bytes / expected_total if expected_total else 100.0
            eta = (expected_total - current_bytes) / speed if speed > 0 else None
            eta_text = f"{eta / 60:.1f} min" if eta is not None else "calculating"
            log_line(
                log_handle,
                f"progress: {percent:5.1f}% | {format_size(current_bytes)} / {format_size(expected_total)} "
                f"| {format_size(speed)}/s | ETA {eta_text}",
            )
            previous_bytes, previous_time = current_bytes, now
            next_status = now + 10
        time.sleep(0.2)
    return process.wait()


def verify_records(records: list[ObjectRecord]) -> list[ObjectRecord]:
    failures: list[ObjectRecord] = []
    for record in records:
        local = Path(record.local_path)
        record.local_size = local.stat().st_size if local.exists() else 0
        if record.local_size != record.expected_size:
            record.action = "failed"
            record.hdf5_check = "not checked"
            record.reason = "missing file" if not local.exists() else "local size differs from S3 object"
            failures.append(record)
            continue
        record.hdf5_check = hdf5_status(local)
        if record.hdf5_check.startswith("ok") or record.hdf5_check.startswith("opened"):
            record.action = "downloaded"
            record.reason = ""
        else:
            record.action = "failed"
            record.reason = "HDF5 verification failed"
            failures.append(record)
    return failures


def write_report(path: Path, root: Path, records: list[ObjectRecord], free_bytes: int, required_bytes: int) -> None:
    complete_sequences = []
    for group, sequences in (("paper_alignment", PAPER_ALIGNMENT), ("task_oriented", TASK_ORIENTED)):
        for sequence in sequences:
            subset = [record for record in records if record.sequence == sequence]
            if subset and all(record.action in {"downloaded", "skip_verified"} for record in subset):
                complete_sequences.append(sequence)
    failures = [record for record in records if record.action == "failed"]
    total_local = sum(record.local_size for record in records if record.action in {"downloaded", "skip_verified"})
    lines = [
        "# M3ED s5cmd 下载报告",
        "",
        f"- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 数据源：`{BUCKET_PREFIX}/`（匿名公开访问，`--no-sign-request`）",
        f"- 本地根目录：`{root}`",
        f"- 下载前可用空间：{format_size(free_bytes)}",
        f"- 本次待下载量：{format_size(required_bytes)}",
        f"- 校验后本地文件总量：{format_size(total_local)}",
        "",
        "## 文件明细",
        "",
        "| 分组 | 序列 | 文件 | S3 大小 | 本地大小 | 状态 | HDF5 检查 | 本地路径 |",
        "|---|---|---|---:|---:|---|---|---|",
    ]
    for record in records:
        lines.append(
            f"| {record.group} | {record.sequence} | {record.file_type} | {format_size(record.expected_size)} | "
            f"{format_size(record.local_size)} | {record.action} | {record.hdf5_check} | `{record.local_path}` |"
        )
    lines.extend(["", "## 汇总", ""])
    lines.append("成功序列：" + ("、".join(complete_sequences) if complete_sequences else "无"))
    missing_gt = []
    for sequence in PAPER_ALIGNMENT + TASK_ORIENTED:
        subset = {record.file_type: record for record in records if record.sequence == sequence}
        if any(subset[file_type].action == "failed" for file_type in ("depth_gt", "pose_gt")):
            missing_gt.append(sequence)
    lines.append("GT 缺失或未通过校验的序列：" + ("、".join(missing_gt) if missing_gt else "无"))
    lines.append("失败文件：" + (str(len(failures)) if failures else "无"))
    if failures:
        lines.extend(["", "| 序列 | 文件 | 原因 |", "|---|---|---|"])
        for record in failures:
            lines.append(f"| {record.sequence} | {record.file_type} | {record.reason} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/data/M3ED"))
    parser.add_argument("--s5cmd", type=Path, default=Path.home() / ".local/bin/s5cmd")
    parser.add_argument("--download", action="store_true", help="run downloads after preflight")
    args = parser.parse_args()

    root = args.root.resolve()
    meta = root / "_download_meta"
    root.mkdir(parents=True, exist_ok=True)
    meta.mkdir(parents=True, exist_ok=True)
    if not args.s5cmd.is_file():
        raise FileNotFoundError(f"s5cmd not found: {args.s5cmd}")

    records = build_records(root, args.s5cmd)
    pending = pending_records(records)
    free_bytes = shutil.disk_usage(root).free
    required_bytes = sum(record.expected_size for record in pending)
    write_commands(meta / "download_commands.txt", pending)
    write_preflight(meta / "preflight.json", records, free_bytes, required_bytes)
    print(f"Resolved {len(records)} required objects from {BUCKET_PREFIX}/")
    print(f"Verified local files to skip: {len(records) - len(pending)}")
    print(f"Required download size: {format_size(required_bytes)}")
    print(f"Free space: {format_size(free_bytes)}")
    print(f"Required free space (1.2x): {format_size(required_bytes * 1.2)}")
    if free_bytes < required_bytes * 1.2:
        print("STOP: free space is below the required 1.2x threshold.", file=sys.stderr)
        return 2
    if not args.download:
        print("Preflight passed. Run again with --download to start s5cmd.")
        return 0

    log_path = meta / "download.log"
    with log_path.open("a", encoding="utf-8") as log_handle:
        log_line(log_handle, "download session started")
        failures = pending
        for attempt in range(4):
            if not failures:
                break
            workers = 16 if attempt == 0 else 8
            command_file = meta / ("download_commands.txt" if attempt == 0 else f"retry_{attempt}_commands.txt")
            write_commands(command_file, failures)
            log_line(log_handle, f"attempt {attempt + 1}/4, objects={len(failures)}, workers={workers}")
            stream_s5cmd(args.s5cmd, command_file, failures, workers, log_handle)
            failures = verify_records(failures)
            if failures:
                log_line(log_handle, f"attempt {attempt + 1} left {len(failures)} failed objects")
        verify_records(records)
        log_line(log_handle, "download session finished")
    write_report(meta / "download_report.md", root, records, free_bytes, required_bytes)
    print(f"Report: {meta / 'download_report.md'}")
    return 1 if any(record.action == "failed" for record in records) else 0


if __name__ == "__main__":
    raise SystemExit(main())
