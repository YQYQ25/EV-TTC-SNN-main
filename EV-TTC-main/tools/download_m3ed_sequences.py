#!/usr/bin/env python3
import argparse
import json
import os
import pty
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401


SEQUENCES = [
    "car_urban_day_penno_small_loop",
    "falcon_outdoor_night_high_beams",
    "spot_outdoor_night_penno_short_loop",
    "spot_outdoor_night_penno_plaza_lights",
    "spot_outdoor_day_skatepark_2",
    "spot_outdoor_day_srt_green_loop",
]

FILE_SPECS = [
    ("data", "events/m3ed/{seq}/{seq}_data.h5", "event data"),
    ("depth_gt", "depth/m3ed/{seq}/{seq}_depth_gt.h5", "depth GT"),
    ("pose_gt", "depth/m3ed/{seq}/{seq}_pose_gt.h5", "pose GT"),
]

S3_BASES = [
    "https://m3ed-dist.s3.us-west-2.amazonaws.com/processed",
    "https://m3ed-dist.s3.amazonaws.com/processed",
]


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def fmt_size(num):
    if num is None:
        return "unknown"
    value = float(num)
    for unit in ["B", "KiB", "MiB", "GiB", "TiB"]:
        if value < 1024.0 or unit == "TiB":
            return f"{value:.2f} {unit}"
        value /= 1024.0


def head_size(url, timeout=20, retries=5):
    last = None
    for attempt in range(1, retries + 1):
        req = urllib.request.Request(url, method="HEAD")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                length = resp.headers.get("Content-Length")
                return int(length) if length is not None else None
        except Exception as exc:
            last = exc
            time.sleep(min(2 * attempt, 10))
    raise last


def candidate_urls(seq, suffix):
    name = f"{seq}_{suffix}.h5"
    return [f"{base}/{seq}/{name}" for base in S3_BASES]


def resolve_remote(seq, suffix):
    errors = []
    for url in candidate_urls(seq, suffix):
        try:
            return url, head_size(url), None
        except Exception as exc:
            errors.append(f"{url}: {exc}")
    return candidate_urls(seq, suffix)[0], None, "; ".join(errors)


def disk_free(path):
    usage = shutil.disk_usage(path)
    return usage.free


def read_scalar(ds):
    if ds.size == 0:
        raise ValueError(f"empty dataset: {ds.name}")
    if len(ds.shape) == 1:
        _ = ds[0]
        _ = ds[-1]
    else:
        index = tuple(0 for _ in ds.shape)
        _ = ds[index]


def check_h5(path, suffix):
    try:
        with h5py.File(path, "r") as f:
            if suffix == "data":
                left = f["prophesee"]["left"]
                for key in ["t", "x", "y", "p", "ms_map_idx"]:
                    read_scalar(left[key])
                read_scalar(left["calib"]["intrinsics"])
            elif suffix == "depth_gt":
                for key in ["depth", "ts", "ts_map_prophesee_left"]:
                    read_scalar(f[key])
            elif suffix == "pose_gt":
                for key in ["Cn_T_C0", "Ln_T_L0", "ts", "ts_map_prophesee_left"]:
                    read_scalar(f[key])
            else:
                raise ValueError(f"unknown suffix: {suffix}")
        return True, "OK"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def run_with_pty(cmd, log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab") as log:
        log.write(f"\n\n===== {now()} CMD: {' '.join(cmd)} =====\n".encode())
        master_fd, slave_fd = pty.openpty()
        proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=slave_fd, stderr=slave_fd)
        os.close(slave_fd)
        try:
            while True:
                try:
                    data = os.read(master_fd, 4096)
                except OSError:
                    break
                if not data:
                    break
                os.write(sys.stdout.fileno(), data)
                log.write(data)
                log.flush()
        finally:
            os.close(master_fd)
        return proc.wait()


def aria2_download(urls, target, log_path, connections):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    input_file = log_path.with_suffix(log_path.suffix + ".aria2_input.txt")
    input_lines = ["\t".join(urls), f"  dir={target.parent}", f"  out={target.name}"]
    input_file.write_text("\n".join(input_lines) + "\n", encoding="utf-8")
    cmd = [
        "aria2c",
        "-c",
        "-x",
        str(connections),
        "-s",
        str(connections),
        "--min-split-size=16M",
        f"--max-connection-per-server={connections}",
        "--uri-selector=feedback",
        "--summary-interval=10",
        "--console-log-level=warn",
        "--download-result=default",
        "--file-allocation=none",
        "--max-tries=0",
        "--retry-wait=5",
        "-i",
        str(input_file),
    ]
    return run_with_pty(cmd, log_path)


def write_summary(rows, path):
    lines = [
        "# M3ED 多序列下载汇总",
        "",
        f"- 更新时间：`{now()}`",
        f"- 下载根目录：`/data/M3ED/`",
        "",
        "| Sequence | Expected size | Actual size | Status | Integrity check | Local path |",
        "|---|---:|---:|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['sequence']}` | {row['expected_size']} | {row['actual_size']} | "
            f"{row['status']} | {row['integrity']} | `{row['local_path']}` |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def sequence_expected(seq):
    specs = []
    total = 0
    for suffix, rel, label in FILE_SPECS:
        url, size, head_error = resolve_remote(seq, suffix)
        specs.append({"suffix": suffix, "rel": rel, "label": label, "url": url, "size": size, "head_error": head_error})
        if size is not None:
            total += size
    return specs, total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/data/M3ED")
    parser.add_argument("--log-dir", default="logs/ttc_TTCEF/download_logs")
    parser.add_argument("--sequence", action="append", choices=SEQUENCES)
    parser.add_argument("--connections", type=int, default=8)
    args = parser.parse_args()

    root = Path(args.root)
    log_dir = Path(args.log_dir)
    root.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    sequences = args.sequence or SEQUENCES
    rows = []
    summary_path = log_dir / "download_summary.md"

    if shutil.which("aria2c") is None:
        raise SystemExit("aria2c not found; install aria2 or provide another downloader with resume support.")

    for seq in sequences:
        seq_log = log_dir / f"{seq}.log"
        print(f"\n========== {seq} ==========")
        specs, expected_total = sequence_expected(seq)

        free = disk_free(root)
        unknown_sizes = any(item["size"] is None for item in specs)
        expected_text = "unknown" if unknown_sizes else fmt_size(expected_total)
        print(f"[{seq}] 磁盘剩余：{fmt_size(free)}；预计下载：{expected_text}")
        if expected_total > 0 and free < expected_total:
            rows.append(
                {
                    "sequence": seq,
                    "expected_size": fmt_size(expected_total),
                    "actual_size": "0 B",
                    "status": "FAILED",
                    "integrity": f"free space {fmt_size(free)} < expected {fmt_size(expected_total)}",
                    "local_path": str(root),
                }
            )
            write_summary(rows, summary_path)
            continue

        actual_total = 0
        checks = []
        status = "OK"
        seq_paths = []
        downloaded_any = False
        for spec in specs:
            suffix = spec["suffix"]
            target = root / spec["rel"].format(seq=seq)
            seq_paths.append(str(target))
            expected = spec["size"]
            print(f"\n[{seq}] {spec['label']}: {target}")
            print(f"[{seq}] 预计大小：{fmt_size(expected)}")
            if spec["head_error"]:
                print(f"[{seq}] HEAD 大小获取失败，将继续尝试下载：{spec['head_error']}")
            if target.exists():
                print(f"[{seq}] 已有文件：{fmt_size(target.stat().st_size)}")

            aria2_marker = Path(str(target) + ".aria2")
            complete_size = target.exists() and (
                (expected is not None and target.stat().st_size == expected) or expected is None
            ) and not aria2_marker.exists()
            if complete_size:
                ok, msg = check_h5(target, suffix)
                if ok:
                    print(f"[{seq}] 已有完整文件，跳过下载：{target}")
                else:
                    bad = target.with_name(f"{target.name}.bad_{time.strftime('%Y%m%d_%H%M%S')}")
                    print(f"[{seq}] 文件大小匹配但 H5 检查失败，移到：{bad}")
                    target.rename(bad)
                    complete_size = False

            if not complete_size:
                urls = candidate_urls(seq, suffix)
                rc = aria2_download(urls, target, seq_log, args.connections)
                downloaded_any = True
                if rc != 0:
                    status = "FAILED"
                    checks.append(f"{target.name}: aria2 exit {rc}")
                    print(f"[{seq}] 下载失败：{target}，exit={rc}")
                    continue

            if not target.exists():
                status = "FAILED"
                checks.append(f"{target.name}: missing")
                continue

            actual = target.stat().st_size
            actual_total += actual
            size_ok = expected is None or actual == expected
            ok, msg = check_h5(target, suffix) if size_ok else (False, f"size mismatch {fmt_size(actual)} != {fmt_size(expected)}")
            checks.append(f"{target.name}: {'OK' if ok else msg}")
            if not ok:
                status = "FAILED"
            print(f"[{seq}] 实际大小：{fmt_size(actual)}；完整性：{'OK' if ok else msg}")

        if status == "OK":
            status = "OK" if downloaded_any else "SKIPPED"
        rows.append(
            {
                "sequence": seq,
                "expected_size": fmt_size(expected_total),
                "actual_size": fmt_size(actual_total),
                "status": status,
                "integrity": "<br>".join(checks),
                "local_path": str(root),
            }
        )
        write_summary(rows, summary_path)

    print(f"\n下载汇总：{summary_path}")
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
