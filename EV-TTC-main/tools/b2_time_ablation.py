#!/usr/bin/env python3
import argparse
import copy
import json
import sys
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401
import numba
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "TTCEF"))
sys.path.insert(0, str(ROOT / "tools"))

from create_exp import load_camera_params  # noqa: E402
from b1_matched_train_eval import evaluate, train  # noqa: E402


US_TO_S = 1e-6
TIME_CHANNELS = ["mean_time_all", "mean_time_pos", "mean_time_neg", "fresh_pos", "fresh_neg"]


@numba.njit(cache=True)
def process_time_chunk(
    x_ev,
    y_ev,
    p_ev,
    t_ev,
    dist_map,
    out,
    target_indices,
    last_active,
    frame_idx,
    target_ptr,
    window_start,
    frame_interval,
    tau,
    sum_all,
    w_all,
    sum_pos,
    w_pos,
    sum_neg,
    w_neg,
    last_pos,
    last_neg,
):
    for i in range(len(x_ev)):
        time_since_last = (t_ev[i] - last_active) * US_TO_S

        if time_since_last > frame_interval:
            readout_time = t_ev[i] * US_TO_S
            duration = readout_time - window_start
            if duration <= 1e-9:
                duration = frame_interval

            if target_ptr < len(target_indices) and frame_idx == target_indices[target_ptr]:
                for yy in range(360):
                    for xx in range(360):
                        if w_all[yy, xx] > 0:
                            out[target_ptr, 0, yy, xx] = ((sum_all[yy, xx] / w_all[yy, xx]) - window_start) / duration
                        if w_pos[yy, xx] > 0:
                            out[target_ptr, 1, yy, xx] = ((sum_pos[yy, xx] / w_pos[yy, xx]) - window_start) / duration
                        if w_neg[yy, xx] > 0:
                            out[target_ptr, 2, yy, xx] = ((sum_neg[yy, xx] / w_neg[yy, xx]) - window_start) / duration
                        if last_pos[yy, xx] >= 0:
                            out[target_ptr, 3, yy, xx] = np.exp(-(readout_time - last_pos[yy, xx]) / tau)
                        if last_neg[yy, xx] >= 0:
                            out[target_ptr, 4, yy, xx] = np.exp(-(readout_time - last_neg[yy, xx]) / tau)
                target_ptr += 1
                if target_ptr == len(target_indices):
                    return last_active, frame_idx + 1, target_ptr, readout_time

            sum_all.fill(0.0)
            w_all.fill(0.0)
            sum_pos.fill(0.0)
            w_pos.fill(0.0)
            sum_neg.fill(0.0)
            w_neg.fill(0.0)
            last_pos.fill(-1.0)
            last_neg.fill(-1.0)

            last_active = t_ev[i]
            window_start = readout_time
            frame_idx += 1

        update = dist_map[y_ev[i], x_ev[i] - 280]
        x_c = int(update[0])
        y_c = int(update[1])
        t_s = t_ev[i] * US_TO_S
        is_pos = p_ev[i] != 0
        for k in range(4):
            if k == 0:
                xx = x_c
                yy = y_c
                weight = update[2]
            elif k == 1:
                xx = x_c + 1
                yy = y_c
                weight = update[3]
            elif k == 2:
                xx = x_c + 1
                yy = y_c + 1
                weight = update[5]
            else:
                xx = x_c
                yy = y_c + 1
                weight = update[4]

            if weight <= 0:
                continue
            sum_all[yy, xx] += t_s * weight
            w_all[yy, xx] += weight
            if is_pos:
                sum_pos[yy, xx] += t_s * weight
                w_pos[yy, xx] += weight
                last_pos[yy, xx] = t_s
            else:
                sum_neg[yy, xx] += t_s * weight
                w_neg[yy, xx] += weight
                last_neg[yy, xx] = t_s

    return last_active, frame_idx, target_ptr, window_start


def build_time_maps(data_root, seq_name, source_indices, args):
    event_path = Path(data_root) / "events" / "m3ed" / seq_name / f"{seq_name}_data.h5"
    out = np.zeros((len(source_indices), 5, 360, 360), dtype=np.float32)
    with h5py.File(event_path, "r") as f_data:
        dist_map, _ = load_camera_params(f_data)
        left = f_data["prophesee"]["left"]
        ms_map = left["ms_map_idx"][:]
        last_active = int(left["t"][0])
        window_start = last_active * US_TO_S
        frame_idx = 0
        target_ptr = 0

        arrays = [np.zeros((360, 360), dtype=np.float32) for _ in range(8)]
        sum_all, w_all, sum_pos, w_pos, sum_neg, w_neg, last_pos, last_neg = arrays
        last_pos.fill(-1.0)
        last_neg.fill(-1.0)

        num_batches = (len(ms_map) - 1) // args.event_batch_ms
        for batch_idx in range(num_batches):
            start = int(ms_map[batch_idx * args.event_batch_ms])
            end = int(ms_map[(batch_idx + 1) * args.event_batch_ms])
            x_all = left["x"][start:end]
            keep = ~((x_all < 280) | (x_all > 999))
            if keep.sum() == 0:
                continue
            last_active, frame_idx, target_ptr, window_start = process_time_chunk(
                x_all[keep],
                left["y"][start:end][keep],
                left["p"][start:end][keep],
                left["t"][start:end][keep],
                dist_map,
                out,
                source_indices,
                last_active,
                frame_idx,
                target_ptr,
                window_start,
                args.output_time_ms * 1e-3,
                args.tau,
                sum_all,
                w_all,
                sum_pos,
                w_pos,
                sum_neg,
                w_neg,
                last_pos,
                last_neg,
            )
            if target_ptr == len(source_indices):
                break
    if target_ptr != len(source_indices):
        raise RuntimeError(f"Only reconstructed {target_ptr}/{len(source_indices)} time maps for {seq_name}")
    return np.clip(out, 0.0, 1.0)


def ensure_time_maps(args, split_name, b1_full_h5, out_h5):
    if Path(out_h5).exists():
        with h5py.File(out_h5, "r") as f:
            if tuple(f["time_maps"].shape[1:]) == (5, 360, 360):
                return str(out_h5)

    with h5py.File(b1_full_h5, "r") as f:
        seq_name = f.attrs["sequence"]
        source_indices = f["source_indices"][:].astype(np.int64)
    maps = build_time_maps(args.data_root, seq_name, source_indices, args)
    with h5py.File(out_h5, "w") as f:
        f.attrs["split"] = split_name
        f.attrs["sequence"] = seq_name
        f.attrs["channels"] = ",".join(TIME_CHANNELS)
        f.create_dataset("source_indices", data=source_indices, dtype=np.int64)
        f.create_dataset(
            "time_maps",
            data=maps.astype(np.float16),
            dtype=np.float16,
            chunks=(1, 5, 360, 360),
            **hdf5plugin.Blosc2(cname="lz4", clevel=5, filters=hdf5plugin.Blosc.SHUFFLE),
        )
    return str(out_h5)


def ensure_b2_h5(b1_full_h5, time_h5, out_h5, time_indices, method_name):
    expected_channels = 12 + len(time_indices)
    if Path(out_h5).exists():
        with h5py.File(out_h5, "r") as f:
            if tuple(f["exp_filts"].shape[1:]) == (expected_channels, 360, 360):
                return str(out_h5)
    with h5py.File(b1_full_h5, "r") as f_base, h5py.File(time_h5, "r") as f_time, h5py.File(out_h5, "w") as f_out:
        n = len(f_base["exp_filts"])
        seq = f_base.attrs["sequence"]
        f_out.attrs["input_definition"] = method_name
        f_out.attrs["sequence"] = seq
        f_out.attrs["time_channels"] = ",".join([TIME_CHANNELS[i] for i in time_indices])
        f_out.create_dataset("file_names", data=np.asarray([seq], dtype=h5py.string_dtype()))
        f_out.create_dataset("source_indices", data=f_base["source_indices"][:], dtype=np.int64)
        exp = f_out.create_dataset(
            "exp_filts",
            shape=(n, expected_channels, 360, 360),
            dtype=np.float16,
            chunks=(1, expected_channels, 360, 360),
            **hdf5plugin.Blosc2(cname="lz4", clevel=5, filters=hdf5plugin.Blosc.SHUFFLE),
        )
        for start in range(0, n, 64):
            end = min(start + 64, n)
            base = f_base["exp_filts"][start:end].astype(np.float16)
            time = f_time["time_maps"][start:end, time_indices].astype(np.float16)
            exp[start:end] = np.concatenate([base, time], axis=1)
        for key in ["ttc", "mask"]:
            src = f_base[key]
            f_out.create_dataset(
                key,
                data=src[:],
                dtype=src.dtype,
                chunks=(1, *src.shape[1:]),
                **hdf5plugin.Blosc2(cname="lz4", clevel=5, filters=hdf5plugin.Blosc.SHUFFLE),
            )
    return str(out_h5)


def clone_args(base, **kwargs):
    out = copy.copy(base)
    for k, v in kwargs.items():
        setattr(out, k, v)
    return out


def fmt(metrics, name, epochs, channels):
    p = metrics["paper_metrics"]
    s = metrics["safety_metrics"]
    b = p["MAE_B1_B3"]
    return [
        name,
        str(epochs),
        str(channels),
        f"{p['mean_frame_median_AE']:.6f}",
        f"{p['MRE_percent']:.6f}",
        f"{p['RE_10_percent']:.6f}/{p['RE_50_percent']:.6f}/{p['RE_80_percent']:.6f}",
        f"{b[0]:.6f}/{b[1]:.6f}/{b[2]:.6f}",
        f"{s['pixel_MAE']:.6f}",
        f"{s['high_risk_MAE']:.6f}",
        f"{s['high_risk_false_negative_rate']:.6f}",
        f"{s['risk_bins']['0_1s']['false_negative_rate']:.6f}",
    ]


def write_b2_report(args, metrics_by_name):
    b1_full = json.loads((Path(args.b1_full_dir) / "paper_metrics.json").read_text())
    rows = [fmt(b1_full, "B1-Full", 20, 12)]
    rows.extend(
        [
            fmt(metrics_by_name["B2-Lite-MeanTime"], "B2-Lite-MeanTime", args.epochs, 13),
            fmt(metrics_by_name["B2-PolarityMeanTime"], "B2-PolarityMeanTime", args.epochs, 14),
            fmt(metrics_by_name["B2-Freshness"], "B2-Freshness", args.epochs, 14),
        ]
    )
    lines = [
        "# B2 显式时间信息消融报告",
        "",
        "| 方法 | Epochs | 输入通道数 | 逐帧中位 AE 均值 | MRE(%) | RE[10/50/80] | Band MAE(B1-B3) | 像素 MAE | 高风险 MAE<=2s | 高风险漏检<=2s | 漏检<=1s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")

    def val(name, path):
        cur = metrics_by_name[name]
        return cur["safety_metrics"]["high_risk_false_negative_rate"], cur["safety_metrics"]["risk_bins"]["0_1s"]["false_negative_rate"], cur["paper_metrics"]["mean_frame_median_AE"]

    lines.extend(
        [
            "",
            "## 判断重点",
            "- 表中 B1-Full 是 12 通道极性分离基线；B2 在其后追加显式时间通道。",
            "- 重点比较逐帧中位 AE、MRE、RE[50/80]、高风险漏检<=2s、漏检<=1s。",
        ]
    )
    (Path(args.report_out)).write_text("\n".join(lines) + "\n")


def run_one(base_args, name, out_dir, time_indices, time_names, prefix, in_channels):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    method_args = clone_args(
        base_args,
        out_dir=str(out_dir),
        in_channels=in_channels,
        method_name=name,
        input_definition=f"{name}: B1-Full + {', '.join(time_names)}",
        paper_title=f"{name} 论文同口径指标",
        checkpoint_prefix=prefix,
        visual_channel_indices=[0, 2, 5, 6, 8, 11] + list(range(12, in_channels)),
        visual_channel_names=["pos scale0", "pos scale2", "pos scale5", "neg scale0", "neg scale2", "neg scale5"] + time_names,
    )
    (out_dir / "run_config.json").write_text(json.dumps(vars(method_args), indent=2))
    train_h5 = ensure_b2_h5(base_args.b1_full_train, base_args.time_train, out_dir / "train.h5", time_indices, name)
    test_h5 = ensure_b2_h5(base_args.b1_full_test, base_args.time_test, out_dir / "test.h5", time_indices, name)
    rows = train(method_args, train_h5, test_h5, out_dir)
    metrics = evaluate(method_args, out_dir / f"{prefix}_best.pt", test_h5, out_dir)
    summary = {
        "method": name,
        "epochs": base_args.epochs,
        "in_channels": in_channels,
        "time_channels": time_names,
        "train_file": train_h5,
        "test_file": test_h5,
        "final_epoch": rows[-1],
        "best_checkpoint": str(out_dir / f"{prefix}_best.pt"),
        "last_checkpoint": str(out_dir / f"{prefix}_last.pt"),
        "metrics": metrics,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return metrics


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="/home/hello/research_project/event+SNN+TTC/data_m3ed_min")
    p.add_argument("--b1-full-dir", default="/home/hello/research_project/event+SNN+TTC/logs/ttc_TTCEF/b1_polarity_full_20ep")
    p.add_argument("--b1-full-train", default="/home/hello/research_project/event+SNN+TTC/logs/ttc_TTCEF/b1_polarity_full_20ep/b1_full_train.h5")
    p.add_argument("--b1-full-test", default="/home/hello/research_project/event+SNN+TTC/logs/ttc_TTCEF/b1_polarity_full_20ep/b1_full_test.h5")
    p.add_argument("--work-dir", default="/home/hello/research_project/event+SNN+TTC/logs/ttc_TTCEF")
    p.add_argument("--report-out", default="/home/hello/research_project/event+SNN+TTC/logs/ttc_TTCEF/b2_time_ablation_report.md")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--eval-batch-size", type=int, default=16)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--pct-start", type=float, default=0.1)
    p.add_argument("--quick-val-batches", type=int, default=16)
    p.add_argument("--output-time-ms", type=float, default=7.0)
    p.add_argument("--event-batch-ms", type=int, default=1000)
    p.add_argument("--tau", type=float, default=0.007)
    p.add_argument("--rel-eps", type=float, default=1e-6)
    p.add_argument("--high-risk-ttc", type=float, default=2.0)
    p.add_argument("--num-vis-samples", type=int, default=20)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    work = Path(args.work_dir)
    args.time_train = ensure_time_maps(args, "train", args.b1_full_train, work / "b2_time_maps_train.h5")
    args.time_test = ensure_time_maps(args, "test", args.b1_full_test, work / "b2_time_maps_test.h5")
    metrics = {}
    metrics["B2-Lite-MeanTime"] = run_one(
        args,
        "B2-Lite-MeanTime",
        work / "b2_lite_mean_time_20ep",
        [0],
        ["mean_time_all"],
        "b2_lite_mean_time",
        13,
    )
    metrics["B2-PolarityMeanTime"] = run_one(
        args,
        "B2-PolarityMeanTime",
        work / "b2_polarity_mean_time_20ep",
        [1, 2],
        ["mean_time_pos", "mean_time_neg"],
        "b2_polarity_mean_time",
        14,
    )
    metrics["B2-Freshness"] = run_one(
        args,
        "B2-Freshness",
        work / "b2_freshness_20ep",
        [3, 4],
        ["fresh_pos", "fresh_neg"],
        "b2_freshness",
        14,
    )
    write_b2_report(args, metrics)
    print(json.dumps({k: v["paper_metrics"] for k, v in metrics.items()}, indent=2))


if __name__ == "__main__":
    main()
