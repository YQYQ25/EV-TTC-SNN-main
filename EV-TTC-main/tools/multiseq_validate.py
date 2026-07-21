#!/usr/bin/env python3
import argparse
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import h5py
import hdf5plugin  # noqa: F401
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "TTCEF"))
sys.path.insert(0, str(ROOT / "tools"))

from merge import combine_files  # noqa: E402
from b1_matched_train_eval import evaluate, ensure_b1_h5, seed_all  # noqa: E402


def model_args(base, **kwargs):
    defaults = {
        "data_root": base.data_root,
        "in_channels": 6,
        "batch_size": 16,
        "eval_batch_size": 16,
        "workers": 4,
        "rel_eps": 1e-6,
        "high_risk_ttc": 2.0,
        "num_vis_samples": 10,
        "cpu": False,
        "dt_ms": 0.2,
        "output_time_ms": 7.0,
        "alphas": [0.12, 0.06, 0.03, 0.015, 0.0095, 0.0045],
        "event_batch_ms": 1000,
        "seed": 1,
        "method_name": "method",
        "paper_title": "论文同口径指标",
        "input_definition": "",
        "checkpoint_prefix": "checkpoint",
        "visual_channel_indices": list(range(6)),
        "visual_channel_names": [f"ch{i}" for i in range(6)],
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def ensure_single_test_h5(seq, data_root, out_file):
    out_file = Path(out_file)
    if out_file.exists():
        with h5py.File(out_file, "r") as f:
            if "exp_filts" in f and len(f["exp_filts"]) > 0:
                return str(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    combine_files(data_root, str(out_file), [seq], chunk_size=250)
    return str(out_file)


def finite_stats(arr, mask=None):
    x = arr[mask] if mask is not None else arr.reshape(-1)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"min": None, "max": None, "mean": None, "median": None}
    return {
        "min": float(x.min()),
        "max": float(x.max()),
        "mean": float(x.mean(dtype=np.float64)),
        "median": float(np.median(x)),
    }


def save_data_visuals(test_h5, out_dir, num=4):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    with h5py.File(test_h5, "r") as f:
        n = len(f["exp_filts"])
        ids = np.linspace(0, max(n - 1, 0), min(num, n), dtype=int)
        for idx in ids:
            exp = f["exp_filts"][idx].astype(np.float32)
            ttc = f["ttc"][idx].astype(np.float32)
            mask = f["mask"][idx].astype(bool)
            fig, axes = plt.subplots(1, 4, figsize=(13, 3.5), constrained_layout=True)
            event = np.abs(exp[-1])
            axes[0].imshow(event, cmap="magma", vmin=0, vmax=np.percentile(event, 99.5) + 1e-6)
            axes[0].set_title("event density")
            vmax = max(float(np.nanpercentile(np.where(mask, ttc, np.nan), 95)) if mask.any() else 5.0, 2.0)
            axes[1].imshow(np.where(mask, ttc, np.nan), cmap="viridis", vmin=0, vmax=vmax)
            axes[1].set_title("GT TTC")
            axes[2].imshow(mask.astype(float), cmap="gray", vmin=0, vmax=1)
            axes[2].set_title("mask")
            axes[3].imshow(exp[0], cmap="coolwarm")
            axes[3].set_title("input ch0")
            for ax in axes:
                ax.axis("off")
            path = out_dir / f"data_check_sample_{idx:05d}.png"
            fig.savefig(path, dpi=150)
            plt.close(fig)
            paths.append(str(path))
    return paths


def write_data_check(seq, test_h5, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(test_h5, "r") as f:
        exp = f["exp_filts"]
        ttc = f["ttc"]
        mask = f["mask"]
        sample_count = len(exp)
        exp_shape = tuple(exp.shape)
        ttc_shape = tuple(ttc.shape)
        mask_sum = 0
        pix_count = 0
        nan_count = 0
        inf_count = 0
        nz_sum = 0
        ttc_values = []
        for start in range(0, sample_count, 128):
            end = min(start + 128, sample_count)
            e = exp[start:end].astype(np.float32)
            y = ttc[start:end].astype(np.float32)
            m = mask[start:end].astype(bool)
            mask_sum += int(m.sum())
            pix_count += int(m.size)
            nan_count += int(np.isnan(y).sum() + np.isnan(e).sum())
            inf_count += int(np.isinf(y).sum() + np.isinf(e).sum())
            nz_sum += int((np.abs(e) > 1e-6).sum())
            vals = y[m & np.isfinite(y)]
            if vals.size:
                ttc_values.append(vals.astype(np.float32))
        all_ttc = np.concatenate(ttc_values) if ttc_values else np.array([], dtype=np.float32)
        label_stats = finite_stats(all_ttc)
        valid_ratio = mask_sum / max(pix_count, 1)
        event_density = nz_sum / max(sample_count * np.prod(exp_shape[1:]), 1)
    vis = save_data_visuals(test_h5, out_dir / "data_visualizations", num=4)
    summary = {
        "sequence": seq,
        "test_h5": str(test_h5),
        "sample_count": sample_count,
        "input_shape": exp_shape,
        "ttc_shape": ttc_shape,
        "label_range": label_stats,
        "valid_mask_ratio": valid_ratio,
        "nan_count": nan_count,
        "inf_count": inf_count,
        "event_density_nonzero_ratio": event_density,
        "visualizations": vis,
    }
    (out_dir / "data_check.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = [
        f"# {seq} 数据检查",
        "",
        f"- test.h5：`{test_h5}`",
        f"- 样本数：`{sample_count}`",
        f"- 输入 shape：`{exp_shape}`",
        f"- TTC map shape：`{ttc_shape}`",
        f"- label 范围：min `{label_stats['min']}`, max `{label_stats['max']}`, mean `{label_stats['mean']}`, median `{label_stats['median']}`",
        f"- valid mask ratio：`{valid_ratio:.6f}`",
        f"- NaN 数量：`{nan_count}`",
        f"- Inf 数量：`{inf_count}`",
        f"- event density 非零比例：`{event_density:.6f}`",
        "",
        "## 可视化",
    ]
    lines.extend([f"- `{p}`" for p in vis])
    (out_dir / "data_check.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def read_metrics(path):
    return json.loads(Path(path).read_text())


def fmt_metric(metrics, name, channels):
    def fmt(value):
        return "N/A" if value is None else f"{value:.6f}"

    p = metrics["paper_metrics"]
    s = metrics["safety_metrics"]
    b = p["MAE_B1_B3"]
    return [
        name,
        str(channels),
        f"{p['mean_frame_median_AE']:.6f}",
        f"{p['MRE_percent']:.6f}",
        f"{p['RE_10_percent']:.6f}/{p['RE_50_percent']:.6f}/{p['RE_80_percent']:.6f}",
        f"{b[0]:.6f}/{b[1]:.6f}/{b[2]:.6f}",
        f"{s['pixel_MAE']:.6f}",
        f"{s['pixel_median_AE']:.6f}",
        fmt(s["high_risk_MAE"]),
        fmt(s["high_risk_false_negative_rate"]),
        fmt(s["risk_bins"]["0_1s"]["false_negative_rate"]),
    ]


def pct_delta(new, old, lower_is_better=True):
    if new is None or old is None or old == 0:
        return None
    raw = 100.0 * (old - new) / old if lower_is_better else 100.0 * (new - old) / old
    return raw


def improved(new, old):
    return new is not None and old is not None and new < old


def improvement_verdict(new, old):
    if new is None or old is None:
        return "不可计算"
    return "是" if new < old else "否"


def write_general_report(base_out, sequences):
    lines = [
        "# 多序列泛化验证报告",
        "",
        "## 当前固定结论",
        "- B1-Matched 是 6 通道公平对比：整体 paper-style 指标优于 B0，但高风险漏检变差。",
        "- B1-Full 是 12 通道 upper bound：整体指标最优，FN<=1s 明显优于 B0 和 B1-Matched，目前作为主模型候选。",
        "- B2 时间信息实验未超过 B1-Full，当前暂停 B2。",
        "",
        "## 新增序列",
    ]
    if not sequences:
        lines.append("- 本次没有完成新增序列评估。")
    for item in sequences:
        dc = item["data_check"]
        lines.append(
            f"- `{item['sequence']}`：样本数 `{dc['sample_count']}`，valid mask ratio `{dc['valid_mask_ratio']:.6f}`，"
            f"TTC mean `{dc['label_range']['mean']}`，TTC median `{dc['label_range']['median']}`"
        )
    for item in sequences:
        lines.extend(["", f"## {item['sequence']} 指标对比", ""])
        lines.extend(
            [
                "| 方法 | 输入通道数 | 逐帧中位 AE | MRE(%) | RE[10/50/80] | Band MAE(B1-B3) | 像素 MAE | 像素中位 AE | 高风险 MAE<=2s | 高风险漏检<=2s | 漏检<=1s |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        b0 = item["metrics"]["B0"]
        b1 = item["metrics"]["B1-Full"]
        lines.append("| " + " | ".join(fmt_metric(b0, "B0 signed IIR", 6)) + " |")
        lines.append("| " + " | ".join(fmt_metric(b1, "B1-Full polarity split", 12)) + " |")
        p0, p1 = b0["paper_metrics"], b1["paper_metrics"]
        s0, s1 = b0["safety_metrics"], b1["safety_metrics"]
        deltas = {
            "逐帧中位 AE": pct_delta(p1["mean_frame_median_AE"], p0["mean_frame_median_AE"]),
            "MRE": pct_delta(p1["MRE_percent"], p0["MRE_percent"]),
            "RE50": pct_delta(p1["RE_50_percent"], p0["RE_50_percent"]),
            "RE80": pct_delta(p1["RE_80_percent"], p0["RE_80_percent"]),
            "上方 Band MAE": pct_delta(p1["MAE_B1_B3"][0], p0["MAE_B1_B3"][0]),
            "HR FN<=2s": pct_delta(s1["high_risk_false_negative_rate"], s0["high_risk_false_negative_rate"]),
            "FN<=1s": pct_delta(s1["risk_bins"]["0_1s"]["false_negative_rate"], s0["risk_bins"]["0_1s"]["false_negative_rate"]),
        }
        lines.extend(["", "### B1-Full 相对 B0 改变量（正数表示降低/改善）"])
        for k, v in deltas.items():
            lines.append(f"- {k}：`{v:.2f}%`" if v is not None else f"- {k}：不可计算")
        stable = (
            p1["mean_frame_median_AE"] < p0["mean_frame_median_AE"]
            and p1["MRE_percent"] < p0["MRE_percent"]
        )
        near_verdict = improvement_verdict(
            s1["risk_bins"]["0_1s"]["false_negative_rate"],
            s0["risk_bins"]["0_1s"]["false_negative_rate"],
        )
        upper_band_verdict = improvement_verdict(p1["MAE_B1_B3"][0], p0["MAE_B1_B3"][0])
        lines.extend(
            [
                "",
                "### 判断",
                f"- B1-Full 多序列整体指标是否优于 B0：`{'是' if stable else '否'}`",
                f"- B1-Full 极近 TTC<=1s 改善是否稳定存在：`{near_verdict}`",
                f"- 上方 band 是否改善：`{upper_band_verdict}`",
            ]
        )
    lines.extend(
        [
            "",
            "## 下一步建议",
            "- 若 B1-Full 在新增序列上保持整体优势，可进入多序列联合训练验证。",
            "- 若上方 band 或 TTC<=1s 在新增序列中仍不稳定，应先补充多序列数据和重新审视时间通道定义，再考虑 B3。",
            "- 当前报告不包含 B2 继续扩展、B3 风险门控、Flightmare、ROS 或 TensorRT。",
        ]
    )
    (Path(base_out) / "multiseq_generalization_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", action="append", required=True)
    parser.add_argument("--data-root", default="/home/hello/research_project/event+SNN+TTC/data_m3ed_multiseq")
    parser.add_argument("--multiseq-root", default="/home/hello/research_project/event+SNN+TTC/ttcef_multiseq")
    parser.add_argument("--out-root", default="/home/hello/research_project/event+SNN+TTC/logs/ttc_TTCEF/multiseq_validation")
    parser.add_argument("--b0-checkpoint", default="/home/hello/research_project/event+SNN+TTC/logs/ttc_TTCEF/b0_signed_baseline_20ep/b0_signed_baseline_best.pt")
    parser.add_argument("--b1-full-checkpoint", default="/home/hello/research_project/event+SNN+TTC/logs/ttc_TTCEF/b1_polarity_full_20ep/b1_polarity_full_best.pt")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    seed_all(1)
    done = []
    for seq in args.sequence:
        seq_out = Path(args.out_root) / seq
        seq_out.mkdir(parents=True, exist_ok=True)
        test_h5 = ensure_single_test_h5(seq, args.data_root, Path(args.multiseq_root) / seq / "test.h5")
        data_check = write_data_check(seq, test_h5, seq_out)

        b0_args = model_args(
            args,
            method_name="B0 signed IIR",
            paper_title=f"{seq} B0 论文同口径指标",
            in_channels=6,
            workers=args.workers,
            eval_batch_size=args.batch_size,
            num_vis_samples=10,
            cpu=args.cpu,
            visual_channel_indices=[0, 1, 2, 3, 4, 5],
            visual_channel_names=[f"signed scale{i}" for i in range(6)],
        )
        b0_dir = seq_out / "b0_signed_iir"
        b0_dir.mkdir(parents=True, exist_ok=True)
        b0_metrics = evaluate(b0_args, Path(args.b0_checkpoint), test_h5, b0_dir)

        b1_args = model_args(
            args,
            method_name="B1-Full polarity split",
            paper_title=f"{seq} B1-Full 论文同口径指标",
            in_channels=12,
            workers=args.workers,
            eval_batch_size=args.batch_size,
            num_vis_samples=10,
            cpu=args.cpu,
            b1_train_name="unused_train.h5",
            b1_test_name="b1_full_test.h5",
            input_definition="B1-Full polarity split: pos scale0..5, neg scale0..5",
            visual_channel_indices=[0, 2, 5, 6, 8, 11],
            visual_channel_names=["pos scale0", "pos scale2", "pos scale5", "neg scale0", "neg scale2", "neg scale5"],
        )
        b1_full_h5 = seq_out / "b1_full_test.h5"
        _, b1_full_test = ensure_b1_h5(b1_args, test_h5, b1_full_h5)
        b1_dir = seq_out / "b1_full_polarity_split"
        b1_dir.mkdir(parents=True, exist_ok=True)
        b1_metrics = evaluate(b1_args, Path(args.b1_full_checkpoint), b1_full_test, b1_dir)

        done.append(
            {
                "sequence": seq,
                "test_h5": test_h5,
                "b1_full_test_h5": b1_full_test,
                "data_check": data_check,
                "metrics": {"B0": b0_metrics, "B1-Full": b1_metrics},
            }
        )
        (seq_out / "sequence_summary.json").write_text(json.dumps(done[-1], indent=2), encoding="utf-8")
    write_general_report(args.out_root, done)
    print(json.dumps({"completed_sequences": [d["sequence"] for d in done]}, indent=2))


if __name__ == "__main__":
    main()
