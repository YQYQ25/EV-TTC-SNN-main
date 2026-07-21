#!/usr/bin/env python3
"""Finalize B3 report and high-risk visualizations from completed runs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import hdf5plugin  # noqa: F401
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model"))
sys.path.insert(0, str(ROOT / "tools"))

from b3_eval import load_model  # noqa: E402
from b3_temporal_data import TemporalSegmentDataset  # noqa: E402
from evslim import EVSlim  # noqa: E402


METHODS = {
    "b1": {
        "label": "Original B1-Full",
        "metrics": "original_b1_full/metrics.json",
        "checkpoint": "../b1_polarity_full_20ep/b1_polarity_full_best.pt",
        "mode": "c0",
        "b1_checkpoint": True,
    },
    "c0": {
        "label": "C0 B1-ClipFT",
        "metrics": "c0_b1_clipft/metrics.json",
        "checkpoint": "c0_b1_clipft/best.pt",
        "mode": "c0",
        "b1_checkpoint": False,
    },
    "oracle": {
        "label": "B3-Oracle",
        "metrics": "b3_oracle/metrics.json",
        "checkpoint": "b3_oracle/best.pt",
        "mode": "oracle",
        "b1_checkpoint": False,
    },
    "closed": {
        "label": "B3-Closed",
        "metrics": "b3_closed/metrics.json",
        "checkpoint": "b3_closed/best.pt",
        "mode": "closed",
        "b1_checkpoint": False,
    },
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def pct(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{100.0 * value:.2f}%"


def num(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "N/A"
    return f"{value:.{digits}f}"


def rel_change(new: float, old: float) -> float:
    return (new - old) / old if old else float("nan")


def metric_row(metrics: dict, added_latency_ms: float | None) -> list[str]:
    paper = metrics["paper_metrics"]
    safety = metrics["safety_metrics"]
    regional = metrics["regional_safety"]
    fn_1s = safety["risk_bins"]["0_1s"]["false_negative_rate"]
    return [
        metrics["name"],
        num(paper["mean_frame_median_AE"]),
        f"{paper['MRE_percent']:.2f}%",
        num(safety["high_risk_MAE"]),
        pct(safety["high_risk_false_negative_rate"]),
        pct(fn_1s),
        pct(regional["center_danger_recall"]),
        pct(regional["left_right_safe_direction_agreement"]),
        "N/A" if added_latency_ms is None else f"{added_latency_ms:.3f} ms",
    ]


def load_metrics(base: Path) -> dict:
    out = {}
    for key, item in METHODS.items():
        path = base / item["metrics"]
        if not path.exists():
            raise FileNotFoundError(f"missing metrics file: {path}")
        out[key] = read_json(path)
    return out


def select_high_risk_frames(dataset: TemporalSegmentDataset, limit: int) -> list[tuple[int, int, int, int]]:
    candidates: list[tuple[int, int, int, int]] = []
    for segment_index in range(len(dataset)):
        _, ttc_seq, mask_seq, metadata = dataset[segment_index]
        for time_index in range(ttc_seq.shape[0]):
            gt = ttc_seq[time_index, 0].numpy()
            mask = mask_seq[time_index, 0].numpy().astype(bool)
            count_1s = int((mask & np.isfinite(gt) & (gt > 0) & (gt <= 1.0)).sum())
            count_2s = int((mask & np.isfinite(gt) & (gt > 0) & (gt <= 2.0)).sum())
            if count_1s > 0:
                source = int(metadata["source_indices"][time_index])
                candidates.append((count_1s, count_2s, segment_index, time_index, source))
    candidates.sort(reverse=True)
    selected = sorted(candidates[:limit], key=lambda item: (item[2], item[3]))
    return [(segment, time_index, count_1s, source) for count_1s, _, segment, time_index, source in selected]


def plot_frame(
    path: Path,
    title: str,
    gt: np.ndarray,
    mask: np.ndarray,
    predictions: dict[str, np.ndarray],
    gt_risk: np.ndarray,
    closed_risk: np.ndarray,
    closed_gains: np.ndarray,
) -> None:
    panels: list[tuple[str, np.ndarray, str, float | None, float | None]] = [
        ("GT TTC", gt, "viridis", 0.0, 5.0),
        ("C0 prediction", predictions["c0"], "viridis", 0.0, 5.0),
        ("Oracle prediction", predictions["oracle"], "viridis", 0.0, 5.0),
        ("Closed prediction", predictions["closed"], "viridis", 0.0, 5.0),
        ("C0 abs error", np.abs(predictions["c0"] - gt), "magma", 0.0, 2.0),
        ("Oracle abs error", np.abs(predictions["oracle"] - gt), "magma", 0.0, 2.0),
        ("Closed abs error", np.abs(predictions["closed"] - gt), "magma", 0.0, 2.0),
        ("Mask", mask.astype(float), "gray", 0.0, 1.0),
        ("GT 3x3 risk", gt_risk, "inferno", 0.0, 3.0),
        ("Closed 3x3 risk", closed_risk, "inferno", 0.0, 3.0),
        ("Closed gain scale 0", closed_gains[0], "coolwarm", 0.5, 1.5),
        ("Closed gain scale 1", closed_gains[1], "coolwarm", 0.5, 1.5),
        ("Closed gain scale 2", closed_gains[2], "coolwarm", 0.5, 1.5),
        ("Closed gain scale 3", closed_gains[3], "coolwarm", 0.5, 1.5),
        ("Closed gain scale 4", closed_gains[4], "coolwarm", 0.5, 1.5),
        ("Closed gain scale 5", closed_gains[5], "coolwarm", 0.5, 1.5),
    ]
    figure, axes = plt.subplots(4, 4, figsize=(15, 13), constrained_layout=True)
    figure.suptitle(title, fontsize=12)
    for axis, (name, array, cmap, vmin, vmax) in zip(axes.ravel(), panels):
        shown = array if name == "Mask" or array.shape == (3, 3) else np.where(mask, array, np.nan)
        image = axis.imshow(shown, cmap=cmap, vmin=vmin, vmax=vmax)
        axis.set_title(name, fontsize=9)
        axis.axis("off")
        if array.shape == (3, 3):
            for row in range(3):
                for col in range(3):
                    axis.text(col, row, f"{array[row, col]:.2f}", ha="center", va="center", color="white", fontsize=8)
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.savefig(path, dpi=140)
    plt.close(figure)


@torch.no_grad()
def make_visualizations(args, out_dir: Path) -> list[dict]:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    dataset = TemporalSegmentDataset(args.test_h5, args.test_clips)
    selected = select_high_risk_frames(dataset, args.num_visualizations)
    selected_by_segment: dict[int, dict[int, tuple[int, int]]] = {}
    for segment, time_index, count_1s, source in selected:
        selected_by_segment.setdefault(segment, {})[time_index] = (count_1s, source)

    b3_base = Path(args.b3_dir)
    models = {
        key: load_model(b3_base / METHODS[key]["checkpoint"], device, METHODS[key]["mode"], METHODS[key]["b1_checkpoint"])
        for key in ("c0", "oracle", "closed")
    }
    visual_dir = out_dir / "high_risk_visualizations"
    visual_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for segment_index in sorted(selected_by_segment):
        x_seq, ttc_seq, mask_seq, metadata = dataset[segment_index]
        previous_gt = None
        previous_x = None
        previous_predictions = {"c0": None, "oracle": None, "closed": None}
        for time_index in range(x_seq.shape[0]):
            x = x_seq[time_index][None].to(device)
            gt = ttc_seq[time_index][None].to(device)
            mask = mask_seq[time_index][None].to(device)

            outputs = {}
            for key, model in models.items():
                if key == "c0" or previous_gt is None:
                    previous = None
                elif key == "oracle":
                    previous = previous_gt
                else:
                    previous = previous_predictions["closed"]
                prediction, aux = model(x, previous, previous_x, return_aux=True)
                outputs[key] = (prediction, aux)

            if time_index in selected_by_segment[segment_index]:
                count_1s, source = selected_by_segment[segment_index][time_index]
                gt_gains, gt_risk, _ = models["closed"].modulator.compute_gains(previous_gt, previous_x, reference_x=x)
                predictions = {key: outputs[key][0][0, 0].float().cpu().numpy() for key in outputs}
                closed_aux = outputs["closed"][1]
                path = visual_dir / f"highrisk_{len(rows):02d}_seg{segment_index}_t{time_index}_src{source}.png"
                plot_frame(
                    path,
                    f"segment={segment_index}, t={time_index}, source={source}, TTC<=1s pixels={count_1s}",
                    gt[0, 0].cpu().numpy(),
                    mask[0, 0].cpu().numpy().astype(bool),
                    predictions,
                    gt_risk[0].float().cpu().numpy(),
                    closed_aux["risk"][0].float().cpu().numpy(),
                    closed_aux["gains"][0].float().cpu().numpy(),
                )
                rows.append(
                    {
                        "index": len(rows),
                        "segment": segment_index,
                        "time_index": time_index,
                        "source_index": source,
                        "ttc_le_1s_pixels": count_1s,
                        "path": str(path),
                    }
                )

            previous_gt = gt.detach()
            previous_x = x.detach()
            for key in previous_predictions:
                previous_predictions[key] = outputs[key][0].detach()

    lines = ["# B3 高风险连续片段可视化", ""]
    lines.append("| Index | Segment | t | Source frame | TTC<=1s pixels | Path |")
    lines.append("|---:|---:|---:|---:|---:|---|")
    for row in rows:
        lines.append(
            f"| {row['index']} | {row['segment']} | {row['time_index']} | {row['source_index']} | "
            f"{row['ttc_le_1s_pixels']} | `{row['path']}` |"
        )
    (visual_dir / "visualization_index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return rows


def write_report(base: Path, visualization_rows: list[dict]) -> None:
    metrics = load_metrics(base)
    baseline_forward = metrics["b1"]["latency_ms"]["evslim_forward_mean"]
    added = {
        key: (metrics[key]["latency_ms"]["complete_b3_mean"] - baseline_forward)
        if metrics[key]["latency_ms"]["complete_b3_mean"] is not None
        else None
        for key in metrics
    }
    added["b1"] = 0.0
    c0, oracle, closed = metrics["c0"], metrics["oracle"], metrics["closed"]
    c0_hr_fn = c0["safety_metrics"]["high_risk_false_negative_rate"]
    oracle_hr_fn = oracle["safety_metrics"]["high_risk_false_negative_rate"]
    closed_hr_fn = closed["safety_metrics"]["high_risk_false_negative_rate"]
    c0_fn1 = c0["safety_metrics"]["risk_bins"]["0_1s"]["false_negative_rate"]
    oracle_fn1 = oracle["safety_metrics"]["risk_bins"]["0_1s"]["false_negative_rate"]
    closed_fn1 = closed["safety_metrics"]["risk_bins"]["0_1s"]["false_negative_rate"]
    oracle_fn1_gain = c0_fn1 - oracle_fn1
    closed_fn1_gain = c0_fn1 - closed_fn1
    retention = closed_fn1_gain / oracle_fn1_gain if oracle_fn1_gain > 0 else float("nan")

    lines = [
        "# B3 3x3 局部风险自适应最终报告",
        "",
        "## 实验设置",
        "",
        "- 数据：B1-Full 12 通道连续 test segments，来自当前 night car 最小数据集。",
        "- 测试方式：完整连续 segment 递推，只在真实 segment 边界重置上一帧风险状态。",
        "- C0、Oracle、Closed 均从同一个 B1-Full best checkpoint 初始化。",
        "- 当前阶段未做 BPTT、学习门控、风险加权 loss、ROS、TensorRT 或 Flightmare。",
        "",
        "## 核心指标",
        "",
        "| Method | Mean frame median AE | MRE | HR MAE<=2s | HR FN<=2s | FN<=1s | Center danger recall | Direction agreement | Added latency |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key in ("b1", "c0", "oracle", "closed"):
        lines.append("| " + " | ".join(metric_row(metrics[key], added[key])) + " |")

    lines += [
        "",
        "## 相对 C0 的变化",
        "",
        f"- Oracle Mean frame median AE：{rel_change(oracle['paper_metrics']['mean_frame_median_AE'], c0['paper_metrics']['mean_frame_median_AE']) * 100:.2f}%。",
        f"- Oracle HR FN<=2s：{rel_change(oracle_hr_fn, c0_hr_fn) * 100:.2f}%。",
        f"- Oracle FN<=1s：{rel_change(oracle_fn1, c0_fn1) * 100:.2f}%。",
        f"- Closed Mean frame median AE：{rel_change(closed['paper_metrics']['mean_frame_median_AE'], c0['paper_metrics']['mean_frame_median_AE']) * 100:.2f}%。",
        f"- Closed HR FN<=2s：{rel_change(closed_hr_fn, c0_hr_fn) * 100:.2f}%。",
        f"- Closed FN<=1s：{rel_change(closed_fn1, c0_fn1) * 100:.2f}%。",
        f"- Closed 保留 Oracle 在 FN<=1s 上收益的 {retention * 100:.2f}%。",
        "",
        "## 区域与控制代理指标",
        "",
        "| Method | Most-dangerous region accuracy | Center danger recall | Direction agreement |",
        "|---|---:|---:|---:|",
    ]
    for key in ("b1", "c0", "oracle", "closed"):
        regional = metrics[key]["regional_safety"]
        lines.append(
            f"| {metrics[key]['name']} | {pct(regional['most_dangerous_region_accuracy'])} | "
            f"{pct(regional['center_danger_recall'])} | {pct(regional['left_right_safe_direction_agreement'])} |"
        )

    lines += [
        "",
        "## 判断",
        "",
        "- Oracle 相比 C0 的整体 paper-style 指标改善，Mean frame median AE 下降；FN<=1s 明显下降，说明在极近 TTC 区域，正确风险反馈确实有价值。",
        "- Oracle 的 HR FN<=2s 没有下降，反而略高于 C0；因此固定 3x3 尺度调制并没有稳定改善完整 0-2s 高风险漏检。",
        "- Closed 的 FN<=1s 仍明显好于 C0，并保留了 Oracle 大部分极近风险收益；但 Mean frame median AE 与 MRE 相比 C0 变差，说明预测反馈会引入分布漂移。",
        "- 当前 Python 版区域风险计算和调制的额外延迟超过 5% 验收线；若继续推进，需要优化实现或简化风险统计。",
        "- 按指导文件判断：B3 最小机制在 TTC<=1s 有信号，但还没有达到“Oracle 与 Closed 均稳定提升”的标准。暂不建议直接进入 Flightmare/ROS/TensorRT。",
        "",
        "## 建议下一步",
        "",
        "- 优先分析高风险可视化，确认 Closed 是否在局部区域出现反馈偏移。",
        "- 如果继续 B3，先尝试风险平滑、置信度过滤或更保守的尺度增益，而不是增加可学习模块。",
        "- 在进入多序列验证前，建议先把完整 segment 评估作为固定回归测试。",
        "",
        "## 输出文件",
        "",
        f"- 高风险可视化目录：`{base / 'high_risk_visualizations'}`",
        f"- 高风险可视化数量：{len(visualization_rows)}",
        f"- 可视化索引：`{base / 'high_risk_visualizations' / 'visualization_index.md'}`",
        f"- 原始 B1-Full 指标：`{base / METHODS['b1']['metrics']}`",
        f"- C0 指标：`{base / METHODS['c0']['metrics']}`",
        f"- Oracle 指标：`{base / METHODS['oracle']['metrics']}`",
        f"- Closed 指标：`{base / METHODS['closed']['metrics']}`",
    ]
    (base / "b3_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--b3-dir", default="/home/hello/research_project/event+SNN+TTC/logs/ttc_TTCEF/b3")
    parser.add_argument("--test-h5", default="/home/hello/research_project/event+SNN+TTC/logs/ttc_TTCEF/b1_polarity_full_20ep/b1_full_test.h5")
    parser.add_argument("--test-clips", default="/home/hello/research_project/event+SNN+TTC/logs/ttc_TTCEF/temporal_audit/test_clips.json")
    parser.add_argument("--num-visualizations", type=int, default=20)
    parser.add_argument("--skip-visualizations", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base = Path(args.b3_dir)
    if args.skip_visualizations:
        visual_dir = base / "high_risk_visualizations"
        rows = [{"path": str(path)} for path in sorted(visual_dir.glob("*.png"))]
    else:
        rows = make_visualizations(args, base)
    write_report(base, rows)
    print(f"wrote {base / 'b3_report.md'}")
    if rows:
        print(f"wrote {len(rows)} visualizations to {base / 'high_risk_visualizations'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
