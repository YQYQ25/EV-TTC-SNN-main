#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import h5py
import hdf5plugin  # noqa: F401
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model"))
from evslim import EVSlim  # noqa: E402


def load_model(path, device):
    ckpt = torch.load(path, map_location="cpu")
    cfg = SimpleNamespace(**ckpt["cfg"])
    model = EVSlim(cfg)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    return model.to(device).eval()


def select_high_risk_samples(test_file, num_samples):
    counts = []
    with h5py.File(test_file, "r") as f:
        for i in range(len(f["ttc"])):
            gt = f["ttc"][i]
            mask = f["mask"][i].astype(bool)
            high = mask & np.isfinite(gt) & (gt > 0.0) & (gt <= 1.0)
            counts.append((int(high.sum()), i))
    counts.sort(reverse=True)
    return [idx for count, idx in counts[:num_samples]], counts[:num_samples]


@torch.no_grad()
def predict_one(model, exp, device):
    x = torch.from_numpy(exp.astype(np.float32))[None].to(device)
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
        pred = model(x).float().cpu().numpy()[0, 0]
    return pred


def save_panel(path, idx, gt, mask, preds):
    mask = mask.astype(bool)
    vmax = max(float(np.nanpercentile(np.where(mask, gt, np.nan), 95)), 2.0)
    err_vmax = max(
        float(np.nanpercentile(np.where(mask, np.abs(preds["B1-Full"] - gt), np.nan), 95)),
        1e-6,
    )
    fig = plt.figure(figsize=(18, 9), constrained_layout=True)
    gs = fig.add_gridspec(2, 4)
    panels = [
        ("GT TTC", gt, "viridis", 0, vmax),
        ("B0 Prediction", preds["B0"], "viridis", 0, vmax),
        ("B1-Matched Prediction", preds["B1-Matched"], "viridis", 0, vmax),
        ("B1-Full Prediction", preds["B1-Full"], "viridis", 0, vmax),
        ("B0 Error", np.abs(preds["B0"] - gt), "magma", 0, err_vmax),
        ("B1-Matched Error", np.abs(preds["B1-Matched"] - gt), "magma", 0, err_vmax),
        ("B1-Full Error", np.abs(preds["B1-Full"] - gt), "magma", 0, err_vmax),
        ("Mask", mask.astype(float), "gray", 0, 1),
    ]
    for p, (title, arr, cmap, vmin, vmax_) in enumerate(panels):
        ax = fig.add_subplot(gs[p // 4, p % 4])
        ax.imshow(np.where(mask, arr, np.nan) if title != "Mask" else arr, cmap=cmap, vmin=vmin, vmax=vmax_)
        ax.set_title(title)
        ax.axis("off")
    fig.suptitle(f"高风险测试样本 {idx}")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def summarize_selected(selected, stats, metrics):
    lines = [
        "# B1 高风险区域深化分析",
        "",
        "## 样本筛选",
        "- 从 test set 中按 `0 < TTC <= 1.0s` 有效像素数量从高到低选取 20 个样本。",
        f"- 选中样本索引：`{selected}`",
        "",
        "## 全局指标观察",
        "- B1-Matched 是 6 通道公平对比；B1-Full 是 12 通道 upper bound，不是严格公平对比。",
        "- B1-Matched 改善了整体回归指标，但高风险漏检变差。",
        "- B1-Full 基本修复 B1-Matched 的 `<=2s` 高风险漏检问题，并大幅改善 `<=1s` 漏检。",
        "",
        "| 方法 | 逐帧中位 AE 均值 | MRE(%) | Band MAE(B1-B3) | 高风险漏检<=2s | 漏检<=1s |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in ["B0", "B1-Matched", "B1-Full"]:
        m = metrics[name]
        p = m["paper_metrics"]
        s = m["safety_metrics"]
        b = p["MAE_B1_B3"]
        lines.append(
            f"| {name} | {p['mean_frame_median_AE']:.6f} | {p['MRE_percent']:.6f} | "
            f"{b[0]:.6f}/{b[1]:.6f}/{b[2]:.6f} | "
            f"{s['high_risk_false_negative_rate']:.6f} | {s['risk_bins']['0_1s']['false_negative_rate']:.6f} |"
        )
    lines.extend([
        "",
        "## 20 个高风险样本上的局部统计",
        "",
        "| 方法 | <=1s 像素 MAE | <=1s 漏检率 | <=2s 像素 MAE | <=2s 漏检率 |",
        "|---|---:|---:|---:|---:|",
    ])
    for name, item in stats.items():
        lines.append(
            f"| {name} | {item['mae_1s']:.6f} | {item['fn_1s']:.6f} | {item['mae_2s']:.6f} | {item['fn_2s']:.6f} |"
        )
    lines.extend([
        "",
        "## 结论",
        "- B1-Full 在 `TTC<=1s` 区域的局部 MAE 和漏检率均明显优于 B1-Matched，说明完整 6 尺度极性分离确实补回了部分近风险信息。",
        "- B1-Matched 整体指标更好但高风险 FN 变差，主要表现为极近 TTC 区域预测偏大，导致 `prediction > threshold` 的漏检变多。",
        "- B1-Full 的改善主要体现在中部/下部 band 和极近 TTC 安全指标；上方 band 仍不是最优，后续时间信息实验需要重点观察上方 band。",
        "",
        "## 可视化路径",
    ])
    return lines


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="/home/hello/research_project/event+SNN+TTC/logs/ttc_TTCEF/b1_deep_analysis")
    p.add_argument("--test-file", default="/home/hello/research_project/event+SNN+TTC/ttcef_merged_min/test.h5")
    p.add_argument("--b0-test-file", default="/home/hello/research_project/event+SNN+TTC/ttcef_merged_min/test.h5")
    p.add_argument("--b1m-test-file", default="/home/hello/research_project/event+SNN+TTC/logs/ttc_TTCEF/b1_polarity_matched_20ep/b1_matched_test.h5")
    p.add_argument("--b1f-test-file", default="/home/hello/research_project/event+SNN+TTC/logs/ttc_TTCEF/b1_polarity_full_20ep/b1_full_test.h5")
    p.add_argument("--b0-ckpt", default="/home/hello/research_project/event+SNN+TTC/logs/ttc_TTCEF/b0_signed_baseline_20ep/b0_signed_baseline_best.pt")
    p.add_argument("--b1m-ckpt", default="/home/hello/research_project/event+SNN+TTC/logs/ttc_TTCEF/b1_polarity_matched_20ep/b1_polarity_matched_best.pt")
    p.add_argument("--b1f-ckpt", default="/home/hello/research_project/event+SNN+TTC/logs/ttc_TTCEF/b1_polarity_full_20ep/b1_polarity_full_best.pt")
    p.add_argument("--num-samples", type=int, default=20)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    vis_dir = out_dir / "high_risk_visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    models = {
        "B0": load_model(args.b0_ckpt, device),
        "B1-Matched": load_model(args.b1m_ckpt, device),
        "B1-Full": load_model(args.b1f_ckpt, device),
    }
    selected, high_counts = select_high_risk_samples(args.test_file, args.num_samples)
    files = {
        "B0": h5py.File(args.b0_test_file, "r"),
        "B1-Matched": h5py.File(args.b1m_test_file, "r"),
        "B1-Full": h5py.File(args.b1f_test_file, "r"),
    }
    stats = {k: {"ae_1": 0.0, "n_1": 0, "fn_1": 0, "ae_2": 0.0, "n_2": 0, "fn_2": 0} for k in models}
    image_paths = []
    try:
        for idx in selected:
            gt = files["B0"]["ttc"][idx].astype(np.float32)
            mask = files["B0"]["mask"][idx].astype(bool)
            preds = {}
            for name in models:
                exp = files[name]["exp_filts"][idx].astype(np.float32)
                pred = predict_one(models[name], exp, device)
                preds[name] = pred
                valid_1 = mask & np.isfinite(gt) & (gt > 0.0) & (gt <= 1.0)
                valid_2 = mask & np.isfinite(gt) & (gt > 0.0) & (gt <= 2.0)
                if valid_1.any():
                    ae = np.abs(pred[valid_1] - gt[valid_1])
                    stats[name]["ae_1"] += float(ae.sum(dtype=np.float64))
                    stats[name]["n_1"] += int(ae.size)
                    stats[name]["fn_1"] += int((pred[valid_1] > 1.0).sum())
                if valid_2.any():
                    ae = np.abs(pred[valid_2] - gt[valid_2])
                    stats[name]["ae_2"] += float(ae.sum(dtype=np.float64))
                    stats[name]["n_2"] += int(ae.size)
                    stats[name]["fn_2"] += int((pred[valid_2] > 2.0).sum())
            path = vis_dir / f"high_risk_sample_{idx:05d}.png"
            save_panel(path, idx, gt, mask, preds)
            image_paths.append(path)
    finally:
        for f in files.values():
            f.close()

    stats_out = {}
    for name, item in stats.items():
        stats_out[name] = {
            "mae_1s": item["ae_1"] / max(item["n_1"], 1),
            "fn_1s": item["fn_1"] / max(item["n_1"], 1),
            "count_1s": item["n_1"],
            "mae_2s": item["ae_2"] / max(item["n_2"], 1),
            "fn_2s": item["fn_2"] / max(item["n_2"], 1),
            "count_2s": item["n_2"],
        }
    metrics = {
        "B0": json.loads(Path("/home/hello/research_project/event+SNN+TTC/logs/ttc_TTCEF/b0_signed_baseline_20ep/paper_metrics.json").read_text()),
        "B1-Matched": json.loads(Path("/home/hello/research_project/event+SNN+TTC/logs/ttc_TTCEF/b1_polarity_matched_20ep/paper_metrics.json").read_text()),
        "B1-Full": json.loads(Path("/home/hello/research_project/event+SNN+TTC/logs/ttc_TTCEF/b1_polarity_full_20ep/paper_metrics.json").read_text()),
    }
    lines = summarize_selected(selected, stats_out, metrics)
    for path in image_paths:
        lines.append(f"- `{path}`")
    (out_dir / "high_risk_analysis.md").write_text("\n".join(lines) + "\n")
    (out_dir / "high_risk_analysis.json").write_text(json.dumps({
        "selected_indices": selected,
        "high_risk_pixel_counts": high_counts,
        "selected_stats": stats_out,
        "visualizations": [str(p) for p in image_paths],
    }, indent=2))
    print(json.dumps(stats_out, indent=2))


if __name__ == "__main__":
    main()
