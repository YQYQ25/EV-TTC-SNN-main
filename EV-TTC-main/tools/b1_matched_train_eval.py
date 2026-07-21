#!/usr/bin/env python3
import argparse
import csv
import json
import random
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
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model"))
sys.path.insert(0, str(ROOT / "tools"))

from b1_matched_32_overfit import build_b1_inputs  # noqa: E402
from evslim import EVSlim  # noqa: E402


def model_cfg(in_channels=6):
    return SimpleNamespace(
        in_channels=in_channels,
        input=in_channels,
        enc_k_size=[3, 3, 3],
        enc_channels=[16, 16, 16],
        aspp_channels=[32, 16],
        rate=[12, 24, 36],
        dec_k_size=[3, 3, 3],
        dec_channels=[8, 8, 1],
        norm=True,
        act="ReLU",
        alpha=0.45,
    )


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def charbonnier_loss(error, alpha=0.45, mask=None):
    loss = (error**2.0 + 1e-5**2.0) ** alpha
    if mask is None:
        return loss.mean()
    mask = mask.float()
    denom = mask.sum(dim=(1, 2, 3)).clamp_min(1.0)
    return ((loss * mask).sum(dim=(1, 2, 3)) / denom).mean()


class TTCEFH5(Dataset):
    def __init__(self, path):
        self.path = str(path)
        self._h5 = None
        with h5py.File(self.path, "r") as f:
            self.length = len(f["exp_filts"])
            self.height = int(f["ttc"].shape[1])
            self.width = int(f["ttc"].shape[2])

    def __len__(self):
        return self.length

    def _open(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.path, "r", libver="latest")
        return self._h5

    def __getitem__(self, idx):
        f = self._open()
        exp = torch.from_numpy(f["exp_filts"][idx].astype(np.float32))
        ttc = torch.from_numpy(f["ttc"][idx].astype(np.float32))[None]
        mask = torch.from_numpy(f["mask"][idx].astype(np.bool_))[None]
        return exp, ttc, mask


def loader(ds, batch_size, shuffle, workers):
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
    )


def source_indices_for_split(data_root, seq_name):
    gt_path = Path(data_root) / "ttcef" / "m3ed" / f"{seq_name}.h5"
    with h5py.File(gt_path, "r") as f:
        valid = f["valid"][:]
        t = f["T"][:]
        omega = f["Omega"][:]
    if "car" in seq_name:
        keep = valid & (np.linalg.norm(t, axis=1) > 1.3) & (np.linalg.norm(omega, axis=1) < 0.18)
    else:
        keep = valid & (np.linalg.norm(t, axis=1) > 0.25) & (np.linalg.norm(omega, axis=1) < 0.18)
    return np.where(keep)[0].astype(np.int64)


def split_sequence(split_h5):
    with h5py.File(split_h5, "r") as f:
        names = [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in f["file_names"][:]]
        if len(names) != 1:
            raise ValueError(f"Expected one sequence in {split_h5}, got {names}")
        return names[0], len(f["exp_filts"])


def ensure_b1_h5(args, split_h5, out_h5):
    seq_name, expected_len = split_sequence(split_h5)
    if Path(out_h5).exists():
        with h5py.File(out_h5, "r") as f:
            if len(f["exp_filts"]) == expected_len and tuple(f["exp_filts"].shape[1:]) == (args.in_channels, 360, 360):
                return seq_name, str(out_h5)

    source_indices = source_indices_for_split(args.data_root, seq_name)
    if len(source_indices) != expected_len:
        raise ValueError(f"Source index count {len(source_indices)} != merged split length {expected_len}")

    inputs = build_b1_inputs(args, seq_name, source_indices)
    if inputs.shape[1] != args.in_channels:
        raise ValueError(f"Built {inputs.shape[1]} input channels, expected {args.in_channels}")
    out_h5 = Path(out_h5)
    out_h5.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(split_h5, "r") as f_in, h5py.File(out_h5, "w") as f_out:
        f_out.attrs["input_definition"] = args.input_definition
        f_out.attrs["source_split_file"] = str(split_h5)
        f_out.attrs["sequence"] = seq_name
        f_out.attrs["alphas"] = np.asarray(args.alphas, dtype=np.float32)
        f_out.create_dataset("file_names", data=np.asarray([seq_name], dtype=h5py.string_dtype()))
        f_out.create_dataset("source_indices", data=source_indices, dtype=np.int64)
        f_out.create_dataset(
            "exp_filts",
            data=inputs.astype(np.float16),
            dtype=np.float16,
            chunks=(1, args.in_channels, 360, 360),
            **hdf5plugin.Blosc2(cname="lz4", clevel=5, filters=hdf5plugin.Blosc.SHUFFLE),
        )
        for key in ["ttc", "mask"]:
            src = f_in[key]
            f_out.create_dataset(
                key,
                data=src[:],
                dtype=src.dtype,
                chunks=(1, *src.shape[1:]),
                **hdf5plugin.Blosc2(cname="lz4", clevel=5, filters=hdf5plugin.Blosc.SHUFFLE),
            )
    return seq_name, str(out_h5)


@torch.no_grad()
def average_loss(model, data_loader, device, alpha, max_batches=None):
    model.eval()
    losses = []
    for bi, (exp, ttc, mask) in enumerate(data_loader):
        exp = exp.to(device, non_blocking=True)
        ttc = ttc.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            pred = model(exp)
            loss = charbonnier_loss(ttc - pred, alpha=alpha, mask=mask)
        losses.append(float(loss.cpu()))
        if max_batches and bi + 1 >= max_batches:
            break
    return float(np.mean(losses))


def train(args, train_file, test_file, out_dir):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    cfg = model_cfg(args.in_channels)
    train_ds = TTCEFH5(train_file)
    test_ds = TTCEFH5(test_file)
    train_loader = loader(train_ds, args.batch_size, True, args.workers)
    val_loader = loader(test_ds, args.batch_size, False, args.workers)
    model = EVSlim(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = max(1, len(train_loader) * args.epochs)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=total_steps, pct_start=args.pct_start, cycle_momentum=False
    )
    rows = []
    best_val = float("inf")
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for exp, ttc, mask in train_loader:
            exp = exp.to(device, non_blocking=True)
            ttc = ttc.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                pred = model(exp)
                loss = charbonnier_loss(ttc - pred, alpha=cfg.alpha, mask=mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            losses.append(float(loss.detach().cpu()))

        quick_val = average_loss(model, val_loader, device, cfg.alpha, max_batches=args.quick_val_batches)
        row = {
            "epoch": epoch + 1,
            "train_loss": float(np.mean(losses)),
            "quick_val_loss": quick_val,
            "lr": sched.get_last_lr()[0],
        }
        rows.append(row)
        torch.save(
            {"model_state_dict": model.state_dict(), "cfg": vars(cfg), "epoch": epoch + 1, "row": row},
            out_dir / f"{args.checkpoint_prefix}_last.pt",
        )
        if quick_val < best_val:
            best_val = quick_val
            torch.save(
                {"model_state_dict": model.state_dict(), "cfg": vars(cfg), "epoch": epoch + 1, "row": row},
                out_dir / f"{args.checkpoint_prefix}_best.pt",
            )
    write_curve(rows, out_dir)
    return rows


def write_curve(rows, out_dir):
    with open(out_dir / "train_curve.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "quick_val_loss", "lr"])
        writer.writeheader()
        writer.writerows(rows)
    data = np.asarray([[r["epoch"], r["train_loss"], r["quick_val_loss"]] for r in rows], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(data[:, 0], data[:, 1], label="train")
    ax.plot(data[:, 0], data[:, 2], label="quick val")
    ax.set_xlabel("epoch")
    ax.set_ylabel("masked Charbonnier loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "train_curve.png", dpi=160)
    plt.close(fig)


def load_model(checkpoint, device):
    ckpt = torch.load(checkpoint, map_location="cpu")
    cfg = SimpleNamespace(**ckpt["cfg"])
    model = EVSlim(cfg)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    return model.to(device).eval(), ckpt


def init_bin(lo, hi):
    return {"lo": lo, "hi": hi, "count": 0, "sum_abs": 0.0, "sum_gt": 0.0, "sum_pred": 0.0, "fn": 0, "abs": []}


def update_bin(item, gt, pred, ae, valid):
    if not valid.any():
        return
    vals = ae[valid].astype(np.float32)
    item["abs"].append(vals)
    item["count"] += int(vals.size)
    item["sum_abs"] += float(vals.sum(dtype=np.float64))
    item["sum_gt"] += float(gt[valid].sum(dtype=np.float64))
    item["sum_pred"] += float(pred[valid].sum(dtype=np.float64))
    item["fn"] += int((pred[valid] > item["hi"]).sum())


def finish_bin(item):
    count = item["count"]
    vals = np.concatenate(item["abs"]) if item["abs"] else np.array([], dtype=np.float32)
    if count == 0:
        return {
            "range": f"{item['lo']} < TTC <= {item['hi']} seconds",
            "valid_pixel_count": 0,
            "mae": None,
            "median_ae": None,
            "false_negative_rate": None,
            "mean_gt": None,
            "mean_prediction": None,
        }
    return {
        "range": f"{item['lo']} < TTC <= {item['hi']} seconds",
        "valid_pixel_count": count,
        "mae": item["sum_abs"] / count,
        "median_ae": float(np.median(vals)) if vals.size else None,
        "false_negative_rate": item["fn"] / count,
        "mean_gt": item["sum_gt"] / count,
        "mean_prediction": item["sum_pred"] / count,
    }


def band_ranges(height):
    if height == 260:
        return [(0, 87), (87, 174), (174, 260)]
    return [(0, height // 3), (height // 3, (2 * height) // 3), ((2 * height) // 3, height)]


@torch.no_grad()
def evaluate(args, checkpoint, test_file, out_dir):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model, ckpt = load_model(checkpoint, device)
    ds = TTCEFH5(test_file)
    test_loader = loader(ds, args.eval_batch_size, False, args.workers)
    frame_medians = []
    abs_all = []
    count = 0
    sum_abs = 0.0
    sum_rel = 0.0
    re_counts = {0.10: 0, 0.50: 0, 0.80: 0}
    bands = band_ranges(ds.height)
    band_sum = [0.0, 0.0, 0.0]
    band_count = [0, 0, 0]
    high = init_bin(0.0, args.high_risk_ttc)
    bins = {"0_1s": init_bin(0.0, 1.0), "1_2s": init_bin(1.0, 2.0), "2_5s": init_bin(2.0, 5.0)}
    sample_ids = set(int(x) for x in np.linspace(0, len(ds) - 1, args.num_vis_samples, dtype=int))
    saved = {}
    vis_dir = out_dir / "test_visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)

    for batch_idx, (exp, gt, mask) in enumerate(test_loader):
        start = batch_idx * args.eval_batch_size
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            pred = model(exp.to(device, non_blocking=True)).float().cpu().numpy()[:, 0]
        exp_np = exp.numpy()
        gt_np = gt.numpy()[:, 0]
        mask_np = mask.numpy()[:, 0].astype(bool)
        for i in range(gt_np.shape[0]):
            idx = start + i
            valid = mask_np[i] & np.isfinite(gt_np[i]) & np.isfinite(pred[i])
            if valid.any():
                ae = np.abs(pred[i] - gt_np[i]).astype(np.float32)
                vals = ae[valid]
                rel = vals / np.maximum(np.abs(gt_np[i][valid]), args.rel_eps)
                frame_medians.append(float(np.median(vals)))
                abs_all.append(vals)
                count += int(vals.size)
                sum_abs += float(vals.sum(dtype=np.float64))
                sum_rel += float(rel.sum(dtype=np.float64))
                for thr in re_counts:
                    re_counts[thr] += int((rel > thr).sum())
                for bi, (r0, r1) in enumerate(bands):
                    bvalid = valid[r0:r1]
                    if bvalid.any():
                        bvals = ae[r0:r1][bvalid]
                        band_sum[bi] += float(bvals.sum(dtype=np.float64))
                        band_count[bi] += int(bvals.size)
                hmask = valid & (gt_np[i] > 0.0) & (gt_np[i] <= args.high_risk_ttc)
                update_bin(high, gt_np[i], pred[i], ae, hmask)
                for item in bins.values():
                    bmask = valid & (gt_np[i] > item["lo"]) & (gt_np[i] <= item["hi"])
                    update_bin(item, gt_np[i], pred[i], ae, bmask)
            if idx in sample_ids:
                saved[idx] = {
                    "input": exp_np[i],
                    "gt": gt_np[i],
                    "prediction": pred[i],
                    "error": np.abs(pred[i] - gt_np[i]).astype(np.float32),
                    "mask": mask_np[i],
                }
                save_visual(args, vis_dir, idx, saved[idx])

    all_abs = np.concatenate(abs_all) if abs_all else np.array([], dtype=np.float32)
    med = np.asarray(frame_medians, dtype=np.float64)
    metrics = {
        "checkpoint": str(checkpoint),
        "checkpoint_epoch": ckpt.get("epoch"),
        "checkpoint_row": ckpt.get("row"),
        "test_file": str(test_file),
        "num_test_samples": len(ds),
        "image_height": ds.height,
        "image_width": ds.width,
        "valid_pixel_count": count,
        "paper_metrics": {
            "mean_frame_median_AE": float(med.mean()),
            "std_frame_median_AE": float(med.std()),
            "MRE_percent": 100.0 * sum_rel / max(count, 1),
            "RE_10_percent": 100.0 * re_counts[0.10] / max(count, 1),
            "RE_50_percent": 100.0 * re_counts[0.50] / max(count, 1),
            "RE_80_percent": 100.0 * re_counts[0.80] / max(count, 1),
            "band_ranges": bands,
            "MAE_B1_B3": [band_sum[i] / max(band_count[i], 1) for i in range(3)],
            "band_valid_pixel_count": band_count,
        },
        "safety_metrics": {
            "pixel_MAE": sum_abs / max(count, 1),
            "pixel_median_AE": float(np.median(all_abs)),
            "high_risk_definition": f"0 < GT_TTC <= {args.high_risk_ttc} seconds",
            "high_risk_MAE": finish_bin(high)["mae"],
            "high_risk_false_negative_rate": finish_bin(high)["false_negative_rate"],
            "risk_bins": {k: finish_bin(v) for k, v in bins.items()},
        },
    }
    (out_dir / "paper_metrics.json").write_text(json.dumps(metrics, indent=2))
    write_paper_md(metrics, out_dir / "paper_metrics.md", args.paper_title)
    return metrics


def visual_channels(args):
    if hasattr(args, "visual_channel_indices") and args.visual_channel_indices:
        return args.visual_channel_indices, args.visual_channel_names
    if args.in_channels == 12:
        return [0, 2, 5, 6, 8, 11], [
            "pos scale0",
            "pos scale2",
            "pos scale5",
            "neg scale0",
            "neg scale2",
            "neg scale5",
        ]
    return list(range(6)), ["pos short", "pos mid", "pos long", "neg short", "neg mid", "neg long"]


def save_visual(args, vis_dir, idx, item):
    channel_ids, names = visual_channels(args)
    inp, gt, pred, err, mask = item["input"], item["gt"], item["prediction"], item["error"], item["mask"]
    masked_gt = np.where(mask, gt, np.nan)
    vmax = max(float(np.nanpercentile(masked_gt, 95)) if np.isfinite(masked_gt).any() else 5.0, 2.0)
    err_vmax = max(float(np.nanpercentile(np.where(mask, err, np.nan), 95)) if mask.any() else 1.0, 1e-6)
    total_panels = len(channel_ids) + 4
    ncols = int(np.ceil(total_panels / 2))
    fig = plt.figure(figsize=(3.2 * ncols, 8), constrained_layout=True)
    gs = fig.add_gridspec(2, ncols)
    for panel_idx, (channel_idx, name) in enumerate(zip(channel_ids, names)):
        ax = fig.add_subplot(gs[panel_idx // ncols, panel_idx % ncols])
        im_vmax = np.percentile(inp[channel_idx], 99.5) + 1e-6
        ax.imshow(inp[channel_idx], cmap="magma", vmin=0, vmax=im_vmax)
        ax.set_title(name)
        ax.axis("off")
    panels = [
        ("GT TTC", gt, "viridis", 0, vmax),
        ("Prediction", pred, "viridis", 0, vmax),
        ("Abs Error", err, "magma", 0, err_vmax),
        ("Mask", mask.astype(float), "gray", 0, 1),
    ]
    for p, (title, arr, cmap, vmin, vmax_) in enumerate(panels, start=len(channel_ids)):
        ax = fig.add_subplot(gs[p // ncols, p % ncols])
        ax.imshow(np.where(mask, arr, np.nan) if title != "Mask" else arr, cmap=cmap, vmin=vmin, vmax=vmax_)
        ax.set_title(title)
        ax.axis("off")
    path = vis_dir / f"sample_{idx:05d}.png"
    fig.suptitle(f"{args.method_name} test sample {idx}")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_paper_md(metrics, path, title):
    def fmt(value):
        return "N/A" if value is None else f"{value:.6f}"

    p = metrics["paper_metrics"]
    s = metrics["safety_metrics"]
    bands = p["MAE_B1_B3"]
    lines = [
        f"# {title}",
        "",
        "## 数据集",
        f"- 测试文件：`{metrics['test_file']}`",
        f"- checkpoint：`{metrics['checkpoint']}`",
        f"- 测试样本数：`{metrics['num_test_samples']}`",
        f"- 有效像素数：`{metrics['valid_pixel_count']}`",
        "",
        "## EV-TTC 论文同口径指标",
        f"- 逐帧中位绝对误差均值：`{p['mean_frame_median_AE']:.6f} ± {p['std_frame_median_AE']:.6f}`",
        f"- 平均相对误差 MRE(%)：`{p['MRE_percent']:.6f}`",
        f"- 相对误差离群率 RE[10%,50%,80%]：`{p['RE_10_percent']:.6f} / {p['RE_50_percent']:.6f} / {p['RE_80_percent']:.6f}`",
        f"- 三段高度 Band MAE(B1-B3)：`{bands[0]:.6f} / {bands[1]:.6f} / {bands[2]:.6f}`",
        "",
        "## 安全导向指标",
        f"- 像素级 MAE：`{s['pixel_MAE']:.6f}`",
        f"- 像素级中位 AE：`{s['pixel_median_AE']:.6f}`",
        f"- 高风险 MAE（GT TTC <= 2s）：`{fmt(s['high_risk_MAE'])}`",
        f"- 高风险漏检率（GT TTC <= 2s 且预测 > 2s）：`{fmt(s['high_risk_false_negative_rate'])}`",
        "",
        "| 风险区间 | 有效像素数 | MAE | 中位 AE | 漏检率 | GT 均值 | 预测均值 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key in ["0_1s", "1_2s", "2_5s"]:
        item = s["risk_bins"][key]
        lines.append(
            f"| {item['range']} | {item['valid_pixel_count']} | {fmt(item['mae'])} | {fmt(item['median_ae'])} | "
            f"{fmt(item['false_negative_rate'])} | {fmt(item['mean_gt'])} | {fmt(item['mean_prediction'])} |"
        )
    path.write_text("\n".join(lines) + "\n")


def format_method(metrics, name, epochs, channels):
    def fmt(value):
        return "N/A" if value is None else f"{value:.6f}"

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
        fmt(s["high_risk_MAE"]),
        fmt(s["high_risk_false_negative_rate"]),
        fmt(s["risk_bins"]["0_1s"]["false_negative_rate"]),
    ]


def write_comparison(args, current_metrics, out_dir):
    b0_path = Path(args.b0_dir) / "paper_metrics.json"
    b0 = json.loads(b0_path.read_text())
    rows = [format_method(b0, "B0 signed IIR", 20, 6)]

    matched_path = Path(args.b1_matched_dir) / "paper_metrics.json"
    current_metrics_path = out_dir / "paper_metrics.json"
    if matched_path.exists() and matched_path.resolve() != current_metrics_path.resolve():
        matched = json.loads(matched_path.read_text())
        rows.append(format_method(matched, "B1-Matched polarity split", 20, 6))

    rows.append(format_method(current_metrics, args.method_name, args.epochs, args.in_channels))
    header = [
        "方法", "Epochs", "输入通道数", "逐帧中位 AE 均值", "MRE(%)", "RE[10/50/80]",
        "Band MAE(B1-B3)", "像素 MAE", "高风险 MAE<=2s", "高风险漏检<=2s", "漏检<=1s",
    ]
    comparison_out = Path(args.comparison_out) if args.comparison_out else out_dir / "b1_comparison_report.md"
    lines = [
        "# B1 极性分离实验对比报告",
        "",
        f"- B0 指标：`{b0_path}`",
        f"- 当前指标：`{current_metrics_path}`",
        "",
        "|" + "|".join(header) + "|",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append("|" + "|".join(row) + "|")
    lines.extend(
        [
            "",
            "## B1 产物",
            f"- 训练曲线：`{out_dir / 'train_curve.png'}`",
            f"- last checkpoint: `{out_dir / f'{args.checkpoint_prefix}_last.pt'}`",
            f"- best checkpoint: `{out_dir / f'{args.checkpoint_prefix}_best.pt'}`",
            f"- 测试可视化：`{out_dir / 'test_visualizations'}`",
        ]
    )
    comparison_out.write_text("\n".join(lines) + "\n")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train-file", default="/home/hello/research_project/event+SNN+TTC/ttcef_merged_min/train.h5")
    p.add_argument("--test-file", default="/home/hello/research_project/event+SNN+TTC/ttcef_merged_min/test.h5")
    p.add_argument("--data-root", default="/home/hello/research_project/event+SNN+TTC/data_m3ed_min")
    p.add_argument("--b0-dir", default="/home/hello/research_project/event+SNN+TTC/logs/ttc_TTCEF/b0_signed_baseline_20ep")
    p.add_argument("--b1-matched-dir", default="/home/hello/research_project/event+SNN+TTC/logs/ttc_TTCEF/b1_polarity_matched_20ep")
    p.add_argument("--out-dir", default="/home/hello/research_project/event+SNN+TTC/logs/ttc_TTCEF/b1_polarity_matched_20ep")
    p.add_argument("--comparison-out", default=None)
    p.add_argument("--method-name", default="B1-Matched polarity split")
    p.add_argument("--input-definition", default="B1-Matched polarity split: pos short/mid/long, neg short/mid/long")
    p.add_argument("--paper-title", default="B1-Matched Paper-style Metrics")
    p.add_argument("--checkpoint-prefix", default="b1_polarity_matched")
    p.add_argument("--b1-train-name", default="b1_matched_train.h5")
    p.add_argument("--b1-test-name", default="b1_matched_test.h5")
    p.add_argument("--in-channels", type=int, default=6)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--eval-batch-size", type=int, default=16)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--pct-start", type=float, default=0.1)
    p.add_argument("--quick-val-batches", type=int, default=16)
    p.add_argument("--dt-ms", type=float, default=0.2)
    p.add_argument("--output-time-ms", type=float, default=7.0)
    p.add_argument("--alphas", type=float, nargs="+", default=[0.12, 0.03, 0.0045])
    p.add_argument("--event-batch-ms", type=int, default=1000)
    p.add_argument("--rel-eps", type=float, default=1e-6)
    p.add_argument("--high-risk-ttc", type=float, default=2.0)
    p.add_argument("--num-vis-samples", type=int, default=20)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    seed_all(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run_config.json").write_text(json.dumps(vars(args), indent=2))

    train_seq, b1_train = ensure_b1_h5(args, args.train_file, out_dir / args.b1_train_name)
    test_seq, b1_test = ensure_b1_h5(args, args.test_file, out_dir / args.b1_test_name)
    rows = train(args, b1_train, b1_test, out_dir)
    metrics = evaluate(args, out_dir / f"{args.checkpoint_prefix}_best.pt", b1_test, out_dir)
    summary = {
        "method": args.method_name,
        "train_sequence": train_seq,
        "test_sequence": test_seq,
        "train_file": args.train_file,
        "test_file": args.test_file,
        "b1_train_file": b1_train,
        "b1_test_file": b1_test,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "in_channels": args.in_channels,
        "input_definition": args.input_definition,
        "final_epoch": rows[-1],
        "best_checkpoint": str(out_dir / f"{args.checkpoint_prefix}_best.pt"),
        "last_checkpoint": str(out_dir / f"{args.checkpoint_prefix}_last.pt"),
        "metrics": metrics,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    write_comparison(args, metrics, out_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
