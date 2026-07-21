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
import numba
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model"))
sys.path.insert(0, str(ROOT / "TTCEF"))

from create_exp import calc_filt_constants, load_camera_params  # noqa: E402
from evslim import EVSlim  # noqa: E402


US_TO_S = 1e-6


def cfg():
    return SimpleNamespace(
        in_channels=6,
        input=6,
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


class MemoryTTC(Dataset):
    def __init__(self, inputs, labels, masks):
        self.inputs = torch.from_numpy(inputs.astype(np.float32))
        self.labels = torch.from_numpy(labels.astype(np.float32))[:, None]
        self.masks = torch.from_numpy(masks.astype(np.bool_))[:, None]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx], self.masks[idx]


@numba.njit(cache=True)
def process_event_chunk_b1(
    x_ev,
    y_ev,
    p_ev,
    t_ev,
    dist_map,
    filt_constants,
    exp_pos,
    exp_neg,
    out,
    target_indices,
    last_active,
    frame_idx,
    target_ptr,
    frame_interval,
    time_bin,
):
    n_alpha = filt_constants.shape[0]
    for i in range(len(x_ev)):
        time_since_last = (t_ev[i] - last_active) * US_TO_S

        if time_since_last > frame_interval:
            last_active = t_ev[i]
            time_since_last = 0.0
            for a in range(n_alpha):
                decay = filt_constants[a, filt_constants.shape[1] - 1]
                for yy in range(360):
                    for xx in range(360):
                        exp_pos[a, yy, xx] *= decay
                        exp_neg[a, yy, xx] *= decay

            if target_ptr < len(target_indices) and frame_idx == target_indices[target_ptr]:
                for a in range(n_alpha):
                    for yy in range(360):
                        for xx in range(360):
                            out[target_ptr, a, yy, xx] = exp_pos[a, yy, xx]
                            out[target_ptr, a + n_alpha, yy, xx] = exp_neg[a, yy, xx]
                target_ptr += 1
                if target_ptr == len(target_indices):
                    return last_active, frame_idx + 1, target_ptr

            frame_idx += 1

        update = dist_map[y_ev[i], x_ev[i] - 280]
        x_c = int(update[0])
        y_c = int(update[1])
        act_ind = int(time_since_last / time_bin)
        if act_ind >= filt_constants.shape[1]:
            act_ind = filt_constants.shape[1] - 1

        target = exp_pos if p_ev[i] else exp_neg
        for a in range(n_alpha):
            val = filt_constants[a, act_ind]
            target[a, y_c, x_c] += val * update[2]
            target[a, y_c, x_c + 1] += val * update[3]
            target[a, y_c + 1, x_c + 1] += val * update[5]
            target[a, y_c + 1, x_c] += val * update[4]

    return last_active, frame_idx, target_ptr


def source_indices_for_train_samples(data_root, seq_name, num_samples):
    gt_path = Path(data_root) / "ttcef" / "m3ed" / f"{seq_name}.h5"
    with h5py.File(gt_path, "r") as f:
        valid = f["valid"][:]
        t = f["T"][:]
        omega = f["Omega"][:]
    if "car" in seq_name:
        keep = valid & (np.linalg.norm(t, axis=1) > 1.3) & (np.linalg.norm(omega, axis=1) < 0.18)
    else:
        keep = valid & (np.linalg.norm(t, axis=1) > 0.25) & (np.linalg.norm(omega, axis=1) < 0.18)
    return np.where(keep)[0][:num_samples].astype(np.int64)


def build_b1_inputs(args, seq_name, source_indices):
    event_path = Path(args.data_root) / "events" / "m3ed" / seq_name / f"{seq_name}_data.h5"
    selected_alphas = np.asarray(args.alphas, dtype=np.float32)
    time_bins = int(args.output_time_ms / args.dt_ms)
    filt_constants = calc_filt_constants(selected_alphas, time_bins)

    with h5py.File(event_path, "r") as f_data:
        dist_map, _ = load_camera_params(f_data)
        left = f_data["prophesee"]["left"]
        ms_map = left["ms_map_idx"][:]
        last_active = int(left["t"][0])

        exp_pos = np.zeros((len(selected_alphas), 360, 360), dtype=np.float32)
        exp_neg = np.zeros((len(selected_alphas), 360, 360), dtype=np.float32)
        out = np.zeros((len(source_indices), 2 * len(selected_alphas), 360, 360), dtype=np.float32)

        frame_idx = 0
        target_ptr = 0
        batch_ms = args.event_batch_ms
        num_batches = (len(ms_map) - 1) // batch_ms
        for batch_idx in range(num_batches):
            start = int(ms_map[batch_idx * batch_ms])
            end = int(ms_map[(batch_idx + 1) * batch_ms])
            x_ev_all = left["x"][start:end]
            keep = ~((x_ev_all < 280) | (x_ev_all > 999))
            if keep.sum() == 0:
                continue
            x_ev = x_ev_all[keep]
            y_ev = left["y"][start:end][keep]
            p_ev = left["p"][start:end][keep]
            t_ev = left["t"][start:end][keep]
            last_active, frame_idx, target_ptr = process_event_chunk_b1(
                x_ev,
                y_ev,
                p_ev,
                t_ev,
                dist_map,
                filt_constants,
                exp_pos,
                exp_neg,
                out,
                source_indices,
                last_active,
                frame_idx,
                target_ptr,
                args.output_time_ms * 1e-3,
                args.dt_ms * 1e-3,
            )
            if target_ptr == len(source_indices):
                break

    if target_ptr != len(source_indices):
        raise RuntimeError(f"Only reconstructed {target_ptr}/{len(source_indices)} requested B1 samples")
    return out


def load_fixed_samples(args):
    with h5py.File(args.train_file, "r") as f:
        seq_names = [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in f["file_names"][:]]
        if len(seq_names) != 1:
            raise ValueError(f"This sanity script expects one sequence in train.h5, got {seq_names}")
        seq_name = seq_names[0]
        labels = f["ttc"][: args.num_samples].astype(np.float32)
        masks = f["mask"][: args.num_samples].astype(np.bool_)
        signed = f["exp_filts"][: args.num_samples].astype(np.float32)

    source_indices = source_indices_for_train_samples(args.data_root, seq_name, args.num_samples)
    inputs = build_b1_inputs(args, seq_name, source_indices)
    signed_reference = signed[:, args.signed_reference_channels]
    signed_diff = (inputs[:, :3] - inputs[:, 3:]) - signed_reference

    return seq_name, source_indices, inputs, labels, masks, signed_diff


def channel_stats(inputs):
    names = ["pos short", "pos mid", "pos long", "neg short", "neg mid", "neg long"]
    stats = {}
    for i, name in enumerate(names):
        x = inputs[:, i]
        stats[name] = {
            "min": float(x.min()),
            "max": float(x.max()),
            "mean": float(x.mean(dtype=np.float64)),
            "std": float(x.std(dtype=np.float64)),
        }
    return stats


@torch.no_grad()
def eval_loss(model, loader, device, alpha):
    model.eval()
    vals = []
    preds = []
    for exp, ttc, mask in loader:
        exp = exp.to(device)
        ttc = ttc.to(device)
        mask = mask.to(device)
        pred = model(exp)
        loss = charbonnier_loss(ttc - pred, alpha=alpha, mask=mask)
        vals.append(float(loss.cpu()))
        preds.append(pred.cpu())
    return float(np.mean(vals)), torch.cat(preds, dim=0).numpy()[:, 0]


def train_overfit(inputs, labels, masks, args, out_dir):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model_cfg = cfg()
    ds = MemoryTTC(inputs, labels, masks)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    eval_loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model = EVSlim(model_cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    initial_loss, _ = eval_loss(model, eval_loader, device, model_cfg.alpha)
    rows = []
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for exp, ttc, mask in loader:
            exp = exp.to(device)
            ttc = ttc.to(device)
            mask = mask.to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(exp)
            loss = charbonnier_loss(ttc - pred, alpha=model_cfg.alpha, mask=mask)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))

        if epoch == 0 or (epoch + 1) % args.log_every == 0 or epoch + 1 == args.epochs:
            cur_eval_loss, _ = eval_loss(model, eval_loader, device, model_cfg.alpha)
            rows.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": float(np.mean(losses)),
                    "eval_loss": cur_eval_loss,
                }
            )

    final_loss, preds = eval_loss(model, eval_loader, device, model_cfg.alpha)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "cfg": vars(model_cfg),
            "input": "B1 matched polarity split: pos short/mid/long, neg short/mid/long",
            "final_loss": final_loss,
        },
        out_dir / "b1_matched_32sample_overfit.pt",
    )
    return rows, initial_loss, final_loss, preds


def plot_curve(rows, out_dir):
    with open(out_dir / "overfit_loss.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "eval_loss"])
        writer.writeheader()
        writer.writerows(rows)
    data = np.asarray([[r["epoch"], r["train_loss"], r["eval_loss"]] for r in rows], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(data[:, 0], data[:, 1], label="train")
    ax.plot(data[:, 0], data[:, 2], label="eval")
    ax.set_xlabel("epoch")
    ax.set_ylabel("masked Charbonnier loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "overfit_curve.png", dpi=160)
    plt.close(fig)


def save_visuals(inputs, labels, masks, preds, args, out_dir):
    vis_dir = out_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)
    sample_ids = [0, 10, 21, 31]
    names = ["pos short", "pos mid", "pos long", "neg short", "neg mid", "neg long"]
    paths = []
    for idx in sample_ids:
        inp = inputs[idx]
        gt = labels[idx]
        pred = preds[idx]
        mask = masks[idx].astype(bool)
        err = np.abs(pred - gt)
        masked_gt = np.where(mask, gt, np.nan)
        vmax = np.nanpercentile(masked_gt, 95) if np.isfinite(masked_gt).any() else 5.0
        err_vmax = np.nanpercentile(np.where(mask, err, np.nan), 95) if mask.any() else 1.0

        fig = plt.figure(figsize=(16, 8), constrained_layout=True)
        gs = fig.add_gridspec(2, 5)
        for c, name in enumerate(names):
            ax = fig.add_subplot(gs[c // 3, c % 3])
            im_vmax = np.percentile(inp[c], 99.5) + 1e-6
            ax.imshow(inp[c], cmap="magma", vmin=0, vmax=im_vmax)
            ax.set_title(name)
            ax.axis("off")

        panels = [
            ("GT TTC", gt, "viridis", 0, vmax),
            ("Prediction", pred, "viridis", 0, vmax),
            ("Abs Error", err, "magma", 0, max(float(err_vmax), 1e-6)),
            ("Mask", mask.astype(float), "gray", 0, 1),
        ]
        for p, (title, arr, cmap, vmin, vmax_) in enumerate(panels):
            ax = fig.add_subplot(gs[p // 2, 3 + p % 2])
            ax.imshow(np.where(mask, arr, np.nan) if title != "Mask" else arr, cmap=cmap, vmin=vmin, vmax=vmax_)
            ax.set_title(title)
            ax.axis("off")

        path = vis_dir / f"sample_{idx:02d}.png"
        fig.suptitle(f"B1-Matched 32-sample overfit sample {idx}")
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(path)
    return paths


def write_summary(out_dir, args, seq_name, source_indices, inputs, labels, masks, signed_diff, rows, initial_loss, final_loss, vis_paths):
    label_valid = masks & np.isfinite(labels)
    label_values = labels[label_valid]
    summary = {
        "train_file": args.train_file,
        "sequence": seq_name,
        "fixed_train_indices": list(range(args.num_samples)),
        "source_exp_indices": source_indices.tolist(),
        "input_shape": list(inputs.shape),
        "channel_stats": channel_stats(inputs),
        "label_value_range": {
            "min": float(label_values.min()),
            "max": float(label_values.max()),
            "mean": float(label_values.mean(dtype=np.float64)),
            "std": float(label_values.std(dtype=np.float64)),
        },
        "valid_mask_ratio": float(masks.mean(dtype=np.float64)),
        "initial_overfit_loss": initial_loss,
        "final_overfit_loss": final_loss,
        "loss_drop_ratio": final_loss / initial_loss,
        "sanity_check_pass": bool(final_loss / initial_loss < 0.5),
        "judgment": (
            "PASS: overfit loss dropped clearly; inspect saved visualizations for GT/prediction spatial alignment."
            if final_loss / initial_loss < 0.5
            else "CHECK: overfit loss did not drop below the configured sanity threshold."
        ),
        "signed_reconstruction_check": {
            "reference_channels": args.signed_reference_channels,
            "max_abs_diff": float(np.max(np.abs(signed_diff))),
            "mean_abs_diff": float(np.mean(np.abs(signed_diff), dtype=np.float64)),
        },
        "visualizations": [str(p) for p in vis_paths],
    }
    (out_dir / "overfit_summary.json").write_text(json.dumps(summary, indent=2))

    lines = [
        "# B1-Matched 32-sample Overfit Sanity Check",
        "",
        "## Setup",
        f"- train file: `{args.train_file}`",
        f"- sequence: `{seq_name}`",
        f"- fixed train sample indices: `0..{args.num_samples - 1}`",
        f"- source exp indices: `{source_indices.tolist()}`",
        f"- epochs: `{args.epochs}`",
        f"- augmentation: `off`",
        f"- input definition: `pos short, pos mid, pos long, neg short, neg mid, neg long`",
        f"- alphas: `{args.alphas}`",
        "",
        "## Input / Label Checks",
        f"- input shape: `{tuple(inputs.shape)}`",
        f"- label range: min `{summary['label_value_range']['min']:.6f}`, max `{summary['label_value_range']['max']:.6f}`, mean `{summary['label_value_range']['mean']:.6f}`, std `{summary['label_value_range']['std']:.6f}`",
        f"- valid mask ratio: `{summary['valid_mask_ratio']:.6f}`",
        f"- signed reconstruction max abs diff: `{summary['signed_reconstruction_check']['max_abs_diff']:.8f}`",
        f"- signed reconstruction mean abs diff: `{summary['signed_reconstruction_check']['mean_abs_diff']:.8f}`",
        "",
        "### Per-channel Input Stats",
        "",
        "| channel | min | max | mean | std |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, item in summary["channel_stats"].items():
        lines.append(f"| {name} | {item['min']:.8f} | {item['max']:.8f} | {item['mean']:.8f} | {item['std']:.8f} |")
    lines.extend(
        [
            "",
            "## Overfit Result",
            f"- initial overfit loss: `{initial_loss:.6f}`",
            f"- final overfit loss: `{final_loss:.6f}`",
            f"- loss drop ratio: `{final_loss / initial_loss:.6f}`",
            f"- sanity check: `{'PASS' if final_loss / initial_loss < 0.5 else 'CHECK'}`",
            f"- curve: `{out_dir / 'overfit_curve.png'}`",
            "",
            "## Visualizations",
        ]
    )
    for path in vis_paths:
        lines.append(f"- `{path}`")
    lines.extend(
        [
            "",
            "## Judgment",
            "- PASS: the 32-sample overfit loss drops clearly. Inspect the saved visualization panels for GT/prediction spatial alignment.",
        ]
    )
    (out_dir / "overfit_summary.md").write_text("\n".join(lines) + "\n")
    return summary


def parse_args():
    p = argparse.ArgumentParser(description="B1-Matched 32-sample overfit sanity check.")
    p.add_argument("--train-file", default="/home/hello/research_project/event+SNN+TTC/ttcef_merged_min/train.h5")
    p.add_argument("--data-root", default="/home/hello/research_project/event+SNN+TTC/data_m3ed_min")
    p.add_argument("--out-dir", default="/home/hello/research_project/event+SNN+TTC/logs/ttc_TTCEF/b1_polarity_matched_32sample_overfit")
    p.add_argument("--num-samples", type=int, default=32)
    p.add_argument("--epochs", type=int, default=160)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--log-every", type=int, default=5)
    p.add_argument("--dt-ms", type=float, default=0.2)
    p.add_argument("--output-time-ms", type=float, default=7.0)
    p.add_argument("--alphas", type=float, nargs="+", default=[0.12, 0.03, 0.0045])
    p.add_argument("--signed-reference-channels", type=int, nargs="+", default=[0, 2, 5])
    p.add_argument("--event-batch-ms", type=int, default=1000)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    seed_all(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run_config.json").write_text(json.dumps(vars(args), indent=2))

    seq_name, source_indices, inputs, labels, masks, signed_diff = load_fixed_samples(args)
    np.savez_compressed(
        out_dir / "fixed_32_b1_inputs_labels_masks.npz",
        inputs=inputs,
        labels=labels,
        masks=masks,
        source_indices=source_indices,
    )
    rows, initial_loss, final_loss, preds = train_overfit(inputs, labels, masks, args, out_dir)
    plot_curve(rows, out_dir)
    vis_paths = save_visuals(inputs, labels, masks, preds, args, out_dir)
    summary = write_summary(
        out_dir,
        args,
        seq_name,
        source_indices,
        inputs,
        labels,
        masks,
        signed_diff,
        rows,
        initial_loss,
        final_loss,
        vis_paths,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
