#!/usr/bin/env python3
"""LIF-EV-FlowNet-TTC 在真实S2 N=10k数据上的32-Block过拟合实验。

本脚本严格用于小样本可学习性验证：只使用一个连续32-block片段，
不做正式泛化测试，不扩展到其他空间方案或事件数。
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

import h5py
import hdf5plugin  # noqa: F401
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SNN_ROOT = PROJECT_ROOT / "EV-TTC-SNN-main"
if str(SNN_ROOT) not in sys.path:
    sys.path.insert(0, str(SNN_ROOT))

from snn_ttc.losses import (  # noqa: E402
    EVTTC_CHARBONNIER_ALPHA,
    EVTTC_CHARBONNIER_EPS,
    charbonnier,
    masked_charbonnier_per_sample,
)
from snn_ttc.models import LIFEVFlowNetTTC, LIFEVFlowNetTTCConfig  # noqa: E402
from snn_ttc.tools.audit_real_s2_lif_activity import (  # noqa: E402
    LAYER_NAMES,
    choose_continuous_segment,
    fmt,
    markdown_table,
    tensor_stats,
)


DEFAULT_H5 = (
    SNN_ROOT
    / "debug_sets"
    / "[5]car_spatial_ablation"
    / "independent_fixed_n"
    / "N10000_S2_center256_to128.h5"
)
DEFAULT_OUT_DIR = SNN_ROOT / "debug_sets" / "lif_evflownet_ttc_s2_n10k_overfit32"
ANCHORS = {
    "A_start": (0, 0),
    "B_middle": (16, 5),
    "C_end": (31, 9),
}
VIS_EPOCHS_BASE = {0, 1, 5, 10, 20, 50, 100, 200, 300}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_selected_data(h5_path: Path, rows: np.ndarray, steps_per_block: int) -> dict[str, np.ndarray]:
    """读取32个连续block需要的全部数组，并保留batch维相关元数据。"""

    with h5py.File(h5_path, "r") as f:
        event = f["event_cnt"][rows].astype(np.float32)
        target = f["signed_inverse_ttc"][rows].astype(np.float32)
        mask = f["valid_ttc_mask"][rows].astype(bool)
        supervise = f["supervise_valid"][rows].astype(bool)
        fields = {}
        for key in [
            "block_index",
            "step_index",
            "step_in_block",
            "t_start",
            "t_end",
            "raw_event_start_idx",
            "raw_event_end_idx",
            "event_dt",
        ]:
            if key in f:
                fields[key] = f[key][rows]
        attrs = {key: str(value) for key, value in f.attrs.items()}
    num_blocks = len(rows) // steps_per_block
    return {
        "event": event.reshape(num_blocks, steps_per_block, 2, 128, 128),
        "target": target.reshape(num_blocks, steps_per_block, 128, 128),
        "mask": mask.reshape(num_blocks, steps_per_block, 128, 128),
        "supervise": supervise.reshape(num_blocks, steps_per_block),
        "fields": fields,
        "attrs": attrs,
    }


def write_selection_csv(path: Path, data: dict[str, np.ndarray], rows: np.ndarray) -> None:
    fields = data["fields"]
    out_rows: list[dict[str, Any]] = []
    for i, row_id in enumerate(rows):
        item = {"h5_row": int(row_id)}
        for key, values in fields.items():
            item[key] = values[i].item() if hasattr(values[i], "item") else values[i]
        out_rows.append(item)
    write_csv(path, out_rows)


def as_block_tensors(data: dict[str, np.ndarray], device: torch.device) -> dict[str, torch.Tensor]:
    """保留 `[num_blocks,T,...]` 结构，训练时再显式补batch维。"""

    return {
        "event": torch.from_numpy(data["event"]).to(device),
        "target": torch.from_numpy(data["target"]).to(device),
        "mask": torch.from_numpy(data["mask"]).to(device),
        "supervise": torch.from_numpy(data["supervise"]).to(device),
    }


def first_conv_module(model: LIFEVFlowNetTTC) -> torch.nn.Module:
    """第一层ConvLIF中的输入电流卷积。"""

    return model.multires_unetrec.encoders[0].conv.ff


def layer_mem_spike(state: Any) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """把MAVLab不同层的嵌套LIF状态统一取成 `[B,C,H,W]`。

    Encoder和Residual状态形如 `[子层, v/z, B, C, H, W]`，这里取第二个
    子层作为该stage输出状态；Decoder状态形如 `[v/z, B, C, H, W]`。
    """

    if state is None or not torch.is_tensor(state):
        return None, None
    if state.ndim == 6 and state.shape[0] >= 2 and state.shape[1] >= 2:
        stage_state = state[1]
        return stage_state[0].detach(), stage_state[1].detach()
    if state.ndim == 5 and state.shape[0] >= 2:
        return state[0].detach(), state[1].detach()
    return state.detach(), None


def extract_layer_features(model: LIFEVFlowNetTTC) -> dict[str, np.ndarray]:
    """从LIF状态中提取各层膜电位和脉冲，数组保留batch维。"""

    features: dict[str, np.ndarray] = {}
    for idx, state in enumerate(model.multires_unetrec.states):
        layer = LAYER_NAMES[idx] if idx < len(LAYER_NAMES) else f"lif_{idx}"
        mem, spike = layer_mem_spike(state)
        if mem is not None:
            features[f"{layer}_mem"] = mem.detach().float().cpu().numpy()
        if spike is not None:
            features[f"{layer}_spike"] = spike.detach().float().cpu().numpy()
    return features


def scalar_stats(values: torch.Tensor) -> dict[str, float]:
    finite = torch.isfinite(values)
    if not bool(finite.any().detach().cpu()):
        return {"mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
    vals = values[finite].float()
    return {
        "mean": float(vals.mean().detach().cpu()),
        "std": float(vals.std(unbiased=False).detach().cpu()),
        "min": float(vals.min().detach().cpu()),
        "max": float(vals.max().detach().cpu()),
    }


def masked_error_stats(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, supervise: torch.Tensor) -> dict[str, Any]:
    """计算masked MAE、median AE和有效像素数量。"""

    valid = mask.bool() & supervise.bool()[:, None, None] & torch.isfinite(pred[:, 0]) & torch.isfinite(target)
    if not bool(valid.any().detach().cpu()):
        return {"mae": float("nan"), "median_ae": float("nan"), "valid_pixels": 0}
    err = (pred[:, 0] - target).abs()[valid]
    return {
        "mae": float(err.mean().detach().cpu()),
        "median_ae": float(err.median().detach().cpu()),
        "valid_pixels": int(valid.sum().detach().cpu()),
    }


def layer_activity_rows(model: LIFEVFlowNetTTC, epoch: int, block_id: int) -> list[dict[str, Any]]:
    """记录block末尾的主要LIF层活动，用于每epoch聚合。"""

    rows: list[dict[str, Any]] = []
    for idx, state in enumerate(model.multires_unetrec.states):
        layer = LAYER_NAMES[idx] if idx < len(LAYER_NAMES) else f"lif_{idx}"
        mem, spike = layer_mem_spike(state)
        if mem is None:
            continue
        mem_s = tensor_stats(mem.detach())
        if spike is None:
            spike_rate = silent = high = float("nan")
        else:
            spike_bool = spike.detach() != 0
            spike_rate = float(spike_bool.float().mean().detach().cpu())
            per_channel = spike_bool.float().flatten(2).mean(dim=2)
            silent = float((per_channel == 0).float().mean().detach().cpu())
            high = float((per_channel > 0.8).float().mean().detach().cpu())
        rows.append(
            {
                "epoch": epoch,
                "block_local_index": block_id,
                "layer": layer,
                "spike_rate": spike_rate,
                "persistent_silent_ratio_step": silent,
                "persistent_high_firing_ratio_step": high,
                "mem_mean": mem_s["mean"],
                "mem_std": mem_s["std"],
                "mem_abs_p99": mem_s["abs_p99"],
                "finite_ratio": mem_s["finite_ratio"],
            }
        )
    return rows


def select_channels_for_array(arrays: list[np.ndarray], kind: str, num_channels: int = 3) -> list[int]:
    """在Epoch 0上固定代表通道：高活动、中位活动、低但非零。"""

    if not arrays:
        return [0]
    merged = np.concatenate([arr.reshape(arr.shape[0], arr.shape[1], -1) for arr in arrays], axis=0)
    if kind == "spike":
        score = np.mean(merged != 0, axis=(0, 2))
    else:
        score = np.std(merged, axis=(0, 2))
    valid = np.where(score > 1e-12)[0]
    if valid.size == 0:
        return list(range(min(num_channels, merged.shape[1])))
    high = int(valid[np.argmax(score[valid])])
    median = int(valid[np.argsort(score[valid])[len(valid) // 2]])
    low = int(valid[np.argmin(score[valid])])
    selected: list[int] = []
    for ch in [high, median, low]:
        if ch not in selected:
            selected.append(ch)
    for ch in range(merged.shape[1]):
        if len(selected) >= num_channels:
            break
        if ch not in selected:
            selected.append(int(ch))
    return selected


def choose_feature_channels(captures: dict[str, dict[str, np.ndarray]]) -> dict[str, list[int]]:
    """根据训练前锚点特征确定固定通道。"""

    by_key: dict[str, list[np.ndarray]] = defaultdict(list)
    for cap in captures.values():
        for key, arr in cap.items():
            if arr.ndim == 4 and key not in {"prediction", "gt", "mask", "abs_error"}:
                by_key[key].append(arr)
    channels: dict[str, list[int]] = {}
    for key, arrays in sorted(by_key.items()):
        kind = "spike" if key.endswith("_spike") else "continuous"
        channels[key] = select_channels_for_array(arrays, kind=kind, num_channels=3)
    return channels


def capture_anchor_features(
    model: LIFEVFlowNetTTC,
    tensors: dict[str, torch.Tensor],
    *,
    epoch_label: str,
    input_scale: float,
    device: torch.device,
    selected_channels: dict[str, list[int]] | None = None,
) -> dict[str, dict[str, np.ndarray]]:
    """从片段起点重放到三个固定锚点，抓取内部数据流。"""

    model.eval()
    model.reset_states()
    max_block = max(block for block, _ in ANCHORS.values())
    anchor_by_pos = {pos: name for name, pos in ANCHORS.items()}
    captures: dict[str, dict[str, np.ndarray]] = {}
    hook_value: dict[str, torch.Tensor] = {}

    def hook(_module: torch.nn.Module, _inp: tuple[torch.Tensor, ...], out: torch.Tensor) -> None:
        hook_value["first_conv_current"] = out.detach()

    handle = first_conv_module(model).register_forward_hook(hook)
    with torch.no_grad():
        for block_id in range(max_block + 1):
            for step_id in range(tensors["event"].shape[1]):
                event_raw = tensors["event"][block_id : block_id + 1, step_id].float()
                pred = model(event_raw * input_scale)
                pos = (block_id, step_id)
                if pos in anchor_by_pos:
                    anchor_name = anchor_by_pos[pos]
                    target = tensors["target"][block_id : block_id + 1, step_id]
                    mask = tensors["mask"][block_id : block_id + 1, step_id].bool()
                    err = (pred[:, 0] - target).abs()
                    valid = mask & tensors["supervise"][block_id : block_id + 1, step_id].bool()[:, None, None]
                    loss_map = charbonnier(pred[:, 0] - target)
                    loss_val = float((loss_map[valid]).mean().detach().cpu()) if bool(valid.any().detach().cpu()) else float("nan")
                    mae_val = float(err[valid].mean().detach().cpu()) if bool(valid.any().detach().cpu()) else float("nan")
                    cap: dict[str, np.ndarray] = {
                        "input_pos": event_raw[:, 0:1].detach().cpu().numpy(),
                        "input_neg": event_raw[:, 1:2].detach().cpu().numpy(),
                        "input_signed": (event_raw[:, 0:1] - event_raw[:, 1:2]).detach().cpu().numpy(),
                        "first_conv_current": hook_value["first_conv_current"].detach().float().cpu().numpy(),
                        "prediction": pred.detach().float().cpu().numpy(),
                        "gt": target[:, None].detach().float().cpu().numpy(),
                        "mask": mask[:, None].float().detach().cpu().numpy(),
                        "masked_prediction": (pred[:, 0:1] * mask[:, None].float()).detach().float().cpu().numpy(),
                        "abs_error": (err[:, None] * mask[:, None].float()).detach().float().cpu().numpy(),
                    }
                    cap.update(extract_layer_features(model))
                    meta = {
                        "epoch_label": epoch_label,
                        "anchor": anchor_name,
                        "block_local_index": block_id,
                        "step_in_block": step_id,
                        "shape_B_T_C_H_W": [1, tensors["event"].shape[1], 2, 128, 128],
                        "batch_index": 0,
                        "input_scale": input_scale,
                        "raw_event_weight_sum": float(event_raw.sum().detach().cpu()),
                        "scaled_event_weight_sum": float((event_raw * input_scale).sum().detach().cpu()),
                        "masked_mae": mae_val,
                        "charbonnier_loss": loss_val,
                    }
                    cap["_meta_json"] = np.asarray(json.dumps(meta, ensure_ascii=False))
                    if selected_channels is not None:
                        cap = filter_capture_channels(cap, selected_channels)
                        cap["_meta_json"] = np.asarray(json.dumps(meta, ensure_ascii=False))
                    captures[anchor_name] = cap
                if (step_id + 1) % tensors["event"].shape[1] == 0:
                    model.detach_states()
    handle.remove()
    return captures


def filter_capture_channels(cap: dict[str, np.ndarray], selected_channels: dict[str, list[int]]) -> dict[str, np.ndarray]:
    """保存原始数组时只保留固定代表通道，避免NPZ过大。"""

    out: dict[str, np.ndarray] = {}
    for key, arr in cap.items():
        if key == "_meta_json" or arr.ndim != 4 or key in {"prediction", "gt", "mask", "masked_prediction", "abs_error", "input_pos", "input_neg", "input_signed"}:
            out[key] = arr
        elif key in selected_channels:
            out[key] = arr[:, selected_channels[key]]
        else:
            out[key] = arr
    return out


def save_feature_npz(out_dir: Path, epoch_label: str, captures: dict[str, dict[str, np.ndarray]]) -> list[Path]:
    paths: list[Path] = []
    out_dir.mkdir(parents=True, exist_ok=True)
    for anchor, cap in captures.items():
        path = out_dir / f"{epoch_label}_{anchor}.npz"
        np.savez_compressed(path, **cap)
        paths.append(path)
    return paths


def compute_visual_ranges(npz_paths: list[Path]) -> dict[str, list[float]]:
    """使用所有保存特征的1%/99%分位数作为统一显示范围。"""

    vals: dict[str, list[np.ndarray]] = defaultdict(list)
    for path in npz_paths:
        with np.load(path, allow_pickle=False) as data:
            for key in data.files:
                if key == "_meta_json" or key.endswith("_spike") or key == "mask":
                    continue
                arr = data[key].astype(np.float32).reshape(-1)
                if arr.size:
                    vals[key].append(arr)
    ranges: dict[str, list[float]] = {}
    for key, chunks in vals.items():
        merged = np.concatenate(chunks)
        lo, hi = np.percentile(merged[np.isfinite(merged)], [1, 99]) if np.isfinite(merged).any() else (0.0, 1.0)
        if abs(float(hi) - float(lo)) < 1e-12:
            hi = lo + 1e-6
        ranges[key] = [float(lo), float(hi)]
    return ranges


def plot_channels(axes: np.ndarray, arr: np.ndarray, title: str, vmin: float, vmax: float, cmap: str, interpolation: str) -> None:
    for col in range(axes.shape[0]):
        ax = axes[col]
        if col < arr.shape[1]:
            im = ax.imshow(arr[0, col], cmap=cmap, vmin=vmin, vmax=vmax, interpolation=interpolation)
            ax.set_title(f"{title} ch{col}")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        ax.axis("off")


def render_visualizations(npz_paths: list[Path], out_dir: Path, selected_channels: dict[str, list[int]]) -> None:
    """按统一颜色范围渲染输入、内部特征和输出图。"""

    ranges = compute_visual_ranges(npz_paths)
    write_json(out_dir / "feature_display_ranges.json", ranges)
    vis_root = out_dir / "visualizations"
    vis_root.mkdir(parents=True, exist_ok=True)
    for path in npz_paths:
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["_meta_json"].item()))
            epoch_label = meta["epoch_label"]
            anchor = meta["anchor"]
            prefix = vis_root / f"{epoch_label}_{anchor}"

            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            for ax, key, name in zip(axes, ["input_pos", "input_neg", "input_signed"], ["positive events", "negative events", "pos-neg"]):
                vmin, vmax = ranges.get(key, [float(data[key].min()), float(data[key].max())])
                im = ax.imshow(data[key][0, 0], cmap="magma" if key != "input_signed" else "coolwarm", vmin=vmin, vmax=vmax)
                ax.set_title(name)
                ax.axis("off")
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
            fig.suptitle(
                f"{epoch_label} {anchor} | B,T,C,H,W={meta['shape_B_T_C_H_W']} | raw_sum={meta['raw_event_weight_sum']:.1f} | scaled_sum={meta['scaled_event_weight_sum']:.1f}"
            )
            fig.tight_layout()
            fig.savefig(f"{prefix}_input.png", dpi=150)
            plt.close(fig)

            first_keys = ["first_conv_current", "enc1_lif_mem", "enc1_lif_spike"]
            fig, axes = plt.subplots(len(first_keys), 3, figsize=(12, 9))
            for row, key in enumerate(first_keys):
                if key not in data:
                    continue
                is_spike = key.endswith("_spike")
                vmin, vmax = (0.0, 1.0) if is_spike else ranges.get(key, [float(data[key].min()), float(data[key].max())])
                chs = selected_channels.get(key, list(range(data[key].shape[1])))
                for col in range(3):
                    ax = axes[row, col]
                    if col < data[key].shape[1]:
                        im = ax.imshow(data[key][0, col], cmap="gray" if is_spike else "viridis", vmin=vmin, vmax=vmax, interpolation="nearest" if is_spike else "bilinear")
                        ax.set_title(f"{key} orig_ch={chs[col] if col < len(chs) else col} shape={data[key].shape}")
                        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
                    ax.axis("off")
            fig.tight_layout()
            fig.savefig(f"{prefix}_first_layer.png", dpi=150)
            plt.close(fig)

            enc_keys = [f"enc{i}_lif_{kind}" for i in range(1, 5) for kind in ["mem", "spike"]]
            render_feature_grid(data, enc_keys, selected_channels, ranges, f"{prefix}_encoder.png")
            mid_dec_keys = [f"{layer}_{kind}" for layer in ["bottleneck_res1_lif", "bottleneck_res2_lif", "dec1_lif", "dec2_lif", "dec3_lif", "dec4_lif"] for kind in ["mem", "spike"]]
            render_feature_grid(data, mid_dec_keys, selected_channels, ranges, f"{prefix}_bottleneck_decoder.png")
            render_output_panel(data, meta, ranges, f"{prefix}_output.png")


def render_feature_grid(
    data: np.lib.npyio.NpzFile,
    keys: list[str],
    selected_channels: dict[str, list[int]],
    ranges: dict[str, list[float]],
    out_path: str,
) -> None:
    rows = [key for key in keys if key in data]
    fig, axes = plt.subplots(len(rows), 3, figsize=(12, 2.6 * max(len(rows), 1)))
    if len(rows) == 1:
        axes = axes[None, :]
    for row, key in enumerate(rows):
        is_spike = key.endswith("_spike")
        vmin, vmax = (0.0, 1.0) if is_spike else ranges.get(key, [float(data[key].min()), float(data[key].max())])
        chs = selected_channels.get(key, list(range(data[key].shape[1])))
        for col in range(3):
            ax = axes[row, col]
            if col < data[key].shape[1]:
                arr = data[key][0, col]
                im = ax.imshow(arr, cmap="gray" if is_spike else "viridis", vmin=vmin, vmax=vmax, interpolation="nearest" if is_spike else "bilinear")
                spike_rate = float(np.mean(arr != 0)) if is_spike else float("nan")
                title = f"{key} ch={chs[col] if col < len(chs) else col} shape={data[key].shape}"
                if is_spike:
                    title += f" spike={spike_rate:.4f}"
                else:
                    title += f" mean={arr.mean():.3g} std={arr.std():.3g}"
                ax.set_title(title, fontsize=8)
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def render_output_panel(data: np.lib.npyio.NpzFile, meta: dict[str, Any], ranges: dict[str, list[float]], out_path: str) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    panels = [
        ("prediction", "prediction", "coolwarm"),
        ("gt", "GT", "coolwarm"),
        ("abs_error", "abs error", "magma"),
        ("mask", "valid mask", "gray"),
        ("masked_prediction", "masked prediction", "coolwarm"),
    ]
    for ax, (key, title, cmap) in zip(axes.flat[:5], panels):
        arr = data[key][0, 0]
        if key == "mask":
            vmin, vmax = 0.0, 1.0
        else:
            vmin, vmax = ranges.get(key, [float(np.nanmin(arr)), float(np.nanmax(arr))])
        im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    ax = axes.flat[5]
    mask = data["mask"][0, 0].astype(bool)
    if mask.any():
        ax.hist(data["prediction"][0, 0][mask].reshape(-1), bins=40, alpha=0.6, label="pred")
        ax.hist(data["gt"][0, 0][mask].reshape(-1), bins=40, alpha=0.6, label="gt")
    ax.legend()
    ax.set_title("masked histogram")
    fig.suptitle(
        f"{meta['epoch_label']} {meta['anchor']} | block={meta['block_local_index']} step={meta['step_in_block']} | MAE={meta['masked_mae']:.5f} loss={meta['charbonnier_loss']:.5f}"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_checkpoint(
    path: Path,
    model: LIFEVFlowNetTTC,
    optimizer: torch.optim.Optimizer,
    *,
    epoch: int,
    global_step: int,
    args: argparse.Namespace,
    selected_rows: np.ndarray,
    best_loss: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "input_scale": args.input_scale,
            "model_config": asdict(LIFEVFlowNetTTCConfig()),
            "loss_config": {
                "type": "masked_charbonnier_per_sample",
                "alpha": EVTTC_CHARBONNIER_ALPHA,
                "epsilon": EVTTC_CHARBONNIER_EPS,
                "smoothness_weight": 0.0,
            },
            "optimizer_config": {"name": "AdamW", "lr": args.lr, "weight_decay": args.weight_decay},
            "gradient_clip": args.gradient_clip,
            "selected_h5_rows": selected_rows.tolist(),
            "random_seed": args.seed,
            "best_loss": best_loss,
        },
        path,
    )


def train_one_epoch(
    model: LIFEVFlowNetTTC,
    optimizer: torch.optim.Optimizer,
    tensors: dict[str, torch.Tensor],
    *,
    epoch: int,
    input_scale: float,
    gradient_clip: float,
    global_step: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], int]:
    """一个epoch从32-block片段起点开始，block之间只detach不reset。"""

    model.train()
    model.reset_states()
    epoch_loss_values: list[float] = []
    epoch_grad_norms: list[float] = []
    epoch_abs_errors: list[torch.Tensor] = []
    block_rows: list[dict[str, Any]] = []
    activity_raw_rows: list[dict[str, Any]] = []
    valid_step_samples_total = 0
    valid_pixels_total = 0
    optimizer_steps = 0
    pred_stats_acc: list[dict[str, float]] = []
    finite_ok = True

    for block_id in range(tensors["event"].shape[0]):
        optimizer.zero_grad(set_to_none=True)
        loss_sum: torch.Tensor | None = None
        valid_sample_step_count = 0
        block_abs_errors: list[torch.Tensor] = []
        block_valid_pixels = 0
        block_pred_stats: list[dict[str, float]] = []

        for step_id in range(tensors["event"].shape[1]):
            events_t = tensors["event"][block_id : block_id + 1, step_id].float() * input_scale
            target_t = tensors["target"][block_id : block_id + 1, step_id].float()
            mask_t = tensors["mask"][block_id : block_id + 1, step_id].bool()
            supervise_t = tensors["supervise"][block_id : block_id + 1, step_id].bool()
            pred_t = model(events_t)
            finite_ok = finite_ok and bool(torch.isfinite(pred_t).all().detach().cpu())
            per_sample_loss, valid_samples, stats = masked_charbonnier_per_sample(pred_t, target_t, mask_t, supervise_t)
            if bool(valid_samples.any().detach().cpu()):
                current_sum = per_sample_loss[valid_samples].sum()
                loss_sum = current_sum if loss_sum is None else loss_sum + current_sum
                valid_sample_step_count += int(valid_samples.sum().detach().cpu())
                valid_mask = mask_t & supervise_t[:, None, None]
                err = (pred_t[:, 0] - target_t).abs()[valid_mask]
                block_abs_errors.append(err.detach().cpu())
                block_valid_pixels += int(stats.total_valid_pixels)
            block_pred_stats.append(scalar_stats(pred_t.detach()))

        backward_called = False
        optimizer_step_called = False
        grad_norm = float("nan")
        block_loss_value = float("nan")
        if valid_sample_step_count > 0 and loss_sum is not None:
            block_loss = loss_sum / valid_sample_step_count
            block_loss.backward()
            backward_called = True
            grad_norm_tensor = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip)
            grad_norm = float(grad_norm_tensor.detach().cpu())
            if math.isfinite(grad_norm):
                optimizer.step()
                optimizer_step_called = True
                optimizer_steps += 1
                global_step += 1
            block_loss_value = float(block_loss.detach().cpu())
            epoch_loss_values.append(block_loss_value)
            epoch_grad_norms.append(grad_norm)

        activity_raw_rows.extend(layer_activity_rows(model, epoch, block_id))
        model.detach_states()

        if block_abs_errors:
            block_err = torch.cat(block_abs_errors)
            epoch_abs_errors.append(block_err)
            block_mae = float(block_err.mean())
            block_median = float(block_err.median())
        else:
            block_mae = float("nan")
            block_median = float("nan")
        pred_stats = {
            "pred_mean": float(np.mean([x["mean"] for x in block_pred_stats])),
            "pred_std": float(np.mean([x["std"] for x in block_pred_stats])),
            "pred_min": float(np.min([x["min"] for x in block_pred_stats])),
            "pred_max": float(np.max([x["max"] for x in block_pred_stats])),
        }
        pred_stats_acc.append(pred_stats)
        valid_step_samples_total += valid_sample_step_count
        valid_pixels_total += block_valid_pixels
        block_rows.append(
            {
                "epoch": epoch,
                "block_local_index": block_id,
                "block_loss": block_loss_value,
                "masked_mae": block_mae,
                "median_ae": block_median,
                "valid_step_sample_count": valid_sample_step_count,
                "valid_pixel_count": block_valid_pixels,
                "grad_norm": grad_norm,
                "backward_called": int(backward_called),
                "optimizer_step_called": int(optimizer_step_called),
                **pred_stats,
            }
        )

    if epoch_abs_errors:
        all_err = torch.cat(epoch_abs_errors)
        epoch_mae = float(all_err.mean())
        epoch_median = float(all_err.median())
    else:
        epoch_mae = float("nan")
        epoch_median = float("nan")
    epoch_row = {
        "epoch": epoch,
        "mean_block_loss": float(np.mean(epoch_loss_values)) if epoch_loss_values else float("nan"),
        "masked_mae": epoch_mae,
        "median_ae": epoch_median,
        "valid_step_sample_count": valid_step_samples_total,
        "valid_pixel_count": valid_pixels_total,
        "mean_grad_norm": float(np.nanmean(epoch_grad_norms)) if epoch_grad_norms else float("nan"),
        "optimizer_step_count": optimizer_steps,
        "pred_mean": float(np.mean([x["pred_mean"] for x in pred_stats_acc])),
        "pred_std": float(np.mean([x["pred_std"] for x in pred_stats_acc])),
        "pred_min": float(np.min([x["pred_min"] for x in pred_stats_acc])),
        "pred_max": float(np.max([x["pred_max"] for x in pred_stats_acc])),
        "finite_ok": int(finite_ok),
    }
    activity_epoch_rows = aggregate_activity(epoch, activity_raw_rows)
    return epoch_row, block_rows, activity_epoch_rows, global_step


def evaluate_one_epoch(
    model: LIFEVFlowNetTTC,
    tensors: dict[str, torch.Tensor],
    *,
    epoch: int,
    input_scale: float,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """训练前/检查点评估：只forward，不backward，不更新参数。"""

    model.eval()
    model.reset_states()
    epoch_loss_values: list[float] = []
    epoch_abs_errors: list[torch.Tensor] = []
    block_rows: list[dict[str, Any]] = []
    activity_raw_rows: list[dict[str, Any]] = []
    valid_step_samples_total = 0
    valid_pixels_total = 0
    pred_stats_acc: list[dict[str, float]] = []
    finite_ok = True
    with torch.no_grad():
        for block_id in range(tensors["event"].shape[0]):
            block_loss_sum = 0.0
            valid_sample_step_count = 0
            block_abs_errors: list[torch.Tensor] = []
            block_valid_pixels = 0
            block_pred_stats: list[dict[str, float]] = []
            for step_id in range(tensors["event"].shape[1]):
                events_t = tensors["event"][block_id : block_id + 1, step_id].float() * input_scale
                target_t = tensors["target"][block_id : block_id + 1, step_id].float()
                mask_t = tensors["mask"][block_id : block_id + 1, step_id].bool()
                supervise_t = tensors["supervise"][block_id : block_id + 1, step_id].bool()
                pred_t = model(events_t)
                finite_ok = finite_ok and bool(torch.isfinite(pred_t).all().detach().cpu())
                per_sample_loss, valid_samples, stats = masked_charbonnier_per_sample(pred_t, target_t, mask_t, supervise_t)
                if bool(valid_samples.any().detach().cpu()):
                    block_loss_sum += float(per_sample_loss[valid_samples].sum().detach().cpu())
                    valid_sample_step_count += int(valid_samples.sum().detach().cpu())
                    valid_mask = mask_t & supervise_t[:, None, None]
                    err = (pred_t[:, 0] - target_t).abs()[valid_mask]
                    block_abs_errors.append(err.detach().cpu())
                    block_valid_pixels += int(stats.total_valid_pixels)
                block_pred_stats.append(scalar_stats(pred_t.detach()))

            block_loss_value = block_loss_sum / valid_sample_step_count if valid_sample_step_count else float("nan")
            if math.isfinite(block_loss_value):
                epoch_loss_values.append(block_loss_value)
            activity_raw_rows.extend(layer_activity_rows(model, epoch, block_id))
            model.detach_states()

            if block_abs_errors:
                block_err = torch.cat(block_abs_errors)
                epoch_abs_errors.append(block_err)
                block_mae = float(block_err.mean())
                block_median = float(block_err.median())
            else:
                block_mae = float("nan")
                block_median = float("nan")
            pred_stats = {
                "pred_mean": float(np.mean([x["mean"] for x in block_pred_stats])),
                "pred_std": float(np.mean([x["std"] for x in block_pred_stats])),
                "pred_min": float(np.min([x["min"] for x in block_pred_stats])),
                "pred_max": float(np.max([x["max"] for x in block_pred_stats])),
            }
            pred_stats_acc.append(pred_stats)
            valid_step_samples_total += valid_sample_step_count
            valid_pixels_total += block_valid_pixels
            block_rows.append(
                {
                    "epoch": epoch,
                    "block_local_index": block_id,
                    "block_loss": block_loss_value,
                    "masked_mae": block_mae,
                    "median_ae": block_median,
                    "valid_step_sample_count": valid_sample_step_count,
                    "valid_pixel_count": block_valid_pixels,
                    "grad_norm": float("nan"),
                    "backward_called": 0,
                    "optimizer_step_called": 0,
                    **pred_stats,
                }
            )

    if epoch_abs_errors:
        all_err = torch.cat(epoch_abs_errors)
        epoch_mae = float(all_err.mean())
        epoch_median = float(all_err.median())
    else:
        epoch_mae = float("nan")
        epoch_median = float("nan")
    epoch_row = {
        "epoch": epoch,
        "mean_block_loss": float(np.mean(epoch_loss_values)) if epoch_loss_values else float("nan"),
        "masked_mae": epoch_mae,
        "median_ae": epoch_median,
        "valid_step_sample_count": valid_step_samples_total,
        "valid_pixel_count": valid_pixels_total,
        "mean_grad_norm": float("nan"),
        "optimizer_step_count": 0,
        "pred_mean": float(np.mean([x["pred_mean"] for x in pred_stats_acc])),
        "pred_std": float(np.mean([x["pred_std"] for x in pred_stats_acc])),
        "pred_min": float(np.min([x["pred_min"] for x in pred_stats_acc])),
        "pred_max": float(np.max([x["pred_max"] for x in pred_stats_acc])),
        "finite_ok": int(finite_ok),
    }
    return epoch_row, block_rows, aggregate_activity(epoch, activity_raw_rows)


def aggregate_activity(epoch: int, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_layer: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_layer[row["layer"]].append(row)
    out: list[dict[str, Any]] = []
    for layer, vals in sorted(by_layer.items(), key=lambda x: LAYER_NAMES.index(x[0]) if x[0] in LAYER_NAMES else 99):
        out.append(
            {
                "epoch": epoch,
                "layer": layer,
                "mean_spike_rate": float(np.nanmean([v["spike_rate"] for v in vals])),
                "mean_silent_ratio": float(np.nanmean([v["persistent_silent_ratio_step"] for v in vals])),
                "mean_high_firing_ratio": float(np.nanmean([v["persistent_high_firing_ratio_step"] for v in vals])),
                "mean_mem_mean": float(np.nanmean([v["mem_mean"] for v in vals])),
                "mean_mem_std": float(np.nanmean([v["mem_std"] for v in vals])),
                "max_mem_abs_p99": float(np.nanmax([v["mem_abs_p99"] for v in vals])),
                "mean_finite_ratio": float(np.nanmean([v["finite_ratio"] for v in vals])),
            }
        )
    return out


def plot_training_curves(out_dir: Path, epoch_rows: list[dict[str, Any]]) -> None:
    epochs = [int(r["epoch"]) for r in epoch_rows]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(epochs, [float(r["mean_block_loss"]) for r in epoch_rows])
    axes[0].set_title("Mean block loss")
    axes[1].plot(epochs, [float(r["masked_mae"]) for r in epoch_rows])
    axes[1].set_title("Masked MAE")
    axes[2].plot(epochs, [float(r["mean_grad_norm"]) for r in epoch_rows])
    axes[2].set_title("Grad norm")
    for ax in axes:
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "train_curves.png", dpi=160)
    plt.close(fig)


def write_reports(
    out_dir: Path,
    args: argparse.Namespace,
    segment_rows: np.ndarray,
    epoch_rows: list[dict[str, Any]],
    activity_rows: list[dict[str, Any]],
    best_epoch: int,
    npz_paths: list[Path],
    selected_channels: dict[str, list[int]],
) -> None:
    initial = epoch_rows[0]
    final = epoch_rows[-1]
    best = min(epoch_rows, key=lambda r: float(r["mean_block_loss"]))
    loss_drop = 1.0 - float(final["mean_block_loss"]) / max(float(initial["mean_block_loss"]), 1e-12)
    mae_drop = 1.0 - float(final["masked_mae"]) / max(float(initial["masked_mae"]), 1e-12)
    success = loss_drop >= 0.8 and mae_drop >= 0.5 and int(final["finite_ok"]) == 1
    final_activity = [r for r in activity_rows if int(r["epoch"]) == int(final["epoch"])]
    high_layers = [r["layer"] for r in final_activity if float(r["mean_high_firing_ratio"]) > 0.5 or float(r["mean_spike_rate"]) > 0.8]
    silent_decoders = [r["layer"] for r in final_activity if str(r["layer"]).startswith("dec") and float(r["mean_spike_rate"]) < 1e-4]

    main_report = f"""# 32Block过拟合训练报告

## 实验配置
- 数据：`{args.h5}`
- 输出目录：`{out_dir}`
- 选择H5行号：`{int(segment_rows[0])}` 到 `{int(segment_rows[-1])}`
- Batch tensor：`[B,T,C,H,W] = [1,10,2,128,128]`
- input_scale：`{args.input_scale}`，只作用于事件输入。
- optimizer：AdamW，lr=`{args.lr}`，weight_decay=`{args.weight_decay}`
- gradient clip：`{args.gradient_clip}`，来源于 MAVLab `train_SNN.yml` 的 `loss.clip_grad=100.0`
- epochs：`{args.epochs}`

## 训练结果
| 指标 | 初始/第1轮 | 最终 | 变化 |
| --- | ---: | ---: | ---: |
| mean block loss | {fmt(initial['mean_block_loss'])} | {fmt(final['mean_block_loss'])} | {fmt(loss_drop * 100)}%下降 |
| masked MAE | {fmt(initial['masked_mae'])} | {fmt(final['masked_mae'])} | {fmt(mae_drop * 100)}%下降 |
| median AE | {fmt(initial['median_ae'])} | {fmt(final['median_ae'])} | - |
| pred std | {fmt(initial['pred_std'])} | {fmt(final['pred_std'])} | - |

- 最佳epoch：`{best_epoch}`，best loss=`{fmt(best['mean_block_loss'])}`
- 参数更新次数/最终epoch：`{int(final['optimizer_step_count'])}`
- 输出有限性：`{"正常" if int(final["finite_ok"]) == 1 else "异常"}`

## 结论
- 训练是否成功：`{"通过" if success else "未完全通过"}`
- 模型是否记住32个Block：`{"是" if success else "需结合可视化复查"}`
- scale=0.3训练后是否仍合理：`{"是" if not high_layers and not silent_decoders else "需复查"}`
- 高发放层：`{", ".join(high_layers) if high_layers else "无"}`
- 沉默decoder层：`{", ".join(silent_decoders) if silent_decoders else "无"}`

## 文件
- `train_epoch_metrics.csv`
- `train_block_metrics.csv`
- `lif_activity_by_epoch.csv`
- `train_curves.png`
- `checkpoints/best_loss.pt`
- `features_npz/`
- `visualizations/`
"""
    (out_dir / "32Block过拟合训练报告.md").write_text(main_report, encoding="utf-8")

    feature_report = f"""# 32Block过拟合内部特征可视化说明

## 固定锚点
- A：Block 0，Step 0，片段起点。
- B：Block 16，Step 5，片段中部。
- C：Block 31，Step 9，片段末端。

## 固定通道
通道在Epoch 0上按空间标准差/活动程度选定，之后不再改变。

```json
{json.dumps(selected_channels, ensure_ascii=False, indent=2)}
```

## 可视化内容
- 输入：正事件、负事件、pos-neg。
- 第一层：`first_conv_current`、`enc1_lif_mem`、`enc1_lif_spike`。
- Encoder：`enc1` 到 `enc4` 的膜电位和脉冲。
- Bottleneck：两个res LIF块的膜电位和脉冲。
- Decoder：`dec1` 到 `dec4` 的膜电位和脉冲。
- 输出：prediction、GT、mask、masked prediction、abs error和masked直方图。

## 颜色范围
连续特征使用所有保存NPZ的1%到99%分位数统一显示；脉冲图固定0到1。原始数组保存在 `features_npz/`。

## 生成数量
- NPZ数量：`{len(npz_paths)}`
- PNG目录：`visualizations/`
"""
    (out_dir / "32Block过拟合内部特征可视化说明.md").write_text(feature_report, encoding="utf-8")

    decision_report = f"""# 32Block过拟合失败或通过判定

## 判定
- 结果：`{"通过" if success else "未完全通过"}`
- loss下降：`{fmt(loss_drop * 100)}%`
- MAE下降：`{fmt(mae_drop * 100)}%`
- 最佳epoch：`{best_epoch}`

## 逐项回答
1. 32个连续Block是否正确选择：是，H5行 `{int(segment_rows[0])}-{int(segment_rows[-1])}`，block连续且不跨timeout。
2. Batch tensor完整shape：`[1,10,2,128,128]`。
3. input_scale是否只作用于事件输入：是，只执行 `event_cnt.float() * {args.input_scale}`。
4. 训练Loss是否明显下降：`{"是" if loss_drop >= 0.8 else "否"}`。
5. 模型是否能记忆32个Block：`{"是" if success else "需复查"}`。
6. 第一层如何将正负事件转换为特征：通过第一层ConvLIF输入电流卷积从2通道事件计数映射到64个64x64电流特征，再经LIF膜电位与脉冲状态编码。
7. Encoder各级空间尺寸和特征变化：约为64x64、32x32、16x16、8x8，空间分辨率下降、通道数增加，spike/membrane承载更抽象结构。
8. Bottleneck是否保留有效空间结构：见 `bottleneck_decoder` 可视化；最终高发放层 `{", ".join(high_layers) if high_layers else "无"}`。
9. Decoder是否从低分辨率特征恢复TTC结构：见 `bottleneck_decoder` 与 `output` 可视化。
10. 输出层预测是否逐渐接近GT：以loss和MAE下降、固定锚点误差图减弱为依据。
11. scale=0.3是否导致饱和或沉默：`{"否" if not high_layers and not silent_decoders else "存在迹象，需复查"}`。
12. 是否满足进入500-Block训练管线验证：`{"是" if success else "暂不建议"}`。

## 下一步
本脚本已停止在32-block阶段，没有启动500-block训练。
"""
    (out_dir / "32Block过拟合失败或通过判定.md").write_text(decision_report, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", type=Path, default=DEFAULT_H5)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--input-scale", type=float, default=0.3)
    parser.add_argument("--steps-per-block", type=int, default=10)
    parser.add_argument("--num-blocks", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--gradient-clip", type=float, default=100.0)
    parser.add_argument("--checkpoint-interval", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.batch_size != 1:
        raise ValueError("本阶段固定batch_size=1，避免SNN状态管理混入额外变量。")
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for sub in ["checkpoints", "visualizations", "features_npz", "logs"]:
        (args.out_dir / sub).mkdir(exist_ok=True)

    segment = choose_continuous_segment(args.h5, args.num_blocks, args.steps_per_block)
    data = load_selected_data(args.h5, segment.rows, args.steps_per_block)
    write_selection_csv(args.out_dir / "32block_overfit_selection.csv", data, segment.rows)
    tensors = as_block_tensors(data, device)

    config_text = f"""experiment: lif_evflownet_ttc_s2_n10k_overfit32
h5: {args.h5}
spatial_scheme: S2_center256_to128
events_per_step: 10000
input_size: [128, 128]
input_channels: 2
input_scale: {args.input_scale}
steps_per_block: {args.steps_per_block}
num_blocks: {args.num_blocks}
total_steps: {args.num_blocks * args.steps_per_block}
batch_size: 1
shuffle: false
optimizer: AdamW
learning_rate: {args.lr}
weight_decay: {args.weight_decay}
max_epochs: {args.epochs}
checkpoint_interval: {args.checkpoint_interval}
gradient_clip: {args.gradient_clip}
loss:
  type: masked_charbonnier_per_sample
  alpha: {EVTTC_CHARBONNIER_ALPHA}
  epsilon: {EVTTC_CHARBONNIER_EPS}
  smoothness_weight: 0
"""
    (args.out_dir / "config.yaml").write_text(config_text, encoding="utf-8")

    model = LIFEVFlowNetTTC().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    initial_state = copy.deepcopy(model.state_dict())
    save_checkpoint(
        args.out_dir / "checkpoints" / "epoch_000.pt",
        model,
        optimizer,
        epoch=0,
        global_step=0,
        args=args,
        selected_rows=segment.rows,
        best_loss=float("inf"),
    )

    # Epoch 0 先选通道，再用同一通道保存训练前特征。
    model.load_state_dict(initial_state)
    epoch0_full = capture_anchor_features(model, tensors, epoch_label="epoch_000", input_scale=args.input_scale, device=device)
    selected_channels = choose_feature_channels(epoch0_full)
    write_json(
        args.out_dir / "selected_feature_channels.json",
        {"anchors": ANCHORS, "selected_channels": selected_channels, "selection_epoch": 0},
    )
    model.load_state_dict(initial_state)
    epoch0_caps = capture_anchor_features(model, tensors, epoch_label="epoch_000", input_scale=args.input_scale, device=device, selected_channels=selected_channels)
    npz_paths = save_feature_npz(args.out_dir / "features_npz", "epoch_000", epoch0_caps)

    model.load_state_dict(initial_state)
    epoch0_row, epoch0_block_rows, epoch0_activity_rows = evaluate_one_epoch(
        model,
        tensors,
        epoch=0,
        input_scale=args.input_scale,
    )
    epoch_rows: list[dict[str, Any]] = [epoch0_row]
    block_rows_all: list[dict[str, Any]] = epoch0_block_rows[:]
    activity_rows_all: list[dict[str, Any]] = epoch0_activity_rows[:]
    best_loss = float("inf")
    best_epoch = 0
    global_step = 0
    visual_epochs = set(e for e in VIS_EPOCHS_BASE if e <= args.epochs)
    visual_epochs.add(args.epochs)

    for epoch in range(1, args.epochs + 1):
        epoch_row, block_rows, activity_rows, global_step = train_one_epoch(
            model,
            optimizer,
            tensors,
            epoch=epoch,
            input_scale=args.input_scale,
            gradient_clip=args.gradient_clip,
            global_step=global_step,
        )
        epoch_rows.append(epoch_row)
        block_rows_all.extend(block_rows)
        activity_rows_all.extend(activity_rows)

        current_loss = float(epoch_row["mean_block_loss"])
        if math.isfinite(current_loss) and current_loss < best_loss:
            best_loss = current_loss
            best_epoch = epoch
            save_checkpoint(
                args.out_dir / "checkpoints" / "best_loss.pt",
                model,
                optimizer,
                epoch=epoch,
                global_step=global_step,
                args=args,
                selected_rows=segment.rows,
                best_loss=best_loss,
            )

        if epoch % args.checkpoint_interval == 0 or epoch in {args.epochs}:
            save_checkpoint(
                args.out_dir / "checkpoints" / f"epoch_{epoch:03d}.pt",
                model,
                optimizer,
                epoch=epoch,
                global_step=global_step,
                args=args,
                selected_rows=segment.rows,
                best_loss=best_loss,
            )
        save_checkpoint(
            args.out_dir / "checkpoints" / "latest.pt",
            model,
            optimizer,
            epoch=epoch,
            global_step=global_step,
            args=args,
            selected_rows=segment.rows,
            best_loss=best_loss,
        )

        if epoch in visual_epochs:
            caps = capture_anchor_features(model, tensors, epoch_label=f"epoch_{epoch:03d}", input_scale=args.input_scale, device=device, selected_channels=selected_channels)
            npz_paths.extend(save_feature_npz(args.out_dir / "features_npz", f"epoch_{epoch:03d}", caps))

    if best_epoch not in visual_epochs:
        ckpt = torch.load(args.out_dir / "checkpoints" / "best_loss.pt", map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        caps = capture_anchor_features(model, tensors, epoch_label=f"best_epoch_{best_epoch:03d}", input_scale=args.input_scale, device=device, selected_channels=selected_channels)
        npz_paths.extend(save_feature_npz(args.out_dir / "features_npz", f"best_epoch_{best_epoch:03d}", caps))

    write_csv(args.out_dir / "train_epoch_metrics.csv", epoch_rows)
    write_csv(args.out_dir / "train_block_metrics.csv", block_rows_all)
    write_csv(args.out_dir / "lif_activity_by_epoch.csv", activity_rows_all)
    plot_training_curves(args.out_dir, epoch_rows)
    render_visualizations(npz_paths, args.out_dir, selected_channels)
    write_reports(args.out_dir, args, segment.rows, epoch_rows, activity_rows_all, best_epoch, npz_paths, selected_channels)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
