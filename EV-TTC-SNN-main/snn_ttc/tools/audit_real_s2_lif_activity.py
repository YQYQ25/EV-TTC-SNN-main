#!/usr/bin/env python3
"""真实S2数据上的 LIF 脉冲率与膜电位审计。

本脚本只做 forward 审计：不计算loss、不反传、不更新参数。
审计目标是比较原始10k事件计数与统一缩放0.1时，各LIF层是否沉默、
饱和、膜电位异常累积，以及连续状态管理是否按预期生效。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import hdf5plugin  # noqa: F401
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

# 报告保持中文；图内标题使用英文以避免服务器字体回退导致乱码。
plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SNN_ROOT = PROJECT_ROOT / "EV-TTC-SNN-main"
if str(SNN_ROOT) not in sys.path:
    sys.path.insert(0, str(SNN_ROOT))

from snn_ttc.models import LIFEVFlowNetTTC  # noqa: E402


DEFAULT_H5 = (
    SNN_ROOT
    / "debug_sets"
    / "[5]car_spatial_ablation"
    / "independent_fixed_n"
    / "N10000_S2_center256_to128.h5"
)
DEFAULT_OUT_DIR = SNN_ROOT / "debug_sets" / "lif_evflownet_ttc_activity_audit"
LAYER_NAMES = [
    "enc1_lif",
    "enc2_lif",
    "enc3_lif",
    "enc4_lif",
    "bottleneck_res1_lif",
    "bottleneck_res2_lif",
    "dec1_lif",
    "dec2_lif",
    "dec3_lif",
    "dec4_lif",
]
ENCODER_LAYERS = {name for name in LAYER_NAMES if name.startswith("enc")}
BOTTLENECK_LAYERS = {name for name in LAYER_NAMES if name.startswith("bottleneck")}
DECODER_LAYERS = {name for name in LAYER_NAMES if name.startswith("dec")}


@dataclass(frozen=True)
class SelectedSegment:
    rows: np.ndarray
    block_indices: np.ndarray
    step_in_block: np.ndarray
    t_start: np.ndarray
    t_end: np.ndarray


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """写CSV，字段自动取所有行的并集。"""

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


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(str(v) for v in row) + " |" for row in rows)
    return "\n".join(lines)


def fmt(value: Any, digits: int = 4) -> str:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(v):
        return "NA"
    if abs(v) >= 1000:
        return f"{v:.2f}"
    return f"{v:.{digits}f}"


def choose_continuous_segment(h5_path: Path, num_blocks: int, steps_per_block: int) -> SelectedSegment:
    """选择不跨timeout/断裂的连续block片段，不随机打乱。"""

    with h5py.File(h5_path, "r") as f:
        total = int(f["event_cnt"].shape[0])
        block_index = f["block_index"][:].astype(np.int64) if "block_index" in f else np.arange(total) // steps_per_block
        step_in_block = f["step_in_block"][:].astype(np.int64) if "step_in_block" in f else np.arange(total) % steps_per_block
        timeout = f["timeout"][:].astype(bool) if "timeout" in f else np.zeros(total, dtype=bool)
        reset_required = f["reset_required"][:].astype(bool) if "reset_required" in f else np.zeros(total, dtype=bool)
        t_start = f["t_start"][:].astype(np.int64) if "t_start" in f else np.arange(total, dtype=np.int64)
        t_end = f["t_end"][:].astype(np.int64) if "t_end" in f else t_start + 1

    need = num_blocks * steps_per_block
    for start in range(0, total - need + 1, steps_per_block):
        rows = np.arange(start, start + need)
        block_rows = block_index[rows].reshape(num_blocks, steps_per_block)
        step_rows = step_in_block[rows].reshape(num_blocks, steps_per_block)
        complete_blocks = all(set(step_rows[i].tolist()) == set(range(steps_per_block)) for i in range(num_blocks))
        block_ids = block_rows[:, 0]
        block_consistent = bool(np.all(block_rows == block_ids[:, None]))
        block_consecutive = bool(np.all(np.diff(block_ids) == 1))
        no_timeout = not bool(timeout[rows].any())
        time_monotonic = bool(np.all(np.diff(t_start[rows]) >= 0))
        # 第一个step可以是片段开始；片段内部若要求reset，说明出现真实断裂。
        no_inner_reset = not bool(reset_required[rows[1:]].any())
        if complete_blocks and block_consistent and block_consecutive and no_timeout and time_monotonic and no_inner_reset:
            return SelectedSegment(
                rows=rows,
                block_indices=block_index[rows],
                step_in_block=step_in_block[rows],
                t_start=t_start[rows],
                t_end=t_end[rows],
            )
    raise RuntimeError(f"未找到满足 {num_blocks} 个连续block的片段：{h5_path}")


def load_events(h5_path: Path, rows: np.ndarray) -> np.ndarray:
    """只读取选中的200个step，避免加载整个H5。"""

    with h5py.File(h5_path, "r") as f:
        return f["event_cnt"][rows].astype(np.float32)


def state_to_mem_spike(state: Any) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """从ConvLIF状态中取膜电位v和spike z。"""

    if state is None or not torch.is_tensor(state):
        return None, None
    if state.ndim >= 5 and state.shape[0] >= 2:
        return state[0].detach(), state[1].detach()
    return state.detach(), None


def tensor_stats(x: torch.Tensor) -> dict[str, float]:
    """统计有限值比例和常用分位数。"""

    finite = torch.isfinite(x)
    if not bool(finite.any().detach().cpu()):
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "abs_p95": float("nan"),
            "abs_p99": float("nan"),
            "finite_ratio": 0.0,
        }
    vals = x[finite].float()
    return {
        "mean": float(vals.mean().cpu()),
        "std": float(vals.std(unbiased=False).cpu()),
        "min": float(vals.min().cpu()),
        "max": float(vals.max().cpu()),
        "abs_p95": float(torch.quantile(vals.abs(), 0.95).cpu()),
        "abs_p99": float(torch.quantile(vals.abs(), 0.99).cpu()),
        "finite_ratio": float(finite.float().mean().cpu()),
    }


def output_stats(pred: torch.Tensor) -> dict[str, float]:
    stats = tensor_stats(pred)
    finite = torch.isfinite(pred)
    if bool(finite.any().detach().cpu()):
        vals = pred[finite]
        stats.update(
            {
                "positive_ratio": float((vals > 0).float().mean().cpu()),
                "negative_ratio": float((vals < 0).float().mean().cpu()),
            }
        )
    else:
        stats.update({"positive_ratio": float("nan"), "negative_ratio": float("nan")})
    return stats


def collect_layer_step_rows(
    model: LIFEVFlowNetTTC,
    *,
    scale: float,
    step_id: int,
    block_id: int,
    step_in_block: int,
    temporal_spike_sum: dict[str, torch.Tensor],
    temporal_count: dict[str, int],
) -> list[dict[str, Any]]:
    """从模型状态读取每层当前step的LIF活动。"""

    rows: list[dict[str, Any]] = []
    states = model.multires_unetrec.states
    for layer_index, state in enumerate(states):
        layer = LAYER_NAMES[layer_index] if layer_index < len(LAYER_NAMES) else f"lif_{layer_index}"
        mem, spike = state_to_mem_spike(state)
        if mem is None:
            continue
        mem_s = tensor_stats(mem)
        if spike is None:
            spike_rate = zero_spike_ratio = high_firing_ratio = float("nan")
        else:
            spike_bool = spike != 0
            spike_rate = float(spike_bool.float().mean().cpu())
            # 单step中每个元素最多一次发放，因此zero/high分别等价于1-rate/rate。
            zero_spike_ratio = float((~spike_bool).float().mean().cpu())
            high_firing_ratio = spike_rate
            if layer not in temporal_spike_sum:
                temporal_spike_sum[layer] = spike_bool.float().detach().cpu()
                temporal_count[layer] = 1
            else:
                temporal_spike_sum[layer] += spike_bool.float().detach().cpu()
                temporal_count[layer] += 1
        rows.append(
            {
                "scale": scale,
                "step_id": step_id,
                "block_id": block_id,
                "step_in_block": step_in_block,
                "layer": layer,
                "spike_rate": spike_rate,
                "zero_spike_ratio": zero_spike_ratio,
                "high_firing_ratio": high_firing_ratio,
                "mem_mean": mem_s["mean"],
                "mem_std": mem_s["std"],
                "mem_min": mem_s["min"],
                "mem_max": mem_s["max"],
                "mem_abs_p95": mem_s["abs_p95"],
                "mem_abs_p99": mem_s["abs_p99"],
                "finite_ratio": mem_s["finite_ratio"],
            }
        )
    return rows


def state_signature(model: LIFEVFlowNetTTC) -> list[torch.Tensor]:
    """复制状态张量，用于确认detach不会清空或改写数值。"""

    return model.state_value_signature()


def max_state_diff(a: list[torch.Tensor], b: list[torch.Tensor]) -> float:
    if len(a) != len(b):
        return float("inf")
    diffs = [float((x - y).abs().max().cpu()) for x, y in zip(a, b) if x.numel() and y.numel()]
    return max(diffs) if diffs else 0.0


def run_forward_audit(
    *,
    events_np: np.ndarray,
    segment: SelectedSegment,
    scale: float,
    seed: int,
    device: torch.device,
    steps_per_block: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """执行200步连续状态forward审计。"""

    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = LIFEVFlowNetTTC().to(device)
    model.eval()
    model.reset_states()

    layer_rows: list[dict[str, Any]] = []
    output_rows: list[dict[str, Any]] = []
    detach_rows: list[dict[str, Any]] = []
    temporal_spike_sum: dict[str, torch.Tensor] = {}
    temporal_count: dict[str, int] = {}

    with torch.no_grad():
        for local_step in range(events_np.shape[0]):
            block_id = int(segment.block_indices[local_step])
            step_in_block = int(segment.step_in_block[local_step])
            x = torch.from_numpy(events_np[local_step : local_step + 1]).to(device) * float(scale)
            pred = model(x)
            layer_rows.extend(
                collect_layer_step_rows(
                    model,
                    scale=scale,
                    step_id=local_step + 1,
                    block_id=block_id,
                    step_in_block=step_in_block,
                    temporal_spike_sum=temporal_spike_sum,
                    temporal_count=temporal_count,
                )
            )
            out_s = output_stats(pred)
            output_rows.append(
                {
                    "scale": scale,
                    "step_id": local_step + 1,
                    "block_id": block_id,
                    "step_in_block": step_in_block,
                    "output_mean": out_s["mean"],
                    "output_std": out_s["std"],
                    "output_min": out_s["min"],
                    "output_max": out_s["max"],
                    "output_abs_p95": out_s["abs_p95"],
                    "positive_ratio": out_s["positive_ratio"],
                    "negative_ratio": out_s["negative_ratio"],
                    "finite_ratio": out_s["finite_ratio"],
                }
            )
            if (local_step + 1) % steps_per_block == 0:
                before = state_signature(model)
                model.detach_states()
                after = state_signature(model)
                detach_rows.append(
                    {
                        "scale": scale,
                        "boundary_step": local_step + 1,
                        "block_id": block_id,
                        "max_state_abs_diff_before_after_detach": max_state_diff(before, after),
                        "num_state_tensors": len(before),
                    }
                )

    temporal_rows: list[dict[str, Any]] = []
    for layer, summed in temporal_spike_sum.items():
        firing = summed / max(temporal_count[layer], 1)
        temporal_rows.append(
            {
                "scale": scale,
                "layer": layer,
                "temporal_mean_firing_rate": float(firing.mean().cpu()),
                "persistent_silent_neuron_ratio": float((firing == 0).float().mean().cpu()),
                "persistent_high_firing_neuron_ratio": float((firing > 0.8).float().mean().cpu()),
            }
        )
    return layer_rows, output_rows, detach_rows, temporal_rows


def summarize_layers(layer_rows: list[dict[str, Any]], temporal_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """把逐step统计聚合为逐层统计。"""

    by_key: dict[tuple[float, str], list[dict[str, Any]]] = defaultdict(list)
    for row in layer_rows:
        by_key[(float(row["scale"]), str(row["layer"]))].append(row)
    temporal_map = {(float(r["scale"]), str(r["layer"])): r for r in temporal_rows}
    summary: list[dict[str, Any]] = []
    for (scale, layer), rows in sorted(by_key.items(), key=lambda x: (x[0][0], LAYER_NAMES.index(x[0][1]) if x[0][1] in LAYER_NAMES else 99)):
        arr = {name: np.asarray([float(r[name]) for r in rows], dtype=np.float64) for name in [
            "spike_rate",
            "zero_spike_ratio",
            "high_firing_ratio",
            "mem_mean",
            "mem_std",
            "mem_min",
            "mem_max",
            "mem_abs_p95",
            "mem_abs_p99",
            "finite_ratio",
        ]}
        temporal = temporal_map.get((scale, layer), {})
        summary.append(
            {
                "scale": scale,
                "layer": layer,
                "mean_spike_rate": float(np.nanmean(arr["spike_rate"])),
                "max_spike_rate": float(np.nanmax(arr["spike_rate"])),
                "mean_zero_spike_ratio": float(np.nanmean(arr["zero_spike_ratio"])),
                "mean_high_firing_ratio": float(np.nanmean(arr["high_firing_ratio"])),
                "mean_mem_mean": float(np.nanmean(arr["mem_mean"])),
                "mean_mem_std": float(np.nanmean(arr["mem_std"])),
                "max_mem_abs_p95": float(np.nanmax(arr["mem_abs_p95"])),
                "max_mem_abs_p99": float(np.nanmax(arr["mem_abs_p99"])),
                "mean_finite_ratio": float(np.nanmean(arr["finite_ratio"])),
                "temporal_mean_firing_rate": temporal.get("temporal_mean_firing_rate", float("nan")),
                "persistent_silent_neuron_ratio": temporal.get("persistent_silent_neuron_ratio", float("nan")),
                "persistent_high_firing_neuron_ratio": temporal.get("persistent_high_firing_neuron_ratio", float("nan")),
                "silent_layer_flag": bool(np.nanmean(arr["spike_rate"]) < 1e-4),
                "saturated_layer_flag": bool(np.nanmean(arr["spike_rate"]) > 0.8 or temporal.get("persistent_high_firing_neuron_ratio", 0.0) > 0.5),
            }
        )
    return summary


def summarize_outputs(output_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_scale: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for row in output_rows:
        by_scale[float(row["scale"])].append(row)
    summary: list[dict[str, Any]] = []
    for scale, rows in sorted(by_scale.items()):
        arr = {name: np.asarray([float(r[name]) for r in rows], dtype=np.float64) for name in [
            "output_mean",
            "output_std",
            "output_min",
            "output_max",
            "output_abs_p95",
            "positive_ratio",
            "negative_ratio",
            "finite_ratio",
        ]}
        summary.append(
            {
                "scale": scale,
                "mean_output_mean": float(np.nanmean(arr["output_mean"])),
                "mean_output_std": float(np.nanmean(arr["output_std"])),
                "min_output": float(np.nanmin(arr["output_min"])),
                "max_output": float(np.nanmax(arr["output_max"])),
                "max_output_abs_p95": float(np.nanmax(arr["output_abs_p95"])),
                "mean_positive_ratio": float(np.nanmean(arr["positive_ratio"])),
                "mean_negative_ratio": float(np.nanmean(arr["negative_ratio"])),
                "mean_finite_ratio": float(np.nanmean(arr["finite_ratio"])),
                "near_constant_flag": bool(np.nanmean(arr["output_std"]) < 1e-5),
            }
        )
    return summary


def summarize_scales(layer_summary: list[dict[str, Any]], output_summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """聚合为逐scale总表，并给出可解释的风险计数。"""

    by_scale_layers: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for row in layer_summary:
        by_scale_layers[float(row["scale"])].append(row)
    out_by_scale = {float(row["scale"]): row for row in output_summary}
    rows: list[dict[str, Any]] = []
    for scale, layers in sorted(by_scale_layers.items()):
        output = out_by_scale[scale]
        encoder_high = [
            r
            for r in layers
            if r["layer"] in ENCODER_LAYERS
            and bool(r["saturated_layer_flag"])
        ]
        bottleneck_high = [
            r
            for r in layers
            if r["layer"] in BOTTLENECK_LAYERS
            and bool(r["saturated_layer_flag"])
        ]
        decoder_near_silent = [
            r
            for r in layers
            if r["layer"] in DECODER_LAYERS
            and (float(r["mean_spike_rate"]) < 0.005 or float(r["persistent_silent_neuron_ratio"]) > 0.95)
        ]
        finite_ok = all(float(r["mean_finite_ratio"]) == 1.0 for r in layers) and float(output["mean_finite_ratio"]) == 1.0
        max_mem_abs_p99 = max(float(r["max_mem_abs_p99"]) for r in layers)
        output_const = bool(output["near_constant_flag"])
        mem_abnormal = max_mem_abs_p99 > 10.0
        score = (
            4 * len(bottleneck_high)
            + 3 * len(encoder_high)
            + 2 * len(decoder_near_silent)
            + 10 * int(output_const)
            + 10 * int(mem_abnormal)
            + 10 * int(not finite_ok)
        )
        # 轻微偏好更有输出动态的scale；只作为风险计数相同的排序辅助。
        score = float(score) - min(float(output["mean_output_std"]) * 10.0, 0.2)
        rows.append(
            {
                "scale": scale,
                "mean_spike_rate_all_layers": float(np.mean([float(r["mean_spike_rate"]) for r in layers])),
                "max_spike_rate_all_layers": float(max(float(r["max_spike_rate"]) for r in layers)),
                "silent_layer_count": int(sum(1 for r in layers if bool(r["silent_layer_flag"]))),
                "high_firing_layer_count": int(sum(1 for r in layers if bool(r["saturated_layer_flag"]))),
                "encoder_high_firing_layer_count": int(len(encoder_high)),
                "bottleneck_high_firing_layer_count": int(len(bottleneck_high)),
                "decoder_near_silent_layer_count": int(len(decoder_near_silent)),
                "max_mem_abs_p95": float(max(float(r["max_mem_abs_p95"]) for r in layers)),
                "max_mem_abs_p99": max_mem_abs_p99,
                "mean_mem_std": float(np.mean([float(r["mean_mem_std"]) for r in layers])),
                "output_mean": float(output["mean_output_mean"]),
                "output_std": float(output["mean_output_std"]),
                "output_min": float(output["min_output"]),
                "output_max": float(output["max_output"]),
                "output_positive_ratio": float(output["mean_positive_ratio"]),
                "output_negative_ratio": float(output["mean_negative_ratio"]),
                "finite_ok": bool(finite_ok),
                "mem_abnormal_flag": bool(mem_abnormal),
                "output_near_constant_flag": bool(output_const),
                "selection_score_lower_is_better": score,
            }
        )
    return rows


def recommend_scale(scale_summary: list[dict[str, Any]]) -> tuple[dict[str, Any], str]:
    """按用户给定原则推荐输入scale。"""

    viable = [
        r
        for r in scale_summary
        if r["finite_ok"]
        and not r["mem_abnormal_flag"]
        and not r["output_near_constant_flag"]
        and int(r["encoder_high_firing_layer_count"]) == 0
        and int(r["bottleneck_high_firing_layer_count"]) == 0
        and int(r["decoder_near_silent_layer_count"]) == 0
    ]
    if viable:
        best = sorted(viable, key=lambda r: (float(r["selection_score_lower_is_better"]), -float(r["output_std"])))[0]
        reason = "满足encoder不过高发放、bottleneck不饱和、decoder不近沉默、输出非常数和膜电位不过大的全部约束。"
        return best, reason
    best = sorted(scale_summary, key=lambda r: (float(r["selection_score_lower_is_better"]), int(r["decoder_near_silent_layer_count"]), -float(r["output_std"])))[0]
    reason = "没有scale同时满足全部约束，选择综合风险计数最低且输出动态较好的scale。"
    return best, reason


def run_reset_comparison(
    *,
    events_np: np.ndarray,
    segment: SelectedSegment,
    scales: list[float],
    seed: int,
    device: torch.device,
    blocks: int,
    steps_per_block: int,
) -> list[dict[str, Any]]:
    """比较连续状态和每block reset模式，验证状态管理确实生效。"""

    rows: list[dict[str, Any]] = []
    use_steps = blocks * steps_per_block
    for scale in scales:
        mode_records: dict[str, dict[int, dict[str, list[float]]]] = {}
        for mode in ["continuous", "reset_each_block"]:
            torch.manual_seed(seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(seed)
            model = LIFEVFlowNetTTC().to(device)
            model.eval()
            model.reset_states()
            block_values: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
            with torch.no_grad():
                for local_step in range(use_steps):
                    local_block = local_step // steps_per_block
                    if mode == "reset_each_block" and local_step % steps_per_block == 0:
                        model.reset_states()
                    x = torch.from_numpy(events_np[local_step : local_step + 1]).to(device) * float(scale)
                    pred = model(x)
                    out_s = output_stats(pred)
                    block_values[local_block]["output_mean"].append(out_s["mean"])
                    block_values[local_block]["output_std"].append(out_s["std"])
                    for layer_index, state in enumerate(model.multires_unetrec.states):
                        layer = LAYER_NAMES[layer_index] if layer_index < len(LAYER_NAMES) else f"lif_{layer_index}"
                        mem, spike = state_to_mem_spike(state)
                        if mem is not None:
                            block_values[local_block][f"{layer}:mem_mean"].append(float(mem.mean().cpu()))
                        if spike is not None:
                            block_values[local_block][f"{layer}:spike_rate"].append(float((spike != 0).float().mean().cpu()))
                    if (local_step + 1) % steps_per_block == 0:
                        model.detach_states()
            mode_records[mode] = block_values

        for block in range(blocks):
            row = {
                "scale": scale,
                "comparison_block": block,
                "source_block_id": int(segment.block_indices[block * steps_per_block]),
            }
            cont = mode_records["continuous"][block]
            reset = mode_records["reset_each_block"][block]
            for key in sorted(cont.keys()):
                c = float(np.mean(cont[key]))
                r = float(np.mean(reset[key]))
                row[f"{key}_continuous"] = c
                row[f"{key}_reset_each_block"] = r
                row[f"{key}_abs_diff"] = abs(c - r)
            rows.append(row)
    return rows


def plot_layer_bars(layer_summary: list[dict[str, Any]], out_dir: Path, key: str, title: str, filename: str) -> None:
    layers = LAYER_NAMES
    scales = sorted({float(r["scale"]) for r in layer_summary})
    x = np.arange(len(layers))
    width = 0.8 / max(len(scales), 1)
    fig, ax = plt.subplots(figsize=(13, 5))
    for i, scale in enumerate(scales):
        values = []
        by_layer = {r["layer"]: r for r in layer_summary if float(r["scale"]) == scale}
        for layer in layers:
            values.append(float(by_layer[layer][key]) if layer in by_layer else np.nan)
        ax.bar(x + (i - (len(scales) - 1) / 2) * width, values, width=width, label=f"scale={scale:g}")
    ax.set_xticks(x)
    ax.set_xticklabels(layers, rotation=35, ha="right")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / filename, dpi=160)
    plt.close(fig)


def plot_step_curves(layer_rows: list[dict[str, Any]], out_dir: Path, key: str, title: str, filename: str, steps_per_block: int) -> None:
    scales = sorted({float(r["scale"]) for r in layer_rows})
    fig, axes = plt.subplots(len(scales), 1, figsize=(14, 4 * len(scales)), sharex=True)
    if len(scales) == 1:
        axes = [axes]
    for ax, scale in zip(axes, scales):
        rows = [r for r in layer_rows if float(r["scale"]) == scale]
        for layer in LAYER_NAMES:
            xs = [int(r["step_id"]) for r in rows if r["layer"] == layer]
            ys = [float(r[key]) for r in rows if r["layer"] == layer]
            ax.plot(xs, ys, linewidth=1.0, label=layer)
        for boundary in range(steps_per_block, max(xs) + 1, steps_per_block):
            ax.axvline(boundary, color="k", alpha=0.12, linewidth=0.8)
        ax.set_title(f"{title} | scale={scale:g}")
        ax.grid(alpha=0.25)
        ax.legend(ncol=2, fontsize=8)
    axes[-1].set_xlabel("step")
    fig.tight_layout()
    fig.savefig(out_dir / filename, dpi=160)
    plt.close(fig)


def plot_output_curves(output_rows: list[dict[str, Any]], out_dir: Path, steps_per_block: int) -> None:
    scales = sorted({float(r["scale"]) for r in output_rows})
    fig, axes = plt.subplots(len(scales), 1, figsize=(13, 4 * len(scales)), sharex=True)
    if len(scales) == 1:
        axes = [axes]
    for ax, scale in zip(axes, scales):
        rows = [r for r in output_rows if float(r["scale"]) == scale]
        xs = [int(r["step_id"]) for r in rows]
        ax.plot(xs, [float(r["output_mean"]) for r in rows], label="output_mean")
        ax.plot(xs, [float(r["output_std"]) for r in rows], label="output_std")
        for boundary in range(steps_per_block, max(xs) + 1, steps_per_block):
            ax.axvline(boundary, color="k", alpha=0.12, linewidth=0.8)
        ax.set_title(f"Output mean/std over steps | scale={scale:g}")
        ax.grid(alpha=0.25)
        ax.legend()
    axes[-1].set_xlabel("step")
    fig.tight_layout()
    fig.savefig(out_dir / "输出mean_std随step变化.png", dpi=160)
    plt.close(fig)


def plot_scale_layer_curves(layer_summary: list[dict[str, Any]], out_dir: Path, key: str, title: str, filename: str) -> None:
    """绘制每层统计量随输入scale变化的曲线。"""

    fig, ax = plt.subplots(figsize=(12, 6))
    for layer in LAYER_NAMES:
        rows = sorted([r for r in layer_summary if r["layer"] == layer], key=lambda r: float(r["scale"]))
        ax.plot([float(r["scale"]) for r in rows], [float(r[key]) for r in rows], marker="o", linewidth=1.2, label=layer)
    ax.set_xlabel("input scale")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / filename, dpi=160)
    plt.close(fig)


def plot_scale_summary_curves(scale_summary: list[dict[str, Any]], out_dir: Path) -> None:
    rows = sorted(scale_summary, key=lambda r: float(r["scale"]))
    scales = [float(r["scale"]) for r in rows]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(scales, [int(r["silent_layer_count"]) for r in rows], marker="o", label="silent layer count")
    ax.plot(scales, [int(r["high_firing_layer_count"]) for r in rows], marker="o", label="high firing layer count")
    ax.plot(scales, [int(r["decoder_near_silent_layer_count"]) for r in rows], marker="o", label="decoder near-silent count")
    ax.plot(scales, [int(r["encoder_high_firing_layer_count"]) + int(r["bottleneck_high_firing_layer_count"]) for r in rows], marker="o", label="encoder+bottleneck high count")
    ax.set_xlabel("input scale")
    ax.set_ylabel("layer count")
    ax.set_title("Silent/high-firing layer counts over input scale")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "沉默层数_高发放层数随scale变化.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(scales, [float(r["output_std"]) for r in rows], marker="o")
    ax.set_xlabel("input scale")
    ax.set_ylabel("output std")
    ax.set_title("Output std over input scale")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "输出std随scale变化.png", dpi=160)
    plt.close(fig)


def build_report(
    *,
    out_dir: Path,
    h5_path: Path,
    segment: SelectedSegment,
    layer_summary: list[dict[str, Any]],
    output_summary: list[dict[str, Any]],
    detach_rows: list[dict[str, Any]],
    reset_rows: list[dict[str, Any]],
    input_scales: list[float],
) -> None:
    """生成中文审计报告和缩放建议。"""

    by_scale_layers: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for row in layer_summary:
        by_scale_layers[float(row["scale"])].append(row)
    out_by_scale = {float(r["scale"]): r for r in output_summary}

    scale_rows = []
    recommendations = []
    for scale in input_scales:
        layers = by_scale_layers[scale]
        silent_layers = sum(1 for r in layers if r["silent_layer_flag"])
        saturated_layers = sum(1 for r in layers if r["saturated_layer_flag"])
        max_mem_abs_p99 = max(float(r["max_mem_abs_p99"]) for r in layers)
        out = out_by_scale[scale]
        finite_ok = all(float(r["mean_finite_ratio"]) == 1.0 for r in layers) and float(out["mean_finite_ratio"]) == 1.0
        scale_rows.append(
            [
                f"{scale:g}",
                fmt(np.mean([r["mean_spike_rate"] for r in layers])),
                silent_layers,
                saturated_layers,
                fmt(max_mem_abs_p99),
                fmt(out["mean_output_std"]),
                f"{fmt(out['mean_positive_ratio'])}/{fmt(out['mean_negative_ratio'])}",
                "否" if finite_ok else "是",
            ]
        )
        if saturated_layers > len(layers) // 2:
            recommendations.append(f"- scale={scale:g}：多数层饱和，不建议直接使用。")
        elif saturated_layers > 0:
            recommendations.append(f"- scale={scale:g}：有 {saturated_layers}/{len(layers)} 层触发高发放标记，建议用0.2/0.5补充确认。")
        elif silent_layers > len(layers) // 2:
            recommendations.append(f"- scale={scale:g}：多数层沉默，应优先检查阈值/初始化/输入幅值。")
        elif silent_layers > 0:
            recommendations.append(f"- scale={scale:g}：有 {silent_layers}/{len(layers)} 层接近沉默，继续缩小输入需谨慎。")
        else:
            recommendations.append(f"- scale={scale:g}：未见多数层沉默或饱和。")

    raw_layers = by_scale_layers.get(1.0, [])
    raw_output = out_by_scale.get(1.0, {})
    scaled_layers = by_scale_layers.get(0.1, [])
    raw_silent = sum(1 for r in raw_layers if r["silent_layer_flag"])
    raw_sat = sum(1 for r in raw_layers if r["saturated_layer_flag"])
    scaled_silent = sum(1 for r in scaled_layers if r["silent_layer_flag"])
    scaled_sat = sum(1 for r in scaled_layers if r["saturated_layer_flag"])

    detach_max = max(float(r["max_state_abs_diff_before_after_detach"]) for r in detach_rows) if detach_rows else float("nan")
    reset_diff_cols = [key for key in reset_rows[0].keys() if key.endswith("_abs_diff")] if reset_rows else []
    reset_output_diff = float(np.mean([float(r.get("output_mean_abs_diff", 0.0)) + float(r.get("output_std_abs_diff", 0.0)) for r in reset_rows])) if reset_rows else float("nan")
    reset_has_effect = any(float(r.get(col, 0.0)) > 1e-8 for r in reset_rows for col in reset_diff_cols)

    if raw_sat == 0 and raw_silent <= len(raw_layers) // 2:
        scale_decision = "原始scale=1.0更合理：未见明显饱和；0.1会让整体spike率进一步降低。"
    elif raw_sat > 0 and scaled_sat < raw_sat:
        scale_decision = "原始scale=1.0存在饱和迹象，0.1有缓解；建议继续测试0.2、0.5。"
    elif raw_silent > len(raw_layers) // 2:
        scale_decision = "原始scale=1.0已偏沉默，不应继续缩小输入；需先检查模型/阈值/输入幅值。"
    else:
        scale_decision = "两种scale均未出现硬性异常，建议进入小规模过拟合后再结合loss趋势判断。"

    layer_rows = [
        [
            r["layer"],
            f"{r['scale']:g}",
            fmt(r["mean_spike_rate"]),
            fmt(r["max_spike_rate"]),
            fmt(r["max_mem_abs_p99"]),
            fmt(r["temporal_mean_firing_rate"]),
            fmt(r["persistent_silent_neuron_ratio"]),
            fmt(r["persistent_high_firing_neuron_ratio"]),
            "是" if r["silent_layer_flag"] else "否",
            "是" if r["saturated_layer_flag"] else "否",
        ]
        for r in layer_summary
    ]

    report = f"""# 真实S2 LIF活动审计报告

## 数据与运行
- H5：`{h5_path}`
- 选中step数：`{len(segment.rows)}`
- 选中block范围：`{int(segment.block_indices[0])}` 到 `{int(segment.block_indices[-1])}`
- 时间范围：`{int(segment.t_start[0])}` 到 `{int(segment.t_end[-1])}` us
- 模型：LIF-EV-FlowNet-TTC，基线随机初始化，eval模式，`torch.no_grad()`
- 本次只做forward审计；未计算loss，未backward，未更新参数。

## 输入缩放对比
{markdown_table(["输入scale", "各层平均spike_rate", "沉默层数", "饱和层数", "mem_abs_p99最大", "输出std均值", "输出正/负比例", "NaN/Inf"], scale_rows)}

## 逐层统计
{markdown_table(["层", "scale", "平均spike_rate", "最大spike_rate", "最大mem_abs_p99", "时间平均发放率", "持续沉默比例", "持续高发放比例", "沉默层", "饱和层"], layer_rows)}

## reset与detach检查
- 每10步detach前后状态最大数值差：`{fmt(detach_max, 8)}`，说明detach没有清空或改写状态。
- 每block reset与连续状态模式是否产生差异：`{"是" if reset_has_effect else "否"}`。
- reset/连续模式平均输出差异参考：`{fmt(reset_output_diff, 8)}`。

## 判断
1. 原始10k事件计数下明显沉默：`{"是" if raw_silent > len(raw_layers) // 2 else "否"}`，沉默层数 {raw_silent}/{len(raw_layers)}。
2. 原始10k事件计数下明显饱和：`{"是" if raw_sat > len(raw_layers) // 2 else "否"}`，饱和标记层数 {raw_sat}/{len(raw_layers)}；若未超过半数，仍表示存在局部高发放迹象。
3. 膜电位200步异常累积：`{"否" if all(math.isfinite(float(r['max_mem_abs_p99'])) for r in layer_summary) else "是"}`。
4. detach后状态保留：`{"是" if detach_max == 0.0 else "需复查"}`。
5. 每block reset与连续状态模式产生预期差异：`{"是" if reset_has_effect else "否"}`。
6. 最终输出接近常数：`{"是" if raw_output.get('near_constant_flag', False) else "否"}`。
7. NaN/Inf或异常极值：`{"否" if all(float(r['mean_finite_ratio']) == 1.0 for r in layer_summary) and all(float(r['mean_finite_ratio']) == 1.0 for r in output_summary) else "是"}`。
8. 原始计数和0.1哪个更合理：{scale_decision}
9. 是否需要继续测试0.2/0.5：`{"是" if raw_sat > 0 and scaled_sat < raw_sat else "暂不必要"}`。
10. 是否满足进入32-block过拟合：`{"是" if detach_max == 0.0 and not raw_output.get('near_constant_flag', False) and raw_sat <= len(raw_layers) // 2 else "需先复查"}`；若进入，建议同时保留0.2/0.5输入缩放补测。

## 建议
{chr(10).join(recommendations)}
"""
    (out_dir / "真实S2_LIF活动审计报告.md").write_text(report, encoding="utf-8")


def build_scale_scan_report(
    *,
    out_dir: Path,
    h5_path: Path,
    segment: SelectedSegment,
    layer_summary: list[dict[str, Any]],
    scale_summary: list[dict[str, Any]],
) -> None:
    """生成专门面向0.1到1.0输入缩放扫描的中文报告。"""

    best, reason = recommend_scale(scale_summary)
    scale_rows = [
        [
            fmt(r["scale"], 1),
            fmt(r["mean_spike_rate_all_layers"]),
            fmt(r["max_spike_rate_all_layers"]),
            int(r["encoder_high_firing_layer_count"]),
            int(r["bottleneck_high_firing_layer_count"]),
            int(r["decoder_near_silent_layer_count"]),
            int(r["silent_layer_count"]),
            int(r["high_firing_layer_count"]),
            fmt(r["max_mem_abs_p95"]),
            fmt(r["max_mem_abs_p99"]),
            fmt(r["output_std"]),
            f"{fmt(r['output_positive_ratio'])}/{fmt(r['output_negative_ratio'])}",
            "是" if r["finite_ok"] else "否",
            fmt(r["selection_score_lower_is_better"]),
        ]
        for r in sorted(scale_summary, key=lambda x: float(x["scale"]))
    ]
    selected_layers = [r for r in layer_summary if abs(float(r["scale"]) - float(best["scale"])) < 1e-9]
    selected_rows = [
        [
            r["layer"],
            fmt(r["mean_spike_rate"]),
            fmt(r["max_spike_rate"]),
            fmt(r["persistent_silent_neuron_ratio"]),
            fmt(r["persistent_high_firing_neuron_ratio"]),
            fmt(r["mean_mem_mean"]),
            fmt(r["mean_mem_std"]),
            fmt(r["max_mem_abs_p95"]),
            fmt(r["max_mem_abs_p99"]),
        ]
        for r in selected_layers
    ]
    report = f"""# 真实S2 输入缩放扫描报告

## 扫描设置
- H5：`{h5_path}`
- 使用同一批连续step：`{len(segment.rows)}` 个，block `{int(segment.block_indices[0])}` 到 `{int(segment.block_indices[-1])}`
- 每个scale重新实例化模型，并用相同seed得到同一初始权重。
- 状态管理：第一个step前 `reset_states()`；每10步只 `detach_states()`；连续block之间不reset。
- 运行模式：eval + `torch.no_grad()`；未计算loss，未backward，未更新参数。

## 逐scale汇总
{markdown_table(["scale", "平均spike_rate", "最大spike_rate", "encoder高发放层", "bottleneck高发放层", "decoder近沉默层", "沉默层", "高发放层", "max mem p95", "max mem p99", "输出std", "输出正/负比例", "finite", "评分"], scale_rows)}

## 推荐scale
- 推荐：`scale={fmt(best["scale"], 1)}`
- 原因：{reason}
- 该scale输出std：`{fmt(best["output_std"])}`，max mem_abs_p99：`{fmt(best["max_mem_abs_p99"])}`。
- 该scale encoder高发放层：`{int(best["encoder_high_firing_layer_count"])}`，bottleneck高发放层：`{int(best["bottleneck_high_firing_layer_count"])}`，decoder近沉默层：`{int(best["decoder_near_silent_layer_count"])}`。

## 推荐scale逐层细节
{markdown_table(["层", "平均spike_rate", "最大spike_rate", "持续沉默比例", "持续高发放比例", "mem_mean", "mem_std", "max mem p95", "max mem p99"], selected_rows)}

## 判断依据
- encoder不能大面积持续高发放：优先要求 encoder 高发放层数为0。
- bottleneck不能饱和：优先要求 bottleneck 高发放层数为0。
- decoder不能接近沉默：优先要求 decoder 近沉默层数为0。
- 输出不能接近常数：要求输出std不触发 near-constant 标记。
- 膜电位不能异常增大：要求 `max_mem_abs_p99 <= 10` 且无NaN/Inf。

## 曲线图
- `各层spike_rate随scale变化.png`
- `各层mem_abs_p99随scale变化.png`
- `沉默层数_高发放层数随scale变化.png`
- `输出std随scale变化.png`
"""
    (out_dir / "真实S2_输入缩放扫描报告.md").write_text(report, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", type=Path, default=DEFAULT_H5)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--num-blocks", type=int, default=20)
    parser.add_argument("--steps-per-block", type=int, default=10)
    parser.add_argument("--input-scales", type=float, nargs="+", default=[1.0, 0.1])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true", help="保留已有输出目录；本脚本会覆盖同名审计文件。")
    args = parser.parse_args()

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    segment = choose_continuous_segment(args.h5, args.num_blocks, args.steps_per_block)
    events_np = load_events(args.h5, segment.rows)

    all_layer_rows: list[dict[str, Any]] = []
    all_output_rows: list[dict[str, Any]] = []
    all_detach_rows: list[dict[str, Any]] = []
    all_temporal_rows: list[dict[str, Any]] = []
    for scale in args.input_scales:
        layer_rows, output_rows, detach_rows, temporal_rows = run_forward_audit(
            events_np=events_np,
            segment=segment,
            scale=float(scale),
            seed=args.seed,
            device=device,
            steps_per_block=args.steps_per_block,
        )
        all_layer_rows.extend(layer_rows)
        all_output_rows.extend(output_rows)
        all_detach_rows.extend(detach_rows)
        all_temporal_rows.extend(temporal_rows)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    layer_summary = summarize_layers(all_layer_rows, all_temporal_rows)
    output_summary = summarize_outputs(all_output_rows)
    scale_summary = summarize_scales(layer_summary, output_summary)
    reset_rows = run_reset_comparison(
        events_np=events_np,
        segment=segment,
        scales=[float(s) for s in args.input_scales],
        seed=args.seed,
        device=device,
        blocks=min(5, args.num_blocks),
        steps_per_block=args.steps_per_block,
    )

    write_csv(args.out_dir / "真实S2_LIF逐step统计.csv", all_layer_rows)
    write_csv(args.out_dir / "真实S2_LIF逐层统计.csv", layer_summary)
    write_csv(args.out_dir / "真实S2_输入缩放逐scale汇总.csv", scale_summary)
    write_csv(args.out_dir / "真实S2_输出统计.csv", all_output_rows)
    write_csv(args.out_dir / "真实S2_输出汇总.csv", output_summary)
    write_csv(args.out_dir / "真实S2_连续状态与每block_reset对比.csv", reset_rows)
    write_csv(args.out_dir / "真实S2_detach边界状态检查.csv", all_detach_rows)
    write_json(
        args.out_dir / "真实S2_LIF活动审计summary.json",
        {
            "h5": str(args.h5),
            "device": str(device),
            "selected_rows": segment.rows.tolist(),
            "input_scales": [float(s) for s in args.input_scales],
            "layer_summary": layer_summary,
            "scale_summary": scale_summary,
            "recommended_scale": recommend_scale(scale_summary)[0],
            "output_summary": output_summary,
            "detach": all_detach_rows,
        },
    )

    plot_layer_bars(layer_summary, args.out_dir, "mean_spike_rate", "Mean spike_rate by LIF layer", "各层平均spike_rate对比图.png")
    plot_layer_bars(layer_summary, args.out_dir, "max_mem_abs_p99", "Max mem_abs_p99 by LIF layer", "各层mem_abs_p99对比图.png")
    plot_step_curves(all_layer_rows, args.out_dir, "spike_rate", "spike_rate over steps", "spike_rate随step变化图.png", args.steps_per_block)
    plot_step_curves(all_layer_rows, args.out_dir, "mem_mean", "mem_mean over steps", "mem_mean随step变化图.png", args.steps_per_block)
    plot_step_curves(all_layer_rows, args.out_dir, "mem_abs_p99", "mem_abs_p99 over steps", "mem_abs_p99随step变化图.png", args.steps_per_block)
    plot_output_curves(all_output_rows, args.out_dir, args.steps_per_block)
    plot_scale_layer_curves(layer_summary, args.out_dir, "mean_spike_rate", "Layer mean spike_rate over input scale", "各层spike_rate随scale变化.png")
    plot_scale_layer_curves(layer_summary, args.out_dir, "max_mem_abs_p99", "Layer mem_abs_p99 over input scale", "各层mem_abs_p99随scale变化.png")
    plot_scale_summary_curves(scale_summary, args.out_dir)

    build_report(
        out_dir=args.out_dir,
        h5_path=args.h5,
        segment=segment,
        layer_summary=layer_summary,
        output_summary=output_summary,
        detach_rows=all_detach_rows,
        reset_rows=reset_rows,
        input_scales=[float(s) for s in args.input_scales],
    )
    build_scale_scan_report(
        out_dir=args.out_dir,
        h5_path=args.h5,
        segment=segment,
        layer_summary=layer_summary,
        scale_summary=scale_summary,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
