#!/usr/bin/env python3
"""统一评估 ANN EV-Slim、Hybrid SNN EV-Slim 与 LIF-EV-FlowNet-TTC 的资源画像。

本工具只读取既有 checkpoint 和固定 H5 样本；不训练、不改数据、不改网络结构。
SNN 的一次测量始终覆盖完整时间窗口，并在每次推理前重置状态。
"""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import math
import os
import statistics
import subprocess
import sys
import threading
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Iterable

import h5py
import hdf5plugin  # noqa: F401  # 注册 H5 压缩插件。
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402


ROOT = Path(__file__).resolve().parents[3]
EVTTC_MODEL_ROOT = ROOT / "EV-TTC-main" / "model"
SNN_ROOT = ROOT / "EV-TTC-SNN-main"
for import_root in (EVTTC_MODEL_ROOT, SNN_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from evslim import EVSlim  # noqa: E402
from hybrid_snn_evslim import HybridSNNEVSlim  # noqa: E402
from snn_ttc.models import LIFEVFlowNetTTC, LIFEVFlowNetTTCConfig  # noqa: E402
from data.ttc_temporal_dm import TTCEFTemporalDataset  # noqa: E402


DEFAULT_REPORT_DIR = SNN_ROOT / "reports" / "算法资源评估"
DEFAULT_OFFICIAL_ROOT = SNN_ROOT / "debug_sets" / "[13]official_car_single_sequence_evttc_reproduction"
DEFAULT_SNN_H5 = SNN_ROOT / "debug_sets" / "[5]car_spatial_ablation" / "independent_fixed_n" / "N10000_S2_center256_to128.h5"
DEFAULT_SNN_CKPT = Path(
    "/data/evttc_storage/event+SNN+TTC/EV-TTC-SNN-main/debug_sets_offloaded/"
    "[10]lif_evflownet_ttc_s2_n10k_500block_pipeline/checkpoints/best_val_mae.pt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="三类稠密感知模型资源评估")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--ann-checkpoint", type=Path, default=DEFAULT_OFFICIAL_ROOT / "EVTTC_Official_Car_20ep/best_val_loss_20ep.ckpt")
    parser.add_argument("--ann-val-h5", type=Path, default=DEFAULT_OFFICIAL_ROOT / "data/val.h5")
    parser.add_argument("--hybrid-checkpoint", type=Path, default=SNN_ROOT / "reports/06_真实数据端到端验证/checkpoints/best.pt")
    parser.add_argument("--hybrid-val-h5", type=Path, default=DEFAULT_OFFICIAL_ROOT / "data/val.h5")
    parser.add_argument("--lif-checkpoint", type=Path, default=DEFAULT_SNN_CKPT)
    parser.add_argument("--lif-h5", type=Path, default=DEFAULT_SNN_H5)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--power-seconds", type=float, default=60.0)
    parser.add_argument("--power-repeats", type=int, default=3)
    parser.add_argument("--skip-power", action="store_true", help="仅用于调试；正式评估不要启用")
    return parser.parse_args()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    names = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=names, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def gib(value: float) -> float:
    return value / 1024**3


def mib(value: float) -> float:
    return value / 1024**2


def tensor_bytes(tensor: torch.Tensor | None) -> int:
    return 0 if tensor is None else tensor.numel() * tensor.element_size()


def flatten_tensors(value: Any) -> list[torch.Tensor]:
    if torch.is_tensor(value):
        return [value]
    if isinstance(value, (tuple, list)):
        return [tensor for item in value for tensor in flatten_tensors(item)]
    if isinstance(value, dict):
        return [tensor for item in value.values() for tensor in flatten_tensors(item)]
    return []


def evslim_cfg() -> SimpleNamespace:
    """与官方 car 单序列复现实验严格相同的网络配置。"""

    return SimpleNamespace(
        enc_k_size=[3, 3, 3], enc_channels=[16, 16, 16], aspp_channels=[32, 16],
        rate=[12, 24, 36], dec_k_size=[3, 3, 3], dec_channels=[8, 8, 1],
        norm=True, in_channels=6, input=6, ds="TTCEF", task="ttc", act="ReLU",
        alpha=0.45, flip_prob=0.3, res=[360, 360],
    )


class ModuleMacProfiler:
    """用实际前向输出形状统计卷积/反卷积/线性层 MAC，不估计自定义 LIF 算子。"""

    def __init__(self, model: nn.Module) -> None:
        self.rows: list[dict[str, Any]] = []
        self.handles: list[Any] = []
        for name, module in model.named_modules():
            if not name:
                continue
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                self.handles.append(module.register_forward_hook(self._hook(name)))

    def _hook(self, name: str) -> Callable[..., None]:
        def handler(module: nn.Module, inputs: tuple[Any, ...], output: Any) -> None:
            out = output[0] if isinstance(output, (tuple, list)) else output
            inp = inputs[0]
            macs = 0
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)) and torch.is_tensor(out):
                batch, channels_out, height, width = out.shape
                kh, kw = module.kernel_size
                macs = batch * height * width * channels_out * (module.in_channels // module.groups) * kh * kw
            elif isinstance(module, nn.Linear) and torch.is_tensor(out):
                macs = out.numel() * module.in_features
            self.rows.append({
                "layer": name,
                "operator": type(module).__name__,
                "input_shape": list(inp.shape) if torch.is_tensor(inp) else [],
                "output_shape": list(out.shape) if torch.is_tensor(out) else [],
                "macs": int(macs),
            })
        return handler

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()

    def aggregate(self) -> list[dict[str, Any]]:
        grouped: dict[tuple[str, str, str, str], dict[str, Any]] = {}
        for row in self.rows:
            key = (row["layer"], row["operator"], str(row["input_shape"]), str(row["output_shape"]))
            item = grouped.setdefault(key, {**row, "calls": 0, "macs_total": 0})
            item["calls"] += 1
            item["macs_total"] += row["macs"]
        return sorted(grouped.values(), key=lambda item: int(item["macs_total"]), reverse=True)


class NvidiaSmiSampler:
    """借助一个持续运行的 nvidia-smi 进程以约100 ms间隔采样桌面 GPU 参考功耗。"""

    def __init__(self, interval_ms: int = 100) -> None:
        self.interval_ms = interval_ms
        self.records: list[dict[str, float]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._proc: subprocess.Popen[str] | None = None

    def start(self) -> None:
        command = [
            "nvidia-smi", "--query-gpu=power.draw,utilization.gpu,memory.used,temperature.gpu,clocks.sm",
            "--format=csv,noheader,nounits", "-lms", str(self.interval_ms),
        ]
        self._proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)

        def consume() -> None:
            assert self._proc is not None and self._proc.stdout is not None
            for line in self._proc.stdout:
                if self._stop.is_set():
                    break
                parts = [part.strip() for part in line.split(",")]
                if len(parts) != 5:
                    continue
                try:
                    power, util, memory, temp, clock = (float(part) for part in parts)
                except ValueError:
                    continue
                self.records.append({
                    "time_s": time.monotonic(), "power_w": power, "gpu_util_pct": util,
                    "memory_mib": memory, "temperature_c": temp, "sm_clock_mhz": clock,
                })
        self._thread = threading.Thread(target=consume, daemon=True)
        self._thread.start()

    def stop(self) -> list[dict[str, float]]:
        self._stop.set()
        if self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        if self._thread is not None:
            self._thread.join(timeout=3)
        return self.records


class BenchmarkCase:
    """把三种输入接口统一成“完整一次推理”的无参闭包。"""

    def __init__(self, name: str, model: nn.Module, input_tensor: torch.Tensor, run: Callable[[bool], torch.Tensor], checkpoint: Path, description: dict[str, Any]) -> None:
        self.name = name
        self.model = model
        self.input_tensor = input_tensor
        self.run = run
        self.checkpoint = checkpoint
        self.description = description


def activate_case(case: BenchmarkCase, device: torch.device) -> None:
    """只把当前被测模型与输入放到GPU，避免其他模型权重污染峰值显存统计。"""

    case.model.to(device).eval()
    case.input_tensor = case.input_tensor.to(device)


def offload_case(case: BenchmarkCase) -> None:
    """每项测量结束后把模型和输入移回CPU，给下一模型一个干净的显存起点。"""

    case.model.to("cpu")
    case.input_tensor = case.input_tensor.to("cpu")
    torch.cuda.empty_cache()


def load_cases(args: argparse.Namespace, device: torch.device) -> list[BenchmarkCase]:
    """加载真实 checkpoint，并取每类模型训练时使用的数据格式作为固定样本。"""

    ann = EVSlim(evslim_cfg()).eval()
    ann_ckpt = torch.load(args.ann_checkpoint, map_location="cpu", weights_only=False)
    ann.load_state_dict(ann_ckpt["model_state_dict"])
    with h5py.File(args.ann_val_h5, "r") as handle:
        ann_x = torch.from_numpy(handle["exp_filts"][0].astype(np.float32))[None]
    ann_case: BenchmarkCase
    def run_ann(amp: bool) -> torch.Tensor:
        return ann(ann_case.input_tensor)
    ann_case = BenchmarkCase(
        "官方 ANN EV-Slim", ann, ann_x, run_ann, args.ann_checkpoint,
        {"input_shape": list(ann_x.shape), "time_steps": 1, "input_normalization": "H5 exp_filts直接输入", "state_policy": "无状态"},
    )

    hybrid = HybridSNNEVSlim().eval()
    hybrid_ckpt = torch.load(args.hybrid_checkpoint, map_location="cpu", weights_only=False)
    hybrid.load_state_dict(hybrid_ckpt["model_state_dict"])
    hybrid_dataset = TTCEFTemporalDataset(args.hybrid_val_h5, window_length=3, window_stride=3, augment=False)
    hybrid_x = hybrid_dataset[0]["exp_filts"][None]
    hybrid_dataset.close()
    hybrid_case: BenchmarkCase
    def run_hybrid(amp: bool) -> torch.Tensor:
        hybrid.reset_states()
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            return hybrid.forward_sequence(hybrid_case.input_tensor)
    hybrid_case = BenchmarkCase(
        "Hybrid SNN EV-Slim", hybrid, hybrid_x, run_hybrid, args.hybrid_checkpoint,
        {"input_shape": list(hybrid_x.shape), "time_steps": int(hybrid_x.shape[1]), "input_normalization": "H5 exp_filts直接输入", "state_policy": "每个3-step窗口前reset_states"},
    )

    lif_ckpt = torch.load(args.lif_checkpoint, map_location="cpu", weights_only=False)
    lif = LIFEVFlowNetTTC(LIFEVFlowNetTTCConfig(**lif_ckpt["model_config"])).eval()
    lif.load_state_dict(lif_ckpt["model_state_dict"])
    with h5py.File(args.lif_h5, "r") as handle:
        # 500-block 管线使用连续10个step并在block起点重置状态。
        lif_x = torch.from_numpy(handle["event_cnt"][0:10].astype(np.float32))[None]
    lif_scale = float(lif_ckpt["input_scale"])
    lif_case: BenchmarkCase
    def run_lif(amp: bool) -> torch.Tensor:
        lif.reset_states()
        prediction: torch.Tensor | None = None
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            for step in range(lif_case.input_tensor.shape[1]):
                prediction = lif(lif_case.input_tensor[:, step] * lif_scale)
        assert prediction is not None
        return prediction
    lif_case = BenchmarkCase(
        "MAVLab LIF-EV-FlowNet-TTC", lif, lif_x, run_lif, args.lif_checkpoint,
        {"input_shape": list(lif_x.shape), "time_steps": int(lif_x.shape[1]), "input_normalization": f"event_cnt * {lif_scale}", "state_policy": "每个10-step block前reset_states，block末detach但本测量无反传"},
    )
    return [ann_case, hybrid_case, lif_case]


def model_state_tensors(case: BenchmarkCase) -> list[tuple[str, torch.Tensor]]:
    model = case.model
    if isinstance(model, HybridSNNEVSlim):
        raw = {"lif1": model._lif1_state, "lif2": model._lif2_state, "accumulator": model.accumulator}
        return [(name, tensor) for name, tensor in raw.items() if torch.is_tensor(tensor)]
    if isinstance(model, LIFEVFlowNetTTC):
        return [(f"state_{index}", tensor) for index, tensor in enumerate(flatten_tensors(model.states))]
    return []


def collect_snn_activity(case: BenchmarkCase) -> list[dict[str, Any]]:
    """用相同固定序列记录SNN每层的平均脉冲率、膜状态与理论SynOps代理量。"""

    rows: list[dict[str, Any]] = []
    model = case.model
    if isinstance(model, HybridSNNEVSlim):
        model.reset_states()
        with torch.inference_mode():
            model.forward_sequence(case.input_tensor)
        rates = model.get_firing_rates()
        shapes = model.last_step_shapes
        for name, shape_key, downstream in (("lif1", "lif1_spikes", model.conv2), ("lif2", "lif2_spikes", model.conv3)):
            shape = shapes.get(shape_key, ())
            neurons_per_step = int(np.prod(shape)) if shape else 0
            rate = float(rates.get(name, 0.0))
            # SynOps代理：实际输出spike数 × 下一层每个输入通道的卷积扇出。
            spikes = rate * neurons_per_step * case.input_tensor.shape[1]
            kh, kw = downstream.kernel_size
            fanout = downstream.out_channels * kh * kw / downstream.groups
            rows.append({
                "model": case.name, "layer": name, "time_steps": int(case.input_tensor.shape[1]),
                "neurons_per_step": neurons_per_step, "mean_spike_rate": rate,
                "estimated_total_spikes": spikes, "synops_proxy": spikes * fanout,
                "state_update_count": neurons_per_step * int(case.input_tensor.shape[1]),
                "membrane_mean": float("nan"), "membrane_std": float("nan"),
            })
    elif isinstance(model, LIFEVFlowNetTTC):
        # 直接读取每个 ConvLIF/ConvLIFRecurrent 返回状态中的 [v_out, z_out]，
        # 避免把父层堆叠状态的第0维误认作脉冲维度。
        per_layer: dict[str, list[dict[str, float]]] = defaultdict(list)
        handles: list[Any] = []
        for name, module in model.named_modules():
            if type(module).__name__ not in {"ConvLIF", "ConvLIFRecurrent"}:
                continue
            def hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any, layer_name: str = name) -> None:
                if not isinstance(output, tuple) or len(output) < 2 or not torch.is_tensor(output[1]):
                    return
                state = output[1]
                if state.ndim < 1 or state.shape[0] < 2:
                    return
                membrane, spikes = state[0].detach(), state[1].detach()
                per_layer[layer_name].append({
                    "spike_rate": float(spikes.mean().cpu()), "spike_count": float(spikes.sum().cpu()),
                    "neurons": float(spikes.numel()), "membrane_mean": float(membrane.mean().cpu()),
                    "membrane_std": float(membrane.std(unbiased=False).cpu()),
                })
            handles.append(module.register_forward_hook(hook))
        with torch.inference_mode():
            case.run(False)
        for handle in handles:
            handle.remove()
        for name, values in sorted(per_layer.items()):
            rows.append({
                "model": case.name, "layer": name, "time_steps": len(values),
                "neurons_per_step": int(np.median([row["neurons"] for row in values])),
                "mean_spike_rate": float(np.mean([row["spike_rate"] for row in values])),
                "estimated_total_spikes": float(np.sum([row["spike_count"] for row in values])),
                # 递归、残差和concat路径使每层扇出不唯一；保持NaN而非猜测SynOps。
                "synops_proxy": float("nan"), "state_update_count": int(np.sum([row["neurons"] for row in values])),
                "membrane_mean": float(np.mean([row["membrane_mean"] for row in values])),
                "membrane_std": float(np.mean([row["membrane_std"] for row in values])),
            })
        model.reset_states()
    return rows


def profile_static_and_memory(case: BenchmarkCase, device: torch.device) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    profiler = ModuleMacProfiler(case.model)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        output = case.run(False)
    torch.cuda.synchronize(device)
    profiler.close()
    layers = profiler.aggregate()
    param_total = sum(parameter.numel() for parameter in case.model.parameters())
    param_trainable = sum(parameter.numel() for parameter in case.model.parameters() if parameter.requires_grad)
    state = model_state_tensors(case)
    state_bytes = sum(tensor_bytes(tensor) for _, tensor in state)
    op_counts = Counter(type(module).__name__ for module in case.model.modules())
    macs = int(sum(int(row["macs_total"]) for row in layers))
    custom = sorted({type(module).__name__ for module in case.model.modules() if "lif" in type(module).__name__.lower() or "spik" in type(module).__name__.lower()})
    complexity = {
        "model": case.name, "parameter_count": param_total, "trainable_parameter_count": param_trainable,
        "weights_fp32_mib": mib(param_total * 4), "weights_fp16_mib": mib(param_total * 2), "weights_int8_mib": mib(param_total),
        "checkpoint_mib": mib(case.checkpoint.stat().st_size), "full_inference_macs": macs,
        "full_inference_gmacs": macs / 1e9, "full_inference_gflops": (macs * 2) / 1e9,
        "time_steps": int(case.description["time_steps"]), "input_shape": str(tuple(case.input_tensor.shape)),
        "output_shape": str(tuple(output.shape)), "operator_types": json.dumps(dict(op_counts), ensure_ascii=False),
        "custom_or_stateful_operators": "; ".join(custom) if custom else "无",
    }
    memory = {
        "model": case.name, "parameter_mib": mib(param_total * 4), "input_mib": mib(tensor_bytes(case.input_tensor)),
        "output_mib": mib(tensor_bytes(output)), "snn_state_mib": mib(state_bytes),
        "peak_allocated_mib": mib(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_mib": mib(torch.cuda.max_memory_reserved(device)),
    }
    state_detail = {name: {"shape": list(tensor.shape), "mib": mib(tensor_bytes(tensor))} for name, tensor in state}
    memory["state_tensors"] = json.dumps(state_detail, ensure_ascii=False)
    return complexity, layers, memory


def latency_profile(case: BenchmarkCase, device: torch.device, warmup: int, iterations: int, amp: bool) -> dict[str, Any]:
    """CUDA Event测量完整一次推理，SNN闭包内包含reset和全部时间步。"""

    precision = "fp16_amp" if amp else "fp32"
    torch.cuda.empty_cache()
    with torch.inference_mode():
        start = time.perf_counter()
        case.run(amp)
        torch.cuda.synchronize(device)
        first_ms = (time.perf_counter() - start) * 1000.0
        for _ in range(warmup):
            case.run(amp)
        torch.cuda.synchronize(device)
        start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        samples: list[float] = []
        for _ in range(iterations):
            start_event.record()
            case.run(amp)
            end_event.record()
            end_event.synchronize()
            samples.append(float(start_event.elapsed_time(end_event)))
    values = np.asarray(samples, dtype=np.float64)
    return {
        "model": case.name, "precision": precision, "warmup": warmup, "iterations": iterations,
        "first_inference_ms": first_ms, "mean_ms": float(values.mean()), "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)), "p99_ms": float(np.percentile(values, 99)),
        "fps": float(1000.0 / values.mean()),
    }


def integrate_energy(samples: list[dict[str, float]], idle_w: float, inference_count: int) -> tuple[float, float]:
    if len(samples) < 2 or inference_count < 1:
        return float("nan"), float("nan")
    times = np.asarray([row["time_s"] for row in samples], dtype=np.float64)
    powers = np.asarray([max(0.0, row["power_w"] - idle_w) for row in samples], dtype=np.float64)
    energy_j = float(np.trapz(powers, times))
    return energy_j, energy_j * 1000.0 / inference_count


def power_profile(case: BenchmarkCase, device: torch.device, seconds: float, repeats: int, idle_w: float, raw_dir: Path) -> list[dict[str, Any]]:
    """FP16连续完整推理的4090相对功耗测试；不外推到嵌入式平台。"""

    rows: list[dict[str, Any]] = []
    for repeat in range(1, repeats + 1):
        torch.cuda.empty_cache()
        sampler = NvidiaSmiSampler(interval_ms=100)
        sampler.start()
        time.sleep(0.2)
        count = 0
        started = time.monotonic()
        with torch.inference_mode():
            while time.monotonic() - started < seconds:
                case.run(True)
                count += 1
        torch.cuda.synchronize(device)
        elapsed = time.monotonic() - started
        records = sampler.stop()
        energy_j, energy_mj = integrate_energy(records, idle_w, count)
        if records:
            mean_power = float(np.mean([item["power_w"] for item in records]))
            mean_util = float(np.mean([item["gpu_util_pct"] for item in records]))
            peak_memory = float(np.max([item["memory_mib"] for item in records]))
            mean_temp = float(np.mean([item["temperature_c"] for item in records]))
        else:
            mean_power = mean_util = peak_memory = mean_temp = float("nan")
        write_json(raw_dir / f"power_{case.name.replace(' ', '_').replace('/', '_')}_repeat{repeat}.json", records)
        rows.append({
            "model": case.name, "precision": "fp16_amp", "repeat": repeat, "duration_s": elapsed,
            "inference_count": count, "mean_power_w": mean_power, "dynamic_power_w": mean_power - idle_w,
            "energy_dynamic_j": energy_j, "energy_dynamic_mj_per_inference": energy_mj,
            "mean_gpu_util_pct": mean_util, "peak_memory_mib": peak_memory, "mean_temperature_c": mean_temp,
            "sample_count": len(records), "sampling_interval_target_ms": 100,
        })
    return rows


def make_inventory(cases: Iterable[BenchmarkCase]) -> str:
    lines = ["# 模型清单", "", "本评估只读取既有 checkpoint 和固定测试样本。", ""]
    for case in cases:
        description = case.description
        if isinstance(case.model, HybridSNNEVSlim):
            operator_note = "自定义/状态算子：实际为两层 LIF（lif1、lif2）加第三层实值累加器；ASPP 与 decoder 仅在序列末执行一次。"
        elif isinstance(case.model, LIFEVFlowNetTTC):
            operator_note = "自定义/状态算子：ConvLIF、ConvLIFRecurrent、脉冲残差块和脉冲上采样块；完整10步都会更新状态。"
        else:
            operator_note = "自定义/状态算子：无；使用 Conv、BatchNorm、ReLU、ASPP 和普通上采样/卷积层。"
        lines += [
            f"## {case.name}", "",
            f"- 模型入口：`{inspect.getfile(type(case.model))}`，`{type(case.model).__module__}.{type(case.model).__name__}`",
            f"- checkpoint：`{case.checkpoint}`",
            f"- checkpoint 精度：FP32 权重；运行时分别测 FP32 与 CUDA AMP FP16。",
            f"- 输入：`{description['input_shape']}`，{description['input_normalization']}",
            f"- 输出：单通道稠密 TTC / signed inverse TTC map，形状由实际前向统计。",
            f"- 时间步：`{description['time_steps']}`；状态管理：{description['state_policy']}",
            f"- {operator_note}",
            "- 推理入口：本资源评估脚本；数据格式与原训练/验证入口一致。",
            "",
        ]
    return "\n".join(lines)


def hardware_rows(complexities: list[dict[str, Any]], memories: list[dict[str, Any]]) -> list[dict[str, Any]]:
    memory_by_name = {row["model"]: row for row in memories}
    rows: list[dict[str, Any]] = []
    for item in complexities:
        gmac = float(item["full_inference_gmacs"])
        mem = memory_by_name[item["model"]]
        peak_mib = float(mem["peak_allocated_mib"])
        state_mib = float(mem["snn_state_mib"])
        # 这里的建议RAM是权重、峰值运行张量和2倍工程缓冲的保守和，不等同于显卡显存。
        minimum_ram_mib = max(512.0, 2.0 * (float(mem["parameter_mib"]) + peak_mib + state_mib))
        if "LIF" in item["model"]:
            level, risk = "FPGA/类脑专用映射", "PyTorch稠密仿真不具事件稀疏加速；递归LIF/自定义状态算子需部署适配"
        elif "Hybrid" in item["model"]:
            level, risk = "Jetson Orin Nano级", "LIF状态与ANN ASPP混合；NPU算子兼容性需要验证"
        else:
            level, risk = "Jetson Orin Nano级", "ASPP空洞卷积、360分辨率和内存带宽可能限制实际FPS"
        rows.append({
            "model": item["model"], "gmac_per_inference": gmac,
            "30hz_gmac_s": gmac * 30, "60hz_gmac_s": gmac * 60, "100hz_gmac_s": gmac * 100,
            "recommended_hardware_gmac_s_2x": gmac * 60 * 2, "recommended_hardware_gmac_s_4x": gmac * 60 * 4,
            "minimum_weight_storage_fp16_mib": item["weights_fp16_mib"], "estimated_minimum_ram_mib": minimum_ram_mib,
            "resource_level": level, "main_risk": risk,
        })
    return rows


def plot_figures(out_dir: Path, latency: list[dict[str, Any]], complexity: list[dict[str, Any]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fp16 = [row for row in latency if row["precision"] == "fp16_amp"]
    if fp16:
        fig, axis = plt.subplots(figsize=(8, 4))
        labels = ["ANN EV-Slim", "Hybrid SNN", "LIF-EV-FlowNet"]
        axis.bar(labels[:len(fp16)], [row["mean_ms"] for row in fp16])
        axis.set_ylabel("Full inference latency (ms)")
        axis.tick_params(axis="x", rotation=18)
        fig.tight_layout()
        fig.savefig(out_dir / "fp16_latency.png", dpi=150)
        plt.close(fig)
    fig, axis = plt.subplots(figsize=(8, 4))
    labels = ["ANN EV-Slim", "Hybrid SNN", "LIF-EV-FlowNet"]
    axis.bar(labels[:len(complexity)], [row["full_inference_gmacs"] for row in complexity])
    axis.set_ylabel("Full inference GMACs")
    axis.tick_params(axis="x", rotation=18)
    fig.tight_layout()
    fig.savefig(out_dir / "full_inference_gmacs.png", dpi=150)
    plt.close(fig)


def summary_markdown(complexity: list[dict[str, Any]], memory: list[dict[str, Any]], latency: list[dict[str, Any]], power: list[dict[str, Any]], activity: list[dict[str, Any]], hardware: list[dict[str, Any]]) -> str:
    by_memory = {row["model"]: row for row in memory}
    fp16_latency = {row["model"]: row for row in latency if row["precision"] == "fp16_amp"}
    power_by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in power:
        power_by_model[row["model"]].append(row)
    activity_by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in activity:
        activity_by_model[row["model"]].append(row)
    lines = ["# 算法资源评估摘要", "", "RTX 4090 数据仅用于三种算法的同平台相对比较，不能直接换算为 Orin、RK3588、FPGA 或类脑芯片功耗。", "", "| 模型 | 参数量 | MACs | 完整SNN仿真MACs | 峰值显存 | FP16延迟 | 动态功率 | 能量/次 | 平均发放率 | SynOps |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for item in complexity:
        name = item["model"]
        mem = by_memory[name]
        lat = fp16_latency.get(name, {})
        power_rows = power_by_model.get(name, [])
        act = activity_by_model.get(name, [])
        avg_power = float(np.nanmean([row["dynamic_power_w"] for row in power_rows])) if power_rows else float("nan")
        avg_energy = float(np.nanmean([row["energy_dynamic_mj_per_inference"] for row in power_rows])) if power_rows else float("nan")
        avg_rate = float(np.nanmean([row["mean_spike_rate"] for row in act])) if act else float("nan")
        finite_synops = [float(row["synops_proxy"]) for row in act if math.isfinite(float(row["synops_proxy"]))]
        synops = float(np.sum(finite_synops)) if finite_synops else float("nan")
        dense_snn = item["full_inference_gmacs"] if int(item["time_steps"]) > 1 else "-"
        lines.append(f"| {name} | {int(item['parameter_count']):,} | {item['full_inference_gmacs']:.3f} GMAC | {dense_snn if dense_snn == '-' else f'{dense_snn:.3f} GMAC'} | {mem['peak_allocated_mib']:.1f} MiB | {lat.get('mean_ms', float('nan')):.3f} ms | {avg_power:.2f} W | {avg_energy:.3f} mJ | {avg_rate:.4f} | {synops:.3e} |")
    lines += ["", "| 模型 | 30 Hz需求 | 60 Hz需求 | 100 Hz需求 | 建议最低RAM | 推荐硬件级别 | 主要风险 |", "| --- | ---: | ---: | ---: | ---: | --- | --- |"]
    for row in hardware:
        lines.append(f"| {row['model']} | {row['30hz_gmac_s']:.2f} GMAC/s | {row['60hz_gmac_s']:.2f} GMAC/s | {row['100hz_gmac_s']:.2f} GMAC/s | {row['estimated_minimum_ram_mib']:.0f} MiB | {row['resource_level']} | {row['main_risk']} |")
    lines += ["", "## 口径", "", "- `1 MAC = 2 FLOPs`；MAC 只统计 Conv2d、ConvTranspose2d 与 Linear。", "- SNN GPU MAC 是普通 PyTorch 稠密卷积仿真的实际层调用总量，不因低发放率而减少。", "- SynOps 是理想事件驱动硬件代理指标；MAVLab递归网络的连接拓扑未从封装层可靠恢复时以 `NaN` 标注，不作猜测。", "- INT8 未测试：当前三模型没有已验证的可靠 INT8 导出路径。", ""]
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("本任务要求使用 cuda:0，但当前 CUDA 不可用。")
    device = torch.device("cuda:0")
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(1)
    np.random.seed(1)
    for path in (args.ann_checkpoint, args.ann_val_h5, args.hybrid_checkpoint, args.hybrid_val_h5, args.lif_checkpoint, args.lif_h5):
        if not path.is_file():
            raise FileNotFoundError(path)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = args.out_dir / "raw_logs"
    cases = load_cases(args, device)
    (args.out_dir / "model_inventory.md").write_text(make_inventory(cases), encoding="utf-8")

    complexity: list[dict[str, Any]] = []
    memory: list[dict[str, Any]] = []
    all_layers: list[dict[str, Any]] = []
    activity: list[dict[str, Any]] = []
    for case in cases:
        activate_case(case, device)
        item, layers, mem = profile_static_and_memory(case, device)
        complexity.append(item)
        memory.append(mem)
        all_layers.extend([{**row, "model": case.name, "mac_share_pct": 100.0 * row["macs_total"] / max(item["full_inference_macs"], 1)} for row in layers])
        activity.extend(collect_snn_activity(case))
        offload_case(case)
    write_csv(args.out_dir / "model_complexity.csv", complexity)
    write_csv(args.out_dir / "memory_profile.csv", memory)
    write_csv(args.out_dir / "layer_macs_ranked.csv", all_layers)
    write_csv(args.out_dir / "snn_activity.csv", activity)

    latency: list[dict[str, Any]] = []
    for case in cases:
        activate_case(case, device)
        for amp in (False, True):
            latency.append(latency_profile(case, device, args.warmup, args.iterations, amp))
        offload_case(case)
    write_csv(args.out_dir / "latency_profile.csv", latency)

    power: list[dict[str, Any]] = []
    idle_records: list[dict[str, float]] = []
    if not args.skip_power:
        sampler = NvidiaSmiSampler(interval_ms=100)
        sampler.start()
        time.sleep(args.power_seconds)
        idle_records = sampler.stop()
        idle_w = float(np.mean([row["power_w"] for row in idle_records])) if idle_records else float("nan")
        write_json(raw_dir / "power_idle.json", idle_records)
        for case in cases:
            activate_case(case, device)
            power.extend(power_profile(case, device, args.power_seconds, args.power_repeats, idle_w, raw_dir))
            offload_case(case)
    else:
        idle_w = float("nan")
    write_csv(args.out_dir / "power_profile.csv", power)
    write_json(raw_dir / "power_idle_summary.json", {"idle_power_w": idle_w, "samples": len(idle_records), "duration_s": args.power_seconds})

    hardware = hardware_rows(complexity, memory)
    write_csv(args.out_dir / "hardware_requirement.csv", hardware)
    plot_figures(args.out_dir / "figures", latency, complexity)
    (args.out_dir / "summary.md").write_text(summary_markdown(complexity, memory, latency, power, activity, hardware), encoding="utf-8")
    write_json(raw_dir / "run_manifest.json", {
        "device": str(device), "gpu": torch.cuda.get_device_name(device), "torch": torch.__version__,
        "warmup": args.warmup, "iterations": args.iterations, "power_seconds": args.power_seconds,
        "power_repeats": args.power_repeats, "models": [case.name for case in cases],
    })
    print(f"resource evaluation completed: {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
