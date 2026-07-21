#!/usr/bin/env python3
"""Hybrid SNN-EV-Slim 在 360x360 随机输入上的显存与速度基准测试。"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_ROOT = PROJECT_ROOT / "EV-TTC-main/model"
if str(MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_ROOT))

from hybrid_block_training import train_hybrid_block  # noqa: E402
from hybrid_snn_evslim import HybridSNNEVSlim  # noqa: E402


DEFAULT_OUT_DIR = PROJECT_ROOT / "EV-TTC-SNN-main/reports/05_hybrid_snn_evslim"
BATCH_SIZES = (2, 4, 8)
PRECISIONS = (("fp32", False), ("amp_fp16", True))
WARMUP_BLOCKS = 3
MEASURED_BLOCKS = 10


def _memory_mib(bytes_value: int) -> float:
    return float(bytes_value) / (1024.0 * 1024.0)


def _stats(tensor: torch.Tensor | None) -> dict[str, float | None]:
    if tensor is None:
        return {"mean": None, "std": None}
    value = tensor.detach().float()
    return {
        "mean": float(value.mean().cpu()),
        "std": float(value.std(unbiased=False).cpu()),
    }


def _is_oom(error: RuntimeError) -> bool:
    return "out of memory" in str(error).lower()


def benchmark_one(batch_size: int, precision: str, use_amp: bool) -> dict[str, Any]:
    """以固定随机张量执行 warmup 和 10 个完整训练 Block。"""

    device = torch.device("cuda:0")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    torch.manual_seed(20260722 + batch_size + int(use_amp) * 100)
    model: HybridSNNEVSlim | None = None
    optimizer: torch.optim.Optimizer | None = None
    try:
        model = HybridSNNEVSlim().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        events = torch.randn(batch_size, 3, 6, 360, 360, device=device)
        target = torch.randn(batch_size, 1, 360, 360, device=device)
        mask = torch.rand(batch_size, 1, 360, 360, device=device) > 0.2
        scaler = None

        # 预热使 CUDA kernel 和内存分配稳定；预热不计入正式峰值和时间。
        for _ in range(WARMUP_BLOCKS):
            result = train_hybrid_block(
                model, optimizer, events, target, mask, use_amp=use_amp, scaler=scaler
            )
            scaler = result.scaler
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

        losses: list[float] = []
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        for _ in range(MEASURED_BLOCKS):
            result = train_hybrid_block(
                model, optimizer, events, target, mask, use_amp=use_amp, scaler=scaler
            )
            scaler = result.scaler
            losses.append(result.loss)
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
        average_seconds = elapsed / MEASURED_BLOCKS
        blocks_per_second = 1.0 / average_seconds
        peak_mib = _memory_mib(torch.cuda.max_memory_allocated(device))
        current_mib = _memory_mib(torch.cuda.memory_allocated(device))
        accumulator = _stats(model.accumulator)
        row = {
            "batch_size": batch_size,
            "precision": precision,
            "success": True,
            "oom": False,
            "warmup_blocks": WARMUP_BLOCKS,
            "measured_blocks": MEASURED_BLOCKS,
            "average_block_seconds": average_seconds,
            "blocks_per_second": blocks_per_second,
            "readouts_per_second": batch_size * 3 * blocks_per_second,
            "peak_memory_mib": peak_mib,
            "current_memory_mib": current_mib,
            "mean_loss": float(sum(losses) / len(losses)),
            "lif1_firing_rate": model.get_firing_rates()["lif1"],
            "lif2_firing_rate": model.get_firing_rates()["lif2"],
            "accumulator_mean": accumulator["mean"],
            "accumulator_std": accumulator["std"],
            "scaler_scale": float(scaler.get_scale()) if scaler is not None else None,
            "error": None,
        }
    except RuntimeError as error:
        if not _is_oom(error):
            raise
        row = {
            "batch_size": batch_size,
            "precision": precision,
            "success": False,
            "oom": True,
            "warmup_blocks": WARMUP_BLOCKS,
            "measured_blocks": MEASURED_BLOCKS,
            "average_block_seconds": None,
            "blocks_per_second": None,
            "readouts_per_second": None,
            "peak_memory_mib": _memory_mib(torch.cuda.max_memory_allocated(device)),
            "current_memory_mib": _memory_mib(torch.cuda.memory_allocated(device)),
            "mean_loss": None,
            "lif1_firing_rate": None,
            "lif2_firing_rate": None,
            "accumulator_mean": None,
            "accumulator_std": None,
            "scaler_scale": None,
            "error": str(error),
        }
    finally:
        del model, optimizer
        gc.collect()
        torch.cuda.empty_cache()
    return row


def recommend(rows: list[dict[str, Any]], total_memory_mib: float) -> dict[str, Any]:
    """按 85% 显存阈值优先选择 AMP 下最大可稳定运行 batch。"""

    threshold = total_memory_mib * 0.85
    suitable = [
        row
        for row in rows
        if row["success"] and row["peak_memory_mib"] is not None and row["peak_memory_mib"] < threshold
    ]
    amp_suitable = [row for row in suitable if row["precision"] == "amp_fp16"]
    candidates = amp_suitable or suitable
    if not candidates:
        return {
            "recommended_batch_size": None,
            "recommended_precision": None,
            "reason": "没有配置同时满足成功运行和峰值显存低于单卡总显存85%。",
            "memory_threshold_mib": threshold,
        }
    chosen = max(candidates, key=lambda row: (int(row["batch_size"]), float(row["blocks_per_second"])))
    note = (
        "B=8 稳定运行，按规则优先使用 B=8。"
        if chosen["batch_size"] == 8
        else (
            "B=8 不可用或超过85%显存阈值，选择 B=4。"
            if chosen["batch_size"] == 4
            else "仅 B=2 满足显存与稳定性条件；后续可用梯度累积增大有效 batch，但不会改善 BatchNorm 统计。"
        )
    )
    return {
        "recommended_batch_size": chosen["batch_size"],
        "recommended_precision": chosen["precision"],
        "reason": note,
        "memory_threshold_mib": threshold,
    }


def render_report(payload: dict[str, Any]) -> str:
    rows = payload["rows"]
    lines = [
        "# Hybrid SNN-EV-Slim 360x360 显存与速度测试",
        "",
        f"- GPU：`{payload['gpu_name']}`",
        f"- 总显存：{payload['total_memory_mib']:.1f} MiB",
        f"- 输入：`[B,3,6,360,360]`，监督目标：`[B,1,360,360]`",
        f"- 每组：预热 {WARMUP_BLOCKS} 个 Block，正式计时 {MEASURED_BLOCKS} 个 Block。",
        "",
        "| Batch | 精度 | 成功 | OOM | 平均Block(s) | Blocks/s | Readouts/s | 峰值显存(MiB) | 当前显存(MiB) | Loss | LIF1/LIF2 | Acc mean/std |",
        "|---:|---|---|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        def value(name: str, digits: int = 3) -> str:
            item = row[name]
            return "-" if item is None else f"{float(item):.{digits}f}"

        lines.append(
            f"| {row['batch_size']} | {row['precision']} | {'是' if row['success'] else '否'} | "
            f"{'是' if row['oom'] else '否'} | {value('average_block_seconds', 4)} | "
            f"{value('blocks_per_second', 3)} | {value('readouts_per_second', 3)} | "
            f"{value('peak_memory_mib', 1)} | {value('current_memory_mib', 1)} | "
            f"{value('mean_loss', 5)} | {value('lif1_firing_rate', 4)}/{value('lif2_firing_rate', 4)} | "
            f"{value('accumulator_mean', 4)}/{value('accumulator_std', 4)} |"
        )
    recommendation = payload["recommendation"]
    lines.extend(
        [
            "",
            "## 正式训练建议",
            f"- 建议 batch size：`{recommendation['recommended_batch_size']}`",
            f"- 建议精度：`{recommendation['recommended_precision']}`",
            f"- 85% 显存阈值：{recommendation['memory_threshold_mib']:.1f} MiB",
            f"- 原因：{recommendation['reason']}",
            "",
        ]
    )
    return "\n".join(lines)


def run_benchmark(out_dir: str | Path = DEFAULT_OUT_DIR) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("当前环境没有 CUDA，无法执行要求的 GPU 显存与速度测试。")
    device = torch.device("cuda:0")
    total_memory_mib = _memory_mib(torch.cuda.get_device_properties(device).total_memory)
    rows: list[dict[str, Any]] = []
    for batch_size in BATCH_SIZES:
        for precision, use_amp in PRECISIONS:
            print(f"[START] B={batch_size}, precision={precision}", flush=True)
            row = benchmark_one(batch_size, precision, use_amp)
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
    payload = {
        "gpu_name": torch.cuda.get_device_name(device),
        "total_memory_mib": total_memory_mib,
        "input_shape_template": ["B", 3, 6, 360, 360],
        "rows": rows,
    }
    payload["recommendation"] = recommend(rows, total_memory_mib)
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "hybrid_benchmark_360.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "HYBRID_BENCHMARK_360.md").write_text(render_report(payload), encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    run_benchmark(args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
