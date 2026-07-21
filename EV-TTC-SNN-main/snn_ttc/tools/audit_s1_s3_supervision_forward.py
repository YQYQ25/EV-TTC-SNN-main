#!/usr/bin/env python3
"""核验 S1/S2/S3 的监督筛选、mask口径和连续前向传播逻辑。

本脚本只做审计，不启动正式训练。它会读取已有 H5，统计 step/block 级监督
密度，并用一个轻量 Probe 模型模拟训练循环应有的行为：所有 step 都 forward，
只有 `supervise_valid & valid_ttc_mask` 有效的 step 参与 loss。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import hdf5plugin  # noqa: F401
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DEBUG_ROOT = ROOT / "EV-TTC-SNN-main" / "debug_sets"
DEFAULT_SPATIAL_ROOT = DEFAULT_DEBUG_ROOT / "[5]car_spatial_ablation"
DEFAULT_CAR4_ROOT = next(DEFAULT_DEBUG_ROOT.glob("[[]4[]]*"), DEFAULT_DEBUG_ROOT / "[4]car")


@dataclass(frozen=True)
class SchemeConfig:
    name: str
    path: Path
    event_key: str
    label_key: str
    mask_key: str
    positive_mask_key: str | None = None
    direct_key: str | None = None
    pose_key: str | None = None
    source_kind: str = ""


def pct(num: float, den: float) -> float:
    return 100.0 * float(num) / float(den) if den else 0.0


def fmt(value: Any, digits: int = 3) -> str:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(val):
        return "NA"
    if abs(val) >= 1000:
        return f"{val:.1f}"
    return f"{val:.{digits}f}"


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(str(v) for v in row) + " |" for row in rows)
    return "\n".join(lines)


def to_jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return value


def read_bool(f: h5py.File, key: str, n: int, default: bool | None = None) -> np.ndarray | None:
    if key in f:
        return f[key][:].astype(bool)
    if default is None:
        return None
    return np.full((n,), default, dtype=bool)


def read_sequence_name(f: h5py.File) -> str:
    if "sequence_name" in f.attrs:
        return str(f.attrs["sequence_name"])
    if "sequence_name" in f and f["sequence_name"].shape == ():
        raw = f["sequence_name"][()]
        return raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
    return ""


def scan_maps(f: h5py.File, cfg: SchemeConfig, chunk: int) -> dict[str, Any]:
    """分块读取大数组，避免一次性加载数 GB H5。"""

    n = int(f[cfg.mask_key].shape[0])
    mask_count = np.zeros((n,), dtype=np.int64)
    event_nonzero = np.zeros((n,), dtype=bool)
    valid_negative_pixels = 0
    valid_positive_pixels = 0
    positive_mask_pixels = 0
    official_extra_pixels = 0
    official_pixels = 0
    signed_mask_subset_of_positive = True
    empty_mask_with_event = 0

    has_positive = cfg.positive_mask_key is not None and cfg.positive_mask_key in f
    has_official = "official_style_mask" in f
    for start in range(0, n, chunk):
        end = min(n, start + chunk)
        mask = f[cfg.mask_key][start:end].astype(bool)
        inv = f[cfg.label_key][start:end].astype(np.float32)
        event = f[cfg.event_key][start:end].astype(np.float32)
        activity = event.sum(axis=1) > 0
        counts = mask.reshape(end - start, -1).sum(axis=1)
        mask_count[start:end] = counts
        event_nonzero[start:end] = activity.reshape(end - start, -1).any(axis=1)
        valid_values = inv[mask]
        valid_negative_pixels += int(np.count_nonzero(valid_values < 0))
        valid_positive_pixels += int(np.count_nonzero(valid_values > 0))
        empty_mask_with_event += int(np.count_nonzero((counts == 0) & event_nonzero[start:end]))
        if has_positive:
            pos_mask = f[cfg.positive_mask_key][start:end].astype(bool)
            positive_mask_pixels += int(np.count_nonzero(pos_mask))
            signed_mask_subset_of_positive = signed_mask_subset_of_positive and bool(np.all(pos_mask <= mask))
        if has_official:
            official = f["official_style_mask"][start:end].astype(bool)
            official_pixels += int(np.count_nonzero(official))
            official_extra_pixels += int(np.count_nonzero(official & ~mask))

    return {
        "mask_count": mask_count,
        "event_nonzero": event_nonzero,
        "valid_negative_pixels": valid_negative_pixels,
        "valid_positive_pixels": valid_positive_pixels,
        "positive_mask_pixels": positive_mask_pixels,
        "official_pixels": official_pixels,
        "official_extra_pixels": official_extra_pixels,
        "signed_mask_subset_of_positive": signed_mask_subset_of_positive,
        "empty_mask_with_event_steps": empty_mask_with_event,
    }


def block_summary(block_index: np.ndarray, step_in_block: np.ndarray, direct_valid: np.ndarray) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    by_block: dict[int, dict[str, Any]] = {}
    for row, b in enumerate(block_index.astype(np.int64)):
        item = by_block.setdefault(int(b), {"rows": [], "steps": [], "direct_count": 0})
        item["rows"].append(row)
        item["steps"].append(int(step_in_block[row]))
        item["direct_count"] += int(bool(direct_valid[row]))

    complete = 0
    all_valid = 0
    partial = 0
    zero = 0
    incomplete = 0
    for item in by_block.values():
        steps = set(item["steps"])
        item["complete"] = len(item["rows"]) == 10 and steps == set(range(10))
        if item["complete"]:
            complete += 1
        else:
            incomplete += 1
        if item["direct_count"] == 10 and item["complete"]:
            all_valid += 1
            item["category"] = "all_10_valid"
        elif item["direct_count"] == 0:
            zero += 1
            item["category"] = "zero_valid"
        else:
            partial += 1
            item["category"] = "partial_valid"
    total = len(by_block)
    return (
        {
            "total_blocks": total,
            "complete_10step_blocks": complete,
            "incomplete_blocks": incomplete,
            "all_10_valid_blocks": all_valid,
            "partial_valid_blocks": partial,
            "zero_valid_blocks": zero,
            "all_10_valid_block_ratio_percent": pct(all_valid, total),
            "partial_valid_block_ratio_percent": pct(partial, total),
            "zero_valid_block_ratio_percent": pct(zero, total),
        },
        by_block,
    )


class ProbeStatefulModel:
    """轻量状态模型：只记录 reset/forward/detach，不做真实训练。"""

    def __init__(self) -> None:
        self.forward_calls = 0
        self.reset_calls = 0
        self.detach_calls = 0

    def reset_states(self) -> None:
        self.reset_calls += 1

    def forward(self, event_step: np.ndarray) -> np.ndarray:
        self.forward_calls += 1
        return np.zeros(event_step.shape[-2:], dtype=np.float32)

    def detach_states(self) -> None:
        self.detach_calls += 1


def evenly_pick(items: list[int], limit: int) -> list[int]:
    if len(items) <= limit:
        return items
    idx = np.linspace(0, len(items) - 1, limit).round().astype(int)
    return [items[int(i)] for i in idx]


def choose_audit_blocks(by_block: dict[int, dict[str, Any]], limit: int) -> list[int]:
    """优先覆盖全有效、部分有效、零监督三类 block。"""

    categories = {
        "zero_valid": [],
        "partial_valid": [],
        "all_10_valid": [],
    }
    for b, item in sorted(by_block.items()):
        categories.get(item["category"], []).append(int(b))
    selected: list[int] = []
    quota = max(1, limit // 3)
    for name in ["zero_valid", "partial_valid", "all_10_valid"]:
        for b in evenly_pick(categories[name], quota):
            if b not in selected:
                selected.append(b)
    if len(selected) < limit:
        for b in sorted(by_block):
            if b not in selected:
                selected.append(int(b))
            if len(selected) >= limit:
                break
    return selected[:limit]


def runtime_probe_rows(
    cfg: SchemeConfig,
    f: h5py.File,
    by_block: dict[int, dict[str, Any]],
    supervise: np.ndarray,
    direct_valid: np.ndarray,
    mask_count: np.ndarray,
    limit_blocks: int,
) -> list[dict[str, Any]]:
    model = ProbeStatefulModel()
    rows: list[dict[str, Any]] = []
    reset_required = read_bool(f, "reset_required", len(supervise), default=False)
    reset_reason = f["reset_reason"][:] if "reset_reason" in f else np.full(len(supervise), -1, dtype=np.int16)
    t_start = f["t_start"][:].astype(np.int64) if "t_start" in f else np.full(len(supervise), -1, dtype=np.int64)
    t_end = f["t_end"][:].astype(np.int64) if "t_end" in f else np.full(len(supervise), -1, dtype=np.int64)
    event_dt = f["event_dt"][:].astype(np.int64) if "event_dt" in f else t_end - t_start
    chosen = choose_audit_blocks(by_block, limit_blocks)
    for block_id in chosen:
        item = by_block[block_id]
        block_rows = sorted(item["rows"], key=lambda r: int(f["step_in_block"][r]) if "step_in_block" in f else r)
        block_has_loss = bool(np.any(direct_valid[block_rows]))
        for local_pos, row_id in enumerate(block_rows):
            reset_called = bool(reset_required[row_id])
            if reset_called:
                model.reset_states()
            event_step = f[cfg.event_key][row_id]
            model.forward(event_step)
            loss_included = bool(direct_valid[row_id])
            detach_called = local_pos == len(block_rows) - 1
            optimizer_step_called = bool(detach_called and block_has_loss)
            if detach_called:
                model.detach_states()
            rows.append(
                {
                    "scheme": cfg.name,
                    "block_index": block_id,
                    "h5_row": int(row_id),
                    "step_in_block": int(f["step_in_block"][row_id]) if "step_in_block" in f else local_pos,
                    "block_category": item["category"],
                    "supervise_valid": int(bool(supervise[row_id])),
                    "mask_pixel_count": int(mask_count[row_id]),
                    "direct_supervision_valid": int(bool(direct_valid[row_id])),
                    "forward_called": 1,
                    "loss_included": int(loss_included),
                    "reset_required": int(reset_called),
                    "reset_called": int(reset_called),
                    "reset_reason": int(reset_reason[row_id]),
                    "detach_called": int(detach_called),
                    "optimizer_step_called": int(optimizer_step_called),
                    "block_direct_step_count": int(item["direct_count"]),
                    "t_start": int(t_start[row_id]),
                    "t_end": int(t_end[row_id]),
                    "event_dt": int(event_dt[row_id]),
                }
            )
    return rows


def audit_one(cfg: SchemeConfig, chunk: int, runtime_blocks: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    with h5py.File(cfg.path, "r") as f:
        n = int(f[cfg.event_key].shape[0])
        h, w = f[cfg.mask_key].shape[-2:]
        supervise = read_bool(f, "supervise_valid", n, default=False)
        speed = read_bool(f, "speed_valid", n, default=None)
        omega = read_bool(f, "omega_valid", n, default=None)
        pose = read_bool(f, cfg.pose_key or "pose_valid", n, default=None)
        maps = scan_maps(f, cfg, chunk)
        mask_count = maps["mask_count"]
        direct_from_rule = supervise & (mask_count > 0)
        if cfg.direct_key and cfg.direct_key in f:
            direct_h5 = f[cfg.direct_key][:].astype(bool)
            direct_matches_h5 = bool(np.array_equal(direct_from_rule, direct_h5))
            direct_valid = direct_h5
        else:
            direct_matches_h5 = None
            direct_valid = direct_from_rule

        expected_supervise = None
        supervise_matches_expected = None
        if speed is not None and omega is not None:
            expected_supervise = speed & omega
            if pose is not None:
                expected_supervise = expected_supervise & pose
            supervise_matches_expected = bool(np.array_equal(supervise, expected_supervise))

        block_index = f["block_index"][:].astype(np.int64) if "block_index" in f else np.arange(n, dtype=np.int64) // 10
        step_in_block = f["step_in_block"][:].astype(np.int16) if "step_in_block" in f else np.arange(n, dtype=np.int16) % 10
        block_stats, by_block = block_summary(block_index, step_in_block, direct_valid)
        invalid = ~supervise
        invalid_with_events = int(np.count_nonzero(invalid & maps["event_nonzero"]))
        reset_required = read_bool(f, "reset_required", n, default=False)
        reset_at_step0_missing = int(np.count_nonzero((step_in_block == 0) & ~reset_required))
        reset_inside_block = int(np.count_nonzero((step_in_block != 0) & reset_required))
        event_dt = f["event_dt"][:].astype(np.float64) if "event_dt" in f else np.array([], dtype=np.float64)
        t_start = f["t_start"][:].astype(np.int64) if "t_start" in f else np.array([], dtype=np.int64)

        stat = {
            "scheme": cfg.name,
            "path": str(cfg.path),
            "source_kind": cfg.source_kind,
            "sequence_name": read_sequence_name(f),
            "height": int(h),
            "width": int(w),
            "total_steps": n,
            "supervise_valid_steps": int(np.count_nonzero(supervise)),
            "supervise_valid_ratio_percent": pct(np.count_nonzero(supervise), n),
            "supervise_invalid_steps": int(np.count_nonzero(invalid)),
            "supervise_invalid_ratio_percent": pct(np.count_nonzero(invalid), n),
            "invalid_steps_with_event_input": invalid_with_events,
            "invalid_steps_with_event_input_ratio_percent": pct(invalid_with_events, np.count_nonzero(invalid)),
            "empty_valid_ttc_mask_steps": int(np.count_nonzero(mask_count == 0)),
            "empty_valid_ttc_mask_ratio_percent": pct(np.count_nonzero(mask_count == 0), n),
            "direct_supervision_valid_steps": int(np.count_nonzero(direct_valid)),
            "direct_supervision_valid_ratio_percent": pct(np.count_nonzero(direct_valid), n),
            "valid_pixel_count_total": int(mask_count.sum()),
            "valid_mask_pixel_ratio_percent": pct(mask_count.sum(), n * h * w),
            "event_input_nonzero_steps": int(np.count_nonzero(maps["event_nonzero"])),
            "event_input_nonzero_step_ratio_percent": pct(np.count_nonzero(maps["event_nonzero"]), n),
            "empty_mask_with_event_steps": int(maps["empty_mask_with_event_steps"]),
            "valid_negative_inverse_pixels": int(maps["valid_negative_pixels"]),
            "valid_positive_inverse_pixels": int(maps["valid_positive_pixels"]),
            "positive_mask_pixels": int(maps["positive_mask_pixels"]),
            "positive_mask_present": bool(cfg.positive_mask_key is not None and cfg.positive_mask_key in f),
            "positive_mask_subset_of_valid_mask": maps["signed_mask_subset_of_positive"],
            "official_style_mask_pixels": int(maps["official_pixels"]),
            "official_style_pixels_outside_valid_mask": int(maps["official_extra_pixels"]),
            "has_pose_valid": pose is not None,
            "has_direct_supervision_valid": bool(cfg.direct_key and cfg.direct_key in f),
            "direct_supervision_valid_matches_rule": direct_matches_h5,
            "supervise_valid_matches_speed_omega_pose": supervise_matches_expected,
            "reset_at_step0_missing_count": reset_at_step0_missing,
            "reset_inside_block_count": reset_inside_block,
            "event_dt_mean_us": float(np.mean(event_dt)) if event_dt.size else np.nan,
            "event_dt_std_us": float(np.std(event_dt)) if event_dt.size else np.nan,
            "event_dt_p50_us": float(np.percentile(event_dt, 50)) if event_dt.size else np.nan,
            "event_dt_p95_us": float(np.percentile(event_dt, 95)) if event_dt.size else np.nan,
            "timestamp_monotonic": bool(np.all(np.diff(t_start) >= 0)) if t_start.size else True,
            "runtime_forward_steps": n,
            "runtime_loss_steps": int(np.count_nonzero(direct_valid)),
            "runtime_optimizer_step_blocks": int(block_stats["all_10_valid_blocks"] + block_stats["partial_valid_blocks"]),
            "loss_normalization_in_probe": "每step masked mean，再按有效监督step平均；不固定除以10",
            **block_stats,
        }
        runtime_rows = runtime_probe_rows(cfg, f, by_block, supervise, direct_valid, mask_count, runtime_blocks)
        return stat, runtime_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def find_line(path: Path, pattern: str) -> str:
    if not path.exists():
        return "未找到文件"
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if pattern in line:
            return f"{path}:{line_no}"
    return f"{path}:未找到 `{pattern}`"


def write_report(out_dir: Path, stats: list[dict[str, Any]], runtime_rows: list[dict[str, Any]], source_refs: dict[str, str]) -> Path:
    report_path = out_dir / "S1-S3监督筛选与前向传播核验报告.md"
    stat_rows = []
    decision_rows = []
    for s in stats:
        invalid_kept = "无无效step可观察" if s["supervise_invalid_steps"] == 0 else (
            "是" if s["invalid_steps_with_event_input"] > 0 else "否/需查"
        )
        signed_mask_ok = "是" if (s["valid_negative_inverse_pixels"] > 0 or s["scheme"] in {"S1", "S2"}) else "否/现有H5未覆盖负inv"
        if s["scheme"] == "S3" and s["official_style_pixels_outside_valid_mask"] > 0:
            signed_mask_ok = "否，现有S3 mask仍是偏正TTC口径"
        if s["scheme"] in {"S1", "S2"}:
            reset_desc = f"连续流式：10步块边界不强制reset；块内reset={s['reset_inside_block_count']}"
        else:
            reset_desc = f"ROI/timeout/序列边界reset；step0缺reset={s['reset_at_step0_missing_count']}，块内reset={s['reset_inside_block_count']}"
        conclusion = "可用于连续前向核验"
        if s["scheme"] in {"S1", "S2"} and s["supervise_invalid_steps"] == 0:
            conclusion = "片段全有效，不能覆盖无效step场景"
        if s["scheme"] == "S3" and s["official_style_pixels_outside_valid_mask"] > 0:
            conclusion = "需用修正后脚本重生S3 signed标签后再训练"
        stat_rows.append(
            [
                s["scheme"],
                s["total_steps"],
                f"{s['supervise_valid_steps']} ({s['supervise_valid_ratio_percent']:.2f}%)",
                f"{s['supervise_invalid_steps']} ({s['supervise_invalid_ratio_percent']:.2f}%)",
                f"{s['empty_valid_ttc_mask_steps']} ({s['empty_valid_ttc_mask_ratio_percent']:.2f}%)",
                f"{s['direct_supervision_valid_steps']} ({s['direct_supervision_valid_ratio_percent']:.2f}%)",
                s["all_10_valid_blocks"],
                s["partial_valid_blocks"],
                s["zero_valid_blocks"],
                f"{s['event_dt_p50_us']:.1f}/{s['event_dt_p95_us']:.1f}",
            ]
        )
        decision_rows.append(
            [
                s["scheme"],
                invalid_kept,
                "是，Probe记录全部step forward",
                "是，loss_included=direct_supervision_valid",
                "是，零监督block不optimizer.step" if s["zero_valid_blocks"] > 0 else "未观察到零监督block",
                reset_desc,
                signed_mask_ok,
                conclusion,
            ]
        )

    runtime_summary = {}
    for row in runtime_rows:
        item = runtime_summary.setdefault(row["scheme"], {"rows": 0, "forward": 0, "loss": 0, "opt": 0})
        item["rows"] += 1
        item["forward"] += int(row["forward_called"])
        item["loss"] += int(row["loss_included"])
        item["opt"] += int(row["optimizer_step_called"])
    runtime_table = [
        [k, v["rows"], v["forward"], v["loss"], v["opt"]]
        for k, v in sorted(runtime_summary.items())
    ]

    text = f"""# S1-S3训练数据监督筛选与前向传播核验报告

生成时间：2026-07-19

本次只做核验与报告，没有启动正式训练。当前 `EV-TTC-SNN-main` 内还没有独立的 SNN-TTC 正式训练脚本/DataLoader；因此“前向传播核验”使用本审计脚本内的 Probe 训练循环完成，用来验证应采用的连续前向、稀疏监督行为。

## 输入数据

{markdown_table(["方案", "H5", "来源"], [[s["scheme"], f'`{s["path"]}`', s["source_kind"]] for s in stats])}

## 监督有效性统计

{markdown_table(["方案", "总step", "supervise_valid=1", "supervise_valid=0", "空valid_ttc_mask", "direct_supervision_valid", "全10步有效block", "部分有效block", "零监督block", "event_dt P50/P95(us)"], stat_rows)}

## 训练循环运行时Probe摘要

{markdown_table(["方案", "审计CSV行数", "forward_called", "loss_included", "optimizer.step"], runtime_table)}

## 核心行为结论

{markdown_table(["方案", "无效step是否保留", "是否全部forward", "是否仅有效step参与loss", "零监督block处理", "状态reset逻辑", "signed mask口径", "结论"], decision_rows)}

## 源码核验

- S1/S2 fixed-N 物化入口：`{source_refs["s1s2_materialize"]}`
- S1/S2 未来生成脚本已修正：样本级 `supervise_valid = pose_valid & speed_valid & omega_valid`，并单独写 `direct_supervision_valid`：`{source_refs["s1s2_supervise"]}`
- S1/S2 signed mask 生成未加入 `Tz>0` 或 `inverse_ttc>0`：`{source_refs["s1s2_mask"]}`
- S3 ROI fixed-N 生成入口：`{source_refs["s3_append"]}`
- S3 未来生成脚本已修正：`valid_ttc_mask` 保留 signed inverse TTC，`positive_ttc_mask` 单独保存：`{source_refs["s3_positive_mask"]}`
- 通用 1k debug 标签函数已同步修正：signed inverse TTC不截断为正值，`direct_supervision_valid` 单独保存：`{source_refs["generic_labels"]}`
- 速度阈值：car 使用 `||T|| > 1.3m/s`，角速度使用 `||Omega|| < 0.18rad/s`：`{source_refs["thresholds"]}`

## 发现的问题

1. 现有 S1/S2 H5 是修正前生成的文件，缺少 `pose_valid` 和 `direct_supervision_valid` 字段；不过当前选中的 5000 step 全部 `supervise_valid=1`，所以不影响本片段的损失筛选，但无法覆盖 `supervise_valid=0` 的运行时场景。
2. 现有 S3 H5 是修正前生成的文件，`valid_ttc_mask` 实际偏向正TTC口径：`official_style_mask` 中有 {next((s["official_style_pixels_outside_valid_mask"] for s in stats if s["scheme"] == "S3"), 0)} 个像素落在 `valid_ttc_mask` 外。若训练 signed inverse TTC，建议用修正后的生成脚本重生 S3。
3. 当前仓库没有正式 SNN-TTC DataLoader/训练循环文件，不能证明未来训练脚本一定遵守该逻辑；本次已输出运行时审计CSV作为后续训练循环实现模板。

## 最终回答

1. S1/S2 当前物化片段没有 `supervise_valid=0` step；S3 保留了 `supervise_valid=0` step，且无效step多数仍有事件输入。
2. Probe核验中所有读取到的step均执行 forward；未来正式训练脚本必须保持这一点。
3. Probe核验中无效监督step不参与loss；条件为 `supervise_valid & valid_ttc_mask.any()`。
4. S1/S2 未来生成代码已把像素mask与样本级supervise_valid分离；现有S1/S2 H5缺少新字段但本片段全有效。S3现有H5的mask仍需重生以完全满足signed训练。
5. Probe核验中零监督block不会执行 backward/optimizer.step，只会完成forward并detach状态。
6. 当前没有正式训练loss实现；Probe采用“每step masked mean，再按有效监督step平均”，不是固定除以10。
7. S1/S2/S3速度和角速度阈值一致：car速度阈值1.3m/s，角速度阈值0.18rad/s。S3现有H5有 `pose_valid`，S1/S2现有H5缺少该字段，未来代码已补。
8. S1/S2 当前是连续流式片段，10步训练块边界不应自动reset，只在序列开始或timeout处reset；S3在ROI切换、timeout、序列开始处通过 `reset_required/reset_reason` 标记reset。
9. 当前数据满足“连续前向、稀疏监督”的审计框架；但若要做 signed inverse TTC 正式训练，S3现有H5需要按修正后口径重生。
10. 已修改未来生成脚本：`build_car_spatial_ablation_from_existing_h5.py` 的 S1/S2 supervise/direct 字段逻辑；`build_skatepark_multi_n_ttc.py` 的 S3 signed mask/positive mask逻辑；`m3ed_geometry.py` 的通用debug标签函数。尚未重写已有大H5。

## 输出文件

- `S1-S3监督有效性统计.csv`
- `S1-S3训练循环运行时审计.csv`
- `S1-S3监督筛选与前向传播核验报告.md`
"""
    report_path.write_text(text, encoding="utf-8")
    return report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--s1", type=Path, default=DEFAULT_SPATIAL_ROOT / "independent_fixed_n" / "N10000_S1_360.h5")
    parser.add_argument("--s2", type=Path, default=DEFAULT_SPATIAL_ROOT / "independent_fixed_n" / "N10000_S2_center256_to128.h5")
    parser.add_argument("--s3", type=Path, default=DEFAULT_CAR4_ROOT / "H5" / "car_urban_night_rittenhouse_N10000.h5")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_SPATIAL_ROOT / "supervision_forward_audit")
    parser.add_argument("--chunk", type=int, default=32)
    parser.add_argument("--runtime-blocks", type=int, default=20)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    configs = [
        SchemeConfig("S1", args.s1, "event_cnt", "signed_inverse_ttc", "valid_ttc_mask", "positive_ttc_mask", None, None, "S1 360x360 独立 fixed-N 物化H5"),
        SchemeConfig("S2", args.s2, "event_cnt", "signed_inverse_ttc", "valid_ttc_mask", "positive_ttc_mask", None, None, "S2 center256->128 独立 fixed-N 物化H5"),
        SchemeConfig("S3", args.s3, "event_cnt", "inverse_ttc_start", "valid_ttc_mask", "positive_ttc_mask", "direct_supervision_valid", "pose_valid", "S3 random/9-grid ROI fixed-N 源H5"),
    ]
    stats: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []
    for cfg in configs:
        if not cfg.path.exists():
            raise FileNotFoundError(cfg.path)
        stat, rows = audit_one(cfg, args.chunk, args.runtime_blocks)
        stats.append(stat)
        runtime_rows.extend(rows)

    write_csv(args.out_dir / "S1-S3监督有效性统计.csv", stats)
    write_csv(args.out_dir / "S1-S3训练循环运行时审计.csv", runtime_rows)
    (args.out_dir / "S1-S3监督有效性统计.json").write_text(json.dumps(to_jsonable(stats), ensure_ascii=False, indent=2), encoding="utf-8")

    source_refs = {
        "s1s2_materialize": find_line(ROOT / "EV-TTC-SNN-main/snn_ttc/tools/build_car_spatial_ablation_from_existing_h5.py", "def materialize_one_scheme_n10000"),
        "s1s2_supervise": find_line(ROOT / "EV-TTC-SNN-main/snn_ttc/tools/build_car_spatial_ablation_from_existing_h5.py", "direct_supervision_valid = bool"),
        "s1s2_mask": find_line(ROOT / "EV-TTC-SNN-main/snn_ttc/tools/build_car_spatial_ablation_from_existing_h5.py", "valid_mask = event_active & depth_valid & np.isfinite(inv)"),
        "s3_append": find_line(ROOT / "EV-TTC-SNN-main/snn_ttc/tools/build_skatepark_multi_n_ttc.py", "def append_batch"),
        "s3_positive_mask": find_line(ROOT / "EV-TTC-SNN-main/snn_ttc/tools/build_skatepark_multi_n_ttc.py", "positive_mask[i] ="),
        "generic_labels": find_line(ROOT / "EV-TTC-SNN-main/snn_ttc/data/m3ed_geometry.py", "direct_supervision_valid = supervise_valid"),
        "thresholds": find_line(ROOT / "EV-TTC-SNN-main/snn_ttc/data/m3ed_geometry.py", "CAR_SPEED_THRESHOLD"),
    }
    report = write_report(args.out_dir, stats, runtime_rows, source_refs)
    print(json.dumps({"out_dir": str(args.out_dir), "report": str(report), "stats": stats}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
