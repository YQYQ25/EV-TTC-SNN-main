"""阶段测试报告的小工具。"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


DEFAULT_OUT_DIR = Path(__file__).resolve().parents[2] / "debug_sets" / "lif_evflownet_ttc_stage1_3"


def ensure_out_dir(out_dir: str | Path | None = None) -> Path:
    path = Path(out_dir) if out_dir is not None else DEFAULT_OUT_DIR
    path.mkdir(parents=True, exist_ok=True)
    return path


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


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend("| " + " | ".join(str(v) for v in row) + " |" for row in rows)
    return "\n".join(lines)


def pass_text(value: bool) -> str:
    return "通过" if value else "未通过"
