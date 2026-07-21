"""Hybrid SNN-EV-Slim 使用的严格连续时序窗口 Dataset。"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable

import h5py
import hdf5plugin  # noqa: F401  # 注册当前 H5 使用的 Blosc 压缩过滤器。
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2


TensorTransform = Callable[[torch.Tensor], torch.Tensor]


def _decode_sequence(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


class TTCEFTemporalDataset(Dataset):
    """把 H5 中严格连续的 7 ms 读出组织为固定长度 Block。

    初始化阶段只载入 source/time/sequence 元数据；约 46 GiB 的 exp_filts、
    TTC 和 mask 均在 ``__getitem__`` 中按窗口延迟读取。
    """

    def __init__(
        self,
        h5_path: str | Path,
        *,
        window_length: int = 3,
        window_stride: int = 3,
        delta_t_us: float = 7_000.0,
        delta_t_tolerance_us: float = 500.0,
        augment: bool = False,
        flip_prob: float = 0.3,
        augmentation_transform: TensorTransform | None = None,
    ) -> None:
        super().__init__()
        self.h5_path = Path(h5_path).resolve()
        self.window_length = int(window_length)
        self.window_stride = int(window_stride)
        self.delta_t_us = float(delta_t_us)
        self.delta_t_tolerance_us = float(delta_t_tolerance_us)
        self.augment = bool(augment)
        if self.window_length < 1:
            raise ValueError("window_length 必须至少为 1。")
        if self.window_stride < 1:
            raise ValueError("window_stride 必须至少为 1。")
        if not self.h5_path.is_file():
            raise FileNotFoundError(self.h5_path)

        self._h5: h5py.File | None = None
        self._h5_pid: int | None = None
        with h5py.File(self.h5_path, "r", libver="latest") as handle:
            required = ("exp_filts", "ttc", "mask", "source_index", "exp_time")
            missing = [name for name in required if name not in handle]
            if missing:
                raise KeyError(f"H5 缺少连续窗口所需字段：{missing}")
            counts = [int(handle[name].shape[0]) for name in required]
            if len(set(counts)) != 1:
                raise ValueError(f"H5 各字段样本数不一致：{dict(zip(required, counts))}")
            self.sample_count = counts[0]
            self.sample_exp_shape = tuple(int(value) for value in handle["exp_filts"].shape[1:])
            self.sample_ttc_shape = tuple(int(value) for value in handle["ttc"].shape[1:])
            self.source_indices = np.asarray(handle["source_index"], dtype=np.int64)
            self.exp_times = np.asarray(handle["exp_time"], dtype=np.float64)
            self.sequence_names = self._read_sequence_names(handle)

        self.contiguous_edges = self._build_contiguous_edges()
        self.contiguous_runs = self._build_contiguous_runs()
        self.break_after_rows = np.flatnonzero(~self.contiguous_edges).astype(np.int64)
        self.window_starts = self._build_window_starts()
        self.transforms: TensorTransform = augmentation_transform or v2.Compose(
            [
                v2.RandomHorizontalFlip(p=flip_prob),
                v2.RandomVerticalFlip(p=flip_prob),
                v2.RandomRotation(degrees=(0, 180)),
            ]
        )

    def _read_sequence_names(self, handle: h5py.File) -> np.ndarray:
        for name in ("sequence_name", "sequence", "file_id"):
            if name in handle and isinstance(handle[name], h5py.Dataset):
                raw = np.asarray(handle[name]).reshape(-1)
                if raw.size != self.sample_count:
                    raise ValueError(f"{name} 长度为 {raw.size}，应为 {self.sample_count}")
                return np.asarray([_decode_sequence(value) for value in raw], dtype=object)
        sequence = _decode_sequence(handle.attrs.get("sequence_name", "unknown"))
        return np.full(self.sample_count, sequence, dtype=object)

    def _build_contiguous_edges(self) -> np.ndarray:
        if self.sample_count < 2:
            return np.zeros(0, dtype=bool)
        source_delta = np.diff(self.source_indices)
        time_delta = np.diff(self.exp_times)
        same_sequence = self.sequence_names[1:] == self.sequence_names[:-1]
        return (
            (source_delta == 1)
            & (time_delta > 0.0)
            & (np.abs(time_delta - self.delta_t_us) <= self.delta_t_tolerance_us)
            & same_sequence
        )

    def _build_contiguous_runs(self) -> list[tuple[int, int]]:
        """返回最大连续区间，端点均为包含关系的 H5 行号。"""

        if self.sample_count == 0:
            return []
        starts = [0]
        ends: list[int] = []
        for edge_index in np.flatnonzero(~self.contiguous_edges).tolist():
            ends.append(int(edge_index))
            starts.append(int(edge_index + 1))
        ends.append(self.sample_count - 1)
        return list(zip(starts, ends))

    def _build_window_starts(self) -> np.ndarray:
        starts: list[int] = []
        for run_start, run_end in self.contiguous_runs:
            final_start = run_end - self.window_length + 1
            if final_start < run_start:
                continue
            starts.extend(range(run_start, final_start + 1, self.window_stride))
        return np.asarray(starts, dtype=np.int64)

    def _ensure_h5_open(self) -> h5py.File:
        """每个进程持有自己的只读 HDF5 handle，禁止跨 worker 共享。"""

        current_pid = os.getpid()
        if self._h5 is not None and self._h5_pid != current_pid:
            try:
                self._h5.close()
            except Exception:
                pass
            self._h5 = None
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r", libver="latest")
            self._h5_pid = current_pid
        return self._h5

    def close(self) -> None:
        if self._h5 is not None:
            try:
                self._h5.close()
            finally:
                self._h5 = None
                self._h5_pid = None

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_h5"] = None
        state["_h5_pid"] = None
        return state

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def __len__(self) -> int:
        return int(self.window_starts.size)

    def __getitem__(self, index: int) -> dict[str, Any]:
        start = int(self.window_starts[index])
        end = start + self.window_length - 1
        handle = self._ensure_h5_open()

        # 连续切片比 HDF5 花式索引更高效；标签严格取 Block 最后一行。
        exp_np = np.asarray(handle["exp_filts"][start : end + 1], dtype=np.float32)
        ttc_np = np.asarray(handle["ttc"][end], dtype=np.float32)
        mask_np = np.asarray(handle["mask"][end], dtype=bool)
        exp = torch.from_numpy(exp_np)
        ttc = torch.from_numpy(ttc_np)[None]
        mask = torch.from_numpy(mask_np)[None]

        if self.augment:
            steps, channels, height, width = exp.shape
            # 所有 step、最后一步标签和 mask 合并后只采样一次增强参数。
            combined = torch.cat(
                [exp.reshape(steps * channels, height, width), ttc, mask.float()], dim=0
            )
            combined = self.transforms(combined)
            exp = combined[: steps * channels].reshape(steps, channels, height, width).float()
            ttc = combined[steps * channels : steps * channels + 1].float()
            mask = combined[steps * channels + 1 : steps * channels + 2].bool()

        return {
            "exp_filts": exp.float(),
            "ttc": ttc.float(),
            "mask": mask.bool(),
            "source_indices": torch.from_numpy(self.source_indices[start : end + 1].copy()),
            "exp_times": torch.from_numpy(self.exp_times[start : end + 1].copy()),
            "sequence_name": str(self.sequence_names[start]),
            "start_row": start,
            "end_row": end,
        }

    def summary(self) -> dict[str, Any]:
        lengths = [end - start + 1 for start, end in self.contiguous_runs]
        return {
            "h5_path": str(self.h5_path),
            "sample_count": self.sample_count,
            "sample_exp_shape": list(self.sample_exp_shape),
            "sample_ttc_shape": list(self.sample_ttc_shape),
            "window_length": self.window_length,
            "window_stride": self.window_stride,
            "window_count": len(self),
            "contiguous_run_count": len(self.contiguous_runs),
            "contiguous_run_length_min": min(lengths) if lengths else 0,
            "contiguous_run_length_max": max(lengths) if lengths else 0,
            "break_count": int(self.break_after_rows.size),
        }


def make_temporal_dataloader(
    dataset: TTCEFTemporalDataset,
    *,
    batch_size: int = 4,
    shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: bool = True,
) -> DataLoader:
    """构造只打乱窗口、不改变窗口内部时间顺序的 DataLoader。"""

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
