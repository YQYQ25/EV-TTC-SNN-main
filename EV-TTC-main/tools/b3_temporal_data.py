#!/usr/bin/env python3
"""Temporal clip datasets for B3, backed by B1-Full H5 inputs and audited clips."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import h5py
import hdf5plugin  # noqa: F401 - register Blosc compression filters.
import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class ClipSpec:
    sequence_name: str
    sample_indices: tuple[int, ...]
    source_indices: tuple[int, ...]
    timestamps: tuple[float, ...]
    segment_id: int

    @property
    def length(self) -> int:
        return len(self.sample_indices)


def _read_payload(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _to_spec(item: dict) -> ClipSpec:
    return ClipSpec(
        sequence_name=str(item["sequence_name"]),
        sample_indices=tuple(int(value) for value in item["h5_sample_indices"]),
        source_indices=tuple(int(value) for value in item["source_frame_indices"]),
        timestamps=tuple(float(value) for value in item["timestamps"]),
        segment_id=int(item["segment_id"]),
    )


def load_segments(clips_json: str | Path) -> list[ClipSpec]:
    payload = _read_payload(clips_json)
    return [_to_spec(item) for item in payload["maximal_continuous_segments"]]


def make_fixed_clips(
    clips_json: str | Path,
    clip_length: int = 16,
    stride: int = 8,
    max_clips: int | None = None,
) -> list[ClipSpec]:
    """Create deterministic windows within audited segments; no cross-boundary windows."""
    if clip_length <= 0 or stride <= 0:
        raise ValueError("clip_length and stride must be positive")
    specs: list[ClipSpec] = []
    for segment in load_segments(clips_json):
        for start in range(0, segment.length - clip_length + 1, stride):
            end = start + clip_length
            specs.append(
                ClipSpec(
                    sequence_name=segment.sequence_name,
                    sample_indices=segment.sample_indices[start:end],
                    source_indices=segment.source_indices[start:end],
                    timestamps=segment.timestamps[start:end],
                    segment_id=segment.segment_id,
                )
            )
    return specs[:max_clips] if max_clips is not None else specs


class _B1FullTemporalBase(Dataset):
    def __init__(self, h5_path: str | Path, specs: Iterable[ClipSpec], cache_in_memory: bool = False):
        self.h5_path = str(Path(h5_path).resolve())
        self.specs = list(specs)
        if not self.specs:
            raise ValueError("no temporal clips were selected")
        self._h5: h5py.File | None = None
        self._cache: dict[str, np.ndarray] | None = None
        with h5py.File(self.h5_path, "r") as handle:
            required = {"exp_filts", "ttc", "mask", "source_indices", "file_names"}
            missing = required - set(handle.keys())
            if missing:
                raise KeyError(f"{self.h5_path} missing {sorted(missing)}")
            self.channel_count = int(handle["exp_filts"].shape[1])
            if self.channel_count != 12:
                raise ValueError(f"B3 requires 12-channel B1-Full input, got {self.channel_count}")
            name = handle["file_names"][0]
            self.sequence_name = name.decode("utf-8") if isinstance(name, bytes) else str(name)
            self.source_indices = handle["source_indices"][:].astype(np.int64)
            if cache_in_memory:
                self._cache = {
                    "exp_filts": handle["exp_filts"][:],
                    "ttc": handle["ttc"][:],
                    "mask": handle["mask"][:],
                }
        self._validate_specs()

    def _validate_specs(self) -> None:
        for spec in self.specs:
            indices = np.asarray(spec.sample_indices, dtype=np.int64)
            source = np.asarray(spec.source_indices, dtype=np.int64)
            timestamps = np.asarray(spec.timestamps, dtype=np.float64)
            if spec.sequence_name != self.sequence_name:
                raise ValueError(f"clip sequence {spec.sequence_name} != H5 sequence {self.sequence_name}")
            if len(indices) != len(source) or len(indices) != len(timestamps):
                raise ValueError("clip metadata lengths disagree")
            if np.any(np.diff(indices) != 1):
                raise ValueError("clip H5 indices are not consecutive")
            if np.any(np.diff(source) != 1):
                raise ValueError("clip source frame indices are not consecutive")
            if np.any(np.diff(timestamps) <= 0):
                raise ValueError("clip timestamps are not strictly increasing")
            if np.any(np.abs(np.diff(timestamps) - 7000.0) > 500.0):
                raise ValueError("clip timestamp interval is outside the 7 ms audit tolerance")
            if not np.array_equal(self.source_indices[indices], source):
                raise ValueError("B1-Full source_indices do not match the audited clip JSON")

    def _open(self) -> h5py.File:
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r", libver="latest")
        return self._h5

    def __len__(self) -> int:
        return len(self.specs)

    def __getitem__(self, item: int):
        spec = self.specs[item]
        indices = np.asarray(spec.sample_indices, dtype=np.int64)
        if self._cache is None:
            handle = self._open()
            exp, ttc_data, mask_data = handle["exp_filts"][indices], handle["ttc"][indices], handle["mask"][indices]
        else:
            exp, ttc_data, mask_data = self._cache["exp_filts"][indices], self._cache["ttc"][indices], self._cache["mask"][indices]
        x = torch.from_numpy(exp.astype(np.float32))
        ttc = torch.from_numpy(ttc_data.astype(np.float32))[:, None]
        mask = torch.from_numpy(mask_data.astype(np.bool_))[:, None]
        metadata = {
            "segment_id": torch.tensor(spec.segment_id, dtype=torch.int64),
            "sample_indices": torch.tensor(spec.sample_indices, dtype=torch.int64),
            "source_indices": torch.tensor(spec.source_indices, dtype=torch.int64),
            "timestamps": torch.tensor(spec.timestamps, dtype=torch.float64),
        }
        return x, ttc, mask, metadata


class TemporalClipDataset(_B1FullTemporalBase):
    """Fixed-length clips for training, shuffled only at clip granularity."""

    def __init__(
        self,
        h5_path: str | Path,
        clips_json: str | Path,
        clip_length: int = 16,
        stride: int = 8,
        max_clips: int | None = None,
        cache_in_memory: bool = False,
    ):
        self.clip_length = clip_length
        self.stride = stride
        super().__init__(h5_path, make_fixed_clips(clips_json, clip_length, stride, max_clips), cache_in_memory)


class TemporalSegmentDataset(_B1FullTemporalBase):
    """Maximal continuous segments for final closed-loop evaluation."""

    def __init__(self, h5_path: str | Path, clips_json: str | Path, cache_in_memory: bool = False):
        super().__init__(h5_path, load_segments(clips_json), cache_in_memory)
