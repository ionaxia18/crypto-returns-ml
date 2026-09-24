"""Read-only access to an anonymous student tensor dataset (schema 1 or 2).

The dataset is a directory of immutable daily NumPy shards:

    <root>/
      dataset.json
      manifest.json
      _SUCCESS
      feature_ids.npy
      dates/YYYY-MM-DD/{X.npy, episode_id.npy, meta.json[, y.npy, w.npy]}

``X.npy`` is float32 ``[1440, S_d, K]``; ``y.npy``/``w.npy`` are float32
``[1440, S_d]`` and exist only on visible dates. ``episode_id.npy`` is an
anonymous int32 listing-episode id per symbol column. All arrays are loaded
with read-only memory mapping; a flattened row view is a zero-copy reshape.
"""

from __future__ import annotations

import datetime
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np

from . import contract


class DataError(RuntimeError):
    """Raised when the dataset root violates the published schema."""


def _sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except FileNotFoundError as exc:
        raise DataError(f"Missing required file: {path}") from exc
    if not isinstance(payload, dict):
        raise DataError(f"Expected a JSON object at {path}")
    return payload


@dataclass(frozen=True)
class DateShard:
    """One immutable daily shard."""

    date: datetime.date
    path: Path
    split: str  # "visible" | "hidden"
    n_symbols: int
    meta: dict

    @property
    def has_labels(self) -> bool:
        return (self.path / "y.npy").is_file() and (self.path / "w.npy").is_file()

    def load_x(self) -> np.ndarray:
        """float32 [1440, S, K], read-only memory map."""
        return np.load(self.path / "X.npy", mmap_mode="r", allow_pickle=False)

    def load_y(self) -> np.ndarray:
        """float32 [1440, S]; only on labeled dates."""
        return np.load(self.path / "y.npy", mmap_mode="r", allow_pickle=False)

    def load_w(self) -> np.ndarray:
        """float32 [1440, S] effective evaluation weight; only on labeled dates."""
        return np.load(self.path / "w.npy", mmap_mode="r", allow_pickle=False)

    def load_episode_id(self) -> np.ndarray:
        """int32 [S] anonymous listing-episode ids (alignment/grouping only)."""
        return np.load(self.path / "episode_id.npy", mmap_mode="r", allow_pickle=False)

    def flat_rows(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Zero-copy (X2d [1440*S, K], y [1440*S], w [1440*S]) row views."""
        x = self.load_x()
        n = x.shape[0] * x.shape[1]
        x2d = x.reshape(n, x.shape[2])
        y = self.load_y().reshape(n)
        w = self.load_w().reshape(n)
        return x2d, y, w

    def flat_x(self) -> np.ndarray:
        """Zero-copy X2d [1440*S, K] row view (works on hidden dates too)."""
        x = self.load_x()
        return x.reshape(x.shape[0] * x.shape[1], x.shape[2])

    def verify(self, deep: bool = False) -> None:
        """Validate shard structure against its own meta.json.

        ``deep=True`` also re-hashes every array file (slow on real shards).
        """
        arrays = self.meta.get("arrays")
        if not isinstance(arrays, dict) or not arrays:
            raise DataError(f"{self.path}: meta.json missing arrays block")
        on_disk = {p.name for p in self.path.glob("*.npy")}
        if set(arrays) != on_disk:
            raise DataError(
                f"{self.path}: meta arrays {sorted(arrays)} != files {sorted(on_disk)}"
            )
        for name, descriptor in arrays.items():
            path = self.path / name
            arr = np.load(path, mmap_mode="r", allow_pickle=False)
            if list(arr.shape) != list(descriptor.get("shape", [])):
                raise DataError(f"{path}: shape {list(arr.shape)} != meta {descriptor.get('shape')}")
            if str(arr.dtype) != descriptor.get("dtype"):
                raise DataError(f"{path}: dtype {arr.dtype} != meta {descriptor.get('dtype')}")
            if deep:
                digest = _sha256_file(path)
                if digest != descriptor.get("sha256"):
                    raise DataError(f"{path}: sha256 mismatch")


class Dataset:
    """A validated read-only handle on one dataset root."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.info = _load_json(self.root / "dataset.json")
        self.schema_version = int(self.info.get("schema_version", -1))
        if self.schema_version not in contract.SUPPORTED_SCHEMA_VERSIONS:
            raise DataError(
                f"Unsupported schema_version {self.info.get('schema_version')!r} "
                f"(kit supports {contract.SUPPORTED_SCHEMA_VERSIONS})"
            )
        if self.schema_version == 1:
            # v1 datasets predate the field and are pinned to the v1 rule;
            # a root that nevertheless declares a different rule is corrupt.
            declared = self.info.get("assignment_rule")
            if declared is not None and declared != contract.DEFAULT_ASSIGNMENT_RULE:
                raise DataError(
                    f"schema_version 1 is pinned to "
                    f"{contract.DEFAULT_ASSIGNMENT_RULE!r} but dataset.json "
                    f"declares assignment_rule {declared!r}"
                )
            self.assignment_rule = contract.DEFAULT_ASSIGNMENT_RULE
        else:
            rule = self.info.get("assignment_rule")
            if rule not in contract.ASSIGNMENT_RULES:
                raise DataError(
                    f"schema_version {self.schema_version} requires a known "
                    f"assignment_rule; got {rule!r} "
                    f"(known: {contract.ASSIGNMENT_RULES})"
                )
            self.assignment_rule = str(rule)
        self.dataset_id: str = str(self.info["dataset_id"])
        self.n_features: int = int(self.info["n_features"])
        self.time_rows: int = int(
            self.info.get("daily_time_rows", contract.DAILY_TIME_ROWS)
        )
        self.hidden_start: datetime.date = datetime.date.fromisoformat(
            str(self.info["hidden_start"])
        )
        dates_root = self.root / "dates"
        if not dates_root.is_dir():
            raise DataError(f"Missing dates directory under {self.root}")
        self._date_dirs: dict[datetime.date, Path] = {}
        for entry in sorted(dates_root.iterdir()):
            if not entry.is_dir():
                continue
            try:
                date = datetime.date.fromisoformat(entry.name)
            except ValueError as exc:
                raise DataError(f"Unexpected date directory name: {entry}") from exc
            self._date_dirs[date] = entry

    # -- date listing --------------------------------------------------------
    def dates(self, split: str = "all") -> list[datetime.date]:
        """List shard dates. ``split`` is one of all|visible|hidden."""
        if split == "all":
            return sorted(self._date_dirs)
        if split == "visible":
            return sorted(d for d in self._date_dirs if d < self.hidden_start)
        if split == "hidden":
            return sorted(d for d in self._date_dirs if d >= self.hidden_start)
        raise ValueError(f"Unknown split {split!r}")

    def split_of(self, date: datetime.date) -> str:
        return "hidden" if date >= self.hidden_start else "visible"

    # -- shard access --------------------------------------------------------
    def shard(self, date: datetime.date) -> DateShard:
        try:
            path = self._date_dirs[date]
        except KeyError as exc:
            raise DataError(f"No shard for {date} under {self.root}") from exc
        meta = _load_json(path / "meta.json")
        if str(meta.get("dataset_id")) != self.dataset_id:
            raise DataError(f"{path}: dataset_id mismatch with dataset.json")
        x_descriptor = meta.get("arrays", {}).get("X.npy", {})
        shape = list(x_descriptor.get("shape", []))
        if len(shape) != 3 or shape[0] != self.time_rows or shape[2] != self.n_features:
            raise DataError(f"{path}: unexpected X shape metadata {shape}")
        return DateShard(
            date=date,
            path=path,
            split=self.split_of(date),
            n_symbols=int(shape[1]),
            meta=meta,
        )

    def iter_shards(self, dates: Sequence[datetime.date]) -> Iterator[DateShard]:
        for date in dates:
            yield self.shard(date)

    # -- validation ----------------------------------------------------------
    def verify(
        self,
        dates: Sequence[datetime.date] | None = None,
        deep: bool = False,
    ) -> int:
        """Verify shard structure (and hashes with ``deep=True``).

        Returns the number of shards checked. Raises DataError on the first
        violation.
        """
        targets = list(dates) if dates is not None else self.dates("all")
        for date in targets:
            shard = self.shard(date)
            shard.verify(deep=deep)
            if shard.split == "visible" and not shard.has_labels:
                raise DataError(f"{shard.path}: visible date missing y/w")
            if shard.split == "hidden" and shard.has_labels:
                raise DataError(f"{shard.path}: hidden date must not carry labels")
        return len(targets)
