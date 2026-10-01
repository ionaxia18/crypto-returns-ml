"""Build the experiment split: ONE fit, cached as flat .npy memmaps.

No rolling. The split copies a single refit of the official 4w schedule:
the first anchor whose predicted dates start on/after ``VAL_AFTER``
(2024-04-01, just before the contract's dev window, which stays untouched).

- train: the last ``train_weeks`` weekly groups before that anchor (fit
  exclusions applied, zero-weight rows dropped, the kit's half-life decay
  toward the anchor multiplied into ``w``), keeping every
  ``minute_stride``-th minute of each day;
- val: exactly the ~4 weeks of dates the official run scores with that
  anchor, every positive-weight row. Zero-weight rows add nothing to any
  sufficient statistic, so metrics on val equal ``f522kit score`` on
  those dates.

The split is fully deterministic: no random sampling anywhere. The only
knobs are ``--train-weeks`` and ``--minute-stride``; change them only to
measure what the subsampling costs (README roadmap stage 2).

Calendar, exclusions and decay come from the kit (``resolve_schedule``,
``Anchor``) so they cannot drift from the contract. Train rows are read
minute by minute from the [1440, S, K] memmap, so stride 15 reads ~1/15
of each training shard.

This reads shards directly and caches labels: fine for private tooling,
out of contract inside a submitted ``fit``.

Layout of ``<out>/`` (``<part>`` is ``train`` or ``val``)::

    meta.json                 settings, date lists, row counts
    <part>_X.npy       float32 [n, K]
    <part>_y.npy       float32 [n]
    <part>_w.npy       float64 [n]   (train: decay already applied)
    <part>_date.npy    int32   [n]   index into meta["<part>_dates"]
    <part>_row.npy     int32   [n]   flat row (minute * S + symbol) in that date
    <part>_minute.npy  int16   [n]   minute of day

meta.json is written last; a crashed build has none and will not open.

Usage::

    python -m experiment.split --data $DATA_ROOT --out $SCRATCH/core980-work/experiment/v1
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from f522kit.calendar import Anchor
from f522kit.data import Dataset
from f522kit.driver import KIT_VERSION, resolve_schedule

SPLIT_VERSION = 2
VAL_AFTER = datetime.date(2024, 4, 1)
TRAIN_WEEKS = 26
MINUTE_STRIDE = 15
WRITE_CHUNK_ROWS = 32_768  # bounds per-worker RAM (~128 MB of X at K=980)


def choose_anchor(dataset: Dataset) -> tuple[Anchor, list[datetime.date], dict]:
    """The 4w anchor whose scored dates start on/after VAL_AFTER.

    Returns the anchor, the dates it scores (= val), and the weekly groups.
    """
    schedule = resolve_schedule(dataset, "4w")
    by_anchor: dict[datetime.date, list[datetime.date]] = {}
    for date, anchor in schedule.assignment.items():
        by_anchor.setdefault(anchor.key, []).append(date)
    candidates = sorted(k for k, ds in by_anchor.items() if min(ds) >= VAL_AFTER)
    if not candidates:
        raise ValueError(f"no 4w anchor scores dates on/after {VAL_AFTER}")
    key = candidates[0]
    anchor = next(a for a in schedule.anchors if a.key == key)
    return anchor, sorted(by_anchor[key]), schedule.groups


def train_dates_for(anchor: Anchor, groups: dict, train_weeks: int) -> list[datetime.date]:
    """Last ``train_weeks`` groups of the anchor's window, minus exclusions.

    The anchor key is unchanged, so decay values match the full window.
    """
    keys = anchor.training_keys[-train_weeks:]
    truncated = Anchor(key=anchor.key, position=anchor.position, training_keys=keys)
    return truncated.training_dates(groups)


def _select_rows(w: np.ndarray, n_symbols: int, minute_stride: int) -> np.ndarray:
    """Sorted flat row indices of positive-weight rows at minutes 0, s, 2s, ..."""
    w2 = np.asarray(w).reshape(-1, n_symbols)
    minutes = np.arange(0, w2.shape[0], minute_stride)
    m_idx, s_idx = np.nonzero(w2[minutes] > 0)
    return (minutes[m_idx] * n_symbols + s_idx).astype(np.int64)


def _gather_x(x3: np.ndarray, rows: np.ndarray, n_symbols: int) -> np.ndarray:
    """Read rows minute by minute so only touched minutes are paged in."""
    minutes = rows // n_symbols
    syms = rows % n_symbols
    out = np.empty((rows.size, x3.shape[2]), dtype=np.float32)
    # rows are sorted, so each minute is one contiguous run in ``out``.
    bounds = np.flatnonzero(np.diff(minutes)) + 1
    starts = np.concatenate([[0], bounds])
    ends = np.concatenate([bounds, [rows.size]])
    for s, e in zip(starts, ends):
        out[s:e] = x3[minutes[s]][syms[s:e]]
    return out


def _open(path: Path, shape: tuple, dtype) -> np.ndarray:
    return np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)


def _fill_part(
    dataset: Dataset,
    out: Path,
    part: str,
    dates: list[datetime.date],
    row_sel: dict,
    decay: dict | None,
    workers: int,
    log_every: int = 10,
) -> dict:
    """Write one part (train or val) of the cache; return its row counts."""
    counts = [row_sel[d].size for d in dates]
    offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    n = int(offsets[-1])
    X = _open(out / f"{part}_X.npy", (n, dataset.n_features), np.float32)
    Y = _open(out / f"{part}_y.npy", (n,), np.float32)
    W = _open(out / f"{part}_w.npy", (n,), np.float64)
    D = _open(out / f"{part}_date.npy", (n,), np.int32)
    R = _open(out / f"{part}_row.npy", (n,), np.int32)
    M = _open(out / f"{part}_minute.npy", (n,), np.int16)

    def work(i: int) -> None:
        date = dates[i]
        shard = dataset.shard(date)
        S = shard.n_symbols
        x3 = shard.load_x()
        y = np.asarray(shard.load_y()).reshape(-1)
        w = np.asarray(shard.load_w()).reshape(-1)
        factor = decay[date] if decay is not None else 1.0
        rows = row_sel[date]
        # Each thread owns a disjoint slice [offsets[i], offsets[i+1]); no locks.
        for c in range(0, rows.size, WRITE_CHUNK_ROWS):
            r = rows[c:c + WRITE_CHUNK_ROWS]
            s = offsets[i] + c
            e = s + r.size
            X[s:e] = _gather_x(x3, r, S)
            Y[s:e] = y[r]
            W[s:e] = w[r].astype(np.float64) * factor
            D[s:e] = i
            R[s:e] = r
            M[s:e] = r // S

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for j, _ in enumerate(pool.map(work, range(len(dates))), 1):
            if j % log_every == 0 or j == len(dates):
                print(f"  [{part}] {j}/{len(dates)} dates, {time.time() - t0:.0f}s",
                      flush=True)
    for arr in (X, Y, W, D, R, M):
        arr.flush()
    return {"n_rows": n, "rows_per_date": counts}


def build_split(
    data_root: str | Path,
    out_dir: str | Path,
    train_weeks: int = TRAIN_WEEKS,
    minute_stride: int = MINUTE_STRIDE,
    workers: int = 8,
    force: bool = False,
) -> dict:
    """Build and write the experiment split; returns its meta dictionary."""
    t0 = time.time()
    out = Path(out_dir)
    if out.exists() and any(out.iterdir()):
        if not force:
            raise FileExistsError(
                f"{out} is not empty; results may point at it. Use a new --out "
                "or pass --force to overwrite.")
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    dataset = Dataset(data_root)
    anchor, val_dates, groups = choose_anchor(dataset)
    train_dates = train_dates_for(anchor, groups, train_weeks)
    print(f"anchor {anchor.key}: train {train_dates[0]}..{train_dates[-1]} "
          f"({len(train_dates)} dates, {train_weeks}w, minute stride {minute_stride}), "
          f"val {val_dates[0]}..{val_dates[-1]} ({len(val_dates)} dates, all rows)",
          flush=True)

    def select(dates, stride):
        def one(d):
            shard = dataset.shard(d)
            if not shard.has_labels:
                raise ValueError(f"{d}: no labels")
            return d, _select_rows(shard.load_w(), shard.n_symbols, stride)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return dict(pool.map(one, dates))

    train_sel = select(train_dates, minute_stride)
    val_sel = select(val_dates, 1)
    decay = {d: anchor.decay(d) for d in train_dates}

    train_info = _fill_part(dataset, out, "train", train_dates, train_sel, decay, workers)
    val_info = _fill_part(dataset, out, "val", val_dates, val_sel, None, workers)

    meta = {
        "split_version": SPLIT_VERSION,
        "kit_version": KIT_VERSION,
        "dataset_id": dataset.dataset_id,
        "data_root": str(data_root),
        "n_features": dataset.n_features,
        "anchor": anchor.key.isoformat(),
        "train_weeks": train_weeks,
        "minute_stride": minute_stride,
        "train_dates": [d.isoformat() for d in train_dates],
        "val_dates": [d.isoformat() for d in val_dates],
        "train": train_info,
        "val": val_info,
        "build_seconds": round(time.time() - t0, 1),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    gb = (train_info["n_rows"] + val_info["n_rows"]) * dataset.n_features * 4 / 1e9
    print(f"done: train {train_info['n_rows']:,} rows, val {val_info['n_rows']:,} rows, "
          f"~{gb:.1f} GB X, {meta['build_seconds']:.0f}s", flush=True)
    return meta


class ExperimentSplit:
    """Read-only handle on a built split (X memory-mapped)."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.meta = json.loads((self.root / "meta.json").read_text(encoding="utf-8"))
        self.train_dates = [datetime.date.fromisoformat(d) for d in self.meta["train_dates"]]
        self.val_dates = [datetime.date.fromisoformat(d) for d in self.meta["val_dates"]]

    def _load(self, name: str, mmap: bool = True) -> np.ndarray:
        return np.load(self.root / f"{name}.npy", mmap_mode="r" if mmap else None)

    def train(self, in_ram: bool = False):
        """(X, y, w, date_idx); X stays memory-mapped unless ``in_ram``."""
        return (self._load("train_X", not in_ram), self._load("train_y", False),
                self._load("train_w", False), self._load("train_date", False))

    def val(self, in_ram: bool = False):
        """(X, y, w, date_idx)."""
        return (self._load("val_X", not in_ram), self._load("val_y", False),
                self._load("val_w", False), self._load("val_date", False))

    def val_rows(self) -> np.ndarray:
        return self._load("val_row", False)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m experiment.split", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--train-weeks", type=int, default=TRAIN_WEEKS)
    p.add_argument("--minute-stride", type=int, default=MINUTE_STRIDE)
    p.add_argument("--workers", type=int,
                   default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    p.add_argument("--force", action="store_true", help="overwrite a non-empty --out")
    a = p.parse_args(argv)
    build_split(a.data, a.out, train_weeks=a.train_weeks, minute_stride=a.minute_stride,
                workers=a.workers, force=a.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
