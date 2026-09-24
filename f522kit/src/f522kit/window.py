"""Training-window views handed to student ``fit`` implementations.

A :class:`TrainWindow` never exposes dates after its anchor, applies the
frozen fit-exclusion list and half-life decay, and provides the frozen
deterministic row sampler. Students receive it as the only argument of
``fit``; everything else about the rolling protocol lives in the harness.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass
from typing import Iterator

import numpy as np

from .calendar import Anchor
from .data import Dataset


@dataclass(frozen=True)
class DateBlock:
    """All valid training rows of one date, decay-weighted toward the anchor."""

    date: datetime.date
    x: np.ndarray        # float32 [n, K] (view into the mmap where possible)
    y: np.ndarray        # float32 [n]
    w_train: np.ndarray  # float64 [n] = effective weight * halflife decay
    decay: float


class TrainWindow:
    """Lazy view over the training window of one refit anchor.

    Rows with zero effective weight carry no fit information under the
    weighted contract, so they are dropped from every block and sample.

    The public surface (``dates``, ``iter_dates``, ``sample``,
    ``valid_row_counts``) exposes only pre-anchor training data. The
    underlying dataset handle is private; reaching around it from student
    code is out of contract and treated as a protocol violation in review.
    """

    def __init__(
        self,
        dataset: Dataset,
        anchor: Anchor,
        dates: list[datetime.date],
        row_budget: int | None = None,
    ):
        self._dataset = dataset
        self.anchor = anchor
        self.row_budget = row_budget
        self.dates = [d for d in dates if d in set(dataset.dates("visible"))]
        self.dates.sort()

    @property
    def anchor_date(self) -> datetime.date:
        return self.anchor.key

    def __len__(self) -> int:
        return len(self.dates)

    # -- streaming access ----------------------------------------------------
    def iter_dates(self) -> Iterator[DateBlock]:
        """Yield per-date valid-row blocks in chronological order."""
        for date in self.dates:
            shard = self._dataset.shard(date)
            x2d, y, w = shard.flat_rows()
            valid = np.flatnonzero(w > 0)
            if valid.size == 0:
                continue
            decay = self.anchor.decay(date)
            yield DateBlock(
                date=date,
                x=x2d[valid],
                y=np.asarray(y[valid]),
                w_train=np.asarray(w[valid], dtype=np.float64) * decay,
                decay=decay,
            )

    # -- frozen deterministic sampler ---------------------------------------
    def valid_row_counts(self) -> dict[datetime.date, int]:
        """Number of positive-weight rows per training date (reads w only)."""
        counts: dict[datetime.date, int] = {}
        for date in self.dates:
            shard = self._dataset.shard(date)
            w = shard.load_w().reshape(-1)
            counts[date] = int(np.count_nonzero(w > 0))
        return counts

    def sample(
        self,
        max_rows: int,
        seed: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Materialize a deterministic date-stratified training sample.

        Rows are allocated across dates proportionally to each date's
        positive-weight row count (largest-remainder rounding so the total
        is exact), then chosen uniformly without replacement within each
        date using a per-date child seed. The rule and its seeds are part
        of the frozen contract: the same (window, max_rows, seed) always
        yields the same rows.

        Returns ``(X [n, K] float32, y [n] float32, w_train [n] float64)``.
        """
        if max_rows <= 0:
            raise ValueError("max_rows must be positive")
        if self.row_budget is not None and max_rows > self.row_budget:
            raise ValueError(
                f"max_rows {max_rows} exceeds the frozen training-row budget "
                f"{self.row_budget} for this run"
            )
        counts = self.valid_row_counts()
        dates = [d for d in self.dates if counts[d] > 0]
        total = sum(counts[d] for d in dates)
        if total == 0:
            raise ValueError("Training window has no positive-weight rows")
        budget = min(max_rows, total)

        # Largest-remainder allocation, capped by per-date availability.
        raw = {d: budget * counts[d] / total for d in dates}
        alloc = {d: min(int(raw[d]), counts[d]) for d in dates}
        remainder = budget - sum(alloc.values())
        by_frac = sorted(
            dates, key=lambda d: (raw[d] - int(raw[d]), d.toordinal()), reverse=True
        )
        idx = 0
        while remainder > 0 and idx < 10 * len(by_frac):
            date = by_frac[idx % len(by_frac)]
            if alloc[date] < counts[date]:
                alloc[date] += 1
                remainder -= 1
            idx += 1

        xs: list[np.ndarray] = []
        ys: list[np.ndarray] = []
        ws: list[np.ndarray] = []
        for date in dates:
            n_take = alloc[date]
            if n_take == 0:
                continue
            shard = self._dataset.shard(date)
            x2d, y, w = shard.flat_rows()
            valid = np.flatnonzero(w > 0)
            rng = np.random.default_rng(
                np.random.SeedSequence([int(seed), date.toordinal()])
            )
            take = valid if n_take >= valid.size else np.sort(
                rng.choice(valid, size=n_take, replace=False)
            )
            decay = self.anchor.decay(date)
            xs.append(np.asarray(x2d[take], dtype=np.float32))
            ys.append(np.asarray(y[take], dtype=np.float32))
            ws.append(np.asarray(w[take], dtype=np.float64) * decay)

        return (
            np.concatenate(xs, axis=0),
            np.concatenate(ys, axis=0),
            np.concatenate(ws, axis=0),
        )
