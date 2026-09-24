"""Canonical evaluation metrics (frozen contract).

Per scored date, with alpha ``a`` (post-calibration, post-clip), target ``y``
(as stored), and effective weight ``w`` over all rows of that date:

    awa   = sum(w * a * a)
    awy   = sum(w * a * y)
    ywy   = sum(w * y * y)
    sum_w = sum(w)

    COR = awy / sqrt(awa * ywy)          (weighted uncentered cosine)
    AR  = awy / awa                      (slope of target on alpha)
    APS = awy / sqrt(awa) / sqrt(sum_w)  (return units; reported in bps)
    AS  = sqrt(awa / sum_w)
    RS  = sqrt(ywy / sum_w)

Aggregates over a window are the simple mean of the daily metrics
("mean-daily"). The stability statistic is the annualized Sharpe of the
daily APS series: mean / std(ddof=1) * sqrt(365). Paired comparison between
two runs uses the daily APS difference series and its Sharpe.

COR and *unclipped* APS are invariant to positive rescaling of predictions;
AR scales inversely with the prediction scale, and once the frozen alpha
clip binds, APS becomes scale-sensitive too. This is why the calibration
rule (TBD-1) matters for APS/AR reporting.
"""

from __future__ import annotations

import datetime
import math
from dataclasses import dataclass

import numpy as np

from . import contract


def clip_alpha(alpha: np.ndarray) -> np.ndarray:
    """Apply the frozen prediction clip (+/- ALPHA_CLIP)."""
    return np.clip(alpha, -contract.ALPHA_CLIP, contract.ALPHA_CLIP)


@dataclass(frozen=True)
class DailyStats:
    """Sufficient statistics of one scored date."""

    date: datetime.date
    awa: float
    awy: float
    ywy: float
    sum_w: float

    @property
    def cor(self) -> float:
        eps = contract.NUMERICAL_EPS
        denominator = math.sqrt(max(self.awa, eps) * max(self.ywy, eps))
        return self.awy / max(denominator, eps)

    @property
    def ar(self) -> float:
        return self.awy / max(self.awa, contract.NUMERICAL_EPS)

    @property
    def aps(self) -> float:
        eps = contract.NUMERICAL_EPS
        return self.awy / math.sqrt(max(self.awa, eps)) / math.sqrt(max(self.sum_w, eps))

    @property
    def alpha_size(self) -> float:
        eps = contract.NUMERICAL_EPS
        return math.sqrt(max(self.awa, 0.0) / max(self.sum_w, eps))

    @property
    def return_size(self) -> float:
        eps = contract.NUMERICAL_EPS
        return math.sqrt(max(self.ywy, 0.0) / max(self.sum_w, eps))


def daily_stats(
    date: datetime.date,
    alpha: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    already_clipped: bool = False,
) -> DailyStats:
    """Compute one date's sufficient statistics in float64."""
    a = np.asarray(alpha, dtype=np.float64).reshape(-1)
    yv = np.asarray(y, dtype=np.float64).reshape(-1)
    wv = np.asarray(w, dtype=np.float64).reshape(-1)
    if not (a.shape == yv.shape == wv.shape):
        raise ValueError(
            f"{date}: mismatched shapes alpha={a.shape} y={yv.shape} w={wv.shape}"
        )
    if not already_clipped:
        a = clip_alpha(a)
    wa = wv * a
    return DailyStats(
        date=date,
        awa=float(np.dot(wa, a)),
        awy=float(np.dot(wa, yv)),
        ywy=float(np.dot(wv * yv, yv)),
        sum_w=float(np.sum(wv)),
    )


def annualized_sharpe(values: np.ndarray) -> float:
    """mean / std(ddof=1) * sqrt(365) of a daily series (nan if degenerate)."""
    values = np.asarray(values, dtype=np.float64)
    if values.size < 2:
        return float("nan")
    std = float(values.std(ddof=1))
    if abs(std) <= contract.NUMERICAL_EPS:
        return float("nan")
    return float(values.mean()) / std * math.sqrt(contract.ANNUALIZATION_DAYS)


def aggregate(stats: list[DailyStats]) -> dict:
    """Mean-daily aggregate metrics for one window of scored dates."""
    if not stats:
        raise ValueError("No scored dates in window")
    stats = sorted(stats, key=lambda s: s.date)
    daily_aps = np.array([s.aps for s in stats])
    daily_cor = np.array([s.cor for s in stats])
    daily_ar = np.array([s.ar for s in stats])
    return {
        "n_dates": len(stats),
        "start_date": stats[0].date.isoformat(),
        "end_date": stats[-1].date.isoformat(),
        "APS_mean_bps": float(daily_aps.mean()) * 1e4,
        "COR_mean": float(daily_cor.mean()),
        "AR_mean": float(daily_ar.mean()),
        "APS_SR": annualized_sharpe(daily_aps),
        "AS_mean": float(np.mean([s.alpha_size for s in stats])),
        "RS_mean": float(np.mean([s.return_size for s in stats])),
    }


def paired_diff(
    model: list[DailyStats],
    baseline: list[DailyStats],
) -> dict:
    """Daily APS difference (model - baseline) over common dates."""
    base_by_date = {s.date: s for s in baseline}
    common = [s for s in model if s.date in base_by_date]
    if not common:
        raise ValueError("No overlapping scored dates for paired comparison")
    diffs = np.array([s.aps - base_by_date[s.date].aps for s in common])
    return {
        "n_dates": len(common),
        "dAPS_mean_bps": float(diffs.mean()) * 1e4,
        "diff_sharpe": annualized_sharpe(diffs),
        "win_rate": float(np.mean(diffs > 0)),
    }
