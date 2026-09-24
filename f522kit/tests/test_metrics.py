import datetime
import math

import numpy as np
import pytest

from f522kit import contract
from f522kit.metrics import (
    aggregate,
    annualized_sharpe,
    clip_alpha,
    daily_stats,
    paired_diff,
)

DATE = datetime.date(2024, 1, 2)


def test_daily_stats_match_hand_computation():
    alpha = np.array([0.001, -0.002, 0.0005, 0.0])
    y = np.array([0.002, -0.001, -0.0005, 0.01])
    w = np.array([1.0, 2.0, 0.5, 0.0])
    stats = daily_stats(DATE, alpha, y, w)

    awa = float(np.sum(w * alpha * alpha))
    awy = float(np.sum(w * alpha * y))
    ywy = float(np.sum(w * y * y))
    sum_w = float(np.sum(w))
    assert stats.awa == pytest.approx(awa, rel=1e-15)
    assert stats.awy == pytest.approx(awy, rel=1e-15)
    assert stats.ywy == pytest.approx(ywy, rel=1e-15)
    assert stats.sum_w == pytest.approx(sum_w, rel=1e-15)

    assert stats.cor == pytest.approx(awy / math.sqrt(awa * ywy), rel=1e-12)
    assert stats.ar == pytest.approx(awy / awa, rel=1e-12)
    assert stats.aps == pytest.approx(awy / math.sqrt(awa) / math.sqrt(sum_w), rel=1e-12)


def test_scale_invariance_below_clip():
    rng = np.random.default_rng(7)
    alpha = rng.standard_normal(1000) * 1e-3
    y = alpha * 0.5 + rng.standard_normal(1000) * 1e-3
    w = rng.uniform(0.5, 2.0, 1000)
    base = daily_stats(DATE, alpha, y, w)
    scaled = daily_stats(DATE, alpha * 3.0, y, w)  # still below the clip
    # COR and unclipped APS are invariant to positive rescaling; AR is not.
    assert scaled.cor == pytest.approx(base.cor, rel=1e-12)
    assert scaled.aps == pytest.approx(base.aps, rel=1e-9)
    assert scaled.ar == pytest.approx(base.ar / 3.0, rel=1e-9)


def test_aps_becomes_scale_sensitive_once_clip_binds():
    rng = np.random.default_rng(9)
    alpha = rng.standard_normal(1000) * 1e-3
    y = alpha + rng.standard_normal(1000) * 1e-3
    w = np.ones(1000)
    base = daily_stats(DATE, alpha, y, w)
    # Scale far past the clip bound: most alphas saturate at +/- ALPHA_CLIP.
    saturated = daily_stats(DATE, alpha * 1e4, y, w)
    assert saturated.aps != pytest.approx(base.aps, rel=1e-3)


def test_alpha_clip_binds_at_contract_bound():
    bound = contract.ALPHA_CLIP
    assert bound == pytest.approx(0.02 * math.sqrt(30))
    alpha = np.array([bound * 10, -bound * 10, bound / 2])
    clipped = clip_alpha(alpha)
    assert clipped[0] == pytest.approx(bound)
    assert clipped[1] == pytest.approx(-bound)
    assert clipped[2] == pytest.approx(bound / 2)


def test_daily_stats_clips_by_default():
    alpha = np.array([10.0, -10.0])
    y = np.array([0.01, -0.01])
    w = np.array([1.0, 1.0])
    stats = daily_stats(DATE, alpha, y, w)
    assert stats.awa == pytest.approx(2 * contract.ALPHA_CLIP**2, rel=1e-12)


def test_annualized_sharpe():
    values = np.array([1.0, 2.0, 3.0, 4.0])
    expected = values.mean() / values.std(ddof=1) * math.sqrt(365.0)
    assert annualized_sharpe(values) == pytest.approx(expected, rel=1e-12)
    assert math.isnan(annualized_sharpe(np.array([1.0])))
    assert math.isnan(annualized_sharpe(np.ones(10)))


def test_aggregate_is_mean_daily():
    rng = np.random.default_rng(11)
    stats = []
    for i in range(30):
        alpha = rng.standard_normal(200) * 1e-3
        y = alpha + rng.standard_normal(200) * 2e-3
        w = rng.uniform(0.5, 2.0, 200)
        stats.append(
            daily_stats(DATE + datetime.timedelta(days=i), alpha, y, w)
        )
    agg = aggregate(stats)
    assert agg["n_dates"] == 30
    assert agg["COR_mean"] == pytest.approx(
        float(np.mean([s.cor for s in stats])), rel=1e-12
    )
    assert agg["APS_mean_bps"] == pytest.approx(
        float(np.mean([s.aps for s in stats])) * 1e4, rel=1e-12
    )


def test_paired_diff_alignment_and_sign():
    rng = np.random.default_rng(3)
    model, base = [], []
    for i in range(20):
        d = DATE + datetime.timedelta(days=i)
        alpha = rng.standard_normal(100) * 1e-3
        y = alpha + rng.standard_normal(100) * 1e-3
        w = np.ones(100)
        model.append(daily_stats(d, alpha, y, w))
        # Baseline: sign-degraded predictions (mostly anti-correlated).
        base.append(daily_stats(d, -alpha, y, w))
    diff = paired_diff(model, base)
    assert diff["n_dates"] == 20
    assert diff["dAPS_mean_bps"] > 0
    assert diff["win_rate"] == pytest.approx(1.0)


def test_zero_weight_day_is_guarded_not_crashing():
    stats = daily_stats(DATE, np.zeros(5), np.zeros(5), np.zeros(5))
    assert stats.cor == 0.0
    assert stats.aps == 0.0
