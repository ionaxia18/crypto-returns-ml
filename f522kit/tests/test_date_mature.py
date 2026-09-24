"""Date-mature assignment rule + schema-2 contract profile tests."""

import datetime
import json

import numpy as np
import pytest

from f522kit import contract
from f522kit.calendar import build_schedule, week_end
from f522kit.data import DataError, Dataset
from f522kit.driver import run
from f522kit.fixture import build_fixture
from f522kit.scoring import score


def _dates(start: str, days: int) -> list[datetime.date]:
    first = datetime.date.fromisoformat(start)
    return [first + datetime.timedelta(days=i) for i in range(days)]


# ---------------------------------------------------------------------------
# Rule semantics
# ---------------------------------------------------------------------------

def test_date_mature_own_group_tue_sun_monday_fallback():
    dates = _dates("2023-01-02", 8 * 7)  # Mondays start each week
    schedule = build_schedule(dates, stride_weeks=1, rule="date_mature")
    for date, anchor in schedule.assignment.items():
        own = week_end(date)
        if date.weekday() == 0:  # Monday
            assert anchor.key == own - datetime.timedelta(days=7), date
        else:
            assert anchor.key == own, date


def test_date_mature_boundary_date_assignments():
    """Pinned examples at the beginning, middle, and end of the date range."""
    dates = _dates("2021-05-07", (datetime.date(2025, 12, 30) - datetime.date(2021, 5, 7)).days + 1)
    schedule = build_schedule(dates, stride_weeks=1, rule="date_mature")
    expect = {
        "2021-05-17": "2021-05-16",
        "2024-07-01": "2024-06-30",
        "2025-12-30": "2026-01-04",
    }
    for d, key in expect.items():
        anchor = schedule.assignment[datetime.date.fromisoformat(d)]
        assert anchor.key.isoformat() == key, (d, anchor.key)


def test_date_mature_maturity_invariant():
    """Every assigned model's training labels mature strictly before the date.

    h30 endpoints of training date t land early on t+1, so the invariant is
    date >= last_training_date + 2 days.
    """
    dates = _dates("2023-01-02", 12 * 7)
    schedule = build_schedule(dates, stride_weeks=1, rule="date_mature")
    for date, anchor in schedule.assignment.items():
        train_dates = anchor.training_dates(schedule.groups, apply_exclusions=False)
        assert train_dates, date
        assert date >= max(train_dates) + datetime.timedelta(days=2), date


def test_date_mature_scored_set_equals_previous_group_scored_set():
    dates = _dates("2021-05-07", 120)
    weekly = build_schedule(dates, stride_weeks=1, rule="previous_group")
    mature = build_schedule(dates, stride_weeks=1, rule="date_mature")
    assert weekly.scored_dates == mature.scored_dates  # pinned scored range


def test_date_mature_ledger_counts():
    # Full dataset range: 244 calendar groups, 243 used models.
    dates = _dates("2021-05-07", (datetime.date(2025, 12, 30) - datetime.date(2021, 5, 7)).days + 1)
    # thin to weekdays-present shape is unnecessary; every calendar day exists
    schedule = build_schedule(dates, stride_weeks=1, rule="date_mature")
    assert len(schedule.weekly_keys) == 244
    assert len(schedule.anchors) == 243  # positions 1..243 all used
    weekly = build_schedule(dates, stride_weeks=1, rule="previous_group")
    assert len(weekly.anchors) == 242  # last group's model unused under v1


def test_date_mature_stride_quantization():
    dates = _dates("2023-01-02", 16 * 7)
    for k in (2, 4):
        schedule = build_schedule(dates, stride_weeks=k, rule="date_mature")
        positions = sorted({a.position for a in schedule.anchors})
        assert all((p - 1) % k == 0 for p in positions)
        keys = schedule.weekly_keys
        for date, anchor in schedule.assignment.items():
            i = keys.index(week_end(date))
            target = i - 1 if date.weekday() == 0 else i
            expected = 1 + ((target - 1) // k) * k
            assert anchor.position == expected, (date, k)


def test_unknown_rule_rejected():
    with pytest.raises(ValueError, match="Unknown assignment rule"):
        build_schedule(_dates("2023-01-02", 14), rule="own_group")


# ---------------------------------------------------------------------------
# Schema-2 dataset contract
# ---------------------------------------------------------------------------

def test_schema2_fixture_selects_date_mature(tmp_path):
    root = build_fixture(tmp_path / "v2", schema_version=2)
    dataset = Dataset(root)
    assert dataset.schema_version == 2
    assert dataset.assignment_rule == "date_mature"
    info = json.loads((root / "dataset.json").read_text())
    # v0.1 readers gate on schema_version == 1, so they must reject this root.
    assert info["schema_version"] == 2


def test_schema1_fixture_pinned_to_previous_group(tmp_path):
    dataset = Dataset(build_fixture(tmp_path / "v1", schema_version=1))
    assert dataset.assignment_rule == "previous_group"


def test_schema2_requires_known_rule(tmp_path):
    root = build_fixture(tmp_path / "bad", schema_version=2)
    info = json.loads((root / "dataset.json").read_text())
    del info["assignment_rule"]
    (root / "dataset.json").write_text(json.dumps(info))
    with pytest.raises(DataError, match="assignment_rule"):
        Dataset(root)
    info["assignment_rule"] = "own_group"
    (root / "dataset.json").write_text(json.dumps(info))
    with pytest.raises(DataError, match="assignment_rule"):
        Dataset(root)


def test_unknown_schema_rejected(tmp_path):
    root = build_fixture(tmp_path / "v3", schema_version=1)
    info = json.loads((root / "dataset.json").read_text())
    info["schema_version"] = 3
    (root / "dataset.json").write_text(json.dumps(info))
    with pytest.raises(DataError, match="schema_version"):
        Dataset(root)


def test_schema1_contradictory_rule_rejected(tmp_path):
    root = build_fixture(tmp_path / "v1bad", schema_version=1)
    info = json.loads((root / "dataset.json").read_text())
    info["assignment_rule"] = "date_mature"
    (root / "dataset.json").write_text(json.dumps(info))
    with pytest.raises(DataError, match="pinned"):
        Dataset(root)
    # An explicit-but-matching declaration remains acceptable.
    info["assignment_rule"] = "previous_group"
    (root / "dataset.json").write_text(json.dumps(info))
    assert Dataset(root).assignment_rule == "previous_group"


# ---------------------------------------------------------------------------
# End-to-end on the schema-2 fixture
# ---------------------------------------------------------------------------

RIDGE = """
import numpy as np
class TinyRidge:
    def fit(self, train):
        X, y, w = train.sample(20000, seed=0)
        Xw = X * w[:, None]
        self.beta = np.linalg.solve(Xw.T @ X + 1e-6 * np.eye(X.shape[1]), Xw.T @ y)
    def predict(self, X):
        return X @ self.beta
"""


def test_schema2_end_to_end_recovers_signal(tmp_path):
    root = build_fixture(tmp_path / "data", schema_version=2)
    dataset = Dataset(root)
    model = tmp_path / "model.py"
    model.write_text(RIDGE, encoding="utf-8")
    out = tmp_path / "run"
    manifest = run(
        dataset,
        model_spec=f"{model}:TinyRidge",
        out_dir=out,
        stride="1w",
        row_budget=20000,
        seed=0,
        start=dataset.dates("visible")[0],
        verbose=False,
    )
    assert manifest["assignment_rule"] == "date_mature"
    # Non-Monday dates must use their own week's model key.
    for d, entry in manifest["predictions"].items():
        date = datetime.date.fromisoformat(d)
        own = week_end(date).isoformat()
        if date.weekday() != 0:
            assert entry["anchor"] == own, d
        else:
            assert entry["anchor"] < d
    first = dataset.dates("visible")[0]
    last = dataset.dates("visible")[-1]
    report = score(dataset, preds_dir=out, windows={"all": (first, last)})
    assert report["windows"]["all"]["COR_mean"] > 0.2
