import datetime

import pytest

from f522kit import contract
from f522kit.calendar import build_schedule, group_dates_by_week, week_end  # noqa: F401


def _dates(start: str, days: int) -> list[datetime.date]:
    first = datetime.date.fromisoformat(start)
    return [first + datetime.timedelta(days=i) for i in range(days)]


def test_week_end_is_sunday_on_or_after():
    # 2023-01-02 is a Monday; its ISO week ends Sunday 2023-01-08.
    assert week_end(datetime.date(2023, 1, 2)) == datetime.date(2023, 1, 8)
    # Sunday maps to itself.
    assert week_end(datetime.date(2023, 1, 8)) == datetime.date(2023, 1, 8)
    assert week_end(datetime.date(2023, 1, 7)) == datetime.date(2023, 1, 8)


def test_group_dates_by_week_keys_and_membership():
    dates = _dates("2023-01-02", 21)
    groups = group_dates_by_week(dates)
    assert sorted(groups) == [
        datetime.date(2023, 1, 8),
        datetime.date(2023, 1, 15),
        datetime.date(2023, 1, 22),
    ]
    assert all(len(v) == 7 for v in groups.values())
    for key, members in groups.items():
        assert all(week_end(d) == key for d in members)


def test_first_two_groups_are_unpredictable():
    dates = _dates("2023-01-02", 35)  # 5 full weeks
    schedule = build_schedule(dates, stride_weeks=1)
    scored = schedule.scored_dates
    # Weeks ending 01-08 and 01-15 are unpredictable; scoring starts 01-16.
    assert scored[0] == datetime.date(2023, 1, 16)


def test_weekly_assignment_uses_previous_group_model():
    dates = _dates("2023-01-02", 42)
    schedule = build_schedule(dates, stride_weeks=1)
    for date, anchor in schedule.assignment.items():
        eval_key = week_end(date)
        # The model key must be exactly one weekly group before the eval key.
        idx = schedule.weekly_keys.index(eval_key)
        assert anchor.key == schedule.weekly_keys[idx - 1]
        # And trained only on groups strictly before its own key.
        assert all(k < anchor.key for k in anchor.training_keys)


def test_stride_anchor_position_formula():
    """anchor_position(i, k) = 1 + floor((i-2)/k)*k for every stride."""
    dates = _dates("2023-01-02", 16 * 7)
    for k in (1, 2, 4):
        schedule = build_schedule(dates, stride_weeks=k)
        for date, anchor in schedule.assignment.items():
            i = schedule.weekly_keys.index(week_end(date))
            expected = 1 + ((i - 2) // k) * k
            assert anchor.position == expected, (date, k, anchor.position, expected)
    # k=1 must reproduce exact weekly previous-group semantics.
    weekly = build_schedule(dates, stride_weeks=1)
    for date, anchor in weekly.assignment.items():
        i = weekly.weekly_keys.index(week_end(date))
        assert anchor.position == i - 1


def test_stride_4_uses_latest_anchor_not_future():
    dates = _dates("2023-01-02", 12 * 7)
    schedule = build_schedule(dates, stride_weeks=4)
    anchor_keys = {a.key for a in schedule.anchors}
    for date, anchor in schedule.assignment.items():
        assert anchor.key in anchor_keys
        # Causality: the anchor is at least one full week before the date's
        # own weekly group.
        assert anchor.key <= week_end(date) - datetime.timedelta(days=7)
    # Anchors are every 4th weekly position starting at the first valid one.
    positions = sorted(a.position for a in schedule.anchors)
    assert positions[0] == 1
    assert all(b - a == 4 for a, b in zip(positions, positions[1:]))


def test_training_window_truncates_at_104_groups():
    dates = _dates("2020-01-06", 120 * 7)
    schedule = build_schedule(dates, stride_weeks=1)
    last_anchor = max(schedule.anchors, key=lambda a: a.position)
    assert len(last_anchor.training_keys) == contract.TRAINING_WINDOW_WEEKS
    early_anchor = min(schedule.anchors, key=lambda a: a.position)
    assert len(early_anchor.training_keys) < contract.TRAINING_WINDOW_WEEKS


def test_excluded_dates_removed_from_training_but_still_scored():
    excluded = datetime.date(2022, 11, 8)
    assert excluded in contract.EXCLUDED_FIT_DATES
    dates = _dates("2022-10-03", 63)
    schedule = build_schedule(dates, stride_weeks=1)
    # Still predicted:
    assert excluded in schedule.assignment
    # Never in any anchor's training dates:
    for anchor in schedule.anchors:
        assert excluded not in anchor.training_dates(schedule.groups)


def test_decay_halflife():
    dates = _dates("2023-01-02", 28)
    schedule = build_schedule(dates, stride_weeks=1)
    anchor = schedule.anchors[-1]
    d = anchor.key - datetime.timedelta(days=int(contract.HALFLIFE_DAYS))
    assert anchor.decay(d) == pytest.approx(0.5)
    assert anchor.decay(anchor.key) == pytest.approx(1.0)
