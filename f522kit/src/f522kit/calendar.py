"""Canonical rolling calendar: weekly groups, refit anchors, model assignment.

Semantics (frozen; see docs/CONTRACT.md):

- Dates group into weekly model groups keyed by the ISO week-end Sunday.
- The weekly model at key ``G_p`` trains on at most ``TRAINING_WINDOW_WEEKS``
  weekly groups strictly before ``G_p`` (time-decayed toward ``G_p``).
- Two assignment rules exist, selected by the dataset (schema v2 datasets
  name theirs in ``dataset.json``):

  * ``previous_group`` (v1 datasets): dates in evaluation group ``G_i`` are
    scored with the model keyed at the previous group ``G_{i-1}``.
  * ``date_mature`` (v2 datasets): a date uses the newest model whose entire
    training window's h30 label endpoints mature strictly before the date's
    00:00. The model keyed ``G_i`` trains through Sunday ``G_{i-1}``, whose
    last labels mature early Monday — so Tuesday..Sunday of group ``G_i``
    use their own group's model and Monday falls back to ``G_{i-1}``.

- Under both rules the first weekly group has no training history and the
  scored range starts with the third weekly group (the contract-pinned
  scored set).
- A refit stride of ``k`` weeks keeps every ``k``-th *valid* weekly model
  position, starting at the first valid one; between anchors the most
  recent anchored model at-or-before the rule's target applies. ``k = 1``
  reproduces the rule's weekly semantics exactly.
- Fit-excluded dates are removed from training windows only; they are still
  predicted and scored.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

from . import contract


def week_end(date: datetime.date) -> datetime.date:
    """The Sunday on-or-after ``date`` (ISO week end)."""
    return date + datetime.timedelta(days=(6 - date.weekday()) % 7)


def group_dates_by_week(
    dates: Iterable[datetime.date],
) -> dict[datetime.date, list[datetime.date]]:
    """Group dates by ISO week, keyed by the week-end Sunday."""
    groups: dict[datetime.date, list[datetime.date]] = {}
    for date in sorted(dates):
        groups.setdefault(week_end(date), []).append(date)
    return groups


@dataclass(frozen=True)
class Anchor:
    """One refit anchor: a weekly model position that is actually fitted."""

    key: datetime.date                     # weekly group key (Sunday)
    position: int                          # index in the sorted weekly key list
    training_keys: tuple[datetime.date, ...]  # <=104 weekly keys strictly before key

    def training_dates(
        self,
        groups: Mapping[datetime.date, Sequence[datetime.date]],
        apply_exclusions: bool = True,
    ) -> list[datetime.date]:
        """Flattened training dates, minus the frozen fit exclusions."""
        out: list[datetime.date] = []
        for key in self.training_keys:
            for date in groups[key]:
                if apply_exclusions and date in contract.EXCLUDED_FIT_DATES:
                    continue
                out.append(date)
        return out

    def decay(self, date: datetime.date) -> float:
        """Frozen training decay of one date toward this anchor."""
        age = (self.key - date).days
        return 0.5 ** (age / contract.HALFLIFE_DAYS)


@dataclass(frozen=True)
class Schedule:
    """A resolved refit schedule over one set of dataset dates."""

    stride_weeks: int
    weekly_keys: tuple[datetime.date, ...]
    groups: dict[datetime.date, list[datetime.date]]
    anchors: tuple[Anchor, ...]
    rule: str = contract.DEFAULT_ASSIGNMENT_RULE
    # eval date -> anchor used to predict it (dates in unpredictable groups
    # are absent)
    assignment: dict[datetime.date, Anchor] = field(default_factory=dict)

    @property
    def scored_dates(self) -> list[datetime.date]:
        return sorted(self.assignment)


def _target_model_position(rule: str, group_index: int, date: datetime.date) -> int:
    """The weekly model position the rule aims at, before stride quantization."""
    if rule == "previous_group":
        return group_index - 1
    if rule == "date_mature":
        # Monday: the own-group model's final training labels (previous
        # Sunday's h30 endpoints) mature only during Monday, so fall back one
        # group. Tuesday..Sunday: the own-group model is mature.
        return group_index - 1 if date.weekday() == 0 else group_index
    raise ValueError(f"Unknown assignment rule {rule!r}")


def build_schedule(
    dates: Sequence[datetime.date],
    stride_weeks: int = 1,
    training_window_weeks: int = contract.TRAINING_WINDOW_WEEKS,
    rule: str = contract.DEFAULT_ASSIGNMENT_RULE,
) -> Schedule:
    """Build the frozen refit schedule for ``dates`` under ``rule``.

    ``dates`` should normally be every dataset date in the modeling range
    (training history plus scored range); the schedule derives weekly groups,
    valid anchors, and the per-date model assignment from it.
    """
    if stride_weeks < 1:
        raise ValueError("stride_weeks must be >= 1")
    if rule not in contract.ASSIGNMENT_RULES:
        raise ValueError(f"Unknown assignment rule {rule!r}")
    groups = group_dates_by_week(dates)
    keys = tuple(sorted(groups))

    def anchor_at(position: int) -> Anchor:
        start = max(0, position - training_window_weeks)
        return Anchor(
            key=keys[position],
            position=position,
            training_keys=keys[start:position],
        )

    # Valid model positions are those with at least one training group:
    # position 0 never has one. Anchors are every stride-th valid position.
    first_valid = 1
    anchor_positions = [
        p for p in range(first_valid, len(keys)) if (p - first_valid) % stride_weeks == 0
    ]
    anchors = {p: anchor_at(p) for p in anchor_positions}

    assignment: dict[datetime.date, Anchor] = {}
    first_scored_group = 2  # contract-pinned scored set under both rules
    for i, key in enumerate(keys):
        if i < first_scored_group:
            continue
        for date in groups[key]:
            target = _target_model_position(rule, i, date)
            if target < first_valid:
                continue
            anchored = (
                first_valid
                + ((target - first_valid) // stride_weeks) * stride_weeks
            )
            assignment[date] = anchors[anchored]

    used_positions = sorted({anchor.position for anchor in assignment.values()})
    used = tuple(anchors[p] for p in used_positions)
    return Schedule(
        stride_weeks=stride_weeks,
        weekly_keys=keys,
        groups=groups,
        anchors=used,
        rule=rule,
        assignment=assignment,
    )
