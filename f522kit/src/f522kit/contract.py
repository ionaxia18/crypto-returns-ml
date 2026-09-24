"""Frozen protocol constants for the anonymous student tensor datasets.

Everything in this module is part of the published evaluation contract.
Students must not change these values; the canonical evaluator always uses
this module, so a locally modified copy only breaks reproduction.

Values marked TBD are policy placeholders awaiting team ratification. They
default to the safest interpretation and will be frozen before hidden
scoring begins.
"""

from __future__ import annotations

import datetime
import math

SCHEMA_VERSION = 1  # v1 datasets; kept for backward compatibility
SUPPORTED_SCHEMA_VERSIONS = (1, 2)

# ---------------------------------------------------------------------------
# Model-assignment rules (schema-selected per dataset)
# ---------------------------------------------------------------------------
# Schema-1 datasets use "previous_group".
# Schema-2 datasets MUST name their rule in dataset.json ("assignment_rule");
# the v2 core-980 dataset uses "date_mature". See calendar.py for semantics.
ASSIGNMENT_RULES = ("previous_group", "date_mature")
DEFAULT_ASSIGNMENT_RULE = "previous_group"

# ---------------------------------------------------------------------------
# Dataset identity
# ---------------------------------------------------------------------------
DAILY_TIME_ROWS = 1440
N_FEATURES = 522
TARGET_NAME = "y"
TARGET_HORIZON_MINUTES = 30
WEIGHT_NAME = "w"

# ---------------------------------------------------------------------------
# Rolling protocol
# ---------------------------------------------------------------------------
# Weekly model groups are keyed by the ISO week-end Sunday. The model for an
# evaluation group is the one keyed at the *previous* weekly group, trained on
# at most TRAINING_WINDOW_WEEKS weekly groups strictly before that key.
TRAINING_WINDOW_WEEKS = 104
HALFLIFE_DAYS = 1460.0

# Development / promotion / final-audit refit strides, in weekly groups.
STRIDES = {"4w": 4, "2w": 2, "1w": 1}
DEFAULT_STRIDE = "4w"

# ---------------------------------------------------------------------------
# Target / prediction scaling
# ---------------------------------------------------------------------------
# The stored y is already clipped to +/- TARGET_CLIP_SCALE * sqrt(horizon)
# and zero-filled where the source target was missing (those rows carry
# exactly zero evaluation weight). Predictions are clipped by the evaluator
# to +/- ALPHA_CLIP_SCALE * sqrt(horizon) after calibration.
TARGET_CLIP_SCALE = 0.03
ALPHA_CLIP_SCALE = 0.02
TARGET_CLIP = TARGET_CLIP_SCALE * math.sqrt(TARGET_HORIZON_MINUTES)
ALPHA_CLIP = ALPHA_CLIP_SCALE * math.sqrt(TARGET_HORIZON_MINUTES)

# Reference linear baseline regularization (normalized-space ridge; see
# docs/CONTRACT.md section "Reference ridge solve").
RIDGE_LAMBDA = 0.01

# ---------------------------------------------------------------------------
# Metric conventions
# ---------------------------------------------------------------------------
NUMERICAL_EPS = 1e-12
ANNUALIZATION_DAYS = 365.0

# ---------------------------------------------------------------------------
# Splits (dataset-relative; hidden labels are never published)
# ---------------------------------------------------------------------------
DATASET_START = datetime.date(2021, 5, 7)
DATASET_END = datetime.date(2025, 12, 30)
HIDDEN_START = datetime.date(2025, 1, 1)

# Scored report range starts here (earlier dates are training history only).
REPORT_START = datetime.date(2022, 7, 1)

# Default labeled scoring windows on the visible period.
DEFAULT_WINDOWS = {
    "is": (datetime.date(2022, 7, 1), datetime.date(2024, 6, 30)),
    "dev": (datetime.date(2024, 7, 1), datetime.date(2024, 12, 31)),
}

# ---------------------------------------------------------------------------
# Fit-excluded dates in the released labeled period (fit only; retained in
# prediction and reporting). Student fitting uses visible labels only.
# ---------------------------------------------------------------------------
EXCLUDED_FIT_DATES = frozenset(
    datetime.date.fromisoformat(s)
    for s in (
        "2021-05-19",
        "2021-09-07",
        "2021-12-04",
        "2022-05-11",
        "2022-05-12",
        "2022-06-07",
        "2022-11-03",
        "2022-11-08",
        "2022-11-09",
        "2023-06-10",
        "2023-08-17",
        "2023-11-09",
        "2024-01-03",
        "2024-04-12",
    )
)

# ---------------------------------------------------------------------------
# Policy placeholders (TBD; frozen before hidden scoring)
# ---------------------------------------------------------------------------
# TBD-1: causal prediction-scale calibration rule. "identity" scores raw
# predictions (COR is scale-invariant; APS/AR are not once the alpha clip
# binds, so keep your prediction scale in return units).
CALIBRATION_RULE = "identity"

# TBD-3: fair-compute training-row budget per fit. None means "not yet
# frozen"; the driver refuses to run without an explicit --row-budget until
# it is. The budget caps the frozen sampler (train.sample()); full-window
# streaming via train.iter_dates() is the sanctioned full-coverage path for
# the linear reference and is audited through the run manifest (fit time,
# training-date counts) rather than mechanically capped.
DEFAULT_ROW_BUDGET: int | None = None
