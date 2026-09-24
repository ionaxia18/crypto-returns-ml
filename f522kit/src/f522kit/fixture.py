"""Synthetic smoke fixture in the exact dataset schema (checklist item 3).

Generates a tiny dataset with a planted linear signal so the full pipeline
(reader -> calendar -> harness -> evaluator) can run end-to-end in seconds.
Entirely synthetic: no real features, symbols, dates beyond the calendar
shape, or statistics from the production dataset appear here.
"""

from __future__ import annotations

import datetime
import hashlib
import json
from pathlib import Path

import numpy as np

from . import contract

FIXTURE_DATASET_ID = "f522kit_smoke_fixture_v1"
FIXTURE_N_FEATURES = 8
FIXTURE_TIME_ROWS = 24  # miniature "day" so the fixture stays tiny
FIXTURE_START = datetime.date(2023, 1, 2)
FIXTURE_N_DATES = 70
FIXTURE_HIDDEN_START = datetime.date(2023, 3, 6)
# Planted linear coefficients (first features carry signal, rest are noise).
FIXTURE_BETA = np.array([8e-4, -6e-4, 4e-4, 0.0, 0.0, 0.0, 0.0, 0.0])
FIXTURE_NOISE = 2e-3


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _descriptor(path: Path) -> dict:
    arr = np.load(path, mmap_mode="r", allow_pickle=False)
    return {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "sha256": _sha256_file(path),
    }


def build_fixture(
    root: str | Path,
    time_rows: int = FIXTURE_TIME_ROWS,
    schema_version: int = 1,
    assignment_rule: str | None = None,
) -> Path:
    """Write the deterministic smoke fixture under ``root`` and return it.

    ``schema_version=1`` reproduces a v1-style dataset (previous-group rule
    implied). ``schema_version=2`` writes the required ``assignment_rule``
    field (default ``date_mature``), exercising the v2 contract profile.
    """
    if schema_version not in contract.SUPPORTED_SCHEMA_VERSIONS:
        raise ValueError(f"Unsupported schema_version {schema_version}")
    if schema_version == 2 and assignment_rule is None:
        assignment_rule = "date_mature"
    root = Path(root)
    dates_root = root / "dates"
    dates_root.mkdir(parents=True, exist_ok=True)

    dates = [
        FIXTURE_START + datetime.timedelta(days=i) for i in range(FIXTURE_N_DATES)
    ]
    for date in dates:
        rng = np.random.default_rng(
            np.random.SeedSequence([20260806, date.toordinal()])
        )
        n_symbols = int(rng.integers(3, 7))
        x = rng.standard_normal((time_rows, n_symbols, FIXTURE_N_FEATURES)).astype(
            np.float32
        )
        noise = rng.standard_normal((time_rows, n_symbols)) * FIXTURE_NOISE
        y = (x.astype(np.float64) @ FIXTURE_BETA + noise).astype(np.float32)
        y = np.clip(y, -contract.TARGET_CLIP, contract.TARGET_CLIP)
        w = rng.uniform(0.5, 2.0, size=(time_rows, n_symbols)).astype(np.float32)
        # A sprinkle of zero-weight (invalid-target) rows, y zero-filled there.
        invalid = rng.random((time_rows, n_symbols)) < 0.05
        w[invalid] = 0.0
        y[invalid] = 0.0
        episode_id = np.arange(n_symbols, dtype=np.int32)

        split = "hidden" if date >= FIXTURE_HIDDEN_START else "visible"
        shard = dates_root / date.isoformat()
        shard.mkdir(parents=True, exist_ok=True)
        np.save(shard / "X.npy", x, allow_pickle=False)
        np.save(shard / "episode_id.npy", episode_id, allow_pickle=False)
        if split == "visible":
            np.save(shard / "y.npy", y, allow_pickle=False)
            np.save(shard / "w.npy", w, allow_pickle=False)
        arrays = {
            path.name: _descriptor(path) for path in sorted(shard.glob("*.npy"))
        }
        meta = {
            "schema_version": contract.SCHEMA_VERSION,
            "dataset_id": FIXTURE_DATASET_ID,
            "date": date.isoformat(),
            "split": split,
            "private": False,
            "arrays": arrays,
        }
        (shard / "meta.json").write_text(
            json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8"
        )

    dataset_info = {
        "schema_version": schema_version,
        "dataset_id": FIXTURE_DATASET_ID,
        "daily_time_rows": time_rows,
        "n_dates": FIXTURE_N_DATES,
        "n_features": FIXTURE_N_FEATURES,
        "start_date": dates[0].isoformat(),
        "end_date": dates[-1].isoformat(),
        "hidden_start": FIXTURE_HIDDEN_START.isoformat(),
        "target": contract.TARGET_NAME,
        "target_dtype": "float32",
        "weight": contract.WEIGHT_NAME,
        "weight_dtype": "float32",
        "feature_dtype": "float32",
        "missing_feature_fill": 0.0,
        "missing_target_fill": 0.0,
        "missing_target_weight": 0.0,
    }
    if schema_version >= 2:
        dataset_info["assignment_rule"] = assignment_rule
    (root / "dataset.json").write_text(
        json.dumps(dataset_info, indent=2, sort_keys=True), encoding="utf-8"
    )
    (root / "_SUCCESS").write_text(
        json.dumps({"dataset_id": FIXTURE_DATASET_ID}), encoding="utf-8"
    )
    return root
