"""The rolling harness: fit at anchors, predict scored dates, persist artifacts.

The driver owns the calendar, masking, and prediction persistence; the
student model sees only its :class:`~f522kit.window.TrainWindow` and per-date
feature matrices. Runs are resumable at date granularity: a date whose
prediction file already exists with a matching manifest entry is skipped.

Prediction artifact schema (checklist item 5):

    <out>/
      run_manifest.json     # kit/dataset identity, model spec, seeds, anchors
      preds/YYYY-MM-DD.npy  # float32 [1440, S_d] raw scores (pre-calibration)
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

from . import contract
from .adapter import check_predictions, load_model_factory
from .calendar import Schedule, build_schedule
from .data import Dataset
from .window import TrainWindow

KIT_VERSION = "0.2.3rc2"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True))
        handle.flush()
        os.fsync(handle.fileno())
    tmp.replace(path)
    _fsync_dir(path.parent)


def _atomic_save_npy(path: Path, array: np.ndarray) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    with tmp.open("wb") as handle:
        np.save(handle, array, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    tmp.replace(path)
    _fsync_dir(path.parent)


def _fsync_dir(path: Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def resolve_schedule(
    dataset: Dataset,
    stride: str,
    include_hidden: bool = False,
) -> Schedule:
    """Build the frozen schedule over the dataset's modeled date range."""
    if stride not in contract.STRIDES:
        raise ValueError(f"stride must be one of {sorted(contract.STRIDES)}")
    dates = dataset.dates("all") if include_hidden else dataset.dates("visible")
    return build_schedule(
        dates,
        stride_weeks=contract.STRIDES[stride],
        rule=dataset.assignment_rule,
    )


def run(
    dataset: Dataset,
    model_spec: str,
    out_dir: str | Path,
    stride: str = contract.DEFAULT_STRIDE,
    row_budget: int | None = None,
    seed: int = 0,
    start: datetime.date | None = None,
    end: datetime.date | None = None,
    include_hidden: bool = False,
    verbose: bool = True,
) -> dict:
    """Execute a rolling run and return the manifest dictionary."""
    if row_budget is None:
        row_budget = contract.DEFAULT_ROW_BUDGET
    if row_budget is None:
        raise ValueError(
            "row_budget is required until the fair-compute policy (TBD-3) "
            "freezes a default; pass --row-budget explicitly"
        )
    factory = load_model_factory(model_spec)
    # Hash the student model file so resume can detect code changes. This
    # covers the spec file only; helper modules it imports are the student's
    # responsibility to keep frozen within one run directory.
    model_path = Path(model_spec.rsplit(":", 1)[0])
    model_sha256 = _sha256_file(model_path)
    out = Path(out_dir)
    preds_dir = out / "preds"
    preds_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "run_manifest.json"

    schedule = resolve_schedule(dataset, stride, include_hidden=include_hidden)
    scored = [
        d
        for d in schedule.scored_dates
        if d >= (start or contract.REPORT_START) and (end is None or d <= end)
    ]
    if not scored:
        raise ValueError("No scored dates in the requested range")

    settings = {
        "kit_version": KIT_VERSION,
        "dataset_id": dataset.dataset_id,
        "model_spec": model_spec,
        "model_sha256": model_sha256,
        "stride": stride,
        "row_budget": row_budget,
        "seed": seed,
        "calibration_rule": contract.CALIBRATION_RULE,
        "assignment_rule": dataset.assignment_rule,
    }
    manifest: dict = {}
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            raise RuntimeError(
                f"{manifest_path} is unreadable or corrupt ({exc}); "
                "remove it or use a fresh --out directory"
            ) from exc
        for key, value in settings.items():
            if manifest.get(key) != value:
                hint = (
                    "the model file changed since the previous run"
                    if key == "model_sha256"
                    else "settings differ"
                )
                raise RuntimeError(
                    f"{manifest_path} was produced with different settings "
                    f"({key}: {manifest.get(key)!r} != {value!r}; {hint}); "
                    "use a fresh --out directory"
                )
    manifest.update(settings)
    manifest.setdefault("anchors", {})
    manifest.setdefault("predictions", {})

    # Group scored dates by their assigned anchor, chronological.
    by_anchor: dict[datetime.date, list[datetime.date]] = {}
    for date in scored:
        by_anchor.setdefault(schedule.assignment[date].key, []).append(date)

    for anchor_key in sorted(by_anchor):
        dates = by_anchor[anchor_key]
        pending = [
            d
            for d in dates
            if d.isoformat() not in manifest["predictions"]
            or not (preds_dir / f"{d.isoformat()}.npy").is_file()
        ]
        if not pending:
            continue
        anchor = schedule.assignment[dates[0]]
        window = TrainWindow(
            dataset,
            anchor,
            anchor.training_dates(schedule.groups),
            row_budget=row_budget,
        )
        if verbose:
            print(
                f"[anchor {anchor_key}] fit on {len(window)} dates, "
                f"predict {len(pending)} dates"
            )
        model = factory()
        fit_start = time.time()
        # The row budget is enforced through the frozen sampler; models that
        # stream via iter_dates() instead are audited via the manifest.
        model.fit(window)
        fit_seconds = time.time() - fit_start
        manifest["anchors"][anchor_key.isoformat()] = {
            "n_training_dates": len(window),
            "row_budget": row_budget,
            "fit_seconds": round(fit_seconds, 3),
        }

        for date in pending:
            shard = dataset.shard(date)
            x2d = shard.flat_x()
            scores = check_predictions(
                model.predict(x2d), x2d.shape[0], context=str(date)
            )
            pred = scores.reshape(dataset.time_rows, shard.n_symbols)
            path = preds_dir / f"{date.isoformat()}.npy"
            _atomic_save_npy(path, pred)
            manifest["predictions"][date.isoformat()] = {
                "anchor": anchor_key.isoformat(),
                "n_symbols": shard.n_symbols,
                "sha256": _sha256_file(path),
            }
        _atomic_write_json(manifest_path, manifest)

    _atomic_write_json(manifest_path, manifest)
    if verbose:
        print(f"Run complete: {len(manifest['predictions'])} dates in {preds_dir}")
    return manifest
