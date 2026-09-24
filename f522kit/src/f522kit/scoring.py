"""Score prediction artifacts against visible labels (canonical evaluator).

The pipeline per scored date is frozen: ``calibrate -> clip -> daily stats``.
Aggregates are mean-daily per labeled window, plus guardrails and an
optional paired daily-APS comparison against a baseline run.
"""

from __future__ import annotations

import datetime
import json
import re
from pathlib import Path

import numpy as np

_DATE_STEM = re.compile(r"\d{4}-\d{2}-\d{2}")

from . import contract
from .calendar import build_schedule
from .calibration import calibrate
from .data import Dataset
from .metrics import DailyStats, aggregate, clip_alpha, daily_stats, paired_diff


class ScoringError(RuntimeError):
    pass


def _load_pred(
    preds_dir: Path, date: datetime.date, time_rows: int, n_symbols: int
) -> np.ndarray:
    path = preds_dir / f"{date.isoformat()}.npy"
    if not path.is_file():
        raise ScoringError(f"Missing prediction file: {path}")
    pred = np.load(path, allow_pickle=False)
    expected = (time_rows, n_symbols)
    if pred.shape != expected:
        raise ScoringError(f"{path}: shape {pred.shape} != {expected}")
    if not np.all(np.isfinite(pred)):
        raise ScoringError(f"{path}: non-finite predictions")
    return pred


def collect_daily_stats(
    dataset: Dataset,
    preds_dir: str | Path,
    dates: list[datetime.date],
) -> tuple[list[DailyStats], dict]:
    """Score one run over ``dates``; returns (daily stats, guardrails)."""
    preds_dir = Path(preds_dir)
    stats: list[DailyStats] = []
    clip_hits = 0
    total_weighted_rows = 0
    pred_sq_sum = 0.0
    pred_count = 0
    for date in dates:
        shard = dataset.shard(date)
        if not shard.has_labels:
            raise ScoringError(f"{date}: no labels available for scoring")
        pred = _load_pred(preds_dir, date, dataset.time_rows, shard.n_symbols)
        alpha = calibrate(pred.reshape(-1).astype(np.float64))
        clipped = clip_alpha(alpha)
        _, y, w = shard.flat_rows()
        weighted = np.asarray(w) > 0
        clip_hits += int(np.count_nonzero((alpha != clipped) & weighted))
        total_weighted_rows += int(np.count_nonzero(weighted))
        pred_sq_sum += float(np.dot(alpha[weighted], alpha[weighted]))
        pred_count += int(np.count_nonzero(weighted))
        stats.append(daily_stats(date, clipped, y, w, already_clipped=True))
    guardrails = {
        "n_dates": len(stats),
        "clip_rate": clip_hits / max(total_weighted_rows, 1),
        "pred_rms_weighted_rows": float(np.sqrt(pred_sq_sum / max(pred_count, 1))),
    }
    return stats, guardrails


def score(
    dataset: Dataset,
    preds_dir: str | Path,
    windows: dict[str, tuple[datetime.date, datetime.date]] | None = None,
    baseline_dir: str | Path | None = None,
    require_all_dates: bool = True,
) -> dict:
    """Produce the report card for one prediction run.

    ``windows`` defaults to the contract's labeled windows. Every scored
    window date must have a prediction file when ``require_all_dates`` —
    partial coverage is a broken run, not a smaller score.
    """
    windows = windows or contract.DEFAULT_WINDOWS
    preds_dir = Path(preds_dir)
    search_dir = preds_dir / "preds" if (preds_dir / "preds").is_dir() else preds_dir
    available = set()
    for p in search_dir.glob("*.npy"):
        if _DATE_STEM.fullmatch(p.stem):
            available.add(datetime.date.fromisoformat(p.stem))
        # Non-date-named .npy files (debug artifacts etc.) are ignored.

    # Coverage is owed only on dates that are predictable under the frozen
    # calendar (the first two weekly groups never have a model; stride does
    # not change predictability).
    predictable = set(
        build_schedule(
            dataset.dates("visible"),
            stride_weeks=1,
            rule=dataset.assignment_rule,
        ).scored_dates
    )
    report: dict = {"windows": {}, "guardrails": {}}
    all_stats: dict[str, list[DailyStats]] = {}
    for label, (start, end) in windows.items():
        window_dates = sorted(
            d for d in predictable if start <= d <= end and d >= contract.REPORT_START
        )
        if not window_dates:
            if require_all_dates:
                raise ScoringError(
                    f"Window {label!r} ({start}..{end}) contains no predictable "
                    "dates; check the window bounds (use --allow-partial to skip)"
                )
            report["guardrails"][label] = {"n_dates": 0, "empty_window": True}
            continue
        scorable = [d for d in window_dates if d in available]
        missing = [d for d in window_dates if d not in available]
        if missing and require_all_dates:
            raise ScoringError(
                f"Window {label!r}: {len(missing)} scored dates missing predictions "
                f"(first: {missing[0]})"
            )
        if not scorable:
            report["guardrails"][label] = {
                "n_dates": 0,
                "missing_dates": len(missing),
            }
            continue
        stats, guardrails = collect_daily_stats(dataset, search_dir, scorable)
        all_stats[label] = stats
        report["windows"][label] = aggregate(stats)
        report["guardrails"][label] = guardrails
        if missing:
            report["guardrails"][label]["missing_dates"] = len(missing)

    if baseline_dir is not None:
        base_dir = Path(baseline_dir)
        base_search = base_dir / "preds" if (base_dir / "preds").is_dir() else base_dir
        report["vs_baseline"] = {}
        for label, stats in all_stats.items():
            base_stats, _ = collect_daily_stats(
                dataset, base_search, [s.date for s in stats]
            )
            report["vs_baseline"][label] = paired_diff(stats, base_stats)

    return report


def format_report(report: dict) -> str:
    lines = ["=" * 72]
    for label, metrics in report.get("windows", {}).items():
        guard = report.get("guardrails", {}).get(label, {})
        lines.append(
            f"[{label}] {metrics['start_date']}..{metrics['end_date']} "
            f"({metrics['n_dates']} dates)"
        )
        lines.append(
            f"  APS {metrics['APS_mean_bps']:.4f} bps | COR {metrics['COR_mean']*100:.4f}% | "
            f"AR {metrics['AR_mean']:.4f} | APS_SR {metrics['APS_SR']:.2f}"
        )
        lines.append(
            f"  guardrails: clip_rate {guard.get('clip_rate', 0):.2e}, "
            f"pred_rms {guard.get('pred_rms_weighted_rows', 0):.3e}"
        )
        diff = report.get("vs_baseline", {}).get(label)
        if diff:
            lines.append(
                f"  vs baseline: dAPS {diff['dAPS_mean_bps']:+.4f} bps | "
                f"diff-Sharpe {diff['diff_sharpe']:.2f} | win {diff['win_rate']*100:.1f}%"
            )
        lines.append("-" * 72)
    return "\n".join(lines)


def save_report(report: dict, path: str | Path) -> None:
    Path(path).write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
