"""Combine full rolling runs (kit-format prediction dirs) and score with the kit.

    python -m experiment.combine_rolling --data $DATA_ROOT --out runs/cmb_ridge_lgb_4w \\
        --part ridge=runs/linear_4w:1 --part lgb=runs/lgbm_4w:1 --norm rms

The rolling counterpart of experiment/combine.py: every predictable date of the
window (default: the contract ``is`` period; never the dev window) must exist
in every part. Each date's prediction is

    norm "none": sum_k w_k * pred_k / sum_k w_k
    norm "rms":  (sum_k w_k * pred_k / rms_k) / sum_k w_k  *  (sum_k w_k * rms_k) / sum_k w_k

where rms_k is the RMS of part k's predictions over the whole window
(label-free, one constant per model: a global scale choice, which leaves APS
unchanged unless the prediction clip binds). Writes ``<out>/preds/``, a
``run_manifest.json`` naming its parts, and ``<out>/report.json`` from
``f522kit.scoring.score`` with the paired daily-APS comparison against
``--baseline`` (default: the official ridge, runs/linear_4w).
"""

from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path

import numpy as np

from f522kit import contract
from f522kit.calendar import build_schedule
from f522kit.data import Dataset
from f522kit.scoring import save_report, score

from .rolling import window_bounds


def predictable_dates(dataset: Dataset, start, end) -> list[datetime.date]:
    sched = build_schedule(dataset.dates("visible"), stride_weeks=1, rule=dataset.assignment_rule)
    return sorted(d for d in sched.scored_dates
                  if start <= d <= end and d >= contract.REPORT_START)


def _pred_path(run_dir: Path, date) -> Path:
    base = run_dir / "preds" if (run_dir / "preds").is_dir() else run_dir
    return base / f"{date.isoformat()}.npy"


def combine_runs(data_root, parts: dict, out, norm="rms", start=None, end=None,
                 baseline="runs/linear_4w") -> dict:
    """``parts``: {alias: (run_dir, weight)}."""
    if norm not in ("none", "rms"):
        raise ValueError("norm must be none|rms")
    dataset = Dataset(data_root)
    start, end = window_bounds(start, end)
    dates = predictable_dates(dataset, start, end)
    parts = {a: (Path(d), float(w)) for a, (d, w) in parts.items()}
    for alias, (run_dir, _) in parts.items():
        missing = [d for d in dates if not _pred_path(run_dir, d).is_file()]
        if missing:
            raise SystemExit(f"part {alias} ({run_dir}) lacks {len(missing)} dates "
                             f"(first {missing[0]})")

    rms = {}
    for alias, (run_dir, _) in parts.items():
        ss = n = 0.0
        for d in dates:
            p = np.load(_pred_path(run_dir, d)).astype(np.float64)
            ss += float(np.dot(p.ravel(), p.ravel())); n += p.size
        rms[alias] = float(np.sqrt(ss / n))
    total_w = sum(w for _, w in parts.values())
    back = sum(w * rms[a] for a, (_, w) in parts.items()) / total_w

    out = Path(out)
    (out / "preds").mkdir(parents=True, exist_ok=True)
    for d in dates:
        acc = None
        for alias, (run_dir, w) in parts.items():
            p = np.load(_pred_path(run_dir, d)).astype(np.float64)
            term = w * (p / rms[alias] if norm == "rms" else p)
            acc = term if acc is None else acc + term
        comb = acc / total_w * (back if norm == "rms" else 1.0)
        np.save(out / "preds" / f"{d.isoformat()}.npy", comb.astype(np.float32))

    def spec_of(run_dir):
        m = run_dir / "run_manifest.json"
        return json.loads(m.read_text(encoding="utf-8")).get("model_spec") if m.is_file() else None

    manifest = {
        "combination": {a: {"run": str(d), "weight": w, "rms": rms[a], "model_spec": spec_of(d)}
                        for a, (d, w) in parts.items()},
        "norm": norm, "window": [start.isoformat(), end.isoformat()], "n_dates": len(dates),
        "dataset_id": dataset.dataset_id,
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    report = score(dataset, out, windows={"is": (start, end)},
                   baseline_dir=baseline if baseline and Path(baseline).exists() else None)
    save_report(report, out / "report.json")
    m = report["windows"]["is"]
    vs = report.get("vs_baseline", {}).get("is")
    print(f"{out}: APS {m['APS_mean_bps']:.4f} bps over {m['n_dates']} days"
          + (f" | vs {baseline}: dAPS {vs['dAPS_mean_bps']:+.4f}, diff Sharpe "
             f"{vs['diff_sharpe']:.2f}, win {vs['win_rate']*100:.1f}%" if vs else ""))
    return report


def _part(text: str):
    alias, rest = text.split("=", 1)
    run, _, weight = rest.rpartition(":")
    return alias, (run, float(weight)) if run else (rest, 1.0)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m experiment.combine_rolling", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--part", action="append", required=True, type=_part,
                   help="alias=run_dir[:weight] (repeatable)")
    p.add_argument("--norm", choices=("none", "rms"), default="rms")
    p.add_argument("--baseline", default="runs/linear_4w")
    p.add_argument("--start", type=datetime.date.fromisoformat, default=None)
    p.add_argument("--end", type=datetime.date.fromisoformat, default=None)
    a = p.parse_args(argv)
    combine_runs(a.data, dict(a.part), a.out, a.norm, a.start, a.end, a.baseline)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
