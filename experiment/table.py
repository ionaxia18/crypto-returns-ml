"""Collect experiment-split runs into one comparison table.

    python -m experiment.table --split $SCRATCH/core980-work/experiment/v2_52w \\
        --baseline results/v2_52w/ridge --extra runs/linear_4w --md results/v2_52w/TABLE.md

Rows are every run directory under ``--results`` (written by
experiment/run.py) plus any ``--extra`` kit run directories, e.g. the full
104-week ridge, which needs no record: its preds already cover the val
dates. Every row is scored with the kit's own ``collect_daily_stats`` +
``aggregate`` on the split's val dates, and the paired columns (dAPS,
diff-Sharpe, win rate) use the kit's ``paired_diff`` vs ``--baseline``,
so all comparisons are on identical dates and rows.

Runs recorded on a different split are refused rather than mixed in.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from f522kit.data import Dataset
from f522kit.metrics import aggregate, paired_diff
from f522kit.scoring import collect_daily_stats

from .split import ExperimentSplit

REPO = Path(__file__).resolve().parent.parent


def _preds_dir(run_dir: Path) -> Path:
    return run_dir / "preds" if (run_dir / "preds").is_dir() else run_dir


def build_table(split_dir, results, baseline=None, extra=(), data_root=None):
    split = ExperimentSplit(split_dir)
    dataset = Dataset(data_root or split.meta["data_root"])
    results = Path(results)

    runs = []  # (name, run_dir, record or None)
    for rec_path in sorted(results.glob("*/record.json")):
        rec = json.loads(rec_path.read_text(encoding="utf-8"))
        if Path(rec["split"]).resolve() != Path(split_dir).resolve():
            raise SystemExit(f"{rec_path} was run on split {rec['split']}, not {split_dir}")
        runs.append((rec["name"], rec_path.parent, rec))
    runs += [(Path(e).name, Path(e), None) for e in extra]
    if not runs:
        raise SystemExit(f"no runs under {results} and no --extra")

    def stats_of(run_dir):
        stats, guard = collect_daily_stats(dataset, _preds_dir(Path(run_dir)), split.val_dates)
        return stats, guard

    base_stats = stats_of(baseline)[0] if baseline else None
    rows = []
    for name, run_dir, rec in runs:
        stats, guard = stats_of(run_dir)
        m = aggregate(stats)
        row = {
            "name": name, "model": rec["model"] if rec else "kit run",
            "APS_bps": m["APS_mean_bps"], "COR_%": m["COR_mean"] * 100,
            "APS_SR": m["APS_SR"], "AR": m["AR_mean"], "clip_rate": guard["clip_rate"],
            # Same rows_id = trained on identical rows (models/*.py fingerprints).
            "fit_rows": rec.get("fit_info", {}).get("n_fit_rows") if rec else None,
            "rows_id": rec.get("fit_info", {}).get("fit_rows_sha") if rec else None,
            "fit_s": rec["timing_s"]["fit"] if rec else None,
            "total_s": rec["timing_s"]["total"] if rec else None,
            "peak_GB": rec["peak_rss_gb"]["final"] if rec else None,
            "cpus": rec["cpus"] if rec else None,
        }
        if base_stats is not None:
            d = paired_diff(stats, base_stats)
            row.update({"dAPS_bps": d["dAPS_mean_bps"], "diff_SR": d["diff_sharpe"],
                        "win_%": d["win_rate"] * 100})
        rows.append(row)
    rows.sort(key=lambda x: -x["APS_bps"])

    meta = split.meta
    head = (f"Experiment split: anchor {meta['anchor']}, train {meta['train_dates'][0]}.."
            f"{meta['train_dates'][-1]} ({meta['train_weeks']}w, minute stride "
            f"{meta['minute_stride']}, {meta['train']['n_rows']:,} rows), val "
            f"{meta['val_dates'][0]}..{meta['val_dates'][-1]} "
            f"({len(meta['val_dates'])} dates, all rows)"
            + (f"; paired columns vs `{baseline}`" if baseline else ""))
    cols = list(rows[0].keys())

    def fmt(v):
        if v is None:
            return "–"
        if isinstance(v, float):
            if v != v:
                return "nan"
            return f"{v:.2e}" if 0 < abs(v) < 1e-3 else f"{v:.3f}" if abs(v) < 100 else f"{v:.0f}"
        if isinstance(v, int) and not isinstance(v, bool):
            return f"{v:,}"
        return str(v)

    lines = [head, "", "| " + " | ".join(cols) + " |",
             "|" + "|".join("---" if c in ("name", "model") else "---:" for c in cols) + "|"]
    lines += ["| " + " | ".join(fmt(row[c]) for c in cols) + " |" for row in rows]
    return rows, "\n".join(lines)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m experiment.table", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", required=True)
    p.add_argument("--results", default=None, help="default: results/<split dir name>")
    p.add_argument("--baseline", default=None, help="run dir for the paired columns")
    p.add_argument("--extra", action="append", default=[],
                   help="extra kit run dir to include (repeatable), e.g. runs/linear_4w")
    p.add_argument("--data", default=None, help="dataset root (default: the split's)")
    p.add_argument("--md", default=None, help="also write the markdown table here")
    a = p.parse_args(argv)
    results = a.results or str(REPO / "results" / Path(a.split).resolve().name)
    _, md = build_table(a.split, results, a.baseline, a.extra, a.data)
    print(md)
    if a.md:
        Path(a.md).write_text(md + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
