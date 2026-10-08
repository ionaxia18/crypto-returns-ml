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

``--by-config`` gives one row per configuration instead: runs with the same
model and settings, ignoring only ``seed`` and ``threads``, are combined by
averaging their daily APS, and the paired columns compare that against the
baseline's configuration (``--baseline`` is then a run name). It reads the
records' kit-scored daily APS only, so it needs no dataset access:

    python -m experiment.table --results results/v2_52w --by-config --baseline ridge_eqall
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from f522kit.data import Dataset
from f522kit.metrics import aggregate, annualized_sharpe, paired_diff
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
    return rows, _markdown(head, rows, left=("name", "model", "rows_id"))


# Settings that do not define a configuration: runs differing only in these
# are seeds (or hardware / data-feeding variants) of the same configuration.
# in_ram only changes how the MLP's rows are fed (loaded once vs streamed).
NON_CONFIG_KEYS = {"seed", "threads", "in_ram"}
# Models with no randomness: extra seeds cannot change them, so no seed warning.
DETERMINISTIC_MODELS = {"ridge"}


def _canon(cfg: dict) -> str:
    cfg = {k: v for k, v in cfg.items() if k not in NON_CONFIG_KEYS}
    return json.dumps(cfg, sort_keys=True, separators=(",", ":"))


def _resolve_now(rec: dict) -> dict:
    """Full settings for a record written before run.py stored them: rebuild the
    model from its typed config with the current code's defaults. Exact as long
    as no default the run relied on has changed since; the one change so far
    (LightGBM max_rows 2M -> 0) was backfilled into the older records."""
    from .run import MODELS, load_class, resolved_config  # deferred: heavy imports
    spec = MODELS.get(rec["model"], rec.get("spec", rec["model"]))
    return resolved_config(load_class(spec)(**rec.get("config", {})))


def config_key(rec: dict) -> tuple[str, str]:
    """Group by the model's full effective settings. Recorded values win;
    settings a record predates (options added later, which always default to
    the old behaviour) are filled in from the current code's defaults."""
    full = {**_resolve_now(rec), **rec.get("resolved_config", {})}
    return rec["model"], _canon(full)


def build_config_table(results, baseline=None, min_seeds=3):
    """One row per configuration, seeds combined; reads records only.

    A configuration's daily APS is the mean over its seeds, day by day; the
    paired columns compare that series with the baseline configuration's
    (the group containing run ``baseline``) using the kit's ``annualized_sharpe``.
    """
    results = Path(results)
    recs = [json.loads(p.read_text(encoding="utf-8"))
            for p in sorted(results.glob("*/record.json"))]
    if not recs:
        raise SystemExit(f"no run records under {results}")
    splits = {r["split"] for r in recs}
    if len(splits) > 1:
        raise SystemExit(f"records come from different splits: {sorted(splits)}")

    groups: dict[tuple[str, str], list[dict]] = {}
    for r in recs:
        groups.setdefault(config_key(r), []).append(r)

    def daily(members):
        series = [m["report"]["daily_APS_bps"] for m in members]
        if len({len(s) for s in series}) != 1:
            raise SystemExit(f"{[m['name'] for m in members]}: different val date counts")
        return np.mean(np.array(series), axis=0)

    base_key = None
    if baseline:
        by_name = {r["name"]: r for r in recs}
        if baseline not in by_name:
            raise SystemExit(f"baseline {baseline!r} is not a run under {results}")
        base_key = config_key(by_name[baseline])
        base_daily = daily(groups[base_key])

    rows = []
    for key, members in groups.items():
        members.sort(key=lambda m: m["config"].get("seed", 0))
        seed_aps = [m["report"]["windows"]["val"]["APS_mean_bps"] for m in members]
        shas = {m.get("fit_info", {}).get("fit_rows_sha") for m in members}
        model = key[0]
        row = {
            # Label with the short typed settings; grouping used the full ones.
            "config": f"{model} {_canon(members[0].get('config', {}))}",
            "runs": ", ".join(m["name"] for m in members),
            "seeds": len(members),
            "APS_bps": float(np.mean(seed_aps)),
            "seed_range": (f"{min(seed_aps):.2f}-{max(seed_aps):.2f}"
                           if len(members) > 1 else None),
            "fit_rows": members[0].get("fit_info", {}).get("n_fit_rows"),
            # Seeds may draw different row samples (e.g. LightGBM max_rows).
            "rows": ("varies" if len(shas) > 1 else "same") if len(members) > 1 else None,
            "fit_s": float(np.mean([m["timing_s"]["fit"] for m in members])),
        }
        if base_key is not None:
            diff = daily(members) - base_daily
            is_base = key == base_key
            row.update({
                "dAPS_bps": float(diff.mean()) if not is_base else None,
                "diff_SR": annualized_sharpe(diff) if not is_base else None,
                "win_%": float(np.mean(diff > 0) * 100) if not is_base else None,
            })
        needs_seeds = model not in DETERMINISTIC_MODELS and len(members) < min_seeds
        row["note"] = ("baseline" if key == base_key else
                       f"⚠ <{min_seeds} seeds" if needs_seeds else "")
        rows.append(row)

    sort_key = "diff_SR" if base_key is not None else "APS_bps"
    rows.sort(key=lambda x: (x["note"] != "baseline",
                             -(x[sort_key] if x.get(sort_key) is not None
                               and x[sort_key] == x[sort_key] else -np.inf)))

    sm = recs[0].get("split_meta", {})
    head = (f"Configurations in {results} (split {sm.get('train_weeks', '?')}w, stride "
            f"{sm.get('minute_stride', '?')}, anchor {sm.get('anchor', '?')}); seeds "
            f"combined by averaging daily APS"
            + (f"; paired columns vs the configuration of `{baseline}`" if baseline else "")
            + ". diff_SR = annualized Sharpe of the daily APS difference.")
    return rows, _markdown(head, rows, left=("config", "runs", "seed_range", "rows", "note"))


def _fmt(v):
    if v is None:
        return "–"
    if isinstance(v, float):
        if v != v:
            return "nan"
        return f"{v:.2e}" if 0 < abs(v) < 1e-3 else f"{v:.3f}" if abs(v) < 100 else f"{v:.0f}"
    if isinstance(v, int) and not isinstance(v, bool):
        return f"{v:,}"
    return str(v).replace("|", "\\|")


def _markdown(head, rows, left=("name", "model")):
    cols = list(rows[0].keys())
    lines = [head, "", "| " + " | ".join(cols) + " |",
             "|" + "|".join("---" if c in left else "---:" for c in cols) + "|"]
    lines += ["| " + " | ".join(_fmt(row[c]) for c in cols) + " |" for row in rows]
    return "\n".join(lines)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m experiment.table", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", default=None,
                   help="split dir (required unless --by-config with --results)")
    p.add_argument("--by-config", action="store_true",
                   help="one row per configuration, seeds combined; reads records only "
                        "(no dataset access); --baseline is then a run name")
    p.add_argument("--min-seeds", type=int, default=3,
                   help="--by-config: flag configurations with fewer seeds")
    p.add_argument("--results", default=None, help="default: results/<split dir name>")
    p.add_argument("--baseline", default=None, help="run dir for the paired columns")
    p.add_argument("--extra", action="append", default=[],
                   help="extra kit run dir to include (repeatable), e.g. runs/linear_4w")
    p.add_argument("--data", default=None, help="dataset root (default: the split's)")
    p.add_argument("--md", default=None, help="also write the markdown table here")
    a = p.parse_args(argv)
    if not (a.results or a.split):
        raise SystemExit("pass --split (or --results with --by-config)")
    if not a.by_config and not a.split:
        raise SystemExit("--split is required for the per-run table")
    results = a.results or str(REPO / "results" / Path(a.split).resolve().name)
    if a.by_config:
        if a.extra:
            raise SystemExit("--extra needs dataset scoring; use the per-run table for it")
        _, md = build_config_table(results, a.baseline, a.min_seeds)
    else:
        _, md = build_table(a.split, results, a.baseline, a.extra, a.data)
    print(md)
    if a.md:
        Path(a.md).write_text(md + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
