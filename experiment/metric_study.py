"""Which smooth measure tracks APS? Score every saved run on a split's val rows with
several candidate measures and compare them as model-selection signals.

    python -m experiment.metric_study --split $SCRATCH/core980-work/experiment/v2_52w \\
        --baseline ridge_eqall --other results/v3_52w_jan24

APS is noisy because each day's APS = COR x that day's return size, so a few
volatile days dominate it. Candidates (each computed per day, then averaged over
days, like the kit's APS):

    APS            kit-style daily APS (bps)
    APS_yclip3     APS against y clipped at +-3 rms_w(y) (val-wide): tails tamed
    COR            weighted (uncentered) correlation = APS without the
                   return-size factor, so every day counts equally
    COR_unweighted the same with all rows weighted equally
    R2             weighted R^2 = 1 - sum w (y - a)^2 / sum w y^2 (scale-dependent)
    neg_huber      minus weighted Huber loss on y / rms, delta 1 (scale-dependent)

Runs that differ only in seed are one configuration (experiment/table.py's
grouping); their daily series are averaged. For each measure:

    rank_corr_APS   Spearman correlation with APS across configurations, same window
    median_abs_t    median |t| of the daily difference vs the --baseline
                    configuration: how clearly the measure separates models
    xwin_rank_corr  with --other (another window's results dir): Spearman between
                    this measure here and APS there, over configurations run on
                    both windows. The selection question: does picking on this
                    measure pick models that score well on another period?
                    (APS here vs APS there is the reference row.)

Combination runs are skipped. Writes <results>/metric_study.md and .json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .combine import gather
from .run import default_results_dir
from .split import ExperimentSplit
from .table import config_key

MEASURES = ["APS", "APS_yclip3", "COR", "COR_unweighted", "R2", "neg_huber"]


def daily_measures(pred, y, w, days, y_rms):
    """{measure: daily series} for one prediction vector over the val rows;
    ``days`` holds each day's row indices."""
    yc = np.clip(y, -3 * y_rms, 3 * y_rms)
    out = {m: np.empty(len(days)) for m in MEASURES}
    for i, s in enumerate(days):
        a, yy, ww, cc = pred[s], y[s], w[s], yc[s]
        swaa = max(float(np.sum(ww * a * a)), 1e-30)
        sw = float(ww.sum())
        out["APS"][i] = float(np.sum(ww * a * yy)) / np.sqrt(swaa) / np.sqrt(sw) * 1e4
        out["APS_yclip3"][i] = float(np.sum(ww * a * cc)) / np.sqrt(swaa) / np.sqrt(sw) * 1e4
        out["COR"][i] = float(np.sum(ww * a * yy)) / np.sqrt(swaa * float(np.sum(ww * yy * yy)))
        out["COR_unweighted"][i] = float(a @ yy) / np.sqrt(max(float(a @ a), 1e-30) * float(yy @ yy))
        out["R2"][i] = 1.0 - float(np.sum(ww * (yy - a) ** 2)) / float(np.sum(ww * yy * yy))
        e = np.abs(yy - a) / y_rms
        hub = np.where(e <= 1.0, 0.5 * e * e, e - 0.5)
        out["neg_huber"][i] = -float(np.sum(ww * hub)) / sw
    return out


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.size < 3:
        return float("nan")
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def _records(results: Path, need_preds: bool):
    out = []
    for p in sorted(results.glob("*/record.json")):
        rec = json.loads(p.read_text(encoding="utf-8"))
        if rec["model"] == "combo" or (need_preds and not (p.parent / "preds").is_dir()):
            continue
        out.append(rec)
    return out


def study(split_dir, results=None, baseline="ridge_eqall", other=None, models=None):
    split = ExperimentSplit(split_dir)
    results = Path(results or default_results_dir(split_dir))
    _, y, w, d = split.val()
    y = np.asarray(y, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    n_days = len(split.val_dates)
    y_rms = float(np.sqrt(np.sum(w * y * y) / np.sum(w)))
    days = [np.flatnonzero(d == i) for i in range(n_days)]

    def keep(r):  # --models filter; the baseline always stays in
        return not models or r["model"] in models or r["name"] == baseline

    recs = [r for r in _records(results, need_preds=True) if keep(r)]
    if not recs:
        raise SystemExit(f"no runs with predictions under {results}")
    groups: dict[tuple, dict] = {}
    for r in recs:
        key = config_key(r)
        series = daily_measures(gather(split, results / r["name"]), y, w, days, y_rms)
        g = groups.setdefault(key, {"runs": [], "series": []})
        g["runs"].append(r["name"])
        g["series"].append(series)
    for g in groups.values():
        g["daily"] = {m: np.mean([s[m] for s in g["series"]], axis=0) for m in MEASURES}
        g["mean"] = {m: float(g["daily"][m].mean()) for m in MEASURES}
        del g["series"]

    by_run = {n: k for k, g in groups.items() for n in g["runs"]}
    if baseline not in by_run:
        raise SystemExit(f"baseline {baseline!r} has no predictions under {results}")
    base = groups[by_run[baseline]]

    other_aps = {}
    if other:
        for r in _records(Path(other), need_preds=False):
            if keep(r):
                other_aps.setdefault(config_key(r), []).append(
                    r["report"]["windows"]["val"]["APS_mean_bps"])
    shared = [k for k in groups if k in other_aps]

    keys = list(groups)
    rows = []
    for m in MEASURES:
        ts = []
        for k in keys:
            if k == by_run[baseline]:
                continue
            diff = groups[k]["daily"][m] - base["daily"][m]
            sd = diff.std(ddof=1)
            if sd > 0:
                ts.append(abs(diff.mean()) / (sd / np.sqrt(n_days)))
        rows.append({
            "measure": m,
            "rank_corr_APS": spearman([groups[k]["mean"][m] for k in keys],
                                      [groups[k]["mean"]["APS"] for k in keys]),
            "median_abs_t": float(np.median(ts)) if ts else float("nan"),
            "xwin_rank_corr": (spearman([groups[k]["mean"][m] for k in shared],
                                        [np.mean(other_aps[k]) for k in shared])
                               if len(shared) >= 3 else float("nan")),
        })

    configs = sorted(({"runs": ", ".join(g["runs"]), **g["mean"]} for g in groups.values()),
                     key=lambda r: -r["APS"])
    return {"split": split.meta.get("anchor"), "n_configs": len(groups),
            "n_runs": len(recs), "n_shared_with_other": len(shared), "other": other,
            "baseline": baseline, "measures": rows, "configs": configs}


def to_markdown(rep) -> str:
    L = [f"# Metric study: anchor {rep['split']}", "",
         f"{rep['n_runs']} runs = {rep['n_configs']} configurations (seeds averaged); "
         f"baseline `{rep['baseline']}`"
         + (f"; {rep['n_shared_with_other']} configurations also run in `{rep['other']}`"
            if rep["other"] else "") + ".", "",
         "| measure | rank corr with APS (same window) | median abs t vs baseline "
         "| rank corr with other window's APS |", "|---|---:|---:|---:|"]
    for r in rep["measures"]:
        L.append(f"| {r['measure']} | {r['rank_corr_APS']:.2f} | {r['median_abs_t']:.2f} "
                 f"| {r['xwin_rank_corr']:.2f} |")
    L += ["", "Higher is better in every column. The `APS` row is the reference: a "
          "measure that separates models more clearly (higher t) and predicts the other "
          "window's APS at least as well is a better selection signal than APS itself.",
          "", "## Configurations", "",
          "| runs | " + " | ".join(MEASURES) + " |", "|---|" + "---:|" * len(MEASURES)]
    for c in rep["configs"]:
        L.append(f"| {c['runs']} | " + " | ".join(f"{c[m]:.4g}" for m in MEASURES) + " |")
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m experiment.metric_study", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", required=True)
    p.add_argument("--results", default=None, help="default: results/<split dir name>")
    p.add_argument("--baseline", default="ridge_eqall", help="run name")
    p.add_argument("--other", default=None,
                   help="another window's results dir (records only), e.g. results/v3_52w_jan24")
    p.add_argument("--models", default=None, help="comma list, e.g. mlp,lgbm (default: all)")
    p.add_argument("--out", default=None, help="default: <results>/metric_study.md")
    a = p.parse_args(argv)
    models = set(a.models.split(",")) if a.models else None
    rep = study(a.split, a.results, a.baseline, a.other, models)
    out = Path(a.out or Path(a.results or default_results_dir(a.split)) / "metric_study.md")
    out.write_text(to_markdown(rep), encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(to_markdown(rep))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
