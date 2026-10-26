"""Post-prediction analysis: where does each model's APS (and a model's gain over
another) come from?

    python -m experiment.analyze --split $SCRATCH/core980-work/experiment/v2_52w \\
        --model ridge=ridge_eqall --model lgb='lgb_lr01_loss_s*' --model combo=cmb_ridge_lgb

Models are saved runs in results/<split>/ (a glob averages seeds); the first is
the reference. Uses the split's val rows (every positive-weight row, so this is
the same APS the kit reports, up to the +-0.11 prediction clip, which binds on
~1e-6 of rows).

**Exact decomposition.** On each day, APS = sum over rows of
    c_row = w * a * y / ( sqrt(sum w a^2) * sqrt(sum w) ),
so a group's contribution = its rows' c summed per day, averaged over days. Group
contributions add up exactly to the mean daily APS, and the difference between two
models' contributions shows where one gains over the other.

Groupings: |y| decile (tails vs normal rows), day volatility (thirds of days by
that day's return size), hour of day, instrument weight decile (instruments ranked
by their share of total weight; episode_id is used for grouping only, as the
contract allows), and whether the reference and the second model agree on the
sign. Also: prediction correlations between models, and how concentrated the
gain is across instruments.

Writes results/<split>/analysis/ANALYSIS.md and analysis.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from f522kit.data import Dataset

from .combine import gather, resolve_runs
from .run import default_results_dir
from .split import ExperimentSplit


def contributions(pred, y, w, d, n_days):
    """Per-row APS contributions (bps); per day they sum to that day's APS."""
    c = np.zeros_like(pred)
    for i in range(n_days):
        s = np.flatnonzero(d == i)
        a, yy, ww = pred[s], y[s], w[s]
        den = np.sqrt(max(float(np.sum(ww * a * a)), 1e-12)) * np.sqrt(max(float(ww.sum()), 1e-12))
        c[s] = ww * a * yy / den * 1e4
    return c


def wcorr(a, b, w):
    am, bm = np.average(a, weights=w), np.average(b, weights=w)
    cov = np.average((a - am) * (b - bm), weights=w)
    return float(cov / np.sqrt(np.average((a - am) ** 2, weights=w) * np.average((b - bm) ** 2, weights=w)))


def group_table(labels, names, contrib, w, n_days, order=None):
    """Rows: group, row share, weight share, contribution per model (bps/day)."""
    keys = order if order is not None else sorted(set(labels.tolist()))
    tot_w = float(w.sum())
    out = []
    for k in keys:
        m = labels == k
        row = {"group": str(k), "row_share": float(m.mean()), "weight_share": float(w[m].sum() / tot_w)}
        for n in names:
            row[n] = float(contrib[n][m].sum() / n_days)
        out.append(row)
    return out


def analyze(split_dir, models: dict, data_root=None, results=None):
    split = ExperimentSplit(split_dir)
    dataset = Dataset(data_root or split.meta["data_root"])
    results = Path(results or default_results_dir(split_dir))
    _, y, w, d = split.val()
    y = np.asarray(y, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    rows = split.val_rows()
    minute = np.load(split.root / "val_minute.npy")
    n_days = len(split.val_dates)

    preds, used = {}, {}
    for name, pat in models.items():
        runs = resolve_runs(results, [pat])
        if not runs:
            raise SystemExit(f"no runs for {name} ({pat}) under {results}")
        preds[name] = np.mean([gather(split, results / r) for r in runs], axis=0)
        used[name] = runs
    names = list(models)
    ref, other = names[0], names[1] if len(names) > 1 else names[0]
    contrib = {n: contributions(preds[n], y, w, d, n_days) for n in names}

    # instrument (episode) per row
    episode = np.empty(rows.size, dtype=np.int64)
    for i, date in enumerate(split.val_dates):
        s = np.flatnonzero(d == i)
        shard = dataset.shard(date)
        episode[s] = np.asarray(shard.load_episode_id())[rows[s] % shard.n_symbols]

    report = {"split": split.meta.get("anchor"), "val": [split.meta["val_dates"][0],
              split.meta["val_dates"][-1]], "models": used,
              "aps_bps": {n: float(contrib[n].sum() / n_days) for n in names}}
    report["prediction_correlation"] = {
        f"{a}~{b}": wcorr(preds[a], preds[b], w) for i, a in enumerate(names) for b in names[i + 1:]}

    absy = np.abs(y)
    dec = np.minimum((np.searchsorted(np.quantile(absy, np.linspace(0, 1, 11)[1:-1]), absy, side="right")), 9)
    day_rs = np.array([np.sqrt(np.sum(w[d == i] * y[d == i] ** 2) / np.sum(w[d == i])) for i in range(n_days)])
    terc = np.searchsorted(np.quantile(day_rs, [1 / 3, 2 / 3]), day_rs, side="right")
    vol_label = np.array(["low", "mid", "high"])[terc][d]
    uniq, inv = np.unique(episode, return_inverse=True)
    w_by = np.bincount(inv, weights=w)
    rank = np.argsort(np.argsort(w_by))  # 0 = lightest
    inst_dec = (rank * 10 // len(uniq))[inv]
    agree = np.where(np.sign(preds[ref]) == np.sign(preds[other]), "agree", "disagree")

    report["groups"] = {
        "abs_y_decile (0 = smallest moves, 9 = largest)": group_table(dec, names, contrib, w, n_days, list(range(10))),
        "day_volatility": group_table(vol_label, names, contrib, w, n_days, ["low", "mid", "high"]),
        "hour_of_day": group_table(minute // 60, names, contrib, w, n_days, list(range(24))),
        "instrument_weight_decile (9 = heaviest)": group_table(inst_dec, names, contrib, w, n_days, list(range(10))),
        f"sign: {ref} vs {other}": group_table(agree, names, contrib, w, n_days, ["agree", "disagree"]),
    }

    # concentration of the gain (other - ref) across instruments
    if other != ref:
        g = np.bincount(inv, weights=(contrib[other] - contrib[ref])) / n_days
        order = np.argsort(-g)
        total = float(g.sum())
        pos = g[g > 0].sum()
        report["instrument_gain"] = {
            "n_instruments": int(len(uniq)),
            "total_gain_bps": total,
            "share_of_instruments_with_gain": float((g > 0).mean()),
            "top10_gain_bps": float(g[order[:10]].sum()),
            "top10pct_instruments_share_of_positive_gain": float(
                g[order[: max(1, len(uniq) // 10)]].sum() / pos) if pos > 0 else None,
            "worst10_gain_bps": float(g[order[-10:]].sum()),
        }
    return report


def to_markdown(rep, names) -> str:
    ref, other = names[0], names[1] if len(names) > 1 else names[0]
    L = [f"# Post-prediction analysis: anchor {rep['split']}, val {rep['val'][0]}..{rep['val'][1]}", "",
         "APS (bps): " + " · ".join(f"{n} {v:.3f}" for n, v in rep["aps_bps"].items()), "",
         "Prediction correlation (weighted): " + " · ".join(f"{k} {v:.2f}"
                                                            for k, v in rep["prediction_correlation"].items()), ""]
    for title, rows in rep["groups"].items():
        L += [f"## {title}", "", "| group | rows | weight | " + " | ".join(names) +
              (f" | {other} − {ref} |" if other != ref else " |"),
              "|---|---:|---:|" + "---:|" * len(names) + ("---:|" if other != ref else "")]
        for r in rows:
            L.append(f"| {r['group']} | {r['row_share']:.1%} | {r['weight_share']:.1%} | " +
                     " | ".join(f"{r[n]:+.3f}" for n in names) +
                     (f" | {r[other] - r[ref]:+.3f} |" if other != ref else " |"))
        L.append("")
    if "instrument_gain" in rep:
        g = rep["instrument_gain"]
        L += [f"## Gain of {other} over {ref} across instruments", "",
              f"- {g['n_instruments']} instruments; total gain {g['total_gain_bps']:+.3f} bps/day",
              f"- instruments where {other} gains: {g['share_of_instruments_with_gain']:.0%}",
              f"- top 10 instruments: {g['top10_gain_bps']:+.3f}; worst 10: {g['worst10_gain_bps']:+.3f}",
              f"- top 10% of instruments carry {g['top10pct_instruments_share_of_positive_gain']:.0%} "
              "of the positive gain" if g["top10pct_instruments_share_of_positive_gain"] is not None else "", ""]
    L.append("Contributions are in bps per day; each grouping's rows sum to the model's APS.")
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m experiment.analyze", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", required=True)
    p.add_argument("--model", action="append", required=True,
                   help="name=run_or_glob (repeatable; first = reference)")
    p.add_argument("--data", default=None)
    p.add_argument("--results", default=None)
    p.add_argument("--out", default=None, help="default: results/<split>/analysis")
    a = p.parse_args(argv)
    models = dict(m.split("=", 1) for m in a.model)
    rep = analyze(a.split, models, a.data, a.results)
    out = Path(a.out or (Path(a.results or default_results_dir(a.split)) / "analysis"))
    out.mkdir(parents=True, exist_ok=True)
    (out / "analysis.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    md = to_markdown(rep, list(models))
    (out / "ANALYSIS.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
