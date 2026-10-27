"""Overall evaluation table: each final model's APS on both validation windows plus
its rolling (official) metrics over the contract ``is`` period.

    python -m experiment.overall            # prints and writes results/OVERALL.md

Window columns read the saved records under results/<split>/ (mean over seeds
for single models; combination runs as written by experiment/combine.py).
Rolling columns read runs/<name>/report.json (experiment/rolling.py merge or
experiment/combine_rolling.py); a model whose report doesn't exist yet shows
"pending". Rerun this after each rolling run finishes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
WINDOWS = {"W1": "v2_52w", "W2": "v3_52w_jan24"}

# label, run glob per window (None = not run there), rolling run dir (None = no plan)
MODELS = [
    ("Ridge (official)", {"W1": "ridge_eqall", "W2": "ridge_eqall"}, "runs/linear_4w"),
    ("Linear MLP", {"W1": "lin2_lr3e-5_con_ema_loss_s*", "W2": "lin_ema_s*"}, "runs/mlplin_4w"),
    ("2-layer MLP", {"W1": "ladder_h256x128_s*", "W2": "h256x128_s*"}, "runs/mlp2_4w"),
    ("LightGBM", {"W1": "lgb_lr01_loss_s*", "W2": "lgb_lr01_loss_s*"}, "runs/lgbm3k_4w"),
    ("Ridge + LightGBM", {"W1": "cmb_ridge_lgb", "W2": "cmb_ridge_lgb"}, "runs/cmb_ridge_lgb_4w"),
    ("Linear MLP + LightGBM", {"W1": "cmb_mlplin_lgb", "W2": "cmb_mlplin_lgb"},
     "runs/cmb_mlplin_lgb_4w"),
    ("Ridge + LightGBM + MLP", {"W1": "cmb_ridge_lgb_mlp2", "W2": "cmb_ridge_lgb_mlp2"},
     "runs/cmb_ridge_lgb_mlp2_4w"),
]

COLUMNS = ["Model", "W1 APS", "W2 APS", "Rolling APS", "Sharpe", "COR %", "AR",
           "diff Sharpe", "win %"]


def window_aps(results: Path, pattern: str | None) -> tuple[float | None, int]:
    """Mean val APS over the runs matching ``pattern`` (seeds), and how many."""
    if pattern is None:
        return None, 0
    recs = [json.loads(p.read_text(encoding="utf-8"))
            for p in sorted(results.glob(f"{pattern}/record.json"))]
    if not recs:
        return None, 0
    return float(np.mean([r["report"]["windows"]["val"]["APS_mean_bps"] for r in recs])), len(recs)


def rolling_metrics(run_dir: Path) -> dict | None:
    rep = run_dir / "report.json"
    if not rep.is_file():
        return None
    r = json.loads(rep.read_text(encoding="utf-8"))
    w, vs = r["windows"]["is"], r.get("vs_baseline", {}).get("is")
    out = {"APS": w["APS_mean_bps"], "SR": w["APS_SR"], "COR": w["COR_mean"] * 100,
           "AR": w["AR_mean"], "days": w["n_dates"],
           "period": f"{w['start_date']}..{w['end_date']}"}
    # The baseline itself is compared with itself: no paired columns.
    if vs and vs["dAPS_mean_bps"] != 0:
        out.update(dSR=vs["diff_sharpe"], win=vs["win_rate"] * 100)
    return out


def build(repo: Path = REPO, models=MODELS) -> tuple[list[dict], str]:
    rows, periods = [], set()
    for label, pats, run in models:
        row = {"Model": label}
        for w, split in WINDOWS.items():
            aps, n = window_aps(repo / "results" / split, pats.get(w))
            row[f"{w} APS"] = None if aps is None else (f"{aps:.2f}" + (f" ({n} seeds)" if n > 1 else ""))
        m = rolling_metrics(repo / run) if run else None
        if m is None:
            row.update({c: "pending" for c in COLUMNS[3:]})
        else:
            periods.add((m["period"], m["days"]))
            row.update({"Rolling APS": f"{m['APS']:.2f}", "Sharpe": f"{m['SR']:.1f}",
                        "COR %": f"{m['COR']:.2f}", "AR": f"{m['AR']:.2f}",
                        "diff Sharpe": f"{m['dSR']:+.1f}" if "dSR" in m else "baseline",
                        "win %": f"{m['win']:.0f}" if "win" in m else "–"})
        rows.append(row)

    period = ", ".join(f"{p} ({d} days)" for p, d in sorted(periods)) or "not yet run"
    lines = ["# Overall evaluation", "",
             "W1 = val 2024-04-09..05-06, W2 = val 2024-01-16..02-12 (28 days each, "
             "52 weeks of training, stride 5, identical rows; single models averaged "
             "over seeds, combinations of seed-averaged parts).",
             f"Rolling = official kit evaluation, 104-week training, 4-week refit, {period}; "
             "one seed, ~10M sampled rows per anchor (ridge uses all rows); paired "
             "columns vs the official ridge (`runs/linear_4w`) on the same days.", "",
             "| " + " | ".join(COLUMNS) + " |",
             "|---|" + "---:|" * (len(COLUMNS) - 1)]
    lines += ["| " + " | ".join("–" if r[c] is None else str(r[c]) for c in COLUMNS) + " |"
              for r in rows]
    lines += ["", "APS in bps; Sharpe = annualized Sharpe of daily APS; diff Sharpe = the "
              "same for the daily APS difference vs ridge (not the gap between the Sharpes: "
              "it measures how consistently the model beats ridge); win % = days beating ridge."]
    return rows, "\n".join(lines) + "\n"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m experiment.overall", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--md", default=str(REPO / "results" / "OVERALL.md"))
    a = p.parse_args(argv)
    _, md = build()
    print(md)
    Path(a.md).write_text(md, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
