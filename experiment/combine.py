"""Combined models built from saved predictions: no retraining.

    python -m experiment.combine --split $SCRATCH/core980-work/experiment/v2_52w \\
        --spec experiments/combos_v1.json

A spec names, per split, which saved runs make up each component (seeds are
averaged into one prediction: a seed ensemble), and a list of combinations:

    {"components": {"v2_52w": {"ridge": ["ridge_eqall"], "lgb": ["lgb_lr01_loss_s*"]}},
     "combos": [{"name": "cmb_ridge_lgb", "parts": {"ridge": 0.5, "lgb": 0.5},
                 "norm": "rms"}]}

Each combination = sum_k weight_k * scale_k * component_k, where
  norm "none": scale_k = 1 (raw average: the model with larger predictions
               dominates);
  norm "rms":  scale_k = 1 / RMS of component k's predictions on the val rows
               (label-free), then the sum is rescaled to the weighted average
               RMS, so each model contributes in proportion to its weight.

It is written like any run (results/<split>/<name>/{preds/, record.json},
model "combo") and scored by the kit, so the seed-aware table compares it:

    python -m experiment.table --results results/v2_52w --by-config --baseline ridge_eqall

Caveat: picking the best weights or combination on the same 28 val days
flatters it, exactly like picking hyperparameters there; report all of them.
"""

from __future__ import annotations

import argparse
import datetime
import fnmatch
import json
import shutil
import time
from pathlib import Path

import numpy as np

from f522kit.data import Dataset

from .run import REPO, default_results_dir, score_val, write_preds
from .split import ExperimentSplit


def gather(split: ExperimentSplit, run_dir: Path) -> np.ndarray:
    """A run's saved kit-format predictions as a flat vector over the val rows."""
    _, _, _, d = split.val()
    rows = split.val_rows()
    out = np.empty(rows.shape[0], dtype=np.float64)
    for i, date in enumerate(split.val_dates):
        sel = np.flatnonzero(d == i)
        out[sel] = np.load(run_dir / "preds" / f"{date.isoformat()}.npy").reshape(-1)[rows[sel]]
    return out


def resolve_runs(results: Path, patterns: list[str]) -> list[str]:
    """Run names (with a finished record and predictions) matching the patterns."""
    have = sorted(p.parent.name for p in results.glob("*/record.json")
                  if (p.parent / "preds").is_dir())
    found = []
    for pat in patterns:
        found += [n for n in have if fnmatch.fnmatch(n, pat) and n not in found]
    return found


def build(split_dir, spec_path, data_root=None, results=None, force=False, only=None):
    split = ExperimentSplit(split_dir)
    dataset = Dataset(data_root or split.meta["data_root"])
    results = Path(results or default_results_dir(split_dir))
    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    window = Path(split_dir).resolve().name
    comp_spec = spec["components"].get(window)
    if comp_spec is None:
        raise SystemExit(f"spec has no components for split {window!r}")

    cache: dict[str, tuple[np.ndarray, list[str]]] = {}

    def component(alias):
        if alias not in cache:
            runs = resolve_runs(results, comp_spec[alias])
            if not runs:
                raise LookupError(f"no finished runs for component {alias!r} ({comp_spec[alias]})")
            cache[alias] = (np.mean([gather(split, results / r) for r in runs], axis=0), runs)
        return cache[alias]

    made = []
    for combo in spec["combos"]:
        name, parts, norm = combo["name"], combo["parts"], combo.get("norm", "none")
        if only and name not in only:
            continue
        if norm not in ("none", "rms"):
            raise SystemExit(f"{name}: norm must be none|rms")
        out = results / name
        if out.exists():
            if not force:
                print(f"skip {name}: exists (use --force)")
                continue
            shutil.rmtree(out)
        try:
            comps = {a: component(a) for a in parts}
        except LookupError as exc:
            print(f"skip {name}: {exc}")
            continue
        t0 = time.time()
        total_w = float(sum(parts.values()))
        rms = {a: float(np.sqrt(np.mean(comps[a][0] ** 2))) for a in parts}
        if norm == "rms":
            pred = sum(w * comps[a][0] / rms[a] for a, w in parts.items()) / total_w
            pred *= sum(w * rms[a] for a, w in parts.items()) / total_w
        else:
            pred = sum(w * comps[a][0] for a, w in parts.items()) / total_w
        out.mkdir(parents=True)
        write_preds(dataset, split, pred.astype(np.float32), out / "preds")
        report = score_val(dataset, split, out)
        first = json.loads((results / comps[next(iter(parts))][1][0] / "record.json")
                           .read_text(encoding="utf-8"))
        config = {"parts": parts, "norm": norm,
                  "components": {a: comps[a][1] for a in parts}}
        record = {
            "name": name, "model": "combo", "spec": "combo", "config": config,
            "resolved_config": config, "split": first["split"],
            "split_meta": first.get("split_meta", {}), "report": report,
            "fit_info": {"component_rms": rms},
            "timing_s": {"fit": 0.0, "total": round(time.time() - t0, 2)},
            "peak_rss_gb": {"after_fit": 0.0, "final": 0.0}, "cpus": 0,
            "finished": datetime.datetime.now().isoformat(timespec="seconds"),
        }
        (out / "record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        m = report["windows"]["val"]
        print(f"[{name}] APS {m['APS_mean_bps']:.4f} | COR {m['COR_mean']*100:.3f}% | "
              f"AR {m['AR_mean']:.3f} | parts {parts} ({norm})", flush=True)
        made.append(record)
    return made


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m experiment.combine", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", required=True)
    p.add_argument("--spec", required=True)
    p.add_argument("--data", default=None)
    p.add_argument("--results", default=None, help="default: results/<split dir name>")
    p.add_argument("--force", action="store_true", help="rebuild combinations that exist")
    p.add_argument("--only", nargs="*", default=None, help="build only these combinations")
    a = p.parse_args(argv)
    build(a.split, a.spec, a.data, a.results, a.force, a.only)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
