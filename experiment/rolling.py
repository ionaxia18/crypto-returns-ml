"""Official rolling evaluation, split into one parallel job per refit anchor.

``f522kit run`` fits its anchors one after another (~34 fits at 4w); for a
model that takes ~1 h per fit that is more than a day. Each anchor's fit is
independent, so this runs the SAME driver once per anchor -- restricted with
``--start/--end`` to exactly the dates that anchor predicts -- in separate
array tasks, then merges the parts into one kit-format run:

    python -m experiment.rolling count --data $DATA_ROOT
    python -m experiment.rolling run   --data $DATA_ROOT --model models/mlp_final.py:TwoLayer \\
        --out runs/mlp2_4w --index 3 --row-budget 10000000
    python -m experiment.rolling merge --data $DATA_ROOT --out runs/mlp2_4w

Each part lives in ``<out>/parts/<anchor>/`` with its own run_manifest.json;
``merge`` checks every part used identical settings (kit version, dataset,
model file hash, stride, row budget, seed), that every predictable date in
the window has a prediction, and writes ``<out>/preds/`` (hard links) plus a
combined ``<out>/run_manifest.json``. Score the merged run with the kit:

    f522kit score --data $DATA_ROOT --preds runs/mlp2_4w --baseline runs/linear_4w \\
        --window is,2022-07-01,2024-06-30

The window defaults to the contract's ``is`` period. Dates in the dev window
(Jul-Dec 2024) are never predicted, so dev stays untouched.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import shutil
from pathlib import Path

from f522kit import contract
from f522kit.data import Dataset
from f522kit.driver import resolve_schedule, run

SETTING_KEYS = ("kit_version", "dataset_id", "model_spec", "model_sha256", "stride",
                "row_budget", "seed", "calibration_rule", "assignment_rule")


def window_bounds(start=None, end=None):
    is_start, is_end = contract.DEFAULT_WINDOWS["is"]
    start, end = start or is_start, end or is_end
    dev_start, _ = contract.DEFAULT_WINDOWS["dev"]
    if end >= dev_start:
        raise ValueError(f"--end {end} reaches the dev window (from {dev_start}), "
                         "which stays untouched")
    return max(start, contract.REPORT_START), end


def anchor_plan(dataset: Dataset, stride="4w", start=None, end=None):
    """[(anchor_key, first_date, last_date)] for anchors scoring dates in the window."""
    start, end = window_bounds(start, end)
    schedule = resolve_schedule(dataset, stride)
    by_anchor: dict[datetime.date, list[datetime.date]] = {}
    for date, anchor in schedule.assignment.items():
        if start <= date <= end:
            by_anchor.setdefault(anchor.key, []).append(date)
    return [(k, min(v), max(v)) for k, v in sorted(by_anchor.items())]


def run_part(data_root, model, out, index, row_budget, stride="4w", seed=0,
             start=None, end=None):
    dataset = Dataset(data_root)
    plan = anchor_plan(dataset, stride, start, end)
    if not 1 <= index <= len(plan):
        raise SystemExit(f"--index {index} out of range 1..{len(plan)}")
    key, first, last = plan[index - 1]
    part = Path(out) / "parts" / key.isoformat()
    print(f"anchor {key}: predicting {first}..{last} into {part}", flush=True)
    run(dataset, model_spec=model, out_dir=part, stride=stride, row_budget=row_budget,
        seed=seed, start=first, end=last)


def merge(data_root, out, stride="4w", start=None, end=None):
    dataset = Dataset(data_root)
    out = Path(out)
    plan = anchor_plan(dataset, stride, start, end)
    manifests = {}
    for key, _, _ in plan:
        path = out / "parts" / key.isoformat() / "run_manifest.json"
        if not path.is_file():
            raise SystemExit(f"missing part for anchor {key}: {path} (did that task fail?)")
        manifests[key] = json.loads(path.read_text(encoding="utf-8"))
    first = next(iter(manifests.values()))
    for key, m in manifests.items():
        diff = [k for k in SETTING_KEYS if m.get(k) != first.get(k)]
        if diff:
            raise SystemExit(f"anchor {key} used different settings: {diff}")

    preds = out / "preds"
    preds.mkdir(parents=True, exist_ok=True)
    combined = {k: first[k] for k in SETTING_KEYS if k in first}
    combined.update({"anchors": {}, "predictions": {},
                     "merged_from_parts": [k.isoformat() for k in manifests]})
    for key, m in manifests.items():
        combined["anchors"].update(m["anchors"])
        for date, entry in m["predictions"].items():
            src = out / "parts" / key.isoformat() / "preds" / f"{date}.npy"
            dst = preds / f"{date}.npy"
            if dst.exists():
                dst.unlink()
            try:
                os.link(src, dst)
            except OSError:
                shutil.copy2(src, dst)
            combined["predictions"][date] = entry

    wanted = {d.isoformat() for _, a, b in plan
              for d in dataset.dates("visible") if a <= d <= b}
    scored = {d.isoformat() for d in resolve_schedule(dataset, stride).scored_dates}
    missing = sorted((wanted & scored) - set(combined["predictions"]))
    if missing:
        raise SystemExit(f"{len(missing)} predictable dates missing (first {missing[0]})")
    (out / "run_manifest.json").write_text(json.dumps(combined, indent=2, sort_keys=True),
                                          encoding="utf-8")
    print(f"merged {len(manifests)} anchors, {len(combined['predictions'])} dates into {out}")
    return combined


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m experiment.rolling", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=("count", "plan", "run", "merge"))
    p.add_argument("--data", required=True)
    p.add_argument("--model", help="path.py:Class (run)")
    p.add_argument("--out", help="run directory (run, merge)")
    p.add_argument("--index", type=int,
                   default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)) or None,
                   help="1-based anchor (default: $SLURM_ARRAY_TASK_ID)")
    p.add_argument("--row-budget", type=int, default=10_000_000)
    p.add_argument("--stride", default="4w")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--start", type=datetime.date.fromisoformat, default=None)
    p.add_argument("--end", type=datetime.date.fromisoformat, default=None)
    a = p.parse_args(argv)
    if a.command in ("count", "plan"):
        plan = anchor_plan(Dataset(a.data), a.stride, a.start, a.end)
        if a.command == "count":
            print(len(plan))
        else:
            for i, (k, f, l) in enumerate(plan, 1):
                print(f"{i:3d}  anchor {k}  predicts {f}..{l}")
    elif a.command == "run":
        if not (a.model and a.out and a.index):
            raise SystemExit("run needs --model, --out and --index (or SLURM_ARRAY_TASK_ID)")
        run_part(a.data, a.model, a.out, a.index, a.row_budget, a.stride, a.seed,
                 a.start, a.end)
    else:
        if not a.out:
            raise SystemExit("merge needs --out")
        merge(a.data, a.out, a.stride, a.start, a.end)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
