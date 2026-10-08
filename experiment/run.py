"""Run one model on the experiment split and log metrics, runtime and peak memory.

    python -m experiment.run --split $SCRATCH/core980-work/experiment/v2_52w --model lgbm \\
        --config '{"max_rows": 2000000}' --name lgbm_default

``--model`` is ``ridge|lgbm|mlp`` (-> models/<name>.py:MyModel) or any
``path.py:Class`` exposing ``fit_arrays(X, y, w, date_idx)`` and
``predict(X)``. ``--config`` is JSON passed as constructor kwargs.

Several runs: ``--list experiments/<file>.txt`` with one
``name model {json config}`` per line runs entry ``--index`` (default
``$SLURM_ARRAY_TASK_ID``); this is what scripts/experiment_run.sbatch uses as
an array job. ``--list FILE --count`` prints the number of entries.

Writes a run directory ``<out>/<name>/`` (``--out`` defaults to
``results/<split dir name>``, so runs on different splits never mix):

    preds/YYYY-MM-DD.npy   float32 [1440, S] per val date, the kit's
                           prediction format (0 on rows the split skips,
                           which all have w = 0 and so score nothing)
    record.json            config, kit report, daily APS, timings,
                           peak RSS, host/CPU info, git commit

Scoring is ``f522kit.scoring.score`` itself over the val dates, so the
numbers are the canonical evaluator's by construction. Because the preds
are in kit format, any kit run (e.g. the full 104-week ``runs/linear_4w``)
can be scored and compared on the same dates; see experiment/table.py.
``--baseline <dir>`` adds the kit's paired daily-APS diff vs that run.

Peak RSS is ``ru_maxrss`` of this process; it includes resident pages of
memory-mapped split files, so it upper-bounds the anonymous memory a model
needs. SLURM's ``sacct -o MaxRSS`` is the number that matters for --mem.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import platform
import resource
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from f522kit.adapter import load_model_factory
from f522kit.data import Dataset
from f522kit.scoring import collect_daily_stats, score

from .split import ExperimentSplit

REPO = Path(__file__).resolve().parent.parent
MODELS = {name: f"{REPO}/models/{name}.py:MyModel" for name in ("ridge", "lgbm", "mlp")}


def default_results_dir(split_dir) -> Path:
    """results/<split dir name>, e.g. results/v2_52w."""
    return REPO / "results" / Path(split_dir).resolve().name


def resolved_config(model) -> dict:
    """The model's settings after construction (plain JSON-able attributes)."""
    simple = (int, float, str, bool, type(None))

    def ok(v):
        if isinstance(v, simple):
            return True
        if isinstance(v, (list, tuple)):
            return all(ok(x) for x in v)
        if isinstance(v, dict):
            return all(isinstance(k, str) and ok(x) for k, x in v.items())
        return False

    return {k: (list(v) if isinstance(v, tuple) else v)
            for k, v in sorted(vars(model).items()) if not k.startswith("_") and ok(v)}


def load_class(spec: str):
    """The model class behind ``path.py:Class``, loaded the way the kit does."""
    # The kit's factory constructs with no arguments; we need kwargs.
    return type(load_model_factory(spec)())


def peak_rss_gb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / (1e9 if sys.platform == "darwin" else 1e6)  # bytes vs KiB


def git_commit() -> str:
    try:
        out = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain",
                                "models", "experiment"], capture_output=True, text=True).stdout
        return out + ("-dirty" if dirty.strip() else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def cpus() -> int:
    env = os.environ.get("SLURM_CPUS_PER_TASK")
    if env:
        return int(env)
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def write_preds(dataset: Dataset, split: ExperimentSplit, pred: np.ndarray, preds_dir: Path) -> None:
    """Scatter flat val predictions into per-date kit-format [T, S] files."""
    preds_dir.mkdir(parents=True, exist_ok=True)
    _, _, _, d = split.val()
    rows = split.val_rows()
    for i, date in enumerate(split.val_dates):
        sel = np.flatnonzero(d == i)
        grid = np.zeros(dataset.time_rows * dataset.shard(date).n_symbols, dtype=np.float32)
        grid[rows[sel]] = pred[sel]
        np.save(preds_dir / f"{date.isoformat()}.npy", grid.reshape(dataset.time_rows, -1))


def score_val(dataset: Dataset, split: ExperimentSplit, run_dir, baseline_dir=None) -> dict:
    """Kit report over the split's val dates, plus the daily APS series (bps)."""
    window = {"val": (split.val_dates[0], split.val_dates[-1])}
    report = score(dataset, run_dir, windows=window, baseline_dir=baseline_dir)
    preds = Path(run_dir) / "preds" if (Path(run_dir) / "preds").is_dir() else Path(run_dir)
    stats, _ = collect_daily_stats(dataset, preds, split.val_dates)
    report["daily_APS_bps"] = [s.aps * 1e4 for s in stats]
    return report


def run_one(split_dir, model, config, name, out_dir, in_ram=False, baseline=None,
            data_root=None, force=False) -> dict:
    split = ExperimentSplit(split_dir)
    dataset = Dataset(data_root or split.meta["data_root"])
    if dataset.dataset_id != split.meta["dataset_id"]:
        raise ValueError("dataset root does not match the split's dataset_id")
    if baseline:
        # Fail before the fit, not after: in an array job the baseline run may
        # not have finished (table.py does the pairing afterwards anyway).
        missing = [d for d in split.val_dates
                   if not (Path(baseline) / "preds" / f"{d.isoformat()}.npy").is_file()
                   and not (Path(baseline) / f"{d.isoformat()}.npy").is_file()]
        if missing:
            raise FileNotFoundError(
                f"baseline {baseline} lacks predictions for {len(missing)} val dates "
                f"(first {missing[0]}); run it first or drop --baseline")
    run_dir = Path(out_dir) / name
    if run_dir.exists():
        if not force:
            raise FileExistsError(f"{run_dir} exists; pick a new --name or pass --force")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)

    spec = MODELS.get(model, model)
    n_cpus = cpus()
    config = {"threads": n_cpus, **config} if model in ("lgbm", "mlp") else config
    record = {
        "name": name, "model": model, "spec": spec, "config": config,
        "split": str(split_dir), "split_meta": {k: split.meta[k] for k in (
            "split_version", "anchor", "train_weeks", "minute_stride", "dataset_id")},
        "cpus": n_cpus, "host": socket.gethostname(), "platform": platform.platform(),
        "slurm_job": os.environ.get("SLURM_JOB_ID"), "git": git_commit(),
        "started": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    print(f"[{name}] {spec} cpus={n_cpus} config={config}", flush=True)

    t0 = time.time()
    X, y, w, d = split.train(in_ram=in_ram)
    load_s = time.time() - t0

    model_obj = load_class(spec)(**config)
    # Every setting the model actually uses, defaults included, so results
    # stay comparable even if a model's defaults change later.
    record["resolved_config"] = resolved_config(model_obj)
    t1 = time.time()
    fit_info = model_obj.fit_arrays(X, y, w, date_idx=d) or {}
    fit_s = time.time() - t1
    rss_after_fit = peak_rss_gb()
    del X, y, w, d

    Xv, _, _, _ = split.val()
    t2 = time.time()
    pred = np.asarray(model_obj.predict(Xv), dtype=np.float32)
    pred_s = time.time() - t2
    write_preds(dataset, split, pred, run_dir / "preds")

    report = score_val(dataset, split, run_dir, baseline_dir=baseline)
    record.update({
        "fit_info": fit_info,
        "report": report,
        "baseline": str(baseline) if baseline else None,
        # With a memmapped X (no --in-ram) "load" is ~0 and disk reads are
        # inside "fit"; compare fit times only between runs with the same mode.
        "x_in_ram": bool(in_ram),
        "timing_s": {"load": round(load_s, 2), "fit": round(fit_s, 2),
                     "predict": round(pred_s, 2), "total": round(time.time() - t0, 2)},
        "peak_rss_gb": {"after_fit": round(rss_after_fit, 3), "final": round(peak_rss_gb(), 3)},
        "finished": datetime.datetime.now().isoformat(timespec="seconds"),
    })
    (run_dir / "record.json").write_text(json.dumps(record, indent=2, default=str),
                                         encoding="utf-8")
    m = report["windows"]["val"]
    print(f"[{name}] APS {m['APS_mean_bps']:.4f} bps | COR {m['COR_mean']*100:.3f}% | "
          f"AR {m['AR_mean']:.3f} | APS_SR {m['APS_SR']:.2f} | fit {fit_s:.0f}s | "
          f"peak {record['peak_rss_gb']['final']:.1f} GB", flush=True)
    return record


def read_list(path) -> list[tuple[str, str, dict]]:
    """Experiment list: one ``name model {json config}`` per line; # comments."""
    entries = []
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split(None, 2)
        if len(parts) < 2:
            raise SystemExit(f"{path}:{n}: expected 'name model [json]'")
        try:
            config = json.loads(parts[2]) if len(parts) == 3 else {}
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{path}:{n}: bad JSON config ({exc})") from exc
        entries.append((parts[0], parts[1], config))
    names = [e[0] for e in entries]
    if len(set(names)) != len(names):
        raise SystemExit(f"{path}: duplicate run names")
    return entries


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m experiment.run", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", required=False)
    p.add_argument("--model", default=None)
    p.add_argument("--config", default="{}", help="JSON constructor kwargs")
    p.add_argument("--name", default=None)
    p.add_argument("--list", default=None,
                   help="experiment list file (replaces --model/--config/--name)")
    p.add_argument("--index", type=int, default=None,
                   help="1-based entry of --list (default: $SLURM_ARRAY_TASK_ID)")
    p.add_argument("--count", action="store_true", help="print the number of --list entries")
    p.add_argument("--suffix", default="", help="appended to the run name, e.g. _c32")
    p.add_argument("--out", default=None, help="default: results/<split dir name>")
    p.add_argument("--in-ram", action="store_true", help="load train X fully into RAM")
    p.add_argument("--baseline", default=None,
                   help="run dir to pair against (e.g. results/v2_52w/ridge or runs/linear_4w)")
    p.add_argument("--data", default=None, help="dataset root (default: the split's)")
    p.add_argument("--force", action="store_true", help="overwrite an existing run dir")
    a = p.parse_args(argv)
    if a.list:
        entries = read_list(a.list)
        if a.count:
            print(len(entries))
            return 0
        index = a.index or int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
        if not 1 <= index <= len(entries):
            raise SystemExit(f"--index {index} out of range 1..{len(entries)} for {a.list}")
        name, model, config = entries[index - 1]
    else:
        if not a.model:
            raise SystemExit("pass --model (or --list)")
        model, config = a.model, json.loads(a.config)
        name = a.name or f"{Path(a.model).stem}_{cpus()}cpu"
    if not a.split:
        raise SystemExit("--split is required")
    out = a.out or str(default_results_dir(a.split))
    run_one(a.split, model, config, name + a.suffix, out, in_ram=a.in_ram,
            baseline=a.baseline, data_root=a.data, force=a.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
