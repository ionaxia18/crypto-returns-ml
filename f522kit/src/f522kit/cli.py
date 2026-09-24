"""Command-line interface: verify-data, anchors, run, score, smoke."""

from __future__ import annotations

import argparse
import datetime
import json
import sys
import tempfile
from pathlib import Path

from . import contract
from .data import Dataset
from .driver import resolve_schedule, run
from .fixture import build_fixture
from .scoring import format_report, save_report, score


def _parse_date(value: str) -> datetime.date:
    return datetime.date.fromisoformat(value)


def _parse_windows(values: list[str] | None):
    if not values:
        return None
    windows = {}
    for item in values:
        label, start, end = item.split(",")
        windows[label] = (_parse_date(start), _parse_date(end))
    return windows


def cmd_verify_data(args: argparse.Namespace) -> int:
    dataset = Dataset(args.data)
    dates = dataset.dates("all")
    if args.sample and args.sample < len(dates):
        step = max(1, len(dates) // args.sample)
        dates = dates[::step][: args.sample]
    checked = dataset.verify(dates=dates, deep=args.deep)
    print(
        f"OK: dataset {dataset.dataset_id} — {checked} shards verified"
        f" ({'deep' if args.deep else 'structural'})"
    )
    return 0


def cmd_anchors(args: argparse.Namespace) -> int:
    dataset = Dataset(args.data)
    schedule = resolve_schedule(dataset, args.stride)
    print(f"stride={args.stride}  anchors={len(schedule.anchors)}")
    for anchor in schedule.anchors:
        n_dates = sum(len(schedule.groups[k]) for k in anchor.training_keys)
        predicted = sorted(
            d for d, a in schedule.assignment.items() if a.key == anchor.key
        )
        print(
            f"  {anchor.key}  train_groups={len(anchor.training_keys):3d} "
            f"train_dates={n_dates:4d}  predicts {predicted[0]}..{predicted[-1]} "
            f"({len(predicted)} dates)"
        )
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    dataset = Dataset(args.data)
    run(
        dataset,
        model_spec=args.model,
        out_dir=args.out,
        stride=args.stride,
        row_budget=args.row_budget,
        seed=args.seed,
        start=args.start,
        end=args.end,
        include_hidden=args.include_hidden,
    )
    return 0


def cmd_score(args: argparse.Namespace) -> int:
    dataset = Dataset(args.data)
    report = score(
        dataset,
        preds_dir=args.preds,
        windows=_parse_windows(args.window),
        baseline_dir=args.baseline,
        require_all_dates=not args.allow_partial,
    )
    print(format_report(report))
    if args.out:
        save_report(report, args.out)
        print(f"Report written to {args.out}")
    return 0


def cmd_smoke(args: argparse.Namespace) -> int:
    """Build the synthetic fixture and run the whole pipeline on it."""
    with tempfile.TemporaryDirectory(prefix="f522kit_smoke_") as tmp:
        tmp_path = Path(tmp)
        data_root = build_fixture(tmp_path / "fixture")
        dataset = Dataset(data_root)
        dataset.verify(deep=True)
        print(f"fixture: {len(dataset.dates('all'))} dates verified (deep)")

        model_path = tmp_path / "smoke_model.py"
        model_path.write_text(
            "import numpy as np\n"
            "class SmokeRidge:\n"
            "    def fit(self, train):\n"
            "        X, y, w = train.sample(20000, seed=0)\n"
            "        Xw = X * w[:, None]\n"
            "        xx = Xw.T @ X + 1e-6 * np.eye(X.shape[1])\n"
            "        xy = Xw.T @ y\n"
            "        self.beta = np.linalg.solve(xx, xy)\n"
            "    def predict(self, X):\n"
            "        return X @ self.beta\n",
            encoding="utf-8",
        )
        out_dir = tmp_path / "run"
        dataset_start = dataset.dates("visible")[0]
        run(
            dataset,
            model_spec=f"{model_path}:SmokeRidge",
            out_dir=out_dir,
            stride="1w",
            row_budget=20000,
            seed=0,
            start=dataset_start,
            verbose=True,
        )
        first = dataset.dates("visible")[0]
        last = dataset.dates("visible")[-1]
        report = score(
            dataset,
            preds_dir=out_dir,
            windows={"smoke": (first, last)},
        )
        print(format_report(report))
        cor = report["windows"]["smoke"]["COR_mean"]
        if cor < 0.2:
            print(f"SMOKE FAIL: planted-signal COR {cor:.3f} < 0.2", file=sys.stderr)
            return 1
        print(f"SMOKE OK: planted-signal COR {cor:.3f}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="f522kit",
        description="Nonlinear modeling on anonymous core980 features",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("verify-data", help="validate a dataset root")
    p.add_argument("--data", required=True)
    p.add_argument("--deep", action="store_true", help="re-hash every array file")
    p.add_argument("--sample", type=int, default=None, help="check only N spread dates")
    p.set_defaults(func=cmd_verify_data)

    p = sub.add_parser("anchors", help="print the refit schedule")
    p.add_argument("--data", required=True)
    p.add_argument("--stride", choices=sorted(contract.STRIDES), default=contract.DEFAULT_STRIDE)
    p.set_defaults(func=cmd_anchors)

    p = sub.add_parser("run", help="execute a rolling run of a student model")
    p.add_argument("--data", required=True)
    p.add_argument("--model", required=True, help="path/to/module.py:ClassName")
    p.add_argument("--out", required=True)
    p.add_argument("--stride", choices=sorted(contract.STRIDES), default=contract.DEFAULT_STRIDE)
    p.add_argument(
        "--row-budget",
        type=int,
        required=contract.DEFAULT_ROW_BUDGET is None,
        default=contract.DEFAULT_ROW_BUDGET,
        help="max rows per fit via train.sample(); required until TBD-3 freezes a default",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--start", type=_parse_date, default=None)
    p.add_argument("--end", type=_parse_date, default=None)
    p.add_argument("--include-hidden", action="store_true")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("score", help="score a prediction run")
    p.add_argument("--data", required=True)
    p.add_argument("--preds", required=True)
    p.add_argument("--baseline", default=None)
    p.add_argument(
        "--window",
        action="append",
        help="label,start,end (repeatable); defaults to contract windows",
    )
    p.add_argument("--allow-partial", action="store_true")
    p.add_argument("--out", default=None, help="write JSON report here")
    p.set_defaults(func=cmd_score)

    p = sub.add_parser("smoke", help="end-to-end self-test on a synthetic fixture")
    p.set_defaults(func=cmd_smoke)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
