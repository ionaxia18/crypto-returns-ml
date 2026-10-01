"""What does the minute stride cost? Ridge at several strides, one data pass.

    python -m experiment.checks.stride_test --data $DATA_ROOT --out results/stride_test.json

Uses the experiment split's dates (same anchor, same ``--train-weeks``) but reads
shards directly, so no cache is built. Ridge needs only X'X and X'y, so a
single pass over the training dates yields every stride at once:

    each minute m belongs to exactly one class: the largest stride s in
    STRIDES with m % s == 0. Nested strides (30 | 15 | 5 | 1 by default)
    make stride s = the sum of the classes of every stride >= s.

Val is then streamed once and all fits are scored per date with the kit's
``daily_stats`` (every positive-weight row; same numbers as f522kit score).
Output: mean-daily APS / COR / AR / APS_SR per stride, plus the paired
daily-APS diff of each stride vs. stride 1.

Cost is one full read of the training and val dates (~270 GB at 26 weeks)
plus one stride-1 ridge accumulation. Reads run in background threads
(``--readers`` dates ahead, ~1.3 GB RAM each) while BLAS uses the CPUs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np

from f522kit.data import Dataset
from f522kit.metrics import aggregate, daily_stats, paired_diff

from experiment.split import TRAIN_WEEKS, _gather_x, choose_anchor, train_dates_for

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from models.ridge import accumulate, ridge_solve  # noqa: E402

STRIDES = (30, 15, 5, 1)
CHUNK = 65_536


def minute_class(n_minutes: int, strides: tuple[int, ...]) -> np.ndarray:
    """Class index per minute: position in ``strides`` (descending) it falls in."""
    cls = np.full(n_minutes, len(strides) - 1)
    for i in reversed(range(len(strides))):
        cls[np.arange(n_minutes) % strides[i] == 0] = i
    return cls


def prefetch(dataset: Dataset, dates, readers: int):
    """Yield (date, shard, x3, y, w) in order, read into RAM by background threads.

    A reader may start date i only once i < consumed + readers, so at most
    ``readers`` dates (~1.3 GB each) are held ahead of the consumer, and the
    date the consumer needs next is never blocked behind later ones.
    """
    cond = threading.Condition()
    state = {"next": 0, "consumed": 0}
    ready: dict[int, tuple] = {}

    def reader():
        while True:
            with cond:
                cond.wait_for(lambda: state["next"] >= len(dates)
                              or state["next"] < state["consumed"] + readers)
                i = state["next"]
                if i >= len(dates):
                    return
                state["next"] += 1
            shard = dataset.shard(dates[i])
            item = (dates[i], shard, np.array(shard.load_x()),  # np.array forces the read
                    np.array(shard.load_y()).reshape(-1), np.array(shard.load_w()).reshape(-1))
            with cond:
                ready[i] = item
                cond.notify_all()

    for _ in range(readers):
        threading.Thread(target=reader, daemon=True).start()
    for i in range(len(dates)):
        with cond:
            cond.wait_for(lambda: i in ready)
            item = ready.pop(i)
            state["consumed"] = i + 1
            cond.notify_all()
        yield item


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m experiment.checks.stride_test", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True)
    p.add_argument("--train-weeks", type=int, default=TRAIN_WEEKS)
    p.add_argument("--strides", default=",".join(map(str, STRIDES)),
                   help="descending, each dividing the previous; must end in 1")
    p.add_argument("--readers", type=int, default=4, help="concurrent date reads")
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)

    strides = tuple(int(s) for s in a.strides.split(","))
    if strides[-1] != 1 or any(strides[i] % strides[i + 1] for i in range(len(strides) - 1)):
        raise SystemExit("--strides must be descending, nested, and end in 1")
    t0 = time.time()
    ds = Dataset(a.data)
    anchor, val_dates, groups = choose_anchor(ds)
    train_dates = train_dates_for(anchor, groups, a.train_weeks)
    print(f"anchor {anchor.key}: train {train_dates[0]}..{train_dates[-1]} "
          f"({len(train_dates)} dates), val {val_dates[0]}..{val_dates[-1]} "
          f"({len(val_dates)} dates), strides {strides}, "
          f"BLAS threads {os.environ.get('OMP_NUM_THREADS', 'unset')}", flush=True)

    k = ds.n_features
    cls = minute_class(ds.time_rows, strides)
    XX = np.zeros((len(strides), k, k))
    XY = np.zeros((len(strides), k))
    n_rows = np.zeros(len(strides), dtype=np.int64)
    for j, (d, shard, x3, y, w) in enumerate(prefetch(ds, train_dates, a.readers), 1):
        S = shard.n_symbols
        decay = anchor.decay(d)
        w2 = w.reshape(-1, S)
        for c in range(len(strides)):
            minutes = np.flatnonzero(cls == c)
            m_idx, s_idx = np.nonzero(w2[minutes] > 0)
            rows = minutes[m_idx] * S + s_idx
            n_rows[c] += rows.size
            for s in range(0, rows.size, CHUNK):
                r = rows[s:s + CHUNK]
                accumulate(XX[c], XY[c], _gather_x(x3, r, S), y[r],
                           w[r].astype(np.float64) * decay)
        if j % 10 == 0 or j == len(train_dates):
            print(f"  train {j}/{len(train_dates)}  {time.time() - t0:.0f}s", flush=True)
    t_train = time.time() - t0

    # Stride s = classes 0..i (coarsest first), i.e. cumulative sums.
    B = np.stack([ridge_solve(0.5 * (xx + xx.T), xy)
                  for xx, xy in zip(np.cumsum(XX, 0), np.cumsum(XY, 0))], axis=1)
    rows_per_stride = np.cumsum(n_rows)

    stats = {s: [] for s in strides}
    for j, (d, shard, x3, y, w) in enumerate(prefetch(ds, val_dates, a.readers), 1):
        rows = np.flatnonzero(w > 0)
        pred = np.empty((rows.size, len(strides)))
        for s in range(0, rows.size, CHUNK):
            r = rows[s:s + CHUNK]
            pred[s:s + r.size] = _gather_x(x3, r, shard.n_symbols).astype(np.float64) @ B
        for i, st in enumerate(strides):
            stats[st].append(daily_stats(d, pred[:, i], y[rows], w[rows]))
        if j % 10 == 0 or j == len(val_dates):
            print(f"  val {j}/{len(val_dates)}  {time.time() - t0:.0f}s", flush=True)

    out = {"anchor": anchor.key.isoformat(), "train_weeks": a.train_weeks,
           "train_dates": [train_dates[0].isoformat(), train_dates[-1].isoformat()],
           "val_dates": [val_dates[0].isoformat(), val_dates[-1].isoformat()],
           "seconds": {"train_pass": round(t_train, 1), "total": round(time.time() - t0, 1)},
           "results": {}}
    print(f"\n{'stride':>6} {'train rows':>12} {'APS bps':>8} {'COR %':>7} {'AR':>6} "
          f"{'APS_SR':>7} {'dAPS vs 1':>10} {'diff SR':>8} {'win %':>6}")
    for i, st in enumerate(strides):
        m = aggregate(stats[st])
        dd = paired_diff(stats[st], stats[1])
        out["results"][str(st)] = {"train_rows": int(rows_per_stride[i]), **m,
                                   "vs_stride1": dd,
                                   "daily_APS_bps": [s.aps * 1e4 for s in stats[st]]}
        print(f"{st:>6} {rows_per_stride[i]:>12,} {m['APS_mean_bps']:>8.4f} "
              f"{m['COR_mean'] * 100:>7.3f} {m['AR_mean']:>6.3f} {m['APS_SR']:>7.2f} "
              f"{dd['dAPS_mean_bps']:>+10.4f} {dd['diff_sharpe']:>8.2f} "
              f"{dd['win_rate'] * 100:>6.1f}")
    print(f"\ntotal {out['seconds']['total']:.0f}s (train pass {t_train:.0f}s)")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
