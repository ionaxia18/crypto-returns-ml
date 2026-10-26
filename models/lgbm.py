"""Track A: LightGBM regressor on the weighted target.

- Target is fit in standardized units (y / rms_w(y)) with weights normalized
  to mean 1, and predictions are mapped back to return units.
- Early stopping is in-window: the most recent ``es_days`` training dates
  (after an ``embargo_days`` gap, since h30 labels overlap) are held out and
  scored with pooled weighted APS. Nothing outside the training window is
  ever consulted.
- Memory: LightGBM bins the whole fit matrix in RAM (~1 byte/value at
  max_bin <= 255). By default (``max_rows=0``) every fit row is used, the
  same rows the MLP and ridge with ``es_days=14`` train on; a memmapped
  prefix is passed through without a float copy (52w split: ~19M rows,
  needs ~128 GB). ``max_rows`` > 0 caps the fit rows with a deterministic
  uniform subsample.

- ``es_metric``: early stopping on pooled holdout APS (``"aps"``, default) or
  on holdout weighted squared error (``"loss"``), the steadier signal: daily
  APS = COR x return size, so a few volatile holdout days dominate it.
- ``base="ridge"``: fit the contract-style normalized ridge (lambda
  ``base_lam``) on the same fit rows first and start boosting from its
  predictions (LightGBM ``init_score``), so the trees learn only the
  nonlinear corrections. Prediction = ridge + trees. Default ``"none"``.

Kit path (``fit(train)``) draws ``train.sample(train.row_budget, seed)``;
that sample is date-ordered but carries no date labels, so the holdout is
the trailing ``es_frac`` of rows instead of whole dates.

Run:
    f522kit run --data $DATA_ROOT --model models/lgbm.py:MyModel \
                --out runs/lgbm_4w --stride 4w --row-budget 2000000 --seed 0
"""

from __future__ import annotations

import hashlib
import os

import numpy as np

PARAMS = {
    "objective": "regression",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 2000,
    "min_sum_hessian_in_leaf": 0.0,
    "feature_fraction": 0.5,
    "bagging_fraction": 0.5,
    "bagging_freq": 1,
    "lambda_l2": 10.0,
    "max_bin": 63,
    "verbose": -1,
}
PREDICT_CHUNK = 262_144
RIDGE_CHUNK = 65_536


def ridge_beta(X, y, w, rows, lam):
    """Normalized weighted ridge (as models/ridge.py) on ``rows`` of X, in float64."""
    k = X.shape[1]
    XX, XY = np.zeros((k, k)), np.zeros(k)
    for s in range(0, rows.size, RIDGE_CHUNK):
        r = rows[s:s + RIDGE_CHUNK]
        x = np.asarray(X[r], dtype=np.float64)
        xw = x * w[r, None]
        XX += xw.T @ x
        XY += xw.T @ y[r]
    XX = 0.5 * (XX + XX.T)
    d = np.maximum(np.sqrt(np.maximum(np.diag(XX), 0.0)), 1e-12)
    beta_n = np.linalg.solve(XX / np.outer(d, d) + lam * np.eye(k), XY / d)
    return beta_n / d


def linear_pred(X, beta, start=0, stop=None):
    stop = X.shape[0] if stop is None else stop
    out = np.empty(stop - start)
    for s in range(start, stop, PREDICT_CHUNK):
        e = min(s + PREDICT_CHUNK, stop)
        out[s - start:e - start] = np.asarray(X[s:e], dtype=np.float64) @ beta
    return out


def holdout_split(n, date_idx, es_days, embargo_days, es_frac):
    """(fit_end, hold_start): rows [0, fit_end) fit, [hold_start, n) hold out.

    Rows must be in chronological order (true for both the experiment split and
    ``train.sample``).
    """
    if date_idx is not None:
        last = int(date_idx[-1])
        hold_first = last - es_days + 1
        fit_last = hold_first - embargo_days - 1
        hold_start = int(np.searchsorted(date_idx, hold_first, side="left"))
        fit_end = int(np.searchsorted(date_idx, fit_last, side="right"))
    else:
        hold_start = int(n * (1.0 - es_frac))
        fit_end = hold_start
    if fit_end <= 0 or hold_start >= n:
        raise ValueError("training window too short for the early-stopping holdout")
    return fit_end, hold_start


def rows_fingerprint(idx):
    """Short hash of the fit-row indices (same as models/ridge.py)."""
    return hashlib.sha1(np.ascontiguousarray(idx, dtype=np.int64).tobytes()).hexdigest()[:12]


def pooled_aps(pred, y, w):
    awy = float(np.dot(w * pred, y))
    awa = float(np.dot(w * pred, pred))
    return awy / np.sqrt(max(awa, 1e-24)) / np.sqrt(max(float(w.sum()), 1e-24))


class MyModel:
    def __init__(self, max_rows=0, num_boost_round=2000,
                 early_stopping_rounds=100, es_days=14, embargo_days=1,
                 es_frac=0.1, threads=0, seed=0, es_metric="aps", base="none",
                 base_lam=0.01, log_every=0, **params):
        self.max_rows = int(max_rows)
        self.num_boost_round = int(num_boost_round)
        self.early_stopping_rounds = int(early_stopping_rounds)
        self.es_days = int(es_days)
        self.embargo_days = int(embargo_days)
        self.es_frac = float(es_frac)
        self.threads = int(threads) or int(os.environ.get("SLURM_CPUS_PER_TASK", 0)) \
            or (os.cpu_count() or 1)
        self.seed = int(seed)
        if es_metric not in ("aps", "loss"):
            raise ValueError(f"es_metric must be aps|loss, got {es_metric!r}")
        if base not in ("none", "ridge"):
            raise ValueError(f"base must be none|ridge, got {base!r}")
        self.es_metric = es_metric
        self.base = base
        self.base_lam = float(base_lam)
        self.log_every = int(log_every)  # print holdout score every N trees (0 = silent)
        self.params = {**PARAMS, **params}

    def fit(self, train):
        X, y, w = train.sample(train.row_budget, seed=self.seed)
        self.fit_arrays(X, y, w, date_idx=None)

    def fit_arrays(self, X, y, w, date_idx=None):
        import lightgbm as lgb

        n = y.shape[0]
        fit_end, hold_start = holdout_split(
            n, date_idx, self.es_days, self.embargo_days, self.es_frac)
        y = np.asarray(y, dtype=np.float64)
        w = np.asarray(w, dtype=np.float64)
        self.y_scale = float(np.sqrt(np.sum(w[:fit_end] * y[:fit_end] ** 2)
                                     / np.sum(w[:fit_end])))
        w_scale = float(w[:fit_end].mean())

        if self.max_rows and fit_end > self.max_rows:
            rng = np.random.default_rng(self.seed)
            idx = np.sort(rng.choice(fit_end, size=self.max_rows, replace=False))
            Xf = np.asarray(X[idx], dtype=np.float32)
        else:
            idx = np.arange(fit_end)
            Xf = X[:fit_end]  # contiguous memmap slice: no float copy
        Xh = np.asarray(X[hold_start:], dtype=np.float32)
        yh, wh = y[hold_start:], w[hold_start:]

        # Optional linear base: trees start from ridge's predictions (scaled units).
        self.beta = None
        init_fit = init_hold = None
        if self.base == "ridge":
            self.beta = ridge_beta(X, y, w, idx, self.base_lam)
            init_fit = linear_pred(Xf, self.beta) / self.y_scale
            init_hold = linear_pred(Xh, self.beta) / self.y_scale

        params = {**self.params, "num_threads": self.threads, "seed": self.seed,
                  "deterministic": True, "force_row_wise": True,
                  "metric": "None" if self.es_metric == "aps" else "l2"}
        dtrain = lgb.Dataset(Xf, label=y[idx] / self.y_scale, weight=w[idx] / w_scale,
                             init_score=init_fit,
                             params={"max_bin": params["max_bin"]}, free_raw_data=True)
        dhold = lgb.Dataset(Xh, label=yh / self.y_scale, weight=wh / w_scale,
                            init_score=init_hold, reference=dtrain)

        def aps_eval(preds, data):  # preds include init_score on the holdout
            return "aps", pooled_aps(preds, yh, wh), True

        self.booster = lgb.train(
            params, dtrain, num_boost_round=self.num_boost_round,
            valid_sets=[dhold], valid_names=["holdout"],
            feval=aps_eval if self.es_metric == "aps" else None,
            callbacks=[lgb.early_stopping(self.early_stopping_rounds, verbose=False)]
            + ([lgb.log_evaluation(self.log_every)] if self.log_every else []),
        )
        self.best_iteration = int(self.booster.best_iteration or self.num_boost_round)
        best = self.booster.best_score["holdout"]
        hold_pred = self.booster.predict(Xh, num_iteration=self.best_iteration,
                                         num_threads=self.threads)
        if init_hold is not None:
            hold_pred = hold_pred + init_hold
        return {
            "n_fit_rows": int(idx.size),
            "fit_rows_sha": rows_fingerprint(idx),
            "n_holdout_rows": int(n - hold_start),
            "best_iteration": self.best_iteration,
            "es_metric": self.es_metric,
            "base": self.base,
            "holdout_aps_bps": pooled_aps(hold_pred, yh, wh) * 1e4,
            "holdout_loss": float(best["l2"]) if "l2" in best else None,
            "y_scale": self.y_scale,
        }

    def predict(self, X):
        out = np.empty(X.shape[0], dtype=np.float32)
        for s in range(0, X.shape[0], PREDICT_CHUNK):
            chunk = np.asarray(X[s:s + PREDICT_CHUNK], dtype=np.float32)
            trees = self.booster.predict(
                chunk, num_iteration=self.best_iteration, num_threads=self.threads
            ) * self.y_scale
            if self.beta is not None:
                trees = trees + chunk.astype(np.float64) @ self.beta
            out[s:s + PREDICT_CHUNK] = trees
        return out
