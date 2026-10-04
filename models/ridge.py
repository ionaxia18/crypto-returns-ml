"""Reference linear baseline: normalized weighted ridge (docs/CONTRACT.md §3).

Streams every positive-weight row of the training window via
``train.iter_dates()`` (no subsampling), accumulates float64 sufficient
statistics, and solves the normalized ridge system:

    XX = Σ w_train · x xᵀ,   XY = Σ w_train · x y
    d_i = max(sqrt(max(XX_ii, 0)), 1e-12)
    (XX / d dᵀ + λ I) βn = XY / d,   β = βn / d

``w_train`` from the kit already includes the half-life decay and zero-weight
rows are already dropped, so no extra masking or decay is applied here.

``fit_arrays`` is the same estimator on in-memory/memmapped arrays, used by
the private experiment-split harness (``experiment``).

Run:
    f522kit run --data $DATA_ROOT --model models/ridge.py:MyModel \
                --out runs/linear_4w --stride 4w --row-budget 1000000 --seed 0
"""

import hashlib

import numpy as np

LAMBDA = 0.01      # contract.RIDGE_LAMBDA
D_FLOOR = 1e-12
CHUNK_ROWS = 65_536  # bounds the float64 copy per chunk (~0.5 GB at K=980)


def ridge_solve(XX: np.ndarray, XY: np.ndarray, lam: float = LAMBDA) -> np.ndarray:
    """Normalized ridge solve from sufficient statistics (CONTRACT §3)."""
    d = np.maximum(np.sqrt(np.maximum(np.diag(XX), 0.0)), D_FLOOR)
    XXn = XX / np.outer(d, d)
    XYn = XY / d
    beta_n = np.linalg.solve(XXn + lam * np.eye(len(d)), XYn)
    return beta_n / d


def accumulate(XX: np.ndarray, XY: np.ndarray, X, y, w) -> None:
    """Add one block's weighted sufficient statistics in place (float64)."""
    for s in range(0, y.shape[0], CHUNK_ROWS):
        x = np.asarray(X[s:s + CHUNK_ROWS], dtype=np.float64)
        yc = np.asarray(y[s:s + CHUNK_ROWS], dtype=np.float64)
        xw = x * w[s:s + CHUNK_ROWS, None]
        XX += xw.T @ x
        XY += xw.T @ yc


def holdout_split(n, date_idx, es_days, embargo_days, es_frac):
    """(fit_end, hold_start); identical to models/lgbm.py (kept self-contained
    because the kit hashes only the model spec file)."""
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


def rows_fingerprint(idx: np.ndarray) -> str:
    """Short hash of the fit-row indices, to show runs trained on identical rows."""
    return hashlib.sha1(np.ascontiguousarray(idx, dtype=np.int64).tobytes()).hexdigest()[:12]


class MyModel:
    """Contract ridge. The optional ``max_rows`` / ``es_days`` (both off by
    default) make ``fit_arrays`` train on exactly the rows models/lgbm.py and
    models/mlp.py train on for the same settings and seed: drop the last
    ``es_days`` dates plus ``embargo_days`` (their early-stopping holdout),
    then the same seeded uniform subsample. Ridge does no early stopping; the
    held-out dates are simply unused. For like-for-like comparisons only.
    """

    def __init__(self, lam: float = LAMBDA, max_rows: int = 0, es_days: int = 0,
                 embargo_days: int = 1, seed: int = 0):
        self.lam = lam
        self.max_rows = int(max_rows)
        self.es_days = int(es_days)
        self.embargo_days = int(embargo_days)
        self.seed = int(seed)

    def fit(self, train):
        XX = XY = None
        for block in train.iter_dates():
            if XX is None:
                k = block.x.shape[1]
                XX = np.zeros((k, k))
                XY = np.zeros(k)
            accumulate(XX, XY, block.x, block.y, block.w_train)
        if XX is None:
            raise ValueError("Training window has no positive-weight rows")
        self._solve(XX, XY)

    def fit_arrays(self, X, y, w, date_idx=None):
        """Experiment-split path: X may be a memmap; streamed in CHUNK_ROWS blocks."""
        n, k = X.shape
        w = np.asarray(w, dtype=np.float64)
        fit_end = n
        if self.es_days:
            if date_idx is None:
                raise ValueError("es_days needs date_idx")
            fit_end, _ = holdout_split(n, date_idx, self.es_days, self.embargo_days, 0.0)
        XX = np.zeros((k, k))
        XY = np.zeros(k)
        if self.max_rows and fit_end > self.max_rows:
            rng = np.random.default_rng(self.seed)  # same draw as lgbm.py / mlp.py
            idx = np.sort(rng.choice(fit_end, size=self.max_rows, replace=False))
            for s in range(0, idx.size, CHUNK_ROWS):
                r = idx[s:s + CHUNK_ROWS]
                accumulate(XX, XY, X[r], y[r], w[r])
        else:
            idx = np.arange(fit_end)
            accumulate(XX, XY, X[:fit_end], y[:fit_end], w[:fit_end])
        self._solve(XX, XY)
        return {"n_fit_rows": int(idx.size), "fit_rows_sha": rows_fingerprint(idx)}

    def _solve(self, XX, XY):
        XX = 0.5 * (XX + XX.T)  # remove float round-off asymmetry
        self.beta = ridge_solve(XX, XY, self.lam)

    def predict(self, X):
        out = np.empty(X.shape[0], dtype=np.float32)
        for s in range(0, X.shape[0], CHUNK_ROWS):
            chunk = np.asarray(X[s:s + CHUNK_ROWS], dtype=np.float64)
            out[s:s + CHUNK_ROWS] = chunk @ self.beta
        return out
