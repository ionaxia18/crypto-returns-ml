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


class MyModel:
    def __init__(self, lam: float = LAMBDA):
        self.lam = lam

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
        k = X.shape[1]
        XX = np.zeros((k, k))
        XY = np.zeros(k)
        accumulate(XX, XY, X, y, np.asarray(w, dtype=np.float64))
        self._solve(XX, XY)
        return {"n_rows": int(y.shape[0])}

    def _solve(self, XX, XY):
        XX = 0.5 * (XX + XX.T)  # remove float round-off asymmetry
        self.beta = ridge_solve(XX, XY, self.lam)

    def predict(self, X):
        out = np.empty(X.shape[0], dtype=np.float32)
        for s in range(0, X.shape[0], CHUNK_ROWS):
            chunk = np.asarray(X[s:s + CHUNK_ROWS], dtype=np.float64)
            out[s:s + CHUNK_ROWS] = chunk @ self.beta
        return out
