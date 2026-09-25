"""Reference linear baseline: normalized weighted ridge (docs/CONTRACT.md §3).

Streams every positive-weight row of the training window via
``train.iter_dates()`` (no subsampling), accumulates float64 sufficient
statistics, and solves the normalized ridge system:

    XX = Σ w_train · x xᵀ,   XY = Σ w_train · x y
    d_i = max(sqrt(max(XX_ii, 0)), 1e-12)
    (XX / d dᵀ + λ I) βn = XY / d,   β = βn / d

``w_train`` from the kit already includes the half-life decay and zero-weight
rows are already dropped, so no extra masking or decay is applied here.

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


class MyModel:
    def fit(self, train):
        XX = XY = None
        for block in train.iter_dates():
            if XX is None:
                k = block.x.shape[1]
                XX = np.zeros((k, k))
                XY = np.zeros(k)
            for s in range(0, block.y.shape[0], CHUNK_ROWS):
                x = np.asarray(block.x[s:s + CHUNK_ROWS], dtype=np.float64)
                y = np.asarray(block.y[s:s + CHUNK_ROWS], dtype=np.float64)
                xw = x * block.w_train[s:s + CHUNK_ROWS, None]
                XX += xw.T @ x
                XY += xw.T @ y
        if XX is None:
            raise ValueError("Training window has no positive-weight rows")
        XX = 0.5 * (XX + XX.T)  # remove float round-off asymmetry
        self.beta = ridge_solve(XX, XY)

    def predict(self, X):
        out = np.empty(X.shape[0], dtype=np.float32)
        for s in range(0, X.shape[0], CHUNK_ROWS):
            chunk = np.asarray(X[s:s + CHUNK_ROWS], dtype=np.float64)
            out[s:s + CHUNK_ROWS] = chunk @ self.beta
        return out
