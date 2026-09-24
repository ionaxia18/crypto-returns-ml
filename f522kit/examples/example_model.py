"""Minimal adapter example: a (bad) constant-scale momentum-free model.

This exists to show the API shape only — it carries no signal and will
score near zero. Do not use it as a modeling starting point; implement the
ridge baseline from docs/CONTRACT.md section 3 first.

Run:
    f522kit run --data $DATA_ROOT --model examples/example_model.py:MeanModel \
                --out runs/example --stride 4w --row-budget 1000000 --seed 0
"""

import numpy as np


class MeanModel:
    """Predicts the training sample's weighted mean target for every row."""

    def fit(self, train):
        X, y, w = train.sample(100_000, seed=0)
        self.mu = float(np.average(y, weights=w))

    def predict(self, X):
        return np.full(X.shape[0], self.mu, dtype=np.float32)
