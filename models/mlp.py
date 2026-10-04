"""Track B: feed-forward MLP (PyTorch, CPU) on streamed minibatches.

- Inputs: per-feature mean/std from a deterministic subsample of the fit
  rows, then clipped to ±``clip_sigma`` (non-finite -> 0).
- Loss: weighted MSE on y / rms_w(y), weights normalized to mean 1;
  predictions are mapped back to return units.
- Data feeding: X may be a memmap larger than RAM. Each epoch visits
  contiguous chunks of ``chunk_rows`` in random order and shuffles within
  the chunk, so reads stay sequential. ``in_ram=True`` loads the fit rows
  once instead (fastest when they fit).
- Early stopping is in-window, as in models/lgbm.py: pooled weighted APS on
  the most recent ``es_days`` training dates after an ``embargo_days`` gap;
  the best epoch's weights are kept.

Run:
    f522kit run --data $DATA_ROOT --model models/mlp.py:MyModel \
                --out runs/mlp_4w --stride 4w --row-budget 2000000 --seed 0
"""

from __future__ import annotations

import copy
import hashlib
import os

import numpy as np

PREDICT_CHUNK = 262_144


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


def rows_fingerprint(idx):
    """Short hash of the fit-row indices (same as models/ridge.py)."""
    return hashlib.sha1(np.ascontiguousarray(idx, dtype=np.int64).tobytes()).hexdigest()[:12]


def pooled_aps(pred, y, w):
    awy = float(np.dot(w * pred, y))
    awa = float(np.dot(w * pred, pred))
    return awy / np.sqrt(max(awa, 1e-24)) / np.sqrt(max(float(w.sum()), 1e-24))


class MyModel:
    def __init__(self, hidden=(256, 128), dropout=0.1, lr=1e-3, weight_decay=1e-5,
                 batch_size=4096, max_epochs=20, patience=3, es_days=14,
                 embargo_days=1, es_frac=0.1, max_rows=0, chunk_rows=262_144,
                 in_ram=True, clip_sigma=5.0, stats_rows=500_000, threads=0, seed=0):
        self.hidden = tuple(int(h) for h in hidden)
        self.dropout = float(dropout)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.batch_size = int(batch_size)
        self.max_epochs = int(max_epochs)
        self.patience = int(patience)
        self.es_days = int(es_days)
        self.embargo_days = int(embargo_days)
        self.es_frac = float(es_frac)
        self.max_rows = int(max_rows)
        self.chunk_rows = int(chunk_rows)
        self.in_ram = bool(in_ram)
        self.clip_sigma = float(clip_sigma)
        self.stats_rows = int(stats_rows)
        self.threads = int(threads) or int(os.environ.get("SLURM_CPUS_PER_TASK", 0)) \
            or (os.cpu_count() or 1)
        self.seed = int(seed)

    # -- kit path -------------------------------------------------------------
    def fit(self, train):
        X, y, w = train.sample(train.row_budget, seed=self.seed)
        self.fit_arrays(X, y, w, date_idx=None)

    # -- shared ---------------------------------------------------------------
    def _transform(self, x):
        z = (np.asarray(x, dtype=np.float32) - self.mu) / self.sd
        np.nan_to_num(z, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        np.clip(z, -self.clip_sigma, self.clip_sigma, out=z)
        return z

    def _net(self, k):
        import torch.nn as nn

        layers, d = [], k
        for h in self.hidden:
            layers += [nn.Linear(d, h), nn.SiLU(), nn.Dropout(self.dropout)]
            d = h
        layers.append(nn.Linear(d, 1))
        return nn.Sequential(*layers)

    def fit_arrays(self, X, y, w, date_idx=None):
        import torch

        torch.set_num_threads(self.threads)
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)

        n, k = X.shape
        fit_end, hold_start = holdout_split(
            n, date_idx, self.es_days, self.embargo_days, self.es_frac)
        y = np.asarray(y, dtype=np.float64)
        w = np.asarray(w, dtype=np.float64)
        self.y_scale = float(np.sqrt(np.sum(w[:fit_end] * y[:fit_end] ** 2)
                                     / np.sum(w[:fit_end])))
        w_scale = float(w[:fit_end].mean())

        fit_idx = np.arange(fit_end)
        if self.max_rows and fit_end > self.max_rows:
            fit_idx = np.sort(rng.choice(fit_end, size=self.max_rows, replace=False))

        stats_idx = fit_idx if fit_idx.size <= self.stats_rows else np.sort(
            rng.choice(fit_idx, size=self.stats_rows, replace=False))
        xs = np.asarray(X[stats_idx], dtype=np.float64)
        xs[~np.isfinite(xs)] = np.nan
        self.mu = np.nan_to_num(np.nanmean(xs, axis=0)).astype(np.float32)
        sd = np.nan_to_num(np.nanstd(xs, axis=0))
        self.sd = np.where(sd > 1e-12, sd, 1.0).astype(np.float32)
        del xs

        Xfit = self._transform(X[fit_idx]) if self.in_ram else None
        yt = (y[fit_idx] / self.y_scale).astype(np.float32)
        wt = (w[fit_idx] / w_scale).astype(np.float32)
        Xh = torch.from_numpy(self._transform(X[hold_start:]))
        yh, wh = y[hold_start:], w[hold_start:]

        net = self._net(k)
        opt = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        best, best_state, bad, history = -np.inf, None, 0, []
        n_fit = fit_idx.size
        chunks = np.arange(0, n_fit, self.chunk_rows)
        for epoch in range(self.max_epochs):
            net.train()
            loss_sum = 0.0
            for c in rng.permutation(chunks):
                sl = slice(c, min(c + self.chunk_rows, n_fit))
                xc = Xfit[sl] if Xfit is not None else self._transform(X[fit_idx[sl]])
                perm = rng.permutation(sl.stop - sl.start)
                xb_all = torch.from_numpy(np.ascontiguousarray(xc[perm]))
                yb_all = torch.from_numpy(yt[sl][perm])
                wb_all = torch.from_numpy(wt[sl][perm])
                for b in range(0, perm.size, self.batch_size):
                    xb = xb_all[b:b + self.batch_size]
                    yb = yb_all[b:b + self.batch_size]
                    wb = wb_all[b:b + self.batch_size]
                    loss = (wb * (net(xb).squeeze(1) - yb) ** 2).sum() / wb.sum()
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()
                    loss_sum += loss.item() * xb.shape[0]
            aps = pooled_aps(self._forward(net, Xh), yh, wh)
            history.append({"epoch": epoch, "train_loss": loss_sum / n_fit,
                            "holdout_aps_bps": aps * 1e4})
            if aps > best:
                best, best_state, bad = aps, copy.deepcopy(net.state_dict()), 0
            else:
                bad += 1
                if bad >= self.patience:
                    break
        net.load_state_dict(best_state)
        self.net = net.eval()
        return {"n_fit_rows": int(n_fit), "fit_rows_sha": rows_fingerprint(fit_idx),
                "n_holdout_rows": int(n - hold_start),
                "best_epoch": int(np.argmax([h["holdout_aps_bps"] for h in history])),
                "holdout_aps_bps": best * 1e4, "history": history,
                "y_scale": self.y_scale}

    @staticmethod
    def _forward(net, xt):
        import torch

        net.eval()
        out = []
        with torch.no_grad():
            for s in range(0, xt.shape[0], PREDICT_CHUNK):
                out.append(net(xt[s:s + PREDICT_CHUNK]).squeeze(1).numpy())
        return np.concatenate(out).astype(np.float64)

    def predict(self, X):
        import torch

        torch.set_num_threads(self.threads)
        out = np.empty(X.shape[0], dtype=np.float32)
        for s in range(0, X.shape[0], PREDICT_CHUNK):
            xt = torch.from_numpy(self._transform(X[s:s + PREDICT_CHUNK]))
            out[s:s + PREDICT_CHUNK] = self._forward(self.net, xt) * self.y_scale
        return out
