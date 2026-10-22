"""MLP candidates for the official rolling evaluation (``f522kit run``).

The kit constructs models with no arguments, so each candidate is a class
with its settings fixed here, which also pins them in the run manifest
(the manifest hashes this file; the shared code lives in models/mlp.py).

Settings come from the experiment-split work (see FINDINGS / results):
EMA weight averaging, learning rate 3e-5, constant schedule, early stopping
on holdout loss, no input clipping, batch 4096, patience 5.

Kit-path specifics:
- ``train.sample(row_budget, seed)`` draws rows at random across the full
  104-week window (not by minute stride), so the run's ``--row-budget`` sets
  the data size; ~10M rows keeps memory under 128 GB (the sampler briefly
  holds two copies).
- The sample carries no dates, so the early-stopping holdout is the most
  recent ``es_frac`` of rows (2% ~ the last 14-15 days of 104 weeks).

    f522kit run --data $DATA_ROOT --model models/mlp_final.py:TwoLayer \\
        --out runs/mlp2_4w --stride 4w --row-budget 10000000 --seed 0 --end 2024-06-30
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.mlp import MyModel as _MLP  # noqa: E402

COMMON = {
    "lr": 3e-5, "lr_schedule": "constant", "weight_avg": "ema", "es_metric": "loss",
    "clip_sigma": 1e6, "batch_size": 4096, "weight_decay": 0.0, "max_epochs": 30,
    "patience": 5, "es_frac": 0.02, "in_ram": False, "seed": 0,
}


class TwoLayer(_MLP):
    """256 -> 128, dropout 0.1: best two-layer network; beat one layer on both windows."""

    def __init__(self):
        super().__init__(hidden=(256, 128), dropout=0.1, **COMMON)


class Linear(_MLP):
    """No hidden layer: the linear reference trained the same way."""

    def __init__(self):
        super().__init__(hidden=(), **COMMON)
