"""LightGBM candidate for the official rolling evaluation (``f522kit run``).

Settings from LightGBM rounds 1-2 on both validation windows (see FINDINGS.md
4.1-4.5): learning rate 0.01, early stopping on holdout loss (not the jumpy
APS), 63 leaves and the other models/lgbm.py defaults. Best single LightGBM:
+0.16 vs ridge pooled over both windows; its main use is combined with ridge
(+0.36, FINDINGS.md 5.1).

The kit constructs models with no arguments, so the settings are fixed here,
which also pins them in the run manifest (shared code: models/lgbm.py).

Kit-path specifics (as models/mlp_final.py):
- ``train.sample(row_budget, seed)`` draws rows at random across the 104-week
  window; every sampled row is used (max_rows 0). ~10M rows needs ~100 GB at
  peak (the sampler briefly holds two copies), within 128 GB.
- The sample carries no dates, so the early-stopping holdout is the most recent
  2% of rows (~ the last 14-15 days).

    scripts/submit_rolling.sh models/lgbm_final.py:Best runs/lgbm_4w --time=10:00:00
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.lgbm import MyModel as _LGBM  # noqa: E402


class Best(_LGBM):
    """lr 0.01, stop on holdout loss, 63 leaves, all sampled rows, seed 0."""

    def __init__(self):
        super().__init__(max_rows=0, num_boost_round=20000, early_stopping_rounds=200,
                         es_frac=0.02, seed=0, learning_rate=0.01, es_metric="loss")
