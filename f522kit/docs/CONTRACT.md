# Evaluation Contract — anonymous student tensor datasets

This document is the normative specification. The kit implements it; if kit
behavior and this document ever disagree, that is a bug — report it.

Everything dataset-specific — the feature count `K`, the schema version,
and the model-assignment rule — comes from `dataset.json`. The core980
assignment uses `K = 980`, schema 2, and `date_mature`. The reader also
supports schema 1, which is pinned to `previous_group`.

## 1. Dataset schema

One immutable directory per trading date:

```
<root>/
  dataset.json            # identity, dtypes, date range, hidden boundary
  manifest.json           # per-date inventory + hashes
  feature_ids.npy         # x000..x(K−1) feature identifiers
  dates/YYYY-MM-DD/
    X.npy                 # float32 [1440, S_d, K]     anonymized features
    episode_id.npy        # int32   [S_d]              listing-episode ids
    y.npy                 # float32 [1440, S_d]        visible dates only
    w.npy                 # float32 [1440, S_d]        visible dates only
    meta.json             # shapes, dtypes, sha256 per array
```

Semantics you must not re-derive or second-guess:

- **`y` is final.** The 30-minute forward target is already aligned, clipped
  to `±0.03·√30`, and zero-filled where the source value was missing.
- **`w` is final.** It is the effective evaluation weight: the date-aligned
  base weight with exactly `0` wherever the target was missing or the weight
  was non-finite. A zero-weight row contributes nothing to fitting or
  scoring; do not invent your own validity mask.
- **`episode_id` is for alignment/grouping only** (e.g., grouped
  cross-validation). Pending policy TBD-4 it must not be used as a model
  input feature.
- Features are anonymous by design. Attempting to de-anonymize them is out
  of contract.
- The feature set was selected using outcomes that include the hidden
  evaluation period. Withholding its labels does not make that period an
  independent holdout for feature selection. Use it for comparison with
  the matched baseline; do not interpret absolute performance as unbiased
  evidence of performance on new data.

## 2. Calendar and rolling protocol

- Dates group into **weekly model groups** keyed by the ISO week-end Sunday
  (`key(d) = d + (6 − weekday(d)) mod 7` days).
- **Model assignment is dataset-selected** (`dataset.json`; schema-1
  datasets are pinned to `previous_group`, schema-2 datasets name their
  rule):
  - `previous_group`: dates in weekly group `G_i` are scored with the model
    anchored at the previous group key `G_{i−1}`.
  - `date_mature`: a date uses the newest model whose entire training
    window's h30 label endpoints mature strictly before the date's 00:00 —
    Tuesday..Sunday of group `G_i` use the model anchored at `G_i` (trained
    through Sunday `G_{i−1}`), Mondays fall back to `G_{i−1}`.
  Under both rules the first two weekly groups of the dataset are never
  scored.
- The model anchored at `G_p` trains on at most **104** weekly groups
  strictly before `G_p`.
- **Training decay**: each training date `d` carries the factor
  `0.5^((anchor − d) / 1460)` (days), multiplied into `w`.
- **Fit exclusions**: the frozen date list in `f522kit.contract`
  (`EXCLUDED_FIT_DATES`) covers the released labeled period and is removed
  from every student training window. Excluded dates are still predicted
  and scored. The public training API uses released visible labels only.
- **Refit stride**: development runs refit every 4th valid weekly anchor
  (`4w`), promotion every 2nd (`2w`), final audits weekly (`1w`). Between
  anchors, the most recent anchored model applies. Stride only thins the
  refit points; it never changes group membership or the assignment rule.
- **No-lookahead**: your `fit` may consume only the `TrainWindow`'s public
  API. Preprocessing statistics, hyperparameter choices, early stopping,
  and any calibration must be computed within that window. Bypassing the
  API (private attributes, direct dataset paths, state cached across
  anchors) is a protocol violation: submitted code is reviewed, and a
  violation voids the run.

## 3. Reference linear baseline (what "reproduce the baseline" means)

The published baseline is a weighted ridge regression, no intercept, no
centering, computed in float64 from sufficient statistics over the full
training window (all positive-weight rows, no subsampling):

```
XX = Σ_rows  w_train · x xᵀ          (K × K)
XY = Σ_rows  w_train · x y           (K)
w_train = w · 0.5^((anchor − date)/1460)
```

The solve is **normalized ridge** — the penalty is applied in the scaled
feature space, which is *not* the same as naive `(XX + λI)⁻¹XY`:

```
d_i    = sqrt(max(XX_ii, 0)), floored at 1e-12
XXn    = XX_ij / (d_i d_j)
XYn    = XY_i / d_i
solve    (XXn + λ I) βn = XYn        with λ = 0.01
β_i    = βn_i / d_i
```

Equivalently: `(XX + λ·diag(XX)) β = XY`. Predictions are `X β`, submitted
raw (the evaluator applies calibration and the clip). Matching the published
reference metrics within the stated tolerance through your own adapter is
the entry gate for model work.

## 4. Evaluation pipeline (frozen)

Per scored date, in order: **calibrate → clip → sufficient stats**.

1. Calibration: `identity` until TBD-1 is frozen (scores are used as
   return-unit alphas exactly as submitted).
2. Clip: `a ← clip(a, ±0.02·√30)`.
3. Daily sufficient statistics over every row of the date:

```
awa = Σ w a²      awy = Σ w a y      ywy = Σ w y²      sum_w = Σ w
```

4. Daily metrics (ε = 1e-12 guards):

```
COR = awy / sqrt(awa · ywy)          weighted uncentered cosine — NOT Pearson
AR  = awy / awa
APS = awy / sqrt(awa) / sqrt(sum_w)  reported in bps (×10⁴)
```

5. Window aggregates are **mean-daily** (simple mean of the daily series).
   The stability statistic is `APS_SR = mean/std(ddof=1) · √365` of daily
   APS. Model-versus-baseline comparison uses the daily APS difference
   series and its annualized Sharpe over common dates.
   The only valid comparison target is the reference baseline scored by
   this evaluator on this dataset — numbers from any other source or
   convention are not comparable and must not be used as benchmarks.
6. Scale-awareness: COR and unclipped APS are invariant to positive
   rescaling of your predictions; AR is not, and APS becomes
   scale-sensitive once the clip binds. Keep predictions in return units.

## 5. Prediction artifact schema

```
<out>/
  run_manifest.json       # kit version, dataset id, model spec, stride,
                          # row budget, seed, per-anchor fit records,
                          # per-date sha256
  preds/YYYY-MM-DD.npy    # float32 [1440, S_d] raw scores
```

Every predictable date in a scored window must be present — partial
coverage fails scoring rather than shrinking it. Runs must be
deterministic given (model code, stride, row budget, seed).

## 6. Open policy boxes (TBD)

| Box | What | Interim behavior |
|---|---|---|
| TBD-1 | Causal calibration rule + promotion threshold | `identity` calibration |
| TBD-2 | Hidden-score query budget + diagnostics | hidden scoring not yet open |
| TBD-3 | Fair-compute training-row budget | `--row-budget` required per run instructions |
| TBD-4 | `episode_id` as model input | grouping/alignment only |
| TBD-5 | Reference metrics + reproduction tolerance | published separately per dataset once the tolerance is frozen |
