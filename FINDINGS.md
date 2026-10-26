# Findings: what works, what doesn't, and why

A running synthesis of the core980 research, organized by lesson rather than by
run. Each entry states the claim, the evidence behind it, the reason we think it
holds, and how confident we are. It is meant to be read before planning the next
experiment, by a person or by an autoresearch agent.

**Confidence levels.** *High*: holds on both validation windows (or the full
rolling evaluation) with 3 seeds. *Medium*: one window, or both windows with a
small or uncertain effect. *Low*: suggestive only. *Refuted*: looked true on one
window and did not hold on another.

**Numbers.** APS is in bps, mean of daily APS over the scored days. "vs ridge"
is the mean daily APS difference against the contract ridge (lambda = 0.01)
trained on the identical rows; "diff Sharpe" is the annualized Sharpe of that
daily difference.

Last updated: 2026-10-26.

---

## Setup this is based on

- **Experiment split**: one refit of the official 4w schedule instead of 34.
  Training: 52 weeks, every 5th minute (~20M rows). Validation: the 28 days that
  refit predicts, every positive-weight row, scored with the kit's own scorer.
  - Window 1: val 2024-04-09 to 2024-05-06 (anchor 2024-04-14).
  - Window 2: val 2024-01-16 to 2024-02-12 (anchor 2024-01-21), never used to
    choose any setting.
- **Rolling evaluation**: the official protocol (27 refits, 731 days of the
  `is` period, 2022-07-01 to 2024-06-30), run one anchor per parallel job.
- **Standard comparison**: every model trains on identical rows (the last 15
  training days are held out for early stopping by LightGBM/MLP and skipped by
  ridge too), verified by row fingerprints, with 3 seeds for random models.
- The **dev window** (Jul-Dec 2024) has not been predicted or looked at. 2025 is
  the hidden period (no labels).

---

## 1. Evaluation lessons

### 1.1 APS is a jumpy metric, so don't select on small samples of it — *High*
- **Why**: per day, APS = COR x RS (prediction quality x that day's return
  volatility). Daily COR is noisy (signal ~4%, and overlapping 30-minute targets
  mean far fewer independent observations than rows), and multiplying by
  volatility lets a few volatile days dominate any average.
- **Evidence**: one run's daily APS ranged 0.05 to 9.3 bps over 28 days. MLP
  holdout APS (14 days) jumped between epochs, e.g. 0.97 -> 2.30 -> 0.79.
- **Implication**: stop training on a smoother signal (holdout loss) and choose
  between close models on many days, not one 28-day window. Candidate selection
  targets not yet tried: R^2, Sharpe of daily APS, mean daily COR.

### 1.2 Single-window selection overfits the window — *High*
- **Evidence**: several window-1 conclusions reversed on window 2:
  ridge lambda=1 vs 0.01 (+0.20 -> -0.20), linear MLP vs ridge (+0.10 -> -0.15),
  MLP dropout 0.2 vs 0.1 (+0.05 -> -0.03), 4 vs 2 layers (+0.05 -> -0.08),
  2-layer MLP vs ridge (+0.37 -> -0.09).
- **Why**: picking the best of many settings on the same 28 days flatters the
  winner, and different periods favour different amounts of shrinkage.
- **Implication**: a within-window diff Sharpe of +8 can turn into -5 on another
  window; it measures day-to-day consistency inside that period only. Require a
  result to hold on both windows (better: the rolling evaluation).

### 1.3 Compare models only on identical data — *High*
- **Evidence**: on 26 weeks, LightGBM looked far behind ridge (1.17 vs 1.64)
  while it trained on 2M of 11.4M rows. On identical rows it tied (1.70 vs 1.76).
- **Why**: more data helps every model here (see 2.2); a 6x data difference
  dominates any model difference.
- **Open trap**: the rolling MLP result (6.1) compares MLPs on a 10M-row random
  sample with ridge on all ~250M rows, so it is not yet a fair comparison.

### 1.4 Seed noise is large for random models — *High*
- **Evidence**: same settings, different seed: LightGBM 2M rows 1.50 / 1.39 /
  1.39; MLP (before EMA) 1.33 / 1.52 / 1.49. EMA cut MLP seed spread to ~0.05.
- **Implication**: always run 3 seeds; single-run gaps under ~0.15 are not evidence.

---

## 2. Data

### 2.1 Minute stride 5 keeps ~96% of APS with 1/5 of the rows — *High*
- **Evidence** (ridge, 26 weeks): stride 30 / 15 / 5 / 1 -> 1.13 / 1.46 / 1.64 /
  1.71. At 52 and 104 weeks, stride 5 costs 4% and 2%.
- **Why**: 30-minute targets one minute apart share 29 of 30 minutes, so
  neighbouring rows are near-duplicates.

### 2.2 Training history matters more than minute density — *High*
- **Evidence** (ridge, stride 5): 26 / 52 / 104 weeks -> 1.64 / 1.84 / 2.02.
  52 weeks at stride 5 (20M rows, 1.84) beats 26 weeks at every minute (57M
  rows, 1.71). 104 weeks at stride 1 reproduces the official ridge exactly on
  those days (2.053).
- **Implication**: given a data budget, spend it on more weeks, not more minutes.
  104 weeks for the nonlinear models is untested.

---

## 3. MLP

### 3.1 The MLP's early failures were a training problem, not a capacity problem — *High*
- **Evidence**: a linear MLP (no hidden layer) is the same function class as ridge
  but scored 1.63 vs 1.76 with standard training (round 1, 108 runs).
- **Why**: each minibatch update on a ~4% signal is mostly noise; the weights
  jitter around the optimum and early stopping keeps a lucky snapshot.

### 3.2 EMA weight averaging fixes noisy training — *High*
- **What**: keep a running average of the weights (decay 0.9998 per step, about
  one epoch at batch 4096) and evaluate/keep the average.
- **Evidence**: linear MLP 1.63 -> 1.86; every EMA setting beat every non-EMA
  one; seed spread 0.2 -> 0.05; holdout curves became smooth. Seed spread stayed
  small on window 2.
- **Why**: averaging cancels the random jitter of individual updates.

### 3.3 Training settings that help: low learning rate, no input clipping — *High*
- Lower learning rate was better at every setting (best 3e-5 with EMA).
- Clipping standardized inputs at +-5 sigma hurt every time (~0.1-0.3 bps): the
  extreme feature values carry signal.
- Weight decay had no effect at 1e-5..1e-1 (too weak to matter under early
  stopping and EMA). Cosine learning-rate decay: no gain. Stopping on holdout
  loss vs APS: identical once training is smooth.

### 3.4 Two hidden layers beat one; more layers or width don't help — *High* (direction), *Medium* (size)
- **Evidence**: 2 layers (256 -> 128) vs 1 layer: +0.18 (W1), +0.07 (W2),
  pooled +0.13 (diff Sharpe 7.0). Vs linear MLP: +0.27 / +0.06. Rolling
  evaluation (731 days, same data for both): 2-layer 2.41 vs linear 2.34.
- One layer at 16 / 64 / 256 units: all ~1.93 (W1). Six 2-layer sizes from
  64->32 to 512->256: all 2.06-2.14. 3-4 layers: tie on W1, worse on W2.
- **Why (hypothesis)**: a second layer adds "combinations of combinations"
  (interactions), a new kind of relationship; extra width only adds more room to
  fit noise.

### 3.5 Fine-tuning the 2-layer network is flat — *High*
- 54 variants (size, shape, dropout 0-0.5, learning rate, weight decay, depth)
  all within ~+-0.05 on W1; the best-looking ones reversed on W2 (1.2).
  Not worth more tuning until evaluation is on many more days.

---

## 4. LightGBM

### 4.1 Fixing training turns LightGBM from behind ridge to ahead of it, on both windows — *High*
- **Evidence** (all rows, 3 seeds, vs ridge on identical rows; W1 / W2 / pooled):
  default (lr 0.05, stop on APS) -0.10 / -0.13 / -0.12;
  lr 0.02, stop on APS +0.07 / +0.08 / +0.08;
  lr 0.02, stop on loss +0.06 / +0.17 / +0.11;
  **lr 0.01, stop on loss +0.12 / +0.21 / +0.16 (diff Sharpe 3.0)**.
- The first model that beats the contract ridge on both windows.
- **Why**: same lesson as the MLP's EMA: slow, steady learning and a smooth
  stopping signal (holdout loss rather than the jumpy holdout APS). Lower
  learning rates also use many more trees (~800 vs ~110).

### 4.2 The gain comes from feature interactions — *High*
- **Evidence** (lr 0.02, stop on loss; pooled vs ridge): depth-1 trees (no
  interactions) -0.26; depth 2 -0.06; depth 3 +0.10; 15 leaves +0.13; 63 leaves
  +0.11.
- **Why**: one-feature-at-a-time trees are a poor additive model next to ridge;
  the value is in combining features. Gains plateau from ~15 leaves.
- Matches the MLP finding (3.4): the nonlinear benefit is interactions.

### 4.3 "Trees on top of ridge" (init from ridge predictions) did not work as built — *Medium*
- **Evidence**: the trees stopped after 1-2 rounds and added nothing (= ridge).
- **Hypothesis**: starting from ridge, the first trees did not lower the holdout
  loss, so early stopping ended immediately. Not yet investigated; a plain
  average of ridge and LightGBM tests the same idea more directly (see 5).

### 4.4 More rows help LightGBM — *Medium*
- **Evidence** (W1, 1 seed): 2M 1.50 -> 6M 1.65 -> all 19.3M rows 1.70. All rows
  need ~128 GB.

### 4.5 LightGBM tuning has plateaued — *High*
- **Evidence** (round 2, both windows, pooled vs ridge): lr 0.01 + 15 leaves
  +0.15; lr 0.005 + 15 leaves +0.14; lr 0.01 depth 3 +0.09; lr 0.01 + 15 leaves
  + min_data_in_leaf 5000 +0.15. None beats round 1's lr 0.01 + 63 leaves (+0.16).
- **Implication**: like the MLP (3.5), further gains are not in hyperparameters;
  combining models is where the gain is (5).

---

## 5. Combining models

Built from saved predictions with `experiment/combine.py` (`experiments/combos_v1.json`).
"Scale-matched" = each model rescaled to the same prediction RMS before averaging.

### 5.1 A linear model + LightGBM is the best model so far, on both windows — *High* (windows), rolling pending
- **Evidence** (vs ridge on identical rows; W1 / W2 / pooled, days ahead):
  ridge + LightGBM scale-matched 50/50: **+0.33 / +0.39 / +0.36** (diff Sharpe
  12.7, standard error 0.07), ahead on 77% of days; that is +0.20 over the best
  single LightGBM (+0.16). 75/25 ridge-heavy: +0.23 pooled, diff Sharpe 16.1,
  ahead on 84% of days. Every linear + LightGBM combination (ridge or linear MLP,
  with or without the 2-layer MLP) lands at +0.36 to +0.40 pooled.
- **Why**: the two capture different structure (straight-line effects vs feature
  interactions) and their errors only partly overlap, so averaging keeps what each
  gets right and cancels part of each one's noise. Each model alone is noisy,
  which makes the variance reduction large.
- **Caveat**: 12 combinations scored on the windows where the LightGBM settings
  were chosen; but the gain is large and consistent on both windows, far beyond
  what selection alone would explain. Needs the rolling evaluation.

### 5.2 Seed averaging is a small, reliable gain — *High*
- LightGBM's 3 seeds averaged: +0.04 over a single seed on both windows
  (diff Sharpe 29: almost every day). Near-free.

### 5.3 The MLP adds diversity, but window-dependently — *Medium*
- LightGBM + 2-layer MLP: +0.51 (W1) / +0.29 (W2); ridge + MLP: +0.31 / +0.06.
  Strong on window 1, weaker on window 2, like the MLP alone (1.2).

### 5.4 Weighting details matter less than combining the two kinds of model — *Medium*
- Scale-matched vs raw average: +0.36 vs +0.32 pooled (raw leans to ridge's
  larger predictions). 25/75 and 75/25 both strongly positive; 75/25 is steadier
  day to day, 50/50 has the larger mean. Weights were not tuned.

---

## 6. Rolling (official) evaluation

### 6.1 MLP over 731 days: behind the official ridge, but on unequal data — *Medium*
- **Evidence** (`is` period, 27 refits): official ridge 2.59; 2-layer MLP 2.41
  (-0.18, diff Sharpe -4.0, wins 36% of days); linear MLP 2.34 (-0.24).
- **Caveat**: the MLPs train on the kit's 10M-row random sample (~4% of the
  104-week window); ridge streams all rows. On the experiment split, cutting
  ridge to 2M rows cost it 0.25. The fair comparison is ridge on the same 10M
  sample (not yet run).
- **What it does show**: 2-layer beats linear over two years on identical data
  (3.4). Each rolling fit took ~4 h, mostly reading the 10M-row sample.

---

## Open questions

1. Does the MLP beat ridge on equal data over the rolling period (ridge on the
   same 10M-row sample)?
2. Does ridge + LightGBM hold up in the rolling evaluation (731 days)? Needs a
   rolling LightGBM run, then combining with the official ridge predictions.
3. Answered on both windows: yes, linear + LightGBM (5.1). Remaining: does adding
   the MLP help reliably, and do per-instrument weights help?
4. Where do the nonlinear models gain: which instruments, tails vs normal days?
   (post-prediction analysis, not started)
5. Do steadier selection targets (R^2, Sharpe of daily APS) pick better models?
6. Does 104 weeks of training help the nonlinear models more than ridge?

---

## How the experiments are run (for reproduction)

- Build an experiment split: `scripts/experiment_build.sbatch` (or automatically
  via `scripts/submit_experiment.sh`). Second window: `--val-after 2024-01-01`.
- Batches: one line per run in `experiments/*.txt`, submitted as an array job
  with `scripts/submit_experiment.sh`. Results: `results/<split>/<run>/record.json`.
- Compare: `python -m experiment.table --results results/<split> --by-config
  --baseline ridge_eqall` (seeds combined; diff Sharpe vs the baseline).
- Combinations: `experiment/combine.py` with a JSON spec.
- Rolling: `scripts/submit_rolling.sh MODEL OUT` (one job per anchor, then merge
  and kit score against `runs/linear_4w`). Dev is never predicted.
