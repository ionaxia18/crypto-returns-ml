# f522kit — nonlinear modeling on anonymous core980 features

Your task is to beat a matched linear baseline using nonlinear models on
980 anonymous features. The data location is supplied with your assignment.
The kit reads the feature count, schema, and model-assignment rule from
`dataset.json`; this assignment uses schema 2 and `date_mature`.

Older kit versions refuse to open a schema-2 root; that is deliberate —
upgrade rather than work around it. `K` below always means the feature
count of the dataset you were given.

You implement **one class with two methods**; the kit owns everything else —
data reading, the rolling calendar, no-lookahead masking, training decay,
prediction persistence, and the canonical evaluator.

```python
# my_model.py
class MyModel:
    def fit(self, train):
        # train: f522kit TrainWindow
        #   train.sample(n_rows, seed) -> (X [n,K] f32, y [n] f32, w [n] f64)
        #   train.iter_dates()         -> per-date streaming blocks
        # w already includes the frozen half-life decay. Zero-weight rows
        # are pre-dropped.
        ...

    def predict(self, X):
        # X: float32 [n, K] — return one finite score per row,
        # in return units.
        ...
```

## Quickstart

```bash
pip install -e .            # numpy only; add .[lgbm] or .[torch] as needed
f522kit smoke               # end-to-end self-test on a synthetic fixture
f522kit verify-data --data $DATA_ROOT --sample 20
f522kit anchors --data $DATA_ROOT --stride 4w
f522kit run   --data $DATA_ROOT --model my_model.py:MyModel \
              --out runs/exp01 --stride 4w --row-budget <budget> --seed 0
f522kit score --data $DATA_ROOT --preds runs/exp01 \
              --baseline runs/linear_ref --out runs/exp01/report.json
```

## Roadmap

The stages below are **directions, not scripts** — deciding how to get
through each one is part of the research. Every stage runs under the same
contract rules.

1. **Reproduce the linear baseline.** Implement the ridge specification in
   [docs/CONTRACT.md](docs/CONTRACT.md) §3 inside your own adapter and match
   the reference metrics published for your dataset (distributed with your
   assignment) within the stated tolerance. This proves your pipeline
   before any model work counts.
2. **Work out subsampling before scaling up.** The dataset is multi-TB
   (see *Data scale* below), so tune your pipeline and first models on a
   subsample you choose and can justify: row sampling via
   `train.sample(n_rows, seed)`, time downsampling through
   `train.iter_dates()` (e.g. keeping one row every few minutes), fewer
   training dates, or a scheme of your own. Then **measure what your
   subsample costs in metric terms** before trusting conclusions drawn on
   it — no measured answer exists yet; producing one is part of the task.
3. **Beat the baseline.** Track A: gradient-boosted trees (LightGBM-class
   trainers typically want the full training matrix resident in memory —
   plan around that). Track B: feed-forward MLP on streamed minibatches
   (what to feed it, normalization, use of the weight `w` in the loss,
   and in-window early stopping are yours to design). Develop at
   `--stride 4w`; only promoted candidates run `2w`/`1w`.
4. **Prove the lift is real.** Promotion needs a stable daily APS
   difference curve versus the matched linear baseline (`--baseline`), not
   one aggregate number.
5. **Stretch — autoresearch.** Once your manual loop works, try driving it
   with an agent: propose a change, train, score, read the report, propose
   again. The same rules bind the agent that bind you — TrainWindow-only
   data access, deterministic manifests, honest logs of failures. Treat
   this as a long-horizon direction, not a requirement.

## Data scale, memory and I/O (read before planning)

- The v2 dataset is ≈ **2.1 TiB** on disk, roughly **1.3 GiB per date**;
  one pass over a single 104-week training window reads on the order of
  **1 TiB** of `X`. Storage is provisioned for you — **repeated reading
  is your problem to engineer**.
- **Do not plan to load the dataset, or even one full training window,
  into RAM.** The expected pattern is streaming: `iter_dates()` yields
  per-date blocks for minibatch training; `sample(n_rows, seed)` gives a
  budgeted, reproducible random sample.
- Model families have different memory floors: minibatch-trained networks
  stream naturally, while gradient-boosted trees usually need their
  training matrix in memory — one more reason stage 2 comes before large
  models.

## Rules that are enforced, not advisory

- The harness decides what data your `fit` can see. Everything you compute
  (preprocessing, hyperparameters, early stopping, calibration) must come
  from inside the `TrainWindow`'s public API (`dates`, `iter_dates`,
  `sample`). Reaching around it — private attributes, direct dataset
  paths, cached label files — is a protocol violation; submitted runs are
  audited and violations void the result.
- Runs must be deterministic given (code, stride, row budget, seed); the
  manifest records all of them.
- Scoring covers every predictable date in a window or fails — no partial
  scores.
- `episode_id` is for grouping/alignment only. Feature de-anonymization is
  out of contract.

## Working mode and deliverables (GitHub)

- Work in your **own private GitHub repository** from day one and invite
  the reviewers you are given. Do not share code with anyone else working
  on the dataset — parallel independent attempts are intentional.
- A result exists when your repo contains it: the adapter code, the
  `run_manifest.json`, the scored `report.json`, and an experiment-log
  entry (date, hypothesis, configuration, outcome). **Failed experiments
  are logged with the same care as successes** — the log is a reviewed
  deliverable, not a diary.
- Building faster private tooling for iteration is encouraged, but a
  number only counts once it is reproduced through `f522kit run` +
  `f522kit score` under the frozen contract; those are the only
  artifacts review accepts.

See [docs/CONTRACT.md](docs/CONTRACT.md) for the frozen protocol and the
open policy boxes (TBD-1..5), and `examples/` for a minimal adapter.

The feature set was selected using outcomes that include the hidden
evaluation period. Hidden labels test your modeling procedure and relative
improvement over the matched baseline; they do not make this period an
independent, unbiased test of the feature set or its absolute performance.
