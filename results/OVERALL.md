# Overall evaluation

W1 = val 2024-04-09..05-06, W2 = val 2024-01-16..02-12 (28 days each, 52 weeks of training, stride 5, identical rows; single models averaged over seeds, combinations of seed-averaged parts).
Rolling = official kit evaluation, 104-week training, 4-week refit, 2022-07-01..2024-06-30 (731 days); one seed, ~10M sampled rows per anchor (ridge uses all rows); paired columns vs the official ridge (`runs/linear_4w`) on the same days.

| Model | W1 APS | W2 APS | Rolling APS | Sharpe | COR % | AR | diff Sharpe | win % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Ridge (official) | 1.76 | 1.80 | 2.59 | 27.7 | 6.31 | 0.70 | baseline | – |
| Linear MLP | 1.86 (3 seeds) | 1.66 (3 seeds) | 2.34 | 25.5 | 5.78 | 0.58 | -5.4 | 35 |
| 2-layer MLP | 2.13 (3 seeds) | 1.72 (3 seeds) | 2.41 | 25.8 | 5.90 | 0.65 | -4.0 | 36 |
| LightGBM | 1.87 (3 seeds) | 2.01 (3 seeds) | pending | pending | pending | pending | pending | pending |
| Ridge + LightGBM | 2.09 | 2.19 | pending | pending | pending | pending | pending | pending |
| Linear MLP + LightGBM | 2.18 | 2.15 | pending | pending | pending | pending | pending | pending |
| Ridge + LightGBM + MLP | 2.21 | 2.11 | pending | pending | pending | pending | pending | pending |

APS in bps; Sharpe = annualized Sharpe of daily APS; diff Sharpe = the same for the daily APS difference vs ridge (not the gap between the Sharpes: it measures how consistently the model beats ridge); win % = days beating ridge.
