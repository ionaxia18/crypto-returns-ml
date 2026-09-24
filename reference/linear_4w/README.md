# Linear reference for the core980 assignment

Dataset ID: `faf0d629f2fb41a78f59a3e1bcb4e439`.

The reference uses the normalized ridge specification in `docs/CONTRACT.md`:
all positive-weight rows in each full training window, no subsampling,
104-week history, 1460-day half-life, no intercept, lambda 0.01,
`date_mature` model assignment, and a **4w refit stride**. Predictions are
scored under the student contract with identity calibration.

| Window | Dates | Mean daily APS (bps) | Mean daily COR |
|---|---|---:|---:|
| is | 2022-07-01 through 2024-06-30 | 2.5857007794942892 | 0.06309831452251272 |
| dev | 2024-07-01 through 2024-12-31 | 1.9922952289600018 | 0.04397372689283276 |

`report.json` contains the full visible-period reference metrics. The
instructor must confirm the reproduction tolerance before results receive
a pass/fail judgment; this file does not define a tolerance.

Implement the linear model in your own adapter and compare its report with
this reference. This folder contains **summary metrics only**, not prediction
arrays. The `f522kit score --baseline` option expects the directory holding
your reproduced linear predictions (`preds/YYYY-MM-DD.npy`), not this JSON
report. Compare each nonlinear run against a linear run with matching
stride, windows, and evaluation settings. These 4w metrics are not the
reference for a 1w or 2w run.

The feature set was selected using outcomes that include the hidden
evaluation period. Withholding its labels does not turn that period into
an independent test of feature selection or absolute performance.
