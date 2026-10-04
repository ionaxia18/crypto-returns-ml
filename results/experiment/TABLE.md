Experiment split: anchor 2024-04-14, train 2023-10-09..2024-04-07 (26w, minute stride 5, 11,389,131 rows), val 2024-04-09..2024-05-06 (28 dates, all rows); paired columns vs `results/experiment/ridge`

| name | model | APS_bps | COR_% | APS_SR | AR | clip_rate | fit_s | total_s | peak_GB | cpus | dAPS_bps | diff_SR | win_% |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| linear_4w | kit run | 2.053 | 4.281 | 22.511 | 0.649 | 0.000 | – | – | – | – | 0.410 | 11.065 | 82.143 |
| ridge | ridge | 1.643 | 3.468 | 26.124 | 0.533 | 0.000 | 173 | 226 | 45.234 | 16 | 0.000 | nan | 0.000 |
| mlp_base | mlp | 1.436 | 2.986 | 27.505 | 0.657 | 0.000 | 1108 | 1212 | 53.138 | 16 | -0.207 | -4.397 | 46.429 |
| lgbm_base | lgbm | 1.174 | 2.624 | 25.815 | 0.463 | 0.000 | 270 | 530 | 36.726 | 16 | -0.468 | -6.985 | 28.571 |
