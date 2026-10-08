Configurations in results/v2_52w (split 52w, stride 5, anchor 2024-04-14); seeds combined by averaging daily APS; paired columns vs the configuration of `ridge_eqall`. diff_SR = annualized Sharpe of the daily APS difference.

| config | runs | seeds | APS_bps | seed_range | fit_rows | rows | fit_s | dAPS_bps | diff_SR | win_% | note |
|---|---|---:|---:|---|---:|---|---:|---:|---:|---:|---|
| ridge {"es_days":14} | ridge_eqall | 1 | 1.755 | – | 19,305,374 | – | 323 | – | – | – | baseline |
| ridge {} | ridge | 1 | 1.837 | – | 20,366,066 | – | 245 | 0.082 | 10.017 | 64.286 |  |
| lgbm {"learning_rate":0.02,"max_rows":6000000,"num_boost_round":5000} | lgbm_rows6m_lr02 | 1 | 1.740 | – | 6,000,000 | – | 439 | -0.016 | -0.271 | 46.429 | ⚠ <3 seeds |
| lgbm {"max_rows":0} | lgbm_allrows | 1 | 1.695 | – | 19,305,374 | – | 1318 | -0.060 | -0.955 | 35.714 | ⚠ <3 seeds |
| lgbm {"max_rows":6000000,"min_data_in_leaf":5000,"num_leaves":15} | lgbm_rows6m_nl15 | 1 | 1.675 | – | 6,000,000 | – | 355 | -0.080 | -1.114 | 42.857 | ⚠ <3 seeds |
| lgbm {"max_rows":6000000} | lgbm_rows6m | 1 | 1.650 | – | 6,000,000 | – | 613 | -0.105 | -1.735 | 35.714 | ⚠ <3 seeds |
| lgbm {} | lgbm_base, lgbm_base_s1, lgbm_base_s2 | 3 | 1.425 | 1.39-1.50 | 2,000,000 | varies | 430 | -0.331 | -5.878 | 39.286 |  |
| ridge {"es_days":14,"max_rows":2000000} | ridge_eq2m | 1 | 1.515 | – | 2,000,000 | – | 340 | -0.241 | -7.175 | 28.571 |  |
| mlp {} | mlp_base, mlp_base_s1, mlp_base_s2 | 3 | 1.446 | 1.33-1.52 | 19,305,374 | same | 2019 | -0.309 | -7.804 | 35.714 |  |
| mlp {"max_rows":2000000} | mlp_eq2m | 1 | 0.821 | – | 2,000,000 | – | 625 | -0.934 | -14.810 | 17.857 | ⚠ <3 seeds |
