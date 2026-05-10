# Results CSV schemas

Reference for the directory layout under `results/` and the column schema of
each CSV file produced by `scripts/train_svr.py`. The underlying results-writer functions live in
`scripts/helpers_svr.py` (`save_results_csv`, `save_predictions_csv`, and
the `compute_permutation_importance` step inside `run_posthoc_inspection`).

## Directory structure

Each invocation of train_svr creates a fresh, timestamped
subdirectory under `results/` and writes its three CSVs there:

```
results/
├── RESULTS_CSV_SCHEMAS.md            # this file
├── YYYY-MM-DD_HHMMSS_TZ/             # one subdir per invocation
│   ├── svr_run_results.csv           # one row per requested run
│   ├── svr_participant_results.csv   # one row per (run, subject)
│   └── svr_perm_importance.csv       # one row per feature, best run only
├── YYYY-MM-DD_HHMMSS_TZ/
└── ...
```
## `svr_run_results.csv`

One row per requested run, in run order. 14 columns.

| Column | Description |
|---|---|
| `run` | Run name from `RUN_FEATURE_SOURCES` (e.g., `mean_predictor`, `egemaps_only`, `whisper_whole`, `whisper_whole_double_en`, `whisper_only`). |
| `n_features` | Width of this run's X matrix. `0` for `mean_predictor`. |
| `RMSE` | Nested-LOO root-mean-squared error of OOF predictions vs PHQ-9 truth. |
| `MAE` | Nested-LOO mean absolute error. |
| `R2` | Nested-LOO coefficient of determination (can be negative for `mean_predictor`). |
| `PI_alpha` | Miscoverage rate used for the jackknife prediction intervals (default `0.05` → 95% PI). Recorded so the PI columns are interpretable later. |
| `PI_coverage` | Empirical fraction of the 52 subjects whose true PHQ-9 falls inside their (1-α) PI; target ≈ 1-α. |
| `PI_mean_width` | Mean width (`pi_upper − pi_lower`) of per-subject PIs in PHQ-9 units; lower = tighter intervals. |
| `best_params` | JSON dict of GridSearchCV best hyperparameters from the post-hoc refit on all 52 subjects (e.g., `{"svr__C": 100, "svr__epsilon": 0.1, "svr__gamma": "scale"}`). `{}` for `mean_predictor`. |
| `feature_matrices_used` | JSON list of source matrices left-joined to build this run's X (e.g., `["egemaps", "whisper_whole"]`). Mirrors `RUN_FEATURE_SOURCES[run]`. `[]` for `mean_predictor`. |
| `n_egemaps_selected` | Count of EN-retained features whose `modality_of(name) == "egemaps"`. `0` for `mean_predictor`. |
| `n_whisper_selected` | Count of EN-retained features whose `modality_of(name)` is `whisper_whole` or `whisper_interview` (whisper modalities combined). `0` for `mean_predictor`. |
| `selected_features` | JSON list of all EN-retained feature names (input-axis order). `[]` for `mean_predictor`. |
| `top_5_perm_importance` | JSON list of `{"feature": ..., "perm_imp_mse_increase_train": ...}` dicts — top 5 EN-selected features by training-set permutation importance. `n_repeats` defaults to 20, bumped to 50 when `run == "whisper_whole"` for tighter CIs. `[]` for `mean_predictor`. |

## `svr_participant_results.csv`

One row per `(run, subject)` pair. With N runs and 52 subjects, this file
has `N × 52` rows. 8 columns. Useful for PI calibration plots, identifying
subjects always outside their PI, and per-subgroup coverage.

| Column | Description |
|---|---|
| `run` | Run name; joins to `svr_run_results.csv` via this column. |
| `subject_id` | MODMA subject ID as int (leading zeros stripped, e.g. `2010002`). 52 distinct values. |
| `y_true` | Ground-truth PHQ-9 score for this subject (range 0–27). |
| `y_pred` | OOF prediction for this subject under this run. For SVR runs, this is the held-out prediction from the GridSearchCV best estimator fit on the other 51 subjects; for `mean_predictor`, it is `mean(y[j] for j != i)`. |
| `RMSE` | Per-participant RMSE = `\|y_true − y_pred\|`. RMSE for n=1 reduces to absolute residual; included for symmetry with the per-run `RMSE` column in `svr_run_results.csv`. |
| `pi_lower` | Lower bound of the (1-α) jackknife PI: `y_pred[i] − q_{−i}` where `q_{−i}` is the (1-α) quantile of `\|residual\|` across the OTHER 51 subjects. |
| `pi_upper` | Upper bound of the (1-α) jackknife PI (symmetric around `y_pred`). |
| `in_interval` | Boolean: `True` iff `pi_lower ≤ y_true ≤ pi_upper`. The mean of this column within a run equals that run's `PI_coverage` in `svr_run_results.csv`. |

## `svr_perm_importance.csv`

One row per feature in the **best non-mean-predictor run's X matrix** (the
run with the lowest nested-LOO RMSE). Other runs are not represented here —
their per-run summaries live in `svr_run_results.csv`. Sorted descending by
`perm_imp_mse_increase_train`. 5 columns.

| Column | Description |
|---|---|
| `feature` | Feature name. eGeMAPS features are unprefixed (e.g., `F0semitoneFrom27.5Hz_sma3nz_amean`); whisper features carry `whisper_whole__` or `whisper_iv__` prefix; demographics carry `demo__`. |
| `modality` | One of `egemaps`, `whisper_whole`, `whisper_interview`, `demographics`, derived from the feature name's prefix by `modality_of()`. |
| `selected_by_EN` | Boolean: `True` iff the SelectFromModel(ElasticNet) step retained this feature when the pipeline was refit on all 52 subjects. For `whisper_whole_double_en` runs, reflects features surviving BOTH ElasticNet stages. |
| `perm_imp_mse_increase_train` | Mean MSE increase under feature shuffle, across `n_repeats` repeats. **Training-set** permutation importance — computed on the same data the post-hoc model was fit on, NOT held-out. EN-dropped features show ≈0 by construction (the selector removes them upstream of the SVR). |
| `perm_imp_mse_increase_train_std` | Standard deviation of the MSE-increase across the `n_repeats` shuffles. |

## Notes

- **JSON columns.** `best_params`, `feature_matrices_used`, `selected_features`,
  and `top_5_perm_importance` (in `svr_run_results.csv`) are stored as JSON-
  encoded strings; load with `json.loads(...)` after reading the CSV.
- **Units.** All RMSE/MAE/PI columns are in PHQ-9 score units (0–27 scale).
  `perm_imp_mse_increase_train` is MSE-increase, NOT RMSE-increase.
- **Reproducibility.** The pipeline uses `random_state=42` (RANDOM_STATE in
  `helpers_svr.py`) for ElasticNet selection and for permutation_importance.
- **Joining files within a run.** `svr_run_results.csv` and
  `svr_participant_results.csv` share the `run` column; the latter expands
  one row per subject per run. The perm-importance CSV is best-run only and
  does not have a `run` column.
