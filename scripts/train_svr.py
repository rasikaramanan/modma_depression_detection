from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV, LeaveOneOut, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
EGEMAPS_CSV = REPO_ROOT / "data" / "features" / "egemaps.csv"
SUBJECT_INFO_CSV = REPO_ROOT / "data" / "metadata" / "subject_info_map.csv"
ISSUES_CSV = REPO_ROOT / "data" / "metadata" / "data_quality_issues.csv"

RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Read in data from CSVs
# ---------------------------------------------------------------------------

subject_info = pd.read_csv(SUBJECT_INFO_CSV,dtype={"subject_id": str},).set_index("subject_id")
print(f"Loaded {SUBJECT_INFO_CSV.name}: {subject_info.shape}")

egemaps = pd.read_csv(EGEMAPS_CSV, dtype={"subject_id": str, "file_number": int})
print(f"Loaded {EGEMAPS_CSV.name}: {egemaps.shape}")

issues = pd.read_csv(ISSUES_CSV, dtype={"subject_id": str, "file_number": int})
print(f"Loaded {ISSUES_CSV.name}: {issues.shape}")

# ---------------------------------------------------------------------------
# Filter out datafiles flagged in data/metadata/data_quality_issues.csv
# ---------------------------------------------------------------------------

exclude_if = (issues["severity"] == "exclude")  & (issues["source"] == "disk_missing")
exclude_pairs = set(zip(issues.loc[exclude_if, "subject_id"],
                        issues.loc[exclude_if, "file_number"]))
before = len(egemaps)
egemaps = egemaps[
    ~pd.MultiIndex.from_arrays([egemaps["subject_id"], egemaps["file_number"]]).isin(exclude_pairs)
].reset_index(drop=True)
print(f"Filtered out {before - len(egemaps)} rows via {ISSUES_CSV.name}; remaining {egemaps.shape}")

# ---------------------------------------------------------------------------
# Aggregate per-subject mean across the 88 egemaps features 
# ---------------------------------------------------------------------------

feature_cols = [c for c in egemaps.columns if c not in ("subject_id", "file_number")]
subject_mean_88 = egemaps.groupby("subject_id")[feature_cols].mean()
subject_mean_88 = subject_mean_88.loc[subject_info.index]
print(f"Per-subject means: {subject_mean_88.shape}")

# ---------------------------------------------------------------------------
# Build feature matrix and target — no holdout split; nested LOO below
# uses each of the 52 subjects as the held-out point exactly once
# ---------------------------------------------------------------------------

X_all = subject_mean_88.to_numpy()              # (52, 88)
y_all = subject_info["PHQ-9"].to_numpy()        # (52,)

print(f"All subjects: X={X_all.shape}  groups={subject_info['group'].value_counts().to_dict()}  "
      f"PHQ-9 range {y_all.min()}-{y_all.max()}")

# ---------------------------------------------------------------------------
# Pipeline: standardize → Elastic-Net feature selection → SVR-RBF
# Wrapping all three steps in a Pipeline ensures every step refits per LOO
# fold (no feature-selection leak). Inner GridSearchCV tunes SVR hyperparams.
# Outer LOO via cross_val_predict gives 52 honest out-of-fold predictions.
# Note: ElasticNet uses fixed alpha/l1_ratio (not ElasticNetCV) to keep the
# nested CV tractable — tuning EN inside outer LOO is too expensive.
# ---------------------------------------------------------------------------

pipe = Pipeline([
    ("scale", StandardScaler()),
    ("select", SelectFromModel(
        ElasticNet(alpha=0.1, l1_ratio=0.7, max_iter=20000, random_state=42),
        threshold=1e-10,
    )),
    ("svr", SVR(kernel="rbf")),
])

inner = GridSearchCV(
    pipe,
    param_grid={
        "svr__C":       [0.1, 1, 10, 100],
        "svr__epsilon": [0.1, 0.5, 1.0, 2.0],
        "svr__gamma":   ["scale", "auto"],
    },
    cv=LeaveOneOut(), scoring="neg_mean_squared_error", n_jobs=1,
)

y_pred_oof = cross_val_predict(inner, X_all, y_all, cv=LeaveOneOut(), n_jobs=-1)

# ---------------------------------------------------------------------------
# Summary stats — nested-LOO metrics + leave-one-out mean baseline
# ---------------------------------------------------------------------------

nested_rmse = np.sqrt(mean_squared_error(y_all, y_pred_oof))
nested_mae  = mean_absolute_error(y_all, y_pred_oof)
nested_r2   = r2_score(y_all, y_pred_oof)
print(f"\nNested-LOO Pipeline (scale → EN-select → SVR)")
print(f"  RMSE: {nested_rmse:.3f}  MAE: {nested_mae:.3f}  R²: {nested_r2:.3f}")

baseline_pred = np.array([y_all[np.arange(len(y_all)) != i].mean() for i in range(len(y_all))])
baseline_rmse = np.sqrt(mean_squared_error(y_all, baseline_pred))
print(f"Mean-predictor baseline (leave-one-out): RMSE {baseline_rmse:.3f}")
print(f"SVR {'beats' if (d := baseline_rmse - nested_rmse) > 0 else 'loses to'} mean baseline by {abs(d):.3f} RMSE")