from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import GridSearchCV, LeaveOneOut, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

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

exclude_if = (issues["severity"] == "exclude")
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
# 80/20 train/test split, STRATIFIED by MDD/HC group
# ---------------------------------------------------------------------------

X_all = subject_mean_88.to_numpy()              # (52, 88)
y_all = subject_info["PHQ-9"].to_numpy()           # (52,)
groups_all = subject_info["group"].to_numpy()       # (52,) — MDD or HC
idx_all = subject_info.index.to_numpy()             # subject_ids

X_train, X_test, y_train, y_test, grp_train, grp_test, idx_train, idx_test = train_test_split(
    X_all, y_all, groups_all, idx_all,
    test_size=0.20, stratify=groups_all, random_state=RANDOM_STATE,
)

print(f"Train: X={X_train.shape}  groups={pd.Series(grp_train).value_counts().to_dict()}  "
      f"PHQ-9 range {y_train.min()}-{y_train.max()}")
print(f"Test : X={X_test.shape}   groups={pd.Series(grp_test).value_counts().to_dict()}  "
      f"PHQ-9 range {y_test.min()}-{y_test.max()}")
print(f"\nTest subject_ids: {list(idx_test)}")

# ---------------------------------------------------------------------------
# Standardize features (fit on train only)
# ---------------------------------------------------------------------------

scaler = StandardScaler().fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s = scaler.transform(X_test)

# ---------------------------------------------------------------------------
# Elastic Net feature selection 
# ---------------------------------------------------------------------------

loo = LeaveOneOut()
enet_cv = ElasticNetCV(
    l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0],
    alphas=np.logspace(-3, 1, 40),
    cv=loo, max_iter=20000, n_jobs=-1, random_state=42,
).fit(X_train_s, y_train)

# Get the nonzero feature indices
nonzero_mask = enet_cv.coef_ != 0
n_selected = nonzero_mask.sum()
print(f"Elastic Net selected {n_selected} features out of 88")
print(f"Elastic Net test RMSE: {np.sqrt(mean_squared_error(y_test, enet_cv.predict(X_test_s))):.3f}")

# ---------------------------------------------------------------------------
#  SVR-RBF on selected features
# ---------------------------------------------------------------------------

X_tr_sel = X_train_s[:, nonzero_mask]
X_te_sel = X_test_s[:, nonzero_mask]

svr_sel = GridSearchCV(
    SVR(kernel="rbf"),
    {"C": [0.1, 1, 10, 100], "epsilon": [0.1, 0.5, 1.0, 2.0], "gamma": ["scale", "auto"]},
    cv=LeaveOneOut(), scoring="neg_mean_squared_error", n_jobs=-1,
).fit(X_tr_sel, y_train)

y_pred_svr_sel = svr_sel.best_estimator_.predict(X_te_sel)
print(f"\nSVR on {n_selected} selected features → test RMSE={np.sqrt(mean_squared_error(y_test, y_pred_svr_sel)):.3f}, "
      f"R²={r2_score(y_test, y_pred_svr_sel):.3f}")

# ---------------------------------------------------------------------------
#  Summary stats
# ---------------------------------------------------------------------------
y_pred = svr_sel.best_estimator_.predict(X_te_sel)
print(f"\nBest SVR params: {svr_sel.best_params_}")
print(f"Test RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.3f}  "
      f"MAE: {mean_absolute_error(y_test, y_pred):.3f}  "
      f"R²: {r2_score(y_test, y_pred):.3f}")


baseline_pred = np.full_like(y_test, fill_value=y_train.mean(), dtype=float)
print(f"SVR {'beats' if (d := np.sqrt(mean_squared_error(y_test, baseline_pred)) - np.sqrt(mean_squared_error(y_test, y_pred))) > 0 else 'loses to'} mean baseline by {abs(d):.3f} RMSE")