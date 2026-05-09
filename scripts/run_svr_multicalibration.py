#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, LeaveOneOut, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.multicalibration import (  # noqa: E402
    build_group_masks,
    compute_score_bin_edges,
    mean_multicalibrate,
    multicalibration_report,
)


EGEMAPS_CSV = REPO_ROOT / "data" / "features" / "egemaps.csv"
SUBJECT_INFO_CSV = REPO_ROOT / "data" / "metadata" / "subject_info_map.csv"
ISSUES_CSV = REPO_ROOT / "data" / "metadata" / "data_quality_issues.csv"
RESULTS_DIR = REPO_ROOT / "data" / "results" / "svr_multicalibration"
RANDOM_STATE = 42


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def leave_one_out_mean_baseline(y: np.ndarray) -> np.ndarray:
    return np.array([y[np.arange(len(y)) != i].mean() for i in range(len(y))])


def load_subject_level_egemaps() -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    subject_info = pd.read_csv(
        SUBJECT_INFO_CSV,
        dtype={"subject_id": str},
        encoding="utf-8-sig",
    ).set_index("subject_id")
    print(f"Loaded {SUBJECT_INFO_CSV.name}: {subject_info.shape}")

    egemaps = pd.read_csv(EGEMAPS_CSV, dtype={"subject_id": str, "file_number": int})
    print(f"Loaded {EGEMAPS_CSV.name}: {egemaps.shape}")

    issues = pd.read_csv(ISSUES_CSV, dtype={"subject_id": str, "file_number": int})
    print(f"Loaded {ISSUES_CSV.name}: {issues.shape}")

    exclude_if = (issues["severity"] == "exclude") & (issues["source"] == "disk_missing")
    exclude_pairs = set(
        zip(
            issues.loc[exclude_if, "subject_id"],
            issues.loc[exclude_if, "file_number"],
        )
    )

    before = len(egemaps)
    egemaps = egemaps[
        ~pd.MultiIndex.from_arrays(
            [egemaps["subject_id"], egemaps["file_number"]]
        ).isin(exclude_pairs)
    ].reset_index(drop=True)
    print(
        f"Filtered out {before - len(egemaps)} rows via {ISSUES_CSV.name}; "
        f"remaining {egemaps.shape}"
    )

    feature_cols = [c for c in egemaps.columns if c not in ("subject_id", "file_number")]
    subject_mean_88 = egemaps.groupby("subject_id")[feature_cols].mean()
    subject_mean_88 = subject_mean_88.loc[subject_info.index]
    print(f"Per-subject means: {subject_mean_88.shape}")

    files_per_subject = egemaps.groupby("subject_id").size().reindex(
        subject_info.index,
        fill_value=0,
    )
    print(
        "Files per subject:",
        {
            "min": int(files_per_subject.min()),
            "median": int(files_per_subject.median()),
            "max": int(files_per_subject.max()),
        },
    )
    return subject_info, subject_mean_88, files_per_subject.to_numpy()


def build_svr_search(*, quick: bool) -> GridSearchCV:
    pipe = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "select",
                SelectFromModel(
                    ElasticNet(
                        alpha=0.1,
                        l1_ratio=0.7,
                        max_iter=20000,
                        random_state=RANDOM_STATE,
                    ),
                    threshold=1e-10,
                ),
            ),
            ("svr", SVR(kernel="rbf")),
        ]
    )

    if quick:
        param_grid = {
            "svr__C": [1, 10],
            "svr__epsilon": [0.5, 1.0],
            "svr__gamma": ["scale"],
        }
    else:
        param_grid = {
            "svr__C": [0.1, 1, 10, 100],
            "svr__epsilon": [0.1, 0.5, 1.0, 2.0],
            "svr__gamma": ["scale", "auto"],
        }

    return GridSearchCV(
        pipe,
        param_grid=param_grid,
        cv=LeaveOneOut(),
        scoring="neg_mean_squared_error",
        n_jobs=1,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run standalone SVR out-of-fold predictions with multicalibration."
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use a smaller SVR hyperparameter grid for faster experimentation.",
    )
    parser.add_argument(
        "--score-bins",
        type=int,
        default=4,
        help="Number of prediction-score quantile bins for multicalibration.",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=1.0,
        help="Stop once every subgroup mean residual is within this PHQ-9 tolerance.",
    )
    parser.add_argument(
        "--max-iters",
        type=int,
        default=50,
        help="Maximum multicalibration update iterations.",
    )
    parser.add_argument(
        "--min-group-size",
        type=int,
        default=5,
        help="Smallest subgroup size retained for calibration/reporting.",
    )
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    subject_info, subject_mean_88, files_per_subject = load_subject_level_egemaps()

    X_all = subject_mean_88.to_numpy()
    y_all = subject_info["PHQ-9"].to_numpy()
    subject_ids = subject_info.index.to_numpy()

    print(
        f"All subjects: X={X_all.shape}  "
        f"groups={subject_info['group'].value_counts().to_dict()}  "
        f"PHQ-9 range {y_all.min()}-{y_all.max()}"
    )

    print(
        "\nRunning nested LOO SVR"
        + (" with quick grid..." if args.quick else " with full train_svr.py grid...")
    )
    inner = build_svr_search(quick=args.quick)
    y_pred_oof = cross_val_predict(inner, X_all, y_all, cv=LeaveOneOut(), n_jobs=-1)

    baseline_pred = leave_one_out_mean_baseline(y_all)
    meta_cols = [
        c
        for c in ("group", "gender")
        if c in subject_info.columns and subject_info[c].notna().any()
    ]
    group_frame = subject_info[meta_cols].reset_index(drop=True)

    score_bin_edges = compute_score_bin_edges(y_pred_oof, n_bins=args.score_bins)
    masks = build_group_masks(
        n=len(y_all),
        group_frame=group_frame,
        scores=y_pred_oof,
        score_bin_edges=score_bin_edges,
        include_crosses=True,
        min_group_size=args.min_group_size,
    )

    mc = mean_multicalibrate(
        y_true=y_all,
        y_pred=y_pred_oof,
        group_masks=masks,
        tol=args.tol,
        max_iters=args.max_iters,
        step_size=1.0,
        min_group_size=args.min_group_size,
        clip=(0.0, 27.0),
    )
    y_pred_mc = mc["y_pred_calibrated"]

    report = multicalibration_report(
        y_true=y_all,
        y_pred_before=y_pred_oof,
        y_pred_after=y_pred_mc,
        group_masks=masks,
    )

    metrics = {
        "svr_oof": regression_metrics(y_all, y_pred_oof),
        "svr_oof_multicalibrated": regression_metrics(y_all, y_pred_mc),
        "loo_mean_baseline": regression_metrics(y_all, baseline_pred),
        "config": {
            "quick": bool(args.quick),
            "score_bins": int(args.score_bins),
            "tol": float(args.tol),
            "max_iters": int(args.max_iters),
            "min_group_size": int(args.min_group_size),
            "n_subjects": int(len(y_all)),
            "group_columns": meta_cols,
        },
        "feature_coverage": {
            "min_files_per_subject": int(files_per_subject.min()),
            "median_files_per_subject": int(np.median(files_per_subject)),
            "max_files_per_subject": int(files_per_subject.max()),
        },
    }

    predictions = pd.DataFrame(
        {
            "subject_id": subject_ids,
            "group": subject_info["group"].to_numpy(),
            "gender": subject_info["gender"].to_numpy()
            if "gender" in subject_info.columns
            else np.nan,
            "PHQ9_actual": y_all,
            "PHQ9_pred_svr_oof": y_pred_oof,
            "PHQ9_pred_svr_mc": y_pred_mc,
            "PHQ9_pred_loo_mean": baseline_pred,
        }
    )

    predictions.to_csv(RESULTS_DIR / "svr_multicalibration_predictions.csv", index=False)
    report.to_csv(RESULTS_DIR / "svr_multicalibration_report.csv", index=False)
    mc["history"].to_csv(RESULTS_DIR / "svr_multicalibration_history.csv", index=False)
    with open(RESULTS_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\nSaved SVR multicalibration outputs to:")
    print(f"  {RESULTS_DIR}")
    print("\nOOF SVR vs multicalibrated OOF SVR:")
    print(f"  svr_rmse    = {metrics['svr_oof']['rmse']:.3f}")
    print(f"  mc_rmse     = {metrics['svr_oof_multicalibrated']['rmse']:.3f}")
    print(f"  svr_mae     = {metrics['svr_oof']['mae']:.3f}")
    print(f"  mc_mae      = {metrics['svr_oof_multicalibrated']['mae']:.3f}")
    print(f"  baseline_rmse = {metrics['loo_mean_baseline']['rmse']:.3f}")

    print("\nWorst subgroup gaps after calibration:")
    if report.empty:
        print("  No subgroup report generated.")
    else:
        print(
            report[
                [
                    "group",
                    "n",
                    "true_mean",
                    "pred_mean_before",
                    "pred_mean_after",
                    "abs_gap_before",
                    "abs_gap_after",
                ]
            ]
            .head(10)
            .to_string(index=False)
        )


if __name__ == "__main__":
    main()
