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
from sklearn.model_selection import (
    GridSearchCV,
    LeaveOneOut,
    cross_val_predict,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.multicalibration import (  # noqa: E402
    apply_multicalibration_updates,
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


def normalize_subject_id(value: object) -> int:
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    text = text.lstrip("0")
    return int(text) if text else 0


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value)


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


def metadata_columns(subject_info: pd.DataFrame) -> list[str]:
    return [
        c
        for c in ("group", "gender")
        if c in subject_info.columns and subject_info[c].notna().any()
    ]


def load_subject_metadata_normalized() -> pd.DataFrame:
    subject_info = pd.read_csv(SUBJECT_INFO_CSV, encoding="utf-8-sig")
    subject_info["subject_id"] = subject_info["subject_id"].apply(normalize_subject_id)
    return subject_info.set_index("subject_id").sort_index()


def run_from_predictions(
    *,
    args: argparse.Namespace,
) -> dict[str, object]:
    predictions_path = Path(args.predictions_csv)
    if not predictions_path.exists():
        raise FileNotFoundError(f"Predictions CSV not found: {predictions_path}")

    pred_df = pd.read_csv(predictions_path)
    required = {"run", "subject_id", "y_true", "y_pred"}
    missing = sorted(required - set(pred_df.columns))
    if missing:
        raise ValueError(
            f"{predictions_path} is missing required columns: {missing}"
        )

    available_runs = sorted(pred_df["run"].dropna().unique())
    if args.run_name is None:
        if len(available_runs) == 1:
            run_name = available_runs[0]
        else:
            raise ValueError(
                "Specify --run-name. Available runs: " + ", ".join(available_runs)
            )
    else:
        run_name = args.run_name

    run_df = pred_df[pred_df["run"] == run_name].copy()
    if run_df.empty:
        raise ValueError(
            f"No rows found for --run-name {run_name!r}. "
            f"Available runs: {', '.join(available_runs)}"
        )

    run_df["subject_id"] = run_df["subject_id"].apply(normalize_subject_id)
    subject_info = load_subject_metadata_normalized()
    meta_cols = metadata_columns(subject_info)
    run_df = run_df.join(subject_info[meta_cols], on="subject_id")

    missing_meta = run_df[meta_cols].isna().any(axis=1) if meta_cols else pd.Series(False)
    if bool(missing_meta.any()):
        missing_ids = run_df.loc[missing_meta, "subject_id"].tolist()
        raise ValueError(f"Missing metadata for subject_id values: {missing_ids}")

    y_true = run_df["y_true"].to_numpy(dtype=float)
    y_pred = run_df["y_pred"].to_numpy(dtype=float)
    group_frame = run_df[meta_cols].reset_index(drop=True)

    score_bin_edges = compute_score_bin_edges(y_pred, n_bins=args.score_bins)
    masks = build_group_masks(
        n=len(run_df),
        group_frame=group_frame,
        scores=y_pred,
        score_bin_edges=score_bin_edges,
        include_crosses=True,
        min_group_size=args.min_group_size,
    )

    mc = mean_multicalibrate(
        y_true=y_true,
        y_pred=y_pred,
        group_masks=masks,
        tol=args.tol,
        max_iters=args.max_iters,
        step_size=1.0,
        min_group_size=args.min_group_size,
        clip=(0.0, 27.0),
    )
    y_pred_mc = mc["y_pred_calibrated"]

    report = multicalibration_report(
        y_true=y_true,
        y_pred_before=y_pred,
        y_pred_after=y_pred_mc,
        group_masks=masks,
    )

    out_stem = f"from_predictions_{safe_name(run_name)}"
    out_df = run_df[
        ["run", "subject_id", *meta_cols, "y_true", "y_pred"]
    ].copy()
    out_df["y_pred_multicalibrated"] = y_pred_mc
    out_df["abs_error_before"] = np.abs(out_df["y_true"] - out_df["y_pred"])
    out_df["abs_error_after"] = np.abs(
        out_df["y_true"] - out_df["y_pred_multicalibrated"]
    )

    out_df.to_csv(RESULTS_DIR / f"{out_stem}_predictions.csv", index=False)
    report.to_csv(RESULTS_DIR / f"{out_stem}_report.csv", index=False)
    mc["history"].to_csv(RESULTS_DIR / f"{out_stem}_history.csv", index=False)

    return {
        "from_predictions_run": run_name,
        "from_predictions_csv": str(predictions_path),
        "from_predictions_base": regression_metrics(y_true, y_pred),
        "from_predictions_multicalibrated": regression_metrics(y_true, y_pred_mc),
        "from_predictions_n": int(len(run_df)),
        "from_predictions_group_columns": meta_cols,
        "from_predictions_outputs": {
            "predictions": str(RESULTS_DIR / f"{out_stem}_predictions.csv"),
            "report": str(RESULTS_DIR / f"{out_stem}_report.csv"),
            "history": str(RESULTS_DIR / f"{out_stem}_history.csv"),
        },
    }


def run_oof_diagnostic(
    *,
    args: argparse.Namespace,
    subject_info: pd.DataFrame,
    X_all: np.ndarray,
    y_all: np.ndarray,
    subject_ids: np.ndarray,
) -> dict[str, object]:
    print(
        "\nRunning nested LOO SVR diagnostic"
        + (" with quick grid..." if args.quick else " with full train_svr.py grid...")
    )
    inner = build_svr_search(quick=args.quick)
    y_pred_oof = cross_val_predict(inner, X_all, y_all, cv=LeaveOneOut(), n_jobs=-1)

    baseline_pred = leave_one_out_mean_baseline(y_all)
    meta_cols = metadata_columns(subject_info)
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

    return {
        "svr_oof": regression_metrics(y_all, y_pred_oof),
        "svr_oof_multicalibrated": regression_metrics(y_all, y_pred_mc),
        "loo_mean_baseline": regression_metrics(y_all, baseline_pred),
    }


def run_holdout_evaluation(
    *,
    args: argparse.Namespace,
    subject_info: pd.DataFrame,
    X_all: np.ndarray,
    y_all: np.ndarray,
    subject_ids: np.ndarray,
) -> dict[str, object]:
    print(
        "\nRunning held-out SVR multicalibration"
        + (" with quick grid..." if args.quick else " with full train_svr.py grid...")
    )

    groups_all = subject_info["group"].to_numpy()
    X_train, X_test, y_train, y_test, idx_train, idx_test, grp_train, grp_test = (
        train_test_split(
            X_all,
            y_all,
            subject_ids,
            groups_all,
            test_size=args.test_size,
            stratify=groups_all,
            random_state=RANDOM_STATE,
        )
    )

    train_search = build_svr_search(quick=args.quick)
    y_pred_train_oof = cross_val_predict(
        train_search,
        X_train,
        y_train,
        cv=LeaveOneOut(),
        n_jobs=-1,
    )

    final_search = build_svr_search(quick=args.quick)
    final_search.fit(X_train, y_train)
    y_pred_test = final_search.predict(X_test)

    meta_cols = metadata_columns(subject_info)
    train_meta = subject_info.loc[idx_train, meta_cols].reset_index(drop=True)
    test_meta = subject_info.loc[idx_test, meta_cols].reset_index(drop=True)

    score_bin_edges = compute_score_bin_edges(y_pred_train_oof, n_bins=args.score_bins)
    train_masks = build_group_masks(
        n=len(y_train),
        group_frame=train_meta,
        scores=y_pred_train_oof,
        score_bin_edges=score_bin_edges,
        include_crosses=True,
        min_group_size=args.min_group_size,
    )

    mc = mean_multicalibrate(
        y_true=y_train,
        y_pred=y_pred_train_oof,
        group_masks=train_masks,
        tol=args.tol,
        max_iters=args.max_iters,
        step_size=1.0,
        min_group_size=args.min_group_size,
        clip=(0.0, 27.0),
    )
    y_pred_train_mc = mc["y_pred_calibrated"]

    test_masks = build_group_masks(
        n=len(y_test),
        group_frame=test_meta,
        scores=y_pred_test,
        score_bin_edges=score_bin_edges,
        include_crosses=True,
        min_group_size=1,
    )
    y_pred_test_mc = apply_multicalibration_updates(
        y_pred=y_pred_test,
        group_masks=test_masks,
        updates=mc["updates"],
        clip=(0.0, 27.0),
    )

    train_report = multicalibration_report(
        y_true=y_train,
        y_pred_before=y_pred_train_oof,
        y_pred_after=y_pred_train_mc,
        group_masks=train_masks,
    )
    test_report = multicalibration_report(
        y_true=y_test,
        y_pred_before=y_pred_test,
        y_pred_after=y_pred_test_mc,
        group_masks=test_masks,
    )

    train_baseline = leave_one_out_mean_baseline(y_train)
    test_baseline = np.full(len(y_test), float(np.mean(y_train)))

    pd.DataFrame(
        {
            "subject_id": idx_train,
            "group": grp_train,
            "gender": subject_info.loc[idx_train, "gender"].to_numpy()
            if "gender" in subject_info.columns
            else np.nan,
            "PHQ9_actual": y_train,
            "PHQ9_pred_svr_oof": y_pred_train_oof,
            "PHQ9_pred_svr_mc": y_pred_train_mc,
            "PHQ9_pred_loo_mean": train_baseline,
        }
    ).to_csv(RESULTS_DIR / "svr_holdout_train_predictions.csv", index=False)

    pd.DataFrame(
        {
            "subject_id": idx_test,
            "group": grp_test,
            "gender": subject_info.loc[idx_test, "gender"].to_numpy()
            if "gender" in subject_info.columns
            else np.nan,
            "PHQ9_actual": y_test,
            "PHQ9_pred_svr": y_pred_test,
            "PHQ9_pred_svr_mc": y_pred_test_mc,
            "PHQ9_pred_train_mean": test_baseline,
        }
    ).to_csv(RESULTS_DIR / "svr_holdout_test_predictions.csv", index=False)

    train_report.to_csv(RESULTS_DIR / "svr_holdout_train_report.csv", index=False)
    test_report.to_csv(RESULTS_DIR / "svr_holdout_test_report.csv", index=False)
    mc["history"].to_csv(RESULTS_DIR / "svr_holdout_history.csv", index=False)

    return {
        "svr_holdout_train_oof": regression_metrics(y_train, y_pred_train_oof),
        "svr_holdout_train_oof_multicalibrated": regression_metrics(
            y_train,
            y_pred_train_mc,
        ),
        "svr_holdout_test": regression_metrics(y_test, y_pred_test),
        "svr_holdout_test_multicalibrated": regression_metrics(y_test, y_pred_test_mc),
        "train_loo_mean_baseline": regression_metrics(y_train, train_baseline),
        "test_train_mean_baseline": regression_metrics(y_test, test_baseline),
        "split": {
            "train_n": int(len(y_train)),
            "test_n": int(len(y_test)),
            "train_groups": pd.Series(grp_train).value_counts().to_dict(),
            "test_groups": pd.Series(grp_test).value_counts().to_dict(),
        },
        "best_params": final_search.best_params_,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run standalone SVR predictions with multicalibration."
    )
    parser.add_argument(
        "--mode",
        choices=("holdout", "oof", "both", "predictions"),
        default="holdout",
        help=(
            "Use held-out evaluation, all-subject LOO diagnostic, both, "
            "or calibrate an existing train_svr predictions CSV."
        ),
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
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.20,
        help="Held-out test fraction used in --mode holdout or --mode both.",
    )
    parser.add_argument(
        "--predictions-csv",
        type=Path,
        default=None,
        help="Path to train_svr.py's svr_participant_results.csv.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Run name inside the predictions CSV, such as egemaps_whisper_demo.",
    )
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.mode == "predictions":
        if args.predictions_csv is None:
            raise SystemExit("--mode predictions requires --predictions-csv")
        metrics = {
            "config": {
                "mode": args.mode,
                "score_bins": int(args.score_bins),
                "tol": float(args.tol),
                "max_iters": int(args.max_iters),
                "min_group_size": int(args.min_group_size),
                "run_name": args.run_name,
            },
        }
        metrics.update(run_from_predictions(args=args))
        with open(RESULTS_DIR / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        print("\nSaved SVR multicalibration outputs to:")
        print(f"  {RESULTS_DIR}")
        print("\nFinal SVR predictions vs multicalibrated predictions:")
        print(f"  run       = {metrics['from_predictions_run']}")
        print(f"  base_rmse = {metrics['from_predictions_base']['rmse']:.3f}")
        print(
            "  mc_rmse   = "
            f"{metrics['from_predictions_multicalibrated']['rmse']:.3f}"
        )
        print(f"  base_mae  = {metrics['from_predictions_base']['mae']:.3f}")
        print(
            "  mc_mae    = "
            f"{metrics['from_predictions_multicalibrated']['mae']:.3f}"
        )
        return

    subject_info, subject_mean_88, files_per_subject = load_subject_level_egemaps()

    X_all = subject_mean_88.to_numpy()
    y_all = subject_info["PHQ-9"].to_numpy()
    subject_ids = subject_info.index.to_numpy()

    print(
        f"All subjects: X={X_all.shape}  "
        f"groups={subject_info['group'].value_counts().to_dict()}  "
        f"PHQ-9 range {y_all.min()}-{y_all.max()}"
    )

    metrics = {
        "config": {
            "mode": args.mode,
            "quick": bool(args.quick),
            "score_bins": int(args.score_bins),
            "tol": float(args.tol),
            "max_iters": int(args.max_iters),
            "min_group_size": int(args.min_group_size),
            "test_size": float(args.test_size),
            "n_subjects": int(len(y_all)),
            "group_columns": metadata_columns(subject_info),
        },
        "feature_coverage": {
            "min_files_per_subject": int(files_per_subject.min()),
            "median_files_per_subject": int(np.median(files_per_subject)),
            "max_files_per_subject": int(files_per_subject.max()),
        },
    }

    if args.mode in ("holdout", "both"):
        metrics.update(
            run_holdout_evaluation(
                args=args,
                subject_info=subject_info,
                X_all=X_all,
                y_all=y_all,
                subject_ids=subject_ids,
            )
        )

    if args.mode in ("oof", "both"):
        metrics.update(
            run_oof_diagnostic(
                args=args,
                subject_info=subject_info,
                X_all=X_all,
                y_all=y_all,
                subject_ids=subject_ids,
            )
        )

    with open(RESULTS_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\nSaved SVR multicalibration outputs to:")
    print(f"  {RESULTS_DIR}")

    if "svr_holdout_test" in metrics:
        print("\nHeld-out SVR vs multicalibrated held-out SVR:")
        print(f"  svr_test_rmse = {metrics['svr_holdout_test']['rmse']:.3f}")
        print(
            "  mc_test_rmse  = "
            f"{metrics['svr_holdout_test_multicalibrated']['rmse']:.3f}"
        )
        print(f"  svr_test_mae  = {metrics['svr_holdout_test']['mae']:.3f}")
        print(
            "  mc_test_mae   = "
            f"{metrics['svr_holdout_test_multicalibrated']['mae']:.3f}"
        )

    if "svr_oof" in metrics:
        print("\nOOF SVR diagnostic vs multicalibrated OOF SVR:")
        print(f"  svr_oof_rmse = {metrics['svr_oof']['rmse']:.3f}")
        print(f"  mc_oof_rmse  = {metrics['svr_oof_multicalibrated']['rmse']:.3f}")
        print(f"  svr_oof_mae  = {metrics['svr_oof']['mae']:.3f}")
        print(f"  mc_oof_mae   = {metrics['svr_oof_multicalibrated']['mae']:.3f}")


if __name__ == "__main__":
    main()
