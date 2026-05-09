#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneOut, cross_val_predict, train_test_split
from sklearn.preprocessing import StandardScaler

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


PROJECT_ROOT = REPO_ROOT / "CSCI567 Project"
AUDIO_ROOT = PROJECT_ROOT / "modma_data" / "audio_lanzhou_2015"
SUBJECT_INFO_CSV = PROJECT_ROOT / "subject_info_map.csv"
AUDIO_FILE_MAP_CSV = PROJECT_ROOT / "audio_file_map.csv"
RESULTS_DIR = REPO_ROOT / "data" / "results" / "multicalibration"

DROP_COLS = ("file", "start", "end")
RANDOM_STATE = 42


def load_functional_frames(audio_root: Path) -> dict[int, pd.DataFrame]:
    func_dfs: dict[int, pd.DataFrame] = {}
    subject_dirs = sorted(p for p in audio_root.iterdir() if p.is_dir())

    for file_idx in range(1, 30):
        stem = f"{file_idx:02d}"
        rows: dict[str, pd.Series] = {}
        feature_cols: list[str] | None = None

        for subj_dir in subject_dirs:
            func_csv = subj_dir / f"{stem}_openSMILE_func.csv"
            if not func_csv.exists():
                continue

            df = pd.read_csv(func_csv)
            df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

            if feature_cols is None:
                feature_cols = list(df.columns)
            elif list(df.columns) != feature_cols:
                raise ValueError(
                    f"Column mismatch in {func_csv}: "
                    f"expected first columns {feature_cols[:3]}, "
                    f"got {list(df.columns)[:3]}"
                )

            rows[subj_dir.name] = df.iloc[0]

        if feature_cols is None:
            raise ValueError(f"No functional CSVs found for file index {file_idx}")

        out = pd.DataFrame.from_dict(rows, orient="index", columns=feature_cols)
        out.index.name = "subject_id"
        func_dfs[file_idx] = out

    return func_dfs


def aggregate_subject_mean(
    func_dfs: dict[int, pd.DataFrame],
    subject_index: pd.Index,
) -> tuple[pd.DataFrame, pd.Series]:
    stacked = pd.concat(func_dfs.values(), axis=0)
    subject_mean_88 = stacked.groupby(level=0).mean()
    subject_mean_88.index.name = "subject_id"
    subject_mean_88 = subject_mean_88.loc[subject_index]
    files_per_subject = stacked.groupby(level=0).size().reindex(subject_index, fill_value=0)
    return subject_mean_88, files_per_subject


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if not AUDIO_ROOT.exists():
        raise FileNotFoundError(f"Audio root not found: {AUDIO_ROOT}")
    if not SUBJECT_INFO_CSV.exists():
        raise FileNotFoundError(f"Missing subject info CSV: {SUBJECT_INFO_CSV}")
    if not AUDIO_FILE_MAP_CSV.exists():
        raise FileNotFoundError(f"Missing audio file map CSV: {AUDIO_FILE_MAP_CSV}")

    print(f"Loading features from {AUDIO_ROOT}")
    func_dfs = load_functional_frames(AUDIO_ROOT)

    subject_info = pd.read_csv(SUBJECT_INFO_CSV, dtype={"subject_id": str}).set_index("subject_id")
    _ = pd.read_csv(AUDIO_FILE_MAP_CSV)

    subject_mean_88, files_per_subject = aggregate_subject_mean(func_dfs, subject_info.index)
    print(
        "Files per subject:",
        {
            "min": int(files_per_subject.min()),
            "median": int(files_per_subject.median()),
            "max": int(files_per_subject.max()),
        },
    )

    X_all = subject_mean_88.values
    y_all = subject_info["PHQ-9"].values
    groups_all = subject_info["group"].values
    idx_all = subject_info.index.values

    X_train, X_test, y_train, y_test, grp_train, grp_test, idx_train, idx_test = train_test_split(
        X_all,
        y_all,
        groups_all,
        idx_all,
        test_size=0.20,
        stratify=groups_all,
        random_state=RANDOM_STATE,
    )

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    loo = LeaveOneOut()
    enet_cv = ElasticNetCV(
        l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0],
        alphas=np.logspace(-3, 1, 40),
        cv=loo,
        max_iter=20000,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    ).fit(X_train_s, y_train)

    final_enet = ElasticNet(
        alpha=enet_cv.alpha_,
        l1_ratio=enet_cv.l1_ratio_,
        max_iter=20000,
        random_state=RANDOM_STATE,
    )
    loso_pred_train = cross_val_predict(final_enet, X_train_s, y_train, cv=loo)
    y_pred_test = enet_cv.predict(X_test_s)

    train_meta = subject_info.loc[idx_train, ["group", "gender"]].reset_index(drop=True)
    test_meta = subject_info.loc[idx_test, ["group", "gender"]].reset_index(drop=True)

    score_bin_edges = compute_score_bin_edges(loso_pred_train, n_bins=4)
    train_masks = build_group_masks(
        n=len(y_train),
        group_frame=train_meta,
        scores=loso_pred_train,
        score_bin_edges=score_bin_edges,
        include_crosses=True,
        min_group_size=5,
    )

    mc = mean_multicalibrate(
        y_true=y_train,
        y_pred=loso_pred_train,
        group_masks=train_masks,
        tol=1.0,
        max_iters=50,
        step_size=1.0,
        min_group_size=5,
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
        y_pred_before=loso_pred_train,
        y_pred_after=y_pred_train_mc,
        group_masks=train_masks,
    )
    test_report = multicalibration_report(
        y_true=y_test,
        y_pred_before=y_pred_test,
        y_pred_after=y_pred_test_mc,
        group_masks=test_masks,
    )

    metrics = {
        "base_train": regression_metrics(y_train, loso_pred_train),
        "mc_train": regression_metrics(y_train, y_pred_train_mc),
        "base_test": regression_metrics(y_test, y_pred_test),
        "mc_test": regression_metrics(y_test, y_pred_test_mc),
        "elastic_net": {
            "alpha": float(enet_cv.alpha_),
            "l1_ratio": float(enet_cv.l1_ratio_),
            "n_nonzero": int((enet_cv.coef_ != 0).sum()),
        },
        "split": {
            "train_n": int(len(y_train)),
            "test_n": int(len(y_test)),
            "train_groups": pd.Series(grp_train).value_counts().to_dict(),
            "test_groups": pd.Series(grp_test).value_counts().to_dict(),
        },
        "feature_coverage": {
            "min_files_per_subject": int(files_per_subject.min()),
            "median_files_per_subject": int(files_per_subject.median()),
            "max_files_per_subject": int(files_per_subject.max()),
        },
    }

    pd.DataFrame(
        {
            "subject_id": idx_train,
            "group": grp_train,
            "PHQ9_actual": y_train,
            "PHQ9_pred_base": loso_pred_train,
            "PHQ9_pred_mc": y_pred_train_mc,
        }
    ).to_csv(RESULTS_DIR / "train_predictions.csv", index=False)

    pd.DataFrame(
        {
            "subject_id": idx_test,
            "group": grp_test,
            "PHQ9_actual": y_test,
            "PHQ9_pred_base": y_pred_test,
            "PHQ9_pred_mc": y_pred_test_mc,
        }
    ).to_csv(RESULTS_DIR / "test_predictions.csv", index=False)

    train_report.to_csv(RESULTS_DIR / "train_multicalibration_report.csv", index=False)
    test_report.to_csv(RESULTS_DIR / "test_multicalibration_report.csv", index=False)
    mc["history"].to_csv(RESULTS_DIR / "multicalibration_history.csv", index=False)
    with open(RESULTS_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\nSaved multicalibration outputs to:")
    print(f"  {RESULTS_DIR}")
    print("\nBase vs multicalibrated test metrics:")
    print(f"  base_test_rmse = {metrics['base_test']['rmse']:.3f}")
    print(f"  mc_test_rmse   = {metrics['mc_test']['rmse']:.3f}")
    print(f"  base_test_mae  = {metrics['base_test']['mae']:.3f}")
    print(f"  mc_test_mae    = {metrics['mc_test']['mae']:.3f}")
    print("\nWorst subgroup gaps on test after calibration:")
    if test_report.empty:
        print("  No test subgroup report generated.")
    else:
        print(
            test_report[
                ["group", "n", "true_mean", "pred_mean_before", "pred_mean_after", "abs_gap_before", "abs_gap_after"]
            ]
            .head(10)
            .to_string(index=False)
        )


if __name__ == "__main__":
    main()
