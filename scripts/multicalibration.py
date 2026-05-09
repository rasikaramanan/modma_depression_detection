#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MultiCalibrationUpdate:
    group_name: str
    group_size: int
    mean_residual_before: float
    applied_shift: float


def _as_numpy_1d(values: Iterable[float], *, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1-D array-like, got shape {arr.shape}")
    return arr


def _validate_same_length(**arrays: np.ndarray) -> int:
    lengths = {name: len(arr) for name, arr in arrays.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"All arrays must have the same length, got {lengths}")
    return next(iter(lengths.values()))


def compute_score_bin_edges(scores: Iterable[float], *, n_bins: int) -> np.ndarray:
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")

    score_arr = _as_numpy_1d(scores, name="scores")
    if len(score_arr) == 0:
        raise ValueError("scores must be non-empty")

    try:
        _, edges = pd.qcut(
            score_arr,
            q=n_bins,
            duplicates="drop",
            retbins=True,
        )
    except ValueError as exc:
        raise ValueError("Unable to compute score-bin edges") from exc

    edges = np.asarray(edges, dtype=float)
    if edges.ndim != 1 or len(edges) < 2:
        raise ValueError(f"Invalid score-bin edges: {edges}")
    return edges


def _score_bin_masks_from_edges(
    scores: np.ndarray,
    *,
    edges: np.ndarray,
    prefix: str = "score_bin",
) -> Dict[str, np.ndarray]:
    if len(scores) == 0:
        return {}

    if edges.ndim != 1 or len(edges) < 2:
        raise ValueError(f"edges must be 1-D with length >= 2, got {edges}")

    edges = edges.astype(float).copy()
    edges[0] = np.nextafter(edges[0], -np.inf)
    edges[-1] = np.nextafter(edges[-1], np.inf)

    cats = pd.cut(scores, bins=edges, include_lowest=True)
    masks: Dict[str, np.ndarray] = {}
    for idx, interval in enumerate(cats.categories):
        left = float(interval.left)
        right = float(interval.right)
        masks[f"{prefix}_{idx}=[{left:.2f}, {right:.2f}]"] = (cats == interval)
    return masks


def build_group_masks(
    *,
    n: int,
    groups: Mapping[str, Iterable[bool]] | None = None,
    group_frame: pd.DataFrame | None = None,
    scores: Iterable[float] | None = None,
    n_score_bins: int = 0,
    score_bin_edges: Iterable[float] | None = None,
    include_crosses: bool = True,
    min_group_size: int = 1,
) -> Dict[str, np.ndarray]:
    masks: Dict[str, np.ndarray] = {}

    if groups is not None:
        for name, mask in groups.items():
            arr = np.asarray(mask, dtype=bool)
            if arr.shape != (n,):
                raise ValueError(
                    f"Mask {name!r} must have shape {(n,)}, got {arr.shape}"
                )
            masks[name] = arr

    if group_frame is not None:
        if len(group_frame) != n:
            raise ValueError(
                f"group_frame must have {n} rows, got {len(group_frame)} rows"
            )
        for col in group_frame.columns:
            series = group_frame[col]
            if not (
                pd.api.types.is_object_dtype(series)
                or pd.api.types.is_categorical_dtype(series)
                or pd.api.types.is_bool_dtype(series)
            ):
                continue
            for value in series.dropna().unique():
                masks[f"{col}={value}"] = (series == value).to_numpy()

    score_masks: Dict[str, np.ndarray] = {}
    if scores is not None and score_bin_edges is not None:
        score_arr = _as_numpy_1d(scores, name="scores")
        if len(score_arr) != n:
            raise ValueError(f"scores must have length {n}, got {len(score_arr)}")
        edges = _as_numpy_1d(score_bin_edges, name="score_bin_edges")
        score_masks = _score_bin_masks_from_edges(score_arr, edges=edges)
        masks.update(score_masks)
    elif scores is not None and n_score_bins > 0:
        score_arr = _as_numpy_1d(scores, name="scores")
        if len(score_arr) != n:
            raise ValueError(f"scores must have length {n}, got {len(score_arr)}")
        edges = compute_score_bin_edges(score_arr, n_bins=n_score_bins)
        score_masks = _score_bin_masks_from_edges(score_arr, edges=edges)
        masks.update(score_masks)

    if include_crosses and score_masks:
        base_groups = {
            name: mask
            for name, mask in masks.items()
            if not name.startswith("score_bin")
        }
        for group_name, group_mask in base_groups.items():
            for score_name, score_mask in score_masks.items():
                masks[f"{group_name} & {score_name}"] = group_mask & score_mask

    filtered = {
        name: mask
        for name, mask in masks.items()
        if int(np.sum(mask)) >= min_group_size
    }
    return filtered


def mean_multicalibrate(
    *,
    y_true: Iterable[float],
    y_pred: Iterable[float],
    group_masks: Mapping[str, Iterable[bool]],
    tol: float = 1.0,
    max_iters: int = 100,
    step_size: float = 1.0,
    min_group_size: int = 1,
    clip: tuple[float, float] | None = None,
) -> dict[str, object]:
    y_true_arr = _as_numpy_1d(y_true, name="y_true")
    y_pred_arr = _as_numpy_1d(y_pred, name="y_pred")
    _validate_same_length(y_true=y_true_arr, y_pred=y_pred_arr)

    current = y_pred_arr.copy()
    updates: list[MultiCalibrationUpdate] = []
    history_rows: list[dict[str, float | int | str]] = []

    processed_masks: Dict[str, np.ndarray] = {}
    for name, mask in group_masks.items():
        arr = np.asarray(mask, dtype=bool)
        if arr.shape != (len(y_true_arr),):
            raise ValueError(
                f"group mask {name!r} must have shape {(len(y_true_arr),)}, "
                f"got {arr.shape}"
            )
        if int(np.sum(arr)) >= min_group_size:
            processed_masks[name] = arr

    if not processed_masks:
        raise ValueError("No valid group masks remained after filtering")

    for iteration in range(1, max_iters + 1):
        residuals = y_true_arr - current

        best_name = None
        best_mask = None
        best_gap = 0.0

        for name, mask in processed_masks.items():
            mean_residual = float(np.mean(residuals[mask]))
            if abs(mean_residual) > abs(best_gap):
                best_name = name
                best_mask = mask
                best_gap = mean_residual

        assert best_name is not None and best_mask is not None

        history_rows.append(
            {
                "iteration": iteration,
                "group": best_name,
                "group_size": int(np.sum(best_mask)),
                "max_abs_mean_residual": abs(best_gap),
                "signed_mean_residual": best_gap,
            }
        )

        if abs(best_gap) <= tol:
            break

        shift = step_size * best_gap
        current[best_mask] += shift
        if clip is not None:
            current = np.clip(current, clip[0], clip[1])

        updates.append(
            MultiCalibrationUpdate(
                group_name=best_name,
                group_size=int(np.sum(best_mask)),
                mean_residual_before=best_gap,
                applied_shift=shift,
            )
        )

    return {
        "y_pred_calibrated": current,
        "updates": updates,
        "history": pd.DataFrame(history_rows),
        "group_masks": processed_masks,
    }


def apply_multicalibration_updates(
    y_pred: Iterable[float],
    group_masks: Mapping[str, Iterable[bool]],
    updates: Iterable[MultiCalibrationUpdate],
    *,
    clip: tuple[float, float] | None = None,
) -> np.ndarray:
    current = _as_numpy_1d(y_pred, name="y_pred").copy()
    for update in updates:
        if update.group_name not in group_masks:
            continue
        mask = np.asarray(group_masks[update.group_name], dtype=bool)
        current[mask] += update.applied_shift
    if clip is not None:
        current = np.clip(current, clip[0], clip[1])
    return current


def multicalibration_report(
    *,
    y_true: Iterable[float],
    y_pred_before: Iterable[float],
    y_pred_after: Iterable[float],
    group_masks: Mapping[str, Iterable[bool]],
) -> pd.DataFrame:
    y_true_arr = _as_numpy_1d(y_true, name="y_true")
    before = _as_numpy_1d(y_pred_before, name="y_pred_before")
    after = _as_numpy_1d(y_pred_after, name="y_pred_after")
    _validate_same_length(y_true=y_true_arr, y_pred_before=before, y_pred_after=after)

    rows = []
    for name, mask in group_masks.items():
        arr = np.asarray(mask, dtype=bool)
        n_group = int(np.sum(arr))
        if n_group == 0:
            continue
        true_mean = float(np.mean(y_true_arr[arr]))
        pred_mean_before = float(np.mean(before[arr]))
        pred_mean_after = float(np.mean(after[arr]))
        rows.append(
            {
                "group": name,
                "n": n_group,
                "true_mean": true_mean,
                "pred_mean_before": pred_mean_before,
                "pred_mean_after": pred_mean_after,
                "gap_before": pred_mean_before - true_mean,
                "gap_after": pred_mean_after - true_mean,
                "abs_gap_before": abs(pred_mean_before - true_mean),
                "abs_gap_after": abs(pred_mean_after - true_mean),
                "mae_before": float(np.mean(np.abs(before[arr] - y_true_arr[arr]))),
                "mae_after": float(np.mean(np.abs(after[arr] - y_true_arr[arr]))),
            }
        )

    report = pd.DataFrame(rows)
    if report.empty:
        return report
    return report.sort_values(["abs_gap_after", "n"], ascending=[False, False])


def demo_group_frame(
    *,
    groups: Iterable[str] | None = None,
    genders: Iterable[str] | None = None,
    ages: Iterable[int] | None = None,
) -> pd.DataFrame:
    data: dict[str, Iterable[object]] = {}
    if groups is not None:
        data["group"] = groups
    if genders is not None:
        data["gender"] = genders
    if ages is not None:
        data["age"] = ages
    return pd.DataFrame(data)
