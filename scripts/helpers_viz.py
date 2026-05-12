#!/usr/bin/env python3
"""
helpers_viz.py
==============
Visualization and table helpers for the SVR-PHQ-9 results.

Designed to be imported from a notebook or called standalone.  All public
functions accept a `results_dir` (Path or str) pointing to one timestamped
subdirectory under ``results/`` (e.g. ``results/2026-05-09_175958_PDT``) and
return either a ``matplotlib.figure.Figure`` or a ``pandas.DataFrame`` — you
decide whether to save or display.

Quick usage in a notebook::

    from pathlib import Path
    import helpers_viz as viz

    RD = Path("../results/2026-05-09_175958_PDT")

    fig = viz.plot_model_comparison(RD)
    fig = viz.plot_pi_coverage(RD)
    fig = viz.plot_pred_vs_true(RD, run="egemaps_whisper_demo")
    fig = viz.plot_perm_importance(RD, top_n=15)
    fig = viz.plot_perm_importance_by_modality(RD)
    fig = viz.plot_feature_counts(RD)

    df  = viz.table_model_comparison(RD)
    df  = viz.table_perm_importance(RD, top_n=20)
    df  = viz.table_per_subject(RD, run="egemaps_whisper_demo")

Public API
----------
Loaders (return DataFrames, cached for the session):
    load_run_results          svr_run_results.csv
    load_participant_results  svr_participant_results.csv
    load_perm_importance      svr_perm_importance.csv

Figures:
    plot_model_comparison           bar chart of RMSE / MAE / R² across runs
    plot_pi_coverage                PI coverage & width across runs
    plot_pred_vs_true               scatter of predicted vs true PHQ-9
    plot_perm_importance            horizontal bar chart of top-N features
    plot_perm_importance_by_modality stacked bar of importance summed by modality
    plot_feature_counts             grouped bar of eGeMAPS / whisper features selected

Tables (return styled DataFrames ready for display or LaTeX export):
    table_model_comparison   per-run metrics table, delta-RMSE column included
    table_perm_importance    top-N feature importance table
    table_per_subject        per-subject predictions + PI for one run
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Aesthetic defaults  (override before calling any plot function if needed)
# ---------------------------------------------------------------------------
MODALITY_COLORS: dict[str, str] = {
    "egemaps":       "#4C72B0",
    "whisper":       "#DD8452",
    "demographics":  "#55A868",
    "other":         "#C44E52",
}

RUN_DISPLAY_NAMES: dict[str, str] = {
    "mean_predictor":       "Mean predictor\n(baseline)",
    "egemaps_only":         "eGeMAPS only",
    "whisper_only":         "Whisper only",
    "egemaps_demo":         "eGeMAPS + Demo",
    "whisper_demo":         "Whisper + Demo",
    "egemaps_whisper":      "eGeMAPS + Whisper",
    "egemaps_whisper_demo": "eGeMAPS + Whisper\n+ Demo",
}

_FIGURE_DPI = 150
_FONT_SIZE   = 10
plt.rcParams.update({
    "font.size":        _FONT_SIZE,
    "axes.titlesize":   _FONT_SIZE + 1,
    "axes.labelsize":   _FONT_SIZE,
    "xtick.labelsize":  _FONT_SIZE - 1,
    "ytick.labelsize":  _FONT_SIZE - 1,
    "legend.fontsize":  _FONT_SIZE - 1,
    "figure.dpi":       _FIGURE_DPI,
})


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _resolve(results_dir: str | Path) -> Path:
    return Path(results_dir).expanduser().resolve()


def _run_label(run: str) -> str:
    return RUN_DISPLAY_NAMES.get(run, run.replace("_", " "))


def _modality_color(modality: str) -> str:
    for key, color in MODALITY_COLORS.items():
        if modality.startswith(key):
            return color
    return MODALITY_COLORS["other"]


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
@lru_cache(maxsize=16)
def load_run_results(results_dir: str | Path) -> pd.DataFrame:
    """
    Load ``svr_run_results.csv`` from *results_dir* and parse JSON columns.

    JSON string columns (``best_params``, ``feature_matrices_used``,
    ``selected_features``, ``top_5_perm_importance``) are decoded into Python
    objects and stored under the same column name.

    Returns a DataFrame with one row per run.
    """
    path = _resolve(results_dir) / "svr_run_results.csv"
    df = pd.read_csv(path)
    for col in ("best_params", "feature_matrices_used", "selected_features"):
        if col in df.columns:
            df[col] = df[col].apply(lambda x: json.loads(x) if pd.notna(x) else x)
    # top_N_perm_importance — column name may vary with top_n_perm parameter
    top_cols = [c for c in df.columns if c.startswith("top_") and "perm_importance" in c]
    for col in top_cols:
        df[col] = df[col].apply(lambda x: json.loads(x) if pd.notna(x) else x)
    return df


@lru_cache(maxsize=16)
def load_participant_results(results_dir: str | Path) -> pd.DataFrame:
    """
    Load ``svr_participant_results.csv`` from *results_dir*.

    Returns a DataFrame with one row per (run, subject) pair.
    """
    path = _resolve(results_dir) / "svr_participant_results.csv"
    return pd.read_csv(path)


@lru_cache(maxsize=16)
def load_perm_importance(results_dir: str | Path) -> pd.DataFrame:
    """
    Load ``svr_perm_importance.csv`` from *results_dir*.

    Returns a DataFrame sorted descending by ``perm_imp_mse_increase_train``,
    one row per feature in the best run's X matrix.
    """
    path = _resolve(results_dir) / "svr_perm_importance.csv"
    df = pd.read_csv(path)
    return df.sort_values("perm_imp_mse_increase_train", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Figure 1 — Model comparison (RMSE / MAE / R²)
# ---------------------------------------------------------------------------
def plot_model_comparison(
    results_dir: str | Path,
    *,
    metrics: list[str] | None = None,
    exclude_baseline: bool = False,
    figsize: tuple[float, float] | None = None,
) -> plt.Figure:
    """
    Grouped bar chart comparing RMSE, MAE, and R² across all runs.

    Parameters
    ----------
    results_dir:
        Path to the timestamped results directory.
    metrics:
        Subset of ``["RMSE", "MAE", "R2"]`` to plot.  Defaults to all three.
    exclude_baseline:
        If True, the ``mean_predictor`` row is omitted.
    figsize:
        Override the default figure size.

    Returns
    -------
    matplotlib Figure
    """
    if metrics is None:
        metrics = ["RMSE", "MAE", "R2"]

    df = load_run_results(results_dir)
    if exclude_baseline:
        df = df[df["run"] != "mean_predictor"].reset_index(drop=True)

    runs   = df["run"].tolist()
    labels = [_run_label(r) for r in runs]
    x      = np.arange(len(runs))
    width  = 0.25
    n_met  = len(metrics)
    offsets = np.linspace(-(n_met - 1) / 2, (n_met - 1) / 2, n_met) * width

    fig_w = max(8, len(runs) * 1.4)
    fig, ax = plt.subplots(figsize=figsize or (fig_w, 4.5))

    palette = ["#4C72B0", "#DD8452", "#55A868"]
    for k, (metric, offset, color) in enumerate(zip(metrics, offsets, palette)):
        vals = df[metric].tolist()
        bars = ax.bar(x + offset, vals, width, label=metric, color=color, alpha=0.85)
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.03,
                f"{val:.2f}",
                ha="center", va="bottom", fontsize=7, rotation=0,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Model comparison — nested-LOO regression on PHQ-9 (n=52)")
    ax.legend(loc="upper right")
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 2 — Prediction interval coverage & width
# ---------------------------------------------------------------------------
def plot_pi_coverage(
    results_dir: str | Path,
    *,
    exclude_baseline: bool = False,
    figsize: tuple[float, float] | None = None,
) -> plt.Figure:
    """
    Two-panel figure: (left) empirical PI coverage per run with target line;
    (right) mean PI width per run.

    Parameters
    ----------
    results_dir:
        Path to the timestamped results directory.
    exclude_baseline:
        If True, ``mean_predictor`` is excluded.

    Returns
    -------
    matplotlib Figure
    """
    df = load_run_results(results_dir)
    if exclude_baseline:
        df = df[df["run"] != "mean_predictor"].reset_index(drop=True)

    runs   = df["run"].tolist()
    labels = [_run_label(r) for r in runs]
    x      = np.arange(len(runs))
    alpha  = float(df["PI_alpha"].iloc[0]) if "PI_alpha" in df.columns else 0.05
    target = 1.0 - alpha

    fig_w = max(10, len(runs) * 1.4)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize or (fig_w, 4.5))

    # — coverage —
    ax1.bar(x, df["PI_coverage"], color="#4C72B0", alpha=0.85)
    ax1.axhline(target, color="red", linewidth=1.4, linestyle="--",
                label=f"target {target:.0%}")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=20, ha="right")
    ax1.set_ylim(0, 1.05)
    ax1.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax1.set_ylabel("Empirical coverage")
    ax1.set_title(f"PI coverage  (target = {target:.0%})")
    ax1.legend()

    # — width —
    ax2.bar(x, df["PI_mean_width"], color="#DD8452", alpha=0.85)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=20, ha="right")
    ax2.set_ylabel("Mean PI width  (PHQ-9 units)")
    ax2.set_title("PI mean width  (lower = tighter)")

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 3 — Predicted vs. true PHQ-9 (per run)
# ---------------------------------------------------------------------------
def plot_pred_vs_true(
    results_dir: str | Path,
    *,
    run: str | None = None,
    show_pi: bool = True,
    figsize: tuple[float, float] | None = None,
) -> plt.Figure:
    """
    Scatter plot of OOF predicted vs. true PHQ-9 for one run.

    Parameters
    ----------
    results_dir:
        Path to the timestamped results directory.
    run:
        Run name to plot.  Defaults to the run with the lowest RMSE
        (excluding ``mean_predictor``).
    show_pi:
        If True, draw vertical error bars for the jackknife PI.

    Returns
    -------
    matplotlib Figure
    """
    run_df = load_run_results(results_dir)
    part   = load_participant_results(results_dir)

    if run is None:
        candidates = run_df[run_df["run"] != "mean_predictor"]
        run = candidates.loc[candidates["RMSE"].idxmin(), "run"]

    sub = part[part["run"] == run].copy()
    if sub.empty:
        raise ValueError(f"Run '{run}' not found in participant results.")

    rmse = float(run_df.loc[run_df["run"] == run, "RMSE"].iloc[0])
    r2   = float(run_df.loc[run_df["run"] == run, "R2"].iloc[0])

    lo   = sub["pi_lower"].to_numpy()
    hi   = sub["pi_upper"].to_numpy()
    err  = np.array([sub["y_pred"] - lo, hi - sub["y_pred"]])

    fig, ax = plt.subplots(figsize=figsize or (5.5, 5.5))

    color_in  = "#4C72B0"
    color_out = "#C44E52"
    colors = sub["in_interval"].map({True: color_in, False: color_out}).tolist()

    if show_pi:
        ax.errorbar(
            sub["y_true"], sub["y_pred"],
            yerr=err,
            fmt="none", ecolor="lightgrey", capsize=3, linewidth=0.8, zorder=1,
        )
    ax.scatter(
        sub["y_true"], sub["y_pred"],
        c=colors, s=45, zorder=2, edgecolors="white", linewidths=0.4,
    )

    lims = [
        min(sub["y_true"].min(), sub["y_pred"].min()) - 1,
        max(sub["y_true"].max(), sub["y_pred"].max()) + 1,
    ]
    ax.plot(lims, lims, "k--", linewidth=1, label="y = x  (perfect)")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("True PHQ-9")
    ax.set_ylabel("Predicted PHQ-9  (OOF)")
    ax.set_title(
        f"{_run_label(run)}\n"
        f"RMSE={rmse:.2f}, R²={r2:.3f}  "
        f"({int(sub['in_interval'].sum())}/{len(sub)} inside 95% PI)",
        fontsize=_FONT_SIZE,
    )
    # legend handles
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=color_in,
               markersize=7, label="Inside PI"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=color_out,
               markersize=7, label="Outside PI"),
        Line2D([0], [0], color="k", linestyle="--", label="y = x"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=8)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 4 — Permutation importance (top-N features, best run)
# ---------------------------------------------------------------------------
def plot_perm_importance(
    results_dir: str | Path,
    *,
    top_n: int = 20,
    selected_only: bool = True,
    figsize: tuple[float, float] | None = None,
) -> plt.Figure:
    """
    Horizontal bar chart of training-set permutation importance for the best
    run (stored in ``svr_perm_importance.csv``).

    Parameters
    ----------
    results_dir:
        Path to the timestamped results directory.
    top_n:
        How many top features to show.
    selected_only:
        If True (default), show only EN-selected features.

    Returns
    -------
    matplotlib Figure
    """
    imp = load_perm_importance(results_dir)
    if selected_only:
        imp = imp[imp["selected_by_EN"]].reset_index(drop=True)
    imp = imp.head(top_n).iloc[::-1].reset_index(drop=True)  # plot bottom→top

    colors = [_modality_color(m) for m in imp["modality"]]

    fig_h = max(4, top_n * 0.38)
    fig, ax = plt.subplots(figsize=figsize or (8, fig_h))

    y_pos = np.arange(len(imp))
    ax.barh(
        y_pos,
        imp["perm_imp_mse_increase_train"],
        xerr=imp["perm_imp_mse_increase_train_std"],
        color=colors, alpha=0.85, capsize=3, error_kw={"linewidth": 0.8},
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(imp["feature"], fontsize=8)
    ax.set_xlabel("MSE increase under permutation  (training set)")
    ax.set_title(f"Permutation importance — top {len(imp)} EN-selected features")

    # modality legend
    seen: set[str] = set()
    legend_handles = []
    from matplotlib.patches import Patch
    for mod in imp["modality"]:
        key = next((k for k in MODALITY_COLORS if mod.startswith(k)), "other")
        if key not in seen:
            seen.add(key)
            legend_handles.append(Patch(facecolor=MODALITY_COLORS[key], label=key))
    ax.legend(handles=legend_handles, loc="lower right", fontsize=8)

    ax.axvline(0, color="black", linewidth=0.5)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 5 — Permutation importance summed by modality
# ---------------------------------------------------------------------------
def plot_perm_importance_by_modality(
    results_dir: str | Path,
    *,
    selected_only: bool = True,
    figsize: tuple[float, float] | None = None,
) -> plt.Figure:
    """
    Bar chart of total permutation importance aggregated by feature modality.

    Parameters
    ----------
    results_dir:
        Path to the timestamped results directory.
    selected_only:
        If True (default), restrict to EN-selected features.

    Returns
    -------
    matplotlib Figure
    """
    imp = load_perm_importance(results_dir)
    if selected_only:
        imp = imp[imp["selected_by_EN"]]

    # Collapse fine-grained modality labels to top-level buckets
    def _bucket(m: str) -> str:
        if m.startswith("egemaps"):     return "egemaps"
        if m.startswith("whisper"):     return "whisper"
        if m.startswith("demographics"): return "demographics"
        return "other"

    imp = imp.copy()
    imp["modality_bucket"] = imp["modality"].apply(_bucket)
    agg = (
        imp.groupby("modality_bucket")["perm_imp_mse_increase_train"]
        .sum()
        .sort_values(ascending=False)
    )

    fig, ax = plt.subplots(figsize=figsize or (5, 3.5))
    colors = [_modality_color(m) for m in agg.index]
    ax.bar(agg.index, agg.values, color=colors, alpha=0.85)
    for i, (label, val) in enumerate(zip(agg.index, agg.values)):
        ax.text(i, val + 0.05, f"{val:.1f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Total MSE increase under permutation")
    ax.set_title("Permutation importance by modality\n(EN-selected features, training set)")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 6 — Number of features selected per run (eGeMAPS vs Whisper)
# ---------------------------------------------------------------------------
def plot_feature_counts(
    results_dir: str | Path,
    *,
    exclude_baseline: bool = True,
    figsize: tuple[float, float] | None = None,
) -> plt.Figure:
    """
    Grouped bar chart of the number of EN-selected eGeMAPS and Whisper
    features per run.

    Parameters
    ----------
    results_dir:
        Path to the timestamped results directory.
    exclude_baseline:
        If True (default), ``mean_predictor`` is excluded.

    Returns
    -------
    matplotlib Figure
    """
    df = load_run_results(results_dir)
    if exclude_baseline:
        df = df[df["run"] != "mean_predictor"].reset_index(drop=True)

    runs   = df["run"].tolist()
    labels = [_run_label(r) for r in runs]
    x      = np.arange(len(runs))
    width  = 0.35

    fig_w = max(8, len(runs) * 1.4)
    fig, ax = plt.subplots(figsize=figsize or (fig_w, 4))

    ax.bar(x - width / 2, df["n_egemaps_selected"], width,
           label="eGeMAPS", color=MODALITY_COLORS["egemaps"], alpha=0.85)
    ax.bar(x + width / 2, df["n_whisper_selected"], width,
           label="Whisper", color=MODALITY_COLORS["whisper"], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Features selected by ElasticNet")
    ax.set_title("EN-selected feature counts per run")
    ax.legend()
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 7 — PHQ-9 distribution in the audio sub-cohort (HC vs MDD)
# ---------------------------------------------------------------------------
PHQ9_GROUP_COLORS: dict[str, str] = {
    "HC":  "#4C72B0",   # seaborn-deep blue (matches HC color used in plot_pred_vs_true)
    "MDD": "#C44E52",   # seaborn-deep red  (matches MDD color used in plot_pred_vs_true)
}


def plot_phq9_distribution(
    metadata_csv: str | Path,
    *,
    figsize: tuple[float, float] | None = None,
) -> plt.Figure:
    """
    Stacked histogram of PHQ-9 scores in the MODMA audio sub-cohort, split by
    HC vs MDD.  Uses the same blue/red pair as the per-subject scatter plots
    so the HC↔blue, MDD↔red association is consistent across the report.
    A dotted vertical line at PHQ-9 = 5 marks the MDD-cohort inclusion
    threshold.

    Parameters
    ----------
    metadata_csv:
        Path to ``data/metadata/subject_info_map.csv`` (or equivalent).  Must
        contain columns ``PHQ-9`` and ``group`` with values ``HC`` and ``MDD``.
    figsize:
        Override the default figure size.

    Returns
    -------
    matplotlib Figure
    """
    df = pd.read_csv(_resolve(metadata_csv))
    PHQ9_MAX = 27  # theoretical PHQ-9 max (9 items × 3 max each)
    bins = range(0, PHQ9_MAX + 1)

    hc  = df.loc[df["group"] == "HC",  "PHQ-9"].to_numpy()
    mdd = df.loc[df["group"] == "MDD", "PHQ-9"].to_numpy()

    fig, ax = plt.subplots(figsize=figsize or (7, 4))

    ax.hist(
        [hc, mdd], bins=bins, stacked=True,
        color=[PHQ9_GROUP_COLORS["HC"], PHQ9_GROUP_COLORS["MDD"]],
        edgecolor="white", linewidth=0.4,
        label=[f"HC  (n={len(hc)})", f"MDD  (n={len(mdd)})"],
    )

    ax.axvline(
        5, color="black", linestyle=":", linewidth=1.2,
        label="MDD inclusion (≥5)",
    )

    ax.set_xlabel("PHQ-9 score")
    ax.set_ylabel("Number of subjects")
    ax.set_title(f"PHQ-9 distribution — MODMA audio sub-cohort (n={len(df)})")
    ax.set_xticks([0, 5, 10, 15, 20, 25, 27])
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


def plot_demographics_distribution(
    metadata_csv: str | Path,
    *,
    figsize: tuple[float, float] | None = None,
) -> plt.Figure:
    """
    Three-panel stacked histogram of age, gender, and education-years in the
    MODMA audio sub-cohort, split by HC vs MDD.  Uses the same blue/red HC/MDD
    pair as ``plot_phq9_distribution`` so the colour semantics are consistent
    across the report.

    Parameters
    ----------
    metadata_csv:
        Path to ``data/metadata/subject_info_map.csv`` (or equivalent).  Must
        contain columns ``group`` (``HC``/``MDD``), ``age``, ``gender``
        (``F``/``M``), and ``edu_years``.
    figsize:
        Override the default figure size.

    Returns
    -------
    matplotlib Figure
    """
    df = pd.read_csv(_resolve(metadata_csv))

    hc_mask  = df["group"] == "HC"
    mdd_mask = df["group"] == "MDD"
    n_hc, n_mdd = int(hc_mask.sum()), int(mdd_mask.sum())

    hc_color  = PHQ9_GROUP_COLORS["HC"]
    mdd_color = PHQ9_GROUP_COLORS["MDD"]

    fig, axes = plt.subplots(1, 3, figsize=figsize or (12, 3.5))

    # --- Age ---
    ax = axes[0]
    age_bins = range(15, 56, 5)
    ax.hist(
        [df.loc[hc_mask, "age"], df.loc[mdd_mask, "age"]],
        bins=age_bins, stacked=True,
        color=[hc_color, mdd_color],
        edgecolor="white", linewidth=0.4,
    )
    ax.set_xlabel("Age (years)")
    ax.set_ylabel("Number of subjects")
    ax.set_title("Age")
    ax.set_xticks(list(age_bins))

    # --- Gender (stacked bars over categorical F/M) ---
    ax = axes[1]
    genders = ["F", "M"]
    hc_counts  = [int((hc_mask  & (df["gender"] == g)).sum()) for g in genders]
    mdd_counts = [int((mdd_mask & (df["gender"] == g)).sum()) for g in genders]
    x = list(range(len(genders)))
    ax.bar(x, hc_counts, color=hc_color,
           edgecolor="white", linewidth=0.4)
    ax.bar(x, mdd_counts, bottom=hc_counts, color=mdd_color,
           edgecolor="white", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(genders)
    ax.set_xlabel("Gender")
    ax.set_ylabel("Number of subjects")
    ax.set_title("Gender")

    # --- Education ---
    ax = axes[2]
    edu_bins = range(6, 25, 2)
    ax.hist(
        [df.loc[hc_mask, "edu_years"], df.loc[mdd_mask, "edu_years"]],
        bins=edu_bins, stacked=True,
        color=[hc_color, mdd_color],
        edgecolor="white", linewidth=0.4,
    )
    ax.set_xlabel("Education (years)")
    ax.set_ylabel("Number of subjects")
    ax.set_title("Education")
    ax.set_xticks(list(edu_bins))

    # Shared legend at the figure level (one entry per group)
    hc_patch  = mpatches.Patch(color=hc_color,  label=f"HC  (n={n_hc})")
    mdd_patch = mpatches.Patch(color=mdd_color, label=f"MDD  (n={n_mdd})")
    fig.legend(
        handles=[hc_patch, mdd_patch],
        loc="upper right", bbox_to_anchor=(0.995, 0.995),
        frameon=True, ncol=2,
    )

    fig.suptitle(
        f"Demographic distributions — MODMA audio sub-cohort (n={len(df)})",
        y=1.00,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig


# ---------------------------------------------------------------------------
# Table 1 — Model comparison
# ---------------------------------------------------------------------------
def table_model_comparison(
    results_dir: str | Path,
    *,
    baseline_run: str = "mean_predictor",
    round_digits: int = 3,
) -> pd.DataFrame:
    """
    Return a clean per-run metrics DataFrame suitable for display or LaTeX
    export.  Includes a ΔRMSE column relative to *baseline_run*.

    Columns: run, n_features, RMSE, MAE, R², PI_coverage, PI_mean_width, ΔRMSE

    Parameters
    ----------
    results_dir:
        Path to the timestamped results directory.
    baseline_run:
        Run name to use as the RMSE reference.  Pass ``None`` to omit ΔRMSE.
    round_digits:
        Decimal places for numeric columns.

    Returns
    -------
    pandas DataFrame (not styled — call ``.style`` yourself if needed)
    """
    df = load_run_results(results_dir)
    keep = ["run", "n_features", "RMSE", "MAE", "R2", "PI_coverage", "PI_mean_width"]
    out  = df[keep].copy()
    out.rename(columns={"R2": "R²"}, inplace=True)

    if baseline_run and baseline_run in df["run"].values:
        base_rmse = float(df.loc[df["run"] == baseline_run, "RMSE"].iloc[0])
        out["ΔRMSE"] = (out["RMSE"] - base_rmse).round(round_digits)
        out.loc[out["run"] == baseline_run, "ΔRMSE"] = float("nan")

    for col in ("RMSE", "MAE", "R²", "PI_coverage", "PI_mean_width"):
        if col in out.columns:
            out[col] = out[col].round(round_digits)

    out["run"] = out["run"].apply(_run_label)
    out = out.rename(columns={"run": "Run"})
    out = out.reset_index(drop=True)
    return out


# ---------------------------------------------------------------------------
# Table 2 — Permutation importance
# ---------------------------------------------------------------------------
def table_perm_importance(
    results_dir: str | Path,
    *,
    top_n: int = 20,
    selected_only: bool = True,
    round_digits: int = 3,
) -> pd.DataFrame:
    """
    Return a clean permutation-importance DataFrame.

    Columns: feature, modality, selected_by_EN, importance (mean ± std)

    Parameters
    ----------
    results_dir:
        Path to the timestamped results directory.
    top_n:
        Number of top features to include.
    selected_only:
        If True, restrict to EN-selected features.
    round_digits:
        Decimal places for numeric columns.

    Returns
    -------
    pandas DataFrame
    """
    imp = load_perm_importance(results_dir)
    if selected_only:
        imp = imp[imp["selected_by_EN"]].reset_index(drop=True)
    imp = imp.head(top_n).reset_index(drop=True)

    out = imp[["feature", "modality", "selected_by_EN",
               "perm_imp_mse_increase_train",
               "perm_imp_mse_increase_train_std"]].copy()
    out.rename(columns={
        "perm_imp_mse_increase_train":     "MSE increase (mean)",
        "perm_imp_mse_increase_train_std": "MSE increase (std)",
        "selected_by_EN":                  "EN selected",
    }, inplace=True)
    out["MSE increase (mean)"] = out["MSE increase (mean)"].round(round_digits)
    out["MSE increase (std)"]  = out["MSE increase (std)"].round(round_digits)
    out.index = range(1, len(out) + 1)
    return out


# ---------------------------------------------------------------------------
# Table 3 — Per-subject predictions
# ---------------------------------------------------------------------------
def table_per_subject(
    results_dir: str | Path,
    *,
    run: str | None = None,
    round_digits: int = 2,
) -> pd.DataFrame:
    """
    Return a per-subject prediction table for one run.

    Columns: subject_id, y_true, y_pred, residual, pi_lower, pi_upper, in_PI

    Parameters
    ----------
    results_dir:
        Path to the timestamped results directory.
    run:
        Run name to retrieve.  Defaults to the best (lowest RMSE) non-baseline
        run.
    round_digits:
        Decimal places for float columns.

    Returns
    -------
    pandas DataFrame sorted by subject_id
    """
    run_df = load_run_results(results_dir)
    part   = load_participant_results(results_dir)

    if run is None:
        candidates = run_df[run_df["run"] != "mean_predictor"]
        run = candidates.loc[candidates["RMSE"].idxmin(), "run"]

    sub = part[part["run"] == run].sort_values("subject_id").copy()
    if sub.empty:
        raise ValueError(f"Run '{run}' not found in participant results.")

    sub["residual"] = (sub["y_true"] - sub["y_pred"]).round(round_digits)
    out = sub[["subject_id", "y_true", "y_pred", "residual",
               "pi_lower", "pi_upper", "in_interval"]].copy()
    out.rename(columns={"in_interval": "in_PI"}, inplace=True)
    for col in ("y_true", "y_pred", "pi_lower", "pi_upper"):
        out[col] = out[col].round(round_digits)
    out = out.reset_index(drop=True)
    return out


# ---------------------------------------------------------------------------
# Convenience: save all figures to a directory
# ---------------------------------------------------------------------------
def save_all_figures(
    results_dir: str | Path,
    output_dir: str | Path | None = None,
    *,
    fmt: str = "pdf",
    dpi: int = 300,
) -> list[Path]:
    """
    Generate every figure and save to *output_dir* (defaults to
    ``<results_dir>/figures/``).

    Returns a list of saved file paths.

    Parameters
    ----------
    results_dir:
        Path to the timestamped results directory.
    output_dir:
        Destination folder.  Created if it doesn't exist.
    fmt:
        File format passed to ``savefig`` (e.g. ``"pdf"``, ``"png"``).
    dpi:
        Resolution for raster formats.

    Returns
    -------
    list of Path objects
    """
    rd = _resolve(results_dir)
    if output_dir is None:
        out_dir = rd / "figures"
    else:
        out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks: list[tuple[str, callable]] = [
        ("model_comparison",            lambda: plot_model_comparison(rd)),
        ("pi_coverage",                 lambda: plot_pi_coverage(rd)),
        ("pred_vs_true_best_run",       lambda: plot_pred_vs_true(rd)),
        ("perm_importance_top20",       lambda: plot_perm_importance(rd, top_n=20)),
        ("perm_importance_by_modality", lambda: plot_perm_importance_by_modality(rd)),
        ("feature_counts",              lambda: plot_feature_counts(rd)),
    ]

    saved: list[Path] = []
    for name, fn in tasks:
        fig = fn()
        fpath = out_dir / f"{name}.{fmt}"
        fig.savefig(fpath, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved → {fpath}")
        saved.append(fpath)

    return saved
