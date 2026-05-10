"""
analyze_intervals_and_groupings.py
======================
Generate the tables and figures backing Ari's results sections from a
train_svr.py results directory:

  * SVR results by elicitation task         (per-task runs, 4 tasks x 2 modalities)
  * SVR results by emotional valence        (per-valence runs, 3 valences x 2 modalities)
  * Prediction interval ablation analysis   (which feature combos tighten the PI?)

Inputs (read from --results-dir):
  - svr_run_results.csv         (one row per run; metrics + PI cov/width)
  - svr_participant_results.csv (one row per (run, subject); y_pred + PI bounds)

Outputs (written under --results-dir/figures/ and --results-dir/tables/):
  - figures/predicted_vs_actual_<run>.{png,pdf}     for best baseline / task / valence
  - figures/rmse_by_task.{png,pdf}                  per-task bar chart
  - figures/rmse_by_valence.{png,pdf}               per-valence bar chart
  - figures/pi_width_by_run.{png,pdf}               sorted PI-width comparison
  - figures/pi_width_vs_rmse.{png,pdf}              decoupling scatter
  - tables/by_task.tex
  - tables/by_valence.tex
  - tables/pi_ablation.tex

Usage:
    python analyze_intervals_and_groupings.py --results-dir results/2026-05-09_153021_PDT
    python analyze_intervals_and_groupings.py --results-dir <dir> --metadata-dir data/metadata
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch


# ------
# Constants — matched to make_per_task_runs / make_per_valence_runs in helpers_svr.py
# ------
TASKS      = ["interview", "passage_reading", "picture_description", "word_reading"]
VALENCES   = ["negative", "neutral", "positive"]
MODALITIES = ["egemaps", "whisper"]

CATEGORY_COLORS = {"baseline": "C0", "task": "C2", "valence": "C3"}


# ------
# Run-name parsing
# ------
def parse_run_name(run: str) -> tuple[str, str | None, str | None]:
    """
    Categorize a run name. Returns (category, subset_name, modality).

      'mean_predictor'                   -> ('baseline', None, None)
      'egemaps_only', 'whisper_demo' ... -> ('baseline', None, <prefix>)
      'egemaps_interview_only'           -> ('task',     'interview', 'egemaps')
      'whisper_negative_only'            -> ('valence',  'negative',  'whisper')

    The exact match against `f"{m}_{v}_only"` keeps full-corpus baselines like
    `egemaps_only` from being mistaken for a per-subset run.
    """
    for t in TASKS:
        for m in MODALITIES:
            if run == f"{m}_{t}_only":
                return ("task", t, m)
    for v in VALENCES:
        for m in MODALITIES:
            if run == f"{m}_{v}_only":
                return ("valence", v, m)
    return ("baseline", None, None)


def annotate_categories(runs: pd.DataFrame) -> pd.DataFrame:
    """Add 'category', 'subset', 'modality' columns derived from run name."""
    parsed = runs["run"].apply(parse_run_name).tolist()
    out = runs.copy()
    out[["category", "subset", "modality"]] = pd.DataFrame(parsed, index=out.index)
    return out


# ------
# Loading
# ------
def load_results(results_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    runs = pd.read_csv(results_dir / "svr_run_results.csv")
    parts = pd.read_csv(results_dir / "svr_participant_results.csv")
    return runs, parts


def load_subject_groups(metadata_csv: Path | None) -> pd.Series | None:
    """
    Try to load subject_id -> group ('MDD' / 'HC') from subject_info_map.csv.
    Returns None if the file isn't found — caller falls back to coloring by
    in_interval instead of group.
    """
    if metadata_csv is None or not metadata_csv.exists():
        return None
    info = pd.read_csv(metadata_csv)
    info["subject_id"] = info["subject_id"].apply(
        lambda s: int(str(s).lstrip("0") or "0")
    )
    return info.set_index("subject_id")["group"]


# ------
# Plots
# ------
def plot_predicted_vs_actual(
    participants: pd.DataFrame,
    run_row: pd.Series,
    groups: pd.Series | None,
    out_dir: Path,
) -> None:
    """
    Predicted vs actual PHQ-9 for one run, with 95% jackknife-conformal PI as
    vertical error bars. Color-coded by group (MDD/HC) when subject metadata
    is available, else by in_interval.
    """
    run_name = run_row["run"]
    sub = participants[participants["run"] == run_name].copy().reset_index(drop=True)
    if sub.empty:
        print(f"  [skip] no per-subject rows for run '{run_name}'")
        return

    if groups is not None:
        sub["group"] = sub["subject_id"].map(groups)

    fig, ax = plt.subplots(figsize=(8, 7))

    # y=x reference
    lo = float(min(sub["y_true"].min(), sub["pi_lower"].min()) - 1)
    hi = float(max(sub["y_true"].max(), sub["pi_upper"].max()) + 1)
    ax.plot([lo, hi], [lo, hi], "k--", alpha=0.4, label="y = x")

    def _errbars(mask, color, label):
        if not mask.any():
            return
        yerr = np.array([
            (sub.loc[mask, "y_pred"] - sub.loc[mask, "pi_lower"]).to_numpy(),
            (sub.loc[mask, "pi_upper"] - sub.loc[mask, "y_pred"]).to_numpy(),
        ])
        ax.errorbar(
            sub.loc[mask, "y_true"], sub.loc[mask, "y_pred"], yerr=yerr,
            fmt="o", color=color, alpha=0.75, label=label, capsize=2,
            markersize=6, ecolor=color, elinewidth=0.8,
        )

    if groups is not None and "group" in sub.columns and sub["group"].notna().any():
        _errbars(sub["group"] == "MDD", "C3", "MDD")
        _errbars(sub["group"] == "HC",  "C0", "HC")
    else:
        _errbars(sub["in_interval"] == True,  "C2", "covered")
        _errbars(sub["in_interval"] == False, "C3", "missed")

    title = (
        f"{run_name}\n"
        f"RMSE={run_row['RMSE']:.2f}  "
        f"MAE={run_row['MAE']:.2f}  "
        f"R\u00b2={run_row['R2']:.2f}  "
        f"PI cov={run_row['PI_coverage']*100:.1f}%  "
        f"PI width={run_row['PI_mean_width']:.1f}"
    )
    ax.set_xlabel("Actual PHQ-9")
    ax.set_ylabel("Predicted PHQ-9 (with 95% PI)")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()

    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"predicted_vs_actual_{run_name}.{ext}", dpi=150)
    plt.close(fig)
    print(f"  predicted_vs_actual_{run_name}")


def plot_rmse_by_subset(
    runs: pd.DataFrame,
    subset_kind: str,            # 'task' | 'valence'
    subset_values: list[str],
    out_dir: Path,
    baseline_rmse: float | None,
) -> None:
    """
    Bar chart of nested-LOO RMSE per subset, with paired bars for eGeMAPS vs
    Whisper. Mean-predictor baseline drawn as a horizontal dashed line.
    """
    eg_rmse = []
    wh_rmse = []
    for v in subset_values:
        eg_run = f"egemaps_{v}_only"
        wh_run = f"whisper_{v}_only"
        eg_rmse.append(runs.loc[runs["run"] == eg_run, "RMSE"].values[0]
                       if (runs["run"] == eg_run).any() else np.nan)
        wh_rmse.append(runs.loc[runs["run"] == wh_run, "RMSE"].values[0]
                       if (runs["run"] == wh_run).any() else np.nan)

    if all(np.isnan(eg_rmse)) and all(np.isnan(wh_rmse)):
        print(f"  [skip] no per-{subset_kind} runs found")
        return

    x = np.arange(len(subset_values))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width/2, eg_rmse, width, label="eGeMAPS", color="C0", edgecolor="black")
    ax.bar(x + width/2, wh_rmse, width, label="Whisper", color="C1", edgecolor="black")

    for xi, eg, wh in zip(x, eg_rmse, wh_rmse):
        if not np.isnan(eg):
            ax.text(xi - width/2, eg + 0.05, f"{eg:.2f}", ha="center", fontsize=9)
        if not np.isnan(wh):
            ax.text(xi + width/2, wh + 0.05, f"{wh:.2f}", ha="center", fontsize=9)

    if baseline_rmse is not None:
        ax.axhline(baseline_rmse, color="black", linestyle="--", alpha=0.6,
                   label=f"Mean predictor ({baseline_rmse:.2f})")

    ax.set_xticks(x)
    ax.set_xticklabels([v.replace("_", " ").title() for v in subset_values])
    ax.set_ylabel("Nested-LOO RMSE (PHQ-9 points)")
    ax.set_title(f"PHQ-9 prediction RMSE by {subset_kind}")
    ax.legend(loc="best")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()

    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"rmse_by_{subset_kind}.{ext}", dpi=150)
    plt.close(fig)
    print(f"  rmse_by_{subset_kind}")


def plot_pi_width_by_run(runs: pd.DataFrame, out_dir: Path) -> None:
    """Horizontal sorted bar chart of PI mean width across all runs."""
    df = annotate_categories(runs).sort_values("PI_mean_width").reset_index(drop=True)
    colors = [CATEGORY_COLORS[c] for c in df["category"]]

    fig, ax = plt.subplots(figsize=(10, max(5, 0.32 * len(df))))
    ax.barh(np.arange(len(df)), df["PI_mean_width"],
            color=colors, edgecolor="black", alpha=0.85)

    for i, w in enumerate(df["PI_mean_width"]):
        ax.text(w + 0.1, i, f"{w:.1f}", va="center", fontsize=8)

    ax.set_yticks(np.arange(len(df)))
    ax.set_yticklabels(df["run"], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("PI mean width (PHQ-9 points)")
    ax.set_title("95% jackknife-conformal PI width by run (smaller = better)")

    legend_handles = [Patch(color=c, label=cat) for cat, c in CATEGORY_COLORS.items()]
    ax.legend(handles=legend_handles, loc="upper right")
    ax.grid(alpha=0.3, axis="x")
    ax.set_xlim(0, df["PI_mean_width"].max() * 1.08)
    fig.tight_layout()

    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"pi_width_by_run.{ext}", dpi=150)
    plt.close(fig)
    print("  pi_width_by_run")


def plot_pi_width_vs_rmse(runs: pd.DataFrame, out_dir: Path) -> None:
    """Scatter showing decoupling of point accuracy (RMSE) from PI width."""
    df = annotate_categories(runs)

    fig, ax = plt.subplots(figsize=(9, 7))
    for cat, sub in df.groupby("category"):
        ax.scatter(sub["RMSE"], sub["PI_mean_width"],
                   color=CATEGORY_COLORS[cat], s=80, alpha=0.85,
                   edgecolor="black", label=cat)

    for _, row in df.iterrows():
        ax.annotate(row["run"], (row["RMSE"], row["PI_mean_width"]),
                    fontsize=7, alpha=0.7,
                    xytext=(4, 4), textcoords="offset points")

    ax.set_xlabel("Nested-LOO RMSE (PHQ-9 points)")
    ax.set_ylabel("95% PI mean width (PHQ-9 points)")
    ax.set_title("PI width vs RMSE: each run plotted")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()

    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"pi_width_vs_rmse.{ext}", dpi=150)
    plt.close(fig)
    print("  pi_width_vs_rmse")


# ------
# Tables
# ------
def _tex_escape(s: str) -> str:
    return s.replace("_", r"\_")


def write_table_by_subset(
    runs: pd.DataFrame,
    subset_kind: str,            # 'task' | 'valence'
    subset_values: list[str],
    tables_dir: Path,
) -> None:
    """LaTeX comparison table: rows = (subset, modality), cols = metrics."""
    rows = []
    for v in subset_values:
        for m in MODALITIES:
            run_name = f"{m}_{v}_only"
            mask = runs["run"] == run_name
            if not mask.any():
                continue
            r = runs.loc[mask].iloc[0]
            rows.append({
                "Subset": v.replace("_", " ").title(),
                "Modality": m,
                "n_features": int(r["n_features"]),
                "RMSE": r["RMSE"],
                "MAE": r["MAE"],
                "R2": r["R2"],
                "PI_coverage": r["PI_coverage"],
                "PI_width": r["PI_mean_width"],
            })
    if not rows:
        print(f"  [skip] no per-{subset_kind} runs to tabulate")
        return
    df = pd.DataFrame(rows)

    lines = [
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        f"{subset_kind.capitalize()} & Modality & $n_{{feat}}$ & RMSE & MAE & "
        r"$R^2$ & PI cov & PI width \\",
        r"\midrule",
    ]
    for _, row in df.iterrows():
        lines.append(
            f"{row['Subset']} & {row['Modality']} & {row['n_features']} & "
            f"{row['RMSE']:.2f} & {row['MAE']:.2f} & {row['R2']:.2f} & "
            f"{row['PI_coverage']*100:.1f}\\% & {row['PI_width']:.1f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    out = tables_dir / f"by_{subset_kind}.tex"
    out.write_text("\n".join(lines) + "\n")
    print(f"  wrote {out}")

    print(f"\n--- BY {subset_kind.upper()} ---")
    print(df.to_string(index=False))


def write_table_pi_ablation(runs: pd.DataFrame, tables_dir: Path) -> None:
    """All runs sorted by PI width ascending — answers 'which run has tightest PI?'"""
    df = annotate_categories(runs)[
        ["run", "category", "RMSE", "PI_coverage", "PI_mean_width"]
    ].sort_values("PI_mean_width").reset_index(drop=True)

    lines = [
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"Run & Category & RMSE & PI coverage & PI width \\",
        r"\midrule",
    ]
    for _, row in df.iterrows():
        lines.append(
            f"\\texttt{{{_tex_escape(row['run'])}}} & {row['category']} & "
            f"{row['RMSE']:.2f} & "
            f"{row['PI_coverage']*100:.1f}\\% & "
            f"{row['PI_mean_width']:.1f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    out = tables_dir / "pi_ablation.tex"
    out.write_text("\n".join(lines) + "\n")
    print(f"  wrote {out}")

    print("\n--- PI ABLATION (sorted by PI width) ---")
    print(df.to_string(index=False))


# ------
# Main
# ------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--results-dir", type=Path, required=True,
        help="Path to a timestamped results dir produced by train_svr.py")
    parser.add_argument("--metadata-dir", type=Path, default=Path("data/metadata"),
        help="Path to metadata dir containing subject_info_map.csv "
             "(for HC/MDD coloring on predicted-vs-actual plots). "
             "Default: data/metadata")
    args = parser.parse_args(argv)

    results_dir: Path = args.results_dir
    if not results_dir.exists():
        print(f"ERROR: {results_dir} does not exist", file=sys.stderr)
        return 1

    figures_dir = results_dir / "figures"
    tables_dir  = results_dir / "tables"
    figures_dir.mkdir(exist_ok=True)
    tables_dir.mkdir(exist_ok=True)

    runs, participants = load_results(results_dir)
    print(f"Loaded {len(runs)} runs and {len(participants)} per-participant rows")
    print(f"Runs present: {runs['run'].tolist()}")

    metadata_csv = (args.metadata_dir / "subject_info_map.csv"
                    if args.metadata_dir else None)
    groups = load_subject_groups(metadata_csv)
    if groups is None:
        print(f"  (subject groups not loaded from {metadata_csv}; "
              "predicted-vs-actual plots will color by in_interval)")
    else:
        print(f"  loaded subject groups for {len(groups)} subjects")

    # baseline RMSE for horizontal reference lines on bar charts
    baseline_rmse = None
    if (runs["run"] == "mean_predictor").any():
        baseline_rmse = float(runs.loc[runs["run"] == "mean_predictor", "RMSE"].values[0])

    runs_annot = annotate_categories(runs)

    # --- Figures ---
    print("\nGenerating figures...")

    # Predicted-vs-actual for the best non-mean-predictor run from each category
    for category, label in [("baseline", "best baseline"),
                            ("task",     "best per-task"),
                            ("valence",  "best per-valence")]:
        cat_runs = runs_annot[runs_annot["category"] == category]
        cat_runs = cat_runs[cat_runs["run"] != "mean_predictor"]
        if cat_runs.empty:
            continue
        best = cat_runs.loc[cat_runs["RMSE"].idxmin()]
        print(f"  ({label}: {best['run']})")
        plot_predicted_vs_actual(participants, best, groups, figures_dir)

    plot_rmse_by_subset(runs, "task", TASKS, figures_dir, baseline_rmse)
    plot_rmse_by_subset(runs, "valence", VALENCES, figures_dir, baseline_rmse)
    plot_pi_width_by_run(runs, figures_dir)
    plot_pi_width_vs_rmse(runs, figures_dir)

    # --- Tables ---
    print("\nGenerating LaTeX tables...")
    write_table_by_subset(runs, "task", TASKS, tables_dir)
    write_table_by_subset(runs, "valence", VALENCES, tables_dir)
    write_table_pi_ablation(runs, tables_dir)

    print(f"\nAll outputs written under {results_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())