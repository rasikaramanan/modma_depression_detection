#!/usr/bin/env python3
"""
helpers_svr.py
==============
Shared infrastructure for nested-LOO SVR-RBF regression on PHQ-9.

Consumed by `train_svr.py` — the main SVR analysis script. It exposes all
configured runs (baseline + whisper-fusion variants + per-task / per-valence
eGeMAPS and whisper ablations) through an interactive menu or via `--runs`.

The train script wires the helpers together by:
  1. defining a `RUN_FEATURE_SOURCES` dict (run name → ordered list of
     source-matrix names) and deriving `KNOWN_RUNS` from its keys,
  2. resolving `requested_runs` via CLI parse (parse_and_validate_args)
     and/or interactive menu (prompt_user_for_runs), then
     ensure_mean_predictor_included,
  3. loading data via load_subject_info_and_egemaps +
     load_feature_matrices_for_specs (lazy, scoped to selected runs),
  4. building per-run X matrices via build_run_matrices, then iterating
     run_one_configuration to fit + score + collect per-run posthoc data,
  5. emitting the printed comparison table + the three CSVs documented in
     `results/RESULTS_CSV_SCHEMAS.md` and the verbose post-hoc report.

Layout:
  Constants                  — shared paths, hyperparameters, feature lists
  Subject loading            — normalize_subject_id, load_subject_info,
                               make_run_results_dir
  Audio file map / slicing   — load_audio_file_map, discover_task_groups,
                               discover_valences, make_per_task_runs,
                               make_per_valence_runs
  Per-modality matrices      — build_egemaps/whisper/demographic_subject_matrix,
                               _summarize_counts
  Pipelines                  — make_pipeline_default, DoubleElasticNetSelector,
                               make_pipeline_double_en, pipeline_for_run
  Inner CV                   — nested_loo_predict, evaluate, jackknife PIs,
                               loo_mean_predictor_predictions
  Modality grouping          — modality_of
  CLI / interactive menu     — build_arg_parser, parse_and_validate_args,
                               parse_run_selection, prompt_user_for_runs,
                               ensure_mean_predictor_included
  Run-orchestration helpers  — load_subject_info_and_egemaps,
                               _maybe_warn_sample_size,
                               load_feature_matrices_for_specs,
                               prompt_for_sample_size_acknowledgment,
                               print_coverage_summary, build_run_matrices,
                               print_run_configurations, run_one_configuration,
                               print_results_table, save_results_csv,
                               save_predictions_csv
  Post-hoc inspection        — refit_best_on_all_subjects,
                               print_selection_summary,
                               _compute_perm_importance_df,
                               print_perm_importance_report,
                               compute_permutation_importance,
                               compute_posthoc_for_run,
                               run_posthoc_inspection
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT        = Path(__file__).resolve().parent.parent
EGEMAPS_CSV      = REPO_ROOT / "data" / "features" / "egemaps.csv"
WHISPER_PARQUET  = REPO_ROOT / "data" / "features" / "transcripts_features.parquet"
SUBJECT_INFO_CSV   = REPO_ROOT / "data" / "metadata" / "subject_info_map.csv"
ISSUES_CSV         = REPO_ROOT / "data" / "metadata" / "data_quality_issues.csv"
AUDIO_FILE_MAP_CSV = REPO_ROOT / "data" / "metadata" / "audio_file_map.csv"

# Each train-script invocation writes into a fresh timestamped subdir under
# RESULTS_BASE_DIR; the exact path is computed at runtime by make_run_results_dir().
RESULTS_BASE_DIR = REPO_ROOT / "results"
PACIFIC_TZ       = ZoneInfo("America/Los_Angeles")


# ---------------------------------------------------------------------------
# Hyperparameters / feature config
# ---------------------------------------------------------------------------
RANDOM_STATE = 42

WHISPER_FEATURE_COLS = [
    "lex_first_person_sg_rate",
    "lex_first_person_pl_rate",
    "lex_negation_rate",
    "lex_ttr",
    "lex_mattr50",
    "syn_mean_tokens_per_sent",
    "syn_sd_tokens_per_sent",
    "syn_punct_density",
    "sent_le_rate",
    "sent_hao_rate",
    "sent_nu_rate",
    "sent_ai_rate",
    "sent_ju_rate",
    "sent_e_rate",
    "sent_jing_rate",
    "sent_net_polarity",
]
# Files 1-18 of the MODMA stimulus protocol are "Interview" prompts (see
# data/metadata/audio_file_map.csv). Used by today's whisper_iv slice; will
# be retired in Prompt 3 in favor of discover_task_groups(...)["interview"].
INTERVIEW_FILE_RANGE = (1, 18)

# Threshold for the per-source sample-size guard added in Prompt 3
# (load_feature_matrices_for_specs warns when a slice yields any subject with
# fewer than this many contributing files). Kept here as a default so train
# scripts can override per-invocation if needed.
MIN_FILES_PER_SUBJECT_DEFAULT = 3

PARAM_GRID = {
    "svr__C":       [0.1, 1, 10, 100],
    "svr__epsilon": [0.1, 0.5, 1.0, 2.0],
    "svr__gamma":   ["scale", "auto"],
}


# ---------------------------------------------------------------------------
# Subject ID + info loading
# ---------------------------------------------------------------------------
def normalize_subject_id(s) -> int:
    """
    Convert a subject_id from any source representation into the canonical int
    form used as the join key throughout this module. Handles the zero-padded
    string convention from the MODMA metadata CSVs ('02010002' → 2010002) and
    is also a no-op on already-int input. An empty / all-zero string maps to
    0 — defensive, not expected in practice.

    Used everywhere subject IDs cross a CSV/parquet boundary: load_subject_info,
    build_egemaps_subject_matrix, build_whisper_subject_matrix.
    """
    s = str(s).lstrip("0")
    return int(s) if s else 0


def load_subject_info() -> pd.DataFrame:
    """
    Load `data/metadata/subject_info_map.csv` (path: SUBJECT_INFO_CSV) into a
    DataFrame indexed by int subject_id, carrying PHQ-9 / group / demographic
    columns (age, gender, edu_years, plus the auxiliary clinical scales).

    Wrapped by load_subject_info_and_egemaps for the standard load+align
    flow used by both train scripts; rarely called directly elsewhere.
    """
    info = pd.read_csv(SUBJECT_INFO_CSV)
    info["subject_id"] = info["subject_id"].apply(normalize_subject_id)
    return info.set_index("subject_id").sort_index()


def make_run_results_dir() -> Path:
    """
    Create and return a fresh, timestamped results directory under RESULTS_BASE_DIR.

    Subdir name format: 'YYYY-MM-DD_HHMMSS_<TZ>' in Pacific time, e.g.
    '2026-05-09_153021_PDT' (PDT in summer, PST in winter — Pacific time
    automatically resolves the correct abbreviation). Falls back to a
    counter-suffixed name if a same-second invocation collides.
    """
    now = datetime.now(PACIFIC_TZ)
    base = now.strftime("%Y-%m-%d_%H%M%S_%Z")
    candidate = RESULTS_BASE_DIR / base
    counter = 2
    while candidate.exists():
        candidate = RESULTS_BASE_DIR / f"{base}_{counter}"
        counter += 1
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


# ---------------------------------------------------------------------------
# Audio file map (stimulus-definition table) + slice discovery
# ---------------------------------------------------------------------------
# data/metadata/audio_file_map.csv is GLOBAL per-file_number, not per-subject:
# each subject performs the same N stimulus prompts in the same order, so the
# slice for a task or valence is just a set of file_numbers. The matrix
# builders below filter rows by file_number membership; subject scoping is
# enforced downstream by the join with subject_info.

def _slug(s: str) -> str:
    """Lowercase + collapse whitespace runs to underscores. Used so task names
    like 'Word Reading' become 'word_reading' (suitable for embedding in
    source-name strings like 'egemaps_task_word_reading')."""
    return "_".join(s.lower().split())


def load_audio_file_map() -> pd.DataFrame:
    """
    Load data/metadata/audio_file_map.csv — the global stimulus-definition
    table mapping file_number → (task, valence). Each subject performs the
    same set of files in the same order, so the table has no subject_id
    column.

    Cleanup applied:
      - file_number cast to int and used as the index.
      - task / valence columns whitespace-stripped and lowercased.
      - empty-string valence → NaN (a few files in the MODMA protocol are
        unvalenced, e.g. the phonetic Passage Reading and the TAT picture).

    Returns:
      DataFrame indexed by file_number (int), with at least 'task' (str) and
      'valence' (str | NaN) columns. Other columns from the CSV (e.g.
      task_number, notes) are passed through.
    """
    df = pd.read_csv(AUDIO_FILE_MAP_CSV, encoding="utf-8-sig")
    df["file_number"] = df["file_number"].astype(int)
    for col in ("task", "valence"):
        if col not in df.columns:
            continue
        normalized = df[col].astype("string").str.strip().str.lower()
        # treat empty strings (CSV blanks) as NaN so dropna() catches them
        normalized = normalized.where(normalized != "", pd.NA)
        df[col] = normalized
    return df.set_index("file_number").sort_index()


def discover_task_groups(audio_file_map: pd.DataFrame) -> dict[str, set[int]]:
    """
    Map each unique non-null task → set of file_numbers labeled with it.
    Task names are slugified ('Word Reading' → 'word_reading') so they're
    safe to embed in source-name strings.

    Insertion order: lexicographic on the slugified task name.

    NOTE: returns set[int] (file_numbers), not set[tuple[int, int]] (subject,
    file pairs). The audio file map is global per-file_number — every
    subject performs the same files — so the per-subject expansion is
    unnecessary and would just be a 52× cross-product. Subject scoping is
    enforced at matrix-build time by the join with subject_info.
    """
    tasks = sorted(audio_file_map["task"].dropna().unique())
    return {
        _slug(t): set(int(fn) for fn in audio_file_map.index[audio_file_map["task"] == t])
        for t in tasks
    }


def discover_valences(audio_file_map: pd.DataFrame) -> dict[str, set[int]]:
    """
    Map each unique non-null valence label → set of file_numbers with it.
    Files without a valence (NaN — see load_audio_file_map) are excluded.

    Insertion order: lexicographic on the valence label.

    Returns set[int] (file_numbers); see discover_task_groups for the
    rationale on int-vs-tuple.
    """
    valences = sorted(audio_file_map["valence"].dropna().unique())
    return {
        v: set(int(fn) for fn in audio_file_map.index[audio_file_map["valence"] == v])
        for v in valences
    }


def make_per_task_runs(
    task_groups: dict[str, set[int]],
) -> dict[str, list[str]]:
    """
    Generate per-task ablation runs from a TASK_GROUPS spec. For each task `t`,
    yields two runs: an eGeMAPS-only and a whisper-only ablation restricted
    to that task's files.

      f"egemaps_{t}_only": [f"egemaps_task_{t}"]
      f"whisper_{t}_only": [f"whisper_task_{t}"]

    Insertion order: lexicographic by task name; egemaps run before whisper run
    for each task. The two source-name strings (egemaps_task_<t>,
    whisper_task_<t>) are dispatched by load_feature_matrices_for_specs to the
    appropriate slice-aware builder.
    """
    runs: dict[str, list[str]] = {}
    for t in sorted(task_groups):
        runs[f"egemaps_{t}_only"] = [f"egemaps_task_{t}"]
        runs[f"whisper_{t}_only"] = [f"whisper_task_{t}"]
    return runs


def make_per_valence_runs(
    valences: dict[str, set[int]],
) -> dict[str, list[str]]:
    """
    Generate per-valence ablation runs from a VALENCES spec. For each valence
    label `v`, yields two runs:

      f"egemaps_{v}_only": [f"egemaps_valence_{v}"]
      f"whisper_{v}_only": [f"whisper_valence_{v}"]

    Insertion order: lexicographic by valence label; egemaps run before
    whisper run for each valence. Same dispatch convention as
    make_per_task_runs.
    """
    runs: dict[str, list[str]] = {}
    for v in sorted(valences):
        runs[f"egemaps_{v}_only"] = [f"egemaps_valence_{v}"]
        runs[f"whisper_{v}_only"] = [f"whisper_valence_{v}"]
    return runs


# ---------------------------------------------------------------------------
# Per-modality subject-matrix builders
# ---------------------------------------------------------------------------
def build_egemaps_subject_matrix(
    keep_files: set[int] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Returns (matrix, file_counts):
      matrix:      DataFrame of subject-mean eGeMAPS features (88 columns).
      file_counts: Series indexed by subject_id with #files contributing to
                   each subject's mean.

    Args:
      keep_files: optional set of file_numbers to restrict aggregation to.
                  When None (default), aggregates over ALL files (current
                  full-corpus behavior, yielding a (52, 88) matrix). When a
                  set, restricts to those file_numbers BEFORE the
                  disk-missing-issue filter; subjects with zero contributing
                  files are dropped from the returned matrix and counts
                  (they're NOT zero-imputed).
    """
    egemaps = pd.read_csv(EGEMAPS_CSV)
    egemaps["subject_id"] = egemaps["subject_id"].apply(normalize_subject_id)

    if keep_files is not None:
        egemaps = egemaps[egemaps["file_number"].isin(keep_files)].reset_index(drop=True)

    issues = pd.read_csv(ISSUES_CSV)
    issues["subject_id"] = issues["subject_id"].apply(normalize_subject_id)
    drop_pairs = set(zip(
        issues.loc[(issues.severity == "exclude") & (issues.source == "disk_missing"), "subject_id"],
        issues.loc[(issues.severity == "exclude") & (issues.source == "disk_missing"), "file_number"],
    ))
    if drop_pairs:
        before = len(egemaps)
        mask = pd.MultiIndex.from_arrays(
            [egemaps["subject_id"], egemaps["file_number"]]
        ).isin(drop_pairs)
        egemaps = egemaps[~mask].reset_index(drop=True)
        print(f"  Filtered {before - len(egemaps)} disk_missing eGeMAPS rows")

    feat_cols = [c for c in egemaps.columns if c not in ("subject_id", "file_number")]
    matrix = egemaps.groupby("subject_id")[feat_cols].mean().sort_index()
    counts = egemaps.groupby("subject_id").size().sort_index().rename("n_files")
    return matrix, counts


def build_whisper_subject_matrix(
    keep_files: set[int] | None = None,
    prefix: str = "whisper",
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Returns (matrix, file_counts):
      matrix:      DataFrame of subject-mean whisper features (16 columns,
                   prefixed by `prefix`).
      file_counts: Series indexed by subject_id with #transcripts contributing
                   per subject.

    Args:
      keep_files: optional set of file_numbers to restrict aggregation to.
                  When None (default), aggregates over ALL transcripts (the
                  whole-corpus 'whisper' slice). When a set, restricts
                  to those file_numbers; subjects with zero contributing
                  transcripts are dropped (NOT zero-imputed).

                  NOTE: the whisper parquet's column for the file index is
                  `file_num` (not `file_number` like eGeMAPS); keep_files is
                  the same int set either way.
      prefix:     column-name prefix, e.g. 'whisper',
                  'whisper_task_interview', 'whisper_valence_neg'.
    """
    df = pd.read_parquet(WHISPER_PARQUET)
    df["subject_id"] = df["subject_id"].apply(normalize_subject_id)
    if keep_files is not None:
        df = df[df["file_num"].isin(keep_files)]
    matrix = df.groupby("subject_id")[WHISPER_FEATURE_COLS].mean().sort_index()
    counts = df.groupby("subject_id").size().sort_index().rename("n_files")
    matrix.columns = [f"{prefix}__{c}" for c in matrix.columns]
    return matrix, counts


def build_demographic_subject_features(info: pd.DataFrame) -> pd.DataFrame:
    """
    Build per-subject demographic feature matrix from subject_info_map.csv
    (already loaded as `info`, indexed by int subject_id).

    Returns:
      demo_df: (n_subj, 3) DataFrame, columns
               ['demo__age', 'demo__gender_M', 'demo__edu_years'].
               gender is encoded as a single binary indicator (1 = M, 0 = F).

    Note: an earlier version of this helper also returned auxiliary clinical
    scales (CTQ-SF / LES / SSRS / GAD-7 / PSQI), but those were dropped because
    GAD-7 (r=0.89) and PSQI (r=0.79) correlate too strongly with PHQ-9 to use
    as model inputs under the project's speech-based-detection framing.
    """
    info = info.copy()

    # Gender encoded as binary (1 if M, 0 if F). Robust to source being numeric
    # (already 0/1), python str, pyarrow string, or pandas string dtype.
    g = info["gender"].astype(str).str.strip().str.upper()
    gender_M = g.isin({"M", "MALE", "1", "1.0"}).astype(int)

    return pd.DataFrame({
        "demo__age":       info["age"].astype(float),
        "demo__gender_M":  gender_M.astype(float),
        "demo__edu_years": info["edu_years"].astype(float),
    }, index=info.index).sort_index()


def _summarize_counts(name: str, counts: pd.Series, expected_max: int) -> str:
    """One-line summary of a per-subject file-count Series, with min-subject callout."""
    lo, hi, med, n_subj = int(counts.min()), int(counts.max()), float(counts.median()), len(counts)
    line = (f"  {name:<14} n_subj={n_subj:>2}  files/subject "
            f"min={lo:>3} median={med:>5.1f} max={hi:>3} (expected_max={expected_max})")
    if lo < expected_max:
        worst = counts[counts == lo].index.tolist()
        line += f"\n    {'':<14} subjects below max: {worst} have {lo} files each"
    return line


# ---------------------------------------------------------------------------
# Pipelines
# ---------------------------------------------------------------------------
def make_pipeline_default() -> Pipeline:
    """
    Build a fresh sklearn Pipeline used by every run except
    whisper_double_en (which uses make_pipeline_double_en for its
    custom 2x ElasticNet selector). Returned by pipeline_for_run for any
    run name not specifically handled there, and consumed per outer LOO
    fold inside nested_loo_predict + once in compute_posthoc_for_run for
    the all-subjects refit.

    Steps (also documented inline):
      1. SimpleImputer(median)        — handles NaN cells (e.g. whisper_iv
                                        rows from subjects with no interview
                                        transcripts). Refit per fold so the
                                        impute median doesn't leak.
      2. StandardScaler              — zero-mean / unit-variance scaling.
      3. SelectFromModel(ElasticNet) — fixed alpha=0.1, l1_ratio=0.7. EN's
                                        alpha / l1_ratio are NOT tuned inside
                                        the nested CV: doing so (e.g. via
                                        ElasticNetCV) would multiply the
                                        per-fold cost of an already-expensive
                                        nested LOO. Filed for revisit on
                                        post_may11 follow-ups.
      4. SVR(rbf)                    — kernel SVR; C / epsilon / gamma are
                                        the tuned hyperparameters in
                                        PARAM_GRID.
    """
    return Pipeline([
        # Step 1: median impute (handles NaN MATTR-interview etc.).
        # Trained on the 51-subject inner training set, no leak.
        ("impute", SimpleImputer(strategy="median")),
        # Step 2: zero-mean unit-variance scaling.
        ("scale",  StandardScaler()),
        # Step 3: ElasticNet selection. Threshold 1e-10 keeps any non-zero coef.
        ("select", SelectFromModel(
            ElasticNet(
                alpha=0.1, l1_ratio=0.7,
                max_iter=20000, random_state=RANDOM_STATE,
            ),
            threshold=1e-10,
        )),
        # Step 4: RBF SVR on the EN-selected features.
        ("svr",    SVR(kernel="rbf")),
    ])


# Custom selector for whisper_double_en: TWO sequential rounds of
# ElasticNet feature selection. Stage 1 fits EN on all input features; stage 2
# fits another EN on the stage-1 survivors. The motivation: a second EN pass
# on the smaller, partially-decorrelated post-stage-1 feature set may further
# prune redundancies that the first pass couldn't see at full feature
# dimensionality. Both stages use the same EN hyperparameters by default; the
# differential pruning (if any) comes purely from the changed covariate
# structure between the two passes.
class DoubleElasticNetSelector(BaseEstimator, TransformerMixin):
    """
    Two-stage ElasticNet feature selection. Exposes a single get_support()
    over the original (input) feature axis indicating which features survive
    BOTH stages, so it slots into the existing post-hoc inspection code with
    no special-casing.

    Parameters
    ----------
    en_alpha : float
        ElasticNet alpha for both stages.
    en_l1_ratio : float
        ElasticNet l1_ratio for both stages.
    en_max_iter : int
        ElasticNet max_iter for both stages.
    random_state : int | None
        ElasticNet random_state for both stages.
    threshold : float
        |coef| threshold for selection at each stage. Defaults to 1e-10
        (keep any nonzero coef, matching SelectFromModel's default behavior).

    Attributes set during fit:
        selected_mask_         (n_features_in_,) bool — features surviving both stages.
        n_selected_stage1_     int — count of features surviving stage 1.
        n_selected_stage2_     int — count of features surviving both stages
                                     (== selected_mask_.sum()).
    """

    def __init__(
        self,
        en_alpha: float = 0.1,
        en_l1_ratio: float = 0.7,
        en_max_iter: int = 20000,
        random_state=42,
        threshold: float = 1e-10,
    ):
        self.en_alpha     = en_alpha
        self.en_l1_ratio  = en_l1_ratio
        self.en_max_iter  = en_max_iter
        self.random_state = random_state
        self.threshold    = threshold

    def _make_en(self) -> ElasticNet:
        return ElasticNet(
            alpha=self.en_alpha,
            l1_ratio=self.en_l1_ratio,
            max_iter=self.en_max_iter,
            random_state=self.random_state,
        )

    def fit(self, X, y):
        # ---- Stage 1: EN on all input features ----
        en1 = self._make_en()
        en1.fit(X, y)
        mask1 = np.abs(en1.coef_) > self.threshold
        self.n_selected_stage1_ = int(mask1.sum())

        # If stage 1 dropped everything, nothing to feed stage 2.
        if self.n_selected_stage1_ == 0:
            self.selected_mask_      = mask1.copy()
            self.n_selected_stage2_  = 0
            return self

        # ---- Stage 2: EN on the stage-1 survivors ----
        en2 = self._make_en()
        en2.fit(X[:, mask1], y)
        mask2 = np.abs(en2.coef_) > self.threshold

        # Compose the final mask back onto the ORIGINAL feature axis so
        # downstream code (perm importance, modality grouping, etc.) sees
        # the full-width support vector.
        final = mask1.copy()
        final[mask1] = mask2
        self.selected_mask_     = final
        self.n_selected_stage2_ = int(final.sum())
        return self

    def transform(self, X):
        return X[:, self.selected_mask_]

    def get_support(self) -> np.ndarray:
        return self.selected_mask_


def make_pipeline_double_en() -> Pipeline:
    """
    Pipeline used by the whisper_double_en run. Identical to the default
    pipeline except the single SelectFromModel(ElasticNet) step is replaced
    by DoubleElasticNetSelector — two sequential EN selection rounds with
    the same hyperparameters as the default pipeline's single round.
    """
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale",  StandardScaler()),
        ("select", DoubleElasticNetSelector(
            en_alpha=0.1,
            en_l1_ratio=0.7,
            en_max_iter=20000,
            random_state=RANDOM_STATE,
            threshold=1e-10,
        )),
        ("svr",    SVR(kernel="rbf")),
    ])


def pipeline_for_run(run_name: str, feature_names: list[str]):
    """
    Run-name → pipeline-factory dispatcher. Returns a zero-arg callable that
    builds a fresh sklearn Pipeline for `run_name`. Used as the default
    `pipeline_factory_lookup` by both run_one_configuration (per outer LOO
    fold) and compute_posthoc_for_run (single refit on all 52 subjects).

    Current dispatch:
      'whisper_double_en' → make_pipeline_double_en  (2x EN selector)
      everything else            → make_pipeline_default

    `feature_names` is currently unused but is part of the signature so future
    pipeline factories can close over column-name information (e.g. a per-
    modality scaler or selector) without changing callers.
    """
    if run_name == "whisper_double_en":
        return make_pipeline_double_en
    return make_pipeline_default


# ---------------------------------------------------------------------------
# Inner CV
# ---------------------------------------------------------------------------
def nested_loo_predict(
    X: np.ndarray,
    y: np.ndarray,
    desc: str,
    n_jobs: int,
    pipeline_factory=None,
) -> np.ndarray:
    """
    Manual outer LOO with tqdm progress and inner GridSearchCV(LOO) per fold.
    Called by run_one_configuration for every non-mean-predictor run; emits the
    per-fold progress bar visible in the train-script output.

    Let n = len(X). Each outer iteration:
      - Leaves out one subject's row.
      - Builds a fresh GridSearchCV around pipeline_factory() with PARAM_GRID.
      - Inner CV is LOO over the n-1 training rows.
      - The best estimator predicts the held-out row.

    n is normally 52 (the full MODMA cohort), but may be smaller when X comes
    from a per-task / per-valence slice that drops subjects with zero
    contributing files (see build_egemaps_subject_matrix /
    build_whisper_subject_matrix `keep_files` semantics).

    Args:
      X, y:             feature matrix and PHQ-9 vector, both length n.
      desc:             tqdm progress-bar prefix (typically the run name).
      n_jobs:           passed through to inner GridSearchCV.
      pipeline_factory: zero-arg callable returning a fresh sklearn Pipeline.
                        Defaults to make_pipeline_default; in practice
                        callers pass pipeline_for_run(run_name, ...).

    Returns: y_pred (length n) of OOF predictions.
    """
    if pipeline_factory is None:
        pipeline_factory = make_pipeline_default

    n = len(X)
    y_pred = np.full(n, np.nan)
    done_mask = np.zeros(n, dtype=bool)

    pbar = tqdm(
        LeaveOneOut().split(X),
        total=n,
        desc=desc,
        unit="fold",
        leave=True,
        dynamic_ncols=True,
    )
    for train_idx, test_idx in pbar:
        inner = GridSearchCV(
            pipeline_factory(),
            param_grid=PARAM_GRID,
            cv=LeaveOneOut(),
            scoring="neg_mean_squared_error",
            n_jobs=n_jobs,
            refit=True,
        )
        inner.fit(X[train_idx], y[train_idx])
        y_pred[test_idx] = inner.predict(X[test_idx])[0]
        done_mask[test_idx] = True

        # Running RMSE on completed folds (estimate while folds finish).
        n_done = int(done_mask.sum())
        if n_done >= 5:
            running = float(np.sqrt(np.mean((y_pred[done_mask] - y[done_mask]) ** 2)))
            pbar.set_postfix({"running_RMSE": f"{running:.3f}", "done": f"{n_done}/{n}"})

    return y_pred


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute the three regression metrics reported per run: RMSE, MAE, R².
    Called by run_one_configuration on the OOF prediction vector returned by
    nested_loo_predict (or loo_mean_predictor_predictions for the mean
    predictor). Returns a plain {str: float} dict; run_one_configuration
    extends it with PI_coverage / PI_mean_width before storing on the result.
    """
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE":  float(mean_absolute_error(y_true, y_pred)),
        "R2":   float(r2_score(y_true, y_pred)),
    }


def compute_jackknife_prediction_intervals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    alpha: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """
    Jackknife (leave-one-out conformal) symmetric prediction intervals.

    For each subject i, the half-width is the (1 - alpha) quantile of |residuals|
    from all OTHER subjects. The PI is symmetric around y_pred[i]:
        PI_i = [y_pred[i] - q_{-i},  y_pred[i] + q_{-i}]
    where q_{-i} = quantile_{1-alpha}({|y_true[j] - y_pred[j]| : j != i}).

    With our LOO setup, y_pred is already an OOF vector — every prediction was
    made on data that didn't include the held-out subject. Under exchangeability
    of (X_i, Y_i), this construction has approximately (1 - alpha) marginal
    coverage. Note the conformal-correct quantile would be the
    ceil((n-1)(1-alpha) + 1) / (n-1) order statistic; for n=52, alpha=0.05 the
    difference vs np.quantile(., 0.95) is sub-percent and we use the simpler
    np.quantile here.

    Args:
        y_true: (n,) ground truth.
        y_pred: (n,) leave-one-out predictions.
        alpha:  desired miscoverage rate (0.05 -> 95% PI).

    Returns:
        lower:      (n,) PI lower bounds.
        upper:      (n,) PI upper bounds.
        coverage:   empirical fraction of y_true inside its PI (target: 1-alpha).
        mean_width: average PI width across subjects (in the y units; PHQ-9 here).
    """
    n = len(y_true)
    abs_resid = np.abs(y_true - y_pred)
    lower = np.empty(n, dtype=float)
    upper = np.empty(n, dtype=float)
    for i in range(n):
        others = np.concatenate([abs_resid[:i], abs_resid[i + 1:]])
        q = float(np.quantile(others, 1.0 - alpha))
        lower[i] = float(y_pred[i] - q)
        upper[i] = float(y_pred[i] + q)
    coverage   = float(((y_true >= lower) & (y_true <= upper)).mean())
    mean_width = float((upper - lower).mean())
    return lower, upper, coverage, mean_width


def loo_mean_predictor_predictions(y: np.ndarray) -> np.ndarray:
    """
    Return the leave-one-out mean-predictor's OOF predictions:
        pred[i] = mean(y[j] for j != i)

    Plugged into the same evaluate() as the SVR runs so the comparison table
    treats the mean predictor as a normal first-row run.
    """
    n = len(y)
    return np.array([y[np.arange(n) != i].mean() for i in range(n)])


# ---------------------------------------------------------------------------
# Modality grouping (used by post-hoc inspection)
# ---------------------------------------------------------------------------
def modality_of(name: str) -> str:
    """
    Map a feature column name to its modality label, used wherever per-run
    feature breakdowns are needed (print_run_configurations,
    print_selection_summary, save_results_csv's n_egemaps_selected /
    n_whisper_selected counts, and the 'modality' column in
    svr_perm_importance.csv via _compute_perm_importance_df).

    Recognized prefixes (checked in order so slice-specific names resolve
    before the legacy whisper / whisper_iv branches):
      'egemaps_task_<X>__…'    → 'egemaps_task_<X>'
      'egemaps_valence_<v>__…' → 'egemaps_valence_<v>'
      'whisper_task_<X>__…'    → 'whisper_task_<X>'
      'whisper_valence_<v>__…' → 'whisper_valence_<v>'
      'whisper__…'             → 'whisper'             (full-corpus whisper)
      'whisper_iv__…'          → 'whisper_interview'   (display label; whisper_iv is the source name)
      'demo__…'                → 'demographics'
      anything else            → 'egemaps'             (full-corpus fallback)

    Callers that need to roll per-task / per-valence slice modalities up to
    the parent 'egemaps' / 'whisper' totals do prefix matching on the
    returned label — both save_results_csv (for the n_egemaps_selected /
    n_whisper_selected columns) and print_run_configurations (for the
    per-run breakdown line) handle this via .startswith("egemaps") /
    .startswith("whisper").
    """
    # Slice prefixes (egemaps/whisper × task/valence). Checked before the
    # legacy whisper / whisper_iv branches so a column like
    # "whisper_task_interview__lex_ttr" routes to its task-specific modality
    # rather than falling through. Each branch returns the part before "__"
    # so post-hoc summaries group features by their fine-grained source.
    if name.startswith("egemaps_task_") and "__" in name:
        return name.split("__", 1)[0]
    if name.startswith("egemaps_valence_") and "__" in name:
        return name.split("__", 1)[0]
    if name.startswith("whisper_task_") and "__" in name:
        return name.split("__", 1)[0]
    if name.startswith("whisper_valence_") and "__" in name:
        return name.split("__", 1)[0]
    if name.startswith("whisper__"):
        return "whisper"
    if name.startswith("whisper_iv__"):
        return "whisper_interview"
    if name.startswith("demo__"):
        return "demographics"
    return "egemaps"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_arg_parser(
    known_runs: tuple[str, ...],
    description: str | None = None,
) -> argparse.ArgumentParser:
    """
    Build the standard SVR-script CLI parser. Almost always invoked through
    parse_and_validate_args (which adds --alpha / --runs validation and
    handles the None-vs-list outcome); surfaced separately so a caller can
    grab the parser to extend it with extra flags before parsing.

    `known_runs` populates the --runs help-string choices list — pass the
    train script's KNOWN_RUNS tuple. `description` is typically the train
    script's `__doc__` so `--help` reproduces the module banner.

    All flags have sensible defaults; `--runs` defaults to None to signal
    "no CLI selection" so the caller can route to its own interactive menu
    (train_svr.py uses prompt_user_for_runs in that case).
    """
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--runs",
        type=str,
        default=None,
        help=(
            f"Comma-separated list of runs. If omitted, the calling script "
            f"falls back to its default selection behavior (interactive menu "
            f"in train_svr.py). Choices: {','.join(known_runs)}"
        ),
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="n_jobs for inner GridSearchCV and permutation_importance (default: -1).",
    )
    parser.add_argument(
        "--n-perm-repeats",
        type=int,
        default=20,
        help="Permutation-importance repeats on post-hoc full fit (default: 20).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help=(
            "Miscoverage rate for jackknife prediction intervals (default: 0.05 -> 95%% PI). "
            "Per-subject PIs are reported alongside RMSE/MAE/R²; coverage and mean width "
            "are added to the comparison table and CSV."
        ),
    )
    return parser


def parse_and_validate_args(
    known_runs: tuple[str, ...],
    description: str | None = None,
    argv: list[str] | None = None,
) -> tuple[argparse.Namespace, list[str] | None]:
    """
    First call inside the train-script main(): parse CLI args via
    build_arg_parser, then validate `--alpha ∈ (0, 1)` and that every entry
    of the parsed `--runs` list is in `known_runs`. Calls sys.exit(1) on
    validation failure (with an error printed to stderr).

    `argv` defaults to None → argparse uses sys.argv; pass an explicit list
    for testing.

    Returns (args, requested_runs):
      - requested_runs is None when --runs was not supplied on the CLI.
        The caller is responsible for resolving the None case — train_svr.py
        drops into prompt_user_for_runs() and then
        ensure_mean_predictor_included().
      - Otherwise it's the parsed, validated list of run names from --runs
        (still passed through ensure_mean_predictor_included by callers
        that want the mean-predictor baseline auto-added).
    """
    parser = build_arg_parser(known_runs, description=description)
    args = parser.parse_args(argv)
    if not (0.0 < args.alpha < 1.0):
        print(f"ERROR: --alpha must be in (0, 1), got {args.alpha}", file=sys.stderr)
        sys.exit(1)
    if args.runs is None:
        return args, None
    requested_runs = [r.strip() for r in args.runs.split(",") if r.strip()]
    for r in requested_runs:
        if r not in known_runs:
            print(f"ERROR: unknown run name '{r}'. Choices: {','.join(known_runs)}", file=sys.stderr)
            sys.exit(1)
    return args, requested_runs


def parse_run_selection(text: str, n_runs: int) -> list[int]:
    """
    Parse a menu selection string into a sorted, deduplicated list of 1-indexed
    run numbers within [1, n_runs].

    Grammar:
      "all" / "ALL"  — shortcut for [1..n_runs].
      "N"            — single index.
      "N1,N2,..."    — comma-separated indices.
      "lo-hi"        — inclusive range (yields [lo..hi]).
      Mixing and whitespace OK: e.g. "1, 3-5,  7".

    Raises:
      ValueError on empty input, non-numeric tokens, malformed ranges, or
      out-of-range indices. Out-of-range messages embed the offending number
      so callers (e.g. prompt_user_for_runs) can echo it back to the user.
    """
    text = text.strip()
    if not text:
        raise ValueError("empty selection")
    if text.lower() == "all":
        return list(range(1, n_runs + 1))

    indices: set[int] = set()
    for part in (p.strip() for p in text.split(",")):
        if not part:
            continue
        if "-" in part:
            try:
                lo_str, hi_str = part.split("-", 1)
                lo, hi = int(lo_str.strip()), int(hi_str.strip())
            except ValueError:
                raise ValueError(f"malformed range: {part!r}")
            if lo > hi:
                raise ValueError(f"malformed range: {part!r}")
            for i in range(lo, hi + 1):
                if not (1 <= i <= n_runs):
                    raise ValueError(f"out of range: {i}")
                indices.add(i)
        else:
            try:
                i = int(part)
            except ValueError:
                raise ValueError(f"invalid token: {part!r}")
            if not (1 <= i <= n_runs):
                raise ValueError(f"out of range: {i}")
            indices.add(i)

    if not indices:
        raise ValueError("empty selection")
    return sorted(indices)


def prompt_user_for_runs(
    run_specs: dict[str, list[str]],
) -> list[str]:
    """
    Display a numbered menu of runs from `run_specs`, accept a selection from
    stdin, ask for confirmation, and return the selected run names in
    `run_specs` insertion order (independent of the order the user typed them).

    Loops until the user confirms a selection with 'y'/'yes':
      - Invalid or empty input re-prompts (with a one-line error for invalid
        and an offending-number echo for out-of-range cases).
      - 'n' (or any non-yes confirmation) re-shows the menu.

    The selection grammar is described inline in the input prompt; see
    parse_run_selection for the full spec.
    """
    run_names = list(run_specs.keys())
    n = len(run_names)
    n_digits   = len(str(n))
    name_width = max(len(name) for name in run_names)
    n_left     = (n + 1) // 2  # ceil(n/2)

    while True:
        # Two-column layout: left column holds indices 1..n_left, right column
        # holds the rest. For n=22 the rows are 1↔12, 2↔13, …, 11↔22; for n=23
        # the rows are 1↔13, …, 12↔(blank). Drops the bracketed source list to
        # keep the menu compact as RUN_FEATURE_SOURCES grows.
        print()
        print("Available run configurations:")
        for row in range(n_left):
            left_idx  = row
            right_idx = row + n_left
            left_entry = f"  {left_idx + 1:>{n_digits}}. {run_names[left_idx]:<{name_width}}"
            if right_idx < n:
                right_entry = f"  {right_idx + 1:>{n_digits}}. {run_names[right_idx]:<{name_width}}"
            else:
                right_entry = ""
            print((left_entry + "    " + right_entry).rstrip())
        print()
        text = input("Select runs (e.g. 1,3-5 or 'all'): ")

        try:
            indices = parse_run_selection(text, n)
        except ValueError as e:
            print(f"  Invalid selection: {e}. Try again.")
            continue

        selected_names = [run_names[i - 1] for i in indices]
        print()
        print(f"Selected {len(selected_names)} run(s):")
        for name in selected_names:
            print(f"  - {name}")
        confirm = input(f"Run these {len(selected_names)} configurations? [y/n] ").strip().lower()
        if confirm in ("y", "yes"):
            print()
            return selected_names


def ensure_mean_predictor_included(
    requested_runs: list[str],
    known_runs: tuple[str, ...],
) -> list[str]:
    """
    Ensure 'mean_predictor' is in requested_runs (prepended) when it's a known
    run. The mean predictor is virtually free to compute and provides the
    trivial baseline that every SVR run is compared against — keeping it in
    every CSV by default avoids a round trip when the user wants the baseline
    for ΔRMSE later.

    No-op if 'mean_predictor' is not in known_runs (so a future train script
    that doesn't define it isn't forced to). Returns a new list — does not
    mutate the input. Prints a one-line note when it auto-includes so the
    behavior is visible to the user.
    """
    if "mean_predictor" not in known_runs:
        return list(requested_runs)
    if "mean_predictor" in requested_runs:
        return list(requested_runs)
    print("  (Note: mean_predictor auto-included as baseline — present in all output CSVs by default.)")
    return ["mean_predictor"] + list(requested_runs)


# ---------------------------------------------------------------------------
# Run-orchestration helpers (one per main() block)
# ---------------------------------------------------------------------------
def load_subject_info_and_egemaps() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    First step of the train-script load flow: load subject_info, build the
    full-corpus eGeMAPS subject-mean matrix (no slicing), and align both
    DataFrames to their intersection on subject_id (warning on any
    mismatch). Prints the standard "Loading data..." status block to stdout
    so the train script's output begins with a recognizable header.

    eGeMAPS is loaded eagerly here because every train-script run uses it in
    some form (full-corpus or per-task/valence slice) and the slice-aware
    builds in load_feature_matrices_for_specs reuse the alignment performed
    here. Whisper / demographic / per-task slice matrices are loaded lazily
    by load_feature_matrices_for_specs based on the user's run selection.

    Returns:
      info:         DataFrame indexed by int subject_id, carrying PHQ-9,
                    group (HC / MDD), and demographic columns.
      egemaps_subj: (n_subj, 88) DataFrame of subject-mean eGeMAPS features.
      eg_counts:    Series of #eGeMAPS files contributing to each subject's
                    mean (passed downstream as the eGeMAPS coverage entry).
    """
    print("Loading data...")
    info = load_subject_info()
    print(f"  subject_info: {info.shape}; group counts: {info['group'].value_counts().to_dict()}")

    egemaps_subj, eg_counts = build_egemaps_subject_matrix()
    print(f"  eGeMAPS subject-mean: {egemaps_subj.shape}")

    if not egemaps_subj.index.equals(info.index):
        common = egemaps_subj.index.intersection(info.index)
        missing_eg = info.index.difference(egemaps_subj.index).tolist()
        missing_in = egemaps_subj.index.difference(info.index).tolist()
        if missing_eg or missing_in:
            print(f"  WARN subject mismatch: in_info_only={missing_eg}  in_egemaps_only={missing_in}")
        egemaps_subj = egemaps_subj.loc[common]
        eg_counts    = eg_counts.loc[common]
        info         = info.loc[common]
        print(f"  Restricted to {len(common)} common subjects")

    return info, egemaps_subj, eg_counts


def _maybe_warn_sample_size(
    warnings: list[str],
    source: str,
    counts: pd.Series,
    expected_max: int,
    threshold: int,
) -> None:
    """
    Append a sample-size warning to `warnings` if any subject's contributing-
    file count for `source` falls below `threshold`. Format matches the spec
    in load_feature_matrices_for_specs's docstring. No-op for empty counts.
    """
    if len(counts) == 0:
        return
    min_count = int(counts.min())
    if min_count >= threshold:
        return
    worst_sid = int(counts.idxmin())
    median_count = float(counts.median())
    warnings.append(
        f"{source}: subject {worst_sid:08d} has {min_count} file(s); "
        f"median {median_count:.1f}, expected_max {expected_max} (threshold {threshold})"
    )


def load_feature_matrices_for_specs(
    info: pd.DataFrame,
    egemaps_subj: pd.DataFrame,
    eg_counts: pd.Series,
    requested_runs: list[str],
    run_specs: dict[str, list[str]],
    task_groups: dict[str, set[int]],
    valences: dict[str, set[int]],
    min_files_per_subject: int = MIN_FILES_PER_SUBJECT_DEFAULT,
) -> tuple[
    dict[str, pd.DataFrame],
    dict[str, tuple[pd.Series, int]],
    list[str],
]:
    """
    Lazily load every feature-source matrix referenced by the SELECTED runs
    (not the full RUN_FEATURE_SOURCES table). Replaces the older
    load_whisper_aggregations_if_needed + load_demographics_if_needed pair.

    Dispatches by source-name pattern:
      "egemaps"                  → already passed in via egemaps_subj.
      "whisper"            → build_whisper_subject_matrix(keep_files=None,
                                       prefix="whisper"), expected_max=29.
      "whisper_iv"               → build_whisper_subject_matrix(
                                       keep_files=task_groups["interview"],
                                       prefix="whisper_iv"), expected_max=18.
                                   (Legacy alias; behavior identical to today's
                                   whisper_iv slice.)
      "egemaps_task_<X>"         → build_egemaps_subject_matrix(
                                       keep_files=task_groups[X]).
                                   expected_max = len(task_groups[X]).
      "egemaps_valence_<v>"      → build_egemaps_subject_matrix(
                                       keep_files=valences[v]).
                                   expected_max = len(valences[v]).
      "whisper_task_<X>"         → build_whisper_subject_matrix(
                                       keep_files=task_groups[X],
                                       prefix=f"whisper_task_{X}").
      "whisper_valence_<v>"      → build_whisper_subject_matrix(
                                       keep_files=valences[v],
                                       prefix=f"whisper_valence_{v}").
      "demo"                     → build_demographic_subject_features(info).
                                   No counts_by_modality entry (demographics
                                   are per-subject, not per-file).

    For every loaded slice (excluding demo), checks counts.min() against
    `min_files_per_subject` and accumulates a warning of the form:
      "<source>: subject {SID:08d} has {min} file(s); median {med:.1f},
       expected_max {emax} (threshold {threshold})"
    The eGeMAPS full-corpus slice is also checked, even though its matrix is
    pre-loaded by load_subject_info_and_egemaps.

    Returns:
      feature_matrices:     {source_name: matrix}, ready for build_run_matrices.
      counts_by_modality:   {label: (counts_series, expected_max)} for
                            print_coverage_summary. The eGeMAPS entry is keyed
                            "eGeMAPS" (matching the historical label); slice
                            entries are keyed by their source name.
      sample_size_warnings: list of warning strings; empty if no slice tripped
                            the threshold. Pass to
                            prompt_for_sample_size_acknowledgment.
    """
    needed_sources = {s for run in requested_runs for s in run_specs.get(run, [])}

    feature_matrices: dict[str, pd.DataFrame] = {"egemaps": egemaps_subj}
    counts_by_modality: dict[str, tuple[pd.Series, int]] = {"eGeMAPS": (eg_counts, 29)}
    sample_size_warnings: list[str] = []

    # Check eGeMAPS itself even though it's pre-loaded — guard catches global
    # subjects-below-threshold issues that affect every run.
    _maybe_warn_sample_size(sample_size_warnings, "egemaps", eg_counts, 29, min_files_per_subject)

    # Sort for deterministic loading order; "egemaps" is already loaded.
    for src in sorted(needed_sources - {"egemaps"}):
        if src == "whisper":
            mat, c = build_whisper_subject_matrix(keep_files=None, prefix="whisper")
            print(f"  whisper subject-mean: {mat.shape}; NaN cells: {int(mat.isna().sum().sum())}")
            feature_matrices[src] = mat
            counts_by_modality["whisper"] = (c, 29)
            _maybe_warn_sample_size(sample_size_warnings, src, c, 29, min_files_per_subject)

        elif src == "whisper_iv":
            kf = task_groups["interview"]
            mat, c = build_whisper_subject_matrix(keep_files=kf, prefix="whisper_iv")
            # Preserve the historical 4-space alignment + median-imputed annotation.
            print(f"  whisper_iv    subject-mean: {mat.shape}; NaN cells: {int(mat.isna().sum().sum())}  "
                  f"(median-imputed inside Pipeline per fold)")
            feature_matrices[src] = mat
            counts_by_modality["whisper_iv"] = (c, len(kf))
            _maybe_warn_sample_size(sample_size_warnings, src, c, len(kf), min_files_per_subject)

        elif src.startswith("egemaps_task_"):
            task = src[len("egemaps_task_"):]
            kf = task_groups[task]
            mat, c = build_egemaps_subject_matrix(keep_files=kf)
            print(f"  {src} subject-mean: {mat.shape}; NaN cells: {int(mat.isna().sum().sum())}")
            feature_matrices[src] = mat
            counts_by_modality[src] = (c, len(kf))
            _maybe_warn_sample_size(sample_size_warnings, src, c, len(kf), min_files_per_subject)

        elif src.startswith("egemaps_valence_"):
            v = src[len("egemaps_valence_"):]
            kf = valences[v]
            mat, c = build_egemaps_subject_matrix(keep_files=kf)
            print(f"  {src} subject-mean: {mat.shape}; NaN cells: {int(mat.isna().sum().sum())}")
            feature_matrices[src] = mat
            counts_by_modality[src] = (c, len(kf))
            _maybe_warn_sample_size(sample_size_warnings, src, c, len(kf), min_files_per_subject)

        elif src.startswith("whisper_task_"):
            task = src[len("whisper_task_"):]
            kf = task_groups[task]
            mat, c = build_whisper_subject_matrix(keep_files=kf, prefix=src)
            print(f"  {src} subject-mean: {mat.shape}; NaN cells: {int(mat.isna().sum().sum())}")
            feature_matrices[src] = mat
            counts_by_modality[src] = (c, len(kf))
            _maybe_warn_sample_size(sample_size_warnings, src, c, len(kf), min_files_per_subject)

        elif src.startswith("whisper_valence_"):
            v = src[len("whisper_valence_"):]
            kf = valences[v]
            mat, c = build_whisper_subject_matrix(keep_files=kf, prefix=src)
            print(f"  {src} subject-mean: {mat.shape}; NaN cells: {int(mat.isna().sum().sum())}")
            feature_matrices[src] = mat
            counts_by_modality[src] = (c, len(kf))
            _maybe_warn_sample_size(sample_size_warnings, src, c, len(kf), min_files_per_subject)

        elif src == "demo":
            demo_df = build_demographic_subject_features(info)
            print(f"  demographics:    {demo_df.shape};  cols: {list(demo_df.columns)}")
            feature_matrices[src] = demo_df
            # No counts_by_modality entry — demographics are per-subject.

        else:
            raise ValueError(
                f"Unrecognized feature source name: {src!r}. Expected one of "
                f"'egemaps', 'whisper', 'whisper_iv', 'demo', or a "
                f"prefixed slice like 'egemaps_task_<X>', 'whisper_valence_<v>'."
            )

    return feature_matrices, counts_by_modality, sample_size_warnings


def prompt_for_sample_size_acknowledgment(warnings: list[str]) -> None:
    """
    If `warnings` is non-empty, print a clearly-delimited block listing each
    warning and prompt the user to confirm proceeding. 'y' / 'yes' returns
    silently; anything else prints "Aborted." and exits with status 0.
    KeyboardInterrupt is NOT caught — callers / users should be free to
    Ctrl+C the run with the default Python behavior.

    No-op (silent) when warnings is empty.
    """
    if not warnings:
        return
    print()
    print("⚠ Sample-size warnings — some subjects contribute very few files to the slices below:")
    for w in warnings:
        print(f"  {w}")
    confirm = input("Proceed with selected runs? [y/n] ").strip().lower()
    if confirm in ("y", "yes"):
        return
    print("Aborted.")
    sys.exit(0)


def print_coverage_summary(
    counts_by_modality: dict[str, tuple[pd.Series, int]],
    gap_pair: tuple[str, str] | None = None,
) -> None:
    """
    Print the "Per-subject file coverage" block (modality asymmetry check)
    in the train-script's standard output. Called from main() right after
    load_feature_matrices_for_specs returns its counts dict, so the block
    appears between the matrix-load status lines and the subsequent
    "Run configurations:" listing.

    counts_by_modality: ordered dict mapping display-label →
                        (counts_series, expected_max). Insertion order
                        determines display order. Arbitrary modality sets
                        are supported; load_feature_matrices_for_specs keys
                        slice modalities by their source name (e.g.
                        'whisper_task_naming') alongside the eGeMAPS /
                        whisper / whisper_iv labels.
    gap_pair:           optional (label_a, label_b). If given, also prints the
                        largest per-subject coverage gap between modality A
                        and modality B; both labels must be keys of
                        counts_by_modality. train_svr.py passes
                        ('eGeMAPS', 'whisper') when whisper is
                        loaded; None otherwise.

    Why this matters: subjects with low audio quality (e.g., 02010036)
    contribute fewer whisper rows than acoustic rows, so their whisper-mean
    is computed from a smaller sample — the coverage block makes that
    asymmetry visible before any modeling happens. Per-row formatting comes
    from _summarize_counts (one line per modality, with a "subjects below
    max" callout when counts.min() < expected_max).
    """
    print("\nPer-subject file coverage (modality asymmetry check):")
    for label, (counts, expected_max) in counts_by_modality.items():
        print(_summarize_counts(label, counts, expected_max=expected_max))
    if gap_pair is not None:
        a, b = gap_pair
        ca, _ = counts_by_modality[a]
        cb, _ = counts_by_modality[b]
        gap = (ca.reindex(cb.index, fill_value=0) - cb).abs()
        if int(gap.max()) > 0:
            worst = int(gap.idxmax())
            print(f"  -> largest {a} vs {b} coverage gap: subject {worst:08d}  "
                  f"({a}={int(ca[worst])}, {b}={int(cb[worst])}, gap={int(gap[worst])})")


def build_run_matrices(
    feature_matrices: dict[str, pd.DataFrame],
    run_specs: dict[str, list[str]],
) -> dict[str, pd.DataFrame | None]:
    """
    Compose per-run feature matrices by left-joining named feature sources.

    feature_matrices: dict mapping source-name → subject-indexed DataFrame
                      (e.g., {"egemaps": egemaps_subj, "whisper": ww, ...}).
    run_specs:        dict mapping run-name → ordered list of source names. The
                      first source is the join anchor (and is .copy()'d to keep
                      the run independent of the source matrix); subsequent
                      sources are joined with how="left" onto it. An empty list
                      yields None (used by the leave-one-out mean predictor).

    Returns:
      run_matrices: dict mapping run-name → DataFrame (or None). Each non-None
                    DataFrame is independent of the source matrices and of every
                    other run's matrix, even when two runs share the same spec.
    """
    run_matrices: dict[str, pd.DataFrame | None] = {}
    for run_name, sources in run_specs.items():
        if not sources:
            run_matrices[run_name] = None
            continue
        df = feature_matrices[sources[0]].copy()
        for src in sources[1:]:
            df = df.join(feature_matrices[src], how="left")
        run_matrices[run_name] = df
    return run_matrices


def print_run_configurations(
    run_matrices: dict[str, pd.DataFrame | None],
    requested_runs: list[str],
    modality_of_fn=modality_of,
) -> None:
    """
    Print the "Run configurations:" block — for each requested run, one line
    showing X.shape, NaN cell count, and a per-modality feature breakdown.
    Called from main() right before the per-run nested-LOO loop so the user
    can sanity-check shapes before any expensive work runs.

    Modality counts are PREFIX-BASED (mirroring save_results_csv): any
    feature whose modality_of() value starts with "egemaps" rolls up into
    n_egemaps, likewise for "whisper". This keeps the line honest for runs
    that use task / valence slice modalities (egemaps_task_<X>,
    whisper_valence_<v>, ...) instead of reporting misleading zeros against
    a hardcoded legacy bucket set.
    """
    print("\nRun configurations:")
    for name in requested_runs:
        X_df = run_matrices[name]
        if X_df is None:
            print(f"  {name:<24}  (no features; LOO mean predictor)")
            continue
        cols = list(X_df.columns)
        n_eg  = sum(1 for c in cols if modality_of_fn(c).startswith("egemaps"))
        n_wp  = sum(1 for c in cols if modality_of_fn(c).startswith("whisper"))
        n_dem = sum(1 for c in cols if modality_of_fn(c) == "demographics")
        nan_cells = int(X_df.isna().sum().sum())
        suffix = "  [2x ElasticNet selection]" if name == "whisper_double_en" else ""
        print(
            f"  {name:<24}  X={X_df.shape}  "
            f"egemaps={n_eg} whisper={n_wp} demo={n_dem}  "
            f"NaN={nan_cells}{suffix}"
        )


def run_one_configuration(
    run_name: str,
    X_df: pd.DataFrame | None,
    y_series: pd.Series,
    alpha: float,
    n_jobs: int,
    run_idx: int,
    n_runs: int,
    pipeline_factory_lookup=pipeline_for_run,
    n_perm_repeats: int = 20,
    compute_posthoc_data: bool = True,
    modality_of_fn=modality_of,
    bump_repeats_for: tuple[str, ...] = ("whisper",),
    bump_to: int = 50,
) -> dict:
    """
    Run a single configuration end-to-end. Prints the section banner, computes
    OOF predictions (mean predictor when run_name == 'mean_predictor', nested
    LOO with the run-specific pipeline otherwise), computes RMSE/MAE/R² and
    jackknife PIs, prints the per-run summary line, and (unless disabled by
    compute_posthoc_data=False or this is the mean_predictor) collects per-run
    post-hoc characterization data: best inner-CV params, EN-selected feature
    mask + names, and full permutation-importance DataFrame.

    `y_series` is the per-subject PHQ-9 Series indexed by subject_id (i.e.
    `info["PHQ-9"].astype(float)`). Per run, the y vector used for nested LOO
    + posthoc is aligned to X_df.index — this matters for runs whose anchor
    source is a sliced matrix (e.g. whisper_only, egemaps_<task>_only,
    whisper_<valence>_only) that may drop subjects with zero contributing
    files. For mean_predictor the full y_series is used (X_df is None).

    Returns a results dict with keys
        {'X', 'subject_ids', 'y_true', 'y_pred',
         'pi_lower', 'pi_upper', 'metrics', 'n_features', 'posthoc'}
    where:
      'subject_ids' is the list of subject_ids the predictions correspond to
                    (X_df.index for SVR runs; full y_series.index for
                    mean_predictor) — consumed by save_predictions_csv.
      'y_true'      is the run-aligned ground-truth ndarray (same length as
                    y_pred) — consumed by save_predictions_csv.
      'posthoc'     is the dict from compute_posthoc_for_run, or None for
                    mean_predictor / when posthoc is disabled.
    The dict is consumed by print_results_table / save_*_csv /
    run_posthoc_inspection downstream.
    """
    if run_name == "mean_predictor":
        # Mean predictor uses the full subject set; X is None.
        subject_ids = list(y_series.index)
        y           = y_series.astype(float).to_numpy()
        banner = (f"\n{'='*78}\n[{run_idx+1}/{n_runs}] RUN: {run_name}   "
                  f"(no features; LOO mean of training PHQ-9)\n{'='*78}")
        print(banner, flush=True)
        y_pred  = loo_mean_predictor_predictions(y)
        X_store = None
        n_feat  = 0
    else:
        # Align y to X's subject index — sliced X may have <len(y_series) rows
        # if any subject contributed zero files to the slice (per Prompt 1's
        # "omit, don't zero-impute" semantics in build_*_subject_matrix).
        # Without this alignment, nested_loo_predict would silently mis-pair
        # X rows with y values when the row counts diverge.
        subject_ids = list(X_df.index)
        y           = y_series.loc[subject_ids].astype(float).to_numpy()
        X = X_df.to_numpy()
        banner = f"\n{'='*78}\n[{run_idx+1}/{n_runs}] RUN: {run_name}   X={X.shape}\n{'='*78}"
        print(banner, flush=True)
        factory = pipeline_factory_lookup(run_name, list(X_df.columns))
        y_pred = nested_loo_predict(
            X, y,
            desc=f"{run_name:<24}",
            n_jobs=n_jobs,
            pipeline_factory=factory,
        )
        X_store = X_df
        n_feat  = X_df.shape[1]

    metrics = evaluate(y, y_pred)
    pi_lower, pi_upper, pi_cov, pi_mean_width = compute_jackknife_prediction_intervals(
        y, y_pred, alpha=alpha,
    )
    metrics["PI_coverage"]   = pi_cov
    metrics["PI_mean_width"] = pi_mean_width

    target_cov = 1.0 - alpha
    cov_status = (
        f"{pi_cov:.3f}  "
        f"(target {target_cov:.3f}; "
        f"{'over' if pi_cov > target_cov else 'under' if pi_cov < target_cov else 'on'}-covered)"
    )
    print(
        f"  -> {run_name}: "
        f"RMSE={metrics['RMSE']:.4f}  MAE={metrics['MAE']:.4f}  R²={metrics['R2']:.4f}  "
        f"PI_cov={cov_status}  PI_width={pi_mean_width:.3f}"
    )

    # Per-run post-hoc characterization (silent — refits on the run's aligned
    # (X, y) via GridSearchCV(LOO) and computes permutation importance).
    # Skipped for mean_predictor (no model) and skippable via
    # compute_posthoc_data=False for callers that want bare metrics only.
    # Populates the columns in svr_run_results.csv (best_params, EN-selected
    # counts/names, top-5 perm).
    posthoc = None
    if compute_posthoc_data and X_store is not None:
        n_repeats_eff = max(n_perm_repeats, bump_to) if run_name in bump_repeats_for else n_perm_repeats
        bump_note = f" (bumped from {n_perm_repeats})" if n_repeats_eff != n_perm_repeats else ""
        print(f"  (posthoc: refit on {len(y)} subjects + perm importance n_repeats={n_repeats_eff}{bump_note}...)",
              flush=True)
        posthoc = compute_posthoc_for_run(
            X_df=X_store,
            y_all=y,
            run_name=run_name,
            n_jobs=n_jobs,
            n_perm_repeats=n_perm_repeats,
            pipeline_factory_lookup=pipeline_factory_lookup,
            modality_of_fn=modality_of_fn,
            bump_repeats_for=bump_repeats_for,
            bump_to=bump_to,
        )

    return {
        "X":           X_store,
        "subject_ids": subject_ids,
        "y_true":      y,
        "y_pred":      y_pred,
        "pi_lower":    pi_lower,
        "pi_upper":    pi_upper,
        "metrics":     metrics,
        "n_features": n_feat,
        "posthoc":    posthoc,
    }


def print_results_table(
    results: dict[str, dict],
    requested_runs: list[str],
    alpha: float,
    baseline_run: str | None,
    delta_column_label: str | None = None,
) -> None:
    """
    Print the "RESULTS — nested-LOO regression on PHQ-9" comparison table to
    stdout (one row per run, in `requested_runs` order). Called from main()
    after the run loop completes, before save_results_csv. The printed table
    is stdout-only — `svr_run_results.csv` carries the same per-run metrics
    plus the posthoc-derived columns.

    baseline_run:        the run name used as the ΔRMSE reference point. The
                         ΔRMSE column shows `run_RMSE − baseline_RMSE` for
                         every other row, '-' for the baseline row itself.
                         If None (or the baseline isn't in `results`), the
                         column is '-' for ALL rows and the explanatory
                         preamble line is suppressed.

                         Per-policy: train_svr.main() passes 'mean_predictor'
                         when it's in the requested runs, else None — so the
                         ΔRMSE column is always anchored to the trivial LOO-
                         mean baseline (or absent if the user explicitly ran
                         without it).
    delta_column_label:  header label for the ΔRMSE column. Defaults to None,
                         which auto-derives the label from `baseline_run`:
                         'ΔRMSE_vs_{baseline_run}' if a baseline is set,
                         else 'ΔRMSE'. Pass an explicit string only if you
                         need a custom header (e.g. external tooling keyed on
                         a specific column name).
    """
    if delta_column_label is None:
        delta_column_label = f"ΔRMSE_vs_{baseline_run}" if baseline_run else "ΔRMSE"

    target_cov = 1.0 - alpha
    delta_w = max(18, len(delta_column_label))
    header = (
        f"{'run':<26} {'n_features':>11} {'RMSE':>8} {'MAE':>8} {'R²':>8} "
        f"{'PI_cov':>8} {'PI_width':>9} {delta_column_label:>{delta_w}}"
    )
    banner_w = max(116, len(header))
    print()
    print("=" * banner_w)
    print(f"RESULTS — nested-LOO regression on PHQ-9 (n=52); jackknife {int(round(target_cov*100))}% prediction intervals")
    if baseline_run:
        print(f"ΔRMSE column = run_RMSE − {baseline_run}_RMSE   (negative = better than {baseline_run})")
    print(f"PI_cov target = {target_cov:.3f}; PI_width in PHQ-9 units (lower = tighter)")
    print("=" * banner_w)
    print(header)
    print("-" * len(header))
    base_rmse = (
        results[baseline_run]["metrics"]["RMSE"]
        if (baseline_run and baseline_run in results) else None
    )
    for run_name in requested_runs:
        m = results[run_name]["metrics"]
        n_feat = results[run_name]["n_features"]
        if base_rmse is None or run_name == baseline_run:
            delta = "-"
        else:
            d = m["RMSE"] - base_rmse
            delta = f"{d:+.4f}"
        n_feat_str = f"{n_feat:d}" if n_feat else "0"
        mae_str    = f"{m['MAE']:.4f}"
        r2_str     = f"{m['R2']:.4f}"
        pi_cov_str   = f"{m['PI_coverage']:.3f}"
        pi_width_str = f"{m['PI_mean_width']:.3f}"
        print(
            f"{run_name:<26} {n_feat_str:>11} {m['RMSE']:>8.4f} {mae_str:>8} {r2_str:>8} "
            f"{pi_cov_str:>8} {pi_width_str:>9} {delta:>{delta_w}}"
        )


def save_results_csv(
    results: dict[str, dict],
    requested_runs: list[str],
    alpha: float,
    output_path: Path,
    run_specs: dict[str, list[str]],
    modality_of_fn=modality_of,
    top_n_perm: int = 5,
) -> None:
    """
    Save per-run numeric results to `output_path` as `svr_run_results.csv` —
    one row per requested run, in run order. Combines the OOF-prediction
    metrics (computed by nested LOO inside run_one_configuration) with the
    per-run posthoc characterization (also computed by run_one_configuration
    via compute_posthoc_for_run, attached as results[run]['posthoc']).

    Called from main() after the run loop completes; must run before
    run_posthoc_inspection if you want the CSV to land before the verbose
    best-run report on stdout.

    Canonical column reference: `results/RESULTS_CSV_SCHEMAS.md` —
    keep that doc in sync any time the column set or column meaning changes
    here (per the "Results documentation discipline" rule in CLAUDE.md).

    Columns:
      run, n_features, RMSE, MAE, R2,
      PI_alpha, PI_coverage, PI_mean_width,
      best_params                — JSON dict of GridSearchCV best params (post-hoc refit).
      feature_matrices_used      — JSON list of source names from run_specs[run].
      n_egemaps_selected         — int, # eGeMAPS features retained by EN.
                                   Counts any feature whose modality_of() value
                                   starts with "egemaps" — so it covers the
                                   full-corpus eGeMAPS modality plus any future
                                   per-task / per-valence eGeMAPS slice
                                   modalities (egemaps_task_<X>,
                                   egemaps_valence_<v>).
      n_whisper_selected         — int, # whisper features retained by EN.
                                   Counts any feature whose modality_of() value
                                   starts with "whisper" — covers whisper
                                   (full-corpus), whisper_interview (display
                                   label for the whisper_iv source), and the
                                   per-task / per-valence whisper slice
                                   modalities (whisper_task_<X>,
                                   whisper_valence_<v>).
      selected_features          — JSON list of all EN-retained feature names.
      top_{N}_perm_importance    — JSON list of {feature, perm_imp_mse_increase_train}
                                   dicts, top N by perm importance among EN-selected
                                   features (N=top_n_perm, default 5).

    For mean_predictor (no model): the posthoc-derived columns are written as
    empty containers ('{}', '[]') / 0. mean_predictor is included as a row by
    default — see ensure_mean_predictor_included.
    """
    top_col = f"top_{top_n_perm}_perm_importance"
    rows = []
    for run_name in requested_runs:
        m = results[run_name]["metrics"]
        n_feat = results[run_name]["n_features"]
        posthoc = results[run_name].get("posthoc")

        if posthoc is None:
            # mean_predictor case (no model, no EN, no perm importance) OR
            # compute_posthoc_data=False was passed.
            best_params_json    = json.dumps({})
            n_egemaps_sel       = 0
            n_whisper_sel       = 0
            selected_feats_json = json.dumps([])
            top_perm_json       = json.dumps([])
        else:
            mask       = posthoc["mask"]
            feat_names = posthoc["feature_names"]

            # Prefix-based modality counts: any feature whose modality starts
            # with "egemaps" / "whisper" rolls up into the corresponding total.
            # This covers full-corpus eGeMAPS / whisper plus every per-task and
            # per-valence slice (egemaps_task_<X>, whisper_valence_<v>, etc.)
            # without enumerating each new modality string.
            n_egemaps_sel = sum(
                1 for i, keep in enumerate(mask)
                if keep and modality_of_fn(feat_names[i]).startswith("egemaps")
            )
            n_whisper_sel = sum(
                1 for i, keep in enumerate(mask)
                if keep and modality_of_fn(feat_names[i]).startswith("whisper")
            )

            best_params_json    = json.dumps(posthoc["best_params"])
            selected_feats_json = json.dumps(posthoc["selected_features"])

            imp_df = posthoc["imp_df"]
            top = imp_df[imp_df["selected_by_EN"]].head(top_n_perm)
            top_perm_records = [
                {
                    "feature": str(rec["feature"]),
                    "perm_imp_mse_increase_train": float(rec["perm_imp_mse_increase_train"]),
                }
                for rec in top.to_dict("records")
            ]
            top_perm_json = json.dumps(top_perm_records)

        feature_sources_json = json.dumps(list(run_specs.get(run_name, [])))

        rows.append({
            "run":                    run_name,
            "n_features":             n_feat,
            "RMSE":                   m["RMSE"],
            "MAE":                    m["MAE"],
            "R2":                     m["R2"],
            "PI_alpha":               alpha,
            "PI_coverage":            m["PI_coverage"],
            "PI_mean_width":          m["PI_mean_width"],
            "best_params":            best_params_json,
            "feature_matrices_used":  feature_sources_json,
            "n_egemaps_selected":     n_egemaps_sel,
            "n_whisper_selected":     n_whisper_sel,
            "selected_features":      selected_feats_json,
            top_col:                  top_perm_json,
        })
    pd.DataFrame(rows).to_csv(output_path, index=False)


def save_predictions_csv(
    results: dict[str, dict],
    requested_runs: list[str],
    output_path: Path,
) -> None:
    """
    Save per-subject predictions + PIs to `output_path` as
    `svr_participant_results.csv` — one row per (run, subject). For runs whose
    anchor source is a sliced matrix that drops subjects with zero
    contributing files, that run will have FEWER rows in the CSV (only the
    subjects it actually predicted on). The full-corpus runs and
    mean_predictor each contribute one row per subject in the global subject
    set. Called from main() after save_results_csv. Useful downstream for PI
    calibration plots, identifying subjects always outside their PI, and
    computing per-subgroup coverage (e.g., bucketing on info["age"] or
    info["gender"]).

    Per-run subject_ids and y_true are read from the results dict (populated
    by run_one_configuration). This avoids re-aligning the global y vector to
    each run's potentially-sliced subject index at write time.

    Canonical column reference: `results/RESULTS_CSV_SCHEMAS.md` — keep that
    doc in sync with any column change here (per CLAUDE.md's "Results
    documentation discipline").

    Columns:
      run, subject_id, y_true, y_pred,
      RMSE         — per-participant RMSE = |y_true - y_pred| (RMSE for n=1
                     reduces to the absolute residual; included for symmetry
                     with the per-run RMSE column in svr_run_results.csv).
      pi_lower, pi_upper, in_interval

    mean_predictor is included as a set of rows by default — see
    ensure_mean_predictor_included.
    """
    pred_rows = []
    for run_name in requested_runs:
        r = results[run_name]
        run_subj_ids = r["subject_ids"]
        y_true       = r["y_true"]
        y_pred       = r["y_pred"]
        pi_lower     = r["pi_lower"]
        pi_upper     = r["pi_upper"]
        for i, sid in enumerate(run_subj_ids):
            y_t  = float(y_true[i])
            y_p  = float(y_pred[i])
            in_pi = bool((y_t >= pi_lower[i]) and (y_t <= pi_upper[i]))
            pred_rows.append({
                "run":         run_name,
                "subject_id":  int(sid),
                "y_true":      y_t,
                "y_pred":      y_p,
                "RMSE":        abs(y_t - y_p),
                "pi_lower":    float(pi_lower[i]),
                "pi_upper":    float(pi_upper[i]),
                "in_interval": in_pi,
            })
    pd.DataFrame(pred_rows).to_csv(output_path, index=False)


# ---------------------------------------------------------------------------
# Post-hoc inspection
# ---------------------------------------------------------------------------
def refit_best_on_all_subjects(
    X: np.ndarray,
    y: np.ndarray,
    pipeline_factory,
    n_jobs: int,
    param_grid: dict | None = None,
) -> tuple[Pipeline, dict]:
    """
    Refit a fresh pipeline on ALL subjects via GridSearchCV(LeaveOneOut). The
    "post-hoc" refit — distinct from the per-fold refits inside
    nested_loo_predict (which leave one subject out for OOF prediction).
    The all-subjects refit is what gets characterized for the per-run
    posthoc CSV columns and for the verbose best-run report.

    Used solely by compute_posthoc_for_run today; surfaced as a top-level
    helper so a user could call it ad-hoc from a notebook against a custom
    pipeline factory + grid.

    Returns (best_estimator, best_params). param_grid defaults to PARAM_GRID.
    """
    if param_grid is None:
        param_grid = PARAM_GRID
    full_grid = GridSearchCV(
        pipeline_factory(),
        param_grid=param_grid,
        cv=LeaveOneOut(),
        scoring="neg_mean_squared_error",
        n_jobs=n_jobs,
        refit=True,
    )
    full_grid.fit(X, y)
    return full_grid.best_estimator_, full_grid.best_params_


def print_selection_summary(
    pipe_full: Pipeline,
    feature_names: list[str],
    modality_of_fn=modality_of,
) -> None:
    """
    Print the "ElasticNet retained X of Y features" block grouped by
    modality (as labeled by `modality_of_fn`). Used inside
    run_posthoc_inspection on the best run's posthoc data; not called from
    main() directly.

    Expects pipe_full to have a 'select' step exposing get_support() over
    the input feature axis (true for both make_pipeline_default and
    make_pipeline_double_en, since DoubleElasticNetSelector deliberately
    exposes the same get_support() interface as SelectFromModel).
    """
    selector = pipe_full.named_steps["select"]
    mask = selector.get_support()
    selected_names = [feature_names[i] for i, keep in enumerate(mask) if keep]

    by_mod_total: dict[str, int] = {}
    by_mod_kept:  dict[str, int] = {}
    for n in feature_names:
        m = modality_of_fn(n)
        by_mod_total[m] = by_mod_total.get(m, 0) + 1
        if n in selected_names:
            by_mod_kept[m] = by_mod_kept.get(m, 0) + 1

    print(f"\n  ElasticNet retained {len(selected_names)} of {len(feature_names)} features:")
    for mod in sorted(by_mod_total):
        kept  = by_mod_kept.get(mod, 0)
        total = by_mod_total[mod]
        print(f"    {mod:<14} {kept:>3d} / {total:>3d}")


def _compute_perm_importance_df(
    pipe_full: Pipeline,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    modality_of_fn=modality_of,
    n_repeats: int = 20,
    n_jobs: int = -1,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """
    Internal: compute the training-set permutation-importance DataFrame for
    `pipe_full` on (X, y). No prints. Used by both compute_permutation_importance
    (which adds verbose output) and compute_posthoc_for_run (silent per-run
    posthoc data collection).

    Returns: imp_df sorted descending by perm_imp_mse_increase_train, with
             columns [feature, modality, selected_by_EN,
                      perm_imp_mse_increase_train, perm_imp_mse_increase_train_std].
    """
    perm = permutation_importance(
        pipe_full, X, y,
        n_repeats=n_repeats,
        scoring="neg_mean_squared_error",
        random_state=random_state,
        n_jobs=n_jobs,
    )
    mask = pipe_full.named_steps["select"].get_support()
    imp_df = pd.DataFrame({
        "feature":                          feature_names,
        "modality":                         [modality_of_fn(n) for n in feature_names],
        "selected_by_EN":                   mask,
        "perm_imp_mse_increase_train":      perm.importances_mean,
        "perm_imp_mse_increase_train_std":  perm.importances_std,
    }).sort_values("perm_imp_mse_increase_train", ascending=False).reset_index(drop=True)
    return imp_df


def print_perm_importance_report(imp_df: pd.DataFrame, top_n: int = 20) -> None:
    """
    Print the standard permutation-importance caveat + the top-N EN-selected
    features (by perm importance) from a precomputed imp_df. Used by
    run_posthoc_inspection on the best run's posthoc data.
    """
    n_selected = int(imp_df["selected_by_EN"].sum())
    print(
        "\n  CAVEAT: 'perm_imp_mse_increase_train' is TRAINING-SET permutation\n"
        "  importance (computed on the same data used to fit the post-hoc model).\n"
        "  Units = MSE-increase under feature shuffle (NOT RMSE-increase).\n"
        "  EN-dropped features show ~0 by construction (the selector removes them\n"
        "  upstream of the SVR), so we report only EN-retained features below."
    )
    print(f"\n  Top {min(top_n, n_selected)} EN-selected features by permutation importance:")
    with pd.option_context("display.max_colwidth", 60):
        print(imp_df[imp_df["selected_by_EN"]].head(top_n).to_string(index=False))


def compute_permutation_importance(
    pipe_full: Pipeline,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    modality_of_fn=modality_of,
    n_repeats: int = 20,
    n_jobs: int = -1,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """
    Verbose-public wrapper: compute training-set permutation importance for
    `pipe_full` on (X, y), print the standard caveat plus the top-20 EN-selected
    features by importance. Returns the importance DataFrame for CSV writing
    by the caller. Equivalent to _compute_perm_importance_df + a "Computing..."
    progress line + print_perm_importance_report.
    """
    print(f"\n  Computing permutation importance (n_repeats={n_repeats})...")
    imp_df = _compute_perm_importance_df(
        pipe_full, X, y, feature_names,
        modality_of_fn=modality_of_fn,
        n_repeats=n_repeats,
        n_jobs=n_jobs,
        random_state=random_state,
    )
    print_perm_importance_report(imp_df)
    return imp_df


def compute_posthoc_for_run(
    X_df: pd.DataFrame | None,
    y_all: np.ndarray,
    run_name: str,
    n_jobs: int,
    n_perm_repeats: int = 20,
    pipeline_factory_lookup=pipeline_for_run,
    modality_of_fn=modality_of,
    bump_repeats_for: tuple[str, ...] = ("whisper",),
    bump_to: int = 50,
    random_state: int = RANDOM_STATE,
) -> dict | None:
    """
    Compute per-run post-hoc characterization data (silent — no internal prints):
      - Refit the run's pipeline on the full (X, y_all) pair via
        GridSearchCV(LOO). For runs whose anchor source is a sliced matrix
        with subjects dropped, that's <52 rows; run_one_configuration is
        responsible for passing a y_all already aligned to X_df.index.
      - Get the EN-selected feature mask + selected feature names.
      - Compute training-set permutation importance.

    n_perm_repeats is bumped to >= bump_to when run_name is in bump_repeats_for
    — added because earlier runs at n_repeats=20 showed ±15-30% relative SDs
    on the importance estimates for the whisper run, and tighter CIs
    were wanted for the report. Behavior is per-run, not per-best-run, so the
    run's posthoc columns (e.g. top-5 in svr_run_results.csv) get the tighter
    CIs whenever that run is executed.

    Returns a dict with keys:
      'pipe_full'         — refit Pipeline (its 'select' step exposes get_support()).
      'best_params'       — GridSearchCV best_params_ dict.
      'mask'              — bool ndarray over the input feature axis.
      'selected_features' — list of EN-selected feature names (in input order).
      'feature_names'     — list of all feature names (X_df.columns).
      'imp_df'            — full perm-importance DataFrame (descending by importance).
      'n_repeats_used'    — effective n_repeats (post-bump).

    Returns None when X_df is None (mean_predictor case — no features, no model).
    """
    if X_df is None:
        return None
    feature_names = list(X_df.columns)
    X = X_df.to_numpy()
    factory = pipeline_factory_lookup(run_name, feature_names)
    pipe_full, best_params = refit_best_on_all_subjects(X, y_all, factory, n_jobs=n_jobs)
    n_repeats_used = max(n_perm_repeats, bump_to) if run_name in bump_repeats_for else n_perm_repeats
    imp_df = _compute_perm_importance_df(
        pipe_full, X, y_all, feature_names,
        modality_of_fn=modality_of_fn,
        n_repeats=n_repeats_used,
        n_jobs=n_jobs,
        random_state=random_state,
    )
    mask = pipe_full.named_steps["select"].get_support()
    selected_features = [feature_names[i] for i, keep in enumerate(mask) if keep]
    return {
        "pipe_full":         pipe_full,
        "best_params":       best_params,
        "mask":              mask,
        "selected_features": selected_features,
        "feature_names":     feature_names,
        "imp_df":            imp_df,
        "n_repeats_used":    n_repeats_used,
    }


def run_posthoc_inspection(
    requested_runs: list[str],
    results: dict[str, dict],
    output_perm_csv: Path,
    modality_of_fn=modality_of,
    top_n_print: int = 20,
) -> None:
    """
    Print the verbose post-hoc report for the best NON-mean-predictor run and
    persist its full perm-importance table to output_perm_csv. Consumes the
    precomputed posthoc data attached to each result by run_one_configuration
    (so the refit + permutation_importance work is NOT duplicated here).

    1. Selects the lowest-RMSE non-mean-predictor run.
    2. Prints best inner-CV params + EN-retained features grouped by modality.
    3. Prints the standard caveat + top-N EN-selected features by importance.
    4. Writes the full importance table to output_perm_csv.

    No-op if all requested runs are mean_predictor (or if the best-run's
    posthoc data is missing — e.g., compute_posthoc_data was disabled).
    """
    posthoc_candidates = [r for r in requested_runs if r != "mean_predictor"]
    if not posthoc_candidates:
        print("\nNo SVR runs requested; skipping post-hoc inspection.")
        return
    best_run = min(posthoc_candidates, key=lambda k: results[k]["metrics"]["RMSE"])
    posthoc = results[best_run].get("posthoc")
    if posthoc is None:
        print(f"\nBest run {best_run} has no posthoc data attached; skipping post-hoc inspection.")
        return

    print()
    print("=" * 84)
    print(f"POST-HOC INSPECTION OF BEST RUN: {best_run}  "
          f"(nested-LOO RMSE = {results[best_run]['metrics']['RMSE']:.4f})")
    print("=" * 84)

    pipe_label = "2x ElasticNet selection" if best_run == "whisper_double_en" else "default"
    print(f"Refit pipeline (used for posthoc): {pipe_label}; "
          f"perm importance n_repeats={posthoc['n_repeats_used']}.")
    print(f"  Best inner-CV params: {posthoc['best_params']}")

    print_selection_summary(posthoc["pipe_full"], posthoc["feature_names"], modality_of_fn=modality_of_fn)
    print_perm_importance_report(posthoc["imp_df"], top_n=top_n_print)

    posthoc["imp_df"].to_csv(output_perm_csv, index=False)
    print(f"\n  Permutation-importance table written to {output_perm_csv}")
