"""
train_svr.py
============
Main SVR analysis script — nested-LOO SVR-RBF regression on PHQ-9 across
single- and multi-configuration runs.

Saves results as CSVs under results/ under a subdir corresponding to the 
script execution's date/time

To inspect or extend the set of runs, edit _BASE_RUNS.

Usage:
    python scripts/train_svr.py --runs mean_predictor,whisper_only
    python scripts/train_svr.py --alpha 0.10                     # 90% PIs
    python scripts/train_svr.py --n-jobs 4 --n-perm-repeats 10
"""

from __future__ import annotations

import sys

from helpers_svr import *


# ---------------------------------------------------------------------------
# Run configuration
# ---------------------------------------------------------------------------
# Adding a run = adding an entry in _BASE_RUNS. Key can be any name, value list must
# consist of "source names".
# Source names refer to keys of the feature_matrices dict that enumerates all the feature 
# sets available. Here are the source names that may be added as values (ie as elements of 
# the value list for a particular key) in _BASE_RUNS:
#
#     "egemaps"                    — eGeMAPS subject-mean matrix (always loaded)
#     "whisper"                    — whole-corpus whisper-feature subject-mean
#     "egemaps_task_<X>"           — eGeMAPS over files belonging to task X
#     "egemaps_valence_<v>"        — eGeMAPS over files with valence v
#     "whisper_task_<X>"           — whisper over files belonging to task X
#     "whisper_valence_<v>"        — whisper over files with valence v
#     "demo"                       — demographic features (age, gender_M, edu_years)
#
_audio_file_map = load_audio_file_map()
TASK_GROUPS: dict[str, set[int]] = discover_task_groups(_audio_file_map)
VALENCES:    dict[str, set[int]] = discover_valences(_audio_file_map)

_BASE_RUNS: dict[str, list[str]] = {
    "mean_predictor":       [],                              # Run 1: DO NOT RENAME. LOO mean of training PHQ-9 (no features).
    "egemaps_only":         ["egemaps"],                     # Run 2: 88 eGeMAPS subject-mean acoustic features only.
    "whisper_only":         ["whisper"],                     # Run 3: 16 whisper subject-mean transcript-derived features only.
    "egemaps_demo":         ["egemaps", "demo"],             # Run 4: eGeMAPS + 3 demographics (age, gender_M, edu_years).
    "whisper_demo":         ["whisper", "demo"],             # Run 5: whisper transcript features + 3 demographics.
    "egemaps_whisper":      ["egemaps", "whisper"],          # Run 6: eGeMAPS + whisper
    "egemaps_whisper_demo": ["egemaps", "whisper", "demo"],  # Run 7: eGeMAPS + whisper + demographics
}

RUN_FEATURE_SOURCES: dict[str, list[str]] = { # all the runs this script will execute
    **_BASE_RUNS, # the ones configured above
    **make_per_task_runs(TASK_GROUPS), # contains configs for each of the task groups for egemaps and whisper: 
                                       #       interview, passage_reading, picture_description, word_reading

    **make_per_valence_runs(VALENCES), # contains configs for each of the valences for egemaps and whisper:
                                        #       positive, negative, neutral
}
KNOWN_RUNS: tuple[str, ...] = tuple(RUN_FEATURE_SOURCES.keys())

def main() -> int:
    args, requested_runs = parse_and_validate_args(KNOWN_RUNS, description=__doc__)
    if requested_runs is None:
        # No --runs on the CLI → drop into the interactive menu.
        requested_runs = prompt_user_for_runs(RUN_FEATURE_SOURCES)
    # mean_predictor is the trivial baseline; auto-include if absent so its
    # row is present in every output CSV by default.
    requested_runs = ensure_mean_predictor_included(requested_runs, KNOWN_RUNS)

    # ----------------------------------------------------------------------
    # Create a fresh timestamped results directory for this invocation.
    # ----------------------------------------------------------------------
    run_dir = make_run_results_dir()
    output_results_csv     = run_dir / "svr_run_results.csv"
    output_perm_csv        = run_dir / "svr_perm_importance.csv"
    output_predictions_csv = run_dir / "svr_participant_results.csv"
    print(f"Run results directory: {run_dir}")

    # ----------------------------------------------------------------------
    # Load + sanity-check data. Whisper, demographic, and any task / valence
    # slice matrices are loaded lazily, scoped to the SELECTED runs only —
    # not the full RUN_FEATURE_SOURCES table — so picking one ablation
    # doesn't load matrices for the others.
    # ----------------------------------------------------------------------
    info, egemaps_subj, eg_counts = load_subject_info_and_egemaps()

    feature_matrices, counts_by_modality, sample_size_warnings = (
        load_feature_matrices_for_specs(
            info=info,
            egemaps_subj=egemaps_subj,
            eg_counts=eg_counts,
            requested_runs=requested_runs,
            run_specs=RUN_FEATURE_SOURCES,
            task_groups=TASK_GROUPS,
            valences=VALENCES,
        )
    )

    # ----------------------------------------------------------------------
    # Per-subject file coverage summary (exposes modality asymmetry).
    # ----------------------------------------------------------------------
    gap_pair = ("eGeMAPS", "whisper") if "whisper" in counts_by_modality else None
    print_coverage_summary(counts_by_modality, gap_pair=gap_pair)

    # ----------------------------------------------------------------------
    # Sample-size guard — if any selected slice has subjects below the
    # threshold (default 3 files), prompt the user to ack-or-abort before
    # any nested-LOO work runs.
    # ----------------------------------------------------------------------
    prompt_for_sample_size_acknowledgment(sample_size_warnings)

    # ----------------------------------------------------------------------
    # Build feature matrices for each run. Filtered to selected runs to match
    # the lazy-loaded feature_matrices contents — building entries for
    # unselected runs would KeyError on missing source matrices.
    # ----------------------------------------------------------------------
    selected_specs = {r: RUN_FEATURE_SOURCES[r] for r in requested_runs}
    run_matrices = build_run_matrices(feature_matrices, selected_specs)
    # PHQ-9 as a Series indexed by subject_id; run_one_configuration aligns it
    # to each run's X.index per-run (sliced runs may drop subjects).
    y_series = info["PHQ-9"].astype(float)

    print_run_configurations(run_matrices, requested_runs)

    # ----------------------------------------------------------------------
    # Run all selected runs
    # ----------------------------------------------------------------------
    results: dict[str, dict] = {}
    for run_idx, run_name in enumerate(requested_runs):
        results[run_name] = run_one_configuration(
            run_name=run_name,
            X_df=run_matrices[run_name],
            y_series=y_series,
            alpha=args.alpha,
            n_jobs=args.n_jobs,
            run_idx=run_idx,
            n_runs=len(requested_runs),
            n_perm_repeats=args.n_perm_repeats,
        )

    # ----------------------------------------------------------------------
    # Results summary + CSV outputs.
    # ΔRMSE in the printed comparison table is anchored to mean_predictor
    # whenever it's in the requested runs (which it almost always is, thanks
    # to ensure_mean_predictor_included). When mean_predictor is absent, the
    # ΔRMSE column is shown as '-' for every row. ΔRMSE is stdout-only — it
    # is NOT a column in svr_run_results.csv.
    # ----------------------------------------------------------------------
    baseline_run = "mean_predictor" if "mean_predictor" in requested_runs else None
    print_results_table(results, requested_runs, alpha=args.alpha, baseline_run=baseline_run)

    save_results_csv(
        results, requested_runs,
        alpha=args.alpha,
        output_path=output_results_csv,
        run_specs=RUN_FEATURE_SOURCES,
    )
    print(f"\nPer-run results written to {output_results_csv}")

    save_predictions_csv(
        results, requested_runs,
        output_path=output_predictions_csv,
    )
    print(f"Per-participant results+PIs written to {output_predictions_csv}")

    # ----------------------------------------------------------------------
    # Post-hoc inspection of the best NON-mean-predictor run (consumes
    # precomputed per-run posthoc data, no recompute).
    # ----------------------------------------------------------------------
    run_posthoc_inspection(
        requested_runs=requested_runs,
        results=results,
        output_perm_csv=output_perm_csv,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
