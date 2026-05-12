# Depression Detection Using the Multimodal Open Dataset for Mental Disorder Analysis (MODMA)

## Repository layout

```
data/
  features/     # eGeMAPS + Whisper feature matrices 
  metadata/     # subject info, task/valence map, quality issues, wav stats
  external/     # third-party assets (DUTIR Chinese emotion lexicon)
notes/          # methods notes, literature review, pipeline write-ups
results/        # one timestamped subdirectory per train_svr.py invocation
scripts/        # see below
```


## Commands to reproduce results

```
# Install dependencies (use Python 3.11)
pip install scikit-learn pandas numpy matplotlib tqdm faster-whisper funasr opencc spacy_pkuseg opensmile

# Modeling, saves results to a results subdirectory
python scripts/train_svr.py

# Post-hoc figures and tables for one results directory
python scripts/analyze_intervals_and_groupings.py --results-dir results/<TIMESTAMP>

# Multi-calibration over the SVR pipeline
python scripts/run_svr_multicalibration.py
```

Each `train_svr.py` invocation writes a fresh `results/YYYY-MM-DD_HHMMSS_TZ/` subdirectory containing per-run metrics, per-participant predictions with intervals, and permutation importance for the best run. The full schema is in `results/RESULTS_CSV_SCHEMAS.md`.

## Scripts, grouped by purpose

**Feature extraction.** Produces the matrices in `data/features/` from raw audio.
- Acoustic (eGeMAPS via openSMILE): `extract_opensmile_features.py`, then `agg_egemaps.py` collates per-file outputs into a single CSV.
- Linguistic (Whisper transcripts → 16 lexical / syntactic / sentiment features): `extract_whisper_features.py` → `audit_whisper_transcripts.py` → `preprocess_transcripts.py` → `extract_text_features.py`.
- Utilities: `get_wav_stats.py` (per-file durations), `extract_wavlm_features.py` (WavLM embeddings; not used in the final submission).

**Modeling.** Nested leave-one-subject-out (LOSO) SVR with RBF kernel, ElasticNet feature selection, and jackknife prediction intervals.
- `train_svr.py` is the entry point; `helpers_svr.py` implements the LOSO loop, inner grid search, prediction intervals, and post-hoc permutation importance.

**Multi-calibration.** Mean recalibration of predictions across demographic subgroups (age, gender, education).
- `multicalibration.py` (algorithm), `run_multicalibration.py` (over ElasticNet), `run_svr_multicalibration.py` (over the SVR pipeline).

**Post-hoc analysis and visualization.**
- `analyze_intervals_and_groupings.py` — per-task and per-valence tables and figures, plus prediction-interval ablation, computed from a `train_svr.py` results directory.
- `helpers_viz.py` — plotting and table helpers used by the notebook.


