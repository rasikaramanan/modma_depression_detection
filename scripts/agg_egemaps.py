#!/usr/bin/env python3
"""Aggregate per-file eGeMAPS functional CSVs into one long-form CSV.

Reads:  {AUDIO_ROOT}/{subject_id}/{NN}_openSMILE_func.csv  (NN = 01..29 per subject)
Writes: {repo_root}/data/features/egemaps.csv  with shape ~ (1503, 90)
        Columns: subject_id, file_number, then the 88 eGeMAPS features in source order.

The 5 corrupt audio files at subject 02010004 (file numbers 24-28) have no
_func.csv on disk; they are skipped, so the expected row count is 1503
(52 subjects x 29 files - 5 missing).

Usage:
    python scripts/agg_egemaps.py
    python scripts/agg_egemaps.py --audio-root /path/to/audio_lanzhou_2015
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_AUDIO_ROOT = REPO_ROOT / "CSCI567 Project" / "modma_data" / "audio_lanzhou_2015"
OUTPUT_DIR = REPO_ROOT / "data" / "features"
OUTPUT_PATH = OUTPUT_DIR / "egemaps.csv"

DROP_COLS = ("file", "start", "end")  # openSMILE metadata, not features
N_FILES_PER_SUBJECT = 29


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--audio-root",
        type=Path,
        default=DEFAULT_AUDIO_ROOT,
        help=f"Directory containing per-subject subdirs (default: {DEFAULT_AUDIO_ROOT}).",
    )
    args = parser.parse_args()

    audio_root: Path = args.audio_root
    if not audio_root.is_dir():
        raise SystemExit(f"AUDIO_ROOT does not exist: {audio_root}")

    print(f"AUDIO_ROOT: {audio_root}")
    print(f"Output:     {OUTPUT_PATH}")
    print()

    subject_dirs = sorted(p for p in audio_root.iterdir() if p.is_dir())
    if not subject_dirs:
        raise SystemExit("No subject subdirectories found under AUDIO_ROOT.")

    print(f"Found {len(subject_dirs)} subject directories. Reading per-file CSVs...")

    rows: list[dict] = []
    feature_cols: list[str] | None = None
    n_read_total = 0
    missing_by_subject: dict[str, list[int]] = defaultdict(list)

    for s_idx, subj_dir in enumerate(subject_dirs, start=1):
        subject_id = subj_dir.name
        n_read_subj = 0

        for file_idx in range(1, N_FILES_PER_SUBJECT + 1):
            stem = f"{file_idx:02d}"
            func_csv = subj_dir / f"{stem}_openSMILE_func.csv"
            if not func_csv.exists():
                missing_by_subject[subject_id].append(file_idx)
                continue

            df = pd.read_csv(func_csv)
            df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

            if df.empty:
                raise SystemExit(f"Empty CSV (no data row): {func_csv}")

            if feature_cols is None:
                feature_cols = list(df.columns)
            elif list(df.columns) != feature_cols:
                raise SystemExit(
                    f"Column mismatch in {func_csv}\n"
                    f"  expected ({len(feature_cols)} cols): {feature_cols[:3]}...\n"
                    f"  got      ({len(df.columns)} cols): {list(df.columns)[:3]}..."
                )

            row = {"subject_id": subject_id, "file_number": file_idx, **df.iloc[0].to_dict()}
            rows.append(row)
            n_read_subj += 1

        n_read_total += n_read_subj
        miss = missing_by_subject.get(subject_id, [])
        line = f"  [{s_idx:2d}/{len(subject_dirs)}] {subject_id}: {n_read_subj} files"
        if miss:
            line += f"  (missing: {miss})"
        print(line)

    print()
    n_missing_total = sum(len(v) for v in missing_by_subject.values())
    print(f"Total CSVs read:    {n_read_total}")
    print(f"Total CSVs missing: {n_missing_total}")

    if not rows or feature_cols is None:
        raise SystemExit("No CSVs were read - nothing to write.")

    print(f"\nBuilding long-form DataFrame from {len(rows)} rows...")
    ordered = ["subject_id", "file_number", *feature_cols]
    out = pd.DataFrame(rows, columns=ordered)
    print(f"Final shape: {out.shape}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
