#!/usr/bin/env python3
"""
get_wav_stats.py
================
Traverse the MODMA audio directory, read every wav file's duration (seconds),
join with the task label from audio_file_map.csv, and write one row per
(subject, file_number) to data/metadata/wav_file_stats.csv.

Expected output: 29 files * 52 subjects = 1508 rows.
"""

from pathlib import Path
import math
import sys

import pandas as pd
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parent.parent
AUDIO_DIR = (
    REPO_ROOT
    / "CSCI567 Project"
    / "modma_data"
    / "audio_lanzhou_2015"
)
AUDIO_FILE_MAP = REPO_ROOT / "data" / "metadata" / "audio_file_map.csv"
OUT_CSV = REPO_ROOT / "data" / "metadata" / "wav_file_stats.csv"


def gen_wav_file_csv():
    file_map = pd.read_csv(AUDIO_FILE_MAP)
    task_by_num = dict(zip(file_map["file_number"], file_map["task"]))

    rows = []
    failures = []
    subject_dirs = sorted(p for p in AUDIO_DIR.iterdir() if p.is_dir())
    for subj_dir in subject_dirs:
        subject_id = subj_dir.name
        for wav_path in sorted(subj_dir.glob("*.wav")):
            try:
                file_number = int(wav_path.stem)
            except ValueError:
                continue
            try:
                info = sf.info(str(wav_path))
                duration = info.duration
            except Exception as e:
                duration = math.nan
                failures.append((subject_id, file_number, str(e)))
            rows.append({
                "subject_id": subject_id,
                "file_number": file_number,
                "task": task_by_num.get(file_number, ""),
                "duration_seconds": duration,
            })

    df = pd.DataFrame(rows).sort_values(["subject_id", "file_number"])
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    print(f"Wrote {len(df)} rows to {OUT_CSV}")
    if failures:
        print(f"{len(failures)} files failed to read (duration=NaN):")
        for subj, num, err in failures:
            print(f"  {subj}/{num:02d}.wav -> {err}")


def print_wav_stats():
    if not OUT_CSV.exists():
        print(f"{OUT_CSV} not found. Run gen_wav_file_csv() first.")
        sys.exit(1)

    df = pd.read_csv(OUT_CSV)
    file_map = pd.read_csv(AUDIO_FILE_MAP)
    task_order = file_map["task"].drop_duplicates().tolist()

    stats = (
        df.groupby("task")["duration_seconds"]
        .agg(
            n="count",
            min="min",
            p25=lambda x: x.quantile(0.25),
            median="median",
            mean="mean",
            p75=lambda x: x.quantile(0.75),
            max="max",
        )
        .reindex(task_order)
    )

    print("Wav duration (seconds) by task:")
    print(stats.round(2).to_string())


def main():
    gen_wav_file_csv()
    print_wav_stats()


if __name__ == "__main__":
    main()
