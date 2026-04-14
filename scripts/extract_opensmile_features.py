#!/usr/bin/env python3
"""
extract_opensmile_features.py
=============================
Extracts openSMILE eGeMAPS features from all MODMA audio wav files.

For each wav file, produces two CSV outputs:
  1. *_openSMILE_lld.csv  — 25 Low-Level Descriptors (LLDs) per frame
                            (~100 frames/sec: 20ms window, 10ms hop)
  2. *_openSMILE_func.csv — 88 eGeMAPS functionals (one row per file)

Output files are saved in the same directory structure as the source wav files
under audio_lanzhou_2015/, creating directories as needed.

Usage:
    python scripts/extract_opensmile_features.py

Requirements:
    pip install opensmile tqdm
"""

import os
import sys
import time
import logging
import argparse
from pathlib import Path

import opensmile
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Default paths — adjust if your repo root differs
REPO_ROOT = Path(__file__).resolve().parent.parent
AUDIO_DIR = (
    REPO_ROOT
    / "CSCI567 Project"
    / "modma_data"
    / "audio_lanzhou_2015"
)

# openSMILE feature sets
FEATURE_SET = opensmile.FeatureSet.eGeMAPSv02
LLD_FEATURE_LEVEL = opensmile.FeatureLevel.LowLevelDescriptors
FUNC_FEATURE_LEVEL = opensmile.FeatureLevel.Functionals

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------


def extract_features_for_file(
    wav_path: Path,
    smile_lld: opensmile.Smile,
    smile_func: opensmile.Smile,
) -> tuple[bool, str]:
    """
    Extract LLD and functional features for a single wav file.

    Returns (success: bool, message: str).
    """
    stem = wav_path.stem  # e.g. "01"
    out_dir = wav_path.parent

    lld_path = out_dir / f"{stem}_openSMILE_lld.csv"
    func_path = out_dir / f"{stem}_openSMILE_func.csv"

    # Skip if both outputs already exist (resume-friendly)
    if lld_path.exists() and func_path.exists():
        return True, f"SKIP (already exists): {wav_path}"

    try:
        # Extract LLD features (per-frame)
        lld_df = smile_lld.process_file(str(wav_path))
        lld_df.to_csv(str(lld_path))

        # Extract functional features (per-file summary)
        func_df = smile_func.process_file(str(wav_path))
        func_df.to_csv(str(func_path))

        return True, f"OK: {wav_path}"

    except Exception as e:
        return False, f"FAIL: {wav_path} — {e}"


def main():
    parser = argparse.ArgumentParser(
        description="Extract openSMILE eGeMAPS features from MODMA audio files."
    )
    parser.add_argument(
        "--audio-dir",
        type=Path,
        default=AUDIO_DIR,
        help="Root directory of audio_lanzhou_2015 (default: auto-detected from repo).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List files that would be processed without extracting.",
    )
    args = parser.parse_args()

    audio_dir = args.audio_dir.resolve()

    # ------------------------------------------------------------------
    # Validate input directory
    # ------------------------------------------------------------------
    if not audio_dir.exists():
        logger.error(f"Audio directory not found: {audio_dir}")
        logger.error(
            "Make sure the MODMA data is located at:\n"
            "  <repo>/CSCI567 Project/modma_data/audio_lanzhou_2015/\n"
            "Or pass --audio-dir explicitly."
        )
        sys.exit(1)

    # ------------------------------------------------------------------
    # Discover all wav files
    # ------------------------------------------------------------------
    wav_files = sorted(audio_dir.rglob("*.wav"))
    if not wav_files:
        logger.error(f"No .wav files found under {audio_dir}")
        sys.exit(1)

    # Gather subject IDs for summary
    subject_ids = sorted({f.parent.name for f in wav_files})
    logger.info(
        f"Found {len(wav_files)} wav files across {len(subject_ids)} subjects "
        f"in {audio_dir}"
    )

    if args.dry_run:
        for f in wav_files:
            print(f"  {f.relative_to(audio_dir)}")
        print(f"\nTotal: {len(wav_files)} files. Use without --dry-run to extract.")
        return

    # ------------------------------------------------------------------
    # Initialise openSMILE extractors
    # ------------------------------------------------------------------
    logger.info("Initialising openSMILE extractors (eGeMAPSv02)...")
    smile_lld = opensmile.Smile(
        feature_set=FEATURE_SET,
        feature_level=LLD_FEATURE_LEVEL,
    )
    smile_func = opensmile.Smile(
        feature_set=FEATURE_SET,
        feature_level=FUNC_FEATURE_LEVEL,
    )

    lld_feature_names = smile_lld.feature_names
    func_feature_names = smile_func.feature_names
    logger.info(
        f"LLD features: {len(lld_feature_names)} per frame  |  "
        f"Functional features: {len(func_feature_names)} per file"
    )

    # ------------------------------------------------------------------
    # Process all wav files
    # ------------------------------------------------------------------
    t_start = time.time()
    n_ok, n_skip, n_fail = 0, 0, 0
    failures = []

    for wav_path in tqdm(wav_files, desc="Extracting openSMILE features", unit="file"):
        success, msg = extract_features_for_file(wav_path, smile_lld, smile_func)

        if success:
            if "SKIP" in msg:
                n_skip += 1
            else:
                n_ok += 1
        else:
            n_fail += 1
            failures.append(msg)
            tqdm.write(f"  {msg}")

    elapsed = time.time() - t_start

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info(f"Done in {elapsed:.1f}s")
    logger.info(
        f"  Extracted: {n_ok}  |  Skipped (existing): {n_skip}  |  Failed: {n_fail}"
    )
    logger.info(f"  Total wav files: {len(wav_files)}")

    if failures:
        logger.warning("Failed files:")
        for msg in failures:
            logger.warning(f"  {msg}")

    logger.info(
        f"\nOutput format per wav file:\n"
        f"  {{stem}}_openSMILE_lld.csv  — {len(lld_feature_names)} LLDs × T frames\n"
        f"  {{stem}}_openSMILE_func.csv — {len(func_feature_names)} functionals × 1 row"
    )


if __name__ == "__main__":
    main()
