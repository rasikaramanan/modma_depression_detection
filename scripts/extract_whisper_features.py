#!/usr/bin/env python3
"""
extract_whisper_features.py
===========================
Generates transcripts of all MODMA audio wav files using faster-whisper
(CTranslate2). The script produces Mandarin and/or English transcripts per
WAV file and nothing else.

DEFAULT behaviour: writes BOTH a Mandarin and an English transcript per wav
file.

Per-wav outputs:
  - *_transcript_faster_whisper_zh.txt — Mandarin transcript (task="transcribe")
  - *_transcript_faster_whisper_en.txt — English translation (task="translate")
                                         (One or both depending on --tasks;
                                          BOTH by default.)

This version is optimized for Apple Silicon Macs:
  - defaults to CPU instead of trying to use MPS
  - defaults to int8 compute for better memory/speed tradeoff on CPU
  - removes torch/torchaudio preprocessing entirely
  - passes the original file path directly to faster-whisper

Output files are saved alongside the source wav files, mirroring the
directory structure of audio_lanzhou_2015/. Existing outputs are
overwritten on every run.

Usage:
    python scripts/extract_faster_whisper_features.py
    python scripts/extract_faster_whisper_features.py --tasks transcribe
    python scripts/extract_faster_whisper_features.py --tasks translate
    python scripts/extract_faster_whisper_features.py --device cpu
    python scripts/extract_faster_whisper_features.py --compute-type float32
    python scripts/extract_faster_whisper_features.py --dry-run

Requirements:
    pip install tqdm faster-whisper
"""

import sys
import os
import time
import logging
import argparse
from pathlib import Path

from tqdm import tqdm
from faster_whisper import WhisperModel

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
AUDIO_DIR = (
    REPO_ROOT
    / "CSCI567 Project"
    / "modma_data"
    / "audio_lanzhou_2015"
)

MODEL_NAME = "large-v3"
SOURCE_LANGUAGE = "zh"

TASK_TO_LANG_CODE = {
    "transcribe": SOURCE_LANGUAGE,
    "translate": "en",
}

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


def extract_whisper_for_file(
    wav_path: Path,
    model: WhisperModel,
    tasks: list[str],
) -> tuple[bool, str]:
    """
    Generate transcripts for a single wav file using the provided faster-whisper
    model. Each requested task in `tasks` ("transcribe" and/or "translate")
    produces a separate transcript file suffixed with its output language code
    (_zh for transcribe, _en for translate).

    Always overwrites existing outputs.

    Saves:
      - {stem}_transcript_faster_whisper_{lang}.txt     (one per task)

    Returns (success: bool, message: str).
    """
    stem = wav_path.stem
    out_dir = wav_path.parent

    transcript_paths = {
        task: out_dir / f"{stem}_transcript_faster_whisper_{TASK_TO_LANG_CODE[task]}.txt"
        for task in tasks
    }

    try:
        audio_path = str(wav_path)

        for task in tasks:
            segments, info = model.transcribe(
                audio_path,
                language=SOURCE_LANGUAGE,
                task=task,
                beam_size=5,
                condition_on_previous_text=False,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500)
            )

            full_text = "".join(segment.text for segment in segments).strip()
            transcript_paths[task].write_text(full_text + "\n", encoding="utf-8")

        return True, "OK"

    except Exception as e:
        return False, f"FAIL: {wav_path} — {e}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Extract Mandarin and/or English faster-whisper transcripts from MODMA audio files."
    )
    parser.add_argument(
        "--audio-dir",
        type=Path,
        default=AUDIO_DIR,
        help="Root directory of audio_lanzhou_2015 (default: auto-detected from repo).",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=["transcribe", "translate"],
        choices=["transcribe", "translate"],
        help=(
            "Which Whisper tasks to run. 'transcribe' produces a Mandarin "
            "transcript ({stem}_transcript_faster_whisper_zh.txt); 'translate' "
            "produces an English transcript ({stem}_transcript_faster_whisper_en.txt). "
            "Default: both."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help=(
            "Device to run on. Default: cpu. "
            "For Apple Silicon, keep this as cpu."
        ),
    )
    parser.add_argument(
        "--compute-type",
        type=str,
        default="int8",
        choices=["int8", "float32", "float16"],
        help=(
            "Numeric precision to use when loading the model. "
            "Default: int8 for a better CPU memory/speed tradeoff."
        ),
    )
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=max(1, os.cpu_count() or 1),
        help="Number of CPU threads to give faster-whisper. Default: all available cores.",
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

    subject_ids = sorted({f.parent.name for f in wav_files})
    logger.info(
        f"Found {len(wav_files)} wav files across {len(subject_ids)} subjects "
        f"in {audio_dir}"
    )
    logger.info(f"Device: {args.device}")
    logger.info(f"Compute type: {args.compute_type}")
    logger.info(f"CPU threads: {args.cpu_threads}")

    if args.dry_run:
        for f in wav_files:
            print(f"  {f.relative_to(audio_dir)}")
        print(f"\nTotal: {len(wav_files)} files. Use without --dry-run to extract.")
        return

    # ------------------------------------------------------------------
    # Load faster-whisper model
    # ------------------------------------------------------------------
    logger.info(f"Loading {MODEL_NAME} (this may take a minute on first run)...")

    try:
        model = WhisperModel(
            MODEL_NAME,
            device=args.device,
            compute_type=args.compute_type,
            cpu_threads=args.cpu_threads,
        )
    except Exception as e:
        logger.error(f"Failed to load model {MODEL_NAME}: {e}")
        sys.exit(1)

    logger.info("Model loaded and ready for transcription.")

    tasks = sorted(set(args.tasks))
    logger.info(
        f"Transcript tasks: {tasks}  →  files: "
        + ", ".join(
            f"{{stem}}_transcript_faster_whisper_{TASK_TO_LANG_CODE[t]}.txt"
            for t in tasks
        )
    )
    logger.info("Overwrite policy: existing output files are OVERWRITTEN.")

    # ------------------------------------------------------------------
    # Process all wav files
    # ------------------------------------------------------------------
    t_start = time.time()
    n_ok, n_fail = 0, 0
    failures = []

    for wav_path in tqdm(wav_files, desc="Extracting faster-whisper transcripts", unit="file"):
        success, msg = extract_whisper_for_file(
            wav_path=wav_path,
            model=model,
            tasks=tasks,
        )

        if success:
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
    logger.info(f"  Extracted (overwritten): {n_ok}  |  Failed: {n_fail}")
    logger.info(f"  Total wav files: {len(wav_files)}")

    if failures:
        logger.warning("Failed files:")
        for msg in failures:
            logger.warning(f"  {msg}")

    transcript_lines = "".join(
        f"  {{stem}}_transcript_faster_whisper_{TASK_TO_LANG_CODE[t]}.txt  — "
        f"{'Mandarin' if t == 'transcribe' else 'English'} transcript (plain text)\n"
        for t in tasks
    )

    logger.info("\nOutput format per wav file:\n" + transcript_lines)


if __name__ == "__main__":
    main()