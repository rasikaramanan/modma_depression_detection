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
directory structure of audio_lanzhou_2015/.

Versioning policy: existing transcripts are NEVER overwritten. If the base
output filename already exists, the new transcript is written to
`{stem}..._v2.txt`, then `_v3.txt`, and so on. Run the script repeatedly
with different parameters to accumulate alternative transcriptions per wav.

Usage:
    # Full corpus, default parameters
    python scripts/extract_whisper_features.py

    # Re-transcribe a specific subset (e.g., audit-flagged failures)
    python scripts/extract_whisper_features.py --files paths.txt

    # Re-transcribe with looser settings (target empty / hallucinated outputs)
    python scripts/extract_whisper_features.py \
        --files audit_failures.txt \
        --no-vad \
        --initial-prompt "这是一段中文心理访谈录音。" \
        --compute-type float32

    # Constrain to one task or one language
    python scripts/extract_whisper_features.py --tasks transcribe
    python scripts/extract_whisper_features.py --tasks translate

    # Custom temperature schedule (default is 0.0,0.2,0.4,0.6,0.8,1.0)
    python scripts/extract_whisper_features.py --temperature 0.6

    # Local model snapshot path (when huggingface_hub can't download)
    python scripts/extract_whisper_features.py \
        --model ~/.cache/huggingface/hub/models--Systran--faster-whisper-large-v3/snapshots/<hash>

    # Dry-run shows the file list without transcribing
    python scripts/extract_whisper_features.py --files paths.txt --dry-run

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
# Helpers
# ---------------------------------------------------------------------------


def read_manifest(manifest_path: Path) -> list[Path]:
    """Read a manifest file listing wav paths to process.

    Format: one path per line. Lines starting with '#' are comments; blank
    lines are ignored.
    """
    paths: list[Path] = []
    missing: list[str] = []
    with manifest_path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            p = Path(line).expanduser().resolve()
            if not p.exists():
                missing.append(line)
            else:
                paths.append(p)
    if missing:
        preview = "\n".join(f"  {m}" for m in missing[:10])
        more = f"\n  ... and {len(missing) - 10} more" if len(missing) > 10 else ""
        raise FileNotFoundError(
            f"{len(missing)} path(s) in manifest do not exist:\n{preview}{more}"
        )
    return paths


def next_versioned_path(base_path: Path) -> Path:
    """Return a non-existent output path, auto-versioning if `base_path` is
    already taken. Versions are suffixed `_v2`, `_v3`, ... before the file
    extension. Existing files are never overwritten.
    """
    if not base_path.exists():
        return base_path
    stem = base_path.stem
    suffix = base_path.suffix
    parent = base_path.parent
    v = 2
    while True:
        candidate = parent / f"{stem}_v{v}{suffix}"
        if not candidate.exists():
            return candidate
        v += 1


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------


def extract_whisper_for_file(
    wav_path: Path,
    model: WhisperModel,
    tasks: list[str],
    *,
    initial_prompt: str | None = None,
    vad_filter: bool = True,
    temperature: list[float] | float = 0.0,
) -> tuple[bool, str, list[Path]]:
    """
    Generate transcripts for a single wav file using the provided faster-whisper
    model. Each requested task in `tasks` ("transcribe" and/or "translate")
    produces a separate transcript file suffixed with its output language code
    (_zh for transcribe, _en for translate).

    Output filenames are auto-versioned: if the base output path already
    exists, the new transcript is written to `{stem}..._v2.txt`, then
    `_v3.txt`, etc. The base path is never overwritten.

    Returns (success: bool, message: str, output_paths: list[Path]).
    """
    stem = wav_path.stem
    out_dir = wav_path.parent

    base_paths = {
        task: out_dir / f"{stem}_transcript_faster_whisper_{TASK_TO_LANG_CODE[task]}.txt"
        for task in tasks
    }
    transcript_paths = {task: next_versioned_path(p) for task, p in base_paths.items()}

    try:
        audio_path = str(wav_path)

        for task in tasks:
            transcribe_kwargs = dict(
                language=SOURCE_LANGUAGE,
                task=task,
                beam_size=5,
                condition_on_previous_text=False,
                vad_filter=vad_filter,
                temperature=temperature,
            )
            if vad_filter:
                transcribe_kwargs["vad_parameters"] = dict(min_silence_duration_ms=500)
            if initial_prompt:
                transcribe_kwargs["initial_prompt"] = initial_prompt

            segments, info = model.transcribe(audio_path, **transcribe_kwargs)
            full_text = "".join(segment.text for segment in segments).strip()
            transcript_paths[task].write_text(full_text + "\n", encoding="utf-8")

        return True, "OK", list(transcript_paths.values())

    except Exception as e:
        return False, f"FAIL: {wav_path} — {e}", []


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Extract Mandarin and/or English faster-whisper transcripts from MODMA audio files."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=MODEL_NAME,
        help=(
            "Model identifier passed to faster-whisper's WhisperModel(). "
            "Accepts either a HuggingFace model id (e.g. 'large-v3') OR a "
            "local snapshot directory path. Use a local path when the conda "
            "env's huggingface_hub cannot download (e.g. missing httpx)."
        ),
    )
    parser.add_argument(
        "--audio-dir",
        type=Path,
        default=AUDIO_DIR,
        help=(
            "Root directory of audio_lanzhou_2015 (default: auto-detected "
            "from repo). Ignored if --files is provided."
        ),
    )
    parser.add_argument(
        "--files",
        type=Path,
        default=None,
        help=(
            "Optional path to a manifest file listing specific .wav paths to "
            "process (one absolute or relative path per line; lines starting "
            "with '#' are comments; blank lines are ignored). When provided, "
            "--audio-dir is ignored."
        ),
    )
    parser.add_argument(
        "--initial-prompt",
        type=str,
        default="",
        help=(
            "Optional text prompt to seed Whisper's decoder. Useful for "
            "anchoring language choice or domain context. Empty by default."
        ),
    )
    parser.add_argument(
        "--no-vad",
        action="store_true",
        help=(
            "Disable Silero VAD filtering. By default VAD is on with "
            "min_silence_duration_ms=500."
        ),
    )
    parser.add_argument(
        "--temperature",
        type=str,
        default="0.0,0.2,0.4,0.6,0.8,1.0",
        help=(
            "Comma-separated list of temperatures for faster-whisper's "
            "temperature-fallback decoding. Default: "
            "'0.0,0.2,0.4,0.6,0.8,1.0' (faster-whisper's own default)."
        ),
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

    # Parse --temperature
    try:
        temperature_list = [float(t.strip()) for t in args.temperature.split(",") if t.strip()]
    except ValueError as e:
        logger.error(f"Failed to parse --temperature {args.temperature!r}: {e}")
        sys.exit(1)
    if not temperature_list:
        logger.error("--temperature parsed to an empty list")
        sys.exit(1)
    temperature_arg: float | list[float] = (
        temperature_list[0] if len(temperature_list) == 1 else temperature_list
    )
    initial_prompt: str | None = args.initial_prompt or None
    vad_filter = not args.no_vad

    # ------------------------------------------------------------------
    # Discover wav files: from manifest if --files given, else rglob
    # ------------------------------------------------------------------
    audio_dir: Path | None = None
    if args.files is not None:
        if not args.files.exists():
            logger.error(f"Manifest file not found: {args.files}")
            sys.exit(1)
        try:
            wav_files = read_manifest(args.files)
        except FileNotFoundError as e:
            logger.error(str(e))
            sys.exit(1)
        if not wav_files:
            logger.error(f"Manifest at {args.files} is empty (no usable paths).")
            sys.exit(1)
        logger.info(f"Loaded {len(wav_files)} wav paths from manifest {args.files}")
    else:
        audio_dir = args.audio_dir.resolve()
        if not audio_dir.exists():
            logger.error(f"Audio directory not found: {audio_dir}")
            logger.error(
                "Make sure the MODMA data is located at:\n"
                "  <repo>/CSCI567 Project/modma_data/audio_lanzhou_2015/\n"
                "Or pass --audio-dir explicitly, or provide --files."
            )
            sys.exit(1)
        wav_files = sorted(audio_dir.rglob("*.wav"))
        if not wav_files:
            logger.error(f"No .wav files found under {audio_dir}")
            sys.exit(1)

    subject_ids = sorted({f.parent.name for f in wav_files})
    logger.info(
        f"Processing {len(wav_files)} wav files across {len(subject_ids)} subjects"
    )
    logger.info(f"Device: {args.device}")
    logger.info(f"Compute type: {args.compute_type}")
    logger.info(f"CPU threads: {args.cpu_threads}")
    logger.info(f"VAD filter: {'OFF' if args.no_vad else 'ON (Silero, min_silence_duration_ms=500)'}")
    logger.info(f"Temperature: {temperature_list}")
    logger.info(
        f"Initial prompt: {initial_prompt!r}" if initial_prompt
        else "Initial prompt: (none)"
    )

    if args.dry_run:
        for f in wav_files:
            try:
                display = f.relative_to(audio_dir) if audio_dir is not None else f
            except ValueError:
                display = f
            print(f"  {display}")
        print(f"\nTotal: {len(wav_files)} files. Use without --dry-run to extract.")
        return

    # ------------------------------------------------------------------
    # Load faster-whisper model
    # ------------------------------------------------------------------
    logger.info(f"Loading {args.model} (this may take a minute on first run)...")

    try:
        model = WhisperModel(
            args.model,
            device=args.device,
            compute_type=args.compute_type,
            cpu_threads=args.cpu_threads,
        )
    except Exception as e:
        logger.error(f"Failed to load model {args.model}: {e}")
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
    logger.info(
        "Versioning policy: if base output already exists, transcripts are "
        "written to {stem}..._v2.txt, then _v3.txt, and so on. Existing "
        "files are NEVER overwritten."
    )

    # ------------------------------------------------------------------
    # Process all wav files
    # ------------------------------------------------------------------
    t_start = time.time()
    n_ok, n_fail = 0, 0
    failures = []

    for wav_path in tqdm(wav_files, desc="Extracting faster-whisper transcripts", unit="file"):
        success, msg, written_paths = extract_whisper_for_file(
            wav_path=wav_path,
            model=model,
            tasks=tasks,
            initial_prompt=initial_prompt,
            vad_filter=vad_filter,
            temperature=temperature_arg,
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
    logger.info(f"  Extracted: {n_ok}  |  Failed: {n_fail}")
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