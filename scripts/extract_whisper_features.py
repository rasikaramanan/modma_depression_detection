#!/usr/bin/env python3
"""
extract_whisper_features.py
===========================
Generates transcripts of all MODMA audio wav files using Whisper-large-v3,
and optionally saves the encoder's hidden-state features as well.

DEFAULT behaviour: writes BOTH a Mandarin and an English transcript per wav
file, and nothing else. Encoder features are opt-in via --save-pooled /
--save-frames.

Per-wav outputs (only those whose flags are set):
  - *_transcript_whisper_zh.txt — Mandarin transcript (task="transcribe")
  - *_transcript_whisper_en.txt — English transcript  (task="translate")
                                  (One or both depending on --tasks;
                                   BOTH by default.)

  - *_whisper_pooled.pt         — Mean-pooled encoder hidden states per layer.
                                  Dict: 'pooled' (n_layers, 1280) float32,
                                        'layers' list of layer indices.
                                  (Opt-in via --save-pooled.)

  - *_whisper_frames.pt         — Full temporal encoder hidden states per layer.
                                  Dict: 'hidden_states' (n_layers, T, 1280) float16,
                                        'sample_rate' 16000,
                                        'frame_rate_ms' 20,
                                        'layers' list of layer indices.
                                  T ≈ duration_seconds × 50 (20ms frame rate).
                                  (Opt-in via --save-frames.)

Audio is resampled from 44.1 kHz → 16 kHz and processed in 30-second segments
(Whisper's native window; zero-padded if shorter). Segment outputs are
concatenated along time to reconstruct the full temporal sequence.

Output files are saved alongside the source wav files, mirroring the
directory structure of audio_lanzhou_2015/.

Usage:
    python scripts/extract_whisper_features.py                              # BOTH transcripts, no encoder features (default)
    python scripts/extract_whisper_features.py --tasks transcribe            # Mandarin transcript only
    python scripts/extract_whisper_features.py --tasks translate             # English transcript only
    python scripts/extract_whisper_features.py --save-pooled                 # also save pooled encoder features
    python scripts/extract_whisper_features.py --save-pooled --save-frames   # also save frame-level features
    python scripts/extract_whisper_features.py --layers 0 16 32              # limit which encoder layers are saved
    python scripts/extract_whisper_features.py --device cpu                  # force CPU
    python scripts/extract_whisper_features.py --dry-run                     # list files without processing

Requirements:
    pip install torch torchaudio transformers tqdm soundfile
"""

import sys
import time
import logging
import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from tqdm import tqdm
from transformers import WhisperModel, WhisperProcessor, WhisperForConditionalGeneration

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

MODEL_NAME = "openai/whisper-large-v3"   # 32 encoder layers, 1280-dim
TARGET_SR = 16_000                        # Whisper expects 16 kHz
SEGMENT_SECONDS = 30                      # Whisper's native input window
SEGMENT_SAMPLES = TARGET_SR * SEGMENT_SECONDS  # 480,000 samples per segment

# Source language of MODMA audio. Whisper will translate → English.
SOURCE_LANGUAGE = "zh"   # Mandarin Chinese — source language of MODMA audio

# Mapping from Whisper task → language code used in the transcript filename.
# "transcribe" preserves the source language (Mandarin, "zh");
# "translate"  always produces English ("en").
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
# Audio loading & resampling
# ---------------------------------------------------------------------------


def load_and_resample(wav_path: Path, target_sr: int = TARGET_SR) -> torch.Tensor:
    """
    Load a wav file and resample to target_sr.

    Uses soundfile for loading (avoids torchaudio 2.8+'s TorchCodec
    dependency) and torchaudio.functional.resample for resampling.

    Returns a 1-D float32 tensor of audio samples.
    """
    # soundfile returns (num_samples,) for mono or (num_samples, channels)
    # for multi-channel. Always return float32.
    data, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)

    if data.ndim == 2:
        # Mix to mono by averaging channels
        data = data.mean(axis=1)

    waveform = torch.from_numpy(np.ascontiguousarray(data))  # (num_samples,) float32

    # Resample if needed
    if sr != target_sr:
        waveform = torchaudio.functional.resample(
            waveform, orig_freq=sr, new_freq=target_sr
        )

    return waveform  # (num_samples,)

# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------


def segment_audio(waveform: torch.Tensor, segment_samples: int) -> list[torch.Tensor]:
    """
    Split waveform into fixed-length segments, zero-padding the last one
    if it's shorter than segment_samples.

    Returns a list of 1-D tensors, each of length segment_samples.
    """
    total = waveform.shape[0]
    segments = []

    for start in range(0, total, segment_samples):
        seg = waveform[start : start + segment_samples]
        if seg.shape[0] < segment_samples:
            pad = torch.zeros(segment_samples - seg.shape[0], dtype=seg.dtype)
            seg = torch.cat([seg, pad])
        segments.append(seg)

    return segments

# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------


@torch.no_grad()
def extract_whisper_for_file(
    wav_path: Path,
    encoder_model: WhisperModel,
    asr_model: WhisperForConditionalGeneration,
    processor: WhisperProcessor,
    device: torch.device,
    layer_indices: list[int],
    tasks: list[str],
    save_pooled: bool = False,
    save_frames: bool = False,
) -> tuple[bool, str]:
    """
    Generate Whisper transcript(s) for a single wav file, and optionally save
    encoder hidden states. Each requested task in `tasks` ("transcribe" and/or
    "translate") produces a separate transcript file suffixed with its output
    language code (_zh for transcribe, _en for translate).

    Saves (as needed — skips individual outputs that already exist):
      - {stem}_transcript_whisper_{lang}.txt     (one per task)
      - {stem}_whisper_pooled.pt                 (only if save_pooled=True)
      - {stem}_whisper_frames.pt                 (only if save_frames=True)

    Returns (success: bool, message: str).
    """
    stem = wav_path.stem
    out_dir = wav_path.parent

    frames_path = out_dir / f"{stem}_whisper_frames.pt"
    pooled_path = out_dir / f"{stem}_whisper_pooled.pt"
    transcript_paths = {
        task: out_dir / f"{stem}_transcript_whisper_{TASK_TO_LANG_CODE[task]}.txt"
        for task in tasks
    }

    # Figure out what still needs to be produced
    need_pooled = save_pooled and not pooled_path.exists()
    need_frames = save_frames and not frames_path.exists()
    tasks_to_run = [t for t in tasks if not transcript_paths[t].exists()]

    if not need_pooled and not need_frames and not tasks_to_run:
        return True, "SKIP"

    try:
        # Load and resample audio
        waveform = load_and_resample(wav_path, TARGET_SR)

        # Segment into 30s chunks (Whisper's native window)
        segments = segment_audio(waveform, SEGMENT_SAMPLES)
        total_real_samples = waveform.shape[0]

        # ---------------------------------------------------------------
        # Per-segment: encoder forward (once) + decoder generation per task
        # ---------------------------------------------------------------
        all_segment_hidden = []  # list of (n_layers, T_seg, 1280) tensors
        transcript_pieces: dict[str, list[str]] = {t: [] for t in tasks_to_run}

        want_encoder = need_pooled or need_frames

        # Match input dtype to the model's parameter dtype (handles the case
        # where the model ended up in fp16 while the processor returns fp32).
        model_dtype = next(asr_model.parameters()).dtype

        for seg in segments:
            inputs = processor(
                seg.numpy(),
                sampling_rate=TARGET_SR,
                return_tensors="pt",
            )
            input_features = inputs.input_features.to(device=device, dtype=model_dtype)

            # --- Encoder forward (only if we still need pooled/frames) ---
            if want_encoder:
                encoder_outputs = encoder_model.encoder(
                    input_features,
                    output_hidden_states=True,
                )
                selected = torch.stack(
                    [
                        encoder_outputs.hidden_states[i].squeeze(0).cpu()
                        for i in layer_indices
                    ],
                    dim=0,
                )  # (n_layers, T_seg, 1280)
                all_segment_hidden.append(selected)

            # --- Decoder generation, once per requested task ---
            for task in tasks_to_run:
                forced_decoder_ids = processor.get_decoder_prompt_ids(
                    language=SOURCE_LANGUAGE,
                    task=task,
                )
                generated_ids = asr_model.generate(
                    input_features,
                    forced_decoder_ids=forced_decoder_ids,
                    max_new_tokens=440,
                )
                text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                transcript_pieces[task].append(text.strip())

        # ---------------------------------------------------------------
        # Save encoder features
        # ---------------------------------------------------------------
        if want_encoder:
            full_hidden = torch.cat(all_segment_hidden, dim=1)  # (n_layers, T_total, 1280)
            # Whisper encoder outputs ~50 frames/sec (20 ms/frame).
            real_frames = int(total_real_samples / TARGET_SR * 50)
            real_frames = min(real_frames, full_hidden.shape[1])
            full_hidden = full_hidden[:, :real_frames, :]

            if need_pooled:
                pooled = full_hidden.float().mean(dim=1)  # (n_layers, 1280) float32
                torch.save(
                    {"pooled": pooled, "layers": layer_indices},
                    str(pooled_path),
                )

            if need_frames:
                torch.save(
                    {
                        "hidden_states": full_hidden.half(),  # (n_layers, T, 1280) float16
                        "sample_rate": TARGET_SR,
                        "frame_rate_ms": 20,
                        "n_frames": real_frames,
                        "layers": layer_indices,
                    },
                    str(frames_path),
                )

        # ---------------------------------------------------------------
        # Save transcripts
        # ---------------------------------------------------------------
        for task in tasks_to_run:
            full_transcript = " ".join(p for p in transcript_pieces[task] if p).strip()
            transcript_paths[task].write_text(full_transcript + "\n", encoding="utf-8")

        return True, "OK"

    except Exception as e:
        return False, f"FAIL: {wav_path} — {e}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Extract frozen Whisper encoder hidden states and English "
            "transcripts from MODMA audio files."
        )
    )
    parser.add_argument(
        "--audio-dir",
        type=Path,
        default=AUDIO_DIR,
        help="Root directory of audio_lanzhou_2015 (default: auto-detected from repo).",
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Which Whisper encoder layers to extract (0-indexed, 0 = post-conv "
            "embedding, 1..N = transformer layers). "
            "Default: all encoder hidden states (whisper-large-v3 → 0..32)."
        ),
    )
    parser.add_argument(
        "--save-pooled",
        action="store_true",
        help=(
            "Also save mean-pooled encoder hidden states "
            "({stem}_whisper_pooled.pt). Off by default — default output is "
            "transcripts only."
        ),
    )
    parser.add_argument(
        "--save-frames",
        action="store_true",
        help=(
            "Also save full frame-level encoder hidden states "
            "({stem}_whisper_frames.pt). Off by default; frames are large "
            "(~92 GB across all MODMA wavs at all layers). "
            "Implies --save-pooled."
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
            "transcript ({stem}_transcript_whisper_zh.txt); 'translate' "
            "produces an English transcript ({stem}_transcript_whisper_en.txt). "
            "Default: both."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run on: 'cuda', 'mps', or 'cpu'. Default: auto-detect.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List files that would be processed without extracting.",
    )
    args = parser.parse_args()

    audio_dir = args.audio_dir.resolve()

    # ------------------------------------------------------------------
    # Resolve device
    # ------------------------------------------------------------------
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

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
    logger.info(f"Device: {device}")

    if args.dry_run:
        for f in wav_files:
            print(f"  {f.relative_to(audio_dir)}")
        print(f"\nTotal: {len(wav_files)} files. Use without --dry-run to extract.")
        return

    # ------------------------------------------------------------------
    # Load model + processor
    # ------------------------------------------------------------------
    logger.info(f"Loading {MODEL_NAME} (this may take a minute on first run)...")
    processor = WhisperProcessor.from_pretrained(MODEL_NAME)

    # Separate model handles for encoder features and generation. Both wrap
    # the same weights under the hood via `from_pretrained`, but keeping two
    # objects keeps the encoder path and the ASR generation path tidy.
    # Force float32 to avoid fp16 bias/input mismatches (seen on MPS).
    encoder_model = (
        WhisperModel.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
        .to(device)
        .to(torch.float32)
        .eval()
    )
    asr_model = (
        WhisperForConditionalGeneration.from_pretrained(
            MODEL_NAME, torch_dtype=torch.float32
        )
        .to(device)
        .to(torch.float32)
        .eval()
    )
    logger.info("Model loaded and set to eval mode (frozen).")

    # ------------------------------------------------------------------
    # Resolve layers
    # ------------------------------------------------------------------
    # Whisper-large-v3 has 32 encoder transformer layers → hidden_states tuple
    # of length 33 (index 0 = post-conv embedding, 1..32 = transformer layers).
    n_encoder_layers = encoder_model.config.encoder_layers
    n_hidden_states = n_encoder_layers + 1  # includes post-conv embedding

    if args.layers is not None:
        layer_indices = sorted(set(args.layers))
        for idx in layer_indices:
            if idx < 0 or idx >= n_hidden_states:
                logger.error(
                    f"Invalid layer index {idx}. Must be 0..{n_hidden_states - 1}."
                )
                sys.exit(1)
    else:
        layer_indices = list(range(n_hidden_states))

    tasks = sorted(set(args.tasks))
    # --save-frames implies --save-pooled (frames already require encoder pass)
    save_pooled = args.save_pooled or args.save_frames
    save_frames = args.save_frames
    logger.info(f"Layers to extract: {layer_indices} ({len(layer_indices)} total)")
    logger.info(f"Save pooled encoder features: {save_pooled}")
    logger.info(f"Save frame-level encoder features: {save_frames}")
    logger.info(
        f"Transcript tasks: {tasks}  →  "
        f"files: "
        + ", ".join(
            f"{{stem}}_transcript_whisper_{TASK_TO_LANG_CODE[t]}.txt" for t in tasks
        )
    )

    # ------------------------------------------------------------------
    # Process all wav files
    # ------------------------------------------------------------------
    t_start = time.time()
    n_ok, n_skip, n_fail = 0, 0, 0
    failures = []

    for wav_path in tqdm(wav_files, desc="Extracting Whisper features", unit="file"):
        success, msg = extract_whisper_for_file(
            wav_path=wav_path,
            encoder_model=encoder_model,
            asr_model=asr_model,
            processor=processor,
            device=device,
            layer_indices=layer_indices,
            tasks=tasks,
            save_pooled=save_pooled,
            save_frames=save_frames,
        )

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

    hidden_dim = encoder_model.config.d_model  # 1280 for large-v3
    transcript_lines = "".join(
        f"  {{stem}}_transcript_whisper_{TASK_TO_LANG_CODE[t]}.txt  — "
        f"{'Mandarin' if t == 'transcribe' else 'English'} transcript (plain text)\n"
        for t in tasks
    )
    encoder_lines = ""
    if save_pooled:
        encoder_lines += (
            f"  {{stem}}_whisper_pooled.pt       — mean-pooled: "
            f"({len(layer_indices)}, {hidden_dim}) float32\n"
        )
    else:
        encoder_lines += "  (pooled encoder features not saved — pass --save-pooled to enable)\n"
    if save_frames:
        encoder_lines += (
            f"  {{stem}}_whisper_frames.pt       — frame-level: "
            f"({len(layer_indices)}, T, {hidden_dim}) float16\n"
        )
    else:
        encoder_lines += "  (frame-level encoder features not saved — pass --save-frames to enable)\n"

    logger.info(
        "\nOutput format per wav file:\n"
        + transcript_lines
        + encoder_lines
        + f"  Layers (if encoder features saved): {layer_indices}"
    )


if __name__ == "__main__":
    main()
