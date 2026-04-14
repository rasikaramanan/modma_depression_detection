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

Transcription uses the HuggingFace ASR pipeline with 30s chunks and 5s
stride for long-form alignment, and Whisper's built-in temperature
fallback + `no_repeat_ngram_size=5` guardrail to prevent degenerate
decoder loops.

Encoder-feature extraction still segments the audio into 30s chunks
(Whisper's native log-mel window) and zero-pads the final segment, then
trims the concatenated hidden states back to the real audio length.
Padding is harmless here because only encoder activations are consumed
— no decoder generation runs on the padded frames.

Output files are saved alongside the source wav files, mirroring the
directory structure of audio_lanzhou_2015/. Existing outputs are
overwritten on every run.

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
from transformers import (
    WhisperModel,
    WhisperProcessor,
    WhisperForConditionalGeneration,
    pipeline,
)

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

# Long-form transcription chunking (HF ASR pipeline).
CHUNK_LENGTH_S = 30    # Whisper's native window
STRIDE_LENGTH_S = 5    # overlap on each side for long-form stitching

# Source language of MODMA audio. Whisper will translate → English.
SOURCE_LANGUAGE = "zh"   # Mandarin Chinese — source language of MODMA audio

# Mapping from Whisper task → language code used in the transcript filename.
# "transcribe" preserves the source language (Mandarin, "zh");
# "translate"  always produces English ("en").
TASK_TO_LANG_CODE = {
    "transcribe": SOURCE_LANGUAGE,
    "translate": "en",
}

# Generation guardrails. These are applied inside the HF ASR pipeline.
#   - temperature tuple → temperature fallback (retry at higher T if the
#     greedy decode tripped compression_ratio_threshold or logprob_threshold).
#   - no_repeat_ngram_size=5 → mechanically breaks degenerate loops like
#     "红红红红红红…" without distorting normal prose.
#   - condition_on_previous_text=False → prevents errors in one chunk from
#     biasing the next (Whisper's default True setting is a known driver of
#     cascading hallucinations on MODMA-style long, quiet speech).
GEN_KWARGS = {
    "no_repeat_ngram_size": 5,
    "temperature": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    "compression_ratio_threshold": 2.4,
    "logprob_threshold": -1.0,
    "condition_on_previous_text": False,
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
    data, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)

    if data.ndim == 2:
        data = data.mean(axis=1)

    waveform = torch.from_numpy(np.ascontiguousarray(data))  # (num_samples,) float32

    if sr != target_sr:
        waveform = torchaudio.functional.resample(
            waveform, orig_freq=sr, new_freq=target_sr
        )

    return waveform  # (num_samples,)

# ---------------------------------------------------------------------------
# Segmentation (encoder-feature path only)
# ---------------------------------------------------------------------------


def segment_audio(waveform: torch.Tensor, segment_samples: int) -> list[torch.Tensor]:
    """
    Split waveform into fixed-length 30s segments, zero-padding the last one
    if it's shorter. Only used when extracting encoder hidden states (which
    require a fixed 3000-frame log-mel input); the transcription path now
    uses the HF pipeline's own chunking with overlap and no generation on
    padded tails.
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
    processor: WhisperProcessor,
    asr_pipe,
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

    Always overwrites existing outputs.

    Saves:
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

    try:
        # Load and resample audio
        waveform = load_and_resample(wav_path, TARGET_SR)
        audio_np = waveform.numpy()
        total_real_samples = waveform.shape[0]

        # ---------------------------------------------------------------
        # Transcription — HF pipeline with chunking + stride + guardrails
        # ---------------------------------------------------------------
        for task in tasks:
            result = asr_pipe(
                audio_np,
                generate_kwargs={
                    "task": task,
                    "language": SOURCE_LANGUAGE,
                    **GEN_KWARGS,
                },
            )
            text = (result.get("text") or "").strip()
            transcript_paths[task].write_text(text + "\n", encoding="utf-8")

        # ---------------------------------------------------------------
        # Encoder features (opt-in) — fixed 30s log-mel windows
        # ---------------------------------------------------------------
        want_encoder = save_pooled or save_frames
        if want_encoder:
            segments = segment_audio(waveform, SEGMENT_SAMPLES)
            all_segment_hidden = []

            # Match input dtype to the encoder's parameter dtype.
            model_dtype = next(encoder_model.parameters()).dtype

            for seg in segments:
                inputs = processor(
                    seg.numpy(),
                    sampling_rate=TARGET_SR,
                    return_tensors="pt",
                )
                input_features = inputs.input_features.to(
                    device=device, dtype=model_dtype
                )
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

            full_hidden = torch.cat(all_segment_hidden, dim=1)  # (n_layers, T_total, 1280)
            # Whisper encoder outputs ~50 frames/sec (20 ms/frame).
            real_frames = int(total_real_samples / TARGET_SR * 50)
            real_frames = min(real_frames, full_hidden.shape[1])
            full_hidden = full_hidden[:, :real_frames, :]

            if save_pooled:
                pooled = full_hidden.float().mean(dim=1)  # (n_layers, 1280) float32
                torch.save(
                    {"pooled": pooled, "layers": layer_indices},
                    str(pooled_path),
                )

            if save_frames:
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

        return True, "OK"

    except Exception as e:
        return False, f"FAIL: {wav_path} — {e}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Extract frozen Whisper encoder hidden states and Mandarin/"
            "English transcripts from MODMA audio files."
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
    # Build long-form ASR pipeline (chunking + stride + guardrails)
    # ------------------------------------------------------------------
    # The pipeline handles Whisper's canonical long-form decoding: it slices
    # the audio into 30s chunks with 5s of overlap on each side, decodes each
    # chunk, and stitches outputs on matching token timestamps. Words that
    # straddle a boundary end up in at least one chunk's interior, and the
    # padded tail is never decoded as if it were real speech.
    asr_pipe = pipeline(
        task="automatic-speech-recognition",
        model=asr_model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        chunk_length_s=CHUNK_LENGTH_S,
        stride_length_s=STRIDE_LENGTH_S,
        torch_dtype=torch.float32,
        device=device,
    )

    # ------------------------------------------------------------------
    # Resolve layers
    # ------------------------------------------------------------------
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
    logger.info(
        f"Long-form chunking: chunk_length_s={CHUNK_LENGTH_S}, "
        f"stride_length_s={STRIDE_LENGTH_S}"
    )
    logger.info(f"Generation guardrails: {GEN_KWARGS}")
    logger.info("Overwrite policy: existing output files are OVERWRITTEN.")

    # ------------------------------------------------------------------
    # Process all wav files
    # ------------------------------------------------------------------
    t_start = time.time()
    n_ok, n_fail = 0, 0
    failures = []

    for wav_path in tqdm(wav_files, desc="Extracting Whisper features", unit="file"):
        success, msg = extract_whisper_for_file(
            wav_path=wav_path,
            encoder_model=encoder_model,
            processor=processor,
            asr_pipe=asr_pipe,
            device=device,
            layer_indices=layer_indices,
            tasks=tasks,
            save_pooled=save_pooled,
            save_frames=save_frames,
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
