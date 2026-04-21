#!/usr/bin/env python3
"""
extract_wavlm_features.py
=========================
Extracts frozen WavLM-Large hidden states from all MODMA audio wav files.

For each wav file, produces two .pt (PyTorch) outputs:
  1. *_wavlm_frames.pt  — Full temporal hidden states per layer.
                          Dict with:
                            'hidden_states': float16 tensor (n_layers, T, 1024)
                            'sample_rate': 16000
                            'frame_rate_ms': 20
                            'layers': list of layer indices extracted
                          T ≈ duration_seconds × 50 (20ms frame rate).

  2. *_wavlm_pooled.pt  — Mean-pooled across time per layer.
                          Dict with:
                            'pooled': float32 tensor (n_layers, 1024)
                            'layers': list of layer indices extracted

The frame-level file uses float16 to manage disk usage (~24 layers × T × 1024
× 2 bytes). The pooled file uses float32 since it's tiny (24 × 1024 × 4 bytes
= 98 KB per file).

Audio is resampled from 44.1 kHz → 16 kHz and processed in 10-second segments
(zero-padded if shorter) to manage GPU memory. Segment outputs are concatenated
along time to reconstruct the full temporal sequence.

Output files are saved alongside the source wav files, mirroring the directory
structure of audio_lanzhou_2015/.

Usage:
    python scripts/extract_wavlm_features.py                     # all 24 layers, GPU if available
    python scripts/extract_wavlm_features.py --layers 3 7 24     # specific layers only
    python scripts/extract_wavlm_features.py --pooled-only        # skip frame-level (saves disk)
    python scripts/extract_wavlm_features.py --device cpu         # force CPU
    python scripts/extract_wavlm_features.py --dry-run            # list files without processing

Requirements:
    pip install torch torchaudio transformers tqdm
"""

import sys
import time
import logging
import argparse
from pathlib import Path

import torch
import torchaudio
from tqdm import tqdm
from transformers import WavLMModel, Wav2Vec2FeatureExtractor

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

MODEL_NAME = "microsoft/wavlm-large"   # 24 transformer layers, 1024-dim
TARGET_SR = 16_000                      # WavLM expects 16 kHz
SEGMENT_SECONDS = 10                    # process in 10s chunks for memory
SEGMENT_SAMPLES = TARGET_SR * SEGMENT_SECONDS  # 160,000 samples per segment

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

    Returns a 1-D float32 tensor of audio samples.
    """
    waveform, sr = torchaudio.load(str(wav_path))

    # Convert to mono if multi-channel
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample if needed
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resampler(waveform)

    return waveform.squeeze(0)  # (num_samples,)

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
            # Zero-pad the final segment
            pad = torch.zeros(segment_samples - seg.shape[0], dtype=seg.dtype)
            seg = torch.cat([seg, pad])
        segments.append(seg)

    return segments

# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------


@torch.no_grad()
def extract_wavlm_for_file(
    wav_path: Path,
    model: WavLMModel,
    feature_extractor: Wav2Vec2FeatureExtractor,
    device: torch.device,
    layer_indices: list[int],
    pooled_only: bool = False,
) -> tuple[bool, str]:
    """
    Extract WavLM hidden states for a single wav file.

    Saves:
      - {stem}_wavlm_frames.pt  (unless pooled_only=True)
      - {stem}_wavlm_pooled.pt

    Returns (success: bool, message: str).
    """
    stem = wav_path.stem
    out_dir = wav_path.parent

    frames_path = out_dir / f"{stem}_wavlm_frames.pt"
    pooled_path = out_dir / f"{stem}_wavlm_pooled.pt"

    # Skip if outputs already exist (resume-friendly)
    if pooled_only:
        if pooled_path.exists():
            return True, "SKIP"
    else:
        if frames_path.exists() and pooled_path.exists():
            return True, "SKIP"

    try:
        # Load and resample audio
        waveform = load_and_resample(wav_path, TARGET_SR)

        # Segment into 10s chunks
        segments = segment_audio(waveform, SEGMENT_SAMPLES)

        # Track how many real (non-padded) frames exist
        total_real_samples = waveform.shape[0]

        # Process each segment through WavLM
        all_segment_hidden = []  # list of (n_layers, T_seg, 1024) tensors

        for seg in segments:
            # Feature extractor normalises and converts to model input
            inputs = feature_extractor(
                seg.numpy(),
                sampling_rate=TARGET_SR,
                return_tensors="pt",
                padding=False,
            )
            input_values = inputs.input_values.to(device)

            # Forward pass with all hidden states
            outputs = model(input_values, output_hidden_states=True)

            # outputs.hidden_states is a tuple of (batch=1, T_seg, 1024)
            # for layers 0..24 (layer 0 = CNN feature encoder output)
            # We select the requested layer indices
            selected = torch.stack(
                [outputs.hidden_states[i].squeeze(0).cpu() for i in layer_indices],
                dim=0,
            )  # (n_layers, T_seg, 1024)

            all_segment_hidden.append(selected)

        # Concatenate segments along time dimension
        full_hidden = torch.cat(all_segment_hidden, dim=1)  # (n_layers, T_total, 1024)

        # Trim to actual audio length (remove frames from zero-padded tail)
        # WavLM's CNN encoder has a receptive field that maps ~20ms per frame
        # Approximate real frame count from real sample count
        real_frames = int(total_real_samples / TARGET_SR * 50)  # 50 frames/sec = 20ms/frame
        real_frames = min(real_frames, full_hidden.shape[1])
        full_hidden = full_hidden[:, :real_frames, :]

        # --- Save pooled (mean across time) ---
        pooled = full_hidden.float().mean(dim=1)  # (n_layers, 1024), float32
        torch.save(
            {
                "pooled": pooled,
                "layers": layer_indices,
            },
            str(pooled_path),
        )

        # --- Save frame-level (float16 to save disk) ---
        if not pooled_only:
            torch.save(
                {
                    "hidden_states": full_hidden.half(),  # (n_layers, T, 1024), float16
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
        description="Extract frozen WavLM-Large hidden states from MODMA audio files."
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
            "Which WavLM layers to extract (0-indexed, 0=CNN encoder, 1-24=transformer). "
            "Default: all 25 layers (0-24). "
            "Example: --layers 3 7 24 for the most informative layers per Maji et al."
        ),
    )
    parser.add_argument(
        "--pooled-only",
        action="store_true",
        help="Only save mean-pooled embeddings (skip frame-level .pt files to save disk).",
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
    # Resolve layers
    # ------------------------------------------------------------------
    # WavLM-Large has 25 hidden states: index 0 = CNN feature encoder,
    # indices 1-24 = transformer layers 1-24.
    if args.layers is not None:
        layer_indices = sorted(set(args.layers))
        for idx in layer_indices:
            if idx < 0 or idx > 24:
                logger.error(f"Invalid layer index {idx}. Must be 0-24.")
                sys.exit(1)
    else:
        layer_indices = list(range(25))  # 0 through 24

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
    logger.info(f"Layers to extract: {layer_indices} ({len(layer_indices)} total)")
    logger.info(f"Pooled only: {args.pooled_only}")

    if args.dry_run:
        for f in wav_files:
            print(f"  {f.relative_to(audio_dir)}")
        print(f"\nTotal: {len(wav_files)} files. Use without --dry-run to extract.")
        return

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    logger.info(f"Loading {MODEL_NAME} (this may take a minute on first run)...")
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
    model = WavLMModel.from_pretrained(MODEL_NAME)
    model = model.to(device)
    model.eval()
    logger.info("Model loaded and set to eval mode (frozen).")

    # ------------------------------------------------------------------
    # Process all wav files
    # ------------------------------------------------------------------
    t_start = time.time()
    n_ok, n_skip, n_fail = 0, 0, 0
    failures = []

    for wav_path in tqdm(wav_files, desc="Extracting WavLM features", unit="file"):
        success, msg = extract_wavlm_for_file(
            wav_path=wav_path,
            model=model,
            feature_extractor=feature_extractor,
            device=device,
            layer_indices=layer_indices,
            pooled_only=args.pooled_only,
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

    logger.info(
        f"\nOutput format per wav file:\n"
        f"  {{stem}}_wavlm_pooled.pt  — mean-pooled: ({len(layer_indices)}, 1024) float32\n"
        + (
            f"  {{stem}}_wavlm_frames.pt — frame-level: ({len(layer_indices)}, T, 1024) float16\n"
            if not args.pooled_only
            else "  (frame-level files skipped — --pooled-only)\n"
        )
        + f"  Layers: {layer_indices}"
    )


if __name__ == "__main__":
    main()
