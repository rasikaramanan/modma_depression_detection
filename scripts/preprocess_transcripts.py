#!/usr/bin/env python3
"""
preprocess_transcripts.py
=========================
Preprocess Whisper Mandarin transcripts into a feature-ready Parquet table.

Pipeline (per `notes/Feature_Pipeline_Explainer.md` and the audit-derived
preprocessing design):

  Pipeline-level (gates which transcripts get processed):
    1. Filter exclusions from data/metadata/data_quality_issues.csv
    2. Latest-version selection per (subject_id, file_num)

  Cleaned-text layer (8 steps; produces one string per transcript):
    1. NFKC Unicode normalization
    2. Whitespace normalization
    3. Lowercase Latin characters
    4+5. Traditional → Simplified Chinese + variant normalization (OpenCC)
    6. Punctuation restoration (FunASR ct-punc CT-Transformer)
    7. Punctuation symbol unification (ASCII → CJK)
    8. Repeated-punctuation collapse

  Segmentation layer (downstream of cleaned-text):
    9. Sentence segmentation (CJK sentence-boundary regex)
   10. Word segmentation (spacy_pkuseg)

Output:
    Single Parquet at data/features/transcripts_preprocessed.parquet with
    one row per processed (subject, file_num). Columns include cleaned_text,
    sentences (list[str]), tokens (list[str]), tokens_by_sentence
    (list[list[str]]), plus diagnostics. Excluded transcripts simply do not
    appear in the table.

Usage:
    # Full corpus
    python scripts/preprocess_transcripts.py

    # Smoke test on 5 specific transcripts
    python scripts/preprocess_transcripts.py \
        --files 02010001/1,02010001/19,02010002/29,02010003/13,02020025/24 \
        --output data/features/transcripts_preprocessed_smoke.parquet
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from opencc import OpenCC
from funasr import AutoModel
import spacy_pkuseg as pkuseg


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
AUDIO_DIR = REPO_ROOT / "CSCI567 Project" / "modma_data" / "audio_lanzhou_2015"
EXCLUDES_CSV = REPO_ROOT / "data" / "metadata" / "data_quality_issues.csv"
OUTPUT_PARQUET_DEFAULT = REPO_ROOT / "data" / "features" / "transcripts_preprocessed.parquet"

# ---------------------------------------------------------------------------
# Discovery patterns (mirror audit_whisper_transcripts.py)
# ---------------------------------------------------------------------------
TRANSCRIPT_GLOB = "*_transcript_faster_whisper_zh*.txt"
TRANSCRIPT_VERSION_RE = re.compile(
    r"^(?P<stem>\d{2})_transcript_faster_whisper_zh(?:_v(?P<version>\d+))?\.txt$"
)

# ---------------------------------------------------------------------------
# Step 7 (cleaned-text): ASCII → CJK punctuation mapping
# ---------------------------------------------------------------------------
# CT-Punc usually emits CJK punctuation, but mixes in ASCII forms occasionally
# (especially around Latin code-switched content). Map all ASCII punctuation
# that has a CJK equivalent to that equivalent so downstream sentence-
# segmentation regexes see one consistent form.
ASCII_TO_CJK_PUNCT = {
    ",": "，",
    ".": "。",
    "?": "？",
    "!": "！",
    ";": "；",
    ":": "：",
    "(": "（",
    ")": "）",
    "<": "《",
    ">": "》",
}
PUNCT_TRANSLATION_TABLE = str.maketrans(ASCII_TO_CJK_PUNCT)

# ---------------------------------------------------------------------------
# Step 8 (cleaned-text): punctuation-cleanup patterns
# ---------------------------------------------------------------------------
# Step 8a -- drop sentence-internal punctuation that immediately follows a
# sentence-ending punctuation (with optional whitespace between).
# ct-punc occasionally emits BOTH a sentence-ender ("。") AND a clause-internal
# punct ("，") at the same boundary, producing patterns like
# "结束了。 ，然后..." which segment into orphan sentences starting with "，".
SENTENCE_END_THEN_INTERNAL_PATTERN = re.compile(r"([。！？；…])\s*[，、：]+")

# Step 8b -- collapse runs of identical punctuation ("。。。" -> "。", "，，" -> "，").
REPEATED_PUNCT_PATTERN = re.compile(r"([。，！？；：…])\1+")

# ---------------------------------------------------------------------------
# Sentence segmentation: split on CJK sentence-boundary punctuation while
# keeping the punctuation attached to the preceding sentence.
# ---------------------------------------------------------------------------
SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[。！？；…])")


# ===========================================================================
# Pipeline-level filtering
# ===========================================================================

def load_exclusions(csv_path: Path) -> set[tuple[str, int]]:
    """
    PIPELINE STEP 1 — Filter exclusions from data_quality_issues.csv.

    Returns the set of (subject_id, file_num) pairs marked severity=exclude.
    Transcripts in this set will be skipped entirely; they will not appear
    in the output Parquet.
    """
    excludes: set[tuple[str, int]] = set()
    if not csv_path.exists():
        print(f"WARN: exclusions CSV not found at {csv_path}; nothing excluded", file=sys.stderr)
        return excludes
    with csv_path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("severity", "").strip() == "exclude":
                try:
                    excludes.add((row["subject_id"].strip(), int(row["file_number"])))
                except (KeyError, ValueError) as e:
                    print(f"WARN: skipping malformed exclusion row {row!r}: {e}", file=sys.stderr)
    return excludes


def discover_latest_transcripts(
    audio_dir: Path,
    excludes: set[tuple[str, int]],
    file_filter: set[tuple[str, int]] | None = None,
) -> list[tuple[str, int, Path, int]]:
    """
    PIPELINE STEP 2 — Latest-version selection per (subject_id, file_num).

    Walks `audio_dir` for transcripts matching `_transcript_faster_whisper_zh*.txt`,
    parses the `_vN` version suffix (no suffix = v1), groups by
    (subject_id, file_num), and returns only the highest-version path per
    group. Skips any (subject_id, file_num) in the excludes set, and any
    not in `file_filter` if provided.
    """
    candidates = sorted(audio_dir.rglob(TRANSCRIPT_GLOB))
    latest: dict[tuple[str, int], tuple[Path, int]] = {}
    for tpath in candidates:
        m = TRANSCRIPT_VERSION_RE.match(tpath.name)
        if not m:
            continue
        sid = tpath.parent.name
        fn = int(m.group("stem"))
        ver = int(m.group("version")) if m.group("version") else 1
        key = (sid, fn)
        if key in excludes:
            continue
        if file_filter is not None and key not in file_filter:
            continue
        if key not in latest or ver > latest[key][1]:
            latest[key] = (tpath, ver)
    return [(sid, fn, p, v) for (sid, fn), (p, v) in sorted(latest.items())]


# ===========================================================================
# Cleaned-text layer (8 steps)
# ===========================================================================

def clean_text(
    raw: str,
    opencc: OpenCC,
    punc_model: AutoModel,
) -> str:
    """
    Apply the 8-step cleaned-text pipeline to a raw Whisper transcript.

    Returns the cleaned string (NFKC-normalized, simplified Chinese,
    punctuation-restored, punctuation-unified, repeated-punct-collapsed).
    """
    # ---- CLEANED-TEXT STEP 1 — NFKC Unicode normalization ----
    # Unifies precomposed/decomposed forms, half-width/full-width digits and
    # Latin chars, and compatibility decompositions (e.g., circled chars).
    # Foundational: ensures all downstream regex/match operations see
    # consistent codepoints.
    text = unicodedata.normalize("NFKC", raw)

    # ---- CLEANED-TEXT STEP 2 — Whitespace normalization ----
    # Collapse runs of whitespace (spaces, tabs, newlines, NBSPs that NFKC
    # converted) into single ASCII spaces; strip leading/trailing.
    text = re.sub(r"\s+", " ", text).strip()

    # ---- CLEANED-TEXT STEP 3 — Lowercase Latin characters ----
    # Chinese chars are case-invariant. Lowercase normalizes 'iPhone' /
    # 'iphone' / 'IPHONE' so lexical features treat them as one token.
    text = text.lower()

    # ---- CLEANED-TEXT STEPS 4 + 5 — Traditional → Simplified + variant normalization ----
    # OpenCC's t2s config uses the STCharacters and STPhrases mappings to:
    #   (a) convert traditional → simplified chars (戶 → 户, 學 → 学)
    #   (b) handle many within-simplified variant characters (異體字)
    # For more aggressive within-simplified variant normalization a separate
    # library (zhconv) or custom mapping would be needed; t2s is sufficient
    # for our corpus.
    text = opencc.convert(text)

    # ---- CLEANED-TEXT STEP 6 — Punctuation restoration via ct-punc ----
    # ct-punc (CT-Transformer punctuation model from FunASR / Alibaba DAMO)
    # adds Chinese sentence-boundary punctuation (。，？！；) at predicted
    # positions. Skips empty input — model errors on empty.
    if text:
        result = punc_model.generate(input=text)
        if result and isinstance(result, list) and len(result) > 0:
            # ct-punc returns [{'key': '...', 'text': '...'}]
            text = result[0].get("text", text)

    # ---- CLEANED-TEXT STEP 7 — Punctuation symbol unification (ASCII → CJK) ----
    # ct-punc usually emits CJK punctuation but mixes in ASCII around
    # Latin code-switched content. Translate ASCII variants to their CJK
    # equivalents so downstream segmentation regexes see one consistent form.
    text = text.translate(PUNCT_TRANSLATION_TABLE)

    # ---- CLEANED-TEXT STEP 8 — Punctuation cleanup (two sub-operations) ----
    # Step 8a: drop sentence-internal punctuation immediately following a
    # sentence-ender (with optional whitespace between). Catches the
    # "。 ，..." pattern that ct-punc occasionally emits at confident
    # boundaries, which would otherwise produce orphan sentences starting
    # with "，" after sentence segmentation.
    text = SENTENCE_END_THEN_INTERNAL_PATTERN.sub(r"\1", text)
    # Step 8b: collapse runs of identical punctuation ("。。。" -> "。",
    # "，，" -> "，"). Runs after 8a so any new repeats created by 8a
    # collapse correctly.
    text = REPEATED_PUNCT_PATTERN.sub(r"\1", text)

    return text


# ===========================================================================
# Segmentation layer (downstream of cleaned-text)
# ===========================================================================

def segment_sentences(text: str) -> list[str]:
    """
    SEGMENTATION STEP 9 — Sentence segmentation.

    Split cleaned text on CJK sentence-boundary punctuation (。！？；…)
    while keeping the punctuation attached to the preceding sentence
    (lookbehind split). Empty/whitespace-only fragments are dropped.
    """
    parts = SENTENCE_SPLIT_PATTERN.split(text)
    return [s.strip() for s in parts if s.strip()]


def segment_words(
    sentences: list[str],
    segmenter: pkuseg.pkuseg,
) -> tuple[list[str], list[list[str]]]:
    """
    SEGMENTATION STEP 10 — Word segmentation via spacy_pkuseg.

    Word-segments each sentence with the pkuseg CRF model. Returns both:
      - flat: a single flat list of all word tokens across the transcript
      - by_sentence: a list of per-sentence token lists (preserves sentence
        structure for sentence-level features)
    """
    by_sentence = [segmenter.cut(s) for s in sentences]
    flat = [tok for sent in by_sentence for tok in sent]
    return flat, by_sentence


# ===========================================================================
# Main
# ===========================================================================

def parse_file_filter(spec: str) -> set[tuple[str, int]]:
    """Parse comma-separated SUBJ/FN spec into a set of (subject_id, file_num)."""
    out: set[tuple[str, int]] = set()
    for entry in spec.split(","):
        entry = entry.strip()
        if not entry:
            continue
        sid, fn = entry.split("/")
        out.add((sid.strip(), int(fn)))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Preprocess Whisper Mandarin transcripts into a feature-ready Parquet."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PARQUET_DEFAULT,
        help=f"Output Parquet path (default: {OUTPUT_PARQUET_DEFAULT}).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N transcripts after filtering. Useful for smoke tests.",
    )
    parser.add_argument(
        "--files",
        type=str,
        default=None,
        help="Comma-separated list of SUBJ/FN to process exclusively, e.g. '02010001/19,02010002/29'.",
    )
    parser.add_argument(
        "--opencc-config",
        type=str,
        default="t2s",
        help="OpenCC config name for trad→simp + variant normalization (default: t2s).",
    )
    parser.add_argument(
        "--punc-model-revision",
        type=str,
        default="v2.0.4",
        help="FunASR ct-punc model revision (default: v2.0.4).",
    )
    parser.add_argument(
        "--print-sample",
        action="store_true",
        help="Print a per-transcript before/after sample to stdout (smoke-test friendly).",
    )
    args = parser.parse_args()

    file_filter = parse_file_filter(args.files) if args.files else None

    # ----------------------------------------------------------------------
    # Pipeline step 1 — Load exclusions
    # ----------------------------------------------------------------------
    excludes = load_exclusions(EXCLUDES_CSV)
    print(f"Loaded {len(excludes)} excluded (subject, file) pairs from {EXCLUDES_CSV.name}")

    # ----------------------------------------------------------------------
    # Pipeline step 2 — Discover latest-version transcripts
    # ----------------------------------------------------------------------
    transcripts = discover_latest_transcripts(AUDIO_DIR, excludes, file_filter)
    if args.limit is not None:
        transcripts = transcripts[: args.limit]
    n_v2plus = sum(1 for *_, v in transcripts if v > 1)
    print(f"Will process {len(transcripts)} transcripts ({n_v2plus} from re-extracted v2+ versions)")
    if not transcripts:
        print("ERROR: no transcripts to process", file=sys.stderr)
        return 1

    # ----------------------------------------------------------------------
    # Load models (one-time setup)
    # ----------------------------------------------------------------------
    print(f"Loading OpenCC config '{args.opencc_config}'...")
    opencc = OpenCC(args.opencc_config)

    print(f"Loading ct-punc punctuation-restoration model (rev {args.punc_model_revision})...")
    print("  (first run downloads ~1.1 GB to ~/.cache/modelscope/)")
    punc_model = AutoModel(model="ct-punc", model_revision=args.punc_model_revision)

    print("Loading spacy_pkuseg word segmenter (default model)...")
    segmenter = pkuseg.pkuseg()
    print("All models loaded.")

    # ----------------------------------------------------------------------
    # Process each transcript through cleaned-text + segmentation layers
    # ----------------------------------------------------------------------
    rows: list[dict] = []
    failures: list[tuple[str, int, str]] = []

    for sid, fn, tpath, ver in tqdm(transcripts, desc="Preprocessing", unit="file"):
        try:
            raw = tpath.read_text(encoding="utf-8").strip()

            # Cleaned-text layer (8 steps inside clean_text)
            cleaned = clean_text(raw, opencc=opencc, punc_model=punc_model)

            # Segmentation layer
            sentences = segment_sentences(cleaned)
            tokens_flat, tokens_by_sent = segment_words(sentences, segmenter)

            row = {
                "subject_id": sid,
                "file_num": fn,
                "transcript_version": ver,
                "source_path": str(tpath),
                "raw_text": raw,
                "cleaned_text": cleaned,
                "sentences": sentences,
                "tokens": tokens_flat,
                "tokens_by_sentence": tokens_by_sent,
                "n_chars": len(cleaned),
                "n_sentences": len(sentences),
                "n_tokens": len(tokens_flat),
            }
            rows.append(row)

            if args.print_sample:
                print(f"\n=== {sid}/{fn:02d} (v{ver}) ===")
                print(f"  raw     [{len(raw):4d} chars]: {raw[:120]}{'…' if len(raw) > 120 else ''}")
                print(f"  cleaned [{len(cleaned):4d} chars]: {cleaned[:120]}{'…' if len(cleaned) > 120 else ''}")
                print(f"  sentences [{len(sentences)}]: {sentences[:3]}{'…' if len(sentences) > 3 else ''}")
                print(f"  tokens [{len(tokens_flat)}]: {tokens_flat[:15]}{'…' if len(tokens_flat) > 15 else ''}")

        except Exception as e:
            failures.append((sid, fn, str(e)))
            tqdm.write(f"FAIL: {sid}/{fn:02d} -- {e}")

    # ----------------------------------------------------------------------
    # Write Parquet
    # ----------------------------------------------------------------------
    if not rows:
        print("ERROR: no transcripts successfully processed", file=sys.stderr)
        return 1

    df = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output, index=False)
    print(f"\nWrote {len(df)} rows -> {args.output}")
    print(f"  columns: {list(df.columns)}")
    print(f"  failures: {len(failures)}")
    for sid, fn, err in failures:
        print(f"    {sid}/{fn:02d}  {err}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
