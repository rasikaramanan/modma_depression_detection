#!/usr/bin/env python3
"""
extract_text_features.py
========================
Extract Pipeline 2 (linguistic) features from preprocessed Whisper transcripts.

Reads the Parquet produced by `preprocess_transcripts.py` and adds a fixed set
of feature columns per transcript. Designed for downstream subject-level
aggregation (mean across each subject's available files) before regression /
classification with n=52 subjects.

Pipeline (per `notes/Feature_Pipeline_Explainer.md`, Pipeline 2):

  Setup:
    1. Load preprocessed Parquet (input from preprocess_transcripts.py)
    2. Bind lexical word lists (module-level constants, lit-grounded)
    3. Load DUTIR Chinese Emotion Vocabulary Ontology (大连理工大学情感词汇本体)

  Per-transcript feature extraction:

    Family A -- Lexical features (5):
      A1. lex_first_person_sg_rate -- 我/我的/俺 rate over word tokens
      A2. lex_first_person_pl_rate -- 我们/咱们/咱 rate (control feature)
      A3. lex_negation_rate         -- 不/没/没有/别/无/未/否/非/莫 rate
      A4. lex_ttr                   -- type-token ratio over word tokens
      A5. lex_mattr50               -- moving-average TTR, window=50

    Family B -- Simple syntactic features (3):
      B1. syn_mean_tokens_per_sent -- mean word tokens per sentence
      B2. syn_sd_tokens_per_sent   -- population SD of word tokens per sentence
      B3. syn_punct_density        -- punct tokens / all tokens

    Family C -- DUTIR sentiment features (8):
      C1. sent_le_rate    -- 乐 (joy) rate (DUTIR macro class from PA + PE)
      C2. sent_hao_rate   -- 好 (like) rate (PD + PH + PG + PB + PK)
      C3. sent_nu_rate    -- 怒 (anger) rate (NA)
      C4. sent_ai_rate    -- 哀 (sadness) rate (NB + NJ + NH + PF) -- key marker
      C5. sent_ju_rate    -- 惧 (fear) rate (NI + NC + NG)
      C6. sent_e_rate     -- 恶 (disgust) rate (NE + ND + NN + NK + NL)
      C7. sent_jing_rate  -- 惊 (surprise) rate (PC)
      C8. sent_net_polarity -- (positive_count - negative_count) / n_word_tokens
                                using DUTIR 极性 column (1=pos, 2=neg).

  Plus one diagnostic column:
      lex_n_word_tokens -- count of non-punct tokens (denominator for rates).

Output:
    Single Parquet at data/features/transcripts_features.parquet containing
    the input columns plus 17 new columns (1 diagnostic + 16 features).

Usage:
    # Full corpus
    python scripts/extract_text_features.py

    # Smoke on the smoke Parquet from preprocess_transcripts.py
    python scripts/extract_text_features.py \
        --input data/features/transcripts_preprocessed_smoke.parquet \
        --output data/features/transcripts_features_smoke.parquet \
        --print-sample
"""

from __future__ import annotations

import argparse
import statistics
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import pandas as pd
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
INPUT_PARQUET_DEFAULT = REPO_ROOT / "data" / "features" / "transcripts_preprocessed.parquet"
OUTPUT_PARQUET_DEFAULT = REPO_ROOT / "data" / "features" / "transcripts_features.parquet"
DUTIR_CSV_DEFAULT = REPO_ROOT / "data" / "external" / "dutir" / "DUTIR_emotion_ontology.csv"


# ---------------------------------------------------------------------------
# SETUP STEP 2 -- Lexical word lists (module-level, lit-grounded)
# ---------------------------------------------------------------------------

# A1 -- First-person singular pronouns. Closed grammatical class in Mandarin.
# Source: Li & Thompson 1981 *Mandarin Chinese: A Functional Reference Grammar*,
# cross-checked against the SC-LIWC 第一人称单数 (i) category (Huang et al. 2012).
# Excludes 我们 (we, plural) -- patterns oppositely to singular in depression
# text per Eichstaedt et al. 2018 PNAS.
FIRST_PERSON_SG = frozenset({"我", "我的", "俺"})

# A2 -- First-person plural pronouns. Tracked separately as a control feature.
# Same sources as A1.
FIRST_PERSON_PL = frozenset({"我们", "咱们", "咱"})

# A3 -- Negation. Closed grammatical class. Source: Li & Thompson 1981; cross-
# validated against the SC-LIWC 否定 (negate) category. The first 4 entries
# cover ~95% of conversational usage; 莫 is mostly classical but kept for
# coverage of fixed expressions.
NEGATION = frozenset({"不", "没", "没有", "别", "无", "未", "否", "非", "莫"})


# ---------------------------------------------------------------------------
# DUTIR fine-grained -> macro emotion mapping (Xu et al. 2008)
# ---------------------------------------------------------------------------
# DUTIR's 21 fine-grained emotion codes roll up to 7 macro categories, which
# is what the depression-NLP literature usually uses. Macro names are pinyin
# to keep ASCII-safe column names downstream.
EMOTION_CLASS_TO_MACRO: dict[str, str] = {
    "PA": "le",   # 快乐 happiness
    "PE": "le",   # 安心 calm/peace
    "PD": "hao",  # 尊敬 respect
    "PH": "hao",  # 赞扬 praise
    "PG": "hao",  # 相信 trust
    "PB": "hao",  # 喜爱 love
    "PK": "hao",  # 祝愿 wish/blessing
    "NA": "nu",   # 愤怒 anger
    "NB": "ai",   # 悲伤 sadness  -- key depression marker
    "NJ": "ai",   # 失望 disappointment
    "NH": "ai",   # 内疚 guilt
    "PF": "ai",   # 思 longing/missing
    "NI": "ju",   # 慌 panic
    "NC": "ju",   # 恐惧 fear
    "NG": "ju",   # 羞 shame
    "NE": "e",    # 烦闷 annoyance/dejection
    "ND": "e",    # 憎恶 disgust
    "NN": "e",    # 贬责 derogation
    "NK": "e",    # 妒忌 jealousy
    "NL": "e",    # 怀疑 doubt
    "PC": "jing", # 惊奇 surprise
}
MACRO_EMOTIONS: tuple[str, ...] = ("le", "hao", "nu", "ai", "ju", "e", "jing")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def is_punct_token(tok: str) -> bool:
    """
    Return True if `tok` is composed entirely of Unicode punctuation chars.
    Used to filter pkuseg-emitted punctuation tokens (。 ， ！ ？ ; etc.) out
    of word-level denominators per LIWC convention.
    """
    if not tok:
        return False
    return all(unicodedata.category(c).startswith("P") for c in tok)


def _coerce_to_list(seq) -> list:
    """Parquet list columns may come back as numpy arrays; force python list."""
    return list(seq) if seq is not None else []


# ---------------------------------------------------------------------------
# SETUP STEP 3 -- DUTIR loader
# ---------------------------------------------------------------------------

def load_dutir(csv_path: Path) -> tuple[dict[str, str], dict[str, int]]:
    """
    SETUP STEP 3 -- Load DUTIR emotion lexicon.

    Returns:
      word_to_macro:    dict[word -> macro emotion code (le/hao/nu/ai/ju/e/jing)]
      word_to_polarity: dict[word -> int polarity (0=neutral, 1=positive, 2=negative)]

    Notes:
      - The CSV from yizhanmiao/DLUT-Emotionontology has leading-space column
        names; we strip them.
      - DUTIR contains polysemous entries (one word -> multiple senses,
        indexed by 词义序号). We keep only the primary sense (smallest
        词义序号) per word, which is the standard treatment in DUTIR papers.
      - Polarity values 3 and 7 occur in <0.3% of rows (likely data-entry
        errors); we drop them.
    """
    df = pd.read_csv(csv_path, on_bad_lines="skip", engine="python")
    df.columns = [c.strip() for c in df.columns]

    # Keep first sense per word (primary).
    df = df.sort_values(["词语", "词义序号"]).drop_duplicates(subset="词语", keep="first")

    word_to_macro: dict[str, str] = {}
    word_to_polarity: dict[str, int] = {}
    unknown_codes: Counter = Counter()
    for _, row in df.iterrows():
        word = str(row["词语"]).strip()
        if not word:
            continue
        emo_code = str(row["情感分类"]).strip()
        macro = EMOTION_CLASS_TO_MACRO.get(emo_code)
        if macro is None:
            unknown_codes[emo_code] += 1
            continue
        word_to_macro[word] = macro
        try:
            pol = int(row["极性"])
        except (TypeError, ValueError):
            continue
        if pol in (0, 1, 2):
            word_to_polarity[word] = pol

    if unknown_codes:
        print(f"  WARN: DUTIR rows with unknown emotion codes: {dict(unknown_codes)}", file=sys.stderr)
    return word_to_macro, word_to_polarity


# ===========================================================================
# FEATURE FAMILY A -- Lexical features
# ===========================================================================

def _mattr(tokens: list[str], window: int = 50) -> float:
    """
    Moving-average TTR (Covington & McFall 2010). Computes TTR over each
    `window`-token sliding window and averages. Length-invariant alternative
    to raw TTR. Returns NaN if len(tokens) < window.
    """
    n = len(tokens)
    if n < window:
        return float("nan")
    ratios = []
    for i in range(n - window + 1):
        chunk = tokens[i:i + window]
        ratios.append(len(set(chunk)) / window)
    return float(sum(ratios) / len(ratios))


def lexical_features(word_tokens: list[str]) -> dict:
    """
    FEATURE FAMILY A -- Lexical (5 features).
    Operates on word tokens with punctuation already filtered out.
    """
    n = len(word_tokens)
    if n == 0:
        return {
            "lex_first_person_sg_rate": float("nan"),
            "lex_first_person_pl_rate": float("nan"),
            "lex_negation_rate":         float("nan"),
            "lex_ttr":                   float("nan"),
            "lex_mattr50":               float("nan"),
        }

    # A1, A2, A3 -- closed-class word-list rates.
    sg_count  = sum(1 for t in word_tokens if t in FIRST_PERSON_SG)
    pl_count  = sum(1 for t in word_tokens if t in FIRST_PERSON_PL)
    neg_count = sum(1 for t in word_tokens if t in NEGATION)

    # A4 -- type-token ratio (length-dependent; reported alongside MATTR).
    ttr = len(set(word_tokens)) / n

    # A5 -- moving-average TTR over a 50-token window. Returns NaN for short
    # transcripts so downstream code can decide how to handle.
    mattr = _mattr(word_tokens, window=50)

    return {
        "lex_first_person_sg_rate": sg_count / n,
        "lex_first_person_pl_rate": pl_count / n,
        "lex_negation_rate":        neg_count / n,
        "lex_ttr":                  ttr,
        "lex_mattr50":              mattr,
    }


# ===========================================================================
# FEATURE FAMILY B -- Simple syntactic features
# ===========================================================================

def syntactic_features(
    word_tokens_by_sent: list[list[str]],
    n_punct_tokens: int,
    n_total_tokens: int,
) -> dict:
    """
    FEATURE FAMILY B -- Simple syntactic features (3).

    B1. mean word tokens per sentence
    B2. population SD of word tokens per sentence (NaN if <2 sentences)
    B3. punctuation density = n_punct_tokens / n_total_tokens
    """
    if not word_tokens_by_sent or n_total_tokens == 0:
        return {
            "syn_mean_tokens_per_sent": float("nan"),
            "syn_sd_tokens_per_sent":   float("nan"),
            "syn_punct_density":        float("nan"),
        }

    sent_lens = [len(s) for s in word_tokens_by_sent]
    mean_len = float(sum(sent_lens) / len(sent_lens))
    sd_len = float(statistics.pstdev(sent_lens)) if len(sent_lens) >= 2 else float("nan")

    return {
        "syn_mean_tokens_per_sent": mean_len,
        "syn_sd_tokens_per_sent":   sd_len,
        "syn_punct_density":        n_punct_tokens / n_total_tokens,
    }


# ===========================================================================
# FEATURE FAMILY C -- DUTIR sentiment features
# ===========================================================================

def sentiment_features(
    word_tokens: list[str],
    word_to_macro: dict[str, str],
    word_to_polarity: dict[str, int],
) -> dict:
    """
    FEATURE FAMILY C -- DUTIR sentiment features (8).

    C1-C7. One rate per macro emotion class (le/hao/nu/ai/ju/e/jing).
    C8.    Net polarity = (positive - negative) / n_word_tokens.

    All rates are over word tokens (punct already removed).
    """
    n = len(word_tokens)
    if n == 0:
        out = {f"sent_{m}_rate": float("nan") for m in MACRO_EMOTIONS}
        out["sent_net_polarity"] = float("nan")
        return out

    macro_counts: Counter = Counter()
    pos_count = 0
    neg_count = 0
    for tok in word_tokens:
        m = word_to_macro.get(tok)
        if m is not None:
            macro_counts[m] += 1
        p = word_to_polarity.get(tok)
        if p == 1:
            pos_count += 1
        elif p == 2:
            neg_count += 1

    out: dict = {f"sent_{m}_rate": macro_counts.get(m, 0) / n for m in MACRO_EMOTIONS}
    out["sent_net_polarity"] = (pos_count - neg_count) / n
    return out


# ===========================================================================
# Per-row orchestrator
# ===========================================================================

# Stable column order for output / NaN fallback rows.
FEATURE_COLUMNS: list[str] = [
    "lex_n_word_tokens",
    "lex_first_person_sg_rate",
    "lex_first_person_pl_rate",
    "lex_negation_rate",
    "lex_ttr",
    "lex_mattr50",
    "syn_mean_tokens_per_sent",
    "syn_sd_tokens_per_sent",
    "syn_punct_density",
] + [f"sent_{m}_rate" for m in MACRO_EMOTIONS] + [
    "sent_net_polarity",
]


def _nan_features() -> dict:
    return {col: float("nan") for col in FEATURE_COLUMNS}


def extract_features_for_row(
    row: pd.Series,
    word_to_macro: dict[str, str],
    word_to_polarity: dict[str, int],
) -> dict:
    """Apply all 3 feature families to a single transcript row."""
    tokens = _coerce_to_list(row["tokens"])
    tokens_by_sent_raw = _coerce_to_list(row["tokens_by_sentence"])
    tokens_by_sent = [_coerce_to_list(s) for s in tokens_by_sent_raw]

    # Filter punctuation tokens for word-level features.
    word_tokens = [t for t in tokens if not is_punct_token(t)]
    word_tokens_by_sent = [
        [t for t in s if not is_punct_token(t)] for s in tokens_by_sent
    ]
    n_punct = len(tokens) - len(word_tokens)

    feats: dict = {"lex_n_word_tokens": len(word_tokens)}
    feats.update(lexical_features(word_tokens))
    feats.update(syntactic_features(word_tokens_by_sent, n_punct, len(tokens)))
    feats.update(sentiment_features(word_tokens, word_to_macro, word_to_polarity))
    return feats


# ===========================================================================
# Main
# ===========================================================================

def _fmt_float(x: float, prec: int = 3) -> str:
    """Pretty-print a float that might be NaN."""
    if x is None or (isinstance(x, float) and (x != x)):  # NaN check
        return "  nan"
    return f"{x:.{prec}f}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract Pipeline-2 (linguistic) features from preprocessed transcripts."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=INPUT_PARQUET_DEFAULT,
        help=f"Input Parquet (default: {INPUT_PARQUET_DEFAULT}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PARQUET_DEFAULT,
        help=f"Output Parquet (default: {OUTPUT_PARQUET_DEFAULT}).",
    )
    parser.add_argument(
        "--dutir-csv",
        type=Path,
        default=DUTIR_CSV_DEFAULT,
        help=f"DUTIR emotion-ontology CSV (default: {DUTIR_CSV_DEFAULT}).",
    )
    parser.add_argument(
        "--print-sample",
        action="store_true",
        help="Print a one-line per-transcript feature snapshot to stdout.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N rows of the input Parquet. For smoke testing.",
    )
    args = parser.parse_args()

    # ----------------------------------------------------------------------
    # Setup step 1 -- Load preprocessed Parquet
    # ----------------------------------------------------------------------
    print(f"Loading preprocessed transcripts from {args.input}...")
    if not args.input.exists():
        print(f"ERROR: input Parquet not found: {args.input}", file=sys.stderr)
        return 1
    df_in = pd.read_parquet(args.input)
    if args.limit is not None:
        df_in = df_in.head(args.limit).reset_index(drop=True)
    print(f"  {len(df_in)} rows; columns: {list(df_in.columns)}")

    required = {"subject_id", "file_num", "transcript_version", "tokens", "tokens_by_sentence"}
    missing = required - set(df_in.columns)
    if missing:
        print(f"ERROR: input Parquet missing required columns: {sorted(missing)}", file=sys.stderr)
        return 1

    # ----------------------------------------------------------------------
    # Setup step 2 -- Lexical word lists are module constants; just report.
    # ----------------------------------------------------------------------
    print(
        f"Lexical lists: "
        f"{len(FIRST_PERSON_SG)} 1st-sg, "
        f"{len(FIRST_PERSON_PL)} 1st-pl, "
        f"{len(NEGATION)} negation"
    )

    # ----------------------------------------------------------------------
    # Setup step 3 -- Load DUTIR
    # ----------------------------------------------------------------------
    print(f"Loading DUTIR emotion ontology from {args.dutir_csv}...")
    if not args.dutir_csv.exists():
        print(f"ERROR: DUTIR CSV not found: {args.dutir_csv}", file=sys.stderr)
        return 1
    word_to_macro, word_to_polarity = load_dutir(args.dutir_csv)
    macro_dist = Counter(word_to_macro.values())
    print(f"  {len(word_to_macro)} words mapped to 7 macro emotions")
    print(f"  per-macro counts: " + ", ".join(f"{m}={macro_dist[m]}" for m in MACRO_EMOTIONS))
    print(f"  {len(word_to_polarity)} words with polarity (0=neu, 1=pos, 2=neg)")

    # ----------------------------------------------------------------------
    # Per-row feature extraction
    # ----------------------------------------------------------------------
    feat_rows: list[dict] = []
    failures: list[tuple[str, int, str]] = []

    for _, row in tqdm(df_in.iterrows(), total=len(df_in), desc="Extracting", unit="file"):
        sid = row.get("subject_id")
        fn = row.get("file_num")
        try:
            feats = extract_features_for_row(row, word_to_macro, word_to_polarity)
            feat_rows.append(feats)

            if args.print_sample:
                ver = row.get("transcript_version", "?")
                print(f"\n=== {sid}/{int(fn):02d} (v{ver}) ===")
                print(
                    f"  n_word_tokens={feats['lex_n_word_tokens']:4d}  "
                    f"sg={_fmt_float(feats['lex_first_person_sg_rate'])}  "
                    f"pl={_fmt_float(feats['lex_first_person_pl_rate'])}  "
                    f"neg={_fmt_float(feats['lex_negation_rate'])}  "
                    f"ttr={_fmt_float(feats['lex_ttr'])}  "
                    f"mattr={_fmt_float(feats['lex_mattr50'])}"
                )
                print(
                    f"  syn: mean_sent_len={_fmt_float(feats['syn_mean_tokens_per_sent'], 2)}  "
                    f"sd_sent_len={_fmt_float(feats['syn_sd_tokens_per_sent'], 2)}  "
                    f"punct_dens={_fmt_float(feats['syn_punct_density'])}"
                )
                emo = "  ".join(
                    f"{m}={_fmt_float(feats[f'sent_{m}_rate'])}" for m in MACRO_EMOTIONS
                )
                print(f"  emo: {emo}")
                print(f"  net_polarity={_fmt_float(feats['sent_net_polarity'])}")

        except Exception as e:
            failures.append((str(sid), int(fn) if pd.notna(fn) else -1, str(e)))
            tqdm.write(f"FAIL: {sid}/{fn}  --  {e}")
            feat_rows.append(_nan_features())

    # ----------------------------------------------------------------------
    # Assemble + write output
    # ----------------------------------------------------------------------
    if not feat_rows:
        print("ERROR: no rows processed", file=sys.stderr)
        return 1

    df_feats = pd.DataFrame(feat_rows, columns=FEATURE_COLUMNS)
    df_out = pd.concat(
        [df_in.reset_index(drop=True), df_feats.reset_index(drop=True)],
        axis=1,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_parquet(args.output, index=False)

    print(f"\nWrote {len(df_out)} rows -> {args.output}")
    print(f"  feature columns added ({len(FEATURE_COLUMNS)}): {FEATURE_COLUMNS}")
    print(f"  failures: {len(failures)}")
    for sid, fn, err in failures:
        print(f"    {sid}/{fn:02d}  {err}")

    # Quick sanity print of feature ranges across the run.
    print("\nFeature summary across processed rows:")
    summary_cols = [c for c in FEATURE_COLUMNS if c != "lex_n_word_tokens"]
    desc = df_feats[summary_cols].describe(percentiles=[0.5]).T[["mean", "std", "min", "50%", "max"]]
    print(desc.to_string())

    return 0


if __name__ == "__main__":
    sys.exit(main())
