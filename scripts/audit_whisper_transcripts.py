#!/usr/bin/env python3
"""
audit_whisper_transcripts.py
============================
Re-audit current Whisper Mandarin transcripts on disk for known
faster-whisper / Whisper-large-v3 failure modes specific to Mandarin
transcription.

Why a fresh audit:
The pre-existing audit at archive/1_fix_whisper_transcript_issues.csv was
generated against an older transcription run (filename pattern
*_transcript_whisper_zh.txt). Spot checks confirm the current run
(*_transcript_faster_whisper_zh.txt) produced different text for the same
wav files, so the old flags do not apply 1:1.

Output:
A CSV at <repo_root>/whisper_transcript_audit.csv with one row per
transcript file. Columns:
  - subject_id      : participant ID (parent dir name)
  - file_num        : 1..29 (the file index within the subject's task battery)
  - passed_<test>   : boolean — TRUE = transcript passed that audit test
  - passed_all      : convenience boolean, AND of all per-test booleans
  - transcript_text : raw transcript content for inspection

Each `passed_<test>` column corresponds to a documented failure mode for
faster-whisper on Mandarin audio. See the AUDIT TESTS section below for
the test-by-test motivation; matching writeup is provided alongside the
script invocation that produced this audit.

Usage:
    python scripts/audit_whisper_transcripts.py
"""

from __future__ import annotations

import csv
import re
import sys
import zlib
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
AUDIO_DIR = REPO_ROOT / "CSCI567 Project" / "modma_data" / "audio_lanzhou_2015"
OUTPUT_CSV = REPO_ROOT / "whisper_transcript_audit.csv"

# Discovery glob: matches both unversioned originals
# (`{stem}_transcript_faster_whisper_zh.txt`) and re-extracted versions
# (`{stem}_transcript_faster_whisper_zh_v2.txt`, `_v3.txt`, etc.). The
# version-parsing regex below distinguishes them; for each (subject,
# file_num) the audit reads ONLY the highest-version file present on disk.
TRANSCRIPT_GLOB = "*_transcript_faster_whisper_zh*.txt"
TRANSCRIPT_VERSION_RE = re.compile(
    r"^(?P<stem>\d{2})_transcript_faster_whisper_zh(?:_v(?P<version>\d+))?\.txt$"
)

# ---------------------------------------------------------------------------
# Configuration: thresholds and known-bad phrases
# ---------------------------------------------------------------------------

# Per-task minimum character length for `passed_min_length`.
# Thresholds calibrated against the observed length distribution on the
# 2026-05-08 audit run (see writeup):
#   - File 19 = passage reading (fixed long text). Min non-zero observed = 138
#     chars; threshold 100 leaves a 38-char buffer below the floor.
#   - Files 20-25 = 10-word reading lists (positive/neutral/negative valence).
#     Min non-zero observed = 14 chars; threshold 10 leaves a 4-char buffer.
#   - Files 26-28 = picture description (CFAPS positive/neutral/negative).
#   - File 29 = picture description (TAT clinical "crying woman").
#   - Files 1-18 = interview (spontaneous). Real subjects legitimately give
#     very short answers ("嗯", "没有", "不知道"), so threshold stays loose.
TASK_MIN_LENGTH: dict[int, int] = {n: 2 for n in range(1, 30)}
TASK_MIN_LENGTH[19] = 100
for n in (20, 21, 22, 23, 24, 25):
    TASK_MIN_LENGTH[n] = 10
for n in (26, 27, 28, 29):
    TASK_MIN_LENGTH[n] = 5

# Compression-ratio threshold (Whisper's own internal heuristic for repetition).
# Reference: openai/whisper transcribe.py -- compression_ratio_threshold = 2.4.
COMPRESSION_RATIO_THRESHOLD = 2.4
COMPRESSION_RATIO_MIN_CHARS = 20  # below this, gzip ratio is unstable

# Repetition-loop n-gram threshold.
# Any character n-gram of length [N_GRAM_MIN, N_GRAM_MAX] appearing
# >= MAX_NGRAM_REPEATS times in a transcript flags a loop.
#
# N_GRAM_MIN was raised from 5 to 6 on 2026-05-08 after observing that
# 5-grams in Mandarin map to ~2-3 lexical units and trigger false positives
# on natural parallel constructions ("X的时候 / X的时候 / X的时候"), topic-echo
# backchannels in interview tasks, and referential repetition in picture
# descriptions ("这个男主人 ... 这个男主人 ..."). Real Whisper loops
# repeat longer verbatim sequences (the model regenerates from prior context),
# so 6-grams retain detection of true loops while excluding the
# Mandarin-specific noise floor at n=5.
N_GRAM_MIN = 6
N_GRAM_MAX = 10
MAX_NGRAM_REPEATS = 3

# Language-content thresholds (over non-whitespace characters).
MIN_CJK_RATIO = 0.7
MAX_LATIN_RATIO = 0.10  # tightened from 0.20 on 2026-05-08; the 0.10-0.20 band on the corpus consisted entirely of real Whisper substitution failures (Jazz->寨子) with no observed false positives.

# Excessive-digit-content threshold. Set ABOVE the highest observed
# legitimate digit ratio (0.115 on 02010013/26's '种了500万') with ~1 SD
# headroom, plus a min-length floor so short transcripts with one or two
# digits ('2号目前沟通2号') don't false-positive on the ratio arithmetic.
# Defensive only on current data (0 hits); catches future hallucinations
# like phone numbers / time codes / numeric loops.
MAX_DIGIT_RATIO_NWS = 0.20
DIGIT_RATIO_MIN_CHARS = 20

# Repeated single-character run threshold (e.g. "好好好好好好" -> fail).
MAX_SAME_CHAR_RUN = 5

# Minimum transcript length below which the cross-subject-uniqueness check
# is skipped. Short common Mandarin answers ("沒有", "好的", "知道") legitimately
# repeat across subjects on different prompts and are not a quality concern;
# the uniqueness check is intended for catching long hallucinated boilerplate
# (Amara credits, TV outros, etc.) that Whisper produces verbatim across many
# subjects. Documented boilerplate hallucinations are typically >= 10 chars.
MIN_LENGTH_FOR_UNIQUENESS_CHECK = 10

# Sound-effect / non-speech labels: bracketed or parenthesized Latin-only
# tokens like "[Music]", "(applause)", "[laughs]", "(typing)". Whisper's
# suppress-tokens mechanism can leak these when training-data subtitles
# labeled background sounds; they should never appear in transcribed
# Mandarin clinical interview speech. The regex matches `[X]` or `(X)`
# where X is 1-30 chars of Latin letters / spaces / hyphens (catches
# multi-word labels like "[door closing]").
_SOUND_EFFECT_PATTERN = re.compile(
    r"\[\s*[A-Za-z][A-Za-z \-]{0,29}\s*\]"
    r"|\(\s*[A-Za-z][A-Za-z \-]{0,29}\s*\)"
)

# URL / social-media artifacts: training-data leakage from web/subtitle
# datasets. Should never appear in clinical interview audio.
_URL_SOCIAL_PATTERN = re.compile(
    r"https?://"
    r"|www\.[A-Za-z]"
    r"|\b[\w-]+\.(?:com|cn|net|org|gov|edu|io|tv|cc|hk)\b"
    r"|(?<![A-Za-z0-9_])[@#][A-Za-z][A-Za-z0-9_]{2,}",
    re.IGNORECASE,
)

# Invisible / non-printable characters: ASCII control codes (except \t and
# \n which are valid whitespace), zero-width chars, BOM markers, bidi
# formatting overrides, and word-joiner / invisible operators. These are
# training-data artifacts that silently corrupt downstream tokenization
# and substring matching.
_CONTROL_AND_ZW_PATTERN = re.compile(
    "[\x00-\x08\x0B-\x1F\x7F]"   # ASCII control except \t (\x09) and \n (\x0A)
    "|[​-‏]"            # zero-width space/non-joiner/joiner + LRM/RLM
    "|﻿"                      # BOM / zero-width no-break space
    "|[‪-‮]"            # bidi formatting overrides
    "|[⁠-⁯]"            # word joiner + invisible operators
)

# Known hallucination substrings produced by Whisper when given silent /
# noisy / hard-to-transcribe input. Sourced from openai/whisper community
# discussions (see writeup that accompanies this script).
KNOWN_HALLUCINATION_PHRASES_ZH: list[str] = [
    # Amara.org subtitle attribution -- a documented training-data bias
    "字幕由Amara.org社区提供",
    "字幕由 Amara.org 社群提供",
    "由Amara.org社群提供的字幕",
    "由 Amara.org 社群提供的字幕",
    "由Amara.org社区提供",
    "由 Amara.org 社區提供",
    "Amara.org",
    # Volunteer subtitle credits -- training-data bias from YouTube subtitles.
    # The fabricated "translator" name varies across hallucinations (杨茜茜
    # in whisper #1873, 李宗盛 in whisper #2685, etc.), so we match on the
    # phrase prefix instead of specific names. Historically reported named
    # variants include "中文字幕志愿者 杨茜茜", "字幕志愿者 杨茜茜",
    # "字幕志願者 楊茜茜".
    "字幕志愿者",
    "字幕志願者",
    "中文字幕志愿者",
    "中文字幕志願者",
    "中文字幕——YK",
    "字幕by索兰娅",
    "字幕by索蘭婭",
    # Chinese video-outro hallucinations -- training-data bias from videos
    # where these phrases appear over silence at video endings (analogous
    # to the English "Thanks for watching" hallucination already covered in
    # KNOWN_HALLUCINATION_PHRASES_EN below).
    "感谢观看",
    "感謝觀看",
    "谢谢观看",
    "謝謝觀看",
    "感谢您的收看",
    "感謝您的收看",
    "下期再见",
    "下期再見",
    "下期见",
    "下期見",
    # Chinese newspaper / publication boilerplate -- training-data bias
    # from newspaper articles in subtitle/caption datasets. Ming Pao (明報,
    # with Hong Kong / Canada / Toronto editions) is the most-observed
    # variant; see whisper #2685.
    "明报",
    "明報",
    # The "明镜与点点" YouTube subscribe-button hallucination -- one of the
    # most-cited Mandarin Whisper hallucinations.
    "请不吝点赞",
    "請不吝點贊",
    "明镜与点点",
    "明鏡與點點",
    "點贊 訂閱 轉發 打賞",
    "点赞 订阅 转发 打赏",
    "订阅 转发 打赏",
    "訂閱 轉發 打賞",
    # Additional subscribe-button / channel-promotion variants documented
    # in whisper #1873 master list and openai community forum threads.
    "欢迎订阅",
    "歡迎訂閱",
    "请关注",
    "請關注",
    "请订阅",
    "請訂閱",
    # Sound-effect / non-speech tokens. Whisper's internal suppress-tokens
    # mechanism can leak these when training-data subtitles labeled
    # background sounds. None of these should ever appear in transcribed
    # clinical interview audio.
    "[音乐]",
    "[音樂]",
    "[掌声]",
    "[掌聲]",
    "[笑声]",
    "[笑聲]",
    "[噪音]",
    "（音乐）",
    "（音樂）",
    "（掌声）",
    "（掌聲）",
    "（笑声）",
    "（笑聲）",
    "(音乐)",
    "(音樂)",
    "(掌声)",
    "(掌聲)",
    "(笑声)",
    "(笑聲)",
    # Generic Chinese subtitle / fan-translation attributions
    "字幕组",
    "字幕組",
    "字幕製作",
    "字幕制作",
    "本字幕由",
]

KNOWN_HALLUCINATION_PHRASES_EN: list[str] = [
    "thanks for watching",
    "thank you for watching",
    "don't forget to subscribe",
    "like and subscribe",
    "please subscribe",
    "subtitles by",
    "translated by",
    "transcription by castingwords",
    "subscribe to",
    "ming pao",  # newspaper-name hallucination; see whisper #2685
    # Sound-effect / non-speech tokens (English bracketed/parenthesized
    # variants). These are music/applause/laughter labels that leak when
    # Whisper's suppress-tokens mechanism fails on non-speech audio.
    "[music]",
    "[applause]",
    "[laughter]",
    "[silence]",
    "[noise]",
    "[inaudible]",
    "(music)",
    "(applause)",
    "(laughter)",
    # Subtitle-credit boilerplate variants
    "subtitles by the amara.org community",
    "subtitled by",
    "captions by",
    "captioning by",
    # Social-media / URL leakage from training data
    "youtube.com",
    "bilibili.com",
    "youtu.be",
]

# ---------------------------------------------------------------------------
# Character-class helpers
# ---------------------------------------------------------------------------

def cjk_count(s: str) -> int:
    """Count CJK Unified Ideographs (BMP block + Extension A)."""
    return sum(
        1 for ch in s
        if ("一" <= ch <= "鿿") or ("㐀" <= ch <= "䶿")
    )


def latin_count(s: str) -> int:
    """Count Latin-alphabet characters (A-Z, a-z)."""
    return sum(1 for ch in s if ("a" <= ch <= "z") or ("A" <= ch <= "Z"))


def digit_count(s: str) -> int:
    """Count ASCII digit characters (0-9)."""
    return sum(1 for ch in s if "0" <= ch <= "9")


def other_script_count(s: str) -> int:
    """Count characters in non-CJK, non-Latin scripts that should never
    appear in Mandarin transcription. Catches Whisper language-confusion
    leakage from multilingual training data: Hangul (Korean), Hiragana /
    Katakana (Japanese), Cyrillic (Russian / Ukrainian / etc.), Arabic.
    """
    return sum(
        1 for ch in s
        if ("가" <= ch <= "힯")  # Hangul Syllables
        or ("぀" <= ch <= "ゟ")  # Hiragana
        or ("゠" <= ch <= "ヿ")  # Katakana
        or ("Ѐ" <= ch <= "ӿ")  # Cyrillic
        or ("؀" <= ch <= "ۿ")  # Arabic
    )


def control_or_zero_width_count(s: str) -> int:
    """Count invisible / non-printable characters that should not appear
    in a clean transcript: ASCII control chars (excluding \\t and \\n),
    zero-width chars, BOM, and bidi formatting overrides. These are
    training-data artifacts that silently corrupt downstream tokenization
    and string matching."""
    return len(_CONTROL_AND_ZW_PATTERN.findall(s))


def longest_same_char_run(s: str) -> int:
    """Length of the longest run of a single character repeated
    consecutively. Returns 0 for empty string, 1 for any non-empty string
    with no consecutive repeats."""
    if not s:
        return 0
    longest = 1
    current = 1
    for i in range(1, len(s)):
        if s[i] == s[i - 1]:
            current += 1
            if current > longest:
                longest = current
        else:
            current = 1
    return longest


def informative_char_count(s: str) -> int:
    """Count language-bearing characters: CJK ideographs + Latin alphabet.

    Excludes punctuation, digits, and whitespace -- those characters do not
    indicate which language was transcribed, so including them in the
    denominator of a 'predominantly Chinese' check produces false-positive
    flags on word-reading transcripts that Whisper formats as
    comma-separated lists (e.g. '不凡,宝贝,自在,中奖,只好,辉煌,高手,美好,
    优胜,团圆。' -- 20 CJK chars but only 0.667 ratio against
    non-whitespace because of 9 ASCII commas + 1 CJK period)."""
    return cjk_count(s) + latin_count(s)


def non_whitespace_len(s: str) -> int:
    return sum(1 for ch in s if not ch.isspace())


def compression_ratio(s: str) -> float:
    """Whisper's compression_ratio: utf8-bytes / zlib-compressed bytes.

    Higher = more repetitive. Whisper's default rejection threshold is 2.4.
    """
    if not s:
        return 0.0
    b = s.encode("utf-8")
    return len(b) / max(1, len(zlib.compress(b)))


def max_ngram_repetition(s: str) -> int:
    """Highest count of any character n-gram in [N_GRAM_MIN, N_GRAM_MAX]
    over the whitespace-stripped transcript."""
    s = re.sub(r"\s+", "", s)
    if len(s) < N_GRAM_MIN:
        return 0
    max_count = 0
    for n in range(N_GRAM_MIN, N_GRAM_MAX + 1):
        if len(s) < n:
            break
        counts: dict[str, int] = defaultdict(int)
        for i in range(len(s) - n + 1):
            counts[s[i:i + n]] += 1
        if counts:
            cur = max(counts.values())
            if cur > max_count:
                max_count = cur
    return max_count


# ---------------------------------------------------------------------------
# Audit tests (each returns True = passed)
# ---------------------------------------------------------------------------

def test_non_empty(text: str, **_) -> bool:
    return len(text.strip()) > 0


def test_min_length(text: str, file_num: int, **_) -> bool:
    return len(text.strip()) >= TASK_MIN_LENGTH.get(file_num, 2)


def test_no_repetition_loop(text: str, **_) -> bool:
    return max_ngram_repetition(text) < MAX_NGRAM_REPEATS


def test_compression_ratio(text: str, **_) -> bool:
    if non_whitespace_len(text) < COMPRESSION_RATIO_MIN_CHARS:
        return True  # too short for stable compression statistics
    return compression_ratio(text) < COMPRESSION_RATIO_THRESHOLD


def test_no_known_hallucination_phrase(text: str, **_) -> bool:
    for phrase in KNOWN_HALLUCINATION_PHRASES_ZH:
        if phrase in text:
            return False
    lower = text.lower()
    for phrase in KNOWN_HALLUCINATION_PHRASES_EN:
        if phrase in lower:
            return False
    return True


def test_predominantly_chinese(text: str, **_) -> bool:
    """Whether the transcript is predominantly CJK among LANGUAGE-BEARING
    characters (CJK + Latin), ignoring punctuation/digits/whitespace.

    Patch (2026-05-08): denominator switched from non-whitespace-length to
    informative-char-count so that legitimately-Chinese word-reading
    transcripts no longer false-positive due to ASCII commas inserted by
    Whisper between words. Empty or pure-punctuation strings return True
    (test not applicable; emptiness caught by passed_non_empty)."""
    informative = informative_char_count(text)
    if informative == 0:
        return True
    return (cjk_count(text) / informative) >= MIN_CJK_RATIO


def test_no_excessive_latin(text: str, **_) -> bool:
    nws = non_whitespace_len(text)
    if nws == 0:
        return True
    return (latin_count(text) / nws) <= MAX_LATIN_RATIO


def test_no_repeated_char_run(text: str, **_) -> bool:
    pattern = r"(.)\1{" + str(MAX_SAME_CHAR_RUN) + r",}"
    return re.search(pattern, text) is None


def test_no_other_scripts(text: str, **_) -> bool:
    """No characters from Hangul / Hiragana / Katakana / Cyrillic / Arabic
    scripts. These should never appear in a Mandarin transcript; their
    presence indicates Whisper language-confusion leakage from multilingual
    training data (e.g. the Korean `명` observed in subject 02010011/04)."""
    return other_script_count(text) == 0


def test_no_sound_effect_tokens(text: str, **_) -> bool:
    """No bracketed or parenthesized Latin-only tokens that look like
    sound-effect labels ([Music], (applause), [laughs], (typing)).
    Catches Whisper's suppress-tokens leakage on non-speech audio."""
    return _SOUND_EFFECT_PATTERN.search(text) is None


def test_no_url_or_social_artifacts(text: str, **_) -> bool:
    """No URL fragments, domain names, or social-media handles. These are
    training-data leakage from web/subtitle datasets and should never
    appear in clinical interview audio."""
    return _URL_SOCIAL_PATTERN.search(text) is None


def test_no_control_or_zero_width_chars(text: str, **_) -> bool:
    """No ASCII control chars (other than \\t / \\n), zero-width characters,
    BOM, bidi overrides, or word-joiner / invisible operators. These
    invisibly corrupt downstream tokenization and string matching even
    when the transcript looks clean to the eye."""
    return _CONTROL_AND_ZW_PATTERN.search(text) is None


def test_no_excessive_digits(text: str, **_) -> bool:
    """Digit content does not exceed MAX_DIGIT_RATIO_NWS of non-whitespace
    chars. Skipped for very short transcripts where one or two digits
    dominate the ratio by arithmetic rather than indicating a real issue.
    Defends against Whisper hallucinations like phone numbers, time codes,
    or runaway numeric loops; legitimate ages / years / durations stay
    well below the threshold (highest legit ratio observed on corpus:
    0.115 on '种了500万')."""
    nws = non_whitespace_len(text)
    if nws < DIGIT_RATIO_MIN_CHARS:
        return True
    return (digit_count(text) / nws) <= MAX_DIGIT_RATIO_NWS


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    if not AUDIO_DIR.exists():
        print(f"ERROR: audio dir not found at {AUDIO_DIR}", file=sys.stderr)
        return 1

    candidates = sorted(AUDIO_DIR.rglob(TRANSCRIPT_GLOB))
    if not candidates:
        print(f"ERROR: no transcripts matched {TRANSCRIPT_GLOB} under {AUDIO_DIR}", file=sys.stderr)
        return 1

    # For each (subject_id, file_num) keep only the highest-version file.
    # Version 1 = unversioned (no suffix); versions 2+ are the `_v2`, `_v3`,
    # ... files written by re-runs of extract_whisper_features.py.
    latest: dict[tuple[str, int], tuple[Path, int]] = {}
    skipped_unparseable = 0
    for tpath in candidates:
        m = TRANSCRIPT_VERSION_RE.match(tpath.name)
        if not m:
            skipped_unparseable += 1
            continue
        subject_id = tpath.parent.name
        file_num = int(m.group("stem"))
        version = int(m.group("version")) if m.group("version") else 1
        key = (subject_id, file_num)
        if key not in latest or version > latest[key][1]:
            latest[key] = (tpath, version)

    if skipped_unparseable:
        print(
            f"WARN: skipped {skipped_unparseable} files matching the glob "
            f"but not the version pattern",
            file=sys.stderr,
        )

    n_v1 = sum(1 for (_, v) in latest.values() if v == 1)
    n_higher = sum(1 for (_, v) in latest.values() if v > 1)
    if n_higher:
        print(
            f"Discovered {len(latest)} (subject,file) keys: "
            f"{n_v1} using v1 (unversioned), {n_higher} using v2+ (re-extracted)"
        )

    # Pass 1: read each selected transcript and build the duplicate-text index.
    rows: list[tuple[str, int, str, int]] = []  # (subject, file_num, text, version)
    text_to_locs: dict[str, list[tuple[str, int]]] = defaultdict(list)

    for (subject_id, file_num), (tpath, version) in sorted(latest.items()):
        text = tpath.read_text(encoding="utf-8").strip()
        text_to_locs[text].append((subject_id, file_num))
        rows.append((subject_id, file_num, text, version))

    # Pass 2: per-row audit results.
    output_rows: list[dict] = []
    for subject_id, file_num, text, version in rows:
        # Compute diagnostics once and reuse for tests + CSV columns.
        char_n = len(text)
        cjk_n = cjk_count(text)
        latin_n = latin_count(text)
        digit_n = digit_count(text)
        other_n = other_script_count(text)
        ctrl_n = control_or_zero_width_count(text)
        nws_n = non_whitespace_len(text)
        informative_n = informative_char_count(text)
        cjk_ratio_informative = (cjk_n / informative_n) if informative_n > 0 else 0.0
        latin_ratio_nws = (latin_n / nws_n) if nws_n > 0 else 0.0
        digit_ratio_nws = (digit_n / nws_n) if nws_n > 0 else 0.0
        comp_ratio = compression_ratio(text) if nws_n >= COMPRESSION_RATIO_MIN_CHARS else 0.0
        worst_ngram_n = max_ngram_repetition(text)
        longest_run = longest_same_char_run(text)

        # `passed_unique_across_subjects`: for the same text, are all
        # occurrences confined to the same file_num? (file 19 = passage
        # reading is identical across subjects by design; that's expected
        # and should pass.)
        if text and len(text) >= MIN_LENGTH_FOR_UNIQUENESS_CHECK:
            locs = text_to_locs[text]
            same_text_other_filenum = any(fn != file_num for (_sid, fn) in locs)
            passed_unique = not same_text_other_filenum
        else:
            # Skip: empties (caught by passed_non_empty) and short common
            # answers below MIN_LENGTH_FOR_UNIQUENESS_CHECK chars are not
            # uniqueness-test concerns.
            passed_unique = True

        results: dict = {
            "subject_id": subject_id,
            "file_num": file_num,
            # Boolean test results
            "passed_non_empty": test_non_empty(text),
            "passed_min_length": test_min_length(text, file_num=file_num),
            "passed_no_repetition_loop": test_no_repetition_loop(text),
            "passed_compression_ratio": test_compression_ratio(text),
            "passed_no_known_hallucination_phrase": test_no_known_hallucination_phrase(text),
            "passed_predominantly_chinese": test_predominantly_chinese(text),
            "passed_no_excessive_latin": test_no_excessive_latin(text),
            "passed_no_repeated_char_run": test_no_repeated_char_run(text),
            "passed_no_other_scripts": test_no_other_scripts(text),
            "passed_no_sound_effect_tokens": test_no_sound_effect_tokens(text),
            "passed_no_url_or_social_artifacts": test_no_url_or_social_artifacts(text),
            "passed_no_control_or_zero_width_chars": test_no_control_or_zero_width_chars(text),
            "passed_no_excessive_digits": test_no_excessive_digits(text),
            "passed_unique_across_subjects": passed_unique,
        }
        results["passed_all"] = all(
            v for k, v in results.items()
            if k.startswith("passed_") and k != "passed_all"
        )
        # Diagnostic columns (S1) -- raw counts and ratios for manual
        # inspection. Don't affect pass/fail; help triage near-threshold
        # cases without re-running the audit.
        results.update({
            "transcript_version": version,
            "char_count": char_n,
            "cjk_count": cjk_n,
            "latin_count": latin_n,
            "digit_count": digit_n,
            "other_script_count": other_n,
            "control_or_zero_width_count": ctrl_n,
            "non_whitespace_len": nws_n,
            "informative_char_count": informative_n,
            "cjk_ratio_informative": round(cjk_ratio_informative, 4),
            "latin_ratio_nws": round(latin_ratio_nws, 4),
            "digit_ratio_nws": round(digit_ratio_nws, 4),
            "compression_ratio": round(comp_ratio, 4),
            "worst_repeated_ngram_count": worst_ngram_n,
            "longest_same_char_run": longest_run,
        })
        results["transcript_text"] = text
        output_rows.append(results)

    # Write CSV.
    fieldnames = [
        "subject_id",
        "file_num",
        # Boolean test results
        "passed_non_empty",
        "passed_min_length",
        "passed_no_repetition_loop",
        "passed_compression_ratio",
        "passed_no_known_hallucination_phrase",
        "passed_predominantly_chinese",
        "passed_no_excessive_latin",
        "passed_no_repeated_char_run",
        "passed_no_other_scripts",
        "passed_no_sound_effect_tokens",
        "passed_no_url_or_social_artifacts",
        "passed_no_control_or_zero_width_chars",
        "passed_no_excessive_digits",
        "passed_unique_across_subjects",
        "passed_all",
        # Diagnostic columns
        "transcript_version",
        "char_count",
        "cjk_count",
        "latin_count",
        "digit_count",
        "other_script_count",
        "control_or_zero_width_count",
        "non_whitespace_len",
        "informative_char_count",
        "cjk_ratio_informative",
        "latin_ratio_nws",
        "digit_ratio_nws",
        "compression_ratio",
        "worst_repeated_ngram_count",
        "longest_same_char_run",
        # Raw text last (longest column)
        "transcript_text",
    ]
    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in output_rows:
            writer.writerow(row)

    # ------------------------------------------------------------------
    # S2: Per-subject summary CSV.
    # ------------------------------------------------------------------
    SUMMARY_CSV = REPO_ROOT / "whisper_transcript_audit_per_subject.csv"

    by_subject: dict[str, list[dict]] = defaultdict(list)
    for row in output_rows:
        by_subject[row["subject_id"]].append(row)

    test_columns = [c for c in fieldnames if c.startswith("passed_") and c != "passed_all"]

    summary_rows: list[dict] = []
    for sid in sorted(by_subject.keys()):
        subj_rows = by_subject[sid]
        n_files = len(subj_rows)
        n_failed_any = sum(1 for r in subj_rows if not r["passed_all"])
        s = {
            "subject_id": sid,
            "n_files": n_files,
            "n_failed_any_test": n_failed_any,
            "failure_rate": round(n_failed_any / n_files, 4),
        }
        for t in test_columns:
            short_name = t.replace("passed_", "n_failed_")
            s[short_name] = sum(1 for r in subj_rows if not r[t])
        s["total_nws_chars"] = sum(r["non_whitespace_len"] for r in subj_rows)
        s["mean_text_len"] = round(
            sum(r["char_count"] for r in subj_rows) / n_files, 1
        )
        s["max_text_len"] = max(r["char_count"] for r in subj_rows)
        s["min_text_len"] = min(r["char_count"] for r in subj_rows)
        summary_rows.append(s)

    summary_fieldnames = (
        ["subject_id", "n_files", "n_failed_any_test", "failure_rate"]
        + [t.replace("passed_", "n_failed_") for t in test_columns]
        + ["total_nws_chars", "mean_text_len", "max_text_len", "min_text_len"]
    )

    with SUMMARY_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    # ------------------------------------------------------------------
    # Stdout summary.
    # ------------------------------------------------------------------
    n_total = len(output_rows)
    print(f"Audited {n_total} transcripts -> {OUTPUT_CSV}")
    print(f"Per-subject summary ({len(summary_rows)} subjects) -> {SUMMARY_CSV}")
    print()
    print(f"{'TEST':45s}  {'PASS':>6s}  {'FAIL':>6s}  {'FAIL %':>7s}")
    print("-" * 75)
    for field in fieldnames:
        if not field.startswith("passed_"):
            continue
        n_pass = sum(1 for r in output_rows if r[field])
        n_fail = n_total - n_pass
        pct = 100.0 * n_fail / max(1, n_total)
        print(f"{field:45s}  {n_pass:6d}  {n_fail:6d}  {pct:6.2f}%")

    # Top 5 worst subjects by failure rate (helps surface systemic issues
    # like 02010036 without manual aggregation).
    worst = sorted(summary_rows, key=lambda s: -s["n_failed_any_test"])[:5]
    if worst[0]["n_failed_any_test"] > 0:
        print()
        print("Top 5 subjects by failed-test count:")
        print(f"  {'subject_id':>10s}  {'n_files':>7s}  {'n_failed':>8s}  {'rate':>6s}")
        for s in worst:
            print(
                f"  {s['subject_id']:>10s}  {s['n_files']:>7d}"
                f"  {s['n_failed_any_test']:>8d}  {s['failure_rate']:>6.2%}"
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
