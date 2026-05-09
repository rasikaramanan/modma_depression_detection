# Whisper Linguistic Feature Pipeline

**Document purpose:** Ground-truth reference for how every feature in
`data/features/transcripts_features.parquet` was derived. The narrative below
will be adapted into the Methods section of the final report.

**Target deliverable:** `data/features/transcripts_features.parquet`
— 1471 rows × 29 columns (12 input + 1 diagnostic + **16 features**),
covering 52 subjects (29 HC, 23 MDD), median 29 audio files per subject.

---

## 1. Pipeline overview

The journey from raw audio to feature matrix runs through four scripts, in
order, all on the `alicia_branch` working tree:

| Stage | Script | Input | Output |
|---|---|---|---|
| 1. Audio → text | `scripts/extract_whisper_features.py` | `audio_lanzhou_2015/**/*.wav` | per-file `*_transcript_faster_whisper_zh.txt` (with `_v2` / `_v3` re-extractions for the audit-flagged subset) |
| 2. Quality audit | `scripts/audit_whisper_transcripts.py` | the transcripts | `data/metadata/data_quality_issues.csv` (37 excludes) |
| 3. Cleaning + segmentation | `scripts/preprocess_transcripts.py` | latest-version transcripts minus exclusion list | `data/features/transcripts_preprocessed.parquet` (1471 × 12) |
| 4. Feature extraction | `scripts/extract_text_features.py` | the preprocessed Parquet | `data/features/transcripts_features.parquet` (1471 × 29) |

Each downstream script reads only the immediately upstream artifact, so the
stages are independently reproducible.

---

## 2. Stage 1 — Audio transcription (WAV → text)

**Script:** `scripts/extract_whisper_features.py`
**Library:** [faster-whisper](https://github.com/SYSTRAN/faster-whisper) 1.0+
(CTranslate2 reimplementation of OpenAI Whisper).
**Model:** `large-v3` (1.55 B parameters), pulled from the Systran HuggingFace
mirror at `Systran/faster-whisper-large-v3`.

### 2.1 Why Whisper-large-v3

Whisper (Radford et al. 2023, *Robust Speech Recognition via Large-Scale Weak
Supervision*, OpenAI) was trained on 680 k hours of multilingual web audio
including ~23 k hours of Mandarin. The `large-v3` checkpoint released in
November 2023 reduces Mandarin word-error rate by ~10–20 % over `large-v2` on
the FLEURS-zh and CommonVoice-zh test sets per OpenAI's own benchmarks.

It is the de-facto field standard for zero-shot Mandarin transcription in
recent depression-from-speech work (see e.g. the various MODMA-derived
re-transcription projects on GitHub since 2024). No domain-specific Mandarin
ASR model is publicly available for clinical-interview audio, so a
strong general-purpose model is the right starting point.

### 2.2 Why faster-whisper rather than openai-whisper

faster-whisper wraps the same model weights through CTranslate2, which gives
a 2–4× speedup on CPU and supports `int8` quantization with negligible WER
impact on Whisper-class models (CTranslate2 benchmarks). MODMA inference runs
on a Mac M-series CPU; the speedup matters in practice (about 12 minutes for
1508 files vs about 50 minutes with vanilla `openai-whisper`).

### 2.3 Default inference parameters

```
beam_size                    = 5
condition_on_previous_text   = False
temperature                  = 0.0  (no fallback schedule on the default pass)
vad_filter                   = True
vad_parameters               = {min_silence_duration_ms: 500}  (Silero VAD)
compute_type                 = int8
language                     = zh
task                         = transcribe
```

Two of these are non-default in faster-whisper and chosen deliberately:

- `condition_on_previous_text = False`: prevents Whisper from carrying its
  own decoded text forward between 30-second windows. This is the standard
  mitigation for Whisper's well-documented "boilerplate hallucination" mode
  (Koenecke et al. 2024 *Careless Whisper*) where the model spirals into
  subscribe-button text, Amara.org credits, or repeated Chinese subtitle
  attributions once it makes one error early in a recording.
- `vad_filter = True` with Silero VAD: skips long silences before the model
  ever sees them. Reduces hallucination on quiet recordings — necessary for
  this corpus where some subjects speak very softly.

### 2.4 Re-extraction pass for audit failures

After the Stage 2 quality audit (Section 3), 40 transcripts were flagged as
empty or hallucinated. These were re-run through the same script with looser
parameters:

```
vad_filter        = False
initial_prompt    = "这是一段中文心理访谈录音。"   ("This is a Chinese psychological interview recording.")
compute_type      = float32                       (full precision)
temperature       = 0.0,0.2,0.4,0.6,0.8,1.0      (fallback schedule)
```

The `--initial-prompt` is the standard Whisper-prompting trick to bias decoding
toward in-domain Mandarin and away from English subtitle boilerplate. The
re-extraction outputs are written to `_v2.txt` (auto-versioned, never
overwriting the original), so every input file retains its full transcription
history.

For 26 of these 40 files the v2 re-extraction recovered usable speech; for
the other 14 (15 of which belong to subject 02010036) the re-extraction
produced new hallucinations and was excluded — see Section 8 on data integrity.

---

## 3. Stage 2 — Transcript quality audit

**Script:** `scripts/audit_whisper_transcripts.py`
**Output:** `data/metadata/data_quality_issues.csv`
(schema: `subject_id, file_number, severity, source, notes`).

The audit applies 14 boolean tests to every transcript and writes one row to
`data_quality_issues.csv` for each transcript that fails one or more tests
with `severity = exclude`. The Stage 3 script then refuses to process any
`(subject_id, file_number)` pair appearing on this list, so excluded files
never enter the feature matrix.

The 14 tests, grouped by failure mode:

1. **Length / emptiness**
   - `passed_non_empty` — file has any decoded characters.
   - `passed_min_length` — at least 2 CJK characters.
2. **Pathological repetition**
   - `passed_no_repetition_loop` — no n-gram of length ≥ 6 repeats > 3 times.
   - `passed_compression_ratio` — bzip2 compression ratio is realistic for
     fluent Chinese speech (catches degenerate "啊啊啊啊…" loops).
3. **Known Whisper hallucinations** — built from the literature
   (Koenecke et al. 2024) plus a project-specific list of phrases observed in
   our own outputs. ~70 phrase patterns including the canonical
   `请不吝点赞 订阅 转发 打赏支持明镜与点点栏目` subscribe-button text,
   Amara.org subtitle credits, Yang Donglian volunteer attributions, the
   Ming Pao newspaper byline, and the prompt-echo `这是一段中文心理访谈录音。`
   that the v2 re-extraction occasionally regurgitated.
   - `passed_no_known_hallucination_phrase`
4. **Script-level integrity**
   - `passed_predominantly_chinese` — ≥ 70 % CJK characters.
   - `passed_no_excessive_latin` — ≤ 10 % Latin characters.
   - `passed_no_other_scripts` — no Cyrillic / Arabic / Devanagari leakage.
   - `passed_no_repeated_char_run` — no single character repeats ≥ 8 times.
5. **Artifact tokens** observed empirically in Whisper-zh hallucinations
   - `passed_no_sound_effect_tokens` — `(笑)`, `(哭)`, `[Music]`, etc.
   - `passed_no_url_or_social_artifacts` — URLs, `@username`, `#hashtag`.
   - `passed_no_control_or_zero_width_chars`
   - `passed_no_excessive_digits` — ≤ 20 % digits among non-whitespace chars.
6. **Cross-corpus uniqueness**
   - `passed_unique_across_subjects` — same transcript text appearing in
     multiple subjects' files is almost certainly a hallucination
     (a real subject saying the exact same sentence as another subject is
     vanishingly unlikely on free-response prompts).

As of the 2026-05-08 audit there are 37 excluded `(subject, file)` pairs.
Subject 02010036 contributes 16 of them; subject 02010004 contributes 5
(physical wav-file corruption, not a Whisper failure). The remaining 16 are
distributed across other subjects.

---

## 4. Stage 3 — Cleaning + segmentation

**Script:** `scripts/preprocess_transcripts.py`
**Output:** `data/features/transcripts_preprocessed.parquet` (1471 rows ×
12 columns).

This stage takes the latest-version transcript on disk for each
`(subject_id, file_number)` pair (auto-detecting `_v2`/`_v3` suffixes) and
runs it through a 10-step cleaned-text + segmentation pipeline. The output
schema is

| column | type | description |
|---|---|---|
| `subject_id`, `file_num`, `transcript_version`, `source_path` | str/int | provenance |
| `raw_text` | str | exact contents of the source `.txt` |
| `cleaned_text` | str | output of cleaned-text steps 1–8 |
| `sentences` | list[str] | sentence-segmented `cleaned_text` |
| `tokens` | list[str] | flat word tokens across the transcript |
| `tokens_by_sentence` | list[list[str]] | tokens grouped by sentence |
| `n_chars`, `n_sentences`, `n_tokens` | int | length diagnostics |

### 4.1 Cleaned-text pipeline (8 steps)

These eight steps are applied in order to `raw_text`, producing
`cleaned_text`. Each step is idempotent given any reasonable input.

1. **Unicode NFKC normalization** (`unicodedata.normalize("NFKC", text)`).
   Unifies precomposed/decomposed CJK forms, half-width vs full-width Latin
   and digit characters, and compatibility decompositions (circled chars,
   superscripts, etc.). Foundational — every subsequent regex assumes a
   single canonical codepoint per character.
2. **Whitespace normalization** — collapse runs of any whitespace
   (including the NBSPs that NFKC creates) to single ASCII spaces; strip.
3. **Lowercase Latin characters** (`text.lower()`). Chinese characters are
   case-invariant; this only affects code-switched English. Ensures
   "iPhone" and "iphone" hash to the same token downstream.
4. **Traditional → Simplified Chinese.** OpenCC `t2s` config (see Section 7
   for justification).
5. **Within-Simplified variant normalization** — same `OpenCC.convert(text)`
   call also collapses many within-simplified variant character pairs
   (異體字) via the STCharacters / STPhrases mappings.
6. **Punctuation restoration.** FunASR `ct-punc` CT-Transformer model,
   revision `v2.0.4` (see Section 7). Whisper's Mandarin output is largely
   unpunctuated (it does not insert sentence-final 。 reliably);
   ct-punc inserts predicted 。 ， ？ ！ ； at sentence/clause boundaries.
   This step is what makes downstream sentence segmentation possible at all.
7. **ASCII → CJK punctuation unification.** ct-punc occasionally emits
   ASCII punctuation around code-switched Latin tokens; we translate
   `, . ? ! ; : ( ) < >` to their CJK equivalents `， 。 ？ ！ ； ： （ ） 《 》`
   so step 9's segmentation regex sees one consistent boundary form.
8. **Punctuation cleanup.** Two sub-operations applied in order:
   - 8a. drop sentence-internal punctuation that immediately follows a
     sentence-ender (regex `([。！？；…])\s*[，、：]+` → group 1). Catches
     a ct-punc quirk where it occasionally emits both `。` and `，` at the
     same boundary, which would otherwise produce sentences starting with
     orphan `，` after step 9.
   - 8b. collapse runs of identical punctuation
     (regex `([。，！？；：…])\1+` → group 1).

### 4.2 Segmentation pipeline (2 steps)

9. **Sentence segmentation.** Lookbehind regex split on CJK sentence-final
   punctuation `。 ！ ？ ； …`, keeping the punctuation attached to the
   preceding sentence. Empty fragments dropped.
10. **Word segmentation.** spacy_pkuseg (default model) — see Section 7 for
    why pkuseg over jieba. Each sentence is segmented independently, then
    flattened into the `tokens` column for whole-transcript features and
    preserved as `tokens_by_sentence` for per-sentence features.

---

## 5. Stage 4 — Feature extraction

**Script:** `scripts/extract_text_features.py`
**Output:** `data/features/transcripts_features.parquet` (1471 rows ×
29 columns; the 12 input columns plus 1 diagnostic + 16 features).

This stage reads the preprocessed Parquet and adds 17 new columns. All rates
are computed over **word tokens only** — the `tokens` column from Stage 3
includes punctuation tokens emitted by pkuseg, and these are filtered out
per LIWC convention (Pennebaker et al.) before any rate is computed.
Punctuation detection uses Unicode general category `P*`.

Three feature families are computed per transcript:

- **Family A — Lexical (5 features).** Closed-class word-list rates plus
  type-token-ratio measures.
- **Family B — Syntactic (3 features).** Simple counts derived from the
  existing token / sentence structure.
- **Family C — Sentiment via DUTIR (8 features).** Lexicon-based
  emotion-category rates over the 7 macro emotions of the Dalian University
  of Technology Chinese Emotion Vocabulary Ontology, plus a net polarity
  score from the same lexicon's positive/negative tags.

The diagnostic column `lex_n_word_tokens` (count of non-punctuation tokens)
is the denominator for Family A and Family C rates and is retained in the
Parquet for downstream auditing.

Full definitions of the 16 features are in Section 6.

---

## 6. Feature definitions (the 16 features)

Notation:
- $W = (w_1, w_2, \ldots, w_n)$ — the sequence of word tokens (punctuation
  excluded) for one transcript. $|W| = n$.
- $S = (s_1, s_2, \ldots, s_k)$ — the sequence of sentences. $|s_j|$ is
  the word-token count of sentence $j$.
- $T$ — the original token sequence including punctuation; $|T_{\text{punct}}|$
  is the count of punctuation tokens within $T$.

All 16 features yield NaN when their natural denominator is zero
(e.g. $n = 0$ or $k < 2$); see Section 8 for NaN-rate diagnostics.

### Family A — Lexical (5 features)

#### A1. `lex_first_person_sg_rate`
$$\text{lex\_first\_person\_sg\_rate} = \frac{|\{i : w_i \in \mathcal{P}_{\text{sg}}\}|}{n}$$
where $\mathcal{P}_{\text{sg}} = \{\text{我}, \text{我的}, \text{俺}\}$ is
the closed-class set of Mandarin first-person singular pronouns.
**Excludes** plural forms — see A2.
**Literature motivation:** Eichstaedt et al. 2018, *PNAS*; Tausczik &
Pennebaker 2010, *Journal of Language and Social Psychology* (LIWC review).
Both establish elevated first-person singular as one of the most replicable
linguistic markers of depression in spontaneous text.

#### A2. `lex_first_person_pl_rate`
$$\text{lex\_first\_person\_pl\_rate} = \frac{|\{i : w_i \in \mathcal{P}_{\text{pl}}\}|}{n}$$
where $\mathcal{P}_{\text{pl}} = \{\text{我们}, \text{咱们}, \text{咱}\}$.
**Tracked separately as a control feature**: depression elevates first-person
*singular* but not *plural* — sometimes the plural inverses (Eichstaedt et al.
2018). Including both lets the model see the contrast.

#### A3. `lex_negation_rate`
$$\text{lex\_negation\_rate} = \frac{|\{i : w_i \in \mathcal{N}\}|}{n}$$
where $\mathcal{N} = \{\text{不}, \text{没}, \text{没有}, \text{别}, \text{无}, \text{未}, \text{否}, \text{非}, \text{莫}\}$
is the standard closed-class set of Mandarin negators (Li & Thompson 1981).
**Literature motivation:** SC-LIWC 否定 (Negate) category, validated for
Chinese in Huang et al. 2012. Elevated negation is a long-standing LIWC
finding for depressed text (Tausczik & Pennebaker 2010) and has replicated
on Chinese Weibo depression text (Lyu et al. 2023, *Frontiers in Psychiatry*).

#### A4. `lex_ttr` — type-token ratio
$$\text{lex\_ttr} = \frac{|\{w_i : i \in 1..n\}|}{n}$$
The classical lexical-diversity ratio: unique types divided by total tokens.
**Caveat:** strongly length-dependent — short transcripts trivially get
TTR ≈ 1. Reported alongside MATTR (A5), which corrects for length.

#### A5. `lex_mattr50` — moving-average TTR, window = 50
$$\text{lex\_mattr50} = \frac{1}{n - W + 1} \sum_{i=1}^{n - W + 1} \frac{|\{w_j : j \in i..i+W-1\}|}{W}, \quad W = 50$$
NaN if $n < 50$.
**Literature motivation:** Covington & McFall 2010, *Journal of Quantitative
Linguistics* — proposes MATTR specifically to remove the length confound in
TTR. Used in clinical NLP for spontaneous-speech analysis. Reduced lexical
diversity is part of the "psychomotor slowing" linguistic signature of
depression (Mundt et al. 2007).

### Family B — Syntactic (3 features)

#### B1. `syn_mean_tokens_per_sent`
$$\text{syn\_mean\_tokens\_per\_sent} = \frac{1}{k} \sum_{j=1}^{k} |s_j|$$
Mean word tokens per sentence. **Literature motivation:** Mundt et al. 2007
*Journal of Neurolinguistics*; Cohn et al. 2018 — depression is associated
with shorter, simpler sentences in spontaneous speech.

#### B2. `syn_sd_tokens_per_sent`
$$\text{syn\_sd\_tokens\_per\_sent} = \sqrt{\frac{1}{k} \sum_{j=1}^{k} (|s_j| - \bar{s})^2}$$
Population standard deviation of sentence length (NaN if $k < 2$). Captures
within-transcript syntactic variability.

#### B3. `syn_punct_density`
$$\text{syn\_punct\_density} = \frac{|T_{\text{punct}}|}{|T|}$$
Fraction of all tokens (punct + word) that are punctuation. Higher density
roughly corresponds to more clausal boundaries per unit content — a coarse
proxy for syntactic chunking. Computed over the full token list $T$, not
just word tokens.

### Family C — Sentiment via DUTIR (8 features)

The DUTIR Chinese Emotion Vocabulary Ontology (Xu et al. 2008, see
Section 7) classifies each of its 27 466 entries into one of 21 fine-grained
emotion classes which are conventionally rolled up to **7 macro categories**
following Xu et al.'s original mapping:

| macro (pinyin) | Chinese | meaning | DUTIR fine-grained codes |
|---|---|---|---|
| `le` | 乐 | joy / happiness | PA + PE |
| `hao` | 好 | like / goodness | PD + PH + PG + PB + PK |
| `nu` | 怒 | anger | NA |
| `ai` | 哀 | sadness | NB + NJ + NH + PF |
| `ju` | 惧 | fear | NI + NC + NG |
| `e` | 恶 | disgust / dejection | NE + ND + NN + NK + NL |
| `jing` | 惊 | surprise | PC |

For every word $w_i$ in $W$, we look it up in DUTIR (primary sense, smallest
词义序号) and obtain at most one macro category and one polarity tag
(0 = neutral, 1 = positive, 2 = negative).

#### C1–C7. Macro emotion rates
For each macro category $m \in \{\text{le}, \text{hao}, \text{nu}, \text{ai}, \text{ju}, \text{e}, \text{jing}\}$:
$$\text{sent\_}m\text{\_rate} = \frac{|\{i : \text{macro}(w_i) = m\}|}{n}$$

The seven feature columns are
`sent_le_rate, sent_hao_rate, sent_nu_rate, sent_ai_rate, sent_ju_rate,
sent_e_rate, sent_jing_rate`.

`sent_ai_rate` (sadness) is the most directly interpretable depression
marker; the others are reported for completeness and as control features.

#### C8. `sent_net_polarity` — net polarity score
$$\text{sent\_net\_polarity} = \frac{|\{i : \text{pol}(w_i) = 1\}| - |\{i : \text{pol}(w_i) = 2\}|}{n}$$
where $\text{pol}(w_i)$ is DUTIR's 极性 column (1 = positive, 2 = negative,
0 = neutral, missing = not in lexicon). Positive values mean the transcript
contains more positive than negative emotion words by token rate.

**Literature motivation for the DUTIR family:** Xu et al. 2008 is the
original DUTIR paper. The lexicon is the most-cited Chinese affective
lexicon and is used in several Chinese text-based depression detection
papers (Yang & Cai 2020, *JMIR*; Lyu et al. 2023, *Frontiers in Psychiatry*).
Reduced positive-affect language and elevated negative-affect language are
established LIWC depression markers (Tausczik & Pennebaker 2010); DUTIR is
the Chinese-language operationalization.

### Note on the absent "absolutist" feature

The original linguistic-features plan (Pipeline 2 in
`notes/Feature_Pipeline_Explainer.md`) included an absolutist-thinking
category modeled on Al-Mosaiwi & Johnstone 2018 (*Clinical Psychological
Science*), which built a 19-word English absolutist dictionary
(`absolutely, all, always, complete, ...`). **No published Chinese-native
absolutist word list exists** for clinical text — TextMind / SC-LIWC has
a *Discrepancy* category but not *Absolutist*; the closest Chinese resource
is Lin et al. 2024's *cognitive-distortion sentence dataset*
(arXiv:2405.15334), which provides labeled sentences but not a word list.
Rather than translate Al-Mosaiwi's English list into Mandarin under
the project's own authority, the absolutist feature was dropped from this
pipeline. This is documented in the feature roster as a transparent
omission.

---

## 7. External libraries — literature justifications

### faster-whisper / Whisper-large-v3 (Stage 1)
- Radford et al. 2023, *Robust Speech Recognition via Large-Scale Weak
  Supervision* (OpenAI) — original Whisper paper.
- Koenecke et al. 2024, *Careless Whisper: Speech-to-Text Hallucination
  Harms*, *FAccT* — characterizes the hallucination patterns we audited
  for in Stage 2 and motivated `condition_on_previous_text=False`.
- CTranslate2 / faster-whisper: chosen for the int8 CPU speedup (see 2.2).

### OpenCC `t2s` (Stage 3, steps 4–5)
- Industry-standard Traditional ↔ Simplified Chinese converter
  (https://github.com/BYVoid/OpenCC). The `t2s` config uses STCharacters
  and STPhrases mappings, which also cover much within-Simplified variant
  normalization (異體字). No clear better alternative for this task.

### FunASR ct-punc / CT-Transformer (Stage 3, step 6)
- Chen et al. 2020, *Controllable Time-Delay Transformer for Real-Time
  Punctuation Prediction and Disfluency Detection*, ICASSP — the original
  CT-Transformer paper from Alibaba DAMO. Trained on Mandarin meeting
  transcripts and broadcast text; explicitly designed for ASR-output
  punctuation restoration.
- We use the FunASR (Gao et al. 2023, INTERSPEECH) packaged release,
  revision `v2.0.4`. FunASR is currently the most-cited and best-maintained
  Mandarin punctuation-restoration toolkit; the closest alternative is
  pre-2020 CRF-based punct restorers which have lower F1 on conversational
  Mandarin per the FunASR benchmarks.

### spacy_pkuseg / pkuseg (Stage 3, step 10)
- Luo et al. 2019, *PKUSEG: A Toolkit for Multi-Domain Chinese Word
  Segmentation*, arXiv:1906.11455 — the pkuseg paper.
- Chosen over **jieba** (the more popular but unmaintained alternative).
  jieba's last release was January 2020; it uses a fixed prefix-dictionary
  + HMM that has not been updated for modern conversational Mandarin.
  pkuseg is a CRF segmenter trained on multi-domain corpora and reports
  ~1–2 F1 points higher than jieba on social/news/web text in its own
  benchmarks. For an n=52 study where every misseg propagates into 16
  features, pkuseg's accuracy advantage matters.
- We use `spacy_pkuseg` (the maintained spaCy fork of pkuseg) with the
  default model, which is the Wikipedia-trained general-domain model.

### DUTIR Chinese Emotion Vocabulary Ontology (Stage 4, Family C)
- Xu et al. 2008, *Construction of a Chinese Emotion Lexicon (情感词汇本体)*,
  Journal of Chinese Information Processing — the original paper introducing
  the lexicon and its 21 → 7 emotion-class mapping.
- 27 466 entries across 21 fine-grained classes, with intensity (1–9)
  and polarity (0/1/2) tags.
- Field standard for Chinese text-based depression detection: used in
  Yang & Cai 2020 (*JMIR Mental Health*) and Lyu et al. 2023 (*Frontiers
  in Psychiatry*), among others.
- Considered alternatives explicitly rejected:
  - **HowNet** — older, binary positive/negative only, less granular.
  - **NTUSD** — Taiwanese Mandarin lexicon; smaller, older, less coverage.
  - **`IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment`** — a generic Chinese
    e-commerce-trained sentiment Transformer. We initially considered it
    but rejected after a literature scan found no published Chinese
    speech-based depression detection paper using it; its training corpus
    (ASAP-SENT, ChnSentiCorp) is product-review domain, not clinical.
  - **Chinese-MentalBERT** (Zhai et al. 2024) — domain-relevant but only
    informative when fine-tuned, which would require a labeled mental-health
    text corpus we do not have for $n = 52$ regression.

### Lexical-feature word lists (Stage 4, Family A) — DIY rather than library

- **First-person pronouns and negation lists** were sourced from
  Li & Thompson 1981 (*Mandarin Chinese: A Functional Reference Grammar*)
  as closed grammatical classes, cross-checked against the SC-LIWC 第一人称
  and 否定 categories (Huang et al. 2012, validated in Gao et al. 2013,
  *Cyberpsychology, Behavior, and Social Networking*).
- **TTR / MATTR** — pure formula, no library required (Covington &
  McFall 2010 for MATTR's window-50 default).
- We did not use the full SC-LIWC dictionary directly because (a) it is
  not openly redistributable and the official CAS portal downloads are
  reportedly broken, and (b) for $n = 52$ the marginal value of 80+ LIWC
  categories over the 3 lit-validated ones (1st-person sg, 1st-person pl,
  negation) is overwhelmed by overfitting risk.

---

## 8. Data integrity caveats

### Excluded files (37 of 1508)

`data/metadata/data_quality_issues.csv` lists 37 `(subject, file)` pairs
with `severity = exclude`. After exclusion the corpus has **1471 transcripts
across 52 subjects** (median 29 files/subject; min 13, max 29).

Two subject-level patterns dominate:

- **Subject 02010036 (16 excluded files).** Wav-level energy analysis
  (subject-relative voicing thresholds on 25 ms frames) confirmed the audio
  is real but unusually quiet — voiced-frame percentages 5–26 % vs this
  subject's own 20.6 % baseline and other subjects' 25–39 %. v2
  re-extraction with `--no-vad --initial-prompt --compute-type float32`
  produced 0 recoveries (14 of 15 generated new boilerplate hallucinations).
  Cause: subject spoke too softly for Whisper-large-v3's reliable
  transcription threshold on 16 of 29 files. Affects text features only;
  the wavs are intact and acoustic features (eGeMAPS) compute correctly.
- **Subject 02010004 (5 excluded files: 24–28).** Wav-level corruption
  confirmed at the file level (no openSMILE outputs either). Distinct
  data-integrity issue from 02010036's transcript-only problem.

The remaining 16 excludes are scattered across ~10 other subjects, mostly
from the audit's hallucination-phrase and predominantly-Chinese tests.

### NaN distribution in the feature matrix

Two features are NaN for some rows:

- `lex_mattr50` is NaN on 1255 of 1471 transcripts. Reason: MATTR-50
  requires $\geq 50$ word tokens; only longer transcripts (mostly the
  interview tasks) qualify. 216 transcripts have it populated.
- `syn_sd_tokens_per_sent` is NaN on 970 of 1471 transcripts. Reason:
  SD requires $\geq 2$ sentences; many short word-reading or single-clause
  responses are 1-sentence after ct-punc. 501 transcripts have it populated.

All other 14 features are populated on all 1471 rows.

### Sentiment features are task-confounded

DUTIR sentiment rates correlate strongly with the prompt's intended valence
(file 24 is a negative-valence word-reading list; file 26 is a positive-
valence word-reading list). When aggregating to subject level by mean
across all 29 files per subject, the task-mandated emotion-word loadings
dominate the per-subject mean and **wash out the depression signal**.
The right way to use these features is per-task subset modeling — see
`scripts/exploration.ipynb` cells 14–15 for the existing per-task / per-
valence split structure, which carries directly over to these features.

This was confirmed empirically: the whole-cohort MDD-vs-HC test on these
16 features finds 3 / 16 significant at p < .05 (sent_net_polarity,
sent_le_rate, lex_negation_rate); the interview-only re-test
(files 1–18) instead surfaces `lex_mattr50` (p = 0.017, t-test) which was
non-significant on the whole cohort — exactly because the word-reading
tasks force every subject to use the same predetermined vocabulary, killing
the diversity signal.

---

## 9. Reproducibility

### Code
All four scripts live in `scripts/` on the `alicia_branch` working tree:
`extract_whisper_features.py`, `audit_whisper_transcripts.py`,
`preprocess_transcripts.py`, `extract_text_features.py`.

### External assets
- DUTIR lexicon CSV at `data/external/dutir/DUTIR_emotion_ontology.csv`
  (1.49 MB; downloaded from
  https://github.com/yizhanmiao/DLUT-Emotionontology). 16 of 27 461 rows
  drop on parse due to a malformed escape; this is acceptable (< 0.06 %
  loss).
- Whisper-large-v3 weights cached at
  `~/.cache/huggingface/hub/models--Systran--faster-whisper-large-v3/`
  (~ 3 GB).
- ct-punc weights cached at
  `~/.cache/modelscope/hub/models/iic/punc_ct-transformer_cn-en-common-vocab471067-large/`
  (~ 1.1 GB).
- spacy_pkuseg model cached on first call.

### Environment
Python 3.11, conda env `csci567`. Key dependencies:
`faster-whisper`, `funasr` (>= 1.3.1), `opencc`, `spacy_pkuseg`, `pandas`,
`tqdm`, `scipy`. Installed via `pip install -U funasr` and similar
ad-hoc; no `requirements.txt` is committed yet (TODO).

### Order to re-run from scratch
```
python scripts/extract_whisper_features.py            # ~50 min (default settings)
# (optional) python scripts/extract_whisper_features.py --files audit_failure_manifest.txt --no-vad --initial-prompt "..."
python scripts/audit_whisper_transcripts.py           # ~30 s; updates data_quality_issues.csv
python scripts/preprocess_transcripts.py              # ~18 min (after first run; ct-punc + pkuseg cached)
python scripts/extract_text_features.py               # < 5 s (no models, just dict lookups)
```

Final artifact: `data/features/transcripts_features.parquet`.
