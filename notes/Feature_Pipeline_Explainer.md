# How Our Two Feature Pipelines Answer RQ1 and RQ2

**Project:** MODMA Depression Detection (CSCI 567)
**Date:** April 2026

---

## The Two Pipelines at a Glance

We extract features from each audio file using two independent pipelines. They capture fundamentally different information from the same recording and are designed to be complementary, not redundant.

| | **openSMILE (eGeMAPS)** | **Whisper Transcripts** |
|---|---|---|
| **What it reads** | Raw audio waveform | Raw audio waveform (internally) |
| **What it outputs** | 88 numeric acoustic features per utterance | Chinese text transcript per utterance |
| **What it captures** | *How* the person sounds | *What* the person says |
| **Signal type** | Continuous, low-level, speaker-physiology-driven | Discrete, high-level, language/cognition-driven |

---

## Pipeline 1: openSMILE / eGeMAPS — Acoustic Features

openSMILE processes the raw audio signal directly. It computes the eGeMAPS (extended Geneva Minimalistic Acoustic Parameter Set) feature set — 88 features per audio file, grouped into three families.

**Prosody** — rhythm, timing, intonation. Includes fundamental frequency (F0 / pitch), loudness contour, speaking rate, and jitter (cycle-to-cycle pitch variation). Depression is associated with slower speech, longer pauses, reduced pitch range, and monotone intonation.

**Spectral** — frequency-domain energy distribution. Includes MFCCs (mel-frequency cepstral coefficients), spectral flux, formant frequencies (F1–F3), and harmonic energy ratios. Depression is associated with decreased spectral energy variance — a "flatter," less dynamically expressive voice.

**Voice quality** — physical vocal fold characteristics. Includes shimmer (amplitude variation), harmonics-to-noise ratio (HNR), and harmonic difference measures (H1–H2, H1–A3). Depression is associated with a breathy, weak, or "lifeless" vocal quality.

These features are entirely independent of language content. A participant reading a happy word list versus answering a sad interview question will produce different eGeMAPS vectors even if the acoustic delivery is identical, only because the recording conditions differ — but the features themselves do not encode *what* was said, only *how* it was said.

---

## Pipeline 2: Whisper Transcripts — Linguistic / Semantic Features

Whisper (large-v3) transcribes each audio file into Mandarin Chinese text. The transcript itself is not the final feature — it is the intermediate representation from which we extract linguistic features using downstream NLP tools. These fall into several families.

**Lexical features** — what words the person chooses. First-person pronoun rate (我/我的), negation density (不/没/别), absolutist word rate (总是/永远/完全), type-token ratio (vocabulary diversity). Depression is associated with elevated self-referential language, more negation, more absolutist phrasing, and reduced lexical diversity.

**Sentiment / affective features** — emotional tone of the content. Computed via Chinese sentiment lexicons (SCLIWC, HowNet) or neural classifiers (Chinese-MentalBERT). Depression is associated with more negative-emotion words and fewer positive-emotion words, especially in spontaneous speech tasks.

**Syntactic features** — structural complexity of speech. Mean sentence length, clause count, subordination depth, dependency-parse complexity. Depression is associated with syntactically simpler output — shorter sentences, fewer embedded clauses. (These features require punctuation in the transcript to compute reliably.)

**Discourse coherence** — how well ideas connect across sentences. Measured by cosine similarity between adjacent sentence embeddings (using a Chinese sentence encoder like BGE). Depression and psychosis are both associated with reduced local coherence — sentences that relate less to each other.

**Neural embeddings** — dense vector representations of the full utterance from a pre-trained Chinese language model (Chinese-MentalBERT or BGE). These capture overall semantic content in a high-dimensional space that a classifier can learn from directly.

---

## Why Both Pipelines Are Needed

Neither pipeline alone captures the full picture.

A participant could speak with flat, monotone prosody (strong eGeMAPS signal for depression) while saying perfectly coherent, lexically rich content — or vice versa. The acoustic and linguistic channels carry partially independent information about mental state.

For **RQ1** (predicting continuous PHQ-9 severity with prediction intervals): Fusing acoustic and linguistic features into a single model gives the model access to both channels of depression signal. Prior work on the DAIC-WOZ corpus (the standard English-language benchmark) consistently shows that multimodal acoustic + text models outperform either modality alone for severity regression (RMSE improvements of 1–2 points). Our multi-task learning framework with auxiliary GAD-7 and PSQI targets benefits from having more diverse input features to share across prediction heads.

For **RQ2** (which speech task and emotional valence best discriminates MDD vs. HC): We compute both eGeMAPS and transcript features *per task per subject*, then compare discriminative power across the five MODMA tasks (interview, word reading, passage reading, picture description) and three valence conditions (positive, neutral, negative). The key question is whether certain tasks are discriminative because of *how* people sound (acoustic), *what* they say (linguistic), or both — and only by having both feature sets can we answer that.

---

## Summary Table

| Feature family | Source pipeline | Requires punctuation? | Example depression marker |
|---|---|---|---|
| Pitch / F0 | openSMILE | N/A | Reduced pitch range, monotone |
| Spectral (MFCCs) | openSMILE | N/A | Flatter spectral energy |
| Voice quality | openSMILE | N/A | Breathy, low HNR |
| Lexical (pronouns, negation) | Whisper → NLP | No | Elevated first-person singular |
| Sentiment / emotion | Whisper → NLP | No | More negative-emotion words |
| Syntactic complexity | Whisper → NLP | **Yes** | Shorter, simpler sentences |
| Discourse coherence | Whisper → NLP | **Yes** | Lower adjacent-sentence similarity |
| Neural embeddings | Whisper → NLP | Helpful | Learned depression-relevant semantics |
