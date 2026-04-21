# MODMA Dataset: A Comprehensive Guide for ML Researchers

## What Is This Document?

This document introduces the MODMA (Multi-modal Open Dataset for Mental-disorder Analysis) dataset to researchers who will be working with it extensively. It covers what the data are, where they came from, how they were collected, and what you'll see when you open the files. If you've never worked with EEG or clinical speech data before, this document will get you oriented.

---

## 1. Overview

MODMA is a Chinese-language multimodal dataset designed for computational depression and anxiety research. It was collected at Lanzhou University Second Hospital (Gansu Province, China) and published as a data descriptor in *Nature Scientific Data* in 2022 (Cai et al., 2022; DOI: 10.1038/s41597-022-01211-x).

The dataset contains three types of data collected from the same clinical population:

- **128-channel EEG** (laboratory-grade brain recordings)
- **3-channel EEG** (wearable/portable brain recordings)
- **Audio** (spoken responses to clinical interview questions, read-aloud passages, and picture descriptions)

Each participant also has clinical labels: a **PHQ-9** depression severity score, a **GAD-7** anxiety severity score, and a binary diagnostic label (MDD vs. healthy control).

---

## 2. Participants

### Who Was Recruited

Participants fell into two groups:

**MDD (Major Depressive Disorder) patients:**
- Recruited from the inpatient and outpatient psychiatric wards at Lanzhou University Second Hospital
- Diagnosed by a psychiatrist using the MINI (Mini-International Neuropsychiatric Interview), a standardized structured diagnostic interview based on DSM-IV criteria — not self-diagnosed
- Required PHQ-9 score ≥ 5 for inclusion
- Required to be medication-free: no psychotropic drugs in the preceding two weeks. This is important because antidepressants, anxiolytics, and antipsychotics all affect EEG signals and speech patterns, so medication-free recording gives a cleaner physiological baseline
- **24 patients** (12 male, 12 female)

**Healthy Controls (HC):**
- Recruited from the Lanzhou community via university posters and flyers
- Screened to exclude any personal or family history of mental disorders
- Required primary education or higher
- **29 controls** (gender breakdown varies by sub-dataset; approximately 15 male, 9 female in the 128-channel EEG subset)

### Demographics

- **Age range:** 16–52 years (inclusion criterion: 18–55)
- **Education:** Primary or higher (required for both groups, to ensure participants could complete reading tasks and questionnaires)

### Exclusion Criteria

For MDD patients: history of psychosis, bipolar disorder, substance abuse, neurological disorders, pregnancy, or any contraindication for EEG (e.g., pacemakers, metallic implants).

For healthy controls: any current or past psychiatric diagnosis, any current psychoactive medication.

### Important Note on Sub-Dataset Overlap

Not all participants appear in all three sub-datasets. The counts are:

| Sub-dataset | Total | MDD | HC |
|---|---|---|---|
| 128-channel EEG (resting + ERP) | 53 | 24 | 29 |
| 3-channel wearable EEG (resting only) | 55 | — | — |
| Audio (interview + reading + pictures) | 52 | 23 | 29 |

The 128-channel EEG and audio subsets overlap substantially but are not identical — a few participants have EEG but not audio or vice versa. The 3-channel wearable EEG was collected on 55 subjects (resting state only). When planning multimodal fusion, you'll need to identify the intersection of participants who have both EEG and audio.

---

## 3. Clinical Labels

### PHQ-9 (Patient Health Questionnaire — 9 Items)

**What it measures:** Depression severity over the past two weeks.

**How it works:** The participant self-reports on 9 items corresponding to DSM-IV depression criteria. Each item is rated 0–3:
- 0 = "Not at all"
- 1 = "Several days"
- 2 = "More than half the days"
- 3 = "Nearly every day"

**The 9 items ask about:**
1. Anhedonia (little interest or pleasure in doing things)
2. Depressed mood (feeling down, depressed, or hopeless)
3. Sleep disturbance (trouble falling/staying asleep, or sleeping too much)
4. Fatigue (feeling tired or having little energy)
5. Appetite changes (poor appetite or overeating)
6. Guilt/worthlessness (feeling bad about yourself)
7. Concentration difficulty (trouble concentrating on things)
8. Psychomotor changes (moving or speaking noticeably slowly/restlessly)
9. Suicidal ideation (thoughts that you would be better off dead)

**Total score range:** 0–27.

**Clinical severity thresholds:**

| Score | Severity |
|---|---|
| 0–4 | None / Minimal |
| 5–9 | Mild |
| 10–14 | Moderate |
| 15–19 | Moderately severe |
| 20–27 | Severe |

**In MODMA:** All MDD patients scored ≥ 5 (this was an inclusion criterion). Most scored ≥ 10 (moderate or above). Healthy controls are expected to score 0–4. This score is stored in the metadata Excel files and can be used as a continuous regression target, not just a binary label.

**Important for modeling:** PHQ-9 is self-reported, which introduces noise — depressed patients may underreport (lack of insight, social desirability) or overreport (negative cognitive bias). This is a known property of the label, not a defect.

### GAD-7 (Generalized Anxiety Disorder — 7 Items)

**What it measures:** Anxiety severity over the past two weeks.

**How it works:** Same format as PHQ-9 — 7 items, each rated 0–3.

**The 7 items ask about:**
1. Feeling nervous, anxious, or on edge
2. Not being able to stop or control worrying
3. Worrying too much about different things
4. Trouble relaxing
5. Being so restless that it's hard to sit still
6. Becoming easily annoyed or irritable
7. Feeling afraid as if something awful might happen

**Total score range:** 0–21.

**Clinical severity thresholds:** 5 (mild), 10 (moderate), 15 (severe).

**In MODMA:** Collected alongside PHQ-9. Depression and anxiety frequently co-occur (roughly 50% of MDD patients have comorbid anxiety). To date, no published paper has used the GAD-7 labels from MODMA — they have been completely ignored by the research community.

### Binary Diagnostic Label

In addition to continuous scores, each participant has a binary label: MDD or HC, determined by the clinician-administered MINI interview. This is the label that all 36 published papers have used for classification.

---

## 4. Recording Equipment

### 128-Channel EEG System

| Specification | Detail |
|---|---|
| Device | HydroCel Geodesic Sensor Net (Electrical Geodesics Inc., Eugene, Oregon, USA) |
| Electrodes | 128 Ag/AgCl sensors in a geodesic arrangement |
| Sampling rate | 250 Hz |
| Reference electrode | Cz (vertex) |
| Impedance threshold | < 50 kΩ per electrode |
| Acquisition software | Net Station version 4.5.4 |
| Environment | Soundproof room, no electromagnetic interference |

The geodesic arrangement covers the entire scalp with roughly uniform spacing. It follows the 10-20 system principles but with much higher density. Key regions for depression research include the frontal electrodes (F3, F4 and neighbors — where alpha asymmetry is measured) and prefrontal electrodes.

### 3-Channel Wearable EEG

| Specification | Detail |
|---|---|
| Electrode positions | Fp1, Fpz, Fp2 (all prefrontal) |
| Electrode type | Ag/AgCl |
| Placement standard | International 10-20 system |
| Data format | TXT files (M × N arrays: M = electrodes, N = sample points) |

This wearable captures only prefrontal activity. It's clinically relevant because prefrontal regions are most implicated in depression, and wearable form factors matter for real-world screening. But the spatial resolution is dramatically lower than the 128-channel system.

### Audio Recording

| Specification | Detail |
|---|---|
| Microphone | Neumann TLM102 (high-end large-diaphragm condenser) |
| Audio interface | RME Fireface UCX |
| Sampling rate | 44.1 kHz |
| Bit depth | 24-bit |
| File format | Uncompressed WAV |
| Ambient noise | < 60 dB |
| SNR | ~20–30 dB |
| Mic distance | ~20 cm from participant |
| Environment | Same soundproof room as EEG |

The Neumann TLM102 is a professional studio-quality microphone. Combined with the RME interface, the audio quality is high — considerably better than most clinical speech datasets, which often use laptop microphones or handheld recorders.

---

## 5. Experiment Protocol

Each participant completed a single recording session lasting approximately 25 minutes at the hospital's clinical research lab. The session was structured with 1-minute rest periods between tasks to reduce fatigue and carryover effects.

### Task 1: Resting-State EEG (No Speech)

**What happens:** The participant sits quietly with eyes closed for 5 minutes while EEG is recorded.

**Purpose:** Establishes a neural baseline. Resting-state EEG is the most common paradigm in depression EEG research. The primary biomarker of interest is **frontal alpha asymmetry** — depressed individuals tend to show relatively less left-frontal alpha power compared to right-frontal, reflecting a withdrawal-oriented motivational style.

**Data produced:** 5 minutes of continuous 128-channel EEG per participant (250 Hz × 300 seconds × 128 channels = ~9.6 million data points per participant).

### Task 2: Dot-Probe Task / Event-Related Potentials (EEG, No Speech)

**What happens:** The participant views pairs of faces (one emotional, one neutral) displayed side by side on a screen. After a brief display, a small dot appears on one side, and the participant presses a button to indicate which side the dot appeared on.

**Stimuli:**
- 60 emotional-neutral face pairs total, drawn from CFAPS (Chinese Facial Affective Picture System)
- 20 fear-neutral pairs
- 20 sad-neutral pairs
- 20 happy-neutral pairs
- Faces are gender-balanced (equal male/female), with non-facial features (hair, clothing) cropped out
- Image size: 5.16 cm × 5.95 cm, with 12 cm between paired images
- Viewing angle: 14.25°

**Conditions:**
- *Congruent:* the dot appears on the same side as the emotional face
- *Incongruent:* the dot appears on the opposite side

**Measured:** Reaction time (RT) and accuracy. RTs < 100 ms or > 2000 ms are excluded as outliers. Non-response trials are excluded.

**Purpose:** Measures attentional bias. Depressed individuals tend to show faster responses when the dot replaces a sad/fearful face (attentional capture by negative stimuli) and slower disengagement from negative stimuli compared to healthy controls.

**Data produced:** EEG time-locked to stimulus events (for ERP analysis), plus behavioral data (.edat files with RT and accuracy per trial).

### Task 3: Structured Clinical Interview (Audio)

**What happens:** A trained clinical interviewer asks the participant 18 questions derived from DSM-IV depression criteria and the Hamilton Rating Scale for Depression (HRSD). The participant's spoken responses are recorded.

**The 18 questions are organized by emotional valence:**

- **Questions 1–6 (Positive):** Probe positive experiences and planning ability.
  - Examples: "If you have a vacation, please describe your travel plans." / "What is the best gift you have ever received, and how did you feel?"
- **Questions 7–12 (Neutral):** Probe general self-description and daily functioning.
  - Examples: "Please describe one of your friends, including age, job, characters, and hobbies." / "How do you evaluate yourself?"
- **Questions 13–18 (Negative):** Probe distress, sleep difficulties, and hopelessness.
  - Examples: "What would you like to do when you are unable to fall asleep?" / "What makes you desperate?"

**Language:** All questions and responses are in Mandarin Chinese.

**Purpose:** Spontaneous speech is considered the most ecologically valid speech task for depression assessment. Depressed speech is characterized by slower rate, longer pauses, lower pitch, reduced pitch variability (monotone), lower energy, and specific spectral changes. The emotional valence manipulation (positive → neutral → negative questions) is designed to probe whether depressed individuals show blunted positive affect, heightened negative affect, or both.

**Data produced:** One or more .wav files per participant containing their interview responses.

### Task 4: Read-Aloud Word Lists and Passage (Audio)

**What happens:** The participant reads aloud from printed materials.

**Stimuli:**
- **Positive words:** 10 common Chinese words with positive emotional connotations (selected from Hongfei Lin's affective ontology corpus)
- **Neutral words:** 10 common Chinese words with neutral connotations (selected from the Chinese sentimental words extremum table)
- **Negative words:** 10 common Chinese words with negative emotional connotations (same source as positive)
- **"The North Wind and the Sun":** A phonetically balanced fable from *The Principles of the International Phonetic Association*, Chinese version. This passage is used worldwide in speech research because it samples the phoneme inventory of a language in a naturalistic narrative context.

**Word selection criteria:** Commonly used, avoiding educational bias, with approximately equal stroke counts across groups (important for Chinese — stroke count correlates with word complexity/familiarity).

**Purpose:** Reading tasks control for content — every participant says the same words, eliminating the confound of different topics and sentence structures in spontaneous speech. The emotional word lists probe whether depressed individuals show different acoustic patterns when producing positive vs. negative content. The North Wind and the Sun provides a standardized connected-speech sample for comparing prosody, rate, and articulation.

**Data produced:** .wav files per task per participant.

### Task 5: Picture Description (Audio)

**What happens:** The participant is shown images and asked to describe them freely.

**Stimuli:**
- 3 face images from CFAPS: one positive expression, one neutral, one negative
- 1 image from the TAT (Thematic Apperception Test): a "crying woman" scene

**Purpose:** Semi-spontaneous speech — the content is less constrained than reading but more standardized than a free interview. The emotional images are designed to elicit affective language and observe whether depressed individuals produce less positive content or more negative content. The TAT image is a classic clinical tool for assessing narrative style and emotional processing.

**Data produced:** .wav files per participant.

---

## 6. File Structure and Organization

### Download Packages

The dataset is distributed as separate zip archives:

```
EEG_128channels_resting_lanzhou_2015.zip      ← 128-ch resting state EEG
EEG_128channels_ERP_lanzhou_2015.zip          ← 128-ch dot-probe task EEG
EEG_3channels_resting_lanzhou_2015.zip        ← 3-ch wearable resting EEG
audio_lanzhou_2015.zip                        ← All audio recordings
```

There is also a BIDS-format conversion available (EDF files).

### File Naming Convention

- Files prefixed **`0201`** → MDD patients (e.g., `020101`, `020102`, …)
- Files prefixed **`0203`** → Healthy controls (e.g., `020301`, `020302`, …)

In BIDS format, these are mapped to `sub-01`, `sub-02`, etc. The mapping is provided in the metadata Excel file.

### EEG Files

**128-channel:**
- Original format: `.mff` (Geodesic native) → converted to `.raw` (via Net Station Waveform Tools) → converted to `.mat` (MATLAB) → converted to `.EDF` (BIDS)
- Each file is a matrix: 128 channels × N time points (at 250 Hz)
- Can be loaded in Python with `mne` (for .EDF or .raw) or `scipy.io.loadmat` (for .mat)

**3-channel:**
- Format: `.txt` files
- Each file is an M × N array (M = 3 electrodes, N = sample points)
- Straightforward to load with `numpy.loadtxt`

### Audio Files

- Format: `.wav` (uncompressed, 44.1 kHz, 24-bit)
- One or more files per participant per task
- Can be loaded with `librosa`, `soundfile`, `scipy.io.wavfile`, or any standard audio library

### Behavioral Data (Dot-Probe Task)

- Format: `.edat` files (E-Prime format)
- Contains: reaction time, accuracy, cell number (condition identifier) per trial
- File naming: `Dot_Detection-0201XX` (MDD) or `Dot_Detection-0203XX` (HC)
- Can be parsed with specialized E-Prime readers or converted to CSV

### Metadata / Labels

- Format: Excel (`.xlsx`)
- Primary file: `participants_information_EEG_128channels_resting_lanzhou_2015.xlsx`
- Contents: participant ID, group (MDD/HC), age, sex, PHQ-9 score, GAD-7 score, original-to-BIDS ID mapping

---

## 7. Preprocessing Applied by Dataset Creators

### EEG

The creators applied the following preprocessing to the EEG data and provide both raw and preprocessed versions:

1. **Band-pass filtering:** 1–45 Hz (FIR filter) — removes slow drift below 1 Hz and high-frequency noise above 45 Hz
2. **Notch filter:** 50 Hz — removes power-line interference (China uses 50 Hz AC, unlike the US which uses 60 Hz)
3. **Artifact removal:** Adaptive noise canceller for eye-blink artifacts

The raw (unprocessed) data is also available, which is important if you want to apply your own preprocessing pipeline (e.g., ICA for artifact removal, different filter cutoffs, or re-referencing to average reference).

### Audio

The audio files are provided as raw, unprocessed WAV recordings. No feature extraction, noise reduction, or segmentation has been applied by the creators. You'll need to do your own preprocessing (e.g., silence removal, normalization, feature extraction via OpenSMILE, librosa, Praat, etc.).

---

## 8. What You'll Actually Work With: A Practical Summary

If you're building models on MODMA, here's what you'll load and what you'll predict:

### Inputs (Features You Extract)

**From EEG (128-channel):**
- Frequency band powers: delta (0.5–4 Hz), theta (4–8 Hz), alpha (8–13 Hz), beta (13–30 Hz), gamma (30+ Hz) — computed per channel via FFT or Welch's method
- Frontal alpha asymmetry: ln(alpha power at F4) − ln(alpha power at F3)
- Functional connectivity: coherence, phase-locking value (PLV), or mutual information between channel pairs
- Time-domain features: signal variance, Hjorth parameters, zero-crossing rate
- Or: raw signal fed directly into a CNN/LSTM/Transformer

**From Audio:**
- Standard acoustic features via OpenSMILE (eGeMAPSv02: 88 features including F0 statistics, jitter, shimmer, HNR, MFCCs, spectral features, loudness, speech rate)
- Or: mel-spectrograms / MFCCs fed into a CNN
- Or: wav2vec2 / HuBERT embeddings for learned representations
- You'll extract these separately per task (interview, reading, picture description) or combine them

**From 3-Channel Wearable EEG:**
- Same frequency band powers but only at 3 prefrontal locations
- More limited but clinically practical

### Targets (What You Predict)

| Target | Type | Range | Used in literature? |
|---|---|---|---|
| MDD vs. HC | Binary classification | {0, 1} | Yes — all 36 papers |
| PHQ-9 score | Continuous regression | 0–27 | No — never done |
| GAD-7 score | Continuous regression | 0–21 | No — never done |

### Sample Sizes to Expect

After identifying the overlap between sub-datasets, you'll likely have roughly 50 participants with both EEG and audio. This is small by ML standards but typical for clinical neuroscience. Plan for leave-one-subject-out (LOSO) cross-validation rather than a simple train/test split, and be cautious about overfitting.

---

## 9. Known Limitations

1. **Small sample size:** ~50 participants is modest. Models must be regularized, and results should use LOSO or repeated k-fold CV — not a single train/test split.

2. **Language:** All speech is in Mandarin Chinese. Acoustic features (F0, jitter, shimmer, energy) are largely language-independent, but prosodic patterns and some spectral features may be language-specific. Cross-linguistic generalization should not be assumed.

3. **Cross-sectional:** Each participant was recorded once. There is no longitudinal follow-up, no treatment response data, and no test-retest reliability assessment.

4. **Self-reported severity labels:** PHQ-9 and GAD-7 are self-report instruments. They are well-validated but inherently noisy. The clinician-administered MINI diagnosis provides a more reliable binary label but no severity gradation.

5. **Medication washout:** The 2-week medication-free requirement is a strength for signal purity but a limitation for ecological validity — in practice, most patients are medicated. Models trained on unmedicated patients may not generalize to medicated populations.

6. **Sub-dataset mismatch:** Not all participants have all modalities. Cross-modal work requires careful identification of the overlapping subset.

---

## 10. Access

- **URL:** http://modma.lzu.edu.cn/data/index/
- **Process:** Select desired packages → download and sign the EULA → upload signed EULA → receive download links
- **Cost:** Free
- **GitHub:** https://github.com/UAIS-LANZHOU/MODMA-Dataset
- **Also available on:** Kaggle (https://www.kaggle.com/datasets/mimino12/modma-dataset)

---

## 11. Citation

If you use this dataset, cite:

> Cai, H., Yuan, Z., Gao, Y., et al. A multi-modal open dataset for mental-disorder analysis. *Scientific Data* **9**, 178 (2022). https://doi.org/10.1038/s41597-022-01211-x
