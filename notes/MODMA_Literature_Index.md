# Comprehensive Guide to Papers Using the MODMA Dataset

**Last updated:** 2026-04-13
**Total papers identified:** 58 (48 using MODMA directly + 5 foundational/pre-MODMA papers by dataset creators + 5 related papers using other datasets)

---

## How to Use This Guide

This document catalogs every published paper we could identify that uses the MODMA (Multi-modal Open Dataset for Mental-disorder Analysis) dataset, organized by category. For each paper we provide a link, the modality used (EEG, audio, or multimodal), and a two-sentence summary. The final section lists closely related papers that do NOT use MODMA but are methodologically relevant.

---

## Category A: Dataset Descriptor & Foundational Papers by MODMA Creators

These are the papers produced by the Lanzhou University team that created MODMA. They used the same underlying data (sometimes before the public release) and are essential reading for understanding the dataset.

### A1. MODMA Dataset: A Multi-modal Open Dataset for Mental-disorder Analysis
- **Authors:** Cai, H., Yuan, Z., Gao, Y., Sun, S., Li, N., Tian, F., Xiao, H., Li, J., Yang, Z., Li, X., Hu, B.
- **Year:** 2022 (arXiv preprint 2020)
- **Venue:** Scientific Data (Nature)
- **Link:** [https://www.nature.com/articles/s41597-022-01211-x](https://www.nature.com/articles/s41597-022-01211-x)
- **Modality:** EEG + Audio (dataset descriptor)
- **Summary:** The foundational dataset descriptor paper introducing MODMA, which contains 128-channel EEG, 3-channel wearable EEG, and speech recordings from 53 participants (24 MDD, 29 HC) recruited at Lanzhou University Second Hospital. Provides benchmark binary classification results and distributes PHQ-9, GAD-7, PSQI, CTQ-SF, LES, and SSRS scores alongside the raw signals.
- **EEG Preprocessing:**
  - **128-channel resting-state EEG:** Recorded at 250 Hz with 128-ch HydroCel Geodesic Sensor Net (Cz online reference, impedance <50 kΩ). Benchmark preprocessing documented by the creators: 1 Hz high-pass + 45 Hz low-pass Hamming-windowed sinc FIR filter; TrimOutlier plugin for EOG/EMG artifact rejection; threshold-based (mean ± SD) bad-channel rejection with spherical interpolation; REST re-referencing; Artifact Subspace Reconstruction (ASR) for high-power epochs; 40 × 2-s artifact-free epochs (80 s) selected per subject; a 16-electrode subset (Fp1/2, F3/4, C3/4, P3/4, O1/2, F7/8, T3/4, T5/6) feeds linear, nonlinear, and PLI-based functional-connectivity feature extraction.
  - **128-channel dot-probe/ERP EEG:** Same acquisition setup as the resting recording (250 Hz, 128-ch HCGSN, Cz reference). Trigger markers for fixation, cue, interval, target, and response are provided for event-locked analysis. The dataset descriptor distributes the raw ERP streams and trigger files but does not specify a standardized benchmark preprocessing pipeline for the dot-probe task itself.
  - **3-channel wearable resting EEG:** Electrodes Fp1, Fpz, Fp2 acquired at 250 Hz. Benchmark preprocessing: 1–45 Hz FIR bandpass; adaptive noise canceller for eye-blink artifacts. ICA/ASR are not applied to the wearable data because of the limited channel count.
- **Audio Preprocessing:** Benchmark uses two feature types: (1) **Fbank** — 40-dimensional mel-scale filterbank energies extracted with Hamming window, 25ms frame length, 10ms hop; (2) **Spectrogram** — 1024-point FFT applied per frame, energy spectra concatenated across frames. No pre-trained models; standard DSP pipeline.

### A2. A Study of Resting-State EEG Biomarkers for Depression Recognition
- **Authors:** Sun, S., Li, J., Chen, H., Gong, T., Li, X., Hu, B.
- **Year:** 2020
- **Venue:** arXiv (preprint)
- **Link:** [https://arxiv.org/abs/2002.11039](https://arxiv.org/abs/2002.11039)
- **Modality:** 128-channel EEG
- **Summary:** An early study by the MODMA team investigating resting-state EEG biomarkers (spectral power, connectivity) for distinguishing MDD from healthy controls using the same 53-subject cohort. Identifies candidate biomarkers in alpha and theta bands that correlate with depressive state.
- **EEG Preprocessing:**
  - **128-channel resting-state EEG:** 5-min eyes-closed recording at 250 Hz on 128-ch HCGSN (Cz reference, impedance <50 kΩ). 1–40 Hz Hamming-windowed sinc FIR bandpass (removes 50 Hz line noise and baseline drift); TrimOutlier plugin for EOG/EMG rejection; threshold-based bad-channel rejection with spherical interpolation; REST re-referencing; ASR removes residual high-power epochs; signal segmented into 40 × 2-s clean epochs per subject; 16-electrode subset (standard 10-20 positions) selected. Features per channel: 8 linear features (including PSD sub-band power and Hjorth parameters), 6 nonlinear features (e.g. entropy), and phase-lagging-index (PLI) functional connectivity (120 entries from the 16×16 FC matrix). Features z-score normalized to [-1, 1]; ReliefF feature selection feeds a logistic-regression classifier.
  - **128-channel dot-probe/ERP EEG:** Not used by this paper.
  - **3-channel wearable resting EEG:** Not used by this paper.
- **Audio Preprocessing:** N/A — EEG only.

### A3. Graph Theory Analysis of Functional Connectivity in Major Depression Disorder With High-Density Resting State EEG Data
- **Authors:** Cai, H., Gao, Y., Sun, S., et al.
- **Year:** 2020
- **Venue:** IEEE Access
- **Link:** [https://modma.lzu.edu.cn/static/references/Graph%20Theory%20Analysis%20of%20Functional%20Connectivity%20in%20Major%20Depression%20Disorder%20With%20High-Density%20Resting%20State%20EEG%20Data.pdf](https://modma.lzu.edu.cn/static/references/Graph%20Theory%20Analysis%20of%20Functional%20Connectivity%20in%20Major%20Depression%20Disorder%20With%20High-Density%20Resting%20State%20EEG%20Data.pdf)
- **Modality:** 128-channel EEG
- **Summary:** Applies graph theory metrics (clustering coefficient, path length, small-worldness) to functional connectivity networks constructed from the MODMA 128-channel resting EEG. Finds that MDD patients exhibit altered network topology, with hubs predominating in the left hemisphere.
- **EEG Preprocessing:**
  - **128-channel resting-state EEG:** 128-channel resting-state EEG used; preprocessing focuses on functional-connectivity construction (imaginary part of coherence, Cluster-Span Threshold) followed by graph-theoretic metrics (clustering coefficient, path length, small-worldness). Specific bandpass cutoffs, artifact-removal method, and epoch parameters are not publicly available in the accessible portions of the paper.
  - **128-channel dot-probe/ERP EEG:** Not used by this paper.
  - **3-channel wearable resting EEG:** Not used by this paper.
- **Audio Preprocessing:** N/A — EEG only.

### A4. Multivariate Pattern Analysis of EEG-Based Functional Connectivity: A Study on the Identification of Depression
- **Authors:** Cai, H., Sha, X., Han, X., Wei, S., Hu, B.
- **Year:** 2019
- **Venue:** IEEE Access
- **Link:** [https://ieeexplore.ieee.org/document/8756209/](https://ieeexplore.ieee.org/document/8756209/)
- **Modality:** 128-channel EEG
- **Summary:** Uses phase lag index to construct functional connectivity matrices from MODMA's resting-state EEG and applies multivariate pattern analysis (MVPA) for MDD identification. Demonstrates that altered functional connectivity patterns, particularly in alpha band, can reliably discriminate MDD from HC.
- **EEG Preprocessing:**
  - **128-channel resting-state EEG:** 128-channel resting-state EEG used (5 min, 250 Hz). ICA applied for artifact removal; per-band (delta/theta/alpha/beta) Phase Lag Index (PLI) computed across channels to form functional-connectivity matrices; altered Kendall rank-correlation used for feature selection; binary linear SVM classification with permutation tests. Specific bandpass cutoffs, re-referencing choice, and epoch length are not publicly available in the accessible portions of the paper.
  - **128-channel dot-probe/ERP EEG:** Not used by this paper.
  - **3-channel wearable resting EEG:** Not used by this paper.
- **Audio Preprocessing:** N/A — EEG only.

### A5. A Resting-State Brain Functional Network Study in MDD Based on Minimum Spanning Tree Analysis and Hierarchical Clustering
- **Authors:** Cai, H., et al.
- **Year:** 2019
- **Venue:** Complexity (Hindawi)
- **Link:** Available at [MODMA publications page](https://modma.lzu.edu.cn/data/publications/)
- **Modality:** 128-channel EEG
- **Summary:** Constructs minimum spanning tree (MST) representations of brain functional networks from MODMA's resting-state EEG and applies hierarchical clustering. Identifies that MST topological metrics differ significantly between MDD and HC groups, suggesting disrupted information integration in depression.
- **EEG Preprocessing:**
  - **128-channel resting-state EEG:** 128-channel resting-state EEG used; coherence computed in the theta band, a Minimum Spanning Tree (MST) built from the coherence matrix, and hierarchical clustering applied to topological metrics (leaf fraction, clustering coefficient). Specific bandpass cutoffs, artifact-removal pipeline, epoch length, and re-referencing are not publicly available in the accessible portions of the paper.
  - **128-channel dot-probe/ERP EEG:** Not used by this paper.
  - **3-channel wearable resting EEG:** Not used by this paper.
- **Audio Preprocessing:** N/A — EEG only.

---

## Category B: EEG-Only Papers (128-Channel Resting-State)

### B1. A Depression Prediction Algorithm Based on Spatiotemporal Feature of EEG Signal
- **Authors:** Liu, S., et al.
- **Year:** 2022
- **Venue:** Brain Sciences (MDPI)
- **Link:** [https://www.mdpi.com/2076-3425/12/5/630](https://www.mdpi.com/2076-3425/12/5/630)
- **Modality:** 128-channel EEG
- **Summary:** Proposes a CNN+GRU architecture to capture spatiotemporal EEG features in theta, alpha, and beta bands for binary MDD vs. HC classification on MODMA. Achieves 89.63% accuracy using the combined spatial and temporal representations.
- **EEG Preprocessing:**
  - **128-channel resting-state EEG:** 0.5 Hz high-pass + 100 Hz low-pass bandpass with 50 Hz notch; FastICA used to remove ocular and muscular artifacts; signals converted into brain-map sequences spanning theta, alpha, and beta bands; the sequence of brain maps feeds a CNN-GRU architecture for joint spatial and temporal feature learning.
  - **128-channel dot-probe/ERP EEG:** Not used by this paper.
  - **3-channel wearable resting EEG:** Not used by this paper.

### B2. LSDD-EEGNet: An Efficient End-to-End Framework for EEG-Based Depression Detection
- **Authors:** Song, X., et al.
- **Year:** 2022
- **Venue:** Biomedical Signal Processing and Control
- **Link:** [https://www.sciencedirect.com/science/article/abs/pii/S1746809422001343](https://www.sciencedirect.com/science/article/abs/pii/S1746809422001343)
- **Modality:** 128-channel EEG
- **Summary:** Introduces a CNN-LSTM framework with a domain discriminator for cross-subject EEG-based depression detection, aiming to reduce subject-level distribution shift. Achieves up to 93.98% accuracy on the gamma band of MODMA's 128-channel resting-state EEG.
- **EEG Preprocessing:**
  - **128-channel resting-state EEG:** Wavelet transform used for noise elimination; EEG decomposed into five canonical bands (delta 1–4 Hz, theta 4–7 Hz, alpha 8–13 Hz, beta 13–30 Hz, gamma >30 Hz); per-band signals fed to a CNN-LSTM backbone (LSDD-EEGNet) coupled with a domain discriminator to reduce training/test subject-level distribution shift.
  - **128-channel dot-probe/ERP EEG:** Not used by this paper.
  - **3-channel wearable resting EEG:** Not used by this paper.

### B3. MAMF-GCN: Multi-Scale Adaptive Multi-Channel Fusion Deep Graph Convolutional Network for Predicting Mental Disorder
- **Authors:** Chen, X., et al.
- **Year:** 2022
- **Venue:** Computers in Biology and Medicine
- **Link:** [https://www.sciencedirect.com/science/article/abs/pii/S0010482522005832](https://www.sciencedirect.com/science/article/abs/pii/S0010482522005832)
- **Modality:** 128-channel EEG
- **Summary:** Proposes a multi-scale graph convolutional network with attention-based multi-channel fusion to integrate features from different brain network atlases. Evaluated on MODMA, DAIC-WOZ, and D-Vlog, exploiting multi-channel correlation for depression classification.
- **EEG Preprocessing:**
  - **128-channel resting-state EEG:** 128-channel resting-state EEG used; specific preprocessing steps (bandpass cutoffs, artifact removal, epoch length, feature definition) are not publicly available in the accessible (paywalled) portions of the paper, beyond the high-level description of multi-scale graph construction and attention-based multi-channel fusion.
  - **128-channel dot-probe/ERP EEG:** Not used by this paper.
  - **3-channel wearable resting EEG:** Not used by this paper.

### B4. EEG Based Depression Recognition Using Improved Graph Convolutional Neural Network
- **Authors:** Zhu, J., Jiang, C., Chen, J., Lin, X., Yu, R., Li, X., Hu, B.
- **Year:** 2022
- **Venue:** Computers in Biology and Medicine
- **Link:** [https://www.sciencedirect.com/science/article/abs/pii/S0010482522005765](https://www.sciencedirect.com/science/article/abs/pii/S0010482522005765)
- **Modality:** 128-channel EEG
- **Summary:** Enhances GCN by assigning learnable weights to each edge in the brain network adjacency matrix, continuously updated during training to reveal which functional connections are most important for depression classification. Validated on MODMA and a self-collected EDRA dataset, providing insights into the pathological mechanism of depression via EEG functional connectivity.
- **EEG Preprocessing:**
  - **128-channel resting-state EEG:** Four linear per-channel features extracted — activity, mobility, complexity (Hjorth parameters) and power spectral density (PSD); a Pearson-correlation-based brain functional-connectivity adjacency is built across channels; adjacency edges receive learnable weights that are updated during training; improved GCN with attention classifies. Specific bandpass cutoffs, artifact-removal method, and epoch length are not publicly available in the accessible portions of the paper.
  - **128-channel dot-probe/ERP EEG:** Not used by this paper.
  - **3-channel wearable resting EEG:** Not used by this paper.

### B5. Identifying Depression Disorder Using Multi-View High-Order Brain Function Network Derived from EEG Signal
- **Authors:** Zhao, S., Gao, Y., Cao, J., Chen, X., Mao, Y., Mao, G., Ren, F.
- **Year:** 2022
- **Venue:** Frontiers in Computational Neuroscience
- **Link:** [https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2022.1046310/full](https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2022.1046310/full)
- **Modality:** 128-channel EEG
- **Summary:** Constructs multi-view high-order brain function networks from MODMA EEG data using matrix variate normal distributions to capture higher-order interactions beyond pairwise connectivity. Demonstrates that high-order network features improve MDD identification accuracy over standard pairwise functional connectivity.
- **EEG Preprocessing:**
  - **128-channel resting-state EEG:** Artifact Subspace Reconstruction (ASR) removes bad epochs arising from eye blinks, muscle activity, and sensor motion (~89% clean data retained); leading and trailing segments discarded to ensure stable state; FFT-derived delta (1–4 Hz) and theta (4–8 Hz) power serve as channel-level features; Phase Lag Index (PLI) matrices computed across channels; multi-view low-order and high-order brain function networks generated via a matrix variate normal distribution and fed to the classifier.
  - **128-channel dot-probe/ERP EEG:** Not used by this paper.
  - **3-channel wearable resting EEG:** Not used by this paper.

### B6. Feature-Level Fusion Based on Spatial-Temporal of Pervasive EEG for Depression Recognition
- **Authors:** Zhang, B., Wei, D., Yan, G., Lei, T., Cai, H., Yang, Z.
- **Year:** 2022
- **Venue:** Computer Methods and Programs in Biomedicine
- **Link:** [https://www.sciencedirect.com/science/article/abs/pii/S0169260722004941](https://www.sciencedirect.com/science/article/abs/pii/S0169260722004941)
- **Modality:** 3-channel EEG
- **Summary:** Transforms 3-channel wearable EEG from MODMA into visibility graph representations and fuses spatial-temporal features for depression recognition. Achieves 95.2% accuracy, demonstrating the viability of low-cost wearable EEG for depression screening.
- **EEG Preprocessing:**
  - **128-channel resting-state EEG:** Not used by this paper.
  - **128-channel dot-probe/ERP EEG:** Not used by this paper.
  - **3-channel wearable resting EEG:** Three-electrode (Fp1, Fpz, Fp2) signals mapped to a spatial complex network via a visibility graph (VG); temporal EEG features and spatial VG metric features extracted and selected; feature-level fusion applies correlation-based contribution weights; a cascade forest of three decision forests performs classification.

### B7. Depression Signal Correlation Identification from Different EEG Channels Based on CNN Feature Extraction
- **Authors:** Wang, Y., et al.
- **Year:** 2022
- **Venue:** Psychiatry Research: Neuroimaging
- **Link:** [https://www.sciencedirect.com/science/article/abs/pii/S0925492722001391](https://www.sciencedirect.com/science/article/abs/pii/S0925492722001391)
- **Modality:** 128-channel EEG
- **Summary:** Investigates the classification contribution of individual EEG channels on MODMA using AlexNet, identifying which channels carry the strongest depression-discriminative signal. Provides channel-level importance rankings useful for electrode reduction and clinical deployment.
- **EEG Preprocessing:**
  - **128-channel resting-state EEG:** 3rd-order Butterworth bandpass filters produce ten 4-Hz-wide sub-bands (4–8, 8–12, 12–16, 16–20, 20–24, 24–28, 28–32, 32–36, 36–40, 40–44 Hz); each EEG channel is trained independently through an AlexNet-style CNN; per-channel classification contribution is ranked, identifying channels 13, 17, 28, 40, 46, 66, 69 as most correlated with depression.
  - **128-channel dot-probe/ERP EEG:** Not used by this paper.
  - **3-channel wearable resting EEG:** Not used by this paper.

### B8. An End-to-End Depression Recognition Method Based on EEGNet
- **Authors:** Liu, B., Chang, H., Peng, K., Wang, X.
- **Year:** 2022
- **Venue:** Frontiers in Psychiatry
- **Link:** [https://www.frontiersin.org/journals/psychiatry/articles/10.3389/fpsyt.2022.864393/full](https://www.frontiersin.org/journals/psychiatry/articles/10.3389/fpsyt.2022.864393/full)
- **Modality:** 128-channel EEG (ERP/dot-probe task)
- **Summary:** Applies EEGNet (a compact CNN) to event-related potential data from MODMA's dot-probe attentional bias task, one of the few papers to use the ERP data rather than resting-state. Finds that happy-neutral face pairs yield the best MDD vs. HC classification performance, suggesting differential attentional processing in depression.
- **EEG Preprocessing:**
  - **128-channel resting-state EEG:** Not used by this paper.
  - **128-channel dot-probe/ERP EEG:** Hamming-windowed sinc FIR 0.3–100 Hz bandpass with 50 Hz line-noise removal; sampling rate 250 Hz; continuous EEG epoched time-locked to emotional–neutral face cues at [-100 ms, 500 ms]; baseline correction over [-100 ms, 0 ms]; separate analyses for happy-neutral, sad-neutral, and fear-neutral cue types (happy-neutral yields the best MDD vs. HC accuracy). Epoched signal fed directly into EEGNet for end-to-end feature extraction and classification.
  - **3-channel wearable resting EEG:** Not used by this paper.

### B9. EEG Diagnosis of Depression Based on Multi-Channel Data Fusion and Clipping Augmentation and CNN
- **Authors:** Wang, B., Kang, Y., Huo, D., Feng, G., Zhang, J., Li, J.
- **Year:** 2022
- **Venue:** Frontiers in Physiology
- **Link:** [https://www.frontiersin.org/journals/physiology/articles/10.3389/fphys.2022.1029298/full](https://www.frontiersin.org/journals/physiology/articles/10.3389/fphys.2022.1029298/full)
- **Modality:** 3-channel EEG
- **Summary:** Uses multi-channel data fusion with clipping-based data augmentation (8× expansion) and a CNN on MODMA's 3-channel wearable EEG. Reports 99.36% accuracy, though the heavy augmentation on a small dataset raises questions about generalizability.
- **EEG Preprocessing:**
  - **128-channel resting-state EEG:** Not used by this paper.
  - **128-channel dot-probe/ERP EEG:** Not used by this paper.
  - **3-channel wearable resting EEG:** Three 1-D wearable channels fused into 2-D image representations via two methods — separate-channels layout in a single image, and RGB-channel synthesis of the three signals; clipping-based data augmentation expands the set ~8×; a classical VGG-style CNN (13 conv + 3 FC layers) performs classification.

### B10. Few-Electrode EEG from Wearable Devices Using Domain Adaptation for Depression Detection
- **Authors:** Wu, W., Ma, L., Lian, B., Cai, W., Zhao, X.
- **Year:** 2022
- **Venue:** Biosensors (MDPI)
- **Link:** [https://www.mdpi.com/2079-6374/12/12/1087](https://www.mdpi.com/2079-6374/12/12/1087)
- **Modality:** 128-channel → 3-channel EEG (domain adaptation)
- **Summary:** Proposes domain adaptation from MODMA's 128-channel EEG to 3-channel wearable EEG, aiming to transfer knowledge from high-density to low-density recordings. Demonstrates that domain adaptation can bridge the gap between clinical-grade and consumer-grade EEG for depression screening.
- **EEG Preprocessing:**
  - **128-channel resting-state EEG:** 128-channel resting-state EEG used as the source domain for transfer learning; acquisition follows the MODMA baseline described in the dataset descriptor. Paper-specific additional preprocessing beyond that baseline is not detailed in accessible portions.
  - **128-channel dot-probe/ERP EEG:** Not used by this paper.
  - **3-channel wearable resting EEG:** Target domain — Fp1/Fpz/Fp2 signals converted to images via two transformations (three-channels-in-one-image layout and RGB synthesis); a domain-adaptation network transfers features learned on the 128-channel source to the 3-channel target to mitigate subject-level distribution shift.

### B11. Automated Accurate Detection of Depression Using Twin Pascal's Triangles Lattice Pattern with EEG Signals
- **Authors:** Tasci, G., et al.
- **Year:** 2023
- **Venue:** Knowledge-Based Systems
- **Link:** [https://www.sciencedirect.com/science/article/abs/pii/S0950705122012862](https://www.sciencedirect.com/science/article/abs/pii/S0950705122012862)
- **Modality:** 128-channel EEG
- **Summary:** Introduces a novel handcrafted feature extraction method (Twin Pascal's Triangles Lattice Pattern, TPTLP) applied to MODMA's 128-channel EEG. Achieves 83.96% accuracy with LOSO cross-validation, a more realistic evaluation than sample-level splitting.
- **EEG Preprocessing:**
  - **128-channel resting-state EEG:** Multilevel discrete wavelet transform (DWT) with Daubechies-4 mother wavelet decomposes the signal into 8 sub-bands; the Twin Pascal's Triangles Lattice Pattern (TPTLP) handcrafted textural feature extractor is applied to each sub-band; neighborhood component analysis selects the most discriminative features; k-NN performs classification, evaluated with LOSO cross-validation. Specific bandpass filter cutoffs and artifact-removal details are not reported.
  - **128-channel dot-probe/ERP EEG:** Not used by this paper.
  - **3-channel wearable resting EEG:** Not used by this paper.

### B12. EEG-Based High-Performance Depression State Recognition
- **Authors:** Wang, Z., Hu, C., Liu, W., Zhou, X., Zhao, X.
- **Year:** 2023
- **Venue:** Frontiers in Neuroscience
- **Link:** [https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2023.1301214/full](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2023.1301214/full)
- **Modality:** 128-channel EEG
- **Summary:** Proposes a W-GCN-GRU model combining weighted graph convolutional networks with gated recurrent units for temporal modeling of MODMA resting-state EEG. Achieves 94.72% accuracy, highlighting the benefit of combining spatial (graph) and temporal (recurrent) architectures.
- **EEG Preprocessing:**
  - **128-channel resting-state EEG:** MATLAB/EEGLAB-based artifact removal; per-channel 23-dimensional feature set spanning time domain, frequency domain, and nonlinear descriptors; Spearman rank-correlation reduction to 6 sensitive features; AUC-weighted feature fusion; W-GCN-GRU architecture combines a weighted graph convolutional network with a gated recurrent unit for spatial-plus-temporal modeling. Specific bandpass cutoffs and epoch parameters are not reported in the accessible portions of the paper.
  - **128-channel dot-probe/ERP EEG:** Not used by this paper.
  - **3-channel wearable resting EEG:** Not used by this paper.

### B13. Electroencephalography-Based Depression Detection Using Multiple Machine Learning Techniques
- **Authors:** Ksibi, A., Zakariah, M., Menzli, L.J., Saidani, O., Almuqren, L., Hanafieh, R.A.M.
- **Year:** 2023
- **Venue:** Diagnostics (MDPI)
- **Link:** [https://www.mdpi.com/2075-4418/13/10/1779](https://www.mdpi.com/2075-4418/13/10/1779)
- **Modality:** 128-channel EEG
- **Summary:** Systematically compares multiple ML and DL techniques (SVM, RF, CNN, LSTM) across different EEG frequency bands on MODMA for binary depression classification. Provides a useful benchmark comparison showing that multiband analysis generally outperforms single-band approaches.
- **EEG Preprocessing:**
  - **128-channel resting-state EEG:** Multiband analysis across standard EEG bands feeds a comparison of SVM, Random Forest, CNN, and LSTM classifiers (CNN achieved ~97% at 25 epochs). Specific bandpass cutoffs, artifact-removal method, and epoch length are not publicly available in the accessible open-access text.
  - **128-channel dot-probe/ERP EEG:** Not used by this paper.
  - **3-channel wearable resting EEG:** Not used by this paper.

### B14. A Novel EEG-Based Graph Convolution Network for Depression Detection: Incorporating Secondary Subject Partitioning and Attention Mechanism (SSPA-GCN)
- **Authors:** Zhang, B., et al.
- **Year:** 2023
- **Venue:** Expert Systems with Applications
- **Link:** [https://www.sciencedirect.com/science/article/abs/pii/S0957417423028580](https://www.sciencedirect.com/science/article/abs/pii/S0957417423028580)
- **Modality:** 128-channel EEG
- **Summary:** Introduces secondary subject partitioning to handle inter-subject variability and adds an attention mechanism to weight the importance of different EEG channels and features in a GCN framework. Achieves the highest classification accuracy of 92.8% on MODMA with strong cross-subject generalization.
- **EEG Preprocessing:**
  - **128-channel resting-state EEG:** First and last 10 s of each recording discarded; 0.3–30 Hz FIR bandpass; window-wise baseline subtraction; ICA for ocular and muscular artifact removal; secondary subject partitioning to reduce inter-subject variability; a two-dimensional attention matrix at the GCN input automatically weights channels and features.
  - **128-channel dot-probe/ERP EEG:** Not used by this paper.
  - **3-channel wearable resting EEG:** Not used by this paper.

### B15. EEG-Based Subject-Independent Depression Detection Using Dynamic Convolution and Feature Adaptation (DCAAN)
- **Authors:** Jiang, W., et al.
- **Year:** 2023
- **Venue:** Advances in Swarm Intelligence (Springer)
- **Link:** [https://link.springer.com/chapter/10.1007/978-3-031-36625-3_22](https://link.springer.com/chapter/10.1007/978-3-031-36625-3_22)
- **Modality:** 128-channel EEG
- **Summary:** Combines dynamic convolution with adversarial domain adaptation for subject-independent depression detection on MODMA. Achieves 86.85% accuracy, demonstrating the challenge and promise of cross-subject generalization in small clinical EEG datasets.
- **EEG Preprocessing:**
  - **128-channel resting-state EEG:** 5-min eyes-closed signal at 250 Hz; 50 Hz trap/notch filter for line-noise removal; LMS adaptive-noise-cancellation for blink-artifact removal; decomposition into canonical bands; differential entropy (DE) features computed over fixed time intervals; a DE sequence is selected per subject; dynamic convolution and adversarial domain adaptation train a subject-independent classifier.
  - **128-channel dot-probe/ERP EEG:** Not used by this paper.
  - **3-channel wearable resting EEG:** Not used by this paper.

### B16. Depression Screening Using Hybrid Neural Network
- **Authors:** Zhang, J., Xu, B., Yin, H.
- **Year:** 2023
- **Venue:** Multimedia Tools and Applications (Springer)
- **Link:** [https://link.springer.com/article/10.1007/s11042-023-14860-w](https://link.springer.com/article/10.1007/s11042-023-14860-w)
- **Modality:** 3-channel EEG
- **Summary:** Proposes a hybrid neural network combining CNN and RNN components for depression screening using MODMA's 3-channel wearable EEG data. Demonstrates that even minimal-electrode setups can achieve competitive depression detection performance with appropriate model design.
- **EEG Preprocessing:**
  - **128-channel resting-state EEG:** 128-channel resting-state EEG used via a hybrid CNN-LSTM architecture for temporal and sequence learning. Detailed filter/artifact parameters not fully reported in accessible portions.
  - **128-channel dot-probe/ERP EEG:** Not used by this paper.
  - **3-channel wearable resting EEG:** Frontal (Fp1, Fp2, Fpz) resting-state EEG used; a lightweight Conv1D + spectral-statistical descriptor fusion model combines raw resting-state windows with per-channel spectral-statistical features; predictions aggregated at the window level. Specific filter cutoffs and epoch length not fully reported.

### B17. Graph-Based EEG Approach for Depression Prediction: Integrating Time-Frequency Complexity and Spatial Topology
- **Authors:** Liu, W., Jia, K., Wang, Z.
- **Year:** 2024
- **Venue:** Frontiers in Neuroscience
- **Link:** [https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2024.1367212/full](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2024.1367212/full)
- **Modality:** 128-channel EEG
- **Summary:** Merges EEG time-frequency complexity features with electrode spatial topology using graph convolutional networks, deriving soft labels from PHQ-9 and GAD-7 scores (the only MODMA paper to incorporate these clinical scales into the model). Achieves 98.30% accuracy; notably the sole paper to reference both PHQ-9 and GAD-7 in its methodology, though for soft-label derivation rather than continuous regression.
- **EEG Preprocessing:**
  - **128-channel resting-state EEG:** Differential Entropy (DE) computed per frequency band and z-score normalized; BiLSTM extracts temporal dynamics of the DE sequence; Pearson correlation between channels forms the spatial adjacency; the resulting graph feeds a GCN stack. Specific bandpass cutoffs and artifact-removal pipeline are not explicitly detailed in accessible portions.
  - **128-channel dot-probe/ERP EEG:** Not used by this paper.
  - **3-channel wearable resting EEG:** Not used by this paper.

### B18. EDT: An EEG-Based Attention Model for Feature Learning and Depression Recognition
- **Authors:** Ying, M., Shao, X., Zhu, J., Zhao, Q., Li, X., Hu, B.
- **Year:** 2024
- **Venue:** Biomedical Signal Processing and Control
- **Link:** [https://www.sciencedirect.com/science/article/abs/pii/S1746809424002404](https://www.sciencedirect.com/science/article/abs/pii/S1746809424002404)
- **Modality:** 128-channel EEG
- **Summary:** Introduces a transformer-inspired attention architecture (EDT) that learns discriminative EEG features for depression recognition without relying on handcrafted feature engineering. Achieves 94.0% accuracy on MODMA, demonstrating that self-attention mechanisms can effectively capture depression-relevant EEG patterns.
- **EEG Preprocessing:**
  - **128-channel resting-state EEG:** EEG decomposed into delta, theta, alpha, beta, and gamma bands; frequency-domain and spatial-domain feature maps constructed; a transformer-style attention architecture (EDT) combined with convolutional layers learns discriminative features end-to-end without handcrafted feature engineering. Specific filter cutoffs and artifact-removal details are not publicly available.
  - **128-channel dot-probe/ERP EEG:** Not used by this paper.
  - **3-channel wearable resting EEG:** Not used by this paper.

### B19. Depression Detection Based on Temporal-Spatial-Frequency Feature Fusion of EEG
- **Authors:** (see ScienceDirect page for full list)
- **Year:** 2024
- **Venue:** Biomedical Signal Processing and Control
- **Link:** [https://www.sciencedirect.com/science/article/abs/pii/S1746809424009881](https://www.sciencedirect.com/science/article/abs/pii/S1746809424009881)
- **Modality:** 128-channel EEG
- **Summary:** Fuses temporal, spatial, and frequency-domain EEG features for depression detection, capturing complementary information across all three dimensions. Achieves 97.24% accuracy on MODMA with a structured feature fusion pipeline.
- **EEG Preprocessing:**
  - **128-channel resting-state EEG:** EEG split into delta/theta/alpha/beta/gamma sub-bands; channel selection driven by frequency-domain weighting; a multiscale convolutional attention module extracts spatial features while a temporal trend-aware self-attention module captures long-range temporal dependencies; channel weights are automatically adjusted. Specific bandpass cutoffs and artifact-removal details are not publicly available.
  - **128-channel dot-probe/ERP EEG:** Not used by this paper.
  - **3-channel wearable resting EEG:** Not used by this paper.

### B20. A Machine Learning Based Depression Screening Framework Using Temporal Domain Features of EEG Signals
- **Authors:** Khan, S., Umar Saeed, S.M., Frnda, J., Arsalan, A., Amin, R., Gantassi, R., et al.
- **Year:** 2024
- **Venue:** PLOS ONE
- **Link:** [https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0299127](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0299127)
- **Modality:** 3-channel EEG
- **Summary:** Extracts 12 temporal domain features from MODMA's 3-channel wearable EEG and evaluates multiple ML classifiers, with Best-First Tree achieving 96.36% accuracy. Provides a lightweight, interpretable approach suitable for resource-constrained clinical settings.
- **EEG Preprocessing:**
  - **128-channel resting-state EEG:** Not used by this paper.
  - **128-channel dot-probe/ERP EEG:** Not used by this paper.
  - **3-channel wearable resting EEG:** Resting-state EEG from frontal electrodes Fp1, Fpz, Fp2; non-overlapping 10-s windows; 12 selected temporal-domain handcrafted features (minimum/maximum amplitude, mean absolute value of the first and second differences, etc.); classifiers include Best-First Tree (top performer at 96.36%), k-NN, and AdaBoost.

### B21. EEG Based Generative Depression Discriminator
- **Authors:** Mao, Z., Wu, H., Tan, Y., Jin, Y.
- **Year:** 2024
- **Venue:** arXiv
- **Link:** [https://arxiv.org/abs/2402.09421](https://arxiv.org/abs/2402.09421)
- **Modality:** 128-channel EEG
- **Summary:** Trains two separate generative networks—one modeling the EEG distribution of MDD subjects and one for HC—and classifies new subjects based on which generator better reconstructs their data. Achieves 92.30% accuracy on MODMA, offering a novel generative approach compared to the predominant discriminative methods.
- **EEG Preprocessing:**
  - **128-channel resting-state EEG:** Bandpass filter 4–14 Hz; FFT-based frequency-domain analysis complemented by wavelet transform; two class-conditional generative networks — one modeling MDD, one HC — are trained to reconstruct electrode signals, and new subjects are classified by which generator reconstructs them better.
  - **128-channel dot-probe/ERP EEG:** Not used by this paper.
  - **3-channel wearable resting EEG:** Not used by this paper.

### B22. A Hybrid Graph Neural Network for Enhanced EEG-Based Depression Detection (HybGNN)
- **Authors:** Wang, Y., Zheng, W., Li, Y., Yang, H.
- **Year:** 2024
- **Venue:** arXiv
- **Link:** [https://arxiv.org/abs/2410.18103](https://arxiv.org/abs/2410.18103)
- **Modality:** 128-channel EEG
- **Summary:** Proposes HybGNN with two branches—a Common GNN (CGNN) capturing shared depression patterns and an Individualized GNN (IGNN) addressing subject-specific variability—with graph pooling/unpooling modules. Achieves 95.42% on MODMA and 93.50% on HUSM, balancing population-level and individual-level feature extraction.
- **EEG Preprocessing:**
  - **128-channel resting-state EEG:** Channel count reduced from 128 to 19 electrodes (standard 10-20 subset) for consistency across datasets; 4-s windows with 75% overlap; 1-D CNN extracts temporal features from raw windowed signal; a dual-branch GNN is then applied — a Common-GNN (CGNN) with a fixed adjacency matrix capturing shared depression patterns and an Individualized-GNN (IGNN) with an adaptive adjacency matrix for subject-specific variability — with graph pooling/unpooling for hierarchical aggregation.
  - **128-channel dot-probe/ERP EEG:** Not used by this paper.
  - **3-channel wearable resting EEG:** Not used by this paper.

### B23. SAD-TIME: A Spatiotemporal-Fused Network for Depression Detection
- **Authors:** Xu, C.-Y., Wang, H.-G., Zhang, L., Zhang, Y.-H., Hou, H.-R., Meng, Q.-H.
- **Year:** 2024
- **Venue:** arXiv
- **Link:** [https://arxiv.org/pdf/2411.08521](https://arxiv.org/pdf/2411.08521)
- **Modality:** 128-channel EEG
- **Summary:** Introduces a multi-scale depth-wise CNN combined with a domain adversarial learner for cross-subject EEG depression detection on MODMA. Achieves 94.00% accuracy by fusing spatiotemporal EEG features while mitigating domain shift between subjects.
- **EEG Preprocessing:**
  - **128-channel resting-state EEG:** A multi-scale depth-wise 1-D CNN operates directly on raw signals to extract per-channel features; a common-feature extractor preserves each channel's unique information; time-interval embedding provides temporal positional encoding; a spatial sector models electrode-position structure while a temporal sector captures time-window continuity; domain-adversarial learning improves cross-subject generalization. Specific bandpass cutoffs and artifact-removal pipeline are not detailed in accessible portions.
  - **128-channel dot-probe/ERP EEG:** Not used by this paper.
  - **3-channel wearable resting EEG:** Not used by this paper.

### B24. Exploring Large-Scale Language Models to Evaluate EEG-Based Multimodal Data for Mental Health (MultiEEG-GPT)
- **Authors:** Hu, Y., Zhang, S., Dang, T., Jia, H., Salim, F.D., Hu, W., Quigley, A.J.
- **Year:** 2024
- **Venue:** ACM UbiComp (Companion)
- **Link:** [https://arxiv.org/abs/2408.07313](https://arxiv.org/abs/2408.07313)
- **Modality:** 128-channel EEG (+ facial expression or audio in multimodal variants)
- **Summary:** Pioneers the use of GPT-4o with zero-shot and few-shot prompting to evaluate EEG-based multimodal data for depression and emotion recognition on MODMA and other datasets. Finds that multimodal LLM approaches outperform single-modality models, with few-shot prompting improving over zero-shot by 5–8%.
- **EEG Preprocessing:**
  - **128-channel resting-state EEG:** EEG converted into topology-map images that are fed directly to GPT-4o through multimodal zero-shot and few-shot prompting; classical signal-processing steps (filtering, artifact removal, re-referencing, epoching) are largely bypassed in favor of image-based LLM evaluation. Multimodal variants combine the EEG topology images with facial-expression or audio inputs.
  - **128-channel dot-probe/ERP EEG:** Not used by this paper.
  - **3-channel wearable resting EEG:** Not used by this paper.

### B25. A Multi-Stage Hemisphere Asymmetry Fusion Network (MSHAF-Net)
- **Authors:** (see ScienceDirect page for full list)
- **Year:** 2025
- **Venue:** Information Fusion (ScienceDirect)
- **Link:** [https://www.sciencedirect.com/science/article/abs/pii/S1566253525004154](https://www.sciencedirect.com/science/article/abs/pii/S1566253525004154)
- **Modality:** 128-channel EEG
- **Summary:** Explicitly models hemisphere asymmetry—a known EEG biomarker of depression—through a multi-stage fusion network that separately processes left and right hemisphere EEG before integrating them. Achieves 98.94% accuracy on MODMA, leveraging the neurobiological finding that right-hemisphere EEG signals are more distinctive in depression.
- **EEG Preprocessing:**
  - **128-channel resting-state EEG:** Left-hemisphere and right-hemisphere EEG routed through separate Adaptive Spatiotemporal Graph Convolution (ASTGCN) branches with learnable graph structure; hemisphere features fused via cross-attention; Maximum Mean Discrepancy (MMD) domain-adaptation loss aligns cross-subject distributions. Specific bandpass cutoffs, artifact-removal method, and epoch length are not publicly available (paywalled).
  - **128-channel dot-probe/ERP EEG:** Not used by this paper.
  - **3-channel wearable resting EEG:** Not used by this paper.

### B26. Attention-Based Multi-Scale Convolution and Conformer for EEG-Based Depression Detection (AMCCBDep)
- **Authors:** Wan, Y., et al.
- **Year:** 2025
- **Venue:** Frontiers in Psychiatry
- **Link:** [https://www.frontiersin.org/journals/psychiatry/articles/10.3389/fpsyt.2025.1584474/full](https://www.frontiersin.org/journals/psychiatry/articles/10.3389/fpsyt.2025.1584474/full)
- **Modality:** 128-channel EEG
- **Summary:** Combines multi-scale parallel convolutions with ECA attention and a Conformer module (capturing both local and global temporal dependencies) plus BiGRU for end-to-end depression recognition. Achieves 98.68% ± 0.45% accuracy on MODMA's 128-channel resting-state EEG.
- **EEG Preprocessing:**
  - **128-channel resting-state EEG:** Minimal classical preprocessing (bandpass filtering + normalization) feeds an Attention-based Multi-scale Parallel Convolution (AMPC) block that uses depthwise-separable convolutions, followed by ECA channel attention, a Conformer module capturing both local and global temporal dependencies, and a BiGRU for bidirectional sequence modeling. Specific bandpass cutoffs and artifact-removal details are not explicitly stated.
  - **128-channel dot-probe/ERP EEG:** Not used by this paper.
  - **3-channel wearable resting EEG:** Not used by this paper.

### B27. Multichannel Convolutional Transformer for Detecting Mental Disorders Using EEG Records
- **Authors:** Dia, M., Khodabandelou, G., Anwar, S.M., Othmani, A.
- **Year:** 2025
- **Venue:** Scientific Reports (Nature)
- **Link:** [https://www.nature.com/articles/s41598-025-98264-w](https://www.nature.com/articles/s41598-025-98264-w)
- **Modality:** 128-channel EEG
- **Summary:** Integrates CNN and transformer architectures in a multichannel framework for mental disorder detection, evaluated on MODMA among other datasets. Achieves 89.84% accuracy on MODMA, demonstrating the general applicability of convolutional transformers beyond single-disorder settings.
- **EEG Preprocessing:**
  - **128-channel resting-state EEG:** Common Spatial Pattern (CSP) filtering; Signal Space Projection (SSP); wavelet denoising; Continuous Wavelet Transform (CWT) converts time-domain EEG into a time-frequency representation; the resulting tokens feed convolutional layers and then a transformer encoder that captures long-range temporal dependencies across all channels; training uses an entropy-based loss.
  - **128-channel dot-probe/ERP EEG:** Not used by this paper.
  - **3-channel wearable resting EEG:** Not used by this paper.

### B28. ELPG-DTFS: Prior-Guided Adaptive Time-Frequency Graph Neural Network for EEG Depression Diagnosis
- **Authors:** Qiu, J., Liang, J., Fan, X., Zhang, M., He, Z.
- **Year:** 2025
- **Venue:** arXiv
- **Link:** [https://arxiv.org/abs/2509.24860](https://arxiv.org/abs/2509.24860)
- **Modality:** 128-channel EEG
- **Summary:** Introduces channel-band attention with a residual knowledge-graph module that incorporates neuroscientific priors (e.g., known asymmetry patterns) into the graph construction for EEG depression diagnosis. Achieves 97.63% accuracy and 97.33% F1-score on MODMA.
- **EEG Preprocessing:**
  - **128-channel resting-state EEG:** Standard MODMA preprocessing: 0.3–30 Hz FIR bandpass; window-wise baseline subtraction; ICA for ocular and muscular artifact removal. Downstream modeling applies channel-band attention with cross-band mutual-information terms, a learnable adjacency for dynamic functional-connectivity graphs, and a residual knowledge-graph pathway that injects neuroscience priors (e.g. known hemispheric asymmetry patterns).
  - **128-channel dot-probe/ERP EEG:** Not used by this paper.
  - **3-channel wearable resting EEG:** Not used by this paper.

### B29. TWD-DepNet: A Deep Network Enhanced by Three-Way Decisions for EEG-Based Depression Detection
- **Authors:** Shi, Y. & Yan, Z.
- **Year:** 2025
- **Venue:** Journal of King Saud University (Springer)
- **Link:** [https://link.springer.com/article/10.1007/s44443-025-00196-y](https://link.springer.com/article/10.1007/s44443-025-00196-y)
- **Modality:** 128-channel EEG
- **Summary:** Incorporates three-way decision theory into a deep network, allowing the model to defer ambiguous cases rather than forcing a binary classification, improving reliability. Outperforms all baseline methods on MODMA across accuracy, precision, F1-score, sensitivity, and specificity.
- **EEG Preprocessing:**
  - **128-channel resting-state EEG:** 0.5–70 Hz FIR bandpass; ICA-based denoising with ICLabel-driven component removal (blinks, muscle, line noise); common-average reference; signal segmented into 25-s epochs with 5-s overlap; 19-channel subset (Fp1/2, F3/4, C3/4, P3/4, O1/2, F7/8, T3/4, T5/6, Cz, Fz, Pz); z-score normalization; convolutional backbone with depthwise-separable layers; a three-way-decision head defers ambiguous samples instead of forcing a binary verdict.
  - **128-channel dot-probe/ERP EEG:** Same preprocessing pipeline as the resting-state branch applied to the dot-probe/ERP recordings.
  - **3-channel wearable resting EEG:** Not used by this paper.

### B30. A Novel Demographic Indicator Fusion Network (DIFNet)
- **Authors:** Wang, C., Zhou, Q., Li, M., Li, J., Zhao, J.
- **Year:** 2025
- **Venue:** Sensors (MDPI)
- **Link:** [https://www.mdpi.com/1424-8220/25/21/6549](https://www.mdpi.com/1424-8220/25/21/6549)
- **Modality:** 128-channel EEG + demographics
- **Summary:** Dynamically fuses EEG features with demographic indicators (age, sex, years of education) using a gated fusion mechanism for depression detection on MODMA. Achieves 99.66% accuracy, the highest reported on MODMA, though the use of demographics alongside EEG in a 53-subject dataset warrants caution about overfitting.
- **EEG Preprocessing:**
  - **128-channel resting-state EEG:** 250 Hz sampling with Cz reference; 1–40 Hz FIR bandpass + 50 Hz notch; 16-channel electrode subset retained for efficiency; 4-s epochs with 75% overlap; Autoreject automatic artifact suppression; z-score normalization; analyses repeated per frequency band (δ/θ/α/β/γ, with beta yielding the highest accuracy); demographic indicators (age, sex, years of education) fused with EEG embeddings via a gated fusion mechanism.
  - **128-channel dot-probe/ERP EEG:** Not used by this paper.
  - **3-channel wearable resting EEG:** Not used by this paper.

### B31. Decentralized EEG-Based Detection of MDD via Transformer Architectures and Split Learning
- **Authors:** Ahmad, U., et al.
- **Year:** 2025
- **Venue:** Frontiers in Computational Neuroscience
- **Link:** [https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2025.1569828/full](https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2025.1569828/full)
- **Modality:** 128-channel EEG
- **Summary:** Uses split learning to distribute transformer-based depression detection across multiple nodes for privacy-preserving computation on MODMA EEG data. Achieves 99% accuracy with Transformer+RF, demonstrating that decentralized architectures need not sacrifice performance.
- **EEG Preprocessing:**
  - **128-channel resting-state EEG:** 0.5–70 Hz bandpass + 50 Hz notch; resampled to 256 Hz; infinity reference (linked-mastoids / average reference); transformer and autoencoder feature-extraction backbones; split-learning distributes the model across three clients for privacy-preserving training (best single-client 96.23%, centralized Transformer+RF 99%).
  - **128-channel dot-probe/ERP EEG:** Not used by this paper.
  - **3-channel wearable resting EEG:** Not used by this paper.

### B32. A Novel Multichannel EEG Analysis Method Using Multiscale Graph Convolution and Cross Attention Transformer (MGFormer)
- **Authors:** Chen, X., Liu, Y., Liu, Z., Liu, Y., Coatrieux, J.L., Shu, H.
- **Year:** 2025
- **Venue:** Springer (ICONIP 2024 proceedings)
- **Link:** [https://link.springer.com/chapter/10.1007/978-981-95-0033-8_28](https://link.springer.com/chapter/10.1007/978-981-95-0033-8_28)
- **Modality:** 128-channel EEG
- **Summary:** Combines multiscale graph convolution (to capture channel interactions at different propagation depths) with cross-attention transformers for dynamic modality fusion in EEG depression detection. Achieves 91.67% accuracy on MODMA, with the cross-attention mechanism enabling flexible information exchange across graph scales.
- **EEG Preprocessing:**
  - **128-channel resting-state EEG:** Multiscale graph convolution builds channel adjacency from inter-channel correlations and captures interactions at different propagation depths; a cross-attention transformer handles dynamic fusion across graph scales. Specific bandpass cutoffs, sampling rate, artifact-removal method, and epoch length are not publicly available (paywalled ICONIP proceedings).
  - **128-channel dot-probe/ERP EEG:** Not used by this paper.
  - **3-channel wearable resting EEG:** Not used by this paper.

### B33. A Novel EEG-Based Depression Detection Model Based on AKRC-C and Random Forest
- **Authors:** Kan, J., Tong, W., Chen, K., Wu, B., Wang, B.
- **Year:** 2025
- **Venue:** Springer (conference proceedings)
- **Link:** [https://link.springer.com/chapter/10.1007/978-981-96-5314-0_39](https://link.springer.com/chapter/10.1007/978-981-96-5314-0_39)
- **Modality:** 128-channel EEG
- **Summary:** Proposes an Adaptive Kernel Ridge Classification with Constraint (AKRC-C) feature extraction method paired with Random Forest classification for EEG depression detection. Achieves 95.03% accuracy on MODMA with an interpretable, relatively simple pipeline.
- **EEG Preprocessing:**
  - **128-channel resting-state EEG:** ~0.1–30 Hz bandpass plus 48–52 Hz notch for 50 Hz line-noise removal; ICA to reject EOG, EMG, and ECG artifacts; signal segmented into 3-s epochs (a 1-s full-band variant is also evaluated); Phase Lag Index (PLI) computed to form functional-connectivity matrices; the Altered Kendall's Rank Correlation with Convergence (AKRC-C) feature-selection scheme picks discriminative PLI entries; Random Forest performs classification.
  - **128-channel dot-probe/ERP EEG:** Not used by this paper.
  - **3-channel wearable resting EEG:** Not used by this paper.

---

## Category C: Audio/Speech-Only Papers

### C1. A Novel Study for Depression Detecting Using Audio Signals Based on Graph Neural Network
- **Authors:** Sun, C., Jiang, M., Gao, L., Xin, Y., Dong, Y.
- **Year:** 2023
- **Venue:** Biomedical Signal Processing and Control
- **Link:** [https://www.sciencedirect.com/science/article/abs/pii/S1746809423011084](https://www.sciencedirect.com/science/article/abs/pii/S1746809423011084)
- **Modality:** Audio
- **Summary:** Applies GRU with sequential graph neural networks to audio signals from MODMA for binary depression classification, modeling temporal dependencies in speech as a graph structure. One of the earlier papers to use graph-based methods specifically on MODMA's audio modality.
- **Audio Preprocessing:** Frame-level **MFCC** features extracted from raw wav files; GRU processes the MFCC time series to produce temporal embeddings which are then structured as a graph for the GNN. Specific MFCC parameters (number of coefficients, window size, hop length) not publicly available (paywalled).

### C2. A Deep Learning Model for Depression Detection Based on MFCC and CNN Generated Spectrogram Features
- **Authors:** Das, A.K. & Naskar, R.
- **Year:** 2024
- **Venue:** Biomedical Signal Processing and Control
- **Link:** [https://www.sciencedirect.com/science/article/abs/pii/S1746809423013319](https://www.sciencedirect.com/science/article/abs/pii/S1746809423013319)
- **Modality:** Audio
- **Summary:** Combines MFCC features with CNN-generated spectrogram features from MODMA audio using a novel CNN with optimized residual blocks and "glorot uniform" kernel initialization. Achieves over 90% accuracy on MODMA, providing a multimodal audio feature fusion approach.
- **Audio Preprocessing:** Two parallel feature streams: (1) **MFCC** features extracted directly from raw audio; (2) **Mel-spectrogram** images generated and fed into a residual-based CNN (Spectro_CNN) for learned spectrogram features. Both streams are fused for classification. Specific extraction parameters (number of MFCCs, mel bands, FFT size) not publicly available (paywalled).

### C3. Multilevel Hybrid Handcrafted Feature Extraction Based Depression Recognition Method Using Speech
- **Authors:** Taşcı, B.
- **Year:** 2024
- **Venue:** Journal of Affective Disorders
- **Link:** [https://www.sciencedirect.com/science/article/abs/pii/S0165032724012151](https://www.sciencedirect.com/science/article/abs/pii/S0165032724012151)
- **Modality:** Audio
- **Summary:** Applies multilevel discrete wavelet transform (DWT) with iterative feature selection to extract handcrafted speech features for depression detection on MODMA. Achieves 94.63% accuracy with low computational complexity, demonstrating that handcrafted features remain competitive with deep learning.
- **Audio Preprocessing:** **Multilevel DWT** applied to raw wav signals to produce wavelet subbands at multiple decomposition levels. From each subband: (1) **1D Local Binary Pattern (1D-LBP)** textural features and (2) **20 statistical moments** are extracted. Features from all levels are concatenated, then reduced via **Iterative Neighborhood Component Analysis (INCA)** for feature selection. No spectrograms or MFCCs — purely time-domain wavelet handcrafted features.

### C4. Vision Transformer for Audio-Based Depression Detection on Multi-Lingual Audio Data
- **Authors:** Pratiwi, M., Sanjaya, S.A.
- **Year:** 2024
- **Venue:** ACM DMIP 2024
- **Link:** [https://dl.acm.org/doi/10.1145/3705927.3705934](https://dl.acm.org/doi/10.1145/3705927.3705934)
- **Modality:** Audio
- **Summary:** Applies Vision Transformer (ViT) and variants (DeiT, Swin Transformer) to audio spectrograms from a merged MODMA + DAIC-WOZ + D-Vlog dataset for cross-lingual depression detection. Achieves over 90% accuracy on both MODMA and DAIC-WOZ, demonstrating transferability of visual-transformer approaches across Mandarin and English speech.
- **Audio Preprocessing:** Raw wav files converted to **mel-spectrogram** images, then fed as 2D image input to Vision Transformer variants (ViT, DeiT, Swin Transformer). Treats depression detection as an image classification task on spectrogram representations. Specific mel-spectrogram parameters (mel bands, FFT size, hop length) not publicly available (paywalled).

### C5. Application of Pre-trained Model-Based Speech Analysis in Depression Detection
- **Authors:** Xu, G., Zhou, C.
- **Year:** 2024
- **Venue:** Scientific Journal of Intelligent Systems Research
- **Link:** [https://bcpublication.org/index.php/SJISR/article/view/7549](https://bcpublication.org/index.php/SJISR/article/view/7549)
- **Modality:** Audio
- **Summary:** Leverages pre-trained speech models (e.g., wav2vec, HuBERT) for depression detection on MODMA's Chinese-language audio, addressing the challenge of limited labeled data through transfer learning. Demonstrates that pre-trained representations can effectively adapt to the Chinese depression detection context despite being trained on predominantly English corpora.
- **Audio Preprocessing:** Raw wav files resampled from 44.1kHz → **16kHz** via FFmpeg, then fed into **Wav2Vec2-large-XLSR-53-Chinese** (Hugging Face). Outputs **1024-dimensional** embeddings per temporal frame, pooled across time. Task-dependent sliding windows: interview questions 5s (2.5s overlap), paragraph reading 10s (5s overlap), picture description 8s (4s overlap), word reading uses the entire file with no segmentation.

### C6. Mixture of Experts for Recognizing Depression from Interview and Reading Tasks
- **Authors:** Ilias, L., Askounis, D.
- **Year:** 2025
- **Venue:** arXiv
- **Link:** [https://arxiv.org/abs/2502.20213](https://arxiv.org/abs/2502.20213)
- **Modality:** Audio
- **Summary:** First study to jointly model both spontaneous (interview) and read speech from MODMA in a single Mixture-of-Experts (MoE) network, addressing the common limitation of using only one speech type. Finds that interview tasks outperform reading tasks for depression detection, with MoE gating enabling input-conditional computation across task types. **Highly relevant to RQ2.**
- **Audio Preprocessing:** Raw wav files converted to 3-channel spectrogram images via **librosa**: Channel 1 = **log-Mel spectrogram** (224 mel bands, hop length 512, Hanning window), Channel 2 = **delta** (Δ, velocity), Channel 3 = **delta-delta** (ΔΔ, acceleration). Images resized to **224×224 pixels**. Fed into pre-trained AlexNet → 768-dim embeddings per file. Interview and reading task embeddings fused via BLOCK tensor decomposition before MoE classification.

### C7. Enhancing Depression Recognition Through a Mixed Expert Model by Integrating Speaker-Related and Emotion-Related Features
- **Authors:** Guo, W., He, Q., Lin, Z., et al.
- **Year:** 2025
- **Venue:** Scientific Reports (Nature)
- **Link:** [https://www.nature.com/articles/s41598-025-88313-9](https://www.nature.com/articles/s41598-025-88313-9)
- **Modality:** Audio
- **Summary:** Disentangles speaker-related features (from a speaker recognition pre-trained TDNN) from emotion-related features (from a speech emotion pre-trained model) and fuses them via a Mixture-of-Experts framework for depression recognition on MODMA. Addresses the often-overlooked confound between speaker identity and emotional content in depression detection.
- **Audio Preprocessing:** Raw wav files converted to **spectrograms**, then processed through two parallel pre-trained feature extractors: (1) **TDNN** pre-trained on speaker recognition → speaker-identity embeddings; (2) **speech emotion recognition model** (pre-trained) → emotion embeddings. Both streams use **ResNet-50** convolutional layers on the spectrogram representation. Outputs fused via MoE gating. Specific spectrogram parameters not detailed in accessible sources.

### C8. Hierarchical Self-Supervised Representation Learning for Depression Detection from Speech
- **Authors:** Li, Y., Chng, E.S., Guan, C.
- **Year:** 2025
- **Venue:** arXiv / Interspeech
- **Link:** [https://arxiv.org/abs/2510.08593](https://arxiv.org/abs/2510.08593)
- **Modality:** Audio
- **Summary:** Proposes a hierarchical adaptive representation encoder that disentangles and re-aligns acoustic and semantic information through asymmetric cross-attention with CTC auxiliary supervision. Achieves state-of-the-art Macro F1 of 0.82 on MODMA and 0.81 on DAIC-WOZ in upper-bound evaluation.
- **Audio Preprocessing:** Raw wav files resampled to **16kHz**, then segmented into **10-second** random segments with amplitude normalization (zero mean, unit variance) and mask-based padding for variable lengths. Fed into **WavLM-Large** (frozen) which extracts hidden representations from all **24 transformer layers** at 20ms frame rate → **1024-dim** per frame per layer. The hierarchical encoder then disentangles acoustic vs. semantic information across layers via asymmetric cross-attention + CTC supervision.

### C9. Enhanced Depression Detection Through Optimally Weighted Spectrogram Feature Fusion
- **Authors:** Das, A.K., Naskar, R.
- **Year:** 2024
- **Venue:** ACM ICCPR 2024
- **Link:** [https://dl.acm.org/doi/10.1145/3704323.3704375](https://dl.acm.org/doi/10.1145/3704323.3704375)
- **Modality:** Audio (+ EEG spectrogram)
- **Summary:** Extracts and optimally weights MFCC and mel-spectrogram features from audio alongside STFT spectrograms from EEG for a combined depression detection approach on MODMA. Focuses on finding the optimal feature weighting strategy for spectrogram-level fusion.
- **Audio Preprocessing:** Three parallel spectrogram-based feature streams: (1) **MFCC** — 13 coefficients, 40 mel-filterbank, Hamming window, 25ms frame, 10ms hop, 1024-point FFT; (2) **Mel-spectrogram** — 40 mel bands, same windowing; (3) additional **STFT spectrogram**. Three fused representations created and **optimally weighted** via "leading goat" simulated annealing algorithm. Per wav file output: 13 MFCCs × T frames + 40 mel bins × T frames + spectrogram, dynamically weighted.

---

## Category D: Multimodal Papers (EEG + Audio/Speech)

### D1. Multimodal Fusion of EEG and Audio Spectrogram for MDD Recognition Using Modified DenseNet121
- **Authors:** Yousufi, M., Damaševičius, R., et al.
- **Year:** 2024
- **Venue:** Brain Sciences (MDPI)
- **Link:** [https://www.mdpi.com/2076-3425/14/10/1018](https://www.mdpi.com/2076-3425/14/10/1018)
- **Modality:** EEG + Audio
- **Summary:** Uses transfer learning with modified DenseNet121 to fuse EEG STFT spectrograms and audio mel-spectrograms from MODMA at the decision level. Achieves 97.53% accuracy with the multimodal fusion, outperforming either unimodal approach alone.
- **EEG Preprocessing:**
  - **128-channel resting-state EEG:** 5-min eyes-closed signal at 250 Hz; EEG converted to spectrograms via Short-Time Fourier Transform (STFT); spectrograms fed as image input into a modified DenseNet121 and fused with audio-derived features. Specific bandpass cutoffs, artifact-removal method, and epoch parameters are not explicitly stated in accessible portions.
  - **128-channel dot-probe/ERP EEG:** Not used by this paper.
  - **3-channel wearable resting EEG:** Not used by this paper.

### D2. DPD (DePression Detection) Net: A Deep Neural Network for Multimodal Depression Detection
- **Authors:** He, M., et al.
- **Year:** 2024
- **Venue:** Health Information Science and Systems (Springer)
- **Link:** [https://link.springer.com/article/10.1007/s13755-024-00311-9](https://link.springer.com/article/10.1007/s13755-024-00311-9)
- **Modality:** EEG + Audio
- **Summary:** Proposes a Graph Neural Network-enhanced Transformer that processes EEG through graph convolutions and audio through convolutional encoders before fusing them with cross-modal attention. Demonstrates that structured multimodal integration outperforms naive concatenation on MODMA.
- **EEG Preprocessing:**
  - **128-channel resting-state EEG:** 128-channel resting-state EEG used in the EEG branch (DPD-E Net variant); a Graph Neural Network extracts spatial structural features from EEG, which are then fused with audio/text branches. Specific bandpass filter, artifact-removal method, and epoch parameters are not publicly available in the accessible portions of the paper.
  - **128-channel dot-probe/ERP EEG:** Not used by this paper.
  - **3-channel wearable resting EEG:** Not used by this paper.

### D3. An Adaptive Multi-Graph Neural Network with Multimodal Feature Fusion Learning for MDD Detection (EMO-GCN)
- **Authors:** Xing, T., Dou, Y., Chen, X., Zhou, J., Xie, X., Peng, S.
- **Year:** 2024
- **Venue:** Scientific Reports (Nature)
- **Link:** [https://www.nature.com/articles/s41598-024-79981-0](https://www.nature.com/articles/s41598-024-79981-0)
- **Modality:** EEG + Audio
- **Summary:** Constructs separate graph neural networks for EEG structural features and acoustic features, then fuses them with an attention mechanism for MDD detection on MODMA. Achieves 96.30% accuracy with the multimodal approach, demonstrating that modality-specific graph representations can be effectively combined.
- **EEG Preprocessing:**
  - **128-channel resting-state EEG:** Signal segmented/epoched prior to feature extraction; differential entropy (DE) computed per canonical frequency band (δ/θ/α/β/γ) and used as node features on a channel graph; multi-graph GNN with adaptive edges performs spatial modeling and fuses with other modalities. Specific bandpass cutoffs and artifact-removal pipeline are not publicly available in the accessible portions of the paper.
  - **128-channel dot-probe/ERP EEG:** Not used by this paper.
  - **3-channel wearable resting EEG:** Not used by this paper.

### D4. A Multimodal Fusion Model with Multi-Level Attention Mechanism for Depression Detection (MFM-Att)
- **Authors:** Fang, M., Peng, S., Liang, Y., Hung, C.-C., Liu, S.
- **Year:** 2022
- **Venue:** Biomedical Signal Processing and Control
- **Link:** [https://www.sciencedirect.com/science/article/abs/pii/S1746809422010151](https://www.sciencedirect.com/science/article/abs/pii/S1746809422010151)
- **Modality:** EEG + Audio (+ visual in some variants)
- **Summary:** Combines audio, EEG, and visual modalities using multi-level attention to extract effective intra-modal and inter-modal features for depression detection. Achieves over 90% accuracy on both DAIC-WOZ and MODMA datasets.
- **EEG Preprocessing:**
  - **128-channel resting-state EEG:** Multi-view EEG feature extraction via an LSTM with attention; a Bi-LSTM plus attention-based fusion block integrates EEG with audio features. Specific bandpass cutoffs, artifact-removal method, epoch length, and channel selection are not publicly available (paywalled).
  - **128-channel dot-probe/ERP EEG:** Not used by this paper.
  - **3-channel wearable resting EEG:** Not used by this paper.

### D5. Multimodal Depression Detection Based on Attention Graph Convolution and Transformer (MHA-GCN_ViT)
- **Authors:** Jia, X., Chen, J., Liu, K., Wang, Q., He, J.
- **Year:** 2025
- **Venue:** Mathematical Biosciences and Engineering (AIMS)
- **Link:** [https://www.aimspress.com/article/doi/10.3934/mbe.2025024?viewType=HTML](https://www.aimspress.com/article/doi/10.3934/mbe.2025024?viewType=HTML)
- **Modality:** EEG + Audio
- **Summary:** Leverages multi-head attention GCN for EEG spatial features and Vision Transformer for audio frequency features, fusing both modalities for MODMA depression detection. Achieves 89.03% accuracy, 90.16% precision, 89.04% recall, and 88.83% F1-score with 5-fold cross-validation.
- **EEG Preprocessing:**
  - **128-channel resting-state EEG:** Signal segmented into ~2–4 s epochs; Short-Time Fourier Transform (STFT) produces spectral features; Differential Entropy (DE) computed per canonical frequency band from the segmented windows; a multi-head GCN operates on DE node features for spatial modeling while a Vision Transformer processes the frequency-domain features; the two streams are fused for classification. Specific bandpass cutoffs and artifact-removal pipeline are not publicly available.
  - **128-channel dot-probe/ERP EEG:** Not used by this paper.
  - **3-channel wearable resting EEG:** Not used by this paper.

### D6. Multimodal Transformer for Depression Detection Based on EEG and Interview Data
- **Authors:** Esmi, N., Shahbahrami, A., Gaydadjiev, G.
- **Year:** 2025
- **Venue:** Biomedical Signal Processing and Control
- **Link:** [https://www.sciencedirect.com/science/article/pii/S1746809425015502](https://www.sciencedirect.com/science/article/pii/S1746809425015502)
- **Modality:** EEG + Audio (interview)
- **Summary:** Jointly models spectral, spatial, and temporal EEG features alongside linguistic and paralinguistic cues from interviews using synchronized multi-head cross-attention and self-attention transformers. Achieves 91.22% on MODMA and 94.17% on DAIC-WOZ, with a 4.7% improvement over prior SOTA.
- **EEG Preprocessing:**
  - **128-channel resting-state EEG:** Spectral, spatial, and temporal EEG feature representations modeled via synchronized multi-head cross-attention plus self-attention transformers; flexible temporal-sequence matching reduces channel requirements. Specific bandpass cutoffs, artifact-removal method, and epoch length are not publicly available (paywalled).
  - **128-channel dot-probe/ERP EEG:** Not used by this paper.
  - **3-channel wearable resting EEG:** Not used by this paper.

### D7. Cross-Modal Knowledge Distillation for Enhanced Depression Detection
- **Authors:** (see Springer page for full list)
- **Year:** 2025
- **Venue:** Complex & Intelligent Systems (Springer)
- **Link:** [https://link.springer.com/article/10.1007/s40747-025-02035-z](https://link.springer.com/article/10.1007/s40747-025-02035-z)
- **Modality:** EEG + Audio → Audio (distillation)
- **Summary:** Trains a multimodal teacher model (EEG + audio) and distills its knowledge into a unimodal speech student model, enabling practical deployment without requiring EEG at inference time. The distilled speech model achieves 83.19% accuracy on MODMA, a 3.47% improvement over direct speech-only training.
- **EEG Preprocessing:**
  - **128-channel resting-state EEG:** First and last 10 s of each recording discarded (280 s retained to suppress acclimation/fatigue effects); 0.3–30 Hz FIR bandpass; window-wise baseline subtraction; ICA for ocular and muscular artifact removal. A MultimodalTeacherNet trains jointly on EEG + audio, and knowledge is distilled into a speech-only StudentNet for inference-time deployment without EEG.
  - **128-channel dot-probe/ERP EEG:** Not used by this paper.
  - **3-channel wearable resting EEG:** Not used by this paper.

### D8. TRI-DEP: A Trimodal Comparative Study for Depression Detection Using Speech, Text, and EEG
- **Authors:** Nurfidausi, et al.
- **Year:** 2025
- **Venue:** arXiv
- **Link:** [https://arxiv.org/abs/2510.14922](https://arxiv.org/abs/2510.14922)
- **Modality:** EEG + Speech + Text
- **Summary:** Compares unimodal (EEG, speech, text) and multimodal fusion strategies including majority voting for depression detection on MODMA. Achieves F1 = 0.874 with trimodal majority voting, providing a systematic comparison of all three modalities.
- **EEG Preprocessing:**
  - **128-channel resting-state EEG:** Evaluates two preprocessing branches: (A) 29-channel subset retained at 250 Hz with 10-s segmentation, feeding handcrafted features; (B) 19-channel subset downsampled to 200 Hz with 5-s segmentation, replicating the CBraMod/MUMTAZ pipeline and feeding pre-trained brain embeddings. Both branches are compared under speech-, text-, and EEG-only and trimodal configurations. Specific bandpass cutoffs and artifact-removal details are not explicitly stated in the accessible version.
  - **128-channel dot-probe/ERP EEG:** Not used by this paper.
  - **3-channel wearable resting EEG:** Not used by this paper.

### D9. Synergistic Fusion of Clinical Interview EEG and Video for Depression Detection: A Cross-Modal Attention Approach
- **Authors:** Hymavathi, J., Anuradha, C.
- **Year:** 2024/2026
- **Venue:** Journal of Computing & Biomedical Informatics
- **Link:** [https://jcbi.org/index.php/Main/article/view/1222](https://jcbi.org/index.php/Main/article/view/1222)
- **Modality:** EEG + Video
- **Summary:** Proposes a dual-stream architecture using Graph Convolutional Networks for 128-channel EEG and LSTM for video, fused with cross-modal attention from MODMA interview data. Demonstrates that EEG+video fusion captures complementary neural and behavioral markers of depression.
- **EEG Preprocessing:**
  - **128-channel resting-state EEG:** Dual-stream architecture: a GCN+LSTM captures spatiotemporal brain dynamics from EEG (correlation-based functional connectivity or DE-derived node features on a channel graph), and cross-modal attention fuses EEG with video. Specific bandpass cutoffs, artifact-removal method, and epoch length are not publicly available in the accessible portions of the paper.
  - **128-channel dot-probe/ERP EEG:** Not used by this paper.
  - **3-channel wearable resting EEG:** Not used by this paper.

---

## Category E: Federated Learning & Privacy-Preserving Papers

### E1. Cross-Silo, Privacy-Preserving, and Lightweight Federated Multimodal System for MDD Identification
- **Authors:** Gupta, C., Khullar, V., Goyal, N., Saini, K., Baniwal, R., Kumar, S., Rastogi, R.
- **Year:** 2023
- **Venue:** PMC / Sensors
- **Link:** [https://pmc.ncbi.nlm.nih.gov/articles/PMC10795654/](https://pmc.ncbi.nlm.nih.gov/articles/PMC10795654/)
- **Modality:** EEG + Audio (federated)
- **Summary:** Implements cross-silo federated learning for multimodal (EEG + audio) depression detection on MODMA, enabling hospitals to collaboratively train models without sharing raw patient data. Reports 99.9% accuracy under both IID and non-IID data distributions, though such high numbers warrant scrutiny given the small dataset.
- **Audio Preprocessing:** **MFCC** features extracted via **librosa** with 25ms frame length and 10ms hop. Produces **161 total features** per audio file (aggregated across MFCC coefficients and summary statistics). Fed into LSTM / Bi-LSTM / 1D-CNN classifiers within the federated framework. Bi-LSTM achieved best performance on the audio branch.

### E2. Privacy Preserving Collaboratively Training Framework for Classification of MDD Using Non-IID 3-Channel EEG
- **Authors:** Gupta, C., Khullar, V., Goyal, N., Saini, K., Baniwal, R., Kumar, S., et al.
- **Year:** 2024
- **Venue:** Procedia Computer Science (ScienceDirect)
- **Link:** [https://www.sciencedirect.com/science/article/pii/S1877050924006756](https://www.sciencedirect.com/science/article/pii/S1877050924006756)
- **Modality:** 3-channel EEG (federated)
- **Summary:** Develops a federated learning framework for 3-channel EEG-based MDD screening using LSTM, Bi-LSTM, GRU, and 1D-CNN on MODMA's wearable EEG data under non-IID conditions. Bi-LSTM achieves 95.99% training accuracy and 95% testing accuracy with IID data, demonstrating federated viability.
- **Audio Preprocessing:** N/A — 3-channel EEG only.

### E3. RA3-FDA: Resource-Adaptive Federated Domain Adaptation for EEG-Based Depression Detection
- **Authors:** He, L., Liu, Y., et al.
- **Year:** 2024
- **Venue:** SSRN (preprint)
- **Link:** [https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5674500](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5674500)
- **Modality:** 128-channel EEG (federated)
- **Summary:** Addresses resource heterogeneity in federated EEG depression detection by adaptively adjusting model complexity per client, enabling collaboration between hospitals with different computational resources. Achieves 96.02% accuracy on MODMA with the federated domain adaptation framework.
- **Audio Preprocessing:** N/A — 128-channel EEG only.

### E4. Modality Independent Federated Multimodal Classification System for EEG, Audio and Text Data
- **Authors:** (see ScienceDirect page for full list)
- **Year:** 2025
- **Venue:** Biomedical Signal Processing and Control
- **Link:** [https://www.sciencedirect.com/science/article/abs/pii/S1746809425004495](https://www.sciencedirect.com/science/article/abs/pii/S1746809425004495)
- **Modality:** EEG + Audio + Text (federated)
- **Summary:** Proposes a modality-independent federated learning framework that can handle heterogeneous data types (EEG, audio, text) across different institutions for depression detection on MODMA. Enables privacy-preserving multimodal learning even when institutions contribute different modalities.
- **Audio Preprocessing:** **MFCC** with optimized parameters: **30 coefficients**, **800ms frame length** (optimized from default), **10ms hop**. Also applies mel-spectrogram conversion and audio denoising/segmentation. Optimized MFCC parameters improved accuracy from 80.96% (default) to 87.16%. Fed into LSTM / Bi-LSTM / CNN within federated framework.

### E5. Cognitively Inspired Federated Learning Framework for Interpretable and Privacy-Secured EEG Biomarker Prediction of Depression Relapse
- **Authors:** (see MDPI page for full list)
- **Year:** 2024
- **Venue:** Bioengineering (MDPI)
- **Link:** [https://www.mdpi.com/2306-5354/12/10/1032](https://www.mdpi.com/2306-5354/12/10/1032)
- **Modality:** EEG (federated)
- **Summary:** Combines cognitively inspired feature extraction with federated learning for interpretable depression relapse prediction (not just detection) using MODMA EEG data. Achieves 92% accuracy while maintaining both interpretability and privacy.
- **Audio Preprocessing:** N/A — EEG only.

---

## Category F: Cross-Dataset & Domain Adaptation Papers

### F1. Uncertainty Aware Domain Incremental Learning for Cross Domain Depression Detection (UDIL-DD)
- **Authors:** (see Nature Scientific Reports page for full list)
- **Year:** 2025
- **Venue:** Scientific Reports (Nature)
- **Link:** [https://www.nature.com/articles/s41598-025-10917-y](https://www.nature.com/articles/s41598-025-10917-y)
- **Modality:** EEG (cross-dataset)
- **Summary:** Proposes evidential deep learning with adaptive thresholds for depression detection that incrementally adapts across multiple EEG datasets including MODMA. Achieves F1 63.65% when tested on MODMA in a cross-domain setting, revealing the significant challenge of cross-dataset generalization.

### F2. Mental-Perceiver: Audio-Textual Multi-Modal Learning for Estimating Mental Disorders
- **Authors:** Qin, J., Liu, C., Tang, T., Liu, D., Wang, M., Huang, Q., Zhang, R.
- **Year:** 2024
- **Venue:** arXiv
- **Link:** [https://arxiv.org/abs/2408.12088](https://arxiv.org/abs/2408.12088)
- **Modality:** Audio + Text
- **Summary:** Develops a multimodal perceiver architecture for anxiety and depression estimation using audio and text, evaluated on MODMA among other datasets. Addresses the challenge of estimating multiple mental disorders from a unified multimodal representation.

### F3. FAIRWELL: Fair Multimodal Self-Supervised Learning for Wellbeing Prediction
- **Authors:** Cheong, J., Mogharabin, A., Liang, P., Gunes, H., Kalkan, S.
- **Year:** 2024
- **Venue:** arXiv
- **Link:** [https://arxiv.org/abs/2508.16748](https://arxiv.org/abs/2508.16748)
- **Modality:** Multimodal
- **Summary:** Proposes fairness-aware self-supervised multimodal learning for wellbeing and depression prediction, evaluated on MODMA among other datasets. Addresses demographic bias in depression detection models, ensuring equitable performance across subgroups.

### F4. The First MPDD Challenge: Multimodal Personality-aware Depression Detection
- **Authors:** Fu, Y., et al.
- **Year:** 2025
- **Venue:** ACM International Conference on Multimedia
- **Link:** [https://arxiv.org/abs/2505.10034](https://arxiv.org/abs/2505.10034)
- **Modality:** Multimodal
- **Summary:** Introduces the MPDD challenge which uses PHQ-9 scale score annotations for depression detection, incorporating personality traits as a factor. While inspired by MODMA's design, uses its own data collection; references MODMA as a foundational dataset in the field.

### F5. Early Detection of Mental Health Disorders Using Machine Learning Models Using Behavioral and Voice Data Analysis
- **Authors:** Sharma, S.K., Alutaibi, A.I., Khan, A.R., Tejani, G.G., Ahmad, F., Mousavirad, S.J.
- **Year:** 2025
- **Venue:** Scientific Reports (Nature)
- **Link:** [https://www.nature.com/articles/s41598-025-00386-8](https://www.nature.com/articles/s41598-025-00386-8)
- **Modality:** Audio (+ behavioral data from separate dataset)
- **Summary:** Combines MODMA voice data with behavioral data from the Mental Disorder Classification dataset using a hybrid framework (Improved Random Forest + LightGBM for behavioral; SVM + KNN for voice). Demonstrates that fusing voice-based depression cues with behavioral indicators improves early mental health disorder detection.

---

## Category G: Related Papers NOT Using MODMA (Methodologically Relevant)

These papers do not use MODMA data but are included because they demonstrate methodologies directly relevant to your research questions (continuous severity regression, task comparison).

### G1. Continuous Scoring of Depression from EEG Signals via CNN-TCN Hybrid
- **Authors:** Hashempour, S., Boostani, R., Mohammadi, M., Sanei, S.
- **Year:** 2022
- **Venue:** IEEE Transactions on Neural Systems and Rehabilitation Engineering
- **Link:** [https://ieeexplore.ieee.org/document/9681891/](https://ieeexplore.ieee.org/document/9681891/)
- **Summary:** One of the few papers to attempt continuous depression severity regression (using BDI, not PHQ-9) from EEG, achieving MSE = 5.64 ± 1.6 (eyes-open). Methodologically relevant to RQ1 as a precedent for regression rather than classification.

### G2. Prediction of Beck Depression Inventory Score in EEG: Application of Deep-Asymmetry Method
- **Authors:** Kang, M., Kang, S., Lee, Y., et al.
- **Year:** 2022
- **Venue:** Applied Sciences (MDPI)
- **Link:** [https://www.mdpi.com/2076-3417/11/19/9218](https://www.mdpi.com/2076-3417/11/19/9218)
- **Summary:** Predicts continuous BDI scores from EEG using asymmetry features, another precedent for severity regression. Demonstrates that hemisphere asymmetry features can serve as continuous predictors, not just binary discriminators.

### G3. DSFMANet: An Automated Approach for Predicting HAMD-17 Scores
- **Authors:** (see ScienceDirect page for full list)
- **Year:** 2024
- **Venue:** Brain Research Bulletin
- **Link:** [https://www.sciencedirect.com/science/article/pii/S0361923024001175](https://www.sciencedirect.com/science/article/pii/S0361923024001175)
- **Summary:** Performs continuous HAMD-17 regression from EEG using a divergent selective focused multi-heads self-attention network. One of the most advanced continuous severity prediction models in the depression-EEG literature.

### G4. PDCH Dataset: A Multimodal Depression Consultation Dataset of Speech and Text with HAMD-17 Assessments
- **Authors:** (see Nature Scientific Data page for full list)
- **Year:** 2025
- **Venue:** Scientific Data (Nature)
- **Link:** [https://www.nature.com/articles/s41597-025-05817-9](https://www.nature.com/articles/s41597-025-05817-9)
- **Summary:** A new Chinese-language multimodal depression dataset (speech + text) with HAMD-17 scores, structurally similar to MODMA but larger and with different clinical scales. Potential cross-dataset validation partner for MODMA-trained models.

### G5. MOGAM: A Multimodal Object-Oriented Graph Attention Model for Depression Detection
- **Authors:** Cha, J., Kim, S., Kim, D., Park, E.
- **Year:** 2024
- **Venue:** arXiv
- **Link:** [https://arxiv.org/abs/2403.15485](https://arxiv.org/abs/2403.15485)
- **Summary:** Proposes object-oriented graph attention for multimodal depression detection, primarily evaluated on vlog datasets but cites MODMA as a reference dataset. Introduces a novel graph construction method that could be adapted for MODMA's multimodal data.

---

## Summary Statistics

| Category | Count | Description |
|---|---|---|
| A: Foundational / Dataset creators | 5 | Papers by the Lanzhou University team using the underlying MODMA data |
| B: EEG-only | 33 | Binary classification from 128-ch or 3-ch EEG |
| C: Audio/Speech-only | 9 | Binary classification from speech recordings |
| D: Multimodal (EEG + Audio) | 9 | Fusing EEG and speech for depression detection |
| E: Federated / Privacy | 5 | Privacy-preserving and distributed learning |
| F: Cross-dataset / Other | 5 | Cross-domain evaluation or related challenges |
| G: Related (non-MODMA) | 5 | Methodologically relevant but different datasets |

### Key Gaps in the Literature (Relevant to Your RQs)

1. **Zero papers attempt PHQ-9 continuous regression on MODMA** — all 48+ papers frame the task as binary MDD vs. HC classification.
2. **Zero papers attempt GAD-7 prediction on MODMA** — only one paper (B17) even mentions GAD-7, and only for soft-label derivation.
3. **Only one paper (C6) systematically compares interview vs. reading speech tasks** — the Mixture of Experts paper, which is directly relevant to RQ2 but does not examine all task types (picture description, valence conditions).
4. **Only one paper (B8) uses the dot-probe ERP data** — the vast majority of EEG papers use resting-state only.
5. **Cross-corpus validation is rare** — only a handful of papers test on both MODMA and DAIC-WOZ.
