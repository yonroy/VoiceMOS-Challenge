# 19 — Paper draft v1 (English, ICASSP 2027)

> **Start version** — first full English draft, generated 5/6/2026 from results in `04_experiments_log.md` / `18_leaderboard_history.md`.
> Status: DEV-set results only (eval released 31/7/2026). `[TODO]` markers = to fill before submission. Vietnamese thinking notes live in `15_paper_draft.md`.

---

## Title

**Fusing Complementary Self-Supervised Emotion Representations for Multi-Task Emotional Speech MOS Prediction**

*Alternative:* A Unified Multi-Task Model for Quality and Emotion MOS Prediction of Emotional TTS

**Authors:** Tran Minh Toan¹, [Mentor — TBD]¹  
¹ [Affiliation — TODO]  
Contact: tranminhtoan14062001@gmail.com

---

## Abstract

Automatic mean opinion score (MOS) prediction is central to evaluating text-to-speech (TTS) systems at scale, but the VoiceMOS Challenge 2026 Track 2 extends the task to *emotional* speech, requiring joint prediction of perceived quality (QMOS), emotion-similarity to a target (EMOS), categorical emotion distribution (CAT), and continuous valence/arousal/dominance (VAD). The official baseline stitches together three independent models — UTMOS for quality, emotion2vec for categories, and a zero-shot large language model for EMOS/VAD — yielding weak emotional predictions and ignoring the available human-labelled training data. We show that two self-supervised emotion encoders are *complementary* (emotion2vec is strongest for EMOS while a WavLM-based SER model is strongest for VAD), and propose a single multi-task model that fuses their frozen representations through a shared trunk with four task heads, jointly predicting all six leaderboard columns. We further fine-tune the WavLM backbone (warm-started from the SER model) to push the emotional dimensions beyond the frozen ceiling. On the official development set, our approach improves EMOS Spearman correlation from 0.194 to **0.811**, valence from 0.34 to **0.66**, reduces categorical error from 0.19 to **0.13**, and raises QMOS from 0.41 to **0.55**, outperforming every single-model variant on all six columns. Results indicate that *fusing complementary emotion representations and supervised fine-tuning* is markedly more effective than assembling zero-shot models for emotional MOS prediction.

*(~180 words — trim to journal limit; [TODO] add one eval-set sentence after 31/7.)*

---

## 1. Introduction

Text-to-speech (TTS) now underpins everyday applications — voice assistants, audiobook and podcast narration, game and virtual characters, film dubbing, emotionally aware customer service, companion and mental-health robots, and accessibility tools for users who cannot speak. In all of these, *what* is said matters less than it once did; *how* it is said — with the right emotion at the right moment — increasingly determines whether the interaction feels natural and human. The research frontier has therefore shifted from *quality* alone to *emotional expressiveness*. Progress on emotional TTS, however, is bottlenecked by *evaluation*. The gold standard is to collect human mean opinion scores (MOS), but human listening tests are slow, costly, and impossible to run continuously during model development, motivating automatic MOS predictors that act as a stand-in "judge". Crucially, today's automatic judges score only quality; they cannot yet assess *emotion*, leaving emotional-TTS developers without a cheap, scalable evaluation signal.

An automatic *emotional* MOS predictor is not merely a measurement convenience: it is the enabling feedback signal for *building* expressive speech systems. A reliable, reproducible judge of emotional quality can be queried millions of times to compare checkpoints, perform model selection, and gate releases — and, more ambitiously, can serve as the reward in preference-based optimization (e.g., RLHF) that teaches a TTS model to express the intended emotion. In this sense, learning to *measure* emotion in speech is a prerequisite for learning to *generate* it at scale: a trustworthy "emotional ruler" is the foundation on which emotionally intelligent voice AI can be developed.

The VoiceMOS Challenge has driven progress on automatic MOS prediction, and its 2026 edition introduces **Track 2 (Emotional TTS)** — the first track to evaluate not only *quality* but also *emotional* aspects of synthesized speech, directly targeting this gap.

Track 2 is substantially harder than prior MOS tasks because a single utterance must be scored along six axes: quality MOS (QMOS), emotion-similarity MOS to a specified target emotion (EMOS), a categorical emotion vote distribution over five classes (CAT), and three continuous affective dimensions — valence, arousal and dominance (VAD). A good system must therefore reason about both signal quality and emotional content simultaneously.

The official baseline addresses these axes with a **patchwork of three independent systems**: UTMOS (SpeechMOS) for QMOS, emotion2vec for CAT, and a zero-shot LLM-as-judge (Gemini) for EMOS and VAD. This design has three weaknesses: (i) the zero-shot emotional predictions are weak (baseline EMOS Spearman ≈ 0.19); (ii) the LLM incurs per-call API cost and cannot economically score the full set; and (iii) it ignores the 12,746 human-labelled training utterances supplied with the challenge.

We make the following contributions:

- **(C1) Empirical finding — representations are complementary.** emotion2vec dominates EMOS (0.637) while a WavLM-based SER model (SAILER) dominates VAD (arousal 0.712); neither wins everywhere, so *fusion* outperforms picking the single best model.
- **(C2) A unified six-column multi-task model.** Two frozen SSL encoders are concatenated into a shared trunk feeding four heads (QMOS, EMOS, CAT, VAD). The QMOS head consumes the trunk together with the UTMOS score as a residual anchor, giving the first improvement of QMOS (0.414→0.548) with no negative transfer to the emotional columns.
- **(C3) Supervised fine-tuning breaks the frozen ceiling.** Unfreezing the top WavLM layers (warm-started from the SER model) surpasses every frozen configuration on all five emotional columns (EMOS 0.811, CAT err 0.133, VAD 0.659/0.793/0.751).
- **(C4) Ablations and analysis** quantifying each component and explaining why rank-based SRCC is robust to value compression in VAD predictions.
- **(C5 — candidate) Practicality / efficiency.** `[TODO]` Unlike the recent trend of accuracy-only mega-ensembles (5–9 large models) that ignore latency and deployment cost, our system runs fully offline on a single commodity GPU with no API calls; we report parameter count and inference cost alongside correlation to argue competitive quality at a fraction of the cost — an underexplored, application-oriented angle for emotional MOS.

> `[TODO]` Position novelty precisely against prior fusion-for-MOS work (see `03_literature_notes.md`); confirm framing with mentor.

---

## 2. Related Work

**Automatic MOS prediction.** Learned MOS predictors began with MOSNet \cite{mosnet}, which framed naturalness rating as a deep regression problem, followed by listener-dependent and SSL-based variants such as LDNet \cite{ldnet} and SSL-MOS \cite{sslmos}. The VoiceMOS Challenge series \cite{voicemos2022,voicemos2024} has since standardised the benchmark; its quality winners UTMOS \cite{utmos} and the recent UTMOSv2 \cite{utmosv2} combine self-supervised features with auxiliary signals (listener/domain embeddings, spectrogram-image branches) and ensembling, and UTMOS provides the QMOS baseline in our task. All of this prior work targets a *single* quality/naturalness axis; VoiceMOS 2026 Track 2 is the first to additionally require *emotional* axes (EMOS/CAT/VAD), which existing quality predictors do not model.

**Self-supervised speech representations.** wav2vec 2.0 \cite{wav2vec2}, HuBERT \cite{hubert} and WavLM \cite{wavlm} learn transferable acoustic representations from large unlabelled corpora and are the de-facto backbones for MOS and paralinguistic tasks. WavLM, pre-trained with a denoising and speaker-aware objective on 94k hours, is particularly strong for affect-related tasks; we use WavLM-large both as a frozen feature extractor and as the backbone we fine-tune.

**Speech emotion representations.** emotion2vec \cite{emotion2vec} is a universal self-supervised emotion representation that tops many categorical SER benchmarks and supplies the categorical-emotion baseline in our task, but it does not output continuous valence/arousal/dominance (VAD). WavLM-based SER models from the Vox-Profile suite \cite{voxprofile} — notably the categorical-emotion model we refer to as SAILER \cite{sailer}, an Interspeech 2025 SER winner — expose both class probabilities and VAD, and the audeering wav2vec 2.0 MSP-Dim model \cite{audeering} predicts dimensional VAD directly. We find these encoders to be *complementary* across the emotional axes (emotion2vec is strongest for EMOS, the WavLM-based SER model for arousal/dominance), which motivates fusing them rather than selecting a single best model.

**Fusion and ensembling for speech assessment.** Feature fusion is a recurring theme among VoiceMOS/AudioMOS winners: the 2024 Track-1 winner fuses SSL features with a spectrogram-as-image branch \cite{t05}, PS-SQA \cite{pssqa} adds pitch and codec features for singing-voice MOS, and ensembles of 5–9 models are used by every top team \cite{voicemos2024}. Fusion of multiple SSL models specifically for MOS has also been studied \cite{sslfusionmos}. These works fuse representations to predict a *quality* score; we instead fuse complementary *emotion* representations and extend the multi-task output to the emotional axes.

**Emotional-TTS and emotion-similarity evaluation.** Quantifying how well synthesized speech matches a target emotion is emerging in emotional-TTS research: EmoSphere++ \cite{emosphere} introduces a speech-to-vector affective similarity signal, and acoustic emotional similarity has been studied via SSL embeddings \cite{acousticsim}. These are reference- or embedding-similarity measures internal to TTS pipelines, rather than predictors trained to match human opinion scores on a public benchmark. To our knowledge, predicting human-rated EMOS (target-emotion-similarity MOS) jointly with CAT and VAD — the Track-2 setting — has not been studied systematically; this is the gap our work addresses.

**Sequence models and LLM judges.** Beyond mean-pooling SSL frames, state-space models such as Mamba \cite{mamba} offer linear-time sequence encoding; MambaRate \cite{mambarate} and HighRateMOS \cite{highratemos} apply SSM/SSL hybrids to recent AudioMOS tracks, and we explore a Mamba temporal head as an alternative to pooling (§5). Orthogonally, audio large language models have been proposed as zero-shot speech-quality judges — the Track-2 EMOS/VAD baseline uses a multimodal LLM, while ALLD \cite{alld} and SpeechQualityLLM \cite{speechqualityllm} fine-tune or prompt audio LLMs for quality assessment. Such judges are flexible but uncalibrated and incur per-call API cost, whereas our model runs fully offline.

**Multi-task learning.** Jointly learning related tasks through a shared encoder is well established. The chief difficulty here is balancing six heterogeneous losses (regression for EMOS/VAD, soft cross-entropy for CAT) of differing scales; we adopt homoscedastic uncertainty weighting \cite{kendall}, which learns a per-task variance instead of hand-tuned weights.

---

## 3. Proposed Method

### 3.1 Overview
Given a 16 kHz waveform and (for EMOS) a target emotion label, the model predicts QMOS, EMOS, the five-class CAT distribution, and the three VAD values.

### 3.2 Unified frozen-fusion model (six columns)
Two **frozen** SSL encoders are used as feature extractors:
- **emotion2vec** (`iic/emotion2vec_plus_large`) → utterance embedding + 5-class emotion probabilities;
- **SAILER** (`tiantiaf/wavlm-large-categorical-emotion`, WavLM-large) → embedding + 9-class probabilities + VAD.

Features are concatenated as `[e2v_emb | e2v_probs | sailer_emb | sailer_probs | sailer_vad]` (a `USE_CLASSPROB` switch drops the probability terms for ablation) and passed to a shared **trunk** (two Linear→ReLU layers, width 512, dropout 0.3). Four heads follow:
- **QMOS** — input `[trunk | UTMOS score]`, regression (UTMOS as a residual anchor);
- **EMOS** — input `[trunk | one-hot target emotion]`, regression (EMOS depends on both audio and target);
- **CAT** — five logits with softmax (soft vote distribution);
- **VAD** — three-way regression.

> **[FIGURE 1] Architecture:** two frozen encoders → concat → shared trunk → four heads; QMOS additionally takes the UTMOS score.

### 3.3 Training objective
The emotional heads are trained with MSE (EMOS, VAD) and soft cross-entropy (CAT) against soft vote labels. Task losses are balanced by **homoscedastic uncertainty weighting** (Kendall et al.), learning a log-variance per task, `L = Σ_i (1/2σ_i²) L_i + log σ_i`; a `USE_UNCERTAINTY` switch reverts to manual weights. Continuous targets are z-scored so their MSE terms share a scale. Gold labels are aggregated per utterance: EMOS = mean `eMOS`, VAD = mean of `val/aro/dom`, CAT = the five-class vote ratio of the (multi-label) `emoCat` annotations.

### 3.4 Fine-tuned variant (best emotional results)
Head-only training over frozen backbones has a ceiling because the backbone was optimized for a different objective. We therefore unfreeze the **top 6 Transformer layers** of a WavLM-large backbone **warm-started from SAILER**, keeping the feature extractor and lower layers frozen, with the **audeering MSP-Dim** model as a frozen auxiliary branch for valence; emotion2vec is dropped in this variant. To fit a 16 GB T4 GPU we use fp16 AMP, gradient checkpointing, batch 4 × accumulation 8 (effective 32), 8-second crops, and learning rates 1e-5 (backbone) / 1e-3 (heads) with early stopping on validation mean SRCC. The QMOS column is taken from the frozen six-head model (§3.2); the final system therefore **combines columns** from both variants.

> **[FIGURE 2 — optional] Fine-tuned variant.**

---

## 4. Experimental Setup

**Dataset.** Track 2 data are newly collected on top of ESD and DailyTalk with 13 TTS systems: 12,746 training / 2,730 validation / 2,730 evaluation utterances over five emotions (neutral, happy, angry, sad, surprised). Training ratings total ≈91k judgments (~7.2 listeners/utterance). External resources: ESD, DailyTalk, and the pretrained checkpoints below. `[TODO declare links]`

**Metrics.** Utterance-level Spearman rank correlation (UTT-SRCC; higher is better) for QMOS/EMOS/VAD, and categorical error (sum |gt−pred| / total label mass; lower is better) for CAT. See `14_leaderboard_metrics.md`.

**Implementation.** Frozen features are cached as `.npz` per backbone (≈12–15 min one-off extraction on a T4, resumable), after which the trunk+heads train in minutes (trunk 512, head 128, dropout 0.3, lr 1e-3, batch 64, ≤80 epochs, early-stop on validation mean SRCC). Pretrained checkpoints: `iic/emotion2vec_plus_large`; SAILER `tiantiaf/wavlm-large-categorical-emotion` (Open RAIL, non-commercial); audeering MSP-Dim (CC BY-NC-SA, non-commercial); SpeechMOS UTMOS22. All runs on Kaggle T4.

---

## 5. Results and Analysis

### 5.1 Main results (development set, CodaBench)

| System | QMOS↑ | EMOS↑ | CAT err↓ | VAL↑ | ARO↑ | DOM↑ |
|---|---|---|---|---|---|---|
| Baseline (UTMOS+e2v+LLM) | 0.414 | 0.194* | 0.193 | — | — | — |
| exp01 — EMOS←emotion2vec (zero-shot) | 0.414 | 0.637 | 0.193 | — | — | — |
| exp03 — SAILER, single model (zero-shot) | 0.414 | 0.562 | 0.190 | 0.341 | 0.712 | 0.630 |
| exp04 — frozen fusion, 5 heads | 0.414 | 0.788 | 0.145 | 0.578 | 0.754 | 0.706 |
| exp07 — frozen fusion, 6 columns | **0.548** | 0.795 | 0.153 | 0.581 | 0.752 | 0.705 |
| exp08 — fine-tuned WavLM (emotion) | 0.414† | **0.811** | **0.133** | **0.659** | **0.793** | **0.751** |
| **Proposed (column merge: QMOS←exp07, emotion←exp08)** | **0.548** | **0.811** | **0.133** | **0.659** | **0.793** | **0.751** |
| Evaluation set | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` |

\* partial submission (496/2730 LLM-scored). † fine-tuned submission lacked the exp07 answer file and fell back to UTMOS (a packaging artifact, not a model deficiency). Bold = best per column.

### 5.2 Ablation

| Configuration | EMOS↑ | CAT err↓ | VAL↑ | ARO↑ | DOM↑ | Tests |
|---|---|---|---|---|---|---|
| Frozen fusion (e2v+SAILER, uncertainty) — exp04 | 0.788 | 0.145 | 0.578 | 0.754 | 0.706 | base fusion |
| Fine-tune top-6 WavLM (warm-start SAILER) — exp08 | **0.811** | **0.133** | **0.659** | **0.793** | **0.751** | frozen vs fine-tuned |
| − SAILER (emotion2vec only) | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | SAILER's VAD role |
| − emotion2vec (SAILER only) | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | e2v's EMOS role |
| − uncertainty (manual weights) | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | loss balancing |
| − class-prob (embeddings only) | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | `[TODO]` | value of probs |
| emotion2vec zero-shot (single) | 0.637 | — | — | — | — | vs trained head |
| SAILER zero-shot (single) | — | — | 0.341 | 0.712 | 0.630 | vs fusion |

### 5.3 Analysis
- **Why fusion wins.** emotion2vec and SAILER capture different facets of emotion (class discrimination → EMOS vs continuous axes → VAD); the shared trunk learns to combine them, lifting every emotional column above any single model.
- **Fine-tuning breaks the frozen ceiling.** Unfreezing the top WavLM layers (exp08) beats the frozen fusion (exp04) on all five emotional columns, at the cost of losing feature caching and requiring AMP + gradient checkpointing + accumulation to fit a T4.
- **Rank metric robustness.** Although predicted VAD values are compressed (≈2.5–3.6), SRCC scores *ordering*, so valence still rises 0.341→0.659 — a useful, counter-intuitive observation for practitioners.
- **Cost.** The proposed system runs fully offline on a single T4 and scores all 2,730 utterances, unlike the API-bound LLM baseline.
- **Remaining weakness.** QMOS reaches 0.548 but remains the relatively weakest column; emotional fine-tuning does not target it. Future work: replace the UTMOS anchor with UTMOSv2 or fine-tune a dedicated quality branch.

---

## 6. Conclusion

We presented a unified multi-task model for emotional-speech MOS prediction that fuses complementary self-supervised emotion representations and is optionally fine-tuned to break the frozen-feature ceiling. On the VoiceMOS 2026 Track 2 development set it improves all six leaderboard columns over the patchwork baseline and over every single-model variant, while running offline on commodity hardware. Future work includes leveraging external emotional corpora (IEMOCAP, MSP-Podcast), strengthening QMOS with UTMOSv2, and reporting evaluation-set results. `[TODO finalize after eval]`

---

## References (placeholder — verify every arXiv ID + add full BibTeX before submission)

> `\cite{key}` markers in §2 map to the keys below. arXiv IDs are from `03_literature_notes.md` and recalled from memory — **must be checked against the actual papers** before camera-ready.

**MOS prediction & VoiceMOS.**
- `mosnet` — Lo et al., *MOSNet: Deep Learning-Based Objective Assessment for Voice Conversion*, Interspeech 2019 (arXiv:1904.08352).
- `ldnet` — Huang et al., *LDNet: Listener-Dependent MOS Prediction*, 2021. `[TODO id]`
- `sslmos` — Cooper et al., *Generalization Ability of MOS Prediction Networks (SSL-MOS)*, ICASSP 2022 (arXiv:2110.02635).
- `utmos` — Saeki et al., *UTMOS: UTokyo-SaruLab System for VoiceMOS 2022*, Interspeech 2022 (arXiv:2204.02152).
- `utmosv2` — UTMOSv2, VoiceMOS/AudioMOS 2024 quality system. `[TODO id]`
- `voicemos2022` — Huang et al., *The VoiceMOS Challenge 2022* (arXiv:2203.11389). `[TODO verify]`
- `voicemos2024` — *The VoiceMOS Challenge 2024: Beyond Speech Quality Prediction* (arXiv:2409.07001).
- `t05` — 2024 Track-1 winner, SSL + spectrogram-image fusion (arXiv:2409.09305).
- `pssqa` — *PS-SQA: Pitch-and-Spectrum-Aware Singing Quality Assessment* (arXiv:2411.11123).
- `sslfusionmos` — *Fusion of Self-Supervised Learned Models for MOS Prediction* (arXiv:2204.04855).

**SSL backbones.**
- `wav2vec2` — Baevski et al., *wav2vec 2.0*, NeurIPS 2020 (arXiv:2006.11477).
- `hubert` — Hsu et al., *HuBERT*, TASLP 2021 (arXiv:2106.07447).
- `wavlm` — Chen et al., *WavLM*, JSTSP 2022 (arXiv:2110.13900).

**Emotion representations.**
- `emotion2vec` — Ma et al., *emotion2vec: Self-Supervised Pre-Training for Speech Emotion Representation*, Findings of ACL 2024 (arXiv:2312.15185).
- `voxprofile` — *Vox-Profile* speaker/affect profiling suite (arXiv:2505.14648).
- `sailer` — categorical-emotion WavLM-large SER model, Interspeech 2025 (arXiv:2505.22133).
- `audeering` — Wagner et al., audeering wav2vec 2.0 MSP-Dim dimensional emotion model (CC BY-NC-SA). `[TODO id]`

**Emotion-similarity / emotional TTS.**
- `emosphere` — *EmoSphere++* emotional TTS with affective similarity (arXiv:2411.02625).
- `acousticsim` — *Acoustic Similarity in Emotional Speech via SSL* (arXiv:2409.17899).

**Sequence models & LLM judges.**
- `mamba` — Gu & Dao, *Mamba: Linear-Time Sequence Modeling with Selective State Spaces* (arXiv:2312.00752).
- `mambarate` — *MambaRate*, AudioMOS 2025 (arXiv:2507.12090).
- `highratemos` — *HighRateMOS*, AudioMOS 2025 (arXiv:2506.21951).
- `alld` — *Audio LLMs Can Be Descriptive Speech Quality Evaluators*, ICLR 2025 (arXiv:2501.17202).
- `speechqualityllm` — *SpeechQualityLLM* (arXiv:2512.08238).

**Multi-task learning.**
- `kendall` — Kendall, Gal, Cipolla, *Multi-Task Learning Using Uncertainty to Weigh Losses*, CVPR 2018 (arXiv:1705.07115).

---

## Submission checklist
- [ ] Fill eval-set results (after 31/7/2026)
- [ ] Run remaining ablation rows
- [ ] Draw Figure 1 (and optional Figure 2)
- [ ] Complete BibTeX references
- [ ] Declare all external data/resources + licenses (SAILER Open RAIL, audeering CC BY-NC-SA non-commercial)
- [ ] Fit ICASSP 2027 template + page limit; final proofread
