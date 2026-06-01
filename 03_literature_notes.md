# 03 — Ghi chú Literature

Tóm tắt các paper liên quan. Dùng trực tiếp cho phần **Related Work** của paper sau này.

> Mẹo: với mỗi paper ghi 3 thứ — (1) họ làm gì, (2) điểm mạnh/yếu, (3) mình học/dùng được gì.

---

## Baseline của challenge

### UTMOS (QMOS Baseline)
- **Nguồn:** Saeki et al., Interspeech 2022
- **Link:** https://www.isca-archive.org/interspeech_2022/saeki22c_interspeech.html
- **Họ làm gì:** [Điền — hệ thống dự đoán MOS dùng SSL features + ensemble]
- **Điểm mạnh/yếu:** [Điền]
- **Mình dùng được gì:** [Điền]

### Emotion2vec (Emotion Categories Baseline)
- **Nguồn:** Findings of ACL 2024
- **Link:** https://aclanthology.org/2024.findings-acl.931.pdf
- **Họ làm gì:** [Điền — self-supervised emotion representation]
- **Điểm mạnh/yếu:** [Điền]
- **Mình dùng được gì:** [Điền]

### Gemini LLM-as-judge (EMOS + VAD Baseline)
- **Họ làm gì:** Dùng LLM đa phương thức để chấm điểm cảm xúc
- **Điểm mạnh/yếu:** [Điền — prompt phụ thuộc, có thể cải tiến]
- **Mình dùng được gì:** [Điền]

---

## SSL Speech Models (backbone tiềm năng)

### Wav2Vec2
- [Điền tóm tắt]

### HuBERT
- [Điền tóm tắt]

### WavLM
- [Điền tóm tắt]

---

## MOS Prediction (background)

### Paper liên quan
| Paper | Năm | Ý chính | Note |
|---|---|---|---|
| MOSNet | 2019 | MOS prediction đầu tiên dùng DNN | |
| LDNet | | | |
| SSL-MOS | | | |
| [Điền] | | | |

---

## Emotional TTS & Emotion Recognition

### Public datasets emotional speech
| Dataset | Mô tả | Cảm xúc | Public? |
|---|---|---|---|
| IEMOCAP | | | ✅ |
| ESD (Emotional Speech Database) | | | ✅ |
| MSP-Podcast | | | ✅ |
| RAVDESS | | | ✅ |
| [Điền] | | | |

### Valence-Arousal-Dominance
- [Điền về mô hình VAD trong emotion]

---

## Bài học từ VoiceMOS Challenge 2024 (phiên bản trước)

> Paper tổng kết: "The VoiceMOS Challenge 2024: Beyond Speech Quality Prediction" — arXiv 2409.07001
> Đọc kỹ phần này vì các kỹ thuật thắng cuộc 2024 áp dụng được cho 2026.

### 3 track của 2024
| Track | Chủ đề | Dataset | Metric |
|---|---|---|---|
| 1 | "Zoomed-in" TTS chất lượng cao | zoomed-in BVCC | SRCC (ranking) |
| 2 | Singing voice (SVS/SVC) | SingMOS | SRCC (ranking) |
| 3 | Semi-supervised: noisy/clean/enhanced | TMHINT-QI mở rộng | LCC (Pearson) |

Track 3 cực kỳ hạn chế dữ liệu: chỉ 60 câu train + 40 câu validation có nhãn, không được dùng thêm data có nhãn MOS chủ quan, nhưng được dùng data khác để augmentation/pretraining.

### Kết quả tổng quát
- 8 đội tham gia (cả academia + industry)
- Baseline mỗi track đều bị ít nhất 1 hệ thống vượt qua
- Kỹ thuật chủ đạo: **fine-tune SSL model** (wav2vec2, HuBERT) — giống VMC 2022, 2023

---

### ⭐ Hệ thống thắng cuộc đáng học

#### T05 — Vô địch Track 1 (Naturalness MOS cho TTS chất lượng cao)
- **Nguồn:** arXiv 2409.09305
- **Ý tưởng cốt lõi:** Kết hợp 2 luồng đặc trưng
  1. **SSL speech feature** (wav2vec2/HuBERT) — bắt thông tin ngữ nghĩa/âm học
  2. **Image feature extractor** (EfficientNetV2 pretrain trên ImageNet) — xử lý **spectrogram như ảnh** để bắt khác biệt giữa các hệ thống TTS
- **Phương pháp:** train riêng 2 MOS predictor → fine-tune bằng **fusion** 2 luồng feature
- **Kết quả:** nhất 7/16 metric, nhì 9/16 metric, vượt xa hạng 3 trở xuống
- **🔑 Học được gì:** (1) **feature fusion** rất mạnh; (2) ý tưởng coi spectrogram như ảnh + transfer learning từ image classifier là novelty đáng giá

#### PS-SQA — Vô địch Track 2 (Singing MOS)
- **Nguồn:** arXiv 2411.11123 (Pitch-and-Spectrum-Aware Singing Quality Assessment)
- **Ý tưởng cốt lõi:** SSL MOS predictor + thông tin chuyên biệt cho giọng hát
  1. **Pitch histogram** — bắt thông tin cao độ
  2. **Non-quantized neural codec** — bắt thông tin phổ (spectral)
  3. **Bias correction** — sửa lệch dự đoán do data ít (low-resource)
  4. **Model fusion** — gộp nhiều model tăng độ chính xác
- **🔑 Học được gì:** (1) thêm **domain-specific features** (pitch cho hát) vào SSL backbone; (2) **bias correction** quan trọng khi data ít; (3) lại là **fusion**

#### LE-SSL-MOS (mạnh ở VMC 2023, nền tảng cho sau này)
- **Nguồn:** arXiv 2311.10656
- **Ý tưởng:** fuse phương pháp **supervised + unsupervised**
  - Listener enhancement branch (dùng điểm của từng listener)
  - SpeechLMScore (metric unsupervised từ speech-LM)
  - ASR confidence như metric phụ + ensemble
- **🔑 Học được gì:** kết hợp tín hiệu unsupervised (ASR, LM score) bổ trợ cho supervised MOS

---

### 🎯 Pattern chung của các hệ thống thắng cuộc
1. **SSL backbone là nền tảng** — wav2vec2 / HuBERT / WavLM fine-tune gần như bắt buộc
2. **Feature fusion thắng lớn** — kết hợp nhiều nguồn feature (SSL + spectrogram-as-image + domain features) luôn vượt single-feature
3. **Domain-specific knowledge** — thêm đặc trưng phù hợp bài toán (pitch cho hát; → với mình có thể là **emotion embedding** cho Track 2)
4. **Bias correction** khi data ít
5. **Model ensemble/fusion** ở bước cuối

---

## Ý tưởng rút ra cho project (Track 2 — Emotional TTS 2026)
> Tổng hợp những gì đọc được → định hình hướng nghiên cứu

- [ ] **Áp dụng fusion pattern:** SSL backbone (WavLM) + **emotion embedding** (Emotion2vec) — tương tự T05 fuse SSL + image, nhưng thay image bằng emotion feature. Đây có thể là novelty cho EMOS.
- [ ] **Spectrogram-as-image:** thử thêm luồng EfficientNet trên mel-spectrogram cho QMOS (tái dùng ý tưởng T05)
- [ ] **Multi-task:** dự đoán QMOS + EMOS chung backbone, 2 head riêng — chia sẻ representation
- [ ] **Bias correction** nếu emotional data ít nhãn
- [ ] Thử dùng VAD (valence/arousal/dominance) như **auxiliary task** để bổ trợ EMOS — ít người làm, dễ novelty
- [ ] Ensemble nhiều seed/model ở bước cuối để đẩy SRCC
