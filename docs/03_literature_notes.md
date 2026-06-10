# 03 — Ghi chú Literature

Tóm tắt các paper liên quan. Dùng trực tiếp cho phần **Related Work** của paper sau này.

> Mẹo: với mỗi paper ghi 3 thứ — (1) họ làm gì, (2) điểm mạnh/yếu, (3) mình học/dùng được gì.

---

## 🧠 5 mắt xích hiểu cách hệ thống hoạt động (ghi 8/6 — buổi học khái niệm, người mới)

> Buổi học "cách hoạt động" của chính hệ thống Track 2, đi từ trực giác → ví dụ → code thật. Nền cho phần **Method** của paper. Chi tiết kiến trúc model: [16_model_architectures.md](16_model_architectures.md); từ điển DL: [17_dl_keywords.md](17_dl_keywords.md).

```
audio ─[1]WavLM SSL─▶ [200×1024] ─[2]pool+head─▶ điểm ─[3]fusion+multitask(exp07)─ ... ─[4]freeze/finetune ─ ... ─[5]MSE vs SRCC
```

1. **WavLM / SSL backbone:** "self-supervised" = tự che dữ liệu rồi đoán → học "nghe" từ ~94k giờ audio KHÔNG nhãn → biến waveform thô (64k số) thành **chuỗi** đặc trưng `[~200 khung × 1024]`. 2 khối: CNN encoder ("tai sơ cấp", gộp 20ms/khung) + Transformer (**self-attention** = mỗi khung nhìn cả câu → bắt ngữ điệu = nơi cảm xúc ẩn). Dữ liệu mình quá ít (~12.7k) để học nghe từ 0 → buộc đứng trên vai WavLM.
2. **Feature → điểm:** chuỗi 200 vector → **pooling** (gộp 200→1; mean-pool = exp08, attention/Mamba = có trọng số) → **head** (mạng nhỏ Linear→ReLU→Linear → 1 số). Backbone to (316M), head bé → cái khó backbone lo rồi, head chỉ ánh xạ đặc trưng→điểm nên ít data vẫn train được.
3. **Fusion + Multi-task (= exp07, hệ chính):** **Fusion** = nối embedding 2 backbone (emotion2vec chuyên cảm xúc + SAILER/WavLM tổng quát) ở **đầu vào** (`np.concatenate`). **Multi-task** = 1 trunk chung → 6 head ở **đầu ra**; task bổ trợ nhau ("học sinh giỏi toàn diện") → thắng model lẻ (exp04 EMOS 0.788 > emotion2vec 0.637). **Uncertainty weighting** (`log_var` model tự học) cân 6 loss khác thang để loss to không lấn loss nhỏ. Code: class `FusionMTL6` trong `exp07_fusion_qmos_pipeline.py`.
4. **Freeze vs Fine-tune:** freeze 🔒 (backbone khóa, chỉ train head = exp07) vs fine-tune 🔓 (mở băng `UNFREEZE_TOP_LAYERS=6` lớp TRÊN, dùng `requires_grad`, 2 LR backbone 1e-5/head 1e-3, warm-start SAILER = exp08). Fine-tune giỏi cảm xúc (EMOS 0.79→0.81) NHƯNG làm rớt QMOS 0.548→0.414 (đặc trưng lệch khỏi trục chất lượng = "quên kiến thức cũ") → đẻ ra **chiến lược TRỘN CỘT** (QMOS←exp07, 5 cảm xúc←exp08). Code khóa/mở: dòng 176-186 `exp08_finetune_emotion_pipeline.py`.
5. **Metric ≠ Loss:** train bằng **MSE** (đoán đúng *con số*, mượt dễ backprop) nhưng leaderboard chấm **SRCC** (đúng *thứ hạng* — model lệch đều vẫn SRCC=1). 2 cái thường đồng chiều nhưng không luôn → exp15 thêm **ranking loss** (phạt khi xếp ngược cặp thứ tự) để tối ưu thẳng cái được chấm.

---

## 🎓 8 bài kinh nghiệm train/fine-tune (ghi 8/6 — Phiên 12, rút từ chính dự án)

1. **Fine-tune > freeze nhưng có trần:** freeze chỉ train head (exp04/07, EMOS~0.79) có trần vì backbone "nghĩ" theo task cũ; mở băng vài lớp trên (exp08) phá trần (EMOS 0.811). Mở vài lớp, không mở hết.
2. **Val nội bộ đẹp = bẫy overfit:** exp11 val 0.83 nhưng DEV 0.66 → val nội bộ chỉ để chọn epoch dừng, điểm thật phải nộp DEV. Số đẹp bất thường = nghi rò rỉ.
3. **Warm-start đã đỉnh → train thêm vô ích:** exp08b resume ≈ exp08. Muốn lên phải **đổi chất** (data/kiến trúc/loss/ensemble), không "đổ thêm giờ".
4. **Checkpoint:** lưu ĐỦ (backbone+head+optimizer, `weights_only=False`) + lưu MỖI best + Save Version ngay. Sự cố exp08 mất backbone vì chỉ lưu heads.
5. **Data nhỏ (12k) đừng from-scratch** (exp12): pretrain ~94k giờ, 12k quá ít để dạy từ 0 → overfit. Data ít → mở băng ít lớp, LR nhỏ.
6. **Loss khớp metric:** chấm SRCC (thứ hạng) nhưng train MSE (giá trị) → exp15 thêm ranking loss; lưu ý ranking cần batch lớn (exp06/07) mới mạnh.
7. **Mẹo T4 16GB:** AMP fp16 · gradient checkpointing (layerdrop=0!) · gradient accumulation (batch hiệu dụng) · 2 LR (backbone 1e-5 / head 1e-3).
8. **Fusion ≠ Ensemble:** fusion = nối đặc trưng trong 1 model (đang có); ensemble = trung bình KẾT QUẢ nhiều model riêng (chưa có) → đòn rẻ giảm gap dev→eval (train 3 seed → TB).

> Phân biệt 3 vị trí trong kiến trúc: **fusion** (nối nguồn ở đầu vào, concat) → **pooling** (gộp chuỗi frame→1 vector; mean-pool=exp08, Mamba=exp15) → **trunk** (thân chung MLP cho multi-task) → **heads**. Mamba thay *pooling*, không phải *trunk*.

---

## 🤖 Audio-LLM-as-Judge cho MOS (ghi 8/6 — Phiên 12, nền cho exp16 + Related Work)

- **Ý tưởng:** đưa audio thẳng cho **audio-LLM** (Gemini, GPT-4o-audio) qua API + prompt → bắt nó chấm MOS. Đây là hướng "LLM-as-judge" đang nóng, nhưng cho **MOS cảm xúc** gần như chưa ai khảo sát có hệ thống → **góc novelty** cho paper.
- **Prior art nặng (Phiên 10):** ALLD (ICLR 2025, fine-tune audio-LLM end-to-end, MSE 0.17) — SOTA nhưng quá nặng để train T4. Cách API zero/few-shot né train, vẫn ra số để so sánh.
- **Giả thuyết cần kiểm:** LLM hiểu **ngữ nghĩa cảm xúc** → kỳ vọng khá ở **EMOS/CAT**; nhưng artifact chất lượng tinh vi (méo/robot) khó nghe → kỳ vọng **yếu ở QMOS**. exp16 sẽ cho bảng số để khẳng định/bác bỏ.
- **Bài học từ baseline Gemini cũ:** EMOS 0.194 vì prompt sơ sài + chấm thiếu mẫu → làm bài bản (prompt rõ định nghĩa metric, đủ 2730 mẫu, temp=0) mới công bằng.

---

## SOTA mới khảo sát (ghi 8/6 — Phiên 10, theo gợi ý mentor "đọc/apply SOTA mới")

### Hướng 1 — LLM-based MOS prediction (audio-LLM chấm chất lượng)
- **ALLD** — *Audio LLMs Can Be Descriptive Speech Quality Evaluators* (ICLR 2025, arXiv:2501.17202). (1) Dùng audio-LLM + alignment/distillation để vừa cho điểm MOS vừa mô tả bằng lời. (2) Mạnh: MSE 0.17, LCC/SRCC 0.93, **vượt cả wav2vec2 & WavLM**; yếu: rất nặng (tỉ tham số). (3) Dùng được: làm baseline "related work" + có thể chạy **zero-shot/feature** cho nhánh cảm xúc (không train).
- **SpeechQualityLLM** (arXiv:2512.08238): MAE 0.41, Pearson 0.86; giao diện hỏi tự do từng khía cạnh chất lượng.
- → **Kết luận:** quá nặng để train trên T4 → để dành làm **section khảo sát / zero-shot** cho paper, KHÔNG ưu tiên tăng điểm.

### Hướng 2 — Mamba / State Space Model (CHỌN LÀM TRƯỚC)
- **Mamba** = SSM xử lý chuỗi, **độ phức tạp tuyến tính** (Transformer là bậc 2) → nhẹ, nhanh, hợp chuỗi audio dài. Thay được phần attention/pooling bằng khối Mamba.
- **MambaRate** (AudioMOS 2025, arXiv:2507.12090): ghép **SSL embedding + Mamba** dự đoán MOS ổn định qua nhiều sampling rate. **HighRateMOS** (2506.21951): hạng 1 ở 5/8 metric AudioMOS 2025 Track 3.
- → **Dùng được (exp14/exp15):** giữ đặc trưng WavLM **frame-level** (chưa pool) → Mamba học **temporal dynamics** (lên/xuống giọng, ngắt quãng, run giọng) → kỳ vọng hơn mean-pool. Cài: `mamba-ssm` + `causal-conv1d` (kernel CUDA, build hay lỗi Kaggle) hoặc bản thuần PyTorch (mamba-minimal, chậm hơn).
- **Lưu ý kỹ thuật:** Mamba cần **chuỗi**, không phải 1 vector đã pool → đây là lý do exp03–08 (mean-pool) không "thử Mamba" trực tiếp được mà phải đổi luồng đặc trưng.

---

## Khái niệm nền (ghi 8/6 — Phiên 9)

### Fusion vs Ensemble (phân biệt cho paper)
- **Fusion (feature fusion):** nối ĐẶC TRƯNG nhiều nguồn TRONG 1 model → trunk chung → train 1 lượt. Hệ hiện tại: exp04/07/08/11.
- **Ensemble:** chạy NHIỀU model riêng → trung bình KẾT QUẢ ở cuối (rồi mới xếp hạng SRCC). Hệ chưa có thật; exp10 (WavLM+audeering) và "exp08 nhiều seed" là ensemble.
- Học được: 2 cái bổ sung nhau; ensemble thường ổn định hơn (giảm gap dev↔eval), là đòn các đội mạnh hay dùng.

### From-scratch vs fine-tune (bối cảnh gợi ý mentor)
- Với ~12k mẫu, fine-tune SSL pretrained gần như chắc chắn > train from-scratch: SSL (WavLM) pretrain trên ~94.000 GIỜ audio; 12k câu quá ít để học "nghe" từ đầu → from-scratch overfit/underfit. exp12 dựng để kiểm chứng bằng số (scratch/base/sailer).
- VAL nội bộ ≠ DEV: exp11 cho VAD nội bộ 0.80/0.87/0.80 nhưng DEV (exp08) chỉ 0.66/0.79/0.75 → vắt internal val có thể KHÔNG tăng (thậm chí giảm) điểm leaderboard. Luôn nộp DEV để biết điểm thật.

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

## 🔎 Khảo sát model SER tốt hơn emotion2vec (cập nhật 4/6/2026)

> Bối cảnh: mentor lưu ý emotion2vec (ACL 2024) đã ra lâu → khảo sát model mới hơn cho **EMOS/CAT/VAD**.
> ⚠️ Mọi điểm dưới đo trên IEMOCAP/MSP-Podcast/MELD/EmoBox — **KHÔNG phải data challenge** (TTS + ESD + DailyTalk, có Surprise + tiếng Trung). Phải tự đo SRCC trên validation mới biết thắng thật (lệch miền).

### Bảng tổng hợp model tốt hơn emotion2vec

| Model | Hơn ở mặt nào | Cỡ / chạy Kaggle T4? | Đầu ra | License | Hợp việc của mình |
|---|---|---|---|---|---|
| **Whisper-large-v3 encoder** | Cảm xúc rời rạc — **đứng nhất 30/32 bộ** trên EmoBox | ~0.6B · ✅ được | Embedding → tự gắn head | MIT (thoáng) | 🟢 Thay backbone cho **CAT/EMOS** |
| **WavLM-large + SAILER** (vô địch IS2025 SER) | Cảm xúc rời rạc, naturalistic | ~0.3B · ✅ được | Xác suất lớp (soft label) | Mở trên HF | 🟢 EMOS/CAT |
| **audeering wav2vec2-MSP-dim** | **VAD** (emotion2vec không làm được) | ~0.2B · ✅ dễ | V/A/D (0–1) + embedding 768 | ⚠️ CC-BY-NC (phi thương mại) | 🟢 **Mở 3 cột VAD** |
| **tiantiaf WavLM-large-MSP-dim** | VAD, mạnh hơn audeering | ~0.3B · ✅ được | V/A/D | kiểm tra lại | 🟢 VAD |
| **C²SER** (2025) | Vượt Qwen2-Audio, SOTA WA/UA/F1 | Lớn · 🟡 nặng | Nhãn/text | Mở | 🟡 tham khảo (Related Work) |
| **MERaLiON-SER** (11/2025) | Đa ngôn ngữ (Anh + Đông Nam Á), bền vững | Lớn · 🟡 nặng | Nhãn + attribute | Mở | 🟡 nếu cần đa ngữ |
| **Qwen2-Audio / AudioLLM reasoning** | Vượt emotion2vec+ large trên MELD (44.7%→53%) | **7B+ · 🔴 khó nổi T4** | **Chỉ text** | Mở | 🔴 nặng + chỉ ra nhãn |

### Benchmark IEMOCAP (paper emotion2vec, cột WA — để so backbone)
- **Base** (chỉ gắn 1 linear): wav2vec2 63.43 · HuBERT 64.92 · WavLM 65.94 · data2vec2.0 68.58 · **emotion2vec 71.79** (cao nhất nhóm base).
- **Large:** wav2vec2 65.64 · HuBERT 67.62 · WavLM 70.03.
- 👉 emotion2vec (base) **nhỉnh hơn cả WavLM-large** ở cảm xúc rời rạc → "mới/to hơn ≠ tốt hơn"; emotion2vec thắng vì **pre-train chuyên cảm xúc** (chuyên khoa) còn WavLM là SSL tổng quát (đa khoa).

### Kết luận để chốt hướng
1. **Khả thi trên T4, đáng thử:** `Whisper-large-v3 encoder` (CAT/EMOS) + `WavLM/audeering MSP-dim` (VAD).
2. **Khoảng trống thật = VAD** (emotion2vec không xuất V/A/D), KHÔNG phải độ chính xác cảm xúc → đây mới là lý do thêm model mới.
3. **Audio-LLM (Qwen2/C²SER/MERaLiON):** chỉ ghi Related Work — nặng + chỉ ra nhãn (vướng đúng vấn đề EMOS cần điểm liên tục, giống SenseVoice).
4. **Ablation cho paper:** emotion2vec vs Whisper-large-v3 (EMOS) + MSP-dim (VAD).

**Nguồn:** EmoBox leaderboard (emo-box.github.io/leaderboard1.html) · SAILER arXiv 2505.22133 · C²SER arXiv 2502.18186 · MERaLiON-SER arXiv 2511.04914 · audeering MSP-dim (HF) · emotion2vec arXiv 2312.15185.

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

---

## ⭐ Ý tưởng chủ lực: FUSION multi-task (chốt 4/6/2026)
> Áp dụng đúng "công thức thắng cuộc 2024" (T05 / PS-SQA / LE-SSL-MOS đều **fusion**) vào Track 2, có **bằng chứng thực nghiệm** từ chính kết quả của mình.

**🔑 Bằng chứng để fusion ăn điểm (đo được 4/6):** không model SER đơn lẻ nào tối ưu mọi chiều cảm xúc —
- **emotion2vec** thắng cột **EMOS** (0.637, exp01) nhưng **không có VAD**.
- **SAILER** (WavLM-large) thắng **VAD** (ARO 0.712 / DOM 0.630, exp03) nhưng EMOS thấp hơn (0.562).
→ Hai model nhìn cảm xúc theo **góc bổ sung nhau** (không trùng lặp) → đúng điều kiện để **gộp lại mạnh hơn cả hai**.

**Kiến trúc đề xuất (fusion + multi-task):**
```
audio ─┬─► emotion2vec ─► embedding A ─┐
       ├─► SAILER      ─► embedding B ─┼─► concat (+ one-hot cảm xúc target) ─► MLP head ─┬─► QMOS
       └─► (WavLM/SSL) ─► embedding C ─┘                                                   ├─► EMOS
                                                                                          ├─► CAT
                                                                                          └─► VAD
```
- Nối (concat) embedding emotion2vec + SAILER (+ tùy chọn WavLM) → 1 vector giàu thông tin.
- Train **head multi-task** trên 12.746 nhãn thật (gộp TB theo wav). Đây là **exp02 mở rộng** (thêm SAILER vào fusion).
- EMOS cần cả audio LẪN target → feed thêm one-hot cảm xúc target (từ `metadata.csv`).

**Phụ trợ (theo pattern 2024):**
- **QMOS** (đang kẹt 0.414): thêm luồng **spectrogram-as-image** (EfficientNet) như T05 — chiêu vô địch Track 1 2024.
- **Bias correction** + **ensemble nhiều seed** ở bước cuối để vắt thêm SRCC.

**Câu chuyện novelty cho ICASSP 2027:**
> *"Không một SER model đơn lẻ nào tối ưu mọi chiều cảm xúc (emotion2vec giỏi EMOS, SAILER giỏi arousal). Chúng tôi **fuse** chúng trong một model multi-task → thắng cả cụm."*
→ Novelty **có bằng chứng thực nghiệm** (bảng so sánh exp01 vs exp03), an toàn để viết.

**Bước tiếp:** code fusion ở session mới (context sạch) → chạy Kaggle → so với mốc EMOS 0.637.

**✅ Khả thi trên Kaggle T4 (16GB) — chốt 4/6:** khả thi nhờ chiến lược **freeze backbone + cache embedding + chỉ train head nhỏ**.
- VRAM: chạy **từng backbone một** (xong giải phóng) → mỗi WavLM-large ~300M ≈ 1–2GB → thừa trong 16GB.
- Việc nặng = **trích đặc trưng 1 LẦN** (~30–60 phút/model × 2–3 model ≈ 1,5–2,5h) → lưu `.npz` (~150MB, mean-pool 1 vector/wav).
- Train head sau đó **vài phút/lần thử** (chỉ ăn vector) → thử siêu tham số thoải mái, khỏi chạy lại backbone. 1 phiên Kaggle 12h dư sức.
- ⚠️ **Chỉ né fine-tune backbone end-to-end** lúc đầu (đó mới là chỗ T4 chật → cần gradient checkpointing / LoRA / batch nhỏ). Giai đoạn đầu đóng băng là đủ kiểm chứng fusion.

**🎯 THIẾT KẾ ĐẦU RA — CHỐT 4/6: "QMOS riêng + 5 cảm xúc chung"**
> Lý do: QMOS đo **chất lượng/độ tự nhiên** (khác trục với cảm xúc) → ép vào head cảm xúc dễ **negative transfer**. 5 cột EMOS/CAT/VAL/ARO/DOM cùng **không gian cảm xúc** → multi-task giúp nhau.
```
                    ┌─► QMOS   (nhánh CHẤT LƯỢNG riêng: giữ SpeechMOS / head riêng)
audio ─► backbone ──┤
   (fusion emotion) └─► head cảm xúc multi-task ─┬─► EMOS ├─► CAT ├─► VAL ├─► ARO └─► DOM
```
- **QMOS:** tách riêng, không chung backbone cảm xúc (mặc định giữ SpeechMOS 0.414; có thể train head chất lượng riêng sau).
- **5 cột cảm xúc:** 1 backbone fusion (emotion2vec + SAILER + tùy chọn WavLM) → head multi-task 5 đầu ra.
- ⚠️ **Khó nhất = CÂN LOSS** (6 loss khác loại/thang: MSE cho EMOS/VAD, cross-entropy/KL cho CAT) → dùng trọng số thủ công hoặc uncertainty weighting; tránh để 1 loss át.
- 🛡️ **Lưới an toàn:** `answer.txt` KHÔNG cần 6 cột từ 1 model → worst case điền mỗi cột bằng model tốt nhất (bản lai). Multi-task không bao giờ tệ hơn hiện tại.
- **Quy trình đo từng bước:** (1) multi-task 5 cảm xúc, giữ QMOS=SpeechMOS → so 0.637 (EMOS)+VAD; (2) cột nào thua bản đơn lẻ → quay lại bản đơn lẻ cột đó.

**📌 Bối cảnh & prior art (chốt 4/6 — CẦN web-search xác minh trước khi viết paper):**
- **Cảm xúc là track MỚI của VoiceMOS 2026** (2022–2024 chỉ có chất lượng/naturalness/singing) → ít tiền lệ → lợi thế novelty. *(Nói "mới trong bối cảnh MOS cảm xúc", KHÔNG nói "chưa ai làm cảm xúc" — tránh reviewer bắt lỗi.)*
- Fusion cho **nhận diện cảm xúc** đã RẤT nhiều: multimodal A+T+V (Tensor Fusion Network 1707.07250, MulT 1906.00295), multi-feature SER (SSL+prosody+spectrogram), gộp nhiều SSL backbone → **ý tưởng kỹ thuật không mới**.
- Fusion cho **EMOS (điểm MOS cảm xúc chủ quan)** gần như **chưa ai làm** → đây là khoảng trống. ⚠️ **Chưa chắc 100% → phải search literature** trước khi khẳng định.
- **Metric VoiceMOS chuẩn (mọi năm):** MSE / LCC (Pearson) / SRCC / KTAU, ở 2 cấp utterance + system. 2024: Track1/2 xếp hạng theo SRCC, Track3 theo LCC. 2026 Track 2: UTT-SRCC (5 cột) + CAT-ERR.

**Link học fusion:** survey Baltrušaitis 1705.09406 (early/late/hybrid) · UTMOS 2204.02152 · SSL-MOS 2110.02635 · đội thắng 2024: 2409.07001 (tổng kết) / 2409.09305 (T05) / 2411.11123 (PS-SQA) / 2311.10656 (LE-SSL-MOS). *(ID nhớ từ kiến thức — kiểm lại khi trích.)*

**🔎 KẾT QUẢ WEB-SEARCH prior art (4/6) — BUỘC ĐỊNH VỊ LẠI NOVELTY:**
- ⚠️ "Fuse nhiều model để đoán cảm xúc" **KHÔNG mới**:
  - **Fusion SSL cho MOS** đã có: *Fusion of Self-supervised Learned Models for MOS Prediction* (**arXiv 2204.04855**) → đối chứng BẮT BUỘC trích Related Work.
  - **Fusion cho SER** rất nhiều: ensemble HuBERT+wav2vec2+WavLM; attention-fusion wav2vec2+prosody (2104.03502, 2411.02964).
  - **Đoán độ tương đồng cảm xúc giọng tổng hợp** cũng có: **SVAS** trong EmoSphere++ (**2411.02625**, SER→VAD→cosine sim); *Acoustic Similarity in Emotional Speech via SSL* (**2409.17899**) — nhưng kiểu reference-based / nội bộ TTS, KHÔNG train khớp điểm ý kiến người trên benchmark.
- ✅ **Khoảng trống còn lại (novelty hẹp, đứng được):**
  1. **EMOS prediction** (điểm MOS độ khớp cảm xúc target do người chấm) — task MỚI VoiceMOS 2026; chưa có nghiên cứu hệ thống.
  2. **Phát hiện thực nghiệm:** emotion2vec (cũ) **vượt** SAILER (SOTA SER) ở EMOS do SRCC chấm thứ hạng → có số liệu.
  3. **Tính bổ sung** giữa SER model theo từng chiều (emotion2vec↔EMOS, SAILER↔arousal) → động lực fuse, đo được.
  4. **Multi-task hợp nhất** EMOS+CAT+VAD trong bối cảnh đánh giá emotional TTS.
- 🎯 **Câu novelty nên viết:** "First systematic study of **EMOS prediction**; không SER đơn lẻ nào tối ưu mọi chiều; model cũ vượt SOTA do tính chất ranking → đề xuất bộ dự đoán **multi-task hợp nhất**." → Novelty ở **task + phát hiện + multi-task**, KHÔNG ở "fusion".
- ⚠️ Search mới 1 vòng → trước khi nộp phải đọc kỹ 2411.02625 + 2409.17899 để tránh trùng.

---

## 📊 Ghi chú phiên 5/6 (Phiên 8) — kết quả 2024/2025, dev↔eval, ensemble, ICASSP

### Kết quả đội thắng VoiceMOS 2024 (từ 2409.07001, bản HTML — ⚠️ số đọc nhanh, lệch tên đội T05/T06, kiểm PDF khi trích)
- **Track 1 (zoomed-in TTS):** đội đầu **UTT-SRCC ~0.676** · system-SRCC ~0.943 / LCC ~0.931 / KTAU ~0.793 (nhất 9/16 metric). Baseline system-SRCC ~0.745.
- **Track 2 (singing):** đội đầu UTT **SRCC ~0.625 / LCC ~0.637**; ⚠️ **không ai vượt baseline system-SRCC ~0.859**.
- **Track 3 (semi-sup, data ít):** LCC SIG ~0.297 · OVRL ~0.713 · BAK ~0.867.
- 🔑 **Bài học:** UTT-level vốn rất khó (~0.62–0.68 là vô địch); system-level cao hơn nhiều — **đừng so điểm utt-level của mình với system-level**. → các cột cảm xúc của mình (EMOS 0.811…) là cao so mặt bằng utt-level.

### dev ↔ eval: thường lệch, UTT-level lệch mạnh hơn system-level
- Gần như luôn có khoảng **tụt sang eval** (eval = hệ TTS/listener/domain chưa thấy). 2024 cố tình tạo lệch (dev nhóm zoom 50% vs eval 25%+12%).
- Metric chính 2026 = **UTT-SRCC** → nhạy → **phải dự phòng eval < dev**.
- ⚠️ **exp08 (fine-tune) rủi ro tụt hơn exp07 (đóng băng)** vì dễ overfit dev → giữ exp07 làm fallback *cả về độ ổn định*.
- Giảm gap: **ensemble** (giảm variance) · chọn bản final theo **robustness > đỉnh dev** · dùng val nội bộ.

### Ensemble — "công thức thắng cuộc" (AudioMOS 2025)
- **T09** (vô địch T1): **9 model** = 5 seed (smoothed CE) + 2 rank/ordinal loss + 2 chỉnh nhẹ.
- **T12** (vô địch T2): **5 model** (4 KAN + 1 VERSA). Ensemble dùng bởi 8/24 đội nhưng **mọi đội đầu đều dùng**. Cỡ điển hình **5–9 model**.
- 🔑 **Cùng kiến trúc + khác SEED rồi trung bình = đã là ensemble hợp lệ** (seed đổi init/shuffle/dropout → lỗi độc lập → triệt tiêu khi trung bình). Không bắt buộc model khác loại.
- Thang đa dạng: (1) cùng model khác seed → (2) đổi loss/hyperparam → (3) khác kiến trúc (lợi nhất). **Mình có sẵn mức 3 free: exp07 (đóng băng) ↔ exp08 (fine-tune)**.
- **Phân biệt 2 chữ "fusion":** *feature fusion* (exp04/07 — 2 backbone → 1 trunk → **1 model**, gộp ở mức đặc trưng) ≠ *ensemble/late fusion* (**nhiều model** hoàn chỉnh → trung bình điểm). Hai cái **bổ sung**, nên làm cả hai. Hiện mình mới có feature fusion, **chưa ensemble**.
- ➡️ Kế hoạch: exp08 train **3 seed → trung bình** + gộp exp07 cho cảm xúc (+ tùy chọn 1 bản rank-loss). Lưu ý exp08 fine-tune không cache → mỗi seed tốn nhiều giờ T4 → 3 seed là thực tế. **Trung bình điểm rồi mới xếp hạng** (SRCC).

### ICASSP 2027 (xác nhận qua CFP 5/6) — mốc & quy mô
- **Hạn full paper 16/9/2026** · acceptance 13/1/2027 · camera-ready 27/1/2027 · **hội nghị 16–21/5/2027, Toronto** (trang CFP in nhầm năm "2027" cho hạn nộp — sai).
- Quy mô: flagship IEEE SPS, **CORE A**; 2024 ~5.796 nộp / ~2.812 nhận (**~45%**); 2025 nhận >3.300; dự ~3–4k người. Không "mời" — được trình bày nếu **bài được nhận** + đăng ký + bay sang (no-show → rút khỏi Xplore). Cần visa Canada; có thể xin **student travel grant**.
- Tham gia challenge thường được **trích dẫn trong paper tổng kết BTC** (như 2409.07001) → chỉ cần nộp eval + system description đã có mặt trong 1 publication.
