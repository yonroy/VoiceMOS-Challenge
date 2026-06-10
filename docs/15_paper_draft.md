# 15 — Bản nháp Paper (ICASSP 2027)

> Cập nhật ngày: 8/6/2026.
>
> ⚠️ **Đây là bản nháp tiếng Việt để TƯ DUY.** Bản nộp ICASSP **bắt buộc bằng TIẾNG ANH**. Viết ý bằng tiếng Việt trước cho chắc, dịch sau.
>
> 🎯 **Cách dùng file này — "vừa làm vừa viết":** mỗi khi chạy xong 1 experiment trong [04_experiments_log.md](04_experiments_log.md) → đổ kết quả + nhận xét sang đúng mục dưới đây ngay. Đừng để cuối mới viết.
>
> **Mốc thời gian:** kết quả nộp 7/8/2026 · **hạn paper ICASSP 2027: 16/9/2026** (chỉ ~6 tuần sau kết quả → phải viết song song).
> ✅ *Đã xác nhận qua CFP chính thức (5/6/2026):* full paper **16/9/2026** · acceptance 13/1/2027 · camera-ready 27/1/2027 · hội nghị **16–21/5/2027, Toronto, Canada**. (Trang CFP in nhầm năm "September 16, 2027" — sai, vì acceptance là 1/2027.)

---

## 📋 Trạng thái viết (tick khi xong)

| Mục | Trạng thái | Nguồn dữ liệu |
|---|---|---|
| Tiêu đề + tác giả | ⬜ nháp | — |
| Abstract | ⬜ (viết cuối cùng) | — |
| 1. Introduction | 🟡 có khung | [07](07_project_summary.md), [08](08_track2_spec.md) |
| 2. Related Work | 🟢 nháp xong (EN ở [19](19_paper_v1_en.md) §2) — cần kiểm arXiv ID | [03_literature_notes.md](03_literature_notes.md) |
| 3. Method | 🟢 fusion (exp07) + fine-tune (exp08) | [12](12_system_description.md) Track 2 |
| 4. Experimental Setup | 🟡 có khung | [08](08_track2_spec.md), [14](14_leaderboard_metrics.md) |
| 5. Results & Ablation | 🟢 kết quả thật exp01/03/04/07/08 | [04_experiments_log.md](04_experiments_log.md), [18](18_leaderboard_history.md) |
| 6. Conclusion | ⬜ (viết cuối) | — |
| Hình kiến trúc | ⬜ BẮT BUỘC | — |

> 🆕 **Cập nhật 5/6:** kết quả tốt nhất hiện tại = **trộn cột**: QMOS 0.548 (exp07, fusion 6-head đóng băng) + 5 cột cảm xúc từ **exp08 (fine-tune WavLM warm-start SAILER)**. exp08 **thắng cả 5 cột cảm xúc** so với exp07 (EMOS 0.811, CAT err 0.133, VAD 0.659/0.793/0.751). Câu chuyện paper mở rộng: *(i)* fusion 2 biểu diễn SSL đóng băng → 1 model 6 cột (exp07); *(ii)* fine-tune backbone đẩy thêm cảm xúc (exp08). Số liệu theo ngày: [18_leaderboard_history.md](18_leaderboard_history.md).

---

## ✍️ Tiêu đề (nháp các phương án)
- *A Unified Multi-Task Model for Emotional Speech MOS Prediction* (nhấn "1 model thay cụm 3 model")
- *Beyond Patchwork Baselines: Joint Quality and Emotion MOS Prediction for Emotional TTS*
- *Fusing Complementary SSL Emotion Representations for Multi-Task Emotional MOS Prediction* (nhấn phát hiện emotion2vec + SAILER bổ sung nhau — sát exp04)
- [thêm phương án...]

**Tác giả / đơn vị:** [điền — Tran Minh Toan + mentor (co-author chờ xác nhận)]

---

## Abstract  *(viết SAU CÙNG — ~150 từ, 5 câu)*
> Công thức 5 câu: (1) bối cảnh bài toán → (2) khoảng trống/điểm yếu của baseline → (3) cách của bạn → (4) kết quả chính (số) → (5) ý nghĩa.

[ ] điền (giờ đã đủ số để viết — dùng kết quả tốt nhất exp07 + exp08). Bản nháp ý 5 câu:
> (1) MOS tự động là chuẩn đánh giá TTS, Track 2 VoiceMOS 2026 mở rộng sang giọng **cảm xúc** (QMOS + EMOS + CAT + VAD). (2) Baseline là cụm 3 model rời (UTMOS + emotion2vec + LLM zero-shot) → EMOS/VAD yếu, không tận dụng nhãn. (3) Chúng tôi hợp nhất **2 biểu diễn SSL cảm xúc bổ sung nhau** (emotion2vec + SAILER) qua trunk chung + head multi-task hợp nhất **trọn 6 cột trong 1 model**, rồi **fine-tune backbone WavLM (warm-start SAILER)** để đẩy thêm các cột cảm xúc. (4) Trên DEV CodaBench: EMOS SRCC 0.194→**0.811**, VAL 0.34→**0.66**, ARO→**0.79**, CAT err 0.19→**0.13**, QMOS 0.41→**0.55** — **vượt mọi model lẻ ở cả 6 cột**. (5) Cho thấy *fusion biểu diễn cảm xúc + fine-tune có giám sát* hiệu quả hơn hẳn *chắp vá nhiều model zero-shot* cho đánh giá MOS cảm xúc.

---

## 1. Introduction

> Mục tiêu: thuyết phục người đọc bài toán này **quan trọng** và **chưa được giải tốt**.

**Khung 4 đoạn:**

1. **Bối cảnh:** MOS (Mean Opinion Score) là chuẩn đánh giá chất lượng giọng tổng hợp; chấm bằng người tốn kém → cần **mô hình tự động dự đoán MOS**. VoiceMOS Challenge là sân chơi chuẩn cho hướng này.
2. **Cái mới của 2026 — Track 2 (Emotional TTS):** không chỉ chấm *chất lượng* (QMOS) mà còn *độ khớp cảm xúc target* (EMOS), phân bố cảm xúc (CAT), và Valence/Arousal/Dominance (VAD). → khó hơn vì phải hiểu **cả chất lượng lẫn cảm xúc**.
3. **Điểm yếu của baseline (động lực của bài):** baseline chính thức là **"chắp vá 3 model rời"** — UTMOS (QMOS) + emotion2vec (CAT) + Gemini LLM-as-judge (EMOS/VAD). Hệ quả: (a) EMOS/VAD zero-shot → yếu (EMOS chỉ ~0.19 SRCC); (b) tốn chi phí API; (c) không tận dụng 12.746 mẫu **có nhãn** sẵn có.
4. **Đóng góp của chúng tôi (cập nhật 5/6):**
   - (C1) **Phát hiện thực nghiệm:** hai biểu diễn SSL cảm xúc **bổ sung nhau** — emotion2vec mạnh EMOS (0.637), SAILER/WavLM mạnh VAD (ARO 0.712) — nên *fusion* hơn hẳn việc chỉ chọn 1 model mạnh nhất.
   - (C2) **Mô hình fusion multi-task hợp nhất 6 cột (exp07):** 2 backbone **đóng băng** → nối đặc trưng → **trunk chung** → 4 head (QMOS dùng `[trunk|UTMOS]` / EMOS có điều kiện one-hot cảm xúc target / CAT soft-CE / VAD), cân loss bằng **uncertainty weighting**. → **1 model trọn 6 cột**, lần đầu cải thiện QMOS (0.414→0.548) mà không kéo tụt cảm xúc (không negative transfer).
   - (C3) **Fine-tune có giám sát đẩy thêm cảm xúc (exp08):** mở băng 6 lớp Transformer trên của **WavLM warm-start từ SAILER** + nhánh audeering đóng băng → vượt mọi cấu hình đóng băng ở cả 5 cột cảm xúc (EMOS 0.811, CAT 0.133, VAD 0.659/0.793/0.751). → fine-tune phá trần của head-only.
   - (C4) **Ablation** tắt từng nhánh (USE_E2V / USE_SAILER / USE_UNCERTAINTY / USE_CLASSPROB; frozen vs UNFREEZE_TOP_LAYERS) chứng minh mỗi thành phần đóng góp; và phân tích vì sao SRCC (thứ hạng) khiến hiện tượng "nén giá trị VAD" không gây hại.

> 💡 **Quan sát của user (5/6) — góc novelty TIỀM NĂNG "practical/efficient MOS":** xu hướng hiện nay (VoiceMOS/AudioMOS) là **đua độ chính xác bằng model khủng + ensemble 5–9 model**, **bỏ qua latency & tính ứng dụng thực tế** — vì challenge chỉ chấm correlation, không chấm chi phí. Khoảng trống ít người chạm. Hệ của mình **chạy offline trên 1 T4, không API** → có thể định vị "**đạt chất lượng cạnh tranh với chi phí/inference thấp**": báo cáo **#params / thời gian inference / GPU-giờ** cạnh SRCC. *Chiến lược cân bằng:* vẫn ensemble vừa phải để giữ hạng challenge, nhưng kể câu chuyện "vừa tốt vừa rẻ" trong paper (reviewer ICASSP thích tính ứng dụng). → cân nhắc thành **C5** + 1 bảng accuracy-vs-cost.

> 💡 Ghi chú: C1–C3 là định vị novelty sau khi web-search prior art (4/6, xem [03](03_literature_notes.md)) — fusion-cho-MOS không hoàn toàn mới, nhưng *fusion cho EMOS/VAD cảm xúc + phát hiện e2v>SAILER* là phần mới. Vẫn nên **chốt lại với mentor**.

---

## 2. Related Work

> ✅ **Đã viết bản tiếng Anh đầy đủ trong [19_paper_v1_en.md](19_paper_v1_en.md) §2** (7 đoạn + danh mục ref kèm arXiv ID). Dưới đây là tóm tắt tiếng Việt để tư duy + định vị novelty. ⚠️ Mọi arXiv ID **phải kiểm lại** trước khi nộp.

**7 đoạn (mỗi đoạn 2–4 câu, dẫn về bài của mình):**

1. **MOS prediction tự động:** MOSNet (đầu tiên, DNN regression) → LDNet, SSL-MOS → VoiceMOS Challenge chuẩn hóa benchmark; **UTMOS / UTMOSv2** (vô địch chất lượng 2022/2024, SSL + tín hiệu phụ + ensemble) là baseline QMOS. → *Mọi paper trên chỉ chấm 1 trục chất lượng; Track 2 2026 mới thêm trục cảm xúc (EMOS/CAT/VAD).*
2. **SSL backbone:** wav2vec 2.0 / HuBERT / **WavLM** (pretrain 94k giờ, mạnh cho paralinguistic) — mình dùng WavLM-large vừa làm feature đóng băng vừa làm backbone fine-tune.
3. **Biểu diễn cảm xúc:** **emotion2vec** (SSL cảm xúc, baseline CAT, KHÔNG có VAD) · **SAILER / Vox-Profile** (WavLM SER, có class-prob + VAD) · **audeering MSP-Dim** (ra thẳng VAD). → *Phát hiện chúng BỔ SUNG nhau theo từng trục → động lực fusion.*
4. **Fusion & ensemble cho đánh giá giọng:** đội thắng VoiceMOS/AudioMOS hay fusion (T05 SSL+spectrogram-ảnh, PS-SQA pitch/codec, ensemble 5–9 model); *Fusion of SSL models for MOS* (2204.04855). → *Họ fusion để đoán CHẤT LƯỢNG; mình fusion biểu diễn CẢM XÚC + mở multi-task sang trục cảm xúc.*
5. **Đánh giá TTS cảm xúc / độ tương đồng cảm xúc:** EmoSphere++ (SVAS, 2411.02625), Acoustic similarity via SSL (2409.17899) — đều là **reference/embedding-similarity nội bộ TTS**, KHÔNG train khớp điểm người trên benchmark. → *Dự đoán EMOS do người chấm + đồng thời CAT/VAD (đúng Track 2) gần như chưa ai làm hệ thống → khoảng trống của mình.*
6. **Sequence model & LLM-judge:** Mamba (SSM tuyến tính), MambaRate/HighRateMOS (AudioMOS 2025) — mình thử **Mamba head thay pooling** (exp14/15); audio-LLM judge (ALLD ICLR 2025, SpeechQualityLLM) linh hoạt nhưng chưa hiệu chỉnh + tốn API, còn hệ mình chạy offline (exp16 là khảo sát).
7. **Multi-task learning:** chia sẻ encoder cho nhiều task; khó nhất là cân 6 loss khác thang → dùng **uncertainty weighting** (Kendall CVPR 2018) thay vì chỉnh tay.

> 🎯 **Câu novelty chốt (an toàn theo web-search 4/6):** novelty nằm ở **(a) task EMOS mới**, **(b) phát hiện thực nghiệm e2v↔SAILER bổ sung nhau / model cũ vượt SOTA do SRCC chấm thứ hạng**, **(c) multi-task hợp nhất 6 cột** — KHÔNG khẳng định "fusion là mới" (đã có 2204.04855). Trước khi nộp phải đọc kỹ 2411.02625 + 2409.17899 để tránh trùng.

---

## 3. Method (Hệ thống đề xuất)

> Đây là phần lõi. Phải mô tả đủ để người khác **tái lập**. **BẮT BUỘC có hình kiến trúc** (guideline BTC).

### 3.1 Tổng quan
- Input: 1 đoạn waveform (16 kHz) + thông tin cảm xúc target (cho EMOS).
- Output: QMOS, EMOS, CAT (5 lớp), VAD (3 trục).

### 3.2 Kiến trúc fusion hợp nhất 6 cột (chốt theo exp07)
- **Hai backbone SSL đóng băng (không fine-tune):**
  - **emotion2vec** (`iic/emotion2vec_plus_large`) → embedding + xác suất 5 lớp cảm xúc.
  - **SAILER** (`tiantiaf/wavlm-large-categorical-emotion`, WavLM-large, vô địch Interspeech 2025 SER) → embedding + xác suất 9 lớp + VAD 3 trục.
- **Nối đặc trưng:** `[e2v_emb | e2v_probs5 | sailer_emb | sailer_probs9 | sailer_vad3]` (cờ `USE_CLASSPROB` bật/tắt phần xác suất để ablation).
- **Trunk chung:** Linear→ReLU ×2 (TRUNK_HIDDEN 512, dropout 0.3) → vector đặc trưng dùng chung.
- **4 head multi-task (hợp nhất 6 cột):**
  - Head **QMOS** (exp07): đầu vào `[trunk | điểm UTMOS]` → regression (UTMOS làm neo residual) → cải thiện QMOS 0.414→**0.548** mà không hại cảm xúc.
  - Head **EMOS**: nối thêm **one-hot cảm xúc target** vào đặc trưng → regression 1 điểm (EMOS phụ thuộc cả audio lẫn target).
  - Head **CAT**: 5 logits → softmax (phân bố vote 5 lớp).
  - Head **VAD**: regression 3 giá trị.
- **[HÌNH 1: sơ đồ kiến trúc]** ← vẽ sau: 2 backbone đóng băng → concat → trunk → 4 head (QMOS/EMOS/CAT/VAD); QMOS thêm đầu vào điểm UTMOS.

### 3.3 Hàm mất mát (loss)
- Multi-task chỉ trên 3 head cảm xúc: EMOS/VAD = **MSE**, CAT = **soft cross-entropy** (nhãn mềm = tỉ lệ vote).
- **Cân loss = uncertainty weighting** (Kendall et al.): học log σ² riêng cho từng task thay vì chọn trọng số tay → `L = Σ_i (1/2σ_i²)·L_i + log σ_i`. Cờ `USE_UNCERTAINTY` tắt được (dùng trọng số tay khi debug).
- Nhãn liên tục (EMOS/VAD) **z-score** để các MSE cùng thang trước khi cộng.
- Nhãn vàng gộp **theo wavID**: EMOS = TB `eMOS` · VAD = TB `val/aro/dom` · CAT = tỉ lệ vote 5 lớp `emoCat` (đa nhãn).

### 3.4 Biến thể fine-tune cho cảm xúc (exp08 — cho kết quả cảm xúc tốt nhất)
- Động lực: head-only trên backbone đóng băng có **trần** (backbone học cho task khác). Fine-tune phá trần.
- **Backbone:** WavLM-large **warm-start từ SAILER** (đã giỏi cảm xúc) — **mở băng 6 lớp Transformer trên**, đóng băng feature-extractor + các lớp dưới; nhánh phụ **audeering MSP-dim đóng băng** (bổ trợ valence). Bỏ emotion2vec ở biến thể này (funasr khó fine-tune).
- **Kỹ thuật T4 (16GB):** AMP fp16 · gradient checkpointing · `BATCH 4 × ACCUM 8` (hiệu dụng 32) · `MAX_SECONDS 8` · LR backbone 1e-5 / head 1e-3 · early-stop theo TB SRCC val.
- **Kết quả:** thắng cả 5 cột cảm xúc vs biến thể đóng băng (xem mục 5.1). QMOS vẫn lấy từ exp07 (head fusion) → **hệ cuối = trộn cột**.
- ⚠️ **Lưu ý tái lập (bài học):** checkpoint fine-tune phải lưu **cả backbone** (`ft_emotion_full.pt`), không chỉ head — bản đầu chỉ lưu head nên mất backbone khi kernel chết.
- **[HÌNH 2 (tùy chọn): sơ đồ biến thể fine-tune]**

---

## 4. Experimental Setup

### 4.1 Dataset (Track 2)
- Thu mới, nền **ESD + DailyTalk** + 13 hệ thống TTS.
- **Train 12,746 · Val 2,730 · Eval 2,730.** 5 cảm xúc: neutral/happy/angry/sad/surprised.
- External data đã dùng: [khai báo — ESD, DailyTalk; pre-trained SSL checkpoint + link]. Eval set công bố 31/7/2026.

### 4.2 Metric
- **UTT-SRCC** (Spearman, utterance-level, cao=tốt) cho QMOS/EMOS/VAD.
- **Categorical error** cho CAT (tổng `|gt−pred|` / tổng label, thấp=tốt).
- (BTC còn báo MSE/LCC/KTAU ở utt + system level — nêu nếu cần.)
- Chi tiết: [14_leaderboard_metrics.md](14_leaderboard_metrics.md).

### 4.3 Cấu hình train (exp04)
- Backbone: **đóng băng** (emotion2vec + SAILER), chỉ train trunk + 3 head.
- TRUNK_HIDDEN 512 · HEAD_HIDDEN 128 · DROPOUT 0.3 · LR 1e-3 · BATCH 64 · EPOCHS 80 · early-stop theo **TB SRCC val nội bộ**.
- **Cache đặc trưng `.npz` riêng từng backbone** (trích 1 lần ~12–15 phút trên T4, resume mỗi 500 file) → train head chỉ vài phút. GPU: Kaggle T4.
- [Còn để ngỏ: augmentation; bias correction; có nên fine-tune nhẹ backbone cho QMOS không].

---

## 5. Results & Ablation

### 5.1 Kết quả chính (DEV — CodaBench)

| Hệ thống | QMOS↑ | EMOS↑ | CAT err↓ | VAL↑ | ARO↑ | DOM↑ |
|---|---|---|---|---|---|---|
| **Baseline** (UTMOS+e2v+Gemini) | 0.414 | 0.194 ⚠️ | 0.193 | — | — | — |
| exp01 (EMOS←emotion2vec, zero-shot) | 0.414 | 0.637 | 0.193 | — | — | — |
| exp03 (SAILER 1 model, zero-shot) | 0.414 | 0.562 | 0.190 | 0.341 | 0.712 | 0.630 |
| exp04 (FUSION 5-head, đóng băng) | 0.414 | 0.788 | 0.145 | 0.578 | 0.754 | 0.706 |
| exp07 (FUSION+QMOS 6-head, đóng băng) | **0.548** | 0.795 | 0.153 | 0.581 | 0.752 | 0.705 |
| exp08 (FINE-TUNE WavLM, cảm xúc) | 0.414¹ | **0.811** | **0.133** | **0.659** | **0.793** | **0.751** |
| **➡️ Đề xuất = trộn cột (QMOS←exp07 · cảm xúc←exp08)** | **0.548** | **0.811** | **0.133** | **0.659** | **0.793** | **0.751** |
| Eval set (cuối) | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |

> ¹ exp08 QMOS rớt 0.414 vì **bản nộp không kèm answer.txt exp07** → rơi về fallback UTMOS (lỗi ghép file, **không** phải model kém). Hệ đề xuất lấy QMOS 0.548 từ exp07.
> 📌 **Hệ đề xuất = trộn cột** (best-per-column): tiến trình hai bước — (a) **exp07** fusion 6-head đóng băng cho QMOS 0.548 (lần cải thiện QMOS đầu tiên, **không negative transfer**); (b) **exp08** fine-tune 6 lớp trên của WavLM (warm-start SAILER) đẩy **cả 5 cột cảm xúc** vượt mọi cấu hình đóng băng (EMOS 0.795→0.811, CAT 0.153→0.133, VAL 0.581→0.659, ARO 0.752→0.793, DOM 0.705→0.751). Số đậm = tốt nhất từng cột.
> ⚠️ Baseline EMOS 0.194 là submission **một phần** (chỉ 496/2730 mẫu Gemini thật) → khi so sánh trong paper phải ghi rõ điều kiện.

### 5.2 Ablation (chứng minh từng thành phần — exp04 có sẵn 4 cờ)

> Cờ trong code: `USE_E2V` / `USE_SAILER` / `USE_UNCERTAINTY` / `USE_CLASSPROB`. **Chưa chạy đủ các dòng** — cần chạy để điền (todo phiên sau).

| Cấu hình | EMOS↑ | CAT err↓ | VAL↑ | ARO↑ | DOM↑ | Mục đích chứng minh |
|---|---|---|---|---|---|---|
| **Full frozen (e2v + SAILER, uncertainty) — exp04** | 0.788 | 0.145 | 0.578 | 0.754 | 0.706 | hệ fusion đóng băng |
| **Fine-tune 6 lớp WavLM (warm-start SAILER) — exp08** | **0.811** | **0.133** | **0.659** | **0.793** | **0.751** | **frozen vs fine-tuned** (fine-tune thắng cả 5 cột) |
| − SAILER (chỉ emotion2vec) | [ ] | [ ] | [ ] | [ ] | [ ] | SAILER đóng góp VAD? |
| − emotion2vec (chỉ SAILER) | [ ] | [ ] | [ ] | [ ] | [ ] | e2v đóng góp EMOS? |
| − uncertainty (trọng số tay) | [ ] | [ ] | [ ] | [ ] | [ ] | cân loss tự động có lợi? |
| − class-prob (chỉ embedding) | [ ] | [ ] | [ ] | [ ] | [ ] | xác suất lớp có cần? |
| EMOS zero-shot (emotion2vec lẻ) | 0.637 | — | — | — | — | so head fusion có train |
| VAD zero-shot (SAILER lẻ) | — | — | 0.341 | 0.712 | 0.630 | so fusion |

### 5.3 Phân tích
- **Vì sao fusion thắng:** emotion2vec và SAILER bắt **khía cạnh khác nhau** của cảm xúc (e2v mạnh phân biệt lớp → EMOS; SAILER mạnh trục liên tục → VAD); trunk chung học cách phối hợp → mọi cột đều lên.
- **SRCC chấm thứ hạng, không chấm giá trị:** dù VAD dự đoán bị "nén" quanh 2.5–3.6, **thứ tự** vẫn khớp người chấm nên VAL nhảy 0.341→0.578. Đây là điểm phân tích đáng đưa vào paper (chống trực giác).
- **Chi phí:** exp04 chạy **offline trên T4** (không gọi API) → rẻ hơn hẳn baseline EMOS/VAD dùng Gemini, lại đủ 2730 mẫu (Gemini chỉ kịp 496).
- **Fine-tune phá trần đóng băng:** mở băng 6 lớp trên của WavLM (exp08) thắng cấu hình đóng băng (exp04) ở **cả 5 cột cảm xúc** → khẳng định fine-tune có giám sát đáng giá hơn head-only, dù tốn compute hơn (mất cache, AMP + grad-ckpt + grad-accum để vừa T4).
- **Điểm yếu còn lại:** QMOS cải thiện lên **0.548** (exp07, head fusion) nhưng vẫn là cột yếu nhất tương đối; fine-tune cảm xúc (exp08) **không nhắm QMOS** → hướng tiếp: thay UTMOS bằng **UTMOSv2** làm neo / fine-tune riêng cho chất lượng.

---

## 6. Conclusion

> 3-4 câu: tóm tắt đóng góp + kết quả + hướng tương lai (vd: thêm data public IEMOCAP/MSP-Podcast, mở rộng VAD).

[ ] viết cuối.

---

## Phụ lục — Checklist nộp paper & system description
- [ ] Khai báo external data/resource cho cả 3 track (link tái lập)
- [ ] Đủ 3 hình kiến trúc (T1 URGENT-MOS, T2 hệ thống mình, T3 ECAPA)
- [ ] Mô tả đủ chi tiết để tái lập
- [ ] Điền đủ kết quả dev + eval
- [ ] **DỊCH SANG TIẾNG ANH**
- [ ] (Tùy chọn) arXiv + công khai code GitHub
- [ ] Đúng template + giới hạn trang ICASSP 2027

> Tham chiếu chéo: kết quả ← [04_experiments_log.md](04_experiments_log.md) · mô tả hệ thống ← [12_system_description.md](12_system_description.md) · metric ← [14_leaderboard_metrics.md](14_leaderboard_metrics.md).
