# 12 — System Description (nháp)

> Tài liệu BẮT BUỘC nộp cho BTC cuối challenge. Khung điền dần trong suốt quá trình làm.
>
> ⚠️ **Đây là bản nháp tiếng Việt để tư duy. Bản nộp cho BTC phải bằng TIẾNG ANH.**
>
> **Guideline BTC (bắt buộc tuân thủ):**
> 1. Khai báo rõ **có dùng external data/resource hay không**; nếu có → ghi pointer (link) hoặc mô tả đủ để tái lập.
> 2. Mô tả **kiến trúc hệ thống + chiến lược training + xử lý dữ liệu** đủ chi tiết để **tái lập (reproducible)**.
> 3. **Phải có HÌNH minh họa kiến trúc.**
> 4. Khuyến khích (không bắt buộc) đăng arXiv + open-source code.

---

## Thông tin chung
- **Team / tác giả:** [điền]
- **Đơn vị:** [điền]
- **Track tham gia:** Track 1, Track 2, Track 3
- **Tóm tắt 1 dòng:** Track 2 là hệ thống chính (tự phát triển); Track 1 & 3 dùng baseline chính thức.

---

# Track 1 — Speech Enhancement (ACR + CCR)

> Cách tiếp cận: **dùng baseline chính thức** (không phát triển hệ thống riêng).

## 1. Tổng quan hệ thống
- Hệ thống: **URGENT-MOS** (baseline BTC, dùng inference với checkpoint pre-trained).
- [Mô tả ngắn: chỉ chạy inference, không fine-tune / có chỉnh gì không]

## 2. Kiến trúc
- Multi-encoder (WavLM, Kimi-Audio, Qwen3-Omni, Audio Flamingo) → căn chỉnh + fusion → AMPM (ACR) + NCPM (CCR).
- **[HÌNH 1: sơ đồ kiến trúc URGENT-MOS]** ← cần chèn
- Tham khảo: arXiv 2601.18438.

## 3. Dữ liệu & external resources
- **Có dùng external resource:** ✅
  - Checkpoint pre-trained: `urgent-challenge/urgent-mos-f1c1m5dcorpus` (HuggingFace).
  - Dev data: `urgent-challenge/vmc2026-track1-dev` (HuggingFace).
- Không dùng data huấn luyện bổ sung (chỉ inference).

## 4. Chiến lược training
- Không train (zero-shot inference từ checkpoint có sẵn). [Xác nhận lại nếu có thay đổi]

## 5. Kết quả
| Set | ACR UTT-SRCC | CCR UTT-SRCC |
|---|---|---|
| Dev (3/6/2026) | 0.662 | 0.411 |
| Eval | [điền] | [điền] |

---

# Track 2 — Emotional TTS ⭐ (hệ thống chính)

> Cách tiếp cận: **phát triển hệ thống riêng**, bắt đầu từ baseline rồi cải tiến.

## 1. Tổng quan hệ thống
- Sub-task làm: QMOS + EMOS (bắt buộc) [+ CAT / VAD nếu có].
- Ý tưởng cốt lõi / novelty: [điền — vd: fusion SSL backbone + emotion embedding; multi-task QMOS+EMOS]

## 2. Kiến trúc
- Backbone: [WavLM / HuBERT / Wav2Vec2 — điền sau khi chốt]
- Đầu ra: [head dự đoán QMOS, EMOS, ...]
- **[HÌNH 2: sơ đồ kiến trúc hệ thống của mình]** ← BẮT BUỘC, cần chèn
- [Mô tả từng khối: input → feature → pooling → head → score]

## 3. Dữ liệu & external resources
- **Data challenge:** Track 2 (train 12.746 / val 2.730 / eval 2.730).
- **Có dùng external data:** [✅/❌ — khai báo rõ]
  - Dataset nền (BTC cung cấp): ESD, DailyTalk.
  - Public data bổ sung (nếu dùng): [IEMOCAP / MSP-Podcast / ... + link]
  - Pre-trained model: [WavLM checkpoint + link HuggingFace]
- Xử lý dữ liệu: [resample 16kHz, cắt/pad, normalize, ...]

## 4. Chiến lược training
- Loss: [vd MSE cho QMOS/EMOS; multi-task weighting]
- Optimizer / LR / batch size / epochs: [điền]
- Kỹ thuật: [augmentation / fine-tune toàn bộ hay freeze backbone / bias correction / ensemble]

## 5. Kết quả
| Set | QMOS SRCC | EMOS SRCC | CAT err | VAL/ARO/DOM SRCC |
|---|---|---|---|---|
| Baseline (DEV, 3/6/2026) | 0.414 | 0.194 ⚠️ | 0.193 | — (bỏ VAD) |
| exp03 SAILER (DEV, 4/6/2026) | 0.414 | 0.562 | 0.190 | 0.341 / 0.712 / 0.630 |
| **exp01 emotion2vec (DEV, 4/6/2026)** | **0.414** | **0.637** | **0.193** | — (không có VAD) |
| **exp04 FUSION multi-task (DEV, 4/6/2026)** | 0.414 | 0.788 | 0.145 | 0.578 / 0.754 / 0.706 |
| **exp07 FUSION+QMOS 6-head (DEV, 4/6/2026)** | **0.548** 🏆 | 0.795 | 0.153 | 0.581 / 0.752 / 0.705 |
| **exp08 FINE-TUNE WavLM (DEV NỘP, 5/6/2026)** | 0.4139 (UTMOS) | **0.811** 🏆 | **0.133** 🏆 | **0.659** 🏆 / **0.793** 🏆 / **0.751** 🏆 |
| **exp08b RESUME exp08 (DEV NỘP, 6/6/2026)** | 0.4167 (UTMOS) | 0.8116 | 0.1331 | 0.6605 / 0.7904 / 0.7539 |
| **🏆 exp_mix TRỘN CỘT (DEV NỘP, 9/6/2026)** | **0.548** 🏆 | **0.811** 🏆 | **0.133** 🏆 | **0.659** 🏆 / **0.793** 🏆 / **0.751** 🏆 |

> 🏆 **exp_mix (9/6, NỘP) — HỆ 6 CỘT MẠNH NHẤT:** ghép cột answer.txt — QMOS←exp07 (0.548) + EMOS/CAT/VAD←exp08 → điểm thật **khớp đúng best-per-column** (QMOS 0.5480 · EMOS 0.8111 · CAT 0.1331 · VAL 0.6590 · ARO 0.7933 · DOM 0.7509). Đây là **bản fallback an toàn** cho phase Evaluation. Folder: `submissions/Track2/exp_mix_q07_emo08/`.
> 🏆 **exp08 (5/6, NỘP):** fine-tune WavLM (mở băng 6 lớp, warm-start SAILER) + audeering frozen → **thắng cả 5 cột cảm xúc** vs exp07. QMOS rớt 0.414 (bản nộp không mượn exp07). → **Hệ tốt nhất 6 cột = trộn cột: 5 cảm xúc exp08 + QMOS exp07 (0.548)** (✅ đã nộp 9/6 = exp_mix). ⚠️ Backbone bị mất (kernel chết, ckpt cũ chỉ lưu heads) → đã vá lưu `ft_emotion_full.pt`, phải train lại. License thêm: **UTMOSv2 MIT** (nếu dùng cho QMOS).
> 📌 **exp08b (6/6, NỘP):** resume từ ckpt full + cache → điểm gần như TRÙNG exp08 (chênh vài phần nghìn). Xác nhận checkpoint exp08 **đã hội tụ**, train thêm trên cùng data không đổi.
| Hệ thống của mình (final) | [điền] | [điền] | [điền] | [điền] |

> 🏆 **EMOS tốt nhất hiện tại = emotion2vec (0.637)**, vượt cả SAILER (0.562). Nhưng **VAD tốt nhất = SAILER**. → hướng: **fusion** (gộp 2 model, xem `03_literature_notes.md`). Bản lai tốt nhất: QMOS←SpeechMOS · EMOS+CAT←emotion2vec · VAD←SAILER.

> ✅ **exp03 (4/6):** thay emotion2vec/Gemini bằng **SAILER** (`tiantiaf/wavlm-large-categorical-emotion`, WavLM-large, vô địch IS2025 SER) — 1 model lo EMOS+CAT+VAD. EMOS 0.194→**0.562**, mở luôn 3 cột VAD. QMOS giữ SpeechMOS.
> 🏆 **exp04 FUSION (4/6):** gộp **emotion2vec + SAILER** (đóng băng) → trunk chung → 3 head (EMOS/CAT/VAD), train multi-task + **uncertainty weighting**; QMOS để riêng (SpeechMOS). Thắng cả 5 cột cảm xúc: EMOS 0.637→**0.788**, CAT err 0.190→**0.145**, VAL 0.341→**0.578**, ARO 0.712→**0.754**, DOM 0.630→**0.706**.
> 🏆 **exp07 FUSION+QMOS 6-head (4/6) — hệ 1-model trọn 6 cột; QMOS 0.548 = mốc tốt nhất (5 cột cảm xúc sau đó bị exp08 vượt, xem mục exp08 ở trên):** thêm **head QMOS thứ 4** vào trunk exp04 (đầu vào `[trunk | điểm UTMOS]`) → **1 model dự đoán trọn 6 cột**. **QMOS 0.414→0.548** 🚀 (+0.134, lần cải thiện QMOS đầu tiên), **KHÔNG negative transfer** (EMOS 0.788→0.795, VAD ≈ giữ). Chỉ CAT hơi tệ (0.145→0.153). → kiến trúc chính cho paper ("1 model multi-task hợp nhất 6 cột thay cụm baseline chắp vá"). Xác nhận giả thuyết: trunk cảm xúc + UTMOS dự đoán chất lượng tốt hơn UTMOS đơn lẻ mà không hại cảm xúc.
> ⚠️ **External resource phải khai báo:** SAILER license **Open RAIL** + **audeering CC BY-NC-SA** (nếu dùng exp05) — đều phi thương mại; emotion2vec `iic/emotion2vec_plus_large`; SpeechMOS UTMOS22. Backbone đóng băng, chỉ train head; nhãn gộp TB theo wav, VAD/EMOS z-score, CAT = tỉ lệ vote 5 lớp.

> Lấy số liệu từ `04_experiments_log.md`.

---

# Track 3 — Codec-based Synthesis (speaker + accent similarity)

> Cách tiếp cận: **dùng baseline chính thức** (theo định hướng mentor: kéo source làm demo).

## 1. Tổng quan hệ thống
- Hệ thống: **ECAPA-TDNN** similarity (baseline BTC).
- Baseline dùng: [Baseline 1 zero-shot cosine / Baseline 2 fine-tuned — chọn cái nào]

## 2. Kiến trúc
- `speechbrain/spkrec-ecapa-voxceleb` → embedding 2 mẫu → [cosine sim / projection head] → spk_sim, acc_sim.
- **[HÌNH 3: sơ đồ ECAPA similarity]** ← cần chèn

## 3. Dữ liệu & external resources
- **Có dùng external resource:** ✅
  - Pre-trained: `speechbrain/spkrec-ecapa-voxceleb` (HuggingFace).
  - Checkpoint fine-tuned: có sẵn trong repo baseline (`official-egs/`).
- Data challenge: Track 3 (train 2.800 / val 600 / eval 600), nền VCTK.

## 4. Chiến lược training
- [Zero-shot, hoặc dùng checkpoint fine-tuned có sẵn — xác nhận]

## 5. Kết quả
| Set | Speaker UTT-SRCC | Accent UTT-SRCC |
|---|---|---|
| Dev (3/6/2026) | 0.451 | 0.440 |
| Eval | [điền] | [điền] |

> Baseline 2 (ECAPA fine-tuned) reproduce đúng mốc tham khảo (~0.45 / ~0.44). 600 cặp dev, không lỗi.

---

## Checklist trước khi nộp
- [ ] Đã khai báo external data/resource cho cả 3 track (có pointer/link)
- [ ] Có đủ 3 hình kiến trúc
- [ ] Mô tả đủ chi tiết để người khác tái lập
- [ ] Điền đầy đủ kết quả (dev + eval)
- [ ] **Đã DỊCH SANG TIẾNG ANH**
- [ ] (Tùy chọn) Upload arXiv + công khai code GitHub
