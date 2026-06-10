# 16 — Kiến trúc các model dùng trong Track 2

> Tài liệu nền cho người mới + nguồn cho phần **Method / Related Work** của paper và `12_system_description.md`.
> Mỗi model: kiến trúc → vai trò trong dự án → exp dùng → bài báo gốc → license.
> Cập nhật ngày: 5/6/2026.

---

## 0. Bản đồ nhanh: model nào lo cột nào

| Cột leaderboard | Model chính | Exp |
|---|---|---|
| **QMOS** (chất lượng) | UTMOS / SpeechMOS → (exp07) head trên trunk + UTMOS-feature | baseline → exp07 |
| **EMOS** (khớp cảm xúc) | emotion2vec (vô địch lẻ) · SAILER · fusion | exp01/03/04/07 |
| **CAT** (5 lớp cảm xúc) | emotion2vec · SAILER | exp03/04/07 |
| **VAD** (Valence/Arousal/Dominance) | SAILER · **audeering** (chuyên valence) · WavLM fine-tune | exp03/04/05/07/08 |

> 3 kiểu xử lý: **lấy embedding (frozen)** · **train head (mới)** · **fine-tune (mở băng backbone)**. exp08 là exp đầu tiên *fine-tune*.

---

## 1. WavLM ⭐ (backbone trung tâm — đang fine-tune ở exp08)

**Là gì:** model SSL (self-supervised learning — học không cần nhãn) cho giọng nói, "full-stack" (1 backbone xài cho ASR, người nói, cảm xúc...). Nuốt **waveform thô** → nhả **chuỗi vector đặc trưng**.

**Kiến trúc — 2 khối lớn:**
```
 waveform thô (16kHz)
        │
        ▼
 ┌───────────────────────┐
 │ 1. CNN feature encoder │  7 lớp conv → nén audio thành "khung" ~20ms
 │    (feature extractor) │  (~50 khung/giây), mỗi khung 512 chiều
 └───────────────────────┘
        │  + feature projection (512 → 1024)
        ▼
 ┌───────────────────────┐
 │ 2. Transformer encoder │  WavLM-large: 24 lớp · hidden 1024 · 16 head
 │    (self-attention)    │  → ~316 triệu tham số
 └───────────────────────┘
        │
        ▼
 hidden states [T khung × 1024 chiều]
```
- **Khối 1 (CNN):** biến sóng âm thô thành chuỗi khung gọn; học đặc trưng **âm học cục bộ** (như "tai" sơ cấp).
- **Khối 2 (Transformer):** mỗi lớp dùng **self-attention** để mỗi khung "nhìn" mọi khung khác → hiểu ngữ cảnh cả câu. `hidden 1024` = `WAVLM_DIM` trong code; `24 lớp` = lý do `UNFREEZE_TOP_LAYERS=6` nghĩa là đóng băng 18 lớp dưới, mở 6 lớp trên.

**2 cải tiến riêng của WavLM (so wav2vec2/HuBERT):**
1. **Gated relative position bias:** thêm "thiên lệch vị trí" có cổng vào attention → nắm thứ tự thời gian tốt hơn.
2. **Masked speech denoising:** lúc pre-train cố tình trộn nhiễu + giọng người khác chồng lên, bắt model đoán lại phần *sạch* → học tách giọng/chịu nhiễu → mạnh cả task người nói & cảm xúc.

**Điểm mấu chốt — các lớp học thứ khác nhau:** lớp **dưới** → âm học/phonetic; lớp **trên** → trừu tượng/người nói/**cảm xúc**. → exp08 chỉ fine-tune 6 lớp trên (cảm xúc ở tầng trên), vừa tiết kiệm VRAM vừa chống overfit.

**Vai trò trong dự án:** backbone của SAILER; **exp08 fine-tune trực tiếp** WavLM-large (warm-start từ SAILER).
**Bài báo:** Chen et al., *"WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing"*, IEEE JSTSP 2022 — **arXiv:2110.13900**.
**License backbone gốc:** MIT (microsoft/wavlm-large).

---

## 2. SAILER ⭐ (WavLM-large đã fine-tune cảm xúc — warm-start của exp08)

**Là gì:** chính WavLM-large nhưng **đã fine-tune sẵn trên MSP-Podcast** cho nhận diện cảm xúc → vô địch **Interspeech 2025 SER Challenge (Task 1)** (1 hệ đơn vượt >95% bài nộp, Macro-F1 > 0.4).

**Kiến trúc:** WavLM-large (mục 1) + head phân loại 9 lớp cảm xúc (Anger, Contempt, Disgust, Fear, Happiness, Neutral, Sadness, Surprise, Other) + xuất sẵn VAD. `forward(return_feature=True)` trả **6 giá trị**: `logits, feat, _det, arousal, valence, dominance`.

**Vai trò:** EMOS+CAT+VAD (exp03), nhánh trong fusion (exp04/07), **warm-start backbone exp08** (lấy WavLM bên trong wrapper ra để fine-tune tiếp).
**Bài báo:** Feng, Lertpetchpun, Byrd, Narayanan, *"Developing a Top-tier Framework in Naturalistic Conditions Challenge for Categorized Emotion Prediction…"*, Interspeech 2025 — **arXiv:2505.22133**. Benchmark/repo: **Vox-Profile, arXiv:2505.14648**, code: github.com/tiantiaf0627/vox-profile-release.
**License:** ⚠️ **Open RAIL** (phi thương mại) — sản phẩm fine-tune cũng phi thương mại → **bắt buộc khai báo** ở `12_`.

---

## 3. emotion2vec (vô địch EMOS lẻ của dự án)

**Là gì:** SSL chuyên **biểu diễn cảm xúc giọng nói** (kiểu data2vec — học bằng cách dự đoán biểu diễn ẩn của chính mình). Pre-train trên nhiều corpus cảm xúc.

**Kiến trúc:** Transformer encoder cảm xúc → xuất **embedding** + **xác suất lớp cảm xúc** (categorical). **KHÔNG có VAD**.

**Vai trò:** EMOS = `1 + 4·P(target)` (exp01, đạt **0.637** — vượt SAILER 0.562); CAT (exp03/04); nhánh trong fusion exp04/07.
**Điểm mạnh/yếu:** giỏi **categorical/EMOS**; "dồn 2 cực" (overconfident) nhưng thứ hạng vẫn khớp người chấm → SRCC cao. **Không làm VAD.**
**Bài báo:** Ma et al., *"emotion2vec: Self-Supervised Pre-Training for Speech Emotion Representation"*, ACL Findings 2024 — **arXiv:2312.15185**. Model: `iic/emotion2vec_plus_large` (funasr/ModelScope).
**Lưu ý:** funasr API **khó fine-tune** → lý do exp08 không fine-tune emotion2vec mà chọn WavLM.

---

## 4. audeering MSP-dim (chuyên Valence — nhánh phụ frozen exp08)

**Là gì:** wav2vec2-large-robust **fine-tune trên MSP-Podcast** cho cảm xúc **dimensional** (xuất thẳng Valence/Arousal/Dominance liên tục ∈ [0,1]).

**Kiến trúc:** backbone **wav2vec2** (mục 6) + regression head `Linear→Tanh→Linear` ra **3 số** [arousal, dominance, valence]. ⚠️ thứ tự xuất là **[A, D, V]** → code đổi về [VAL, ARO, DOM] thang 1–5.

**Vai trò:** thay cả 3 cột VAD (exp05); **nhánh phụ đóng băng** trong exp08 (`[emb 1024 | VAD 3]`, cache `aud_*.npz`).
**Điểm mạnh:** **valence** — đúng câu chuyện bài báo *"closing the valence gap"* (valence là chiều khó nhất với acoustic-only).
**Bài báo:** Wagner et al., *"Dawn of the Transformer Era in Speech Emotion Recognition: Closing the Valence Gap"*, IEEE TPAMI 2023 — **arXiv:2203.07378**. Model: `audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim`.
**License:** ⚠️ **CC BY-NC-SA 4.0** (phi thương mại) → khai báo `12_`.
**Gotcha:** subclass `Wav2Vec2PreTrainedModel` lỗi version transformers mới → dùng `Wav2Vec2Model` + nạp tay trọng số head.

---

## 5. UTMOS / SpeechMOS (QMOS)

**Là gì:** model dự đoán **MOS chất lượng/naturalness**, vô địch **VoiceMOS Challenge 2022**.

**Kiến trúc:** SSL backbone (wav2vec2/ssl) + head hồi quy MOS; có dùng đặc trưng phụ (data-domain id, listener id...) thời 2022.

**Vai trò:** QMOS mọi exp (zero-shot, **0.414**); làm **đầu vào** cho QMOS head exp06/07 (neo residual).
**Hạn chế:** train trên giọng **không cảm xúc** 2022 → **lệch domain** với Track 2 → lý do QMOS kẹt 0.414 cho tới khi exp07 train head (→ 0.548).
**Bài báo:** Saeki et al., *"UTMOS: UTokyo-SaruLab System for VoiceMOS Challenge 2022"*, Interspeech 2022 — **arXiv:2204.02152**. Dùng qua `tarepan/SpeechMOS` (torch.hub, `utmos22_strong`).

---

## 6. wav2vec 2.0 (backbone nền của audeering)

**Là gì:** model SSL giọng nói "ông tổ" — CNN feature encoder + Transformer + **contrastive learning** (đoán đúng đoạn lượng tử hóa bị che giữa các phương án nhiễu).
**Vai trò:** không dùng trực tiếp; là **backbone bên trong audeering**.
**Bài báo:** Baevski et al., *"wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations"*, NeurIPS 2020 — **arXiv:2006.11477**.

> **HuBERT** (Hsu et al., arXiv:2106.07447) — họ hàng (masked prediction + cụm pseudo-label k-means); *chưa dùng*, chỉ cite ở Related Work nếu so backbone.

---

## 7. Multi-task uncertainty weighting (cách cân loss — exp04/07/08)

**Là gì:** kỹ thuật **tự học trọng số** cho nhiều loss khác thang (EMOS/CAT/VAD), thay vì chỉnh tay. Mỗi task học 1 tham số `log σ²`.
**Công thức (trong code mục 6 exp08):**
```
tổng_loss = Σ_task [ exp(−log σ²_task) · L_task + log σ²_task ]
```
Task "khó/nhiễu" (σ² lớn) tự được giảm trọng số → các task không lấn nhau.
**Bài báo:** Kendall, Gal, Cipolla, *"Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics"*, CVPR 2018 — **arXiv:1705.07115**.

---

## 8. Kiến trúc HỆ THỐNG của mình (fusion — exp04/07 frozen, exp08 fine-tune)

**exp04/07 (frozen fusion):**
```
 wav ─┬─► emotion2vec (frozen) ─► [emb | p5] ─┐
      └─► SAILER (frozen) ─► [emb | p9 | vad3]┘─► TRUNK chung ─┬─► EMOS (+target)
                                                                ├─► CAT (5)
                                                                ├─► VAD (3)
                                                                └─► QMOS (+UTMOS) [exp07]
```
**exp08 (fine-tune — 1 backbone train, 1 phụ frozen):**
```
 wav ─┬─► WavLM (warm-start SAILER) ── 6 lớp trên FINE-TUNE ─► emb 1024 ┐
      └─► audeering (FROZEN, cache) ─► [emb | vad3] ──────────────────┘─► TRUNK ─┬─► EMOS (+target)
                                                                                  ├─► CAT (5)
                                                                                  └─► VAD (3)
 QMOS: mượn exp07 (0.548) hoặc UTMOS.
```
- **3 kiểu:** 🔥 fine-tune (6 lớp WavLM, LR 1e-5) · 🆕 train từ đầu (trunk+head, LR 1e-3) · ❄️ lấy embedding (audeering + lớp WavLM dưới, frozen, cache).

---

## Liên kết
- Chi tiết exp + điểm: [04_experiments_log.md](04_experiments_log.md) · Khai báo license/Method: [12_system_description.md](12_system_description.md) · Paper: [15_paper_draft.md](15_paper_draft.md)
- Pipeline code: `kaggle_baseline/track2/` (`exp08_finetune_emotion_pipeline.py` = exp08)
