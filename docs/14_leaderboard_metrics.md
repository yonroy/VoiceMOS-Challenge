# 14 — Giải thích các cột điểm trên Leaderboard

> Giải nghĩa từng cột số bạn thấy trên CodaBench, chia theo track. Dành cho người mới.

---

## Đọc trước: 3 ký hiệu lặp lại ở mọi cột

| Ký hiệu | Nghĩa | Quy tắc |
|---|---|---|
| **UTT** | *Utterance-level* — chấm theo **từng câu nói** (mỗi audio 1 điểm), rồi tính tương quan trên toàn bộ câu. (Khác *system-level* = gộp trung bình theo từng hệ thống TTS.) | — |
| **SRCC** | *Spearman's Rank Correlation Coefficient* — đo **thứ hạng** điểm model dự đoán có khớp thứ hạng điểm người chấm không. Giá trị từ −1 đến 1. | **Càng gần 1 càng tốt** ⬆️ |
| **ERR** | *Error* — sai số (ở đây là sai lệch phân bố cảm xúc). | **Càng thấp càng tốt** ⬇️ |

> 💡 SRCC quan tâm **thứ tự**, không phải sai số tuyệt đối: nếu model luôn chấm thấp hơn người 0.5 điểm nhưng **xếp hạng đúng** thì SRCC vẫn cao. Đây là metric chính của challenge.

---

## 🟦 Track 1 — Speech Enhancement (2 cột)

| Cột | Tên đầy đủ | Là gì | Tốt khi |
|---|---|---|---|
| **ACR UTT-SRCC** | *Absolute Category Rating* | Người nghe chấm **chất lượng tuyệt đối** của 1 audio (thang 1–5, không so sánh với cái khác). Cột này = model đoán điểm ACR đó khớp người tới đâu. | ⬆️ gần 1 |
| **CCR UTT-SRCC** | *Comparative Category Rating* | Người nghe **so sánh 2 audio** (cái này tốt hơn/kém hơn cái kia bao nhiêu). Model đoán điểm so sánh đó. | ⬆️ gần 1 |

> Track 1 mình dùng baseline URGENT-MOS (chỉ inference).

---

## 🟥 Track 2 — Emotional TTS ⭐ (6 cột — track chính của mình)

| Cột | Tên đầy đủ | Là gì | Tốt khi | Bắt buộc? |
|---|---|---|---|---|
| **MOS UTT-SRCC** | Quality MOS (QMOS) | Đoán điểm **chất lượng giọng** (1–5): nghe có tự nhiên/sạch không. | ⬆️ gần 1 | ✅ Bắt buộc |
| **EMOS UTT-SRCC** | Emotion MOS | Đoán điểm **độ khớp cảm xúc target** (1–5): giọng có thể hiện đúng cảm xúc được yêu cầu (happy/sad/...) không. | ⬆️ gần 1 | ✅ Bắt buộc |
| **CAT UTT-ERR** | Categorical Error | Sai lệch giữa **phân bố cảm xúc** model dự đoán và phân bố người nghe cảm nhận (5 lớp: angry/happy/neutral/sad/surprised). | ⬇️ càng thấp | ⬜ Tùy chọn |
| **Valence UTT-SRCC** | Valence (V) | Đoán mức **tích cực ↔ tiêu cực** của cảm xúc trong giọng. | ⬆️ gần 1 | ⬜ Tùy chọn |
| **Arousal UTT-SRCC** | Arousal (A) | Đoán mức **kích thích/năng lượng** (bình tĩnh ↔ phấn khích). | ⬆️ gần 1 | ⬜ Tùy chọn |
| **Dominance UTT-SRCC** | Dominance (D) | Đoán mức **chi phối/áp đảo** (rụt rè ↔ mạnh mẽ) trong giọng. | ⬆️ gần 1 | ⬜ Tùy chọn |

> **Valence/Arousal/Dominance (VAD)** là cách mô tả cảm xúc bằng 3 trục số học thay vì nhãn rời rạc.
> Baseline: QMOS = SpeechMOS(UTMOS) · CAT = emotion2vec · EMOS & VAD = Gemini LLM-as-judge.
> ⚠️ Chú ý: **CAT là ERR (thấp = tốt)**, 5 cột còn lại là **SRCC (cao = tốt)** — đừng nhầm chiều.

---

## 🟩 Track 3 — Codec-based Synthesis (2 cột)

| Cột | Tên đầy đủ | Là gì | Tốt khi |
|---|---|---|---|
| **SPK UTT-SRCC** | Speaker similarity | Đoán độ **giống người nói** giữa audio sinh ra và giọng tham chiếu (cùng 1 người không). | ⬆️ gần 1 |
| **ACC UTT-SRCC** | Accent similarity | Đoán độ **giống chất giọng vùng miền (accent)** so với tham chiếu. | ⬆️ gần 1 |

> Track 3 mình dùng baseline ECAPA-TDNN (cosine similarity của embedding).

---

## Tóm tắt 1 dòng để nhớ
- **Hầu hết cột = UTT-SRCC → càng CAO (gần 1) càng tốt.**
- **Chỉ riêng `CAT UTT-ERR` → càng THẤP càng tốt.**
- **UTT** = chấm theo từng câu; cả 10 cột đều ở mức utterance.
