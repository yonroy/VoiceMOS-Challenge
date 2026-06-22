# 18 — Lịch sử Leaderboard qua các ngày

> Bảng theo dõi điểm leaderboard (DEV, CodaBench) theo thời gian. Cập nhật ngày: 16/6/2026.
> Metric chính = **UTT-SRCC** (càng cao càng tốt), riêng **CAT err** càng thấp càng tốt.
> Nguồn số liệu chuẩn: `04_experiments_log.md` + `12_system_description.md`. Khi có bản nộp mới → thêm 1 hàng.

---

## A. Track 2 — "tốt nhất từng cột" theo ngày (best-per-column, tích lũy)

> Mỗi ô = điểm tốt nhất đạt được **tính đến hết ngày đó** (gộp mọi bản nộp). `—` = chưa có.

| Ngày | QMOS ↑ | EMOS ↑ | CAT err ↓ | VAL ↑ | ARO ↑ | DOM ↑ |
|---|---|---|---|---|---|---|
| 3/6 | 0.414 | 0.194 | 0.193 | — | — | — |
| 4/6 | **0.548** | 0.795 | 0.145 | 0.581 | 0.754 | 0.706 |
| 5/6 | **0.548** | **0.811** | **0.133** | **0.659** | **0.793** | **0.751** |

**Δ 4/6 → 5/6:** EMOS +0.016 · CAT −0.020 · VAL +0.078 · ARO +0.041 · DOM +0.046 · QMOS giữ 0.548.
→ 5/6 cải thiện **5/6 cột** (đều nhờ exp08 fine-tune). QMOS đứng yên vì chưa có bản nộp QMOS mới hơn exp07.

| 8/6 | **0.548** | 0.8116 | **0.133** | 0.6605 | **0.793** | 0.7539 |

**Δ 5/6 → 8/6 (exp08b resume):** EMOS/VAL/DOM nhích vài phần nghìn, ARO/CAT ~giữ → **không đáng kể** (exp08b ≈ exp08, checkpoint đã hội tụ). Best 6 cột vẫn = QMOS exp07 + 5 cảm xúc exp08/exp08b.

| 9/6 | **0.548** | **0.811** | **0.133** | **0.659** | **0.793** | **0.751** |

**Δ 8/6 → 9/6 (TRỘN CỘT exp_mix_q07_emo08 — ĐÃ NỘP):** lần đầu **gom đủ best-per-column vào 1 bản nộp** (QMOS exp07 0.548 + 5 cảm xúc exp08) → điểm thật khớp đúng kỳ vọng. **Không phải kỷ lục cột mới** mà là **chốt hệ 6 cột mạnh nhất bằng điểm thật** (trước đó best-per-column chỉ là mục tiêu lý thuyết, chưa có bản nộp đơn nào đạt cả 6).

| 10/6 | **0.6296** 🚀 | **0.8116** | **0.1331** | **0.6605** | **0.7978** 🚀 | **0.7539** |

**Δ 9/6 → 10/6 (exp13 + exp15 — 2 kỷ lục cột trong 1 ngày):**
- **QMOS 0.548 → 0.6296** (+0.082, exp13 fine-tune UTMOS trên nhãn `qMOS` thật) — cột QMOS nhúc nhích lần đầu từ 4/6.
- **ARO 0.7933 → 0.7978** (+0.0045, exp15 Mamba head) — temporal modeling giúp đúng cột Arousal.
- EMOS/CAT/VAL/DOM best vẫn của exp08/exp08b (exp15 thua sát nút 3 cột này).

> ⚠️ Best-per-column hiện **chưa gom đủ trong 1 bản nộp**. Bản trộn thế hệ mới = QMOS←exp13 + ARO←exp15 + EMOS/CAT/VAL/DOM←exp08 — **chưa nộp**.

| 16/6 | **0.6296** | **0.8144** 🚀 | **0.1331** | **0.6605** | **0.7978** | **0.7539** |

**Δ 10/6 → 16/6 (exp18 cross-attention):** EMOS 0.8116 → **0.8144** (+0.0028, kỷ lục cột — cross-attn frozen WavLM⟷audeering, chỉ train ~1.7M tham số). Các cột exp18 còn lại (CAT 0.1351 · VAL 0.6403 · ARO 0.7917 · DOM 0.7426) **KHÔNG vượt** best cũ → giữ nguyên. Bản trộn best-per-column mới: **EMOS←exp18** + QMOS←exp13 + ARO←exp15 + CAT/VAL/DOM←exp08 — **chưa nộp**.

---

## B. Track 2 — chi tiết từng bản nộp (theo thứ tự thời gian)

| Ngày | Exp | Hệ thống | QMOS | EMOS | CAT err | VAL | ARO | DOM |
|---|---|---|---|---|---|---|---|---|
| 3/6 | baseline | UTMOS + emotion2vec + Gemini | 0.414 | 0.194 | 0.193 | — | — | — |
| 4/6 | exp01 | EMOS = emotion2vec target-prob | 0.414 | 0.637 | 0.193 | — | — | — |
| 4/6 | exp03 | SAILER (EMOS+CAT+VAD) | 0.414 | 0.562 | 0.190 | 0.341 | 0.712 | 0.630 |
| 4/6 | exp04 | FUSION e2v+SAILER (3 head) | 0.414 | 0.788 | 0.145 | 0.578 | 0.754 | 0.706 |
| 4/6 | exp07 | FUSION + QMOS head (6 cột) | **0.548** | 0.795 | 0.153 | 0.581 | 0.752 | 0.705 |
| 5/6 | exp08 | FINE-TUNE WavLM (warm-start SAILER) | 0.414¹ | **0.811** | **0.133** | **0.659** | **0.793** | **0.751** |
| 6/6 | exp08b | RESUME exp08 (train tiếp từ ckpt) | 0.4167¹ | 0.8116 | 0.1331 | 0.6605 | 0.7904 | 0.7539 |
| **9/6** | **exp_mix** | **TRỘN CỘT: QMOS←exp07 + 5 cảm xúc←exp08** | **0.548** | **0.811** | **0.133** | **0.659** | **0.793** | **0.751** |
| **10/6** | **exp13** | **FINE-TUNE thẳng UTMOS cho QMOS** | **0.63** 🚀 | ⁴ | ⁴ | ⁴ | ⁴ | ⁴ |
| **10/6** | **exp15** | **WavLM ft + MAMBA head (QMOS←ckpt exp13)** | **0.6296** | 0.8070 | 0.1349 | 0.6545 | **0.7978** 🚀 | 0.7506 |

¹ exp08/exp08b QMOS ~0.414 vì **bản nộp không kèm answer.txt exp07** → rơi về fallback UTMOS (không phải model kém).
² exp08b ≈ exp08 (chênh không đáng kể) → xác nhận checkpoint exp08 **đã hội tụ**, resume thêm không đổi.
³ exp_mix = ghép cột từ exp07 (QMOS) + exp08 (EMOS/CAT/VAD) → điểm thật khớp best-per-column tại 9/6. Bản fallback an toàn cho phase Evaluation.
⁴ exp13: QMOS 0.63 trên leaderboard (`benchmark/final.png`); cột cảm xúc bản nộp đó không bằng exp08. Số 4 chữ số 0.6296 xác nhận qua bản nộp exp15 (cùng ckpt `ft_qmos_utmos.pt`).
⁵ exp15 (`submissions/Track2/exp15_predict/`): Mamba head **gần hòa** mean-pool exp08 — thua sát EMOS/CAT/VAL, **thắng ARO** (+0.0045, kỷ lục cột). **Việc tiếp: nộp bản trộn QMOS←exp13 + ARO←exp15 + EMOS/CAT/VAL/DOM←exp08.**

---

## C. Track 1 & Track 3 — theo ngày

> Chỉ chạy baseline, chưa phát triển hệ thống riêng → điểm giữ nguyên từ 3/6.

| Ngày | T1 ACR ↑ | T1 CCR ↑ | T3 SPK ↑ | T3 ACC ↑ |
|---|---|---|---|---|
| 3/6 | 0.662 | 0.411 | 0.451 | 0.440 |
| 4/6 | 0.662 | 0.411 | 0.451 | 0.440 |
| 5/6 | 0.662 | 0.411 | 0.451 | 0.440 |
| 8/6 | 0.662 | 0.411 | 0.451 | 0.440 |

> Track 1/3 giữ nguyên qua các phiên — chỉ chạy baseline, không có bản nộp mới.

---

## Cách cập nhật bảng này
1. Có bản nộp mới → thêm 1 hàng vào **mục B** (ngày · exp · điểm 6 cột).
2. Nếu bản nộp phá kỷ lục cột nào → cập nhật hàng ngày mới ở **mục A** (in đậm cột phá kỷ lục).
3. Track 1/3 có điểm mới → thêm hàng **mục C**.
4. Đổi dòng "Cập nhật ngày" ở đầu file.
