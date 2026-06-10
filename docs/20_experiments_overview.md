# 20 — Bảng tổng quan trạng thái Experiment (Track 2)

> **Mục đích:** nhìn 1 phát biết **đã làm được gì** trong từng exp, cái nào đã nộp/có điểm, cái nào mới code.
> Đây là **bảng trạng thái nhanh** — chi tiết config/kết quả đầy đủ vẫn ở [04_experiments_log.md](04_experiments_log.md).
>
> Cập nhật ngày: 9/6/2026 (Phiên 14).

---

## 🔑 Chú thích trạng thái

| Ký hiệu | Nghĩa |
|---|---|
| ✅ **NỘP** | Đã nộp CodaBench → có **điểm thật** trên leaderboard (cái này mới "tính") |
| 🟡 **CHẠY** | Đã chạy nhưng **chỉ có val nội bộ**, chưa nộp (val nội bộ ≠ điểm DEV thật) |
| ⚪ **CODE** | Mới viết code, **chưa chạy lần nào** |

> ⚠️ Phân biệt quan trọng: **val nội bộ** (tự chia 10% train để tự chấm) thường **CAO hơn** điểm DEV thật trên leaderboard (dễ overfit). Chỉ điểm ✅ NỘP mới là điểm thật.

---

## 1️⃣ ĐÃ NỘP — có điểm thật trên leaderboard (DEV)

| Exp | Là gì | Điểm chính | Vai trò |
|---|---|---|---|
| baseline | UTMOS + emotion2vec + Gemini (chắp vá 3 model) | QMOS 0.414 · EMOS 0.194 · CAT 0.193 | mốc xuất phát |
| **exp01** | EMOS = xác suất lớp target của emotion2vec (offline) | EMOS **0.637** | phát hiện emotion2vec > SAILER ở EMOS |
| **exp03** | SAILER (WavLM-large) lo EMOS + CAT + VAD | EMOS 0.562 · mở 3 cột VAD (0.341/0.712/0.630) | lần đầu có VAD |
| **exp04** | FUSION: emotion2vec + SAILER (đóng băng) → trunk chung → 3 head (5 cột cảm xúc) | EMOS 0.788 · CAT 0.145 · VAD 0.578/0.754/0.706 | thắng mọi model lẻ ở cảm xúc |
| **exp07** | exp04 + thêm head QMOS thứ 4 → 1 model trọn 6 cột | **QMOS 0.548 🏆** · EMOS 0.795 · CAT 0.153 · VAD 0.581/0.752/0.705 | 🏆 **QMOS tốt nhất** |
| **exp08** | FINE-TUNE WavLM (mở băng 6 lớp, warm-start SAILER) + audeering frozen | **EMOS 0.811 🏆 · CAT 0.133 🏆 · VAD 0.659/0.793/0.751 🏆** | 🏆 **cảm xúc tốt nhất** (QMOS rớt 0.414) |
| **exp08b** | exp08 train tiếp (resume từ checkpoint) | MOS 0.4167 · EMOS 0.8116 · CAT 0.1331 · VAD 0.6605/0.7904/0.7539 | ≈ exp08 → xác nhận **đã hội tụ** |
| **exp_mix** | TRỘN CỘT: QMOS←exp07 + 5 cảm xúc←exp08 (ghép answer.txt) | **QMOS 0.548 · EMOS 0.811 · CAT 0.133 · VAD 0.659/0.793/0.751** | 🏆 **hệ 6 cột mạnh nhất — NỘP 9/6, điểm thật khớp best-per-column** |

---

## 2️⃣ ĐÃ CHẠY nhưng CHƯA NỘP (chỉ val nội bộ)

| Exp | Là gì | Kết quả (val nội bộ) | Vì sao chưa nộp |
|---|---|---|---|
| **exp11** | fine-tune CẢ WavLM + audeering trong 1 model (fusion) | mean SRCC 0.83 (EMOS 0.835 / VAD 0.803·0.874·0.808) | warm-start đã đỉnh, train thêm không cải thiện; nghi **overfit** (val nội bộ 0.80 >> DEV exp08 0.66) |

---

## 3️⃣ MỚI CODE — chưa chạy lần nào

| Exp | Là gì | Ngày code | Ghi chú |
|---|---|---|---|
| exp02 | train head EMOS trên emotion2vec đóng băng | 3/6 | bỏ ngỏ (đã có hướng fusion tốt hơn) |
| exp05 | VAD bằng audeering riêng (đẩy VAL) | 4/6 | |
| exp06 | train head QMOS riêng (đặc trưng đóng băng + neo UTMOS) | 4/6 | để A/B với exp07 |
| exp09a | probe UTMOSv2 vs UTMOS cho QMOS (không tốn lượt nộp) | 5/6 | |
| exp10 | fine-tune audeering riêng + ensemble cột VAD | 5/6 | |
| exp12 | ablation khởi tạo WavLM (scratch / base / sailer) | 8/6 | chưa chạy đủ 3 mode (trả lời mentor) |
| **exp13** | **fine-tune UTMOS cho QMOS + nạp ckpt cảm xúc exp08 → answer 6 cột** | **8/6** | HOÀN THIỆN 9/6: **sửa lỗi ranking loss** (gom MSE+pred cả cửa sổ → backward 1 lần). Sẵn sàng chạy — chưa chạy thật |
| exp14 | Mamba head cộng vào fusion exp07 (WavLM frame đóng băng) | 8/6 | chỉ có .py, chưa convert .ipynb |
| **exp15** | **WavLM fine-tune + Mamba head** (thay mean-pool exp08) + ranking loss + tự dò DATA_ROOT | 8/6 | smoke-test 8/6 (mamba-ssm fail→PyTorch); chưa chạy thật |
| **exp16** | **Audio-LLM-as-Judge** (API Gemini/GPT-4o-audio chấm 6 cột) — novelty cho paper, KHÔNG GPU | 8/6 | code xong (.py+.ipynb), cache+resume; chưa chạy (cần API key) |

---

## ✅ Bản TRỘN CỘT — ĐÃ NỘP (9/6) 🎉
- `submissions/Track2/exp_mix_q07_emo08/` = QMOS←exp07 (0.548) + 5 cột cảm xúc←exp08 → **hệ 6 cột mạnh nhất**.
- **Điểm thật (scores.json):** QMOS **0.5480** · EMOS **0.8111** · CAT err **0.1331** · VAL **0.6590** · ARO **0.7933** · DOM **0.7509** → khớp đúng best-per-column.
- Đây là **bản fallback an toàn** cho phase Evaluation (chỉ cần đổi input eval → ráp lại answer.txt).

---

## 🎯 Hệ thống mạnh nhất hiện tại — "TRỘN CỘT"

Grader chỉ chấm `answer.txt` → được phép ghép cột từ nhiều exp:

| Cột | Lấy từ | Điểm |
|---|---|---|
| QMOS | **exp07** | 0.548 🏆 |
| EMOS | **exp08** | 0.811 🏆 |
| CAT (err) | **exp08** | 0.133 🏆 |
| VAL / ARO / DOM | **exp08** | 0.659 / 0.793 / 0.751 🏆 |

> ✅ **ĐÃ NỘP 9/6** — điểm thật khớp best-per-column (QMOS 0.548 · EMOS 0.811 · CAT 0.133 · VAD 0.659/0.793/0.751). Món nợ kéo dài nhiều phiên đã chốt.

---

## 🧭 exp13 đang thử cải thiện gì?

Thay vì mượn QMOS 0.548 của exp07, **exp13 fine-tune thẳng UTMOS** trên nhãn qMOS thật → xem QMOS có vượt 0.548 không.
- **Nếu vượt** → exp13 = hệ mạnh nhất mới (QMOS exp13 + 5 cột cảm xúc exp08).
- **Nếu không vượt** → giữ QMOS exp07; vẫn là kết quả cho paper.
- Checkpoint cảm xúc dùng cho exp13 = **`ft_emotion_full_20epoch.pt`** (bản TỐT NHẤT, không dùng `ft_emotion_full.pt`).

---

## 📌 Đọc thêm
- Chi tiết config → kết quả → nhận xét mỗi exp: [04_experiments_log.md](04_experiments_log.md)
- Bảng điểm từng track + leaderboard: [12_system_description.md](12_system_description.md) · [18_leaderboard_history.md](18_leaderboard_history.md)
- Giải thích từng cột metric: [14_leaderboard_metrics.md](14_leaderboard_metrics.md)
