# 11 — Báo cáo tiến độ (gửi mentor)

> Nhật ký báo cáo tiến độ theo ngày. Báo cáo mới nhất ở trên cùng.

---

## Báo cáo ngày 1/6/2026

**Người thực hiện:** Tran Minh Toan
**Track:** Track 2 — Emotional TTS (chính); Track 1 & 3 — baseline/demo

### 1. Thủ tục & hạ tầng — đã xong
- ✅ Đăng ký challenge + **join CodaBench** (competition 16419).
- ✅ Chốt compute: dùng **Kaggle T4** (theo gợi ý của thầy/cô).
- 🔄 Đã gửi **license form Track 2 & 3** cho BTC (Erica Cooper). Lần đầu bị trả lại do thiếu chữ ký → **đã ký lại và gửi lần 2**, đang chờ xác nhận để nhận link data.

### 2. Nghiên cứu yêu cầu đề bài — đã xong
- ✅ Đọc & hệ thống hóa đặc tả **cả 3 track**: dataset, kích thước (Track 2: train 12.746 / val 2.730 / eval 2.730), 6 metric (QMOS/EMOS/VAD SRCC + categorical error), format nộp (`answer.txt`).
- ✅ Xác định dataset nền Track 2 = **ESD + DailyTalk** + 13 hệ thống TTS; đã có sẵn 2 bộ này qua Kaggle Dataset.

### 3. Baseline — đã dựng pipeline
- ✅ Viết pipeline chạy trên Kaggle cho **cả 3 track** (notebook sẵn sàng).
- ✅ **Track 1** (URGENT-MOS): chạy được ngay (data công khai, checkpoint pre-trained) → chuẩn bị nộp lần đầu (thỏa luật "nộp ≥1 lần trong training phase").
- ✅ **Track 3** (ECAPA): có code + checkpoint train sẵn → sẵn sàng cho mục đích demo UI.
- ✅ **Track 2**: pipeline QMOS (SpeechMOS) + EmoCat (emotion2vec) test được trên ESD ngay; EMOS/VAD (Gemini) chờ data.

### 4. Đang làm / chờ
- ⏳ Chờ data Track 2/3 (license).
- 🔄 Bắt đầu đọc literature (UTMOS, WavLM) + xây nền kiến thức.

### 5. Câu hỏi cần định hướng
- **Hướng novelty Track 2** nên đi theo đâu? (cân nhắc: fusion SSL backbone + emotion embedding, hoặc multi-task QMOS+EMOS)
- **Backbone** nên chọn WavLM hay HuBERT/Wav2Vec2 cho emotion MOS?
- Thầy/cô muốn báo cáo **theo tuần hay theo milestone**?

---

<!-- Mẫu cho báo cáo sau — copy block dưới:

## Báo cáo ngày DD/MM/YYYY
### Đã làm tuần này
-
### Kết quả / số liệu
-
### Khó khăn / cần hỗ trợ
-
### Kế hoạch tuần tới
-
-->
