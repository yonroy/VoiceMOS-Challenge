# 08 — Đặc tả Track 2 (chính thức)

> Nguồn: trang mô tả Track 2 + Submission guideline trên CodaBench (lưu tại `reference/content_btc/`). Cập nhật: 1/6/2026.
> File này là **nguồn chân lý** cho format đầu ra của hệ thống — code submission phải bám theo đây.

---

## 1. Nhiệm vụ
Cho 1 đoạn giọng nói cảm xúc (TTS hoặc người thật), dự đoán:

| # | Sub-task | Mô tả | Bắt buộc? |
|---|---|---|---|
| 1 | **QMOS** | MOS chất lượng giọng | ✅ |
| 2 | **EMOS** | MOS độ giống cảm xúc target | ✅ |
| 3 | **CAT** | Phân bố cảm xúc người nghe cảm nhận (5 lớp) | tùy chọn |
| 4 | **VAD** | Valence / Arousal / Dominance | tùy chọn |

5 cảm xúc: **neutral, happy, angry, sad, surprised**.

## 2. Dataset
- **Thu mới**, dựa trên **ESD + DailyTalk** (mẫu cảm xúc tự nhiên) + mẫu tổng hợp từ **13 hệ thống TTS** khác.

| Set | Số mẫu |
|---|---|
| Training | **12,746** |
| Validation | **2,730** |
| Evaluation | **2,730** |

- **Cách lấy data:** tải license form Track 2 → điền → email `ecooper@nict.go.jp` → BTC gửi link sau khi xác nhận.
  - License form: https://drive.google.com/file/d/1XYFRWCzKHRelz6KGZJlTsR-PAdacJ__d/view
- Eval data: công bố 31/7/2026 (To be announced).
- ⚠️ ESD/DailyTalk bản Kaggle chỉ để chuẩn bị/khám phá; **data chính thức có label MOS** phải lấy qua license.

## 3. Metric trên leaderboard (6 cái)
1. **UTT-SRCC** cho QMOS
2. **UTT-SRCC** cho EMOS
3. **Categorical error** cho CAT (xem dưới)
4. **UTT-SRCC** cho Valence
5. **UTT-SRCC** cho Arousal
6. **UTT-SRCC** cho Dominance

> Trong summary paper, BTC còn báo cáo MSE, LCC, KTAU ở cả utterance-level lẫn system-level.

### Categorical error (CAT)
- Ground-truth mỗi câu = **vector tỉ lệ vote** trên 5 cảm xúc (mỗi giá trị = tỉ lệ annotator chọn cảm xúc đó).
- Prediction = vector **xác suất** [0,1] cùng 5 lớp.
- Metric = tổng `|gt − pred|` trên **mọi cảm xúc & mọi câu**, chia cho **tổng số label** → mean error (càng **thấp** càng tốt).

## 4. Format nộp — `answer.txt`
- File **BẮT BUỘC tên `answer.txt`** (khác Track 1 dùng `predictions.csv`).
- Header đầy đủ:
  ```
  wav,QMOS,EMOS,CAT,VAL,ARO,DOM
  ```
- **QMOS, EMOS bắt buộc**; CAT/VAL/ARO/DOM tùy chọn → được nộp tập con, ví dụ:
  - `wav,QMOS,EMOS`
  - `wav,QMOS,EMOS,CAT`
  - `wav,QMOS,EMOS,VAL,ARO,DOM`
  - `wav,QMOS,EMOS,VAL,DOM` ...
- Cột **CAT** ghi dạng `emotion:prob` nối bằng `|`:
  ```
  angry:7.36e-07|happy:0.00014|neutral:0.98|sad:0.018|surprised:4.15e-08
  ```
- Số thực được phép (không cần làm tròn về integer).
- Ví dụ 2 dòng:
  ```
  wav,QMOS,EMOS,CAT,VAL,ARO,DOM
  vmc2026-track2-sys012-spk026-utt041.wav,3.5880108,1,angry:7.37e-07|happy:0.00014|neutral:0.98|sad:0.018|surprised:4.15e-08,1,5,5
  vmc2026-track2-sys002-spk012-utt177.wav,3.7573109,2,angry:1.19e-05|happy:0.0022|neutral:0.93|sad:0.0015|surprised:0.0045,4,4,4
  ```
- Nén: `zip -j <tên_tùy_ý>.zip answer.txt` (tên zip không ràng buộc; `-j` để tránh tạo thư mục lồng).

## 5. Quy trình nộp trên CodaBench
1. Tab **My Submissions**.
2. Pulldown **Selected Tracks** → chọn **Track 2** và **BỎ CHỌN** các track khác.
3. Bấm icon **kẹp giấy (📎)** → upload file zip.
4. Trạng thái chấm: **Preparing → Running → Scoring → Finished** (có nút cancel).
5. Chấm xong → chọn đưa lên leaderboard hay không (đưa bao nhiêu lần tùy ý).
6. Click submission → tab **LOGS** để xem stdout/stderr nếu lỗi.

### ⚠️ Bẫy & lưu ý
- **Mỗi lần chỉ nộp 1 track:** `answer.txt` chỉ chứa mẫu của Track 2.
- **Nhớ BỎ CHỌN track không nộp.** Nếu không → track thừa bị sanity-check fail và gán điểm rác trên leaderboard (MSE=100, LCC/SRCC/KTAU=0). Bỏ chọn đúng → track thừa hiện **n/a**.
- Có vài **cột dummy** giữa track 1&2 và track 3 do thiết kế CodaBench — bình thường.

## 6. Luật nộp (số lần)
- **Training phase:** bắt buộc nộp **≥ 1 lần**; tối đa **30 lần/ngày**.
- **Evaluation phase:** tối đa **3 lần**; chọn **1** làm đáp án cuối.
- Bắt buộc nộp **system description** cuối challenge.
- Được dùng **mọi public dataset** (phải khai báo); **cấm** proprietary / tự thu MOS trừ khi đã public.

## 7. Checklist nộp lần đầu
- [ ] Sign up + join CodaBench (competition 16419)
- [ ] Nhận data Track 2 (license đã gửi Erica Cooper)
- [ ] Sinh `answer.txt` đúng header, đúng tên file wav
- [ ] `zip -j sub.zip answer.txt`
- [ ] My Submissions → chọn Track 2, bỏ chọn track khác → upload
- [ ] Chờ Finished → kiểm tra điểm → add to leaderboard
- [ ] Đọc FAQ CodaBench: https://voicemos-challenge-2023.github.io/codalab.html

---

## Liên kết
- Hướng dẫn nộp/FAQ CodaBench: https://voicemos-challenge-2023.github.io/codalab.html
- CodaBench competition: https://www.codabench.org/competitions/16419/?secret_key=<XEM_EMAIL_BTC>
- Baseline: https://github.com/voicemos-challenge/vmc2026-baselines
