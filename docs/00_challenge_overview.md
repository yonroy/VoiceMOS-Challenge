# 00 — Tổng quan VoiceMOS Challenge 2026

## Giới thiệu
VoiceMOS Challenge (VMC) thành lập năm 2022, mục tiêu phát triển các phương pháp **tự động dự đoán điểm MOS** (Mean Opinion Score) cho giọng nói, thay thế việc kiểm tra thính giác bằng con người vốn tốn kém và chậm.

Năm 2026 quay lại tập trung vào **speech** (sau khi 2025 mở rộng sang music/singing trong AudioMOS Challenge).

---

## Metric đánh giá chính
**SRCC** — Spearman's Rank Correlation Coefficient, dùng ở cấp độ từng câu nói (**UTT-SRCC**). Đo tương quan giữa điểm model dự đoán và điểm người chấm; càng gần 1 càng tốt.

> Track 2 có **6 metric** trên leaderboard: UTT-SRCC cho QMOS, EMOS, Valence, Arousal, Dominance + **Categorical error** cho phân bố cảm xúc (càng thấp càng tốt). Chi tiết: `08_track2_spec.md`.
> Summary paper sẽ báo cáo thêm MSE, LCC, KTAU ở cả utterance-level và system-level.

---

## 3 Tracks

### Track 1 — Speech Enhancement
- Dựa trên dữ liệu listening test của ICASSP 2026 URGENT Challenge
- 840 câu nói đa ngôn ngữ (9 ngôn ngữ), top 6 hệ thống
- Dự đoán: **ACR** (Absolute Category Rating) + **CCR** (Comparative Category Rating)

### Track 2 — Emotional TTS ⭐ (Track của mình)
- Giọng nói cảm xúc từ TTS + người thật
- Dataset **thu mới**, dựa trên **ESD + DailyTalk** + mẫu từ **13 hệ thống TTS**; 5 cảm xúc (neutral/happy/angry/sad/surprised)
- Kích thước: **train 12,746 · val 2,730 · eval 2,730**
- Dự đoán:
  1. **QMOS** — MOS chất lượng giọng (bắt buộc)
  2. **EMOS** — MOS độ giống cảm xúc target (bắt buộc)
  3. Phân loại cảm xúc người nghe cảm nhận (tùy chọn)
  4. Valence / Arousal / Dominance (tùy chọn)
- Format nộp: file `answer.txt` (`wav,QMOS,EMOS,CAT,VAL,ARO,DOM`) → xem `08_track2_spec.md`

### Track 3 — Codec-based Speech Synthesis
- Giọng tiếng Anh có accent từ hệ thống codec
- Dataset CodecMOS-Accent: 4.000 mẫu, 24 hệ thống, 32 người nói, 10 accent
- Dự đoán: độ tương đồng **speaker** + **accent** (cần cả reference speech)

---

## Timeline

| Mốc | Ngày | Ghi chú |
|---|---|---|
| Website & đăng ký mở | 16/4/2026 | — |
| Training data release | 22/5/2026 | Trên CodaBench |
| Email hướng dẫn data | 20/5/2026 | Đã gửi |
| **Evaluation data release** | **31/7/2026** | — |
| **🔴 Hạn nộp kết quả** | **7/8/2026** | QUAN TRỌNG NHẤT |
| Đăng ký đóng | 7/8/2026 | — |
| Công bố kết quả | 31/8/2026 | — |
| **Hạn nộp paper ICASSP 2027** | **16/9/2026** | Nếu muốn publish |

---

## Quy định
- **Training phase:** bắt buộc nộp leaderboard **≥ 1 lần**; tối đa **30 lần/ngày**.
- **Evaluation phase:** tối đa **3 lần nộp**, chọn **1** làm đáp án cuối.
- **Mỗi lần chỉ nộp 1 track** (file `answer.txt` chỉ chứa mẫu của track đó); nhớ bỏ chọn track không nộp trên CodaBench.
- **Bắt buộc** nộp system description sau khi challenge kết thúc (mô tả kiến trúc, training, data; có hình minh họa; khuyến khích arXiv + open-source code).
- Được dùng **bất kỳ public dataset** nào (phải khai báo trong system description).
- **Không** được dùng proprietary dataset hoặc tự thu thập MOS rating (trừ khi đã public).

---

## Baseline Track 2
- **QMOS Baseline:** UTMOS (Interspeech 2022)
- **Emotion Categories Baseline:** Emotion2vec
- **EMOS Baseline:** Gemini LLM-as-judge
- **Valence/Arousal/Dominance Baseline:** Gemini LLM-as-judge

Repo: https://github.com/voicemos-challenge/vmc2026-baselines

---

## Ban tổ chức
- Wen-Chin Huang & Tomoki Toda (Nagoya University, Japan)
- Erica Cooper (NICT, Japan)
- Wei Wang (Shanghai Jiao Tong University, China)
- Marvin Sach (TU Braunschweig, Germany)
- Xiaoxue Gao (A*STAR, Singapore)
- Nicholas Sanders (University of Edinburgh, UK)

Venue dự kiến: special session / satellite workshop tại **ICASSP 2027**.
