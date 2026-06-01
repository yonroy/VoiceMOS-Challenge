# VoiceMOS Challenge 2026 — Track 2 (Emotional TTS)

Dự án tham gia VoiceMOS Challenge 2026, tập trung vào **Track 2: Dự đoán MOS chất lượng giọng nói và độ tương đồng cảm xúc cho hệ thống Emotional TTS**.

---

## Mục tiêu
- Xây dựng hệ thống tự động dự đoán điểm MOS cho giọng nói cảm xúc
- Đạt thứ hạng tốt trên CodaBench (metric: UTT-SRCC)
- Viết paper submit ICASSP 2027

## Bài toán Track 2
Cho một đoạn giọng nói cảm xúc (TTS hoặc người thật), dự đoán:
1. **QMOS** — MOS chất lượng giọng nói (bắt buộc)
2. **EMOS** — MOS độ giống cảm xúc target (bắt buộc)
3. Phân loại cảm xúc người nghe cảm nhận (tùy chọn)
4. Valence / Arousal / Dominance (tùy chọn)

---

## Cấu trúc tài liệu
| File | Nội dung |
|---|---|
| `00_challenge_overview.md` | Tóm tắt cuộc thi, timeline, deadline |
| `01_research_plan.md` | Kế hoạch nghiên cứu, milestone |
| `02_mentor_questions.md` | Câu hỏi & trả lời từ mentor |
| `03_literature_notes.md` | Ghi chú đọc paper |
| `04_experiments_log.md` | Nhật ký thí nghiệm |
| `05_setup_environment.md` | Hướng dẫn môi trường, setup |
| `06_baseline_repos.md` | Repo baseline đã clone & cách setup |
| `07_project_summary.md` | Tóm tắt toàn bộ dự án (tra cứu nhanh) |
| `08_track2_spec.md` | Đặc tả Track 2: dataset, metric, format nộp `answer.txt` |
| `09_tracks_overview.md` | Đặc tả & so sánh cả 3 track (chính thức) |
| `10_learning_roadmap.md` | Lộ trình học cho người mới (gắn với dự án) |
| `11_progress_reports.md` | Nhật ký báo cáo tiến độ gửi mentor |

---

## Cách bắt đầu
1. Đọc `00_challenge_overview.md` để nắm thông tin cuộc thi
2. Setup môi trường theo `05_setup_environment.md`
3. Theo dõi kế hoạch trong `01_research_plan.md`
4. Ghi lại mọi thí nghiệm vào `04_experiments_log.md`

---

## Liên kết quan trọng
- Website challenge: https://sites.google.com/view/voicemos-challenge/voicemos-challenge-2026
- Baseline repo: https://github.com/voicemos-challenge/vmc2026-baselines
- Nền tảng thi: https://www.codabench.org/
- Liên hệ BTC: voicemos.challenge@gmail.com

## Thông tin liên hệ
- **Người làm:** [Điền tên]
- **Mentor:** [Điền tên mentor]
- **Đơn vị:** [Điền tên trường/lab]
