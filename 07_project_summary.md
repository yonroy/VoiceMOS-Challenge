# 07 — Tóm tắt toàn bộ dự án

> Tài liệu tổng hợp nhanh để tra cứu / gửi mentor. Cập nhật ngày: 1/6/2026.

---

## 1. Dự án là gì
Tham gia **VoiceMOS Challenge 2026 — Track 2 (Emotional TTS)**: xây hệ thống **tự động dự đoán điểm MOS** cho giọng nói cảm xúc. Mục tiêu kép:
- Đạt thứ hạng tốt trên CodaBench (metric: UTT-SRCC)
- Viết **paper ICASSP 2027**

## 2. Bài toán Track 2
Cho 1 đoạn giọng cảm xúc (TTS hoặc người thật), dự đoán:

| Sub-task | Mô tả | Bắt buộc? |
|---|---|---|
| **QMOS** | MOS chất lượng giọng (1–5) | ✅ |
| **EMOS** | MOS độ khớp cảm xúc target (1–5) | ✅ |
| **EmoCat** | Tỉ lệ vote 5 cảm xúc (Neutral/Happy/Sad/Angry/Surprise) | tùy chọn |
| **VAD** | Valence / Arousal / Dominance (mỗi cái 1–5) | tùy chọn |

**Metric chính:** UTT-SRCC — Utterance-level Spearman's Rank Correlation Coefficient (càng gần 1 càng tốt). Track 2 có 6 metric leaderboard (QMOS/EMOS/VAD SRCC + categorical error).

**Dataset Track 2:** thu mới trên nền ESD + DailyTalk + 13 hệ thống TTS — **train 12,746 · val 2,730 · eval 2,730**. Nộp file `answer.txt` (`wav,QMOS,EMOS,CAT,VAL,ARO,DOM`). Chi tiết: `08_track2_spec.md`.

## 3. Timeline (hôm nay 1/6/2026)

| Mốc | Ngày | Ghi chú |
|---|---|---|
| Training data release | 22/5/2026 | ✅ đã có |
| Eval data release | 31/7/2026 | |
| **🔴 Hạn nộp kết quả** | **7/8/2026** | quan trọng nhất |
| Công bố kết quả | 31/8/2026 | |
| Hạn nộp paper ICASSP 2027 | 16/9/2026 | nếu publish |

→ Còn **~10 tuần** đến deadline nộp kết quả.

## 4. Baseline (đã clone về máy)
```
d:\VFS\VoiceMOS Challenge 2026\
├── vmc2026-baselines/   # baseline chính thức (track1/2/3)
├── UTMOS22/             # QMOS baseline
└── emotion2vec/         # EmoCat baseline
```

| Sub-task | Baseline | Trạng thái |
|---|---|---|
| QMOS | UTMOS (fairseq fork, torch 1.11) | clone xong, **chưa tải checkpoint** |
| EmoCat | Emotion2vec+ large (funasr) | clone xong, có `run_vmc2026.py` |
| EMOS / VAD | Gemini `gemini-3-flash-preview` | code sẵn, **cần API key** |

> Chi tiết setup + cách chạy trên Kaggle: xem `06_baseline_repos.md`.

## 5. Hướng nghiên cứu đang cân nhắc
- Fine-tune SSL backbone (WavLM / HuBERT / Wav2Vec2) cho MOS prediction
- Multi-task learning: dự đoán QMOS + EMOS đồng thời
- Kết hợp acoustic features + emotional embeddings (Emotion2vec)
- Cải tiến LLM-as-judge (prompt engineering trên Gemini)
- Khai thác phần optional VAD (ít người làm → dễ tạo novelty)

> ⚠️ Điểm novelty **chưa chốt** — chờ trao đổi mentor (xem `02_mentor_questions.md`).

> **Scope đã chốt (1/6/2026):** tham gia **cả 3 track**, dồn sức **Track 2** (hệ thống riêng), **Track 1 & 3 chỉ chạy baseline**.

## 6. Trạng thái & vấn đề
- **Đăng ký challenge:** ✅ đã xác nhận + ✅ **đã join CodaBench** (competition 16419).
- **GPU:** ✅ chốt dùng **Kaggle T4** (16GB, 30h/tuần). Cần verify phone để bật GPU+internet. Dự phòng: Colab Pro / cloud nếu thiếu giờ.
- **Data Track 2:** ✅ **đã gửi license** cho Erica Cooper (ecooper@nict.go.jp) — ⏳ **đang chờ BTC xác nhận** rồi gửi link data. (ESD/DailyTalk dùng Kaggle dataset — xem mục Liên kết.)
- **Mentor:** ⬜ còn câu hỏi **novelty** + co-author chưa có câu trả lời (scope & GPU đã tự chốt).
  > ⚠️ Lưu ý: data Track 2 **không** nằm sẵn trên CodaBench (chỉ Track 1 có sẵn ở tab description).

## 7. Tài liệu dự án

| File | Nội dung | Tình trạng |
|---|---|---|
| `README.md` | Tổng quan + index | ✅ |
| `00_challenge_overview.md` | Cuộc thi, 3 track, timeline | ✅ đầy đủ |
| `01_research_plan.md` | Kế hoạch 10 tuần, milestone | ✅ khung |
| `02_mentor_questions.md` | Câu hỏi mentor | ⬜ chờ trả lời |
| `03_literature_notes.md` | Ghi chú paper | ⬜ phần lớn trống |
| `04_experiments_log.md` | Nhật ký thí nghiệm | ⬜ chưa có exp |
| `05_setup_environment.md` | Setup môi trường, GPU | ✅ khung |
| `06_baseline_repos.md` | Clone repo + chạy Kaggle | ✅ |
| `07_project_summary.md` | Tóm tắt toàn bộ (file này) | ✅ |
| `08_track2_spec.md` | Đặc tả Track 2: dataset, 6 metric, format `answer.txt`, quy trình nộp | ✅ |
| `09_tracks_overview.md` | Đặc tả & so sánh cả 3 track | ✅ |

## 8. Việc cần làm tiếp (ưu tiên)
- ✅ Đăng ký + join CodaBench · ✅ gửi license Track 2/3 · ✅ chốt GPU=Kaggle · ✅ chốt scope · ✅ ESD/DailyTalk dùng Kaggle dataset
1. 🟢 **Quick win — nộp baseline Track 1 NGAY** (không cần license): chạy `kaggle_baseline/track1_baseline_pipeline.py` → `predictions.csv` → nộp. Thỏa luôn luật "≥1 lần training phase".
2. ⏳ **Chờ BTC xác nhận license** → nhận link data Track 2 + Track 3 (không chặn việc khác)
3. 🔴 **Verify phone Kaggle** → bật GPU T4 + internet; lấy **Gemini API key** → Kaggle Secrets
4. 🔴 **Test baseline Track 2** trên ESD (QMOS+EmoCat) bằng `track2_baseline_pipeline.py` (chưa cần data thật)
5. 🟡 Khi có data Track 2/3: chạy đủ baseline → gộp `answer.txt` → reproduce điểm → ghi mốc vào `04_experiments_log.md`
6. 🟢 Hỏi mentor **hướng novelty** track 2 + điền literature notes (UTMOS, Emotion2vec)

> Pipeline cả 3 track đã viết sẵn trong `kaggle_baseline/` (script sinh `predictions.csv`/`answer.txt` + zip).

---

## Liên kết quan trọng
- Website challenge: https://sites.google.com/view/voicemos-challenge/voicemos-challenge-2026
- Baseline chính thức: https://github.com/voicemos-challenge/vmc2026-baselines
- Nền tảng thi: https://www.codabench.org/
- **CodaBench competition (đã đăng ký):** https://www.codabench.org/competitions/16419/?secret_key=<XEM_EMAIL_BTC>
- Đăng ký (form gốc): https://forms.gle/L6YdkUf1PJdSSwLU7
- Liên hệ BTC: voicemos.challenge@gmail.com

### Data Track 2
- **License form Track 2** (điền + gửi Erica Cooper): https://drive.google.com/file/d/1XYFRWCzKHRelz6KGZJlTsR-PAdacJ__d/view
- License form Track 3 (nếu cần): https://drive.google.com/file/d/1wXHW784k5KQwik0YacXDjG95vo3zm5Ak/view
- Gửi license tới: **Erica Cooper — ecooper@nict.go.jp**
- ESD dataset (gốc): https://github.com/HLTSingapore/Emotional-Speech-Data
- DailyTalk dataset (gốc): https://github.com/keonlee9420/DailyTalk
- Đầu mối BTC: Wen-Chin Huang (Nagoya University, Toda Lab)

**→ Cách dùng trên Kaggle (đã chọn): KHÔNG tải về máy, dùng Kaggle Dataset có sẵn.**
- ESD trên Kaggle: `nguyenthanhlim/emotional-speech-dataset-esd` → mount tại `/kaggle/input/emotional-speech-dataset-esd/`
- DailyTalk trên Kaggle: `phngtvit/dailytalk` → mount tại `/kaggle/input/dailytalk/`
- Trong Notebook: panel phải → **Add Input / + Add Data** → search slug → **Add**.
- ⚠️ Đây là bản mirror cộng đồng (không phải bản chính thức) → khi nhận data Track 2 từ BTC nhớ đối chiếu số speaker/utterance/cấu trúc. ESD chuẩn: 20 speaker (10 EN + 10 ZH), 350 câu, 5 cảm xúc, ~29h.
