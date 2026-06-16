# 07 — Tóm tắt toàn bộ dự án

> Tài liệu tổng hợp nhanh để tra cứu / gửi mentor. Cập nhật ngày: 16/6/2026.

> **Trạng thái leaderboard (cập nhật 10/6 — tốt nhất TỪNG CỘT):** Track 1 ACR/CCR = **0.662 / 0.411** · **Track 2 (best per-column): QMOS 0.6296 🏆(exp13) · EMOS 0.8116 🏆(exp08b) · CAT err 0.1331 🏆(exp08) · VAL 0.6605 🏆(exp08b) · ARO 0.7978 🏆(exp15 Mamba) · DOM 0.7539 🏆(exp08b)** · Track 3 SPK/ACC = **0.451 / 0.440**. → Hệ mạnh nhất 6 cột MỚI = **trộn cột QMOS←exp13 + ARO←exp15 + EMOS/CAT/VAL/DOM←exp08 (CHƯA NỘP bản trộn)**. Chi tiết: `04_experiments_log.md`, `12_system_description.md`.
>
> 🚀 **Mốc 10/6 (Phiên 21) — 2 KỶ LỤC CỘT TRONG 1 NGÀY:** (1) **exp13** fine-tune thẳng UTMOS → QMOS **0.548 → 0.6296** (+0.082, cột đứng yên từ 4/6); (2) **exp15** Mamba head → **ARO 0.7933 → 0.7978** (NỘP bản exp15_predict 2-ckpt: cảm xúc←exp15 + QMOS←exp13; Mamba ≈ mean-pool ở 4 cột còn lại → ablation cho paper). `submissions/Track2/exp15_predict/`.
>
> 🎉 **Mốc 4/6 — exp07 hợp nhất 6 cột (HỆ THỐNG CHÍNH cho paper):** thêm **head QMOS thứ 4** (đầu vào `[trunk | UTMOS]`) vào trunk fusion exp04 → 1 model dự đoán trọn 6 cột. **QMOS 0.414→0.548** 🚀 (lần cải thiện QMOS đầu tiên, +0.134) mà **KHÔNG kéo tụt** 5 cột cảm xúc (EMOS 0.788→0.795). Tiền đề exp04: gộp emotion2vec + SAILER (đóng băng) → trunk chung → multi-task + uncertainty weighting → thắng mọi model lẻ ở cảm xúc. Điểm trừ nhỏ: CAT 0.145→0.153.
> ➡️ **Việc tiếp:** chạy exp06 (head QMOS riêng) A/B với 0.548 để biết QMOS lên nhờ chia sẻ cảm xúc hay nhờ UTMOS-feature; ablation cho paper.
>
> 🏆 **Mốc 5/6 (Phiên 7) — exp08 FINE-TUNE NỘP (điểm thật):** EMOS **0.811** · CAT err **0.133** · VAL/ARO/DOM **0.659/0.793/0.751** → **THẮNG cả 5 cột cảm xúc** vs exp07. QMOS rớt 0.414 (bản nộp không mượn exp07). → **Hệ tốt nhất 6 cột = trộn cột: 5 cảm xúc exp08 + QMOS exp07 (0.548)** (chưa nộp).
> 🔴 **Sự cố:** ckpt exp08 gốc chỉ lưu `heads` → backbone fine-tune **mất khi kernel chết** → đã vá lưu `ft_emotion_full.pt` mỗi best; **phải train lại exp08**.
> 🔬 **Research + code mới (chưa chạy):** UTMOSv2 (T05 vô địch VMC2024, MIT) thay UTMOS cho QMOS — notebook probe A/B; `exp08b_finetune_resume` (resume từ ckpt+cache); `exp10_finetune_audeering` (exp10: fine-tune audeering riêng + ensemble VAD — Hướng A an toàn T4). Tài liệu nền: `16_model_architectures.md`, `17_dl_keywords.md`.
>
> 📌 **Mốc 8/6 (Phiên 9):** (1) **exp08b** (resume exp08) NỘP — MOS 0.4167·EMOS 0.8116·CAT 0.1331·VAD 0.6605/0.7904/0.7539 ≈ exp08 → **checkpoint đã hội tụ**. (2) **exp11** (fine-tune CẢ WavLM+audeering, fusion 1 model) đã chạy — VAL nội bộ 0.8298 nhưng **train thêm KHÔNG cải thiện** (warm-start đã đỉnh); chưa nộp DEV. (3) **exp12** (ablation khởi tạo scratch/base/sailer) code xong theo gợi ý mentor "from-scratch vs fine-tune" — chưa đủ số 3 mode. ⚠️ Best-per-column hầu như giữ nguyên (exp08b ≈ exp08). Việc gấp: **nộp DEV bản trộn cột** (chưa làm).

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

## 3. Timeline (hôm nay 16/6/2026)

| Mốc | Ngày | Ghi chú |
|---|---|---|
| Training data release | 22/5/2026 | ✅ đã có |
| Eval data release | 31/7/2026 | |
| **🔴 Hạn nộp kết quả** | **7/8/2026** | quan trọng nhất |
| Công bố kết quả | 31/8/2026 | |
| Hạn nộp paper ICASSP 2027 | 16/9/2026 | nếu publish |

→ Còn **~7–8 tuần** đến deadline nộp kết quả (52 ngày từ 16/6 → 7/8).

## 4. Baseline (đã clone về máy)
```
d:\VFS\VoiceMOS Challenge 2026\
└── baselines/               # (gom 3 repo, sắp xếp lại 3/6)
    ├── vmc2026-baselines/   # baseline chính thức (track1/2/3)
    ├── UTMOS22/             # QMOS baseline
    └── emotion2vec/         # EmoCat baseline
```

| Sub-task | Baseline | Trạng thái |
|---|---|---|
| QMOS | SpeechMOS (UTMOS22_strong, qua torch.hub) | ✅ chạy GPU, QMOS SRCC 0.414 |
| EmoCat | Emotion2vec+ large (funasr) | ✅ chạy GPU, CAT err 0.193 |
| EMOS / VAD | Gemini `gemini-3-flash-preview` | ✅ chạy (billing prepaid); EMOS mới 496/2730, VAD bỏ |

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
- **GPU:** ✅ Kaggle T4 (16GB, 30h/tuần), đã verify phone, GPU+internet bật ổn. Dự phòng: Colab Pro nếu thiếu giờ.
- **Data Track 2 & 3:** ✅ **đã nhận đủ (3/6)**. Track 2 = gói BTC + ESD + DailyTalk (ráp 15.477 wav). Track 3 = gói `_syn` + `_vctk` → 3.548 wav.
- **Mentor:** 🟢 đã có **nhiều gợi ý kỹ thuật được áp dụng** — fine-tune (→ exp08), ablation from-scratch (→ exp12), thử SOTA mới như Mamba/LLM-based (→ exp14/15). ⬜ câu hỏi **novelty cuối** + co-author cho ICASSP 2027 còn ngỏ.
- **Vấn đề hiện tại (9/6):**
  - ✅ **ĐÃ nộp DEV bản trộn cột 9/6** (exp_mix: exp08 5 cảm xúc + exp07 QMOS 0.548) — điểm thật khớp best-per-column. Món nợ kéo dài nhiều phiên đã chốt. ✅ Checkpoint cảm xúc + cache đã upload Kaggle Dataset (chống mất).
  - 🟠 exp11 (fusion 2 backbone) val nội bộ 0.83 nhưng nghi **overfit** (>> DEV exp08 0.66) → cần nộp DEV 1 lần để biết thật.
  - 🟠 exp12 (ablation khởi tạo) **chưa đủ 3 mode** (scratch/base/sailer) → chưa trả lời được mentor "from-scratch vs fine-tune".
  - 🟡 exp13/exp14/exp15 vừa code 8/6, **chưa smoke test** lần nào.
  > ⚠️ Lưu ý: data Track 2/3 **không** nằm sẵn trên CodaBench (chỉ Track 1 dev có sẵn ở tab description); phải tự ráp từ gói tải về / Kaggle dataset.

## 7. Tài liệu dự án

| File | Nội dung | Tình trạng |
|---|---|---|
| `README.md` | Tổng quan + index | ✅ |
| `00_challenge_overview.md` | Cuộc thi, 3 track, timeline | ✅ đầy đủ |
| `01_research_plan.md` | Kế hoạch 10 tuần, milestone | ✅ khung |
| `02_mentor_questions.md` | Câu hỏi mentor | ⬜ chờ trả lời |
| `03_literature_notes.md` | Ghi chú paper | ⬜ phần lớn trống |
| `04_experiments_log.md` | Nhật ký thí nghiệm | ✅ có baseline (3/6) |
| `05_setup_environment.md` | Setup môi trường, GPU | ✅ khung |
| `06_baseline_repos.md` | Clone repo + chạy Kaggle | ✅ |
| `07_project_summary.md` | Tóm tắt toàn bộ (file này) | ✅ |
| `08_track2_spec.md` | Đặc tả Track 2: dataset, 6 metric, format `answer.txt`, quy trình nộp | ✅ |
| `09_tracks_overview.md` | Đặc tả & so sánh cả 3 track | ✅ |

## 8. Việc cần làm tiếp (ưu tiên — cập nhật 8/6)

### Đã xong (mốc lịch sử — giữ tham chiếu)
- ✅ Đăng ký + join CodaBench · ✅ verify phone · ✅ chốt scope · ✅ ráp 15.477 wav Track 2 + 3.548 wav Track 3
- ✅ **Nộp baseline 3 track (3/6):** T1 0.662/0.411 · T2 0.414/0.194/0.193 · T3 0.451/0.440
- ✅ **Track 2 — vượt baseline xa:** exp01→exp08 đẩy điểm tốt nhất từng cột lên QMOS 0.548 · EMOS 0.811 · CAT 0.133 · VAD 0.659/0.793/0.751 (xem dòng trạng thái đầu file)

### Đang nợ / phải làm (theo độ ưu tiên)
1. ✅ **Nộp DEV bản trộn cột (XONG 9/6)** = exp_mix QMOS 0.548 + EMOS 0.811 + CAT 0.133 + VAD 0.659/0.793/0.751. Checkpoint + cache cũng đã lên Kaggle Dataset.
2. 🔴 **Smoke test exp13** (UTMOS fine-tune cho QMOS) — phá trần 0.548 hay không? Lưới an toàn: chỉ nộp nếu VAL nội bộ > UTMOS zero-shot.
3. 🔴 **Smoke test exp15** (Mamba head trên WavLM ft) — vượt exp08 ở 5 cột cảm xúc?
3b. 🔴 **Smoke test + chạy exp16** (Audio-LLM-as-Judge, API Gemini/GPT-4o-audio) — novelty cho paper; nộp → Bảng A (LLM vs exp07/exp08). Code xong 8/6 (Phiên 12), chưa chạy.
4. 🟠 **Chạy đủ exp12 3 mode** (scratch/base/sailer) → bảng ablation trả lời mentor "from-scratch vs fine-tune".
5. 🟠 **Nộp exp11 DEV** 1 lần để biết điểm thật (val nội bộ 0.83 nghi overfit).
6. 🟠 Chạy ensemble **exp10** (audeering ft + exp08) → so VAD với exp08 thuần.
7. 🟡 **Chuẩn bị Evaluation Phase** (eval thả 31/7, hạn nộp **7/8/2026**): script trộn cột + validate format + lưu ckpt thành Kaggle Dataset. Xem checklist GHIM ở đầu [13_daily_todo.md](13_daily_todo.md).
8. 🟡 Cập nhật [19_paper_v1_en.md](19_paper_v1_en.md) với kết quả mới (exp08b/11/14/15) + bảng "accuracy-vs-cost" cho góc novelty practical/efficient.
9. 🟢 Chốt câu hỏi **novelty cuối** + co-author với mentor cho hạn ICASSP 16/9/2026.

> Pipeline 3 track + 13 experiment Track 2 nằm trong `kaggle_baseline/` (đặt tên `expNN_tên.{ipynb,py}` từ 5/6 Phiên 8).

---

## Liên kết quan trọng
- **Demo UI (Gradio 3 tab, HF Space):** https://huggingface.co/spaces/tranminhtoan140601/voicemos2026-demo (chạy nhanh trên Kaggle T4 qua `kaggle_baseline/demo_run_from_hf.ipynb`)
- **API service 3 track (HF Space, Phiên 19):** https://tranminhtoan140601-voicemos2026-api.hf.space (Swagger `/docs`) — FastAPI REST, code ở `api_service/`
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
