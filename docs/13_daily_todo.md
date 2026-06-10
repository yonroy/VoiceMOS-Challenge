# 13 — Todo

> Tick `[x]` khi xong. Mục mới nhất ở trên cùng.

---

## 📌📅 CHECKLIST EVALUATION PHASE (eval thả 31/7 · HẠN NỘP 7/8/2026) — GHIM
> **Cửa sổ chỉ ~1 tuần.** Chốt ranking chính thức ở phase này (DEV chỉ để luyện). Grader chỉ chấm `answer.txt` → được phép **trộn cột / ensemble**. Track 1/2 xếp theo **SRCC**, Track 3 theo **LCC** (utt + system). **Bắt buộc:** nộp trong phase này + system description + khai báo external data/license.

### 🔧 CHUẨN BỊ TRƯỚC (làm xong TRƯỚC 31/7 — đừng để nước tới chân)
- [x] **Khôi phục & lưu chắc mọi checkpoint** — Kaggle Dataset **`cache-exp8`** (cập nhật 10/6): gốc = `ft_qmos_utmos.pt` (exp13) + `ft_mamba_emotion_full.pt` (exp15); `archive/` = `ft_emotion_full_20epoch.pt` (exp08 TỐT NHẤT) + `ft_emotion_full.pt` + `ft_joint_full.pt` (exp11) + `aud_dev.npz`/`aud_train.npz`
- [ ] **Đóng băng pipeline inference** mỗi cột: exp07 (QMOS) + exp08 (5 cảm xúc) chạy được end-to-end, đã test trên DEV ra đúng `answer.txt`
- [ ] **Viết sẵn script TRỘN CỘT** (ghép QMOS←exp07 + cảm xúc←exp08 thành 1 answer.txt) + **script validate format** (đúng header/số dòng/range) — test trên DEV
- [ ] Ước lượng **thời gian chạy inference 2.730 eval** trên T4 (để biết cần mấy giờ GPU) + còn đủ **quota Kaggle 30h/tuần**?
- [ ] Chuẩn bị **system description (EN)** từ `19_paper_v1_en.md` + bảng license (SAILER Open RAIL · audeering CC BY-NC-SA · WavLM MIT · UTMOS · emotion2vec) — **non-commercial phải khai báo rõ**

### ▶️ KHI EVAL SET THẢ (≈31/7)
- [ ] Tải/Add Input **eval audio** (kiểm số lượng = 2.730, đúng định dạng wav 16kHz)
- [ ] Chạy **trích đặc trưng + inference** từng hệ trên eval → ra answer riêng mỗi nhánh
- [ ] **Trộn cột** → `answer.txt` 6 cột (wav,QMOS,EMOS,CAT,VAL,ARO,DOM) → **validate format** → zip
- [ ] (nếu kịp) tạo thêm **1–2 biến thể** để A/B: vd ensemble VAD (exp10) / đổi neo UTMOSv2

### 📤 NỘP (trước 7/8, chừa ≥1 ngày phòng lỗi)
- [ ] Nộp **Track 2** trên CodaBench (đúng track, bỏ chọn track khác) → đọc điểm eval
- [ ] Nộp **Track 1 & Track 3** (chạy lại baseline trên eval của 2 track đó) — vì phải nộp eval mới được tính hạng
- [ ] Nếu cho nhiều lượt: nộp biến thể tốt nhất, **chọn bản final** theo điểm eval
- [ ] **Ghi điểm eval** vào `18_leaderboard_history.md` + `12_` + `04_`

### 📝 SAU NỘP
- [ ] Hoàn thiện **system description** nộp BTC (EN, có hình kiến trúc) — hạn riêng của BTC
- [ ] Cập nhật `19_paper_v1_en.md` cột "Evaluation set" bằng điểm thật
- [ ] ⚠️ **Lưới an toàn:** nếu hệ chính lỗi giờ chót → có sẵn **bản fallback** (exp07 nguyên bản 6 cột) để nộp kịp hạn

> ⚠️ Rủi ro lớn nhất = **dồn hết vào tuần cuối**. Mọi script (inference/trộn cột/validate) phải **chạy ngon trên DEV trước 31/7**, eval chỉ việc đổi đường dẫn input.

---

## ✅ NGÀY 10/6/2026 (Phiên 21) — 🚀 exp13 NỘP: QMOS 0.548→~0.63 (kỷ lục cột QMOS) + vẽ kiến trúc exp_mix
- [x] 🚀 **Chạy + NỘP exp13** (fine-tune thẳng UTMOS trên nhãn qMOS) → **QMOS ~0.63** DEV — phá trần exp07 0.548 (trả nợ kéo dài từ Phiên 10); cột cảm xúc bản nộp không bằng exp08
- [x] **Vẽ kiến trúc từng layer hệ tốt nhất** (exp_mix: nhánh exp08 + exp07 + exp13 v2) → mục mới trong `04_` + mục 1–4 Track 2 của `12_system_description.md` (sơ đồ ASCII + bảng layer + external resources + training)
- [x] Nhận xét `100audio_emotion_scores.csv` (100/100 OK): neutral-bias rõ (68 neutral/32 happy, khử neutral → 97 happy); QMOS thấp đồng loạt 1.85 (lệch domain — chỉ dùng xếp hạng); VAD dải hẹp nhưng đúng hướng
- [x] exp15_predict: thêm in `val_emos` khi nạp ckpt (biết ngay ckpt smoke-test hay train thật); xác định `ft_mamba_emotion_full.pt` (1.27GB local, 8/6) = ckpt exp15 + hướng dẫn predict DEV
- [x] Đồng bộ điểm exp13 vào `04_` / `07_` / `12_` / `18_` / `20_`

### ➡️ VIỆC TIẾP THEO
- [ ] 🔴 **Tải `scoring_result.zip` exp13** về `submissions/Track2/exp13_finetune_qmos/` → ghi số chính xác 4 chữ số (hiện ~0.63 user báo) + 5 cột còn lại
- [ ] 🔴 **Ghép + NỘP bản trộn cột MỚI**: QMOS←exp13 (0.63) + 5 cảm xúc←exp08 → hệ 6 cột mạnh nhất mới (script ghép đã có từ exp_mix)
- [x] 🔴 **Upload ckpt lên Kaggle Dataset `cache-exp8`** — đã có ở GỐC dataset: `ft_qmos_utmos.pt` (exp13) + `ft_mamba_emotion_full.pt` (exp15); cache + ckpt cũ trong `archive/`. exp15_predict đã trỏ mặc định vào đây
- [ ] 🔒 (vẫn nợ) revoke token HF đã lộ; smoke test exp16 (LLM-judge)

---

## ✅ NGÀY 10/6/2026 (Phiên 20) — client Kaggle gọi API + notebook VoxCPM2 sinh emotion & chấm điểm + sửa metric neutral-bias
- [x] **Client Kaggle gọi API 3 track**: `kaggle_baseline/demo_api_client_kaggle.{ipynb,py}` (urllib, resume, CSV, nháp answer.txt)
- [x] **Notebook VoxCPM2 sinh emotion + chấm điểm** (`Tuần 1/VoxCPM2/VoxCPM2_Emotion_Generate_and_Score.ipynb`): Kaggle GPU, chấm **local** (exp08 + UTMOS), upload ref qua `/kaggle/input`
- [x] **Phát hiện neutral-bias trên tiếng Việt**: argmax luôn ra neutral (acc 20%) nhưng **VAD đúng hướng** (arousal angry/surprised cao, sad thấp; valence happy cao nhất)
- [x] **Sửa metric**: thêm `perceived_nn` (khử neutral), `cat_target_rank`, 2 accuracy, **Bước 8b SRCC theo VAD**
- [x] **Viết lại README** (bảng điểm 3 track + docs 00→21 + Demo UI/API HF) + thêm link Demo UI vào `07_`
- [x] **`.gitignore`** chặn `cache/`/`*.pt`/`*.npz`/`100audio/`/`*.wav` (tránh commit nhầm 4.7GB)
- [x] **Commit + push** `8790aff` lên `origin/main` (127 file reorg + api_service + demo client) — không lộ token

### ➡️ VIỆC TIẾP THEO
- [ ] 🟢 Chạy lại Bước 7→8b → gửi SRCC VAD mới (xác nhận hướng đo đúng)
- [ ] 🟠 Sửa TTS prosody: **reference audio CÓ cảm xúc** + tăng cfg/timesteps + thử prompt tiếng Việt
- [ ] 🟠 (tùy) thêm **emotion2vec** (đa ngôn ngữ) cho CAT tiếng Việt
- [ ] 🔴 (nợ) revoke token HF đã lộ; smoke test exp16 (LLM-judge) + exp13 (QMOS)

---

## ✅ NGÀY 10/6/2026 (Phiên 19) — slide Apple-clean + HTML + DEPLOY API service 3 track
- [x] Slide: thêm **giải thích từng layer + toán** cho cả 3 track — hình MỚI URGENT-MOS (Track 1) & ECAPA cosine (Track 3) + bổ sung toán C2/C3 (Track 2)
- [x] **Redesign Apple-clean** (CSS nhúng: nền trắng, font SF, bảng hairline, bullet accent, blockquote thẻ; title+divider lead)
- [x] **Render HTML** `slide/voicemos2026_slides.html` (marp-cli `--html --no-stdin`)
- [x] ⭐ **Xây API service 3 track** (`api_service/`): FastAPI REST + Docker, lazy-load, Track 2 thêm QMOS → đủ 6 cột
- [x] **PUSH + BUILD + RUNNING** trên HF Space `tranminhtoan140601/voicemos2026-api` (free CPU); verified `/health`,`/docs`; predict thật OK
- [x] **Đính chính:** URGENT-MOS chạy được trên HF free CPU (cả 3 track) — bác bỏ lo ngại "Track 1 quá nặng"
- [x] `score_100audio.py` — batch chấm emotion 100 file qua API → CSV (resume, zero-dep)

### ➡️ VIỆC TIẾP THEO
- [ ] 🔒 **REVOKE token HF** `hf_FZbh…` (lộ phiên này) + token cũ Phiên 16 → token mới
- [ ] 🟢 Đợi 100audio xong → đọc `100audio_emotion_scores.csv` (phân bố cảm xúc, QMOS cao/thấp)
- [ ] 🟠 Export slide PDF/PPTX (bật HTML) gửi mentor
- [ ] 🔴 (vẫn nợ) smoke test exp16 (LLM-judge) + exp13 (QMOS)

---

## ✅ NGÀY 10/6/2026 (Phiên 18) — slide present 3 track (mentor giao)
- [x] Tạo `docs/21_slides_3_tracks.md` — deck Marp tiếng Việt ~21 slide, cấu trúc như bài báo rút gọn (3 track, Track 2 trọng tâm)
- [x] Thêm 3 hình kiến trúc **SVG inline** trong chính file md (overview · fusion C2 · fine-tune C3) — không tạo file ảnh rời
- [x] Ghi hướng dẫn render Marp + lưu ý **bật HTML** (`enableHtml` / cờ `--html`) vào cuối file slide

### ➡️ VIỆC TIẾP THEO
- [ ] 🟠 Mở Marp preview (bật `markdown.marp.enableHtml`) kiểm 3 hình → export PDF/PPTX gửi mentor
- [ ] 🔴 Smoke test exp16 (LLM-judge) + exp13 (QMOS); 🔒 revoke token HF đã lộ (Phiên 16)

---

## 🆕 VIỆC MENTOR GIAO (9/6/2026)
- [x] 🎉 **Làm slide present cho cả 3 track (XONG 10/6, Phiên 18)** → `docs/21_slides_3_tracks.md` (Marp, tiếng Việt, ~21 slide, mạch như bài báo rút gọn) + 3 hình kiến trúc SVG inline (overview / fusion / fine-tune). _(còn: bật enableHtml → export PDF/PPTX gửi mentor)_
- [x] 🎉 **Đẩy TẤT CẢ lên Hugging Face (XONG 9/6, acc `tranminhtoan140601`):** 3 repo — checkpoint `voicemos2026-track2-emotion` (3 file .pt) · UI Space `voicemos2026-demo` (RUNNING) · code `voicemos2026-code` (55 file). Tất cả CC BY-NC-SA 4.0, không kèm data thô. File chuẩn bị ở `huggingface/` (PUSH_GUIDE.md).
- [x] **Tạo demo Gradio gộp 3 track** `kaggle_baseline/demo_all_tracks_gradio` + **`demo_run_from_hf`** (kéo UI từ HF chạy Kaggle GPU) — 3 tab lazy-load, Track 2 dùng exp08 (chỉ cảm xúc, KHÔNG QMOS)
- [x] **Nâng UI Space** clean light + Plotly (gauge/radar/bar + badge màu); fix gradio 4→6 → Space **RUNNING**
- [ ] 🔒 **REVOKE token HF** đã lộ trong chat 9/6 → tạo token mới nếu cần
- [ ] 🟢 (tùy) thêm cột **QMOS** vào tab Track 2 demo (ghép exp07/exp13) để demo đủ 6 cột

---

## ✅ Phiên 14 (9/6/2026) — demo cảm xúc + đổi focus sang 5 cột cảm xúc

### Đã xong
- [x] exp15: thêm RESUME (train tiếp từ ckpt) + fix CACHE_INPUT đệ quy + glob `(2)`
- [x] exp15_predict: file predict-only (nạp ckpt → answer, không train)
- [x] exp14: tạo `.ipynb` (Mamba head vào C2 đóng băng)
- [x] exp13: auto-dò DATA_ROOT cell 0 + chú thích `ft_emotion_full_20epoch.pt`
- [x] demo_track2_emotion_gradio (FILE MỚI, exp08, 5 cột + tab metric) — **đã chạy thật trên Kaggle**
- [x] Related Work §2 (EN + VI) + Introduction §1 (động lực) cho paper
- [x] chuyển `ft_qmos_utmos (1).pt` → `cache/ft_qmos_utmos.pt`

### Còn nợ / tiếp theo
- [x] **NỘP bản trộn cột exp07+exp08** lên CodaBench — đã nộp 9/6 (hệ 6 cột mạnh nhất)
- [ ] Điều tra **neutral-bias**: test demo ~3 mẫu/cảm xúc đã biết nhãn → bảng CAT-top1 vs nhãn thật
    - Đã có: **sad → ✅ đúng** (CAT sad 39% top1, neutral 15%, Phiên 15) · **surprised → ❌ thành neutral 63%** (Phiên 14) → cần thêm happy/angry/neutral để kết luận
- [ ] Nếu bias xác nhận → thử **class-weight / oversample** lớp hiếm (surprised) khi train cảm xúc
- [ ] Viết **exp17** — thêm data cảm xúc ngoài (ESD full / MSP-Podcast) cho CAT/VAD (phiên mới)
- [ ] Kiểm lại arXiv ID trong Related Work `19_` trước khi nộp paper
- [ ] Hỏi mentor: expressiveness = EMOS hay tổ hợp 5 cột; verdict KHỚP/LỆCH có là metric phụ?

---

## ✅ NGÀY 9/6/2026 (Phiên 17) — xác nhận điểm thật exp_mix (từ scoring zip) + bổ sung motivation paper §1
> ⚠️ Session này khởi đầu trên **snapshot docs cũ (Phiên 13)** → ban đầu tưởng exp_mix/checkpoint "chưa làm"; thực tế đã xong từ Phiên 14/16. Giá trị mới = xác nhận điểm + enrich paper.
- [x] **Xác nhận điểm thật exp_mix** đọc từ `submissions/Track2/exp_mix_q07_emo08/scoring_result (2).zip`: QMOS **0.5480** · EMOS **0.8111** · CAT err **0.1331** · VAL **0.6590** · ARO **0.7933** · DOM **0.7509** = khớp best-per-column (trước chỉ ghi "đã nộp" không có số)
- [x] **Bổ sung paper §1** (`19_`): đoạn mới "predictor = tín hiệu phản hồi để phát triển TTS cảm xúc (RLHF/model selection)" + ẩn dụ "emotional ruler" + mở rộng danh sách ứng dụng
- [x] Vẽ flowchart exp16 (trong chat) + giải thích mục đích/ứng dụng Track 2
- [x] Đồng bộ điểm số exp_mix vào `04_/07_/12_/18_/20_/05_` (slug Kaggle `toanminh222/cache-exp8`)

### ➡️ VIỆC TIẾP THEO
- [ ] 🔴 Smoke test **exp16** (Audio-LLM-as-Judge) → novelty cho paper (Bảng A)
- [ ] 🔴 Smoke test **exp13** (ckpt đã trên Kaggle) → phá trần QMOS 0.548?
- [ ] 🟠 exp14 ablation Mamba; điều tra neutral-bias (theo Phiên 14/15)
- [ ] 🔒 (nợ Phiên 16) **REVOKE token HF** đã lộ + làm slide 3 track (mentor)

---

## ✅ NGÀY 9/6/2026 (Phiên 13) — hoàn thiện exp13 (sửa ranking loss) + phân tích cải thiện QMOS
- [x] "Đọc" dự án → phát hiện **checkpoint còn ở `cache/`** (`ft_emotion_full_20epoch.pt` tốt nhất, `ft_emotion_full.pt`, `ft_joint_full.pt`) — không mất hẳn; chỉ cần upload Kaggle
- [x] Phân tích cải thiện **QMOS 0.548**: nguyên nhân (UTMOS lệch domain + MSE vs SRCC); 4 hướng ROI (exp09a UTMOSv2 → exp13 fine-tune → ranking loss → ensemble)
- [x] **🔧 HOÀN THIỆN exp13:** sửa lỗi ranking loss (gom MSE+pred cả cửa sổ ACCUM → backward 1 lần; đếm mẫu hợp lệ; flush dư cuối epoch; cảnh báo VRAM) → `py_compile` OK + đồng bộ `.ipynb`
- [x] Làm rõ "fine-tune ở dưới" trong exp14 = mở băng WavLM = mất cache + OOM T4 = **chính là exp15** → khuyến nghị exp14 đóng băng trước, fine-tune qua exp15

### ➡️ VIỆC TIẾP THEO
- [x] 🔴 (vẫn nợ) **NỘP** `exp_mix_q07_emo08/submission.zip` → ✅ xong 9/6 (Phiên 14)
- [x] 🔴 **Upload `ft_emotion_full_20epoch.pt` + cache audeering** lên Kaggle thành Dataset → ✅ xong 9/6 (Phiên 14)
- [ ] 🔴 Smoke test exp13 (`LIMIT_TRAIN=300, RANK_LAMBDA=0`) → so val SRCC vs UTMOS zero-shot; chưa vượt 0.548 thì thử `RANK_LAMBDA=0.3`
- [ ] 🟠 Chạy exp14 (USE_MAMBA on/off) → ablation Mamba; có tín hiệu → exp15

---

## ✅ NGÀY 8/6/2026 (Phiên 12) — hướng mới Audio-LLM-as-Judge (exp16) + học train/fine-tune
- [x] Buổi học **8 bài kinh nghiệm train/fine-tune** rút từ chính exp08/11/12 (fine-tune>freeze · val nội bộ=bẫy overfit · warm-start đỉnh→train thêm vô ích · checkpoint lưu đủ+mỗi best · data nhỏ đừng scratch · loss khớp metric · mẹo T4 · fusion≠ensemble) → ghi `03_`
- [x] Làm rõ **fusion vs trunk vs pooling**; Mamba (exp15) thay **pooling** không phải trunk; exp14 Mamba là nhánh cộng thêm
- [x] Chốt hướng "thêm GPT/LLM" = **gọi API audio-LLM-as-judge** (không fine-tune LLM nặng), mục tiêu **novelty cho paper**
- [x] **Code exp16** `exp16_llm_judge`: API Gemini/GPT-4o-audio chấm 6 cột → `answer.txt`; cờ `PROVIDER`+`SHOT_MODE`; **cache+resume** `.jsonl` (không trả tiền lại); parse JSON chịu lỗi; temp=0; tái dùng `load_target_emotions()`/format exp07; hàm `ensemble_rank_average` tùy chọn
- [x] Syntax OK (py_compile) + convert `.ipynb` (JSON hợp lệ)
- [x] Plan mode: viết + được duyệt plan exp16

### ➡️ VIỆC TIẾP THEO
- [ ] 🔴 (vẫn nợ) **NỘP bản trộn cột** `exp_mix_q07_emo08/submission.zip` — ROI cao nhất
- [ ] 🔴 **Smoke test exp16** `LIMIT=20`, `PROVIDER=gemini` zero-shot (Secrets `GEMINI_API_KEY`, Internet On) → kiểm JSON parse + cache resume + validate
- [ ] 🔴 Chạy full exp16 (2730) → nộp CodaBench → đọc SRCC → điền **Bảng A** (LLM vs exp07/exp08) cho paper
- [ ] ⚠️ **Xác nhận model ID nhận audio** trước khi chạy (`GEMINI_MODEL`=gemini-2.5-flash / `OPENAI_MODEL`=gpt-4o-audio-preview — có thể đã đổi; baseline từng dùng họ gemini-*-flash-preview)
- [ ] 🟠 Chạy exp16 few-shot và/hoặc OpenAI → **Bảng B** (zero vs few-shot, 2 LLM); tùy chọn ensemble rank-average
- [ ] 🟢 Khai báo external resource API (Gemini/OpenAI) trong `12_`

---

## ✅ NGÀY 8/6/2026 (Phiên 11) — buổi học củng cố hệ thống + báo cáo mentor
- [x] Học "5 mắt xích" hiểu cách hệ thống hoạt động (SSL/WavLM · pooling+head · fusion multi-task+uncertainty · freeze vs fine-tune · MSE-vs-SRCC) — trực giác + ví dụ + soi code thật (exp07/exp08) → ghi `03_literature_notes.md`
- [x] Đào sâu **toán self-attention** `softmax(QKᵀ/√d)·V` (Q/K/V, scaled dot-product, multi-head) + **mắt xích data** (nhãn listener-wise → MOS = `groupby.mean()`, EMOS cần target, train/val/eval split)
- [x] Soạn **báo cáo mentor** (bản ngắn: điểm số + việc hôm nay + câu hỏi) → lưu `11_progress_reports.md`; thêm 2 câu hỏi (kinh nghiệm fine-tune data nhỏ · novelty EMOS) vào `02_mentor_questions.md`
- [ ] (còn nợ) Đào tiếp **Mamba vs Transformer** (O(n) vs O(n²)) + **tiền xử lý audio** — đã đề xuất, chưa làm
- 🔴 (vẫn nợ từ Phiên 7) **nộp DEV bản trộn cột** exp07(QMOS)+exp08(cảm xúc)

---

## ✅ NGÀY 8/6/2026 (Phiên 10) — áp dụng SOTA Mamba (gợi ý mentor): exp13 + exp14 + exp15
- [x] Khảo sát SOTA mới: LLM-based (ALLD/SpeechQualityLLM) vs Mamba (MambaRate) → ghi `03_`; chọn **Mamba** làm trước (khả thi T4)
- [x] **Code exp13** `exp13_finetune_qmos`: FINE-TUNE thẳng UTMOS (`utmos22_strong`) trên nhãn `qMOS` thật + nạp ckpt cảm xúc exp08 20ep (`ft_emotion_full_20epoch.pt`) → answer 6 cột. Mục tiêu phá trần QMOS 0.548 (exp07). BATCH=1+ACCUM=16 (UTMOS không có attention-mask), LR 1e-5, `RANK_LAMBDA` cờ tùy chọn ranking loss, lưu `ft_qmos_utmos.pt` mỗi best
- [x] **Code exp14** `exp14_mamba_head`: nhánh Mamba 2 chiều trên WavLM frame-level (đóng băng) cộng vào fusion exp07 (6 cột); cờ `USE_MAMBA`; cache frame fp16
- [x] **Code exp15** `exp15_wavlm_mamba_emotion` ⭐: WavLM fine-tune (SAILER warm-start) + **Mamba head** thay mean-pool → 5 cột cảm xúc; cờ `USE_MAMBA` = ablation Mamba vs mean-pool
- [x] Nhúng Mamba thuần PyTorch (fp32) + thêm dòng cài `mamba-ssm causal-conv1d` (try/except, fallback an toàn)
- [x] Phòng gotcha: layerdrop=0, checkpoint lưu cả backbone+Mamba+heads, không đụng numpy
- [x] Ghi `04_` (hàng + mục exp13, exp14, exp15)
- [x] Tạo `.ipynb` cho exp13 + exp15 (exp14 còn .py)
- [x] 🎯 **Ghép bản TRỘN CỘT** QMOS(exp07)+cảm xúc(exp08) → `submissions/Track2/exp_mix_q07_emo08/submission.zip` (validate OK, 2730 dòng)
- [x] Nâng cấp exp15: **ranking loss** `RANK_LAMBDA=0.3` + **tự dò DATA_ROOT** + sửa cài mamba-ssm (`--no-build-isolation`+ninja)
- [x] Smoke test exp15 Kaggle: mamba-ssm fail → Mamba thuần PyTorch (fallback OK); học Internet phải On
- [x] Tạo `docs/20_experiments_overview.md` + thêm vào CLAUDE.md

### ➡️ VIỆC TIẾP THEO
- [ ] 🔴 **NỘP `exp_mix_q07_emo08/submission.zip`** lên CodaBench → chốt điểm hệ 6 cột mạnh nhất (đã sẵn, chỉ thiếu upload)
- [ ] 🔴 Chạy thật **exp13** (UTMOS-ft, LIMIT 300→None) → so QMOS vs exp07 0.548; vượt → thay cột QMOS
- [ ] 🔴 Chạy thật **exp15** (Mamba on/off) → bảng ablation Mamba vs mean-pool; thắng exp08 → nộp DEV
- [ ] 🟠 Nếu Mamba thuần PyTorch quá chậm: dùng kernel CUDA / giảm MAMBA_LAYERS/MAX_SECONDS / đổi head BiGRU-Transformer
- [ ] 🟠 Tạo `.ipynb` cho exp14 nếu cần chạy
- [ ] 🟡 (nợ Phiên 9) chạy đủ exp12 3 mode

---

## ✅ NGÀY 8/6/2026 (Phiên 9) — exp08b điểm thật · exp11 fusion 2 backbone · exp12 ablation khởi tạo
- [x] Đọc điểm exp08b (resume) đã nộp: MOS 0.4167 · EMOS 0.8116 · CAT 0.1331 · VAD 0.6605/0.7904/0.7539 (≈ exp08 → hội tụ)
- [x] Dời submission exp08b → `submissions/Track2/exp08b_finetune_resume/` (submission.zip + scoring_result.zip)
- [x] **Code exp11** `exp11_finetune_joint`: fine-tune CẢ WavLM + audeering, fusion 1 model, warm-start exp08; thêm cờ RESUME từ `ft_joint_full.pt`, `RESUME_LR_SCALE`; chống OOM
- [x] **Chạy exp11** → VAL nội bộ EMOS 0.8347/VAD 0.803/0.874/0.808 (mean 0.8298); warm-start đã đỉnh → KHÔNG cải thiện (early stop ep4)
- [x] **Code exp12** `exp12_wavlm_scratch`: ablation INIT_MODE scratch/base/sailer (trả lời mentor from-scratch vs fine-tune)
- [x] Fix **CheckpointError** (layerdrop=0) + **SystemError bad call flags** (khóa numpy + Restart Session) cho exp11/exp12
- [x] Chuẩn bị input ensemble: tách answer.txt exp08+exp07 → `submissions/Track2/_ens_inputs/`
- [x] Dời `ft_emotion_full.pt` + `ft_emotion_full_20epoch.pt` → `cache/`

### ➡️ VIỆC TIẾP THEO
- [ ] 🔴 **Nộp DEV bản trộn cột** (5 cảm xúc exp08/exp08b + QMOS exp07 0.548) — vẫn chưa nộp hệ 6 cột mạnh nhất
- [ ] 🟠 Chạy đủ **exp12** 3 mode (scratch/base/sailer) → ghi bảng ablation vào `04_` → trả lời mentor
- [ ] 🟠 Cân nhắc **nộp exp11 DEV** (đừng tin VAL nội bộ — VAD nội bộ 0.80 >> DEV 0.66 = overfit)
- [ ] 🟡 Hướng tăng điểm thật: **ranking loss** (tối ưu thẳng SRCC) / **ensemble** (exp10) thay vì vặn hyperparameter
- [ ] 🟢 Chạy ensemble exp10 với `_ens_inputs/` (đã sẵn answer exp08+exp07)

---

## ✅ NGÀY 5/6/2026 (Phiên 8) — 🧹 dọn & chuẩn hóa tên file/folder
- [x] Đổi tên 10 cặp pipeline `kaggle_baseline/track2/` → `expNN_tên` (bỏ `track2_`); giữ baseline/prepare_data/demo
- [x] Xóa `__pycache__` cũ (mang tên file cũ)
- [x] Chuẩn hóa `submissions/Track2/`: folder `expNN_tên` (sửa "fussion"/"emtion"); zip → `submission.zip`/`scoring_result.zip`; thư mục giải nén → `scores/`
- [x] Thay tên cũ trong 60 file (docs + README + memory + notebook); grep tên cũ = 0; notebook JSON hợp lệ
- [x] Thêm quy ước `expNN_tên` vào `CLAUDE.md` (mục 4)

### ➡️ VIỆC TIẾP THEO (giữ từ Phiên 7)
- [ ] 🔴 Train lại `exp08_finetune_emotion` → có lại `ft_emotion_full.pt` → Save Version ngay
- [ ] 🔴 Trộn cột: exp08 (5 cảm xúc) + exp07 QMOS (0.548) → nộp hệ 6 cột mạnh nhất
- [ ] 🟠 **Ensemble (giảm gap dev↔eval):** exp08 train **3 seed → trung bình điểm** + gộp exp07 (đa dạng kiến trúc) (+ tùy chọn 1 bản rank-loss). Trung bình điểm RỒI mới xếp hạng (SRCC). exp08 không cache → mỗi seed tốn nhiều giờ T4.
- [ ] 🟡 Nguyên tắc chọn bản nộp eval: **robustness > đỉnh dev**; luôn có fallback exp07
- [ ] ⚠️ `git add`/commit thay đổi đổi tên + dọn docs + file mới `18_`/`19_` (khi user OK)

---

## ✅ NGÀY 5/6/2026 (Phiên 7) — exp08 điểm thật · UTMOSv2 · mất backbone & vá · exp10 audeering ensemble
- [x] Đọc điểm exp08 đã nộp: EMOS **0.811**/CAT **0.133**/VAD **0.659·0.793·0.751** (thắng 5 cột); QMOS rớt 0.414
- [x] Research model QMOS mới → **UTMOSv2 (T05 vô địch VMC2024, MIT)**; đổi fallback QMOS exp08 UTMOS→UTMOSv2
- [x] Code `exp09a_qmos_utmosv2_probe` (A/B UTMOSv2 vs UTMOS trên train, không tốn lượt nộp)
- [x] Phát hiện + xử lý **mất backbone**: ckpt cũ chỉ lưu `heads` → vá exp08 lưu `ft_emotion_full.pt` **mỗi best** + tự copy cache (`CACHE_INPUT`)
- [x] Code `exp08b_finetune_resume` (resume từ ckpt full + cache; fix `weights_only=False`, slug `_`→`-`)
- [x] Code **exp10** `exp10_finetune_audeering` (fine-tune audeering riêng + ensemble VAD với exp08 — Hướng A cho T4, tránh OOM)

### ➡️ VIỆC TIẾP THEO
- [ ] 🔴 **Train lại exp08** (notebook đã vá, tái dùng cache) → có lại `ft_emotion_full.pt` → **Save Version NGAY**
- [ ] 🔴 **Trộn cột:** 5 cảm xúc exp08 + QMOS exp07 (0.548) → nộp hệ mạnh nhất 6 cột
- [ ] 🟠 Chạy **exp10** audeering (LIMIT nhỏ→None) → VAD ≥ exp08? → ensemble → nộp so sánh
- [ ] 🟡 Chạy **probe UTMOSv2**; thắng → exp09 fine-tune QMOS dùng UTMOSv2 làm neo
- [ ] 🟢 Ghi điểm thật vào `12_`; khai báo license UTMOSv2 (MIT)

---

## ✅ NGÀY 5/6/2026 (Phiên 6) — exp08 FINE-TUNE WavLM cho cảm xúc + tài liệu nền DL
- [x] Chốt hướng fine-tune (theo gợi ý mentor): **cho cảm xúc**, WavLM warm-start SAILER, thay e2v bằng audeering frozen
- [x] **Code exp08** `exp08_finetune_emotion_pipeline.py` + `.ipynb`: mở băng 6 lớp WavLM trên + audeering frozen → trunk → 3 head, AMP + grad-ckpt + grad-accum, uncertainty weighting
- [x] Đổi `EPOCHS` 8→12 (trần; early-stop quyết số epoch thật)
- [x] Hướng dẫn chạy Kaggle (LIMIT nhỏ→full, Save Version, OOM mitigation)
- [x] Xác nhận `track2cachingcheckpoint/` KHÔNG cần cho exp08 (tạo cache audeering riêng)
- [x] Tạo [16_model_architectures.md](16_model_architectures.md) + [17_dl_keywords.md](17_dl_keywords.md); web-verify SAILER arXiv 2505.22133
- [x] Học nền: fine-tune/train/embedding · kiến trúc WavLM · SSL · pretrain vs fine-tune · audeering vs e2v · Transformer≠LM · batch/epoch

### ➡️ VIỆC TIẾP THEO
- [ ] 🔴 Chờ exp08 chạy **full xong** → đọc khối `✅ VAL` cuối: EMOS có leo qua 0.795? → nộp nguyên exp08 (thắng cả 5 cột) hay **trộn cột** (EMOS←exp07 · VAD←exp08)
- [ ] 🟠 Nếu nộp → ghi điểm thật vào `04_`/`12_`; nếu trộn cột → viết script ghép answer.txt
- [ ] 🟡 Ablation: `UNFREEZE_TOP_LAYERS=0` vs `6` (frozen vs fine-tuned) · `USE_AUDEERING=False` → bảng cho paper
- [ ] 🟢 Khai báo license exp08 (WavLM MIT · SAILER Open RAIL · audeering CC BY-NC-SA) trong `12_`
- [ ] 🟢 (treo) exp06 head QMOS riêng · ablation exp04 4 cờ

---

## ✅ NGÀY 4/6/2026 (Phiên 5) — tấn công QMOS (exp06 + exp07) + cập nhật paper
- [x] Giải thích metric: ngoài SRCC chỉ còn **CAT-ERR**; khái niệm **UTT** (chấm từng câu) vs system-level
- [x] Làm rõ **QMOS đo gì** (artifact/độ sạch — phần lớn trực giao cảm xúc); QMOS hiện = UTMOS zero-shot (chưa train)
- [x] Lập **kế hoạch cải thiện QMOS** 4 bước (val nội bộ → head riêng → gộp fusion → ranking loss/fine-tune)
- [x] **Code exp06** `exp06_qmos_train_pipeline.py` + `.ipynb`: head QMOS riêng + UTMOS làm đầu vào, ghép vào answer.txt exp04
- [x] **Code exp07** `exp07_fusion_qmos_pipeline.py` + `.ipynb`: fusion 6 head (thêm QMOS), cảnh báo negative transfer, cờ `USE_UTMOS_FEAT`
- [x] Cập nhật **paper `15_`** theo exp04 (Method/Results/Ablation/Abstract)
- [x] Xác nhận `track2cachingcheckpoint/` tái dùng được + workflow copy cache Kaggle (gotcha read-only)

### ➡️ VIỆC TIẾP THEO
- [x] 🟠 Chạy + **NỘP exp07** → **QMOS 0.414→0.548 🚀, KHÔNG negative transfer (EMOS 0.795/VAD ≈ giữ), CAT 0.153** → HỆ THỐNG TỐT NHẤT
- [ ] 🔴 Chạy **exp06** trên Kaggle (LIMIT nhỏ→full) → A/B với exp07 (0.548): QMOS lên nhờ chia sẻ cảm xúc hay nhờ UTMOS-feature?
- [ ] 🟡 Ablation `USE_UTMOS_FEAT=False` (exp07) đo thẳng giả thuyết "chất lượng từ biểu diễn cảm xúc"
- [ ] 🟢 Upload `track2cachingcheckpoint/` thành Kaggle Dataset (đỡ trích lại) + Save Version giữ `utmos_*.npz`
- [ ] 🟢 (treo từ Phiên 4) chạy exp05 audeering · ablation exp04 4 cờ · khai báo license trong `12_`

---

## ✅ NGÀY 4/6/2026 (Phiên 3) — code exp04 fusion + exp05 audeering VAD
- [x] Viết **exp04 fusion multi-task** (`exp04_fusion_pipeline.py` + `.ipynb`): e2v+SAILER → trunk chung → 3 head (EMOS/CAT/VAD), uncertainty weighting, 4 cờ ablation
- [x] Fix `train.csv`: phân tách `|`, `emoCat` đa nhãn `,` → `sep="|"`
- [x] Ép GPU emotion2vec (`device=`) + tắt log funasr ồn (`disable_pbar/log/update`)
- [x] Viết **exp05 VAD audeering** (`exp05_vad_audeering`) — tách file riêng, giữ exp03 nguyên bản
- [x] Fix lỗi version transformers (audeering): bỏ subclass `PreTrainedModel` → `Wav2Vec2Model` + nạp head tay

### ➡️ VIỆC TIẾP THEO
- [x] 🔴 Chạy full **exp04** → nộp → **🏆 EMOS 0.788 · CAT 0.145 · VAL/ARO/DOM 0.578/0.754/0.706 (thắng cả 5 cột!)**
- [ ] 🔴 **Cải thiện QMOS** (vẫn 0.414 — cột DUY NHẤT chưa làm): train head chất lượng / fine-tune SSL
- [ ] 🟠 Chạy **exp05** (LIMIT=20 → None) → xem audeering có đẩy VAL > 0.578 nữa không
- [ ] 🟡 **Ablation exp04** (tắt từng nhánh USE_E2V/USE_SAILER/USE_UNCERTAINTY/USE_CLASSPROB) → bảng cho paper
- [ ] 🟢 **Save Version** giữ cache `fusion_cache/*.npz` (khỏi trích lại ~15 phút)
- [ ] 🟢 Khai báo license **audeering CC BY-NC-SA** + **SAILER Open RAIL** trong `12_system_description.md`

---

## ✅ NGÀY 4/6/2026 (Phiên 2) — exp01 emotion2vec EMOS 0.637 🏆 + chốt hướng fusion
- [x] Chạy exp01 (`EMOS_METHOD="emotion2vec"`) full + nộp CodaBench
- [x] Đọc `scores.json`: QMOS 0.4139 · **EMOS 0.6365** · CAT err 0.1933 → **VƯỢT cả SAILER 0.562**
- [x] Học sâu: SRCC chấm thứ hạng · hiện tượng "dồn 2 cực" · vòng đời điểm số · cấu trúc data (`train.csv` ~7 người/câu)
- [x] Khảo sát đội thắng 2024 (fusion 2–3 luồng) · metric 2024 (MSE/LCC/SRCC/KTAU) · cảm xúc = track mới 2026
- [x] **Chốt hướng fusion multi-task: "QMOS riêng + 5 cảm xúc chung"** → ghi `03_literature_notes.md`
- [x] Ghi `04_`, `12_`, `07_`, `11_` (điểm exp01 + hướng fusion)

### ➡️ VIỆC TIẾP THEO (session sau)
- [ ] 🟠 Nộp **bản lai** (EMOS+CAT←emotion2vec · VAD←SAILER · QMOS←SpeechMOS) → điểm tổng tốt nhất
- [x] 🟡 **Web-search xác minh prior art** (4/6): fusion KHÔNG mới (2204.04855 fusion-MOS; SER fusion nhiều; SVAS 2411.02625) → định vị lại novelty = **task EMOS + phát hiện emotion2vec>SAILER + multi-task hợp nhất**. Xem `03_`.
- [ ] 🟡 **Code fusion multi-task** (5 cảm xúc): emotion2vec+SAILER → concat → head, freeze+cache trên T4
- [ ] 🟢 QMOS: train head chất lượng riêng (vượt 0.414)

---

## ✅ NGÀY 4/6/2026 (Thứ Tư) — exp03 SAILER thành công 🎉
- [x] Khảo sát model thay emotion2vec (SenseVoice ❌ chỉ ra nhãn; chốt **SAILER** WavLM-large)
- [x] Ghi khảo sát + benchmark vào `03_literature_notes.md`
- [x] Chia `kaggle_baseline/` theo track (track1/ track2/ track3/) + cập nhật README
- [x] Viết `track2/exp03_emos_sailer.ipynb` (EMOS+CAT+VAD bằng SAILER, offline)
- [x] Fix lỗi: bỏ `pip install -e .` → clone+sys.path; unpack 6 giá trị forward; đẩy GPU T4
- [x] **Nộp exp03 → EMOS 0.194→0.562 🚀; CAT 0.190; VAD ARO 0.712/DOM 0.630/VAL 0.341**
- [x] Cập nhật `04_`, `12_`, `07_`, `11_`

### ➡️ VIỆC TIẾP THEO
- [ ] 🟠 Đẩy **VAL** (0.341 — thấp nhất): thử model VAD chuyên (audeering / tiantiaf MSP-dim)
- [ ] 🟡 **Cách B:** train head trên SAILER embedding → xem có vượt EMOS 0.562
- [ ] 🟢 Cải thiện **QMOS** (vẫn 0.414) — fine-tune SSL backbone
- [ ] 🟢 Khai báo license Open RAIL (SAILER) trong system description cuối

---

## Todo ngày 3/6/2026 (Thứ Ba)

> Trọng tâm: **data Track 2 đã về** → chạy trọn pipeline Track 2 → nộp baseline Track 2 lần đầu.
> Tick `[x]` khi xong. **Track 3 cũng đã đủ data** (gói VCTK về 3/6) → xem mục 6.

---

## 🔴 1. Ráp data Track 2 đầy đủ (15.477 wav) — notebook `track2_prepare_data.ipynb`
- [ ] Settings → **Internet = On** (bắt buộc để build sv56); GPU không cần
- [ ] + Add Input đủ **3 dataset**: gói Track 2 · ESD (`Emotional Speech Dataset (ESD).zip`) · DailyTalk (`dailytalk.zip`)
- [ ] Run All → ráp ESD (1.379) + DailyTalk (38) + chuẩn hóa sv56
- [ ] Xác nhận log báo **✅ ĐỦ 15.477 file**
- [ ] **Save Version (Commit)** → từ output **Create Dataset** mới

## 🔴 2. Chạy baseline Track 2 → tạo `answer.txt` — notebook `track2_baseline.ipynb`
- [ ] Settings → Accelerator = **GPU T4**, Internet = **On**
- [ ] + Add Input = dataset 15.477 wav vừa tạo
- [ ] Sửa `DATA_ROOT` ở cell 0 cho khớp slug dataset mới
- [ ] Chạy thử `LIMIT = 20` trước → OK rồi đặt `None` chạy full tập DEV (~2.730 mẫu)
- [ ] (EMOS) thêm Secret `GEMINI_API_KEY` + bỏ comment dòng `Gemini_EMOS.py`/`Gemini_VAD.py` — *chưa kịp thì để mặc định, cải thiện sau*
- [ ] Để notebook chạy `build_answer` + `validate` → ra `submission_track2.zip`

## 🟠 3. Nộp Track 2 lên CodaBench
- [ ] My Submissions → chọn **Track 2**, **bỏ chọn** track khác → upload `submission_track2.zip`
- [ ] Ghi điểm vào `04_experiments_log.md` (hàng baseline) + `12_system_description.md` bảng Track 2

## 🟡 4. Chốt nốt Track 1 + ghi tiến độ
- [x] Giải nén `submissions/Track1/scoring_result.zip` → đọc điểm ACR/CCR (0.662 / 0.411)
- [x] Ghi điểm Track 1 vào `12_system_description.md` (bảng Track 1, cột Dev)
- [x] Viết **báo cáo ngày 3/6** vào `11_progress_reports.md`

## 🟢 5. Nếu còn thời gian — học nền
- [ ] Đọc lướt paper **UTMOS** (= QMOS baseline) → ghi 3–5 dòng vào `03_literature_notes.md`

## ✅ 6. Track 3 — ĐÃ NỘP (SPK 0.451 / ACC 0.440)
- [x] Upload `vmc2026-track3` lên Kaggle bằng CLI (gói `_syn` + `_vctk`)
- [x] **Gộp VCTK** → tổng **3.548 wav** (notebook tự dò + gộp)
- [x] Chạy `track3_baseline.ipynb` (ECAPA fine-tuned) → inference 600 cặp dev
- [x] Nộp Track 3 → SPK **0.451** · ACC **0.440** (khớp baseline ~0.45/0.44)

---

**🎯 Tối thiểu hôm nay:** ráp xong data Track 2 + nộp được baseline Track 2 (dù EMOS tạm mặc định).
**⚠️ Nhớ:** phải **Save Version → Create Dataset** ở bước 1 xong mới chạy được bước 2.
**✅ Hết bị chặn:** Track 3 đã đủ data (3/6) — ưu tiên sau khi xong Track 2.

---

### ✅ ĐÃ ĐẠT CUỐI NGÀY 3/6 — ĐỦ CẢ 3 TRACK TRÊN LEADERBOARD 🎉
- ✅ Ráp 15.477 wav Track 2 → dataset `vmc2026-track2-full`
- ✅ Nộp **Track 1**: ACR/CCR = **0.662 / 0.411**
- ✅ Nộp **Track 2**: QMOS **0.414** · EMOS **0.194** (496/2730) · CAT err **0.193**
- ✅ Nộp **Track 3**: SPK **0.451** · ACC **0.440** (ráp 3.548 wav, gộp VCTK)
- ✅ Ghi điểm vào `04_experiments_log.md`, `12_system_description.md`; báo cáo `11_progress_reports.md`

### ➡️ CÒN LẠI (mai/khi rảnh)
- 🟡 `--resume` chấm nốt EMOS 2.234 mẫu → nộp lại Track 2 (cải thiện EMOS 0.194)
- 🟢 Đọc UTMOS (mục 5) → chốt hướng cải tiến QMOS vượt 0.414
- 🟢 Trao đổi mentor hướng novelty Track 2

### 🆕 Định hướng chốt ở phiên chiều 3/6 (xem `11_progress_reports.md`)
- 🔴 **EMOS cách A (ưu tiên):** sửa pipeline → bỏ Gemini, lấy **xác suất lớp target từ emotion2vec** làm điểm EMOS (offline, miễn phí) → nộp lại Track 2.
- 🟠 **exp01 multi-task:** train 1 backbone chung + nhiều head (QMOS/EMOS/CAT/VAD) trên 12.746 mẫu có nhãn — khung đã ghi sẵn ở `04_experiments_log.md`.
- 🟢 **Dọn dẹp:** xác nhận xóa 2 folder `UTMOS22/` + `emotion2vec/` (dư ~23 MB cho workflow Kaggle) — **đang chờ user OK.**
- ℹ️ Đã tạo `CLAUDE.md` (quy ước đọc/“xong”/cảnh báo token). Challenge KHÔNG bắt buộc dùng baseline.

---

### ✅ PHIÊN TỐI 3/6 — sắp xếp dự án + exp01 + mở khóa train
- [x] **Sắp xếp lại thư mục:** docs/ · baselines/ · reference/ · submissions/ (cập nhật CLAUDE.md, README, .gitignore, link nội bộ). **Chưa commit.**
- [x] Tạo `docs/15_paper_draft.md` (khung paper ICASSP 2027) → bắt đầu "vừa làm vừa viết".
- [x] **exp01:** sửa pipeline `.py` + notebook `track2_baseline.ipynb` → `EMOS_METHOD="emotion2vec"` (EMOS = 1+4·P(target), offline). Chạy thử LIMIT=20 OK.
- [x] Chốt: Track 1 KHÔNG có train data · Track 3 có (finetune.py) · Track 2 train được.
- [x] **Mở khóa exp02:** xác nhận `sets/train.csv` đủ nhãn (qMOS/eMOS/emoCat/val/aro/dom theo listener).

### ➡️ VIỆC TIẾP THEO (session sau)
- [ ] 🔴 Chạy exp01 full (`LIMIT=None`) → nộp Track 2 → ghi điểm EMOS mới (so 0.194) vào `04_` + `12_`.
- [x] 🟠 **exp02 (EMOS có train) — ĐÃ CODE 3/6:** gộp train.csv theo wav → emotion2vec (đóng băng) + MLP head → `exp02_train_emos.ipynb`. *Chưa chạy thật.*
- [ ] 🟢 Tinh chỉnh EMOS (bỏ scale / softmax temperature) nếu SRCC bị trùng hạng.
- [ ] 🟢 Đọc UTMOS → `03_literature_notes.md` (cho Related Work của paper).

---

### ✅ PHIÊN ĐÊM 3/6 — code train EMOS (exp02 EMOS-only) + cập nhật CLAUDE.md
- [x] Chốt thiết kế exp02 giai đoạn EMOS-only: **emotion2vec đóng băng + MLP head**, Kaggle T4.
- [x] Viết `kaggle_baseline/exp02_train_emos_pipeline.py` + `exp02_train_emos.ipynb` (jupytext, cú pháp OK).
- [x] Cập nhật `CLAUDE.md` mục 3: model Opus 4.8 / context 1M + mốc token tuyệt đối.
- [x] Tổng hợp câu hỏi mentor (xem `02_mentor_questions.md`).

### ➡️ VIỆC TIẾP (session sau)
- [ ] 🔴 **Chạy `exp02_train_emos.ipynb` trên Kaggle:** sửa `DATA_ROOT` → thử `LIMIT_TRAIN=300` → `None` → xem VAL SRCC (so 0.194).
- [ ] 🟠 Nộp `answer.txt` exp02 lên CodaBench → ghi điểm EMOS thật vào `04_` + `12_` + `07_`.
- [ ] 🟢 Nếu khá → mở rộng **multi-task** (thêm head QMOS/CAT/VAD chung backbone).
