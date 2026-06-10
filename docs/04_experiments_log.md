# 04 — Nhật ký Thí nghiệm

Ghi lại MỌI thí nghiệm. Đây là tài liệu quan trọng nhất khi viết paper — ablation study và bảng kết quả đều lấy từ đây.

> Quy tắc: mỗi experiment ghi đủ **config → kết quả → nhận xét**. Không bao giờ chạy mà không ghi.

---

## Bảng tổng hợp kết quả

| Exp ID | Mô tả | Backbone | QMOS SRCC | EMOS SRCC | Note |
|---|---|---|---|---|---|
| baseline | UTMOS+emotion2vec+Gemini | wav2vec2 | 0.414 | 0.194 | DEV 3/6; CAT err 0.193; EMOS chỉ 496/2730 thật |
| **exp01** | EMOS: Gemini→emotion2vec target-prob (offline) | emotion2vec | 0.414 | **0.637** ✅ | **NỘP 4/6: EMOS 0.194→0.6365 (VƯỢT cả SAILER 0.562!); QMOS 0.4139; CAT err 0.1933** |
| exp02 | EMOS có train: emotion2vec (đóng băng) + MLP head | emotion2vec | (giữ) | mục tiêu >0.194 | ĐÃ CODE 3/6 (pipeline + notebook), chờ chạy Kaggle. Giai đoạn EMOS-only của hướng multi-task |
| **exp03** | **EMOS+CAT+VAD bằng SAILER (offline, không train)** | **WavLM-large (SAILER)** | 0.414 (giữ) | **0.562** ✅ | **NỘP 4/6: EMOS 0.194→0.562 🚀; CAT 0.190; VAD mới ARO 0.712/DOM 0.630/VAL 0.341** |
| **exp04** | **FUSION multi-task:** e2v+SAILER (đóng băng) → trunk chung → 3 head (EMOS/CAT/VAD), uncertainty weighting | emotion2vec + WavLM(SAILER) | 0.414 (giữ) | **0.788** 🏆 | **NỘP 4/6: EMOS 0.637→0.788 🚀; CAT err 0.145; VAL 0.578/ARO 0.754/DOM 0.706 — VƯỢT MỌI model lẻ ở cả 5 cột cảm xúc** |
| **exp05** | VAD bằng **audeering MSP-dim** (thay cả 3 VAD); EMOS/CAT giữ SAILER | wav2vec2 (audeering) | 0.414 (giữ) | 0.562 (giữ SAILER) | ĐÃ CODE 4/6, chờ chạy. Mục tiêu đẩy VAL (SAILER 0.341). Tách file riêng, giữ exp03 nguyên bản |
| **exp06** | **TRAIN head QMOS** trên đặc trưng cache (e2v+SAILER) + điểm UTMOS làm đầu vào | e2v+SAILER (head MLP) | mục tiêu >0.414 | (giữ exp04) | ĐÃ CODE 4/6 (Phiên 5), chờ chạy. Tấn công cột QMOS — cột duy nhất chưa train. Ghép QMOS mới vào answer.txt exp04 |
| **exp07** | **FUSION + QMOS head** (hợp nhất 6 cột): thêm head QMOS thứ 4 vào trunk exp04 | e2v+SAILER (6-head MTL) | **0.548** 🏆 | **0.795** 🏆 | **NỘP 4/6 (Phiên 5): QMOS 0.414→0.548 🚀 (+0.134); EMOS 0.795; CAT err 0.153; VAL 0.581/ARO 0.752/DOM 0.705. KHÔNG negative transfer. QMOS 0.548 = tốt nhất; 5 cột cảm xúc sau đó bị exp08 vượt → hệ mạnh nhất = trộn cột exp08(cảm xúc)+exp07(QMOS)** |
| **exp08** | **FINE-TUNE WavLM** (warm-start SAILER, mở băng 6 lớp trên) + audeering frozen → trunk → 3 head cảm xúc | **WavLM-large (fine-tune)** + audeering | 0.4139 (UTMOS) | **0.811** 🏆 | **NỘP 5/6 (Phiên 7): EMOS 0.811 · CAT err 0.133 · VAL 0.659/ARO 0.793/DOM 0.751 — THẮNG cả 5 cột cảm xúc vs exp07.** QMOS rớt 0.414 (bản nộp KHÔNG mượn exp07). ⚠️ Backbone MẤT (kernel chết, ckpt cũ chỉ lưu heads) → đã vá lưu `ft_emotion_full.pt` mỗi best; phải train lại |
| **exp09a** | **PROBE UTMOSv2 vs UTMOS** cho QMOS (A/B trên train, không tốn lượt nộp) | UTMOSv2 (T05, MIT) | A/B vs 0.414 | — | ĐÃ CODE 5/6 (Phiên 7), CHƯA chạy. UTMOSv2 = vô địch VMC2024 Track 1; xem có hơn UTMOS trên giọng cảm xúc |
| **exp10** | **FINE-TUNE AUDEERING riêng** (1 backbone) → ENSEMBLE cột VAD (trung bình) với exp08 — Hướng A cho T4 | audeering wav2vec2-large (fine-tune) | (mượn exp07) | mục tiêu VAD ≥ exp08 | ĐÃ CODE 5/6 (Phiên 7), CHƯA chạy. Tránh OOM (không nhồi 2 backbone 1 model); ensemble VAL/ARO/DOM |
| **exp08b** | **RESUME exp08** từ ckpt full + cache (train tiếp) | WavLM-large (fine-tune) + audeering frozen | 0.4167 | **0.8116** | **NỘP 6/6 (Phiên 9): MOS 0.4167 · EMOS 0.8116 · CAT err 0.1331 · VAL/ARO/DOM 0.6605/0.7904/0.7539** — gần như TRÙNG exp08 → xác nhận checkpoint đã **hội tụ**. `submissions/Track2/exp08b_finetune_resume/` |
| **exp11** | **FINE-TUNE ĐỒNG THỜI WavLM + audeering, FUSION 1 model** (cả 2 trainable, warm-start exp08) | WavLM-large + audeering (CẢ HAI fine-tune) | (mượn exp07) | warm-start 0.835 (val nội bộ) | CODE + CHẠY 8/6 (Phiên 9). **VAL nội bộ:** EMOS 0.8347 · VAL/ARO/DOM 0.803/0.874/0.808 (mean 0.8298). Warm-start đã ở đỉnh → train thêm **KHÔNG cải thiện** (early stop ep4). CHƯA nộp DEV. Tự resume từ `ft_joint_full.pt`. Fix: layerdrop=0, khóa numpy |
| **exp12** | **Ablation KHỞI TẠO WavLM** (cờ INIT_MODE: scratch/base/sailer), CHỈ WavLM | WavLM-large (scratch‑rand / base‑SSL / sailer‑emotion) | (mượn exp07) | so 3 mode (val nội bộ) | CODE 8/6 (Phiên 9) trả lời mentor "from-scratch vs fine-tune". `scratch`=random (mở băng toàn bộ, LR 1e-4); `base`/`sailer`=fine-tune 6 lớp. CHƯA đủ số 3 mode. Kỳ vọng sailer>base>scratch |
| **exp13** | **FINE-TUNE thẳng UTMOS** cho QMOS + nạp ckpt cảm xúc exp08 → answer 6 cột | UTMOS22_strong (TRAINABLE) + WavLM ft exp08 (chỉ inference) | mục tiêu >0.548 (exp07) | (mượn exp08 20ep) | CODE 8/6 (Phiên 10). Phá trần QMOS 0.548 bằng fine-tune trực tiếp model chất lượng trên nhãn `qMOS` thật (vì exp07 chỉ head đóng băng + neo UTMOS). Dùng ckpt **`ft_emotion_full_20epoch.pt`** (bản tốt nhất, KHÔNG dùng `ft_emotion_full.pt`). **HOÀN THIỆN 9/6 (Phiên 13): sửa lỗi ranking loss** (gom MSE+pred cả cửa sổ ACCUM rồi backward 1 lần → hết lỗi "backward graph a second time"; ranking giờ so ~16 câu/lần). `py_compile`+`.ipynb` OK. CHƯA chạy thật |
| **exp14** | **MAMBA temporal head** (cộng nhánh Mamba 2 chiều trên WavLM frame-level vào fusion exp07) | WavLM-large frame (đóng băng) + e2v/SAILER pooled | (mượn exp07) | so vs exp07 (val nội bộ) | CODE 8/6 (Phiên 10) theo gợi ý mentor "thử Mamba/SOTA mới". Cờ `USE_MAMBA` False=exp07 / True=+Mamba → **ablation Mamba cho paper**. Mamba thuần PyTorch (không cần `mamba-ssm`). CHƯA chạy |
| **exp15** | **WavLM FINE-TUNE + MAMBA head** cho 5 cột cảm xúc (thay mean-pool exp08 bằng Mamba encoder) | WavLM-large (SAILER warm-start, fine-tune) + audeering frozen | (mượn exp07) | so vs exp08 (val nội bộ) | CODE 8/6 (Phiên 10). Mamba head TRÊN WavLM fine-tune, predict cả 5 cột. Cờ `USE_MAMBA` False=exp08 / True=Mamba → **ablation Mamba vs mean-pool**. **CẬP NHẬT P10:** thêm **ranking loss** `RANK_LAMBDA=0.3` (4 cột SRCC) + **tự dò DATA_ROOT**. SMOKE-TEST 8/6: mamba-ssm build fail → Mamba thuần PyTorch (chậm). Chưa chạy thật/nộp |
| **exp16** | **AUDIO-LLM-AS-JUDGE** (API): đưa audio cho audio-LLM + prompt → chấm cả 6 cột | Gemini / GPT-4o-audio (API, KHÔNG train, KHÔNG GPU) | LLM zero/few-shot | LLM zero/few-shot | CODE 8/6 (Phiên 12). Mục tiêu **NOVELTY cho paper** (khảo sát audio-LLM-as-judge cho MOS cảm xúc), so vs exp07/exp08. Cờ `PROVIDER` (gemini/openai) + `SHOT_MODE` (zero/few). Cache+resume `.jsonl` (không trả tiền lại), parse JSON chịu lỗi, temp=0. Syntax OK; **CHƯA chạy** (cần API key + audio Kaggle) |
| **🏆 exp_mix** | **TRỘN CỘT:** ghép answer.txt — QMOS←exp07 + EMOS/CAT/VAD←exp08 | (không train; ghép cột 2 bản nộp) | **0.548** 🏆 | **0.811** 🏆 | **NỘP 9/6 (Phiên 14): QMOS 0.5480 · EMOS 0.8111 · CAT err 0.1331 · VAL 0.6590 · ARO 0.7933 · DOM 0.7509 — khớp đúng best-per-column.** 🏆 HỆ 6 CỘT MẠNH NHẤT + bản fallback an toàn cho Evaluation. `submissions/Track2/exp_mix_q07_emo08/` |

> SRCC = Spearman's Rank Correlation Coefficient (càng cao càng tốt). Metric chính của challenge: UTT-SRCC.

> ✅ **Bản TRỘN CỘT ĐÃ NỘP (9/6, Phiên 14):** QMOS←exp07 (0.548) + 5 cột cảm xúc←exp08 → `submissions/Track2/exp_mix_q07_emo08/` → điểm thật **khớp đúng best-per-column** (QMOS 0.5480 · EMOS 0.8111 · CAT 0.1331 · VAL 0.6590 · ARO 0.7933 · DOM 0.7509). Đây là **hệ 6 cột mạnh nhất**, dùng làm **fallback an toàn** cho phase Evaluation.

---

## Chi tiết từng thí nghiệm

### baseline — Reproduce baseline Track 2 (lần nộp đầu)
- **Ngày:** 3/6/2026
- **Mục tiêu:** Có mốc baseline trên leaderboard để so sánh (thỏa luật nộp ≥1 lần)
- **Config:**
  - QMOS: SpeechMOS (UTMOS22_strong) · CAT: emotion2vec+ large · EMOS: Gemini gemini-3-flash-preview
  - Data: tập DEV Track 2 (~2.730 mẫu), `vmc2026-track2-full` (15.477 wav đã ráp ESD+DailyTalk)
  - VAD: bỏ (tiết kiệm credit). EMOS chỉ chấm 496/2730 mẫu (dừng sớm) → còn lại mặc định 3.
- **Kết quả (DEV, CodaBench):**
  - QMOS UTT-SRCC: **0.414**
  - EMOS UTT-SRCC: **0.194** (một phần — sẽ tăng khi chấm đủ)
  - CAT UTT-ERR: **0.193** (thấp = tốt)
- **Nhận xét:** QMOS & CAT là điểm thật đầy đủ → mốc xuất phát hợp lý. EMOS bị kéo thấp do 82% mẫu mặc định; cần `--resume` chấm nốt rồi nộp lại. Hướng cải tiến: fine-tune SSL backbone cho QMOS, hoàn thiện EMOS.

---

### exp01 — Bỏ Gemini cho EMOS, dùng emotion2vec (ĐÃ CODE 3/6, chờ chạy)
- **Ngày:** code 3/6/2026 · chạy Kaggle [ ] · nộp [ ]
- **Giả thuyết:** EMOS hiện 0.194 vì Gemini zero-shot + 82% mẫu mặc định. Lấy **xác suất lớp cảm xúc target từ emotion2vec** (model đã dùng cho CAT) làm điểm EMOS → offline, đủ 2.730 mẫu → kỳ vọng EMOS tăng rõ.
- **Thay đổi so với baseline (ĐÃ làm trong code):** thêm `EMOS_METHOD="emotion2vec"` vào `kaggle_baseline/track2_baseline_pipeline.py`. Với mỗi wav: đọc cảm xúc target từ `metadata.csv` → lấy `P(target)` từ `emocat_probs` (emotion2vec đã tính cho CAT) → `EMOS = 1 + 4·P` (scale [0,1]→[1,5]). Dùng lại EmoCat nên không tốn thêm tính toán. Giữ nhánh Gemini làm tùy chọn.
- **Config:** model `iic/emotion2vec_plus_large` (funasr, offline trên Kaggle); không train. SRCC bất biến với scale tuyến tính nên chỉ cần thứ hạng P đúng.
- **Cách chạy:** set `EMOS_METHOD="emotion2vec"` (mặc định) → Run All → ra `answer.txt` đủ 2.730 mẫu (không cần GEMINI_API_KEY).
- **Kết quả (DEV, CodaBench 4/6):**
  - QMOS UTT-SRCC: **0.4139** (giữ nguyên — SpeechMOS)
  - **EMOS UTT-SRCC: 0.6365** ✅ (baseline 0.194 → tăng ×3,3; **VƯỢT cả SAILER 0.562**)
  - CAT UTT-ERR: **0.1933**
- **Nhận xét:** Bất ngờ — emotion2vec **vượt SAILER** ở EMOS dù bị "dồn 2 cực" (overconfident, P sát 0/1). Lý do: SRCC chấm **thứ hạng**, thứ hạng emotion2vec khớp người chấm tốt hơn; ties ở cực không hại đủ để thua. → EMOS tốt nhất hiện tại = emotion2vec. Nhưng emotion2vec KHÔNG có VAD (SAILER vẫn giữ phần đó).
- **Bước tiếp theo:** (1) nộp bản lai (EMOS←emotion2vec, VAD←SAILER); (2) fusion multi-task (emotion2vec+SAILER) — xem `03_literature_notes.md`.

### exp02 — EMOS có train: emotion2vec (đóng băng) + MLP head (ĐÃ CODE 3/6, chờ chạy)
- **Ngày:** code 3/6/2026 · chạy Kaggle [ ] · nộp [ ]
- **Phạm vi (chốt 3/6 phiên đêm):** train riêng **EMOS** trước (điểm yếu nhất = 0.194), chưa làm multi-task → dễ debug, nhanh có kết quả; sau đó mới mở rộng thêm head QMOS/CAT/VAD dùng chung backbone.
- **✅ Nhãn train (xác nhận 3/6):** `sets/train.csv` có đủ cột `lisID, wavID, qMOS, emoCat, eMOS, val, dom, aro`. Điểm **theo từng listener** → **gộp trung bình `eMOS` theo wav** = nhãn vàng. **Target cảm xúc** lấy từ `metadata.csv`.
- **Giả thuyết:** train có giám sát trên ~12.746 mẫu có nhãn eMOS sẽ vượt zero-shot (exp01 = 0.194). EMOS phụ thuộc cả audio LẪN target → feed thêm one-hot target.
- **Kiến trúc (ĐÃ code):** backbone **emotion2vec ĐÓNG BĂNG** (trích đặc trưng, không train lại) → feature `[embedding ~D | xác suất 5 lớp | one-hot target(5)]` → **MLP head** (Linear→ReLU→Dropout ×2 → 1).
- **Config:** model `iic/emotion2vec_plus_large` · HIDDEN 256 · DROPOUT 0.3 · LR 1e-3 · BATCH 64 · EPOCHS 60 · early-stop PATIENCE 12 · VAL 10% · Loss MSE · embedding cache .npz.
- **File:** `kaggle_baseline/exp02_train_emos_pipeline.py` + `exp02_train_emos.ipynb` (cú pháp OK; **chưa chạy thật**). Xuất `answer.txt` đầy đủ: QMOS=SpeechMOS, CAT=emotion2vec, EMOS=head.
- **Kết quả:** VAL SRCC nội bộ [ ] · EMOS UTT-SRCC (CodaBench) [ ] (mục tiêu > 0.194) · QMOS [ ] (giữ)
- **Nhận xét:** [ ]
- **Bước tiếp theo:** chạy Kaggle (thử `LIMIT_TRAIN=300` → `None`) → nộp → nếu khá thì lên multi-task đầy đủ.

---

### exp03 — EMOS+CAT+VAD bằng SAILER (offline, không train) ⭐ ĐÃ NỘP 4/6
- **Ngày:** code + chạy + nộp 4/6/2026
- **Giả thuyết:** EMOS baseline thấp (0.194) do Gemini zero-shot + chấm thiếu. Dùng **SAILER** (`tiantiaf/wavlm-large-categorical-emotion`, WavLM-large, vô địch Interspeech 2025 SER) lấy `P(target)` → EMOS sẽ vượt rõ.
- **Thay đổi:** thay emotion2vec/Gemini → **1 model SAILER lo cả EMOS + CAT + VAD**. EMOS = `1+4·P(target)` (softmax 9 lớp); CAT = 5 lớp renorm; VAD = arousal/valence/dominance SAILER xuất sẵn (sigmoid→1–5). QMOS giữ SpeechMOS.
- **Config:** model SAILER qua repo `vox-profile-release` (clone + sys.path, KHÔNG `pip install -e .`); audio 16kHz mono ≤15s; offline, không train; chạy GPU T4. File: `kaggle_baseline/track2/exp03_emos_sailer.ipynb`.
- **Kết quả (DEV, CodaBench 4/6):**
  - QMOS UTT-SRCC: **0.4139** (giữ nguyên — SpeechMOS không đổi)
  - **EMOS UTT-SRCC: 0.5618** 🚀 (baseline 0.194 → tăng gần ×3)
  - CAT UTT-ERR: **0.1903** (baseline 0.193 → tốt hơn chút, thấp = tốt)
  - VAL UTT-SRCC: **0.3410** · ARO UTT-SRCC: **0.7118** · DOM UTT-SRCC: **0.6302** (3 cột VAD trước đây bỏ trống)
- **Nhận xét:** Bước nhảy lớn nhất từ trước tới nay cho Track 2. EMOS tăng mạnh nhờ SAILER (đúng chuyên môn cảm xúc tự nhiên). ARO/DOM rất tốt; **VAL thấp nhất (0.341)** — đúng với literature (valence khó nhất khi chỉ dùng acoustic). License SAILER = **Open RAIL (phi thương mại)** → đã khai báo ở `12_`.
- **Bước tiếp theo:** (1) thử model VAD chuyên (audeering / tiantiaf MSP-dim) để đẩy VAL; (2) train head trên SAILER embedding (Cách B) xem có vượt 0.562; (3) cải thiện QMOS (vẫn 0.414).

---

### exp04 — FUSION multi-task (e2v + SAILER) ⭐ HƯỚNG CHÍNH (ĐÃ CODE 4/6, đang chạy)
- **Ngày:** code 4/6/2026 · chạy Kaggle [ ] · nộp [ ]
- **Giả thuyết:** emotion2vec thắng EMOS (0.637), SAILER thắng VAD (ARO 0.712/DOM 0.630) → 2 model **bổ sung nhau**; gộp đặc trưng + học chung sẽ mạnh hơn từng model lẻ.
- **Kiến trúc ("QMOS riêng + 5 cảm xúc chung"):** 2 backbone **đóng băng** (emotion2vec + SAILER) → nối đặc trưng `[e2v_emb | e2v_probs5 | sailer_emb | sailer_probs9 | sailer_vad3]` → **trunk chung** (Linear+ReLU ×2) → 3 head: **EMOS** (nối thêm one-hot target → 1), **CAT** (5 logits → softmax), **VAD** (3). QMOS để riêng (SpeechMOS).
- **Nhãn (gộp theo wavID):** EMOS = TB `eMOS` · VAD = TB `val/aro/dom` · CAT = tỉ lệ vote 5 lớp `emoCat` (đa nhãn). Nhãn liên tục z-score để các MSE cùng thang.
- **Loss:** EMOS/VAD = MSE · CAT = soft cross-entropy (nhãn mềm) · **cân loss = uncertainty weighting** (log σ² học được, cờ `USE_UNCERTAINTY` tắt được).
- **Config:** TRUNK_HIDDEN 512 · HEAD_HIDDEN 128 · DROPOUT 0.3 · LR 1e-3 · BATCH 64 · EPOCHS 80 · early-stop theo TB SRCC val nội bộ · cache `.npz` riêng từng backbone. **4 cờ ablation:** USE_E2V / USE_SAILER / USE_UNCERTAINTY / USE_CLASSPROB.
- **File:** `kaggle_baseline/track2/exp04_fusion_pipeline.py` + `exp04_fusion.ipynb` (cú pháp OK; **chưa chạy xong**).
- **Lỗi đã xử:** `train.csv` phân tách `|` (emoCat đa nhãn `,`) → `sep="|"`; ép GPU emotion2vec; tắt log funasr.
- **Kết quả (DEV, CodaBench 4/6):**
  - QMOS UTT-SRCC: **0.4139** (giữ — SpeechMOS)
  - **EMOS UTT-SRCC: 0.7878** 🚀 (exp01 emotion2vec 0.637 → +0.15; vượt rõ model lẻ tốt nhất)
  - **CAT UTT-ERR: 0.1454** (exp03 0.190 → tốt hơn, thấp = tốt)
  - **VAL UTT-SRCC: 0.5782** 🚀 (SAILER 0.341 → +0.24) · **ARO 0.7544** (SAILER 0.712) · **DOM 0.7061** (SAILER 0.630)
- **Nhận xét:** Fusion multi-task **thắng TẤT CẢ 5 cột cảm xúc** so với mọi model lẻ — chứng minh giả thuyết "emotion2vec + SAILER bổ sung nhau". Đáng chú ý: lo ngại "VAD nén chặt" khi xem answer.txt là **không đúng** — vì SRCC chấm **thứ hạng**, dù giá trị VAD nén quanh 2.5–3.6 nhưng thứ tự khớp người chấm tốt (VAL 0.341→0.578). → **exp04 là hệ thống tốt nhất hiện tại cho Track 2.** QMOS vẫn 0.414 (chưa đụng).
- **Bước tiếp:** (1) cải thiện **QMOS** (vẫn 0.414 — cột duy nhất chưa làm); (2) ablation (USE_E2V/USE_SAILER/USE_UNCERTAINTY/USE_CLASSPROB) điền bảng dưới cho paper; (3) thử exp05 (audeering) xem có đẩy VAL hơn 0.578 nữa không.

### exp05 — VAD bằng audeering MSP-dim (đẩy VAL) (ĐÃ CODE 4/6, chờ chạy)
- **Ngày:** code 4/6/2026 · chạy Kaggle [ ] · nộp [ ]
- **Giả thuyết:** VAL của SAILER thấp nhất (0.341 — đúng literature: valence khó với acoustic-only). Model **dimensional chuyên** `audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim` có thể đẩy VAL.
- **Thiết kế:** **tách file riêng** (giữ exp03 nguyên). SAILER lo EMOS+CAT, **audeering lo cả 3 VAD** (VAL/ARO/DOM), QMOS=SpeechMOS. 2 forward/wav.
- **Kỹ thuật:** audeering xuất `[arousal, dominance, valence]` ∈ [0,1] → đổi về [VAL,ARO,DOM] thang 1–5. Vì transformers mới lỗi khi subclass `Wav2Vec2PreTrainedModel` → **chỉ dùng `Wav2Vec2Model` + tự nạp tay trọng số regression head** từ checkpoint.
- **File:** `kaggle_baseline/track2/exp05_vad_audeering_pipeline.py` + `.ipynb` (cú pháp OK; **chưa chạy thật**).
- **License:** audeering = **CC BY-NC-SA 4.0** (phi thương mại) → khai báo `12_`.
- **Kết quả (DEV):** VAL [ ] · ARO [ ] · DOM [ ] (so exp03: 0.341 / 0.712 / 0.630)
- **Nhận xét:** [ ]
- **Bước tiếp:** chạy LIMIT=20 kiểm tra VAD ∈ [1,5] → full → nộp → A/B với exp03; nếu audeering chỉ thắng VAL thì trộn cột (VAL←audeering, ARO/DOM←SAILER).

---

### exp06 — TRAIN head QMOS (ĐÃ CODE 4/6 Phiên 5, chờ chạy)
- **Ngày:** code 4/6/2026 · chạy Kaggle [ ] · nộp [ ]
- **Giả thuyết:** QMOS kẹt 0.414 vì UTMOS zero-shot + lệch domain (train trên giọng không cảm xúc 2022), chưa dùng nhãn `qMOS` có sẵn. Train 1 head hồi quy nhỏ trên đặc trưng SSL (đã cache) + **điểm UTMOS làm đầu vào** (neo residual) → kỳ vọng vượt 0.414.
- **Kiến trúc:** đặc trưng `[e2v_emb | sailer_emb | xác suất | UTMOS]` → MLP (Linear→ReLU→Dropout ×2 →1). Backbone **đóng băng**, tái dùng cache `fusion_cache/`. Nhãn vàng = TB `qMOS` theo wav.
- **Config:** HIDDEN 256 · DROPOUT 0.3 · LR 1e-3 · BATCH 64 · EPOCHS 120 · early-stop theo SRCC val nội bộ · `RANK_LAMBDA` (tùy chọn pairwise ranking loss vì metric là thứ hạng) · cờ `USE_UTMOS_FEAT/USE_E2V/USE_SAILER/USE_CLASSPROB`.
- **File:** `kaggle_baseline/track2/exp06_qmos_train_pipeline.py` + `.ipynb`. Val nội bộ in **head SRCC vs UTMOS SRCC** (chỉ nộp khi head>UTMOS). Cuối: **ghép** QMOS mới vào answer.txt exp04 → `submission_track2_exp06_qmos.zip`.
- **Kết quả:** head SRCC nội bộ [ ] · QMOS UTT-SRCC CodaBench [ ] (mục tiêu >0.414)
- **Nhận xét:** [ ]

### exp07 — FUSION + QMOS head (hợp nhất 6 cột) ⭐ HỆ THỐNG TỐT NHẤT — NỘP 4/6
- **Ngày:** code + chạy + nộp 4/6/2026 (Phiên 5)
- **Giả thuyết (của user):** "chất giọng tự nhiên liên quan cảm nhận cảm xúc" → QMOS có thể hưởng lợi từ trunk cảm xúc chung. **Rủi ro:** e2v/SAILER chuyên cảm xúc, chưa chắc bắt lỗi chất lượng → QMOS có thể thua UTMOS, hoặc gộp làm tụt EMOS/VAD (negative transfer).
- **Kiến trúc:** mở rộng exp04 → `FusionMTL6` thêm **head QMOS thứ 4** (đầu vào `[trunk | UTMOS]`); 6 task (qmos/emos/cat/val/aro/dom) cân bằng bằng uncertainty weighting. QMOS giờ từ head, KHÔNG còn SpeechMOS riêng.
- **Config:** giống exp04 (TRUNK 512 · HEAD 128 · LR 1e-3 · EPOCHS 80) + cờ `USE_UTMOS_FEAT`. So mốc exp04 (EMOS 0.788/CAT 0.145/VAL 0.578/ARO 0.754/DOM 0.706).
- **File:** `kaggle_baseline/track2/exp07_fusion_qmos_pipeline.py` + `.ipynb`. Mục 5 in SRCC cả 6 cột + **cảnh báo negative transfer** nếu cảm xúc tụt >0.02 so exp04.
- **Quyết định nộp:** QMOS↑ & cảm xúc không tụt → nộp exp07 (1 model 6 cột, đẹp paper); QMOS↑ nhưng cảm xúc tụt → giữ exp04 + ghép cột QMOS; QMOS không vượt UTMOS → kết luận "chất lượng trực giao cảm xúc" (vẫn là phát hiện cho paper).
- **Ablation đo giả thuyết:** `USE_UTMOS_FEAT=False` → QMOS chỉ từ trunk cảm xúc.
- **Kết quả (DEV, CodaBench 4/6) — `submissions/Track2/exp07_fusion_qmos/scoring_result.zip`:**
  - **QMOS UTT-SRCC: 0.5480** 🚀 (exp04 0.414 → **+0.134**, lần cải thiện QMOS ĐẦU TIÊN kể từ baseline)
  - EMOS UTT-SRCC: **0.7950** (exp04 0.788 → nhích lên, KHÔNG tụt)
  - CAT UTT-ERR: **0.1531** (exp04 0.145 → tệ hơn chút, +0.008)
  - VAL **0.5815** / ARO **0.7515** / DOM **0.7048** (exp04 0.578/0.754/0.706 → gần như giữ nguyên)
- **Nhận xét:** Thành công đúng mục tiêu. (1) QMOS nhảy lớn nhờ trunk cảm xúc + UTMOS-feature; (2) **KHÔNG negative transfer** — gộp QMOS không kéo tụt 5 cột cảm xúc (EMOS còn nhích) → rơi đúng kịch bản đẹp "nộp bản hợp nhất 6 cột". Điểm trừ duy nhất: CAT tệ hơn 0.008 (có thể uncertainty weighting chia lại trọng số khi thêm task thứ 6). → **exp07 thay exp04 làm hệ thống chính cho paper** (1 model trọn 6 cột).
- **Câu hỏi nghiên cứu còn mở (cần exp06 + ablation để trả lời):** QMOS lên là nhờ **chia sẻ biểu diễn cảm xúc** hay chỉ nhờ **UTMOS-feature**? → chạy exp06 (head riêng) A/B với 0.548; và `USE_UTMOS_FEAT=False` để tách phần đóng góp.

### exp08 — FINE-TUNE WavLM cho 5 cột cảm xúc ⭐ FINE-TUNE ĐẦU TIÊN (ĐANG CHẠY 5/6 Phiên 6)
- **Ngày:** code + chạy 5/6/2026 · nộp [ ]
- **Khác mọi exp trước:** exp03–07 đều **đóng băng** backbone (freeze + head). exp08 **MỞ BĂNG (fine-tune)** WavLM-large để học lại đặc trưng riêng cho MOS cảm xúc → theo gợi ý mentor.
- **Giả thuyết:** fine-tune phá trần "freeze + head"; warm-start từ SAILER (đã giỏi cảm xúc) → vượt frozen-fusion exp07 ở cảm xúc, đặc biệt VAD.
- **Kiến trúc:** WavLM-large (lôi backbone HF bên trong wrapper SAILER, fallback WavLM trắng) — **mở băng 6 lớp Transformer trên**, đóng băng feature-extractor + lớp dưới → mean-pool → emb 1024. + **audeering MSP-dim FROZEN** (cache `aud_*.npz`, `[emb|vad3]`) → concat → **trunk** → 3 head: EMOS(+target)/CAT(softmax)/VAD. Uncertainty weighting 5 task. QMOS mượn exp07/UTMOS (không train ở đây).
- **Config:** `UNFREEZE_TOP_LAYERS=6` · TRUNK 512 · HEAD 128 · LR backbone 1e-5 / head 1e-3 · `EPOCHS=12` (trần) · `BATCH 4 × ACCUM 8` (hiệu dụng 32) · `MAX_SECONDS 8` · AMP fp16 · gradient checkpointing · early-stop PATIENCE 3 theo TB SRCC val.
- **File:** `kaggle_baseline/track2/exp08_finetune_emotion_pipeline.py` + `.ipynb`. Cache audeering ở `ft_cache/` (WavLM KHÔNG cache vì đang train).
- **Kết quả (DEV val nội bộ, full, mới epoch 2/12):**
  - EMOS **0.752** (exp07 0.795 → tụt nhẹ, vì bỏ emotion2vec — vô địch EMOS lẻ)
  - **VAL 0.747** 🚀 (exp07 0.581) · **ARO 0.857** 🚀 (0.752) · **DOM 0.783** 🚀 (0.705)
  - TB 4 cột cảm xúc: **0.785** vs exp07 0.708 → **+0.077** ngay epoch 2
  - `cat_err 0.582` = metric **nội bộ tạm** (L1 phân phối), KHÔNG so được CAT-ERR CodaBench
- **Nhận xét (sơ bộ):** fine-tune ăn đậm ở **VAD** (đúng kỳ vọng: mở băng WavLM + audeering chuyên dimensional). EMOS tụt nhẹ là đánh đổi vì bỏ e2v. Chưa chạy xong → **chưa nộp**.
- **Quyết định nộp (chờ khối VAL cuối):** EMOS leo ≥0.795 → nộp nguyên exp08 (thắng cả 5 cột); EMOS kẹt <0.795 → **trộn cột** (EMOS←exp07/e2v · VAL/ARO/DOM←exp08).
- **Ablation cho paper:** `UNFREEZE_TOP_LAYERS=0` (head-only) vs `=6` (fine-tune) → bảng "frozen vs fine-tuned"; `USE_AUDEERING=False`.
- **License (khai báo `12_`):** WavLM MIT · SAILER Open RAIL · audeering CC BY-NC-SA (đều phi thương mại trừ WavLM).

---

### exp08b — RESUME exp08 (train tiếp từ ckpt full) ⭐ ĐÃ NỘP 6/6
- **Ngày:** code 5/6/2026 (Phiên 7) · chạy + nộp 6/6/2026 · đọc điểm 8/6 (Phiên 9)
- **Giả thuyết:** exp08 dừng early-stop ở patience 3 — có thể chưa hội tụ hết, resume LR nhỏ + train tiếp xem có nhích lên không.
- **Kiến trúc:** giống exp08 (WavLM-large mở băng 6 lớp + audeering frozen → trunk → 3 head EMOS/CAT/VAD), warm-start từ `ft_emotion_full.pt` (đủ backbone + heads). Tái dùng cache audeering (`aud_*.npz`).
- **Config:** giống exp08 — LR backbone 1e-5 / head 1e-3 · BATCH 4 × ACCUM 8 · MAX_SECONDS 8 · AMP + grad-ckpt · EPOCHS 12 (trần) · PATIENCE 3. Gotcha: `weights_only=False` cho torch 2.6; slug Kaggle đổi `_`→`-`.
- **File:** `kaggle_baseline/track2/exp08b_finetune_resume_pipeline.py` + `.ipynb`. Submission: `submissions/Track2/exp08b_finetune_resume/`.
- **Kết quả (DEV, CodaBench 6/6):**
  - QMOS UTT-SRCC: **0.4167** (fallback UTMOS — không mượn exp07)
  - EMOS UTT-SRCC: **0.8116** (exp08: 0.811)
  - CAT UTT-ERR: **0.1331** (exp08: 0.133)
  - VAL **0.6605** · ARO **0.7904** · DOM **0.7539** (exp08: 0.659/0.793/0.751)
- **Nhận xét:** Điểm **gần như TRÙNG exp08** (chênh vài phần nghìn ở mọi cột) → **xác nhận checkpoint exp08 đã hội tụ**; train tiếp trên cùng data không thay đổi. Bài học: muốn cải thiện thật thì phải đổi data/loss/kiến trúc, không phải train thêm.
- **Bước tiếp:** ngừng resume cùng data; thử ranking loss hoặc ensemble nhiều seed.

---

### exp09a — PROBE UTMOSv2 vs UTMOS cho QMOS (ĐÃ CODE 5/6 Phiên 7, CHƯA chạy)
- **Ngày:** code 5/6/2026 · chạy [ ] · không tốn lượt nộp
- **Giả thuyết:** UTMOS (2022) lệch domain giọng cảm xúc → kẹt 0.414. **UTMOSv2** (T05 sarulab-speech, **MIT**, vô địch VMC2024 Track 1) là kế nhiệm trực tiếp — có thể tốt hơn ngay cả zero-shot.
- **Thiết kế (probe A/B):** chấm tập train Track 2 (nhãn `qMOS` thật) bằng UTMOS vs UTMOSv2 → so SRCC nội bộ → biết model nào khớp giọng cảm xúc hơn **mà không tốn lượt nộp CodaBench**.
- **File:** `kaggle_baseline/track2/exp09a_qmos_utmosv2_probe.ipynb`. Đồng thời đổi fallback QMOS trong exp08 từ UTMOS sang UTMOSv2.
- **Kết quả:** UTMOS SRCC train [ ] · UTMOSv2 SRCC train [ ]
- **Nhận xét:** [ ]
- **Bước tiếp:** chạy probe → nếu UTMOSv2 thắng → thay UTMOS bằng UTMOSv2 cho fallback và làm neo trong exp06/07.

---

### exp10 — FINE-TUNE AUDEERING riêng + ENSEMBLE VAD (ĐÃ CODE 5/6 Phiên 7, CHƯA chạy)
- **Ngày:** code 5/6/2026 · chạy [ ] · nộp [ ]
- **Giả thuyết (Hướng A cho T4):** nhồi 2 backbone fine-tune vào 1 model → OOM trên T4. Thay vào, **fine-tune audeering RIÊNG** (1 backbone, nhẹ) → **ensemble VAD** (trung bình cột) với exp08 → giảm gap mà không OOM.
- **Kiến trúc:** audeering wav2vec2-large (mở băng) → head VAD 3 chiều, train trên nhãn `val/aro/dom` gộp theo wav. Sau train: ghép answer.txt = QMOS (exp07/UTMOS) + EMOS/CAT (exp08) + VAD = **trung bình** (audeering ft, exp08).
- **Config:** lưu `ft_audeering_full.pt` mỗi best. AMP + grad-ckpt. Tái dùng infra giống exp08.
- **File:** `kaggle_baseline/track2/exp10_finetune_audeering_pipeline.py` + `.ipynb`. Có mục 7 ensemble.
- **Kết quả:** [ ]
- **Nhận xét:** [ ]
- **Bước tiếp:** chạy → đọc VAD riêng audeering có ≥ exp08 không → nếu ≥ thì ensemble → so DEV với exp08 thuần.

---

### exp11 — FINE-TUNE ĐỒNG THỜI WavLM + audeering, FUSION 1 model ⭐ ĐÃ CHẠY 8/6 (chưa nộp DEV)
- **Ngày:** code + chạy 8/6/2026 (Phiên 9) · nộp DEV [ ]
- **Giả thuyết:** exp08 đóng băng audeering (chỉ ft WavLM) — có thể bỏ lỡ tín hiệu nếu mở băng cả 2. Fusion **trong 1 model** + warm-start mạnh (exp08 đỉnh) → kỳ vọng vượt exp08.
- **Kiến trúc:** WavLM-large + audeering (CẢ HAI fine-tune, mở băng 4 lớp/backbone) → nối đặc trưng → trunk → 3 head cảm xúc. Warm-start: WavLM + heads từ `ft_emotion_full_20epoch.pt`; audeering+head random ban đầu, có cờ **RESUME đủ** từ `ft_joint_full.pt`. Chống OOM: BATCH=1 + ACCUM, grad-ckpt cả 2 backbone, AMP, MAX_SECONDS=6.
- **Config:** UNFREEZE 4 lớp/backbone · LR backbone 1e-5 · `RESUME_LR_SCALE` (scale LR nhỏ khi resume) · EPOCHS 12 · PATIENCE 3.
- **File:** `kaggle_baseline/track2/exp11_finetune_joint_pipeline.py` + `.ipynb`. Checkpoint `ft_joint_full.pt` ở thư mục gốc dự án.
- **Kết quả (VAL nội bộ 8/6, KHÔNG phải DEV CodaBench):**
  - EMOS **0.8347** · VAL **0.803** · ARO **0.874** · DOM **0.808** · mean SRCC **0.8298**
  - Warm-start đã ở đỉnh → 4 epoch sau đều thấp hơn → **early stop ep4, KHÔNG cải thiện**.
- **Nhận xét:** Bài học (1) warm-start mạnh + resume LR nhỏ → khó vượt. (2) **VAL nội bộ ≠ DEV** — VAL nội bộ exp11 0.80/0.87/0.80 nhưng DEV exp08 chỉ 0.66/0.79/0.75 → **chênh ~0.10–0.15** = nghi overfit / lệch phân phối. **Chưa nộp DEV** → không biết điểm thật. Lỗi gặp: layerdrop CheckpointError + SystemError numpy ABI — đã fix (xem mục Lỗi & bài học).
- **Bước tiếp:** cân nhắc nộp DEV 1 lần để biết thật (đừng tin VAL nội bộ). Nếu vượt exp08 DEV → exp11 thành hệ 1-model cảm xúc tốt nhất.

---

### exp12 — Ablation KHỞI TẠO WavLM (scratch / base / sailer) (ĐÃ CODE 8/6 Phiên 9, chưa đủ 3 mode)
- **Ngày:** code 8/6/2026 · chạy 1 phần · nộp [ ]
- **Bối cảnh:** mentor gợi ý "12k mẫu có thể train scratch tốt hơn". Code 1 notebook để **đo bằng số** — cô lập biến *khởi tạo backbone*, mọi thứ khác giữ nguyên.
- **Thiết kế (CHỈ WavLM, bỏ audeering để cô lập):** cờ **`INIT_MODE`** 3 giá trị:
  - `scratch` = random init (mở băng TOÀN BỘ, LR 1e-4)
  - `base` = WavLM-large pretrain SSL (fine-tune 6 lớp trên, LR 1e-5)
  - `sailer` = warm-start cảm xúc (fine-tune 6 lớp trên, LR 1e-5)
- **Quan điểm đã trao đổi:** 12k mẫu quá ít so với 94k giờ SSL pretrain → kỳ vọng `sailer > base >> scratch`. exp12 chỉ để **xác nhận bằng số** cho mentor.
- **File:** `kaggle_baseline/track2/exp12_wavlm_scratch_pipeline.py` + `.ipynb`. Mỗi lần chạy đổi `INIT_MODE`.
- **Kết quả:** scratch [ ] · base [ ] · sailer [ ] — chưa chạy đủ 3 mode.
- **Nhận xét:** [ ] — chờ đủ số.
- **Bước tiếp:** chạy đủ 3 mode → bảng ablation "khởi tạo backbone" cho paper + trả lời mentor.

---

### exp13 — FINE-TUNE thẳng UTMOS cho QMOS + ghép cảm xúc exp08 (ĐÃ CODE 8/6 Phiên 10, CHƯA chạy)
- **Ngày:** code 8/6/2026 · chạy [ ] · nộp [ ]
- **Bối cảnh:** QMOS tốt nhất hiện tại = exp07 0.548 (head ĐÓNG BĂNG + neo UTMOS). Muốn phá trần → **fine-tune thẳng UTMOS** trên nhãn `qMOS` thật của Track 2 để kéo model chất lượng về đúng domain giọng cảm xúc. Mượn 5 cột cảm xúc từ checkpoint exp08.
- **Vì sao UTMOS chứ không UTMOSv2:** UTMOS (`utmos22_strong`, tarepan/SpeechMOS) = **1 model đơn**, tải qua `torch.hub`, `nn.Module` chuẩn → backprop được toàn model. UTMOSv2 = ensemble nhiều fold + 2 luồng → khó train.
- **Kiến trúc:** PHẦN A wav→UTMOS (TRAINABLE, warm-start) → QMOS, train trên `qMOS` gold. PHẦN B wav→WavLM exp08 ft + audeering frozen→ EMOS/CAT/VAD, chỉ inference từ ckpt **`ft_emotion_full_20epoch.pt`** (bản tốt nhất, KHÔNG dùng `ft_emotion_full.pt`). PHẦN C ghép QMOS(A) + 5 cột cảm xúc(B) → answer.txt 6 cột. **Không neo UTMOS riêng** — khi fine-tune chính UTMOS thì "neo" nằm sẵn trong trọng số warm-start.
- **Config:** LR 1e-5 (warm-start sẵn tốt) · BATCH 1 × ACCUM 16 (UTMOS không có attention-mask → BATCH=1 an toàn) · MAX_SECONDS 12 (cắt audio chặn OOM backprop) · `FREEZE_FEAT_EXT=True` (đóng băng CNN, đỡ VRAM + chống overfit) · `RANK_LAMBDA=0` (0=MSE thuần; >0=cộng pairwise ranking loss tối ưu thẳng SRCC) · EPOCHS 10 · PATIENCE 3.
- **File:** `kaggle_baseline/track2/exp13_finetune_qmos_pipeline.py` + `.ipynb`. Lưu `ft_qmos_utmos.pt` mỗi best + Save Version NGAY (bài học exp08).
- **Lưới an toàn:** chỉ nộp QMOS fine-tune nếu **SRCC val nội bộ > zero-shot UTMOS** (mục A in cả 2 số).
- **Kết quả:** UTMOS zero-shot 0.414 · UTMOS ft (val nội bộ) [ ] · QMOS UTT-SRCC DEV CodaBench [ ] (mục tiêu >0.548)
- **Nhận xét:** [ ]
- **Bước tiếp:** smoke test LIMIT_TRAIN=300, LIMIT_DEV=20 → full → so 0.548 → nộp nếu vượt.

---

### exp14 — MAMBA temporal head (CỘNG vào fusion exp07) (ĐÃ CODE 8/6 Phiên 10, CHƯA chạy)
- **Ngày:** code 8/6/2026 · chạy Kaggle [ ] · nộp [ ]
- **Bối cảnh:** mentor gợi ý đọc SOTA mới (LLM-based / Mamba) để paper xịn + xem apply được không. Chọn **Mamba** trước vì khả thi trên T4 (LLM-based quá nặng để train). Tham khảo MambaRate (AudioMOS 2025, arXiv:2507.12090).
- **Vấn đề cốt lõi:** exp04/exp07 **mean-pool** đặc trưng SSL → 1 vector/wav → mất động lực thời gian. **Mamba (SSM)** cần **chuỗi frame** → phải giữ đặc trưng frame-level (chưa pool).
- **Thiết kế (CỘNG thêm, không thay thế):** giữ nguyên đặc trưng pooled `[e2v|sailer|UTMOS]` của exp07 (dùng lại cache `fusion_cache/`) + **nhánh mới:** WavLM-large **frame-level** (T×1024, đóng băng) → **Mamba 2 lớp, 2 chiều** → attentive-pool → `z_seq(128)` → concat vào trunk → 6 head (QMOS/EMOS/CAT/VAL/ARO/DOM).
- **Cờ ablation chính `USE_MAMBA`:** `False` → ra **đúng exp07** (kiểm chứng); `True` → bật Mamba. So 2 lần = **ablation Mamba cho paper** ("temporal Mamba có hơn mean-pooling không?").
- **Config:** WavLM `microsoft/wavlm-large` đóng băng · MAX_FRAMES 256 (~5s, cache fp16 từng `.npy`) · MAMBA_DMODEL 256 · MAMBA_LAYERS 2 · BIDIRECTIONAL · Z_DIM 128 · TRUNK 512 · HEAD 128 · LR 1e-3 · BATCH 32 · EPOCHS 80 · uncertainty weighting 6 task.
- **Gotcha đã xử (trong file):** (1) `mamba-ssm` hay lỗi build CUDA trên Kaggle → **nhúng Mamba thuần PyTorch** (selective scan vòng lặp thời gian, theo mamba-minimal); tự dùng `mamba-ssm` nếu import được. (2) Cache frame-level nặng → cap MAX_FRAMES + lưu fp16 (~0.5MB/wav). KHÔNG đụng numpy (tránh lệch ABI — bài học exp12).
- **File:** `kaggle_baseline/track2/exp14_mamba_head_pipeline.py`.
- **Kết quả:** USE_MAMBA=False (=exp07) [ ] · USE_MAMBA=True [ ] (so QMOS 0.548 / EMOS 0.795 / VAD 0.581·0.752·0.705)
- **Nhận xét:** [ ]
- **Bước tiếp:** chạy LIMIT_TRAIN=300/LIMIT_DEV=20 kiểm pipeline → full 2 lần (Mamba off/on) → điền ablation → nếu Mamba thắng thì nộp DEV.

---

### exp15 — WavLM FINE-TUNE + MAMBA head cho 5 cột cảm xúc (ĐÃ CODE 8/6 Phiên 10, CHƯA chạy)
- **Ngày:** code 8/6/2026 · chạy Kaggle [ ] · nộp [ ]
- **Bối cảnh:** mentor gợi ý áp dụng SOTA mới. User chốt: **Mamba head TRÊN WavLM (fine-tune)** để predict emotion, cả 5 cột, SAILER warm-start. Đây là exp08 đổi đúng 1 chỗ kiến trúc.
- **Giả thuyết:** exp08 fine-tune WavLM nhưng vẫn **mean-pool** → mất động lực thời gian (lên/xuống giọng, ngắt quãng, run giọng). Thay mean-pool bằng **Mamba encoder** (SSM học chuỗi, độ phức tạp tuyến tính) → nắm temporal dynamics → kỳ vọng vượt exp08 ở EMOS/VAD.
- **Kiến trúc:** WavLM-large (SAILER warm-start, mở băng 6 lớp trên, TRAINABLE) → hidden states (B,T,1024) **KHÔNG pool** → MambaEncoder (proj 1024→256, Mamba 2 lớp 2 chiều, attentive-pool có mask) → z(256) + audeering frozen [emb|vad3] → trunk → 3 head EMOS(+target)/CAT/VAD. QMOS mượn exp07/UTMOSv2.
- **Cờ ablation chính `USE_MAMBA`:** False → quay về `masked_mean` = **đúng exp08**; True → Mamba head. So 2 lần = ablation "Mamba vs mean-pool" (cùng backbone fine-tune) cho paper.
- **Config:** UNFREEZE 6 · MAX_SECONDS 6 (giảm từ 8 vì Mamba backprop-through-time nặng) · BATCH 2 × ACCUM 16 (eff 32) · MAMBA_DMODEL 256 · LAYERS 2 · DSTATE 16 · BIDIRECTIONAL · LR backbone 1e-5 / head 1e-3 · EPOCHS 12 · PATIENCE 3 · AMP + grad-ckpt + uncertainty weighting 5 task.
- **Gotcha đã xử (trong file):** (1) `layerdrop=0` tránh CheckpointError khi grad-ckpt (bài học exp12). (2) `mamba-ssm` khó cài → nhúng Mamba thuần PyTorch (chạy fp32 cho ổn định), tự dùng `mamba-ssm` nếu import được. (3) Checkpoint `ft_mamba_emotion_full.pt` lưu CẢ backbone + Mamba + heads mỗi best (bài học exp08). (4) Không đụng numpy.
- **File:** `kaggle_baseline/track2/exp15_wavlm_mamba_emotion_pipeline.py`.
- **Kết quả:** USE_MAMBA=True [ ] · USE_MAMBA=False (=exp08) [ ] (mốc exp08: EMOS 0.811 / VAL 0.659 / ARO 0.793 / DOM 0.751)
- **Nhận xét:** [ ]
- **Bước tiếp:** smoke test LIMIT 300/20 (1 epoch không OOM/không CheckpointError) → full True → so exp08 → nếu thắng ráp answer.txt nộp DEV. ⚠️ Mamba thuần PyTorch khi fine-tune RẤT chậm trên full — nếu quá chậm thử cài `mamba-ssm causal-conv1d`.

---

### exp16 — Audio-LLM-as-Judge qua API (ĐÃ CODE 8/6 Phiên 12, CHƯA chạy)
- **Ngày:** code 8/6/2026 · chạy [ ] · nộp [ ]
- **Bối cảnh:** user muốn "thêm SOTA model như GPT/LLM". Phiên 10 từng gác hướng LLM vì train quá nặng cho T4 → chọn cách **gọi API audio-LLM-as-judge** (chỉ inference qua mạng, KHÔNG GPU). Mục tiêu = **NOVELTY cho paper** (không bắt buộc phá trần leaderboard).
- **Câu chuyện paper:** "khảo sát có hệ thống audio-LLM-as-judge cho dự đoán MOS cảm xúc" — so audio-LLM zero/few-shot với hệ SSL đã train (exp07/exp08). Giả thuyết: LLM khá ở EMOS/CAT (hiểu ngữ nghĩa cảm xúc), yếu ở QMOS (artifact tinh vi acoustic-only).
- **Thiết kế:** mỗi wav DEV → đọc audio 16k mono cắt 12s → gửi audio + prompt có cấu trúc (định nghĩa rõ 6 metric + đưa cảm xúc target cho EMOS) → LLM trả **JSON nghiêm ngặt** 6 cột → parse → ráp `answer.txt` đúng format exp07 (`wav,QMOS,EMOS,CAT,VAL,ARO,DOM`).
- **Cờ:** `PROVIDER` gemini/openai (→ bảng so 2 audio-LLM) · `SHOT_MODE` zero/few-shot (few-shot nhét K audio ví dụ có nhãn từ train.csv → ablation Bảng B) · `LIMIT`/`MAX_SECONDS`/`WORKERS`/`MAX_RETRY` · `TEMPERATURE=0` (tái lập).
- **Robust + rẻ:** **cache+resume** `.jsonl` mỗi stem (chạy lại KHÔNG gọi lại API = không trả tiền 2 lần); parse JSON chịu lỗi (regex trích `{...}` + clamp [1,5] + chuẩn hóa CAT 5 lớp + retry); thất bại → điền mặc định + log riêng.
- **Tái dùng:** `load_target_emotions()`/`norm_emotion()`/format `answer.txt` từ baseline+exp07. Kèm hàm `ensemble_rank_average` (tùy chọn) trộn THỨ HẠNG điểm LLM + hệ trained (vd bản trộn cột exp07+exp08).
- **File:** `kaggle_baseline/track2/exp16_llm_judge_pipeline.py` + `.ipynb`. Syntax OK (py_compile) + `.ipynb` JSON hợp lệ.
- **⚠️ Trước khi chạy:** xác nhận model ID nhận audio (`GEMINI_MODEL` mặc định `gemini-2.5-flash`, `OPENAI_MODEL` `gpt-4o-audio-preview` — có thể đã đổi; baseline từng dùng họ `gemini-3-flash-preview`). Secrets `GEMINI_API_KEY`(+`OPENAI_API_KEY`), Internet On, GPU không cần.
- **Kết quả:** Gemini zero-shot DEV [ ] · OpenAI zero-shot [ ] · few-shot [ ] (so exp07 QMOS 0.548 / exp08 EMOS 0.811·CAT 0.133·VAD 0.659·0.793·0.751)
- **Nhận xét:** [ ]
- **Bước tiếp:** smoke test `LIMIT=20` gemini zero-shot → full 2730 → nộp → điền Bảng A/B.

---

> Copy block "exp" ở trên cho mỗi thí nghiệm mới.

---

## Ablation study (cho paper)
> Tổng hợp các thí nghiệm chứng minh từng thành phần đóng góp ra sao

| Cấu hình | QMOS SRCC | EMOS SRCC | Δ |
|---|---|---|---|
| Full model | | | — |
| − component A | | | |
| − component B | | | |

---

## Lỗi & bài học
| Ngày | Lỗi gặp phải | Cách khắc phục |
|---|---|---|
| 8/6 | `CheckpointError: A different number of tensors saved during forward and recomputation` (exp12 khi train) | **layerdrop** (WavLM bỏ lớp ngẫu nhiên) đụng gradient-checkpointing (chạy lại forward khác lần đầu). Fix: `wavlm.config.layerdrop = 0.0` (và `aud.config.layerdrop=0.0`) ngay sau khi dựng backbone |
| 8/6 | `SystemError: bad call flags` khi `import torch` (exp12) | `pip install` đổi version **numpy** → lệch ABI torch. Fix: **Restart Session** + khóa numpy: `pip install ... numpy==<bản gốc>` mỗi lần cài. Chỉ cài gói còn thiếu (Kaggle có sẵn torch/transformers/...) |
| 8/6 | pip fail `Temporary failure in name resolution` (exp15 cell cài) | **Internet đang TẮT** trên Kaggle. Fix: Settings → **Internet = On** (cần verify phone) → Restart → Run All. exp08/13/15 đều cần Internet On (tải SAILER/audeering/WavLM/UTMOS/mamba). |
| 8/6 | `causal-conv1d`/`mamba-ssm` build fail (exp15) | Build CUDA hay fail trên Kaggle. Fix thử: `--no-build-isolation` cho cả causal-conv1d + cài `ninja`. Nếu vẫn fail → code tự fallback **Mamba thuần PyTorch** (chạy được, chậm). Full quá chậm → hạ MAX_SECONDS/LAYERS hoặc đổi head BiGRU/Transformer. |
