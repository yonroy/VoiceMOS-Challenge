# 06 — Clone & Setup Baseline Repos

Ghi lại các repo baseline đã clone và cách setup. Cập nhật khi thêm/đổi repo.

> Ngày clone: 1/6/2026 — đã kéo về máy local trong thư mục dự án.

---

## Tổng quan các repo đã clone

```
d:\VFS\VoiceMOS Challenge 2026\
└── baselines/               # (sắp xếp lại 3/6/2026 — gom 3 repo vào đây)
    ├── vmc2026-baselines/   # Baseline chính thức của BTC (track1/2/3)
    ├── UTMOS22/             # QMOS baseline (Track 2)
    └── emotion2vec/         # EmoCat baseline (Track 2)
```

> ⚠️ **Không commit thư mục `baselines/`** vào repo tài liệu của mình (chúng là repo git riêng + nặng). Đã thêm `baselines/` vào `.gitignore`.

---

## 1. Repo baseline chính thức

```bash
git clone https://github.com/voicemos-challenge/vmc2026-baselines.git
```

Cấu trúc: 3 thư mục `track1/`, `track2/`, `track3/`. Mỗi track có README riêng.

### Baseline theo từng track
| Track | Nhiệm vụ | Baseline | Repo / Vị trí code | Chạy baseline được chưa? |
|---|---|---|---|---|
| Track 1 | Speech Enhancement | URGENT-MOS | https://github.com/vvwangvv/URGENT-MOS (paper: arxiv.org/abs/2601.18438) | ✅ **NGAY** (data dev công khai HF, có checkpoint + script nộp sẵn) |
| **Track 2** ⭐ | Emotional TTS | xem bảng dưới | Trong `baselines/vmc2026-baselines/track2/` | ✅ **data đã đủ** (3/6, gói BTC + ESD + DailyTalk trên Kaggle) |
| Track 3 | Codec-based synthesis | ECAPA-TDNN | `baselines/vmc2026-baselines/track3/` (có code + **checkpoint pre-trained**) | ✅ **data đã đủ** (3/6, gói VCTK là mảnh cuối) |

> **Pipeline gộp sẵn cho cả 3 track** trong `kaggle_baseline/`:
> `track1_baseline_pipeline.py` · `track2_baseline_pipeline.py` · `track3_baseline_pipeline.py`.

### Track 1 — chi tiết (không bị chặn)
- Repo `vvwangvv/URGENT-MOS`: `pip install -e .`; checkpoint `urgent-challenge/urgent-mos-f1c1m5dcorpus` **tự tải từ HF**.
- Script nộp sẵn: `scripts/infer_vmc2026_track1.py --split dev --output predictions_dev.csv` → đúng cột `sample_id,pred_score` (ACR 1008 + CCR 2520).
- Data dev công khai: HF `urgent-challenge/vmc2026-track1-dev` (configs `acr` + `ccr`, FLAC). GPU khuyến nghị.

### Track 3 — chi tiết (data đã đủ 3/6)
- Code đầy đủ trong repo: `inference.py`, `finetune.py`, `model.py` + **checkpoint pre-trained** `official-egs/{spk,acc}_sim_adamw_lr1e-3/model_*_step20000.pt`.
- Baseline 1 zero-shot ECAPA (không train) · Baseline 2 fine-tuned (tốt hơn: spk/acc SRCC ~0.44–0.45 dev).
- Inference: `python inference.py --data-root <DATA_ROOT> --csv-path <DATA_ROOT>/sets/dev.csv --checkpoint <ckpt> --out spk_dev.csv` (chạy riêng spk & acc rồi gộp).
- **Data (đã có local 3/6):** giải nén `..._syn.tar.gz` (3.252 wav + `sets/*.csv`) và `..._vctk.tar.gz` (296 wav: sys008+sys019) → copy `_vctk/wav/*.wav` sang `_syn/wav/` → **3.548 wav** (sys019 = giọng tham chiếu `wav_b`). Train 2.800 cặp / 13.687 rating / 25 listener · dev 600 cặp.

### Baseline Track 2 (4 sub-task)
| Sub-task | Baseline | Có code sẵn trong repo? | Cần thêm |
|---|---|---|---|
| **QMOS** (chất lượng 1–5) | UTMOS | ❌ chỉ README | Clone `UTMOS22` |
| **EmoCat** (tỉ lệ vote 5 cảm xúc) | Emotion2vec+ large | ⚠️ có `run_vmc2026.py` | Clone `emotion2vec` |
| **EMOS** (độ khớp cảm xúc 1–5) | Gemini `gemini-3-flash-preview` | ✅ `Gemini_EMOS.py` | `GEMINI_API_KEY` |
| **VAD** (Valence/Arousal/Dominance 1–5) | Gemini `gemini-3-flash-preview` | ✅ `Gemini_VAD.py` | `GEMINI_API_KEY` |

> 5 lớp cảm xúc EmoCat: **Neutral, Happy, Sad, Angry, Surprise** (người nghe được chọn nhiều lớp → bài toán là dự đoán tỉ lệ vote mỗi lớp).

**Liên hệ baseline:**
- QMOS & EmoCat: ecooper@nict.go.jp (Erica Cooper)
- EMOS & VAD: xiaoxue.gao@u.nus.edu (Xiaoxue Gao)

---

## 2. UTMOS22 (QMOS baseline)

```bash
git clone https://github.com/sarulab-speech/UTMOS22.git
```

- **Paper:** Saeki et al., Interspeech 2022 — UTokyo-SaruLab System for VoiceMOS Challenge 2022.
- **Quản lý môi trường:** poetry (`pyproject.toml` + `poetry.lock`). Dùng fairseq (nặng).
- **Cấu trúc chính:** `strong/` (single model), `stacking/` (ensemble), `fairseq_checkpoints/`.

### ⚠️ Checkpoint chưa được tải sẵn
Repo chỉ chứa script tải, phải chạy để lấy weights:
```bash
cd UTMOS22/fairseq_checkpoints
bash download_strong_checkpoints.sh      # checkpoint cho strong/ + fairseq wav2vec2
bash download_stacking_checkpoints.sh    # checkpoint cho ensemble (nếu cần)
```

### Quick Prediction (dự đoán QMOS)
Theo mục "Quick Prediction" trong README của UTMOS22 — chạy inference trên thư mục wav của data challenge để ra điểm QMOS. (Cập nhật lệnh chính xác sau khi đọc README repo + có data.)

---

## 3. emotion2vec (EmoCat baseline)

```bash
git clone https://github.com/ddlBoJack/emotion2vec.git
```

- **Paper:** Ma et al., Findings of ACL 2024 — emotion2vec: Self-Supervised Pre-Training for Speech Emotion Representation.
- **Cấu trúc chính:** `upstream/`, `src/`, `scripts/`, `iemocap_downstream/`.
- **Model:** `emotion2vec+ large` — tự tải từ HuggingFace/ModelScope khi inference.

### Cách chạy (dùng script BTC sửa sẵn)
File đã sửa nằm trong baseline chính thức:
`baselines/vmc2026-baselines/track2/EmoCat/run_vmc2026.py`
```bash
# 1. Sửa biến `indir` trong run_vmc2026.py → trỏ tới thư mục wav
# 2. Chạy script
python run_vmc2026.py
# 3. Kết quả: category_probs.out (xác suất 5 lớp cảm xúc)
```

> ⚠️ **2 vấn đề của `run_vmc2026.py` gốc — đã xử lý trong pipeline của mình** (`kaggle_baseline/track2_baseline_pipeline.py`):
> 1. **Bug ghi file:** `outl += '\n'` và `outf.write(outl)` bị đặt **trong vòng lặp** → ghi lặp/tích lũy, output sai. Phải đưa ra ngoài, mỗi wav ghi 1 dòng.
> 2. **9 lớp → 5 lớp:** emotion2vec trả 9 lớp (có disgusted/fearful/other/unknown). Format CAT cần đúng 5 lớp (angry/happy/neutral/sad/surprised) và nên **chuẩn hóa lại tổng = 1** sau khi lọc.

---

## 4. Baseline Gemini (EMOS & VAD) — không cần clone

Code nằm sẵn trong `baselines/vmc2026-baselines/track2/EMOS/` và `.../VAD/`.

```bash
pip install pandas google-genai loguru tqdm
export GEMINI_API_KEY=your_api_key_here   # Windows PowerShell: $env:GEMINI_API_KEY="..."

# EMOS — điểm khớp cảm xúc 1–5
python Gemini_EMOS.py --start-row 1 --end-row 10

# VAD — 3 điểm val/aro/dom (1–5)
python Gemini_VAD.py --start-row 1 --end-row 10
```

- Đầu vào: `metadata.csv` (cùng thư mục script) + audio (mặc định `./LT_samples/`, đổi bằng `--base-path`).
- ⚠️ **Free tier Gemini** dễ dính quota → giảm `--workers`, tăng `--retry-sleep`, chạy theo batch nhỏ. Eval quy mô lớn nên dùng paid plan.
- Output Gemini đôi khi thiếu field → kiểm tra CSV, chạy lại sample lỗi (`--resume`).

---

## Checklist setup baseline
- [x] Clone `vmc2026-baselines`, `UTMOS22`, `emotion2vec` ✅ (1/6/2026)
- [x] Viết pipeline gộp Kaggle ✅ `kaggle_baseline/track2_baseline_pipeline.py`
- [ ] Tạo Kaggle Notebook: GPU T4 + Internet On (verify phone) + Add Data (ESD)
- [ ] Test QMOS (SpeechMOS) + EmoCat (emotion2vec) trên ESD — không cần data thật
- [ ] Lấy `GEMINI_API_KEY` → Kaggle Secrets (cho EMOS/VAD)
- [ ] ⏳ Chờ data Track 2 → sửa `WAV_DIR`/`METADATA_CSV` → chạy đủ 4 baseline
- [ ] Reproduce điểm baseline → ghi vào `04_experiments_log.md`
- [ ] Nộp thử `submission.zip` lên CodaBench (thỏa luật ≥1 lần training phase)
- (UTMOS gốc fairseq: chỉ dùng nếu cần khớp tuyệt đối điểm baseline; mặc định dùng SpeechMOS)

---

## Chạy trên Kaggle (GPU free)

Phương án dùng GPU miễn phí khi chưa có GPU lab. **Kết luận: chạy được cả 3 baseline Track 2**, độ dễ khác nhau.

### Hạn mức Kaggle Free
- GPU **T4 16GB** (hoặc P100 16GB); bật được **2×T4**
- **30 giờ GPU/tuần**, mỗi session tối đa **12 giờ**
- Disk: `/kaggle/working` ~20GB (lưu lại) + ~73GB temp
- Cần **bật Internet** trong notebook (Settings → Internet on; phải verify số điện thoại) để tải model / gọi API

### Đánh giá từng baseline
| Thành phần | GPU? | Kaggle? | Độ khó |
|---|---|---|---|
| emotion2vec (EmoCat) | Có (nhẹ) | ✅ Dễ | 🟢 |
| Gemini EMOS / VAD | Không | ✅ Chỉ cần Internet + API key | 🟢 |
| UTMOS22 (QMOS) | Có (nhẹ, inference) | ⚠️ Được nhưng dễ vỡ môi trường | 🟠 |

Inference cả 3 đều fit T4 16GB. Nếu sau này **fine-tune SSL lớn**, giới hạn 30h/tuần + session 12h là rào cản thật → cần checkpoint thường xuyên, chia nhiều session.

### 🟢 emotion2vec (EmoCat) — dễ nhất
```bash
pip install funasr
# Model iic/emotion2vec_plus_large tự tải từ HuggingFace (cần Internet)
# Sửa `indir` trong run_vmc2026.py → thư mục wav, rồi chạy:
python run_vmc2026.py   # output: category_probs.out
```

### 🟢 Gemini EMOS / VAD — không cần GPU
- Chạy trên CPU notebook cũng được.
- Lưu key vào **Kaggle Secrets** (Add-ons → Secrets), tên `GEMINI_API_KEY`, rồi nạp vào env.
```bash
pip install pandas google-genai loguru tqdm
python Gemini_EMOS.py --start-row 1 --end-row 10
```
- ⚠️ Quota free tier Gemini dễ hết với eval lớn → giảm `--workers`, chạy batch nhỏ, `--resume`.

### 🟠 UTMOS22 (QMOS) — chỗ dễ vướng nhất
Vấn đề: `pyproject.toml` pin **torch 1.11, Python ^3.8** + **fork fairseq riêng** (`sarulab-speech/fairseq @ for_utmos`), trong khi Kaggle mặc định Python 3.11 + torch 2.x → dễ xung đột (fairseq = "dependency hell").

Chỉ **inference** ("Quick Prediction"), không train lại → tải nhẹ, fit 16GB. Các lựa chọn:
1. Cài trong **virtualenv riêng** với torch 1.11 (không đụng base env Kaggle).
2. Thử torch mới + patch lỗi import fairseq (có thể phát sinh lỗi).
3. **Phương án nhẹ thay thế:** UTMOS bản pip — `pip install speechmos` ([SpeechMOS](https://github.com/tarepan/SpeechMOS)), không cần fairseq, chạy vài dòng. Phù hợp khi chỉ cần điểm QMOS baseline nhanh.

### Thứ tự nên thử trên Kaggle
**emotion2vec → Gemini → UTMOS (hoặc speechmos)**

---

## Pipeline gộp sẵn (notebook Kaggle)

`kaggle_baseline/track2_baseline_pipeline.py` — chạy cả 4 baseline rồi gộp thành `answer.txt` đúng chuẩn + zip:
- **QMOS:** SpeechMOS qua `torch.hub` (không cần fairseq) → ổn định trên Kaggle.
- **EmoCat:** emotion2vec+ large (đã sửa bug + chuẩn hóa 5 lớp).
- **EMOS/VAD:** gọi script Gemini gốc (cần `metadata.csv` + `GEMINI_API_KEY` qua Kaggle Secrets).
- Tự bỏ cột nếu thiếu dữ liệu → vẫn xuất `answer.txt` hợp lệ (tối thiểu `wav,QMOS,EMOS`).

**Trạng thái (3/6 — data Track 2 đã đủ):** QMOS + EmoCat chạy được trên ESD (Kaggle dataset); EMOS/VAD nay cũng chạy đầy đủ vì đã có `metadata.csv` chứa nhãn cảm xúc target từ gói data thật (ráp qua `track2_prepare_data.ipynb` → 15.477 wav).

Cách dùng: Notebook → GPU T4 + Internet On → Add Data (ESD) → Secrets (`GEMINI_API_KEY`) → sửa `WAV_DIR`/`METADATA_CSV` ở cell 0 → chạy.

---

## Liên kết
- Baseline chính thức: https://github.com/voicemos-challenge/vmc2026-baselines
- UTMOS22: https://github.com/sarulab-speech/UTMOS22
- SpeechMOS (UTMOS bản pip, nhẹ): https://github.com/tarepan/SpeechMOS
- emotion2vec: https://github.com/ddlBoJack/emotion2vec
- URGENT-MOS (Track 1): https://github.com/vvwangvv/URGENT-MOS
- ECAPA-TDNN (Track 3): https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb
