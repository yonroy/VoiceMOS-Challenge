# %% [markdown]
# # VMC2026 Track 2 — Baseline Pipeline (Kaggle)
#
# Chạy 4 baseline → gộp thành `answer.txt` đúng chuẩn nộp CodaBench.
#
# | Sub-task | Baseline | GPU | Cần data thật? |
# |---|---|---|---|
# | QMOS | SpeechMOS (UTMOS bản pip) | có (nhẹ) | không (test trên ESD được) |
# | EmoCat (CAT) | emotion2vec+ large (funasr) | có (nhẹ) | không (test trên ESD được) |
# | EMOS | Gemini LLM-as-judge | không | **có** (cần metadata.csv + API key) |
# | VAD | Gemini LLM-as-judge | không | **có** (cần metadata.csv + API key) |
#
# **Cách dùng trên Kaggle:**
# 1. Tạo Notebook, Settings → Accelerator = **GPU T4**, Internet = **On** (cần verify phone).
# 2. Add Data: `nguyenthanhlim/emotional-speech-dataset-esd` (để test), và dataset Track 2 chính thức khi có.
# 3. Add-ons → Secrets: thêm `GEMINI_API_KEY` (cho EMOS/VAD).
# 4. Copy từng cell (# %%) sang notebook, hoặc upload file này.
#
# Format đích `answer.txt`: `wav,QMOS,EMOS,CAT,VAL,ARO,DOM` — xem `08_track2_spec.md`.
# QMOS & EMOS bắt buộc; CAT/VAD tùy chọn. Có thể nộp tập con cột.

# %% [markdown]
# ## 0. Cấu hình đường dẫn — SỬA Ở ĐÂY

# %%
import os

# Thư mục chứa file .wav cần dự đoán.
# - Test ngay: trỏ tới ESD trên Kaggle.
# - Khi có data thật: trỏ tới thư mục wav của Track 2.
WAV_DIR = "/kaggle/input/emotional-speech-dataset-esd"   # << SỬA khi có data thật

# metadata.csv của data Track 2 (cần cho Gemini EMOS/VAD — chứa nhãn cảm xúc target).
# Chưa có data thì để None, pipeline sẽ bỏ qua bước Gemini.
METADATA_CSV = None   # ví dụ: "/kaggle/input/<track2-data>/metadata.csv"

OUT_DIR = "/kaggle/working"
RUN_QMOS    = True
RUN_EMOCAT  = True
RUN_EMOS    = METADATA_CSV is not None    # cần metadata + API key
RUN_VAD     = METADATA_CSV is not None

EMOTIONS5 = ["angry", "happy", "neutral", "sad", "surprised"]


def list_wavs(d):
    return sorted(f for f in os.listdir(d) if f.lower().endswith(".wav"))


print("WAV_DIR:", WAV_DIR)
print("Số wav:", len(list_wavs(WAV_DIR)) if os.path.isdir(WAV_DIR) else "(chưa thấy thư mục)")

# %% [markdown]
# ## 1. Cài đặt

# %%
# !pip install -q speechmos funasr librosa soundfile pandas google-genai loguru tqdm

# %% [markdown]
# ## 2. QMOS — SpeechMOS (UTMOS)
# Dùng SpeechMOS qua torch.hub (không cần fairseq). Output: dict {wav: score 1-5}.

# %%
def run_qmos(wav_dir):
    import torch, librosa
    # SpeechMOS yêu cầu 16kHz; input shape (Batch, Time); sr truyền dạng keyword.
    predictor = torch.hub.load(
        "tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True
    )
    scores = {}
    for w in list_wavs(wav_dir):
        wave, _ = librosa.load(os.path.join(wav_dir, w), sr=16000, mono=True)
        wave_t = torch.from_numpy(wave).unsqueeze(0)   # (1, Time)
        score = predictor(wave_t, sr=16000)            # → tensor shape (1,)
        scores[w] = float(score.mean().item())
    return scores


qmos_scores = run_qmos(WAV_DIR) if RUN_QMOS else {}
print("QMOS xong:", len(qmos_scores), "mẫu")
list(qmos_scores.items())[:3]

# %% [markdown]
# ## 3. EmoCat — emotion2vec+ large (funasr)
# Sửa bug bản gốc + lọc 5 lớp + **chuẩn hóa tổng = 1** (đúng format CAT).
# Output: dict {wav: {angry:p, happy:p, neutral:p, sad:p, surprised:p}}.

# %%
def run_emocat(wav_dir):
    from funasr import AutoModel
    model = AutoModel(model="iic/emotion2vec_plus_large", hub="hf")
    results = {}
    for w in list_wavs(wav_dir):
        rec = model.generate(
            os.path.join(wav_dir, w),
            granularity="utterance",
            extract_embedding=False,
        )
        labels = rec[0]["labels"]
        scores = rec[0]["scores"]
        # gom điểm 5 lớp quan tâm (label có thể dạng "xx/angry")
        probs = {e: 0.0 for e in EMOTIONS5}
        for lab, sc in zip(labels, scores):
            name = lab.split("/")[-1]
            if name in probs:
                probs[name] = float(sc)
        total = sum(probs.values())
        if total > 0:                      # chuẩn hóa lại trên 5 lớp
            probs = {k: v / total for k, v in probs.items()}
        results[w] = probs
    return results


emocat_probs = run_emocat(WAV_DIR) if RUN_EMOCAT else {}
print("EmoCat xong:", len(emocat_probs), "mẫu")
list(emocat_probs.items())[:2]

# %% [markdown]
# ## 4. EMOS & VAD — Gemini (cần metadata.csv + GEMINI_API_KEY)
# Gọi script baseline gốc trong `vmc2026-baselines/track2/`.
# Trên Kaggle: clone repo + nạp key từ Secrets. Bỏ qua nếu chưa có data.

# %%
def setup_gemini_key():
    try:
        from kaggle_secrets import UserSecretsClient
        os.environ["GEMINI_API_KEY"] = UserSecretsClient().get_secret("GEMINI_API_KEY")
        print("Đã nạp GEMINI_API_KEY từ Kaggle Secrets")
    except Exception as e:
        print("Chưa nạp được key từ Secrets:", e, "→ set thủ công os.environ['GEMINI_API_KEY']")


emos_scores = {}   # {wav: int 1-5}
vad_scores = {}    # {wav: (val, aro, dom)}

if RUN_EMOS or RUN_VAD:
    setup_gemini_key()
    # !git clone -q https://github.com/voicemos-challenge/vmc2026-baselines.git /kaggle/working/vmc2026-baselines
    # Chạy (1-based, inclusive). Với eval lớn nên chia batch + giảm --workers do quota free tier.
    # !cd /kaggle/working/vmc2026-baselines/track2/EMOS && python Gemini_EMOS.py \
    #     --metadata-path {METADATA_CSV} --base-path {WAV_DIR} \
    #     --output-file /kaggle/working/emos.csv --start-row 1 --end-row 50 --workers 4
    # !cd /kaggle/working/vmc2026-baselines/track2/VAD && python Gemini_VAD.py \
    #     --metadata-path {METADATA_CSV} --base-path {WAV_DIR} \
    #     --output-file /kaggle/working/vad.csv --start-row 1 --end-row 50 --workers 4
    import pandas as pd
    if os.path.exists("/kaggle/working/emos.csv"):
        df = pd.read_csv("/kaggle/working/emos.csv")
        emos_scores = dict(zip(df["uttID"], df["emos"]))
    if os.path.exists("/kaggle/working/vad.csv"):
        df = pd.read_csv("/kaggle/working/vad.csv")
        # tên cột VAD tùy script — chỉnh lại nếu khác
        for _, r in df.iterrows():
            vad_scores[r["uttID"]] = (r.get("valence"), r.get("arousal"), r.get("dominance"))

print("EMOS:", len(emos_scores), "| VAD:", len(vad_scores))

# %% [markdown]
# ## 5. Gộp thành `answer.txt`
# QMOS & EMOS bắt buộc. Tự bỏ cột nếu thiếu dữ liệu (nộp tập con hợp lệ).

# %%
def fmt_cat(p):
    return "|".join(f"{e}:{p[e]:.6g}" for e in EMOTIONS5)


def build_answer(out_path):
    wavs = list_wavs(WAV_DIR)
    have_emos = RUN_EMOS and len(emos_scores) > 0
    have_cat  = RUN_EMOCAT and len(emocat_probs) > 0
    have_vad  = RUN_VAD and len(vad_scores) > 0

    cols = ["wav", "QMOS", "EMOS"]          # QMOS+EMOS bắt buộc
    if have_cat:  cols.append("CAT")
    if have_vad:  cols += ["VAL", "ARO", "DOM"]

    n = 0
    with open(out_path, "w") as f:
        f.write(",".join(cols) + "\n")
        for w in wavs:
            row = [w,
                   f"{qmos_scores.get(w, 3.0):.6g}",
                   str(emos_scores.get(w, 3))]
            if have_cat:
                row.append(fmt_cat(emocat_probs.get(w, {e: 0.2 for e in EMOTIONS5})))
            if have_vad:
                v = vad_scores.get(w, (3, 3, 3))
                row += [str(v[0]), str(v[1]), str(v[2])]
            f.write(",".join(row) + "\n")
            n += 1
    print(f"Ghi {n} dòng → {out_path} | cột: {cols}")
    return cols


answer_path = os.path.join(OUT_DIR, "answer.txt")
cols = build_answer(answer_path)

# %% [markdown]
# ## 6. Validate + đóng zip

# %%
def validate(path):
    import csv
    with open(path) as f:
        rows = list(csv.reader(f))
    header = rows[0]
    assert header[0] == "wav" and "QMOS" in header and "EMOS" in header, "Header sai"
    for i, r in enumerate(rows[1:], 2):
        assert len(r) == len(header), f"Dòng {i} sai số cột"
    print(f"OK: {len(rows)-1} dòng, header = {header}")


validate(answer_path)
# !cd /kaggle/working && zip -j submission.zip answer.txt && unzip -l submission.zip
print("Sẵn sàng nộp: /kaggle/working/submission.zip (chứa answer.txt)")

# %% [markdown]
# ## Ghi chú
# - Nộp: My Submissions → chọn **Track 2**, **bỏ chọn** track khác → upload `submission.zip`.
# - `metadata.csv` đi kèm data Track 2 chính thức; chứa nhãn cảm xúc target cho Gemini EMOS/VAD.
# - Quota Gemini free tier dễ hết với eval lớn → chia batch `--start-row/--end-row`, giảm `--workers`, dùng `--resume`.
# - Khi có data thật: sửa `WAV_DIR`, `METADATA_CSV` ở cell 0 rồi chạy lại từ đầu.
