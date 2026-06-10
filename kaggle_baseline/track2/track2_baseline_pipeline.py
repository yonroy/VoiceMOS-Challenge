# %% [markdown]
# # VMC2026 Track 2 — Baseline Pipeline (Kaggle)
#
# Chạy 4 baseline → gộp thành `answer.txt` đúng chuẩn nộp CodaBench.
#
# | Sub-task | Baseline | GPU | Cần label? |
# |---|---|---|---|
# | QMOS | SpeechMOS (UTMOS bản pip) | có (nhẹ) | không (chỉ cần wav) |
# | EmoCat (CAT) | emotion2vec+ large (funasr) | có (nhẹ) | không (chỉ cần wav) |
# | EMOS | **emotion2vec target-prob** (mặc định, offline) HOẶC Gemini | có (nhẹ) | cần `metadata.csv` (nhãn target) |
# | VAD | Gemini LLM-as-judge (chỉ khi EMOS_METHOD="gemini") | không | cần `metadata.csv` + API key |
#
# **Cách dùng trên Kaggle:**
# 1. Tạo Notebook, Settings → Accelerator = **GPU T4**, Internet = **On** (cần verify phone).
# 2. **+ Add Input** → chọn dataset Track 2 đã upload (Kaggle tự giải nén → có thư mục `vmc2026-track2/`).
# 3. Add-ons → Secrets: thêm `GEMINI_API_KEY` (cho EMOS/VAD).
# 4. Sửa `DATA_ROOT` ở cell 0 cho khớp slug dataset, rồi chạy lần lượt từng cell.
#
# Format đích `answer.txt`: `wav,QMOS,EMOS,CAT,VAL,ARO,DOM` — xem `08_track2_spec.md`.
# QMOS & EMOS bắt buộc; CAT/VAD tùy chọn. Có thể nộp tập con cột.
#
# > ⚠️ Ở **training phase** ta dự đoán cho tập **DEV** (`sets/dev.scp`, ~2730 mẫu) rồi nộp.
# > Thư mục `wav/` chứa cả train + dev nên KHÔNG glob hết — chỉ lấy đúng danh sách dev.scp.

# %% [markdown]
# ## 0. Cấu hình đường dẫn — SỬA Ở ĐÂY

# %%
import os, glob

# ── Data Track 2 trên Kaggle ────────────────────────────────────────────────
# Kaggle TỰ giải nén .tar.gz khi tạo Dataset → có sẵn thư mục `vmc2026-track2/`.
# Đổi <track2-data> thành slug dataset của bạn (xem thanh path bên phải khi Add Input).
DATA_ROOT    = "/kaggle/input/<track2-data>/vmc2026-track2"   # << SỬA slug
WAV_DIR      = f"{DATA_ROOT}/wav"
METADATA_CSV = f"{DATA_ROOT}/metadata.csv"      # định dạng: wavID|emotion|transcript (KHÔNG header)
DEV_SCP      = f"{DATA_ROOT}/sets/dev.scp"      # danh sách wav của tập DEV (tập cần nộp ở train phase)

# Muốn test nhanh trên ESD (chưa có data thật): trỏ WAV_DIR vào ESD, đặt DEV_SCP=None, METADATA_CSV=None.
# WAV_DIR = "/kaggle/input/datasets/nguyenthanhlim/emotional-speech-dataset-esd/Emotion Speech Dataset"
# DEV_SCP = None; METADATA_CSV = None

LIMIT = None   # << số nhỏ (vd 20) để chạy thử cho nhanh; None = chạy toàn bộ tập DEV

OUT_DIR = "/kaggle/working"

# ── Cách tính EMOS ──────────────────────────────────────────────────────────
# "emotion2vec": OFFLINE, MIỄN PHÍ (exp01, khuyến nghị) — lấy P(cảm xúc target) từ
#                emotion2vec (model đã chạy cho CAT) rồi scale về thang 1–5. Không cần API.
# "gemini"     : LLM-as-judge qua Gemini API (cần GEMINI_API_KEY, tốn phí). Chỉ cách này có VAD.
EMOS_METHOD = "emotion2vec"

_have_meta  = bool(METADATA_CSV) and os.path.exists(METADATA_CSV)   # cần nhãn cảm xúc target
RUN_QMOS    = True
RUN_EMOCAT  = True
RUN_EMOS    = _have_meta                                  # cả 2 cách đều cần target từ metadata
RUN_VAD     = _have_meta and EMOS_METHOD == "gemini"      # VAD chỉ có ở Gemini

EMOTIONS5 = ["angry", "happy", "neutral", "sad", "surprised"]

# Chuẩn hóa nhãn cảm xúc target (metadata) → đúng 1 trong 5 lớp của emotion2vec.
_EMO_ALIAS = {
    "angry": "angry", "anger": "angry",
    "happy": "happy", "happiness": "happy", "joy": "happy",
    "neutral": "neutral", "calm": "neutral",
    "sad": "sad", "sadness": "sad",
    "surprise": "surprised", "surprised": "surprised", "surprising": "surprised",
}

def norm_emotion(label):
    """Đưa nhãn cảm xúc bất kỳ về 1 trong EMOTIONS5; None nếu không khớp."""
    key = str(label).strip().lower()
    return _EMO_ALIAS.get(key, key if key in EMOTIONS5 else None)


def list_wavs(d):
    """Trả về list đường dẫn .wav đầy đủ cần dự đoán.
    - Có DEV_SCP  → đọc danh sách tên file tập DEV (đúng tập cần nộp, wav nằm phẳng trong wav/).
    - Không có    → quét đệ quy mọi .wav trong thư mục (chế độ test ESD, lồng speaker/emotion)."""
    if DEV_SCP and os.path.exists(DEV_SCP):
        with open(DEV_SCP) as f:
            names = [ln.strip() for ln in f if ln.strip()]
        wavs = [os.path.join(d, n) for n in names]
    else:
        wavs = sorted(glob.glob(os.path.join(d, "**", "*.wav"), recursive=True))
    return wavs[:LIMIT] if LIMIT else wavs


print("WAV_DIR:", WAV_DIR)
print("Số wav:", len(list_wavs(WAV_DIR)) if os.path.isdir(WAV_DIR) else "(chưa thấy thư mục)")
print("Chế độ DEV (dev.scp):", bool(DEV_SCP and os.path.exists(DEV_SCP)))

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
    missing = 0
    for w in list_wavs(wav_dir):
        if not os.path.exists(w):       # mẫu ESD/DailyTalk chưa lấy ngoài → bỏ qua, không crash
            missing += 1
            continue
        wave, _ = librosa.load(w, sr=16000, mono=True)
        wave_t = torch.from_numpy(wave).unsqueeze(0)   # (1, Time)
        score = predictor(wave_t, sr=16000)            # → tensor shape (1,)
        scores[w] = float(score.mean().item())
    if missing:
        print(f"[QMOS] Bỏ qua {missing} file thiếu (chưa có ESD/DailyTalk) → sẽ nhận điểm mặc định.")
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
    missing = 0
    for w in list_wavs(wav_dir):
        if not os.path.exists(w):       # mẫu ESD/DailyTalk chưa lấy ngoài → bỏ qua
            missing += 1
            continue
        rec = model.generate(
            w,
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
    if missing:
        print(f"[EmoCat] Bỏ qua {missing} file thiếu (chưa có ESD/DailyTalk) → sẽ nhận phân bố mặc định.")
    return results


emocat_probs = run_emocat(WAV_DIR) if RUN_EMOCAT else {}
print("EmoCat xong:", len(emocat_probs), "mẫu")
list(emocat_probs.items())[:2]

# %% [markdown]
# ## 4. EMOS — emotion2vec target-prob (mặc định) hoặc Gemini
# **Cách emotion2vec (exp01, offline):** với mỗi wav, lấy xác suất emotion2vec gán cho ĐÚNG cảm xúc
# target (đọc từ `metadata.csv`), scale [0,1] → [1,5]. Vì EMOS chấm bằng SRCC (thứ hạng) nên scale
# tuyến tính không đổi tương quan — chỉ cần thứ tự đúng. KHÔNG cần train, KHÔNG cần API.
# **Cách Gemini:** gọi script baseline gốc (cần `GEMINI_API_KEY`); chỉ cách này mới có VAD.
# `metadata.csv` dạng `wavID|emotion|transcript` (không header); `emotion` = cảm xúc target.

# %%
def load_target_emotions():
    """Đọc metadata.csv (wavID|emotion|transcript, không header) → {stem: emotion_chuẩn}."""
    tgt = {}
    if not (METADATA_CSV and os.path.exists(METADATA_CSV)):
        return tgt
    with open(METADATA_CSV, encoding="utf-8") as f:
        for ln in f:
            parts = ln.strip().split("|")
            if len(parts) < 2:
                continue
            stem = os.path.splitext(os.path.basename(parts[0]))[0]
            tgt[stem] = norm_emotion(parts[1])
    return tgt


def run_emos_emotion2vec(wav_dir, target_map):
    """EMOS offline = P(cảm xúc target) từ emotion2vec, scale [0,1] → [1,5].
    Dùng lại emocat_probs (đã tính ở mục 3) nên KHÔNG tốn thêm tính toán."""
    out, miss_tgt, miss_prob = {}, 0, 0
    for w in list_wavs(wav_dir):
        name = os.path.basename(w)
        tgt = target_map.get(os.path.splitext(name)[0])
        probs = emocat_probs.get(w)
        if tgt is None:
            miss_tgt += 1; continue
        if not probs:
            miss_prob += 1; continue
        out[name] = 1.0 + 4.0 * probs.get(tgt, 0.0)   # p=0→1 điểm, p=1→5 điểm
    if miss_tgt:  print(f"[EMOS-e2v] {miss_tgt} mẫu thiếu nhãn target → mặc định 3.")
    if miss_prob: print(f"[EMOS-e2v] {miss_prob} mẫu thiếu prob emotion2vec → mặc định 3.")
    return out


def setup_gemini_key():
    try:
        from kaggle_secrets import UserSecretsClient
        os.environ["GEMINI_API_KEY"] = UserSecretsClient().get_secret("GEMINI_API_KEY")
        print("Đã nạp GEMINI_API_KEY từ Kaggle Secrets")
    except Exception as e:
        print("Chưa nạp được key từ Secrets:", e, "→ set thủ công os.environ['GEMINI_API_KEY']")


emos_scores = {}   # {uttID(tên file .wav): điểm EMOS}
vad_scores = {}    # {uttID: (val, aro, dom)} — chỉ có khi dùng Gemini

target_map = load_target_emotions()
print("Nhãn cảm xúc target đọc được:", len(target_map))

if RUN_EMOS and EMOS_METHOD == "emotion2vec":
    assert RUN_EMOCAT and emocat_probs, "EMOS theo emotion2vec cần EmoCat chạy trước (RUN_EMOCAT=True)."
    emos_scores = run_emos_emotion2vec(WAV_DIR, target_map)

elif RUN_EMOS and EMOS_METHOD == "gemini":
    setup_gemini_key()
    # !git clone -q https://github.com/voicemos-challenge/vmc2026-baselines.git /kaggle/working/vmc2026-baselines
    # Chạy (1-based, inclusive). Với eval lớn nên chia batch + giảm --workers do quota free tier.
    # Gemini chỉ chấm các dòng metadata.csv trùng với DEV; cách đơn giản: chạy hết rồi lọc lại ở build_answer.
    # !cd /kaggle/working/vmc2026-baselines/track2/EMOS && python Gemini_EMOS.py \
    #     --metadata-path {METADATA_CSV} --base-path {WAV_DIR} \
    #     --output-file /kaggle/working/emos.csv --workers 4 --resume
    # !cd /kaggle/working/vmc2026-baselines/track2/VAD && python Gemini_VAD.py \
    #     --metadata-path {METADATA_CSV} --base-path {WAV_DIR} \
    #     --output-file /kaggle/working/vad.csv --workers 4 --resume
    import pandas as pd
    if os.path.exists("/kaggle/working/emos.csv"):
        df = pd.read_csv("/kaggle/working/emos.csv")
        emos_scores = dict(zip(df["uttID"], df["emos"]))
    if os.path.exists("/kaggle/working/vad.csv"):
        df = pd.read_csv("/kaggle/working/vad.csv")
        # cột output VAD chuẩn của Gemini_VAD.py: uttID, val, aro, dom
        for _, r in df.iterrows():
            vad_scores[r["uttID"]] = (r["val"], r["aro"], r["dom"])

print("EMOS:", len(emos_scores), "| VAD:", len(vad_scores))

# %% [markdown]
# ## 5. Gộp thành `answer.txt`
# QMOS & EMOS bắt buộc. Tự bỏ cột nếu thiếu dữ liệu (nộp tập con hợp lệ).
# Lưu ý key: qmos/emocat theo path đầy đủ; emos/vad theo TÊN FILE → tra cứu bằng basename.

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
            name = os.path.basename(w)              # tên file = key của emos/vad, và là giá trị cột wav
            row = [name,
                   f"{qmos_scores.get(w, 3.0):.6g}",
                   str(emos_scores.get(name, 3))]
            if have_cat:
                row.append(fmt_cat(emocat_probs.get(w, {e: 0.2 for e in EMOTIONS5})))
            if have_vad:
                v = vad_scores.get(name, (3, 3, 3))
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
# !cd /kaggle/working && zip -j submission_track2.zip answer.txt && unzip -l submission_track2.zip
print("Sẵn sàng nộp: /kaggle/working/submission_track2.zip (chứa answer.txt)")

# %% [markdown]
# ## Ghi chú
# - Nộp: My Submissions → chọn **Track 2**, **bỏ chọn** track khác → upload `submission_track2.zip`.
# - `metadata.csv` (wavID|emotion|transcript, không header) chứa nhãn cảm xúc target cho Gemini EMOS/VAD.
# - Train phase: dự đoán tập DEV (`sets/dev.scp`). `sets/train.csv` có nhãn người nghe để train mô hình riêng.
# - Quota Gemini free tier dễ hết với eval lớn → chia batch `--start-row/--end-row`, giảm `--workers`, dùng `--resume`.
# - Khi có data thật: sửa `DATA_ROOT` ở cell 0 rồi chạy lại từ đầu.
