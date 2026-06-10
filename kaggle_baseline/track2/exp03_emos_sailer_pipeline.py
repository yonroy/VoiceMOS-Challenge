# %% [markdown]
# # VMC2026 Track 2 — exp03 (EMOS bằng SAILER, offline) — Kaggle
#
# **Mục tiêu:** chấm **EMOS** (độ khớp cảm xúc target) bằng model **SAILER**
# (`tiantiaf/wavlm-large-categorical-emotion`, vô địch Interspeech 2025 SER),
# thay cho emotion2vec — KHÔNG train, chỉ lấy xác suất lớp cảm xúc target.
#
# ## Ý tưởng (đọc 1 lần cho hiểu)
# SAILER nhận 1 wav → xuất **logits 9 lớp cảm xúc** → softmax → **xác suất từng lớp**.
# EMOS = mức khớp cảm xúc target → lấy thẳng **P(cảm xúc target)** rồi kéo về thang 1–5:
#
# ```
# mỗi wav ─► SAILER (WavLM-large) ─► softmax 9 lớp ─┬─► P(target)  ─► EMOS = 1 + 4·P
#                                                   └─► 5 lớp (renorm) ─► CAT
#       target emotion (metadata.csv) ─────────────────┘
# ```
#
# - **9 lớp SAILER:** `Anger, Contempt, Disgust, Fear, Happiness, Neutral, Sadness, Surprise, Other`.
#   → đủ cả 5 lớp challenge (angry/happy/neutral/sad/surprised).
# - **EMOS** = `1 + 4·P(target)` (scale [0,1]→[1,5]); SRCC bất biến với scale tuyến tính.
# - **CAT** = lấy xác suất 5 lớp challenge từ chính SAILER (renormalize tổng=1).
# - **VAD** = arousal/valence/dominance SAILER xuất sẵn (sigmoid 0–1 → 1–5) → 1 model lo EMOS+CAT+VAD!
# - **QMOS** = SpeechMOS (UTMOS) — bắt buộc để `answer.txt` hợp lệ.
# - KHÔNG train → nộp được ngay. So điểm EMOS với baseline emotion2vec (0.194) và exp01.
#
# **Cách chạy trên Kaggle:** Settings → Accelerator = **GPU T4**, Internet = **On**
# → + Add Input dataset Track 2 (15.477 wav, có `sets/dev.scp`, `metadata.csv`)
# → sửa `DATA_ROOT` ở cell 0 → Run All.
#
# ⚠️ License SAILER = **Open RAIL** (phi thương mại) → phải khai báo trong `docs/12_system_description.md`.

# %% [markdown]
# ## 0. Cấu hình — SỬA Ở ĐÂY

# %%
import os

# ── Data Track 2 (dataset 15.477 wav đã ráp) ────────────────────────────────
DATA_ROOT    = "/kaggle/input/vmc2026-track2-full/vmc2026-track2"   # << SỬA slug cho khớp Add Input
WAV_DIR      = f"{DATA_ROOT}/wav"
METADATA_CSV = f"{DATA_ROOT}/metadata.csv"     # wavID|emotion|transcript (KHÔNG header) → target emotion
DEV_SCP      = f"{DATA_ROOT}/sets/dev.scp"     # danh sách wav tập DEV (tập cần nộp ở training phase)

OUT_DIR = "/kaggle/working"

DEVICE      = "cuda"        # "cuda" trên Kaggle GPU; "cpu" nếu không có GPU
MAX_SECONDS = 15           # SAILER nhận tối đa 15s (giới hạn của model)
SR          = 16000        # SAILER cần 16kHz mono
LIMIT       = None          # đặt số nhỏ (vd 20) để chạy thử nhanh; None = full DEV

# 5 lớp cảm xúc challenge (thứ tự cố định cho cột CAT)
EMOTIONS5 = ["angry", "happy", "neutral", "sad", "surprised"]

# 9 lớp SAILER (đúng thứ tự model xuất) + chỉ số của 5 lớp challenge trong đó
SAILER9 = ["Anger", "Contempt", "Disgust", "Fear", "Happiness", "Neutral", "Sadness", "Surprise", "Other"]
EMO2SAILER = {"angry": 0, "happy": 4, "neutral": 5, "sad": 6, "surprised": 7}   # EMOTIONS5 → index trong SAILER9

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

def stem(path_or_name):
    return os.path.splitext(os.path.basename(str(path_or_name)))[0]

print("DATA_ROOT:", DATA_ROOT)
for p in [WAV_DIR, METADATA_CSV, DEV_SCP]:
    print(("  ✅ " if os.path.exists(p) else "  ❌ THIẾU ") + p)

# %% [markdown]
# ## 1. Cài đặt + tải code SAILER
# SAILER cần file `WavLMWrapper` trong repo `vox-profile-release`.
# ⚠️ **KHÔNG** `pip install -e .` (build wheel của repo hay lỗi trên Kaggle). Thay vào đó:
# chỉ **clone + thêm repo vào `sys.path`** rồi cài đúng vài thư viện model cần
# (`transformers/torch/huggingface_hub` Kaggle đã có sẵn; chỉ thiếu `loralib`, `speechbrain`).

# %%
import sys, subprocess

def pip_install(*pkgs):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], check=True)

REPO_DIR = "/kaggle/working/vox-profile-release"
if not os.path.exists(REPO_DIR):
    subprocess.run(["git", "clone", "--depth", "1",
                    "https://github.com/tiantiaf0627/vox-profile-release.git", REPO_DIR], check=True)

# Deps mà WavLMWrapper cần (xem import trong src/model/emotion/wavlm_emotion.py) + thư viện chấm QMOS.
pip_install("loralib", "speechbrain", "speechmos", "librosa", "soundfile", "scipy", "tqdm")

if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)     # để `from src.model.emotion... import WavLMWrapper` chạy được

# %% [markdown]
# ## 2. Nạp model SAILER

# %%
import torch
import torch.nn.functional as F

device = DEVICE if torch.cuda.is_available() else "cpu"
print("Device:", device)

from src.model.emotion.wavlm_emotion import WavLMWrapper   # noqa: E402

sailer = WavLMWrapper.from_pretrained("tiantiaf/wavlm-large-categorical-emotion").to(device)
sailer.eval()
print("✅ Đã nạp SAILER (wavlm-large-categorical-emotion)")

# %% [markdown]
# ## 3. Đọc cảm xúc target cho mỗi wav (từ metadata.csv)

# %%
def load_target_emotions():
    """metadata.csv (wavID|emotion|transcript, KHÔNG header) → {stem: emotion_chuẩn|None}."""
    tgt = {}
    with open(METADATA_CSV, encoding="utf-8") as f:
        for ln in f:
            parts = ln.strip().split("|")
            if len(parts) < 2:
                continue
            tgt[stem(parts[0])] = norm_emotion(parts[1])
    return tgt

target_map = load_target_emotions()
print(f"Target emotions: {len(target_map)} wav | ví dụ:", dict(list(target_map.items())[:3]))

# %% [markdown]
# ## 4. Hàm chấm 1 wav bằng SAILER → xác suất 9 lớp + VAD
# WavLMWrapper khi `return_feature=True` trả **6 giá trị**:
# `predicted(logits 9 lớp), features, detailed_logits, arousal, valence, dominance` (VAD sigmoid 0–1).
# → 1 model lo cả **EMOS** (P target), **CAT** (5 lớp renorm) **và VAD** (mở 3 cột đang trống!).

# %%
import numpy as np
import librosa

@torch.no_grad()
def sailer_infer(wav_path):
    """→ (probs9: float32[9], vad3: float32[3] theo thứ tự [VAL,ARO,DOM] thang 1–5);
       None nếu thiếu/lỗi file."""
    if not os.path.exists(wav_path):
        return None
    wave, _ = librosa.load(wav_path, sr=SR, mono=True)
    wave = wave[: MAX_SECONDS * SR]                       # cắt tối đa 15s
    data = torch.from_numpy(wave).float().unsqueeze(0).to(device)
    logits, _feat, _det, arousal, valence, dominance = sailer(data, return_feature=True)
    probs9 = F.softmax(logits, dim=1)[0].detach().cpu().numpy().astype(np.float32)
    # VAD sigmoid [0,1] → thang 1–5 cho khớp ví dụ BTC (SRCC bất biến với scale tuyến tính)
    v, a, d = float(valence.item()), float(arousal.item()), float(dominance.item())
    vad3 = np.array([1 + 4 * v, 1 + 4 * a, 1 + 4 * d], dtype=np.float32)   # [VAL, ARO, DOM]
    return probs9, vad3

def emos_from_probs(probs9, target):
    """EMOS = 1 + 4·P(target). None nếu không biết target → để caller xử lý mặc định."""
    if target is None or target not in EMO2SAILER:
        return None
    return 1.0 + 4.0 * float(probs9[EMO2SAILER[target]])

def cat5_from_probs(probs9):
    """Lấy 5 lớp challenge từ 9 lớp SAILER rồi renormalize tổng=1."""
    v = np.array([probs9[EMO2SAILER[e]] for e in EMOTIONS5], dtype=np.float32)
    s = v.sum()
    return v / s if s > 0 else np.full(5, 0.2, dtype=np.float32)

# %% [markdown]
# ## 5. QMOS = SpeechMOS (UTMOS) — bắt buộc cho answer.txt

# %%
@torch.no_grad()
def run_qmos(names):
    predictor = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True).to(device).eval()
    from tqdm.auto import tqdm
    out = {}
    for n in tqdm(names, desc="QMOS"):
        p = os.path.join(WAV_DIR, n)
        if not os.path.exists(p):
            continue
        wave, _ = librosa.load(p, sr=SR, mono=True)
        x = torch.from_numpy(wave).unsqueeze(0).to(device)   # đẩy input lên GPU
        out[n] = float(predictor(x, sr=SR).mean().item())
    return out

# %% [markdown]
# ## 6. Chạy trên DEV → `answer.txt` đầy đủ (QMOS, EMOS, CAT, VAL, ARO, DOM)

# %%
def list_dev():
    with open(DEV_SCP) as f:
        return [ln.strip() for ln in f if ln.strip()]

dev_names = list_dev()
if LIMIT:
    dev_names = dev_names[:LIMIT]
print("DEV:", len(dev_names), "mẫu")

qmos_scores = run_qmos(dev_names)

def fmt_cat(probs5):
    return "|".join(f"{e}:{probs5[i]:.6g}" for i, e in enumerate(EMOTIONS5))

def build_answer(out_path):
    from tqdm.auto import tqdm
    n_emos = n_default = 0
    with open(out_path, "w") as f:
        f.write("wav,QMOS,EMOS,CAT,VAL,ARO,DOM\n")
        for name in tqdm(dev_names, desc="SAILER EMOS/CAT/VAD"):
            sid = stem(name)
            out = sailer_infer(os.path.join(WAV_DIR, name))
            if out is None:
                emos, cat5 = 3.0, np.full(5, 0.2, dtype=np.float32)
                vad3 = np.array([3.0, 3.0, 3.0], dtype=np.float32)
                n_default += 1
            else:
                probs9, vad3 = out
                emos = emos_from_probs(probs9, target_map.get(sid))
                if emos is None:
                    emos = 3.0; n_default += 1
                else:
                    n_emos += 1
                cat5 = cat5_from_probs(probs9)
            qmos = qmos_scores.get(name, 3.0)
            f.write(f"{name},{qmos:.6g},{emos:.6g},{fmt_cat(cat5)},"
                    f"{vad3[0]:.6g},{vad3[1]:.6g},{vad3[2]:.6g}\n")
    print(f"Ghi {len(dev_names)} dòng → {out_path} | EMOS thật {n_emos}, mặc định {n_default}")

answer_path = os.path.join(OUT_DIR, "answer.txt")
build_answer(answer_path)

# %% [markdown]
# ## 7. Validate + đóng zip

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
os.system(f"cd {OUT_DIR} && zip -j submission_track2_exp03_sailer.zip answer.txt && unzip -l submission_track2_exp03_sailer.zip")
print("Sẵn sàng nộp:", os.path.join(OUT_DIR, "submission_track2_exp03_sailer.zip"))

# %% [markdown]
# ## Ghi chú
# - **Chưa chạy thật bao giờ** → lần đầu đặt `LIMIT = 20` ở cell 0 để bắt lỗi setup (clone repo / import / model).
# - Điểm DEV thật phải nộp lên CodaBench mới biết (My Submissions → Track 2, bỏ chọn track khác).
# - Notebook này đổi **EMOS + CAT + VAD** sang SAILER (1 model lo 6 cột metric). QMOS vẫn SpeechMOS cũ.
#   Muốn ablation EMOS sạch (giữ CAT=emotion2vec) thì chỉ lấy cột EMOS từ đây, ghép với CAT của `track2_baseline`.
# - Rủi ro setup duy nhất = import `src.model.emotion.wavlm_emotion` (cần repo vox-profile-release).
#   Nếu lỗi import: kiểm tra `REPO_DIR` đã clone + `sys.path` đã thêm REPO_DIR (KHÔNG dùng pip install -e .).
# - Nhớ ghi config → kết quả → nhận xét vào `docs/04_experiments_log.md` (mục exp03).
