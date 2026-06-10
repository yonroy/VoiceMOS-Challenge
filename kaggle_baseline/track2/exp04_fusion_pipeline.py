# %% [markdown]
# # VMC2026 Track 2 — exp04 (FUSION multi-task) — Kaggle
#
# **Mục tiêu:** gộp 2 backbone bổ sung nhau (**emotion2vec** thắng EMOS · **SAILER/WavLM** thắng VAD)
# thành **1 model multi-task** dự đoán chung 5 đầu ra cảm xúc: **EMOS · CAT · VAL · ARO · DOM**.
# QMOS để **riêng** (giữ SpeechMOS) — đúng thiết kế đã chốt: *"QMOS riêng + 5 cảm xúc chung"*.
#
# ## Ý tưởng (đọc 1 lần cho hiểu)
# Bằng chứng để fusion (từ exp01 & exp03): emotion2vec đứng đầu **EMOS** (0.637), SAILER đứng đầu
# **VAD** (ARO 0.712 / DOM 0.630). Hai model "nhìn" cảm xúc theo cách khác nhau → **nối đặc trưng**
# của cả hai rồi cho một mạng nhỏ học → kỳ vọng mạnh hơn từng model lẻ.
#
# ```
#                 ┌─ emotion2vec ─► embedding ~D1 + xác suất 5 lớp ─┐
#  mỗi wav ──────►│                                                 ├─► NỐI ─► TRUNK chung
#                 └─ SAILER(WavLM) ► embedding ~D2 + 9 lớp + VAD3  ─┘        (Linear+ReLU)
#                                                                              │
#                              ┌───────────────────────────────────────────────┤
#       target emotion(one-hot)│                                                │
#                              ▼                                                ▼
#                       [EMOS head]                              [CAT head]  [VAD head]
#                       (cần target)                             (5 lớp)     (VAL/ARO/DOM)
# ```
#
# - **Cả 2 backbone ĐÓNG BĂNG** → chỉ trích đặc trưng (cache `.npz`), **chỉ train phần trunk + head nhỏ**
#   → nhẹ GPU, train vài phút, hợp T4. (Né fine-tune end-to-end lúc đầu.)
# - **EMOS phụ thuộc target** (cùng audio, target khác → điểm khác) → EMOS head nhận thêm one-hot target.
#   **CAT/VAD** là cảm nhận về chính audio → chỉ cần trunk (không cần target).
# - **Nhãn vàng** gộp theo `wavID` từ `sets/train.csv`:
#   EMOS = TB `eMOS` · VAL/ARO/DOM = TB `val/aro/dom` · CAT = **tỉ lệ vote 5 lớp** của `emoCat`.
# - **Cân loss = uncertainty weighting** (Kendall 2018): mỗi task có 1 trọng số σ **tự học**
#   → không phải dò tay. Có cờ `USE_UNCERTAINTY=False` để quay về trọng số cố định khi cần debug.
# - Cuối cùng xuất `answer.txt` **đủ 7 cột**: `wav,QMOS,EMOS,CAT,VAL,ARO,DOM`
#   (QMOS=SpeechMOS · 5 cột còn lại = model fusion) → nộp được ngay. So mốc: EMOS 0.637 · VAD ARO 0.712.
#
# **Cách chạy trên Kaggle:** Settings → Accelerator = **GPU T4**, Internet = **On**
# → + Add Input dataset Track 2 (15.477 wav, có `sets/train.csv`, `sets/dev.scp`, `metadata.csv`)
# → sửa `DATA_ROOT` ở cell 0 → Run All. Lần đầu nên đặt `LIMIT_TRAIN = 300`, `LIMIT_DEV = 20` để bắt lỗi setup.

# %% [markdown]
# ## 0. Cấu hình — SỬA Ở ĐÂY

# %%
import os

# ── Data Track 2 (dataset 15.477 wav đã ráp, có sets/train.csv) ──────────────
DATA_ROOT    = "/kaggle/input/vmc2026-track2-full/vmc2026-track2"   # << SỬA slug cho khớp Add Input
WAV_DIR      = f"{DATA_ROOT}/wav"
METADATA_CSV = f"{DATA_ROOT}/metadata.csv"     # wavID|emotion|transcript (KHÔNG header) → target emotion
TRAIN_CSV    = f"{DATA_ROOT}/sets/train.csv"   # nhãn người nghe: lisID,wavID,qMOS,emoCat,eMOS,val,dom,aro
DEV_SCP      = f"{DATA_ROOT}/sets/dev.scp"     # danh sách wav tập DEV (tập cần nộp ở training phase)

OUT_DIR   = "/kaggle/working"
CACHE_DIR = "/kaggle/working/fusion_cache"     # cache embedding 2 backbone (tái dùng giữa các lần chạy)
os.makedirs(CACHE_DIR, exist_ok=True)

# ── Siêu tham số train ───────────────────────────────────────────────────────
DEVICE          = "cuda"      # "cuda" trên Kaggle GPU; "cpu" nếu không có GPU
TRUNK_HIDDEN    = 512         # số neuron lớp trunk chung
HEAD_HIDDEN     = 128         # số neuron lớp ẩn mỗi head
DROPOUT         = 0.3
LR              = 1e-3
EPOCHS          = 80
BATCH           = 64
VAL_FRAC        = 0.10        # 10% train → validation nội bộ (đo SRCC từng task)
PATIENCE        = 15          # early stop theo điểm tổng val (xem SCORE_FOR_STOP)
SEED            = 42

USE_UNCERTAINTY = True        # True = tự cân loss (Kendall); False = dùng LOSS_W cố định bên dưới
LOSS_W          = {"emos": 1.0, "cat": 1.0, "val": 1.0, "aro": 1.0, "dom": 1.0}  # chỉ dùng khi tắt uncertainty
USE_E2V         = True        # bật/tắt nhánh emotion2vec trong fusion (để ablation)
USE_SAILER      = True        # bật/tắt nhánh SAILER trong fusion (để ablation)
USE_CLASSPROB   = True        # thêm xác suất lớp (e2v 5 + sailer 9) + VAD3 của SAILER vào feature

LIMIT_TRAIN     = None        # đặt số nhỏ (vd 300) để chạy thử nhanh; None = full
LIMIT_DEV       = None        # đặt số nhỏ (vd 20) để chạy thử nhanh; None = full

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
    """Lấy tên file không đuôi, để khớp wavID giữa train.csv / metadata / dev.scp."""
    return os.path.splitext(os.path.basename(str(path_or_name)))[0]

assert USE_E2V or USE_SAILER, "Phải bật ít nhất 1 backbone (USE_E2V hoặc USE_SAILER)."
print("DATA_ROOT:", DATA_ROOT)
for p in [WAV_DIR, METADATA_CSV, TRAIN_CSV, DEV_SCP]:
    print(("  ✅ " if os.path.exists(p) else "  ❌ THIẾU ") + p)

# %% [markdown]
# ## 1. Cài đặt + tải code SAILER
# emotion2vec qua `funasr` (offline). SAILER cần `WavLMWrapper` trong repo `vox-profile-release`
# → **clone + sys.path** (KHÔNG `pip install -e .` vì build wheel hay lỗi trên Kaggle).

# %%
import sys, subprocess

def pip_install(*pkgs):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], check=True)

pip_install("speechmos", "funasr", "librosa", "soundfile", "pandas", "scipy", "scikit-learn", "tqdm")

if USE_SAILER:
    pip_install("loralib", "speechbrain")   # deps WavLMWrapper cần
    REPO_DIR = "/kaggle/working/vox-profile-release"
    if not os.path.exists(REPO_DIR):
        subprocess.run(["git", "clone", "--depth", "1",
                        "https://github.com/tiantiaf0627/vox-profile-release.git", REPO_DIR], check=True)
    if REPO_DIR not in sys.path:
        sys.path.insert(0, REPO_DIR)

# %% [markdown]
# ## 2. Đọc & gộp nhãn (gộp theo wavID)
# - `train.csv`: mỗi dòng = 1 listener chấm 1 wav → gộp **theo wavID**:
#   EMOS=TB `eMOS` · VAL/ARO/DOM=TB `val/aro/dom` · CAT=**tỉ lệ vote 5 lớp** của `emoCat`.
# - `metadata.csv`: lấy **cảm xúc target** cho mỗi wav (để feed EMOS head).

# %%
import numpy as np
import pandas as pd

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

def _col(cols_map, *names, default_idx=None, df=None):
    for n in names:
        if n in cols_map:
            return cols_map[n]
    return list(df.columns)[default_idx] if default_idx is not None else None

def parse_emocat_votes(cell):
    """1 ô emoCat (có thể đa nhãn, vd 'happy;surprised') → vector đếm 5 lớp (chưa chuẩn hóa)."""
    v = np.zeros(len(EMOTIONS5), dtype=np.float32)
    for tok in str(cell).replace("/", ",").replace(";", ",").replace("|", ",").replace(" ", ",").split(","):
        e = norm_emotion(tok)
        if e in EMOTIONS5:
            v[EMOTIONS5.index(e)] += 1.0
    return v

def load_train_labels():
    """train.csv → DataFrame [wavID, emos, val, aro, dom, cat0..cat4] gộp theo wav.
    CAT = tỉ lệ vote 5 lớp (tổng=1); nếu wav không có vote hợp lệ → phân phối đều."""
    # train.csv phân tách bằng "|"; cột emoCat đa nhãn dùng "," bên trong (vd "Angry,Surprised").
    df = pd.read_csv(TRAIN_CSV, sep="|")
    cols = {c.lower().strip(): c for c in df.columns}
    wav_col  = _col(cols, "wavid", "wav", default_idx=1, df=df)
    emos_col = _col(cols, "emos", "emo", "emomos")
    val_col  = _col(cols, "val", "valence")
    aro_col  = _col(cols, "aro", "arousal")
    dom_col  = _col(cols, "dom", "dominance")
    cat_col  = _col(cols, "emocat", "cat", "emotion")
    assert emos_col, f"Không thấy cột eMOS trong train.csv (cột: {list(df.columns)})"

    df["_stem"] = df[wav_col].map(stem)
    rows = []
    for sid, g in df.groupby("_stem"):
        rec = {"wavID": sid, "emos": float(g[emos_col].mean())}
        rec["val"] = float(g[val_col].mean()) if val_col else np.nan
        rec["aro"] = float(g[aro_col].mean()) if aro_col else np.nan
        rec["dom"] = float(g[dom_col].mean()) if dom_col else np.nan
        votes = np.zeros(len(EMOTIONS5), dtype=np.float32)
        if cat_col:
            for cell in g[cat_col]:
                votes += parse_emocat_votes(cell)
        s = votes.sum()
        cat = votes / s if s > 0 else np.full(len(EMOTIONS5), 1.0 / len(EMOTIONS5), dtype=np.float32)
        for i in range(len(EMOTIONS5)):
            rec[f"cat{i}"] = float(cat[i])
        rows.append(rec)
    return pd.DataFrame(rows)

target_map = load_target_emotions()
train_df = load_train_labels()
HAS_VAD = bool(train_df["val"].notna().any())
print(f"Target emotions: {len(target_map)} | wav train (gộp): {len(train_df)} | có nhãn VAD: {HAS_VAD}")
print("eMOS:", train_df["emos"].describe()[["mean", "std", "min", "max"]].to_dict())
train_df.head()

# %% [markdown]
# ## 3. Trích đặc trưng 2 backbone (có cache riêng từng model)
# - **emotion2vec** → embedding + xác suất 5 lớp (như exp02).
# - **SAILER** → embedding (features) + xác suất 9 lớp + VAD3 (như exp03).
# Mỗi backbone cache riêng (`e2v_<tag>.npz`, `sailer_<tag>.npz`) → chạy nối tiếp được, đổi 1 backbone
# không phải trích lại cái kia. Trích xong **giải phóng GPU** rồi mới nạp backbone sau.

# %%
import torch
import torch.nn.functional as F

device = DEVICE if torch.cuda.is_available() else "cpu"
print("Device:", device)
if device == "cuda":
    print("  ✅ GPU:", torch.cuda.get_device_name(0))
else:
    print("  ⚠️ KHÔNG thấy GPU! Trích đặc trưng ~15k file trên CPU rất lâu.")
    print("     → Settings → Accelerator = GPU T4 rồi chạy lại.")

# ---- emotion2vec ----
def extract_e2v(stems, tag):
    """→ dict {stem: (emb[D1], probs5[5])}. Cache CACHE_DIR/e2v_<tag>.npz."""
    from tqdm.auto import tqdm
    cache_path = os.path.join(CACHE_DIR, f"e2v_{tag}.npz")
    store = {}
    if os.path.exists(cache_path):
        z = np.load(cache_path, allow_pickle=True)
        store = {k: z[k] for k in z.files}
        print(f"[e2v/{tag}] nạp cache: {len(store)}")
    todo = [s for s in stems if s not in store]
    if todo:
        from funasr import AutoModel
        m = AutoModel(model="iic/emotion2vec_plus_large", hub="hf", device=device)   # ép GPU
        miss = 0
        for i, s in enumerate(tqdm(todo, desc=f"e2v {tag}")):
            wav = os.path.join(WAV_DIR, s + ".wav")
            if not os.path.exists(wav):
                miss += 1; continue
            r = m.generate(wav, granularity="utterance", extract_embedding=True)[0]
            emb = np.asarray(r["feats"], dtype=np.float32).reshape(-1)
            probs = {e: 0.0 for e in EMOTIONS5}
            for lab, sc in zip(r["labels"], r["scores"]):
                name = lab.split("/")[-1]
                if name in probs:
                    probs[name] = float(sc)
            tot = sum(probs.values())
            p5 = np.array([probs[e] / tot if tot > 0 else 0.2 for e in EMOTIONS5], dtype=np.float32)
            store[s] = np.concatenate([emb, p5]).astype(np.float32)   # [D1 + 5]
            if (i + 1) % 500 == 0:
                np.savez(cache_path, **store)
        np.savez(cache_path, **store)
        del m
        torch.cuda.empty_cache() if device == "cuda" else None
        if miss:
            print(f"[e2v/{tag}] {miss} file thiếu → bỏ qua.")
    return {s: (v[:-5], v[-5:]) for s, v in store.items()}

# ---- SAILER ----
def _pool_feat(features):
    """features (tensor) → vector 1 chiều (mean-pool nếu còn chiều thời gian)."""
    f = features.detach().cpu().numpy()
    if f.ndim <= 1:
        return f.reshape(-1).astype(np.float32)
    return f.mean(axis=tuple(range(f.ndim - 1))).reshape(-1).astype(np.float32)

def extract_sailer(stems, tag):
    """→ dict {stem: (emb[D2], probs9[9], vad3[3] thang 1–5)}. Cache CACHE_DIR/sailer_<tag>.npz.
    Mỗi mẫu lưu vector [emb | probs9(9) | vad3(3)] → cắt lại khi nạp."""
    import librosa
    from tqdm.auto import tqdm
    cache_path = os.path.join(CACHE_DIR, f"sailer_{tag}.npz")
    store = {}
    if os.path.exists(cache_path):
        z = np.load(cache_path, allow_pickle=True)
        store = {k: z[k] for k in z.files}
        print(f"[sailer/{tag}] nạp cache: {len(store)}")
    todo = [s for s in stems if s not in store]
    if todo:
        from src.model.emotion.wavlm_emotion import WavLMWrapper
        sailer = WavLMWrapper.from_pretrained("tiantiaf/wavlm-large-categorical-emotion").to(device).eval()
        miss = 0
        with torch.no_grad():
            for i, s in enumerate(tqdm(todo, desc=f"sailer {tag}")):
                wav = os.path.join(WAV_DIR, s + ".wav")
                if not os.path.exists(wav):
                    miss += 1; continue
                wave, _ = librosa.load(wav, sr=16000, mono=True)
                wave = wave[: 15 * 16000]
                data = torch.from_numpy(wave).float().unsqueeze(0).to(device)
                logits, feat, _det, arousal, valence, dominance = sailer(data, return_feature=True)
                emb = _pool_feat(feat)
                p9 = F.softmax(logits, dim=1)[0].detach().cpu().numpy().astype(np.float32)
                vad3 = np.array([1 + 4 * float(valence.item()),
                                 1 + 4 * float(arousal.item()),
                                 1 + 4 * float(dominance.item())], dtype=np.float32)  # [VAL,ARO,DOM]
                store[s] = np.concatenate([emb, p9, vad3]).astype(np.float32)   # [D2 + 9 + 3]
                if (i + 1) % 500 == 0:
                    np.savez(cache_path, **store)
        np.savez(cache_path, **store)
        del sailer
        torch.cuda.empty_cache() if device == "cuda" else None
        if miss:
            print(f"[sailer/{tag}] {miss} file thiếu → bỏ qua.")
    return {s: (v[:-12], v[-12:-3], v[-3:]) for s, v in store.items()}

# %% [markdown]
# ## 4. Dựng feature + nhãn cho train
# Feature audio (KHÔNG gồm target) = nối các phần đang bật:
# `[e2v_emb | e2v_probs5 | sailer_emb | sailer_probs9 | sailer_vad3]`.
# One-hot target để **riêng** (chỉ EMOS head dùng). Bỏ wav thiếu feature.

# %%
train_stems = list(train_df["wavID"])
if LIMIT_TRAIN:
    train_stems = train_stems[:LIMIT_TRAIN]

e2v_tr    = extract_e2v(train_stems, "train")    if USE_E2V    else {}
sailer_tr = extract_sailer(train_stems, "train") if USE_SAILER else {}

def audio_feature(sid, e2v_map, sailer_map):
    """Nối đặc trưng audio cho 1 wav. None nếu thiếu phần bắt buộc."""
    parts = []
    if USE_E2V:
        pk = e2v_map.get(sid)
        if pk is None:
            return None
        emb, p5 = pk
        parts.append(emb)
        if USE_CLASSPROB:
            parts.append(p5)
    if USE_SAILER:
        pk = sailer_map.get(sid)
        if pk is None:
            return None
        emb, p9, vad3 = pk
        parts.append(emb)
        if USE_CLASSPROB:
            parts.append(p9); parts.append(vad3)
    return np.concatenate(parts).astype(np.float32)

def onehot_target(tgt):
    v = np.zeros(len(EMOTIONS5), dtype=np.float32)
    if tgt in EMOTIONS5:
        v[EMOTIONS5.index(tgt)] = 1.0
    return v

lab = train_df.set_index("wavID")
X, T, y_emos, y_vad, y_cat = [], [], [], [], []
for s in train_stems:
    f = audio_feature(s, e2v_tr, sailer_tr)
    tgt = target_map.get(s)
    if f is None or tgt is None or s not in lab.index:
        continue
    X.append(f)
    T.append(onehot_target(tgt))
    y_emos.append(lab.loc[s, "emos"])
    y_vad.append([lab.loc[s, "val"], lab.loc[s, "aro"], lab.loc[s, "dom"]])
    y_cat.append([lab.loc[s, f"cat{i}"] for i in range(len(EMOTIONS5))])

X = np.stack(X).astype(np.float32)
T = np.stack(T).astype(np.float32)
y_emos = np.array(y_emos, dtype=np.float32)
y_vad  = np.array(y_vad,  dtype=np.float32)         # [N,3] (VAL,ARO,DOM) — có thể toàn NaN nếu thiếu nhãn
y_cat  = np.array(y_cat,  dtype=np.float32)         # [N,5] phân phối tổng=1
FEAT_DIM = X.shape[1]
print(f"Train: X={X.shape} target={T.shape} emos={y_emos.shape} vad={y_vad.shape} cat={y_cat.shape}")

# Chuẩn hóa feature audio (z-score) — lưu mean/std để áp dụng y hệt lúc dự đoán DEV.
feat_mean = X.mean(0, keepdims=True)
feat_std  = X.std(0, keepdims=True) + 1e-6
Xn = (X - feat_mean) / feat_std

# Chuẩn hóa nhãn liên tục (eMOS, VAD) về z-score → các MSE cùng thang (uncertainty weighting ổn định hơn).
# SRCC bất biến với scale → khi xuất answer.txt chỉ cần đảo z-score về thang gốc cho đẹp.
emos_mu, emos_sd = float(y_emos.mean()), float(y_emos.std() + 1e-6)
y_emos_z = (y_emos - emos_mu) / emos_sd
if HAS_VAD:
    vad_mu = np.nanmean(y_vad, axis=0)
    vad_sd = np.nanstd(y_vad, axis=0) + 1e-6
    y_vad_z = (y_vad - vad_mu) / vad_sd
else:
    vad_mu = np.zeros(3, dtype=np.float32); vad_sd = np.ones(3, dtype=np.float32)
    y_vad_z = np.zeros_like(y_vad)

# %% [markdown]
# ## 5. Model fusion multi-task + train loop
# - **Trunk** chung: `Linear(FEAT_DIM→TRUNK_HIDDEN)+ReLU+Dropout` (×2).
# - **EMOS head**: nối `[trunk | one-hot target]` → MLP → 1 (vì EMOS phụ thuộc target).
# - **CAT head**: trunk → 5 logits → softmax (dự đoán phân phối vote). Loss = soft-CE (KL).
# - **VAD head**: trunk → 3 (VAL/ARO/DOM). Loss = MSE (bỏ qua nếu thiếu nhãn VAD).
# - **Cân loss**: uncertainty weighting — tổng `Σ exp(-sᵢ)·Lᵢ + sᵢ`, `sᵢ=log σᵢ²` **học được**.

# %%
import torch.nn as nn
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split

torch.manual_seed(SEED); np.random.seed(SEED)
N_EMO = len(EMOTIONS5)

idx_all = np.arange(X.shape[0])
tr_idx, va_idx = train_test_split(idx_all, test_size=VAL_FRAC, random_state=SEED)

def to_t(a):
    return torch.tensor(a, dtype=torch.float32, device=device)

Xn_t, T_t = to_t(Xn), to_t(T)
emos_t = to_t(y_emos_z).unsqueeze(1)
vad_t  = to_t(y_vad_z)
cat_t  = to_t(y_cat)

class FusionMTL(nn.Module):
    def __init__(self, d_in, trunk_h, head_h, p, n_emo):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(d_in, trunk_h), nn.ReLU(), nn.Dropout(p),
            nn.Linear(trunk_h, trunk_h), nn.ReLU(), nn.Dropout(p),
        )
        self.emos = nn.Sequential(   # nhận [trunk | target]
            nn.Linear(trunk_h + n_emo, head_h), nn.ReLU(), nn.Dropout(p), nn.Linear(head_h, 1))
        self.cat  = nn.Sequential(
            nn.Linear(trunk_h, head_h), nn.ReLU(), nn.Dropout(p), nn.Linear(head_h, n_emo))
        self.vad  = nn.Sequential(
            nn.Linear(trunk_h, head_h), nn.ReLU(), nn.Dropout(p), nn.Linear(head_h, 3))

    def forward(self, x, tgt):
        h = self.trunk(x)
        emos = self.emos(torch.cat([h, tgt], dim=1))
        cat_logits = self.cat(h)
        vad = self.vad(h)
        return emos, cat_logits, vad

model = FusionMTL(FEAT_DIM, TRUNK_HIDDEN, HEAD_HIDDEN, DROPOUT, N_EMO).to(device)

# Trọng số bất định (log σ²) cho 5 task: emos, cat, val, aro, dom.
TASKS = ["emos", "cat", "val", "aro", "dom"]
log_var = nn.Parameter(torch.zeros(len(TASKS), device=device))
params = list(model.parameters()) + ([log_var] if USE_UNCERTAINTY else [])
opt = torch.optim.Adam(params, lr=LR, weight_decay=1e-5)

mse = nn.MSELoss(reduction="none")

def soft_ce(logits, target_dist):
    """Cross-entropy với nhãn mềm (phân phối): −Σ p·log q."""
    logq = F.log_softmax(logits, dim=1)
    return -(target_dist * logq).sum(dim=1)

def task_losses(emos_p, cat_logits, vad_p, b):
    """Trả về dict loss TB từng task cho 1 batch (chỉ số b)."""
    L = {}
    L["emos"] = mse(emos_p, emos_t[b]).mean()
    L["cat"]  = soft_ce(cat_logits, cat_t[b]).mean()
    if HAS_VAD:
        L["val"] = mse(vad_p[:, 0:1], vad_t[b, 0:1]).mean()
        L["aro"] = mse(vad_p[:, 1:2], vad_t[b, 1:2]).mean()
        L["dom"] = mse(vad_p[:, 2:3], vad_t[b, 2:3]).mean()
    else:
        z = torch.zeros((), device=device)
        L["val"] = L["aro"] = L["dom"] = z
    return L

def combine(L):
    """Gộp 5 loss thành 1 số: uncertainty weighting hoặc trọng số cố định."""
    if USE_UNCERTAINTY:
        tot = 0.0
        for i, t in enumerate(TASKS):
            tot = tot + torch.exp(-log_var[i]) * L[t] + log_var[i]
        return tot
    return sum(LOSS_W[t] * L[t] for t in TASKS)

@torch.no_grad()
def eval_val():
    """SRCC từng task trên tập val nội bộ (CAT báo bằng −KL để 'cao=tốt' cho early-stop)."""
    model.eval()
    ep, cl, vp = model(Xn_t[va_idx], T_t[va_idx])
    ep = ep.cpu().numpy().ravel()
    out = {"emos": spearmanr(ep, y_emos[va_idx]).correlation}
    if HAS_VAD:
        vp = vp.cpu().numpy()
        for j, t in enumerate(["val", "aro", "dom"]):
            out[t] = spearmanr(vp[:, j], y_vad[va_idx, j]).correlation
    # CAT: dùng −KL(p‖q) trung bình (càng gần 0 càng tốt) → đổi dấu để hợp early-stop
    q = F.softmax(cl, dim=1).cpu().numpy()
    p = y_cat[va_idx]
    kl = (p * (np.log(p + 1e-9) - np.log(q + 1e-9))).sum(1).mean()
    out["cat_negkl"] = float(-kl)
    return out

def val_score(m):
    """Điểm tổng để early-stop = TB SRCC các task liên tục có nhãn."""
    keys = ["emos"] + (["val", "aro", "dom"] if HAS_VAD else [])
    return float(np.mean([m[k] for k in keys]))

best_score, best_state, bad = -1e9, None, 0
tr_t = torch.tensor(tr_idx, device=device)
for ep in range(1, EPOCHS + 1):
    model.train()
    perm = tr_t[torch.randperm(len(tr_t), device=device)]
    run = 0.0
    for i in range(0, len(perm), BATCH):
        b = perm[i:i + BATCH]
        opt.zero_grad()
        emos_p, cat_logits, vad_p = model(Xn_t[b], T_t[b])
        L = task_losses(emos_p, cat_logits, vad_p, b)
        loss = combine(L)
        loss.backward(); opt.step()
        run += loss.item() * len(b)
    m = eval_val()
    sc = val_score(m)
    if sc > best_score:
        best_score = sc
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        bad = 0
    else:
        bad += 1
    if ep % 5 == 0 or ep == 1:
        msg = " ".join(f"{k}={m[k]:.3f}" for k in m)
        print(f"epoch {ep:3d} | loss {run/len(perm):.4f} | {msg} | best {best_score:.4f}")
    if bad >= PATIENCE:
        print(f"Early stop ở epoch {ep}.")
        break

model.load_state_dict(best_state)
final = eval_val()
print("\n✅ VAL (nội bộ) tốt nhất:")
print(f"   EMOS SRCC = {final['emos']:.4f}   (so mốc exp01 emotion2vec = 0.637)")
if HAS_VAD:
    print(f"   VAL/ARO/DOM SRCC = {final['val']:.4f} / {final['aro']:.4f} / {final['dom']:.4f}"
          f"   (so mốc SAILER = 0.341 / 0.712 / 0.630)")
if USE_UNCERTAINTY:
    print("   log σ² mỗi task:", {t: round(float(log_var[i]), 3) for i, t in enumerate(TASKS)})

# Lưu model + tham số chuẩn hóa.
torch.save({"state": best_state, "feat_mean": feat_mean, "feat_std": feat_std,
            "emos_mu": emos_mu, "emos_sd": emos_sd, "vad_mu": vad_mu, "vad_sd": vad_sd,
            "FEAT_DIM": FEAT_DIM, "EMOTIONS5": EMOTIONS5, "HAS_VAD": HAS_VAD,
            "USE_E2V": USE_E2V, "USE_SAILER": USE_SAILER, "USE_CLASSPROB": USE_CLASSPROB,
            "TRUNK_HIDDEN": TRUNK_HIDDEN, "HEAD_HIDDEN": HEAD_HIDDEN, "val_score": best_score},
           os.path.join(OUT_DIR, "fusion_mtl.pt"))
print("Đã lưu", os.path.join(OUT_DIR, "fusion_mtl.pt"))

# %% [markdown]
# ## 6. Dự đoán DEV → `answer.txt` đầy đủ 7 cột
# - **EMOS/CAT/VAD** = model fusion (đảo z-score về thang gốc cho EMOS/VAD; CAT = softmax 5 lớp).
# - **QMOS** = SpeechMOS (UTMOS) — để riêng, đúng thiết kế.

# %%
def list_dev():
    with open(DEV_SCP) as f:
        return [ln.strip() for ln in f if ln.strip()]   # tên file .wav

dev_names = list_dev()
if LIMIT_DEV:
    dev_names = dev_names[:LIMIT_DEV]
dev_stems = [stem(n) for n in dev_names]
print("DEV:", len(dev_names), "mẫu")

# 6a. Trích đặc trưng 2 backbone cho DEV (cache riêng)
e2v_dev    = extract_e2v(dev_stems, "dev")    if USE_E2V    else {}
sailer_dev = extract_sailer(dev_stems, "dev") if USE_SAILER else {}

# 6b. Dự đoán 5 cột cảm xúc bằng model fusion
@torch.no_grad()
def predict_emotion(sid):
    f = audio_feature(sid, e2v_dev, sailer_dev)
    if f is None:
        return None
    fn = (f[None, :] - feat_mean) / feat_std
    tgt = onehot_target(target_map.get(sid))[None, :]
    model.eval()
    emos_p, cat_logits, vad_p = model(to_t(fn), to_t(tgt))
    emos = float(emos_p.item()) * emos_sd + emos_mu                      # đảo z-score
    cat5 = F.softmax(cat_logits, dim=1)[0].cpu().numpy()
    vad3 = vad_p[0].cpu().numpy() * vad_sd + vad_mu                      # [VAL,ARO,DOM]
    return emos, cat5, vad3

# 6c. QMOS = SpeechMOS (để riêng)
@torch.no_grad()
def run_qmos(names):
    import librosa
    from tqdm.auto import tqdm
    predictor = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True).to(device).eval()
    out = {}
    for n in tqdm(names, desc="QMOS"):
        p = os.path.join(WAV_DIR, n)
        if not os.path.exists(p):
            continue
        wave, _ = librosa.load(p, sr=16000, mono=True)
        out[n] = float(predictor(torch.from_numpy(wave).unsqueeze(0).to(device), sr=16000).mean().item())
    return out

qmos_scores = run_qmos(dev_names)

# %%
def fmt_cat(probs5):
    return "|".join(f"{e}:{probs5[i]:.6g}" for i, e in enumerate(EMOTIONS5))

def build_answer(out_path):
    from tqdm.auto import tqdm
    n_real = n_default = 0
    with open(out_path, "w") as f:
        f.write("wav,QMOS,EMOS,CAT,VAL,ARO,DOM\n")
        for name in tqdm(dev_names, desc="answer"):
            sid = stem(name)
            pred = predict_emotion(sid)
            if pred is None:
                emos, cat5, vad3 = 3.0, np.full(5, 0.2, np.float32), np.array([3.0, 3.0, 3.0])
                n_default += 1
            else:
                emos, cat5, vad3 = pred
                n_real += 1
            qmos = qmos_scores.get(name, 3.0)
            f.write(f"{name},{qmos:.6g},{emos:.6g},{fmt_cat(cat5)},"
                    f"{vad3[0]:.6g},{vad3[1]:.6g},{vad3[2]:.6g}\n")
    print(f"Ghi {len(dev_names)} dòng → {out_path} | fusion thật {n_real}, mặc định {n_default}")

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
os.system(f"cd {OUT_DIR} && zip -j submission_track2_exp04_fusion.zip answer.txt && unzip -l submission_track2_exp04_fusion.zip")
print("Sẵn sàng nộp:", os.path.join(OUT_DIR, "submission_track2_exp04_fusion.zip"))

# %% [markdown]
# ## Ghi chú
# - **Lần đầu**: đặt `LIMIT_TRAIN=300`, `LIMIT_DEV=20` ở cell 0 để bắt lỗi setup (clone repo / import / model).
#   Chạy OK rồi đặt `None` chạy full.
# - **VAL SRCC** ở mục 5 là ước lượng nội bộ (10% train) → so mốc EMOS 0.637 / ARO 0.712. Điểm DEV thật
#   phải nộp CodaBench mới biết (My Submissions → Track 2, bỏ chọn track khác).
# - Embedding đã cache trong `/kaggle/working/fusion_cache/` → **Save Version** để giữ; lần sau đổi
#   siêu tham số/đổi cách cân loss chỉ train lại head (vài phút), khỏi trích lại.
# - **Ablation cho paper** (đổi cờ ở cell 0, train lại head):
#   `USE_E2V=False` (chỉ SAILER) · `USE_SAILER=False` (chỉ emotion2vec) · `USE_UNCERTAINTY=False` (trọng số tay)
#   · `USE_CLASSPROB=False` (chỉ embedding) → điền bảng ablation `docs/04_experiments_log.md`.
# - License SAILER = **Open RAIL (phi thương mại)** → nhắc trong `docs/12_system_description.md`.
# - Nhớ ghi config → kết quả → nhận xét vào `docs/04_experiments_log.md` (mục exp04).
