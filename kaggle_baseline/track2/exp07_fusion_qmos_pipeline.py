# %% [markdown]
# # VMC2026 Track 2 — exp07 (FUSION + QMOS head, HỢP NHẤT 6 cột) — Kaggle
#
# **Khác exp04 ở đâu:** exp04 để **QMOS riêng** (UTMOS zero-shot). exp07 **gộp luôn QMOS vào trunk chung**
# → 1 model multi-task dự đoán **đủ 6 đầu ra**: QMOS · EMOS · CAT · VAL · ARO · DOM.
#
# ## Giả thuyết (của bạn) cần kiểm chứng
# "Chất giọng tự nhiên có liên quan tới cảm nhận cảm xúc" → nếu đúng, QMOS sẽ **hưởng lợi** từ biểu diễn
# cảm xúc chung (emotion2vec + SAILER). **Rủi ro:** 2 backbone này chuyên *cảm xúc*, chưa chắc bắt tốt
# *lỗi chất lượng/artifact* (thứ UTMOS chuyên trị) → QMOS có thể **thua** UTMOS, hoặc gộp làm **tụt** EMOS/VAD.
#
# ## Lưới an toàn trong thiết kế
# - **Vẫn đưa điểm UTMOS làm 1 đầu vào** cho QMOS head (`USE_UTMOS_FEAT`) → head học **chỉnh sửa** quanh
#   0.414 thay vì học lại từ đầu → khó tệ hơn UTMOS.
# - **In SRCC cả 6 cột + so mốc exp04** (EMOS 0.788 · CAT err 0.145 · VAL 0.578 · ARO 0.754 · DOM 0.706)
#   → cảnh báo ngay nếu gộp QMOS làm tụt 5 cột cảm xúc.
# - **File riêng**, KHÔNG đụng `exp04_fusion_pipeline.py` (exp04 vẫn nguyên).
#
# ```
#  mỗi wav ─► [e2v_emb | e2v_p5 | sailer_emb | sailer_p9 | sailer_vad3] ─► TRUNK chung
#                                                                            │
#         ┌──────────────┬───────────────┬─────────────┬───────────────────┤
#   [QMOS head]      [EMOS head]      [CAT head]    [VAD head]
#   trunk + UTMOS    trunk + target    trunk         trunk
# ```
#
# **Cách chạy:** GPU T4 + Internet On → Add Input dataset Track 2 → sửa `DATA_ROOT` → Run All.
# Lần đầu đặt `LIMIT_TRAIN=300`, `LIMIT_DEV=20`. Dùng CHUNG cache `fusion_cache/` với exp04 (thêm `utmos_*.npz`).

# %% [markdown]
# ## 0. Cấu hình — SỬA Ở ĐÂY

# %%
import os

DATA_ROOT    = "/kaggle/input/vmc2026-track2-full/vmc2026-track2"   # << SỬA slug
WAV_DIR      = f"{DATA_ROOT}/wav"
METADATA_CSV = f"{DATA_ROOT}/metadata.csv"     # wavID|emotion|transcript (KHÔNG header)
TRAIN_CSV    = f"{DATA_ROOT}/sets/train.csv"   # lisID|wavID|qMOS|emoCat|eMOS|val|dom|aro
DEV_SCP      = f"{DATA_ROOT}/sets/dev.scp"

OUT_DIR   = "/kaggle/working"
CACHE_DIR = "/kaggle/working/fusion_cache"     # dùng CHUNG với exp04 (thêm utmos_*.npz)
os.makedirs(CACHE_DIR, exist_ok=True)

# ── Siêu tham số ─────────────────────────────────────────────────────────────
DEVICE          = "cuda"
TRUNK_HIDDEN    = 512
HEAD_HIDDEN     = 128
DROPOUT         = 0.3
LR              = 1e-3
EPOCHS          = 80
BATCH           = 64
VAL_FRAC        = 0.10
PATIENCE        = 15
SEED            = 42

USE_UNCERTAINTY = True        # tự cân 6 loss (Kendall); False = dùng LOSS_W cố định
LOSS_W          = {"qmos": 1.0, "emos": 1.0, "cat": 1.0, "val": 1.0, "aro": 1.0, "dom": 1.0}
USE_E2V         = True
USE_SAILER      = True
USE_CLASSPROB   = True
USE_UTMOS_FEAT  = True        # đưa điểm UTMOS làm đầu vào QMOS head (neo residual quanh 0.414)

LIMIT_TRAIN     = None
LIMIT_DEV       = None

# Mốc exp04 để so (cảnh báo nếu tụt khi gộp QMOS)
EXP04 = {"emos": 0.788, "cat_err": 0.145, "val": 0.578, "aro": 0.754, "dom": 0.706, "qmos_utmos": 0.414}

EMOTIONS5 = ["angry", "happy", "neutral", "sad", "surprised"]
SAILER9 = ["Anger", "Contempt", "Disgust", "Fear", "Happiness", "Neutral", "Sadness", "Surprise", "Other"]
EMO2SAILER = {"angry": 0, "happy": 4, "neutral": 5, "sad": 6, "surprised": 7}

_EMO_ALIAS = {
    "angry": "angry", "anger": "angry",
    "happy": "happy", "happiness": "happy", "joy": "happy",
    "neutral": "neutral", "calm": "neutral",
    "sad": "sad", "sadness": "sad",
    "surprise": "surprised", "surprised": "surprised", "surprising": "surprised",
}

def norm_emotion(label):
    key = str(label).strip().lower()
    return _EMO_ALIAS.get(key, key if key in EMOTIONS5 else None)

def stem(p):
    return os.path.splitext(os.path.basename(str(p)))[0]

assert USE_E2V or USE_SAILER, "Phải bật ít nhất 1 backbone."
print("DATA_ROOT:", DATA_ROOT)
for p in [WAV_DIR, METADATA_CSV, TRAIN_CSV, DEV_SCP]:
    print(("  ✅ " if os.path.exists(p) else "  ❌ THIẾU ") + p)

# %% [markdown]
# ## 1. Cài đặt + tải code SAILER

# %%
import sys, subprocess

def pip_install(*pkgs):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], check=True)

pip_install("speechmos", "funasr", "librosa", "soundfile", "pandas", "scipy", "scikit-learn", "tqdm")

if USE_SAILER:
    pip_install("loralib", "speechbrain")
    REPO_DIR = "/kaggle/working/vox-profile-release"
    if not os.path.exists(REPO_DIR):
        subprocess.run(["git", "clone", "--depth", "1",
                        "https://github.com/tiantiaf0627/vox-profile-release.git", REPO_DIR], check=True)
    if REPO_DIR not in sys.path:
        sys.path.insert(0, REPO_DIR)

# %% [markdown]
# ## 2. Đọc & gộp nhãn (gộp theo wavID) — THÊM cột qMOS
# Khác exp04: gộp thêm **qMOS** (= TB `qMOS` theo wav) làm nhãn cho QMOS head.

# %%
import numpy as np
import pandas as pd

def load_target_emotions():
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
    v = np.zeros(len(EMOTIONS5), dtype=np.float32)
    for tok in str(cell).replace("/", ",").replace(";", ",").replace("|", ",").replace(" ", ",").split(","):
        e = norm_emotion(tok)
        if e in EMOTIONS5:
            v[EMOTIONS5.index(e)] += 1.0
    return v

def load_train_labels():
    """train.csv → DataFrame [wavID, qmos, emos, val, aro, dom, cat0..cat4] gộp theo wav."""
    df = pd.read_csv(TRAIN_CSV, sep="|")
    cols = {c.lower().strip(): c for c in df.columns}
    wav_col  = _col(cols, "wavid", "wav", default_idx=1, df=df)
    qmos_col = _col(cols, "qmos", "mos")
    emos_col = _col(cols, "emos", "emo", "emomos")
    val_col  = _col(cols, "val", "valence")
    aro_col  = _col(cols, "aro", "arousal")
    dom_col  = _col(cols, "dom", "dominance")
    cat_col  = _col(cols, "emocat", "cat", "emotion")
    assert qmos_col, f"Không thấy cột qMOS trong train.csv (cột: {list(df.columns)})"
    assert emos_col, f"Không thấy cột eMOS trong train.csv (cột: {list(df.columns)})"

    df["_stem"] = df[wav_col].map(stem)
    rows = []
    for sid, g in df.groupby("_stem"):
        rec = {"wavID": sid,
               "qmos": float(g[qmos_col].mean()),
               "emos": float(g[emos_col].mean())}
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
print(f"Target: {len(target_map)} | wav train (gộp): {len(train_df)} | có VAD: {HAS_VAD}")
print("qMOS:", train_df["qmos"].describe()[["mean", "std", "min", "max"]].to_dict())
print("eMOS:", train_df["emos"].describe()[["mean", "std", "min", "max"]].to_dict())
train_df.head()

# %% [markdown]
# ## 3. Trích đặc trưng 2 backbone + điểm UTMOS (cache CHUNG với exp04)

# %%
import torch
import torch.nn.functional as F

device = DEVICE if torch.cuda.is_available() else "cpu"
print("Device:", device, ("✅ " + torch.cuda.get_device_name(0)) if device == "cuda" else "⚠️ CPU")

def extract_e2v(stems, tag):
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
        m = AutoModel(model="iic/emotion2vec_plus_large", hub="hf", device=device)
        for i, s in enumerate(tqdm(todo, desc=f"e2v {tag}")):
            wav = os.path.join(WAV_DIR, s + ".wav")
            if not os.path.exists(wav):
                continue
            r = m.generate(wav, granularity="utterance", extract_embedding=True)[0]
            emb = np.asarray(r["feats"], dtype=np.float32).reshape(-1)
            probs = {e: 0.0 for e in EMOTIONS5}
            for lab, sc in zip(r["labels"], r["scores"]):
                name = lab.split("/")[-1]
                if name in probs:
                    probs[name] = float(sc)
            tot = sum(probs.values())
            p5 = np.array([probs[e] / tot if tot > 0 else 0.2 for e in EMOTIONS5], dtype=np.float32)
            store[s] = np.concatenate([emb, p5]).astype(np.float32)
            if (i + 1) % 500 == 0:
                np.savez(cache_path, **store)
        np.savez(cache_path, **store)
        del m
        torch.cuda.empty_cache() if device == "cuda" else None
    return {s: (v[:-5], v[-5:]) for s, v in store.items()}

def _pool_feat(features):
    f = features.detach().cpu().numpy()
    if f.ndim <= 1:
        return f.reshape(-1).astype(np.float32)
    return f.mean(axis=tuple(range(f.ndim - 1))).reshape(-1).astype(np.float32)

def extract_sailer(stems, tag):
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
        with torch.no_grad():
            for i, s in enumerate(tqdm(todo, desc=f"sailer {tag}")):
                wav = os.path.join(WAV_DIR, s + ".wav")
                if not os.path.exists(wav):
                    continue
                wave, _ = librosa.load(wav, sr=16000, mono=True)
                wave = wave[: 15 * 16000]
                data = torch.from_numpy(wave).float().unsqueeze(0).to(device)
                logits, feat, _det, arousal, valence, dominance = sailer(data, return_feature=True)
                emb = _pool_feat(feat)
                p9 = F.softmax(logits, dim=1)[0].detach().cpu().numpy().astype(np.float32)
                vad3 = np.array([1 + 4 * float(valence.item()),
                                 1 + 4 * float(arousal.item()),
                                 1 + 4 * float(dominance.item())], dtype=np.float32)
                store[s] = np.concatenate([emb, p9, vad3]).astype(np.float32)
                if (i + 1) % 500 == 0:
                    np.savez(cache_path, **store)
        np.savez(cache_path, **store)
        del sailer
        torch.cuda.empty_cache() if device == "cuda" else None
    return {s: (v[:-12], v[-12:-3], v[-3:]) for s, v in store.items()}

def extract_utmos(names, tag):
    """Chấm UTMOS từng wav (theo TÊN, vì DEV gọi .wav theo tên). → dict {stem: score}.
    Cache CACHE_DIR/utmos_<tag>.npz. Dùng vừa làm đầu vào QMOS head, vừa làm baseline so sánh."""
    import librosa
    from tqdm.auto import tqdm
    cache_path = os.path.join(CACHE_DIR, f"utmos_{tag}.npz")
    store = {}
    if os.path.exists(cache_path):
        z = np.load(cache_path, allow_pickle=True)
        store = {k: float(z[k]) for k in z.files}
        print(f"[utmos/{tag}] nạp cache: {len(store)}")
    todo = [n for n in names if stem(n) not in store]
    if todo:
        predictor = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong",
                                   trust_repo=True).to(device).eval()
        with torch.no_grad():
            for i, n in enumerate(tqdm(todo, desc=f"utmos {tag}")):
                wav = os.path.join(WAV_DIR, n if str(n).endswith(".wav") else n + ".wav")
                if not os.path.exists(wav):
                    continue
                wave, _ = librosa.load(wav, sr=16000, mono=True)
                store[stem(n)] = float(predictor(torch.from_numpy(wave).unsqueeze(0).to(device),
                                                 sr=16000).mean().item())
                if (i + 1) % 500 == 0:
                    np.savez(cache_path, **{k: np.float32(v) for k, v in store.items()})
        np.savez(cache_path, **{k: np.float32(v) for k, v in store.items()})
        del predictor
        torch.cuda.empty_cache() if device == "cuda" else None
    return store

# %% [markdown]
# ## 4. Dựng feature + nhãn cho train
# Feature audio (cảm xúc) = `[e2v_emb | e2v_p5 | sailer_emb | sailer_p9 | sailer_vad3]` (như exp04).
# Thêm: vector **UTMOS** (1 số/ wav) cho QMOS head, và nhãn **qMOS**.

# %%
train_stems = list(train_df["wavID"])
if LIMIT_TRAIN:
    train_stems = train_stems[:LIMIT_TRAIN]

e2v_tr    = extract_e2v(train_stems, "train")    if USE_E2V    else {}
sailer_tr = extract_sailer(train_stems, "train") if USE_SAILER else {}
utmos_tr  = extract_utmos(train_stems, "train")  if USE_UTMOS_FEAT else {}

def audio_feature(sid, e2v_map, sailer_map):
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
X, T, U, y_qmos, y_emos, y_vad, y_cat = [], [], [], [], [], [], []
for s in train_stems:
    f = audio_feature(s, e2v_tr, sailer_tr)
    tgt = target_map.get(s)
    if f is None or tgt is None or s not in lab.index:
        continue
    if USE_UTMOS_FEAT and s not in utmos_tr:
        continue
    X.append(f)
    T.append(onehot_target(tgt))
    U.append(utmos_tr.get(s, 3.0) if USE_UTMOS_FEAT else 0.0)
    y_qmos.append(lab.loc[s, "qmos"])
    y_emos.append(lab.loc[s, "emos"])
    y_vad.append([lab.loc[s, "val"], lab.loc[s, "aro"], lab.loc[s, "dom"]])
    y_cat.append([lab.loc[s, f"cat{i}"] for i in range(len(EMOTIONS5))])

X = np.stack(X).astype(np.float32)
T = np.stack(T).astype(np.float32)
U = np.array(U, dtype=np.float32).reshape(-1, 1)
y_qmos = np.array(y_qmos, dtype=np.float32)
y_emos = np.array(y_emos, dtype=np.float32)
y_vad  = np.array(y_vad,  dtype=np.float32)
y_cat  = np.array(y_cat,  dtype=np.float32)
FEAT_DIM = X.shape[1]
print(f"Train: X={X.shape} U={U.shape} qmos={y_qmos.shape} emos={y_emos.shape} vad={y_vad.shape}")

# Chuẩn hóa feature audio + UTMOS (z-score), lưu mean/std.
feat_mean = X.mean(0, keepdims=True); feat_std = X.std(0, keepdims=True) + 1e-6
Xn = (X - feat_mean) / feat_std
u_mu, u_sd = float(U.mean()), float(U.std() + 1e-6)
Un = (U - u_mu) / u_sd

# Chuẩn hóa nhãn liên tục về z-score.
qmos_mu, qmos_sd = float(y_qmos.mean()), float(y_qmos.std() + 1e-6)
y_qmos_z = (y_qmos - qmos_mu) / qmos_sd
emos_mu, emos_sd = float(y_emos.mean()), float(y_emos.std() + 1e-6)
y_emos_z = (y_emos - emos_mu) / emos_sd
if HAS_VAD:
    vad_mu = np.nanmean(y_vad, axis=0); vad_sd = np.nanstd(y_vad, axis=0) + 1e-6
    y_vad_z = (y_vad - vad_mu) / vad_sd
else:
    vad_mu = np.zeros(3, dtype=np.float32); vad_sd = np.ones(3, dtype=np.float32)
    y_vad_z = np.zeros_like(y_vad)

# %% [markdown]
# ## 5. Model fusion multi-task (6 head) + train loop
# Thêm so exp04: **QMOS head** nhận `[trunk | UTMOS]` → 1; `qmos` vào uncertainty weighting (6 task).

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

Xn_t, T_t, Un_t = to_t(Xn), to_t(T), to_t(Un)
qmos_t = to_t(y_qmos_z).unsqueeze(1)
emos_t = to_t(y_emos_z).unsqueeze(1)
vad_t  = to_t(y_vad_z)
cat_t  = to_t(y_cat)

class FusionMTL6(nn.Module):
    def __init__(self, d_in, trunk_h, head_h, p, n_emo, use_utmos):
        super().__init__()
        self.use_utmos = use_utmos
        self.trunk = nn.Sequential(
            nn.Linear(d_in, trunk_h), nn.ReLU(), nn.Dropout(p),
            nn.Linear(trunk_h, trunk_h), nn.ReLU(), nn.Dropout(p),
        )
        self.qmos = nn.Sequential(   # nhận [trunk | utmos] nếu bật
            nn.Linear(trunk_h + (1 if use_utmos else 0), head_h), nn.ReLU(), nn.Dropout(p),
            nn.Linear(head_h, 1))
        self.emos = nn.Sequential(   # nhận [trunk | target]
            nn.Linear(trunk_h + n_emo, head_h), nn.ReLU(), nn.Dropout(p), nn.Linear(head_h, 1))
        self.cat  = nn.Sequential(
            nn.Linear(trunk_h, head_h), nn.ReLU(), nn.Dropout(p), nn.Linear(head_h, n_emo))
        self.vad  = nn.Sequential(
            nn.Linear(trunk_h, head_h), nn.ReLU(), nn.Dropout(p), nn.Linear(head_h, 3))

    def forward(self, x, tgt, utmos):
        h = self.trunk(x)
        qmos_in = torch.cat([h, utmos], dim=1) if self.use_utmos else h
        qmos = self.qmos(qmos_in)
        emos = self.emos(torch.cat([h, tgt], dim=1))
        cat_logits = self.cat(h)
        vad = self.vad(h)
        return qmos, emos, cat_logits, vad

model = FusionMTL6(FEAT_DIM, TRUNK_HIDDEN, HEAD_HIDDEN, DROPOUT, N_EMO, USE_UTMOS_FEAT).to(device)

TASKS = ["qmos", "emos", "cat", "val", "aro", "dom"]
log_var = nn.Parameter(torch.zeros(len(TASKS), device=device))
params = list(model.parameters()) + ([log_var] if USE_UNCERTAINTY else [])
opt = torch.optim.Adam(params, lr=LR, weight_decay=1e-5)
mse = nn.MSELoss(reduction="none")

def soft_ce(logits, target_dist):
    logq = F.log_softmax(logits, dim=1)
    return -(target_dist * logq).sum(dim=1)

def task_losses(qmos_p, emos_p, cat_logits, vad_p, b):
    L = {}
    L["qmos"] = mse(qmos_p, qmos_t[b]).mean()
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
    if USE_UNCERTAINTY:
        tot = 0.0
        for i, t in enumerate(TASKS):
            tot = tot + torch.exp(-log_var[i]) * L[t] + log_var[i]
        return tot
    return sum(LOSS_W[t] * L[t] for t in TASKS)

@torch.no_grad()
def eval_val():
    model.eval()
    qp, ep, cl, vp = model(Xn_t[va_idx], T_t[va_idx], Un_t[va_idx])
    qp = qp.cpu().numpy().ravel(); ep = ep.cpu().numpy().ravel()
    out = {"qmos": spearmanr(qp, y_qmos[va_idx]).correlation,
           "emos": spearmanr(ep, y_emos[va_idx]).correlation}
    if USE_UTMOS_FEAT:
        out["qmos_utmos"] = spearmanr(U[va_idx, 0], y_qmos[va_idx]).correlation   # baseline UTMOS đơn lẻ
    if HAS_VAD:
        vp = vp.cpu().numpy()
        for j, t in enumerate(["val", "aro", "dom"]):
            out[t] = spearmanr(vp[:, j], y_vad[va_idx, j]).correlation
    q = F.softmax(cl, dim=1).cpu().numpy(); p = y_cat[va_idx]
    kl = (p * (np.log(p + 1e-9) - np.log(q + 1e-9))).sum(1).mean()
    out["cat_negkl"] = float(-kl)
    return out

def val_score(m):
    """Điểm tổng early-stop = TB SRCC các task liên tục (qmos+emos+VAD)."""
    keys = ["qmos", "emos"] + (["val", "aro", "dom"] if HAS_VAD else [])
    return float(np.mean([m[k] for k in keys]))

best_score, best_state, bad = -1e9, None, 0
tr_t = torch.tensor(tr_idx, device=device)
for ep_i in range(1, EPOCHS + 1):
    model.train()
    perm = tr_t[torch.randperm(len(tr_t), device=device)]
    run = 0.0
    for i in range(0, len(perm), BATCH):
        b = perm[i:i + BATCH]
        opt.zero_grad()
        qmos_p, emos_p, cat_logits, vad_p = model(Xn_t[b], T_t[b], Un_t[b])
        L = task_losses(qmos_p, emos_p, cat_logits, vad_p, b)
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
    if ep_i % 5 == 0 or ep_i == 1:
        msg = " ".join(f"{k}={m[k]:.3f}" for k in ["qmos", "emos", "val", "aro", "dom"] if k in m)
        print(f"epoch {ep_i:3d} | loss {run/len(perm):.4f} | {msg} | best {best_score:.4f}")
    if bad >= PATIENCE:
        print(f"Early stop ở epoch {ep_i}.")
        break

model.load_state_dict(best_state)
final = eval_val()
print("\n✅ VAL (nội bộ) — exp07 (fusion + QMOS head):")
print(f"   QMOS SRCC = {final['qmos']:.4f}", end="")
if "qmos_utmos" in final:
    tag = "✅ vượt UTMOS" if final["qmos"] > final["qmos_utmos"] else "⚠️ CHƯA vượt UTMOS"
    print(f"   (UTMOS đơn lẻ = {final['qmos_utmos']:.4f} → {tag})")
else:
    print()
print(f"   EMOS SRCC = {final['emos']:.4f}   (mốc exp04 = {EXP04['emos']})")
if HAS_VAD:
    print(f"   VAL/ARO/DOM = {final['val']:.4f}/{final['aro']:.4f}/{final['dom']:.4f}"
          f"   (mốc exp04 = {EXP04['val']}/{EXP04['aro']}/{EXP04['dom']})")
# Cảnh báo negative transfer (gộp QMOS làm tụt cảm xúc)
warn = []
if final["emos"] < EXP04["emos"] - 0.02:
    warn.append(f"EMOS {final['emos']:.3f} < {EXP04['emos']}")
if HAS_VAD:
    for t in ["val", "aro", "dom"]:
        if final[t] < EXP04[t] - 0.02:
            warn.append(f"{t.upper()} {final[t]:.3f} < {EXP04[t]}")
if warn:
    print("   ⚠️ NEGATIVE TRANSFER? Cảm xúc tụt so exp04:", "; ".join(warn),
          "\n      → cân nhắc giữ exp04 cho 5 cột cảm xúc + chỉ lấy QMOS từ exp07/exp06.")
else:
    print("   ✅ Không thấy 5 cột cảm xúc tụt rõ so exp04.")
if USE_UNCERTAINTY:
    print("   log σ² mỗi task:", {t: round(float(log_var[i]), 3) for i, t in enumerate(TASKS)})

torch.save({"state": best_state, "feat_mean": feat_mean, "feat_std": feat_std,
            "u_mu": u_mu, "u_sd": u_sd,
            "qmos_mu": qmos_mu, "qmos_sd": qmos_sd, "emos_mu": emos_mu, "emos_sd": emos_sd,
            "vad_mu": vad_mu, "vad_sd": vad_sd, "FEAT_DIM": FEAT_DIM,
            "USE_E2V": USE_E2V, "USE_SAILER": USE_SAILER, "USE_CLASSPROB": USE_CLASSPROB,
            "USE_UTMOS_FEAT": USE_UTMOS_FEAT, "val_score": best_score},
           os.path.join(OUT_DIR, "fusion_qmos_mtl.pt"))
print("Đã lưu", os.path.join(OUT_DIR, "fusion_qmos_mtl.pt"))

# %% [markdown]
# ## 6. Dự đoán DEV → `answer.txt` đủ 6 cột (QMOS giờ từ HEAD, không phải SpeechMOS riêng)

# %%
def list_dev():
    with open(DEV_SCP) as f:
        return [ln.strip() for ln in f if ln.strip()]

dev_names = list_dev()
if LIMIT_DEV:
    dev_names = dev_names[:LIMIT_DEV]
dev_stems = [stem(n) for n in dev_names]
print("DEV:", len(dev_names), "mẫu")

e2v_dev    = extract_e2v(dev_stems, "dev")    if USE_E2V    else {}
sailer_dev = extract_sailer(dev_stems, "dev") if USE_SAILER else {}
utmos_dev  = extract_utmos(dev_names, "dev")  if USE_UTMOS_FEAT else {}

@torch.no_grad()
def predict_all(sid):
    f = audio_feature(sid, e2v_dev, sailer_dev)
    if f is None:
        return None
    fn = (f[None, :] - feat_mean) / feat_std
    tgt = onehot_target(target_map.get(sid))[None, :]
    u = np.array([[utmos_dev.get(sid, 3.0)]], dtype=np.float32)
    un = (u - u_mu) / u_sd
    model.eval()
    qmos_p, emos_p, cat_logits, vad_p = model(to_t(fn), to_t(tgt), to_t(un))
    qmos = float(qmos_p.item()) * qmos_sd + qmos_mu
    emos = float(emos_p.item()) * emos_sd + emos_mu
    cat5 = F.softmax(cat_logits, dim=1)[0].cpu().numpy()
    vad3 = vad_p[0].cpu().numpy() * vad_sd + vad_mu
    return qmos, emos, cat5, vad3

def fmt_cat(probs5):
    return "|".join(f"{e}:{probs5[i]:.6g}" for i, e in enumerate(EMOTIONS5))

def build_answer(out_path):
    from tqdm.auto import tqdm
    n_real = n_default = 0
    with open(out_path, "w") as f:
        f.write("wav,QMOS,EMOS,CAT,VAL,ARO,DOM\n")
        for name in tqdm(dev_names, desc="answer"):
            sid = stem(name)
            pred = predict_all(sid)
            if pred is None:
                # rơi về: QMOS=UTMOS nếu có, còn lại mặc định
                qmos = utmos_dev.get(sid, 3.0)
                emos, cat5, vad3 = 3.0, np.full(5, 0.2, np.float32), np.array([3.0, 3.0, 3.0])
                n_default += 1
            else:
                qmos, emos, cat5, vad3 = pred
                n_real += 1
            f.write(f"{name},{qmos:.6g},{emos:.6g},{fmt_cat(cat5)},"
                    f"{vad3[0]:.6g},{vad3[1]:.6g},{vad3[2]:.6g}\n")
    print(f"Ghi {len(dev_names)} dòng → {out_path} | head thật {n_real}, mặc định {n_default}")

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
os.system(f"cd {OUT_DIR} && zip -j submission_track2_exp07_fusion_qmos.zip answer.txt "
          f"&& unzip -l submission_track2_exp07_fusion_qmos.zip")
print("Sẵn sàng nộp:", os.path.join(OUT_DIR, "submission_track2_exp07_fusion_qmos.zip"))

# %% [markdown]
# ## Ghi chú
# - **Lần đầu** đặt `LIMIT_TRAIN=300`, `LIMIT_DEV=20`; OK rồi đặt `None`.
# - **Đọc kết quả mục 5 theo 2 câu hỏi:**
#   1. QMOS head có **vượt UTMOS đơn lẻ (0.414)** không? (dòng "vượt/CHƯA vượt UTMOS")
#   2. Gộp QMOS có **làm tụt** EMOS/VAD so exp04 không? (dòng "NEGATIVE TRANSFER?")
# - **Quyết định nộp:**
#   - Nếu QMOS↑ và cảm xúc KHÔNG tụt → nộp answer.txt exp07 (1 model trọn 6 cột — đẹp cho paper).
#   - Nếu QMOS↑ nhưng cảm xúc TỤT → giữ exp04 cho 5 cột cảm xúc, chỉ lấy **cột QMOS** của exp07/exp06 ghép vào.
#   - Nếu QMOS không vượt UTMOS → kết luận "chất lượng trực giao cảm xúc" (vẫn là phát hiện cho paper); giữ exp04.
# - **Ablation cho paper**: `USE_UTMOS_FEAT=False` (QMOS chỉ từ trunk cảm xúc) → đo trực tiếp giả thuyết của bạn.
# - Cache dùng CHUNG `fusion_cache/` với exp04 → **Save Version** giữ lại.
# - Ghi config → kết quả → nhận xét vào `docs/04_experiments_log.md` (mục exp07).
