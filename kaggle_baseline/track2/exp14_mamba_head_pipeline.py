# %% [markdown]
# # VMC2026 Track 2 — exp14 (MAMBA temporal head, CỘNG vào FUSION 6 cột) — Kaggle
#
# **Ý tưởng (theo gợi ý mentor "thử Mamba"):** exp04/exp07 đều **mean-pool** đặc trưng SSL →
# mỗi wav thành 1 vector → mất hết **động lực theo thời gian** (lên/xuống giọng, ngắt quãng, rung).
# **Mamba** là State Space Model (SSM) xử lý **chuỗi** với độ phức tạp tuyến tính → cho nó **dãy frame**
# (chưa pool) để học temporal dynamics, rồi mới pool. Tham khảo: MambaRate (AudioMOS 2025), arXiv:2507.12090.
#
# ## exp14 = exp07 + 1 nhánh Mamba (CỘNG thêm, không thay thế)
# ```
#            ┌─ đặc trưng POOLED [e2v_emb|e2v_p5|sailer_emb|sailer_p9|sailer_vad3]  (y hệt exp07 → DÙNG LẠI cache)
#  mỗi wav ──┤
#            └─ WavLM frame-level (chuỗi T×1024) ─► Mamba (2 lớp, 2 chiều) ─► attn-pool ─► z_seq (Z chiều)
#                    │
#         concat ──► TRUNK chung ──► 6 head: QMOS · EMOS · CAT · VAL · ARO · DOM
# ```
# - **Cờ `USE_MAMBA`:** `False` → chạy ra **đúng exp07** (kiểm chứng tái lập ~0.548/0.795). `True` → bật nhánh Mamba.
#   Đây CHÍNH là **ablation "có/không Mamba"** cho paper.
# - WavLM **đóng băng** (chỉ trích đặc trưng) → Mamba head nhỏ → train nhanh, vừa T4.
#
# ## 2 gotcha Kaggle đã xử trong file
# 1. `mamba-ssm` hay lỗi build CUDA → **nhúng sẵn Mamba thuần PyTorch** (không cần pip); tự dùng `mamba-ssm` nếu import được.
# 2. Cache frame-level RẤT nặng → **cap `MAX_FRAMES`** + lưu **fp16**. Ước lượng: MAX_FRAMES=256, 1024 chiều, fp16
#    ≈ 0.5 MB/wav → train ~12k ≈ 6 GB, dev ~2.7k ≈ 1.4 GB (vừa /kaggle/working). **Save Version** để giữ cache.
#
# **Cách chạy:** GPU T4 + Internet On → Add Input dataset Track 2 → sửa `DATA_ROOT` → Run All.
# Lần đầu đặt `LIMIT_TRAIN=300`, `LIMIT_DEV=20` để soi nhanh; OK rồi đặt `None`.

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
CACHE_DIR = "/kaggle/working/fusion_cache"     # DÙNG CHUNG với exp04/exp07 (e2v_*, sailer_*, utmos_*)
SEQ_DIR   = "/kaggle/working/wavlm_seq_cache"  # MỚI: cache frame-level WavLM (fp16)
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(SEQ_DIR, exist_ok=True)

# ── Bật/tắt nhánh Mamba (ablation chính) ─────────────────────────────────────
USE_MAMBA       = True         # False → ra ĐÚNG exp07 (sanity check). True → bật nhánh Mamba.

# ── Siêu tham số nhánh Mamba ─────────────────────────────────────────────────
WAVLM_NAME      = "microsoft/wavlm-large"   # backbone frame-level (đóng băng). Trả chuỗi (T, 1024).
MAX_FRAMES      = 256          # cap độ dài chuỗi (256 frame ≈ 5.1s @ 50Hz). Giảm nếu hết đĩa.
MAMBA_DMODEL    = 256          # chiều ẩn của khối Mamba (proj 1024→256 trước khi vào Mamba)
MAMBA_LAYERS    = 2            # số khối Mamba xếp chồng
MAMBA_DSTATE    = 16           # chiều state SSM
BIDIRECTIONAL   = True         # chạy Mamba cả 2 chiều (xuôi + ngược) rồi cộng
Z_DIM           = 128          # chiều vector z_seq sau attentive-pool, đem concat vào fusion

# ── Siêu tham số fusion (giống exp07) ────────────────────────────────────────
DEVICE          = "cuda"
TRUNK_HIDDEN    = 512
HEAD_HIDDEN     = 128
DROPOUT         = 0.3
LR              = 1e-3
EPOCHS          = 80
BATCH           = 32           # nhỏ hơn exp07 (64) vì có nhánh Mamba tốn RAM hơn
VAL_FRAC        = 0.10
PATIENCE        = 15
SEED            = 42

USE_UNCERTAINTY = True
LOSS_W          = {"qmos": 1.0, "emos": 1.0, "cat": 1.0, "val": 1.0, "aro": 1.0, "dom": 1.0}
USE_E2V         = True
USE_SAILER      = True
USE_CLASSPROB   = True
USE_UTMOS_FEAT  = True

LIMIT_TRAIN     = None
LIMIT_DEV       = None

# Mốc exp07 để so (đây là hệ thống đang tốt nhất)
EXP07 = {"qmos": 0.548, "emos": 0.795, "cat_err": 0.153, "val": 0.581, "aro": 0.752, "dom": 0.705}

EMOTIONS5 = ["angry", "happy", "neutral", "sad", "surprised"]
SAILER9 = ["Anger", "Contempt", "Disgust", "Fear", "Happiness", "Neutral", "Sadness", "Surprise", "Other"]

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

assert USE_E2V or USE_SAILER, "Phải bật ít nhất 1 backbone pooled."
print("USE_MAMBA =", USE_MAMBA, "| nếu False → ra đúng exp07")
print("DATA_ROOT:", DATA_ROOT)
for p in [WAV_DIR, METADATA_CSV, TRAIN_CSV, DEV_SCP]:
    print(("  ✅ " if os.path.exists(p) else "  ❌ THIẾU ") + p)

# %% [markdown]
# ## 1. Cài đặt + tải code SAILER
# Chỉ cài gói còn thiếu (Kaggle có sẵn torch/transformers). KHÔNG đụng numpy (tránh lệch ABI torch — bài học exp12).

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
# ## 2. Đọc & gộp nhãn theo wavID (giống exp07)

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
    df = pd.read_csv(TRAIN_CSV, sep="|")
    cols = {c.lower().strip(): c for c in df.columns}
    wav_col  = _col(cols, "wavid", "wav", default_idx=1, df=df)
    qmos_col = _col(cols, "qmos", "mos")
    emos_col = _col(cols, "emos", "emo", "emomos")
    val_col  = _col(cols, "val", "valence")
    aro_col  = _col(cols, "aro", "arousal")
    dom_col  = _col(cols, "dom", "dominance")
    cat_col  = _col(cols, "emocat", "cat", "emotion")
    assert qmos_col and emos_col, f"Thiếu cột qMOS/eMOS (cột: {list(df.columns)})"
    df["_stem"] = df[wav_col].map(stem)
    rows = []
    for sid, g in df.groupby("_stem"):
        rec = {"wavID": sid, "qmos": float(g[qmos_col].mean()), "emos": float(g[emos_col].mean())}
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

# %% [markdown]
# ## 3. Đặc trưng POOLED (e2v + sailer + UTMOS) — TÁI DÙNG cache exp04/exp07
# (Y hệt exp07; nếu đã chạy exp07 thì cache `fusion_cache/` còn nguyên → không tính lại.)

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
# ## 3b. Đặc trưng FRAME-LEVEL WavLM (chuỗi T×1024) cho nhánh Mamba — cache fp16
# Mỗi wav lưu 1 file `.npy` riêng trong `SEQ_DIR` (mảng fp16 [T, 1024], T ≤ MAX_FRAMES).
# WavLM **đóng băng** (eval, no_grad) → layerdrop tự tắt ở eval, không đụng gotcha checkpoint.

# %%
_wavlm = None
def _get_wavlm():
    """Lazy-load microsoft/wavlm-large (đóng băng). Trả model + feature_extractor."""
    global _wavlm
    if _wavlm is None:
        from transformers import WavLMModel, AutoFeatureExtractor
        fe = AutoFeatureExtractor.from_pretrained(WAVLM_NAME)
        mdl = WavLMModel.from_pretrained(WAVLM_NAME).to(device).eval()
        for p in mdl.parameters():
            p.requires_grad = False
        _wavlm = (mdl, fe)
    return _wavlm

def seq_path(sid):
    return os.path.join(SEQ_DIR, sid + ".npy")

def extract_wavlm_seq(stems, tag):
    """Trích frame-level WavLM cho từng wav, cache fp16 ra .npy. Trả set stem đã có."""
    if not USE_MAMBA:
        return set()
    import librosa
    from tqdm.auto import tqdm
    todo = [s for s in stems if not os.path.exists(seq_path(s))]
    if todo:
        mdl, fe = _get_wavlm()
        with torch.no_grad():
            for i, s in enumerate(tqdm(todo, desc=f"wavlm-seq {tag}")):
                wav = os.path.join(WAV_DIR, s + ".wav")
                if not os.path.exists(wav):
                    continue
                wave, _ = librosa.load(wav, sr=16000, mono=True)
                wave = wave[: 15 * 16000]
                inp = fe(wave, sampling_rate=16000, return_tensors="pt").input_values.to(device)
                hs = mdl(inp).last_hidden_state[0]          # (T, 1024)
                if hs.shape[0] > MAX_FRAMES:                 # cap độ dài (đều theo thời gian)
                    idx = torch.linspace(0, hs.shape[0] - 1, MAX_FRAMES).long()
                    hs = hs[idx]
                np.save(seq_path(s), hs.cpu().numpy().astype(np.float16))
        torch.cuda.empty_cache() if device == "cuda" else None
    return {s for s in stems if os.path.exists(seq_path(s))}

def load_seq(sid):
    """Đọc chuỗi fp16 → tensor float32 (T, 1024). Thiếu file → None."""
    p = seq_path(sid)
    if not os.path.exists(p):
        return None
    return torch.from_numpy(np.load(p).astype(np.float32))

def collate_seqs(sids):
    """Gộp list chuỗi độ dài khác nhau → (B, Lmax, 1024) + mask (B, Lmax) bool (True=thật)."""
    seqs = [load_seq(s) for s in sids]
    lens = [t.shape[0] for t in seqs]
    Lmax = max(lens)
    B = len(seqs)
    x = torch.zeros(B, Lmax, seqs[0].shape[1], dtype=torch.float32)
    mask = torch.zeros(B, Lmax, dtype=torch.bool)
    for i, t in enumerate(seqs):
        x[i, : t.shape[0]] = t
        mask[i, : t.shape[0]] = True
    return x, mask

# %% [markdown]
# ## 4. Dựng feature pooled + nhãn cho train (lọc các wav đủ mọi nguồn)

# %%
train_stems = list(train_df["wavID"])
if LIMIT_TRAIN:
    train_stems = train_stems[:LIMIT_TRAIN]

e2v_tr    = extract_e2v(train_stems, "train")    if USE_E2V    else {}
sailer_tr = extract_sailer(train_stems, "train") if USE_SAILER else {}
utmos_tr  = extract_utmos(train_stems, "train")  if USE_UTMOS_FEAT else {}
seq_tr    = extract_wavlm_seq(train_stems, "train")

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
keep_sids, X, T, U = [], [], [], []
y_qmos, y_emos, y_vad, y_cat = [], [], [], []
for s in train_stems:
    f = audio_feature(s, e2v_tr, sailer_tr)
    tgt = target_map.get(s)
    if f is None or tgt is None or s not in lab.index:
        continue
    if USE_UTMOS_FEAT and s not in utmos_tr:
        continue
    if USE_MAMBA and s not in seq_tr:        # cần có chuỗi WavLM nếu bật Mamba
        continue
    keep_sids.append(s)
    X.append(f)
    T.append(onehot_target(tgt))
    U.append(utmos_tr.get(s, 3.0) if USE_UTMOS_FEAT else 0.0)
    y_qmos.append(lab.loc[s, "qmos"]); y_emos.append(lab.loc[s, "emos"])
    y_vad.append([lab.loc[s, "val"], lab.loc[s, "aro"], lab.loc[s, "dom"]])
    y_cat.append([lab.loc[s, f"cat{i}"] for i in range(len(EMOTIONS5))])

X = np.stack(X).astype(np.float32)
T = np.stack(T).astype(np.float32)
U = np.array(U, dtype=np.float32).reshape(-1, 1)
y_qmos = np.array(y_qmos, dtype=np.float32); y_emos = np.array(y_emos, dtype=np.float32)
y_vad  = np.array(y_vad,  dtype=np.float32); y_cat  = np.array(y_cat,  dtype=np.float32)
FEAT_DIM = X.shape[1]
print(f"Train giữ lại: {len(keep_sids)} wav | X={X.shape} | Mamba={'ON' if USE_MAMBA else 'OFF'}")

# Chuẩn hóa feature pooled + UTMOS + nhãn liên tục (z-score)
feat_mean = X.mean(0, keepdims=True); feat_std = X.std(0, keepdims=True) + 1e-6
Xn = (X - feat_mean) / feat_std
u_mu, u_sd = float(U.mean()), float(U.std() + 1e-6); Un = (U - u_mu) / u_sd
qmos_mu, qmos_sd = float(y_qmos.mean()), float(y_qmos.std() + 1e-6); y_qmos_z = (y_qmos - qmos_mu) / qmos_sd
emos_mu, emos_sd = float(y_emos.mean()), float(y_emos.std() + 1e-6); y_emos_z = (y_emos - emos_mu) / emos_sd
if HAS_VAD:
    vad_mu = np.nanmean(y_vad, axis=0); vad_sd = np.nanstd(y_vad, axis=0) + 1e-6
    y_vad_z = (y_vad - vad_mu) / vad_sd
else:
    vad_mu = np.zeros(3, dtype=np.float32); vad_sd = np.ones(3, dtype=np.float32); y_vad_z = np.zeros_like(y_vad)

# %% [markdown]
# ## 5a. Khối MAMBA (thuần PyTorch, không cần `mamba-ssm`)
# Tự dùng `mamba-ssm` nếu import được (nhanh hơn); nếu không → bản thuần PyTorch (selective scan vòng lặp thời gian).
# Bản này theo "mamba-minimal" (johnma2006) — đúng công thức, chỉ chậm hơn kernel CUDA, nhưng head nhỏ nên OK trên T4.

# %%
import math
import torch.nn as nn

try:
    from mamba_ssm import Mamba as _OfficialMamba   # nếu cài được thì dùng (tùy chọn)
    _HAS_MAMBA_SSM = True
    print("✅ Dùng mamba-ssm (CUDA kernel)")
except Exception:
    _HAS_MAMBA_SSM = False
    print("ℹ️ Không có mamba-ssm → dùng Mamba thuần PyTorch (nhúng sẵn)")

class MambaBlockTorch(nn.Module):
    """Một khối Mamba (selective SSM) thuần PyTorch. d_model = chiều ẩn."""
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_inner = expand * d_model
        self.dt_rank = math.ceil(d_model / 16)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=d_conv,
                                groups=self.d_inner, padding=d_conv - 1, bias=True)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))           # (d_inner, d_state)
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.d_state = d_state

    def forward(self, x):                                 # x: (B, L, d_model)
        B, L, _ = x.shape
        xz = self.in_proj(x)                              # (B, L, 2*d_inner)
        xin, z = xz.chunk(2, dim=-1)
        xin = xin.transpose(1, 2)                         # (B, d_inner, L)
        xin = self.conv1d(xin)[..., :L].transpose(1, 2)   # (B, L, d_inner) causal conv
        xin = F.silu(xin)
        y = self._ssm(xin)                                # (B, L, d_inner)
        y = y * F.silu(z)
        return self.out_proj(y)

    def _ssm(self, x):                                    # x: (B, L, d_inner)
        A = -torch.exp(self.A_log)                        # (d_inner, d_state)
        x_dbl = self.x_proj(x)                            # (B, L, dt_rank + 2*d_state)
        delta, Bm, Cm = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        delta = F.softplus(self.dt_proj(delta))           # (B, L, d_inner)
        dA = torch.exp(delta.unsqueeze(-1) * A)           # (B, L, d_inner, d_state)
        dB_x = delta.unsqueeze(-1) * Bm.unsqueeze(2) * x.unsqueeze(-1)  # (B, L, d_inner, d_state)
        h = torch.zeros(x.shape[0], self.d_inner, self.d_state, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(x.shape[1]):                       # selective scan theo thời gian
            h = dA[:, t] * h + dB_x[:, t]
            ys.append((h * Cm[:, t].unsqueeze(1)).sum(-1))   # (B, d_inner)
        y = torch.stack(ys, dim=1)                        # (B, L, d_inner)
        return y + x * self.D

class MambaLayer(nn.Module):
    """Pre-norm residual quanh 1 khối Mamba (chọn official nếu có)."""
    def __init__(self, d_model, d_state):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        if _HAS_MAMBA_SSM:
            self.mix = _OfficialMamba(d_model=d_model, d_state=d_state, d_conv=4, expand=2)
        else:
            self.mix = MambaBlockTorch(d_model, d_state=d_state)

    def forward(self, x):
        return x + self.mix(self.norm(x))

class MambaEncoder(nn.Module):
    """1024 → d_model → [Mamba ×L] (2 chiều nếu BIDIRECTIONAL) → attentive-pool → Z_DIM."""
    def __init__(self, d_in, d_model, n_layers, d_state, z_dim, bidir):
        super().__init__()
        self.bidir = bidir
        self.proj = nn.Linear(d_in, d_model)
        self.fwd = nn.ModuleList([MambaLayer(d_model, d_state) for _ in range(n_layers)])
        if bidir:
            self.bwd = nn.ModuleList([MambaLayer(d_model, d_state) for _ in range(n_layers)])
        self.attn = nn.Linear(d_model, 1)                 # attentive pooling
        self.out = nn.Linear(d_model, z_dim)

    def _run(self, layers, h):
        for L in layers:
            h = L(h)
        return h

    def forward(self, x, mask):                           # x: (B, L, 1024), mask: (B, L) bool
        h = self.proj(x)
        out = self._run(self.fwd, h)
        if self.bidir:
            rev = torch.flip(h, dims=[1])
            out = out + torch.flip(self._run(self.bwd, rev), dims=[1])
        a = self.attn(out).squeeze(-1)                    # (B, L)
        a = a.masked_fill(~mask, float("-inf"))
        w = torch.softmax(a, dim=1).unsqueeze(-1)         # (B, L, 1)
        pooled = (out * w).sum(1)                          # (B, d_model)
        return self.out(pooled)                            # (B, z_dim)

# %% [markdown]
# ## 5b. Model fusion 6 head + nhánh Mamba + train loop

# %%
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split

torch.manual_seed(SEED); np.random.seed(SEED)
N_EMO = len(EMOTIONS5)
idx_all = np.arange(X.shape[0])
tr_idx, va_idx = train_test_split(idx_all, test_size=VAL_FRAC, random_state=SEED)

def to_t(a):
    return torch.tensor(a, dtype=torch.float32, device=device)

Xn_t, T_t, Un_t = to_t(Xn), to_t(T), to_t(Un)
qmos_t = to_t(y_qmos_z).unsqueeze(1); emos_t = to_t(y_emos_z).unsqueeze(1)
vad_t  = to_t(y_vad_z); cat_t = to_t(y_cat)

class FusionMamba6(nn.Module):
    def __init__(self, d_in, trunk_h, head_h, p, n_emo, use_utmos, use_mamba):
        super().__init__()
        self.use_utmos = use_utmos
        self.use_mamba = use_mamba
        z_extra = Z_DIM if use_mamba else 0
        if use_mamba:
            self.enc = MambaEncoder(1024, MAMBA_DMODEL, MAMBA_LAYERS, MAMBA_DSTATE, Z_DIM, BIDIRECTIONAL)
        self.trunk = nn.Sequential(
            nn.Linear(d_in + z_extra, trunk_h), nn.ReLU(), nn.Dropout(p),
            nn.Linear(trunk_h, trunk_h), nn.ReLU(), nn.Dropout(p))
        self.qmos = nn.Sequential(
            nn.Linear(trunk_h + (1 if use_utmos else 0), head_h), nn.ReLU(), nn.Dropout(p), nn.Linear(head_h, 1))
        self.emos = nn.Sequential(
            nn.Linear(trunk_h + n_emo, head_h), nn.ReLU(), nn.Dropout(p), nn.Linear(head_h, 1))
        self.cat = nn.Sequential(
            nn.Linear(trunk_h, head_h), nn.ReLU(), nn.Dropout(p), nn.Linear(head_h, n_emo))
        self.vad = nn.Sequential(
            nn.Linear(trunk_h, head_h), nn.ReLU(), nn.Dropout(p), nn.Linear(head_h, 3))

    def forward(self, x, tgt, utmos, seq=None, mask=None):
        if self.use_mamba:
            z = self.enc(seq, mask)
            x = torch.cat([x, z], dim=1)
        h = self.trunk(x)
        qmos_in = torch.cat([h, utmos], dim=1) if self.use_utmos else h
        return self.qmos(qmos_in), self.emos(torch.cat([h, tgt], dim=1)), self.cat(h), self.vad(h)

model = FusionMamba6(FEAT_DIM, TRUNK_HIDDEN, HEAD_HIDDEN, DROPOUT, N_EMO, USE_UTMOS_FEAT, USE_MAMBA).to(device)
n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Tham số train được: {n_par/1e6:.2f} M")

TASKS = ["qmos", "emos", "cat", "val", "aro", "dom"]
log_var = nn.Parameter(torch.zeros(len(TASKS), device=device))
params = list(model.parameters()) + ([log_var] if USE_UNCERTAINTY else [])
opt = torch.optim.Adam(params, lr=LR, weight_decay=1e-5)
mse = nn.MSELoss(reduction="none")

def soft_ce(logits, target_dist):
    return -(target_dist * F.log_softmax(logits, dim=1)).sum(dim=1)

def task_losses(qmos_p, emos_p, cat_logits, vad_p, b):
    L = {"qmos": mse(qmos_p, qmos_t[b]).mean(),
         "emos": mse(emos_p, emos_t[b]).mean(),
         "cat":  soft_ce(cat_logits, cat_t[b]).mean()}
    if HAS_VAD:
        L["val"] = mse(vad_p[:, 0:1], vad_t[b, 0:1]).mean()
        L["aro"] = mse(vad_p[:, 1:2], vad_t[b, 1:2]).mean()
        L["dom"] = mse(vad_p[:, 2:3], vad_t[b, 2:3]).mean()
    else:
        z = torch.zeros((), device=device); L["val"] = L["aro"] = L["dom"] = z
    return L

def combine(L):
    if USE_UNCERTAINTY:
        return sum(torch.exp(-log_var[i]) * L[t] + log_var[i] for i, t in enumerate(TASKS))
    return sum(LOSS_W[t] * L[t] for t in TASKS)

# batch theo INDEX (vì nhánh Mamba cần đọc chuỗi theo sid → collate động)
sids_arr = np.array(keep_sids)

def forward_batch(bidx):
    """bidx: numpy index. Trả output model cho batch (tự collate chuỗi nếu bật Mamba)."""
    bt = torch.tensor(bidx, device=device)
    if USE_MAMBA:
        seq, mask = collate_seqs(list(sids_arr[bidx]))
        seq, mask = seq.to(device), mask.to(device)
        return model(Xn_t[bt], T_t[bt], Un_t[bt], seq, mask)
    return model(Xn_t[bt], T_t[bt], Un_t[bt])

@torch.no_grad()
def eval_val():
    model.eval()
    qp, ep, vp = [], [], []
    for i in range(0, len(va_idx), BATCH):
        b = va_idx[i:i + BATCH]
        q, e, _cl, v = forward_batch(b)
        qp.append(q.cpu().numpy().ravel()); ep.append(e.cpu().numpy().ravel()); vp.append(v.cpu().numpy())
    qp = np.concatenate(qp); ep = np.concatenate(ep); vp = np.concatenate(vp)
    out = {"qmos": spearmanr(qp, y_qmos[va_idx]).correlation,
           "emos": spearmanr(ep, y_emos[va_idx]).correlation}
    if USE_UTMOS_FEAT:
        out["qmos_utmos"] = spearmanr(U[va_idx, 0], y_qmos[va_idx]).correlation
    if HAS_VAD:
        for j, t in enumerate(["val", "aro", "dom"]):
            out[t] = spearmanr(vp[:, j], y_vad[va_idx, j]).correlation
    return out

def val_score(m):
    keys = ["qmos", "emos"] + (["val", "aro", "dom"] if HAS_VAD else [])
    return float(np.mean([m[k] for k in keys]))

best_score, best_state, bad = -1e9, None, 0
for ep_i in range(1, EPOCHS + 1):
    model.train()
    perm = np.random.permutation(tr_idx)
    run = 0.0
    for i in range(0, len(perm), BATCH):
        b = perm[i:i + BATCH]
        opt.zero_grad()
        q, e, cl, v = forward_batch(b)
        loss = combine(task_losses(q, e, cl, v, torch.tensor(b, device=device)))
        loss.backward(); opt.step()
        run += loss.item() * len(b)
    m = eval_val(); sc = val_score(m)
    if sc > best_score:
        best_score = sc; bad = 0
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    else:
        bad += 1
    if ep_i % 2 == 0 or ep_i == 1:
        msg = " ".join(f"{k}={m[k]:.3f}" for k in ["qmos", "emos", "val", "aro", "dom"] if k in m)
        print(f"epoch {ep_i:3d} | loss {run/len(perm):.4f} | {msg} | best {best_score:.4f}")
    if bad >= PATIENCE:
        print(f"Early stop ở epoch {ep_i}."); break

model.load_state_dict(best_state)
final = eval_val()
print(f"\n✅ VAL (nội bộ) — exp14 (Mamba={'ON' if USE_MAMBA else 'OFF'}):")
print(f"   QMOS={final['qmos']:.4f} (exp07 {EXP07['qmos']}) | EMOS={final['emos']:.4f} (exp07 {EXP07['emos']})")
if HAS_VAD:
    print(f"   VAL/ARO/DOM={final['val']:.4f}/{final['aro']:.4f}/{final['dom']:.4f}"
          f" (exp07 {EXP07['val']}/{EXP07['aro']}/{EXP07['dom']})")
print("   → So sánh USE_MAMBA True vs False = ablation Mamba cho paper.")

torch.save({"state": best_state, "feat_mean": feat_mean, "feat_std": feat_std,
            "u_mu": u_mu, "u_sd": u_sd, "qmos_mu": qmos_mu, "qmos_sd": qmos_sd,
            "emos_mu": emos_mu, "emos_sd": emos_sd, "vad_mu": vad_mu, "vad_sd": vad_sd,
            "FEAT_DIM": FEAT_DIM, "USE_MAMBA": USE_MAMBA, "val_score": best_score},
           os.path.join(OUT_DIR, "fusion_mamba_mtl.pt"))
print("Đã lưu", os.path.join(OUT_DIR, "fusion_mamba_mtl.pt"))

# %% [markdown]
# ## 6. Dự đoán DEV → `answer.txt` đủ 6 cột

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
seq_dev    = extract_wavlm_seq(dev_stems, "dev")

@torch.no_grad()
def predict_all(sid):
    f = audio_feature(sid, e2v_dev, sailer_dev)
    if f is None:
        return None
    if USE_MAMBA and not os.path.exists(seq_path(sid)):
        return None
    fn = (f[None, :] - feat_mean) / feat_std
    tgt = onehot_target(target_map.get(sid))[None, :]
    u = np.array([[utmos_dev.get(sid, 3.0)]], dtype=np.float32); un = (u - u_mu) / u_sd
    model.eval()
    if USE_MAMBA:
        seq, mask = collate_seqs([sid]); seq, mask = seq.to(device), mask.to(device)
        q, e, cl, v = model(to_t(fn), to_t(tgt), to_t(un), seq, mask)
    else:
        q, e, cl, v = model(to_t(fn), to_t(tgt), to_t(un))
    qmos = float(q.item()) * qmos_sd + qmos_mu
    emos = float(e.item()) * emos_sd + emos_mu
    cat5 = F.softmax(cl, dim=1)[0].cpu().numpy()
    vad3 = v[0].cpu().numpy() * vad_sd + vad_mu
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
                qmos = utmos_dev.get(sid, 3.0)
                emos, cat5, vad3 = 3.0, np.full(5, 0.2, np.float32), np.array([3.0, 3.0, 3.0])
                n_default += 1
            else:
                qmos, emos, cat5, vad3 = pred; n_real += 1
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
os.system(f"cd {OUT_DIR} && zip -j submission_track2_exp14_mamba.zip answer.txt "
          f"&& unzip -l submission_track2_exp14_mamba.zip")
print("Sẵn sàng nộp:", os.path.join(OUT_DIR, "submission_track2_exp14_mamba.zip"))

# %% [markdown]
# ## Ghi chú
# - **Ablation chính cho paper:** chạy 2 lần — `USE_MAMBA=False` (= exp07, mốc) và `USE_MAMBA=True`.
#   So QMOS/EMOS/VAD nội bộ → trả lời "bộ mã hóa thời gian Mamba có hơn mean-pooling không?".
# - **Nếu hết đĩa khi cache chuỗi:** giảm `MAX_FRAMES` (256→160) hoặc xóa `wavlm_seq_cache/` sau khi chạy xong.
# - **Nếu Mamba chậm:** thử `pip install mamba-ssm causal-conv1d` (file tự dùng nếu import được); hoặc giảm
#   `MAMBA_LAYERS`/`MAX_FRAMES`. Bản thuần PyTorch dùng vòng lặp thời gian nên chậm hơn kernel CUDA.
# - **Save Version** để giữ cache `fusion_cache/` + `wavlm_seq_cache/` cho lần sau.
# - Ghi config → kết quả → nhận xét vào `docs/04_experiments_log.md` (mục exp14).
