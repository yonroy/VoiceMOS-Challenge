# %% [markdown]
# # VMC2026 Track 2 — exp08-RESUME (fine-tune TIẾP từ checkpoint + cache) — Kaggle
#
# **Mục đích:** train tiếp model fine-tune cảm xúc (exp08) từ **checkpoint đã lưu** thay vì train lại từ
# đầu — tiết kiệm giờ GPU. Tận dụng:
# - `ft_emotion_full.pt` (CÓ cả backbone WavLM + heads + thống kê chuẩn hóa) → nạp lại đúng trạng thái.
# - **cache audeering** `aud_*.npz` (đặc trưng frozen) → KHÔNG trích lại (~đỡ chục phút).
#
# > ⚠️ Bắt buộc dùng checkpoint **đủ backbone** (`ft_emotion_full.pt` từ cell "TRAIN TIẾP", hoặc bản
# > `ft_emotion_meta.pt` MỚI đã vá để lưu cả `wavlm`). Bản `ft_emotion_meta.pt` CŨ chỉ có `heads` → KHÔNG dùng được.
#
# ## Chuẩn bị input trên Kaggle (Add Input)
# 1. Dataset Track 2 (`vmc2026-track2-full`) — wav + nhãn.
# 2. **Checkpoint**: upload `ft_emotion_full.pt` thành 1 Dataset → trỏ `RESUME_CKPT`.
# 3. **Cache** (tùy chọn nhưng nên có): upload thư mục chứa `aud_train.npz`, `aud_dev.npz` → trỏ `CACHE_INPUT`.
# 4. (tùy chọn) `answer.txt` exp07 để mượn cột QMOS 0.548.
#
# **Cách chạy:** GPU T4 + Internet On → sửa các slug ở cell 0 → Run All. Lần đầu để `LIMIT_TRAIN=300`.

# %% [markdown]
# ## 0. Cấu hình — SỬA Ở ĐÂY

# %%
import os, shutil

DATA_ROOT    = "/kaggle/input/vmc2026-track2-full/vmc2026-track2"   # << SỬA slug
WAV_DIR      = f"{DATA_ROOT}/wav"
METADATA_CSV = f"{DATA_ROOT}/metadata.csv"
TRAIN_CSV    = f"{DATA_ROOT}/sets/train.csv"
DEV_SCP      = f"{DATA_ROOT}/sets/dev.scp"

# ── Checkpoint + cache để RESUME ─────────────────────────────────────────────
RESUME_CKPT  = "/kaggle/input/ft-emotion-full/ft_emotion_full.pt"   # << CHECKPOINT đủ backbone
CACHE_INPUT  = "/kaggle/input/ft-emotion-cache"                     # << thư mục chứa aud_*.npz (hoặc "" nếu không có)
EXP07_ANSWER = "/kaggle/input/exp07-answer/answer.txt"             # << (tùy chọn) mượn QMOS 0.548; không có → UTMOSv2

OUT_DIR   = "/kaggle/working"
CACHE_DIR = "/kaggle/working/ft_cache"     # /kaggle/input read-only → copy cache sang đây để ghi/append được
os.makedirs(CACHE_DIR, exist_ok=True)

# ── Fine-tune / siêu tham số (train TIẾP) ────────────────────────────────────
DEVICE              = "cuda"
SR                  = 16000
MAX_SECONDS         = 8
UNFREEZE_TOP_LAYERS = 6           # PHẢI khớp checkpoint (mặc định exp08 = 6)
TRUNK_HIDDEN        = 512          # PHẢI khớp checkpoint
HEAD_HIDDEN         = 128          # PHẢI khớp checkpoint
DROPOUT             = 0.3
LR_BACKBONE         = 1e-5
LR_HEAD             = 1e-3
RESUME_LR_SCALE     = 1.0          # <1.0 để giảm LR khi train tiếp (vd 0.5 nếu val đã chững)
WEIGHT_DECAY        = 1e-5
EPOCHS              = 10           # số epoch train THÊM (run này)
PATIENCE            = 5            # dừng khi val không lên; LUÔN giữ best
BATCH               = 4
ACCUM               = 8           # effective batch = 32
VAL_FRAC            = 0.10
SEED                = 42
USE_AMP             = True
USE_GRAD_CKPT       = True
USE_AUDEERING       = True         # PHẢI khớp checkpoint (exp08 = True)
USE_UNCERTAINTY     = True

LIMIT_TRAIN         = 300          # << LẦN ĐẦU 300; chạy thật None
LIMIT_DEV           = 20           # << LẦN ĐẦU 20; chạy thật None

# Mốc exp07 + exp08 để so
EXP07 = {"emos": 0.795, "cat_err": 0.153, "val": 0.581, "aro": 0.752, "dom": 0.705}
EXP08 = {"emos": 0.811, "cat_err": 0.133, "val": 0.659, "aro": 0.793, "dom": 0.751}  # bản đã nộp

EMOTIONS5 = ["angry", "happy", "neutral", "sad", "surprised"]
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

print("DATA_ROOT:", DATA_ROOT)
for p in [WAV_DIR, METADATA_CSV, TRAIN_CSV, DEV_SCP, RESUME_CKPT]:
    print(("  ✅ " if os.path.exists(p) else "  ❌ THIẾU ") + p)

# Copy cache (aud_*.npz) từ input read-only sang working để append được
if CACHE_INPUT and os.path.isdir(CACHE_INPUT):
    n = 0
    for fn in os.listdir(CACHE_INPUT):
        if fn.startswith("aud_") and fn.endswith(".npz"):
            shutil.copy(os.path.join(CACHE_INPUT, fn), os.path.join(CACHE_DIR, fn)); n += 1
    print(f"📦 Copy {n} file cache audeering từ {CACHE_INPUT} → {CACHE_DIR}")
else:
    print("ℹ️ Không có CACHE_INPUT → sẽ tự trích audeering (chậm hơn lần đầu).")

# %% [markdown]
# ## 1. Cài đặt + tải code SAILER (để dựng đúng kiến trúc WavLM rồi nạp checkpoint đè lên)

# %%
import sys, subprocess

def pip_install(*pkgs):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], check=True)

pip_install("loralib", "speechbrain", "speechmos", "librosa", "soundfile",
            "scipy", "scikit-learn", "pandas", "tqdm")

REPO_DIR = "/kaggle/working/vox-profile-release"
if not os.path.exists(REPO_DIR):
    subprocess.run(["git", "clone", "--depth", "1",
                    "https://github.com/tiantiaf0627/vox-profile-release.git", REPO_DIR], check=True)
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

# %% [markdown]
# ## 2. Dựng WavLM (như exp08) → NẠP trọng số backbone từ checkpoint
# Dựng đúng kiến trúc (SAILER wrapper → lấy backbone HF; fallback WavLM trắng), rồi `load_state_dict`
# bằng `ckpt["wavlm"]` → khôi phục đúng trạng thái fine-tune đã lưu.

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F

device = DEVICE if torch.cuda.is_available() else "cpu"
print("Device:", device, ("✅ " + torch.cuda.get_device_name(0)) if device == "cuda" else "⚠️ CPU (rất chậm!)")

ckpt = torch.load(RESUME_CKPT, map_location="cpu", weights_only=False)   # ckpt có numpy (vad_mu) → cần False
assert "wavlm" in ckpt, ("❌ Checkpoint KHÔNG có 'wavlm' (backbone). Đây là bản ft_emotion_meta.pt CŨ "
                         "chỉ lưu heads → không resume được. Hãy dùng ft_emotion_full.pt.")
print("✅ Nạp checkpoint:", RESUME_CKPT, "| keys:", list(ckpt.keys()))

def find_hf_backbone(module):
    cands = []
    for name, m in module.named_modules():
        enc = getattr(m, "encoder", None)
        if getattr(m, "feature_extractor", None) is not None and enc is not None \
                and getattr(enc, "layers", None) is not None:
            cands.append((name, m))
    if not cands:
        return None, None
    cands.sort(key=lambda nm: sum(p.numel() for p in nm[1].parameters()), reverse=True)
    return cands[0]

wavlm = None
try:
    from src.model.emotion.wavlm_emotion import WavLMWrapper   # noqa: E402
    _wrapper = WavLMWrapper.from_pretrained("tiantiaf/wavlm-large-categorical-emotion")
    name, wavlm = find_hf_backbone(_wrapper)
    if wavlm is not None:
        print(f"✅ Dựng backbone WavLM từ SAILER wrapper tại '.{name}'")
except Exception as e:
    print("⚠️ Lỗi nạp SAILER wrapper:", repr(e), "→ fallback WavLM trắng.")

if wavlm is None:
    from transformers import WavLMModel
    wavlm = WavLMModel.from_pretrained("microsoft/wavlm-large")
    print("ℹ️ Fallback: microsoft/wavlm-large.")

wavlm = wavlm.to(device)
WAVLM_DIM = int(wavlm.config.hidden_size)

# Nạp trọng số đã fine-tune từ checkpoint (đè lên kiến trúc vừa dựng)
miss, unexp = wavlm.load_state_dict(ckpt["wavlm"], strict=False)
print(f"🔁 load wavlm từ checkpoint: thiếu {len(miss)} / dư {len(unexp)} key (kỳ vọng ~0).")
if len(miss) > 20 or len(unexp) > 20:
    print("   ⚠️ Lệch key nhiều → kiến trúc có thể không khớp checkpoint. Kiểm tra UNFREEZE/USE_AUDEERING.")

# Đóng băng partial: chỉ mở UNFREEZE_TOP_LAYERS lớp trên
for p in wavlm.parameters():
    p.requires_grad = False
enc_layers = wavlm.encoder.layers
n_layers = len(enc_layers)
for layer in enc_layers[max(0, n_layers - UNFREEZE_TOP_LAYERS):]:
    for p in layer.parameters():
        p.requires_grad = True
n_train = sum(p.numel() for p in wavlm.parameters() if p.requires_grad)
print(f"WavLM: {n_layers} lớp · mở băng {min(UNFREEZE_TOP_LAYERS, n_layers)} → {n_train/1e6:.1f}M param train (dim {WAVLM_DIM})")

if USE_GRAD_CKPT:
    wavlm.gradient_checkpointing_enable()
    if hasattr(wavlm, "enable_input_require_grads"):
        wavlm.enable_input_require_grads()

def masked_mean(hidden, attn_mask):
    if attn_mask is None:
        return hidden.mean(dim=1)
    try:
        fm = wavlm._get_feature_vector_attention_mask(hidden.shape[1], attn_mask)
    except Exception:
        return hidden.mean(dim=1)
    fm = fm.unsqueeze(-1).to(hidden.dtype)
    return (hidden * fm).sum(1) / fm.sum(1).clamp(min=1e-6)

def wavlm_embed(input_values, attn_mask):
    out = wavlm(input_values, attention_mask=attn_mask).last_hidden_state
    return masked_mean(out, attn_mask)

# %% [markdown]
# ## 3. audeering FROZEN (đặc trưng phụ) — dùng cache nếu có

# %%
import numpy as np
import librosa
from tqdm.auto import tqdm

AUD_DIM = 0
aud_backbone = aud_head = aud_proc = None
if USE_AUDEERING:
    from transformers import Wav2Vec2Model, Wav2Vec2Config, Wav2Vec2Processor
    from huggingface_hub import hf_hub_download
    AUD_NAME = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
    aud_proc = Wav2Vec2Processor.from_pretrained(AUD_NAME)
    aud_cfg = Wav2Vec2Config.from_pretrained(AUD_NAME)
    aud_backbone = Wav2Vec2Model(aud_cfg)
    try:
        _sd = __import__("safetensors.torch", fromlist=["load_file"]).load_file(
            hf_hub_download(AUD_NAME, "model.safetensors"))
    except Exception:
        _sd = torch.load(hf_hub_download(AUD_NAME, "pytorch_model.bin"), map_location="cpu")
    bb_sd = {k[len("wav2vec2."):]: v for k, v in _sd.items() if k.startswith("wav2vec2.")}
    aud_backbone.load_state_dict(bb_sd, strict=False)
    _hid = _sd["classifier.dense.weight"].shape[0]
    _out = _sd["classifier.out_proj.weight"].shape[0]
    aud_head = nn.Sequential(nn.Linear(_hid, _hid), nn.Tanh(), nn.Linear(_hid, _out))
    aud_head[0].weight.data.copy_(_sd["classifier.dense.weight"]); aud_head[0].bias.data.copy_(_sd["classifier.dense.bias"])
    aud_head[2].weight.data.copy_(_sd["classifier.out_proj.weight"]); aud_head[2].bias.data.copy_(_sd["classifier.out_proj.bias"])
    aud_backbone = aud_backbone.to(device).eval()
    aud_head = aud_head.to(device).eval()
    AUD_DIM = _hid + 3
    print(f"✅ audeering frozen ({AUD_DIM}-D)")

def load_wav(name_or_stem):
    p = name_or_stem if os.path.isabs(str(name_or_stem)) else os.path.join(
        WAV_DIR, name_or_stem if str(name_or_stem).endswith(".wav") else str(name_or_stem) + ".wav")
    if not os.path.exists(p):
        return None
    wave, _ = librosa.load(p, sr=SR, mono=True)
    return wave[: MAX_SECONDS * SR].astype(np.float32)

@torch.no_grad()
def extract_audeering(stems, tag):
    if not USE_AUDEERING:
        return {}
    cache_path = os.path.join(CACHE_DIR, f"aud_{tag}.npz")
    store = {}
    if os.path.exists(cache_path):
        z = np.load(cache_path, allow_pickle=True)
        store = {k: z[k] for k in z.files}
        print(f"[aud/{tag}] nạp cache: {len(store)}")
    todo = [s for s in stems if s not in store]
    for i, s in enumerate(tqdm(todo, desc=f"audeering {tag}")):
        wave = load_wav(s)
        if wave is None:
            continue
        x = aud_proc(wave, sampling_rate=SR).input_values[0]
        x = torch.from_numpy(np.asarray(x, dtype=np.float32)).unsqueeze(0).to(device)
        h = aud_backbone(x)[0].mean(dim=1)
        out = aud_head(h)[0].cpu().numpy()
        vad = np.array([1 + 4 * out[2], 1 + 4 * out[0], 1 + 4 * out[1]], dtype=np.float32)
        store[s] = np.concatenate([h[0].cpu().numpy(), vad]).astype(np.float32)
        if (i + 1) % 500 == 0:
            np.savez(cache_path, **store)
    if todo:
        np.savez(cache_path, **store)
    return store

# %% [markdown]
# ## 4. Đọc & gộp nhãn theo wavID

# %%
import pandas as pd

def load_target_emotions():
    tgt = {}
    with open(METADATA_CSV, encoding="utf-8") as f:
        for ln in f:
            parts = ln.strip().split("|")
            if len(parts) >= 2:
                tgt[stem(parts[0])] = norm_emotion(parts[1])
    return tgt

def _col(cols_map, *names, df=None, default_idx=None):
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
    wav_col = _col(cols, "wavid", "wav", df=df, default_idx=1)
    emos_col = _col(cols, "emos", "emo", "emomos")
    val_col = _col(cols, "val", "valence"); aro_col = _col(cols, "aro", "arousal"); dom_col = _col(cols, "dom", "dominance")
    cat_col = _col(cols, "emocat", "cat", "emotion")
    assert emos_col, f"Không thấy cột eMOS (cột: {list(df.columns)})"
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
        cat = votes / s if s > 0 else np.full(len(EMOTIONS5), 0.2, dtype=np.float32)
        for i in range(len(EMOTIONS5)):
            rec[f"cat{i}"] = float(cat[i])
        rows.append(rec)
    return pd.DataFrame(rows)

target_map = load_target_emotions()
train_df = load_train_labels()
HAS_VAD = bool(train_df["val"].notna().any())
print(f"Target: {len(target_map)} | wav train (gộp): {len(train_df)} | có VAD: {HAS_VAD}")

# %% [markdown]
# ## 5. Dataset/loader — DÙNG thống kê chuẩn hóa TỪ CHECKPOINT (để khớp head đã train)

# %%
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

train_stems = [s for s in train_df["wavID"] if target_map.get(s) is not None]
if LIMIT_TRAIN:
    train_stems = train_stems[:LIMIT_TRAIN]
aud_tr = extract_audeering(train_stems, "train")

lab = train_df.set_index("wavID")

# QUAN TRỌNG: lấy mean/std từ checkpoint (head đã train theo thang này) thay vì tính lại.
emos_mu = float(ckpt["emos_mu"]); emos_sd = float(ckpt["emos_sd"])
vad_mu = np.asarray(ckpt["vad_mu"], dtype=np.float32); vad_sd = np.asarray(ckpt["vad_sd"], dtype=np.float32)
print(f"Dùng chuẩn hóa từ ckpt: emos μ={emos_mu:.3f} σ={emos_sd:.3f} | vad μ={np.round(vad_mu,2)}")

def onehot_target(tgt):
    v = np.zeros(len(EMOTIONS5), dtype=np.float32)
    if tgt in EMOTIONS5:
        v[EMOTIONS5.index(tgt)] = 1.0
    return v

class EmoDataset(Dataset):
    def __init__(self, stems):
        self.stems = [s for s in stems if (load_wav(s) is not None) and ((not USE_AUDEERING) or s in aud_tr)]
    def __len__(self):
        return len(self.stems)
    def __getitem__(self, i):
        s = self.stems[i]
        wave = load_wav(s)
        emos = (float(lab.loc[s, "emos"]) - emos_mu) / emos_sd
        if HAS_VAD:
            vad = (np.array([lab.loc[s, "val"], lab.loc[s, "aro"], lab.loc[s, "dom"]], np.float32) - vad_mu) / vad_sd
        else:
            vad = np.zeros(3, dtype=np.float32)
        cat = np.array([lab.loc[s, f"cat{j}"] for j in range(len(EMOTIONS5))], dtype=np.float32)
        aud = aud_tr[s] if USE_AUDEERING else np.zeros(0, dtype=np.float32)
        return {"wave": wave, "tgt": onehot_target(target_map.get(s)), "aud": aud,
                "emos": np.float32(emos), "vad": vad, "cat": cat,
                "emos_raw": np.float32(lab.loc[s, "emos"]),
                "vad_raw": np.array([lab.loc[s, "val"], lab.loc[s, "aro"], lab.loc[s, "dom"]], np.float32)}

def collate(batch):
    lens = [len(b["wave"]) for b in batch]
    L = max(lens)
    waves = np.zeros((len(batch), L), dtype=np.float32)
    mask = np.zeros((len(batch), L), dtype=np.float32)
    for i, b in enumerate(batch):
        waves[i, : len(b["wave"])] = b["wave"]; mask[i, : len(b["wave"])] = 1.0
    return {
        "input_values": torch.from_numpy(waves), "attn_mask": torch.from_numpy(mask).long(),
        "tgt": torch.from_numpy(np.stack([b["tgt"] for b in batch])),
        "aud": torch.from_numpy(np.stack([b["aud"] for b in batch])) if USE_AUDEERING else None,
        "emos": torch.from_numpy(np.stack([b["emos"] for b in batch])).unsqueeze(1),
        "vad": torch.from_numpy(np.stack([b["vad"] for b in batch])),
        "cat": torch.from_numpy(np.stack([b["cat"] for b in batch])),
        "emos_raw": np.stack([b["emos_raw"] for b in batch]),
        "vad_raw": np.stack([b["vad_raw"] for b in batch]),
    }

ds = EmoDataset(train_stems)
print("Dataset hợp lệ:", len(ds), "wav")
tr_i, va_i = train_test_split(np.arange(len(ds)), test_size=VAL_FRAC, random_state=SEED)
tr_loader = DataLoader(torch.utils.data.Subset(ds, tr_i), batch_size=BATCH, shuffle=True, collate_fn=collate, num_workers=2)
va_loader = DataLoader(torch.utils.data.Subset(ds, va_i), batch_size=BATCH, shuffle=False, collate_fn=collate, num_workers=2)

# %% [markdown]
# ## 6. Heads (NẠP từ checkpoint) + optimizer + train TIẾP

# %%
from scipy.stats import spearmanr

torch.manual_seed(SEED); np.random.seed(SEED)
N_EMO = len(EMOTIONS5)
TRUNK_IN = WAVLM_DIM + (AUD_DIM if USE_AUDEERING else 0)

class EmoHeads(nn.Module):
    def __init__(self, d_in, trunk_h, head_h, p, n_emo):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(d_in, trunk_h), nn.ReLU(), nn.Dropout(p),
                                   nn.Linear(trunk_h, trunk_h), nn.ReLU(), nn.Dropout(p))
        self.emos = nn.Sequential(nn.Linear(trunk_h + n_emo, head_h), nn.ReLU(), nn.Dropout(p), nn.Linear(head_h, 1))
        self.cat = nn.Sequential(nn.Linear(trunk_h, head_h), nn.ReLU(), nn.Dropout(p), nn.Linear(head_h, n_emo))
        self.vad = nn.Sequential(nn.Linear(trunk_h, head_h), nn.ReLU(), nn.Dropout(p), nn.Linear(head_h, 3))
    def forward(self, feat, tgt):
        h = self.trunk(feat)
        return self.emos(torch.cat([h, tgt], 1)), self.cat(h), self.vad(h)

heads = EmoHeads(TRUNK_IN, TRUNK_HIDDEN, HEAD_HIDDEN, DROPOUT, N_EMO).to(device)
hmiss, hunexp = heads.load_state_dict(ckpt["heads"], strict=False)
print(f"🔁 load heads từ checkpoint: thiếu {len(hmiss)} / dư {len(hunexp)} key (kỳ vọng 0).")
print(f"Trunk input = {TRUNK_IN} (wavlm {WAVLM_DIM} + aud {AUD_DIM if USE_AUDEERING else 0})")

TASKS = ["emos", "cat", "val", "aro", "dom"]
log_var = nn.Parameter(torch.zeros(len(TASKS), device=device))
bb_params = [p for p in wavlm.parameters() if p.requires_grad]
head_params = list(heads.parameters()) + ([log_var] if USE_UNCERTAINTY else [])
opt = torch.optim.AdamW([
    {"params": bb_params, "lr": LR_BACKBONE * RESUME_LR_SCALE},
    {"params": head_params, "lr": LR_HEAD * RESUME_LR_SCALE},
], weight_decay=WEIGHT_DECAY)
scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP and device == "cuda")
mse = nn.MSELoss()

def soft_ce(logits, target_dist):
    return -(target_dist * F.log_softmax(logits, dim=1)).sum(1).mean()

def forward_batch(b):
    feat_wavlm = wavlm_embed(b["input_values"].to(device), b["attn_mask"].to(device))
    feat = torch.cat([feat_wavlm, b["aud"].to(device)], dim=1) if USE_AUDEERING else feat_wavlm
    return heads(feat, b["tgt"].to(device))

def compute_loss(emos_p, cat_l, vad_p, b):
    L = {}
    L["emos"] = mse(emos_p, b["emos"].to(device))
    L["cat"] = soft_ce(cat_l, b["cat"].to(device))
    if HAS_VAD:
        vt = b["vad"].to(device)
        L["val"] = mse(vad_p[:, 0:1], vt[:, 0:1]); L["aro"] = mse(vad_p[:, 1:2], vt[:, 1:2]); L["dom"] = mse(vad_p[:, 2:3], vt[:, 2:3])
    else:
        z = torch.zeros((), device=device); L["val"] = L["aro"] = L["dom"] = z
    if USE_UNCERTAINTY:
        return sum(torch.exp(-log_var[i]) * L[t] + log_var[i] for i, t in enumerate(TASKS))
    return sum(L.values())

@torch.no_grad()
def evaluate():
    wavlm.eval(); heads.eval()
    P = {"emos": [], "val": [], "aro": [], "dom": []}; Y = {"emos": [], "val": [], "aro": [], "dom": []}
    catP, catY = [], []
    for b in va_loader:
        with torch.cuda.amp.autocast(enabled=USE_AMP and device == "cuda"):
            emos_p, cat_l, vad_p = forward_batch(b)
        P["emos"] += emos_p.float().cpu().numpy().ravel().tolist(); Y["emos"] += b["emos_raw"].tolist()
        vad_p = vad_p.float().cpu().numpy()
        for j, t in enumerate(["val", "aro", "dom"]):
            P[t] += vad_p[:, j].tolist(); Y[t] += b["vad_raw"][:, j].tolist()
        catP.append(F.softmax(cat_l, 1).float().cpu().numpy()); catY.append(b["cat"])
    out = {}
    for t in ["emos"] + (["val", "aro", "dom"] if HAS_VAD else []):
        out[t] = spearmanr(P[t], Y[t]).correlation
    q = np.concatenate(catP); p = np.concatenate(catY)
    out["cat_err"] = float(np.abs(q - p).sum(1).mean())
    return out

def mean_srcc(m):
    keys = ["emos"] + (["val", "aro", "dom"] if HAS_VAD else [])
    return float(np.mean([m[k] for k in keys]))

# Init best TỪ checkpoint hiện tại → chỉ lưu nếu train tiếp TỐT HƠN
m0 = evaluate(); best = mean_srcc(m0)
best_state = {"wavlm": {k: v.cpu().clone() for k, v in wavlm.state_dict().items()},
              "heads": {k: v.cpu().clone() for k, v in heads.state_dict().items()}}
print(f"📍 Checkpoint hiện tại: mean SRCC = {best:.4f} | "
      + " ".join(f"{k}={m0[k]:.3f}" for k in ['emos','val','aro','dom'] if k in m0))

bad = 0
for ep in range(1, EPOCHS + 1):
    wavlm.train(); heads.train()
    opt.zero_grad(); run = 0.0; nb = 0
    for step, b in enumerate(tqdm(tr_loader, desc=f"+epoch {ep}")):
        with torch.cuda.amp.autocast(enabled=USE_AMP and device == "cuda"):
            emos_p, cat_l, vad_p = forward_batch(b)
            loss = compute_loss(emos_p, cat_l, vad_p, b) / ACCUM
        scaler.scale(loss).backward()
        if (step + 1) % ACCUM == 0:
            scaler.step(opt); scaler.update(); opt.zero_grad()
        run += loss.item() * ACCUM; nb += 1
    m = evaluate(); sc = mean_srcc(m)
    msg = " ".join(f"{k}={m[k]:.3f}" for k in ["emos", "val", "aro", "dom"] if k in m)
    print(f"+epoch {ep:2d} | loss {run/max(nb,1):.4f} | {msg} | cat_err {m['cat_err']:.3f} | mean {sc:.4f} (best {max(best,sc):.4f})")
    if sc > best:
        best = sc
        best_state = {"wavlm": {k: v.cpu().clone() for k, v in wavlm.state_dict().items()},
                      "heads": {k: v.cpu().clone() for k, v in heads.state_dict().items()}}
        bad = 0
    else:
        bad += 1
        if bad >= PATIENCE:
            print(f"Early stop (resume) ở +epoch {ep}."); break

wavlm.load_state_dict(best_state["wavlm"]); heads.load_state_dict(best_state["heads"])
final = evaluate()
print("\n✅ VAL sau resume:")
print(f"   EMOS={final['emos']:.4f} (ckpt {m0['emos']:.3f} · exp08 nộp {EXP08['emos']})")
if HAS_VAD:
    print(f"   VAL/ARO/DOM={final['val']:.4f}/{final['aro']:.4f}/{final['dom']:.4f} "
          f"(exp08 nộp {EXP08['val']}/{EXP08['aro']}/{EXP08['dom']})")
print(f"   mean SRCC: ckpt {mean_srcc(m0):.4f} → sau resume {mean_srcc(final):.4f} "
      + ("🚀 cải thiện" if mean_srcc(final) > mean_srcc(m0) + 1e-4 else "➖ không cải thiện (giữ ckpt cũ)"))

torch.save({"wavlm": best_state["wavlm"], "heads": best_state["heads"],
            "emos_mu": emos_mu, "emos_sd": emos_sd, "vad_mu": vad_mu, "vad_sd": vad_sd,
            "WAVLM_DIM": WAVLM_DIM, "AUD_DIM": AUD_DIM, "UNFREEZE_TOP_LAYERS": UNFREEZE_TOP_LAYERS,
            "val_emos": final["emos"]}, os.path.join(OUT_DIR, "ft_emotion_full.pt"))
print("Đã lưu FULL (có backbone):", os.path.join(OUT_DIR, "ft_emotion_full.pt"))

# %% [markdown]
# ## 7. Dự đoán DEV → answer.txt (5 cột cảm xúc từ resume; QMOS mượn exp07 hoặc UTMOSv2)

# %%
def list_dev():
    with open(DEV_SCP) as f:
        return [ln.strip() for ln in f if ln.strip()]

dev_names = list_dev()
if LIMIT_DEV:
    dev_names = dev_names[:LIMIT_DEV]
dev_stems = [stem(n) for n in dev_names]
print("DEV:", len(dev_names), "mẫu")
aud_dev = extract_audeering(dev_stems, "dev")

def load_exp07_qmos():
    if EXP07_ANSWER and os.path.exists(EXP07_ANSWER):
        import csv
        d = {}
        with open(EXP07_ANSWER) as f:
            for row in csv.DictReader(f):
                d[row["wav"]] = float(row["QMOS"]); d[stem(row["wav"])] = float(row["QMOS"])
        print(f"✅ Mượn QMOS từ exp07 ({EXP07_ANSWER}): {len(d)//2} wav")
        return d
    return None

qmos_map = load_exp07_qmos()
if qmos_map is None:
    print("ℹ️ Không có answer.txt exp07 → chấm QMOS bằng UTMOSv2 (T05, vô địch VMC2024).")
    pip_install("git+https://github.com/sarulab-speech/UTMOSv2.git")
    import utmosv2
    v2 = utmosv2.create_model(pretrained=True)
    qmos_map = {}
    for n in tqdm(dev_names, desc="UTMOSv2"):
        wav = os.path.join(WAV_DIR, n if str(n).endswith(".wav") else str(n) + ".wav")
        if not os.path.exists(wav):
            continue
        out = v2.predict(input_path=wav)
        qmos_map[n] = float(out["predicted_mos"]) if isinstance(out, dict) else float(out)
    del v2; torch.cuda.empty_cache() if device == "cuda" else None

@torch.no_grad()
def predict_emotion(sid):
    wave = load_wav(sid)
    if wave is None or (USE_AUDEERING and sid not in aud_dev):
        return None
    wavlm.eval(); heads.eval()
    iv = torch.from_numpy(wave).unsqueeze(0).to(device)
    am = torch.ones((1, len(wave)), dtype=torch.long, device=device)
    tgt = torch.from_numpy(onehot_target(target_map.get(sid))).unsqueeze(0).to(device)
    with torch.cuda.amp.autocast(enabled=USE_AMP and device == "cuda"):
        fw = wavlm_embed(iv, am)
        feat = torch.cat([fw, torch.from_numpy(aud_dev[sid]).unsqueeze(0).to(device)], dim=1) if USE_AUDEERING else fw
        emos_p, cat_l, vad_p = heads(feat, tgt)
    emos = float(emos_p.item()) * emos_sd + emos_mu
    cat5 = F.softmax(cat_l, 1)[0].float().cpu().numpy()
    vad3 = vad_p[0].float().cpu().numpy() * vad_sd + vad_mu
    return emos, cat5, vad3

def fmt_cat(p5):
    return "|".join(f"{e}:{p5[i]:.6g}" for i, e in enumerate(EMOTIONS5))

def build_answer(out_path):
    n_real = n_def = 0
    with open(out_path, "w") as f:
        f.write("wav,QMOS,EMOS,CAT,VAL,ARO,DOM\n")
        for name in tqdm(dev_names, desc="answer"):
            sid = stem(name)
            pr = predict_emotion(sid)
            if pr is None:
                emos, cat5, vad3 = 3.0, np.full(5, 0.2, np.float32), np.array([3.0, 3.0, 3.0]); n_def += 1
            else:
                emos, cat5, vad3 = pr; n_real += 1
            qmos = qmos_map.get(name, qmos_map.get(sid, 3.0))
            f.write(f"{name},{qmos:.6g},{emos:.6g},{fmt_cat(cat5)},{vad3[0]:.6g},{vad3[1]:.6g},{vad3[2]:.6g}\n")
    print(f"Ghi {len(dev_names)} dòng → {out_path} | cảm xúc thật {n_real}, mặc định {n_def}")

answer_path = os.path.join(OUT_DIR, "answer.txt")
build_answer(answer_path)

# %% [markdown]
# ## 8. Validate + zip

# %%
def validate(path):
    import csv
    with open(path) as f:
        rows = list(csv.reader(f))
    assert rows[0][0] == "wav" and "QMOS" in rows[0], "Header sai"
    for i, r in enumerate(rows[1:], 2):
        assert len(r) == len(rows[0]), f"Dòng {i} sai số cột"
    print(f"OK: {len(rows)-1} dòng, header = {rows[0]}")

validate(answer_path)
os.system(f"cd {OUT_DIR} && zip -j submission_track2_exp08_resume.zip answer.txt && unzip -l submission_track2_exp08_resume.zip")
print("Sẵn sàng nộp:", os.path.join(OUT_DIR, "submission_track2_exp08_resume.zip"))

# %% [markdown]
# ## Ghi chú
# - **Đầu vào bắt buộc:** `RESUME_CKPT` = `ft_emotion_full.pt` (CÓ backbone). Bản `ft_emotion_meta.pt` cũ chỉ
#   có heads → cell 2 sẽ assert lỗi nhắc dùng file đủ.
# - **Cache:** trỏ `CACHE_INPUT` tới dataset chứa `aud_train.npz`/`aud_dev.npz` → khỏi trích lại audeering.
#   Nếu LIMIT khác lần trước, cache thiếu stem nào sẽ tự trích bù (resume theo stem).
# - **Chuẩn hóa lấy TỪ checkpoint** (`emos_mu/sd`, `vad_mu/sd`) → khớp thang head đã train (đừng tính lại).
# - **best init từ checkpoint** → chỉ lưu nếu train tiếp THỰC SỰ tốt hơn (không sợ tụt).
# - Nếu val chững: đặt `RESUME_LR_SCALE=0.5` (giảm LR) hoặc tăng `UNFREEZE_TOP_LAYERS` (lưu ý: mở thêm lớp
#   thì lớp mới chưa được train trong checkpoint → cần nhiều epoch hơn).
# - QMOS: tốt nhất Add Input `answer.txt` exp07 (0.548). Để trộn cột chuẩn, xem kết quả exp08: 5 cột cảm xúc
#   resume + QMOS exp07 → hệ thống mạnh nhất 6 cột.
# - Ghi config → kết quả → nhận xét vào `docs/04_experiments_log.md`.
