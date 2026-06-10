# %% [markdown]
# # VMC2026 Track 2 — exp12 (WavLM: SCRATCH vs BASE vs SAILER — ablation khởi tạo) — Kaggle T4
#
# **Mục đích:** kiểm chứng giả thuyết của mentor — *"với 12k data, train from scratch có tốt hơn fine-tune không?"*
# Một notebook, đổi cờ `INIT_MODE` để chạy 3 cách khởi tạo backbone WavLM, so trên CÙNG kiến trúc/data:
#
# | INIT_MODE | Khởi tạo WavLM | Train gì | Ý nghĩa |
# |---|---|---|---|
# | `scratch` | **ngẫu nhiên** (không pretrain) | **toàn bộ** backbone | "from scratch" đúng nghĩa mentor nói |
# | `base`    | microsoft/wavlm-large (pretrain SSL, KHÔNG cảm xúc) | mở băng N lớp trên | đo lợi ích của SAILER warm-start |
# | `sailer`  | warm-start cảm xúc (như exp08) | mở băng N lớp trên | bản mạnh hiện tại |
#
# **Chỉ WavLM** (bỏ audeering) để cô lập đúng biến "khởi tạo". QMOS mượn exp07 / UTMOSv2.
#
# ## ⚠️ Kỳ vọng trung thực (để đọc kết quả đúng)
# - `scratch` gần như CHẮC CHẮN yếu hơn `base`/`sailer`: 12k mẫu là quá ít để dạy WavLM "nghe" từ đầu
#   (SSL pretrain dùng ~94.000 GIỜ audio). Đây là ablation để **chứng minh bằng số**, không phải để vượt.
# - `scratch` phải mở băng TOÀN BỘ (mới có gì để học) → **nặng + chậm + dễ OOM** trên T4. Dùng LIMIT nhỏ trước.
# - So sánh bằng **VAL nội bộ** giữa 3 mode đã đủ kết luận; muốn chắc thì nộp mode tốt nhất lên DEV.
#
# **Cách chạy:** GPU T4 + Internet On → sửa cell 0 (`INIT_MODE` + slug) → Run All. Chạy 3 lần đổi INIT_MODE.

# %% [markdown]
# ## 0. Cấu hình — SỬA Ở ĐÂY

# %%
import os

INIT_MODE = "sailer"   # << "scratch" | "base" | "sailer"  (đổi rồi chạy lại để so) — "sailer" = WavLM warm-start cảm xúc

DATA_ROOT    = "/kaggle/input/datasets/minhtoan2/vmc2026-track2-full"   # << SỬA slug cho khớp Add Input
WAV_DIR      = f"{DATA_ROOT}/wav"
METADATA_CSV = f"{DATA_ROOT}/metadata.csv"
TRAIN_CSV    = f"{DATA_ROOT}/sets/train.csv"
DEV_SCP      = f"{DATA_ROOT}/sets/dev.scp"

EXP07_ANSWER = "/kaggle/input/exp07-answer/answer.txt"   # << (tùy chọn) mượn QMOS 0.548; không có → UTMOSv2
OUT_DIR      = "/kaggle/working"

# ── Siêu tham số ─────────────────────────────────────────────────────────────
DEVICE          = "cuda"
SR              = 16000
MAX_SECONDS     = 6
TRUNK_HIDDEN    = 512
HEAD_HIDDEN     = 128
DROPOUT         = 0.3
WEIGHT_DECAY    = 1e-5
EPOCHS          = 15
PATIENCE        = 5
BATCH           = 4
ACCUM           = 8
VAL_FRAC        = 0.10
SEED            = 42
USE_AMP         = True
USE_GRAD_CKPT   = True
USE_UNCERTAINTY = True

# Khởi tạo & LR & mở băng — TỰ đặt theo INIT_MODE (scratch cần LR lớn + mở băng toàn bộ)
if INIT_MODE == "scratch":
    UNFREEZE_TOP_LAYERS = "all"     # random init → phải train tất cả mới học được
    LR_BACKBONE = 1e-4              # random init cần bước lớn hơn fine-tune
elif INIT_MODE in ("base", "sailer"):
    UNFREEZE_TOP_LAYERS = 6         # fine-tune: chỉ mở băng N lớp trên (tiết kiệm VRAM, chống overfit)
    LR_BACKBONE = 1e-5
else:
    raise ValueError(f"INIT_MODE lạ: {INIT_MODE}")
LR_HEAD = 1e-3

LIMIT_TRAIN = 300    # << LẦN ĐẦU 300; chạy thật None
LIMIT_DEV   = 20     # << LẦN ĐẦU 20; chạy thật None

EXP08 = {"emos": 0.811, "cat_err": 0.133, "val": 0.659, "aro": 0.793, "dom": 0.751}  # mốc DEV để tham khảo

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

print(f"INIT_MODE = {INIT_MODE} | UNFREEZE = {UNFREEZE_TOP_LAYERS} | LR_BACKBONE = {LR_BACKBONE}")
print("DATA_ROOT:", DATA_ROOT)
for p in [WAV_DIR, METADATA_CSV, TRAIN_CSV, DEV_SCP]:
    print(("  ✅ " if os.path.exists(p) else "  ❌ THIẾU ") + p)

# %% [markdown]
# ## 1. Cài đặt (clone SAILER chỉ khi INIT_MODE='sailer')

# %%
import sys, subprocess
import numpy as _np

# ⚠️ KHÓA numpy = bản Kaggle đang có → pip KHÔNG được nâng/hạ numpy → tránh "SystemError: bad call flags"
# (lỗi import torch do numpy lệch phiên bản với torch đã biên dịch sẵn).
_NPIN = f"numpy=={_np.__version__}"
print("Khóa numpy ở:", _NPIN)

def pip_install(*pkgs):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs, _NPIN], check=True)

# Kaggle đã có sẵn torch/transformers/librosa/scipy/sklearn/pandas/tqdm/huggingface_hub/safetensors.
# Chỉ cài thêm vài gói speech còn thiếu (kèm khóa numpy ở trên).
pip_install("loralib", "speechmos", "soundfile")
if INIT_MODE == "sailer":
    pip_install("speechbrain")

if INIT_MODE == "sailer":
    REPO_DIR = "/kaggle/working/vox-profile-release"
    if not os.path.exists(REPO_DIR):
        subprocess.run(["git", "clone", "--depth", "1",
                        "https://github.com/tiantiaf0627/vox-profile-release.git", REPO_DIR], check=True)
    if REPO_DIR not in sys.path:
        sys.path.insert(0, REPO_DIR)

# %% [markdown]
# ## 2. Dựng WavLM theo INIT_MODE

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = DEVICE if torch.cuda.is_available() else "cpu"
print("Device:", device, ("✅ " + torch.cuda.get_device_name(0)) if device == "cuda" else "⚠️ CPU (rất chậm!)")

from transformers import WavLMModel, WavLMConfig

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
if INIT_MODE == "scratch":
    # Random init NHƯNG giữ ĐÚNG kiến trúc large (để công bằng với base/sailer)
    cfg = WavLMConfig.from_pretrained("microsoft/wavlm-large")
    wavlm = WavLMModel(cfg)   # KHÔNG load trọng số → ngẫu nhiên
    print("🎲 WavLM-large khởi tạo NGẪU NHIÊN (from scratch, không pretrain).")
elif INIT_MODE == "base":
    wavlm = WavLMModel.from_pretrained("microsoft/wavlm-large")
    print("📦 WavLM-large pretrain SSL (chưa học cảm xúc).")
elif INIT_MODE == "sailer":
    try:
        from src.model.emotion.wavlm_emotion import WavLMWrapper   # noqa: E402
        _wrapper = WavLMWrapper.from_pretrained("tiantiaf/wavlm-large-categorical-emotion")
        name, wavlm = find_hf_backbone(_wrapper)
        print(f"🔥 WavLM warm-start SAILER (cảm xúc) tại '.{name}'")
    except Exception as e:
        print("⚠️ Lỗi nạp SAILER:", repr(e), "→ fallback base pretrained.")
        wavlm = WavLMModel.from_pretrained("microsoft/wavlm-large")

wavlm = wavlm.to(device)
WAVLM_DIM = int(wavlm.config.hidden_size)
wavlm.config.layerdrop = 0.0   # ⚠️ BẮT BUỘC khi dùng gradient-checkpointing (tránh CheckpointError do bỏ lớp ngẫu nhiên)

# Mở băng theo cấu hình
if UNFREEZE_TOP_LAYERS == "all":
    for p in wavlm.parameters():
        p.requires_grad = True
    n_open = "ALL"
else:
    for p in wavlm.parameters():
        p.requires_grad = False
    _wl = wavlm.encoder.layers
    for layer in _wl[max(0, len(_wl) - UNFREEZE_TOP_LAYERS):]:
        for p in layer.parameters():
            p.requires_grad = True
    n_open = f"top {min(UNFREEZE_TOP_LAYERS, len(_wl))}/{len(_wl)}"
print(f"WavLM mở băng: {n_open} → {sum(p.numel() for p in wavlm.parameters() if p.requires_grad)/1e6:.1f}M param train (dim {WAVLM_DIM})")

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
# ## 3. Đọc & gộp nhãn theo wavID

# %%
import librosa
import pandas as pd
from tqdm.auto import tqdm

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
# ## 4. Dataset/loader (chỉ raw wave cho WavLM)

# %%
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

train_stems = [s for s in train_df["wavID"] if target_map.get(s) is not None]
if LIMIT_TRAIN:
    train_stems = train_stems[:LIMIT_TRAIN]
lab = train_df.set_index("wavID")

def _zfit(a):
    a = np.asarray(a, dtype=np.float32); return float(np.nanmean(a)), float(np.nanstd(a) + 1e-6)
emos_mu, emos_sd = _zfit([lab.loc[s, "emos"] for s in train_stems])
if HAS_VAD:
    vad_mu = np.array([_zfit([lab.loc[s, c] for s in train_stems])[0] for c in ["val", "aro", "dom"]], np.float32)
    vad_sd = np.array([_zfit([lab.loc[s, c] for s in train_stems])[1] for c in ["val", "aro", "dom"]], np.float32)
else:
    vad_mu = np.zeros(3, np.float32); vad_sd = np.ones(3, np.float32)
print(f"Chuẩn hóa: emos μ={emos_mu:.3f} σ={emos_sd:.3f}")

def onehot_target(tgt):
    v = np.zeros(len(EMOTIONS5), dtype=np.float32)
    if tgt in EMOTIONS5:
        v[EMOTIONS5.index(tgt)] = 1.0
    return v

def load_wav(sid):
    p = os.path.join(WAV_DIR, sid if str(sid).endswith(".wav") else str(sid) + ".wav")
    if not os.path.exists(p):
        return None
    wave, _ = librosa.load(p, sr=SR, mono=True)
    return wave[: MAX_SECONDS * SR].astype(np.float32)

class EmoDataset(Dataset):
    def __init__(self, stems):
        self.stems = [s for s in stems if load_wav(s) is not None]
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
        return {"wave": wave, "tgt": onehot_target(target_map.get(s)),
                "emos": np.float32(emos), "vad": vad, "cat": cat,
                "emos_raw": np.float32(lab.loc[s, "emos"]),
                "vad_raw": np.array([lab.loc[s, "val"], lab.loc[s, "aro"], lab.loc[s, "dom"]], np.float32)}

def collate(batch):
    L = max(len(b["wave"]) for b in batch)
    waves = np.zeros((len(batch), L), dtype=np.float32)
    mask = np.zeros((len(batch), L), dtype=np.float32)
    for i, b in enumerate(batch):
        waves[i, : len(b["wave"])] = b["wave"]; mask[i, : len(b["wave"])] = 1.0
    return {
        "input_values": torch.from_numpy(waves), "attn_mask": torch.from_numpy(mask).long(),
        "tgt": torch.from_numpy(np.stack([b["tgt"] for b in batch])),
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
# ## 5. Heads + train loop

# %%
from scipy.stats import spearmanr

torch.manual_seed(SEED); np.random.seed(SEED)
N_EMO = len(EMOTIONS5)

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

heads = EmoHeads(WAVLM_DIM, TRUNK_HIDDEN, HEAD_HIDDEN, DROPOUT, N_EMO).to(device)

TASKS = ["emos", "cat", "val", "aro", "dom"]
log_var = nn.Parameter(torch.zeros(len(TASKS), device=device))
bb_params = [p for p in wavlm.parameters() if p.requires_grad]
head_params = list(heads.parameters()) + ([log_var] if USE_UNCERTAINTY else [])
opt = torch.optim.AdamW([{"params": bb_params, "lr": LR_BACKBONE},
                         {"params": head_params, "lr": LR_HEAD}], weight_decay=WEIGHT_DECAY)
scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP and device == "cuda")
mse = nn.MSELoss()

def soft_ce(logits, target_dist):
    return -(target_dist * F.log_softmax(logits, dim=1)).sum(1).mean()

def forward_batch(b):
    feat = wavlm_embed(b["input_values"].to(device), b["attn_mask"].to(device))
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

CKPT_PATH = os.path.join(OUT_DIR, f"ft_wavlm_{INIT_MODE}.pt")
def save_full(state, val_emos=float("nan")):
    torch.save({"wavlm": state["wavlm"], "heads": state["heads"], "INIT_MODE": INIT_MODE,
                "emos_mu": emos_mu, "emos_sd": emos_sd, "vad_mu": vad_mu, "vad_sd": vad_sd,
                "WAVLM_DIM": WAVLM_DIM, "val_emos": float(val_emos)}, CKPT_PATH)

best, best_state, bad = -1e9, None, 0
for ep in range(1, EPOCHS + 1):
    wavlm.train(); heads.train()
    opt.zero_grad(); run = 0.0; nb = 0
    for step, b in enumerate(tqdm(tr_loader, desc=f"[{INIT_MODE}] epoch {ep}")):
        with torch.cuda.amp.autocast(enabled=USE_AMP and device == "cuda"):
            emos_p, cat_l, vad_p = forward_batch(b)
            loss = compute_loss(emos_p, cat_l, vad_p, b) / ACCUM
        scaler.scale(loss).backward()
        if (step + 1) % ACCUM == 0:
            scaler.step(opt); scaler.update(); opt.zero_grad()
        run += loss.item() * ACCUM; nb += 1
    m = evaluate(); sc = mean_srcc(m)
    msg = " ".join(f"{k}={m[k]:.3f}" for k in ["emos", "val", "aro", "dom"] if k in m)
    print(f"[{INIT_MODE}] epoch {ep:2d} | loss {run/max(nb,1):.4f} | {msg} | cat_err {m['cat_err']:.3f} | mean {sc:.4f} (best {max(best,sc):.4f})")
    if sc > best:
        best = sc
        best_state = {"wavlm": {k: v.cpu().clone() for k, v in wavlm.state_dict().items()},
                      "heads": {k: v.cpu().clone() for k, v in heads.state_dict().items()}}
        save_full(best_state, m["emos"]); bad = 0
        print(f"   💾 lưu best → {CKPT_PATH} (epoch {ep}, mean {sc:.4f})")
    else:
        bad += 1
        if bad >= PATIENCE:
            print(f"Early stop ở epoch {ep}."); break

if best_state:
    wavlm.load_state_dict(best_state["wavlm"]); heads.load_state_dict(best_state["heads"])
final = evaluate()
print(f"\n✅ VAL (nội bộ) — exp12 INIT_MODE={INIT_MODE}:")
print(f"   EMOS={final['emos']:.4f}", end="")
if HAS_VAD:
    print(f" | VAL/ARO/DOM={final['val']:.4f}/{final['aro']:.4f}/{final['dom']:.4f}")
else:
    print()
print(f"   cat_err={final['cat_err']:.4f} | mean SRCC={mean_srcc(final):.4f}")
print(f"   (Mốc DEV exp08 để tham khảo: EMOS {EXP08['emos']}, VAD {EXP08['val']}/{EXP08['aro']}/{EXP08['dom']})")
print("   ➜ GHI con số này vào bảng ablation 04_ rồi đổi INIT_MODE chạy lại để so 3 mode.")

# %% [markdown]
# ## 6. Dự đoán DEV → answer.txt (QMOS mượn exp07 / UTMOSv2)

# %%
def list_dev():
    with open(DEV_SCP) as f:
        return [ln.strip() for ln in f if ln.strip()]

dev_names = list_dev()
if LIMIT_DEV:
    dev_names = dev_names[:LIMIT_DEV]
print("DEV:", len(dev_names), "mẫu")

def load_exp07_qmos():
    if EXP07_ANSWER and os.path.exists(EXP07_ANSWER):
        import csv
        d = {}
        with open(EXP07_ANSWER) as f:
            for row in csv.DictReader(f):
                d[row["wav"]] = float(row["QMOS"]); d[stem(row["wav"])] = float(row["QMOS"])
        print(f"✅ Mượn QMOS exp07: {len(d)//2} wav")
        return d
    return None

qmos_map = load_exp07_qmos()
if qmos_map is None:
    print("ℹ️ Không có exp07 → QMOS bằng UTMOSv2.")
    pip_install("git+https://github.com/sarulab-speech/UTMOSv2.git")
    import utmosv2
    v2 = utmosv2.create_model(pretrained=True)
    qmos_map = {}
    for n in tqdm(dev_names, desc="UTMOSv2"):
        wav = os.path.join(WAV_DIR, n if str(n).endswith(".wav") else str(n) + ".wav")
        if os.path.exists(wav):
            o = v2.predict(input_path=wav)
            qmos_map[n] = float(o["predicted_mos"]) if isinstance(o, dict) else float(o)
    del v2; torch.cuda.empty_cache() if device == "cuda" else None

@torch.no_grad()
def predict_emotion(sid):
    wave = load_wav(sid)
    if wave is None:
        return None
    wavlm.eval(); heads.eval()
    iv = torch.from_numpy(wave).unsqueeze(0).to(device)
    am = torch.ones((1, len(wave)), dtype=torch.long, device=device)
    tgt = torch.from_numpy(onehot_target(target_map.get(sid))).unsqueeze(0).to(device)
    with torch.cuda.amp.autocast(enabled=USE_AMP and device == "cuda"):
        feat = wavlm_embed(iv, am)
        emos_p, cat_l, vad_p = heads(feat, tgt)
    emos = float(emos_p.item()) * emos_sd + emos_mu
    cat5 = F.softmax(cat_l, 1)[0].float().cpu().numpy()
    vad3 = vad_p[0].float().cpu().numpy() * vad_sd + vad_mu
    return emos, cat5, vad3

def fmt_cat(p5):
    return "|".join(f"{e}:{p5[i]:.6g}" for i, e in enumerate(EMOTIONS5))

answer_path = os.path.join(OUT_DIR, f"answer_{INIT_MODE}.txt")
n_real = n_def = 0
with open(answer_path, "w") as f:
    f.write("wav,QMOS,EMOS,CAT,VAL,ARO,DOM\n")
    for name in tqdm(dev_names, desc=f"answer[{INIT_MODE}]"):
        sid = stem(name)
        pr = predict_emotion(sid)
        if pr is None:
            emos, cat5, vad3 = 3.0, np.full(5, 0.2, np.float32), np.array([3.0, 3.0, 3.0]); n_def += 1
        else:
            emos, cat5, vad3 = pr; n_real += 1
        qmos = qmos_map.get(name, qmos_map.get(sid, 3.0))
        f.write(f"{name},{qmos:.6g},{emos:.6g},{fmt_cat(cat5)},{vad3[0]:.6g},{vad3[1]:.6g},{vad3[2]:.6g}\n")
print(f"Ghi {len(dev_names)} dòng → {answer_path} | thật {n_real}, mặc định {n_def}")

# %% [markdown]
# ## 7. Validate + zip

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
os.system(f"cd {OUT_DIR} && cp answer_{INIT_MODE}.txt answer.txt && zip -j submission_track2_exp12_{INIT_MODE}.zip answer.txt && unzip -l submission_track2_exp12_{INIT_MODE}.zip")
print("Sẵn sàng nộp:", os.path.join(OUT_DIR, f"submission_track2_exp12_{INIT_MODE}.zip"))

# %% [markdown]
# ## Ghi chú
# - **Chạy 3 lần** đổi `INIT_MODE` ("scratch"→"base"→"sailer"), ghi `mean SRCC` mỗi lần vào BẢNG ABLATION
#   trong `docs/04_experiments_log.md` → trả lời mentor bằng số: from-scratch tốt hơn fine-tune không?
# - **scratch nặng:** mở băng toàn bộ WavLM-large. Nếu OOM → giảm `BATCH` (4→2), `MAX_SECONDS` (6→5),
#   hoặc đổi sang `microsoft/wavlm-base-plus` (sửa cell 2) cho khả thi (lưu ý: khác kiến trúc → so kém công bằng hơn).
# - **scratch chậm + cần nhiều epoch hơn** (random init): để `EPOCHS=15`, `PATIENCE=5`. Vẫn nhiều khả năng < base/sailer.
# - **Đừng nhầm VAL nội bộ với DEV.** So 3 mode bằng VAL nội bộ đã đủ kết luận; muốn chắc thì nộp mode tốt nhất.
# - Checkpoint lưu `ft_wavlm_<mode>.pt`. Save Version sau mỗi lần chạy.
