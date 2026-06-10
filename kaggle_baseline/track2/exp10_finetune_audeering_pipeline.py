# %% [markdown]
# # VMC2026 Track 2 — exp10 (fine-tune AUDEERING riêng + ensemble VAD với exp08) — Kaggle T4
#
# **Ý tưởng (Hướng A — an toàn cho T4):** thay vì nhồi 2 backbone large vào 1 model (dễ OOM),
# ta fine-tune **audeering wav2vec2-large** RIÊNG (1 backbone → vừa T4), rồi **ensemble cột VAD**
# với exp08 (WavLM fine-tune). Mỗi lần chỉ 1 backbone trong VRAM → không OOM.
#
# ```
#  [exp08]  WavLM fine-tune  ─► VAD_wavlm  ┐
#                                          ├─ trung bình ─► VAD cuối (mạnh hơn cả 2)
#  [exp10]  audeering fine-tune ─► VAD_aud ┘
# ```
# audeering vốn là model **dimensional (chuyên VAD)** → fine-tune nó để bổ trợ VAD cho exp08.
#
# **Cách chạy:** GPU T4 + Internet On → sửa slug cell 0 → Run All. Lần đầu `LIMIT_TRAIN=300`.
# Để ensemble: Add Input answer.txt exp08 → trỏ `EXP08_ANSWER`.

# %% [markdown]
# ## 0. Cấu hình

# %%
import os

DATA_ROOT    = "/kaggle/input/datasets/minhtoan2/vmc2026-track2-full"   # << SỬA slug
WAV_DIR      = f"{DATA_ROOT}/wav"
METADATA_CSV = f"{DATA_ROOT}/metadata.csv"
TRAIN_CSV    = f"{DATA_ROOT}/sets/train.csv"
DEV_SCP      = f"{DATA_ROOT}/sets/dev.scp"

OUT_DIR   = "/kaggle/working"

# QMOS mượn exp07 (0.548); ensemble VAD với answer.txt exp08.
EXP07_ANSWER = "/kaggle/input/exp07-answer/answer.txt"     # << (tùy chọn) mượn QMOS; không có → UTMOSv2
EXP08_ANSWER = "/kaggle/input/exp08-answer/answer.txt"     # << (tùy chọn) để ENSEMBLE VAD; không có → chỉ ra answer audeering

# ── Fine-tune audeering (1 backbone) ─────────────────────────────────────────
DEVICE              = "cuda"
SR                  = 16000
MAX_SECONDS         = 8
UNFREEZE_TOP_LAYERS = 6           # số lớp encoder audeering mở băng (T4 thừa sức 1 backbone)
TRUNK_HIDDEN        = 512
HEAD_HIDDEN         = 128
DROPOUT             = 0.3
LR_BACKBONE         = 1e-5
LR_HEAD             = 1e-3
WEIGHT_DECAY        = 1e-5
EPOCHS              = 12
PATIENCE            = 4
BATCH               = 4
ACCUM               = 8
VAL_FRAC            = 0.10
SEED                = 42
USE_AMP             = True
USE_GRAD_CKPT       = True
USE_UNCERTAINTY     = True

# Ensemble: cột nào lấy TRUNG BÌNH giữa exp08 và exp10; cột khác giữ từ exp08.
ENSEMBLE_COLS       = ["VAL", "ARO", "DOM"]   # audeering mạnh VAD → ensemble VAD. Thêm "EMOS" nếu muốn.

LIMIT_TRAIN         = 300         # << LẦN ĐẦU 300; chạy thật None
LIMIT_DEV           = 20          # << LẦN ĐẦU 20; chạy thật None

EXP08 = {"emos": 0.811, "cat_err": 0.133, "val": 0.659, "aro": 0.793, "dom": 0.751}

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
for p in [WAV_DIR, METADATA_CSV, TRAIN_CSV, DEV_SCP]:
    print(("  ✅ " if os.path.exists(p) else "  ❌ THIẾU ") + p)

# %% [markdown]
# ## 1. Cài đặt

# %%
import sys, subprocess

def pip_install(*pkgs):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], check=True)

pip_install("transformers", "huggingface_hub", "safetensors", "speechmos",
            "librosa", "soundfile", "scipy", "scikit-learn", "pandas", "tqdm")

# %% [markdown]
# ## 2. Nạp audeering wav2vec2-large làm backbone FINE-TUNE
# Nạp backbone tay (tránh lỗi subclass `Wav2Vec2PreTrainedModel` ở transformers mới) rồi mở băng lớp trên.

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = DEVICE if torch.cuda.is_available() else "cpu"
print("Device:", device, ("✅ " + torch.cuda.get_device_name(0)) if device == "cuda" else "⚠️ CPU (rất chậm!)")

from transformers import Wav2Vec2Model, Wav2Vec2Config, Wav2Vec2Processor
from huggingface_hub import hf_hub_download

AUD_NAME = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
aud_proc = Wav2Vec2Processor.from_pretrained(AUD_NAME)
aud_cfg = Wav2Vec2Config.from_pretrained(AUD_NAME)
aud = Wav2Vec2Model(aud_cfg)
try:
    _sd = __import__("safetensors.torch", fromlist=["load_file"]).load_file(
        hf_hub_download(AUD_NAME, "model.safetensors"))
except Exception:
    _sd = torch.load(hf_hub_download(AUD_NAME, "pytorch_model.bin"), map_location="cpu")
bb_sd = {k[len("wav2vec2."):]: v for k, v in _sd.items() if k.startswith("wav2vec2.")}
miss, unexp = aud.load_state_dict(bb_sd, strict=False)
print(f"audeering backbone: thiếu {len(miss)} / dư {len(unexp)} key (strict=False)")
aud = aud.to(device)
AUD_DIM = int(aud.config.hidden_size)

# Đóng băng tất cả, mở băng UNFREEZE_TOP_LAYERS lớp encoder trên cùng
for p in aud.parameters():
    p.requires_grad = False
enc_layers = aud.encoder.layers
n_layers = len(enc_layers)
for layer in enc_layers[max(0, n_layers - UNFREEZE_TOP_LAYERS):]:
    for p in layer.parameters():
        p.requires_grad = True
n_train = sum(p.numel() for p in aud.parameters() if p.requires_grad)
print(f"audeering: {n_layers} lớp · mở băng {min(UNFREEZE_TOP_LAYERS, n_layers)} → {n_train/1e6:.1f}M param train (dim {AUD_DIM})")

if USE_GRAD_CKPT:
    aud.gradient_checkpointing_enable()
    if hasattr(aud, "enable_input_require_grads"):
        aud.enable_input_require_grads()

def masked_mean(hidden, attn_mask):
    if attn_mask is None:
        return hidden.mean(dim=1)
    try:
        fm = aud._get_feature_vector_attention_mask(hidden.shape[1], attn_mask)
    except Exception:
        return hidden.mean(dim=1)
    fm = fm.unsqueeze(-1).to(hidden.dtype)
    return (hidden * fm).sum(1) / fm.sum(1).clamp(min=1e-6)

def aud_embed(input_values, attn_mask):
    out = aud(input_values, attention_mask=attn_mask).last_hidden_state
    return masked_mean(out, attn_mask)

# %% [markdown]
# ## 3. Nhãn (gộp theo wavID) — như exp08

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
# ## 4. Dataset/loader (input_values qua audeering processor) + chuẩn hóa nhãn

# %%
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

train_stems = [s for s in train_df["wavID"] if target_map.get(s) is not None]
if LIMIT_TRAIN:
    train_stems = train_stems[:LIMIT_TRAIN]
lab = train_df.set_index("wavID")

def _zfit(arr):
    a = np.asarray(arr, dtype=np.float32)
    return float(np.nanmean(a)), float(np.nanstd(a) + 1e-6)

emos_mu, emos_sd = _zfit([lab.loc[s, "emos"] for s in train_stems])
if HAS_VAD:
    vad_mu = np.array([_zfit([lab.loc[s, c] for s in train_stems])[0] for c in ["val", "aro", "dom"]], dtype=np.float32)
    vad_sd = np.array([_zfit([lab.loc[s, c] for s in train_stems])[1] for c in ["val", "aro", "dom"]], dtype=np.float32)
else:
    vad_mu = np.zeros(3, dtype=np.float32); vad_sd = np.ones(3, dtype=np.float32)

def onehot_target(tgt):
    v = np.zeros(len(EMOTIONS5), dtype=np.float32)
    if tgt in EMOTIONS5:
        v[EMOTIONS5.index(tgt)] = 1.0
    return v

def load_iv(sid):
    """Đọc wav → chuẩn hóa bằng audeering processor → input_values (1D float32)."""
    p = os.path.join(WAV_DIR, sid if str(sid).endswith(".wav") else str(sid) + ".wav")
    if not os.path.exists(p):
        return None
    wave, _ = librosa.load(p, sr=SR, mono=True)
    wave = wave[: MAX_SECONDS * SR]
    iv = aud_proc(wave, sampling_rate=SR).input_values[0]
    return np.asarray(iv, dtype=np.float32)

class AudDataset(Dataset):
    def __init__(self, stems):
        self.stems = [s for s in stems if load_iv(s) is not None]
    def __len__(self):
        return len(self.stems)
    def __getitem__(self, i):
        s = self.stems[i]
        iv = load_iv(s)
        emos = (float(lab.loc[s, "emos"]) - emos_mu) / emos_sd
        if HAS_VAD:
            vad = (np.array([lab.loc[s, "val"], lab.loc[s, "aro"], lab.loc[s, "dom"]], np.float32) - vad_mu) / vad_sd
        else:
            vad = np.zeros(3, dtype=np.float32)
        cat = np.array([lab.loc[s, f"cat{j}"] for j in range(len(EMOTIONS5))], dtype=np.float32)
        return {"iv": iv, "tgt": onehot_target(target_map.get(s)),
                "emos": np.float32(emos), "vad": vad, "cat": cat,
                "emos_raw": np.float32(lab.loc[s, "emos"]),
                "vad_raw": np.array([lab.loc[s, "val"], lab.loc[s, "aro"], lab.loc[s, "dom"]], np.float32)}

def collate(batch):
    L = max(len(b["iv"]) for b in batch)
    ivs = np.zeros((len(batch), L), dtype=np.float32)
    mask = np.zeros((len(batch), L), dtype=np.float32)
    for i, b in enumerate(batch):
        ivs[i, : len(b["iv"])] = b["iv"]; mask[i, : len(b["iv"])] = 1.0
    return {
        "input_values": torch.from_numpy(ivs), "attn_mask": torch.from_numpy(mask).long(),
        "tgt": torch.from_numpy(np.stack([b["tgt"] for b in batch])),
        "emos": torch.from_numpy(np.stack([b["emos"] for b in batch])).unsqueeze(1),
        "vad": torch.from_numpy(np.stack([b["vad"] for b in batch])),
        "cat": torch.from_numpy(np.stack([b["cat"] for b in batch])),
        "emos_raw": np.stack([b["emos_raw"] for b in batch]),
        "vad_raw": np.stack([b["vad_raw"] for b in batch]),
    }

ds = AudDataset(train_stems)
print("Dataset hợp lệ:", len(ds), "wav")
tr_i, va_i = train_test_split(np.arange(len(ds)), test_size=VAL_FRAC, random_state=SEED)
tr_loader = DataLoader(torch.utils.data.Subset(ds, tr_i), batch_size=BATCH, shuffle=True, collate_fn=collate, num_workers=2)
va_loader = DataLoader(torch.utils.data.Subset(ds, va_i), batch_size=BATCH, shuffle=False, collate_fn=collate, num_workers=2)

# %% [markdown]
# ## 5. Heads + train loop (lưu ft_audeering_full.pt mỗi best)

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

heads = EmoHeads(AUD_DIM, TRUNK_HIDDEN, HEAD_HIDDEN, DROPOUT, N_EMO).to(device)

TASKS = ["emos", "cat", "val", "aro", "dom"]
log_var = nn.Parameter(torch.zeros(len(TASKS), device=device))
bb_params = [p for p in aud.parameters() if p.requires_grad]
head_params = list(heads.parameters()) + ([log_var] if USE_UNCERTAINTY else [])
opt = torch.optim.AdamW([{"params": bb_params, "lr": LR_BACKBONE},
                         {"params": head_params, "lr": LR_HEAD}], weight_decay=WEIGHT_DECAY)
scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP and device == "cuda")
mse = nn.MSELoss()

def soft_ce(logits, target_dist):
    return -(target_dist * F.log_softmax(logits, dim=1)).sum(1).mean()

def forward_batch(b):
    feat = aud_embed(b["input_values"].to(device), b["attn_mask"].to(device))
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
    aud.eval(); heads.eval()
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

CKPT_PATH = os.path.join(OUT_DIR, "ft_audeering_full.pt")
def save_full_ckpt(state, val_emos=float("nan")):
    torch.save({"aud": state["aud"], "heads": state["heads"],
                "emos_mu": emos_mu, "emos_sd": emos_sd, "vad_mu": vad_mu, "vad_sd": vad_sd,
                "AUD_DIM": AUD_DIM, "UNFREEZE_TOP_LAYERS": UNFREEZE_TOP_LAYERS,
                "val_emos": float(val_emos)}, CKPT_PATH)

best, best_state, bad = -1e9, None, 0
for ep in range(1, EPOCHS + 1):
    aud.train(); heads.train()
    opt.zero_grad(); run = 0.0; nb = 0
    for step, b in enumerate(tqdm(tr_loader, desc=f"epoch {ep}")):
        with torch.cuda.amp.autocast(enabled=USE_AMP and device == "cuda"):
            emos_p, cat_l, vad_p = forward_batch(b)
            loss = compute_loss(emos_p, cat_l, vad_p, b) / ACCUM
        scaler.scale(loss).backward()
        if (step + 1) % ACCUM == 0:
            scaler.step(opt); scaler.update(); opt.zero_grad()
        run += loss.item() * ACCUM; nb += 1
    m = evaluate(); sc = mean_srcc(m)
    msg = " ".join(f"{k}={m[k]:.3f}" for k in ["emos", "val", "aro", "dom"] if k in m)
    print(f"epoch {ep:2d} | loss {run/max(nb,1):.4f} | {msg} | cat_err {m['cat_err']:.3f} | mean {sc:.4f} (best {max(best,sc):.4f})")
    if sc > best:
        best = sc
        best_state = {"aud": {k: v.cpu().clone() for k, v in aud.state_dict().items()},
                      "heads": {k: v.cpu().clone() for k, v in heads.state_dict().items()}}
        save_full_ckpt(best_state, m["emos"])
        print(f"   💾 lưu best → {CKPT_PATH} (epoch {ep}, mean {sc:.4f})")
        bad = 0
    else:
        bad += 1
        if bad >= PATIENCE:
            print(f"Early stop ở epoch {ep}."); break

if best_state:
    aud.load_state_dict(best_state["aud"]); heads.load_state_dict(best_state["heads"])
final = evaluate()
print("\n✅ VAL (nội bộ) — exp10 (fine-tune audeering):")
print(f"   EMOS={final['emos']:.4f}", end="")
if HAS_VAD:
    print(f" | VAL/ARO/DOM={final['val']:.4f}/{final['aro']:.4f}/{final['dom']:.4f} (exp08 {EXP08['val']}/{EXP08['aro']}/{EXP08['dom']})")
else:
    print()
print(f"   → so exp08: audeering {'mạnh' if HAS_VAD and final['val'] > EXP08['val'] else 'yếu/ngang'} ở VAL. "
      f"Ensemble sẽ lấy trung bình 2 model.")
save_full_ckpt(best_state if best_state else {"aud": aud.state_dict(), "heads": heads.state_dict()}, final["emos"])
print(f"✅ Đã lưu {CKPT_PATH}. NHỚ Save Version!")

# %% [markdown]
# ## 6. Dự đoán DEV → predictions + answer_audeering.txt

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
        print(f"✅ Mượn QMOS từ exp07: {len(d)//2} wav")
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
    iv = load_iv(sid)
    if iv is None:
        return None
    aud.eval(); heads.eval()
    ivt = torch.from_numpy(iv).unsqueeze(0).to(device)
    am = torch.ones((1, len(iv)), dtype=torch.long, device=device)
    tgt = torch.from_numpy(onehot_target(target_map.get(sid))).unsqueeze(0).to(device)
    with torch.cuda.amp.autocast(enabled=USE_AMP and device == "cuda"):
        feat = aud_embed(ivt, am)
        emos_p, cat_l, vad_p = heads(feat, tgt)
    emos = float(emos_p.item()) * emos_sd + emos_mu
    cat5 = F.softmax(cat_l, 1)[0].float().cpu().numpy()
    vad3 = vad_p[0].float().cpu().numpy() * vad_sd + vad_mu
    return emos, cat5, vad3

def fmt_cat(p5):
    return "|".join(f"{e}:{p5[i]:.6g}" for i, e in enumerate(EMOTIONS5))

dev_pred = {}   # name -> (emos, cat5, vad3)
with open(os.path.join(OUT_DIR, "answer_audeering.txt"), "w") as f:
    f.write("wav,QMOS,EMOS,CAT,VAL,ARO,DOM\n")
    for name in tqdm(dev_names, desc="answer_aud"):
        sid = stem(name)
        pr = predict_emotion(sid)
        if pr is None:
            emos, cat5, vad3 = 3.0, np.full(5, 0.2, np.float32), np.array([3.0, 3.0, 3.0])
        else:
            emos, cat5, vad3 = pr
        dev_pred[name] = (emos, cat5, vad3)
        qmos = qmos_map.get(name, qmos_map.get(sid, 3.0))
        f.write(f"{name},{qmos:.6g},{emos:.6g},{fmt_cat(cat5)},{vad3[0]:.6g},{vad3[1]:.6g},{vad3[2]:.6g}\n")
print("Đã ghi answer_audeering.txt")

# %% [markdown]
# ## 7. ENSEMBLE với exp08 → answer.txt cuối (trung bình cột VAD)
# Lấy answer.txt exp08 làm nền; cột trong `ENSEMBLE_COLS` = trung bình (exp08 + exp10). Còn lại giữ exp08.

# %%
import csv
COL_IDX = {"QMOS": 1, "EMOS": 2, "VAL": 4, "ARO": 5, "DOM": 6}   # vị trí cột trong answer.txt
AUD_VAL = {"EMOS": lambda p: p[0], "VAL": lambda p: p[2][0], "ARO": lambda p: p[2][1], "DOM": lambda p: p[2][2]}

answer_path = os.path.join(OUT_DIR, "answer.txt")
if EXP08_ANSWER and os.path.exists(EXP08_ANSWER):
    with open(EXP08_ANSWER) as f:
        rows = list(csv.reader(f))
    header, body = rows[0], rows[1:]
    n_ens = 0
    with open(answer_path, "w") as f:
        f.write(",".join(header) + "\n")
        for r in body:
            name = r[0]; sid = stem(name)
            pr = dev_pred.get(name) or dev_pred.get(sid)
            if pr is not None:
                for col in ENSEMBLE_COLS:
                    if col in COL_IDX and col in AUD_VAL:
                        v08 = float(r[COL_IDX[col]]); vaud = float(AUD_VAL[col](pr))
                        r[COL_IDX[col]] = f"{0.5*(v08+vaud):.6g}"
                n_ens += 1
            f.write(",".join(r) + "\n")
    print(f"✅ Ensemble {ENSEMBLE_COLS}: {n_ens} dòng → {answer_path} (nền exp08 + trung bình audeering)")
else:
    print("ℹ️ Không có EXP08_ANSWER → answer.txt = answer_audeering.txt (chỉ audeering, chưa ensemble).")
    import shutil
    shutil.copy(os.path.join(OUT_DIR, "answer_audeering.txt"), answer_path)

# %% [markdown]
# ## 8. Validate + zip

# %%
def validate(path):
    with open(path) as f:
        rows = list(csv.reader(f))
    assert rows[0][0] == "wav" and "QMOS" in rows[0], "Header sai"
    for i, r in enumerate(rows[1:], 2):
        assert len(r) == len(rows[0]), f"Dòng {i} sai số cột"
    print(f"OK: {len(rows)-1} dòng, header = {rows[0]}")

validate(answer_path)
os.system(f"cd {OUT_DIR} && zip -j submission_track2_exp10_ensemble.zip answer.txt && unzip -l submission_track2_exp10_ensemble.zip")
print("Sẵn sàng nộp:", os.path.join(OUT_DIR, "submission_track2_exp10_ensemble.zip"))

# %% [markdown]
# ## Ghi chú
# - **Hướng A (T4-an toàn):** fine-tune audeering RIÊNG (1 backbone) → ensemble VAD với exp08 → KHÔNG OOM.
# - **Đọc mục 5:** audeering VAL/ARO/DOM có ≥ exp08 không? Nếu ngang/hơn → ensemble đáng giá.
# - **Ensemble (mục 7):** mặc định trung bình VAL/ARO/DOM. Thêm "EMOS" vào `ENSEMBLE_COLS` nếu audeering EMOS tốt.
# - **Checkpoint:** lưu `ft_audeering_full.pt` mỗi best (kernel chết vẫn còn). Save Version sau khi xong.
# - QMOS vẫn mượn exp07 (0.548). So sánh: nộp answer.txt ensemble vs exp08 thuần để xem ensemble có nhích VAD.
# - Ghi config → kết quả → nhận xét vào `docs/04_experiments_log.md` (exp10).
