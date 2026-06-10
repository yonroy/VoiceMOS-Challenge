# %% [markdown]
# # VMC2026 Track 2 — exp08 (FINE-TUNE WavLM cho 5 cột cảm xúc) — Kaggle
#
# **Khác mọi exp trước:** exp03–07 đều **đóng băng** backbone (chỉ trích đặc trưng + train head nhỏ trên cache).
# exp08 **MỞ BĂNG (fine-tune)** WavLM-large để nó học lại đặc trưng riêng cho bài MOS cảm xúc 2026.
#
# ## Thiết kế (chốt với mentor 5/6)
# ```
#  wav ─┬─► WavLM-large (warm-start SAILER, TRAINABLE: chỉ mở băng N lớp trên)  ─► pool ─► emb_wavlm ┐
#       └─► audeering MSP-dim (FROZEN, cache .npz)  ─► [emb_aud | vad3]                                ├─► TRUNK ─┬─► EMOS (+target)
#                                                                                                       ┘          ├─► CAT (5)
#                                                                                                                  └─► VAD (3)
#  QMOS: KHÔNG train ở đây → mượn cột QMOS của exp07 (0.548) hoặc UTMOSv2 (T05, vô địch VMC2024).
# ```
# - **Warm-start:** khởi tạo WavLM từ checkpoint **SAILER** (`tiantiaf/wavlm-large-categorical-emotion`,
#   đã giỏi cảm xúc) thay vì WavLM "trắng" → điểm xuất phát tốt hơn nhiều.
# - **Phụ (frozen):** audeering — dimensional, bổ trợ góc nhìn categorical của WavLM, kỳ vọng kéo **VAL**.
# - **Đóng băng partial:** chỉ train `UNFREEZE_TOP_LAYERS` lớp Transformer trên cùng + feature-extractor giữ băng
#   → tiết kiệm VRAM T4 + chống overfit (chỉ 12.7k mẫu).
#
# ## ⚠️ Đánh đổi phải biết trước (so freeze+head)
# - **Mất lợi thế cache:** mỗi epoch chạy lại cả WavLM (forward+backward) → chậm & đốt giờ GPU (30h/tuần).
#   → **Lần đầu BẮT BUỘC đặt `LIMIT_TRAIN=300`, `LIMIT_DEV=20`** để chỉnh trơn rồi mới `None`.
# - **Dễ overfit / OOM:** nếu OOM → giảm `BATCH`, tăng `ACCUM`, giảm `MAX_SECONDS`, giảm `UNFREEZE_TOP_LAYERS`.
# - **Lưới an toàn:** exp07 vẫn là bản nộp vô địch tới khi exp08 **thắng trên VAL nội bộ** (đừng đốt lượt nộp).
#
# **Cách chạy Kaggle:** GPU **T4** + Internet **On** → Add Input dataset Track 2 → sửa `DATA_ROOT` → Run All.

# %% [markdown]
# ## 0. Cấu hình — SỬA Ở ĐÂY

# %%
import os

DATA_ROOT    = "/kaggle/input/vmc2026-track2-full/vmc2026-track2"   # << SỬA slug cho khớp Add Input
WAV_DIR      = f"{DATA_ROOT}/wav"
METADATA_CSV = f"{DATA_ROOT}/metadata.csv"     # wavID|emotion|transcript (KHÔNG header)
TRAIN_CSV    = f"{DATA_ROOT}/sets/train.csv"   # lisID|wavID|qMOS|emoCat|eMOS|val|dom|aro
DEV_SCP      = f"{DATA_ROOT}/sets/dev.scp"

OUT_DIR   = "/kaggle/working"
CACHE_DIR = "/kaggle/working/ft_cache"         # cache audeering (.npz) — backbone WavLM KHÔNG cache (đang train)
os.makedirs(CACHE_DIR, exist_ok=True)

# (Tùy chọn) TÁI DÙNG cache audeering cũ: trỏ tới dataset chứa aud_train.npz/aud_dev.npz → tự copy sang CACHE_DIR.
# Để "" nếu chạy mới hoàn toàn. /kaggle/input read-only nên phải copy sang working để ghi/append.
CACHE_INPUT = "/kaggle/input/datasets/minhtoan2/cache-exp8"   # << SỬA slug cho khớp (hoặc "")
if CACHE_INPUT and os.path.isdir(CACHE_INPUT):
    import shutil
    _n = 0
    for _fn in os.listdir(CACHE_INPUT):
        if _fn.startswith("aud_") and _fn.endswith(".npz"):
            shutil.copy(os.path.join(CACHE_INPUT, _fn), os.path.join(CACHE_DIR, _fn)); _n += 1
    print(f"📦 Tái dùng cache: copy {_n} file aud_*.npz từ {CACHE_INPUT} → {CACHE_DIR}")

# Mượn cột QMOS của exp07 (tốt nhất 0.548). Trỏ tới answer.txt exp07 nếu có; không thì dùng UTMOSv2.
EXP07_ANSWER = "/kaggle/input/exp07-answer/answer.txt"   # << (tùy chọn) Add Input answer.txt exp07; không có → UTMOSv2

# ── Fine-tune / siêu tham số ─────────────────────────────────────────────────
DEVICE              = "cuda"
SR                  = 16000
MAX_SECONDS         = 8           # cắt audio để chặn bộ nhớ backprop; OOM thì giảm còn 6
UNFREEZE_TOP_LAYERS = 6           # số lớp Transformer trên cùng được train (0 = freeze hết = quay về head-only)
TRUNK_HIDDEN        = 512
HEAD_HIDDEN         = 128
DROPOUT             = 0.3
LR_BACKBONE         = 1e-5        # LR nhỏ cho backbone fine-tune
LR_HEAD             = 1e-3        # LR lớn cho trunk + head (train từ đầu)
WEIGHT_DECAY        = 1e-5
EPOCHS              = 12          # TRẦN; early-stop quyết định số epoch thực (8 hơi thấp cho lần chạy thật)
PATIENCE            = 3            # dừng khi val SRCC không lên 3 epoch; LUÔN giữ best_state
BATCH               = 4           # nhỏ vì backbone to; tăng ACCUM để bù
ACCUM               = 8           # effective batch = BATCH*ACCUM = 32
VAL_FRAC            = 0.10
SEED                = 42
USE_AMP             = True        # mixed precision fp16 — tiết kiệm VRAM
USE_GRAD_CKPT       = True        # gradient checkpointing — tiết kiệm VRAM (đổi lấy chậm hơn)
USE_AUDEERING       = True        # nhánh phụ frozen audeering; False = chỉ WavLM
USE_UNCERTAINTY     = True        # tự cân 5 loss (Kendall); False = trọng số 1.0

LIMIT_TRAIN         = 300         # << LẦN ĐẦU để 300; chạy thật đặt None
LIMIT_DEV           = 20          # << LẦN ĐẦU để 20; chạy thật đặt None

# Mốc exp07 để so (cảnh báo nếu fine-tune KHÔNG thắng → giữ exp07)
EXP07 = {"emos": 0.795, "cat_err": 0.153, "val": 0.581, "aro": 0.752, "dom": 0.705}

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
print(f"Fine-tune: mở băng {UNFREEZE_TOP_LAYERS} lớp trên · BATCH {BATCH}×ACCUM {ACCUM} · MAX {MAX_SECONDS}s")

# %% [markdown]
# ## 1. Cài đặt + tải code SAILER (clone + sys.path, KHÔNG pip install -e .)

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
# ## 2. Nạp SAILER → lấy backbone WavLM bên trong để FINE-TUNE
# Thay vì gọi wrapper như hộp đen, ta **lôi module WavLM-large (HuggingFace) bên trong wrapper** ra
# → toàn quyền đóng băng/mở băng từng lớp + tự pool. Nếu không tìm thấy (cấu trúc lạ) → **fallback**
# nạp `microsoft/wavlm-large` trắng (mất warm-start, có cảnh báo).

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F

device = DEVICE if torch.cuda.is_available() else "cpu"
print("Device:", device, ("✅ " + torch.cuda.get_device_name(0)) if device == "cuda" else "⚠️ CPU (rất chậm!)")

def find_hf_backbone(module):
    """Tìm submodule kiểu HF Wav2Vec2/WavLM backbone: có .feature_extractor và .encoder.layers."""
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
        print(f"✅ Warm-start SAILER: lấy backbone WavLM bên trong wrapper tại '.{name}' "
              f"({sum(p.numel() for p in wavlm.parameters())/1e6:.0f}M params)")
    else:
        print("⚠️ Không tìm thấy backbone HF bên trong wrapper SAILER → sẽ fallback WavLM trắng.")
except Exception as e:
    print("⚠️ Lỗi nạp SAILER wrapper:", repr(e), "→ fallback WavLM trắng.")

if wavlm is None:
    from transformers import WavLMModel
    wavlm = WavLMModel.from_pretrained("microsoft/wavlm-large")
    print("ℹ️ Fallback: nạp microsoft/wavlm-large (KHÔNG warm-start SAILER).")

wavlm = wavlm.to(device)
WAVLM_DIM = int(wavlm.config.hidden_size)

# ── Đóng băng partial: feature-extractor + tất cả trừ UNFREEZE_TOP_LAYERS lớp trên ──
for p in wavlm.parameters():
    p.requires_grad = False
enc_layers = wavlm.encoder.layers
n_layers = len(enc_layers)
for layer in enc_layers[max(0, n_layers - UNFREEZE_TOP_LAYERS):]:
    for p in layer.parameters():
        p.requires_grad = True
n_train = sum(p.numel() for p in wavlm.parameters() if p.requires_grad)
print(f"WavLM: {n_layers} lớp encoder · mở băng {min(UNFREEZE_TOP_LAYERS, n_layers)} lớp trên "
      f"→ {n_train/1e6:.1f}M param train (trên dim {WAVLM_DIM})")

if USE_GRAD_CKPT:
    wavlm.gradient_checkpointing_enable()
    if hasattr(wavlm, "enable_input_require_grads"):
        wavlm.enable_input_require_grads()   # cần khi grad-ckpt + lớp dưới đóng băng

def masked_mean(hidden, attn_mask):
    """Mean-pool theo thời gian, bỏ qua phần pad (giữ gradient)."""
    if attn_mask is None:
        return hidden.mean(dim=1)
    try:
        fm = wavlm._get_feature_vector_attention_mask(hidden.shape[1], attn_mask)
    except Exception:
        return hidden.mean(dim=1)
    fm = fm.unsqueeze(-1).to(hidden.dtype)
    return (hidden * fm).sum(1) / fm.sum(1).clamp(min=1e-6)

def wavlm_embed(input_values, attn_mask):
    out = wavlm(input_values, attention_mask=attn_mask).last_hidden_state   # [B,T,D]
    return masked_mean(out, attn_mask)

# %% [markdown]
# ## 3. Nạp audeering MSP-dim (FROZEN) — đặc trưng phụ
# Lấy `[emb_pool(1024) | vad3(1–5)]` mỗi wav rồi **cache .npz** (chạy 1 lần). Kỹ thuật nạp head tay
# y hệt exp05 (tránh lỗi version transformers khi subclass `Wav2Vec2PreTrainedModel`).

# %%
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
    missing, unexpected = aud_backbone.load_state_dict(bb_sd, strict=False)
    print(f"  audeering backbone: thiếu {len(missing)} / dư {len(unexpected)} key (strict=False)")
    _hid = _sd["classifier.dense.weight"].shape[0]
    _out = _sd["classifier.out_proj.weight"].shape[0]
    aud_head = nn.Sequential(nn.Linear(_hid, _hid), nn.Tanh(), nn.Linear(_hid, _out))
    aud_head[0].weight.data.copy_(_sd["classifier.dense.weight"]); aud_head[0].bias.data.copy_(_sd["classifier.dense.bias"])
    aud_head[2].weight.data.copy_(_sd["classifier.out_proj.weight"]); aud_head[2].bias.data.copy_(_sd["classifier.out_proj.bias"])
    aud_backbone = aud_backbone.to(device).eval()
    aud_head = aud_head.to(device).eval()
    AUD_DIM = _hid + 3   # emb_pool + [VAL,ARO,DOM]
    print(f"✅ audeering frozen (đặc trưng phụ {AUD_DIM}-D = emb {_hid} + vad 3)")

# %%
import numpy as np
import librosa
from tqdm.auto import tqdm

def load_wav(name_or_stem, in_wav_dir=True):
    p = name_or_stem if os.path.isabs(str(name_or_stem)) else os.path.join(
        WAV_DIR, name_or_stem if str(name_or_stem).endswith(".wav") else str(name_or_stem) + ".wav")
    if not os.path.exists(p):
        return None
    wave, _ = librosa.load(p, sr=SR, mono=True)
    return wave[: MAX_SECONDS * SR].astype(np.float32)

@torch.no_grad()
def extract_audeering(stems, tag):
    """→ dict {stem: float32[AUD_DIM]}; cache CACHE_DIR/aud_<tag>.npz (resume mỗi 500)."""
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
        h = aud_backbone(x)[0].mean(dim=1)                    # [1, hid]
        out = aud_head(h)[0].cpu().numpy()                    # [arousal, dominance, valence] ∈[0,1]
        vad = np.array([1 + 4 * out[2], 1 + 4 * out[0], 1 + 4 * out[1]], dtype=np.float32)  # [VAL,ARO,DOM]
        store[s] = np.concatenate([h[0].cpu().numpy(), vad]).astype(np.float32)
        if (i + 1) % 500 == 0:
            np.savez(cache_path, **store)
    if todo:
        np.savez(cache_path, **store)
    return store

# %% [markdown]
# ## 4. Đọc & gộp nhãn theo wavID (EMOS / VAD / CAT) — như exp04/07 nhưng KHÔNG cần qMOS

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
# ## 5. Dataset / DataLoader (load wav theo batch — KHÔNG cache WavLM vì đang train)

# %%
from torch.utils.data import Dataset, DataLoader

train_stems = [s for s in train_df["wavID"] if target_map.get(s) is not None]
if LIMIT_TRAIN:
    train_stems = train_stems[:LIMIT_TRAIN]
aud_tr = extract_audeering(train_stems, "train")

lab = train_df.set_index("wavID")

# Chuẩn hóa nhãn liên tục về z-score (để các MSE cùng thang) — lưu để giải mã lúc dự đoán.
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
    out = {
        "input_values": torch.from_numpy(waves), "attn_mask": torch.from_numpy(mask).long(),
        "tgt": torch.from_numpy(np.stack([b["tgt"] for b in batch])),
        "aud": torch.from_numpy(np.stack([b["aud"] for b in batch])) if USE_AUDEERING else None,
        "emos": torch.from_numpy(np.stack([b["emos"] for b in batch])).unsqueeze(1),
        "vad": torch.from_numpy(np.stack([b["vad"] for b in batch])),
        "cat": torch.from_numpy(np.stack([b["cat"] for b in batch])),
        "emos_raw": np.stack([b["emos_raw"] for b in batch]),
        "vad_raw": np.stack([b["vad_raw"] for b in batch]),
    }
    return out

from sklearn.model_selection import train_test_split
ds = EmoDataset(train_stems)
print("Dataset hợp lệ:", len(ds), "wav")
tr_i, va_i = train_test_split(np.arange(len(ds)), test_size=VAL_FRAC, random_state=SEED)
tr_loader = DataLoader(torch.utils.data.Subset(ds, tr_i), batch_size=BATCH, shuffle=True, collate_fn=collate, num_workers=2)
va_loader = DataLoader(torch.utils.data.Subset(ds, va_i), batch_size=BATCH, shuffle=False, collate_fn=collate, num_workers=2)

# %% [markdown]
# ## 6. Head fusion (trunk + 3 head cảm xúc) + train loop (AMP + grad accumulation)

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
print(f"Trunk input = {TRUNK_IN} (wavlm {WAVLM_DIM} + aud {AUD_DIM if USE_AUDEERING else 0})")

TASKS = ["emos", "cat", "val", "aro", "dom"]
log_var = nn.Parameter(torch.zeros(len(TASKS), device=device))
bb_params = [p for p in wavlm.parameters() if p.requires_grad]
head_params = list(heads.parameters()) + ([log_var] if USE_UNCERTAINTY else [])
opt = torch.optim.AdamW([
    {"params": bb_params, "lr": LR_BACKBONE},
    {"params": head_params, "lr": LR_HEAD},
], weight_decay=WEIGHT_DECAY)
scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP and device == "cuda")
mse = nn.MSELoss()

def soft_ce(logits, target_dist):
    return -(target_dist * F.log_softmax(logits, dim=1)).sum(1).mean()

def forward_batch(b):
    feat_wavlm = wavlm_embed(b["input_values"].to(device), b["attn_mask"].to(device))
    if USE_AUDEERING:
        feat = torch.cat([feat_wavlm, b["aud"].to(device)], dim=1)
    else:
        feat = feat_wavlm
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
    out["cat_err"] = float(np.abs(q - p).sum(1).mean())   # ~ tổng |Δ| trung bình (xấp xỉ CAT-ERR)
    return out

def mean_srcc(m):
    keys = ["emos"] + (["val", "aro", "dom"] if HAS_VAD else [])
    return float(np.mean([m[k] for k in keys]))

# Lưu checkpoint FULL (có backbone WavLM) — gọi NGAY mỗi best để kernel chết giữa chừng vẫn còn file.
CKPT_PATH = os.path.join(OUT_DIR, "ft_emotion_full.pt")
def save_full_ckpt(state, val_emos=float("nan")):
    torch.save({"wavlm": state["wavlm"], "heads": state["heads"],
                "emos_mu": emos_mu, "emos_sd": emos_sd, "vad_mu": vad_mu, "vad_sd": vad_sd,
                "WAVLM_DIM": WAVLM_DIM, "AUD_DIM": AUD_DIM,
                "UNFREEZE_TOP_LAYERS": UNFREEZE_TOP_LAYERS, "val_emos": float(val_emos)}, CKPT_PATH)

best, best_state, bad = -1e9, None, 0
for ep in range(1, EPOCHS + 1):
    wavlm.train(); heads.train()
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
        best_state = {"wavlm": {k: v.cpu().clone() for k, v in wavlm.state_dict().items()},
                      "heads": {k: v.cpu().clone() for k, v in heads.state_dict().items()}}
        save_full_ckpt(best_state, m["emos"])   # LƯU NGAY mỗi best → an toàn nếu kernel chết
        print(f"   💾 lưu best → {CKPT_PATH} (epoch {ep}, mean {sc:.4f})")
        bad = 0
    else:
        bad += 1
        if bad >= PATIENCE:
            print(f"Early stop ở epoch {ep}."); break

if best_state:
    wavlm.load_state_dict(best_state["wavlm"]); heads.load_state_dict(best_state["heads"])
final = evaluate()
print("\n✅ VAL (nội bộ) — exp08 (fine-tune WavLM cho cảm xúc):")
print(f"   EMOS={final['emos']:.4f} (exp07 {EXP07['emos']})")
if HAS_VAD:
    print(f"   VAL/ARO/DOM={final['val']:.4f}/{final['aro']:.4f}/{final['dom']:.4f} "
          f"(exp07 {EXP07['val']}/{EXP07['aro']}/{EXP07['dom']})")
warn = [f"EMOS {final['emos']:.3f}<{EXP07['emos']}"] if final["emos"] < EXP07["emos"] - 0.005 else []
if HAS_VAD:
    warn += [f"{t.upper()} {final[t]:.3f}<{EXP07[t]}" for t in ["val", "aro", "dom"] if final[t] < EXP07[t] - 0.005]
print("   ⚠️ CHƯA thắng exp07 ở:", "; ".join(warn), "→ cân nhắc giữ exp07." if warn else "")
if not warn:
    print("   ✅ Fine-tune thắng/ngang exp07 ở mọi cột cảm xúc → đáng nộp.")
# Lưu lần cuối từ best (đã lưu sẵn mỗi best trong loop; đây là phát cuối cho chắc).
save_full_ckpt(best_state if best_state else
               {"wavlm": wavlm.state_dict(), "heads": heads.state_dict()}, final["emos"])
print(f"✅ Đã lưu {CKPT_PATH} (CÓ backbone WavLM + heads → resume được). "
      f"NHỚ Save Version để file ra Output!")

# %% [markdown]
# ## 7. Dự đoán DEV → answer.txt (5 cột cảm xúc từ exp08; QMOS mượn exp07 hoặc UTMOS)

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

# QMOS: ưu tiên mượn cột QMOS của exp07; không có file → chấm UTMOSv2 (T05, vô địch VMC2024).
def load_exp07_qmos():
    if EXP07_ANSWER and os.path.exists(EXP07_ANSWER):
        import csv
        d = {}
        with open(EXP07_ANSWER) as f:
            r = csv.DictReader(f)
            for row in r:
                d[row["wav"]] = float(row["QMOS"]); d[stem(row["wav"])] = float(row["QMOS"])
        print(f"✅ Mượn QMOS từ exp07 ({EXP07_ANSWER}): {len(d)//2} wav")
        return d
    return None

qmos_map = load_exp07_qmos()
if qmos_map is None:
    print("ℹ️ Không có answer.txt exp07 → chấm QMOS bằng UTMOSv2 (T05, vô địch VMC2024 Track 1).")
    pip_install("git+https://github.com/sarulab-speech/UTMOSv2.git")   # cần Internet On, checkpoint tự tải
    import utmosv2
    v2 = utmosv2.create_model(pretrained=True)
    qmos_map = {}
    for n in tqdm(dev_names, desc="UTMOSv2"):
        wav = os.path.join(WAV_DIR, n if str(n).endswith(".wav") else str(n) + ".wav")
        if not os.path.exists(wav):
            continue
        out = v2.predict(input_path=wav)   # trả float hoặc dict {'predicted_mos': ...} tùy phiên bản
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
        if USE_AUDEERING:
            aud = torch.from_numpy(aud_dev[sid]).unsqueeze(0).to(device)
            feat = torch.cat([fw, aud], dim=1)
        else:
            feat = fw
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
# ## 8. Validate + đóng zip

# %%
def validate(path):
    import csv
    with open(path) as f:
        rows = list(csv.reader(f))
    assert rows[0][0] == "wav" and "QMOS" in rows[0] and "EMOS" in rows[0], "Header sai"
    for i, r in enumerate(rows[1:], 2):
        assert len(r) == len(rows[0]), f"Dòng {i} sai số cột"
    print(f"OK: {len(rows)-1} dòng, header = {rows[0]}")

validate(answer_path)
os.system(f"cd {OUT_DIR} && zip -j submission_track2_exp08_ft-emotion.zip answer.txt "
          f"&& unzip -l submission_track2_exp08_ft-emotion.zip")
print("Sẵn sàng nộp:", os.path.join(OUT_DIR, "submission_track2_exp08_ft-emotion.zip"))

# %% [markdown]
# ## Ghi chú
# - **Lần đầu** `LIMIT_TRAIN=300`, `LIMIT_DEV=20` để kiểm tra chạy trơn (1 epoch xong không OOM); rồi đặt `None`.
# - **OOM trên T4?** giảm theo thứ tự: `MAX_SECONDS` (8→6) → `UNFREEZE_TOP_LAYERS` (6→4→2) → `BATCH` (4→2, tăng `ACCUM`).
# - **Đọc mục 6:** so EMOS/VAD VAL nội bộ với mốc exp07 (EMOS 0.795 · VAL 0.581 · ARO 0.752 · DOM 0.705).
#   - Nếu fine-tune **thắng** → nộp answer.txt exp08 (5 cột cảm xúc của exp08 + QMOS mượn exp07).
#   - Nếu **thua** → giữ exp07; vẫn là kết quả cho paper ("fine-tune chưa vượt frozen-fusion trên data nhỏ").
# - **QMOS:** Add Input answer.txt exp07 vào `/kaggle/input/exp07-answer/answer.txt` để mượn cột QMOS 0.548;
#   không có thì tự chấm UTMOSv2 (T05, vô địch VMC2024 — mạnh hơn UTMOS, cần Internet On).
# - **Ablation cho paper:** `UNFREEZE_TOP_LAYERS=0` (≈ head-only) vs `=6` (fine-tune) → bảng "frozen vs fine-tuned".
#   `USE_AUDEERING=False` → đo đóng góp nhánh phụ.
# - Ghi config → kết quả → nhận xét vào `docs/04_experiments_log.md` (mục exp08).
