# %% [markdown]
# # VMC2026 Track 2 — exp11 (FINE-TUNE ĐỒNG THỜI WavLM + audeering, FUSION 1 model) — Kaggle T4
#
# **Khác exp08:** exp08 chỉ fine-tune WavLM, audeering **đóng băng** (frozen, cache). exp11 **MỞ BĂNG CẢ HAI**
# backbone và fuse đặc trưng **trong cùng 1 model** → cả hai cùng học cho bài MOS cảm xúc 2026.
#
# ```
#  wav ─┬─► WavLM-large   (warm-start exp08, TRAINABLE: mở băng N lớp trên) ─► pool ─► emb_wavlm ┐
#       └─► audeering MSP (TRAINABLE: mở băng N lớp trên) ─► pool ─► [emb_aud(1024) | vad3] ──────┼─► TRUNK ─┬─► EMOS (+target)
#                                                                                                 ┘          ├─► CAT (5)
#                                                                                                            └─► VAD (3)
#  QMOS: KHÔNG train ở đây → mượn cột QMOS exp07 (0.548) hoặc UTMOSv2.
# ```
#
# ## Vì sao "feature fusion + fine-tune cả 2" (khác ensemble exp10)
# - **exp10 = ensemble:** 2 model RIÊNG → trung bình cột VAD ở mức answer. An toàn nhưng 2 model không "nói chuyện".
# - **exp11 = fusion:** 1 model, 2 backbone fuse Ở TRONG → trunk học phối hợp cả hai góc nhìn (WavLM categorical +
#   audeering dimensional) → kỳ vọng mạnh hơn nếu không OOM/overfit.
#
# ## ⚠️ ĐÁNH ĐỔI PHẢI BIẾT — đây là cấu hình NẶNG nhất (2 backbone large cùng có gradient)
# - **Rủi ro OOM cao trên T4 (16GB).** Đã bật sẵn mọi cách giảm bộ nhớ: `BATCH=1` + grad-accum,
#   gradient-checkpointing CẢ 2 backbone, AMP fp16, `MAX_SECONDS=6`, mở băng ÍT lớp (mặc định 4 mỗi backbone).
# - Nếu vẫn OOM: giảm `UNFREEZE_WAVLM`/`UNFREEZE_AUD` → 2, giảm `MAX_SECONDS` → 5, tăng `ACCUM`.
# - **Chậm + đốt giờ GPU** (2 backbone forward+backward, không cache được). **LẦN ĐẦU BẮT BUỘC `LIMIT_TRAIN=300`,
#   `LIMIT_DEV=20`** để chỉnh trơn rồi mới `None`.
# - **Lưới an toàn:** đừng đốt lượt nộp — chỉ nộp khi exp11 thắng exp08 (0.811) TRÊN VAL NỘI BỘ.
#
# **Cách chạy:** GPU **T4** + Internet **On** → Add Input (data + checkpoint exp08 + [tùy chọn] answer exp07) →
# sửa slug cell 0 → Run All. Ghi config→kết quả→nhận xét vào `docs/04_experiments_log.md` (exp11).

# %% [markdown]
# ## 0. Cấu hình — SỬA Ở ĐÂY

# %%
import os

DATA_ROOT    = "/kaggle/input/datasets/minhtoan2/vmc2026-track2-full/vmc2026-track2"   # << SỬA slug cho khớp Add Input
WAV_DIR      = f"{DATA_ROOT}/wav"
METADATA_CSV = f"{DATA_ROOT}/metadata.csv"     # wavID|emotion|transcript (KHÔNG header)
TRAIN_CSV    = f"{DATA_ROOT}/sets/train.csv"   # lisID|wavID|qMOS|emoCat|eMOS|val|dom|aro
DEV_SCP      = f"{DATA_ROOT}/sets/dev.scp"

# ── Warm-start / RESUME: trỏ tới 1 trong 2 loại checkpoint ───────────────────
#   • ft_emotion_full_20epoch.pt (exp08): có 'wavlm'+'heads' → WARM-START (audeering từ pretrained gốc).
#   • ft_joint_full.pt (exp11): có thêm 'aud'+'aud_head'    → RESUME ĐỦ (khôi phục cả 2 backbone đã fine-tune).
# Notebook TỰ nhận biết theo key trong checkpoint. Để "" nếu train WavLM từ SAILER trắng.
WARMSTART_CKPT = "/kaggle/input/ft-joint-full/ft_joint_full.pt"   # << exp08 ckpt (warm-start) HOẶC exp11 ckpt (resume)

# Mượn cột QMOS exp07 (0.548). Không có → UTMOSv2.
EXP07_ANSWER = "/kaggle/input/exp07-answer/answer.txt"   # << (tùy chọn) answer.txt exp07; không có → UTMOSv2

OUT_DIR = "/kaggle/working"

# ── Fine-tune / siêu tham số (CẤU HÌNH NẶNG — đã tối ưu cho T4) ───────────────
DEVICE          = "cuda"
SR              = 16000
MAX_SECONDS     = 6            # ↓ so exp08 (8) để tiết kiệm VRAM (2 backbone)
UNFREEZE_WAVLM  = 4            # số lớp encoder WavLM mở băng (OOM → 2)
UNFREEZE_AUD    = 4            # số lớp encoder audeering mở băng (OOM → 2)
TRUNK_HIDDEN    = 512          # PHẢI khớp checkpoint exp08 nếu warm-start heads
HEAD_HIDDEN     = 128          # PHẢI khớp checkpoint exp08
DROPOUT         = 0.3
LR_BACKBONE     = 1e-5         # LR chung cho 2 backbone
LR_HEAD         = 1e-3
RESUME_LR_SCALE = 1.0          # <1.0 để GIẢM LR khi resume (vd 0.5 nếu val đã chững) — nhân vào cả 2 nhóm LR
WEIGHT_DECAY    = 1e-5
EPOCHS          = 12
PATIENCE        = 4            # dừng khi val không lên; LUÔN giữ best
BATCH           = 1            # ⚠️ 2 backbone large → batch nhỏ
ACCUM           = 16           # effective batch = 16
VAL_FRAC        = 0.10
SEED            = 42
USE_AMP         = True
USE_GRAD_CKPT   = True
USE_UNCERTAINTY = True

LIMIT_TRAIN     = 300          # << LẦN ĐẦU 300; chạy thật None
LIMIT_DEV       = 20           # << LẦN ĐẦU 20; chạy thật None

EXP08 = {"emos": 0.811, "cat_err": 0.133, "val": 0.659, "aro": 0.793, "dom": 0.751}  # mốc để so

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
print(("  ✅ " if (WARMSTART_CKPT and os.path.exists(WARMSTART_CKPT)) else "  ⚠️ KHÔNG có ") + str(WARMSTART_CKPT)
      + ("  → warm-start" if (WARMSTART_CKPT and os.path.exists(WARMSTART_CKPT)) else "  → train từ SAILER trắng"))

# %% [markdown]
# ## 1. Cài đặt + tải code SAILER (dựng đúng kiến trúc WavLM)

# %%
import sys, subprocess

def pip_install(*pkgs):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], check=True)

pip_install("transformers", "huggingface_hub", "safetensors", "loralib", "speechbrain",
            "speechmos", "librosa", "soundfile", "scipy", "scikit-learn", "pandas", "tqdm")

REPO_DIR = "/kaggle/working/vox-profile-release"
if not os.path.exists(REPO_DIR):
    subprocess.run(["git", "clone", "--depth", "1",
                    "https://github.com/tiantiaf0627/vox-profile-release.git", REPO_DIR], check=True)
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

# %% [markdown]
# ## 2A. WavLM TRAINABLE (warm-start SAILER / checkpoint exp08)

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = DEVICE if torch.cuda.is_available() else "cpu"
print("Device:", device, ("✅ " + torch.cuda.get_device_name(0)) if device == "cuda" else "⚠️ CPU (rất chậm!)")

# Nạp checkpoint exp08 (nếu có) — lấy cả 'wavlm', 'heads', thống kê chuẩn hóa
ckpt = None
if WARMSTART_CKPT and os.path.exists(WARMSTART_CKPT):
    ckpt = torch.load(WARMSTART_CKPT, map_location="cpu", weights_only=False)
    print("✅ Nạp checkpoint warm-start:", WARMSTART_CKPT, "| keys:", list(ckpt.keys()))
    if "wavlm" not in ckpt:
        print("   ⚠️ Checkpoint KHÔNG có 'wavlm' (chỉ heads?) → vẫn dựng WavLM từ SAILER, chỉ warm-start heads nếu khớp.")

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
wavlm.config.layerdrop = 0.0   # ⚠️ tắt layerdrop khi dùng gradient-checkpointing (tránh CheckpointError)

# Đè trọng số đã fine-tune từ checkpoint exp08 (nếu có)
if ckpt is not None and "wavlm" in ckpt:
    miss, unexp = wavlm.load_state_dict(ckpt["wavlm"], strict=False)
    print(f"🔁 load wavlm từ checkpoint exp08: thiếu {len(miss)} / dư {len(unexp)} key (kỳ vọng ~0).")

# Đóng băng partial: chỉ mở UNFREEZE_WAVLM lớp trên
for p in wavlm.parameters():
    p.requires_grad = False
_wl = wavlm.encoder.layers
for layer in _wl[max(0, len(_wl) - UNFREEZE_WAVLM):]:
    for p in layer.parameters():
        p.requires_grad = True
print(f"WavLM: {len(_wl)} lớp · mở băng {min(UNFREEZE_WAVLM, len(_wl))} → "
      f"{sum(p.numel() for p in wavlm.parameters() if p.requires_grad)/1e6:.1f}M param train (dim {WAVLM_DIM})")

if USE_GRAD_CKPT:
    wavlm.gradient_checkpointing_enable()
    if hasattr(wavlm, "enable_input_require_grads"):
        wavlm.enable_input_require_grads()

def masked_mean(hidden, attn_mask, model):
    if attn_mask is None:
        return hidden.mean(dim=1)
    try:
        fm = model._get_feature_vector_attention_mask(hidden.shape[1], attn_mask)
    except Exception:
        return hidden.mean(dim=1)
    fm = fm.unsqueeze(-1).to(hidden.dtype)
    return (hidden * fm).sum(1) / fm.sum(1).clamp(min=1e-6)

def wavlm_embed(input_values, attn_mask):
    out = wavlm(input_values, attention_mask=attn_mask).last_hidden_state
    return masked_mean(out, attn_mask, wavlm)

# %% [markdown]
# ## 2B. audeering TRAINABLE (mở băng — khác exp08 là frozen)
# Nạp backbone tay + head dimensional gốc; mở băng `UNFREEZE_AUD` lớp trên. Đặc trưng fuse = [hidden(1024) | vad3].

# %%
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
aud.load_state_dict(bb_sd, strict=False)
_hid = _sd["classifier.dense.weight"].shape[0]
aud_head = nn.Sequential(nn.Linear(_hid, _hid), nn.Tanh(), nn.Linear(_hid, _sd["classifier.out_proj.weight"].shape[0]))
aud_head[0].weight.data.copy_(_sd["classifier.dense.weight"]); aud_head[0].bias.data.copy_(_sd["classifier.dense.bias"])
aud_head[2].weight.data.copy_(_sd["classifier.out_proj.weight"]); aud_head[2].bias.data.copy_(_sd["classifier.out_proj.bias"])
aud = aud.to(device); aud_head = aud_head.to(device)
aud.config.layerdrop = 0.0   # ⚠️ tắt layerdrop khi dùng gradient-checkpointing (tránh CheckpointError)
AUD_DIM = _hid + 3   # = 1027 (khớp exp08 để warm-start heads)

# RESUME: nếu checkpoint là ft_joint_full.pt (có 'aud') → khôi phục audeering ĐÃ fine-tune (đè pretrained)
if ckpt is not None and "aud" in ckpt:
    amiss, aunexp = aud.load_state_dict(ckpt["aud"], strict=False)
    print(f"🔁 RESUME audeering từ checkpoint: thiếu {len(amiss)} / dư {len(aunexp)} key (kỳ vọng ~0).")
    if "aud_head" in ckpt:
        aud_head.load_state_dict(ckpt["aud_head"]); print("🔁 RESUME aud_head từ checkpoint.")
else:
    print("ℹ️ Checkpoint không có 'aud' → audeering khởi từ pretrained gốc (chế độ warm-start exp08).")

# Đóng băng partial audeering: mở UNFREEZE_AUD lớp trên + head dimensional luôn trainable
for p in aud.parameters():
    p.requires_grad = False
_al = aud.encoder.layers
for layer in _al[max(0, len(_al) - UNFREEZE_AUD):]:
    for p in layer.parameters():
        p.requires_grad = True
for p in aud_head.parameters():
    p.requires_grad = True
print(f"audeering: {len(_al)} lớp · mở băng {min(UNFREEZE_AUD, len(_al))} → "
      f"{sum(p.numel() for p in aud.parameters() if p.requires_grad)/1e6:.1f}M param train (hidden {_hid}, fuse dim {AUD_DIM})")

if USE_GRAD_CKPT:
    aud.gradient_checkpointing_enable()
    if hasattr(aud, "enable_input_require_grads"):
        aud.enable_input_require_grads()

def aud_embed(input_values, attn_mask):
    """Trả về [hidden(1024) | vad3] — vad3 từ head dimensional gốc, theo thứ tự VAL,ARO,DOM."""
    h = masked_mean(aud(input_values, attention_mask=attn_mask).last_hidden_state, attn_mask, aud)
    out = aud_head(h)   # [B,3] thứ tự gốc audeering: (arousal, dominance, valence)
    vad = torch.stack([1 + 4 * out[:, 2], 1 + 4 * out[:, 0], 1 + 4 * out[:, 1]], dim=1)  # → VAL,ARO,DOM
    return torch.cat([h, vad], dim=1)

# %% [markdown]
# ## 3. Đọc & gộp nhãn theo wavID (như exp08)

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
# ## 4. Dataset/loader — trả về CẢ raw wave (cho WavLM) + input_values audeering
# Hai backbone cần đầu vào khác nhau: WavLM nhận wave thô; audeering nhận wave đã chuẩn hóa bởi processor.
# Cùng độ dài → dùng chung attention mask theo mức sample.

# %%
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

train_stems = [s for s in train_df["wavID"] if target_map.get(s) is not None]
if LIMIT_TRAIN:
    train_stems = train_stems[:LIMIT_TRAIN]
lab = train_df.set_index("wavID")

# Chuẩn hóa: lấy TỪ checkpoint nếu warm-start (để khớp head đã train); không thì fit từ data.
if ckpt is not None and "emos_mu" in ckpt:
    emos_mu = float(ckpt["emos_mu"]); emos_sd = float(ckpt["emos_sd"])
    vad_mu = np.asarray(ckpt["vad_mu"], dtype=np.float32); vad_sd = np.asarray(ckpt["vad_sd"], dtype=np.float32)
    print(f"Chuẩn hóa TỪ ckpt: emos μ={emos_mu:.3f} σ={emos_sd:.3f} | vad μ={np.round(vad_mu,2)}")
else:
    def _zfit(a):
        a = np.asarray(a, dtype=np.float32); return float(np.nanmean(a)), float(np.nanstd(a) + 1e-6)
    emos_mu, emos_sd = _zfit([lab.loc[s, "emos"] for s in train_stems])
    if HAS_VAD:
        vad_mu = np.array([_zfit([lab.loc[s, c] for s in train_stems])[0] for c in ["val", "aro", "dom"]], np.float32)
        vad_sd = np.array([_zfit([lab.loc[s, c] for s in train_stems])[1] for c in ["val", "aro", "dom"]], np.float32)
    else:
        vad_mu = np.zeros(3, np.float32); vad_sd = np.ones(3, np.float32)
    print(f"Chuẩn hóa fit từ data: emos μ={emos_mu:.3f} σ={emos_sd:.3f}")

def onehot_target(tgt):
    v = np.zeros(len(EMOTIONS5), dtype=np.float32)
    if tgt in EMOTIONS5:
        v[EMOTIONS5.index(tgt)] = 1.0
    return v

def load_pair(sid):
    """Trả về (wave_thô, iv_audeering) cùng độ dài; None nếu thiếu file."""
    p = os.path.join(WAV_DIR, sid if str(sid).endswith(".wav") else str(sid) + ".wav")
    if not os.path.exists(p):
        return None
    wave, _ = librosa.load(p, sr=SR, mono=True)
    wave = wave[: MAX_SECONDS * SR].astype(np.float32)
    iv = np.asarray(aud_proc(wave, sampling_rate=SR).input_values[0], dtype=np.float32)
    return wave, iv

class JointDataset(Dataset):
    def __init__(self, stems):
        self.stems = [s for s in stems if load_pair(s) is not None]
    def __len__(self):
        return len(self.stems)
    def __getitem__(self, i):
        s = self.stems[i]
        wave, iv = load_pair(s)
        emos = (float(lab.loc[s, "emos"]) - emos_mu) / emos_sd
        if HAS_VAD:
            vad = (np.array([lab.loc[s, "val"], lab.loc[s, "aro"], lab.loc[s, "dom"]], np.float32) - vad_mu) / vad_sd
        else:
            vad = np.zeros(3, dtype=np.float32)
        cat = np.array([lab.loc[s, f"cat{j}"] for j in range(len(EMOTIONS5))], dtype=np.float32)
        return {"wave": wave, "iv": iv, "tgt": onehot_target(target_map.get(s)),
                "emos": np.float32(emos), "vad": vad, "cat": cat,
                "emos_raw": np.float32(lab.loc[s, "emos"]),
                "vad_raw": np.array([lab.loc[s, "val"], lab.loc[s, "aro"], lab.loc[s, "dom"]], np.float32)}

def collate(batch):
    L = max(len(b["wave"]) for b in batch)
    waves = np.zeros((len(batch), L), dtype=np.float32)
    ivs = np.zeros((len(batch), L), dtype=np.float32)
    mask = np.zeros((len(batch), L), dtype=np.float32)
    for i, b in enumerate(batch):
        n = len(b["wave"])
        waves[i, :n] = b["wave"]; ivs[i, :len(b["iv"])] = b["iv"]; mask[i, :n] = 1.0
    return {
        "wave": torch.from_numpy(waves), "iv": torch.from_numpy(ivs), "attn_mask": torch.from_numpy(mask).long(),
        "tgt": torch.from_numpy(np.stack([b["tgt"] for b in batch])),
        "emos": torch.from_numpy(np.stack([b["emos"] for b in batch])).unsqueeze(1),
        "vad": torch.from_numpy(np.stack([b["vad"] for b in batch])),
        "cat": torch.from_numpy(np.stack([b["cat"] for b in batch])),
        "emos_raw": np.stack([b["emos_raw"] for b in batch]),
        "vad_raw": np.stack([b["vad_raw"] for b in batch]),
    }

ds = JointDataset(train_stems)
print("Dataset hợp lệ:", len(ds), "wav")
tr_i, va_i = train_test_split(np.arange(len(ds)), test_size=VAL_FRAC, random_state=SEED)
tr_loader = DataLoader(torch.utils.data.Subset(ds, tr_i), batch_size=BATCH, shuffle=True, collate_fn=collate, num_workers=2)
va_loader = DataLoader(torch.utils.data.Subset(ds, va_i), batch_size=BATCH, shuffle=False, collate_fn=collate, num_workers=2)

# %% [markdown]
# ## 5. Heads (warm-start exp08 nếu khớp) + optimizer 2 backbone + train loop

# %%
from scipy.stats import spearmanr

torch.manual_seed(SEED); np.random.seed(SEED)
N_EMO = len(EMOTIONS5)
TRUNK_IN = WAVLM_DIM + AUD_DIM

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
if ckpt is not None and "heads" in ckpt:
    hmiss, hunexp = heads.load_state_dict(ckpt["heads"], strict=False)
    if len(hmiss) == 0 and len(hunexp) == 0:
        print("🔁 warm-start heads từ exp08: KHỚP hoàn toàn.")
    else:
        print(f"⚠️ heads exp08 lệch (thiếu {len(hmiss)}/dư {len(hunexp)}) → có thể TRUNK_IN khác. Heads init mới phần lệch.")
print(f"Trunk input = {TRUNK_IN} (wavlm {WAVLM_DIM} + aud {AUD_DIM})")

TASKS = ["emos", "cat", "val", "aro", "dom"]
log_var = nn.Parameter(torch.zeros(len(TASKS), device=device))
bb_params = [p for p in wavlm.parameters() if p.requires_grad] + \
            [p for p in aud.parameters() if p.requires_grad] + list(aud_head.parameters())
head_params = list(heads.parameters()) + ([log_var] if USE_UNCERTAINTY else [])
opt = torch.optim.AdamW([{"params": bb_params, "lr": LR_BACKBONE * RESUME_LR_SCALE},
                         {"params": head_params, "lr": LR_HEAD * RESUME_LR_SCALE}], weight_decay=WEIGHT_DECAY)
scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP and device == "cuda")
mse = nn.MSELoss()

def soft_ce(logits, target_dist):
    return -(target_dist * F.log_softmax(logits, dim=1)).sum(1).mean()

def forward_batch(b):
    am = b["attn_mask"].to(device)
    fw = wavlm_embed(b["wave"].to(device), am)
    fa = aud_embed(b["iv"].to(device), am)
    return heads(torch.cat([fw, fa], dim=1), b["tgt"].to(device))

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

def set_train(flag):
    wavlm.train(flag); aud.train(flag); aud_head.train(flag); heads.train(flag)

@torch.no_grad()
def evaluate():
    set_train(False)
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

def snapshot():
    return {"wavlm": {k: v.cpu().clone() for k, v in wavlm.state_dict().items()},
            "aud": {k: v.cpu().clone() for k, v in aud.state_dict().items()},
            "aud_head": {k: v.cpu().clone() for k, v in aud_head.state_dict().items()},
            "heads": {k: v.cpu().clone() for k, v in heads.state_dict().items()}}

CKPT_PATH = os.path.join(OUT_DIR, "ft_joint_full.pt")
def save_full(state, val_emos=float("nan")):
    torch.save({**state, "emos_mu": emos_mu, "emos_sd": emos_sd, "vad_mu": vad_mu, "vad_sd": vad_sd,
                "WAVLM_DIM": WAVLM_DIM, "AUD_DIM": AUD_DIM,
                "UNFREEZE_WAVLM": UNFREEZE_WAVLM, "UNFREEZE_AUD": UNFREEZE_AUD,
                "val_emos": float(val_emos)}, CKPT_PATH)

# Init best từ trạng thái warm-start hiện tại → chỉ lưu nếu train tốt hơn
m0 = evaluate(); best = mean_srcc(m0); best_state = snapshot(); save_full(best_state, m0.get("emos", float("nan")))
print(f"📍 Khởi điểm (warm-start): mean SRCC = {best:.4f} | "
      + " ".join(f"{k}={m0[k]:.3f}" for k in ['emos','val','aro','dom'] if k in m0))

bad = 0
for ep in range(1, EPOCHS + 1):
    set_train(True)
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
        best = sc; best_state = snapshot(); save_full(best_state, m["emos"])
        print(f"   💾 lưu best → {CKPT_PATH} (epoch {ep}, mean {sc:.4f})"); bad = 0
    else:
        bad += 1
        if bad >= PATIENCE:
            print(f"Early stop ở epoch {ep}."); break

# Nạp lại best
wavlm.load_state_dict(best_state["wavlm"]); aud.load_state_dict(best_state["aud"])
aud_head.load_state_dict(best_state["aud_head"]); heads.load_state_dict(best_state["heads"])
final = evaluate()
print("\n✅ VAL (nội bộ) — exp11 (fine-tune CẢ 2 + fusion):")
print(f"   EMOS={final['emos']:.4f} (exp08 {EXP08['emos']})")
if HAS_VAD:
    print(f"   VAL/ARO/DOM={final['val']:.4f}/{final['aro']:.4f}/{final['dom']:.4f} "
          f"(exp08 {EXP08['val']}/{EXP08['aro']}/{EXP08['dom']})")
print(f"   mean SRCC: warm-start {mean_srcc(m0):.4f} → exp11 {mean_srcc(final):.4f} "
      + ("🚀 cải thiện" if mean_srcc(final) > mean_srcc(m0) + 1e-4 else "➖ không cải thiện"))
save_full(best_state, final.get("emos", float("nan")))
print("Đã lưu FULL:", CKPT_PATH, "→ NHỚ Save Version!")

# %% [markdown]
# ## 6. Dự đoán DEV → answer.txt (5 cột cảm xúc từ exp11; QMOS mượn exp07 / UTMOSv2)

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
    pair = load_pair(sid)
    if pair is None:
        return None
    wave, iv = pair
    set_train(False)
    w = torch.from_numpy(wave).unsqueeze(0).to(device)
    ivt = torch.from_numpy(iv).unsqueeze(0).to(device)
    am = torch.ones((1, len(wave)), dtype=torch.long, device=device)
    tgt = torch.from_numpy(onehot_target(target_map.get(sid))).unsqueeze(0).to(device)
    with torch.cuda.amp.autocast(enabled=USE_AMP and device == "cuda"):
        feat = torch.cat([wavlm_embed(w, am), aud_embed(ivt, am)], dim=1)
        emos_p, cat_l, vad_p = heads(feat, tgt)
    emos = float(emos_p.item()) * emos_sd + emos_mu
    cat5 = F.softmax(cat_l, 1)[0].float().cpu().numpy()
    vad3 = vad_p[0].float().cpu().numpy() * vad_sd + vad_mu
    return emos, cat5, vad3

def fmt_cat(p5):
    return "|".join(f"{e}:{p5[i]:.6g}" for i, e in enumerate(EMOTIONS5))

answer_path = os.path.join(OUT_DIR, "answer.txt")
n_real = n_def = 0
with open(answer_path, "w") as f:
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
print(f"Ghi {len(dev_names)} dòng → {answer_path} | cảm xúc thật {n_real}, mặc định {n_def}")

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
os.system(f"cd {OUT_DIR} && zip -j submission_track2_exp11_joint.zip answer.txt && unzip -l submission_track2_exp11_joint.zip")
print("Sẵn sàng nộp:", os.path.join(OUT_DIR, "submission_track2_exp11_joint.zip"))

# %% [markdown]
# ## Ghi chú
# - **exp11 = fine-tune CẢ WavLM + audeering, FUSION 1 model** (khác exp08 audeering frozen, khác exp10 ensemble).
# - **Warm-start:** WavLM + heads từ `ft_emotion_full_20epoch.pt` (exp08) → bắt đầu từ điểm tốt; audeering từ
#   pretrained gốc, mở băng để học thêm. Khởi điểm = đúng exp08 → train chỉ có thể tốt lên (giữ best).
# - **OOM:** đây là cấu hình nặng nhất. Nếu CUDA OOM → giảm `UNFREEZE_WAVLM`/`UNFREEZE_AUD` (4→2),
#   `MAX_SECONDS` (6→5), giữ `BATCH=1` + tăng `ACCUM`.
# - **Checkpoint:** lưu `ft_joint_full.pt` mỗi best (đủ cả 2 backbone + heads) → kernel chết vẫn còn. Save Version!
# - **QMOS** vẫn mượn exp07 (0.548). So sánh nộp: exp11 vs exp08(0.811) vs exp10(ensemble) → chọn bản tốt nhất.
# - Ghi config → kết quả → nhận xét vào `docs/04_experiments_log.md` (exp11).
