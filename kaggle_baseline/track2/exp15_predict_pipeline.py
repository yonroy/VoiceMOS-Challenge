# %% [markdown]
# # VMC2026 Track 2 — exp15 PREDICT-ONLY (nạp checkpoint → chấm DEV, KHÔNG train) — Kaggle
#
# **Mục đích:** bạn ĐÃ có checkpoint exp15 (`ft_mamba_emotion_full*.pt`, lưu cả backbone WavLM + Mamba enc + heads).
# File này **chỉ inference**: dựng lại đúng kiến trúc → nạp trọng số + thống kê chuẩn hóa TỪ ckpt →
# dự đoán 5 cột cảm xúc trên tập DEV → ghép QMOS (exp07/UTMOSv2) → `answer.txt` → zip nộp.
# **KHÔNG** train, **KHÔNG** cần train.csv (chỉ cần wav DEV + metadata.csv để lấy cảm xúc target cho EMOS).
#
# ## Vì sao nhanh
# - Không có vòng train → chỉ 1 lượt forward qua DEV (~2730 mẫu). Việc lâu nhất là trích audeering DEV
#   (~vài phút; có cache thì gần như tức thì).
#
# ## Chuẩn bị input trên Kaggle (Add Input)
# 1. Dataset Track 2 (wav + `metadata.csv` + `sets/dev.scp`).
# 2. **Checkpoint** exp15: dataset chứa `ft_mamba_emotion_full*.pt` (vd `cache_exp8`). Auto-dò; hoặc trỏ `CKPT_PATH`.
# 3. (tùy chọn) cache audeering `aud_dev.npz` để khỏi trích lại.
# 4. (tùy chọn) `answer.txt` exp07 để mượn cột QMOS 0.548.
#
# **Cách chạy:** GPU **T4** + Internet **On** → Add Input → Run All.

# %% [markdown]
# ## 0. Cấu hình — SỬA Ở ĐÂY

# %%
import os, glob

# ── TỰ DÒ DATA_ROOT (quét /kaggle/input tìm thư mục có sets + wav/ + metadata.csv) ──
def find_data_root(search_root="/kaggle/input"):
    cands = []
    for dev_scp in glob.glob(os.path.join(search_root, "**", "sets", "dev.scp"), recursive=True):
        root = os.path.dirname(os.path.dirname(dev_scp))
        score = os.path.isdir(os.path.join(root, "wav")) + os.path.exists(os.path.join(root, "metadata.csv"))
        cands.append((score, root))
    cands.sort(reverse=True)
    return cands

_cands = find_data_root("/kaggle/input")
if _cands:
    print("🔎 Ứng viên DATA_ROOT:")
    for sc, r in _cands:
        print(f"   [{sc}/2] {r}")
    DATA_ROOT = _cands[0][1]
    print(f"👉 Tự chọn DATA_ROOT = {DATA_ROOT}")
else:
    DATA_ROOT = "/kaggle/input/datasets/minhtoan2"   # dự phòng — sửa tay
    print(f"❌ Không thấy sets/dev.scp → dùng dự phòng {DATA_ROOT} (đã Add Input chưa?)")

WAV_DIR      = f"{DATA_ROOT}/wav"
METADATA_CSV = f"{DATA_ROOT}/metadata.csv"     # wavID|emotion|transcript (KHÔNG header) — lấy cảm xúc target cho EMOS
DEV_SCP      = f"{DATA_ROOT}/sets/dev.scp"

OUT_DIR   = "/kaggle/working"
CACHE_DIR = "/kaggle/working/ft_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# ── CHECKPOINT exp15 (đủ backbone + Mamba + heads) ───────────────────────────
CKPT_PATH = ""    # << "" = auto-dò ft_mamba_emotion_full*.pt; hoặc "/kaggle/input/<slug>/ft_mamba_emotion_full (2).pt"

def find_ckpt(explicit):
    """Tìm checkpoint exp15. Khớp cả tên bị thêm hậu tố trùng, vd 'ft_mamba_emotion_full (2).pt'."""
    if explicit and os.path.exists(explicit):
        return explicit
    for base in ["/kaggle/input", "/kaggle/working"]:
        hits = sorted(glob.glob(os.path.join(base, "**", "ft_mamba_emotion_full*.pt"), recursive=True))
        if hits:
            return hits[0]
    return ""

CKPT_PATH = find_ckpt(CKPT_PATH)
assert CKPT_PATH, "❌ Không thấy checkpoint ft_mamba_emotion_full*.pt. Đã Add Input dataset chứa ckpt chưa?"
print("✅ Dùng checkpoint:", CKPT_PATH)

# (Tùy chọn) tái dùng cache audeering DEV — quét đệ quy (file có thể nằm trong archive/)
CACHE_INPUT = "/kaggle/input/cache-exp8"   # << SỬA slug (hoặc "")
if CACHE_INPUT and os.path.isdir(CACHE_INPUT):
    import shutil
    _n = 0
    for _fp in glob.glob(os.path.join(CACHE_INPUT, "**", "aud_*.npz"), recursive=True):
        shutil.copy(_fp, os.path.join(CACHE_DIR, os.path.basename(_fp))); _n += 1
    print(f"📦 Copy {_n} file aud_*.npz từ {CACHE_INPUT}")

# Mượn cột QMOS exp07 (0.548). Trỏ answer.txt exp07 nếu có; không thì UTMOSv2.
EXP07_ANSWER = "/kaggle/input/exp07-answer/answer.txt"   # << (tùy chọn)

# ── Siêu tham số PHẢI KHỚP lúc train exp15 (ckpt không lưu các số này của Mamba) ──
MAMBA_DMODEL  = 256
MAMBA_LAYERS  = 2
MAMBA_DSTATE  = 16
BIDIRECTIONAL = True
TRUNK_HIDDEN  = 512
HEAD_HIDDEN   = 128
DROPOUT       = 0.3       # không ảnh hưởng eval (model.eval() tắt dropout) — chỉ để dựng đúng shape

DEVICE       = "cuda"
SR           = 16000
MAX_SECONDS  = 6          # khớp lúc train (exp15 = 6)
USE_AMP      = True
LIMIT_DEV    = None       # << để None chấm ĐỦ 2730; đặt 20 để smoke-test nhanh

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
for p in [WAV_DIR, METADATA_CSV, DEV_SCP, CKPT_PATH]:
    print(("  ✅ " if os.path.exists(p) else "  ❌ THIẾU ") + p)

# %% [markdown]
# ## 1. Cài đặt + tải code SAILER (để dựng đúng kiến trúc WavLM rồi nạp ckpt đè lên)

# %%
import sys, subprocess

def pip_install(*pkgs):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], check=True)

pip_install("loralib", "speechbrain", "speechmos", "librosa", "soundfile",
            "scipy", "scikit-learn", "pandas", "tqdm")

# Mamba kernel CUDA (tùy chọn — không có thì dùng Mamba thuần PyTorch, inference vẫn ổn vì chỉ 1 lượt forward)
INSTALL_MAMBA_SSM = True
if INSTALL_MAMBA_SSM:
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ninja"], check=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-build-isolation", "causal-conv1d>=1.2.0"], check=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-build-isolation", "mamba-ssm"], check=True)
        print("✅ Cài mamba-ssm xong (dùng kernel CUDA nếu import được).")
    except Exception as e:
        print("⚠️ Cài mamba-ssm thất bại:", repr(e), "→ Mamba thuần PyTorch (inference vẫn chạy).")

REPO_DIR = "/kaggle/working/vox-profile-release"
if not os.path.exists(REPO_DIR):
    subprocess.run(["git", "clone", "--depth", "1",
                    "https://github.com/tiantiaf0627/vox-profile-release.git", REPO_DIR], check=True)
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

# %% [markdown]
# ## 2. Nạp checkpoint → dựng WavLM → load trọng số backbone đã fine-tune

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F

device = DEVICE if torch.cuda.is_available() else "cpu"
print("Device:", device, ("✅ " + torch.cuda.get_device_name(0)) if device == "cuda" else "⚠️ CPU (chậm)")

ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)  # ckpt có numpy → cần False
assert "wavlm" in ckpt, "❌ Checkpoint KHÔNG có 'wavlm' (backbone) → không inference được. Cần ft_mamba_emotion_full*.pt đủ."
print("✅ Nạp ckpt | keys:", list(ckpt.keys()))

# Lấy cấu hình KIẾN TRÚC từ ckpt (để dựng đúng shape head)
USE_MAMBA  = bool(ckpt.get("USE_MAMBA", True))
Z_DIM      = int(ckpt.get("Z_DIM", 256))
AUD_DIM    = int(ckpt.get("AUD_DIM", 0))
USE_AUDEERING = AUD_DIM > 0
UNFREEZE_TOP_LAYERS = int(ckpt.get("UNFREEZE_TOP_LAYERS", 6))
print(f"Từ ckpt: USE_MAMBA={USE_MAMBA} · Z_DIM={Z_DIM} · AUD_DIM={AUD_DIM} (audeering={'ON' if USE_AUDEERING else 'OFF'})")

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
wavlm.config.layerdrop = 0.0

miss, unexp = wavlm.load_state_dict(ckpt["wavlm"], strict=False)
print(f"🔁 load wavlm từ ckpt: thiếu {len(miss)} / dư {len(unexp)} key (kỳ vọng ~0)")
if len(miss) > 20 or len(unexp) > 20:
    print("   ⚠️ Lệch key nhiều → kiểm tra backbone có khớp ckpt không.")
wavlm.eval()

def frame_mask(T, attn_mask):
    if attn_mask is None:
        return torch.ones((1, T), dtype=torch.bool, device=device)
    try:
        return wavlm._get_feature_vector_attention_mask(T, attn_mask).bool()
    except Exception:
        return torch.ones((attn_mask.shape[0], T), dtype=torch.bool, device=attn_mask.device)

def masked_mean(hidden, attn_mask):
    if attn_mask is None:
        return hidden.mean(dim=1)
    fm = frame_mask(hidden.shape[1], attn_mask).unsqueeze(-1).to(hidden.dtype)
    return (hidden * fm).sum(1) / fm.sum(1).clamp(min=1e-6)

# %% [markdown]
# ## 3. audeering MSP-dim (FROZEN) — chỉ dựng nếu ckpt có dùng (AUD_DIM>0)

# %%
import numpy as np
import librosa
from tqdm.auto import tqdm

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
    aud_head = nn.Sequential(nn.Linear(_hid, _hid), nn.Tanh(), nn.Linear(_hid, _sd["classifier.out_proj.weight"].shape[0]))
    aud_head[0].weight.data.copy_(_sd["classifier.dense.weight"]); aud_head[0].bias.data.copy_(_sd["classifier.dense.bias"])
    aud_head[2].weight.data.copy_(_sd["classifier.out_proj.weight"]); aud_head[2].bias.data.copy_(_sd["classifier.out_proj.bias"])
    aud_backbone = aud_backbone.to(device).eval()
    aud_head = aud_head.to(device).eval()
    assert _hid + 3 == AUD_DIM, f"⚠️ AUD_DIM dựng ({_hid+3}) ≠ ckpt ({AUD_DIM}) → audeering không khớp!"
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
        vad = np.array([1 + 4 * out[2], 1 + 4 * out[0], 1 + 4 * out[1]], dtype=np.float32)  # [VAL,ARO,DOM]
        store[s] = np.concatenate([h[0].cpu().numpy(), vad]).astype(np.float32)
        if (i + 1) % 500 == 0:
            np.savez(cache_path, **store)
    if todo:
        np.savez(cache_path, **store)
    return store

# %% [markdown]
# ## 4. Cảm xúc target theo wavID (cho one-hot điều kiện của head EMOS)

# %%
def load_target_emotions():
    tgt = {}
    with open(METADATA_CSV, encoding="utf-8") as f:
        for ln in f:
            parts = ln.strip().split("|")
            if len(parts) >= 2:
                tgt[stem(parts[0])] = norm_emotion(parts[1])
    return tgt

target_map = load_target_emotions()
print("Target cảm xúc:", len(target_map), "wav")

def onehot_target(tgt):
    v = np.zeros(len(EMOTIONS5), dtype=np.float32)
    if tgt in EMOTIONS5:
        v[EMOTIONS5.index(tgt)] = 1.0
    return v

# %% [markdown]
# ## 5. Khối Mamba (giống exp15) + MambaEncoder

# %%
import math

try:
    from mamba_ssm import Mamba as _OfficialMamba
    _HAS_MAMBA_SSM = True
    print("✅ Dùng mamba-ssm (CUDA kernel)")
except Exception:
    _HAS_MAMBA_SSM = False
    print("ℹ️ Không có mamba-ssm → Mamba thuần PyTorch")

class MambaBlockTorch(nn.Module):
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
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.d_state = d_state

    def forward(self, x):
        B, L, _ = x.shape
        xin, z = self.in_proj(x).chunk(2, dim=-1)
        xin = xin.transpose(1, 2)
        xin = self.conv1d(xin)[..., :L].transpose(1, 2)
        xin = F.silu(xin)
        y = self._ssm(xin) * F.silu(z)
        return self.out_proj(y)

    def _ssm(self, x):
        A = -torch.exp(self.A_log)
        delta, Bm, Cm = torch.split(self.x_proj(x), [self.dt_rank, self.d_state, self.d_state], dim=-1)
        delta = F.softplus(self.dt_proj(delta))
        dA = torch.exp(delta.unsqueeze(-1) * A)
        dB_x = delta.unsqueeze(-1) * Bm.unsqueeze(2) * x.unsqueeze(-1)
        h = torch.zeros(x.shape[0], self.d_inner, self.d_state, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(x.shape[1]):
            h = dA[:, t] * h + dB_x[:, t]
            ys.append((h * Cm[:, t].unsqueeze(1)).sum(-1))
        return torch.stack(ys, dim=1) + x * self.D

class MambaLayer(nn.Module):
    def __init__(self, d_model, d_state):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mix = _OfficialMamba(d_model=d_model, d_state=d_state, d_conv=4, expand=2) \
            if _HAS_MAMBA_SSM else MambaBlockTorch(d_model, d_state=d_state)
    def forward(self, x):
        return x + self.mix(self.norm(x))

class MambaEncoder(nn.Module):
    def __init__(self, d_in, d_model, n_layers, d_state, z_dim, bidir):
        super().__init__()
        self.bidir = bidir
        self.proj = nn.Linear(d_in, d_model)
        self.fwd = nn.ModuleList([MambaLayer(d_model, d_state) for _ in range(n_layers)])
        if bidir:
            self.bwd = nn.ModuleList([MambaLayer(d_model, d_state) for _ in range(n_layers)])
        self.attn = nn.Linear(d_model, 1)
        self.out = nn.Linear(d_model, z_dim)

    @staticmethod
    def _run(layers, h):
        for L in layers:
            h = L(h)
        return h

    def forward(self, x, mask):
        with torch.cuda.amp.autocast(enabled=False):
            x = x.float()
            h = self.proj(x)
            out = self._run(self.fwd, h)
            if self.bidir:
                out = out + torch.flip(self._run(self.bwd, torch.flip(h, dims=[1])), dims=[1])
            a = self.attn(out).squeeze(-1).masked_fill(~mask, float("-inf"))
            w = torch.softmax(a, dim=1).unsqueeze(-1)
            return self.out((out * w).sum(1))

# %% [markdown]
# ## 6. Dựng enc + heads → nạp trọng số từ ckpt + lấy chuẩn hóa từ ckpt

# %%
N_EMO = len(EMOTIONS5)
WAVLM_BRANCH = Z_DIM if USE_MAMBA else WAVLM_DIM
TRUNK_IN = WAVLM_BRANCH + (AUD_DIM if USE_AUDEERING else 0)

enc = MambaEncoder(WAVLM_DIM, MAMBA_DMODEL, MAMBA_LAYERS, MAMBA_DSTATE, Z_DIM, BIDIRECTIONAL).to(device) \
    if USE_MAMBA else None

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
hm, hu = heads.load_state_dict(ckpt["heads"], strict=False)
print(f"🔁 load heads từ ckpt: thiếu {len(hm)} / dư {len(hu)} key (kỳ vọng 0)")
if USE_MAMBA:
    assert ckpt.get("enc") is not None, "❌ ckpt USE_MAMBA=True nhưng KHÔNG có 'enc' → không inference đúng được."
    em, eu = enc.load_state_dict(ckpt["enc"], strict=False)
    print(f"🔁 load Mamba enc từ ckpt: thiếu {len(em)} / dư {len(eu)} key (kỳ vọng 0)")
heads.eval()
if USE_MAMBA:
    enc.eval()

# Chuẩn hóa LẤY TỪ ckpt (head dự đoán ở thang z-score này → phải giải chuẩn đúng thang)
emos_mu = float(ckpt["emos_mu"]); emos_sd = float(ckpt["emos_sd"])
vad_mu = np.asarray(ckpt["vad_mu"], dtype=np.float32); vad_sd = np.asarray(ckpt["vad_sd"], dtype=np.float32)
print(f"Chuẩn hóa từ ckpt: emos μ={emos_mu:.3f} σ={emos_sd:.3f} | vad μ={np.round(vad_mu,2)}")

def wavlm_branch(input_values, attn_mask):
    out = wavlm(input_values, attention_mask=attn_mask).last_hidden_state
    if USE_MAMBA:
        return enc(out, frame_mask(out.shape[1], attn_mask))
    return masked_mean(out, attn_mask)

print(f"Trunk input = {TRUNK_IN} (wavlm-branch {WAVLM_BRANCH} [{'Mamba' if USE_MAMBA else 'mean-pool'}] + aud {AUD_DIM if USE_AUDEERING else 0})")

# %% [markdown]
# ## 7. Dự đoán DEV → answer.txt (5 cột cảm xúc; QMOS mượn exp07/UTMOSv2)

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
        print(f"✅ Mượn QMOS exp07 ({EXP07_ANSWER}): {len(d)//2} wav")
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
    iv = torch.from_numpy(wave).unsqueeze(0).to(device)
    am = torch.ones((1, len(wave)), dtype=torch.long, device=device)
    tgt = torch.from_numpy(onehot_target(target_map.get(sid))).unsqueeze(0).to(device)
    with torch.cuda.amp.autocast(enabled=USE_AMP and device == "cuda"):
        fw = wavlm_branch(iv, am)
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
os.system(f"cd {OUT_DIR} && zip -j submission_track2_exp15_predict.zip answer.txt "
          f"&& unzip -l submission_track2_exp15_predict.zip")
print("Sẵn sàng nộp:", os.path.join(OUT_DIR, "submission_track2_exp15_predict.zip"))

# %% [markdown]
# ## Ghi chú
# - File này **chỉ inference** — không train, không cần train.csv. Dùng khi đã có `ft_mamba_emotion_full*.pt`.
# - ⚠️ **Siêu tham số Mamba/heads (MAMBA_DMODEL/LAYERS/DSTATE, TRUNK_HIDDEN, HEAD_HIDDEN) PHẢI khớp lúc train**
#   (ckpt không lưu các số này) — nếu lúc train exp15 bạn đổi, hãy sửa cho khớp ở cell 0, nếu không load_state_dict
#   sẽ lệch key / sai shape.
# - `USE_MAMBA`, `Z_DIM`, `AUD_DIM`, `UNFREEZE_TOP_LAYERS` thì **đọc tự động từ ckpt**.
# - QMOS: tốt nhất Add Input `answer.txt` exp07 (0.548); không có thì tự chấm UTMOSv2.
# - Smoke-test: đặt `LIMIT_DEV=20` chạy thử cho nhanh, OK rồi đặt lại `None` để chấm đủ 2730.
