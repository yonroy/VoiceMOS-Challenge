# %% [markdown]
# # VMC2026 Track 2 — exp15 (WavLM FINE-TUNE + MAMBA head cho 5 cột cảm xúc) — Kaggle
#
# **Ý tưởng:** exp08 fine-tune WavLM nhưng vẫn **mean-pool** đặc trưng theo thời gian → 1 vector/wav
# (vứt bỏ động lực thời gian: lên/xuống giọng, ngắt quãng, run giọng — rất quan trọng cho cảm xúc).
# exp15 **thay mean-pool bằng MAMBA head** (bộ mã hóa chuỗi học được, độ phức tạp tuyến tính) → kỳ vọng
# nắm temporal dynamics tốt hơn. Tham khảo: MambaRate (AudioMOS 2025, arXiv:2507.12090).
#
# ## Kiến trúc (= exp08 đổi đúng 1 chỗ: pool → Mamba)
# ```
#  wav ─► WavLM-large (SAILER warm-start, mở băng N lớp, TRAINABLE) ─► hidden states (B, T, 1024)
#                                                                            │  (KHÔNG mean-pool)
#                                                              MambaEncoder (proj 1024→d, Mamba×L 2 chiều,
#                                                              attentive-pool có mask) ─► z (B, Z_DIM)
#                                                                            │
#       (tùy chọn) audeering MSP-dim FROZEN [emb|vad3] ──concat──► TRUNK ─┬─► EMOS (+ one-hot target)
#                                                                          ├─► CAT (5, soft-CE)
#                                                                          └─► VAD (3)
#  QMOS: KHÔNG train ở đây → mượn cột QMOS exp07 (0.548) hoặc UTMOSv2.
# ```
# - **Cờ `USE_MAMBA`:** True = Mamba head; False = quay về `masked_mean` = **đúng exp08**
#   → đây là **ablation chính cho paper** ("Mamba temporal head vs mean-pooling", CÙNG backbone fine-tune).
#
# ## ⚠️ Đánh đổi / gotcha (đã phòng trong code)
# - Fine-tune = chạy lại WavLM mỗi epoch (không cache được) → **lần đầu BẮT BUỘC `LIMIT_TRAIN=300`, `LIMIT_DEV=20`**.
# - `mamba-ssm` khó cài Kaggle → tự fallback **Mamba thuần PyTorch** (vòng-lặp-thời-gian). Bản này khi fine-tune
#   **chậm + nặng RAM hơn** → cap `MAX_SECONDS=6`, `BATCH=2`. OOM/quá chậm → hạ MAX_SECONDS→5, MAMBA_LAYERS→1,
#   hoặc thử cài `mamba-ssm causal-conv1d`.
# - `layerdrop=0` (tránh CheckpointError khi grad-ckpt — bài học exp12). KHÔNG đụng numpy (lệch ABI).
# - **Checkpoint lưu CẢ backbone + Mamba + heads mỗi best** (bài học exp08 mất backbone).
#
# ## 🔁 RESUME (yêu cầu của user): "nếu có checkpoint thì train TIẾP, không train lại từ đầu"
# - Notebook **tự dò** `ft_mamba_emotion_full.pt` trong `/kaggle/input` và `/kaggle/working` (hoặc trỏ tay `RESUME_CKPT`).
# - Có ckpt đủ (backbone WavLM + Mamba enc + heads) → **nạp lại trạng thái + thống kê chuẩn hóa TỪ ckpt** rồi train tiếp;
#   `best` khởi tạo = điểm VAL của ckpt → chỉ ghi đè khi train tiếp **TỐT HƠN** (không sợ tụt). `RESUME_LR_SCALE<1` để hạ LR.
# - KHÔNG có ckpt → train mới từ SAILER warm-start như cũ (hành vi exp15 gốc giữ nguyên).
#
# **Cách chạy Kaggle:** GPU **T4** + Internet **On** → Add Input dataset Track 2 (+ Add Input checkpoint cũ nếu muốn resume)
# → sửa `DATA_ROOT` → Run All.

# %% [markdown]
# ## 0. Cấu hình — SỬA Ở ĐÂY

# %%
import os, glob

# ── TỰ DÒ DATA_ROOT (quét /kaggle/input tìm thư mục có sets/train.csv + wav/ + metadata.csv) ──
def find_data_root(search_root="/kaggle/input"):
    cands = []
    for train_csv in glob.glob(os.path.join(search_root, "**", "sets", "train.csv"), recursive=True):
        root = os.path.dirname(os.path.dirname(train_csv))          # .../<root>/sets/train.csv → <root>
        score = os.path.isdir(os.path.join(root, "wav")) + os.path.exists(os.path.join(root, "metadata.csv"))
        cands.append((score, root))
    cands.sort(reverse=True)                                        # ưu tiên thư mục đủ wav + metadata
    return cands

_cands = find_data_root("/kaggle/input")
if _cands:
    print("🔎 Ứng viên DATA_ROOT (điểm cao = đủ wav+metadata):")
    for sc, r in _cands:
        print(f"   [{sc}/2] {r}")
    DATA_ROOT = _cands[0][1]
    print(f"👉 Tự chọn DATA_ROOT = {DATA_ROOT}")
else:
    DATA_ROOT = "/kaggle/input/datasets/minhtoan2"   # dự phòng — sửa tay nếu auto-dò không thấy
    print(f"❌ Không thấy sets/train.csv trong /kaggle/input → dùng dự phòng {DATA_ROOT} (đã Add Input chưa?)")

WAV_DIR      = f"{DATA_ROOT}/wav"
METADATA_CSV = f"{DATA_ROOT}/metadata.csv"     # wavID|emotion|transcript (KHÔNG header)
TRAIN_CSV    = f"{DATA_ROOT}/sets/train.csv"   # lisID|wavID|qMOS|emoCat|eMOS|val|dom|aro
DEV_SCP      = f"{DATA_ROOT}/sets/dev.scp"

OUT_DIR   = "/kaggle/working"
CACHE_DIR = "/kaggle/working/ft_cache"         # cache audeering (.npz) — WavLM/Mamba KHÔNG cache (đang train)
os.makedirs(CACHE_DIR, exist_ok=True)

# (Tùy chọn) tái dùng cache audeering cũ (read-only /kaggle/input → copy sang working)
# Dataset cache_exp8: aud_*.npz nằm trong thư mục con archive/ → quét ĐỆ QUY để bắt mọi vị trí.
CACHE_INPUT = "/kaggle/input/cache-exp8"   # << SỬA slug (dataset cache_exp8 → Kaggle đổi _→-); hoặc ""
if CACHE_INPUT and os.path.isdir(CACHE_INPUT):
    import shutil
    _n = 0
    for _fp in glob.glob(os.path.join(CACHE_INPUT, "**", "aud_*.npz"), recursive=True):
        shutil.copy(_fp, os.path.join(CACHE_DIR, os.path.basename(_fp))); _n += 1
    print(f"📦 Tái dùng cache: copy {_n} file aud_*.npz (quét đệ quy {CACHE_INPUT})")
else:
    print(f"ℹ️ Không thấy CACHE_INPUT={CACHE_INPUT} → sẽ tự trích audeering.")

# Mượn cột QMOS exp07 (0.548). Trỏ answer.txt exp07 nếu có; không thì UTMOSv2.
EXP07_ANSWER = "/kaggle/input/exp07-answer/answer.txt"   # << (tùy chọn)

# ── Cờ Mamba (ablation chính) ────────────────────────────────────────────────
USE_MAMBA           = True        # True = Mamba head; False = mean-pool = ĐÚNG exp08

# ── Siêu tham số Mamba head ──────────────────────────────────────────────────
MAMBA_DMODEL        = 256
MAMBA_LAYERS        = 2
MAMBA_DSTATE        = 16
BIDIRECTIONAL       = True
Z_DIM               = 256         # chiều vector ra sau attentive-pool, thay cho emb WavLM mean-pool

# ── Fine-tune / siêu tham số (kế thừa exp08) ─────────────────────────────────
DEVICE              = "cuda"
SR                  = 16000
MAX_SECONDS         = 6           # giảm từ 8 (exp08) vì Mamba backprop-through-time nặng RAM hơn
UNFREEZE_TOP_LAYERS = 6           # số lớp Transformer trên cùng được train (0 = freeze hết)
TRUNK_HIDDEN        = 512
HEAD_HIDDEN         = 128
DROPOUT             = 0.3
LR_BACKBONE         = 1e-5
LR_HEAD             = 1e-3        # cho Mamba + trunk + head (train từ đầu)
WEIGHT_DECAY        = 1e-5
EPOCHS              = 12
PATIENCE            = 3
BATCH               = 2           # nhỏ (backbone to + Mamba); bù bằng ACCUM
ACCUM               = 16          # effective batch = 32
VAL_FRAC            = 0.10
SEED                = 42
USE_AMP             = True
USE_GRAD_CKPT       = True
USE_AUDEERING       = True
USE_UNCERTAINTY     = True
RANK_LAMBDA         = 0.3         # 0 = chỉ MSE (cũ). >0 = thêm pairwise ranking loss (tối ưu thẳng SRCC) cho emos/val/aro/dom
                                  # ⚠️ ranking cần NHIỀU cặp/batch mới mạnh → BATCH nhỏ (2) thì tác dụng yếu (xem Ghi chú)

LIMIT_TRAIN         = 300         # << LẦN ĐẦU 300; chạy thật None
LIMIT_DEV           = 20          # << LẦN ĐẦU 20; chạy thật None

# ── RESUME — train TIẾP từ checkpoint, KHÔNG train lại từ đầu ─────────────────
# Để "" + auto-dò: nếu thấy `ft_mamba_emotion_full.pt` (đủ backbone+Mamba+heads) trong /kaggle/input
# hoặc /kaggle/working → nạp lại rồi train tiếp. Trỏ tay RESUME_CKPT nếu muốn chỉ định file cụ thể.
RESUME_CKPT         = ""          # << "" = auto-dò; hoặc "/kaggle/input/<slug>/ft_mamba_emotion_full.pt"
RESUME_LR_SCALE     = 1.0         # <1.0 hạ LR khi train tiếp (vd 0.5 nếu val đã chững)

def find_resume_ckpt(explicit):
    """Tìm checkpoint exp15 để train tiếp. Ưu tiên đường dẫn user trỏ; không thì auto-dò.
    Khớp cả tên bị Kaggle/Windows thêm hậu tố trùng, vd 'ft_mamba_emotion_full (2).pt'."""
    if explicit and os.path.exists(explicit):
        return explicit
    for base in ["/kaggle/input", "/kaggle/working"]:
        hits = sorted(glob.glob(os.path.join(base, "**", "ft_mamba_emotion_full*.pt"), recursive=True))
        if hits:
            return hits[0]
    return ""

RESUME_CKPT = find_resume_ckpt(RESUME_CKPT)
RESUME      = bool(RESUME_CKPT)
print("🔁 RESUME =", RESUME, ("→ train tiếp từ: " + RESUME_CKPT) if RESUME else "(không thấy ckpt → train MỚI từ đầu)")

# Mốc so (exp08 fine-tune + mean-pool — đối thủ trực tiếp của Mamba head)
EXP08 = {"emos": 0.811, "val": 0.659, "aro": 0.793, "dom": 0.751}

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

print("USE_MAMBA =", USE_MAMBA, "(False → ra đúng exp08)")
print("DATA_ROOT:", DATA_ROOT)
for p in [WAV_DIR, METADATA_CSV, TRAIN_CSV, DEV_SCP]:
    print(("  ✅ " if os.path.exists(p) else "  ❌ THIẾU ") + p)
print(f"Fine-tune: mở băng {UNFREEZE_TOP_LAYERS} lớp · BATCH {BATCH}×ACCUM {ACCUM} · MAX {MAX_SECONDS}s")

# %% [markdown]
# ## 1. Cài đặt + tải code SAILER (clone + sys.path)

# %%
import sys, subprocess

def pip_install(*pkgs):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], check=True)

pip_install("loralib", "speechbrain", "speechmos", "librosa", "soundfile",
            "scipy", "scikit-learn", "pandas", "tqdm")

# Cài kernel CUDA Mamba (nhanh + nhẹ RAM hơn bản thuần PyTorch nhiều). Build hay lỗi/chậm trên Kaggle
# → bọc try/except: lỗi thì BỎ QUA, mục 6a tự fallback Mamba thuần PyTorch. KHÔNG để chết notebook.
INSTALL_MAMBA_SSM = True   # đặt False nếu muốn BỎ QUA, dùng thẳng Mamba thuần PyTorch
if INSTALL_MAMBA_SSM and USE_MAMBA:
    try:
        # --no-build-isolation cho CẢ HAI → dùng torch+CUDA sẵn có của Kaggle để biên dịch (đừng kéo torch khác).
        # Cần ninja để build nhanh. -q ẩn log nên bước này có thể "treo" vài phút khi đang compile — bình thường.
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ninja"], check=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                        "--no-build-isolation", "causal-conv1d>=1.2.0"], check=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                        "--no-build-isolation", "mamba-ssm"], check=True)
        print("✅ Cài mamba-ssm + causal-conv1d xong (sẽ dùng kernel CUDA nếu import được).")
    except Exception as e:
        print("⚠️ Cài mamba-ssm thất bại:", repr(e), "→ dùng Mamba thuần PyTorch (chậm hơn).")
        print("   ℹ️ Vẫn chạy bình thường. Nếu chạy THẬT (LIMIT=None) quá chậm → xem Ghi chú cuối notebook.")

REPO_DIR = "/kaggle/working/vox-profile-release"
if not os.path.exists(REPO_DIR):
    subprocess.run(["git", "clone", "--depth", "1",
                    "https://github.com/tiantiaf0627/vox-profile-release.git", REPO_DIR], check=True)
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

# %% [markdown]
# ## 2. Nạp SAILER → lấy backbone WavLM bên trong để FINE-TUNE (warm-start)

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F

device = DEVICE if torch.cuda.is_available() else "cpu"
print("Device:", device, ("✅ " + torch.cuda.get_device_name(0)) if device == "cuda" else "⚠️ CPU (rất chậm!)")

def find_hf_backbone(module):
    """Tìm submodule kiểu HF WavLM backbone: có .feature_extractor và .encoder.layers."""
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
        print(f"✅ Warm-start SAILER: backbone WavLM tại '.{name}' "
              f"({sum(p.numel() for p in wavlm.parameters())/1e6:.0f}M params)")
    else:
        print("⚠️ Không tìm thấy backbone HF trong wrapper SAILER → fallback WavLM trắng.")
except Exception as e:
    print("⚠️ Lỗi nạp SAILER wrapper:", repr(e), "→ fallback WavLM trắng.")

if wavlm is None:
    from transformers import WavLMModel
    wavlm = WavLMModel.from_pretrained("microsoft/wavlm-large")
    print("ℹ️ Fallback: microsoft/wavlm-large (KHÔNG warm-start SAILER).")

wavlm = wavlm.to(device)
WAVLM_DIM = int(wavlm.config.hidden_size)
wavlm.config.layerdrop = 0.0   # ⚠️ tránh CheckpointError khi grad-ckpt (bài học exp12)

# ── RESUME: nạp trọng số backbone đã fine-tune từ checkpoint (đè lên warm-start SAILER) ──
resume_ckpt = None
if RESUME:
    resume_ckpt = torch.load(RESUME_CKPT, map_location="cpu", weights_only=False)  # ckpt có numpy → cần False
    assert "wavlm" in resume_ckpt, ("❌ Checkpoint KHÔNG có 'wavlm' (backbone) → không resume được. "
                                    "Dùng file ft_mamba_emotion_full.pt do exp15 lưu.")
    if resume_ckpt.get("USE_MAMBA", USE_MAMBA) != USE_MAMBA:
        print(f"   ⚠️ ckpt USE_MAMBA={resume_ckpt.get('USE_MAMBA')} ≠ cấu hình hiện tại {USE_MAMBA} → kiến trúc LỆCH! "
              "Đặt USE_MAMBA cho khớp ckpt.")
    miss, unexp = wavlm.load_state_dict(resume_ckpt["wavlm"], strict=False)
    print(f"🔁 RESUME load wavlm từ ckpt: thiếu {len(miss)} / dư {len(unexp)} key (kỳ vọng ~0). keys ckpt:", list(resume_ckpt.keys()))
    if len(miss) > 20 or len(unexp) > 20:
        print("   ⚠️ Lệch key nhiều → kiểm tra UNFREEZE_TOP_LAYERS / backbone có khớp ckpt không.")

# ── Đóng băng partial: feature-extractor + tất cả trừ UNFREEZE_TOP_LAYERS lớp trên ──
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

def frame_mask(T, attn_mask):
    """attn_mask (B, Lwav) → frame-mask (B, T) bool (True=frame thật). Khớp downsample của WavLM."""
    if attn_mask is None:
        return torch.ones((1, T), dtype=torch.bool, device=device)
    try:
        fm = wavlm._get_feature_vector_attention_mask(T, attn_mask)
        return fm.bool()
    except Exception:
        return torch.ones((attn_mask.shape[0], T), dtype=torch.bool, device=attn_mask.device)

def masked_mean(hidden, attn_mask):
    """Mean-pool theo thời gian bỏ pad (đường exp08 khi USE_MAMBA=False)."""
    if attn_mask is None:
        return hidden.mean(dim=1)
    fm = frame_mask(hidden.shape[1], attn_mask).unsqueeze(-1).to(hidden.dtype)
    return (hidden * fm).sum(1) / fm.sum(1).clamp(min=1e-6)

# %% [markdown]
# ## 3. Nạp audeering MSP-dim (FROZEN) — đặc trưng phụ (như exp08)

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
    aud_backbone.load_state_dict(bb_sd, strict=False)
    _hid = _sd["classifier.dense.weight"].shape[0]
    _out = _sd["classifier.out_proj.weight"].shape[0]
    aud_head = nn.Sequential(nn.Linear(_hid, _hid), nn.Tanh(), nn.Linear(_hid, _out))
    aud_head[0].weight.data.copy_(_sd["classifier.dense.weight"]); aud_head[0].bias.data.copy_(_sd["classifier.dense.bias"])
    aud_head[2].weight.data.copy_(_sd["classifier.out_proj.weight"]); aud_head[2].bias.data.copy_(_sd["classifier.out_proj.bias"])
    aud_backbone = aud_backbone.to(device).eval()
    aud_head = aud_head.to(device).eval()
    AUD_DIM = _hid + 3
    print(f"✅ audeering frozen (đặc trưng phụ {AUD_DIM}-D = emb {_hid} + vad 3)")

# %%
import numpy as np
import librosa
from tqdm.auto import tqdm

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
        out = aud_head(h)[0].cpu().numpy()                  # [arousal, dominance, valence] ∈[0,1]
        vad = np.array([1 + 4 * out[2], 1 + 4 * out[0], 1 + 4 * out[1]], dtype=np.float32)  # [VAL,ARO,DOM]
        store[s] = np.concatenate([h[0].cpu().numpy(), vad]).astype(np.float32)
        if (i + 1) % 500 == 0:
            np.savez(cache_path, **store)
    if todo:
        np.savez(cache_path, **store)
    return store

# %% [markdown]
# ## 4. Đọc & gộp nhãn theo wavID (EMOS / VAD / CAT) — như exp08

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

def _zfit(arr):
    a = np.asarray(arr, dtype=np.float32)
    return float(np.nanmean(a)), float(np.nanstd(a) + 1e-6)

if RESUME and resume_ckpt is not None:
    # QUAN TRỌNG: lấy chuẩn hóa TỪ ckpt (head đã train theo thang này) — KHÔNG tính lại để khỏi lệch thang
    emos_mu = float(resume_ckpt["emos_mu"]); emos_sd = float(resume_ckpt["emos_sd"])
    vad_mu = np.asarray(resume_ckpt["vad_mu"], dtype=np.float32)
    vad_sd = np.asarray(resume_ckpt["vad_sd"], dtype=np.float32)
    print(f"🔁 RESUME: dùng chuẩn hóa TỪ ckpt: emos μ={emos_mu:.3f} σ={emos_sd:.3f} | vad μ={np.round(vad_mu,2)}")
else:
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
    L = max(len(b["wave"]) for b in batch)
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

from sklearn.model_selection import train_test_split
ds = EmoDataset(train_stems)
print("Dataset hợp lệ:", len(ds), "wav")
tr_i, va_i = train_test_split(np.arange(len(ds)), test_size=VAL_FRAC, random_state=SEED)
tr_loader = DataLoader(torch.utils.data.Subset(ds, tr_i), batch_size=BATCH, shuffle=True, collate_fn=collate, num_workers=2)
va_loader = DataLoader(torch.utils.data.Subset(ds, va_i), batch_size=BATCH, shuffle=False, collate_fn=collate, num_workers=2)

# %% [markdown]
# ## 6a. Khối MAMBA (thuần PyTorch, fallback nếu không có `mamba-ssm`)
# Theo "mamba-minimal" — đúng công thức selective SSM, chỉ chậm hơn kernel CUDA. Chạy trong fp32 cho ổn định.

# %%
import math

try:
    from mamba_ssm import Mamba as _OfficialMamba
    _HAS_MAMBA_SSM = True
    print("✅ Dùng mamba-ssm (CUDA kernel)")
except Exception:
    _HAS_MAMBA_SSM = False
    print("ℹ️ Không có mamba-ssm → Mamba thuần PyTorch (chậm hơn khi fine-tune)")

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

    def forward(self, x):                                 # x: (B, L, d_model)
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
    """1024 → d_model → [Mamba ×L] (2 chiều) → attentive-pool (có mask) → Z_DIM."""
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

    def forward(self, x, mask):                           # x:(B,L,1024) mask:(B,L) bool
        with torch.cuda.amp.autocast(enabled=False):      # SSM chạy fp32 cho ổn định
            x = x.float()
            h = self.proj(x)
            out = self._run(self.fwd, h)
            if self.bidir:
                out = out + torch.flip(self._run(self.bwd, torch.flip(h, dims=[1])), dims=[1])
            a = self.attn(out).squeeze(-1).masked_fill(~mask, float("-inf"))
            w = torch.softmax(a, dim=1).unsqueeze(-1)
            return self.out((out * w).sum(1))

# %% [markdown]
# ## 6b. Head cảm xúc + train loop (AMP + grad-accum + uncertainty weighting)

# %%
from scipy.stats import spearmanr

torch.manual_seed(SEED); np.random.seed(SEED)
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
print(f"Trunk input = {TRUNK_IN} (wavlm-branch {WAVLM_BRANCH} [{'Mamba' if USE_MAMBA else 'mean-pool'}] + aud {AUD_DIM if USE_AUDEERING else 0})")
if USE_MAMBA:
    print(f"Mamba encoder: {sum(p.numel() for p in enc.parameters())/1e6:.2f}M param")

# ── RESUME: nạp heads (+ Mamba enc) từ checkpoint ──
if RESUME and resume_ckpt is not None:
    hm, hu = heads.load_state_dict(resume_ckpt["heads"], strict=False)
    print(f"🔁 RESUME load heads từ ckpt: thiếu {len(hm)} / dư {len(hu)} key (kỳ vọng 0)")
    if USE_MAMBA and resume_ckpt.get("enc") is not None:
        em, eu = enc.load_state_dict(resume_ckpt["enc"], strict=False)
        print(f"🔁 RESUME load Mamba enc từ ckpt: thiếu {len(em)} / dư {len(eu)} key (kỳ vọng 0)")
    elif USE_MAMBA:
        print("   ⚠️ ckpt KHÔNG có 'enc' (Mamba) → Mamba head train lại từ đầu (chỉ resume backbone+heads).")

TASKS = ["emos", "cat", "val", "aro", "dom"]
log_var = nn.Parameter(torch.zeros(len(TASKS), device=device))
bb_params = [p for p in wavlm.parameters() if p.requires_grad]
head_params = list(heads.parameters()) + (list(enc.parameters()) if USE_MAMBA else []) \
    + ([log_var] if USE_UNCERTAINTY else [])
_lr_scale = RESUME_LR_SCALE if RESUME else 1.0
opt = torch.optim.AdamW([
    {"params": bb_params, "lr": LR_BACKBONE * _lr_scale},
    {"params": head_params, "lr": LR_HEAD * _lr_scale},
], weight_decay=WEIGHT_DECAY)
if RESUME and _lr_scale != 1.0:
    print(f"🔁 RESUME: LR ×{_lr_scale} → backbone {LR_BACKBONE*_lr_scale:.1e} · head {LR_HEAD*_lr_scale:.1e}")
scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP and device == "cuda")
mse = nn.MSELoss()

def soft_ce(logits, target_dist):
    return -(target_dist * F.log_softmax(logits, dim=1)).sum(1).mean()

def wavlm_branch(input_values, attn_mask):
    out = wavlm(input_values, attention_mask=attn_mask).last_hidden_state    # (B,T,D)
    if USE_MAMBA:
        return enc(out, frame_mask(out.shape[1], attn_mask))                  # (B, Z_DIM)
    return masked_mean(out, attn_mask)                                        # (B, D)

def forward_batch(b):
    fw = wavlm_branch(b["input_values"].to(device), b["attn_mask"].to(device))
    feat = torch.cat([fw, b["aud"].to(device)], dim=1) if USE_AUDEERING else fw
    return heads(feat, b["tgt"].to(device))

def pairwise_rank_loss(pred, target):
    """Hinge ranking trên MỌI cặp trong batch → tối ưu thẳng thứ hạng (≈ SRCC). Khả vi (backprop được).
    Cần ≥2 mẫu/batch mới có cặp; batch càng to càng nhiều cặp → tín hiệu càng mạnh."""
    p = pred.reshape(-1); t = target.reshape(-1)
    if p.numel() < 2:
        return torch.zeros((), device=p.device)
    sign = torch.sign(t.unsqueeze(0) - t.unsqueeze(1))      # +1 nếu câu i ĐÁNG cao hơn câu j
    diff = p.unsqueeze(0) - p.unsqueeze(1)                  # chênh lệch model dự đoán
    return torch.relu(-sign * diff).mean()                  # phạt khi xếp sai thứ tự

def compute_loss(emos_p, cat_l, vad_p, b):
    L = {"emos": mse(emos_p, b["emos"].to(device)), "cat": soft_ce(cat_l, b["cat"].to(device))}
    if HAS_VAD:
        vt = b["vad"].to(device)
        L["val"] = mse(vad_p[:, 0:1], vt[:, 0:1]); L["aro"] = mse(vad_p[:, 1:2], vt[:, 1:2]); L["dom"] = mse(vad_p[:, 2:3], vt[:, 2:3])
    else:
        vt = None
        z = torch.zeros((), device=device); L["val"] = L["aro"] = L["dom"] = z
    # Ranking loss CHỈ cho các cột chấm SRCC (emos/val/aro/dom). CAT là ERR phân bố → giữ soft-CE.
    if RANK_LAMBDA > 0:
        L["emos"] = L["emos"] + RANK_LAMBDA * pairwise_rank_loss(emos_p, b["emos"].to(device))
        if HAS_VAD:
            L["val"] = L["val"] + RANK_LAMBDA * pairwise_rank_loss(vad_p[:, 0:1], vt[:, 0:1])
            L["aro"] = L["aro"] + RANK_LAMBDA * pairwise_rank_loss(vad_p[:, 1:2], vt[:, 1:2])
            L["dom"] = L["dom"] + RANK_LAMBDA * pairwise_rank_loss(vad_p[:, 2:3], vt[:, 2:3])
    if USE_UNCERTAINTY:
        return sum(torch.exp(-log_var[i]) * L[t] + log_var[i] for i, t in enumerate(TASKS))
    return sum(L.values())

def set_mode(train):
    wavlm.train(train); heads.train(train)
    if USE_MAMBA:
        enc.train(train)

@torch.no_grad()
def evaluate():
    set_mode(False)
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
    out = {t: spearmanr(P[t], Y[t]).correlation for t in ["emos"] + (["val", "aro", "dom"] if HAS_VAD else [])}
    q = np.concatenate(catP); p = np.concatenate(catY)
    out["cat_err"] = float(np.abs(q - p).sum(1).mean())
    return out

def mean_srcc(m):
    keys = ["emos"] + (["val", "aro", "dom"] if HAS_VAD else [])
    return float(np.mean([m[k] for k in keys]))

CKPT_PATH = os.path.join(OUT_DIR, "ft_mamba_emotion_full.pt")
def save_full_ckpt(state, val_emos=float("nan")):
    torch.save({"wavlm": state["wavlm"], "heads": state["heads"], "enc": state.get("enc"),
                "USE_MAMBA": USE_MAMBA, "emos_mu": emos_mu, "emos_sd": emos_sd,
                "vad_mu": vad_mu, "vad_sd": vad_sd, "WAVLM_DIM": WAVLM_DIM, "AUD_DIM": AUD_DIM,
                "Z_DIM": Z_DIM, "UNFREEZE_TOP_LAYERS": UNFREEZE_TOP_LAYERS,
                "val_emos": float(val_emos)}, CKPT_PATH)

def snapshot():
    s = {"wavlm": {k: v.cpu().clone() for k, v in wavlm.state_dict().items()},
         "heads": {k: v.cpu().clone() for k, v in heads.state_dict().items()}}
    if USE_MAMBA:
        s["enc"] = {k: v.cpu().clone() for k, v in enc.state_dict().items()}
    return s

# RESUME: init best = điểm VAL của ckpt hiện tại → chỉ ghi đè nếu train tiếp TỐT HƠN (không sợ tụt)
if RESUME and resume_ckpt is not None:
    m0 = evaluate(); best = mean_srcc(m0); best_state = snapshot(); bad = 0
    print(f"📍 RESUME — checkpoint hiện tại: mean SRCC={best:.4f} | "
          + " ".join(f"{k}={m0[k]:.3f}" for k in ['emos', 'val', 'aro', 'dom'] if k in m0))
else:
    m0 = None
    best, best_state, bad = -1e9, None, 0
for ep in range(1, EPOCHS + 1):
    set_mode(True)
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
        best = sc; bad = 0
        best_state = snapshot()
        save_full_ckpt(best_state, m["emos"])
        print(f"   💾 lưu best → {CKPT_PATH} (epoch {ep}, mean {sc:.4f})")
    else:
        bad += 1
        if bad >= PATIENCE:
            print(f"Early stop ở epoch {ep}."); break

if best_state:
    wavlm.load_state_dict(best_state["wavlm"]); heads.load_state_dict(best_state["heads"])
    if USE_MAMBA:
        enc.load_state_dict(best_state["enc"])
final = evaluate()
if RESUME and m0 is not None:
    print(f"\n🔁 RESUME: mean SRCC ckpt {mean_srcc(m0):.4f} → sau train tiếp {mean_srcc(final):.4f} "
          + ("🚀 cải thiện → đã ghi đè ckpt" if mean_srcc(final) > mean_srcc(m0) + 1e-4 else "➖ không cải thiện (giữ best cũ)"))
print(f"\n✅ VAL (nội bộ) — exp15 (Mamba={'ON' if USE_MAMBA else 'OFF'}):")
print(f"   EMOS={final['emos']:.4f} (exp08 {EXP08['emos']})")
if HAS_VAD:
    print(f"   VAL/ARO/DOM={final['val']:.4f}/{final['aro']:.4f}/{final['dom']:.4f} "
          f"(exp08 {EXP08['val']}/{EXP08['aro']}/{EXP08['dom']})")
warn = [f"EMOS {final['emos']:.3f}<{EXP08['emos']}"] if final["emos"] < EXP08["emos"] - 0.005 else []
if HAS_VAD:
    warn += [f"{t.upper()} {final[t]:.3f}<{EXP08[t]}" for t in ["val", "aro", "dom"] if final[t] < EXP08[t] - 0.005]
print("   ⚠️ Mamba head CHƯA thắng exp08 ở:", "; ".join(warn), "(vẫn là kết quả cho paper)" if warn else "")
if not warn:
    print("   ✅ Mamba head thắng/ngang exp08 ở mọi cột → temporal modeling có ích!")
save_full_ckpt(best_state if best_state else
               {"wavlm": wavlm.state_dict(), "heads": heads.state_dict(),
                "enc": enc.state_dict() if USE_MAMBA else None}, final["emos"])
print(f"✅ Đã lưu {CKPT_PATH} (CÓ backbone + Mamba + heads). NHỚ Save Version!")

# %% [markdown]
# ## 7. Dự đoán DEV → answer.txt (5 cột cảm xúc exp15; QMOS mượn exp07/UTMOSv2)

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
    set_mode(False)
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
os.system(f"cd {OUT_DIR} && zip -j submission_track2_exp15_mamba-emotion.zip answer.txt "
          f"&& unzip -l submission_track2_exp15_mamba-emotion.zip")
print("Sẵn sàng nộp:", os.path.join(OUT_DIR, "submission_track2_exp15_mamba-emotion.zip"))

# %% [markdown]
# ## Ghi chú
# - **🔁 RESUME (train tiếp, không train lại từ đầu):** Add Input dataset chứa `ft_mamba_emotion_full.pt` của lần
#   chạy trước (hoặc để nó nằm sẵn trong `/kaggle/working` khi chạy nối phiên) → notebook tự dò & train tiếp.
#   `EPOCHS` lúc này là **số epoch train THÊM**. Val chững → đặt `RESUME_LR_SCALE=0.5`. Muốn ép train mới: `RESUME_CKPT="—"`
#   (đường dẫn không tồn tại) hoặc xóa ckpt khỏi input. ⚠️ `USE_MAMBA` phải KHỚP ckpt (code sẽ cảnh báo nếu lệch).
# - **Lần đầu** `LIMIT_TRAIN=300`, `LIMIT_DEV=20` → kiểm 1 epoch không OOM / không CheckpointError; rồi đặt `None`.
# - **Ablation chính cho paper:** chạy `USE_MAMBA=True` vs `USE_MAMBA=False` (=exp08) → so EMOS/VAL/ARO/DOM nội bộ
#   → trả lời "Mamba temporal head có hơn mean-pooling không?".
# - **OOM / quá chậm trên T4 (nhất là khi dùng Mamba thuần PyTorch):** giảm theo thứ tự
#   `MAX_SECONDS` (6→5) → `MAMBA_LAYERS` (2→1) → `UNFREEZE_TOP_LAYERS` (6→4) → `BATCH` (2→1, tăng `ACCUM`).
#   Hoặc thử cài `mamba-ssm causal-conv1d` (nhanh + nhẹ RAM hơn nhiều) — code tự dùng nếu import được.
# - **Ranking loss (`RANK_LAMBDA`):** thêm pairwise ranking cho 4 cột SRCC (emos/val/aro/dom) → khớp metric
#   UTT-SRCC hơn MSE. ⚠️ **Điểm yếu:** ranking tính trên các cặp TRONG 1 mini-batch; `BATCH=2` → mỗi forward
#   chỉ có 1 cặp → tín hiệu YẾU. Muốn ranking mạnh: tăng `BATCH` (4→8 nếu VRAM chịu được). Ở các exp head
#   ĐÓNG BĂNG (exp06/07, BATCH=64) ranking mạnh hơn nhiều. A/B `RANK_LAMBDA=0` vs `0.3` → bảng ablation cho paper.
# - **QMOS:** Add Input answer.txt exp07 vào `/kaggle/input/exp07-answer/answer.txt` để mượn QMOS 0.548;
#   không có thì tự chấm UTMOSv2 (cần Internet On).
# - Ghi config → kết quả → nhận xét vào `docs/04_experiments_log.md` (mục exp15).
