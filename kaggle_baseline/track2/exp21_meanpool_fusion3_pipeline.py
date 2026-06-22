# %% [markdown]
# # VMC2026 Track 2 — exp21 (GỘP 3 ENCODER bằng MEAN-POOL + CONCAT) — Kaggle
#
# **Ý tưởng:** đơn giản hoá exp18/exp20. Mỗi encoder ĐÓNG BĂNG → **mean-pool theo thời gian → 1 vector/wav**,
# rồi **CONCAT cả 3** → trunk MLP chung → 3 head (EMOS/CAT/VAD). KHÔNG cross-attention, KHÔNG frame-level.
#
# ```
#   wav ┬─ WavLM (SAILER) ─ mean-pool ─► e_w  (1024)
#       ├─ audeering ────── mean-pool ─► e_a  (1024)  + VAD3 (3, thang 1–5)
#       └─ Qwen2-Audio ──── pooled ────► e_l  (LLM_DIM ~3584)
#                                         │ concat [e_w | e_a | VAD3 | e_l]
#                                         ▼
#                                  trunk MLP → EMOS / CAT / VAD
# ```
#
# **Vì sao 3 encoder bù nhau:** WavLM = âm học cấp thấp · audeering = VAD chuyên · Qwen2-Audio (audio-LLM) =
# ngữ nghĩa/biểu cảm cấp cao. 3 góc nhìn khác nhau → ensemble đa encoder (kiểu URGENT-MOS của Track 1).
#
# **Khác exp18/exp20:** exp18 = cross-attention WavLM⟷audeering; exp20 = cross-attn + LLM concat.
# exp21 = **mean-pool concat cả 3** (nhẹ nhất, cache chỉ 1 vector/encoder/wav). Đây là một **ablation đẹp**:
# "fusion đơn giản (mean-pool concat) vs phức tạp (cross-attention)".
#
# **Phạm vi:** chỉ 5 cột cảm xúc (EMOS/CAT/VAL/ARO/DOM). QMOS trộn cột từ exp13 ở build_answer.
#
# ## ⚠️ Phải biết trước
# - **Qwen2-Audio-7B fp16 KHÔNG vừa T4 16GB** → nạp **4-bit (bitsandbytes)** ~5GB. Chỉ inference+cache nên đủ.
# - **Lần đầu** `LIMIT_TRAIN=300`, `LIMIT_DEV=20` (forward 7B chậm → smoke test trước).
# - Cờ ablation: `USE_AUDIOLLM`, `USE_AUDEERING`, `USE_VAD3` → bật/tắt từng encoder để đo đóng góp.
# - **Lưới an toàn:** chỉ trộn/thay cột cảm xúc nếu VAL nội bộ **vượt mốc exp08**.
# - **Lưu checkpoint mỗi best + Save Version NGAY** (bài học exp08).
#
# **Cách chạy Kaggle:** GPU **T4** + Internet **On** → Add Input (1) dataset Track 2, (2) tùy chọn `answer.txt`
# exp13 (trộn cột QMOS) → sửa slug cell 0 → Run All (LIMIT nhỏ trước).

# %% [markdown]
# ## 0. Cấu hình — SỬA Ở ĐÂY

# %%
import os, glob

def find_data_root(search_root="/kaggle/input"):
    cands = []
    for train_csv in glob.glob(os.path.join(search_root, "**", "sets", "train.csv"), recursive=True):
        root = os.path.dirname(os.path.dirname(train_csv))
        score = os.path.isdir(os.path.join(root, "wav")) + os.path.exists(os.path.join(root, "metadata.csv"))
        cands.append((score, root))
    cands.sort(reverse=True)
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
METADATA_CSV = f"{DATA_ROOT}/metadata.csv"     # wavID|emotion|transcript
TRAIN_CSV    = f"{DATA_ROOT}/sets/train.csv"   # lisID|wavID|qMOS|emoCat|eMOS|val|dom|aro
DEV_SCP      = f"{DATA_ROOT}/sets/dev.scp"

OUT_DIR   = "/kaggle/working"
CACHE_DIR = "/kaggle/working/pool3_cache"      # persist qua Save Version (KHÁC /kaggle/temp)
POOL_DIR_W = f"{CACHE_DIR}/wavlm_pool"         # .npy fp16 POOLED WavLM (1 vector/wav)
POOL_DIR_A = f"{CACHE_DIR}/aud_pool"           # .npy fp16 POOLED audeering (1 vector/wav)
LLM_EMB_DIR = f"{CACHE_DIR}/qwen_emb"          # .npy fp16 POOLED Qwen2-Audio (1 vector/wav)
for d in (CACHE_DIR, POOL_DIR_W, POOL_DIR_A, LLM_EMB_DIR):
    os.makedirs(d, exist_ok=True)

# ── (tùy chọn) cột QMOS để TRỘN — answer.txt exp13 (Add Input). "" = dùng fallback 3.0 ──
QMOS_ANSWER = "/kaggle/input/exp13-answer/answer.txt"   # << sửa slug; có cột wav,QMOS

# ── 3 encoder (đóng băng) — bật/tắt để ablation ──
USE_AUDEERING   = True     # encoder 2 (VAD chuyên)
USE_VAD3        = True      # ghép thêm VAD3 (thang 1–5) của audeering vào concat
USE_AUDIOLLM    = True      # encoder 3 (audio-LLM); False → chỉ WavLM(+audeering) mean-pool concat

# ── Audio-LLM (Qwen2-Audio) ──
QWEN_MODEL      = "Qwen/Qwen2-Audio-7B-Instruct"
AUDIOLLM_4BIT   = True                                   # BẮT BUỘC True trên T4
AUDIOLLM_PROMPT = "Describe the emotion and speaking style of this speech."
LLM_DIM         = None                                   # tự dò từ cache

# ── Tiền xử lý audio ──
MAX_SECONDS  = 12          # cắt audio trước khi qua backbone
SR           = 16000

# ── Train (frozen → nhẹ) ──
TRUNK_HIDDEN = 512
HEAD_HIDDEN  = 128
DROPOUT      = 0.3
LR           = 1e-3
WEIGHT_DECAY = 1e-5
BATCH        = 64
EPOCHS       = 80
PATIENCE     = 15
VAL_FRAC     = 0.10
SEED         = 42
USE_UNCERTAINTY = True
LAMBDA_RANK = 0.3          # ranking loss cho EMOS/VAL/ARO/DOM (train theo SRCC); CAT giữ soft-CE

LIMIT_TRAIN = 300          # << LẦN ĐẦU 300; chạy thật None
LIMIT_DEV   = 20           # << LẦN ĐẦU 20; chạy thật None

EXP08 = {"emos": 0.8116, "val": 0.6605, "aro": 0.7904, "dom": 0.7539}

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
_enc = ["WavLM(SAILER)"] + (["audeering"] if USE_AUDEERING else []) + (["Qwen2-Audio"] if USE_AUDIOLLM else [])
print(f"GỘP {len(_enc)} encoder (mean-pool concat): {' + '.join(_enc)}"
      + (" + VAD3" if (USE_VAD3 and USE_AUDEERING) else ""))
print(f"Ranking loss: λ={LAMBDA_RANK}" + (" = MSE thuần" if LAMBDA_RANK == 0 else " (MSE + pairwise ranking)"))

# %% [markdown]
# ## 1. Cài đặt + clone SAILER + bitsandbytes (cho audio-LLM 4-bit)

# %%
import sys, subprocess

def pip_install(*pkgs):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], check=True)

_pkgs = ["librosa", "soundfile", "scipy", "scikit-learn", "pandas", "tqdm", "safetensors",
         "loralib", "speechbrain"]
if USE_AUDIOLLM:
    _pkgs += ["bitsandbytes", "accelerate"]
pip_install(*_pkgs)

REPO_DIR = "/kaggle/working/vox-profile-release"
if not os.path.exists(REPO_DIR):
    subprocess.run(["git", "clone", "--depth", "1",
                    "https://github.com/tiantiaf0627/vox-profile-release.git", REPO_DIR], check=True)
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

# %% [markdown]
# ## 2. Nhãn vàng cảm xúc (gộp trung bình theo wav)

# %%
import numpy as np
import pandas as pd

def load_target_emotions():
    tgt = {}
    with open(METADATA_CSV, encoding="utf-8") as f:
        for ln in f:
            parts = ln.strip().split("|")
            if len(parts) >= 2:
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

# %% [markdown]
# ## 3. Trích + cache POOLED vector mỗi encoder (đóng băng → mean-pool 1 vector/wav)

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
from tqdm.auto import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device, ("✅ " + torch.cuda.get_device_name(0)) if device == "cuda" else "⚠️ CPU (rất chậm!)")
torch.manual_seed(SEED); np.random.seed(SEED)

def load_wave(sid):
    p = os.path.join(WAV_DIR, sid + ".wav")
    if not os.path.exists(p):
        return None
    wave, _ = librosa.load(p, sr=SR, mono=True)
    return wave[: MAX_SECONDS * SR].astype(np.float32)

# ── WavLM (SAILER) frozen → mean-pool ──────────────────────────────────
def find_hf_backbone(module):
    cands = []
    for nm, m in module.named_modules():
        enc = getattr(m, "encoder", None)
        if getattr(m, "feature_extractor", None) is not None and enc is not None \
                and getattr(enc, "layers", None) is not None:
            cands.append((nm, m))
    if not cands:
        return None, None
    cands.sort(key=lambda x: sum(p.numel() for p in x[1].parameters()), reverse=True)
    return cands[0]

_wavlm = None
def _get_wavlm():
    global _wavlm
    if _wavlm is None:
        bb = None
        try:
            from src.model.emotion.wavlm_emotion import WavLMWrapper
            _name, bb = find_hf_backbone(WavLMWrapper.from_pretrained("tiantiaf/wavlm-large-categorical-emotion"))
            if bb is not None:
                print(f"✅ Backbone WavLM từ SAILER wrapper tại '.{_name}'")
        except Exception as e:
            print("⚠️ Lỗi nạp SAILER wrapper:", repr(e), "→ fallback WavLM trắng.")
        if bb is None:
            from transformers import WavLMModel
            bb = WavLMModel.from_pretrained("microsoft/wavlm-large")
            print("ℹ️ Fallback: microsoft/wavlm-large.")
        bb = bb.to(device).eval()
        for p in bb.parameters():
            p.requires_grad = False
        _wavlm = bb
    return _wavlm

WAVLM_DIM = 1024
emb_w = {}   # stem -> pooled WavLM (np.float16)

def extract_wavlm_pool(stems, tag):
    for s in stems:
        if s not in emb_w:
            p = os.path.join(POOL_DIR_W, s + ".npy")
            if os.path.exists(p):
                emb_w[s] = np.load(p)
    todo = [s for s in stems if s not in emb_w]
    if todo:
        wavlm = _get_wavlm()
        with torch.no_grad():
            for s in tqdm(todo, desc=f"wavlm-pool {tag}"):
                wave = load_wave(s)
                if wave is None:
                    continue
                iv = torch.from_numpy(wave).unsqueeze(0).to(device)
                hs = wavlm(iv).last_hidden_state[0]        # (T1, 1024)
                e = hs.mean(0).float().cpu().numpy().astype(np.float16)   # mean-pool thời gian
                np.save(os.path.join(POOL_DIR_W, s + ".npy"), e)
                emb_w[s] = e
        torch.cuda.empty_cache() if device == "cuda" else None
    return {s for s in stems if s in emb_w}

# ── audeering frozen → mean-pool + VAD3 ────────────────────────────────
aud_backbone = aud_head = aud_proc = None
AUD_DIM = 1024
def _get_audeering():
    global aud_backbone, aud_head, aud_proc, AUD_DIM
    if aud_backbone is not None:
        return
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
    AUD_DIM = _hid
    print(f"✅ audeering frozen (pool-dim {AUD_DIM}) + VAD3 head")

emb_a = {}     # stem -> pooled audeering (np.float16)
aud_vad = {}   # stem -> [VAL,ARO,DOM] thang 1–5

def extract_aud_pool(stems, tag):
    if not USE_AUDEERING:
        return set(stems)
    vad_path = os.path.join(CACHE_DIR, f"aud_vad_{tag}.npz")
    if os.path.exists(vad_path):
        z = np.load(vad_path, allow_pickle=True)
        for k in z.files:
            aud_vad[k] = z[k]
    for s in stems:
        if s not in emb_a:
            p = os.path.join(POOL_DIR_A, s + ".npy")
            if os.path.exists(p):
                emb_a[s] = np.load(p)
    todo = [s for s in stems if s not in emb_a or s not in aud_vad]
    if todo:
        _get_audeering()
        with torch.no_grad():
            for i, s in enumerate(tqdm(todo, desc=f"aud-pool {tag}")):
                wave = load_wave(s)
                if wave is None:
                    continue
                x = aud_proc(wave, sampling_rate=SR).input_values[0]
                x = torch.from_numpy(np.asarray(x, dtype=np.float32)).unsqueeze(0).to(device)
                h = aud_backbone(x)[0]                     # (1, T2, hid)
                pooled = h.mean(dim=1)                     # (1, hid)
                emb_a[s] = pooled[0].float().cpu().numpy().astype(np.float16)
                np.save(os.path.join(POOL_DIR_A, s + ".npy"), emb_a[s])
                out = aud_head(pooled)[0].cpu().numpy()    # [aro,dom,val] ∈ [0,1]
                aud_vad[s] = np.array([1 + 4 * out[2], 1 + 4 * out[0], 1 + 4 * out[1]], dtype=np.float32)
                if (i + 1) % 500 == 0:
                    np.savez(vad_path, **aud_vad)
        np.savez(vad_path, **aud_vad)
        torch.cuda.empty_cache() if device == "cuda" else None
    return {s for s in stems if s in emb_a}

# ── Qwen2-Audio (audio-LLM) frozen, 4-bit → pooled embedding ───────────
_qwen = _qwen_proc = None
def _get_qwen():
    global _qwen, _qwen_proc
    if _qwen is not None:
        return
    from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
    kw = dict(torch_dtype=torch.float16)
    if AUDIOLLM_4BIT:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
        kw["device_map"] = {"": 0}
    _qwen_proc = AutoProcessor.from_pretrained(QWEN_MODEL)
    _qwen = Qwen2AudioForConditionalGeneration.from_pretrained(QWEN_MODEL, **kw).eval()
    if not AUDIOLLM_4BIT:
        _qwen = _qwen.to(device)
    print(f"✅ Qwen2-Audio nạp xong ({'4-bit' if AUDIOLLM_4BIT else 'fp16'}).")

def _audio_token_mask(ids):
    for attr in ("audio_token_index", "audio_token_id"):
        tok = getattr(_qwen.config, attr, None)
        if tok is not None:
            m = (ids == tok)
            if bool(m.any()):
                return m
    return None

@torch.no_grad()
def _qwen_embed(wave):
    conv = [{"role": "user", "content": [
        {"type": "audio", "audio_url": "x"},
        {"type": "text", "text": AUDIOLLM_PROMPT}]}]
    text = _qwen_proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
    inputs = _qwen_proc(text=text, audios=[wave], sampling_rate=SR, return_tensors="pt")
    inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}
    out = _qwen(**inputs, output_hidden_states=True, return_dict=True)
    last = out.hidden_states[-1][0]            # (seq, hidden)
    m = _audio_token_mask(inputs["input_ids"][0])
    emb = last[m].mean(0) if m is not None else last.mean(0)
    return emb.float().cpu().numpy().astype(np.float16)

emb_l = {}   # stem -> pooled Qwen2-Audio (np.float16)
def extract_audiollm_emb(stems, tag):
    if not USE_AUDIOLLM:
        return set(stems)
    for s in stems:
        if s not in emb_l:
            p = os.path.join(LLM_EMB_DIR, s + ".npy")
            if os.path.exists(p):
                emb_l[s] = np.load(p)
    todo = [s for s in stems if s not in emb_l]
    if todo:
        _get_qwen()
        for s in tqdm(todo, desc=f"qwen-emb {tag}"):
            wave = load_wave(s)
            if wave is None:
                continue
            try:
                e = _qwen_embed(wave)
            except Exception as ex:
                print("⚠️ qwen embed fail", s, repr(ex)); continue
            np.save(os.path.join(LLM_EMB_DIR, s + ".npy"), e)
            emb_l[s] = e
        torch.cuda.empty_cache() if device == "cuda" else None
    return {s for s in stems if s in emb_l}

# %% [markdown]
# ## 4. Concat 3 vector → trunk MLP → 3 head; train multi-task (uncertainty + ranking)

# %%
class EmoHeads(nn.Module):
    """concat feature → trunk MLP chung → EMOS (ghép one-hot target) / CAT / VAD."""
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

# ── Trích cache cho TRAIN (3 encoder) ──
train_stems = list(train_df["wavID"])
if LIMIT_TRAIN:
    train_stems = train_stems[:LIMIT_TRAIN]
have_w = extract_wavlm_pool(train_stems, "train")
have_a = extract_aud_pool(train_stems, "train") if USE_AUDEERING else set(train_stems)
have_l = extract_audiollm_emb(train_stems, "train") if USE_AUDIOLLM else set(train_stems)

if USE_AUDIOLLM and emb_l:
    LLM_DIM = int(np.asarray(next(iter(emb_l.values()))).shape[-1])
    print(f"🆕 LLM_DIM tự dò = {LLM_DIM} (Qwen2-Audio hidden)")

def onehot_target(tgt):
    v = np.zeros(len(EMOTIONS5), dtype=np.float32)
    if tgt in EMOTIONS5:
        v[EMOTIONS5.index(tgt)] = 1.0
    return v

# ── Lọc wav đủ MỌI nguồn bật + dựng nhãn ──
N_EMO = len(EMOTIONS5)
lab = train_df.set_index("wavID")
items = []
for s in train_stems:
    tgt = target_map.get(s)
    if s not in have_w or (USE_AUDEERING and s not in have_a) \
            or (USE_AUDIOLLM and s not in have_l) or tgt is None or s not in lab.index:
        continue
    items.append((s, onehot_target(tgt), float(lab.loc[s, "emos"]),
                  [lab.loc[s, "val"], lab.loc[s, "aro"], lab.loc[s, "dom"]],
                  [lab.loc[s, f"cat{i}"] for i in range(N_EMO)]))
print(f"Train items đủ nguồn: {len(items)}")
assert len(items) >= 10, "Quá ít mẫu — kiểm tra cache/nhãn (LIMIT quá nhỏ?)."

y_emos = np.array([it[2] for it in items], dtype=np.float32)
y_vad  = np.array([it[3] for it in items], dtype=np.float32)
y_cat  = np.array([it[4] for it in items], dtype=np.float32)
T_oh   = np.array([it[1] for it in items], dtype=np.float32)

emos_mu, emos_sd = float(y_emos.mean()), float(y_emos.std() + 1e-6)
if HAS_VAD:
    vad_mu = np.nanmean(y_vad, axis=0); vad_sd = np.nanstd(y_vad, axis=0) + 1e-6
else:
    vad_mu = np.zeros(3, dtype=np.float32); vad_sd = np.ones(3, dtype=np.float32)
y_emos_z = (y_emos - emos_mu) / emos_sd
y_vad_z  = (y_vad - vad_mu) / vad_sd if HAS_VAD else np.zeros_like(y_vad)

from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr
idx_all = np.arange(len(items))
tr_idx, va_idx = train_test_split(idx_all, test_size=VAL_FRAC, random_state=SEED)

# Chiều concat = WavLM + audeering + VAD3 + LLM (theo cờ bật)
TRUNK_IN = WAVLM_DIM \
    + (AUD_DIM if USE_AUDEERING else 0) \
    + (3 if (USE_VAD3 and USE_AUDEERING) else 0) \
    + (LLM_DIM if USE_AUDIOLLM else 0)
print(f"Concat feature = {TRUNK_IN}-D "
      f"(WavLM {WAVLM_DIM}"
      + (f" + audeering {AUD_DIM}" if USE_AUDEERING else "")
      + (f" + VAD3 3" if (USE_VAD3 and USE_AUDEERING) else "")
      + (f" + LLM {LLM_DIM}" if USE_AUDIOLLM else "") + ")")

# ── ⚠️ CHUẨN HOÁ per-dim từng nhánh (thống kê TRAIN) ──────────────────
# LLM hidden (4096-D, lại 4-bit) có vài chiều "outlier" giá trị cực lớn → concat thô sẽ LẤN ÁT
# WavLM/audeering (1024-D) làm trunk chết (VAD hằng → nan, EMOS sập 0.22). Đưa MỖI chiều về mean0/std1.
def _branch_stats(emb_dict):
    X = np.stack([emb_dict[items[j][0]].astype(np.float32) for j in tr_idx])
    return X.mean(0), X.std(0) + 1e-5
W_MU, W_SD = _branch_stats(emb_w)
A_MU = A_SD = L_MU = L_SD = None
if USE_AUDEERING:
    A_MU, A_SD = _branch_stats(emb_a)
if USE_AUDIOLLM:
    L_MU, L_SD = _branch_stats(emb_l)
print("✅ Chuẩn hoá per-dim từng nhánh (mean0/std1 theo TRAIN) — chống LLM outlier lấn.")

heads = EmoHeads(TRUNK_IN, TRUNK_HIDDEN, HEAD_HIDDEN, DROPOUT, N_EMO).to(device)
print(f"Tham số train (chỉ heads, backbone đóng băng): {sum(p.numel() for p in heads.parameters())/1e6:.2f}M")

TASKS = ["emos", "cat", "val", "aro", "dom"]
log_var = nn.Parameter(torch.zeros(len(TASKS), device=device))
params = list(heads.parameters()) + ([log_var] if USE_UNCERTAINTY else [])
opt = torch.optim.Adam(params, lr=LR, weight_decay=WEIGHT_DECAY)
mse = nn.MSELoss(reduction="none")

def soft_ce(logits, target_dist):
    return -(target_dist * F.log_softmax(logits, dim=1)).sum(dim=1)

def pairwise_rank_loss(p, y):
    dp = p.unsqueeze(1) - p.unsqueeze(0)
    dy = y.unsqueeze(1) - y.unsqueeze(0)
    mask = dy > 0
    if mask.sum() == 0:
        return p.new_zeros(())
    return F.softplus(-dp[mask]).mean()

def rank_term(p, y):
    return LAMBDA_RANK * pairwise_rank_loss(p, y) if LAMBDA_RANK > 0 else p.new_zeros(())

def feat_batch(sids):
    """list stem → concat [WavLM | audeering | VAD3 | LLM] (theo cờ, đã chuẩn hoá) → tensor (B, TRUNK_IN).
    VAD3 (thang 1–5, 3-D) GIỮ NGUYÊN — đặc trưng có nghĩa vật lý, scale nhỏ, không cần chuẩn hoá."""
    cols = [(np.stack([emb_w[s].astype(np.float32) for s in sids]) - W_MU) / W_SD]
    if USE_AUDEERING:
        cols.append((np.stack([emb_a[s].astype(np.float32) for s in sids]) - A_MU) / A_SD)
        if USE_VAD3:
            cols.append(np.stack([aud_vad[s].astype(np.float32) for s in sids]))
    if USE_AUDIOLLM:
        cols.append((np.stack([emb_l[s].astype(np.float32) for s in sids]) - L_MU) / L_SD)
    return torch.from_numpy(np.concatenate(cols, axis=1).astype(np.float32)).to(device)

@torch.no_grad()
def eval_val():
    heads.eval()
    preds_e, preds_v, preds_c = [], [], []
    for i in range(0, len(va_idx), BATCH):
        bi = va_idx[i:i + BATCH]
        sids = [items[j][0] for j in bi]
        feat = feat_batch(sids)
        tgt = torch.from_numpy(T_oh[bi]).to(device)
        ep, cl, vp = heads(feat, tgt)
        preds_e.append(ep.cpu().numpy().ravel())
        preds_v.append(vp.cpu().numpy())
        preds_c.append(F.softmax(cl, dim=1).cpu().numpy())
    ep = np.concatenate(preds_e); vp = np.concatenate(preds_v); q = np.concatenate(preds_c)
    out = {"emos": spearmanr(ep, y_emos[va_idx]).correlation}
    if HAS_VAD:
        for j, t in enumerate(["val", "aro", "dom"]):
            out[t] = spearmanr(vp[:, j], y_vad[va_idx, j]).correlation
    p = y_cat[va_idx]
    out["cat_negkl"] = float(-(p * (np.log(p + 1e-9) - np.log(q + 1e-9))).sum(1).mean())
    return out

def val_score(m):
    keys = ["emos"] + (["val", "aro", "dom"] if HAS_VAD else [])
    vals = [m[k] for k in keys if m[k] == m[k]]   # bỏ nan (head hằng) → vẫn lưu được best
    return float(np.mean(vals)) if vals else -1e9

emos_t = torch.from_numpy(y_emos_z).float().unsqueeze(1).to(device)
vad_t  = torch.from_numpy(y_vad_z).float().to(device)
cat_t  = torch.from_numpy(y_cat).float().to(device)

CKPT_PATH = os.path.join(OUT_DIR, "exp21_pool3.pt")
def save_ckpt(val_emos):
    torch.save({"heads": {k: v.cpu() for k, v in heads.state_dict().items()},
                "emos_mu": emos_mu, "emos_sd": emos_sd, "vad_mu": vad_mu, "vad_sd": vad_sd,
                "W_MU": W_MU, "W_SD": W_SD, "A_MU": A_MU, "A_SD": A_SD, "L_MU": L_MU, "L_SD": L_SD,
                "TRUNK_IN": TRUNK_IN, "WAVLM_DIM": WAVLM_DIM, "AUD_DIM": AUD_DIM, "LLM_DIM": LLM_DIM,
                "USE_AUDEERING": USE_AUDEERING, "USE_VAD3": USE_VAD3, "USE_AUDIOLLM": USE_AUDIOLLM,
                "QWEN_MODEL": QWEN_MODEL, "AUDIOLLM_PROMPT": AUDIOLLM_PROMPT, "LAMBDA_RANK": LAMBDA_RANK,
                "TRUNK_HIDDEN": TRUNK_HIDDEN, "HEAD_HIDDEN": HEAD_HIDDEN, "DROPOUT": DROPOUT,
                "EMOTIONS5": EMOTIONS5, "val_emos": float(val_emos)}, CKPT_PATH)

best_score, best_blob, bad = -1e9, None, 0
for ep in range(1, EPOCHS + 1):
    heads.train()
    perm = np.array(tr_idx); np.random.shuffle(perm)
    run = 0.0
    for i in range(0, len(perm), BATCH):
        bi = perm[i:i + BATCH]
        sids = [items[j][0] for j in bi]
        opt.zero_grad()
        feat = feat_batch(sids)
        tgt = torch.from_numpy(T_oh[bi]).to(device)
        ep_, cl, vp = heads(feat, tgt)
        bt = torch.from_numpy(bi).to(device)
        L = {"emos": mse(ep_, emos_t[bt]).mean() + rank_term(ep_.squeeze(1), emos_t[bt].squeeze(1)),
             "cat":  soft_ce(cl, cat_t[bt]).mean()}
        if HAS_VAD:
            for j, t in ((0, "val"), (1, "aro"), (2, "dom")):
                L[t] = mse(vp[:, j:j+1], vad_t[bt, j:j+1]).mean() + rank_term(vp[:, j], vad_t[bt, j])
        else:
            z0 = torch.zeros((), device=device); L["val"] = L["aro"] = L["dom"] = z0
        if USE_UNCERTAINTY:
            loss = sum(torch.exp(-log_var[k]) * L[t] + log_var[k] for k, t in enumerate(TASKS))
        else:
            loss = sum(L[t] for t in TASKS)
        loss.backward(); opt.step()
        run += float(loss.item()) * len(bi)
    m = eval_val(); sc = val_score(m)
    if sc > best_score:
        best_score = sc
        best_blob = {k: v.cpu().clone() for k, v in heads.state_dict().items()}
        save_ckpt(m["emos"]); bad = 0
    else:
        bad += 1
    if ep % 5 == 0 or ep == 1:
        msg = " ".join(f"{k}={m[k]:.3f}" for k in m)
        print(f"epoch {ep:3d} | loss {run/len(perm):.4f} | {msg} | best {best_score:.4f}")
    if bad >= PATIENCE:
        print(f"Early stop ở epoch {ep}."); break

if best_blob is not None:
    heads.load_state_dict(best_blob)
final = eval_val()
print("\n✅ VAL (nội bộ) tốt nhất:")
print(f"   EMOS SRCC = {final['emos']:.4f}   (mốc exp08 = {EXP08['emos']})")
if HAS_VAD:
    print(f"   VAL/ARO/DOM = {final['val']:.4f} / {final['aro']:.4f} / {final['dom']:.4f}"
          f"   (mốc exp08 = {EXP08['val']} / {EXP08['aro']} / {EXP08['dom']})")
    for key in ["val", "aro", "dom", "emos"]:
        if final[key] < EXP08[key] - 0.005:
            print(f"   ⚠️ {key.upper()} {final[key]:.4f} THUA exp08 {EXP08[key]} → giữ exp08 cho cột này.")
if USE_UNCERTAINTY:
    print("   log σ² mỗi task:", {t: round(float(log_var[i]), 3) for i, t in enumerate(TASKS)})
print(f"💾 checkpoint best → {CKPT_PATH}  (⚠️ Save Version NGAY!)")

# %% [markdown]
# ## 5. Dự đoán DEV (đảo z-score)

# %%
def list_dev():
    with open(DEV_SCP) as f:
        return [ln.strip() for ln in f if ln.strip()]

dev_names = list_dev()
if LIMIT_DEV:
    dev_names = dev_names[:LIMIT_DEV]
dev_stems = [stem(n) for n in dev_names]
print("DEV:", len(dev_names), "mẫu")

dev_w = extract_wavlm_pool(dev_stems, "dev")
dev_a = extract_aud_pool(dev_stems, "dev") if USE_AUDEERING else set(dev_stems)
dev_l = extract_audiollm_emb(dev_stems, "dev") if USE_AUDIOLLM else set(dev_stems)

@torch.no_grad()
def predict_emotion(sid):
    if sid not in dev_w or (USE_AUDEERING and sid not in dev_a) or (USE_AUDIOLLM and sid not in dev_l):
        return None
    heads.eval()
    feat = feat_batch([sid])
    tgt = torch.from_numpy(onehot_target(target_map.get(sid))[None, :]).to(device)
    ep, cl, vp = heads(feat, tgt)
    emos = float(np.clip(ep.item() * emos_sd + emos_mu, 1.0, 5.0))
    cat5 = F.softmax(cl, dim=1)[0].cpu().numpy()
    vad3 = np.clip(vp[0].cpu().numpy() * vad_sd + vad_mu, 1.0, 5.0)
    return emos, cat5, vad3

# %% [markdown]
# ## 6. Ghép answer.txt 6 cột (QMOS trộn từ exp13 / fallback)

# %%
def load_qmos_answer(path):
    if not path or not os.path.exists(path):
        print("ℹ️ Không có QMOS_ANSWER → QMOS fallback 3.0 (nhớ trộn cột exp13 khi nộp thật).")
        return {}
    import csv
    out = {}
    with open(path) as f:
        r = csv.reader(f); header = next(r)
        qi = header.index("QMOS") if "QMOS" in header else 1
        for row in r:
            if row:
                out[row[0]] = float(row[qi]); out[stem(row[0])] = float(row[qi])
    print(f"✅ Nạp QMOS từ {path}: {len(out)} khóa")
    return out

qmos_map = load_qmos_answer(QMOS_ANSWER)

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
# ## 7. Validate + zip

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
os.system(f"cd {OUT_DIR} && zip -j submission_track2_exp21_pool3.zip answer.txt "
          f"&& unzip -l submission_track2_exp21_pool3.zip")
print("Sẵn sàng nộp:", os.path.join(OUT_DIR, "submission_track2_exp21_pool3.zip"))

# %% [markdown]
# ## Ghi chú
# - **Lần đầu** `LIMIT_TRAIN=300`, `LIMIT_DEV=20` (forward 7B chậm → smoke test trước); rồi `None` chạy thật.
# - **OOM nạp Qwen?** Đảm bảo `AUDIOLLM_4BIT=True`. Backbone không cần khi train (chỉ dùng cache pooled).
# - **Ablation chính (cho paper):**
#   (1) số encoder: WavLM | +audeering | +audeering+LLM → đo từng encoder thêm bao nhiêu mỗi cột.
#   (2) **fusion: mean-pool concat (exp21) vs cross-attention (exp18/exp20)** — cùng backbone, khác cách gộp.
#   (3) `AUDIOLLM_PROMPT` (prompt cảm xúc vs trung tính); `LAMBDA_RANK ∈ {0, 0.3, 1.0}`.
# - **Lưới an toàn:** chỉ trộn/thay cột cảm xúc nếu VAL nội bộ **vượt mốc exp08**; thua → giữ exp08 (ablation âm).
# - **Cân chiều:** LLM (~3584-D) lớn hơn WavLM/audeering (1024). Trunk MLP tự học trọng số, nhưng nếu nghi LLM
#   "lấn", có thể chuẩn hoá từng nhánh (L2-normalize) hoặc chiếu LLM về 256-D trước concat (thêm 1 Linear) — để ablation.
# - **Checkpoint** `exp21_pool3.pt` đủ predict-only (config + heads); KHÔNG lưu backbone. **Cache pooled**
#   (`wavlm_pool/`, `aud_pool/`, `qwen_emb/`) nên lưu ra Kaggle Dataset (trích 7B tốn thời gian). **Save Version NGAY**.
# - License: SAILER (Open RAIL) · audeering (CC BY-NC-SA) · Qwen2-Audio (Apache-2.0) — khai báo `docs/12_`.
# - Ghi config → kết quả → nhận xét vào `docs/04_experiments_log.md` (mục exp21).
