# %% [markdown]
# # VMC2026 Track 2 — exp20 (THÊM AUDIO-LLM Qwen2-Audio làm LUỒNG THỨ 3, pooled-concat) — Kaggle
#
# **Ý tưởng:** exp18 gộp **2 backbone đóng băng** WavLM(SAILER) ⟷ audeering bằng cross-attention.
# exp20 **thêm 1 "đôi tai" thứ 3 = audio-LLM Qwen2-Audio**: cho audio qua Qwen2-Audio (đóng băng),
# lấy **1 vector embedding/wav** (mean-pool hidden-state của LLM tại vị trí token audio), **cache 1 lần**,
# rồi **CONCAT** vào trunk input cạnh `z` (cross-attn) + VAD3. Đây là kiểu "multi-encoder fusion" của
# URGENT-MOS (Track 1) nhưng áp cho cảm xúc Track 2.
#
# **Vì sao audio-LLM bổ trợ:** WavLM bắt âm học cấp thấp, audeering bắt VAD; audio-LLM (7B) "hiểu" ngữ
# nghĩa/độ tự nhiên/biểu cảm ở tầng cao → 3 góc nhìn khác nhau, bù lỗi cho nhau (ensemble đa encoder).
#
# **Pooled-concat (mức nhẹ, làm TRƯỚC):** chỉ thêm 1 vector/wav → sửa ít, cache nhẹ (1 vector fp16/wav),
# train nhanh. Nếu có tín hiệu tốt mới nâng lên cross-attention stream thứ 3 (mức nặng) ở exp sau.
#
# **Phạm vi:** chỉ 5 cột cảm xúc (EMOS/CAT/VAL/ARO/DOM). QMOS trộn cột từ exp13 ở build_answer.
#
# ## ⚠️ Phải biết trước
# - **Qwen2-Audio-7B fp16 KHÔNG vừa T4 16GB** → nạp **4-bit (bitsandbytes)** ~5GB. Chỉ inference+cache nên 4-bit đủ.
# - **Lần đầu BẮT BUỘC** `LIMIT_TRAIN=300`, `LIMIT_DEV=20` (riêng forward 7B rất chậm → smoke test trước).
# - `USE_AUDIOLLM=False` → tái lập **đúng exp18** (lưới ablation: có/không audio-LLM).
# - **Lưới an toàn:** chỉ trộn/thay cột cảm xúc nếu VAL nội bộ **vượt mốc exp08**.
# - **Lưu checkpoint mỗi best + Save Version NGAY** (bài học exp08: kernel chết là mất).
#
# **Cách chạy Kaggle:** GPU **T4** + Internet **On** → Add Input (1) dataset Track 2, (2) tùy chọn `answer.txt`
# exp13 (trộn cột QMOS) → sửa slug cell 0 → Run All (LIMIT nhỏ trước).

# %% [markdown]
# ## 0. Cấu hình — SỬA Ở ĐÂY

# %%
import os, glob

# ── TỰ DÒ DATA_ROOT (quét /kaggle/input tìm thư mục có sets/train.csv + wav/ + metadata.csv) ──
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
CACHE_DIR = "/kaggle/working/xattn_cache"      # persist qua Save Version (KHÁC /kaggle/temp)
SEQ_DIR_W = f"{CACHE_DIR}/wavlm_seq"           # .npy fp16 frame-level WavLM (SAILER), per wav
SEQ_DIR_A = f"{CACHE_DIR}/aud_seq"             # .npy fp16 frame-level audeering, per wav
LLM_EMB_DIR = f"{CACHE_DIR}/qwen_emb"          # .npy fp16 POOLED embedding Qwen2-Audio, 1 vector/wav
for d in (CACHE_DIR, SEQ_DIR_W, SEQ_DIR_A, LLM_EMB_DIR):
    os.makedirs(d, exist_ok=True)

# ── (tùy chọn) cột QMOS để TRỘN — answer.txt exp13 (Add Input). "" = dùng fallback 3.0 ──
QMOS_ANSWER = "/kaggle/input/exp13-answer/answer.txt"   # << sửa slug; có cột wav,QMOS

# ── CrossAttnFusion (y exp18) ──
D_MODEL    = 256
Z_DIM      = 256
N_HEADS    = 4
N_LAYERS   = 1
XATTN_DIR  = "wavlm_q"     # "wavlm_q" (Q=WavLM,KV=aud) | "aud_q" | "bi"
USE_VAD3   = True          # ghép audeering VAD3 (thang 1–5) vào trunk
ATTN_DROP  = 0.1

# ── 🆕 AUDIO-LLM (Qwen2-Audio) — luồng thứ 3, pooled-concat ──
USE_AUDIOLLM    = True                                   # False → tái lập ĐÚNG exp18 (ablation)
QWEN_MODEL      = "Qwen/Qwen2-Audio-7B-Instruct"
AUDIOLLM_4BIT   = True                                   # BẮT BUỘC True trên T4 (fp16 7B không vừa 16GB)
AUDIOLLM_PROMPT = "Describe the emotion and speaking style of this speech."  # prompt điều hướng audio-LLM
AUDIOLLM_PROJ   = 256                                    # chiếu emb LLM (~3584/4096-D) → 256-D rồi mới concat
LLM_DIM         = None                                   # tự dò từ cache sau khi trích

# ── Cache frame-level (WavLM/audeering) ──
MAX_FRAMES   = 250         # cap số frame/wav (chống nổ dung lượng + OOM attention O(T1·T2))
FRAME_STRIDE = 2           # 1 = giữ nguyên; 2 = lấy 1/2 frame (cảm xúc là tín hiệu chậm → mất ít)
MAX_SECONDS  = 12          # cắt audio trước khi qua backbone
SR           = 16000

# ── Train (frozen → nhẹ, theo exp18) ──
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
USE_AUDEERING   = True
USE_UNCERTAINTY = True

# ── Ranking loss (train THEO SRCC) — y exp18 ──
LAMBDA_RANK = 0.3          # hệ số ranking loss cho 4 cột SRCC (EMOS/VAL/ARO/DOM); CAT giữ soft-CE

LIMIT_TRAIN = 300          # << LẦN ĐẦU 300; chạy thật None
LIMIT_DEV   = 20           # << LẦN ĐẦU 20; chạy thật None

# Mốc exp08 để so (leaderboard DEV) — lưới an toàn
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
print(f"CrossAttn: dir={XATTN_DIR} d_model={D_MODEL} z={Z_DIM} heads={N_HEADS} layers={N_LAYERS} VAD3={USE_VAD3}")
print(f"Audio-LLM: {'BẬT ' + QWEN_MODEL + (' (4-bit)' if AUDIOLLM_4BIT else ' (fp16)') if USE_AUDIOLLM else 'TẮT → = exp18'}"
      + (f" → chiếu {AUDIOLLM_PROJ}-D concat vào trunk" if USE_AUDIOLLM else ""))
print(f"Ranking loss (train theo SRCC): λ={LAMBDA_RANK}"
      + (" = MSE thuần" if LAMBDA_RANK == 0 else " (MSE + pairwise ranking trên EMOS/VAL/ARO/DOM)"))

# Cross-attention CẦN cả 2 luồng (WavLM + audeering) → audeering bắt buộc.
assert USE_AUDEERING, "exp20 cross-attention cần USE_AUDEERING=True (gộp WavLM ⟷ audeering)."
assert XATTN_DIR in ("wavlm_q", "aud_q", "bi"), f"XATTN_DIR lạ: {XATTN_DIR}"

# %% [markdown]
# ## 1. Cài đặt + clone SAILER + bitsandbytes (cho audio-LLM 4-bit)

# %%
import sys, subprocess

def pip_install(*pkgs):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], check=True)

# loralib + speechbrain BẮT BUỘC cho SAILER wrapper (LoRA). bitsandbytes + accelerate cho Qwen2-Audio 4-bit.
_pkgs = ["librosa", "soundfile", "scipy", "scikit-learn", "pandas", "tqdm", "safetensors",
         "loralib", "speechbrain"]
if USE_AUDIOLLM:
    _pkgs += ["bitsandbytes", "accelerate"]
pip_install(*_pkgs)

# Code SAILER (WavLM-large SER vô địch IS2025) — backbone WavLM frozen
REPO_DIR = "/kaggle/working/vox-profile-release"
if not os.path.exists(REPO_DIR):
    subprocess.run(["git", "clone", "--depth", "1",
                    "https://github.com/tiantiaf0627/vox-profile-release.git", REPO_DIR], check=True)
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

# %% [markdown]
# ## 2. Nhãn vàng cảm xúc (gộp trung bình theo wav) — COPY exp18

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
# ## 3. Trích + cache FRAME-LEVEL WavLM + audeering (đóng băng) — COPY exp18

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
from tqdm.auto import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device, ("✅ " + torch.cuda.get_device_name(0)) if device == "cuda" else "⚠️ CPU (rất chậm!)")
torch.manual_seed(SEED); np.random.seed(SEED)

def cap_frames(hs):
    """hs tensor [T,D] → ≤MAX_FRAMES frame, subsample đều theo thời gian (stride hiệu dụng)."""
    T = hs.shape[0]
    tgt = (T + FRAME_STRIDE - 1) // FRAME_STRIDE if FRAME_STRIDE > 1 else T
    tgt = max(1, min(tgt, MAX_FRAMES))
    if tgt < T:
        idx = torch.linspace(0, T - 1, tgt).long()
        hs = hs[idx]
    return hs

def load_wave(sid):
    p = os.path.join(WAV_DIR, sid + ".wav")
    if not os.path.exists(p):
        return None
    wave, _ = librosa.load(p, sr=SR, mono=True)
    return wave[: MAX_SECONDS * SR].astype(np.float32)

# ── WavLM (SAILER) frozen ──────────────────────────────────────────────
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

def extract_wavlm_seq(stems, tag):
    todo = [s for s in stems if not os.path.exists(os.path.join(SEQ_DIR_W, s + ".npy"))]
    if todo:
        wavlm = _get_wavlm()
        with torch.no_grad():
            for s in tqdm(todo, desc=f"wavlm-seq {tag}"):
                wave = load_wave(s)
                if wave is None:
                    continue
                iv = torch.from_numpy(wave).unsqueeze(0).to(device)
                hs = wavlm(iv).last_hidden_state[0]               # (T1, 1024)
                hs = cap_frames(hs.float().cpu())
                np.save(os.path.join(SEQ_DIR_W, s + ".npy"), hs.numpy().astype(np.float16))
        torch.cuda.empty_cache() if device == "cuda" else None
    return {s for s in stems if os.path.exists(os.path.join(SEQ_DIR_W, s + ".npy"))}

# ── audeering frozen (frame-level + VAD3) ──────────────────────────────
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
    _out = _sd["classifier.out_proj.weight"].shape[0]
    aud_head = nn.Sequential(nn.Linear(_hid, _hid), nn.Tanh(), nn.Linear(_hid, _out))
    aud_head[0].weight.data.copy_(_sd["classifier.dense.weight"]); aud_head[0].bias.data.copy_(_sd["classifier.dense.bias"])
    aud_head[2].weight.data.copy_(_sd["classifier.out_proj.weight"]); aud_head[2].bias.data.copy_(_sd["classifier.out_proj.bias"])
    aud_backbone = aud_backbone.to(device).eval()
    aud_head = aud_head.to(device).eval()
    AUD_DIM = _hid
    print(f"✅ audeering frozen (frame-dim {AUD_DIM}) + VAD3 head")

aud_vad = {}   # stem -> [VAL,ARO,DOM] thang 1–5

def extract_aud_seq(stems, tag):
    if not USE_AUDEERING:
        return set()
    vad_path = os.path.join(CACHE_DIR, f"aud_vad_{tag}.npz")
    if os.path.exists(vad_path):
        z = np.load(vad_path, allow_pickle=True)
        for k in z.files:
            aud_vad[k] = z[k]
    todo = [s for s in stems if not os.path.exists(os.path.join(SEQ_DIR_A, s + ".npy")) or s not in aud_vad]
    if todo:
        _get_audeering()
        with torch.no_grad():
            for i, s in enumerate(tqdm(todo, desc=f"aud-seq {tag}")):
                wave = load_wave(s)
                if wave is None:
                    continue
                x = aud_proc(wave, sampling_rate=SR).input_values[0]
                x = torch.from_numpy(np.asarray(x, dtype=np.float32)).unsqueeze(0).to(device)
                h = aud_backbone(x)[0]                          # (1, T2, hid)
                seq = cap_frames(h[0].float().cpu())            # (T2', hid)
                np.save(os.path.join(SEQ_DIR_A, s + ".npy"), seq.numpy().astype(np.float16))
                out = aud_head(h.mean(dim=1))[0].cpu().numpy()  # [aro,dom,val] ∈ [0,1]
                aud_vad[s] = np.array([1 + 4 * out[2], 1 + 4 * out[0], 1 + 4 * out[1]], dtype=np.float32)
                if (i + 1) % 500 == 0:
                    np.savez(vad_path, **aud_vad)
        np.savez(vad_path, **aud_vad)
        torch.cuda.empty_cache() if device == "cuda" else None
    return {s for s in stems if os.path.exists(os.path.join(SEQ_DIR_A, s + ".npy"))}

# %% [markdown]
# ## 3b. 🆕 Trích + cache POOLED EMBEDDING từ audio-LLM Qwen2-Audio (đóng băng, 4-bit)
# - Cho audio qua Qwen2-Audio (4-bit) với 1 prompt cảm xúc → lấy `hidden_states[-1]` (last layer LLM).
# - **Mean-pool tại vị trí token AUDIO** (bỏ token chữ của prompt) → 1 vector/wav → cache fp16.
# - Vector nhỏ (1 × hidden ~3584/4096) → cache rất nhẹ; KHÔNG giữ frame-level (mức nhẹ pooled-concat).

# %%
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

def _audio_token_ids(ids):
    """Trả mask token AUDIO trong input_ids (để mean-pool đúng phần audio, bỏ token chữ của prompt)."""
    for attr in ("audio_token_index", "audio_token_id"):
        tok = getattr(_qwen.config, attr, None)
        if tok is not None:
            m = (ids == tok)
            if bool(m.any()):
                return m
    return None

@torch.no_grad()
def _qwen_embed(wave):
    """1 wav → 1 vector pooled (hidden-state cuối của LLM tại token audio)."""
    conv = [{"role": "user", "content": [
        {"type": "audio", "audio_url": "x"},
        {"type": "text", "text": AUDIOLLM_PROMPT}]}]
    text = _qwen_proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
    inputs = _qwen_proc(text=text, audios=[wave], sampling_rate=SR, return_tensors="pt")
    inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}
    out = _qwen(**inputs, output_hidden_states=True, return_dict=True)
    last = out.hidden_states[-1][0]            # (seq, hidden)
    m = _audio_token_ids(inputs["input_ids"][0])
    emb = last[m].mean(0) if m is not None else last.mean(0)   # ưu tiên token audio; fallback toàn chuỗi
    return emb.float().cpu().numpy().astype(np.float16)

llm_emb = {}   # stem -> pooled embedding (np.float16)

def extract_audiollm_emb(stems, tag):
    """Cache pooled embedding Qwen2-Audio ra LLM_EMB_DIR/<stem>.npy + nạp vào dict llm_emb."""
    if not USE_AUDIOLLM:
        return set(stems)
    # nạp cache có sẵn vào RAM (resume sau khi kernel restart / Save Version)
    for s in stems:
        if s not in llm_emb:
            p = os.path.join(LLM_EMB_DIR, s + ".npy")
            if os.path.exists(p):
                llm_emb[s] = np.load(p)
    todo = [s for s in stems if s not in llm_emb]
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
            llm_emb[s] = e
        torch.cuda.empty_cache() if device == "cuda" else None
    return {s for s in stems if s in llm_emb}

# ── nạp chuỗi từ cache (RAM, fp16 → tensor float32 khi collate) ──
def load_seq(seq_dir, sid):
    p = os.path.join(seq_dir, sid + ".npy")
    return np.load(p) if os.path.exists(p) else None

def collate(sids):
    ws = [torch.from_numpy(load_seq(SEQ_DIR_W, s).astype(np.float32)) for s in sids]
    as_ = [torch.from_numpy(load_seq(SEQ_DIR_A, s).astype(np.float32)) for s in sids]
    def pad(seqs):
        Lmax = max(t.shape[0] for t in seqs); B = len(seqs)
        x = torch.zeros(B, Lmax, seqs[0].shape[1], dtype=torch.float32)
        m = torch.zeros(B, Lmax, dtype=torch.bool)
        for i, t in enumerate(seqs):
            x[i, : t.shape[0]] = t; m[i, : t.shape[0]] = True
        return x, m
    w, wm = pad(ws); a, am = pad(as_)
    return w, wm, a, am

# %% [markdown]
# ## 4. Module CrossAttnFusion + EmoHeads + train multi-task (uncertainty weighting)
# So exp18: chỉ thêm **llm_proj** (Linear LLM_DIM→AUDIOLLM_PROJ) + concat pooled emb vào trunk input.

# %%
class CrossAttnFusion(nn.Module):
    """WavLM frames [B,T1,1024] ⟷ audeering frames [B,T2,Da] → cross-attention → attentive-pool → z[B,Z_DIM]."""
    def __init__(self, d_w, d_a, d_model, n_heads, n_layers, z_dim, direction, p):
        super().__init__()
        self.direction = direction
        self.proj_w = nn.Linear(d_w, d_model)
        self.proj_a = nn.Linear(d_a, d_model)
        def _blocks():
            return (nn.ModuleList([nn.MultiheadAttention(d_model, n_heads, dropout=p, batch_first=True)
                                   for _ in range(n_layers)]),
                    nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)]),
                    nn.ModuleList([nn.Sequential(nn.Linear(d_model, d_model * 2), nn.GELU(),
                                                 nn.Linear(d_model * 2, d_model)) for _ in range(n_layers)]),
                    nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)]))
        self.mha, self.ln, self.ff, self.ln2 = _blocks()
        if direction == "bi":
            self.mha_b, self.ln_b, self.ff_b, self.ln2_b = _blocks()
        self.attn_pool = nn.Linear(d_model, 1)
        self.out = nn.Linear(d_model, z_dim)

    def _stack(self, q, kv, kv_pad, mha, ln, ff, ln2):
        for i in range(len(mha)):
            a, _ = mha[i](q, kv, kv, key_padding_mask=kv_pad, need_weights=False)
            q = ln[i](q + a)
            q = ln2[i](q + ff[i](q))
        return q

    def _pool(self, seq, mask):
        s = self.attn_pool(seq).squeeze(-1).masked_fill(~mask, float("-inf"))
        w = torch.softmax(s, dim=1).unsqueeze(-1)
        return (seq * w).sum(1)

    def forward(self, w, wm, a, am):
        hw, ha = self.proj_w(w), self.proj_a(a)
        w_pad, a_pad = ~wm, ~am
        if self.direction in ("wavlm_q", "bi"):
            q = self._stack(hw, ha, a_pad, self.mha, self.ln, self.ff, self.ln2)
            z = self.out(self._pool(q, wm))
            if self.direction == "wavlm_q":
                return z
            qb = self._stack(ha, hw, w_pad, self.mha_b, self.ln_b, self.ff_b, self.ln2_b)
            return z + self.out(self._pool(qb, am))
        q = self._stack(ha, hw, w_pad, self.mha, self.ln, self.ff, self.ln2)
        return self.out(self._pool(q, am))

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

# ── Trích cache cho TRAIN (WavLM + audeering + audio-LLM) ──
train_stems = list(train_df["wavID"])
if LIMIT_TRAIN:
    train_stems = train_stems[:LIMIT_TRAIN]
have_w = extract_wavlm_seq(train_stems, "train")
have_a = extract_aud_seq(train_stems, "train") if USE_AUDEERING else set(train_stems)
have_llm = extract_audiollm_emb(train_stems, "train")   # 🆕

# Tự dò chiều embedding LLM từ cache
if USE_AUDIOLLM and llm_emb:
    LLM_DIM = int(np.asarray(next(iter(llm_emb.values()))).shape[-1])
    print(f"🆕 LLM_DIM tự dò = {LLM_DIM} (Qwen2-Audio hidden) → chiếu xuống {AUDIOLLM_PROJ}-D")

def onehot_target(tgt):
    v = np.zeros(len(EMOTIONS5), dtype=np.float32)
    if tgt in EMOTIONS5:
        v[EMOTIONS5.index(tgt)] = 1.0
    return v

# ── Lọc wav đủ MỌI nguồn (WavLM + audeering + audio-LLM) + dựng nhãn ──
N_EMO = len(EMOTIONS5)
lab = train_df.set_index("wavID")
items = []
for s in train_stems:
    tgt = target_map.get(s)
    if s not in have_w or (USE_AUDEERING and s not in have_a) \
            or (USE_AUDIOLLM and s not in have_llm) or tgt is None or s not in lab.index:
        continue
    items.append((s,
                  onehot_target(tgt),
                  float(lab.loc[s, "emos"]),
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

# ── Chuẩn hoá LLM emb per-dim (thống kê TRAIN) — chống outlier 4-bit lấn llm_proj (bài học exp21) ──
L_MU = L_SD = None
if USE_AUDIOLLM:
    _X = np.stack([llm_emb[items[j][0]].astype(np.float32) for j in tr_idx])
    L_MU, L_SD = _X.mean(0), _X.std(0) + 1e-5
    print("✅ Chuẩn hoá LLM emb (mean0/std1 theo TRAIN) trước khi chiếu 256-D.")

AUD_BRANCH = AUD_DIM if USE_AUDEERING else 0
TRUNK_IN = Z_DIM + (3 if (USE_VAD3 and USE_AUDEERING) else 0) + (AUDIOLLM_PROJ if USE_AUDIOLLM else 0)

fusion = CrossAttnFusion(WAVLM_DIM, AUD_BRANCH if USE_AUDEERING else WAVLM_DIM,
                         D_MODEL, N_HEADS, N_LAYERS, Z_DIM, XATTN_DIR, ATTN_DROP).to(device)
heads = EmoHeads(TRUNK_IN, TRUNK_HIDDEN, HEAD_HIDDEN, DROPOUT, N_EMO).to(device)
llm_proj = nn.Linear(LLM_DIM, AUDIOLLM_PROJ).to(device) if USE_AUDIOLLM else None   # 🆕
print(f"Trunk input = {TRUNK_IN} (z {Z_DIM} + VAD3 {3 if (USE_VAD3 and USE_AUDEERING) else 0} "
      f"+ LLM {AUDIOLLM_PROJ if USE_AUDIOLLM else 0})")
_nparam = sum(p.numel() for p in fusion.parameters()) + sum(p.numel() for p in heads.parameters()) \
          + (sum(p.numel() for p in llm_proj.parameters()) if USE_AUDIOLLM else 0)
print(f"Tham số train (fusion+heads+llm_proj): {_nparam/1e6:.2f}M (backbone đóng băng, không train)")

TASKS = ["emos", "cat", "val", "aro", "dom"]
log_var = nn.Parameter(torch.zeros(len(TASKS), device=device))
params = list(fusion.parameters()) + list(heads.parameters()) \
    + (list(llm_proj.parameters()) if USE_AUDIOLLM else []) \
    + ([log_var] if USE_UNCERTAINTY else [])
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

def vad3_feat(sids):
    if not (USE_VAD3 and USE_AUDEERING):
        return None
    return torch.from_numpy(np.stack([aud_vad[s] for s in sids]).astype(np.float32)).to(device)

def llm_feat(sids):   # 🆕 pooled embedding LLM theo batch (đã chuẩn hoá per-dim)
    if not USE_AUDIOLLM:
        return None
    x = (np.stack([llm_emb[s] for s in sids]).astype(np.float32) - L_MU) / L_SD
    return torch.from_numpy(x).to(device)

def forward_batch(sids):
    """feature vào trunk = [z cross-attn | (tùy chọn) VAD3 | (tùy chọn) llm_proj(pooled LLM emb)]."""
    w, wm, a, am = collate(sids)
    w, wm, a, am = w.to(device), wm.to(device), a.to(device), am.to(device)
    z = fusion(w, wm, a, am)
    parts = [z]
    vf = vad3_feat(sids)
    if vf is not None:
        parts.append(vf)
    lf = llm_feat(sids)
    if lf is not None:
        parts.append(llm_proj(lf))     # 🆕
    return torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]

def run_heads(feat, tgt):
    return heads(feat, tgt)

@torch.no_grad()
def eval_val():
    fusion.eval(); heads.eval()
    if USE_AUDIOLLM:
        llm_proj.eval()
    preds_e, preds_v, preds_c = [], [], []
    for i in range(0, len(va_idx), BATCH):
        bi = va_idx[i:i + BATCH]
        sids = [items[j][0] for j in bi]
        feat = forward_batch(sids)
        tgt = torch.from_numpy(T_oh[bi]).to(device)
        ep, cl, vp = run_heads(feat, tgt)
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

CKPT_PATH = os.path.join(OUT_DIR, "exp20_audiollm.pt")
def save_ckpt(val_emos):
    blob = {"fusion": {k: v.cpu() for k, v in fusion.state_dict().items()},
            "heads": {k: v.cpu() for k, v in heads.state_dict().items()},
            "emos_mu": emos_mu, "emos_sd": emos_sd, "vad_mu": vad_mu, "vad_sd": vad_sd,
            "D_MODEL": D_MODEL, "Z_DIM": Z_DIM, "N_HEADS": N_HEADS, "N_LAYERS": N_LAYERS,
            "XATTN_DIR": XATTN_DIR, "USE_VAD3": USE_VAD3, "USE_AUDEERING": USE_AUDEERING,
            "USE_AUDIOLLM": USE_AUDIOLLM, "QWEN_MODEL": QWEN_MODEL, "AUDIOLLM_PROJ": AUDIOLLM_PROJ,
            "LLM_DIM": LLM_DIM, "AUDIOLLM_PROMPT": AUDIOLLM_PROMPT,
            "LAMBDA_RANK": LAMBDA_RANK, "AUD_DIM": AUD_DIM, "MAX_FRAMES": MAX_FRAMES,
            "FRAME_STRIDE": FRAME_STRIDE, "TRUNK_HIDDEN": TRUNK_HIDDEN, "HEAD_HIDDEN": HEAD_HIDDEN,
            "DROPOUT": DROPOUT, "EMOTIONS5": EMOTIONS5, "val_emos": float(val_emos)}
    if USE_AUDIOLLM:
        blob["llm_proj"] = {k: v.cpu() for k, v in llm_proj.state_dict().items()}
        blob["L_MU"] = L_MU; blob["L_SD"] = L_SD
    torch.save(blob, CKPT_PATH)

def snapshot():
    d = {"fusion": {k: v.cpu().clone() for k, v in fusion.state_dict().items()},
         "heads": {k: v.cpu().clone() for k, v in heads.state_dict().items()}}
    if USE_AUDIOLLM:
        d["llm_proj"] = {k: v.cpu().clone() for k, v in llm_proj.state_dict().items()}
    return d

def restore(d):
    fusion.load_state_dict(d["fusion"]); heads.load_state_dict(d["heads"])
    if USE_AUDIOLLM:
        llm_proj.load_state_dict(d["llm_proj"])

best_score, best_blob, bad = -1e9, None, 0
for ep in range(1, EPOCHS + 1):
    fusion.train(); heads.train()
    if USE_AUDIOLLM:
        llm_proj.train()
    perm = np.array(tr_idx); np.random.shuffle(perm)
    run = 0.0
    for i in range(0, len(perm), BATCH):
        bi = perm[i:i + BATCH]
        sids = [items[j][0] for j in bi]
        opt.zero_grad()
        feat = forward_batch(sids)
        tgt = torch.from_numpy(T_oh[bi]).to(device)
        ep_, cl, vp = run_heads(feat, tgt)
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
        best_blob = snapshot()
        save_ckpt(m["emos"]); bad = 0
    else:
        bad += 1
    if ep % 5 == 0 or ep == 1:
        msg = " ".join(f"{k}={m[k]:.3f}" for k in m)
        print(f"epoch {ep:3d} | loss {run/len(perm):.4f} | {msg} | best {best_score:.4f}")
    if bad >= PATIENCE:
        print(f"Early stop ở epoch {ep}."); break

if best_blob is not None:
    restore(best_blob)
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

dev_w = extract_wavlm_seq(dev_stems, "dev")
dev_a = extract_aud_seq(dev_stems, "dev") if USE_AUDEERING else set(dev_stems)
dev_llm = extract_audiollm_emb(dev_stems, "dev")   # 🆕

@torch.no_grad()
def predict_emotion(sid):
    if sid not in dev_w or (USE_AUDEERING and sid not in dev_a) or (USE_AUDIOLLM and sid not in dev_llm):
        return None
    fusion.eval(); heads.eval()
    if USE_AUDIOLLM:
        llm_proj.eval()
    feat = forward_batch([sid])
    tgt = torch.from_numpy(onehot_target(target_map.get(sid))[None, :]).to(device)
    ep, cl, vp = run_heads(feat, tgt)
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
os.system(f"cd {OUT_DIR} && zip -j submission_track2_exp20_audiollm.zip answer.txt "
          f"&& unzip -l submission_track2_exp20_audiollm.zip")
print("Sẵn sàng nộp:", os.path.join(OUT_DIR, "submission_track2_exp20_audiollm.zip"))

# %% [markdown]
# ## Ghi chú
# - **Lần đầu** `LIMIT_TRAIN=300`, `LIMIT_DEV=20` (forward 7B chậm → smoke test trước); rồi `None` chạy thật.
# - **OOM nạp Qwen?** Đảm bảo `AUDIOLLM_4BIT=True` (fp16 7B KHÔNG vừa T4). Vẫn OOM → giảm BATCH train (64→32)
#   và đảm bảo trích LLM xong rồi mới train (backbone không cần khi train, chỉ dùng cache).
# - **Ablation chính (cho paper):** `USE_AUDIOLLM=True` vs `False` (=exp18) → đo audio-LLM thêm bao nhiêu mỗi cột.
#   Phụ: `AUDIOLLM_PROJ ∈ {128,256,512}`; đổi `AUDIOLLM_PROMPT` (prompt cảm xúc vs trung tính).
# - **Lưới an toàn:** chỉ trộn/thay cột cảm xúc nếu VAL nội bộ **vượt mốc exp08**; thua → giữ exp08, vẫn là
#   ablation âm "audio-LLM không bổ trợ 2-encoder" cho paper.
# - **Checkpoint** `exp20_audiollm.pt` đủ predict-only (config + fusion + heads + llm_proj); KHÔNG lưu backbone
#   (đóng băng). **Cache pooled emb** `qwen_emb/*.npy` nên lưu ra Kaggle Dataset (trích 7B rất tốn thời gian).
#   **Save Version NGAY** sau khi chạy.
# - License phải khai báo: SAILER (Open RAIL) · audeering (CC BY-NC-SA) · **Qwen2-Audio (Apache-2.0)**.
# - Ghi config → kết quả → nhận xét vào `docs/04_experiments_log.md` (mục exp20).
