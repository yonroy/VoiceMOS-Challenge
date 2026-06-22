# %% [markdown]
# # VMC2026 Track 2 — exp19 (FINE-TUNE WavLM dưới + CROSS-ATTENTION trên) — Kaggle
#
# **Ý tưởng:** exp18 = cross-attention nhưng backbone ĐÓNG BĂNG (sức chứa nhỏ → không phá được kỷ lục VAD).
# exp19 **gỡ băng WavLM** (fine-tune top-K lớp, như exp08) RỒI cross-attention với audeering → cross-attn học
# trên đặc trưng *thích nghi theo nhiệm vụ*. Lấp ô trống: exp08 (ft + concat) · exp18 (frozen + xattn) → **exp19 (ft + xattn)**.
#
# ## 💡 Mẹo chạy được trên T4: fine-tune NỬA, cache NỬA
# - **WavLM (SAILER):** fine-tune **top-K lớp** (đóng băng feature-extractor + lớp dưới) → chạy LIVE mỗi batch.
# - **audeering:** ĐÓNG BĂNG → **cache frame-level** (dùng lại `aud_seq` của exp18) → khỏi tốn GPU/RAM.
# → mỗi step chỉ forward WavLM, audeering lấy từ cache. Nhẹ hơn exp11 (ft cả 2 backbone → overfit) nhiều.
#
# ## ⚠️ Phải biết trước
# - **Lần đầu BẮT BUỘC** `LIMIT_TRAIN=200`, `LIMIT_DEV=20` (fine-tune nặng → test trơn trước).
# - **OOM T4?** giảm `BATCH`(2), tăng `ACCUM`, giảm `MAX_SECONDS`(6), `UNFREEZE_TOP`(4), `MAX_FRAMES`(200).
# - **💾 Lưu CẢ backbone mỗi best + Save Version NGAY** (bài học exp08: kernel chết là mất backbone).
# - Lưới an toàn: chỉ trộn/thay cột nếu VAL nội bộ **vượt mốc exp08**. **Coi chừng OVERFIT** (vết exp11) → phải nộp DEV mới tin.
#
# **Cách chạy Kaggle:** GPU T4 + Internet On → Add Input (1) dataset Track 2, (2) tùy chọn answer.txt exp13 (trộn QMOS).

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
    print("🔎 Ứng viên DATA_ROOT:")
    for sc, r in _cands:
        print(f"   [{sc}/2] {r}")
    DATA_ROOT = _cands[0][1]
    print(f"👉 Tự chọn DATA_ROOT = {DATA_ROOT}")
else:
    DATA_ROOT = "/kaggle/input/datasets/minhtoan2"
    print(f"❌ Không thấy sets/train.csv → dùng dự phòng {DATA_ROOT} (đã Add Input chưa?)")

WAV_DIR      = f"{DATA_ROOT}/wav"
METADATA_CSV = f"{DATA_ROOT}/metadata.csv"
TRAIN_CSV    = f"{DATA_ROOT}/sets/train.csv"
DEV_SCP      = f"{DATA_ROOT}/sets/dev.scp"

OUT_DIR   = "/kaggle/working"
CACHE_DIR = "/kaggle/working/xattn_cache"      # tái dùng cache audeering của exp18 nếu Add Input
SEQ_DIR_A = f"{CACHE_DIR}/aud_seq"             # .npy fp16 frame-level audeering (ĐÓNG BĂNG → cache)
for d in (CACHE_DIR, SEQ_DIR_A):
    os.makedirs(d, exist_ok=True)

QMOS_ANSWER = "/kaggle/input/exp13-answer/answer.txt"   # << answer.txt exp13 để trộn cột QMOS; "" = fallback 3.0

# ── CrossAttnFusion (giống exp18) ──
D_MODEL    = 256
Z_DIM      = 256
N_HEADS    = 4
N_LAYERS   = 1
XATTN_DIR  = "wavlm_q"     # "wavlm_q" | "aud_q" | "bi"
USE_VAD3   = True
ATTN_DROP  = 0.1

# ── FINE-TUNE WavLM (cái mới của exp19) ──
UNFREEZE_TOP = 6           # số lớp encoder TOP của WavLM được mở băng (đóng băng feat-extractor + lớp dưới)
LR_BACKBONE  = 1e-5        # LR thấp cho backbone (giữ warm-start SAILER, chống quên)
LR_HEAD      = 1e-3        # LR cho cross-attn + heads
BATCH        = 2           # nhỏ (WavLM-large fine-tune) → bù bằng ACCUM
ACCUM        = 8           # effective batch = BATCH*ACCUM = 16
MAX_SECONDS  = 8

# ── audeering cache + cross-attn frame ──
MAX_FRAMES   = 250
FRAME_STRIDE = 2
SR           = 16000

# ── Ranking loss (train theo SRCC) — copy exp18 ──
LAMBDA_RANK = 0.3          # 0 = MSE thuần; >0 = MSE + pairwise ranking (EMOS/VAL/ARO/DOM)

# ── Train ──
TRUNK_HIDDEN = 512
HEAD_HIDDEN  = 128
DROPOUT      = 0.3
WEIGHT_DECAY = 1e-5
EPOCHS       = 12          # fine-tune warm-start → ít epoch
PATIENCE     = 4
VAL_FRAC     = 0.10
SEED         = 42
USE_UNCERTAINTY = True

LIMIT_TRAIN = 200          # << LẦN ĐẦU 200; chạy thật None
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
print(f"CrossAttn: dir={XATTN_DIR} d_model={D_MODEL} z={Z_DIM} | Fine-tune WavLM top {UNFREEZE_TOP} lớp "
      f"(LR bb={LR_BACKBONE}, head={LR_HEAD})")
print(f"Ranking loss: λ={LAMBDA_RANK} | BATCH={BATCH}×ACCUM={ACCUM}={BATCH*ACCUM} | MAX_SECONDS={MAX_SECONDS}")

assert XATTN_DIR in ("wavlm_q", "aud_q", "bi"), f"XATTN_DIR lạ: {XATTN_DIR}"

# %% [markdown]
# ## 1. Cài đặt + clone SAILER (backbone WavLM warm-start)

# %%
import sys, subprocess

def pip_install(*pkgs):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], check=True)

# loralib + speechbrain BẮT BUỘC cho SAILER wrapper (thiếu → fallback WavLM trắng → mất warm-start cảm xúc).
pip_install("librosa", "soundfile", "scipy", "scikit-learn", "pandas", "tqdm", "safetensors",
            "loralib", "speechbrain")

REPO_DIR = "/kaggle/working/vox-profile-release"
if not os.path.exists(REPO_DIR):
    subprocess.run(["git", "clone", "--depth", "1",
                    "https://github.com/tiantiaf0627/vox-profile-release.git", REPO_DIR], check=True)
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

# %% [markdown]
# ## 2. Nhãn vàng cảm xúc (gộp theo wav) — COPY exp18

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
# ## 3. Backbone: WavLM (fine-tune top-K, LIVE) + audeering (frozen, CACHE)

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
from tqdm.auto import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device, ("✅ " + torch.cuda.get_device_name(0)) if device == "cuda" else "⚠️ CPU (rất chậm!)")
torch.manual_seed(SEED); np.random.seed(SEED)
WAVLM_DIM = 1024

def cap_frames(hs):
    """hs [T,D] (CÓ THỂ giữ grad) → ≤MAX_FRAMES frame, subsample đều theo thời gian."""
    T = hs.shape[0]
    tgt = (T + FRAME_STRIDE - 1) // FRAME_STRIDE if FRAME_STRIDE > 1 else T
    tgt = max(1, min(tgt, MAX_FRAMES))
    if tgt < T:
        idx = torch.linspace(0, T - 1, tgt, device=hs.device).long()
        hs = hs[idx]
    return hs

_wave_cache = {}
def get_wave(sid):
    if sid in _wave_cache:
        return _wave_cache[sid]
    p = os.path.join(WAV_DIR, sid + ".wav")
    if not os.path.exists(p):
        _wave_cache[sid] = None
        return None
    wave, _ = librosa.load(p, sr=SR, mono=True)
    wave = wave[: MAX_SECONDS * SR].astype(np.float32)
    _wave_cache[sid] = wave
    return wave

# ── WavLM (SAILER) — mở băng top-K lớp ──
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

def build_wavlm():
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
    bb = bb.to(device)
    # đóng băng tất cả, rồi mở băng top-K lớp encoder
    for p in bb.parameters():
        p.requires_grad = False
    enc_layers = bb.encoder.layers
    k = min(UNFREEZE_TOP, len(enc_layers))
    for layer in enc_layers[-k:]:
        for p in layer.parameters():
            p.requires_grad = True
    n_train = sum(p.numel() for p in bb.parameters() if p.requires_grad)
    print(f"✅ WavLM: mở băng {k}/{len(enc_layers)} lớp top → {n_train/1e6:.1f}M tham số train (backbone)")
    return bb

wavlm = build_wavlm()

def wavlm_seq(wave):
    """1 wave (np) → frame-level [T1,1024] CÓ GRAD (qua lớp đã mở băng)."""
    iv = torch.from_numpy(wave).unsqueeze(0).to(device)
    hs = wavlm(iv).last_hidden_state[0]      # (T1,1024)
    return cap_frames(hs)

# ── audeering frozen (frame-level + VAD3) — COPY exp18 ──
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

aud_vad = {}
def extract_aud_seq(stems, tag):
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
                wave = get_wave(s)
                if wave is None:
                    continue
                x = aud_proc(wave, sampling_rate=SR).input_values[0]
                x = torch.from_numpy(np.asarray(x, dtype=np.float32)).unsqueeze(0).to(device)
                h = aud_backbone(x)[0]
                seq = cap_frames(h[0].float().cpu())
                np.save(os.path.join(SEQ_DIR_A, s + ".npy"), seq.numpy().astype(np.float16))
                out = aud_head(h.mean(dim=1))[0].cpu().numpy()
                aud_vad[s] = np.array([1 + 4 * out[2], 1 + 4 * out[0], 1 + 4 * out[1]], dtype=np.float32)
                if (i + 1) % 500 == 0:
                    np.savez(vad_path, **aud_vad)
        np.savez(vad_path, **aud_vad)
        torch.cuda.empty_cache() if device == "cuda" else None
    return {s for s in stems if os.path.exists(os.path.join(SEQ_DIR_A, s + ".npy"))}

def load_aud_seq(sid):
    p = os.path.join(SEQ_DIR_A, sid + ".npy")
    return np.load(p) if os.path.exists(p) else None

# %% [markdown]
# ## 4. CrossAttnFusion + EmoHeads + ranking loss (COPY exp18) + train loop FINE-TUNE

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

# ── Trích cache audeering (WavLM KHÔNG cache — fine-tune) ──
train_stems = list(train_df["wavID"])
if LIMIT_TRAIN:
    train_stems = train_stems[:LIMIT_TRAIN]
have_a = extract_aud_seq(train_stems, "train")

def onehot_target(tgt):
    v = np.zeros(len(EMOTIONS5), dtype=np.float32)
    if tgt in EMOTIONS5:
        v[EMOTIONS5.index(tgt)] = 1.0
    return v

N_EMO = len(EMOTIONS5)
lab = train_df.set_index("wavID")
items = []
for s in train_stems:
    tgt = target_map.get(s)
    if s not in have_a or get_wave(s) is None or tgt is None or s not in lab.index:
        continue
    items.append((s, onehot_target(tgt), float(lab.loc[s, "emos"]),
                  [lab.loc[s, "val"], lab.loc[s, "aro"], lab.loc[s, "dom"]],
                  [lab.loc[s, f"cat{i}"] for i in range(N_EMO)]))
print(f"Train items đủ nguồn: {len(items)}")
assert len(items) >= 8, "Quá ít mẫu — kiểm tra cache/nhãn (LIMIT quá nhỏ?)."

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

AUD_BRANCH = AUD_DIM
TRUNK_IN = Z_DIM + (3 if USE_VAD3 else 0)
fusion = CrossAttnFusion(WAVLM_DIM, AUD_BRANCH, D_MODEL, N_HEADS, N_LAYERS, Z_DIM, XATTN_DIR, ATTN_DROP).to(device)
heads = EmoHeads(TRUNK_IN, TRUNK_HIDDEN, HEAD_HIDDEN, DROPOUT, N_EMO).to(device)
print(f"Trunk input = {TRUNK_IN} | Fusion {sum(p.numel() for p in fusion.parameters())/1e6:.2f}M · "
      f"Heads {sum(p.numel() for p in heads.parameters())/1e6:.2f}M")

TASKS = ["emos", "cat", "val", "aro", "dom"]
log_var = nn.Parameter(torch.zeros(len(TASKS), device=device))
bb_params = [p for p in wavlm.parameters() if p.requires_grad]
head_params = list(fusion.parameters()) + list(heads.parameters()) + ([log_var] if USE_UNCERTAINTY else [])
opt = torch.optim.AdamW([{"params": bb_params, "lr": LR_BACKBONE},
                         {"params": head_params, "lr": LR_HEAD}], weight_decay=WEIGHT_DECAY)
scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))
mse = nn.MSELoss(reduction="none")

def vad3_feat(sids):
    if not USE_VAD3:
        return None
    return torch.from_numpy(np.stack([aud_vad[s] for s in sids]).astype(np.float32)).to(device)

def forward_batch(sids):
    """list stem → feature [B, Z_DIM(+3)]. WavLM chạy LIVE (có grad), audeering từ cache."""
    w_list = [wavlm_seq(get_wave(s)) for s in sids]                # mỗi cái (T1,1024) có grad
    Tw = max(t.shape[0] for t in w_list)
    w  = w_list[0].new_zeros(len(sids), Tw, WAVLM_DIM)
    wm = torch.zeros(len(sids), Tw, dtype=torch.bool, device=device)
    for i, t in enumerate(w_list):
        w[i, : t.shape[0]] = t; wm[i, : t.shape[0]] = True
    a_list = [torch.from_numpy(load_aud_seq(s).astype(np.float32)) for s in sids]
    Ta = max(t.shape[0] for t in a_list)
    a  = torch.zeros(len(sids), Ta, AUD_DIM)
    am = torch.zeros(len(sids), Ta, dtype=torch.bool)
    for i, t in enumerate(a_list):
        a[i, : t.shape[0]] = t; am[i, : t.shape[0]] = True
    a, am = a.to(device), am.to(device)
    z = fusion(w, wm, a, am)
    vf = vad3_feat(sids)
    return torch.cat([z, vf], dim=1) if vf is not None else z

emos_t = torch.from_numpy(y_emos_z).float().unsqueeze(1).to(device)
vad_t  = torch.from_numpy(y_vad_z).float().to(device)
cat_t  = torch.from_numpy(y_cat).float().to(device)

@torch.no_grad()
def eval_val():
    wavlm.eval(); fusion.eval(); heads.eval()
    preds_e, preds_v, preds_c = [], [], []
    for i in range(0, len(va_idx), BATCH):
        bi = va_idx[i:i + BATCH]
        sids = [items[j][0] for j in bi]
        with torch.cuda.amp.autocast(enabled=(device == "cuda")):
            feat = forward_batch(sids)
            tgt = torch.from_numpy(T_oh[bi]).to(device)
            ep, cl, vp = heads(feat, tgt)
        preds_e.append(ep.float().cpu().numpy().ravel())
        preds_v.append(vp.float().cpu().numpy())
        preds_c.append(F.softmax(cl.float(), dim=1).cpu().numpy())
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
    return float(np.mean([m[k] for k in keys]))

CKPT_PATH = os.path.join(OUT_DIR, "ft_xattn_full.pt")
def save_ckpt(val_emos):
    torch.save({"wavlm": {k: v.cpu() for k, v in wavlm.state_dict().items()},   # LƯU CẢ BACKBONE
                "fusion": {k: v.cpu() for k, v in fusion.state_dict().items()},
                "heads": {k: v.cpu() for k, v in heads.state_dict().items()},
                "emos_mu": emos_mu, "emos_sd": emos_sd, "vad_mu": vad_mu, "vad_sd": vad_sd,
                "D_MODEL": D_MODEL, "Z_DIM": Z_DIM, "N_HEADS": N_HEADS, "N_LAYERS": N_LAYERS,
                "XATTN_DIR": XATTN_DIR, "USE_VAD3": USE_VAD3, "AUD_DIM": AUD_DIM,
                "UNFREEZE_TOP": UNFREEZE_TOP, "LAMBDA_RANK": LAMBDA_RANK,
                "MAX_FRAMES": MAX_FRAMES, "FRAME_STRIDE": FRAME_STRIDE, "MAX_SECONDS": MAX_SECONDS,
                "TRUNK_HIDDEN": TRUNK_HIDDEN, "HEAD_HIDDEN": HEAD_HIDDEN, "DROPOUT": DROPOUT,
                "EMOTIONS5": EMOTIONS5, "val_emos": float(val_emos)}, CKPT_PATH)

best_score, best_blob, bad = -1e9, None, 0
for ep in range(1, EPOCHS + 1):
    wavlm.train(); fusion.train(); heads.train()
    perm = np.array(tr_idx); np.random.shuffle(perm)
    run = 0.0; opt.zero_grad(); pending = False
    for step, i in enumerate(range(0, len(perm), BATCH)):
        bi = perm[i:i + BATCH]
        sids = [items[j][0] for j in bi]
        with torch.cuda.amp.autocast(enabled=(device == "cuda")):
            feat = forward_batch(sids)
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
        scaler.scale(loss / ACCUM).backward(); pending = True
        if (step + 1) % ACCUM == 0:
            scaler.step(opt); scaler.update(); opt.zero_grad(); pending = False
        run += float(loss.item()) * len(bi)
    if pending:                                          # flush phần dư cuối epoch (chỉ khi còn grad)
        scaler.step(opt); scaler.update(); opt.zero_grad()
    m = eval_val(); sc = val_score(m)
    if sc > best_score:
        best_score = sc
        best_blob = ({k: v.cpu().clone() for k, v in wavlm.state_dict().items()},
                     {k: v.cpu().clone() for k, v in fusion.state_dict().items()},
                     {k: v.cpu().clone() for k, v in heads.state_dict().items()})
        save_ckpt(m["emos"]); bad = 0
    else:
        bad += 1
    msg = " ".join(f"{k}={m[k]:.3f}" for k in m)
    print(f"epoch {ep:2d} | loss {run/len(perm):.4f} | {msg} | best {best_score:.4f}")
    if bad >= PATIENCE:
        print(f"Early stop ở epoch {ep}."); break

if best_blob is not None:
    wavlm.load_state_dict(best_blob[0]); fusion.load_state_dict(best_blob[1]); heads.load_state_dict(best_blob[2])
final = eval_val()
print("\n✅ VAL (nội bộ) tốt nhất:")
print(f"   EMOS SRCC = {final['emos']:.4f}   (mốc exp08 = {EXP08['emos']})")
if HAS_VAD:
    print(f"   VAL/ARO/DOM = {final['val']:.4f} / {final['aro']:.4f} / {final['dom']:.4f}"
          f"   (mốc exp08 = {EXP08['val']} / {EXP08['aro']} / {EXP08['dom']})")
    for key in ["val", "aro", "dom", "emos"]:
        if final[key] < EXP08[key] - 0.005:
            print(f"   ⚠️ {key.upper()} {final[key]:.4f} THUA exp08 {EXP08[key]} → giữ exp08 cho cột này.")
print(f"💾 checkpoint best (CÓ backbone) → {CKPT_PATH}  (⚠️ Save Version NGAY!)")

# %% [markdown]
# ## 5. Dự đoán DEV (WavLM LIVE, đảo z-score)

# %%
def list_dev():
    with open(DEV_SCP) as f:
        return [ln.strip() for ln in f if ln.strip()]

dev_names = list_dev()
if LIMIT_DEV:
    dev_names = dev_names[:LIMIT_DEV]
dev_stems = [stem(n) for n in dev_names]
print("DEV:", len(dev_names), "mẫu")
dev_a = extract_aud_seq(dev_stems, "dev")

@torch.no_grad()
def predict_emotion(sid):
    if sid not in dev_a or get_wave(sid) is None:
        return None
    wavlm.eval(); fusion.eval(); heads.eval()
    with torch.cuda.amp.autocast(enabled=(device == "cuda")):
        feat = forward_batch([sid])
        tgt = torch.from_numpy(onehot_target(target_map.get(sid))[None, :]).to(device)
        ep, cl, vp = heads(feat, tgt)
    emos = float(np.clip(ep.float().item() * emos_sd + emos_mu, 1.0, 5.0))
    cat5 = F.softmax(cl.float(), dim=1)[0].cpu().numpy()
    vad3 = np.clip(vp[0].float().cpu().numpy() * vad_sd + vad_mu, 1.0, 5.0)
    return emos, cat5, vad3

# %% [markdown]
# ## 6. Ghép answer.txt 6 cột (QMOS trộn từ exp13 / fallback) — COPY exp18

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
os.system(f"cd {OUT_DIR} && zip -j submission_track2_exp19_ft-crossattn.zip answer.txt "
          f"&& unzip -l submission_track2_exp19_ft-crossattn.zip")
print("Sẵn sàng nộp:", os.path.join(OUT_DIR, "submission_track2_exp19_ft-crossattn.zip"))

# %% [markdown]
# ## Ghi chú
# - **Lần đầu** `LIMIT_TRAIN=200`, `LIMIT_DEV=20` để chạy trơn; rồi `None` chạy thật.
# - **OOM T4?** giảm theo thứ tự: `BATCH`(2→1, tăng `ACCUM`) → `MAX_SECONDS`(8→6) → `UNFREEZE_TOP`(6→4) → `MAX_FRAMES`(250→200).
# - **CHẬM** (WavLM live, không cache) → ít epoch (warm-start gần đỉnh); cache audeering tái dùng được từ exp18.
# - **OVERFIT** (vết exp11): val nội bộ đẹp ≠ DEV. LR backbone thấp + mở ít lớp + early-stop SRCC; **phải nộp DEV mới tin**.
# - **Checkpoint `ft_xattn_full.pt` CÓ backbone** → predict-only sau cần nạp cả wavlm. Save Version NGAY (bài học exp08).
# - **Ablation:** `UNFREEZE_TOP ∈ {2,4,6}` · cross-attn(exp19) vs concat(exp08) · `LAMBDA_RANK ∈ {0,0.3}` · so exp18(frozen)/exp11(ft cả 2).
# - License: SAILER (Open RAIL) · audeering (CC BY-NC-SA) — phi thương mại, khai báo `docs/12_system_description.md`.
# - Ghi config → kết quả → nhận xét vào `docs/04_experiments_log.md` (mục exp19).
