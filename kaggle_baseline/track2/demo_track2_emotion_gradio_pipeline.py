# %% [markdown]
# # VMC2026 Track 2 — Demo Gradio "Emotional TTS Evaluator" (model TỐT NHẤT = exp08)
#
# Demo này dùng **checkpoint cảm xúc tốt nhất** (`ft_emotion_full_20epoch.pt`: WavLM fine-tune warm-start
# SAILER + audeering frozen) để chấm **5 cột cảm xúc** của 1 file giọng TTS: **EMOS / CAT / VAL / ARO / DOM**.
# Khác demo cũ (`demo_track2_gradio`) dùng baseline UTMOS+emotion2vec+Gemini — bản này KHÔNG cần API.
#
# **2 tab:**
# 1. *Chấm 1 file TTS* — tải audio + chọn cảm xúc target → ra điểm biểu cảm cảm xúc + diễn giải KHỚP/LỆCH.
# 2. *Metric bộ chấm* — tính UTT-SRCC (EMOS/VAD) + CAT-err trên val nội bộ (train.csv) → cho biết độ tin cậy.
#
# **Cách chạy Kaggle:** GPU **T4** + Internet **On** → Add Input (1) dataset Track 2, (2) dataset chứa
# `ft_emotion_full_20epoch.pt` → Run All → cell cuối in link `*.gradio.live`.

# %% [markdown]
# ## 0. Cấu hình — auto-dò DATA_ROOT + checkpoint

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
    DATA_ROOT = "/kaggle/input/datasets/minhtoan2"   # dự phòng
    print(f"❌ Không thấy sets/train.csv → dự phòng {DATA_ROOT} (đã Add Input chưa?)")

WAV_DIR      = f"{DATA_ROOT}/wav"
METADATA_CSV = f"{DATA_ROOT}/metadata.csv"
TRAIN_CSV    = f"{DATA_ROOT}/sets/train.csv"

# ── Checkpoint cảm xúc exp08 (ưu tiên bản 20 epoch = TỐT NHẤT) ─────────────────
CKPT_PATH = ""    # << "" = auto-dò; hoặc trỏ tay "/kaggle/input/<slug>/ft_emotion_full_20epoch.pt"

def find_ckpt(explicit):
    if explicit and os.path.exists(explicit):
        return explicit
    pats = ["ft_emotion_full_20epoch*.pt", "ft_emotion_full*.pt"]   # ưu tiên bản 20epoch
    for pat in pats:
        for base in ["/kaggle/input", "/kaggle/working"]:
            hits = sorted(glob.glob(os.path.join(base, "**", pat), recursive=True))
            if hits:
                return hits[0]
    return ""

CKPT_PATH = find_ckpt(CKPT_PATH)
assert CKPT_PATH, "❌ Không thấy ft_emotion_full*.pt. Add Input dataset chứa checkpoint exp08 chưa?"
print("✅ Checkpoint:", CKPT_PATH)

# ── Hằng kiến trúc PHẢI khớp exp08 (ckpt không lưu các số này) ────────────────
DEVICE       = "cuda"
SR           = 16000
EMO_MAX_SEC  = 8
TRUNK_HIDDEN = 512
HEAD_HIDDEN  = 128
DROPOUT      = 0.3       # không ảnh hưởng eval
USE_AMP      = True

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

# Mốc exp08 (val nội bộ / DEV) để so trong tab metric
EXP08 = {"emos": 0.811, "cat_err": 0.133, "val": 0.659, "aro": 0.793, "dom": 0.751}

# %% [markdown]
# ## 1. Cài đặt + clone code SAILER

# %%
import sys, subprocess

def pip_install(*pkgs):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], check=True)

pip_install("gradio", "loralib", "speechbrain", "librosa", "soundfile",
            "scipy", "scikit-learn", "pandas", "tqdm")

REPO_DIR = "/kaggle/working/vox-profile-release"
if not os.path.exists(REPO_DIR):
    subprocess.run(["git", "clone", "--depth", "1",
                    "https://github.com/tiantiaf0627/vox-profile-release.git", REPO_DIR], check=True)
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

# %% [markdown]
# ## 2. Nạp model exp08 (backbone WavLM ft + audeering frozen + heads) — 1 lần

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import librosa

device = DEVICE if torch.cuda.is_available() else "cpu"
print("Device:", device, ("✅ " + torch.cuda.get_device_name(0)) if device == "cuda" else "⚠️ CPU (chậm)")

ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)   # ckpt có numpy → cần False
assert "wavlm" in ckpt and "heads" in ckpt, "❌ Checkpoint thiếu 'wavlm'/'heads' → cần ft_emotion_full_20epoch.pt đủ."
AUD_DIM = int(ckpt.get("AUD_DIM", 0))
USE_AUDEERING = AUD_DIM > 0
print("✅ Nạp ckpt | keys:", list(ckpt.keys()), "| AUD_DIM:", AUD_DIM, "(audeering", "ON)" if USE_AUDEERING else "OFF)")

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
    _name, wavlm = find_hf_backbone(_wrapper)
    if wavlm is not None:
        print(f"✅ Dựng backbone WavLM từ SAILER wrapper tại '.{_name}'")
except Exception as e:
    print("⚠️ Lỗi nạp SAILER wrapper:", repr(e), "→ fallback WavLM trắng.")
if wavlm is None:
    from transformers import WavLMModel
    wavlm = WavLMModel.from_pretrained("microsoft/wavlm-large")
    print("ℹ️ Fallback: microsoft/wavlm-large.")

wavlm = wavlm.to(device).eval()
WAVLM_DIM = int(wavlm.config.hidden_size)
wavlm.config.layerdrop = 0.0
_miss, _unexp = wavlm.load_state_dict(ckpt["wavlm"], strict=False)
print(f"🔁 load wavlm: thiếu {len(_miss)} / dư {len(_unexp)} key (kỳ vọng ~0)")

def masked_mean(hidden, attn_mask):
    if attn_mask is None:
        return hidden.mean(dim=1)
    try:
        fm = wavlm._get_feature_vector_attention_mask(hidden.shape[1], attn_mask)
    except Exception:
        return hidden.mean(dim=1)
    fm = fm.unsqueeze(-1).to(hidden.dtype)
    return (hidden * fm).sum(1) / fm.sum(1).clamp(min=1e-6)

@torch.no_grad()
def wavlm_embed(input_values, attn_mask):
    out = wavlm(input_values, attention_mask=attn_mask).last_hidden_state
    return masked_mean(out, attn_mask)

# ── audeering frozen (đặc trưng phụ) — chỉ dựng nếu ckpt có dùng ──
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
    assert _hid + 3 == AUD_DIM, f"⚠️ AUD_DIM dựng ({_hid+3}) ≠ ckpt ({AUD_DIM})"
    print(f"✅ audeering frozen ({AUD_DIM}-D)")

@torch.no_grad()
def audeering_feat(wave):
    x = aud_proc(wave, sampling_rate=SR).input_values[0]
    x = torch.from_numpy(np.asarray(x, dtype=np.float32)).unsqueeze(0).to(device)
    h = aud_backbone(x)[0].mean(dim=1)
    out = aud_head(h)[0].cpu().numpy()
    vad = np.array([1 + 4 * out[2], 1 + 4 * out[0], 1 + 4 * out[1]], dtype=np.float32)  # [VAL,ARO,DOM]
    return np.concatenate([h[0].cpu().numpy(), vad]).astype(np.float32)

# ── EmoHeads (khớp exp08) + nạp trọng số + chuẩn hóa từ ckpt ──
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

heads = EmoHeads(TRUNK_IN, TRUNK_HIDDEN, HEAD_HIDDEN, DROPOUT, N_EMO).to(device).eval()
_hm, _hu = heads.load_state_dict(ckpt["heads"], strict=False)
print(f"🔁 load heads: thiếu {len(_hm)} / dư {len(_hu)} key (kỳ vọng 0)")

emos_mu = float(ckpt["emos_mu"]); emos_sd = float(ckpt["emos_sd"])
vad_mu = np.asarray(ckpt["vad_mu"], dtype=np.float32); vad_sd = np.asarray(ckpt["vad_sd"], dtype=np.float32)
print(f"Chuẩn hóa từ ckpt: emos μ={emos_mu:.3f} σ={emos_sd:.3f} | vad μ={np.round(vad_mu,2)}")

def onehot_target(tgt):
    v = np.zeros(N_EMO, dtype=np.float32)
    if tgt in EMOTIONS5:
        v[EMOTIONS5.index(tgt)] = 1.0
    return v

# %% [markdown]
# ## 3. Hàm suy luận lõi (1 wave numpy → emos/cat5/vad3)

# %%
@torch.no_grad()
def infer_wave(wave, target_emotion):
    """wave: numpy float32 (đã 16k mono). target_emotion: str hoặc None. Trả (emos, cat5, vad3)."""
    wave = wave[: EMO_MAX_SEC * SR].astype(np.float32)
    iv = torch.from_numpy(wave).unsqueeze(0).to(device)
    am = torch.ones((1, len(wave)), dtype=torch.long, device=device)
    tgt = torch.from_numpy(onehot_target(norm_emotion(target_emotion) if target_emotion else None)).unsqueeze(0).to(device)
    with torch.cuda.amp.autocast(enabled=USE_AMP and device == "cuda"):
        fw = wavlm_embed(iv, am)
        if USE_AUDEERING:
            fw = torch.cat([fw, torch.from_numpy(audeering_feat(wave)).unsqueeze(0).to(device)], dim=1)
        emos_p, cat_l, vad_p = heads(fw, tgt)
    emos = float(emos_p.item()) * emos_sd + emos_mu
    cat5 = F.softmax(cat_l, 1)[0].float().cpu().numpy()
    vad3 = vad_p[0].float().cpu().numpy() * vad_sd + vad_mu
    return emos, cat5, vad3

# %% [markdown]
# ## 4. Hàm metric val nội bộ (UTT-SRCC + CAT-err) — đánh giá độ tin cậy bộ chấm

# %%
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

def parse_emocat_votes(cell):
    v = np.zeros(N_EMO, dtype=np.float32)
    for tok in str(cell).replace("/", ",").replace(";", ",").replace("|", ",").replace(" ", ",").split(","):
        e = norm_emotion(tok)
        if e in EMOTIONS5:
            v[EMOTIONS5.index(e)] += 1.0
    return v

def _col(cols_map, *names, df=None, default_idx=None):
    for n in names:
        if n in cols_map:
            return cols_map[n]
    return list(df.columns)[default_idx] if default_idx is not None else None

def load_train_labels():
    df = pd.read_csv(TRAIN_CSV, sep="|")
    cols = {c.lower().strip(): c for c in df.columns}
    wav_col = _col(cols, "wavid", "wav", df=df, default_idx=1)
    emos_col = _col(cols, "emos", "emo", "emomos")
    val_col = _col(cols, "val", "valence"); aro_col = _col(cols, "aro", "arousal"); dom_col = _col(cols, "dom", "dominance")
    cat_col = _col(cols, "emocat", "cat", "emotion")
    df["_stem"] = df[wav_col].map(stem)
    rows = []
    for sid, g in df.groupby("_stem"):
        rec = {"wavID": sid, "emos": float(g[emos_col].mean())}
        rec["val"] = float(g[val_col].mean()) if val_col else np.nan
        rec["aro"] = float(g[aro_col].mean()) if aro_col else np.nan
        rec["dom"] = float(g[dom_col].mean()) if dom_col else np.nan
        votes = np.zeros(N_EMO, dtype=np.float32)
        if cat_col:
            for cell in g[cat_col]:
                votes += parse_emocat_votes(cell)
        s = votes.sum()
        cat = votes / s if s > 0 else np.full(N_EMO, 0.2, dtype=np.float32)
        for i in range(N_EMO):
            rec[f"cat{i}"] = float(cat[i])
        rows.append(rec)
    return pd.DataFrame(rows)

# target cảm xúc theo wav (cho EMOS) từ metadata
def load_target_emotions():
    tgt = {}
    if os.path.exists(METADATA_CSV):
        with open(METADATA_CSV, encoding="utf-8") as f:
            for ln in f:
                parts = ln.strip().split("|")
                if len(parts) >= 2:
                    tgt[stem(parts[0])] = norm_emotion(parts[1])
    return tgt

_target_map = None
_val_df = None
def _prep_eval():
    """Lazy: đọc nhãn + tách 10% val nội bộ (seed 42, khớp exp08)."""
    global _target_map, _val_df
    if _val_df is None:
        _target_map = load_target_emotions()
        df = load_train_labels()
        df = df[df["wavID"].map(lambda s: os.path.exists(os.path.join(WAV_DIR, s + ".wav")))].reset_index(drop=True)
        _, va = train_test_split(np.arange(len(df)), test_size=0.10, random_state=42)
        _val_df = df.iloc[va].reset_index(drop=True)
    return _target_map, _val_df

def eval_metrics(limit):
    tmap, vdf = _prep_eval()
    n = min(int(limit), len(vdf))
    P = {"emos": [], "val": [], "aro": [], "dom": []}; Y = {"emos": [], "val": [], "aro": [], "dom": []}
    catP, catY = [], []
    for i in tqdm(range(n), desc="eval"):
        r = vdf.iloc[i]; sid = r["wavID"]
        wav = os.path.join(WAV_DIR, sid + ".wav")
        wave, _ = librosa.load(wav, sr=SR, mono=True)
        emos, cat5, vad3 = infer_wave(wave, tmap.get(sid))
        P["emos"].append(emos); Y["emos"].append(float(r["emos"]))
        for j, t in enumerate(["val", "aro", "dom"]):
            P[t].append(float(vad3[j])); Y[t].append(float(r[t]))
        catP.append(cat5); catY.append([r[f"cat{k}"] for k in range(N_EMO)])
    rows = []
    for t in ["emos", "val", "aro", "dom"]:
        srcc = spearmanr(P[t], Y[t]).correlation
        rows.append([t.upper(), f"{srcc:.4f}", f"{EXP08.get(t, float('nan')):.3f}"])
    cat_err = float(np.abs(np.array(catP) - np.array(catY)).sum(1).mean())
    rows.append(["CAT-err ↓", f"{cat_err:.4f}", f"{EXP08['cat_err']:.3f}"])
    return rows

# %% [markdown]
# ## 5. Giao diện Gradio (2 tab)

# %%
import gradio as gr

def ui_predict(audio, target_emotion):
    """Trả về: verdict(md) · EMOS(number) · CAT(label) · VAL/ARO/DOM(number)."""
    if not audio:
        return "### ⚠️ Hãy tải audio.", None, {}, None, None, None
    wave, _ = librosa.load(audio, sr=SR, mono=True)
    emos, cat5, vad3 = infer_wave(wave, target_emotion)
    cat_dict = {e: float(cat5[i]) for i, e in enumerate(EMOTIONS5)}
    perceived = EMOTIONS5[int(np.argmax(cat5))]
    if target_emotion:
        match = "✅ **KHỚP** target" if perceived == norm_emotion(target_emotion) else "⚠️ **LỆCH** target"
        band = "🟢 tốt" if emos >= 4 else ("🟡 khá" if emos >= 3 else "🔴 yếu")
        verdict = (f"### Kết luận biểu cảm\n"
                   f"- Cảm xúc cảm nhận: **{perceived}** → {match} (`{target_emotion}`)\n"
                   f"- EMOS = **{emos:.2f}/5** → biểu cảm {band}")
    else:
        verdict = (f"### Kết luận biểu cảm\n"
                   f"- Cảm xúc cảm nhận: **{perceived}**\n"
                   f"- *(Chọn cảm xúc target để bật EMOS — độ khớp ý đồ)*")
        emos = None
    return verdict, (round(emos, 3) if emos is not None else None), cat_dict, \
        round(float(vad3[0]), 3), round(float(vad3[1]), 3), round(float(vad3[2]), 3)

def ui_eval(limit):
    return eval_metrics(limit)

INTRO = (
    "# 🎙️ Emotional TTS Evaluator — VoiceMOS 2026 Track 2\n"
    "Bộ chấm **độ biểu cảm cảm xúc** của giọng TTS, chạy bằng model tốt nhất (**exp08**: WavLM fine-tune + "
    "audeering). Offline, không cần API.\n\n"
    "> **5 output dưới đây CHÍNH LÀ định nghĩa \"expressive emotion\" của Track 2** — mỗi cái trả lời một câu hỏi:\n"
    "> **EMOS** = có đúng cảm xúc được yêu cầu không · **CAT** = người nghe cảm nhận cảm xúc nào · "
    "**VAD** = hóa trị / cường độ / chi phối."
)

with gr.Blocks(title="VMC2026 Track 2 — Emotional TTS Evaluator (exp08)") as demo:
    gr.Markdown(INTRO)
    with gr.Tab("🎯 Chấm 1 file TTS"):
        with gr.Row():
            with gr.Column(scale=1):
                a = gr.Audio(type="filepath", label="Audio (giọng TTS)")
                tgt = gr.Dropdown(EMOTIONS5, label="🎯 Cảm xúc target (cho EMOS)")
                btn = gr.Button("Chấm cảm xúc", variant="primary")
            with gr.Column(scale=2):
                verdict = gr.Markdown()
                with gr.Row():
                    emos_o = gr.Number(label="EMOS — khớp cảm xúc target (1–5)", interactive=False)
                cat_o = gr.Label(label="CAT — phân bố cảm xúc cảm nhận (5 lớp)")
                gr.Markdown("**VAD — toạ độ cảm xúc liên tục (1–5):**")
                with gr.Row():
                    val_o = gr.Number(label="Valence (tích cực↑)", interactive=False)
                    aro_o = gr.Number(label="Arousal (kích động↑)", interactive=False)
                    dom_o = gr.Number(label="Dominance (chi phối↑)", interactive=False)
        btn.click(ui_predict, [a, tgt], [verdict, emos_o, cat_o, val_o, aro_o, dom_o])
    with gr.Tab("📊 Độ tin cậy bộ chấm"):
        gr.Markdown("Đo model tái lập nhãn người tốt tới đâu trên **val nội bộ** (10% train.csv, seed 42) — "
                    "**UTT-SRCC** (EMOS/VAD, cao=tốt) + **CAT-err** (thấp=tốt).\n"
                    "⚠️ Dev label ẩn → đây **KHÔNG** phải điểm leaderboard, chỉ để biết bộ chấm đáng tin cỡ nào.")
        lim = gr.Slider(20, 300, value=100, step=20, label="Số mẫu val để chấm (nhiều = chậm)")
        tbl = gr.Dataframe(headers=["Cột", "Model (val nội bộ)", "Mốc exp08"],
                           label="UTT-SRCC / CAT-err", interactive=False)
        gr.Button("Chạy đánh giá", variant="primary").click(ui_eval, [lim], [tbl])

demo.launch(share=True)

# %% [markdown]
# ## Ghi chú
# - Hằng `TRUNK_HIDDEN/HEAD_HIDDEN` PHẢI khớp exp08 (ckpt không lưu) — sai là lệch key/shape.
# - EMOS cần cảm xúc target → chưa chọn dropdown thì chỉ hiện CAT/VAD.
# - exp08 = mean-pool (không Mamba) → demo dùng `masked_mean`.
# - Metric tab chấm trên val nội bộ train.csv (dev ẩn) → con số ~ mốc exp08 nếu trùng tập val.
# - Cần GPU T4 + Internet On (tải WavLM/SAILER/audeering lần đầu).
