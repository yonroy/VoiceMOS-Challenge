"""
VoiceMOS Challenge 2026 — Demo Gradio GỘP 3 TRACK (Hugging Face Space).

Khác bản Kaggle (`kaggle_baseline/demo_all_tracks_gradio`): checkpoint Track 2 được tải từ HF Models
repo qua `hf_hub_download` (Space không có /kaggle/input). Track 1 & 3 tự clone lúc chạy.

Lazy-load: mỗi track chỉ nạp model khi bấm nút ở tab đó.
"""
import os, sys, subprocess
import numpy as np
import librosa
import torch

# ── Repo Models chứa checkpoint Track 2 (SỬA username cho khớp) ───────────────
HF_MODEL_REPO = "tranminhtoan140601/voicemos2026-track2-emotion"
T2_CKPT_FILE  = "ft_emotion_full_20epoch.pt"
# ──────────────────────────────────────────────────────────────────────────────

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SR = 16000
print("Device:", DEVICE)

def _scalar(x):
    return float(x.item()) if hasattr(x, "item") else float(x)

# =============================================================================
# PLOTLY HELPERS — biểu đồ trực quan (gauge / radar / bar)
# =============================================================================
import plotly.graph_objects as go
from plotly.subplots import make_subplots

_FONT = dict(family="Inter, system-ui, sans-serif", color="#1f2937")
_INK = "#4f46e5"

def _band_color(v, a, b):
    r = (v - a) / (b - a) if b > a else 0.5
    return "#ef4444" if r < 0.4 else ("#f59e0b" if r < 0.7 else "#16a34a")

def _gauge_trace(v, title, a, b):
    return go.Indicator(
        mode="gauge+number", value=round(float(v), 3),
        title={"text": title, "font": {"size": 13}},
        gauge={"axis": {"range": [a, b]}, "bar": {"color": _band_color(v, a, b)},
               "steps": [{"range": [a, a + (b - a) * 0.4], "color": "#fee2e2"},
                         {"range": [a + (b - a) * 0.4, a + (b - a) * 0.7], "color": "#fef3c7"},
                         {"range": [a + (b - a) * 0.7, b], "color": "#dcfce7"}]})

def gauges_row(items):
    """items: list of (value, title, vmin, vmax) → 1 hàng nhiều gauge."""
    n = len(items)
    fig = make_subplots(rows=1, cols=n, specs=[[{"type": "indicator"}] * n])
    for i, (v, t, a, b) in enumerate(items):
        fig.add_trace(_gauge_trace(v, t, a, b), row=1, col=i + 1)
    fig.update_layout(height=230, margin=dict(l=20, r=20, t=45, b=10),
                      paper_bgcolor="white", font=_FONT)
    return fig

def gauge(value, title, a=1.0, b=5.0):
    fig = go.Figure(_gauge_trace(value, title, a, b))
    fig.update_layout(height=230, margin=dict(l=20, r=20, t=45, b=10),
                      paper_bgcolor="white", font=_FONT)
    return fig

def vad_radar(val, aro, dom):
    fig = go.Figure(go.Scatterpolar(
        r=[val, aro, dom, val], theta=["Valence", "Arousal", "Dominance", "Valence"],
        fill="toself", line={"color": _INK}, fillcolor="rgba(79,70,229,0.18)"))
    fig.update_layout(height=300, margin=dict(l=40, r=40, t=45, b=20), showlegend=False,
                      polar={"radialaxis": {"range": [1, 5], "tickvals": [1, 2, 3, 4, 5]}},
                      paper_bgcolor="white", font=_FONT,
                      title={"text": "VAD — toạ độ cảm xúc (1–5)", "font": {"size": 14}})
    return fig

def cat_bar(cat_dict):
    items = sorted(cat_dict.items(), key=lambda kv: kv[1])
    labels = [k for k, _ in items]
    vals = [v * 100 for _, v in items]
    colors = [_INK if i == len(items) - 1 else "#c7d2fe" for i in range(len(items))]
    fig = go.Figure(go.Bar(x=vals, y=labels, orientation="h", marker_color=colors,
                           text=[f"{v:.0f}%" for v in vals], textposition="outside"))
    fig.update_layout(height=300, margin=dict(l=10, r=30, t=45, b=20),
                      xaxis={"range": [0, 100], "ticksuffix": "%"},
                      paper_bgcolor="white", plot_bgcolor="white", font=_FONT,
                      title={"text": "CAT — phân bố cảm xúc cảm nhận", "font": {"size": 14}})
    return fig

def placeholder_fig(text):
    fig = go.Figure()
    fig.add_annotation(text=text, showarrow=False, x=0.5, y=0.5, xref="paper", yref="paper",
                       font={"size": 14, "color": "#9ca3af"})
    fig.update_layout(height=230, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="white",
                      xaxis={"visible": False}, yaxis={"visible": False})
    return fig

# =============================================================================
# TRACK 1 — URGENT-MOS (ACR + CCR)
# =============================================================================
URGENT_REPO = "/home/user/URGENT-MOS"
URGENT_CKPT = "urgent-challenge/urgent-mos-f1c1m5dcorpus"
_T1 = {}

def _t1_load():
    if "m" in _T1:
        return _T1["m"]
    if not os.path.isdir(URGENT_REPO):
        subprocess.run(f"git clone -q https://github.com/vvwangvv/URGENT-MOS.git {URGENT_REPO}",
                       shell=True, check=True)
    if URGENT_REPO not in sys.path:
        sys.path.insert(0, URGENT_REPO)
    import importlib
    importlib.invalidate_caches()
    try:
        importlib.import_module("urgent_mos.api.infer")
    except Exception:
        subprocess.run(f"pip install -q -e {URGENT_REPO}", shell=True, check=False)
        importlib.invalidate_caches()
    from urgent_mos.utils import load_model_from_checkpoint
    m = load_model_from_checkpoint(URGENT_CKPT, DEVICE)
    m.eval()
    _T1["m"] = m
    return m

def t1_predict(audio_a, audio_b):
    if not audio_a:
        return "⚠️ Hãy tải lên ít nhất **Audio A**.", placeholder_fig("Tải Audio A để xem ACR")
    try:
        m = _t1_load()
        from urgent_mos.api.infer import infer, infer_pairs
        wa = torch.from_numpy(librosa.load(audio_a, sr=SR, mono=True)[0]).float()
        acr_a = max(1.0, min(5.0, _scalar(infer(m, [wa], sample_rate=[SR],
                                                batch_frames=None, num_workers=0)[0]["mos_overall"])))
        if audio_b:
            wb = torch.from_numpy(librosa.load(audio_b, sr=SR, mono=True)[0]).float()
            acr_b = max(1.0, min(5.0, _scalar(infer(m, [wb], sample_rate=[SR],
                                                    batch_frames=None, num_workers=0)[0]["mos_overall"])))
            ccr = max(-3.0, min(3.0, _scalar(infer_pairs(m, [(wa, wb)], sample_rate=[(SR, SR)],
                                                         batch_frames=None, num_workers=0)[0]["mos_overall"])))
            md = (f"<div class='verdict ok'>✅ Đã chấm 2 audio</div>\n\n"
                  f"**ACR A {acr_a:.3f}** · **ACR B {acr_b:.3f}** · **CCR(A↔B) {ccr:+.3f}** — >0 nghĩa là A tốt hơn B.")
            fig = gauges_row([(acr_a, "ACR A (1–5)", 1, 5), (acr_b, "ACR B (1–5)", 1, 5), (ccr, "CCR (−3..+3)", -3, 3)])
        else:
            md = "<div class='verdict ok'>✅ Đã chấm Audio A</div>\n\nThêm **Audio B** để có CCR (so sánh cặp)."
            fig = gauges_row([(acr_a, "ACR A (1–5)", 1, 5)])
        return md, fig
    except Exception as e:
        return f"<div class='verdict err'>❌ Track 1 lỗi</div>\n\n`{repr(e)}`", placeholder_fig("Lỗi — xem log")

# =============================================================================
# TRACK 3 — ECAPA fine-tuned (spk_sim + acc_sim)
# =============================================================================
T3_ROOT = "/home/user/vmc2026-baselines"
T3_REPO = f"{T3_ROOT}/track3"
CKPT_SPK = f"{T3_REPO}/official-egs/spk_sim_adamw_lr1e-3/model_spk_sim_step20000.pt"
CKPT_ACC = f"{T3_REPO}/official-egs/acc_sim_adamw_lr1e-3/model_acc_sim_step20000.pt"
_T3 = {}

def _t3_load():
    if "spk" in _T3:
        return _T3
    if not os.path.isdir(T3_ROOT):
        subprocess.run(f"git clone -q https://github.com/voicemos-challenge/vmc2026-baselines.git {T3_ROOT}",
                       shell=True, check=True)
    if T3_REPO not in sys.path:
        sys.path.insert(0, T3_REPO)
    from model import Model
    spk = Model(mlp_heads=["spk_sim"]); spk.load_state_dict(torch.load(CKPT_SPK, map_location="cpu"))
    acc = Model(mlp_heads=["acc_sim"]); acc.load_state_dict(torch.load(CKPT_ACC, map_location="cpu"))
    _T3.update(spk=spk.to(DEVICE).eval(), acc=acc.to(DEVICE).eval())
    return _T3

def t3_predict(audio_test, audio_ref):
    if not audio_test or not audio_ref:
        return "⚠️ Cần **cả 2 file**: audio test + audio reference.", placeholder_fig("Tải đủ 2 audio")
    try:
        M = _t3_load()
        ta = torch.from_numpy(librosa.load(audio_test, sr=SR, mono=True)[0]).float().unsqueeze(0).to(DEVICE)
        tb = torch.from_numpy(librosa.load(audio_ref, sr=SR, mono=True)[0]).float().unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            o_spk = M["spk"](ta, tb)
            spk = float(o_spk["spk_sim"].item())
            acc = float(M["acc"](ta, tb)["acc_sim"].item())
            cos = float(o_spk["cos_sim"].item())
        md = (f"<div class='verdict ok'>✅ Đã so sánh test ↔ reference</div>\n\n"
              f"**Speaker {spk:.3f}** · **Accent {acc:.3f}** (1–5) · cosine zero-shot {cos:.3f}")
        fig = gauges_row([(spk, "Speaker sim (1–5)", 1, 5), (acc, "Accent sim (1–5)", 1, 5)])
        return md, fig
    except Exception as e:
        return f"<div class='verdict err'>❌ Track 3 lỗi</div>\n\n`{repr(e)}`", placeholder_fig("Lỗi — xem log")

# =============================================================================
# TRACK 2 — exp08 Emotional TTS Evaluator (EMOS/CAT/VAD)
# =============================================================================
EMO_MAX_SEC, TRUNK_HIDDEN, HEAD_HIDDEN, DROPOUT, USE_AMP = 8, 512, 128, 0.3, True
EMOTIONS5 = ["angry", "happy", "neutral", "sad", "surprised"]
_EMO_ALIAS = {
    "angry": "angry", "anger": "angry", "happy": "happy", "happiness": "happy", "joy": "happy",
    "neutral": "neutral", "calm": "neutral", "sad": "sad", "sadness": "sad",
    "surprise": "surprised", "surprised": "surprised", "surprising": "surprised",
}
def norm_emotion(label):
    key = str(label).strip().lower()
    return _EMO_ALIAS.get(key, key if key in EMOTIONS5 else None)

_T2 = {}

def _t2_load():
    if "infer" in _T2:
        return _T2["infer"]
    import torch.nn as nn
    from huggingface_hub import hf_hub_download
    ckpt_path = hf_hub_download(repo_id=HF_MODEL_REPO, filename=T2_CKPT_FILE)

    repo = "/home/user/vox-profile-release"
    if not os.path.exists(repo):
        subprocess.run(["git", "clone", "--depth", "1",
                        "https://github.com/tiantiaf0627/vox-profile-release.git", repo], check=True)
    if repo not in sys.path:
        sys.path.insert(0, repo)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert "wavlm" in ckpt and "heads" in ckpt, "Checkpoint thiếu 'wavlm'/'heads'."
    AUD_DIM = int(ckpt.get("AUD_DIM", 0)); USE_AUDEERING = AUD_DIM > 0

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
        from src.model.emotion.wavlm_emotion import WavLMWrapper
        _wrapper = WavLMWrapper.from_pretrained("tiantiaf/wavlm-large-categorical-emotion")
        _name, wavlm = find_hf_backbone(_wrapper)
    except Exception as e:
        print("⚠️ SAILER wrapper lỗi:", repr(e))
    if wavlm is None:
        from transformers import WavLMModel
        wavlm = WavLMModel.from_pretrained("microsoft/wavlm-large")
    wavlm = wavlm.to(DEVICE).eval()
    WAVLM_DIM = int(wavlm.config.hidden_size)
    wavlm.config.layerdrop = 0.0
    wavlm.load_state_dict(ckpt["wavlm"], strict=False)

    def masked_mean(hidden, attn_mask):
        if attn_mask is None:
            return hidden.mean(dim=1)
        try:
            fm = wavlm._get_feature_vector_attention_mask(hidden.shape[1], attn_mask)
        except Exception:
            return hidden.mean(dim=1)
        fm = fm.unsqueeze(-1).to(hidden.dtype)
        return (hidden * fm).sum(1) / fm.sum(1).clamp(min=1e-6)

    aud_backbone = aud_head = aud_proc = None
    if USE_AUDEERING:
        from transformers import Wav2Vec2Model, Wav2Vec2Config, Wav2Vec2Processor
        from huggingface_hub import hf_hub_download as _dl
        AUD_NAME = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
        aud_proc = Wav2Vec2Processor.from_pretrained(AUD_NAME)
        aud_backbone = Wav2Vec2Model(Wav2Vec2Config.from_pretrained(AUD_NAME))
        try:
            _sd = __import__("safetensors.torch", fromlist=["load_file"]).load_file(_dl(AUD_NAME, "model.safetensors"))
        except Exception:
            _sd = torch.load(_dl(AUD_NAME, "pytorch_model.bin"), map_location="cpu")
        bb_sd = {k[len("wav2vec2."):]: v for k, v in _sd.items() if k.startswith("wav2vec2.")}
        aud_backbone.load_state_dict(bb_sd, strict=False)
        _hid = _sd["classifier.dense.weight"].shape[0]
        aud_head = nn.Sequential(nn.Linear(_hid, _hid), nn.Tanh(),
                                 nn.Linear(_hid, _sd["classifier.out_proj.weight"].shape[0]))
        aud_head[0].weight.data.copy_(_sd["classifier.dense.weight"]); aud_head[0].bias.data.copy_(_sd["classifier.dense.bias"])
        aud_head[2].weight.data.copy_(_sd["classifier.out_proj.weight"]); aud_head[2].bias.data.copy_(_sd["classifier.out_proj.bias"])
        aud_backbone = aud_backbone.to(DEVICE).eval(); aud_head = aud_head.to(DEVICE).eval()

    @torch.no_grad()
    def audeering_feat(wave):
        x = aud_proc(wave, sampling_rate=SR).input_values[0]
        x = torch.from_numpy(np.asarray(x, dtype=np.float32)).unsqueeze(0).to(DEVICE)
        h = aud_backbone(x)[0].mean(dim=1)
        out = aud_head(h)[0].cpu().numpy()
        vad = np.array([1 + 4 * out[2], 1 + 4 * out[0], 1 + 4 * out[1]], dtype=np.float32)
        return np.concatenate([h[0].cpu().numpy(), vad]).astype(np.float32)

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

    heads = EmoHeads(TRUNK_IN, TRUNK_HIDDEN, HEAD_HIDDEN, DROPOUT, N_EMO).to(DEVICE).eval()
    heads.load_state_dict(ckpt["heads"], strict=False)
    emos_mu, emos_sd = float(ckpt["emos_mu"]), float(ckpt["emos_sd"])
    vad_mu = np.asarray(ckpt["vad_mu"], dtype=np.float32); vad_sd = np.asarray(ckpt["vad_sd"], dtype=np.float32)

    def onehot_target(tgt):
        v = np.zeros(N_EMO, dtype=np.float32)
        if tgt in EMOTIONS5:
            v[EMOTIONS5.index(tgt)] = 1.0
        return v

    @torch.no_grad()
    def infer_wave(wave, target_emotion):
        wave = wave[: EMO_MAX_SEC * SR].astype(np.float32)
        iv = torch.from_numpy(wave).unsqueeze(0).to(DEVICE)
        am = torch.ones((1, len(wave)), dtype=torch.long, device=DEVICE)
        tgt = torch.from_numpy(onehot_target(norm_emotion(target_emotion) if target_emotion else None)).unsqueeze(0).to(DEVICE)
        with torch.cuda.amp.autocast(enabled=USE_AMP and DEVICE == "cuda"):
            fw = masked_mean(wavlm(iv, attention_mask=am).last_hidden_state, am)
            if USE_AUDEERING:
                fw = torch.cat([fw, torch.from_numpy(audeering_feat(wave)).unsqueeze(0).to(DEVICE)], dim=1)
            emos_p, cat_l, vad_p = heads(fw, tgt)
        emos = float(emos_p.item()) * emos_sd + emos_mu
        cat5 = torch.softmax(cat_l, 1)[0].float().cpu().numpy()
        vad3 = vad_p[0].float().cpu().numpy() * vad_sd + vad_mu
        return emos, cat5, vad3

    print("✅ Track 2 exp08 nạp xong (audeering", "ON)" if USE_AUDEERING else "OFF)")
    _T2["infer"] = infer_wave
    return infer_wave

def t2_predict(audio, target_emotion):
    if not audio:
        return ("⚠️ Hãy tải audio (giọng TTS).", placeholder_fig("Tải audio để chấm EMOS"),
                placeholder_fig("CAT"), placeholder_fig("VAD"))
    try:
        infer_wave = _t2_load()
        wave, _ = librosa.load(audio, sr=SR, mono=True)
        emos, cat5, vad3 = infer_wave(wave, target_emotion)
        cat_dict = {e: float(cat5[i]) for i, e in enumerate(EMOTIONS5)}
        perceived = EMOTIONS5[int(np.argmax(cat5))]
        cat_fig = cat_bar(cat_dict)
        vad_fig = vad_radar(float(vad3[0]), float(vad3[1]), float(vad3[2]))
        if target_emotion:
            ok = perceived == norm_emotion(target_emotion)
            md = (f"<div class='verdict {'ok' if ok else 'warn'}'>"
                  f"{'✅ KHỚP target' if ok else '⚠️ LỆCH target'}</div>\n\n"
                  f"Cảm nhận: **{perceived}** · target `{target_emotion}` · EMOS **{emos:.2f}/5**")
            emos_fig = gauge(emos, "EMOS — khớp target (1–5)", 1, 5)
        else:
            md = (f"<div class='verdict neutral'>Cảm nhận: {perceived}</div>\n\n"
                  f"*(Chọn cảm xúc target để bật điểm EMOS)*")
            emos_fig = placeholder_fig("Chọn cảm xúc target → EMOS")
        return md, emos_fig, cat_fig, vad_fig
    except Exception as e:
        ph = placeholder_fig("Lỗi — xem log")
        return f"<div class='verdict err'>❌ Track 2 lỗi</div>\n\n`{repr(e)}`", ph, ph, ph

# =============================================================================
# Giao diện Gradio GỘP — 3 tab (clean light + Plotly)
# =============================================================================
import gradio as gr

THEME = gr.themes.Soft(primary_hue="indigo", neutral_hue="slate",
                       font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"])

CUSTOM_CSS = """
footer{display:none !important}
.gradio-container{max-width:1080px !important; margin:auto}
#hero{padding:20px 24px; border-bottom:3px solid #4f46e5; margin-bottom:6px}
#hero h1{margin:0; font-size:1.55rem; color:#111827}
#hero p{margin:6px 0 0; color:#6b7280; font-size:.95rem}
.verdict{display:inline-block; padding:6px 14px; border-radius:999px; font-weight:600; font-size:.95rem}
.verdict.ok{background:#dcfce7; color:#166534}
.verdict.warn{background:#fef3c7; color:#92400e}
.verdict.err{background:#fee2e2; color:#991b1b}
.verdict.neutral{background:#eef2ff; color:#3730a3}
#foot{color:#9ca3af; font-size:.85rem; text-align:center; margin:16px 0 4px}
#foot a{color:#6366f1; text-decoration:none}
"""

HERO = (
    "<div id='hero'><h1>🎙️ VoiceMOS Challenge 2026 — Bộ chấm 3 Track</h1>"
    "<p>Đánh giá tự động: <b>chất lượng giọng</b> (T1) · <b>cảm xúc TTS</b> (T2) · "
    "<b>giống người nói / accent</b> (T3). Lần đầu mỗi tab chờ tải model (CPU free → chậm).</p></div>"
)

with gr.Blocks(title="VMC2026 — Demo 3 Track", theme=THEME, css=CUSTOM_CSS) as demo:
    gr.HTML(HERO)

    with gr.Tab("1️⃣ Chất lượng (ACR/CCR)"):
        with gr.Row():
            with gr.Column(scale=1):
                t1a = gr.Audio(type="filepath", label="Audio A (bắt buộc)")
                t1b = gr.Audio(type="filepath", label="Audio B (tùy chọn → CCR)")
                t1btn = gr.Button("Chấm chất lượng", variant="primary")
            with gr.Column(scale=2):
                t1out = gr.Markdown()
                t1plot = gr.Plot(label="")
        t1btn.click(t1_predict, [t1a, t1b], [t1out, t1plot])

    with gr.Tab("2️⃣ Cảm xúc (EMOS/CAT/VAD)"):
        with gr.Row():
            with gr.Column(scale=1):
                t2a = gr.Audio(type="filepath", label="Audio (giọng TTS)")
                t2tgt = gr.Dropdown(EMOTIONS5, label="🎯 Cảm xúc target (cho EMOS)")
                t2btn = gr.Button("Chấm cảm xúc", variant="primary")
            with gr.Column(scale=2):
                t2verdict = gr.Markdown()
                t2emos = gr.Plot(label="")
                with gr.Row():
                    t2cat = gr.Plot(label="")
                    t2vad = gr.Plot(label="")
        t2btn.click(t2_predict, [t2a, t2tgt], [t2verdict, t2emos, t2cat, t2vad])

    with gr.Tab("3️⃣ Speaker / Accent"):
        with gr.Row():
            with gr.Column(scale=1):
                t3t = gr.Audio(type="filepath", label="Audio cần đánh giá (test)")
                t3r = gr.Audio(type="filepath", label="Audio tham chiếu (reference)")
                t3btn = gr.Button("So sánh", variant="primary")
            with gr.Column(scale=2):
                t3out = gr.Markdown()
                t3plot = gr.Plot(label="")
        t3btn.click(t3_predict, [t3t, t3r], [t3out, t3plot])

    gr.HTML(
        "<div id='foot'>VoiceMOS Challenge 2026 · CC BY-NC-SA 4.0 · "
        "<a href='https://huggingface.co/tranminhtoan140601/voicemos2026-track2-emotion'>checkpoint</a> · "
        "<a href='https://huggingface.co/tranminhtoan140601/voicemos2026-code'>code</a></div>"
    )

if __name__ == "__main__":
    # Tự nhận môi trường: HF Space (SPACE_ID có) → bind 7860, KHÔNG share; Kaggle/local → share=True (link gradio.live).
    _on_spaces = os.environ.get("SPACE_ID") is not None
    demo.launch(share=not _on_spaces, server_name="0.0.0.0", server_port=7860)
