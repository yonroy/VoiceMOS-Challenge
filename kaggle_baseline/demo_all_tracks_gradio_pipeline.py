# %% [markdown]
# # VMC2026 — Demo Gradio GỘP 3 TRACK (1 link cho mentor)
#
# Gộp 3 demo lẻ (`track1/`, `track2/`, `track3/`) vào **1 app Gradio 3 tab**:
# - **Track 1** · Speech Enhancement → **ACR** (chất lượng A) + **CCR** (so A vs B). Model: URGENT-MOS.
# - **Track 2** · Emotional TTS → **EMOS / CAT / VAD**. Model TỐT NHẤT = **exp08** (WavLM fine-tune + audeering).
# - **Track 3** · Speaker/Accent → **spk_sim / acc_sim**. Model: ECAPA fine-tuned (baseline BTC).
#
# > **Lazy-load:** mỗi track chỉ nạp model khi bạn bấm "Dự đoán" ở tab đó → tab nào thiếu checkpoint/repo
# > chỉ báo lỗi trong tab đó, KHÔNG sập cả app. Track 1 & 3 chỉ cần Internet; Track 2 cần thêm checkpoint exp08.
#
# ### Cách chạy trên Kaggle
# 1. Settings → **GPU T4 + Internet On**.
# 2. (Cho Track 2) Add Input: dataset Track 2 (`sets/train.csv`, `wav/`, `metadata.csv`) + dataset chứa
#    `ft_emotion_full_20epoch.pt` (slug `toanminh222/cache-exp8`). Thiếu thì 2 tab kia vẫn chạy.
# 3. **Run All** → cell cuối in link `*.gradio.live` (sống ~72h) → gửi mentor.

# %% [markdown]
# ## 1. Cài đặt gói (1 lần cho cả 3 track)

# %%
# !pip install -q gradio librosa soundfile speechbrain torchaudio loralib scipy scikit-learn pandas tqdm

import os, sys, glob, subprocess

def pip_install(*pkgs):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], check=False)

# Cài nhẹ (Kaggle có sẵn torch/transformers/numpy → KHÔNG đụng numpy để tránh lệch ABI)
pip_install("gradio", "librosa", "soundfile", "speechbrain", "torchaudio",
            "loralib", "scipy", "scikit-learn", "pandas", "tqdm")

import librosa
import numpy as np
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SR = 16000
print("Device:", DEVICE, ("✅ " + torch.cuda.get_device_name(0)) if DEVICE == "cuda" else "⚠️ CPU (chậm)")

def _stem(p):
    return os.path.splitext(os.path.basename(str(p)))[0]

def _scalar(x):
    return float(x.item()) if hasattr(x, "item") else float(x)

# %% [markdown]
# ## 2. TRACK 1 — URGENT-MOS (ACR + CCR) · lazy-load

# %%
URGENT_REPO = "/kaggle/working/URGENT-MOS"
URGENT_CKPT = "urgent-challenge/urgent-mos-f1c1m5dcorpus"   # tự tải từ HuggingFace
_T1 = {}

def _t1_load():
    """Nạp URGENT-MOS 1 lần (clone repo + sys.path + checkpoint)."""
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
        return "⚠️ Hãy tải lên ít nhất **Audio A**."
    try:
        m = _t1_load()
        from urgent_mos.api.infer import infer, infer_pairs
        wa = torch.from_numpy(librosa.load(audio_a, sr=SR, mono=True)[0]).float()
        acr_a = max(1.0, min(5.0, _scalar(infer(m, [wa], sample_rate=[SR],
                                                batch_frames=None, num_workers=0)[0]["mos_overall"])))
        out = f"**ACR (Audio A): {acr_a:.3f}**  (chất lượng tuyệt đối, thang 1–5)"
        if audio_b:
            wb = torch.from_numpy(librosa.load(audio_b, sr=SR, mono=True)[0]).float()
            acr_b = max(1.0, min(5.0, _scalar(infer(m, [wb], sample_rate=[SR],
                                                    batch_frames=None, num_workers=0)[0]["mos_overall"])))
            ccr = max(-3.0, min(3.0, _scalar(infer_pairs(m, [(wa, wb)], sample_rate=[(SR, SR)],
                                                         batch_frames=None, num_workers=0)[0]["mos_overall"])))
            out += (f"\n\n**ACR (Audio B): {acr_b:.3f}**"
                    f"\n\n**CCR (A so với B): {ccr:+.3f}**  (>0: A tốt hơn B; thang −3..+3)")
        return out
    except Exception as e:
        return f"❌ Track 1 lỗi: `{repr(e)}`\n\nKiểm tra **Internet On** (cần tải URGENT-MOS từ GitHub/HuggingFace)."

# %% [markdown]
# ## 3. TRACK 3 — ECAPA fine-tuned (spk_sim + acc_sim) · lazy-load

# %%
T3_REPO = "/kaggle/working/vmc2026-baselines/track3"
CKPT_SPK = f"{T3_REPO}/official-egs/spk_sim_adamw_lr1e-3/model_spk_sim_step20000.pt"
CKPT_ACC = f"{T3_REPO}/official-egs/acc_sim_adamw_lr1e-3/model_acc_sim_step20000.pt"
_T3 = {}

def _t3_load():
    if "spk" in _T3:
        return _T3
    repo_root = "/kaggle/working/vmc2026-baselines"
    if not os.path.isdir(repo_root):
        subprocess.run(f"git clone -q https://github.com/voicemos-challenge/vmc2026-baselines.git {repo_root}",
                       shell=True, check=True)
    if T3_REPO not in sys.path:
        sys.path.insert(0, T3_REPO)
    from model import Model
    spk = Model(mlp_heads=["spk_sim"])
    spk.load_state_dict(torch.load(CKPT_SPK, map_location="cpu"))
    acc = Model(mlp_heads=["acc_sim"])
    acc.load_state_dict(torch.load(CKPT_ACC, map_location="cpu"))
    _T3.update(spk=spk.to(DEVICE).eval(), acc=acc.to(DEVICE).eval())
    return _T3

def t3_predict(audio_test, audio_ref):
    if not audio_test or not audio_ref:
        return "⚠️ Cần **cả 2 file**: audio test + audio reference."
    try:
        M = _t3_load()
        ta = torch.from_numpy(librosa.load(audio_test, sr=SR, mono=True)[0]).float().unsqueeze(0).to(DEVICE)
        tb = torch.from_numpy(librosa.load(audio_ref, sr=SR, mono=True)[0]).float().unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            o_spk = M["spk"](ta, tb)
            spk = float(o_spk["spk_sim"].item())
            acc = float(M["acc"](ta, tb)["acc_sim"].item())
            cos = float(o_spk["cos_sim"].item())
        return (f"**Speaker similarity: {spk:.3f}**  (1–5)\n\n"
                f"**Accent similarity : {acc:.3f}**  (1–5)\n\n"
                f"Cosine zero-shot (tham khảo): {cos:.3f}")
    except Exception as e:
        return f"❌ Track 3 lỗi: `{repr(e)}`\n\nKiểm tra **Internet On** (clone repo baseline chứa checkpoint)."

# %% [markdown]
# ## 4. TRACK 2 — exp08 Emotional TTS Evaluator (EMOS/CAT/VAD) · lazy-load
#
# Model TỐT NHẤT: WavLM fine-tune (warm-start SAILER) + audeering frozen → trunk → 3 head.
# Hằng kiến trúc PHẢI khớp exp08 (ckpt không lưu các số này).

# %%
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

def _t2_find_ckpt():
    for pat in ["ft_emotion_full_20epoch*.pt", "ft_emotion_full*.pt"]:
        for base in ["/kaggle/input", "/kaggle/working"]:
            hits = sorted(glob.glob(os.path.join(base, "**", pat), recursive=True))
            if hits:
                return hits[0]
    return ""

_T2 = {}

def _t2_load():
    if "infer" in _T2:
        return _T2["infer"]
    import torch.nn as nn
    ckpt_path = _t2_find_ckpt()
    assert ckpt_path, "Không thấy ft_emotion_full*.pt — Add Input dataset checkpoint exp08 (slug toanminh222/cache-exp8)?"
    # code SAILER để dựng backbone WavLM
    repo = "/kaggle/working/vox-profile-release"
    if not os.path.exists(repo):
        subprocess.run(["git", "clone", "--depth", "1",
                        "https://github.com/tiantiaf0627/vox-profile-release.git", repo], check=True)
    if repo not in sys.path:
        sys.path.insert(0, repo)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert "wavlm" in ckpt and "heads" in ckpt, "Checkpoint thiếu 'wavlm'/'heads' → cần bản đủ ft_emotion_full_20epoch.pt."
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
        print("⚠️ SAILER wrapper lỗi:", repr(e), "→ fallback WavLM trắng.")
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

    # audeering frozen (nếu ckpt dùng)
    aud_backbone = aud_head = aud_proc = None
    if USE_AUDEERING:
        from transformers import Wav2Vec2Model, Wav2Vec2Config, Wav2Vec2Processor
        from huggingface_hub import hf_hub_download
        AUD_NAME = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
        aud_proc = Wav2Vec2Processor.from_pretrained(AUD_NAME)
        aud_backbone = Wav2Vec2Model(Wav2Vec2Config.from_pretrained(AUD_NAME))
        try:
            _sd = __import__("safetensors.torch", fromlist=["load_file"]).load_file(
                hf_hub_download(AUD_NAME, "model.safetensors"))
        except Exception:
            _sd = torch.load(hf_hub_download(AUD_NAME, "pytorch_model.bin"), map_location="cpu")
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
            fw = wavlm(iv, attention_mask=am).last_hidden_state
            fw = masked_mean(fw, am)
            if USE_AUDEERING:
                fw = torch.cat([fw, torch.from_numpy(audeering_feat(wave)).unsqueeze(0).to(DEVICE)], dim=1)
            emos_p, cat_l, vad_p = heads(fw, tgt)
        emos = float(emos_p.item()) * emos_sd + emos_mu
        cat5 = torch.softmax(cat_l, 1)[0].float().cpu().numpy()
        vad3 = vad_p[0].float().cpu().numpy() * vad_sd + vad_mu
        return emos, cat5, vad3

    print(f"✅ Track 2 exp08 nạp xong (audeering {'ON' if USE_AUDEERING else 'OFF'}) từ {ckpt_path}")
    _T2["infer"] = infer_wave
    return infer_wave

def t2_predict(audio, target_emotion):
    """Trả: verdict(md), EMOS(number), CAT(label dict), VAL, ARO, DOM."""
    if not audio:
        return "### ⚠️ Hãy tải audio (giọng TTS).", None, {}, None, None, None
    try:
        infer_wave = _t2_load()
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
            verdict = (f"### Kết luận biểu cảm\n- Cảm xúc cảm nhận: **{perceived}**\n"
                       f"- *(Chọn cảm xúc target để bật EMOS — độ khớp ý đồ)*")
            emos = None
        return verdict, (round(emos, 3) if emos is not None else None), cat_dict, \
            round(float(vad3[0]), 3), round(float(vad3[1]), 3), round(float(vad3[2]), 3)
    except Exception as e:
        return f"### ❌ Track 2 lỗi\n`{repr(e)}`\n\nĐã Add Input checkpoint exp08 + Internet On chưa?", None, {}, None, None, None

# %% [markdown]
# ## 5. Giao diện Gradio GỘP — 3 tab + launch

# %%
import gradio as gr

INTRO = (
    "# 🎙️ VoiceMOS Challenge 2026 — Demo 3 Track\n"
    "Một link cho cả 3 track. Mỗi tab nhận audio → trả điểm bộ chấm tự động.\n\n"
    "| Track | Bài toán | Output |\n|---|---|---|\n"
    "| **1** | Speech Enhancement | ACR (chất lượng) · CCR (so sánh cặp) |\n"
    "| **2** | Emotional TTS | EMOS · CAT · VAD (5 cột cảm xúc) — *model tốt nhất exp08* |\n"
    "| **3** | Speaker/Accent | spk_sim · acc_sim |\n\n"
    "> Model nạp **lần đầu bấm nút** (chờ ~1–2 phút tải). Tab thiếu checkpoint chỉ báo lỗi trong tab đó."
)

with gr.Blocks(title="VMC2026 — Demo 3 Track") as demo:
    gr.Markdown(INTRO)

    with gr.Tab("1️⃣ Track 1 · Chất lượng (ACR/CCR)"):
        gr.Markdown("Tải **Audio A** → ACR (1–5). Tải thêm **Audio B** → CCR (A vs B, −3..+3, >0 = A tốt hơn).")
        t1a = gr.Audio(type="filepath", label="Audio A (bắt buộc)")
        t1b = gr.Audio(type="filepath", label="Audio B (tùy chọn — để tính CCR)")
        t1out = gr.Markdown()
        gr.Button("Dự đoán", variant="primary").click(t1_predict, [t1a, t1b], t1out)

    with gr.Tab("2️⃣ Track 2 · Cảm xúc (EMOS/CAT/VAD)"):
        gr.Markdown("Model tốt nhất **exp08** (WavLM fine-tune + audeering, offline). "
                    "Chọn **cảm xúc target** để bật EMOS (độ khớp ý đồ).")
        with gr.Row():
            with gr.Column(scale=1):
                t2a = gr.Audio(type="filepath", label="Audio (giọng TTS)")
                t2tgt = gr.Dropdown(EMOTIONS5, label="🎯 Cảm xúc target (cho EMOS)")
                t2btn = gr.Button("Chấm cảm xúc", variant="primary")
            with gr.Column(scale=2):
                t2verdict = gr.Markdown()
                t2emos = gr.Number(label="EMOS — khớp cảm xúc target (1–5)", interactive=False)
                t2cat = gr.Label(label="CAT — phân bố cảm xúc cảm nhận (5 lớp)")
                gr.Markdown("**VAD — toạ độ cảm xúc liên tục (1–5):**")
                with gr.Row():
                    t2val = gr.Number(label="Valence (tích cực↑)", interactive=False)
                    t2aro = gr.Number(label="Arousal (kích động↑)", interactive=False)
                    t2dom = gr.Number(label="Dominance (chi phối↑)", interactive=False)
        t2btn.click(t2_predict, [t2a, t2tgt], [t2verdict, t2emos, t2cat, t2val, t2aro, t2dom])

    with gr.Tab("3️⃣ Track 3 · Speaker/Accent"):
        gr.Markdown("Tải **audio cần đánh giá** + **audio tham chiếu** → độ giống người nói & accent (1–5).")
        t3t = gr.Audio(type="filepath", label="Audio cần đánh giá (test)")
        t3r = gr.Audio(type="filepath", label="Audio tham chiếu (reference)")
        t3out = gr.Markdown()
        gr.Button("Dự đoán", variant="primary").click(t3_predict, [t3t, t3r], t3out)

demo.launch(share=True)

# %% [markdown]
# ## Ghi chú
# - **Lazy-load:** mỗi `_tN_load()` nạp model 1 lần rồi cache module-level → tab nào không bấm thì không tốn RAM/VRAM.
# - Track 1 cần URGENT-MOS (GitHub + HuggingFace); Track 3 clone repo baseline (có sẵn checkpoint); Track 2 cần
#   checkpoint exp08 (`ft_emotion_full_20epoch.pt`, slug `toanminh222/cache-exp8`) + tải WavLM/SAILER/audeering.
# - Hằng `TRUNK_HIDDEN/HEAD_HIDDEN/EMO_MAX_SEC` của Track 2 PHẢI khớp exp08 (ckpt không lưu) — sai là lệch shape.
# - 3 tab độc lập: thiếu checkpoint/Internet của 1 track chỉ báo lỗi trong tab đó, 2 tab còn lại vẫn chạy.
# - Cần **GPU T4 + Internet On**. Bản chỉ Track 2 đầy đủ (có tab metric val nội bộ) ở `track2/demo_track2_emotion_gradio`.
