# %% [markdown]
# # VMC2026 Track 2 — Demo Gradio (Emotional TTS: QMOS / CAT / EMOS / VAD)
#
# - **QMOS** (UTMOS) + **CAT** (emotion2vec, 5 cảm xúc): chạy ngay, chỉ cần audio.
# - **EMOS / VAD** (Gemini): tùy chọn — cần dán `GEMINI_API_KEY` + chọn cảm xúc target.
#
# ### Cách dùng trên Kaggle
# 1. Settings → **GPU T4 + Internet On**.
# 2. **Run All** → cell cuối in link `*.gradio.live` (sống ~72h).

# %% [markdown]
# ## 1. Cài đặt

# %%
# !pip install -q gradio speechmos funasr librosa soundfile google-genai

# %% [markdown]
# ## 2. Nạp model + hàm dự đoán

# %%
import re, json, librosa

GEMINI_MODEL = "gemini-2.0-flash"
EMOTIONS5 = ["angry", "happy", "neutral", "sad", "surprised"]

_M = {}

def _qmos():
    if "qmos" not in _M:
        import torch
        _M["qmos"] = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True)
    return _M["qmos"]


def _emocat():
    if "emocat" not in _M:
        from funasr import AutoModel
        _M["emocat"] = AutoModel(model="iic/emotion2vec_plus_large", hub="hf")
    return _M["emocat"]


def _gemini_emos_vad(audio_path, target_emotion, api_key):
    """EMOS (1-5 độ khớp cảm xúc target) + VAD (val/aro/dom 1-5) — bản demo gọn qua Gemini."""
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=api_key)
    part = types.Part.from_bytes(data=open(audio_path, "rb").read(), mime_type="audio/wav")
    cfg = types.GenerateContentConfig(temperature=0.0)

    p_emos = (f"The target emotion is '{target_emotion}'. On a scale of 1 to 5, how well does the "
              f"speaker express that emotion? 5=perfect match, 1=no match. Answer with ONLY one integer 1-5.")
    r = client.models.generate_content(model=GEMINI_MODEL, config=cfg, contents=[p_emos, part])
    mm = re.search(r"[1-5]", getattr(r, "text", "") or "")
    emos = int(mm.group()) if mm else None

    p_vad = ('Rate this speech on three 1-5 scales: Valence (1=very negative,5=very positive), '
             'Arousal (1=very calm,5=very excited), Dominance (1=very submissive,5=very dominant). '
             'Answer ONLY as JSON: {"val":x,"aro":y,"dom":z}.')
    r2 = client.models.generate_content(model=GEMINI_MODEL, config=cfg, contents=[p_vad, part])
    val = aro = dom = None
    try:
        d = json.loads(re.search(r"\{.*\}", getattr(r2, "text", "") or "", re.S).group())
        val, aro, dom = d.get("val"), d.get("aro"), d.get("dom")
    except Exception:
        pass
    return emos, (val, aro, dom)


def predict(audio, target_emotion, gemini_key):
    import torch
    if not audio:
        return "⚠️ Hãy tải audio.", {}
    wav = librosa.load(audio, sr=16000, mono=True)[0]
    # QMOS
    qmos = float(_qmos()(torch.from_numpy(wav).unsqueeze(0), sr=16000).mean().item())
    # CAT
    rec = _emocat().generate(audio, granularity="utterance", extract_embedding=False)
    probs = {e: 0.0 for e in EMOTIONS5}
    for lab, sc in zip(rec[0]["labels"], rec[0]["scores"]):
        name = lab.split("/")[-1]
        if name in probs:
            probs[name] = float(sc)
    tot = sum(probs.values())
    if tot > 0:
        probs = {k: v / tot for k, v in probs.items()}

    lines = [f"QMOS (chất lượng giọng, 1–5): {qmos:.3f}"]
    if gemini_key and target_emotion:
        try:
            emos, (val, aro, dom) = _gemini_emos_vad(audio, target_emotion, gemini_key)
            lines.append(f"EMOS (độ khớp cảm xúc '{target_emotion}', 1–5): {emos}")
            lines.append(f"VAD — Valence: {val} · Arousal: {aro} · Dominance: {dom}")
        except Exception as e:
            lines.append(f"(EMOS/VAD lỗi: {e})")
    else:
        lines.append("(EMOS/VAD: dán GEMINI_API_KEY + chọn cảm xúc target để bật)")
    return "\n".join(lines), probs

# %% [markdown]
# ## 3. Giao diện Gradio + launch

# %%
import gradio as gr

with gr.Blocks(title="VMC2026 Track 2 — Emotional TTS") as demo:
    gr.Markdown("# 🎙️ Track 2 · Emotional TTS (QMOS / CAT / EMOS / VAD)\n"
                "QMOS + phân bố cảm xúc (CAT) chạy ngay. EMOS/VAD cần Gemini key + cảm xúc target.")
    a = gr.Audio(type="filepath", label="Audio")
    with gr.Row():
        tgt = gr.Dropdown(EMOTIONS5, label="Cảm xúc target (cho EMOS, tùy chọn)")
        key = gr.Textbox(label="GEMINI_API_KEY (tùy chọn)", type="password")
    out = gr.Textbox(label="Kết quả số", lines=5)
    lbl = gr.Label(label="CAT — phân bố cảm xúc cảm nhận")
    gr.Button("Dự đoán", variant="primary").click(predict, [a, tgt, key], [out, lbl])

demo.launch(share=True)

# %% [markdown]
# ## Ghi chú
# - EMOS/VAD là bản demo gọn (prompt rút gọn) — KHÔNG hoàn toàn giống script baseline gốc, chỉ minh họa.
