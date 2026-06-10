# %% [markdown]
# # VMC2026 Track 3 — Demo Gradio (Speaker + Accent Similarity)
#
# Baseline **ECAPA fine-tuned**. Tải **audio test** + **audio reference** →
# **speaker similarity** + **accent similarity** (thang 1–5).
#
# > ✅ Demo chỉ cần **2 file audio bất kỳ** + checkpoint có sẵn trong repo → KHÔNG cần data VCTK.
#
# ### Cách dùng trên Kaggle
# 1. Settings → **GPU T4 + Internet On**.
# 2. **Run All** → cell cuối in link `*.gradio.live` (sống ~72h).

# %% [markdown]
# ## 1. Cài đặt + clone repo baseline (có sẵn checkpoint)

# %%
# !pip install -q gradio librosa soundfile speechbrain torchaudio
# !git clone -q https://github.com/voicemos-challenge/vmc2026-baselines.git /kaggle/working/vmc2026-baselines

# %% [markdown]
# ## 2. Nạp 2 checkpoint (spk_sim + acc_sim) + hàm dự đoán

# %%
import librosa

DEVICE = "cuda"
T3_REPO = "/kaggle/working/vmc2026-baselines/track3"
CKPT_SPK = f"{T3_REPO}/official-egs/spk_sim_adamw_lr1e-3/model_spk_sim_step20000.pt"
CKPT_ACC = f"{T3_REPO}/official-egs/acc_sim_adamw_lr1e-3/model_acc_sim_step20000.pt"

_M = {}

def _load():
    if "spk" not in _M:
        import sys, torch
        if T3_REPO not in sys.path:
            sys.path.insert(0, T3_REPO)
        from model import Model
        dev = DEVICE if torch.cuda.is_available() else "cpu"
        spk = Model(mlp_heads=["spk_sim"])
        spk.load_state_dict(torch.load(CKPT_SPK, map_location="cpu"))
        acc = Model(mlp_heads=["acc_sim"])
        acc.load_state_dict(torch.load(CKPT_ACC, map_location="cpu"))
        _M.update(spk=spk.to(dev).eval(), acc=acc.to(dev).eval(), dev=dev)
    return _M


def predict(audio_test, audio_ref):
    import torch
    if not audio_test or not audio_ref:
        return "⚠️ Cần cả 2 file: audio test + audio reference."
    M = _load()
    dev = M["dev"]
    ta = torch.from_numpy(librosa.load(audio_test, sr=16000, mono=True)[0]).float().unsqueeze(0).to(dev)
    tb = torch.from_numpy(librosa.load(audio_ref, sr=16000, mono=True)[0]).float().unsqueeze(0).to(dev)
    with torch.no_grad():
        o_spk = M["spk"](ta, tb)
        spk = float(o_spk["spk_sim"].item())
        acc = float(M["acc"](ta, tb)["acc_sim"].item())
        cos = float(o_spk["cos_sim"].item())
    return (f"Speaker similarity: {spk:.3f}   (1–5)\n"
            f"Accent similarity : {acc:.3f}   (1–5)\n"
            f"Cosine zero-shot (tham khảo): {cos:.3f}")

# %% [markdown]
# ## 3. Giao diện Gradio + launch

# %%
import gradio as gr

with gr.Blocks(title="VMC2026 Track 3 — Speaker/Accent Similarity") as demo:
    gr.Markdown("# 🎙️ Track 3 · Speaker / Accent Similarity\n"
                "Tải **audio cần đánh giá** + **audio tham chiếu** → độ giống người nói & accent.")
    at = gr.Audio(type="filepath", label="Audio cần đánh giá (test)")
    ar = gr.Audio(type="filepath", label="Audio tham chiếu (reference)")
    out = gr.Textbox(label="Kết quả", lines=4)
    gr.Button("Dự đoán", variant="primary").click(predict, [at, ar], out)

demo.launch(share=True)
