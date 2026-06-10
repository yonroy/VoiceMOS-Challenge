# %% [markdown]
# # VMC2026 Track 1 — Demo Gradio (Speech Enhancement: ACR + CCR)
#
# Baseline **URGENT-MOS**. Tải **Audio A** → **ACR** (chất lượng 1–5).
# Tải thêm **Audio B** → **CCR** (so sánh A vs B, thang −3..+3, >0 nghĩa là A tốt hơn).
#
# ### Cách dùng trên Kaggle
# 1. Settings → **GPU T4 + Internet On**.
# 2. **Run All** → cell cuối in link `*.gradio.live` (sống ~72h) → gửi mentor.

# %% [markdown]
# ## 1. Cài đặt + clone URGENT-MOS

# %%
# !pip install -q gradio librosa soundfile
# !git clone -q https://github.com/vvwangvv/URGENT-MOS.git /kaggle/working/URGENT-MOS
# !pip install -q -e /kaggle/working/URGENT-MOS

# %% [markdown]
# ## 2. Nạp model + hàm dự đoán

# %%
import os, sys, subprocess, librosa

DEVICE = "cuda"
URGENT_REPO = "/kaggle/working/URGENT-MOS"
URGENT_CKPT = "urgent-challenge/urgent-mos-f1c1m5dcorpus"   # tự tải từ HuggingFace


def _ensure_urgent_mos():
    """Đảm bảo import được `urgent_mos`: clone repo nếu thiếu, cài deps, thêm vào sys.path."""
    if not os.path.isdir(URGENT_REPO):
        subprocess.run(f"git clone -q https://github.com/vvwangvv/URGENT-MOS.git {URGENT_REPO}",
                       shell=True, check=True)
    # package nằm ở ROOT repo → thêm vào path là import được, KHÔNG phụ thuộc pip install thành công
    if URGENT_REPO not in sys.path:
        sys.path.insert(0, URGENT_REPO)
    import importlib
    importlib.invalidate_caches()
    # thử import; nếu thiếu dependency thì cài editable (kéo theo torchcodec, hydra-core, omegaconf...)
    try:
        importlib.import_module("urgent_mos.api.infer")
    except Exception:
        subprocess.run(f"pip install -q -e {URGENT_REPO}", shell=True)
        importlib.invalidate_caches()


_M = {}

def _load():
    if "m" not in _M:
        _ensure_urgent_mos()
        import torch
        from urgent_mos.utils import load_model_from_checkpoint
        dev = DEVICE if torch.cuda.is_available() else "cpu"
        m = load_model_from_checkpoint(URGENT_CKPT, dev)
        m.eval()
        _M["m"] = m
    return _M["m"]


def _scalar(x):
    return float(x.item()) if hasattr(x, "item") else float(x)


def predict(audio_a, audio_b):
    if not audio_a:
        return "⚠️ Hãy tải lên ít nhất Audio A."
    import torch
    m = _load()                                          # _load() tự ensure repo + sys.path TRƯỚC
    from urgent_mos.api.infer import infer, infer_pairs  # giờ mới import được
    wa = torch.from_numpy(librosa.load(audio_a, sr=16000, mono=True)[0]).float()
    acr_a = max(1.0, min(5.0, _scalar(infer(m, [wa], sample_rate=[16000],
                                            batch_frames=None, num_workers=0)[0]["mos_overall"])))
    out = f"ACR (Audio A): {acr_a:.3f}   (chất lượng tuyệt đối, thang 1–5)"
    if audio_b:
        wb = torch.from_numpy(librosa.load(audio_b, sr=16000, mono=True)[0]).float()
        acr_b = max(1.0, min(5.0, _scalar(infer(m, [wb], sample_rate=[16000],
                                                batch_frames=None, num_workers=0)[0]["mos_overall"])))
        ccr = max(-3.0, min(3.0, _scalar(infer_pairs(m, [(wa, wb)], sample_rate=[(16000, 16000)],
                                                     batch_frames=None, num_workers=0)[0]["mos_overall"])))
        out += (f"\nACR (Audio B): {acr_b:.3f}"
                f"\nCCR (A so với B): {ccr:+.3f}   (>0: A tốt hơn B; thang −3..+3)")
    return out

# %% [markdown]
# ## 3. Giao diện Gradio + launch

# %%
import gradio as gr

with gr.Blocks(title="VMC2026 Track 1 — ACR/CCR") as demo:
    gr.Markdown("# 🎙️ Track 1 · Speech Enhancement (ACR / CCR)\n"
                "Tải **Audio A** để có ACR. Tải thêm **Audio B** để so sánh CCR (A vs B).")
    a = gr.Audio(type="filepath", label="Audio A (bắt buộc)")
    b = gr.Audio(type="filepath", label="Audio B (tùy chọn — để tính CCR)")
    out = gr.Textbox(label="Kết quả", lines=4)
    gr.Button("Dự đoán", variant="primary").click(predict, [a, b], out)

demo.launch(share=True)
