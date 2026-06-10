# %% [markdown]
# # VMC2026 — Chạy demo Gradio trên KAGGLE bằng cách KÉO code UI từ Hugging Face
#
# Chiến lược: **HF = nơi chứa code UI** (Space `tranminhtoan140601/voicemos2026-demo`),
# **Kaggle = nơi chạy** (GPU T4 free). Notebook này tải `app.py` từ Space về rồi chạy →
# ra link `*.gradio.live` (sống ~72h) để gửi mentor. KHÔNG tốn GPU trả phí của HF.
#
# `app.py` tự nhận môi trường: trên Kaggle → `share=True` (link công khai); checkpoint Track 2
# tự tải từ HF Models repo `tranminhtoan140601/voicemos2026-track2-emotion`.
#
# ### Cách chạy
# 1. Settings → **GPU T4 + Internet On**.
# 2. **Run All** → cell cuối in link `*.gradio.live`.

# %% [markdown]
# ## 1. Cài deps (khớp Space) — KHÔNG đụng numpy/torch có sẵn Kaggle

# %%
# !pip install -q gradio==6.17.3 huggingface_hub librosa soundfile speechbrain loralib scipy scikit-learn pandas

import subprocess, sys

def pip_install(*pkgs):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], check=False)

pip_install("gradio==6.17.3", "huggingface_hub", "librosa", "soundfile",
            "speechbrain", "loralib", "scipy", "scikit-learn", "pandas")

# %% [markdown]
# ## 2. Kéo code UI (app.py) từ HF Space về Kaggle

# %%
import os
from huggingface_hub import snapshot_download

SPACE_REPO = "tranminhtoan140601/voicemos2026-demo"
LOCAL_DIR = "/kaggle/working/vmc_demo"

# Tải toàn bộ repo Space (app.py + requirements + README) về local
snapshot_download(repo_id=SPACE_REPO, repo_type="space", local_dir=LOCAL_DIR)
print("✅ Đã kéo Space về:", LOCAL_DIR)
print("Files:", os.listdir(LOCAL_DIR))

# %% [markdown]
# ## 3. Chạy app.py (Kaggle có GPU → nhanh; app.py tự share=True ra link gradio.live)
#
# `app.py` tải checkpoint Track 2 từ HF Models repo, clone URGENT-MOS/SAILER/baseline lúc bấm nút.
# Cell này sẽ **chạy mãi** (server Gradio) — đợi dòng `Running on public URL: https://....gradio.live`.

# %%
# Chạy như tiến trình con để giữ log; KHÔNG có SPACE_ID nên app.py tự bật share=True
subprocess.run([sys.executable, "app.py"], cwd=LOCAL_DIR, check=True)

# %% [markdown]
# ## Ghi chú
# - Đây là cách "1 nguồn code, chạy nơi có GPU free": sửa UI thì sửa trên Space HF → chạy lại notebook này.
# - Nếu muốn chạy bản local trong `kaggle_baseline/demo_all_tracks_gradio` (code inline) thì dùng notebook đó.
# - Lần đầu bấm nút mỗi track sẽ tải model (WavLM/SAILER/URGENT-MOS/ECAPA) → chờ chút; Kaggle có GPU nên inference nhanh.
# - Cần **Internet On** (tải code HF + model) + **GPU T4**.
