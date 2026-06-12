# ─────────────────────────────────────────────────────────────────────────────
# Image Triton + thư viện cho Python backend Track 2.
# Base tritonserver:*-py3 đã có sẵn Python backend + torch/torchaudio + CUDA.
# Ta chỉ thêm: git/ffmpeg/libsndfile (đọc audio) + transformers/librosa... (model).
#
# Build:  docker build -t voicemos-triton ./triton_service
# (xem README.md để biết lệnh `docker run` nạp model_repository)
# ─────────────────────────────────────────────────────────────────────────────
# Chọn tag khớp máy bạn; 24.08 là bản ổn định phổ biến. Xem:
#   https://docs.nvidia.com/deeplearning/frameworks/support-matrix/
FROM nvcr.io/nvidia/tritonserver:24.08-py3

RUN apt-get update && apt-get install -y --no-install-recommends \
        git ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY model_requirements.txt /tmp/model_requirements.txt
RUN pip install --no-cache-dir -r /tmp/model_requirements.txt

# Cache HF + torch.hub ghi vào /tmp (ghi được khi chạy không-root)
ENV HF_HOME=/tmp/hf_cache \
    TORCH_HOME=/tmp/torch_cache \
    MODELS_DIR=/tmp/models \
    PYTHONUNBUFFERED=1
