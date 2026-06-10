# Docker cho Hugging Face Space (sdk: docker). Theo template HF: chạy user uid 1000.
FROM python:3.10-slim

# system deps: git (clone URGENT-MOS / baseline / vox-profile), ffmpeg + libsndfile (đọc audio)
RUN apt-get update && apt-get install -y --no-install-recommends \
        git ffmpeg libsndfile1 build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    TORCH_HOME=/home/user/.cache/torch \
    MODELS_DIR=/home/user/models \
    PYTHONUNBUFFERED=1

WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

COPY --chown=user app ./app

EXPOSE 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
