#!/usr/bin/env bash
# Khởi động Triton + Gateway trên Linux server (NVIDIA Container Toolkit cần được cài sẵn).
# Chạy từ thư mục triton_service/:
#   bash run_server.sh              # dùng HF_TOKEN từ env
#   HF_TOKEN=hf_xxx bash run_server.sh
#   bash run_server.sh --no-gpu    # test CPU (đổi KIND_GPU → KIND_CPU trong config trước)

set -euo pipefail
cd "$(dirname "$0")"

# Tắt "compose bake" (builder mới) — trên vài bản docker compose nó gãy với lỗi
# "failed to execute bake: read |0: file already closed" sau khi build xong.
# Dùng builder cổ điển cho ổn định.
export COMPOSE_BAKE=false

NO_GPU=0
for arg in "$@"; do [[ "$arg" == "--no-gpu" ]] && NO_GPU=1; done

# Kiểm tra Docker + NVIDIA
if ! docker info &>/dev/null; then
    echo "❌ Docker daemon chưa chạy." && exit 1
fi
if [[ $NO_GPU -eq 0 ]] && ! nvidia-smi &>/dev/null; then
    echo "⚠️  nvidia-smi không thấy GPU — chạy với --no-gpu để ép CPU mode." && exit 1
fi

export HF_TOKEN="${HF_TOKEN:-}"
if [[ -z "$HF_TOKEN" ]]; then
    echo "⚠️  HF_TOKEN chưa đặt — nếu checkpoint private thì sẽ lỗi khi tải."
fi

if [[ $NO_GPU -eq 1 ]]; then
    # Tạm patch compose để bỏ GPU device (chỉ test)
    COMPOSE_FILE="docker-compose.yml"
    echo "⚙️  CPU mode — đảm bảo config.pbtxt đã đổi KIND_GPU → KIND_CPU."
    docker compose -f "$COMPOSE_FILE" up --build \
        --scale gateway=1
else
    docker compose up --build
fi
