#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Dựng MÔI TRƯỜNG RIÊNG cho Track 1 (URGENT-MOS) rồi đóng gói (conda-pack) thành
# track1_env.tar.gz để Triton Python backend nạp RIÊNG cho model track1_acr
# (qua tham số EXECUTION_ENV_PATH trong config.pbtxt).
#
# VÌ SAO: URGENT-MOS dùng Qwen3-Omni-MoE → cần `transformers` đời mới + torch 2.7 +
# torchcodec≥0.4 (có AudioDecoder thật). Stack này XUNG ĐỘT với image chung của
# Track 2 (pin transformers<4.50 + torch 2.4 cho WavLM). Cô lập env là cách sạch nhất:
# Track 1 cài đúng đồ của nó, Track 2/3 giữ nguyên image, không đụng nhau.
#
# CHẠY TRÊN SERVER (dùng conda `base` có sẵn — dấu nhắc đang là `(base)`):
#   cd /userdata/nhandt23/VoiceMos/VoiceMOS-Challenge/triton_service/track1_env
#   bash build_track1_env.sh
#
# Sản phẩm: ../model_repository/track1_acr/track1_env.tar.gz  (~1.5–2.5GB, .gitignore bỏ qua)
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

ENV_NAME="${ENV_NAME:-track1env}"
PY_VER="3.10"   # PHẢI khớp Python của Triton 24.08 (3.10) để dùng chung python_backend_stub
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="$HERE/../model_repository/track1_acr/track1_env.tar.gz"
URGENT_REPO="${URGENT_REPO:-/tmp/URGENT-MOS-build}"

echo "==> 1/6  Tạo conda env '$ENV_NAME' (python $PY_VER)"
source "$(conda info --base)/etc/profile.d/conda.sh"
# Gỡ MỌI env đang chồng (vd .venv layer lên) để khỏi cài nhầm chỗ.
while [[ -n "${VIRTUAL_ENV:-}" ]]; do deactivate 2>/dev/null || break; done
conda deactivate 2>/dev/null || true
conda create -y -n "$ENV_NAME" "python=$PY_VER"
conda activate "$ENV_NAME"

# CHỐNG NHẦM MÔI TRƯỜNG: dùng THẲNG python tuyệt đối của env, không phụ thuộc PATH.
ENVPY="$CONDA_PREFIX/bin/python"
echo "    CONDA_PREFIX = $CONDA_PREFIX"
echo "    ENVPY        = $ENVPY ($($ENVPY --version 2>&1))"
if [[ "$(basename "$CONDA_PREFIX")" != "$ENV_NAME" ]]; then
    echo "❌ CONDA_PREFIX không trỏ vào $ENV_NAME (có env khác đang chồng). Hãy 'deactivate' rồi chạy lại." && exit 1
fi
export PYTHONNOUSERSITE=1

echo "==> 2/6  Cài ffmpeg (torchcodec cần lib FFmpeg để import được)"
conda install -y -n "$ENV_NAME" -c conda-forge "ffmpeg<7"

"$ENVPY" -m pip install -U pip

echo "==> 3/6  Lấy source URGENT-MOS"
rm -rf "$URGENT_REPO"
git clone -q https://github.com/vvwangvv/URGENT-MOS.git "$URGENT_REPO"

echo "==> 4/6  Cài URGENT-MOS + toàn bộ stack gốc của nó (torch2.7/torchcodec/transformers Qwen3-Omni...)"
# Cài non-editable → urgent_mos + deps NẰM HẲN trong env (đi theo conda-pack, không phụ thuộc path build).
"$ENVPY" -m pip install "$URGENT_REPO"

echo "==> 5/6  KIỂM TRA import BẰNG PYTHON CỦA ENV (không phải python ngoài)"
# numpy phải nằm ĐÚNG trong env này, nếu thiếu là pack ra sẽ lỗi 'No module named numpy'.
"$ENVPY" -c "import numpy, sys; print('   numpy', numpy.__version__, 'tại', numpy.__file__)"
"$ENVPY" - <<'PY'
import transformers.models.qwen3_omni_moe          # class Track1 cần
from torchcodec.decoders import AudioDecoder        # phải là class THẬT (không stub)
import urgent_mos.model.urgent_mos                   # module từng làm Triton chết
print(">> ENV OK: transformers", __import__("transformers").__version__,
      "| torch", __import__("torch").__version__,
      "| AudioDecoder", AudioDecoder)
PY

echo "==> 6/6  Đóng gói env -> $OUT"
# conda-pack cài vào BASE (công cụ đóng gói), gọi lệnh standalone `conda-pack` cho chắc.
"$(conda info --base)/bin/python" -m pip install -q conda-pack
rm -f "$OUT"
conda-pack -n "$ENV_NAME" -o "$OUT" --force

echo ""
echo "✅ XONG. File env: $OUT"
ls -lh "$OUT"
echo "→ Bước tiếp: config.pbtxt của track1_acr đã trỏ EXECUTION_ENV_PATH vào file này."
echo "  Khởi động lại: cd ..; cd ..; bash run_server.sh --no-gpu"
