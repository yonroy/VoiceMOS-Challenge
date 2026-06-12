# ─────────────────────────────────────────────────────────────────────────────
# Chạy Triton LOCAL trên Windows (Docker Desktop + GPU). Chạy từ thư mục triton_service/:
#     .\run_local.ps1                # build (nếu cần) + chạy server, GPU
#     .\run_local.ps1 -Cpu          # chạy CPU (nhớ sửa config.pbtxt → KIND_CPU)
#     .\run_local.ps1 -HfToken hf_xxx   # nếu HF repo checkpoint để private
#     .\run_local.ps1 -SkipBuild    # bỏ qua build, chạy luôn image đã có
#
# Yêu cầu: ĐÃ MỞ Docker Desktop (daemon chạy). Lần đầu pull image Triton ~vài GB → chờ.
# ─────────────────────────────────────────────────────────────────────────────
param(
    [switch]$Cpu,
    [switch]$SkipBuild,
    [string]$HfToken = $env:HF_TOKEN
)

$ErrorActionPreference = "Stop"
$here = $PSScriptRoot
$image = "voicemos-triton"

# 0) Docker daemon đang chạy?
docker info *> $null
if (-not $?) {
    Write-Host "🔴 Docker daemon chưa chạy. Hãy MỞ Docker Desktop rồi chạy lại." -ForegroundColor Red
    exit 1
}

# 1) Build image (Triton + transformers/librosa...)
if (-not $SkipBuild) {
    Write-Host "🔨 Build image $image (lần đầu lâu do pull base Triton)..." -ForegroundColor Cyan
    docker build -t $image $here
    if (-not $?) { Write-Host "🔴 Build lỗi." -ForegroundColor Red; exit 1 }
}

# 2) Chạy server, mount model_repository
$modelRepo = Join-Path $here "model_repository"
$envArgs = @()
if ($HfToken) { $envArgs += @("-e", "HF_TOKEN=$HfToken") }

$gpuArgs = @("--gpus", "all")
if ($Cpu) {
    $gpuArgs = @()
    Write-Host "⚙️  CPU mode — đảm bảo config.pbtxt đã đổi KIND_GPU → KIND_CPU." -ForegroundColor Yellow
}

Write-Host "🚀 Chạy Triton (HTTP 8000 / gRPC 8001 / metrics 8002). Ctrl+C để dừng." -ForegroundColor Green
docker run --rm @gpuArgs `
    -p 8000:8000 -p 8001:8001 -p 8002:8002 `
    -v "${modelRepo}:/models" `
    @envArgs `
    $image tritonserver --model-repository=/models

# Khi log hiện 'track2_emotion ... READY' → mở terminal khác chạy UI:
#     cd ui ; pip install -r requirements.txt ; python app_ui.py
