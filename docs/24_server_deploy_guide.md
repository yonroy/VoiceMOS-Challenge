# Hướng dẫn deploy & chạy Triton Service trên Linux Server

> Server: `/home/nhandt23/project/VoiceMOS` · GPU 12GB · Ubuntu
> Cập nhật ngày: 15/6/2026

---

## 🔀 2 chế độ chạy — đọc trước khi bắt đầu

| Chế độ | Khi nào dùng | Nạp model | Lệnh khởi động |
|---|---|---|---|
| **CPU test** | GPU đang **đầy / hết VRAM**, chỉ cần kiểm tra pipeline đúng/sai | **Chỉ Track 2** (trước) | `bash run_server.sh --no-gpu` |
| **GPU thật** | GPU rảnh, cần đo tốc độ (Locust, dynamic batching) | **Cả 3 track** | `bash run_server.sh` |

**Trạng thái repo hiện tại = CPU test (đã cấu hình sẵn 15/6):**
- 3 file `config.pbtxt` đặt `kind: KIND_CPU`
- `docker-compose.yml`: comment khối GPU `deploy`, command thêm `--model-control-mode=explicit --load-model=track2_emotion` (chỉ nạp Track 2)

> ⚠️ **CPU test = chỉ kiểm tra ĐÚNG/SAI** (ra JSON 6 cột, điểm khớp `api_service`). **KHÔNG** lấy số tốc độ trên CPU: WavLM-large CPU rất chậm (~chục giây/file) và dynamic batching không tăng tốc như GPU. Locust / so batch 1-vs-8 (mentor yêu cầu) **phải chạy GPU thật**.

👉 Nếu chỉ muốn test nhanh Track 2 bây giờ: nhảy thẳng tới **[Phần A — Chạy nhanh CPU test](#phần-a--chạy-nhanh-cpu-test-track-2)**.
👉 Muốn chạy đầy đủ GPU 3 track + Locust: xem **[Phần B — Chạy GPU đầy đủ](#phần-b--chạy-gpu-đầy-đủ-3-track--locust)**.

---

# Phần A — Chạy nhanh CPU test (Track 2)

> Mục tiêu: dựng pipeline Track 2 trên CPU, chấm 1 file `.wav`, xác nhận ra JSON 6 cột đúng. Đây là việc "🟠 chạy thật + curl test Track 2" còn nợ từ Phiên 24.

## A1 — Lấy code mới về server

```bash
ssh nhandt23@<địa-chỉ-ip-server>
cd /home/nhandt23/project/VoiceMOS
git pull          # kéo 4 file CPU mode vừa sửa (sau khi đã commit/push từ máy local)
```

> Lần đầu chưa có repo: xem **[Phần B / Bước 1](#bước-1--clone-code-lần-đầu)** để clone.

## A2 — Đặt HuggingFace Token (để tải checkpoint Track 2)

```bash
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
echo $HF_TOKEN     # kiểm tra đã đặt
```
> Token: huggingface.co → Settings → Access Tokens → New token (Read). Checkpoint `ft_emotion_full_20epoch.pt` nằm trên repo HF private.

## A3 — Khởi động (CPU, chỉ Track 2)

```bash
cd triton_service
bash run_server.sh --no-gpu
```

Lệnh này build image (lần đầu ~5–10 phút cài torch...) rồi chạy Triton + gateway. Chờ tới khi log hiện:

```
I ... Successfully loaded model 'track2_emotion'
...
[track2_emotion] sẵn sàng trên cpu
INFO:     Uvicorn running on http://0.0.0.0:8080
```

> Request **đầu tiên** sẽ tải WavLM-large + UTMOS + audeering từ HF (~vài GB) → file đầu chậm hơn các file sau.
> Trên CPU **không** thấy dòng `track1_acr` / `track3_sim` — đúng, vì chế độ này chỉ nạp Track 2.

## A4 — Test Track 2 (mở terminal thứ 2)

```bash
ssh nhandt23@<ip-server>
cd /home/nhandt23/project/VoiceMOS

# 1. Gateway sống chưa?
curl http://localhost:8080/health
# Mong đợi: {"status":"ok","triton":"triton:8000"}

# 2. Chấm 1 file (đổi đường dẫn wav cho đúng)
curl -s -F "file=@/duong/dan/sample.wav" -F "target_emotion=happy" \
     http://localhost:8080/track2 | python3 -m json.tool
```

Kết quả mong đợi (JSON 6 cột):
```json
{
  "qmos": 3.8,
  "cat": {"angry": 0.02, "happy": 0.85, "neutral": 0.08, "sad": 0.03, "surprised": 0.02},
  "perceived_emotion": "happy",
  "vad": {"valence": 4.1, "arousal": 3.9, "dominance": 3.5},
  "emos": 4.2,
  "target_emotion": "happy",
  "emos_match": true
}
```
> Bỏ `-F "target_emotion=..."` thì kết quả không có `emos`/`emos_match` (head EMOS cần target one-hot).

## A5 — Đối chiếu tính đúng

Chấm **cùng 1 file** bằng `api_service` cũ (HF Space hoặc local — cùng checkpoint `ft_emotion_full_20epoch.pt`). Điểm 6 cột phải **khớp** → pipeline Triton port đúng. Lệch nhiều → kiểm tra version checkpoint / tiền xử lý audio.

## A6 — Dừng

```bash
cd /home/nhandt23/project/VoiceMOS/triton_service
docker compose down
```

---

# Phần B — Chạy GPU đầy đủ (3 track + Locust)

> Dùng khi GPU rảnh. Trước khi chạy, phải **đảo cấu hình về GPU** (xem B0), nếu không Triton vẫn chạy CPU và chỉ nạp Track 2.

## B0 — Đảo cấu hình CPU → GPU

```bash
cd /home/nhandt23/project/VoiceMOS/triton_service

# 1. 3 config: KIND_CPU → KIND_GPU
sed -i 's/kind: KIND_CPU/kind: KIND_GPU/' \
    model_repository/track2_emotion/config.pbtxt \
    model_repository/track1_acr/config.pbtxt \
    model_repository/track3_sim/config.pbtxt

# 2. docker-compose.yml:
#    - BỎ comment 6 dòng khối deploy/resources/.../nvidia
#    - XÓA 2 dòng: --model-control-mode=explicit và --load-model=track2_emotion
#      (Triton mặc định nạp TẤT CẢ model trong model_repository)
nano docker-compose.yml
```
> Sửa tay `docker-compose.yml` cho chắc (2 chỗ trên đều có ghi chú `# CPU test mode:` ngay trên). Sau đó dùng `bash run_server.sh` (không `--no-gpu`).

## Bước 1 — Clone code (lần đầu)

```bash
mkdir -p /home/nhandt23/project && cd /home/nhandt23/project
git clone https://github.com/yonroy/VoiceMOS-Challenge.git VoiceMOS
cd VoiceMOS
```
> **Repo PRIVATE** (hỏi mật khẩu): tạo Personal Access Token (github.com → Settings → Developer settings → Personal access tokens, scope `repo`) rồi:
> `git clone https://<TOKEN>@github.com/yonroy/VoiceMOS-Challenge.git VoiceMOS`
> Clone ~14MB (data/ + baselines/ không nằm trong git).

Kiểm tra:
```bash
ls triton_service/
# Phải thấy: docker-compose.yml  gateway/  loadtest/  model_repository/  run_server.sh  ui/
```
> Lần sau cập nhật: `cd /home/nhandt23/project/VoiceMOS && git pull`

## Bước 2 — Kiểm tra môi trường

```bash
docker --version                      # Docker 20+
nvidia-smi                            # thấy GPU, VRAM, driver
docker run --rm --gpus all nvidia/cuda:12.0-base-ubuntu20.04 nvidia-smi   # Docker dùng được GPU?
docker compose version || docker-compose version    # v2 hay v1 (run_server.sh dùng v2)
```

> **Kiểm tra `torch` trong Python backend** (image *thường* có sẵn):
> ```bash
> docker run --rm nvcr.io/nvidia/tritonserver:24.08-py3 \
>     python3 -c "import torch; print('torch', torch.__version__, '· cuda', torch.cuda.is_available())"
> ```
> Lỗi `No module named torch` → thêm `torch` + `torchaudio` vào `triton_service/model_requirements.txt`.

> Lỗi `could not select device driver` → cài NVIDIA Container Toolkit:
> ```bash
> distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
> curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
> curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
>   sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
>   sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
> sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
> sudo systemctl restart docker
> ```

## Bước 3 — HuggingFace Token

```bash
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
echo $HF_TOKEN
```

## Bước 4 — Chuẩn bị file audio mẫu cho Locust

Locust cần ≥1 file `.wav` (5–10 file độ dài khác nhau là lý tưởng để test luôn dynamic batching).

> ⚠️ File wav đang ở **máy local**, server chưa có → upload qua `scp` (4A). Nếu server có sẵn data → 4B.

### 4A — Upload từ máy bạn lên server

```bash
# Trên server: tạo thư mục đích
mkdir -p /home/nhandt23/project/VoiceMOS/triton_service/loadtest/samples
```
```powershell
# Trên PowerShell MÁY WINDOWS của bạn (KHÔNG phải SSH):
scp "d:\VFS\VoiceMOS Challenge 2026\data\<thư-mục-wav>\*.wav" `
    nhandt23@<ip-server>:/home/nhandt23/project/VoiceMOS/triton_service/loadtest/samples/
```
```bash
# Quay lại server, xác nhận:
ls -lh /home/nhandt23/project/VoiceMOS/triton_service/loadtest/samples/
```
> Dùng **VS Code Remote-SSH** thì kéo-thả file thẳng vào `samples/`, khỏi `scp`.

### 4B — Data đã có sẵn trên server

```bash
mkdir -p triton_service/loadtest/samples
find /home/nhandt23/project/VoiceMOS -name "*.wav" 2>/dev/null | head -20
cp /đường/dẫn/thật/*.wav triton_service/loadtest/samples/
ls -lh triton_service/loadtest/samples/
```

## Bước 5 — Khởi động (GPU, 3 track)

```bash
cd triton_service
bash run_server.sh            # KHÔNG --no-gpu
```

Chờ thấy **cả 3 dòng**:
```
I ... Successfully loaded model 'track2_emotion'
I ... Successfully loaded model 'track1_acr'
I ... Successfully loaded model 'track3_sim'
```
Và gateway: `INFO:     Uvicorn running on http://0.0.0.0:8080`

> Lần đầu 5–15 phút (tải WavLM-large + audeering + URGENT-MOS + ECAPA từ HF).

## Bước 6 — Kiểm tra server sống (terminal 2)

```bash
ssh nhandt23@<ip-server>; cd /home/nhandt23/project/VoiceMOS

curl http://localhost:8080/health          # {"status":"ok","triton":"triton:8000"}
curl http://localhost:8000/v2/health/ready # HTTP 200, body rỗng
nvidia-smi                                 # ~5-7 GB VRAM (3 model)
```

## Bước 7 — Test từng track bằng curl

```bash
WAV=triton_service/loadtest/samples/$(ls triton_service/loadtest/samples/ | head -1)
echo "Dùng file: $WAV"

# Track 2 — 6 cột
curl -s -F "file=@$WAV" -F "target_emotion=happy" http://localhost:8080/track2 | python3 -m json.tool

# Track 1 — ACR (1 file) / CCR (2 file)
curl -s -F "file_a=@$WAV" http://localhost:8080/track1 | python3 -m json.tool                 # {"acr_a":3.7}
curl -s -F "file_a=@$WAV" -F "file_b=@$WAV" http://localhost:8080/track1 | python3 -m json.tool # {"acr_a","acr_b","ccr"}

# Track 3 — spk_sim + acc_sim
curl -s -F "file_test=@$WAV" -F "file_ref=@$WAV" http://localhost:8080/track3 | python3 -m json.tool
```

## Bước 7B — UI Gradio (tùy chọn)

UI là **client** kéo-thả audio → xem điểm, gọi gateway `:8080`. Chọn 1 cách:

**Cách 1 — chạy UI trên máy Windows (khuyến nghị):**
```powershell
cd "d:\VFS\VoiceMOS Challenge 2026\triton_service\ui"
pip install -r requirements.txt
$env:GATEWAY_URL = "http://<ip-server>:8080"
python app_ui.py     # http://localhost:7860
```
> Cần server mở cổng 8080 cho máy bạn (`curl http://<ip-server>:8080/health` từ máy bạn phải OK).

**Cách 2 — UI trên server + SSH tunnel (khi cổng bị chặn):**
```bash
cd /home/nhandt23/project/VoiceMOS/triton_service/ui
pip install -r requirements.txt
GATEWAY_URL=http://localhost:8080 python app_ui.py    # cổng 7860 trên server
```
```powershell
ssh -N -L 7860:localhost:7860 nhandt23@<ip-server>    # rồi mở http://localhost:7860
```
> UI 4 tab: 🎯 Track 2 · 🔊 Track 1 · 🗣️ Track 3 · 📦 Chấm hàng loạt. Bấm **"Kiểm tra server"** trước khi chấm.

## Bước 8 — Locust loadtest

```bash
pip install -r triton_service/loadtest/requirements.txt
cd triton_service/loadtest && mkdir -p results

# Warmup 5 user, 1 phút
locust -f locustfile.py --host http://localhost:8080 \
       --users 5 --spawn-rate 1 --run-time 1m --headless --csv results/warmup
cat results/warmup_stats.csv

# Thật 20 user, 2 phút (số mentor yêu cầu)
locust -f locustfile.py --host http://localhost:8080 \
       --users 20 --spawn-rate 2 --run-time 2m --headless --csv results/batch8
```

## Bước 9 — So dynamic batching ON vs OFF

```bash
cd /home/nhandt23/project/VoiceMOS/triton_service

sed -i 's/max_batch_size: 8/max_batch_size: 1/' model_repository/track2_emotion/config.pbtxt
docker compose restart triton && sleep 30
cd loadtest && locust -f locustfile.py --host http://localhost:8080 \
       --users 20 --spawn-rate 2 --run-time 1m --headless --csv results/batch1

cd .. && sed -i 's/max_batch_size: 1/max_batch_size: 8/' model_repository/track2_emotion/config.pbtxt
docker compose restart triton && sleep 30
cd loadtest && locust -f locustfile.py --host http://localhost:8080 \
       --users 20 --spawn-rate 2 --run-time 1m --headless --csv results/batch8

echo "=== batch=1 ===" && grep "track2" results/batch1_stats.csv
echo "=== batch=8 ===" && grep "track2" results/batch8_stats.csv
```

## Dừng

```bash
cd /home/nhandt23/project/VoiceMOS/triton_service
docker compose down
```

---

## Xử lý lỗi thường gặp

| Lỗi | Nguyên nhân | Cách sửa |
|---|---|---|
| Chạy CPU mà Triton đòi GPU / `could not select device driver` | Chưa comment khối `deploy` GPU trong compose, hoặc quên `--no-gpu` | Kiểm tra compose đã comment khối `nvidia`; chạy `bash run_server.sh --no-gpu` |
| Chạy GPU nhưng chỉ nạp Track 2 | Còn `--load-model=track2_emotion` trong compose | Xóa 2 dòng `--model-control-mode`/`--load-model` (xem B0) |
| Chạy GPU nhưng log "sẵn sàng trên cpu" | Config còn `KIND_CPU` | Đảo về `KIND_GPU` (xem B0) |
| `track2_emotion UNAVAILABLE` | Checkpoint chưa tải / HF_TOKEN sai | Xem log Triton, `echo $HF_TOKEN` |
| `ValueError: ... torch.load ... require torch >= v2.6 ... CVE-2025-32434` | transformers ≥4.50 cấm `torch.load` khi torch<2.6; `microsoft/wavlm-large` chỉ có `.bin` | Đã pin `transformers<4.50` trong `model_requirements.txt`; rebuild: `docker compose build --no-cache triton` rồi `bash run_server.sh --no-gpu` |
| `No module named 'loralib'` (chỉ WARN) | Thiếu loralib → SAILER wrapper lỗi | **Bỏ qua** — tự fallback WavLM trắng rồi nạp đè trọng số fine-tune từ checkpoint (giống api_service) |
| `PytorchStreamReader failed reading zip archive: failed finding central directory` | File checkpoint (UTMOS/WavLM...) tải **dở dang/đứt mạng** → `.pt` cụt | Cache nay đã để ở named volume (tải 1 lần). Nếu file hỏng còn kẹt trong volume: `docker compose down && docker volume rm triton_service_torch_cache triton_service_hf_cache && docker compose up` để tải lại sạch |
| `502 Bad Gateway` từ curl | Triton chưa READY, gateway gọi sớm | Đợi thêm; `curl localhost:8000/v2/health/ready` |
| `CUDA out of memory` (GPU) | 3 model không vừa VRAM 12GB | Giảm `count` trong config.pbtxt, hoặc chạy ít model hơn |
| `No .wav in samples/` | Chưa có file mẫu Locust | Xem Bước 4 |
| Build image lâu / treo | Đang pull Triton base (~5GB) + cài torch | Chờ; kiểm tra internet server |
| `failed to execute bake: read \|0: file already closed` (sau khi build XONG) | Bug compose "bake"; image thực ra **đã build** | `run_server.sh` đã set `COMPOSE_BAKE=false`. Nếu vẫn gặp: chạy thẳng `docker compose up` (không `--build`, dùng image đã có) |
| `Bind for 0.0.0.0:8080 failed: port is already allocated` | Cổng 8080 đã bị service khác trên server chiếm | Gateway nay mặc định **18080** (`GATEWAY_PORT`). Test ở `localhost:18080`. Kiểm tra cổng bận: `sudo lsof -i:8080` hoặc `docker ps`. Đổi cổng khác: `GATEWAY_PORT=9090 docker compose up` |
| CPU chấm rất chậm (~chục giây/file) | WavLM-large trên CPU | Bình thường — CPU chỉ để test đúng/sai, không đo tốc độ |
