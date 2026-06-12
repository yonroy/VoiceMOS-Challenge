# Hướng dẫn deploy Triton Service trên Linux Server

> Server: `/home/nhandt23/project/VoiceMOS` · GPU 12GB · Ubuntu
> Cập nhật ngày: 12/6/2026

---

## Bước 0 — SSH vào server

```bash
ssh nhandt23@<địa-chỉ-ip-server>
# Vào đúng thư mục project
cd /home/nhandt23/project/VoiceMOS
```

---

## Bước 1 — Clone code (lần đầu — server CHƯA có repo)

```bash
mkdir -p /home/nhandt23/project && cd /home/nhandt23/project

# Repo public:
git clone https://github.com/yonroy/VoiceMOS-Challenge.git VoiceMOS
cd VoiceMOS
```

> **Nếu repo PRIVATE** (git clone hỏi mật khẩu / `Authentication failed`):
> ```bash
> # Tạo Personal Access Token: github.com → Settings → Developer settings →
> #   Personal access tokens → Generate (scope: repo)
> git clone https://<TOKEN>@github.com/yonroy/VoiceMOS-Challenge.git VoiceMOS
> ```
> Clone về **chỉ ~14MB** (data/ và baselines/ không nằm trong git → không bị kéo về).

Kiểm tra thư mục `triton_service/` đã có chưa:
```bash
ls triton_service/
# Phải thấy: docker-compose.yml  gateway/  loadtest/  model_repository/  run_server.sh
```

> **Lần sau cập nhật code** (đã clone rồi): `cd /home/nhandt23/project/VoiceMOS && git pull`

---

## Bước 2 — Kiểm tra môi trường server

```bash
# Docker đã cài chưa?
docker --version
# Cần: Docker version 20+ 

# NVIDIA Container Toolkit đã cài chưa?
nvidia-smi
# Cần thấy: GPU name, VRAM, driver version

# Docker có dùng GPU được không?
docker run --rm --gpus all nvidia/cuda:12.0-base-ubuntu20.04 nvidia-smi
# Cần thấy: cùng thông tin GPU như lệnh trên

# Dùng `docker compose` (v2) hay `docker-compose` (v1)?
docker compose version || docker-compose version
# run_server.sh dùng `docker compose` (v2). Nếu chỉ có v1, đổi lệnh thành docker-compose.
```

> **⚠️ KIỂM TRA QUAN TRỌNG — Python backend có `torch` không?** Code `model.py` cần `import torch`.
> Image `tritonserver:24.08-py3` *thường* có sẵn torch, nhưng nên xác nhận TRƯỚC khi build full (kéo base ~vài GB nhưng nhanh hơn build + tải model rồi mới phát hiện lỗi):
> ```bash
> docker run --rm nvcr.io/nvidia/tritonserver:24.08-py3 \
>     python3 -c "import torch; print('torch', torch.__version__, '· cuda', torch.cuda.is_available())"
> ```
> - **Ra `torch 2.x · cuda True`** → OK, build bình thường.
> - **Lỗi `No module named torch`** → thêm 2 dòng vào `triton_service/model_requirements.txt`:
>   `torch` và `torchaudio` (hoặc đổi base sang tag `*-pyt-python-py3`). Báo tôi nếu gặp.

> Nếu lệnh GPU lỗi "could not select device driver" → cần cài NVIDIA Container Toolkit:
> ```bash
> distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
> curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
> curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
>   sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
>   sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
> sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
> sudo systemctl restart docker
> ```

---

## Bước 3 — Đặt HuggingFace Token

Checkpoint Track 2 (`ft_emotion_full_20epoch.pt`) nằm trên HF repo private. Cần token để tải:

```bash
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
# Kiểm tra đã đặt chưa
echo $HF_TOKEN
```

> Token lấy ở: huggingface.co → Settings → Access Tokens → New token (Read).

---

## Bước 4 — Chuẩn bị file audio mẫu cho Locust

Locust cần ít nhất 1 file `.wav` để bắn test (đọc file vào RAM rồi bắn lặp lại nhiều lần — 1 file là đủ, 5–10 file độ dài khác nhau là lý tưởng để test luôn dynamic batching).

> ⚠️ **Tình huống của dự án này: file wav đang ở MÁY LOCAL của bạn, server CHƯA có.**
> → phải **upload từ máy bạn lên server** bằng `scp` (xem 4A). Nếu data đã có sẵn trên server thì dùng 4B.

### 4A — Upload file wav từ máy bạn lên server (trường hợp hiện tại)

**Bước 1.** Trên server, tạo sẵn thư mục đích (chạy ở terminal đang SSH vào server):
```bash
mkdir -p /home/nhandt23/project/VoiceMOS/triton_service/loadtest/samples
```

**Bước 2.** Mở **PowerShell NGAY TRÊN MÁY WINDOWS của bạn** (KHÔNG phải terminal SSH), chạy `scp`:
```powershell
# Cú pháp: scp <file-máy-bạn> <user>@<ip-server>:<đường-dẫn-đích-trên-server>

# Upload nhiều file .wav trong 1 thư mục (thay đường dẫn nguồn cho đúng máy bạn)
scp "d:\VFS\VoiceMOS Challenge 2026\data\<thư-mục-chứa-wav>\*.wav" `
    nhandt23@<ip-server>:/home/nhandt23/project/VoiceMOS/triton_service/loadtest/samples/

# Hoặc chỉ 1 file
scp "d:\VFS\VoiceMOS Challenge 2026\data\<...>\sample.wav" `
    nhandt23@<ip-server>:/home/nhandt23/project/VoiceMOS/triton_service/loadtest/samples/
```
`scp` sẽ hỏi mật khẩu server (giống lúc SSH). Backtick `` ` `` cuối dòng là nối dòng trong PowerShell.

**Bước 3.** Quay lại terminal SSH server, xác nhận đã lên:
```bash
ls -lh /home/nhandt23/project/VoiceMOS/triton_service/loadtest/samples/
# Phải thấy file .wav, dung lượng > 0
```

> 💡 Nếu bạn kết nối server bằng **VS Code Remote-SSH**: có thể **kéo-thả file** trực tiếp từ Explorer máy bạn vào thư mục `samples/` trên server — không cần gõ `scp`.

### 4B — Nếu data đã có sẵn TRÊN server

```bash
mkdir -p triton_service/loadtest/samples

# Tìm xem file .wav nằm ở đâu trên server
find /home/nhandt23/project/VoiceMOS -name "*.wav" 2>/dev/null | head -20

# Copy đúng đường dẫn vừa tìm được (thay cho đường dẫn ví dụ bên dưới)
cp /đường/dẫn/thật/*.wav triton_service/loadtest/samples/

# Xác nhận (phải KHÔNG rỗng)
ls -lh triton_service/loadtest/samples/
```

---

## Bước 5 — Khởi động server

```bash
cd triton_service

# Khởi động (lần đầu build image ~5-10 phút, kéo Triton base image ~vài GB)
bash run_server.sh
```

Terminal sẽ hiện log liên tục. Chờ đến khi thấy **cả 3 dòng** này:

```
I ... Successfully loaded model 'track2_emotion'
I ... Successfully loaded model 'track1_acr'
I ... Successfully loaded model 'track3_sim'
```

Và gateway:
```
INFO:     Uvicorn running on http://0.0.0.0:8080
```

> Lần đầu có thể mất **5–15 phút** vì cần tải WavLM-large + audeering + URGENT-MOS từ HuggingFace.

---

## Bước 6 — Mở terminal mới, kiểm tra server sống

```bash
# Mở terminal thứ 2 (server vẫn chạy ở terminal 1)
ssh nhandt23@<ip-server>
cd /home/nhandt23/project/VoiceMOS

# 1. Gateway còn sống không?
curl http://localhost:8080/health
# Kết quả mong đợi: {"status":"ok","triton":"triton:8000"}

# 2. Triton trực tiếp
curl http://localhost:8000/v2/health/ready
# Kết quả mong đợi: (HTTP 200, body rỗng)

# 3. GPU dùng bao nhiêu VRAM?
nvidia-smi
# Cần thấy: ~5-7 GB đã dùng (3 model nạp vào VRAM)
```

---

## Bước 7 — Test từng track bằng curl

```bash
# Lấy 1 file wav để test
WAV=triton_service/loadtest/samples/$(ls triton_service/loadtest/samples/ | head -1)
echo "Dùng file: $WAV"

# Track 2 — 6 cột cảm xúc
curl -s -F "file=@$WAV" -F "target_emotion=happy" \
     http://localhost:8080/track2 | python3 -m json.tool

# Kết quả mong đợi dạng:
# {
#   "qmos": 3.8,
#   "cat": {"angry": 0.02, "happy": 0.85, ...},
#   "perceived_emotion": "happy",
#   "vad": {"valence": 4.1, "arousal": 3.9, "dominance": 3.5},
#   "emos": 4.2,
#   "target_emotion": "happy",
#   "emos_match": true
# }

# Track 1 — ACR (1 file)
curl -s -F "file_a=@$WAV" \
     http://localhost:8080/track1 | python3 -m json.tool
# Kết quả: {"acr_a": 3.7}

# Track 1 — CCR (2 file so sánh)
curl -s -F "file_a=@$WAV" -F "file_b=@$WAV" \
     http://localhost:8080/track1 | python3 -m json.tool
# Kết quả: {"acr_a": 3.7, "acr_b": 3.7, "ccr": 0.0}

# Track 3 — spk_sim + acc_sim
curl -s -F "file_test=@$WAV" -F "file_ref=@$WAV" \
     http://localhost:8080/track3 | python3 -m json.tool
# Kết quả: {"spk_sim": 4.5, "acc_sim": 4.3, "cosine": 0.92}
```

---

## Bước 7B — Mở UI Gradio (tùy chọn, demo trực quan)

UI là **client** kéo-thả audio → xem điểm, gọi gateway `:8080`. Có **2 cách chạy**, chọn 1:

### Cách 1 — Chạy UI ngay TRÊN MÁY WINDOWS của bạn (khuyến nghị)
UI không cần GPU, chỉ cần gọi tới gateway server. Mở **PowerShell trên máy bạn**:
```powershell
cd "d:\VFS\VoiceMOS Challenge 2026\triton_service\ui"
pip install -r requirements.txt

# Trỏ thẳng tới gateway trên server (thay <ip-server>)
$env:GATEWAY_URL = "http://<ip-server>:8080"
python app_ui.py
# Mở http://localhost:7860 → kéo audio vào chấm
```
> Cần server **mở cổng 8080** cho máy bạn truy cập (cùng mạng nội bộ/VPN). Kiểm tra:
> `curl http://<ip-server>:8080/health` từ máy bạn phải ra `{"status":"ok"}`.

### Cách 2 — Chạy UI TRÊN SERVER + SSH tunnel (khi cổng server bị chặn)
Chạy UI trên server, "đục" 1 đường hầm về máy bạn:
```bash
# Trên server (terminal mới):
cd /home/nhandt23/project/VoiceMOS/triton_service/ui
pip install -r requirements.txt
GATEWAY_URL=http://localhost:8080 python app_ui.py   # UI chạy ở cổng 7860 trên server
```
```powershell
# Trên MÁY BẠN — mở SSH tunnel đưa cổng 7860 của server về localhost:7860:
ssh -N -L 7860:localhost:7860 nhandt23@<ip-server>
# Rồi mở trình duyệt máy bạn: http://localhost:7860
```

> UI có **4 tab**: 🎯 Track 2 (6 cột cảm xúc) · 🔊 Track 1 (ACR/CCR) · 🗣️ Track 3 (spk/acc/cos) · 📦 Chấm hàng loạt (đo throughput). Bấm **"Kiểm tra server"** đầu trang để xác nhận 🟢 trước khi chấm.

---

## Bước 8 — Chạy Locust loadtest

```bash
# Cài Locust (1 lần)
pip install -r triton_service/loadtest/requirements.txt

cd triton_service/loadtest
mkdir -p results

# Chạy nhẹ trước — 5 người, 1 phút (kiểm tra không có lỗi)
locust -f locustfile.py --host http://localhost:8080 \
       --users 5 --spawn-rate 1 --run-time 1m --headless \
       --csv results/warmup
cat results/warmup_stats.csv

# Chạy thật — 20 người, 2 phút (đây là số mentor yêu cầu)
locust -f locustfile.py --host http://localhost:8080 \
       --users 20 --spawn-rate 2 --run-time 2m --headless \
       --csv results/batch8
```

Kết quả trông như này:

```
Type    Name          Req/s   Fail/s  Avg(ms)  p50   p95
POST    /track2        12.3     0.0      820    750   1200
POST    /track1         5.1     0.0      340    310    520
POST    /track3         5.0     0.0      280    260    410
GET     /health         2.1     0.0        5      5     10
```

---

## Bước 9 (tùy chọn) — So sánh dynamic batching ON vs OFF

```bash
cd /home/nhandt23/project/VoiceMOS/triton_service

# Tắt batching (max_batch_size: 1)
sed -i 's/max_batch_size: 8/max_batch_size: 1/' \
    model_repository/track2_emotion/config.pbtxt

# Restart server
docker compose restart triton
sleep 30   # đợi model reload

# Chạy loadtest
cd loadtest
locust -f locustfile.py --host http://localhost:8080 \
       --users 20 --spawn-rate 2 --run-time 1m --headless \
       --csv results/batch1

# Bật lại batching
cd ..
sed -i 's/max_batch_size: 1/max_batch_size: 8/' \
    model_repository/track2_emotion/config.pbtxt
docker compose restart triton
sleep 30

cd loadtest
locust -f locustfile.py --host http://localhost:8080 \
       --users 20 --spawn-rate 2 --run-time 1m --headless \
       --csv results/batch8

# So sánh throughput
echo "=== batch=1 ===" && grep "track2" results/batch1_stats.csv
echo "=== batch=8 ===" && grep "track2" results/batch8_stats.csv
```

---

## Dừng server

```bash
# Dừng tất cả (Ctrl+C ở terminal 1, hoặc từ terminal khác)
cd /home/nhandt23/project/VoiceMOS/triton_service
docker compose down
```

---

## Xử lý lỗi thường gặp

| Lỗi | Nguyên nhân | Cách sửa |
|---|---|---|
| `track2_emotion UNAVAILABLE` | Checkpoint chưa tải xong / HF_TOKEN sai | Kiểm tra log Triton, xác nhận `echo $HF_TOKEN` |
| `502 Bad Gateway` từ curl | Triton chưa READY, gateway gọi sớm | Đợi thêm, kiểm tra `curl localhost:8000/v2/health/ready` |
| `CUDA out of memory` | 3 model không vừa VRAM | Giảm `count: 2` → `count: 1` trong config.pbtxt |
| `could not select device driver` | NVIDIA Container Toolkit chưa cài | Xem Bước 2 |
| `No .wav in samples/` | Chưa có file mẫu cho Locust | Xem Bước 4 |
| Build image lâu / treo | Đang pull Triton base image (~5GB) | Chờ, kiểm tra internet server |
