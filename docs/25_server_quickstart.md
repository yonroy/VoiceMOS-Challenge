# ⚡ Quickstart server — cầm tay khi lên Triton (1 trang)

> Bản tóm tắt copy-paste. Chi tiết + xử lý lỗi xem [24_server_deploy_guide.md](24_server_deploy_guide.md).
> Thay `<ip-server>` bằng IP thật. Server: `/home/nhandt23/project/VoiceMOS` · GPU 12GB.

---

## 0️⃣ Lên server + CLONE code (lần đầu — server chưa có repo)
```bash
ssh nhandt23@<ip-server>
mkdir -p /home/nhandt23/project && cd /home/nhandt23/project

# Clone repo (public → chạy thẳng; private → xem ghi chú dưới)
git clone https://github.com/yonroy/VoiceMOS-Challenge.git VoiceMOS
cd VoiceMOS

# Kiểm tra đã có triton_service:
ls triton_service/    # thấy: docker-compose.yml gateway/ loadtest/ model_repository/ run_server.sh
```
> **Repo PRIVATE?** Nếu `git clone` hỏi mật khẩu / báo `Authentication failed`:
> ```bash
> # Cách nhanh: dùng Personal Access Token (github.com → Settings → Developer settings → PAT)
> git clone https://<TOKEN>@github.com/yonroy/VoiceMOS-Challenge.git VoiceMOS
> ```
> **Lần sau cập nhật code** (đã clone rồi): `cd VoiceMOS && git pull`

## 1️⃣ Kiểm tra TRƯỚC (đừng bỏ — chặn lỗi sớm)
```bash
nvidia-smi                                  # thấy GPU 12GB
docker compose version || docker-compose version   # v2 hay v1?
# ⭐ QUAN TRỌNG NHẤT — Python backend có torch không:
docker run --rm nvcr.io/nvidia/tritonserver:24.08-py3 \
    python3 -c "import torch; print('torch', torch.__version__, torch.cuda.is_available())"
```
- Ra `torch 2.x True` → đi tiếp.
- `No module named torch` → **DỪNG**, thêm `torch`+`torchaudio` vào `triton_service/model_requirements.txt` (báo Claude).

## 2️⃣ Token + khởi động
```bash
export HF_TOKEN=hf_xxx          # token MỚI (token cũ đã lộ)
cd triton_service
bash run_server.sh              # lần đầu 5–15 phút (build + tải model)
```
✅ Chờ thấy: `track2_emotion ... READY` · `track1_acr ... READY` · `track3_sim ... READY` + `Uvicorn ... 8080`

## 3️⃣ Terminal MỚI — kiểm tra sống
```bash
ssh nhandt23@<ip-server>
curl http://localhost:8080/health                 # {"status":"ok"}
nvidia-smi                                         # VRAM dùng ~5-7GB
```

## 4️⃣ Test điểm — Track 2 TRƯỚC (xác nhận pipeline)
```bash
cd /home/nhandt23/project/VoiceMOS
WAV=triton_service/loadtest/samples/$(ls triton_service/loadtest/samples/ | head -1)

curl -s -F "file=@$WAV" -F "target_emotion=happy" localhost:8080/track2 | python3 -m json.tool
curl -s -F "file_a=@$WAV" localhost:8080/track1 | python3 -m json.tool
curl -s -F "file_test=@$WAV" -F "file_ref=@$WAV" localhost:8080/track3 | python3 -m json.tool
```
⭐ **Đối chiếu điểm Track 2 với `api_service/` cũ — phải GẦN GIỐNG** (cùng ckpt exp08). Lệch nhiều = có bug port.

## 5️⃣ (tùy chọn) UI — chạy trên MÁY BẠN
```powershell
cd "d:\VFS\VoiceMOS Challenge 2026\triton_service\ui"
pip install -r requirements.txt
$env:GATEWAY_URL = "http://<ip-server>:8080"
python app_ui.py                # http://localhost:7860
```

## 6️⃣ (tùy chọn) Locust loadtest
```bash
pip install -r triton_service/loadtest/requirements.txt
cd triton_service/loadtest && mkdir -p results
locust -f locustfile.py --host http://localhost:8080 \
       --users 20 --spawn-rate 2 --run-time 2m --headless --csv results/run
```

## ⏹️ Dừng
```bash
cd /home/nhandt23/project/VoiceMOS/triton_service && docker compose down
```

---

## 🚨 Nếu lỗi
| Thấy gì | Làm gì |
|---|---|
| `No module named torch` | thêm torch vào `model_requirements.txt` (bước 1) |
| model `UNAVAILABLE` | xem log Triton; check `echo $HF_TOKEN` |
| `502 Bad Gateway` | Triton chưa READY — đợi, `curl :8000/v2/health/ready` |
| `CUDA out of memory` | `count: 2`→`1` trong config.pbtxt |
| Track 3 lỗi shape/index | ECAPA batch chưa chắc — báo Claude sửa loop |
| `could not select device driver` | cài NVIDIA Container Toolkit (guide Bước 2) |
| GPU không nhận trong container | thử thêm `runtime: nvidia` / kiểm compose v2 |

**Ưu tiên test: Track 2 → Track 1 → Track 3.** Track 2 ra điểm khớp api_service = thắng lớn.
