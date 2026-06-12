# Triton Service — Những điểm cần hiểu để nắm hệ thống

> Cập nhật ngày: 12/6/2026

---

## 1. Luồng dữ liệu

```
Bạn  →  upload file .wav
         ↓
   FastAPI gateway :8080        ← "cửa trước" — nhận file, kiểm tra, chuyển tiếp
         ↓
   Triton server :8000          ← "nhà máy" — chạy model AI, tính điểm
         ↓
   model.py (Python backend)    ← code thật: decode audio → WavLM → heads → JSON
         ↓
   JSON điểm trả về             ← {qmos, emos, cat, vad, ...}
```

**Gateway không chạy model** — nó chỉ nhận file từ bạn và chuyển sang Triton.  
**Triton không biết HTTP thông thường** — gateway dùng `tritonclient` để nói chuyện với Triton.

---

## 2. Tại sao cần 2 tầng (gateway + Triton)?

| | FastAPI gateway | Triton |
|---|---|---|
| Làm gì | Nhận upload, kiểm tra file, route đúng track | Chạy model AI thật |
| Biết gì | HTTP multipart, JSON | Tensor, batch, GPU |
| Ai gọi | Người dùng / client | Gateway |

Triton không nhận file `.wav` trực tiếp — nó chỉ nhận **tensor** (mảng số). Gateway là người dịch từ "file wav" sang "tensor bytes" rồi đưa vào Triton.

---

## 3. Model được nạp ở đâu, khi nào

```
docker-compose up
      ↓
Triton khởi động → đọc model_repository/
      ↓
Gọi initialize() trong model.py  ← 1 LẦN DUY NHẤT
      ↓  (tải checkpoint từ HuggingFace, nạp vào GPU VRAM)
Server báo "READY"
      ↓
Người dùng gửi request → Triton gọi execute()  ← MỖI LẦN có request
```

`initialize()` chỉ chạy 1 lần khi server bật → model nằm sẵn trong VRAM → mỗi request chỉ tốn thời gian tính toán, không tốn thời gian tải model.

---

## 4. Dynamic batching — 1 dòng tóm tắt

> Thay vì GPU xử lý 1 audio rồi nghỉ, rồi 1 audio rồi nghỉ... → **gom 8 audio xử lý cùng lúc**, thời gian như nhau nhưng ra 8 kết quả.

Điều kiện để làm được: phải **pad** audio ngắn cho bằng audio dài nhất + dùng **attention_mask** để model biết đâu là thật, đâu là padding.

```protobuf
max_batch_size: 8
dynamic_batching {
  max_queue_delay_microseconds: 5000   # chờ tối đa 5ms để gom đủ batch
}
```

---

## 5. 3 model trong VRAM cùng lúc

```
GPU 12GB
├── track2_emotion  (~3 GB)   ← WavLM-large + audeering + heads + UTMOS
├── track1_acr      (~1-2 GB) ← URGENT-MOS
└── track3_sim      (~0.5 GB) ← ECAPA-TDNN × 2
                    ─────────
                    ~5-6 GB tổng → còn dư cho batch
```

Triton nạp cả 3 model khi khởi động, không cần load/unload mỗi request.

---

## 6. Locust — đo sức chịu đựng

Locust không test xem **kết quả đúng không** (đó là việc của unit test).  
Locust test **server có chịu được tải cao không**:

- Khi 20 người cùng gửi audio → RPS là bao nhiêu?
- Latency p95 có vượt ngưỡng chấp nhận được không?
- Có request nào lỗi (timeout, crash) không?

Kịch bản so sánh:

```bash
# Tắt batching: max_batch_size: 1
locust --users 20 --run-time 1m --headless --csv results/batch1

# Bật batching: max_batch_size: 8
locust --users 20 --run-time 1m --headless --csv results/batch8
```

→ So sánh throughput (audio/s) để chứng minh dynamic batching có hiệu quả.

---

## 7. File nào làm gì

| File | Vai trò |
|---|---|
| [docker-compose.yml](../triton_service/docker-compose.yml) | Khởi động Triton + gateway cùng lúc |
| [run_server.sh](../triton_service/run_server.sh) | Script bash chạy compose trên Linux server |
| [gateway/app.py](../triton_service/gateway/app.py) | Nhận upload → gọi Triton → trả JSON |
| [model_repository/track2_emotion/config.pbtxt](../triton_service/model_repository/track2_emotion/config.pbtxt) | Khai báo input/output + bật dynamic batching |
| [model_repository/track2_emotion/1/model.py](../triton_service/model_repository/track2_emotion/1/model.py) | Code AI thật: nạp exp08, tính 6 cột |
| [loadtest/locustfile.py](../triton_service/loadtest/locustfile.py) | Giả lập người dùng đồng thời, đo RPS + latency |
| [ui/app_ui.py](../triton_service/ui/app_ui.py) | UI Gradio 4 tab (3 track + batch) — demo trực quan, gọi gateway :8080 |

---

## 8. Lệnh duy nhất cần chạy trên server

```bash
export HF_TOKEN=hf_xxx          # để tải checkpoint Track 2 từ HuggingFace
bash triton_service/run_server.sh
```

Tất cả còn lại (tải model, cài thư viện, cấu hình GPU) đã được xử lý trong Dockerfile và `initialize()`.

---

## 9. Kiểm tra nhanh sau khi server bật

```bash
# Gateway còn sống không?
curl http://localhost:8080/health

# Track 2 chạy được không?
curl -F "file=@sample.wav" -F "target_emotion=happy" http://localhost:8080/track2

# Triton trực tiếp (không qua gateway)
curl http://localhost:8000/v2/health/ready

# GPU còn dư VRAM không?
nvidia-smi
```
