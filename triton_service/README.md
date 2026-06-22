# Triton Service — VoiceMOS 2026 (3 Track)

Phục vụ **3 track** bằng **NVIDIA Triton Inference Server** + **FastAPI gateway**, chạy trên server Linux nội bộ với **1 GPU 12GB**.

```
client ──POST multipart .wav──> FastAPI gateway :8080 ──tritonclient──> Triton :8000
                                                                          ├ track2_emotion  (WavLM ft + audeering, dyn-batch)
                                                                          ├ track1_acr      (URGENT-MOS)
                                                                          └ track3_sim      (ECAPA-TDNN spk+acc)
                              <──────── JSON score ────────────────────────
         Triton metrics :8002   ·   1×GPU 12GB   ·   docker-compose
```

---

## Cấu trúc thư mục

```
triton_service/
├── docker-compose.yml         # khởi động Triton + gateway cùng lúc
├── run_server.sh              # wrapper bash để chạy trên Linux server
├── run_local.ps1              # tham khảo Windows (test local)
├── Dockerfile                 # image Triton + transformers/librosa
├── model_requirements.txt     # lib cài thêm vào image server
├── model_repository/          # Triton đọc thư mục này
│   ├── track2_emotion/
│   │   ├── config.pbtxt       # max_batch_size:8 + dynamic_batching
│   │   └── 1/model.py         # WavLM ft + audeering + UTMOS, batch thật
│   ├── track1_acr/
│   │   ├── config.pbtxt
│   │   └── 1/model.py         # URGENT-MOS (ACR + CCR)
│   └── track3_sim/
│       ├── config.pbtxt
│       └── 1/model.py         # ECAPA-TDNN (spk_sim + acc_sim)
├── gateway/
│   ├── app.py                 # FastAPI: /track1 /track2 /track3 /health
│   ├── Dockerfile
│   └── requirements.txt
├── loadtest/
│   ├── locustfile.py          # Locust: ramp user, đo RPS + latency
│   └── requirements.txt
├── client/
│   ├── batch_client.py        # CLI bắn audio 1 file / cả thư mục
│   └── requirements.txt
├── ui/
│   ├── app_ui.py              # Gradio UI 4 tab (3 track + batch), gọi gateway :8080
│   └── requirements.txt
└── bench/
    ├── benchmark_vs_fastapi.py
    └── requirements.txt
```

---

## Khởi động nhanh trên Linux server

```bash
# 0) Vào thư mục triton_service/
cd /home/<user>/project/VoiceMOS/triton_service

# 1) Đặt HF Token (nếu checkpoint private)
export HF_TOKEN=hf_xxx

# 2) Khởi động Triton + gateway (build image lần đầu ~vài phút)
bash run_server.sh

# Server sẵn sàng khi log hiện:
#   track2_emotion ... READY
#   track1_acr     ... READY
#   track3_sim     ... READY
#   gateway        : Uvicorn running on http://0.0.0.0:8080
```

### Kiểm tra nhanh

```bash
# Health
curl http://localhost:8080/health

# Track 2 — 6 cột cảm xúc
curl -F "file=@sample.wav" -F "target_emotion=happy" http://localhost:8080/track2

# Track 1 — ACR
curl -F "file_a=@clean.wav" http://localhost:8080/track1

# Track 1 — CCR (so sánh cặp)
curl -F "file_a=@clean.wav" -F "file_b=@noisy.wav" http://localhost:8080/track1

# Track 3 — spk_sim + acc_sim
curl -F "file_test=@tts.wav" -F "file_ref=@ref.wav" http://localhost:8080/track3

# Triton health (trực tiếp, bỏ qua gateway)
curl http://localhost:8000/v2/health/ready

# GPU usage
nvidia-smi
```

---

## Loadtest bằng Locust

```bash
# Cài
pip install -r triton_service/loadtest/requirements.txt

# Đặt file .wav mẫu vào thư mục samples/
mkdir -p triton_service/loadtest/samples
cp some.wav triton_service/loadtest/samples/

cd triton_service/loadtest

# Web UI — mở http://localhost:8089, đặt host = http://localhost:8080
locust -f locustfile.py --host http://localhost:8080

# Headless — ramp 20 user, chạy 2 phút, xuất CSV
locust -f locustfile.py --host http://localhost:8080 \
       --users 20 --spawn-rate 2 --run-time 2m --headless \
       --csv results/locust
```

**Thí nghiệm dynamic batching**: đổi `max_batch_size` từ 1 → 8 trong config.pbtxt của track muốn test, restart server, so sánh RPS:
```bash
# max_batch_size: 1 (tắt batching)
sed -i 's/max_batch_size: 8/max_batch_size: 1/' model_repository/track2_emotion/config.pbtxt
bash run_server.sh &
locust -f loadtest/locustfile.py --host http://localhost:8080 \
       --users 20 --spawn-rate 2 --run-time 1m --headless --csv results/batch1

# max_batch_size: 8 (bật batching)
sed -i 's/max_batch_size: 1/max_batch_size: 8/' model_repository/track2_emotion/config.pbtxt
bash run_server.sh &
locust -f loadtest/locustfile.py --host http://localhost:8080 \
       --users 20 --spawn-rate 2 --run-time 1m --headless --csv results/batch8
```

---

## UI Gradio (demo trực quan)

UI [ui/app_ui.py](ui/app_ui.py) là **client** gọi gateway `:8080` — load audio → xem điểm + latency, không nạp model. 4 tab:

| Tab | Input | Output |
|---|---|---|
| 🎯 Track 2 | 1 audio + emotion | 6 cột (QMOS, EMOS, CAT, VAD) + latency |
| 🔊 Track 1 | audio A (+ B) | ACR (+ ACR_B, CCR) + latency |
| 🗣️ Track 3 | audio Test + Ref | spk_sim, acc_sim, cosine + latency |
| 📦 Batch | nhiều audio (Track 2 / Track 1) | bảng điểm + throughput/latency p50/p95 |

```bash
pip install -r triton_service/ui/requirements.txt
python triton_service/ui/app_ui.py    # http://localhost:7860
# Trỏ ô "Gateway URL" tới server, vd http://<ip-server>:8080
# Hoặc đặt sẵn: GATEWAY_URL=http://<ip>:8080 python triton_service/ui/app_ui.py
```

---

## Dynamic batching — cách hoạt động

| Tầng | Cơ chế |
|---|---|
| **Triton** | `max_batch_size: 8` + `dynamic_batching { max_queue_delay_microseconds: 5000 }` — gom tối đa 8 request / 5ms chờ → 1 `execute()` call |
| **Input audio** | Kiểu `TYPE_STRING shape [1]` — mỗi request đóng 1 blob bytes bất kể độ dài, nên Triton gom batch được |
| **model.py Track 2** | `execute()` nhận N request → decode N audio → **pad tới max_length + attention_mask** → **1 WavLM forward gộp** → split kết quả. Audeering + QMOS vẫn loop nhưng nhẹ hơn WavLM |
| **Track 3** | ECAPA-TDNN cũng pad theo cặp (test+ref) → 1 forward gộp |
| **Track 1** | URGENT-MOS có sẵn `infer(list)` → batch tự nhiên |

---

## Endpoints gateway

| Method | Path | Input | Output |
|---|---|---|---|
| GET | `/health` | — | `{"status":"ok", "triton":"..."}` |
| POST | `/track2` | `file` (wav) + `target_emotion` (optional) | `{qmos, emos, cat{5}, vad{val,aro,dom}, perceived_emotion}` |
| POST | `/track1` | `file_a` + `file_b` (optional) | `{acr_a}` hoặc `{acr_a, acr_b, ccr}` |
| POST | `/track3` | `file_test` + `file_ref` | `{spk_sim, acc_sim, cosine}` |

---

## Cổng dịch vụ

| Cổng | Dịch vụ |
|---|---|
| 8080 | FastAPI gateway (upload audio → score) |
| 8000 | Triton HTTP (KServe v2) |
| 8001 | Triton gRPC |
| 8002 | Triton metrics (Prometheus) |

---

## Ghi chú

- **Checkpoint Track 2** (`ft_emotion_full_20epoch.pt`): nằm trên HF repo `tranminhtoan140601/voicemos2026-track2-emotion`. Server tự tải khi `initialize()`. Repo private → đặt `HF_TOKEN`.
- **VRAM**: 3 model cùng lúc ~5–7GB (Track2 ~3GB + Track1 ~1–2GB + Track3 ~0.5GB) → vừa 12GB.
- **Tăng instance**: nếu VRAM còn dư, tăng `count: 2` trong `config.pbtxt` của track muốn tăng throughput.
