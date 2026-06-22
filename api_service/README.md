---
title: VoiceMOS 2026 3-Track MOS API
emoji: 🎙️
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: cc-by-nc-sa-4.0
---

# VoiceMOS Challenge 2026 — 3-Track MOS API

REST API (FastAPI) chấm MOS tự động cho cả 3 track. Đóng gói **Docker** để deploy lên **Hugging Face Space**.
Mỗi track **lazy-load** (nạp model ở request đầu tiên) → Space khởi động nhanh, chỉ tốn RAM cho track được gọi.

| Endpoint | Track | Input | Output |
|---|---|---|---|
| `POST /track1` | Speech Enhancement | `file_a` (+ `file_b` tùy chọn) | `acr_a` (+ `acr_b`, `ccr`) |
| `POST /track2` | Emotional TTS | `file` (+ `target_emotion`) | `qmos`, `emos`, `cat{5}`, `vad{val,aro,dom}` |
| `POST /track3` | Codec synthesis | `file_test` + `file_ref` | `spk_sim`, `acc_sim`, `cosine` |
| `GET /health` | — | — | trạng thái |
| `GET /docs` | — | — | Swagger UI (thử trực tiếp) |

## Model dùng
- **Track 1:** URGENT-MOS (`urgent-challenge/urgent-mos-f1c1m5dcorpus`) — tự clone repo + tải checkpoint.
- **Track 2:** QMOS ← UTMOS (SpeechMOS) · EMOS/CAT/VAD ← **exp08** (WavLM fine-tune + audeering),
  checkpoint kéo từ HF repo `yonroy/voicemos2026-track2-emotion/ft_emotion_full_20epoch.pt`.
- **Track 3:** ECAPA-TDNN fine-tuned (clone `vmc2026-baselines`, dùng checkpoint kèm repo).

## Biến môi trường (tùy chọn)
| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `T2_HF_REPO` | `yonroy/voicemos2026-track2-emotion` | HF repo chứa checkpoint Track 2 |
| `T2_HF_CKPT` | `ft_emotion_full_20epoch.pt` | tên file checkpoint exp08 |
| `T1_CKPT` | `urgent-challenge/urgent-mos-f1c1m5dcorpus` | checkpoint URGENT-MOS |
| `MODELS_DIR` | `/home/user/models` | nơi clone repo + cache |
| `HF_TOKEN` | — | đặt **Secret** nếu HF repo để private |

## Chạy local (Docker)
```bash
cd api_service
docker build -t voicemos-api .
docker run -p 7860:7860 voicemos-api
# mở http://localhost:7860/docs
```

## Chạy local (không Docker)
```bash
cd api_service
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 7860 --reload
```

## Gọi thử bằng curl
```bash
# Track 1 — chỉ ACR
curl -F "file_a=@a.wav" http://localhost:7860/track1
# Track 1 — ACR + CCR
curl -F "file_a=@a.wav" -F "file_b=@b.wav" http://localhost:7860/track1

# Track 2 — 6 cột (có EMOS khi truyền target_emotion)
curl -F "file=@tts.wav" -F "target_emotion=happy" http://localhost:7860/track2

# Track 3 — speaker/accent similarity
curl -F "file_test=@test.wav" -F "file_ref=@ref.wav" http://localhost:7860/track3
```

Ví dụ response `/track2`:
```json
{
  "qmos": 3.41,
  "cat": {"angry":0.03,"happy":0.71,"neutral":0.12,"sad":0.05,"surprised":0.09},
  "perceived_emotion": "happy",
  "vad": {"valence": 3.92, "arousal": 3.40, "dominance": 3.18},
  "emos": 3.88, "target_emotion": "happy", "emos_match": true
}
```

## Deploy lên Hugging Face Space

### Cách A — script tự động (khuyến nghị)
```powershell
pip install huggingface_hub
$env:HF_TOKEN = "hf_xxx"                 # token WRITE; nhớ REVOKE token cũ đã lộ
python push_to_hf_space.py               # tạo Space + upload (mặc định yonroy/voicemos2026-api)
# tùy chọn:
python push_to_hf_space.py --private --hardware cpu-upgrade --set-token-secret
```
Script tự: `create_repo(sdk=docker)` → `upload_folder` (bỏ qua rác/`*.pt`) → in link Space + `/docs`.
`--set-token-secret` đặt Secret `HF_TOKEN` cho Space (cần nếu repo checkpoint Track 2 để **private**).

### Cách B — thủ công
1. Tạo Space mới → **SDK = Docker**.
2. Push toàn bộ thư mục `api_service/` lên repo của Space (README.md này có sẵn frontmatter Docker).
3. Space tự build → API chạy ở `https://<user>-<space>.hf.space` → thử tại `/docs`.

> ⚙️ **Phần cứng:** cả 3 track đã được xác nhận **chạy được trên HF Space free CPU (16GB)** — bản demo Gradio
> trước đó (cùng code inference) chạy đủ 3 track trên free CPU. Trên CPU sẽ **chậm hơn GPU** (mỗi request vài chục
> giây, nhất là Track 1 URGENT-MOS + Track 2 WavLM-large). Lần đầu mỗi track còn tốn thời gian **tải model**.
> Muốn nhanh hơn → nâng hardware GPU (có phí) hoặc dùng Kaggle T4.
