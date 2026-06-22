---
title: VoiceMOS 2026 — Demo 3 Track
emoji: 🎙️
colorFrom: indigo
colorTo: pink
sdk: gradio
sdk_version: 6.17.3
app_file: app.py
pinned: false
license: cc-by-nc-sa-4.0
---

# VoiceMOS Challenge 2026 — Demo Gradio 3 Track

Một Space cho cả 3 track của VoiceMOS Challenge 2026:

- **Track 1** · Speech Enhancement → ACR (chất lượng) + CCR (so cặp). Model: URGENT-MOS.
- **Track 2** · Emotional TTS → EMOS / CAT / VAD. Model **tốt nhất exp08** (WavLM fine-tune + audeering).
- **Track 3** · Speaker/Accent → spk_sim / acc_sim. Model: ECAPA fine-tuned (baseline BTC).

## Cấu hình cần thiết

- **Checkpoint Track 2** được tải tự động từ HF Models repo
  [`yonroy/voicemos2026-track2-emotion`](https://huggingface.co/yonroy/voicemos2026-track2-emotion)
  qua `hf_hub_download` (xem `app.py`).
- Track 1 & Track 3 tự clone repo/model lúc chạy lần đầu.

## ⚠️ Lưu ý phần cứng

Space **CPU free (16GB RAM)** chạy được nhưng **chậm** (WavLM-large/URGENT-MOS/ECAPA đều nặng).
Lazy-load (chỉ nạp model khi bấm tab đó) giúp tiết kiệm RAM. Để mượt nên **nâng Space lên GPU**.

## License

CC BY-NC-SA 4.0 — phi thương mại (kế thừa SAILER Open RAIL + audeering CC BY-NC-SA).
