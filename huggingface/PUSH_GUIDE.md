# Hướng dẫn đẩy dự án lên Hugging Face

> Mentor yêu cầu đẩy **code + checkpoint + UI** lên HF (9/6/2026). Thư mục này chứa sẵn mọi thứ cần.
> ⚠️ **Trước tiên:** thay `YOUR_HF_USERNAME` thành username HF thật trong: `upload_checkpoints.py`,
> `model_card_README.md`, `space/README.md`, `space/app.py`. (Báo Claude username → sửa giúp 1 lượt.)

## Thứ tự đẩy (quan trọng — Space phụ thuộc Model repo)

```
Bước 1: Model repo (checkpoint)  →  Bước 2: Space (UI tải ckpt từ Model repo)  →  Bước 3: code
```

## Chuẩn bị 1 lần
```bash
pip install -U huggingface_hub
huggingface-cli login        # dán token WRITE (Settings → Access Tokens → New token → Write)
```

## Bước 1 — Đẩy checkpoint (HF Models)
```bash
python huggingface/upload_checkpoints.py
```
- Tạo repo `…/voicemos2026-track2-emotion` + đẩy `ft_emotion_full_20epoch.pt` (1.27GB),
  `ft_qmos_utmos.pt` (411MB), `ft_joint_full.pt` (1.9GB) + model card.
- File lớn → cần mạng khỏe; lệnh có resume nếu đứt.

## Bước 2 — Tạo Space (UI Gradio)
1. huggingface.co → **New Space** → SDK **Gradio** → đặt tên `voicemos2026-demo`.
2. Clone Space về rồi copy 3 file trong `huggingface/space/` vào:
   ```bash
   git clone https://huggingface.co/spaces/YOUR_HF_USERNAME/voicemos2026-demo
   cp huggingface/space/{app.py,requirements.txt,README.md} voicemos2026-demo/
   cd voicemos2026-demo && git add . && git commit -m "VoiceMOS 2026 demo 3 track" && git push
   ```
3. Space tự build. ⚠️ **CPU free chạy chậm** (WavLM-large) → cân nhắc Settings → upgrade GPU.

## Bước 3 — Đẩy code (tùy chọn)
- Có thể đẩy `kaggle_baseline/` (pipeline 3 track + experiment) thành 1 repo Models riêng,
  hoặc thêm thư mục `code/` vào Space.
- ⚠️ **KHÔNG** đẩy data thô (BTC VoiceMOS / ESD / DailyTalk) — license riêng.

## License (khai báo khi đẩy)
WavLM MIT · SAILER Open RAIL · audeering CC BY-NC-SA 4.0 · emotion2vec · UTMOS.
→ Phần lớn **phi thương mại** → repo để **CC BY-NC-SA 4.0** (đã set trong model card + Space README).

## File trong thư mục này
| File | Vai trò |
|---|---|
| `upload_checkpoints.py` | Script đẩy checkpoint lên HF Models |
| `model_card_README.md` | Model card (README) cho repo checkpoint |
| `space/app.py` | App Gradio 3 tab cho Space (tải ckpt từ Model repo) |
| `space/requirements.txt` | Dependencies Space |
| `space/README.md` | README Space (có YAML header bắt buộc) |
