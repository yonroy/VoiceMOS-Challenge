# FastAPI service cho VoiceMOS Challenge 2026 — 3 track.
# Lazy-load: mỗi track nạp model ở request đầu tiên của track đó (boot Space nhanh).
import os
import tempfile
import traceback
from typing import Optional

import librosa
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

SR = 16000

app = FastAPI(
    title="VoiceMOS 2026 — 3-Track MOS API",
    version="1.0.0",
    description=(
        "Bộ chấm MOS tự động cho 3 track của VoiceMOS Challenge 2026.\n\n"
        "- **/track1** — Speech Enhancement → ACR (+ CCR nếu gửi 2 file)\n"
        "- **/track2** — Emotional TTS → QMOS + EMOS + CAT + VAD (6 cột)\n"
        "- **/track3** — Codec synthesis → speaker_sim + accent_sim (cần test + reference)\n\n"
        "Model nạp ở request ĐẦU TIÊN của mỗi track (chờ tải ~1–2 phút lần đầu)."
    ),
)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


def _load_audio(upload: UploadFile):
    suffix = os.path.splitext(upload.filename or "")[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(upload.file.read())
        path = tmp.name
    try:
        wave, _ = librosa.load(path, sr=SR, mono=True)
    finally:
        os.unlink(path)
    if wave is None or len(wave) == 0:
        raise HTTPException(status_code=400, detail=f"Không đọc được audio: {upload.filename}")
    return wave


def _guard(fn, *args):
    try:
        return fn(*args)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}\n{traceback.format_exc()[-600:]}")


@app.get("/")
def root():
    return {"service": "VoiceMOS 2026 3-Track MOS API", "docs": "/docs",
            "endpoints": ["/track1", "/track2", "/track3", "/health"]}


@app.get("/health")
def health():
    return {"status": "ok", "tracks": [1, 2, 3]}


@app.post("/track1", summary="ACR (+ CCR) cho speech enhancement")
async def track1_ep(
    file_a: UploadFile = File(..., description="Audio A (bắt buộc) → ACR"),
    file_b: Optional[UploadFile] = File(None, description="Audio B (tùy chọn) → ACR(B) + CCR(A vs B)"),
):
    from .tracks import track1
    wa = _load_audio(file_a)
    wb = _load_audio(file_b) if file_b is not None else None
    return _guard(track1.predict, wa, wb)


@app.post("/track2", summary="QMOS + EMOS + CAT + VAD cho emotional TTS")
async def track2_ep(
    file: UploadFile = File(..., description="Audio giọng TTS"),
    target_emotion: Optional[str] = Form(None, description="angry|happy|neutral|sad|surprised — để bật EMOS"),
):
    from .tracks import track2
    wave = _load_audio(file)
    return _guard(track2.predict, wave, target_emotion)


@app.post("/track3", summary="speaker_sim + accent_sim cho codec synthesis")
async def track3_ep(
    file_test: UploadFile = File(..., description="Audio cần đánh giá"),
    file_ref: UploadFile = File(..., description="Audio tham chiếu (reference)"),
):
    from .tracks import track3
    wt = _load_audio(file_test)
    wr = _load_audio(file_ref)
    return _guard(track3.predict, wt, wr)
