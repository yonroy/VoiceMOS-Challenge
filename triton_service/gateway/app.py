"""
FastAPI gateway: nhận multipart audio → gọi Triton → trả JSON score.

Endpoints:
  POST /track2  file (wav/flac/mp3) + target_emotion (optional) → 6-col JSON
  POST /track1  file_a + file_b (optional)                      → acr/ccr JSON
  POST /track3  file_test + file_ref                            → spk/acc/cos JSON
  GET  /health                                                  → {"status":"ok"}
"""
import json
import os

import numpy as np
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import JSONResponse

TRITON_URL = os.environ.get("TRITON_URL", "localhost:8000")

app = FastAPI(title="VoiceMOS 2026 Gateway")


def _triton_client():
    import tritonclient.http as httpclient
    return httpclient, httpclient.InferenceServerClient(url=TRITON_URL)


def _str_input(httpclient, name: str, value: bytes) -> object:
    """Đóng gói bytes thành tensor TYPE_STRING shape [1,1] cho Triton."""
    arr = np.array([[value]], dtype=object)
    t = httpclient.InferInput(name, arr.shape, "BYTES")
    t.set_data_from_numpy(arr)
    return t


def _call_triton(model: str, inputs: list, output_name: str) -> dict:
    httpclient, client = _triton_client()
    outputs = [httpclient.InferRequestedOutput(output_name)]
    try:
        result = client.infer(model, inputs=inputs, outputs=outputs)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Triton error: {e}")
    raw = result.as_numpy(output_name).reshape(-1)[0]
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    return json.loads(raw)


@app.get("/health")
def health():
    try:
        httpclient, client = _triton_client()
        if not client.is_server_live():
            raise HTTPException(status_code=503, detail="Triton not live")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))
    return {"status": "ok", "triton": TRITON_URL}


@app.post("/track2")
async def track2(
    file: UploadFile = File(...),
    target_emotion: str = Form(default=""),
):
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="file rỗng")
    httpclient, _ = _triton_client()
    inputs = [
        _str_input(httpclient, "AUDIO_BYTES", raw),
        _str_input(httpclient, "TARGET_EMOTION", target_emotion.encode()),
    ]
    return JSONResponse(_call_triton("track2_emotion", inputs, "RESULT_JSON"))


@app.post("/track1")
async def track1(
    file_a: UploadFile = File(...),
    file_b: UploadFile = File(default=None),
):
    raw_a = await file_a.read()
    if not raw_a:
        raise HTTPException(status_code=400, detail="file_a rỗng")
    httpclient, _ = _triton_client()
    inputs = [_str_input(httpclient, "AUDIO_BYTES_A", raw_a)]
    if file_b is not None:
        raw_b = await file_b.read()
        if raw_b:
            inputs.append(_str_input(httpclient, "AUDIO_BYTES_B", raw_b))
    return JSONResponse(_call_triton("track1_acr", inputs, "RESULT_JSON"))


@app.post("/track3")
async def track3(
    file_test: UploadFile = File(...),
    file_ref: UploadFile = File(...),
):
    raw_t = await file_test.read()
    raw_r = await file_ref.read()
    if not raw_t or not raw_r:
        raise HTTPException(status_code=400, detail="file_test hoặc file_ref rỗng")
    httpclient, _ = _triton_client()
    inputs = [
        _str_input(httpclient, "AUDIO_BYTES_TEST", raw_t),
        _str_input(httpclient, "AUDIO_BYTES_REF", raw_r),
    ]
    return JSONResponse(_call_triton("track3_sim", inputs, "RESULT_JSON"))
