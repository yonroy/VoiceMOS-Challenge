"""
Smoke test cho scaffold FastAPI (KHÔNG cần torch / model nặng).
Xác minh: app boot · /health · /docs · multipart upload · _load_audio (librosa).
Endpoint predict sẽ trả 500 (thiếu torch local) — đó là KỲ VỌNG khi test local không GPU.

Chạy:  cd api_service ; python tests/smoke_test.py
"""
import io
import os
import sys

import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.main import app, _load_audio  # noqa: E402

client = TestClient(app)


def _wav_bytes(seconds=1.0, sr=16000):
    t = np.linspace(0, seconds, int(seconds * sr), endpoint=False)
    wave = (0.1 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)  # tone 220Hz
    buf = io.BytesIO()
    sf.write(buf, wave, sr, format="WAV")
    return buf.getvalue()


def check(name, cond):
    print(("✅" if cond else "❌"), name)
    return cond


ok = True

# 1) routes không cần model
r = client.get("/health")
ok &= check("GET /health == 200 & status ok", r.status_code == 200 and r.json().get("status") == "ok")
ok &= check("GET / liệt kê endpoints", client.get("/").status_code == 200)
ok &= check("GET /docs (Swagger) == 200", client.get("/docs").status_code == 200)
ok &= check("GET /openapi.json có 3 track", all(
    p in client.get("/openapi.json").json()["paths"] for p in ["/track1", "/track2", "/track3"]))

# 2) _load_audio: ghi wav tạm → librosa đọc đúng độ dài
class _Up:  # giả UploadFile tối thiểu
    filename = "tone.wav"
    def __init__(self, b): self.file = io.BytesIO(b)

wave = _load_audio(_Up(_wav_bytes(1.0)))
ok &= check(f"_load_audio trả ~16000 mẫu (got {len(wave)})", 15000 < len(wave) < 17000)

# 3) POST multipart tới /track2: routing + parse file chạy; predict fail vì thiếu torch (kỳ vọng)
r = client.post("/track2", files={"file": ("tone.wav", _wav_bytes(1.0), "audio/wav")},
                data={"target_emotion": "happy"})
detail = str(r.json())
ok &= check("POST /track2 nhận file & vào handler (200 hoặc 500-thiếu-torch)",
            r.status_code == 200 or ("torch" in detail.lower() or "no module" in detail.lower()))
print("   → /track2 status:", r.status_code, "| detail head:", detail[:90])

print("\n", "🎉 SCAFFOLD OK" if ok else "⚠️ CÓ LỖI — xem ❌ ở trên")
sys.exit(0 if ok else 1)
