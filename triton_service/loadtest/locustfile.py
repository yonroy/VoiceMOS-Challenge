"""
Locust loadtest cho VoiceMOS 2026 gateway (FastAPI :8080).

Cài: pip install -r requirements.txt
Chạy web UI:
    locust -f locustfile.py --host http://localhost:8080
Chạy headless (tự động):
    locust -f locustfile.py --host http://localhost:8080 \
           --users 20 --spawn-rate 2 --run-time 2m --headless \
           --csv results/locust

Cần đặt ít nhất 1 file .wav vào thư mục SAMPLE_DIR (mặc định ./samples/).
Đường dẫn cấu hình qua biến môi trường:
    SAMPLE_DIR       thư mục chứa wav mẫu (mặc định ./samples)
    LOCUST_EMOTION   cảm xúc target cho Track 2 (mặc định happy)
"""
import os
import glob
import random
from locust import HttpUser, task, between, events

SAMPLE_DIR = os.environ.get("SAMPLE_DIR", "./samples")
EMOTION = os.environ.get("LOCUST_EMOTION", "happy")

_wavs: list[bytes] = []


@events.init.add_listener
def load_samples(environment, **_):
    paths = glob.glob(os.path.join(SAMPLE_DIR, "**", "*.wav"), recursive=True)
    if not paths:
        print(f"[locust] CẢNH BÁO: không tìm thấy .wav trong {SAMPLE_DIR}. "
              f"Đặt ít nhất 1 file vào {SAMPLE_DIR}/ rồi chạy lại.")
        return
    for p in paths[:100]:  # đọc tối đa 100 file vào RAM để test nhanh
        with open(p, "rb") as f:
            _wavs.append(f.read())
    print(f"[locust] Đã nạp {len(_wavs)} file wav từ {SAMPLE_DIR}")


def _pick() -> bytes:
    if not _wavs:
        raise RuntimeError("Không có file wav mẫu — xem SAMPLE_DIR.")
    return random.choice(_wavs)


class VoiceMOSUser(HttpUser):
    wait_time = between(0.1, 0.5)

    @task(5)
    def score_track2(self):
        raw = _pick()
        with self.client.post(
            "/track2",
            files={"file": ("sample.wav", raw, "audio/wav")},
            data={"target_emotion": EMOTION},
            catch_response=True,
        ) as resp:
            if resp.status_code == 200:
                resp.success()
            else:
                resp.failure(f"HTTP {resp.status_code}: {resp.text[:200]}")

    @task(2)
    def score_track1_acr(self):
        raw = _pick()
        with self.client.post(
            "/track1",
            files={"file_a": ("sample.wav", raw, "audio/wav")},
            catch_response=True,
        ) as resp:
            if resp.status_code == 200:
                resp.success()
            else:
                resp.failure(f"HTTP {resp.status_code}: {resp.text[:200]}")

    @task(2)
    def score_track1_ccr(self):
        raw_a, raw_b = _pick(), _pick()
        with self.client.post(
            "/track1",
            files={
                "file_a": ("a.wav", raw_a, "audio/wav"),
                "file_b": ("b.wav", raw_b, "audio/wav"),
            },
            catch_response=True,
        ) as resp:
            if resp.status_code == 200:
                resp.success()
            else:
                resp.failure(f"HTTP {resp.status_code}: {resp.text[:200]}")

    @task(2)
    def score_track3(self):
        raw_t, raw_r = _pick(), _pick()
        with self.client.post(
            "/track3",
            files={
                "file_test": ("test.wav", raw_t, "audio/wav"),
                "file_ref": ("ref.wav", raw_r, "audio/wav"),
            },
            catch_response=True,
        ) as resp:
            if resp.status_code == 200:
                resp.success()
            else:
                resp.failure(f"HTTP {resp.status_code}: {resp.text[:200]}")

    @task(1)
    def health_check(self):
        self.client.get("/health")
