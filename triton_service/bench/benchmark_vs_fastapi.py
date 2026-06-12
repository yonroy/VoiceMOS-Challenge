#!/usr/bin/env python3
# ─────────────────────────────────────────────────────────────────────────────
# Benchmark ĐỐI ĐẦU: Triton vs FastAPI — Track 2, CÔNG BẰNG.
#
# Cùng 1 bộ audio, cùng các mức concurrency (1/4/8/16...), bắn SONG SONG vào cả 2
# endpoint → đo **throughput (audio/giây)** + **latency p50/p95**. Đây mới là phép
# so có ý nghĩa (UI kéo-thả 1 file thì 2 bên gần như bằng nhau).
#
# Hai endpoint:
#   • Triton  : tritonclient HTTP (mặc định localhost:8000), input AUDIO_BYTES + TARGET_EMOTION.
#   • FastAPI : POST multipart /track2 (file + target_emotion) — service trong api_service/.
#
# Cài: pip install -r requirements.txt  (tritonclient[http], numpy; matplotlib để vẽ — tùy chọn)
#
# Ví dụ:
#   python benchmark_vs_fastapi.py --dir ./wavs --emotion happy \
#       --triton-url localhost:8000 \
#       --fastapi-base https://tranminhtoan140601-voicemos2026-api.hf.space \
#       --concurrency 1,4,8,16 --out bench.csv --plot bench.png
#
# Bỏ qua 1 bên bằng cách KHÔNG truyền --triton-url hoặc --fastapi-base.
# ─────────────────────────────────────────────────────────────────────────────
import argparse
import csv
import glob
import json
import os
import time
import uuid
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

MODEL = "track2_emotion"


# ───────────────────────── Triton backend ─────────────────────────
def _triton_client(url, ssl):
    import tritonclient.http as httpclient
    return httpclient, httpclient.InferenceServerClient(url=url, ssl=ssl)


def call_triton(url, ssl, name, raw, emotion):
    """1 request → latency_ms. Mỗi gọi tự mở client (thread-safe), giống cách FastAPI mở connection/req."""
    httpclient, client = _triton_client(url, ssl)

    def _str(n, v):
        arr = np.array([[v]], dtype=object)
        t = httpclient.InferInput(n, arr.shape, "BYTES")
        t.set_data_from_numpy(arr)
        return t

    inputs = [_str("AUDIO_BYTES", raw), _str("TARGET_EMOTION", emotion or "")]
    outputs = [httpclient.InferRequestedOutput("RESULT_JSON")]
    t0 = time.perf_counter()
    client.infer(MODEL, inputs=inputs, outputs=outputs)
    return (time.perf_counter() - t0) * 1000.0


# ───────────────────────── FastAPI backend ─────────────────────────
def call_fastapi(base, name, raw, emotion, timeout=180):
    """POST multipart /track2 → latency_ms (tự dựng body, không cần `requests`)."""
    boundary = uuid.uuid4().hex
    body = bytearray()
    if emotion:
        body += (f'--{boundary}\r\nContent-Disposition: form-data; name="target_emotion"'
                 f'\r\n\r\n{emotion}\r\n').encode()
    body += (f'--{boundary}\r\nContent-Disposition: form-data; name="file"; '
             f'filename="{name}"\r\nContent-Type: audio/wav\r\n\r\n').encode()
    body += raw + b"\r\n" + f"--{boundary}--\r\n".encode()
    req = urllib.request.Request(
        base.rstrip("/") + "/track2", data=bytes(body),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        r.read()
    return (time.perf_counter() - t0) * 1000.0


# ───────────────────────── chạy 1 mức tải ─────────────────────────
def run_load(call_fn, items, emotion, concurrency):
    """items: list (name, raw_bytes). Trả dict số liệu cho 1 mức concurrency."""
    lats, errs = [], 0
    wall0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = [ex.submit(call_fn, name, raw, emotion) for name, raw in items]
        for fut in as_completed(futs):
            try:
                lats.append(fut.result())
            except Exception:
                errs += 1
    wall = time.perf_counter() - wall0
    ok = len(lats)
    a = np.array(lats) if ok else np.array([0.0])
    return {"n": len(items), "ok": ok, "err": errs, "wall_s": round(wall, 3),
            "throughput": round(ok / wall, 3) if wall > 0 else 0,
            "lat_mean": round(float(a.mean()), 1),
            "lat_p50": round(float(np.percentile(a, 50)), 1),
            "lat_p95": round(float(np.percentile(a, 95)), 1)}


def warmup(call_fn, sample, emotion, n=2, timeout_note=""):
    """Gọi vài lần để kích model nạp (FastAPI lazy-load lần đầu rất chậm)."""
    print(f"  warmup {n} request{timeout_note}...", end=" ", flush=True)
    for i in range(n):
        try:
            call_fn(*sample, emotion)
        except Exception as e:
            print(f"(warmup lỗi: {type(e).__name__})", end=" ")
    print("ok")


# ───────────────────────── vẽ biểu đồ (tùy chọn) ─────────────────────────
def plot(rows, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("⚠️ Không có matplotlib → bỏ qua --plot.")
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    for backend in sorted({r["backend"] for r in rows}):
        rs = sorted([r for r in rows if r["backend"] == backend], key=lambda r: r["concurrency"])
        xs = [r["concurrency"] for r in rs]
        ax1.plot(xs, [r["throughput"] for r in rs], marker="o", label=backend)
        ax2.plot(xs, [r["lat_p95"] for r in rs], marker="o", label=backend)
    ax1.set(title="Throughput (cao = tốt)", xlabel="concurrency", ylabel="audio/giây")
    ax2.set(title="Latency p95 (thấp = tốt)", xlabel="concurrency", ylabel="ms")
    for ax in (ax1, ax2):
        ax.grid(True, alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=130)
    print(f"📈 Biểu đồ → {os.path.abspath(path)}")


# ───────────────────────── main ─────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Benchmark đối đầu Triton vs FastAPI (Track 2)")
    ap.add_argument("--dir", required=True, help="thư mục chứa .wav")
    ap.add_argument("--limit", type=int, default=50, help="số file dùng để bench (mặc định 50)")
    ap.add_argument("--emotion", default="", help="cảm xúc target chung (bật EMOS)")
    ap.add_argument("--triton-url", default=None, help="vd localhost:8000 (bỏ trống = không test Triton)")
    ap.add_argument("--triton-ssl", action="store_true")
    ap.add_argument("--fastapi-base", default=None, help="vd https://...hf.space (bỏ trống = không test FastAPI)")
    ap.add_argument("--concurrency", default="1,4,8", help="các mức song song, vd 1,4,8,16")
    ap.add_argument("--out", default=None, help="xuất CSV")
    ap.add_argument("--plot", default=None, help="xuất biểu đồ PNG (cần matplotlib)")
    args = ap.parse_args()

    if not args.triton_url and not args.fastapi_base:
        ap.error("cần ít nhất 1 trong --triton-url / --fastapi-base")
    levels = [int(x) for x in args.concurrency.split(",") if x.strip()]

    files = sorted(glob.glob(os.path.join(args.dir, "**", "*.wav"), recursive=True))[:args.limit]
    if not files:
        ap.error(f"không thấy .wav trong {args.dir}")
    items = []
    for p in files:
        with open(p, "rb") as f:
            items.append((os.path.basename(p), f.read()))
    print(f"📦 {len(items)} file · concurrency={levels} · emotion={args.emotion or '(không)'}\n")

    backends = []
    if args.triton_url:
        backends.append(("Triton", lambda n, r, e: call_triton(args.triton_url, args.triton_ssl, n, r, e)))
    if args.fastapi_base:
        backends.append(("FastAPI", lambda n, r, e: call_fastapi(args.fastapi_base, n, r, e)))

    rows = []
    for name, fn in backends:
        print(f"▶️  {name}")
        warmup(fn, items[0], args.emotion, n=2,
               timeout_note=" (FastAPI lần đầu có thể vài phút do tải model)" if name == "FastAPI" else "")
        for c in levels:
            m = run_load(fn, items, args.emotion, c)
            m.update(backend=name, concurrency=c)
            rows.append(m)
            print(f"    c={c:<3d} throughput={m['throughput']:>7.2f} audio/s | "
                  f"lat mean={m['lat_mean']:>6.0f} p50={m['lat_p50']:>6.0f} p95={m['lat_p95']:>6.0f} ms | "
                  f"ok={m['ok']}/{m['n']} err={m['err']}")
        print()

    # bảng so sánh throughput
    if len(backends) == 2:
        print("📊 So sánh throughput (audio/s) — Triton / FastAPI:")
        for c in levels:
            t = next((r["throughput"] for r in rows if r["backend"] == "Triton" and r["concurrency"] == c), None)
            f = next((r["throughput"] for r in rows if r["backend"] == "FastAPI" and r["concurrency"] == c), None)
            if t and f:
                print(f"    c={c:<3d}  Triton={t:>7.2f}  FastAPI={f:>7.2f}  → Triton nhanh hơn {t/f:.2f}×")

    if args.out:
        cols = ["backend", "concurrency", "n", "ok", "err", "wall_s",
                "throughput", "lat_mean", "lat_p50", "lat_p95"]
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows([{k: r[k] for k in cols} for r in rows])
        print(f"💾 CSV → {os.path.abspath(args.out)}")
    if args.plot:
        plot(rows, args.plot)


if __name__ == "__main__":
    main()
