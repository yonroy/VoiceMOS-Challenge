#!/usr/bin/env python3
# ─────────────────────────────────────────────────────────────────────────────
# UI Gradio (LOCAL) cho Triton Service — CẢ 3 TRACK, gọi qua GATEWAY.
#   • Load audio → POST multipart sang FastAPI gateway (:8080) → hiện ĐIỂM.
#   • Gateway lo chuyển tiếp sang Triton (:8000) → Triton suy luận + dynamic batching.
#   • UI KHÔNG nạp model, KHÔNG nói chuyện trực tiếp với Triton — chỉ là client HTTP.
#
# Chạy server trước (xem ../README.md: bash run_server.sh), rồi:
#     pip install -r requirements.txt
#     python app_ui.py
# Mặc định mở http://localhost:7860 ; trỏ ô "Gateway URL" tới server (vd http://<ip>:8080).
# ─────────────────────────────────────────────────────────────────────────────
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import gradio as gr
import numpy as np
import requests

EMOTIONS5 = ["angry", "happy", "neutral", "sad", "surprised"]
DEFAULT_URL = os.environ.get("GATEWAY_URL", "http://localhost:8080")


# ───────────────────────── HTTP helpers ─────────────────────────
def _post(base, path, files, data=None, timeout=180):
    """POST multipart sang gateway → (json_dict, latency_ms). Raise nếu lỗi HTTP."""
    url = base.rstrip("/") + path
    t0 = time.perf_counter()
    resp = requests.post(url, files=files, data=data or {}, timeout=timeout)
    latency_ms = (time.perf_counter() - t0) * 1000.0
    resp.raise_for_status()
    return resp.json(), latency_ms


def _read(path):
    with open(path, "rb") as f:
        return f.read()


def check_server(base):
    try:
        r = requests.get(base.rstrip("/") + "/health", timeout=10)
        if r.status_code == 200:
            j = r.json()
            return (f"### ✅ Gateway sẵn sàng\n"
                    f"- status: **{j.get('status')}**  ·  Triton: `{j.get('triton')}`")
        return f"### ⚠️ Gateway trả HTTP {r.status_code}\n`{r.text[:200]}`"
    except Exception as e:
        return (f"### 🔴 Không kết nối được gateway `{base}`\n"
                f"`{type(e).__name__}: {e}`\n\nĐã chạy `bash run_server.sh` chưa?")


# ───────────────────────── Tab 1: Track 2 (cảm xúc) ─────────────────────────
def ui_track2(base, audio_path, target_emotion):
    blank = ("", {}, None, None, None, None, None)
    if not audio_path:
        return ("### ⚠️ Hãy tải audio.",) + blank[1:]
    try:
        files = {"file": ("audio.wav", _read(audio_path), "audio/wav")}
        data = {"target_emotion": target_emotion or ""}
        r, latency = _post(base, "/track2", files, data)
    except Exception as e:
        return (f"### 🔴 Lỗi gọi gateway\n`{type(e).__name__}: {e}`",) + blank[1:]

    cat = r.get("cat", {})
    perceived = r.get("perceived_emotion", "?")
    vad = r.get("vad", {})
    lines = [f"### Kết quả  ·  ⏱️ latency = **{latency:.0f} ms**",
             f"- Cảm xúc cảm nhận (CAT argmax): **{perceived}**",
             f"- QMOS (chất lượng) = **{r.get('qmos')}/5**"]
    if "emos" in r:
        match = "✅ KHỚP" if r.get("emos_match") else "⚠️ LỆCH"
        lines.append(f"- EMOS (khớp `{r.get('target_emotion')}`) = **{r.get('emos')}/5** → {match}")
    else:
        lines.append("- *(Chọn cảm xúc target để bật EMOS)*")
    return ("\n".join(lines),
            {e: float(cat.get(e, 0.0)) for e in EMOTIONS5},
            r.get("qmos"), r.get("emos"),
            vad.get("valence"), vad.get("arousal"), vad.get("dominance"))


# ───────────────────────── Tab 2: Track 1 (ACR/CCR) ─────────────────────────
def ui_track1(base, audio_a, audio_b):
    if not audio_a:
        return "### ⚠️ Hãy tải audio A.", None, None, None
    try:
        files = {"file_a": ("a.wav", _read(audio_a), "audio/wav")}
        if audio_b:
            files["file_b"] = ("b.wav", _read(audio_b), "audio/wav")
        r, latency = _post(base, "/track1", files)
    except Exception as e:
        return f"### 🔴 Lỗi gọi gateway\n`{type(e).__name__}: {e}`", None, None, None

    lines = [f"### Kết quả  ·  ⏱️ latency = **{latency:.0f} ms**",
             f"- ACR (chất lượng A) = **{r.get('acr_a')}/5**"]
    if "ccr" in r:
        ccr = r.get("ccr")
        better = "A tốt hơn B" if ccr > 0 else ("B tốt hơn A" if ccr < 0 else "ngang nhau")
        lines.append(f"- ACR (B) = **{r.get('acr_b')}/5**")
        lines.append(f"- CCR (A so với B, -3..+3) = **{ccr}** → {better}")
    else:
        lines.append("- *(Tải thêm audio B để tính CCR so sánh cặp)*")
    return "\n".join(lines), r.get("acr_a"), r.get("acr_b"), r.get("ccr")


# ───────────────────────── Tab 3: Track 3 (speaker/accent sim) ─────────────────────────
def ui_track3(base, audio_test, audio_ref):
    if not audio_test or not audio_ref:
        return "### ⚠️ Cần đủ 2 file: Test và Reference.", None, None, None
    try:
        files = {
            "file_test": ("test.wav", _read(audio_test), "audio/wav"),
            "file_ref": ("ref.wav", _read(audio_ref), "audio/wav"),
        }
        r, latency = _post(base, "/track3", files)
    except Exception as e:
        return f"### 🔴 Lỗi gọi gateway\n`{type(e).__name__}: {e}`", None, None, None

    lines = [f"### Kết quả  ·  ⏱️ latency = **{latency:.0f} ms**",
             f"- spk_sim (giống người nói) = **{r.get('spk_sim')}/5**",
             f"- acc_sim (giống chất giọng/accent) = **{r.get('acc_sim')}/5**",
             f"- cosine (zero-shot tham khảo) = **{r.get('cosine')}**"]
    return "\n".join(lines), r.get("spk_sim"), r.get("acc_sim"), r.get("cosine")


# ───────────────────────── Tab 4: batch + đo tốc độ ─────────────────────────
T2_COLS = ["file", "latency_ms", "qmos", "perceived", "emos", "val", "aro", "dom",
           "cat_angry", "cat_happy", "cat_neutral", "cat_sad", "cat_surprised"]
T1_COLS = ["file", "latency_ms", "acr_a"]


def ui_batch(base, track, files, target_emotion, workers, progress=gr.Progress()):
    cols = T2_COLS if track == "Track 2 (cảm xúc)" else T1_COLS
    if not files:
        return [], "### ⚠️ Hãy tải ít nhất 1 file.", gr.update(headers=cols)
    paths = [f.name if hasattr(f, "name") else f for f in files]
    is_t2 = track == "Track 2 (cảm xúc)"

    def work(p):
        raw = _read(p)
        if is_t2:
            r, lat = _post(base, "/track2",
                           {"file": (os.path.basename(p), raw, "audio/wav")},
                           {"target_emotion": target_emotion or ""})
        else:
            r, lat = _post(base, "/track1",
                           {"file_a": (os.path.basename(p), raw, "audio/wav")})
        r["_file"] = os.path.basename(p)
        r["_latency_ms"] = lat
        return r

    rows, latencies, n_err = [], [], 0
    wall0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=int(workers)) as ex:
        futs = {ex.submit(work, p): p for p in paths}
        for i, fut in enumerate(as_completed(futs), 1):
            progress(i / len(paths), desc=f"{i}/{len(paths)}")
            try:
                r = fut.result()
                if is_t2:
                    cat = r.get("cat", {})
                    vad = r.get("vad", {})
                    rows.append([r["_file"], round(r["_latency_ms"]), r.get("qmos"),
                                 r.get("perceived_emotion"), r.get("emos"),
                                 round(float(vad.get("valence", 0)), 2),
                                 round(float(vad.get("arousal", 0)), 2),
                                 round(float(vad.get("dominance", 0)), 2),
                                 *[round(float(cat.get(e, 0)), 3) for e in EMOTIONS5]])
                else:
                    rows.append([r["_file"], round(r["_latency_ms"]), r.get("acr_a")])
                latencies.append(r["_latency_ms"])
            except Exception as e:
                n_err += 1
                rows.append([os.path.basename(futs[fut]), None, f"ERR: {e}"]
                            + [None] * (len(cols) - 3))
    wall = time.perf_counter() - wall0

    ok = len(latencies)
    if ok:
        thr = ok / wall
        p50 = float(np.percentile(latencies, 50))
        p95 = float(np.percentile(latencies, 95))
        metrics = (
            f"### 📊 Độ đo serving ({track}, workers={int(workers)})\n"
            f"- Tổng: **{len(paths)}** file · ok **{ok}** · lỗi **{n_err}**\n"
            f"- ⏱️ Tổng thời gian (wall): **{wall:.2f}s**\n"
            f"- 🚀 Throughput: **{thr:.2f} audio/giây**\n"
            f"- Latency mỗi request: trung bình **{np.mean(latencies):.0f}ms** · "
            f"p50 **{p50:.0f}ms** · p95 **{p95:.0f}ms**\n\n"
            f"> Tăng `workers` để 'nuôi' dynamic batching của Triton → throughput tăng tới khi nghẽn GPU."
        )
    else:
        metrics = "### 🔴 Không file nào chấm được — kiểm tra gateway/URL."
    return rows, metrics, gr.update(headers=cols)


# ───────────────────────── Layout ─────────────────────────
INTRO = (
    "# 🎙️ VoiceMOS 2026 — Triton Tester (3 track)\n"
    "UI này là **client** gọi **FastAPI gateway** (`:8080`), gateway chuyển tiếp sang "
    "**Triton Inference Server** (`:8000`, dynamic batching). UI chỉ load audio và hiện điểm + latency.\n\n"
    "> Chạy server trước: `bash run_server.sh`. Trỏ ô **Gateway URL** tới server (vd `http://<ip>:8080`)."
)

with gr.Blocks(title="VMC2026 — Triton Tester (3 track)") as demo:
    gr.Markdown(INTRO)
    with gr.Row():
        url_in = gr.Textbox(DEFAULT_URL, label="Gateway URL (http://host:8080)", scale=4)
        chk_btn = gr.Button("Kiểm tra server", scale=1)
    status = gr.Markdown()
    chk_btn.click(check_server, [url_in], status)

    # ── Track 2 ──
    with gr.Tab("🎯 Track 2 — Cảm xúc"):
        with gr.Row():
            with gr.Column(scale=1):
                t2_audio = gr.Audio(type="filepath", label="Audio (giọng TTS)")
                t2_tgt = gr.Dropdown(EMOTIONS5, label="Cảm xúc target (bật EMOS)")
                t2_go = gr.Button("Chấm", variant="primary")
            with gr.Column(scale=2):
                t2_verdict = gr.Markdown()
                t2_cat = gr.Label(label="CAT — phân bố cảm xúc cảm nhận")
                with gr.Row():
                    t2_qmos = gr.Number(label="QMOS", interactive=False)
                    t2_emos = gr.Number(label="EMOS", interactive=False)
                with gr.Row():
                    t2_val = gr.Number(label="Valence", interactive=False)
                    t2_aro = gr.Number(label="Arousal", interactive=False)
                    t2_dom = gr.Number(label="Dominance", interactive=False)
        t2_go.click(ui_track2, [url_in, t2_audio, t2_tgt],
                    [t2_verdict, t2_cat, t2_qmos, t2_emos, t2_val, t2_aro, t2_dom])

    # ── Track 1 ──
    with gr.Tab("🔊 Track 1 — Chất lượng (ACR/CCR)"):
        gr.Markdown("Audio A → **ACR** (1..5). Thêm audio B → **ACR(B)** + **CCR** (A so với B, -3..+3).")
        with gr.Row():
            with gr.Column(scale=1):
                t1_a = gr.Audio(type="filepath", label="Audio A (cần)")
                t1_b = gr.Audio(type="filepath", label="Audio B (tùy chọn, để tính CCR)")
                t1_go = gr.Button("Chấm", variant="primary")
            with gr.Column(scale=2):
                t1_verdict = gr.Markdown()
                with gr.Row():
                    t1_acra = gr.Number(label="ACR (A)", interactive=False)
                    t1_acrb = gr.Number(label="ACR (B)", interactive=False)
                    t1_ccr = gr.Number(label="CCR (A vs B)", interactive=False)
        t1_go.click(ui_track1, [url_in, t1_a, t1_b],
                    [t1_verdict, t1_acra, t1_acrb, t1_ccr])

    # ── Track 3 ──
    with gr.Tab("🗣️ Track 3 — Speaker/Accent sim"):
        gr.Markdown("Cần **2 file**: Test (giọng TTS cần chấm) + Reference (giọng gốc) → spk_sim, acc_sim, cosine.")
        with gr.Row():
            with gr.Column(scale=1):
                t3_test = gr.Audio(type="filepath", label="Audio Test (cần)")
                t3_ref = gr.Audio(type="filepath", label="Audio Reference (cần)")
                t3_go = gr.Button("Chấm", variant="primary")
            with gr.Column(scale=2):
                t3_verdict = gr.Markdown()
                with gr.Row():
                    t3_spk = gr.Number(label="spk_sim", interactive=False)
                    t3_acc = gr.Number(label="acc_sim", interactive=False)
                    t3_cos = gr.Number(label="cosine", interactive=False)
        t3_go.click(ui_track3, [url_in, t3_test, t3_ref],
                    [t3_verdict, t3_spk, t3_acc, t3_cos])

    # ── Batch ──
    with gr.Tab("📦 Chấm hàng loạt + đo tốc độ"):
        gr.Markdown("Tải **nhiều file** → bắn song song qua gateway → bảng điểm + **throughput/latency**. "
                    "Chỗ thấy rõ hiệu quả dynamic batching của Triton.")
        with gr.Row():
            b_track = gr.Dropdown(["Track 2 (cảm xúc)", "Track 1 (ACR)"],
                                  value="Track 2 (cảm xúc)", label="Track", scale=2)
            b_tgt = gr.Dropdown(EMOTIONS5, label="Cảm xúc target (chỉ Track 2)", scale=2)
            b_workers = gr.Slider(1, 16, value=4, step=1, label="Số request song song", scale=2)
            b_go = gr.Button("Chấm hàng loạt", variant="primary", scale=1)
        b_files = gr.File(file_count="multiple", file_types=["audio"], label="Nhiều audio")
        b_metrics = gr.Markdown()
        b_tbl = gr.Dataframe(headers=T2_COLS, label="Kết quả", interactive=False, wrap=True)
        b_go.click(ui_batch, [url_in, b_track, b_files, b_tgt, b_workers],
                   [b_tbl, b_metrics, b_tbl])

if __name__ == "__main__":
    # Cổng UI đọc từ env (server dùng chung hay đụng cổng). Đổi: GRADIO_SERVER_PORT=7870 python app_ui.py
    _port = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))
    demo.launch(server_name="0.0.0.0", server_port=_port)
