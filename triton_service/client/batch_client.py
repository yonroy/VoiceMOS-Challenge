#!/usr/bin/env python3
# ─────────────────────────────────────────────────────────────────────────────
# Client BẮN AUDIO → Triton (track2_emotion) → nhận điểm 6 cột.
#
# Đây là phần "tạo api service để bắn audio chấm kết quả": gửi 1 file hoặc CẢ
# THƯ MỤC wav tới Triton, chạy NHIỀU REQUEST SONG SONG (tận dụng các instance của
# server) → in JSON / xuất CSV. Đây mới là chỗ "chấm nhanh hàng loạt" phát huy.
#
# Cài: pip install -r requirements.txt   (tritonclient[http], soundfile chỉ để test)
#
# Ví dụ:
#   # 1 file
#   python batch_client.py --url localhost:8000 --audio tts.wav --emotion happy
#   # cả thư mục, 8 luồng song song, xuất CSV
#   python batch_client.py --url localhost:8000 --dir ./wavs --emotion happy \
#          --workers 8 --out scores.csv
# ─────────────────────────────────────────────────────────────────────────────
import argparse
import csv
import glob
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import tritonclient.http as httpclient

MODEL = "track2_emotion"
AUDIO_EXTS = (".wav", ".flac", ".mp3", ".ogg", ".m4a")


def _str_tensor(name, value):
    """Bọc 1 chuỗi/bytes thành input STRING shape [1,1] cho Triton."""
    arr = np.array([[value]], dtype=object)
    t = httpclient.InferInput(name, arr.shape, "BYTES")
    t.set_data_from_numpy(arr)
    return t


def score_one(url, path, emotion, ssl=False):
    """Gửi 1 file → trả dict kết quả (kèm 'file'). Mỗi luồng tự mở 1 client (thread-safe)."""
    with open(path, "rb") as f:
        raw = f.read()
    client = httpclient.InferenceServerClient(url=url, ssl=ssl, verbose=False)
    inputs = [_str_tensor("AUDIO_BYTES", raw)]
    if emotion:
        inputs.append(_str_tensor("TARGET_EMOTION", emotion))
    outputs = [httpclient.InferRequestedOutput("RESULT_JSON")]
    resp = client.infer(MODEL, inputs=inputs, outputs=outputs)
    payload = resp.as_numpy("RESULT_JSON").reshape(-1)[0]
    if isinstance(payload, bytes):
        payload = payload.decode("utf-8")
    result = json.loads(payload)
    result["file"] = os.path.basename(path)
    return result


def collect_files(audio, directory):
    if audio:
        return [audio]
    files = []
    for ext in AUDIO_EXTS:
        files += glob.glob(os.path.join(directory, f"**/*{ext}"), recursive=True)
    return sorted(files)


def flatten(r):
    """Làm phẳng dict 6 cột → 1 hàng CSV."""
    row = {"file": r.get("file"), "qmos": r.get("qmos"),
           "perceived_emotion": r.get("perceived_emotion"),
           "emos": r.get("emos"), "target_emotion": r.get("target_emotion"),
           "emos_match": r.get("emos_match")}
    for k, v in (r.get("cat") or {}).items():
        row[f"cat_{k}"] = v
    for k, v in (r.get("vad") or {}).items():
        row[f"vad_{k}"] = v
    return row


def main():
    ap = argparse.ArgumentParser(description="Bắn audio → Triton track2_emotion → điểm 6 cột")
    ap.add_argument("--url", default="localhost:8000", help="host:port HTTP của Triton (mặc định 8000)")
    ap.add_argument("--audio", help="1 file audio")
    ap.add_argument("--dir", help="thư mục chứa audio (đệ quy)")
    ap.add_argument("--emotion", default="", help="cảm xúc đích để bật EMOS (vd happy)")
    ap.add_argument("--workers", type=int, default=4, help="số request song song")
    ap.add_argument("--out", help="xuất CSV (nếu bỏ trống → in JSON ra màn hình)")
    ap.add_argument("--ssl", action="store_true", help="dùng HTTPS")
    args = ap.parse_args()

    if not args.audio and not args.dir:
        ap.error("cần --audio <file> hoặc --dir <thư mục>")
    files = collect_files(args.audio, args.dir)
    if not files:
        ap.error("không tìm thấy file audio nào.")
    print(f"▶️  {len(files)} file → Triton {args.url} (model={MODEL}, workers={args.workers})")

    results, errors = [], []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(score_one, args.url, p, args.emotion, args.ssl): p for p in files}
        for i, fut in enumerate(as_completed(futs), 1):
            path = futs[fut]
            try:
                r = fut.result()
                results.append(r)
                print(f"  [{i}/{len(files)}] ✅ {r['file']}: qmos={r['qmos']} "
                      f"emo={r['perceived_emotion']}" + (f" emos={r.get('emos')}" if r.get('emos') else ""))
            except Exception as e:
                errors.append((path, str(e)))
                print(f"  [{i}/{len(files)}] ❌ {os.path.basename(path)}: {e}")

    if args.out and results:
        rows = [flatten(r) for r in results]
        cols = sorted({k for row in rows for k in row})
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
        print(f"💾 Đã ghi {len(rows)} dòng → {args.out}")
    elif not args.out:
        print(json.dumps(results, ensure_ascii=False, indent=2))

    if errors:
        print(f"⚠️  {len(errors)} file lỗi.")


if __name__ == "__main__":
    main()
