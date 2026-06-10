#!/usr/bin/env python3
"""
Chấm điểm emotion (Track 2) cho cả 1 thư mục .wav qua API đã deploy trên HF Space.
- Zero-dependency: chỉ dùng thư viện chuẩn (urllib) → KHÔNG cần cài torch/requests.
- Resume: bỏ qua file đã có trong CSV (chạy lại an toàn khi Space ngủ/timeout).
- Mặc định chấm CAT + VAD + QMOS (không cần target). Truyền --target để bật EMOS cho TOÀN BỘ file.

Chạy (PowerShell):
    cd "d:\\VFS\\VoiceMOS Challenge 2026"
    python score_100audio.py
    python score_100audio.py --target happy     # nếu muốn EMOS với target chung
"""
import argparse
import csv
import json
import os
import sys
import time
import urllib.request
import uuid

DEFAULT_BASE = "https://tranminhtoan140601-voicemos2026-api.hf.space"
EMOTIONS5 = ["angry", "happy", "neutral", "sad", "surprised"]


def post_track2(base, wav_path, target=None, timeout=600):
    url = base.rstrip("/") + "/track2"
    boundary = uuid.uuid4().hex
    with open(wav_path, "rb") as f:
        data = f.read()
    body = bytearray()
    if target:
        body += (f'--{boundary}\r\nContent-Disposition: form-data; name="target_emotion"'
                 f'\r\n\r\n{target}\r\n').encode()
    fn = os.path.basename(wav_path)
    body += (f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{fn}"'
             f'\r\nContent-Type: audio/wav\r\n\r\n').encode()
    body += data + b"\r\n"
    body += f"--{boundary}--\r\n".encode()
    req = urllib.request.Request(
        url, data=bytes(body),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "100audio"))
    ap.add_argument("--base", default=os.environ.get("API_BASE", DEFAULT_BASE))
    ap.add_argument("--out", default="100audio_emotion_scores.csv")
    ap.add_argument("--target", default=None, help="cảm xúc target chung (angry|happy|neutral|sad|surprised) → bật EMOS")
    args = ap.parse_args()

    wavs = sorted(f for f in os.listdir(args.dir) if f.lower().endswith(".wav"))
    if not wavs:
        sys.exit(f"Không thấy .wav trong {args.dir}")
    print(f"Tổng {len(wavs)} file · API = {args.base} · target = {args.target or '(không)'}")

    cols = ["file", "qmos", "perceived_emotion"] + \
           (["emos", "target", "emos_match"] if args.target else []) + \
           [f"cat_{e}" for e in EMOTIONS5] + ["valence", "arousal", "dominance", "error"]

    done = set()
    if os.path.exists(args.out):
        with open(args.out, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("file") and not row.get("error"):
                    done.add(row["file"])
        print(f"Resume: đã có {len(done)} file chấm trước đó → bỏ qua.")

    new_file = not os.path.exists(args.out)
    with open(args.out, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        if new_file:
            w.writeheader()
        for i, name in enumerate(wavs, 1):
            if name in done:
                continue
            path = os.path.join(args.dir, name)
            t0 = time.time()
            row = {"file": name}
            try:
                # lần đầu tải model trên Space → để timeout dài; các lần sau nhanh hơn
                res = post_track2(args.base, path, args.target, timeout=600 if i == 1 else 180)
                row["qmos"] = res.get("qmos")
                row["perceived_emotion"] = res.get("perceived_emotion")
                cat = res.get("cat", {})
                for e in EMOTIONS5:
                    row[f"cat_{e}"] = cat.get(e)
                vad = res.get("vad", {})
                row["valence"] = vad.get("valence")
                row["arousal"] = vad.get("arousal")
                row["dominance"] = vad.get("dominance")
                if args.target:
                    row["emos"] = res.get("emos")
                    row["target"] = res.get("target_emotion")
                    row["emos_match"] = res.get("emos_match")
                print(f"[{i}/{len(wavs)}] {name:18s} "
                      f"emo={row['perceived_emotion']:9s} qmos={row['qmos']} "
                      f"V/A/D={row['valence']}/{row['arousal']}/{row['dominance']} "
                      f"({time.time()-t0:.1f}s)")
            except Exception as e:
                row["error"] = repr(e)[:160]
                print(f"[{i}/{len(wavs)}] {name} ❌ {row['error']}")
            w.writerow(row)
            f.flush()

    print(f"\n✅ Xong → {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
