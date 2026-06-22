#!/usr/bin/env python3
"""
demo_api_client_kaggle — Client gọi API service 3 track (VoiceMOS 2026) trên Kaggle.

Service đã deploy trên Hugging Face Space (FastAPI REST):
    https://yonroy-voicemos2026-api.hf.space   (Swagger: /docs)

File này CHỈ là CLIENT (không nạp model, không cần GPU). Dùng trên KAGGLE với
**Internet ON** (hoặc chạy local). Zero-dependency: chỉ thư viện chuẩn `urllib`.

Endpoint:
- Track 1 — POST /track1  (file_a [, file_b])           -> acr_a [, acr_b, ccr]
- Track 2 — POST /track2  (file [, target_emotion])     -> qmos, emos, cat{5}, vad{v,a,d}
- Track 3 — POST /track3  (file_test, file_ref)         -> spk_sim, acc_sim, cosine

Tính năng:
- Batch chấm cả 1 thư mục .wav + **resume** (bỏ qua file đã chấm → an toàn khi Space ngủ/timeout).
- Xuất CSV chi tiết mỗi track.
- Track 2: xuất thêm bản **nháp** answer.txt (wav,QMOS,EMOS,CAT,VAL,ARO,DOM).

⚠️ HF free CPU CHẬM (mỗi file vài chục giây, lần đầu mỗi track còn tải model ~1–2 phút).
   Muốn nhanh → chạy service trên Kaggle T4 (hướng khác), client này giữ nguyên cách gọi.

Chạy CLI (Track 2):
    python demo_api_client_kaggle_pipeline.py --track 2 --dir /kaggle/input/.../wav --out t2.csv
    python demo_api_client_kaggle_pipeline.py --track 2 --dir wavs --target happy
    python demo_api_client_kaggle_pipeline.py --track 1 --dir wavs
    python demo_api_client_kaggle_pipeline.py --track 3 --dir wavs --ref ref.wav
"""
import argparse
import csv
import json
import os
import sys
import time
import urllib.request
import urllib.error
import uuid

DEFAULT_BASE = "https://yonroy-voicemos2026-api.hf.space"
EMOTIONS5 = ["angry", "happy", "neutral", "sad", "surprised"]


# ----------------------------------------------------------------------------- #
# HTTP: multipart/form-data thủ công (không cần `requests`)
# ----------------------------------------------------------------------------- #
def _post_multipart(url, files, fields=None, timeout=600):
    """files: dict {field_name: filepath} · fields: dict {name: str}. Trả JSON đã parse."""
    boundary = uuid.uuid4().hex
    body = bytearray()
    for name, val in (fields or {}).items():
        if val is None:
            continue
        body += (f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"'
                 f'\r\n\r\n{val}\r\n').encode()
    for name, path in files.items():
        fn = os.path.basename(path)
        with open(path, "rb") as f:
            data = f.read()
        body += (f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"; '
                 f'filename="{fn}"\r\nContent-Type: audio/wav\r\n\r\n').encode()
        body += data + b"\r\n"
    body += f"--{boundary}--\r\n".encode()
    req = urllib.request.Request(
        url, data=bytes(body),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def health(base=DEFAULT_BASE, timeout=30):
    with urllib.request.urlopen(base.rstrip("/") + "/health", timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


# ----------------------------------------------------------------------------- #
# Gọi từng track (1 file)
# ----------------------------------------------------------------------------- #
def call_track1(base, wav_a, wav_b=None, timeout=600):
    files = {"file_a": wav_a}
    if wav_b:
        files["file_b"] = wav_b
    return _post_multipart(base.rstrip("/") + "/track1", files, timeout=timeout)


def call_track2(base, wav, target=None, timeout=600):
    return _post_multipart(base.rstrip("/") + "/track2", {"file": wav},
                           {"target_emotion": target}, timeout=timeout)


def call_track3(base, wav_test, wav_ref, timeout=600):
    return _post_multipart(base.rstrip("/") + "/track3",
                           {"file_test": wav_test, "file_ref": wav_ref}, timeout=timeout)


# ----------------------------------------------------------------------------- #
# Batch + resume cho 1 thư mục
# ----------------------------------------------------------------------------- #
def _list_wavs(d):
    return sorted(f for f in os.listdir(d) if f.lower().endswith(".wav"))


def _load_done(out_path):
    done = set()
    if os.path.exists(out_path):
        with open(out_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("file") and not row.get("error"):
                    done.add(row["file"])
    return done


def _writer(out_path, cols):
    new = not os.path.exists(out_path)
    f = open(out_path, "a", newline="", encoding="utf-8")
    w = csv.DictWriter(f, fieldnames=cols)
    if new:
        w.writeheader()
    return f, w


def batch_track2(base, wav_dir, out_csv, target=None, answer_txt=None):
    wavs = _list_wavs(wav_dir)
    if not wavs:
        sys.exit(f"Không thấy .wav trong {wav_dir}")
    cols = (["file", "qmos", "perceived_emotion"]
            + (["emos", "target", "emos_match"] if target else [])
            + [f"cat_{e}" for e in EMOTIONS5]
            + ["valence", "arousal", "dominance", "error"])
    done = _load_done(out_csv)
    print(f"Track2 · {len(wavs)} file · API={base} · target={target or '(không)'} · resume {len(done)}")
    f, w = _writer(out_csv, cols)
    try:
        for i, name in enumerate(wavs, 1):
            if name in done:
                continue
            row = {"file": name}
            t0 = time.time()
            try:
                res = call_track2(base, os.path.join(wav_dir, name), target,
                                  timeout=600 if i == 1 else 180)
                row["qmos"] = res.get("qmos")
                row["perceived_emotion"] = res.get("perceived_emotion")
                for e in EMOTIONS5:
                    row[f"cat_{e}"] = res.get("cat", {}).get(e)
                vad = res.get("vad", {})
                row["valence"], row["arousal"], row["dominance"] = (
                    vad.get("valence"), vad.get("arousal"), vad.get("dominance"))
                if target:
                    row["emos"] = res.get("emos")
                    row["target"] = res.get("target_emotion")
                    row["emos_match"] = res.get("emos_match")
                print(f"[{i}/{len(wavs)}] {name:18s} emo={row['perceived_emotion']:9s} "
                      f"qmos={row['qmos']} V/A/D={row['valence']}/{row['arousal']}/{row['dominance']} "
                      f"({time.time()-t0:.1f}s)")
            except Exception as e:
                row["error"] = repr(e)[:160]
                print(f"[{i}/{len(wavs)}] {name} ❌ {row['error']}")
            w.writerow(row)
            f.flush()
    finally:
        f.close()
    if answer_txt:
        _write_answer_txt(out_csv, answer_txt, target)
    print(f"✅ Track2 xong → {os.path.abspath(out_csv)}")


def batch_track1(base, wav_dir, out_csv):
    """Chấm ACR cho từng file (1 file/lần). CCR cần cặp file → không làm batch tự động."""
    wavs = _list_wavs(wav_dir)
    if not wavs:
        sys.exit(f"Không thấy .wav trong {wav_dir}")
    cols = ["file", "acr", "error"]
    done = _load_done(out_csv)
    print(f"Track1 · {len(wavs)} file · API={base} · resume {len(done)}")
    f, w = _writer(out_csv, cols)
    try:
        for i, name in enumerate(wavs, 1):
            if name in done:
                continue
            row = {"file": name}
            t0 = time.time()
            try:
                res = call_track1(base, os.path.join(wav_dir, name),
                                  timeout=600 if i == 1 else 180)
                row["acr"] = res.get("acr_a", res.get("acr"))
                print(f"[{i}/{len(wavs)}] {name:18s} acr={row['acr']} ({time.time()-t0:.1f}s)")
            except Exception as e:
                row["error"] = repr(e)[:160]
                print(f"[{i}/{len(wavs)}] {name} ❌ {row['error']}")
            w.writerow(row)
            f.flush()
    finally:
        f.close()
    print(f"✅ Track1 xong → {os.path.abspath(out_csv)}")


def batch_track3(base, wav_dir, ref_wav, out_csv):
    """Chấm speaker/accent similarity của từng file so với 1 reference."""
    wavs = _list_wavs(wav_dir)
    if not wavs:
        sys.exit(f"Không thấy .wav trong {wav_dir}")
    cols = ["file", "spk_sim", "acc_sim", "cosine", "error"]
    done = _load_done(out_csv)
    print(f"Track3 · {len(wavs)} file · ref={os.path.basename(ref_wav)} · resume {len(done)}")
    f, w = _writer(out_csv, cols)
    try:
        for i, name in enumerate(wavs, 1):
            if name in done:
                continue
            row = {"file": name}
            t0 = time.time()
            try:
                res = call_track3(base, os.path.join(wav_dir, name), ref_wav,
                                  timeout=600 if i == 1 else 180)
                row["spk_sim"] = res.get("spk_sim")
                row["acc_sim"] = res.get("acc_sim")
                row["cosine"] = res.get("cosine")
                print(f"[{i}/{len(wavs)}] {name:18s} spk={row['spk_sim']} acc={row['acc_sim']} "
                      f"({time.time()-t0:.1f}s)")
            except Exception as e:
                row["error"] = repr(e)[:160]
                print(f"[{i}/{len(wavs)}] {name} ❌ {row['error']}")
            w.writerow(row)
            f.flush()
    finally:
        f.close()
    print(f"✅ Track3 xong → {os.path.abspath(out_csv)}")


# ----------------------------------------------------------------------------- #
# Bản nháp answer.txt cho Track 2 (đối chiếu lại format BTC trước khi nộp!)
# ----------------------------------------------------------------------------- #
def _write_answer_txt(csv_path, answer_path, target):
    """
    ⚠️ NHÁP — KHÔNG nộp thẳng. Cột CAT ở đây = perceived_emotion (nhãn trội), còn
    format BTC có thể yêu cầu phân phối 5 lớp. Đối chiếu 08_track2_spec.md trước khi nộp.
    Khi không có target/EMOS thật → để trống EMOS (điền sau bằng nhánh exp08).
    """
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("error"):
                continue
            rows.append(r)
    with open(answer_path, "w", encoding="utf-8") as out:
        out.write("wav,QMOS,EMOS,CAT,VAL,ARO,DOM\n")
        for r in rows:
            out.write(",".join([
                r["file"],
                str(r.get("qmos", "")),
                str(r.get("emos", "")) if target else "",
                str(r.get("perceived_emotion", "")),
                str(r.get("valence", "")),
                str(r.get("arousal", "")),
                str(r.get("dominance", "")),
            ]) + "\n")
    print(f"📝 answer.txt (NHÁP) → {os.path.abspath(answer_path)} — đối chiếu format BTC trước khi nộp")


# ----------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Client gọi API 3 track VoiceMOS 2026 (HF Space)")
    ap.add_argument("--track", type=int, choices=[1, 2, 3], required=True)
    ap.add_argument("--dir", required=True, help="thư mục chứa .wav cần chấm")
    ap.add_argument("--base", default=os.environ.get("API_BASE", DEFAULT_BASE))
    ap.add_argument("--out", default=None, help="CSV output (mặc định track{N}_scores.csv)")
    ap.add_argument("--target", default=None, help="(track2) cảm xúc target chung → bật EMOS")
    ap.add_argument("--answer", default=None, help="(track2) xuất thêm answer.txt nháp")
    ap.add_argument("--ref", default=None, help="(track3) đường dẫn reference .wav (bắt buộc)")
    args = ap.parse_args()

    out = args.out or f"track{args.track}_scores.csv"
    try:
        print("health:", health(args.base))
    except Exception as e:
        print(f"⚠️ /health lỗi ({e}) — Space có thể đang ngủ, vẫn thử gọi (lần đầu sẽ chậm).")

    if args.track == 1:
        batch_track1(args.base, args.dir, out)
    elif args.track == 2:
        batch_track2(args.base, args.dir, out, args.target, args.answer)
    elif args.track == 3:
        if not args.ref:
            sys.exit("Track 3 cần --ref <reference.wav>")
        batch_track3(args.base, args.dir, args.ref, out)


if __name__ == "__main__":
    main()
