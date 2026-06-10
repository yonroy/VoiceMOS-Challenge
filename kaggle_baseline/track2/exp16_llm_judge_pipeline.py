# %% [markdown]
# # exp16 — Audio-LLM-as-Judge cho MOS cảm xúc (Track 2)
#
# **Ý tưởng:** đưa thẳng audio cho một **audio-LLM** (Gemini / GPT-4o-audio) qua **API** + prompt có
# cấu trúc → bắt nó chấm cả 6 cột (`QMOS, EMOS, CAT, VAL, ARO, DOM`) → ráp `answer.txt` → nộp CodaBench.
#
# **Mục tiêu chính = NOVELTY cho paper** (khảo sát có hệ thống audio-LLM-as-judge cho MOS cảm xúc),
# so với hệ SSL đã train (exp07 QMOS 0.548 · exp08 EMOS 0.811…). KHÔNG cần GPU — thuần gọi API.
#
# | Đặc điểm | Giá trị |
# |---|---|
# | GPU | ❌ không cần (chỉ network I/O) |
# | Tốn phí | ✅ API trả tiền theo token/audio → **cache + resume bắt buộc** |
# | Provider | `gemini` (mặc định, đã có billing) · `openai` (GPT-4o-audio, để so 2 LLM) |
# | Output | `answer.txt` 6 cột giống exp07 |
#
# **Cách dùng Kaggle:** Internet = **On**; Add-ons → Secrets: `GEMINI_API_KEY` (và `OPENAI_API_KEY`
# nếu chạy provider openai). Settings GPU **không cần**. Sửa `DATA_ROOT` cho khớp slug rồi Run All.
#
# ⚠️ **Model ID có thể đã đổi** theo thời gian → kiểm tra `GEMINI_MODEL` / `OPENAI_MODEL` còn nhận
# audio không trước khi chạy full (xem mục 1).

# %% [markdown]
# ## 0. Cấu hình — SỬA Ở ĐÂY

# %%
import os, io, re, json, time, base64, glob

# ── Data Track 2 trên Kaggle ────────────────────────────────────────────────
DATA_ROOT    = "/kaggle/input/vmc2026-track2-full/vmc2026-track2"   # << SỬA slug
WAV_DIR      = f"{DATA_ROOT}/wav"
METADATA_CSV = f"{DATA_ROOT}/metadata.csv"      # wavID|emotion|transcript (KHÔNG header) — nhãn cảm xúc target
DEV_SCP      = f"{DATA_ROOT}/sets/dev.scp"      # danh sách wav DEV cần nộp (train phase)
TRAIN_CSV    = f"{DATA_ROOT}/sets/train.csv"    # chỉ cần khi SHOT_MODE="few_shot"

OUT_DIR   = "/kaggle/working"
CACHE_DIR = "/kaggle/working/exp16_llm_cache"   # nên Save Version / lưu Dataset để KHÔNG gọi lại API
os.makedirs(CACHE_DIR, exist_ok=True)

# ── Provider & model ────────────────────────────────────────────────────────
PROVIDER     = "gemini"                  # "gemini" | "openai"
GEMINI_MODEL = "gemini-2.5-flash"        # << xác nhận model audio hiện hành (baseline dùng họ gemini-*-flash)
OPENAI_MODEL = "gpt-4o-audio-preview"    # << model audio của OpenAI; cần OPENAI_API_KEY
TEMPERATURE  = 0.0                       # cố định để TÁI LẬP (paper)

# ── Chế độ chạy ─────────────────────────────────────────────────────────────
SHOT_MODE    = "zero_shot"   # "zero_shot" | "few_shot" (nhét K ví dụ audio có nhãn từ train.csv)
FEW_K        = 2             # số ví dụ few-shot (mỗi ví dụ = 1 audio + nhãn vàng) — tốn thêm token!
LIMIT        = 20           # << số nhỏ (20) để smoke test; None = full DEV (~2730) — CHẠY THỬ TRƯỚC
MAX_SECONDS  = 12           # cắt audio cho rẻ + nhanh
WORKERS      = 4            # luồng gọi song song (giảm nếu dính rate limit)
MAX_RETRY    = 3            # số lần thử lại 1 wav khi lỗi mạng / JSON hỏng
RETRY_SLEEP  = 2.0          # giây nghỉ giữa các lần thử

TAG = f"{PROVIDER}_{(GEMINI_MODEL if PROVIDER=='gemini' else OPENAI_MODEL)}_{SHOT_MODE}".replace("/", "-")
CACHE_PATH = os.path.join(CACHE_DIR, f"{TAG}.jsonl")   # 1 dòng JSON / wav (raw + parsed) → resume
print("TAG:", TAG, "| cache:", CACHE_PATH)

# %% [markdown]
# ## 0b. Nhãn cảm xúc target + chuẩn hóa lớp (tái dùng quy ước baseline)

# %%
EMOTIONS5 = ["angry", "happy", "neutral", "sad", "surprised"]   # THỨ TỰ chuẩn cho cột CAT

_EMO_ALIAS = {
    "angry": "angry", "anger": "angry",
    "happy": "happy", "happiness": "happy", "joy": "happy",
    "neutral": "neutral", "calm": "neutral",
    "sad": "sad", "sadness": "sad",
    "surprise": "surprised", "surprised": "surprised", "surprising": "surprised",
}

def norm_emotion(label):
    key = str(label).strip().lower()
    return _EMO_ALIAS.get(key, key if key in EMOTIONS5 else None)

def stem(name):
    return os.path.splitext(os.path.basename(name))[0]

def load_target_emotions():
    """metadata.csv (wavID|emotion|transcript, không header) → {stem: emotion_chuẩn}."""
    tgt = {}
    if not (METADATA_CSV and os.path.exists(METADATA_CSV)):
        print("⚠️ Không thấy metadata.csv → EMOS sẽ thiếu cảm xúc target.")
        return tgt
    with open(METADATA_CSV, encoding="utf-8") as f:
        for ln in f:
            parts = ln.strip().split("|")
            if len(parts) < 2:
                continue
            tgt[stem(parts[0])] = norm_emotion(parts[1])
    return tgt

target_map = load_target_emotions()
print("Nhãn cảm xúc target:", len(target_map))

def list_dev():
    with open(DEV_SCP) as f:
        return [ln.strip() for ln in f if ln.strip()]

dev_names = list_dev()
if LIMIT:
    dev_names = dev_names[:LIMIT]
print("DEV cần chấm:", len(dev_names), "mẫu", "| LIMIT =", LIMIT)

# %% [markdown]
# ## 1. Cài SDK + nạp key
#
# Gemini dùng SDK mới `google-genai`; OpenAI dùng `openai`. Trên Kaggle **Internet phải On**.

# %%
# !pip -q install google-genai openai soundfile librosa

def setup_keys():
    """Nạp API key từ Kaggle Secrets (fallback: biến môi trường đã set sẵn)."""
    try:
        from kaggle_secrets import UserSecretsClient
        sec = UserSecretsClient()
        for k in ["GEMINI_API_KEY", "OPENAI_API_KEY"]:
            try:
                os.environ[k] = sec.get_secret(k)
                print(f"Đã nạp {k} từ Secrets")
            except Exception:
                pass
    except Exception as e:
        print("Không dùng được Kaggle Secrets:", e, "→ set tay os.environ[...] nếu cần")

setup_keys()

# %% [markdown]
# ## 2. Đọc + chuẩn hóa audio (16kHz mono, cắt MAX_SECONDS) → bytes WAV trong RAM

# %%
import numpy as np

def load_wav_bytes(path, sr=16000, max_seconds=MAX_SECONDS):
    """Trả (wav_bytes, base64_str). Cắt ≤ max_seconds, resample 16k mono, encode WAV PCM16."""
    import soundfile as sf
    try:
        import librosa
        y, _ = librosa.load(path, sr=sr, mono=True)
    except Exception:
        y, in_sr = sf.read(path)
        if y.ndim > 1:
            y = y.mean(axis=1)
        if in_sr != sr:   # fallback resample tuyến tính nếu không có librosa
            idx = np.linspace(0, len(y) - 1, int(len(y) * sr / in_sr))
            y = np.interp(idx, np.arange(len(y)), y)
    if max_seconds:
        y = y[: int(sr * max_seconds)]
    buf = io.BytesIO()
    sf.write(buf, y.astype(np.float32), sr, format="WAV", subtype="PCM_16")
    raw = buf.getvalue()
    return raw, base64.b64encode(raw).decode("ascii")

# %% [markdown]
# ## 3. Prompt — định nghĩa 6 metric + ép JSON nghiêm ngặt
#
# QMOS = chất lượng/độ tự nhiên (sạch, không méo/robot). EMOS = độ KHỚP với **cảm xúc target**.
# CAT = phân phối vote 5 lớp. VAD = Valence/Arousal/Dominance. Tất cả thang **1–5** (CAT là tỉ lệ 0–1).

# %%
SYSTEM_INSTRUCTION = (
    "You are an expert evaluator of emotional text-to-speech. "
    "Listen to the audio and rate it. Respond with ONLY a compact JSON object, no prose."
)

def build_prompt(target_emo):
    tgt = target_emo if target_emo else "unknown"
    return (
        "Rate this speech utterance. The INTENDED (target) emotion is: "
        f"\"{tgt}\".\n\n"
        "Return a JSON object with EXACTLY these keys (numbers on a 1-5 scale unless stated):\n"
        "  \"qmos\": overall audio QUALITY / naturalness (1=very unnatural/robotic/distorted, 5=clean & human-like).\n"
        "  \"emos\": how well the emotion expressed MATCHES the target emotion above "
        "(1=not matching at all, 5=perfectly matching).\n"
        "  \"cat\": an object with probabilities (summing to 1.0) over the 5 perceived emotions: "
        "{\"neutral\":_, \"happy\":_, \"sad\":_, \"angry\":_, \"surprised\":_}.\n"
        "  \"val\": valence (1=very negative, 5=very positive).\n"
        "  \"aro\": arousal (1=very calm, 5=very excited).\n"
        "  \"dom\": dominance (1=very submissive, 5=very dominant).\n\n"
        "Example format: "
        "{\"qmos\":3.5,\"emos\":4.0,"
        "\"cat\":{\"neutral\":0.1,\"happy\":0.7,\"sad\":0.0,\"angry\":0.1,\"surprised\":0.1},"
        "\"val\":4.0,\"aro\":3.5,\"dom\":3.0}\n"
        "Respond with ONLY the JSON."
    )

# %% [markdown]
# ## 3b. (tùy chọn) Few-shot — lấy K ví dụ audio có nhãn vàng từ train.csv
#
# Bật khi `SHOT_MODE="few_shot"`. Mỗi ví dụ = 1 audio train + nhãn vàng (gộp TB theo wav). Tốn thêm token.

# %%
few_shot_examples = []   # list[(audio_b64, audio_bytes, gold_json_str)]

def _agg_train_labels():
    """Gộp train.csv (sep='|') theo wavID → nhãn vàng trung bình; CAT = tỉ lệ vote."""
    import pandas as pd
    df = pd.read_csv(TRAIN_CSV, sep="|")
    rows = {}
    for wav, g in df.groupby("wavID"):
        votes = np.zeros(5, np.float32)
        for cell in g["emoCat"].astype(str):
            for tok in cell.split(","):
                e = norm_emotion(tok)
                if e in EMOTIONS5:
                    votes[EMOTIONS5.index(e)] += 1
        s = votes.sum()
        cat = (votes / s) if s > 0 else np.full(5, 0.2, np.float32)
        rows[stem(wav)] = dict(
            qmos=float(g["qMOS"].mean()), emos=float(g["eMOS"].mean()),
            val=float(g["val"].mean()), aro=float(g["aro"].mean()), dom=float(g["dom"].mean()),
            cat={EMOTIONS5[i]: round(float(cat[i]), 4) for i in range(5)},
        )
    return rows

def build_few_shot():
    if SHOT_MODE != "few_shot":
        return
    labels = _agg_train_labels()
    picked = list(labels.keys())[:FEW_K]
    for sid in picked:
        wavp = os.path.join(WAV_DIR, sid + ".wav")
        if not os.path.exists(wavp):
            continue
        raw, b64 = load_wav_bytes(wavp)
        gold = labels[sid]
        gold_json = json.dumps({
            "qmos": round(gold["qmos"], 2), "emos": round(gold["emos"], 2),
            "cat": gold["cat"], "val": round(gold["val"], 2),
            "aro": round(gold["aro"], 2), "dom": round(gold["dom"], 2),
        })
        few_shot_examples.append((b64, raw, gold_json))
    print(f"Few-shot: {len(few_shot_examples)} ví dụ")

build_few_shot()

# %% [markdown]
# ## 4. Gọi API — trừu tượng hóa provider (gemini / openai)
#
# Mỗi provider tự dựng message của nó (kèm few-shot nếu có). Trả về **text thô** để parse ở mục 5.

# %%
_client = {"gemini": None, "openai": None}

def _gemini_client():
    if _client["gemini"] is None:
        from google import genai
        _client["gemini"] = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    return _client["gemini"]

def _openai_client():
    if _client["openai"] is None:
        from openai import OpenAI
        _client["openai"] = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return _client["openai"]

def call_gemini(audio_b64, audio_bytes, prompt):
    from google.genai import types
    client = _gemini_client()
    contents = []
    for ex_b64, ex_bytes, ex_gold in few_shot_examples:   # few-shot: audio ví dụ + nhãn vàng
        contents.append(types.Content(role="user", parts=[
            types.Part.from_bytes(data=ex_bytes, mime_type="audio/wav"),
            types.Part.from_text(text=build_prompt(None)),
        ]))
        contents.append(types.Content(role="model", parts=[types.Part.from_text(text=ex_gold)]))
    contents.append(types.Content(role="user", parts=[
        types.Part.from_bytes(data=audio_bytes, mime_type="audio/wav"),
        types.Part.from_text(text=prompt),
    ]))
    resp = client.models.generate_content(
        model=GEMINI_MODEL, contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION, temperature=TEMPERATURE),
    )
    return resp.text

def call_openai(audio_b64, audio_bytes, prompt):
    client = _openai_client()
    messages = [{"role": "system", "content": SYSTEM_INSTRUCTION}]
    for ex_b64, ex_bytes, ex_gold in few_shot_examples:
        messages.append({"role": "user", "content": [
            {"type": "text", "text": build_prompt(None)},
            {"type": "input_audio", "input_audio": {"data": ex_b64, "format": "wav"}},
        ]})
        messages.append({"role": "assistant", "content": ex_gold})
    messages.append({"role": "user", "content": [
        {"type": "text", "text": prompt},
        {"type": "input_audio", "input_audio": {"data": audio_b64, "format": "wav"}},
    ]})
    resp = client.chat.completions.create(
        model=OPENAI_MODEL, messages=messages, temperature=TEMPERATURE,
        modalities=["text"],
    )
    return resp.choices[0].message.content

def call_llm(audio_b64, audio_bytes, prompt):
    return call_gemini(audio_b64, audio_bytes, prompt) if PROVIDER == "gemini" \
        else call_openai(audio_b64, audio_bytes, prompt)

# %% [markdown]
# ## 5. Parse JSON chịu lỗi → 6 cột; clamp [1,5]; chuẩn hóa CAT

# %%
def _clamp(x, lo=1.0, hi=5.0, default=3.0):
    try:
        v = float(x)
    except Exception:
        return default
    return max(lo, min(hi, v))

def parse_response(text):
    """text thô LLM → dict {qmos,emos,cat5(list theo EMOTIONS5),val,aro,dom} hoặc None nếu hỏng."""
    if not text:
        return None
    m = re.search(r"\{.*\}", text, re.DOTALL)   # trích khối JSON đầu tiên
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except Exception:
        return None
    cat_in = d.get("cat", {}) or {}
    cat = np.zeros(5, np.float32)
    for k, v in cat_in.items():
        e = norm_emotion(k)
        if e in EMOTIONS5:
            try:
                cat[EMOTIONS5.index(e)] = max(0.0, float(v))
            except Exception:
                pass
    cat = cat / cat.sum() if cat.sum() > 0 else np.full(5, 0.2, np.float32)
    return dict(
        qmos=_clamp(d.get("qmos")), emos=_clamp(d.get("emos")),
        cat5=cat.tolist(),
        val=_clamp(d.get("val")), aro=_clamp(d.get("aro")), dom=_clamp(d.get("dom")),
    )

# %% [markdown]
# ## 6. Vòng chấm có CACHE + RESUME (KHÔNG gọi lại wav đã có trong cache)

# %%
def load_cache():
    done = {}
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, encoding="utf-8") as f:
            for ln in f:
                try:
                    r = json.loads(ln)
                    done[r["stem"]] = r
                except Exception:
                    continue
    return done

def score_one(name):
    """Gọi LLM cho 1 wav, retry; trả record dict {stem,name,raw,parsed}."""
    sid = stem(name)
    wavp = os.path.join(WAV_DIR, name if name.endswith(".wav") else name + ".wav")
    tgt = target_map.get(sid)
    prompt = build_prompt(tgt)
    last_err = None
    for attempt in range(MAX_RETRY):
        try:
            _, b64 = (None, None)
            raw_bytes, b64 = load_wav_bytes(wavp)
            text = call_llm(b64, raw_bytes, prompt)
            parsed = parse_response(text)
            if parsed is not None:
                return dict(stem=sid, name=name, raw=text, parsed=parsed, ok=True)
            last_err = "parse_fail"
        except Exception as e:
            last_err = str(e)
        time.sleep(RETRY_SLEEP * (attempt + 1))
    return dict(stem=sid, name=name, raw=None, parsed=None, ok=False, err=last_err)

def run_scoring():
    from concurrent.futures import ThreadPoolExecutor, as_completed
    done = load_cache()
    todo = [n for n in dev_names if stem(n) not in done]
    print(f"Cache có {len(done)} | cần chấm thêm {len(todo)} | ước lượng {len(todo)} call API")
    if not todo:
        return done
    n_ok = n_bad = 0
    with open(CACHE_PATH, "a", encoding="utf-8") as fout, \
         ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(score_one, n): n for n in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            rec = fut.result()
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            done[rec["stem"]] = rec
            n_ok += int(rec["ok"]); n_bad += int(not rec["ok"])
            if i % 50 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)} | ok={n_ok} bad={n_bad}")
    if n_bad:
        print(f"⚠️ {n_bad} wav hỏng (parse/API) → sẽ điền mặc định ở build_answer.")
    return done

records = run_scoring()

# %% [markdown]
# ## 7. Ráp `answer.txt` 6 cột (giống exp07) + validate + zip

# %%
def fmt_cat(probs5):
    return "|".join(f"{e}:{probs5[i]:.6g}" for i, e in enumerate(EMOTIONS5))

def build_answer(out_path):
    n_real = n_default = 0
    with open(out_path, "w") as f:
        f.write("wav,QMOS,EMOS,CAT,VAL,ARO,DOM\n")
        for name in dev_names:
            sid = stem(name)
            rec = records.get(sid)
            p = rec["parsed"] if (rec and rec.get("parsed")) else None
            if p is None:
                qmos = emos = val = aro = dom = 3.0
                cat5 = [0.2] * 5
                n_default += 1
            else:
                qmos, emos = p["qmos"], p["emos"]
                val, aro, dom = p["val"], p["aro"], p["dom"]
                cat5 = p["cat5"]; n_real += 1
            f.write(f"{name},{qmos:.6g},{emos:.6g},{fmt_cat(cat5)},"
                    f"{val:.6g},{aro:.6g},{dom:.6g}\n")
    print(f"Ghi {len(dev_names)} dòng → {out_path} | LLM thật {n_real}, mặc định {n_default}")

answer_path = os.path.join(OUT_DIR, "answer.txt")
build_answer(answer_path)

def validate(path):
    import csv
    with open(path) as f:
        rows = list(csv.reader(f))
    header = rows[0]
    assert header[0] == "wav" and "QMOS" in header and "EMOS" in header, "Header sai"
    for i, r in enumerate(rows[1:], 2):
        assert len(r) == len(header), f"Dòng {i} sai số cột"
    print(f"OK: {len(rows)-1} dòng, header = {header}")

validate(answer_path)
# !cd /kaggle/working && zip -j submission_track2_exp16.zip answer.txt && unzip -l submission_track2_exp16.zip
print("Sẵn sàng nộp: /kaggle/working/submission_track2_exp16.zip")

# %% [markdown]
# ## 8. (tùy chọn) Ensemble muộn: trộn THỨ HẠNG điểm LLM + hệ trained
#
# Trung bình rank của exp16 với một `answer.txt` đã có (vd bản trộn cột exp07+exp08) cho từng cột số.
# Đa dạng nguồn → có thể giảm nhiễu. CHỈ chạy khi có sẵn file kia (đặt đường dẫn rồi bỏ comment).

# %%
def ensemble_rank_average(answer_a, answer_b, out_path):
    """Trộn 2 answer.txt theo TRUNG BÌNH THỨ HẠNG cho 5 cột số (QMOS/EMOS/VAL/ARO/DOM); CAT lấy theo A."""
    import pandas as pd
    num_cols = ["QMOS", "EMOS", "VAL", "ARO", "DOM"]
    A = pd.read_csv(answer_a); B = pd.read_csv(answer_b)
    A = A.set_index("wav"); B = B.set_index("wav").reindex(A.index)
    out = A.copy()
    for c in num_cols:
        if c in A.columns and c in B.columns:
            ra = A[c].rank(); rb = B[c].rank()
            out[c] = ((ra + rb) / 2.0)        # SRCC bất biến với scale → để nguyên rank trung bình
    out.reset_index().to_csv(out_path, index=False)
    print("Ensemble →", out_path)

# ensemble_rank_average(answer_path,
#     "/kaggle/input/.../exp_mix_q07_emo08/answer.txt",
#     os.path.join(OUT_DIR, "answer_ens.txt"))

# %% [markdown]
# ## Ghi chú nộp & paper
# - Nộp: My Submissions → **Track 2** (bỏ chọn track khác) → `submission_track2_exp16.zip` → đọc SRCC 6 cột.
# - **Bảng A (paper):** đặt SRCC exp16 (gemini/openai, zero-shot) cạnh exp07 (QMOS 0.548) + exp08
#   (EMOS 0.811 · CAT 0.133 · VAD 0.659/0.793/0.751). Kỳ vọng: LLM khá ở EMOS/CAT, yếu ở QMOS.
# - **Bảng B:** chạy lại `SHOT_MODE="few_shot"` (1 provider) → so zero vs few-shot.
# - **Cache:** Save Version để giữ `exp16_llm_cache/*.jsonl` (không trả tiền lại). Lưu thành Kaggle
#   Dataset nếu muốn dùng cho eval phase.
# - **Khai báo external resource** (API thương mại Gemini/OpenAI) trong `12_system_description.md`.
