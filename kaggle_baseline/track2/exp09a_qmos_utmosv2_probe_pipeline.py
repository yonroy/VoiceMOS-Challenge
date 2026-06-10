# %% [markdown]
# # VMC2026 Track 2 — exp09a (PROBE: UTMOSv2 vs UTMOS cho QMOS) — Kaggle
#
# **Mục đích (rẻ, KHÔNG tốn lượt nộp):** trước khi fine-tune QMOS, kiểm tra xem
# **UTMOSv2** (hệ thống **T05 — vô địch VoiceMOS Challenge 2024 Track 1**, naturalness MOS)
# có **mạnh hơn UTMOS 2022** (đang dùng) trên dữ liệu Track 2 hay không.
#
# ## Ý tưởng A/B không tốn lượt nộp
# Tập **train** Track 2 CÓ nhãn `qMOS` thật (`sets/train.csv`). Ta:
# 1. Chấm một mẫu train bằng **UTMOS** (torch.hub `utmos22_strong`) — baseline đang dùng.
# 2. Chấm cùng mẫu đó bằng **UTMOSv2** (`sarulab-speech/UTMOSv2`, MIT).
# 3. So **SRCC mỗi model vs nhãn qMOS vàng** → biết model nào "xếp hạng" giống người chấm hơn.
#
# > SRCC chấm **thứ hạng** (scale-invariant) → khỏi lo lệch thang điểm. Mẫu ~2.000 wav là đủ ổn định.
#
# ## Vì sao đáng thử
# - UTMOSv2 = #1 ở 7/16 metric VMC2024 Track 1 (bỏ xa hạng 3) → bản kế nhiệm trực tiếp của UTMOS.
# - **Lưu ý:** UTMOSv2 cũng train trên giọng *không* cảm xúc → vẫn có thể lệch domain; A/B này để
#   biết nó có **đáng** làm "neo" mạnh hơn cho head QMOS fine-tune (exp09) hay không.
#
# **Cách chạy:** GPU T4 + **Internet On** (UTMOSv2 cài từ git + tải checkpoint) → Add Input dataset
# Track 2 → sửa `DATA_ROOT` → Run All. Lần đầu để `PROBE_N=300` cho nhanh, OK rồi tăng `2000`.

# %% [markdown]
# ## 0. Cấu hình — SỬA Ở ĐÂY

# %%
import os

DATA_ROOT    = "/kaggle/input/vmc2026-track2-full/vmc2026-track2"   # << SỬA slug
WAV_DIR      = f"{DATA_ROOT}/wav"
TRAIN_CSV    = f"{DATA_ROOT}/sets/train.csv"   # lisID|wavID|qMOS|emoCat|eMOS|val|dom|aro
DEV_SCP      = f"{DATA_ROOT}/sets/dev.scp"

OUT_DIR   = "/kaggle/working"
CACHE_DIR = "/kaggle/working/qmos_probe_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

DEVICE  = "cuda"
PROBE_N = 2000     # số wav train để A/B (lần đầu để 300 cho nhanh). SRCC ~2000 mẫu đã ổn định.
SEED    = 42

# (Tùy chọn) Nếu muốn TẠO LUÔN answer.txt đổi cột QMOS←UTMOSv2 để nộp xác nhận trên DEV:
#   trỏ tới answer.txt của exp07 (giữ nguyên 5 cột cảm xúc, chỉ thay QMOS).
#   Để None nếu chỉ muốn chạy A/B nội bộ.
EXP07_ANSWER = None    # ví dụ: "/kaggle/input/exp07-answer/answer.txt"

def stem(p):
    return os.path.splitext(os.path.basename(str(p)))[0]

for p in [WAV_DIR, TRAIN_CSV, DEV_SCP]:
    print(("  ✅ " if os.path.exists(p) else "  ❌ THIẾU ") + p)

# %% [markdown]
# ## 1. Cài đặt (UTMOS + UTMOSv2)

# %%
import sys, subprocess

def pip_install(*pkgs):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], check=True)

pip_install("speechmos", "librosa", "soundfile", "pandas", "scipy", "scikit-learn", "tqdm")
# UTMOSv2 (T05) — cài từ git, cần Internet On. Checkpoint tự tải lần đầu.
pip_install("git+https://github.com/sarulab-speech/UTMOSv2.git")

# %% [markdown]
# ## 2. Nhãn qMOS vàng (gộp trung bình theo wav)

# %%
import numpy as np
import pandas as pd

def load_qmos_labels():
    """train.csv (sep '|') → dict {stem: qMOS trung bình theo wav}."""
    df = pd.read_csv(TRAIN_CSV, sep="|")
    cols = {c.lower().strip(): c for c in df.columns}
    wav_col  = cols.get("wavid") or cols.get("wav") or list(df.columns)[1]
    qmos_col = cols.get("qmos")  or cols.get("mos")
    assert qmos_col, f"Không thấy cột qMOS (cột: {list(df.columns)})"
    df["_stem"] = df[wav_col].map(stem)
    g = df.groupby("_stem")[qmos_col].mean()
    return {s: float(v) for s, v in g.items()}

qmos_gold = load_qmos_labels()
print(f"Số wav train có nhãn qMOS: {len(qmos_gold)}")

# Chọn mẫu probe (chỉ giữ wav thật sự tồn tại trên đĩa)
rng = np.random.default_rng(SEED)
all_stems = [s for s in qmos_gold if os.path.exists(os.path.join(WAV_DIR, s + ".wav"))]
rng.shuffle(all_stems)
probe_stems = all_stems[:PROBE_N]
print(f"Mẫu probe: {len(probe_stems)} / {len(all_stems)} wav tồn tại")

# %% [markdown]
# ## 3. Hàm chấm: UTMOS (cũ) và UTMOSv2 (mới) — đều cache .npz

# %%
import torch
from scipy.stats import spearmanr, pearsonr

device = DEVICE if torch.cuda.is_available() else "cpu"
print("Device:", device, ("✅ " + torch.cuda.get_device_name(0)) if device == "cuda" else "⚠️ CPU")

def score_utmos(stems, tag):
    """UTMOS 2022 (torch.hub utmos22_strong). → dict {stem: score}. Cache."""
    import librosa
    from tqdm.auto import tqdm
    cache = os.path.join(CACHE_DIR, f"utmos_{tag}.npz")
    store = {}
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        store = {k: float(z[k]) for k in z.files}
        print(f"[utmos/{tag}] nạp cache: {len(store)}")
    todo = [s for s in stems if s not in store]
    if todo:
        predictor = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong",
                                   trust_repo=True).to(device).eval()
        with torch.no_grad():
            for i, s in enumerate(tqdm(todo, desc=f"utmos {tag}")):
                wav = os.path.join(WAV_DIR, s + ".wav")
                wave, _ = librosa.load(wav, sr=16000, mono=True)
                store[s] = float(predictor(torch.from_numpy(wave).unsqueeze(0).to(device),
                                           sr=16000).mean().item())
                if (i + 1) % 500 == 0:
                    np.savez(cache, **{k: np.float32(v) for k, v in store.items()})
        np.savez(cache, **{k: np.float32(v) for k, v in store.items()})
        del predictor
        torch.cuda.empty_cache() if device == "cuda" else None
    return store

def score_utmosv2(stems, tag):
    """UTMOSv2 / T05 (sarulab-speech/UTMOSv2). → dict {stem: score}. Cache."""
    from tqdm.auto import tqdm
    cache = os.path.join(CACHE_DIR, f"utmosv2_{tag}.npz")
    store = {}
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        store = {k: float(z[k]) for k in z.files}
        print(f"[utmosv2/{tag}] nạp cache: {len(store)}")
    todo = [s for s in stems if s not in store]
    if todo:
        import utmosv2
        model = utmosv2.create_model(pretrained=True)   # ensemble, checkpoint tự tải
        for i, s in enumerate(tqdm(todo, desc=f"utmosv2 {tag}")):
            wav = os.path.join(WAV_DIR, s + ".wav")
            out = model.predict(input_path=wav)
            # predict trả về float (hoặc dict có 'predicted_mos') tùy phiên bản
            store[s] = float(out["predicted_mos"]) if isinstance(out, dict) else float(out)
            if (i + 1) % 200 == 0:
                np.savez(cache, **{k: np.float32(v) for k, v in store.items()})
        np.savez(cache, **{k: np.float32(v) for k, v in store.items()})
        del model
        torch.cuda.empty_cache() if device == "cuda" else None
    return store

# %% [markdown]
# ## 4. Chạy A/B trên mẫu train → in SRCC mỗi model vs nhãn qMOS vàng

# %%
utmos_s   = score_utmos(probe_stems, "probe")
utmosv2_s = score_utmosv2(probe_stems, "probe")

# Chỉ so trên các stem cả 2 model đều chấm được (để công bằng)
common = [s for s in probe_stems if s in utmos_s and s in utmosv2_s and s in qmos_gold]
y_gold = np.array([qmos_gold[s] for s in common])
p_v1   = np.array([utmos_s[s]   for s in common])
p_v2   = np.array([utmosv2_s[s] for s in common])
print(f"\nSố mẫu so sánh chung: {len(common)}")

srcc_v1 = spearmanr(p_v1, y_gold).correlation
srcc_v2 = spearmanr(p_v2, y_gold).correlation
lcc_v1  = pearsonr(p_v1, y_gold)[0]
lcc_v2  = pearsonr(p_v2, y_gold)[0]

print("\n📊 A/B trên TRAIN (nhãn qMOS vàng) — UTT-SRCC là metric chính:")
print(f"   UTMOS 2022 (đang dùng) : SRCC = {srcc_v1:.4f} | LCC = {lcc_v1:.4f}")
print(f"   UTMOSv2 / T05 (mới)     : SRCC = {srcc_v2:.4f} | LCC = {lcc_v2:.4f}")
delta = srcc_v2 - srcc_v1
if delta > 0.01:
    print(f"   ✅ UTMOSv2 THẮNG (+{delta:.4f} SRCC) → đáng dùng làm neo cho exp09 / đổi cột QMOS.")
elif delta < -0.01:
    print(f"   ⚠️ UTMOSv2 THUA ({delta:.4f} SRCC) → giữ UTMOS; lệch domain cảm xúc quá mạnh.")
else:
    print(f"   ➖ Ngang nhau ({delta:+.4f}) → ưu tiên model nào tiện hơn; chốt bằng fine-tune.")

# Mốc tham chiếu leaderboard: UTMOS zero-shot DEV = 0.414; head QMOS exp07 = 0.548.
# (SRCC train ≠ SRCC dev nhưng cùng xu hướng → dùng để quyết hướng, không phải điểm nộp.)
print("\nℹ️ Mốc leaderboard DEV để đối chiếu: UTMOS zero-shot 0.414 · head QMOS exp07 0.548.")

# %% [markdown]
# ## 5. (Tùy chọn) Tạo answer.txt đổi cột QMOS←UTMOSv2 để nộp xác nhận DEV
# Chỉ chạy nếu `EXP07_ANSWER` trỏ tới answer.txt exp07. Giữ nguyên 5 cột cảm xúc, chỉ thay QMOS.

# %%
def build_swapped_answer(exp07_answer_path, out_path):
    """Đọc answer.txt exp07 (wav,QMOS,EMOS,CAT,VAL,ARO,DOM), thay QMOS = UTMOSv2(dev)."""
    import csv
    with open(DEV_SCP) as f:
        dev_names = [ln.strip() for ln in f if ln.strip()]
    dev_stems = [stem(n) for n in dev_names]
    utmosv2_dev = score_utmosv2(dev_stems, "dev")    # chấm DEV bằng UTMOSv2 (cache riêng)

    with open(exp07_answer_path) as f:
        rows = list(csv.reader(f))
    header, body = rows[0], rows[1:]
    qi = header.index("QMOS")
    n_swap = 0
    with open(out_path, "w") as f:
        f.write(",".join(header) + "\n")
        for r in body:
            sid = stem(r[0])
            if sid in utmosv2_dev:
                r[qi] = f"{utmosv2_dev[sid]:.6g}"
                n_swap += 1
            f.write(",".join(r) + "\n")
    print(f"Ghi {len(body)} dòng → {out_path} | đổi QMOS được {n_swap} dòng")
    return out_path

if EXP07_ANSWER and os.path.exists(EXP07_ANSWER):
    out = os.path.join(OUT_DIR, "answer.txt")
    build_swapped_answer(EXP07_ANSWER, out)
    os.system(f"cd {OUT_DIR} && zip -j submission_track2_exp09a_utmosv2.zip answer.txt "
              f"&& unzip -l submission_track2_exp09a_utmosv2.zip")
    print("Sẵn sàng nộp:", os.path.join(OUT_DIR, "submission_track2_exp09a_utmosv2.zip"))
else:
    print("Bỏ qua mục 5 (EXP07_ANSWER=None hoặc không tồn tại). Chỉ chạy A/B nội bộ.")

# %% [markdown]
# ## Ghi chú
# - **Đọc kết quả mục 4:** UTMOSv2 SRCC có > UTMOS không?
#   - **Thắng rõ** → dùng UTMOSv2 làm **neo** cho `exp09` (fine-tune WavLM trên nhãn qMOS) thay UTMOS;
#     và/hoặc nộp answer.txt đổi cột (mục 5) để xác nhận trên leaderboard DEV.
#   - **Thua/ngang** → giữ UTMOS làm neo; kết luận "UTMOSv2 vẫn lệch domain cảm xúc" (phát hiện cho paper).
# - **Gotcha Kaggle:** UTMOSv2 cài từ git + tải checkpoint → **Internet On**. Bản nộp Internet-off cần
#   pre-download weights thành Kaggle Dataset.
# - UTMOSv2 là **ensemble nhiều fold** → chậm hơn UTMOS. Nếu lâu, giảm `PROBE_N` hoặc chấm dần (có cache).
# - License: UTMOSv2 **MIT** · UTMOS BSD-3. Ghi vào `docs/12_system_description.md`.
# - Ghi config → kết quả → nhận xét vào `docs/04_experiments_log.md` (mục exp09a).
