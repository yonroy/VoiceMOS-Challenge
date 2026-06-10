# %% [markdown]
# # VMC2026 Track 2 — exp02 (EMOS có train) — Kaggle
#
# **Mục tiêu:** train một model dự đoán **EMOS** (độ khớp cảm xúc target) từ ~12.746 mẫu
# có nhãn người nghe trong `sets/train.csv`, kỳ vọng **vượt baseline 0.194** (exp01 offline).
#
# ## Ý tưởng (đọc 1 lần cho hiểu)
# EMOS phụ thuộc **cả audio LẪN cảm xúc target** (cùng audio "vui": target=happy → điểm cao,
# target=sad → điểm thấp). Vì vậy model phải nhận vào cả hai:
#
# ```
# mỗi wav ─► emotion2vec ─► (a) embedding ~D chiều   ┐
#                           (b) xác suất 5 cảm xúc    ├─► nối ─► MLP head ─► EMOS (1–5)
#       target emotion ───► one-hot 5 chiều           ┘            (CÁI MÌNH TRAIN)
# ```
#
# - **Backbone emotion2vec ĐÓNG BĂNG** (không train lại) → chỉ trích đặc trưng. Nhẹ GPU, ít data vẫn ổn.
# - **Chỉ train MLP head nhỏ** → học ánh xạ `(đặc trưng + target) → điểm người chấm`.
# - **Nhãn vàng** = trung bình `eMOS` của mọi listener trên cùng 1 wav (gộp theo `wavID`).
# - Embedding **trích 1 lần → cache .npz** (12.746 file rất lâu, chạy lại tốn giờ GPU).
# - Tách 10% train làm **validation nội bộ** → đo SRCC trong lúc train (DEV không có nhãn để tự chấm).
# - Cuối cùng xuất `answer.txt` **đầy đủ**: QMOS=SpeechMOS · CAT=emotion2vec · **EMOS=head vừa train** → nộp được ngay.
#
# **Cách chạy trên Kaggle:** Settings → Accelerator = **GPU T4**, Internet = **On** → + Add Input dataset
# Track 2 (15.477 wav, có `sets/train.csv`) → sửa `DATA_ROOT` ở cell 0 → Run All.

# %% [markdown]
# ## 0. Cấu hình — SỬA Ở ĐÂY

# %%
import os, glob, json, time

# ── Data Track 2 (dataset 15.477 wav đã ráp, có sets/train.csv) ──────────────
DATA_ROOT    = "/kaggle/input/vmc2026-track2-full/vmc2026-track2"   # << SỬA slug cho khớp Add Input
WAV_DIR      = f"{DATA_ROOT}/wav"
METADATA_CSV = f"{DATA_ROOT}/metadata.csv"     # wavID|emotion|transcript (KHÔNG header) → target emotion
TRAIN_CSV    = f"{DATA_ROOT}/sets/train.csv"   # nhãn người nghe: lisID,wavID,qMOS,emoCat,eMOS,val,dom,aro
DEV_SCP      = f"{DATA_ROOT}/sets/dev.scp"     # danh sách wav tập DEV (tập cần nộp ở training phase)

OUT_DIR   = "/kaggle/working"
CACHE_DIR = "/kaggle/working/emb_cache"        # nơi lưu embedding đã trích (tái dùng giữa các lần chạy)
os.makedirs(CACHE_DIR, exist_ok=True)

# ── Siêu tham số train (đổi nếu muốn thử nghiệm) ─────────────────────────────
DEVICE        = "cuda"      # "cuda" trên Kaggle GPU; "cpu" nếu không có GPU
HIDDEN        = 256         # số neuron lớp ẩn của MLP head
DROPOUT       = 0.3
LR            = 1e-3
EPOCHS        = 60
BATCH         = 64
VAL_FRAC      = 0.10        # 10% train → validation nội bộ (đo SRCC)
PATIENCE      = 12          # early stop: dừng nếu val-SRCC không cải thiện sau N epoch
SEED          = 42

LIMIT_TRAIN   = None        # đặt số nhỏ (vd 300) để chạy thử nhanh; None = full
USE_CLASSPROB = True        # thêm 5 xác suất cảm xúc của emotion2vec vào feature (tín hiệu exp01)

EMOTIONS5 = ["angry", "happy", "neutral", "sad", "surprised"]

_EMO_ALIAS = {
    "angry": "angry", "anger": "angry",
    "happy": "happy", "happiness": "happy", "joy": "happy",
    "neutral": "neutral", "calm": "neutral",
    "sad": "sad", "sadness": "sad",
    "surprise": "surprised", "surprised": "surprised", "surprising": "surprised",
}

def norm_emotion(label):
    """Đưa nhãn cảm xúc bất kỳ về 1 trong EMOTIONS5; None nếu không khớp."""
    key = str(label).strip().lower()
    return _EMO_ALIAS.get(key, key if key in EMOTIONS5 else None)

def stem(path_or_name):
    """Lấy tên file không đuôi, để khớp wavID giữa train.csv / metadata / dev.scp."""
    return os.path.splitext(os.path.basename(str(path_or_name)))[0]

print("DATA_ROOT:", DATA_ROOT)
for p in [WAV_DIR, METADATA_CSV, TRAIN_CSV, DEV_SCP]:
    print(("  ✅ " if os.path.exists(p) else "  ❌ THIẾU ") + p)

# %% [markdown]
# ## 1. Cài đặt

# %%
# !pip install -q speechmos funasr librosa soundfile pandas scipy scikit-learn tqdm

# %% [markdown]
# ## 2. Đọc & gộp nhãn
# - `train.csv`: mỗi dòng = 1 listener chấm 1 wav → **gộp trung bình eMOS theo wavID** = nhãn vàng.
# - `metadata.csv`: lấy **cảm xúc target** cho mỗi wav (chuẩn hóa về 5 lớp).

# %%
import pandas as pd

def load_target_emotions():
    """metadata.csv (wavID|emotion|transcript, KHÔNG header) → {stem: emotion_chuẩn|None}."""
    tgt = {}
    with open(METADATA_CSV, encoding="utf-8") as f:
        for ln in f:
            parts = ln.strip().split("|")
            if len(parts) < 2:
                continue
            tgt[stem(parts[0])] = norm_emotion(parts[1])
    return tgt

def load_train_labels():
    """train.csv → DataFrame [wavID(stem), emos] đã gộp trung bình theo wav."""
    df = pd.read_csv(TRAIN_CSV)
    # Chuẩn hóa tên cột (phòng khi viết hoa/thường khác nhau)
    cols = {c.lower().strip(): c for c in df.columns}
    wav_col = cols.get("wavid") or cols.get("wav") or list(df.columns)[1]
    emos_col = cols.get("emos") or cols.get("emo") or cols.get("emomos")
    assert emos_col, f"Không thấy cột eMOS trong train.csv (cột hiện có: {list(df.columns)})"
    g = df.groupby(df[wav_col].map(stem))[emos_col].mean()
    out = g.reset_index()
    out.columns = ["wavID", "emos"]
    return out

target_map = load_target_emotions()
train_df = load_train_labels()
print(f"Target emotions: {len(target_map)} | wav train (đã gộp): {len(train_df)}")
print("eMOS thống kê:", train_df["emos"].describe()[["mean", "std", "min", "max"]].to_dict())
train_df.head()

# %% [markdown]
# ## 3. Trích đặc trưng emotion2vec (có cache)
# Mỗi wav → 1 lần `generate(extract_embedding=True)` cho ra **embedding** (cho EMOS) +
# **xác suất 5 lớp** (cho CAT và làm feature). Lưu cache `.npz` để lần sau khỏi chạy lại.

# %%
import numpy as np

_e2v_model = None
def get_e2v():
    global _e2v_model
    if _e2v_model is None:
        from funasr import AutoModel
        _e2v_model = AutoModel(model="iic/emotion2vec_plus_large", hub="hf")
    return _e2v_model

def extract_one(wav_path):
    """→ (emb: np.float32[D], probs5: np.float32[5] tổng=1). None nếu lỗi/thiếu file."""
    if not os.path.exists(wav_path):
        return None
    rec = get_e2v().generate(wav_path, granularity="utterance", extract_embedding=True)
    r = rec[0]
    emb = np.asarray(r["feats"], dtype=np.float32).reshape(-1)
    probs = {e: 0.0 for e in EMOTIONS5}
    for lab, sc in zip(r["labels"], r["scores"]):
        name = lab.split("/")[-1]
        if name in probs:
            probs[name] = float(sc)
    tot = sum(probs.values())
    if tot > 0:
        probs = {k: v / tot for k, v in probs.items()}
    probs5 = np.array([probs[e] for e in EMOTIONS5], dtype=np.float32)
    return emb, probs5

def extract_set(stems, tag):
    """Trích (hoặc nạp cache) cho danh sách stem. Trả về dict {stem: (emb, probs5)}.
    Cache lưu tại CACHE_DIR/<tag>.npz; tự bỏ qua stem đã có để chạy nối tiếp được."""
    from tqdm.auto import tqdm
    cache_path = os.path.join(CACHE_DIR, f"{tag}.npz")
    store = {}
    if os.path.exists(cache_path):
        z = np.load(cache_path, allow_pickle=True)
        store = {k: z[k] for k in z.files}
        print(f"[{tag}] nạp cache: {len(store)} mẫu")
    todo = [s for s in stems if s not in store]
    if not todo:
        print(f"[{tag}] đủ cache, bỏ qua trích.")
    else:
        miss = 0
        for i, s in enumerate(tqdm(todo, desc=f"trích {tag}")):
            res = extract_one(os.path.join(WAV_DIR, s + ".wav"))
            if res is None:
                miss += 1
                continue
            emb, probs5 = res
            store[s] = np.concatenate([emb, probs5]).astype(np.float32)  # [D + 5]
            if (i + 1) % 500 == 0:   # lưu cache định kỳ phòng ngắt session
                np.savez(cache_path, **store)
        np.savez(cache_path, **store)
        if miss:
            print(f"[{tag}] {miss} file thiếu/ lỗi → bỏ qua.")
        print(f"[{tag}] tổng cache: {len(store)} mẫu → {cache_path}")
    # tách lại thành (emb, probs5)
    out = {}
    for s, vec in store.items():
        out[s] = (vec[:-5], vec[-5:])
    return out

# Trích cho tập train
train_stems = list(train_df["wavID"])
if LIMIT_TRAIN:
    train_stems = train_stems[:LIMIT_TRAIN]
train_feat = extract_set(train_stems, "train")
EMB_DIM = next(iter(train_feat.values()))[0].shape[0]
print("EMB_DIM =", EMB_DIM)

# %% [markdown]
# ## 4. Dựng feature + nhãn cho train
# Feature mỗi wav = `[embedding | (probs5 nếu bật) | one-hot target(5)]`. Bỏ wav thiếu target/feature.

# %%
def onehot_target(tgt):
    v = np.zeros(len(EMOTIONS5), dtype=np.float32)
    if tgt in EMOTIONS5:
        v[EMOTIONS5.index(tgt)] = 1.0
    return v

def build_feature(stem_id, feat_map):
    pack = feat_map.get(stem_id)
    if pack is None:
        return None
    emb, probs5 = pack
    tgt = target_map.get(stem_id)
    if tgt is None:           # không biết cảm xúc target → không train được mẫu này
        return None
    parts = [emb]
    if USE_CLASSPROB:
        parts.append(probs5)
    parts.append(onehot_target(tgt))
    return np.concatenate(parts).astype(np.float32)

emos_label = dict(zip(train_df["wavID"], train_df["emos"]))
X, y = [], []
for s in train_stems:
    f = build_feature(s, train_feat)
    if f is None or s not in emos_label:
        continue
    X.append(f); y.append(emos_label[s])
X = np.stack(X); y = np.array(y, dtype=np.float32)
FEAT_DIM = X.shape[1]
print(f"Train: X={X.shape}  y={y.shape}  FEAT_DIM={FEAT_DIM}")

# Chuẩn hóa feature (z-score) — lưu mean/std để áp dụng y hệt lúc dự đoán DEV.
feat_mean = X.mean(0, keepdims=True)
feat_std  = X.std(0, keepdims=True) + 1e-6
Xn = (X - feat_mean) / feat_std

# %% [markdown]
# ## 5. Model (MLP head) + train loop
# Loss = MSE. Theo dõi **SRCC** trên validation nội bộ; lưu model tốt nhất (early stopping).

# %%
import torch, torch.nn as nn
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split

torch.manual_seed(SEED); np.random.seed(SEED)
device = DEVICE if torch.cuda.is_available() else "cpu"
print("Device:", device)

Xtr, Xva, ytr, yva = train_test_split(Xn, y, test_size=VAL_FRAC, random_state=SEED)
Xtr_t = torch.tensor(Xtr, device=device); ytr_t = torch.tensor(ytr, device=device).unsqueeze(1)
Xva_t = torch.tensor(Xva, device=device); yva_t = torch.tensor(yva, device=device).unsqueeze(1)

class EmosHead(nn.Module):
    def __init__(self, d_in, hidden, p):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.ReLU(), nn.Dropout(p),
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(p),
            nn.Linear(hidden // 2, 1),
        )
    def forward(self, x):
        return self.net(x)

model = EmosHead(FEAT_DIM, HIDDEN, DROPOUT).to(device)
opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
lossf = nn.MSELoss()

def val_srcc():
    model.eval()
    with torch.no_grad():
        pred = model(Xva_t).cpu().numpy().ravel()
    return spearmanr(pred, yva).correlation

best_srcc, best_state, bad = -1.0, None, 0
n = Xtr_t.shape[0]
for ep in range(1, EPOCHS + 1):
    model.train()
    perm = torch.randperm(n, device=device)
    tot = 0.0
    for i in range(0, n, BATCH):
        idx = perm[i:i + BATCH]
        opt.zero_grad()
        out = model(Xtr_t[idx])
        loss = lossf(out, ytr_t[idx])
        loss.backward(); opt.step()
        tot += loss.item() * len(idx)
    srcc = val_srcc()
    if srcc > best_srcc:
        best_srcc, best_state, bad = srcc, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
    else:
        bad += 1
    if ep % 5 == 0 or ep == 1:
        print(f"epoch {ep:3d} | train MSE {tot/n:.4f} | val SRCC {srcc:.4f} | best {best_srcc:.4f}")
    if bad >= PATIENCE:
        print(f"Early stop ở epoch {ep} (val SRCC không tăng {PATIENCE} epoch).")
        break

model.load_state_dict(best_state)
print(f"\n✅ VAL SRCC tốt nhất = {best_srcc:.4f}  (baseline exp01 ≈ 0.194 — so ở đây)")

# Lưu model + tham số chuẩn hóa để tái dùng / mô tả hệ thống.
torch.save({"state": best_state, "feat_mean": feat_mean, "feat_std": feat_std,
            "EMB_DIM": EMB_DIM, "FEAT_DIM": FEAT_DIM, "USE_CLASSPROB": USE_CLASSPROB,
            "EMOTIONS5": EMOTIONS5, "val_srcc": float(best_srcc)},
           os.path.join(OUT_DIR, "emos_head.pt"))
print("Đã lưu", os.path.join(OUT_DIR, "emos_head.pt"))

# %% [markdown]
# ## 6. Dự đoán DEV → `answer.txt` đầy đủ
# - **EMOS** = head vừa train (cần embedding + target của từng wav DEV).
# - **CAT** = xác suất 5 lớp emotion2vec (đã có sẵn khi trích đặc trưng).
# - **QMOS** = SpeechMOS (UTMOS) — bắt buộc, chạy thêm ở đây để answer.txt hợp lệ.

# %%
def list_dev():
    with open(DEV_SCP) as f:
        return [ln.strip() for ln in f if ln.strip()]   # tên file .wav

dev_names = list_dev()
dev_stems = [stem(n) for n in dev_names]
print("DEV:", len(dev_names), "mẫu")

# 6a. Trích đặc trưng emotion2vec cho DEV (cache riêng)
dev_feat = extract_set(dev_stems, "dev")

# 6b. EMOS từ head đã train
def predict_emos(stem_id):
    f = build_feature(stem_id, dev_feat)
    if f is None:
        return None
    fn = (f[None, :] - feat_mean) / feat_std
    model.eval()
    with torch.no_grad():
        return float(model(torch.tensor(fn, dtype=torch.float32, device=device)).item())

# 6c. QMOS = SpeechMOS
def run_qmos(names):
    import librosa
    predictor = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True)
    out = {}
    from tqdm.auto import tqdm
    for n in tqdm(names, desc="QMOS"):
        p = os.path.join(WAV_DIR, n)
        if not os.path.exists(p):
            continue
        wave, _ = librosa.load(p, sr=16000, mono=True)
        out[n] = float(predictor(torch.from_numpy(wave).unsqueeze(0), sr=16000).mean().item())
    return out

qmos_scores = run_qmos(dev_names)

# %%
def fmt_cat(probs5):
    return "|".join(f"{e}:{probs5[i]:.6g}" for i, e in enumerate(EMOTIONS5))

def build_answer(out_path):
    n_emos = n_default = 0
    with open(out_path, "w") as f:
        f.write("wav,QMOS,EMOS,CAT\n")
        for name in dev_names:
            sid = stem(name)
            emos = predict_emos(sid)
            if emos is None:
                emos = 3.0; n_default += 1
            else:
                n_emos += 1
            qmos = qmos_scores.get(name, 3.0)
            probs5 = dev_feat[sid][1] if sid in dev_feat else np.full(5, 0.2, dtype=np.float32)
            f.write(f"{name},{qmos:.6g},{emos:.6g},{fmt_cat(probs5)}\n")
    print(f"Ghi {len(dev_names)} dòng → {out_path} | EMOS thật {n_emos}, mặc định {n_default}")

answer_path = os.path.join(OUT_DIR, "answer.txt")
build_answer(answer_path)

# %% [markdown]
# ## 7. Validate + đóng zip

# %%
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
# !cd /kaggle/working && zip -j submission_track2_exp02.zip answer.txt && unzip -l submission_track2_exp02.zip
print("Sẵn sàng nộp: /kaggle/working/submission_track2_exp02.zip")

# %% [markdown]
# ## Ghi chú
# - **VAL SRCC** in ở mục 5 là ước lượng nội bộ (10% train) — so với baseline 0.194 để biết có khá hơn không.
#   Điểm DEV thật phải nộp lên CodaBench mới biết (My Submissions → Track 2, bỏ chọn track khác).
# - Muốn thử nhanh: đặt `LIMIT_TRAIN = 300` ở cell 0.
# - Embedding đã cache trong `/kaggle/working/emb_cache/` → **Save Version** để giữ, lần sau train head khỏi trích lại.
# - Hướng cải tiến tiếp: thêm head QMOS/CAT/VAD dùng chung backbone (exp02 multi-task đầy đủ);
#   thử backbone wav2vec2/WavLM; thêm ranking loss; fine-tune nhẹ backbone.
# - Nhớ ghi config → kết quả → nhận xét vào `docs/04_experiments_log.md` (mục exp02).
