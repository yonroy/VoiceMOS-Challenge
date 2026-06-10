# %% [markdown]
# # VMC2026 Track 2 — exp06 (TRAIN QMOS head) — Kaggle
#
# **Mục tiêu:** QMOS là cột **duy nhất chưa train** (đang dùng UTMOS zero-shot → SRCC kẹt 0.414).
# `train.csv` CÓ sẵn cột `qMOS` → ta train 1 **head hồi quy nhỏ** trên đặc trưng SSL (đã cache ở exp04)
# để vượt 0.414.
#
# ## Ý tưởng (đọc 1 lần cho hiểu)
# - Tái dùng đặc trưng **emotion2vec + SAILER** đã trích & cache trong `fusion_cache/` (exp04) → KHÔNG trích lại.
# - Thêm **chính điểm UTMOS** (SpeechMOS) làm 1 đặc trưng đầu vào → head chỉ cần **học chỉnh sửa (residual)**
#   quanh 0.414 thay vì học lại từ đầu → an toàn, gần như chắc chắn ≥ UTMOS đơn lẻ.
# - Nhãn vàng QMOS = **TB `qMOS` theo wav** (gộp các listener trong `train.csv`).
# - Có **val nội bộ 10%** → đo SRCC, so thẳng với UTMOS trên CÙNG tập val → biết có cải thiện thật
#   **trước khi** tốn lượt nộp CodaBench.
# - Cuối cùng: **GIỮ NGUYÊN exp04** (5 cột cảm xúc đang thắng), chỉ **thay cột QMOS** trong `answer.txt`.
#
# ```
#  mỗi wav ─► [e2v_emb | e2v_probs5 | sailer_emb | sailer_probs9 | sailer_vad3 | UTMOS] ─► MLP ─► QMOS
#                                                                                  (head train)
# ```
#
# **Cách chạy trên Kaggle:** Settings → Accelerator = **GPU T4**, Internet = **On**.
# + Add Input: (1) dataset Track 2 (15.477 wav, có `sets/train.csv`) ; (2) — nếu có — dataset chứa
# `fusion_cache/*.npz` đã Save Version ở exp04 (đỡ ~15') ; (3) file `answer.txt` của exp04 để ghép cột.
# Lần đầu đặt `LIMIT_TRAIN=300`, `LIMIT_DEV=20` để bắt lỗi setup, OK rồi đặt `None`.

# %% [markdown]
# ## 0. Cấu hình — SỬA Ở ĐÂY

# %%
import os

# ── Data Track 2 ─────────────────────────────────────────────────────────────
DATA_ROOT    = "/kaggle/input/vmc2026-track2-full/vmc2026-track2"   # << SỬA slug cho khớp Add Input
WAV_DIR      = f"{DATA_ROOT}/wav"
TRAIN_CSV    = f"{DATA_ROOT}/sets/train.csv"   # nhãn người nghe: lisID|wavID|qMOS|emoCat|eMOS|val|dom|aro
DEV_SCP      = f"{DATA_ROOT}/sets/dev.scp"     # danh sách wav tập DEV

OUT_DIR   = "/kaggle/working"
# Dùng CHUNG cache với exp04. Nếu đã Save Version cache ở exp04, trỏ CACHE_DIR vào dataset đó
# (vd "/kaggle/input/<slug-cache>/fusion_cache") để khỏi trích lại; nếu không, để mặc định sẽ tự trích.
CACHE_DIR = "/kaggle/working/fusion_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# File answer.txt của exp04 (5 cột cảm xúc đang thắng) để GHÉP cột QMOS mới vào.
# Trỏ tới nơi bạn đặt file exp04. Nếu không có, notebook vẫn xuất qmos_dev.csv riêng + cảnh báo.
EXP04_ANSWER = "/kaggle/input/exp04-answer/answer.txt"   # << SỬA; hoặc "/kaggle/working/answer.txt"

# ── Đặc trưng dùng cho QMOS ──────────────────────────────────────────────────
USE_E2V        = True     # nối embedding emotion2vec
USE_SAILER     = True     # nối embedding SAILER/WavLM
USE_CLASSPROB  = True     # nối thêm xác suất lớp (e2v5 + sailer9 + vad3)
USE_UTMOS_FEAT = True     # nối thêm điểm UTMOS làm 1 đặc trưng (neo residual quanh 0.414)

# ── Siêu tham số train head ──────────────────────────────────────────────────
DEVICE      = "cuda"
HIDDEN      = 256
DROPOUT     = 0.3
LR          = 1e-3
EPOCHS      = 120
BATCH       = 64
VAL_FRAC    = 0.10
PATIENCE    = 20
SEED        = 42
RANK_LAMBDA = 0.0         # 0 = chỉ MSE. >0 (vd 0.2) = cộng thêm pairwise ranking loss (tối ưu thứ hạng=SRCC)

LIMIT_TRAIN = None        # số nhỏ (vd 300) để chạy thử; None = full
LIMIT_DEV   = None

def stem(p):
    return os.path.splitext(os.path.basename(str(p)))[0]

assert USE_E2V or USE_SAILER or USE_UTMOS_FEAT, "Phải bật ít nhất 1 nguồn đặc trưng."
print("DATA_ROOT:", DATA_ROOT)
for p in [WAV_DIR, TRAIN_CSV, DEV_SCP]:
    print(("  ✅ " if os.path.exists(p) else "  ❌ THIẾU ") + p)

# %% [markdown]
# ## 1. Cài đặt + (nếu cần) tải code SAILER
# emotion2vec qua `funasr`; SAILER cần `WavLMWrapper` trong repo `vox-profile-release` (clone + sys.path).
# Nếu cache đã đủ thì các model này sẽ KHÔNG được nạp (chỉ nạp khi còn file phải trích).

# %%
import sys, subprocess

def pip_install(*pkgs):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], check=True)

pip_install("speechmos", "funasr", "librosa", "soundfile", "pandas", "scipy", "scikit-learn", "tqdm")

if USE_SAILER:
    pip_install("loralib", "speechbrain")
    REPO_DIR = "/kaggle/working/vox-profile-release"
    if not os.path.exists(REPO_DIR):
        subprocess.run(["git", "clone", "--depth", "1",
                        "https://github.com/tiantiaf0627/vox-profile-release.git", REPO_DIR], check=True)
    if REPO_DIR not in sys.path:
        sys.path.insert(0, REPO_DIR)

# %% [markdown]
# ## 2. Nhãn vàng QMOS (gộp `qMOS` theo wavID)

# %%
import numpy as np
import pandas as pd

def load_qmos_labels():
    """train.csv (sep='|') → DataFrame [wavID, qmos] với qmos = TB theo wav."""
    df = pd.read_csv(TRAIN_CSV, sep="|")
    cols = {c.lower().strip(): c for c in df.columns}
    wav_col  = cols.get("wavid") or cols.get("wav") or list(df.columns)[1]
    qmos_col = cols.get("qmos")  or cols.get("qMOS".lower()) or cols.get("mos")
    assert qmos_col, f"Không thấy cột qMOS trong train.csv (cột: {list(df.columns)})"
    df["_stem"] = df[wav_col].map(stem)
    g = df.groupby("_stem")[qmos_col].mean().reset_index()
    g.columns = ["wavID", "qmos"]
    return g

qmos_df = load_qmos_labels()
print(f"wav train (gộp): {len(qmos_df)}")
print("qMOS:", qmos_df["qmos"].describe()[["mean", "std", "min", "max"]].to_dict())
qmos_df.head()

# %% [markdown]
# ## 3. Trích / nạp đặc trưng (cache CHUNG với exp04) + điểm UTMOS
# - `extract_e2v` / `extract_sailer`: y hệt exp04, cache `e2v_<tag>.npz` / `sailer_<tag>.npz`.
# - `extract_utmos`: chấm UTMOS từng wav → cache `utmos_<tag>.npz` (dùng vừa làm đặc trưng, vừa làm baseline so sánh).

# %%
import torch
import torch.nn.functional as F

device = DEVICE if torch.cuda.is_available() else "cpu"
print("Device:", device, ("✅ " + torch.cuda.get_device_name(0)) if device == "cuda" else "⚠️ CPU")

EMOTIONS5 = ["angry", "happy", "neutral", "sad", "surprised"]

def extract_e2v(stems, tag):
    """→ dict {stem: emb_full[D1+5]}. Cache CACHE_DIR/e2v_<tag>.npz (giống exp04)."""
    from tqdm.auto import tqdm
    cache_path = os.path.join(CACHE_DIR, f"e2v_{tag}.npz")
    store = {}
    if os.path.exists(cache_path):
        z = np.load(cache_path, allow_pickle=True)
        store = {k: z[k] for k in z.files}
        print(f"[e2v/{tag}] nạp cache: {len(store)}")
    todo = [s for s in stems if s not in store]
    if todo:
        from funasr import AutoModel
        m = AutoModel(model="iic/emotion2vec_plus_large", hub="hf", device=device)
        for i, s in enumerate(tqdm(todo, desc=f"e2v {tag}")):
            wav = os.path.join(WAV_DIR, s + ".wav")
            if not os.path.exists(wav):
                continue
            r = m.generate(wav, granularity="utterance", extract_embedding=True)[0]
            emb = np.asarray(r["feats"], dtype=np.float32).reshape(-1)
            probs = {e: 0.0 for e in EMOTIONS5}
            for lab, sc in zip(r["labels"], r["scores"]):
                name = lab.split("/")[-1]
                if name in probs:
                    probs[name] = float(sc)
            tot = sum(probs.values())
            p5 = np.array([probs[e] / tot if tot > 0 else 0.2 for e in EMOTIONS5], dtype=np.float32)
            store[s] = np.concatenate([emb, p5]).astype(np.float32)
            if (i + 1) % 500 == 0:
                np.savez(cache_path, **store)
        np.savez(cache_path, **store)
        del m
        torch.cuda.empty_cache() if device == "cuda" else None
    return store   # mỗi value = [D1 | 5]

def _pool_feat(features):
    f = features.detach().cpu().numpy()
    if f.ndim <= 1:
        return f.reshape(-1).astype(np.float32)
    return f.mean(axis=tuple(range(f.ndim - 1))).reshape(-1).astype(np.float32)

def extract_sailer(stems, tag):
    """→ dict {stem: vec[D2+9+3]}. Cache CACHE_DIR/sailer_<tag>.npz (giống exp04)."""
    import librosa
    from tqdm.auto import tqdm
    cache_path = os.path.join(CACHE_DIR, f"sailer_{tag}.npz")
    store = {}
    if os.path.exists(cache_path):
        z = np.load(cache_path, allow_pickle=True)
        store = {k: z[k] for k in z.files}
        print(f"[sailer/{tag}] nạp cache: {len(store)}")
    todo = [s for s in stems if s not in store]
    if todo:
        from src.model.emotion.wavlm_emotion import WavLMWrapper
        sailer = WavLMWrapper.from_pretrained("tiantiaf/wavlm-large-categorical-emotion").to(device).eval()
        with torch.no_grad():
            for i, s in enumerate(tqdm(todo, desc=f"sailer {tag}")):
                wav = os.path.join(WAV_DIR, s + ".wav")
                if not os.path.exists(wav):
                    continue
                wave, _ = librosa.load(wav, sr=16000, mono=True)
                wave = wave[: 15 * 16000]
                data = torch.from_numpy(wave).float().unsqueeze(0).to(device)
                logits, feat, _det, arousal, valence, dominance = sailer(data, return_feature=True)
                emb = _pool_feat(feat)
                p9 = F.softmax(logits, dim=1)[0].detach().cpu().numpy().astype(np.float32)
                vad3 = np.array([1 + 4 * float(valence.item()),
                                 1 + 4 * float(arousal.item()),
                                 1 + 4 * float(dominance.item())], dtype=np.float32)
                store[s] = np.concatenate([emb, p9, vad3]).astype(np.float32)
                if (i + 1) % 500 == 0:
                    np.savez(cache_path, **store)
        np.savez(cache_path, **store)
        del sailer
        torch.cuda.empty_cache() if device == "cuda" else None
    return store   # mỗi value = [D2 | 9 | 3]

def extract_utmos(names, tag):
    """Chấm UTMOS từng wav (theo TÊN file, vì DEV gọi .wav theo tên). → dict {stem: score}.
    Cache CACHE_DIR/utmos_<tag>.npz. Dùng vừa làm đặc trưng vừa làm baseline so sánh."""
    import librosa
    from tqdm.auto import tqdm
    cache_path = os.path.join(CACHE_DIR, f"utmos_{tag}.npz")
    store = {}
    if os.path.exists(cache_path):
        z = np.load(cache_path, allow_pickle=True)
        store = {k: float(z[k]) for k in z.files}
        print(f"[utmos/{tag}] nạp cache: {len(store)}")
    todo = [n for n in names if stem(n) not in store]
    if todo:
        predictor = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong",
                                   trust_repo=True).to(device).eval()
        with torch.no_grad():
            for i, n in enumerate(tqdm(todo, desc=f"utmos {tag}")):
                wav = os.path.join(WAV_DIR, n if n.endswith(".wav") else n + ".wav")
                if not os.path.exists(wav):
                    continue
                wave, _ = librosa.load(wav, sr=16000, mono=True)
                sc = float(predictor(torch.from_numpy(wave).unsqueeze(0).to(device), sr=16000).mean().item())
                store[stem(n)] = sc
                if (i + 1) % 500 == 0:
                    np.savez(cache_path, **{k: np.float32(v) for k, v in store.items()})
        np.savez(cache_path, **{k: np.float32(v) for k, v in store.items()})
        del predictor
        torch.cuda.empty_cache() if device == "cuda" else None
    return store

# %% [markdown]
# ## 4. Dựng feature + nhãn cho train

# %%
train_stems = list(qmos_df["wavID"])
if LIMIT_TRAIN:
    train_stems = train_stems[:LIMIT_TRAIN]

e2v_tr    = extract_e2v(train_stems, "train")    if USE_E2V    else {}
sailer_tr = extract_sailer(train_stems, "train") if USE_SAILER else {}
utmos_tr  = extract_utmos(train_stems, "train")  if USE_UTMOS_FEAT else {}

def qmos_feature(sid, e2v_map, sailer_map, utmos_map):
    """Nối đặc trưng QMOS cho 1 wav. None nếu thiếu phần bắt buộc."""
    parts = []
    if USE_E2V:
        v = e2v_map.get(sid)
        if v is None:
            return None
        parts.append(v[:-5])                      # emb e2v
        if USE_CLASSPROB:
            parts.append(v[-5:])                  # probs5
    if USE_SAILER:
        v = sailer_map.get(sid)
        if v is None:
            return None
        parts.append(v[:-12])                     # emb sailer
        if USE_CLASSPROB:
            parts.append(v[-12:])                 # probs9 + vad3
    if USE_UTMOS_FEAT:
        u = utmos_map.get(sid)
        if u is None:
            return None
        parts.append(np.array([u], dtype=np.float32))
    return np.concatenate(parts).astype(np.float32)

lab = qmos_df.set_index("wavID")["qmos"]
X, y = [], []
for s in train_stems:
    f = qmos_feature(s, e2v_tr, sailer_tr, utmos_tr)
    if f is None or s not in lab.index:
        continue
    X.append(f)
    y.append(float(lab.loc[s]))

X = np.stack(X).astype(np.float32)
y = np.array(y, dtype=np.float32)
FEAT_DIM = X.shape[1]
print(f"Train: X={X.shape} y={y.shape}")

feat_mean = X.mean(0, keepdims=True)
feat_std  = X.std(0, keepdims=True) + 1e-6
Xn = (X - feat_mean) / feat_std
y_mu, y_sd = float(y.mean()), float(y.std() + 1e-6)
yn = (y - y_mu) / y_sd

# %% [markdown]
# ## 5. Train head QMOS + so với UTMOS trên CÙNG val nội bộ
# - Head = MLP nhỏ (`Linear→ReLU→Dropout ×2 → 1`). Loss = MSE (+ tùy chọn pairwise ranking).
# - In **SRCC head** và **SRCC UTMOS** trên cùng tập val → biết head có thật sự vượt 0.414 không.

# %%
import torch.nn as nn
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split

torch.manual_seed(SEED); np.random.seed(SEED)
idx_all = np.arange(X.shape[0])
tr_idx, va_idx = train_test_split(idx_all, test_size=VAL_FRAC, random_state=SEED)

def to_t(a):
    return torch.tensor(a, dtype=torch.float32, device=device)

Xn_t = to_t(Xn); yn_t = to_t(yn).unsqueeze(1)

class QMOSHead(nn.Module):
    def __init__(self, d_in, h, p):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, h), nn.ReLU(), nn.Dropout(p),
            nn.Linear(h, h), nn.ReLU(), nn.Dropout(p),
            nn.Linear(h, 1),
        )

    def forward(self, x):
        return self.net(x)

model = QMOSHead(FEAT_DIM, HIDDEN, DROPOUT).to(device)
opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
mse = nn.MSELoss()

def pairwise_rank_loss(pred, target):
    """Khuyến khích pred xếp hạng giống target (margin ranking trên các cặp trong batch)."""
    n = pred.shape[0]
    if n < 2:
        return torch.zeros((), device=device)
    pi, pj = pred.unsqueeze(0), pred.unsqueeze(1)
    ti, tj = target.unsqueeze(0), target.unsqueeze(1)
    sign = torch.sign(ti - tj)                       # +1 nếu i nên cao hơn j
    diff = pi - pj
    # hinge: phạt khi thứ tự sai
    return torch.relu(-sign * diff).mean()

@torch.no_grad()
def eval_val():
    model.eval()
    p = model(Xn_t[va_idx]).cpu().numpy().ravel()
    srcc_head = spearmanr(p, y[va_idx]).correlation
    out = {"head": float(srcc_head)}
    if USE_UTMOS_FEAT:
        u = X[va_idx, -1]                            # cột UTMOS (đặc trưng cuối, chưa chuẩn hóa)
        out["utmos"] = float(spearmanr(u, y[va_idx]).correlation)
    return out

best, best_state, bad = -1e9, None, 0
tr_t = torch.tensor(tr_idx, device=device)
for ep in range(1, EPOCHS + 1):
    model.train()
    perm = tr_t[torch.randperm(len(tr_t), device=device)]
    run = 0.0
    for i in range(0, len(perm), BATCH):
        b = perm[i:i + BATCH]
        opt.zero_grad()
        pred = model(Xn_t[b])
        loss = mse(pred, yn_t[b])
        if RANK_LAMBDA > 0:
            loss = loss + RANK_LAMBDA * pairwise_rank_loss(pred.ravel(), yn_t[b].ravel())
        loss.backward(); opt.step()
        run += loss.item() * len(b)
    m = eval_val()
    if m["head"] > best:
        best = m["head"]
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        bad = 0
    else:
        bad += 1
    if ep % 5 == 0 or ep == 1:
        extra = f" | UTMOS={m['utmos']:.4f}" if "utmos" in m else ""
        print(f"epoch {ep:3d} | loss {run/len(perm):.4f} | head SRCC={m['head']:.4f}{extra} | best {best:.4f}")
    if bad >= PATIENCE:
        print(f"Early stop ở epoch {ep}.")
        break

model.load_state_dict(best_state)
final = eval_val()
print("\n✅ VAL (nội bộ):")
print(f"   QMOS head SRCC = {final['head']:.4f}")
if "utmos" in final:
    print(f"   UTMOS  baseline = {final['utmos']:.4f}  (mốc leaderboard 0.414)")
    print("   →", "✅ HEAD VƯỢT UTMOS" if final["head"] > final["utmos"] else "⚠️ chưa vượt — thử tăng EPOCHS / RANK_LAMBDA / bật thêm đặc trưng")

torch.save({"state": best_state, "feat_mean": feat_mean, "feat_std": feat_std,
            "y_mu": y_mu, "y_sd": y_sd, "FEAT_DIM": FEAT_DIM,
            "USE_E2V": USE_E2V, "USE_SAILER": USE_SAILER,
            "USE_CLASSPROB": USE_CLASSPROB, "USE_UTMOS_FEAT": USE_UTMOS_FEAT,
            "val_srcc": best}, os.path.join(OUT_DIR, "qmos_head.pt"))
print("Đã lưu", os.path.join(OUT_DIR, "qmos_head.pt"))

# %% [markdown]
# ## 6. Dự đoán QMOS cho DEV

# %%
def list_dev():
    with open(DEV_SCP) as f:
        return [ln.strip() for ln in f if ln.strip()]

dev_names = list_dev()
if LIMIT_DEV:
    dev_names = dev_names[:LIMIT_DEV]
dev_stems = [stem(n) for n in dev_names]
print("DEV:", len(dev_names), "mẫu")

e2v_dev    = extract_e2v(dev_stems, "dev")    if USE_E2V    else {}
sailer_dev = extract_sailer(dev_stems, "dev") if USE_SAILER else {}
utmos_dev  = extract_utmos(dev_names, "dev")  if USE_UTMOS_FEAT else {}

@torch.no_grad()
def predict_qmos(sid):
    f = qmos_feature(sid, e2v_dev, sailer_dev, utmos_dev)
    if f is None:
        return None
    fn = (f[None, :] - feat_mean) / feat_std
    model.eval()
    return float(model(to_t(fn)).item()) * y_sd + y_mu     # đảo z-score

qmos_pred = {}
n_real = n_def = 0
for n in dev_names:
    sid = stem(n)
    p = predict_qmos(sid)
    if p is None:
        p = utmos_dev.get(sid, 3.0)                          # rơi về UTMOS nếu thiếu feature
        n_def += 1
    else:
        n_real += 1
    qmos_pred[n] = p
print(f"QMOS dự đoán: head thật {n_real}, dự phòng UTMOS {n_def}")

# Lưu riêng (để ghép tay nếu cần)
import csv
qmos_csv = os.path.join(OUT_DIR, "qmos_dev.csv")
with open(qmos_csv, "w", newline="") as f:
    w = csv.writer(f); w.writerow(["wav", "QMOS"])
    for n in dev_names:
        w.writerow([n, f"{qmos_pred[n]:.6g}"])
print("Đã ghi", qmos_csv)

# %% [markdown]
# ## 7. Ghép QMOS mới vào answer.txt của exp04 → bản nộp mới
# Giữ NGUYÊN 5 cột cảm xúc đang thắng (EMOS/CAT/VAL/ARO/DOM), chỉ thay cột QMOS.

# %%
def merge_into_exp04(exp04_path, out_path):
    if not os.path.exists(exp04_path):
        print(f"⚠️ Không thấy {exp04_path} → BỎ QUA ghép. Hãy dùng qmos_dev.csv để thay cột QMOS thủ công,")
        print("   hoặc trỏ EXP04_ANSWER đúng đường dẫn answer.txt của exp04 rồi chạy lại cell này.")
        return False
    with open(exp04_path) as f:
        rows = list(csv.reader(f))
    header = rows[0]
    qi = header.index("QMOS")
    wi = header.index("wav")
    n_swapped = n_miss = 0
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(header)
        for r in rows[1:]:
            name = r[wi]
            if name in qmos_pred:
                r[qi] = f"{qmos_pred[name]:.6g}"; n_swapped += 1
            else:
                n_miss += 1
            w.writerow(r)
    print(f"Ghép xong → {out_path} | thay {n_swapped} cột QMOS, thiếu {n_miss} (giữ QMOS cũ)")
    return True

merged = os.path.join(OUT_DIR, "answer.txt")
ok = merge_into_exp04(EXP04_ANSWER, merged)

if ok:
    # validate + zip
    with open(merged) as f:
        rows = list(csv.reader(f))
    assert rows[0][0] == "wav" and "QMOS" in rows[0]
    for i, r in enumerate(rows[1:], 2):
        assert len(r) == len(rows[0]), f"Dòng {i} sai số cột"
    print(f"OK: {len(rows)-1} dòng, header = {rows[0]}")
    os.system(f"cd {OUT_DIR} && zip -j submission_track2_exp06_qmos.zip answer.txt "
              f"&& unzip -l submission_track2_exp06_qmos.zip")
    print("Sẵn sàng nộp:", os.path.join(OUT_DIR, "submission_track2_exp06_qmos.zip"))

# %% [markdown]
# ## Ghi chú
# - **Lần đầu** đặt `LIMIT_TRAIN=300`, `LIMIT_DEV=20` để bắt lỗi; OK rồi đặt `None`.
# - **So sánh công bằng**: mục 5 in cả `head SRCC` và `UTMOS SRCC` trên CÙNG val nội bộ → chỉ nộp khi head > UTMOS.
# - Nếu head **chưa vượt** 0.414: thử (a) tăng `EPOCHS`; (b) bật `RANK_LAMBDA=0.2` (tối ưu thứ hạng);
#   (c) đảm bảo `USE_UTMOS_FEAT=True` (neo residual); (d) thử bỏ bớt đặc trưng nhiễu (tắt `USE_CLASSPROB`).
# - **Ablation QMOS cho paper**: bật/tắt `USE_E2V/USE_SAILER/USE_UTMOS_FEAT/USE_CLASSPROB` → ghi `docs/04_experiments_log.md` (exp06).
# - Cache dùng CHUNG `fusion_cache/` với exp04 → nhớ **Save Version** giữ lại (gồm `utmos_*.npz` mới).
# - Ghi config → kết quả → nhận xét vào `docs/04_experiments_log.md` (mục exp06).
