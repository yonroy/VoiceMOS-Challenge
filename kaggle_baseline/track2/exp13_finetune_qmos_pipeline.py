# %% [markdown]
# # VMC2026 Track 2 — exp13 (FINE-TUNE UTMOS cho QMOS) + answer 6 cột — Kaggle
#
# **Mục tiêu:** QMOS hiện tốt nhất = 0.548 (exp07, head ĐÓNG BĂNG + neo UTMOS). exp13 thử **MỞ BĂNG
# (fine-tune) thẳng UTMOS** trên nhãn `qMOS` thật của Track 2 → kéo model chất lượng về đúng domain giọng
# cảm xúc. Sau đó **mượn 5 cột cảm xúc từ checkpoint exp08** (`ft_emotion_full_20epoch.pt` — bản TỐT NHẤT)
# → ghép `answer.txt` 6 cột.
#
# ## Vì sao fine-tune UTMOS (không phải UTMOSv2)
# - UTMOS (`utmos22_strong`, tarepan/SpeechMOS) = **1 model đơn**, tải qua `torch.hub`, **bản thân đã dự đoán
#   QMOS** → warm-start hoàn hảo cho cột chất lượng (khác UTMOSv2 = ensemble nhiều fold + 2 luồng → khó train).
# - forward: `model(wave[B,T], sr) -> MOS[B]`, là `nn.Module` chuẩn → backprop được toàn model.
# - **Không dùng neo UTMOS riêng** (đã chốt): khi fine-tune chính UTMOS thì "neo" nằm sẵn trong trọng số
#   warm-start → head/neo ngoài là thừa.
#
# ## Thiết kế
# ```
#  [PHẦN A] wav ─► UTMOS (utmos22_strong, TRAINABLE, warm-start pretrained) ─► QMOS    (train trên qMOS gold)
#  [PHẦN B] wav ─► WavLM(exp08 ft) + audeering(frozen) ─► EMOS/CAT/VAD               (NẠP ckpt, chỉ inference)
#  [PHẦN C] ghép QMOS(A) + 5 cột cảm xúc(B) ─► answer.txt 6 cột ─► validate ─► zip
# ```
#
# ## ⚠️ Phải biết trước
# - Fine-tune = **không cache** (mỗi epoch chạy lại UTMOS forward+backward) → tốn giờ GPU. **Lần đầu BẮT BUỘC
#   `LIMIT_TRAIN=300`, `LIMIT_DEV=20`** để chỉnh trơn rồi mới `None`.
# - Lưới an toàn: chỉ nộp QMOS fine-tune nếu **SRCC val nội bộ > zero-shot UTMOS** (mục A in cả 2 số).
# - **Lưu checkpoint `ft_qmos_utmos.pt` mỗi best + Save Version NGAY** (bài học exp08: kernel chết là mất).
#
# **Cách chạy Kaggle:** GPU **T4** + Internet **On** → Add Input (1) dataset Track 2, (2) dataset chứa
# `ft_emotion_full.pt` (exp08), (3) tùy chọn cache `aud_dev.npz` → sửa slug cell 0 → Run All.

# %% [markdown]
# ## 0. Cấu hình — SỬA Ở ĐÂY

# %%
import os, shutil, glob

# ── TỰ DÒ DATA_ROOT (quét /kaggle/input tìm thư mục có sets/train.csv + wav/ + metadata.csv) ──
def find_data_root(search_root="/kaggle/input"):
    cands = []
    for train_csv in glob.glob(os.path.join(search_root, "**", "sets", "train.csv"), recursive=True):
        root = os.path.dirname(os.path.dirname(train_csv))          # .../<root>/sets/train.csv → <root>
        score = os.path.isdir(os.path.join(root, "wav")) + os.path.exists(os.path.join(root, "metadata.csv"))
        cands.append((score, root))
    cands.sort(reverse=True)                                        # ưu tiên thư mục đủ wav + metadata
    return cands

_cands = find_data_root("/kaggle/input")
if _cands:
    print("🔎 Ứng viên DATA_ROOT (điểm cao = đủ wav+metadata):")
    for sc, r in _cands:
        print(f"   [{sc}/2] {r}")
    DATA_ROOT = _cands[0][1]
    print(f"👉 Tự chọn DATA_ROOT = {DATA_ROOT}")
else:
    DATA_ROOT = "/kaggle/input/datasets/minhtoan2"   # dự phòng — sửa tay nếu auto-dò không thấy
    print(f"❌ Không thấy sets/train.csv trong /kaggle/input → dùng dự phòng {DATA_ROOT} (đã Add Input chưa?)")

WAV_DIR      = f"{DATA_ROOT}/wav"
METADATA_CSV = f"{DATA_ROOT}/metadata.csv"     # wavID|emotion|transcript (cho cột cảm xúc)
TRAIN_CSV    = f"{DATA_ROOT}/sets/train.csv"   # lisID|wavID|qMOS|emoCat|eMOS|val|dom|aro
DEV_SCP      = f"{DATA_ROOT}/sets/dev.scp"

# ── Checkpoint cảm xúc exp08 (để sinh 5 cột EMOS/CAT/VAD) ─────────────────────
# ⭐ TỐT NHẤT = ft_emotion_full_20epoch.pt (bản 20 epoch) — dùng bản này, KHÔNG dùng ft_emotion_full.pt.
EMO_CKPT     = "/kaggle/input/ft-emotion-full/ft_emotion_full_20epoch.pt"   # << ckpt exp08 20ep (CÓ backbone WavLM)
CACHE_INPUT  = "/kaggle/input/ft-emotion-cache"                     # << (tùy chọn) thư mục chứa aud_dev.npz; "" nếu không có

OUT_DIR   = "/kaggle/working"
CACHE_DIR = "/kaggle/working/ft_cache"     # /kaggle/input read-only → copy cache audeering sang đây
os.makedirs(CACHE_DIR, exist_ok=True)

# ── PHẦN A: fine-tune UTMOS (QMOS) ───────────────────────────────────────────
DEVICE          = "cuda"
SR              = 16000
QMOS_MAX_SEC    = 12          # cắt audio chặn bộ nhớ backprop (UTMOS); OOM thì giảm 10/8
LR              = 1e-5        # LR nhỏ cho fine-tune (warm-start sẵn tốt)
WEIGHT_DECAY    = 1e-5
EPOCHS          = 10          # TRẦN; early-stop quyết số epoch thật
PATIENCE        = 3
BATCH           = 1          # UTMOS forward KHÔNG có attention-mask → BATCH=1 an toàn (pad zero sẽ lệch pooling)
ACCUM           = 16         # effective batch = BATCH*ACCUM = 16
VAL_FRAC        = 0.10
SEED            = 42
USE_AMP         = True
RANK_LAMBDA     = 0.0         # 0 = chỉ MSE. >0 (vd 0.3) = cộng pairwise ranking loss (tối ưu thẳng thứ hạng=SRCC)
FREEZE_FEAT_EXT = True        # đóng băng feature-extractor (CNN conv) của UTMOS → đỡ VRAM + chống overfit

# ── PHẦN B: inference cảm xúc (PHẢI khớp kiến trúc exp08) ─────────────────────
EMO_MAX_SEC         = 8
UNFREEZE_TOP_LAYERS = 6       # khớp ckpt exp08
TRUNK_HIDDEN        = 512
HEAD_HIDDEN         = 128
DROPOUT             = 0.3
USE_AUDEERING       = True    # khớp ckpt exp08

LIMIT_TRAIN = 300            # << LẦN ĐẦU 300; chạy thật None
LIMIT_DEV   = 20             # << LẦN ĐẦU 20; chạy thật None

# Mốc QMOS để so (leaderboard DEV)
QMOS_BASELINE = {"utmos_zeroshot": 0.414, "exp07_head": 0.548}

EMOTIONS5 = ["angry", "happy", "neutral", "sad", "surprised"]
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

def stem(p):
    return os.path.splitext(os.path.basename(str(p)))[0]

print("DATA_ROOT:", DATA_ROOT)
for p in [WAV_DIR, METADATA_CSV, TRAIN_CSV, DEV_SCP, EMO_CKPT]:
    print(("  ✅ " if os.path.exists(p) else "  ❌ THIẾU ") + p)
print(f"Fine-tune UTMOS: LR {LR} · BATCH {BATCH}×ACCUM {ACCUM} · MAX {QMOS_MAX_SEC}s · rank λ {RANK_LAMBDA}")

# Copy cache audeering (aud_dev.npz) từ input read-only sang working (để cột cảm xúc khỏi trích lại)
if CACHE_INPUT and os.path.isdir(CACHE_INPUT):
    n = 0
    for fn in os.listdir(CACHE_INPUT):
        if fn.startswith("aud_") and fn.endswith(".npz"):
            shutil.copy(os.path.join(CACHE_INPUT, fn), os.path.join(CACHE_DIR, fn)); n += 1
    print(f"📦 Copy {n} file cache audeering từ {CACHE_INPUT} → {CACHE_DIR}")
else:
    print("ℹ️ Không có CACHE_INPUT → sẽ tự trích audeering cho DEV (chậm hơn lần đầu).")

# %% [markdown]
# ## 1. Cài đặt

# %%
import sys, subprocess

def pip_install(*pkgs):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], check=True)

pip_install("speechmos", "loralib", "speechbrain", "librosa", "soundfile",
            "scipy", "scikit-learn", "pandas", "tqdm")

# Code SAILER (để dựng đúng kiến trúc WavLM rồi nạp ckpt exp08 đè lên) — chỉ cần cho PHẦN B
REPO_DIR = "/kaggle/working/vox-profile-release"
if not os.path.exists(REPO_DIR):
    subprocess.run(["git", "clone", "--depth", "1",
                    "https://github.com/tiantiaf0627/vox-profile-release.git", REPO_DIR], check=True)
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

# %% [markdown]
# ## 2. Nhãn vàng qMOS (gộp trung bình theo wav) — như exp06/exp09a

# %%
import numpy as np
import pandas as pd

def load_qmos_labels():
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
_vals = np.array(list(qmos_gold.values()))
print(f"qMOS gold: mean {_vals.mean():.3f} · std {_vals.std():.3f} · min {_vals.min():.2f} · max {_vals.max():.2f}")

# %% [markdown]
# ## 3. PHẦN A — Fine-tune UTMOS trên qMOS
# UTMOS xuất MOS thang ~1–5 (đã warm-start) → train MSE trên thang GỐC (không z-score, để giữ ý nghĩa warm-start).
# `BATCH=1` + grad-accum: tránh phải pad (UTMOS forward không nhận attention-mask).

# %%
import torch
import torch.nn as nn
import librosa
from tqdm.auto import tqdm
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split

device = DEVICE if torch.cuda.is_available() else "cpu"
print("Device:", device, ("✅ " + torch.cuda.get_device_name(0)) if device == "cuda" else "⚠️ CPU (rất chậm!)")
torch.manual_seed(SEED); np.random.seed(SEED)

# Nạp UTMOS (torch.hub) — model nn.Module, forward(wave[B,T], sr) -> MOS[B]
utmos = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True).to(device)
n_all = sum(p.numel() for p in utmos.parameters())

# (tùy chọn) đóng băng feature-extractor (các lớp conv trích đặc trưng) → đỡ VRAM + chống overfit
if FREEZE_FEAT_EXT:
    n_frozen = 0
    for name, p in utmos.named_parameters():
        if "feature_extractor" in name or "feature_projection" in name or "conv" in name.lower():
            p.requires_grad = False; n_frozen += p.numel()
    print(f"❄️ Đóng băng feature-extractor: {n_frozen/1e6:.1f}M / {n_all/1e6:.1f}M param")
n_train = sum(p.numel() for p in utmos.parameters() if p.requires_grad)
print(f"UTMOS: {n_all/1e6:.1f}M param tổng · {n_train/1e6:.1f}M param sẽ train")

def load_wav_qmos(sid):
    p = os.path.join(WAV_DIR, sid + ".wav")
    if not os.path.exists(p):
        return None
    wave, _ = librosa.load(p, sr=SR, mono=True)
    return wave[: QMOS_MAX_SEC * SR].astype(np.float32)

# Tập train QMOS: chỉ wav tồn tại trên đĩa
train_stems_q = [s for s in qmos_gold if os.path.exists(os.path.join(WAV_DIR, s + ".wav"))]
np.random.shuffle(train_stems_q)
if LIMIT_TRAIN:
    train_stems_q = train_stems_q[:LIMIT_TRAIN]
tr_q, va_q = train_test_split(train_stems_q, test_size=VAL_FRAC, random_state=SEED)
print(f"QMOS train: {len(tr_q)} · val nội bộ: {len(va_q)}")

opt = torch.optim.AdamW([p for p in utmos.parameters() if p.requires_grad],
                        lr=LR, weight_decay=WEIGHT_DECAY)
scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP and device == "cuda")
mse = nn.MSELoss()

def utmos_forward(wave_np):
    """1 wav numpy -> MOS scalar tensor (giữ grad)."""
    x = torch.from_numpy(wave_np).unsqueeze(0).to(device)   # [1, T]
    out = utmos(x, SR)                                       # [1] (hoặc [1,?])
    return out.reshape(-1).mean()                            # scalar an toàn mọi shape

def pairwise_rank_loss(preds, targets):
    """Hinge ranking trên các cặp trong 1 nhóm (khuyến khích đúng thứ hạng = tối ưu SRCC)."""
    p = torch.stack(preds); t = torch.tensor(targets, device=device, dtype=torch.float32)
    if len(p) < 2:
        return torch.zeros((), device=device)
    sign = torch.sign(t.unsqueeze(0) - t.unsqueeze(1))
    diff = p.unsqueeze(0) - p.unsqueeze(1)
    return torch.relu(-sign * diff).mean()

@torch.no_grad()
def eval_qmos_val():
    utmos.eval()
    preds, gts = [], []
    for s in va_q:
        wave = load_wav_qmos(s)
        if wave is None:
            continue
        with torch.cuda.amp.autocast(enabled=USE_AMP and device == "cuda"):
            preds.append(float(utmos_forward(wave).item()))
        gts.append(qmos_gold[s])
    return float(spearmanr(preds, gts).correlation)

# Baseline ZERO-SHOT (trước khi train) trên CÙNG val → mốc phải vượt
srcc_zeroshot = eval_qmos_val()
print(f"\n📍 UTMOS zero-shot (val nội bộ): SRCC = {srcc_zeroshot:.4f}  "
      f"(leaderboard DEV ~{QMOS_BASELINE['utmos_zeroshot']}; exp07 head {QMOS_BASELINE['exp07_head']})")

CKPT_QMOS = os.path.join(OUT_DIR, "ft_qmos_utmos.pt")
def save_qmos_ckpt(srcc):
    torch.save({"utmos_state": {k: v.cpu() for k, v in utmos.state_dict().items()},
                "val_srcc": float(srcc), "raw_scale": True,
                "QMOS_MAX_SEC": QMOS_MAX_SEC, "FREEZE_FEAT_EXT": FREEZE_FEAT_EXT}, CKPT_QMOS)

best, best_state, bad = srcc_zeroshot, {k: v.cpu().clone() for k, v in utmos.state_dict().items()}, 0
save_qmos_ckpt(best)   # lưu sẵn bản zero-shot (worst case vẫn = baseline)

# Gom theo CỬA SỔ = ACCUM mẫu HỢP LỆ (micro). Hai chế độ backward:
#   • RANK off (mặc định) → backward NGAY từng mẫu → đồ thị giải phóng liền → VRAM thấp.
#   • RANK on  → ranking cần SO các pred TRONG cửa sổ → PHẢI giữ đồ thị cả cửa sổ →
#                gom MSE (win_loss) + pred (buf_p) rồi backward MỘT lần (MSE_mean + λ·rank).
#   ⚠️ Lỗi cũ: backward MSE từng bước đã giải phóng đồ thị → rank_loss.backward() sau đó
#      sẽ lỗi "backward through the graph a second time". Bản này gom rồi backward 1 lần → hết lỗi.
for ep in range(1, EPOCHS + 1):
    utmos.train()
    opt.zero_grad()
    np.random.shuffle(tr_q)
    run = 0.0; nb = 0
    micro = 0; win_loss = None; buf_p, buf_t = [], []
    for s in tqdm(tr_q, desc=f"epoch {ep}"):
        wave = load_wav_qmos(s)
        if wave is None:
            continue
        with torch.cuda.amp.autocast(enabled=USE_AMP and device == "cuda"):
            pred = utmos_forward(wave)
            loss = mse(pred, torch.tensor(qmos_gold[s], device=device, dtype=pred.dtype))
        run += float(loss.item()); nb += 1
        if RANK_LAMBDA > 0:
            win_loss = loss if win_loss is None else win_loss + loss   # GIỮ đồ thị (không backward ngay)
            buf_p.append(pred); buf_t.append(qmos_gold[s]); micro += 1
        else:
            scaler.scale(loss / ACCUM).backward(); micro += 1            # backward ngay → VRAM thấp
        if micro == ACCUM:
            if RANK_LAMBDA > 0:
                total = win_loss / micro
                if len(buf_p) >= 2:
                    total = total + RANK_LAMBDA * pairwise_rank_loss(buf_p, buf_t)
                scaler.scale(total).backward()
            scaler.step(opt); scaler.update(); opt.zero_grad()
            micro = 0; win_loss = None; buf_p, buf_t = [], []
    # flush cửa sổ dư cuối epoch (số mẫu không chia hết cho ACCUM)
    if micro > 0:
        if RANK_LAMBDA > 0:
            total = win_loss / micro
            if len(buf_p) >= 2:
                total = total + RANK_LAMBDA * pairwise_rank_loss(buf_p, buf_t)
            scaler.scale(total).backward()
        scaler.step(opt); scaler.update(); opt.zero_grad()
    sc = eval_qmos_val()
    print(f"epoch {ep:2d} | loss {run/max(nb,1):.4f} | val SRCC {sc:.4f} "
          f"(zero-shot {srcc_zeroshot:.4f} · best {max(best,sc):.4f})")
    if sc > best:
        best = sc
        best_state = {k: v.cpu().clone() for k, v in utmos.state_dict().items()}
        save_qmos_ckpt(best)
        print(f"   💾 lưu best → {CKPT_QMOS} (epoch {ep}, SRCC {sc:.4f})")
        bad = 0
    else:
        bad += 1
        if bad >= PATIENCE:
            print(f"Early stop ở epoch {ep}."); break

utmos.load_state_dict(best_state)
print(f"\n✅ PHẦN A xong — QMOS val nội bộ: zero-shot {srcc_zeroshot:.4f} → fine-tune {best:.4f} "
      + ("🚀 cải thiện" if best > srcc_zeroshot + 1e-4 else "➖ KHÔNG vượt zero-shot"))
if best <= srcc_zeroshot + 1e-4:
    print("   ⚠️ Fine-tune chưa vượt zero-shot → cân nhắc tăng EPOCHS / bật RANK_LAMBDA=0.3 / "
          "mở băng feature-extractor (FREEZE_FEAT_EXT=False); hoặc giữ QMOS exp07 (0.548).")

# %% [markdown]
# ## 4. PHẦN A (tiếp) — Dự đoán QMOS cho DEV bằng UTMOS đã fine-tune

# %%
def list_dev():
    with open(DEV_SCP) as f:
        return [ln.strip() for ln in f if ln.strip()]

dev_names = list_dev()
if LIMIT_DEV:
    dev_names = dev_names[:LIMIT_DEV]
print("DEV:", len(dev_names), "mẫu")

@torch.no_grad()
def predict_qmos(name):
    p = os.path.join(WAV_DIR, name if str(name).endswith(".wav") else str(name) + ".wav")
    if not os.path.exists(p):
        return None
    wave, _ = librosa.load(p, sr=SR, mono=True)
    wave = wave[: QMOS_MAX_SEC * SR].astype(np.float32)
    utmos.eval()
    with torch.cuda.amp.autocast(enabled=USE_AMP and device == "cuda"):
        v = float(utmos_forward(wave).item())
    return float(np.clip(v, 1.0, 5.0))

qmos_pred = {}
n_real = n_def = 0
for name in tqdm(dev_names, desc="QMOS dev"):
    v = predict_qmos(name)
    if v is None:
        v = 3.0; n_def += 1
    else:
        n_real += 1
    qmos_pred[name] = v
print(f"QMOS dự đoán: thật {n_real}, mặc định {n_def}")

# Giải phóng UTMOS trước khi nạp backbone cảm xúc (đỡ VRAM T4)
del utmos, opt, scaler
torch.cuda.empty_cache() if device == "cuda" else None

# %% [markdown]
# ## 5. PHẦN B — Nạp ckpt exp08 (WavLM ft + audeering) → 5 cột cảm xúc cho DEV
# Tái dùng nguyên cơ chế load của exp08b: dựng kiến trúc → `load_state_dict` từ `ft_emotion_full_20epoch.pt`.

# %%
import torch.nn.functional as F

ckpt = torch.load(EMO_CKPT, map_location="cpu", weights_only=False)   # ckpt có numpy → weights_only=False
assert "wavlm" in ckpt, ("❌ EMO_CKPT không có 'wavlm' (backbone). Cần ft_emotion_full_20epoch.pt (bản đủ backbone), "
                         "KHÔNG phải ft_emotion_meta.pt cũ.")
print("✅ Nạp ckpt cảm xúc:", EMO_CKPT, "| keys:", list(ckpt.keys()))

def find_hf_backbone(module):
    cands = []
    for nm, m in module.named_modules():
        enc = getattr(m, "encoder", None)
        if getattr(m, "feature_extractor", None) is not None and enc is not None \
                and getattr(enc, "layers", None) is not None:
            cands.append((nm, m))
    if not cands:
        return None, None
    cands.sort(key=lambda x: sum(p.numel() for p in x[1].parameters()), reverse=True)
    return cands[0]

wavlm = None
try:
    from src.model.emotion.wavlm_emotion import WavLMWrapper   # noqa: E402
    _wrapper = WavLMWrapper.from_pretrained("tiantiaf/wavlm-large-categorical-emotion")
    _name, wavlm = find_hf_backbone(_wrapper)
    if wavlm is not None:
        print(f"✅ Dựng backbone WavLM từ SAILER wrapper tại '.{_name}'")
except Exception as e:
    print("⚠️ Lỗi nạp SAILER wrapper:", repr(e), "→ fallback WavLM trắng.")
if wavlm is None:
    from transformers import WavLMModel
    wavlm = WavLMModel.from_pretrained("microsoft/wavlm-large")
    print("ℹ️ Fallback: microsoft/wavlm-large.")

wavlm = wavlm.to(device).eval()
WAVLM_DIM = int(wavlm.config.hidden_size)
miss, unexp = wavlm.load_state_dict(ckpt["wavlm"], strict=False)
print(f"🔁 load wavlm từ ckpt: thiếu {len(miss)} / dư {len(unexp)} key (kỳ vọng ~0).")

def masked_mean(hidden, attn_mask):
    if attn_mask is None:
        return hidden.mean(dim=1)
    try:
        fm = wavlm._get_feature_vector_attention_mask(hidden.shape[1], attn_mask)
    except Exception:
        return hidden.mean(dim=1)
    fm = fm.unsqueeze(-1).to(hidden.dtype)
    return (hidden * fm).sum(1) / fm.sum(1).clamp(min=1e-6)

@torch.no_grad()
def wavlm_embed(input_values, attn_mask):
    out = wavlm(input_values, attention_mask=attn_mask).last_hidden_state
    return masked_mean(out, attn_mask)

# ── audeering FROZEN (đặc trưng phụ) — như exp08 ──
AUD_DIM = 0
aud_backbone = aud_head = aud_proc = None
if USE_AUDEERING:
    from transformers import Wav2Vec2Model, Wav2Vec2Config, Wav2Vec2Processor
    from huggingface_hub import hf_hub_download
    AUD_NAME = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
    aud_proc = Wav2Vec2Processor.from_pretrained(AUD_NAME)
    aud_cfg = Wav2Vec2Config.from_pretrained(AUD_NAME)
    aud_backbone = Wav2Vec2Model(aud_cfg)
    try:
        _sd = __import__("safetensors.torch", fromlist=["load_file"]).load_file(
            hf_hub_download(AUD_NAME, "model.safetensors"))
    except Exception:
        _sd = torch.load(hf_hub_download(AUD_NAME, "pytorch_model.bin"), map_location="cpu")
    bb_sd = {k[len("wav2vec2."):]: v for k, v in _sd.items() if k.startswith("wav2vec2.")}
    aud_backbone.load_state_dict(bb_sd, strict=False)
    _hid = _sd["classifier.dense.weight"].shape[0]
    _out = _sd["classifier.out_proj.weight"].shape[0]
    aud_head = nn.Sequential(nn.Linear(_hid, _hid), nn.Tanh(), nn.Linear(_hid, _out))
    aud_head[0].weight.data.copy_(_sd["classifier.dense.weight"]); aud_head[0].bias.data.copy_(_sd["classifier.dense.bias"])
    aud_head[2].weight.data.copy_(_sd["classifier.out_proj.weight"]); aud_head[2].bias.data.copy_(_sd["classifier.out_proj.bias"])
    aud_backbone = aud_backbone.to(device).eval()
    aud_head = aud_head.to(device).eval()
    AUD_DIM = _hid + 3
    print(f"✅ audeering frozen ({AUD_DIM}-D)")

def load_wav_emo(sid):
    p = os.path.join(WAV_DIR, sid + ".wav")
    if not os.path.exists(p):
        return None
    wave, _ = librosa.load(p, sr=SR, mono=True)
    return wave[: EMO_MAX_SEC * SR].astype(np.float32)

@torch.no_grad()
def extract_audeering(stems, tag):
    if not USE_AUDEERING:
        return {}
    cache_path = os.path.join(CACHE_DIR, f"aud_{tag}.npz")
    store = {}
    if os.path.exists(cache_path):
        z = np.load(cache_path, allow_pickle=True)
        store = {k: z[k] for k in z.files}
        print(f"[aud/{tag}] nạp cache: {len(store)}")
    todo = [s for s in stems if s not in store]
    for i, s in enumerate(tqdm(todo, desc=f"audeering {tag}")):
        wave = load_wav_emo(s)
        if wave is None:
            continue
        x = aud_proc(wave, sampling_rate=SR).input_values[0]
        x = torch.from_numpy(np.asarray(x, dtype=np.float32)).unsqueeze(0).to(device)
        h = aud_backbone(x)[0].mean(dim=1)
        out = aud_head(h)[0].cpu().numpy()
        vad = np.array([1 + 4 * out[2], 1 + 4 * out[0], 1 + 4 * out[1]], dtype=np.float32)  # [VAL,ARO,DOM]
        store[s] = np.concatenate([h[0].cpu().numpy(), vad]).astype(np.float32)
        if (i + 1) % 500 == 0:
            np.savez(cache_path, **store)
    if todo:
        np.savez(cache_path, **store)
    return store

# ── EmoHeads (khớp exp08) + nạp trọng số head + thống kê chuẩn hóa từ ckpt ──
N_EMO = len(EMOTIONS5)
TRUNK_IN = WAVLM_DIM + (AUD_DIM if USE_AUDEERING else 0)

class EmoHeads(nn.Module):
    def __init__(self, d_in, trunk_h, head_h, p, n_emo):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(d_in, trunk_h), nn.ReLU(), nn.Dropout(p),
                                   nn.Linear(trunk_h, trunk_h), nn.ReLU(), nn.Dropout(p))
        self.emos = nn.Sequential(nn.Linear(trunk_h + n_emo, head_h), nn.ReLU(), nn.Dropout(p), nn.Linear(head_h, 1))
        self.cat = nn.Sequential(nn.Linear(trunk_h, head_h), nn.ReLU(), nn.Dropout(p), nn.Linear(head_h, n_emo))
        self.vad = nn.Sequential(nn.Linear(trunk_h, head_h), nn.ReLU(), nn.Dropout(p), nn.Linear(head_h, 3))
    def forward(self, feat, tgt):
        h = self.trunk(feat)
        return self.emos(torch.cat([h, tgt], 1)), self.cat(h), self.vad(h)

heads = EmoHeads(TRUNK_IN, TRUNK_HIDDEN, HEAD_HIDDEN, DROPOUT, N_EMO).to(device).eval()
hmiss, hunexp = heads.load_state_dict(ckpt["heads"], strict=False)
print(f"🔁 load heads từ ckpt: thiếu {len(hmiss)} / dư {len(hunexp)} key (kỳ vọng 0).")

emos_mu = float(ckpt["emos_mu"]); emos_sd = float(ckpt["emos_sd"])
vad_mu = np.asarray(ckpt["vad_mu"], dtype=np.float32); vad_sd = np.asarray(ckpt["vad_sd"], dtype=np.float32)
print(f"Chuẩn hóa từ ckpt: emos μ={emos_mu:.3f} σ={emos_sd:.3f} | vad μ={np.round(vad_mu,2)}")

# Target cảm xúc (cho EMOS head) từ metadata
def load_target_emotions():
    tgt = {}
    with open(METADATA_CSV, encoding="utf-8") as f:
        for ln in f:
            parts = ln.strip().split("|")
            if len(parts) >= 2:
                tgt[stem(parts[0])] = norm_emotion(parts[1])
    return tgt

target_map = load_target_emotions()

def onehot_target(tgt):
    v = np.zeros(N_EMO, dtype=np.float32)
    if tgt in EMOTIONS5:
        v[EMOTIONS5.index(tgt)] = 1.0
    return v

dev_stems = [stem(n) for n in dev_names]
aud_dev = extract_audeering(dev_stems, "dev")

@torch.no_grad()
def predict_emotion(sid):
    wave = load_wav_emo(sid)
    if wave is None or (USE_AUDEERING and sid not in aud_dev):
        return None
    iv = torch.from_numpy(wave).unsqueeze(0).to(device)
    am = torch.ones((1, len(wave)), dtype=torch.long, device=device)
    tgt = torch.from_numpy(onehot_target(target_map.get(sid))).unsqueeze(0).to(device)
    with torch.cuda.amp.autocast(enabled=USE_AMP and device == "cuda"):
        fw = wavlm_embed(iv, am)
        feat = torch.cat([fw, torch.from_numpy(aud_dev[sid]).unsqueeze(0).to(device)], dim=1) if USE_AUDEERING else fw
        emos_p, cat_l, vad_p = heads(feat, tgt)
    emos = float(emos_p.item()) * emos_sd + emos_mu
    cat5 = F.softmax(cat_l, 1)[0].float().cpu().numpy()
    vad3 = vad_p[0].float().cpu().numpy() * vad_sd + vad_mu
    return emos, cat5, vad3

# %% [markdown]
# ## 6. PHẦN C — Ghép QMOS (fine-tune) + 5 cột cảm xúc (exp08) → answer.txt 6 cột

# %%
def fmt_cat(p5):
    return "|".join(f"{e}:{p5[i]:.6g}" for i, e in enumerate(EMOTIONS5))

def build_answer(out_path):
    n_real = n_def = 0
    with open(out_path, "w") as f:
        f.write("wav,QMOS,EMOS,CAT,VAL,ARO,DOM\n")
        for name in tqdm(dev_names, desc="answer"):
            sid = stem(name)
            pr = predict_emotion(sid)
            if pr is None:
                emos, cat5, vad3 = 3.0, np.full(5, 0.2, np.float32), np.array([3.0, 3.0, 3.0]); n_def += 1
            else:
                emos, cat5, vad3 = pr; n_real += 1
            qmos = qmos_pred.get(name, qmos_pred.get(sid, 3.0))
            f.write(f"{name},{qmos:.6g},{emos:.6g},{fmt_cat(cat5)},{vad3[0]:.6g},{vad3[1]:.6g},{vad3[2]:.6g}\n")
    print(f"Ghi {len(dev_names)} dòng → {out_path} | cảm xúc thật {n_real}, mặc định {n_def}")

answer_path = os.path.join(OUT_DIR, "answer.txt")
build_answer(answer_path)

# %% [markdown]
# ## 7. Validate + zip

# %%
def validate(path):
    import csv
    with open(path) as f:
        rows = list(csv.reader(f))
    assert rows[0][0] == "wav" and "QMOS" in rows[0] and "EMOS" in rows[0], "Header sai"
    for i, r in enumerate(rows[1:], 2):
        assert len(r) == len(rows[0]), f"Dòng {i} sai số cột"
    print(f"OK: {len(rows)-1} dòng, header = {rows[0]}")

validate(answer_path)
os.system(f"cd {OUT_DIR} && zip -j submission_track2_exp13_ft-qmos.zip answer.txt "
          f"&& unzip -l submission_track2_exp13_ft-qmos.zip")
print("Sẵn sàng nộp:", os.path.join(OUT_DIR, "submission_track2_exp13_ft-qmos.zip"))

# %% [markdown]
# ## Ghi chú
# - **Lần đầu** `LIMIT_TRAIN=300`, `LIMIT_DEV=20` để chạy trơn (không OOM, 1 epoch xong); rồi đặt `None`.
# - **OOM trên T4?** giảm `QMOS_MAX_SEC` (12→10→8); giữ `FREEZE_FEAT_EXT=True`; `BATCH=1` đã là min.
#   ⚠️ **Bật `RANK_LAMBDA>0` tốn VRAM hơn** vì phải GIỮ đồ thị cả cửa sổ ACCUM (=16) để so thứ hạng →
#   nếu OOM khi bật ranking: giảm `ACCUM` (vd 8, cũng là kích thước nhóm ranking) hoặc `QMOS_MAX_SEC`.
# - **Đọc mục A:** so `val SRCC fine-tune` với `zero-shot`. Chỉ nộp QMOS fine-tune nếu **vượt zero-shot**
#   (lý tưởng vượt cả exp07 0.548); nếu không → giữ QMOS exp07 (Add Input answer.txt exp07, đổi cột QMOS).
# - Nếu chưa vượt: tăng `EPOCHS`, bật `RANK_LAMBDA=0.3` (tối ưu thẳng thứ hạng), hoặc `FREEZE_FEAT_EXT=False`
#   (mở băng feature-extractor — mạnh hơn nhưng dễ overfit + nặng VRAM).
# - **Lưu checkpoint:** `ft_qmos_utmos.pt` lưu mỗi best → **Save Version NGAY** sau khi chạy (bài học exp08).
# - **License QMOS:** UTMOS/SpeechMOS (kiểm tra license tarepan/SpeechMOS) — khai báo `docs/12_system_description.md`.
# - Ghi config → kết quả → nhận xét vào `docs/04_experiments_log.md` (mục exp13).
