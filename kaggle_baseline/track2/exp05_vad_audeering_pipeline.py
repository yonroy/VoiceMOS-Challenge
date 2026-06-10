# %% [markdown]
# # VMC2026 Track 2 — exp05 (VAD bằng audeering MSP-dim) — Kaggle
#
# **Mục tiêu:** đẩy **VAL** (SAILER chỉ 0.341 — thấp nhất) bằng model VAD chuyên
# `audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim` (dimensional, xuất thẳng
# arousal/dominance/valence ∈ [0,1]). **Thay cả 3 cột VAD** bằng audeering.
#
# ## Phân công model (giữ cái tốt của exp03, chỉ đổi VAD)
# ```
# QMOS  ← SpeechMOS (UTMOS)         (để riêng)
# EMOS  ← SAILER  (1 + 4·P(target))  ┐ giữ nguyên exp03
# CAT   ← SAILER  (5 lớp renorm)     ┘
# VAL   ← audeering ┐
# ARO   ← audeering ├─ THAY cả 3 (model VAD chuyên)
# DOM   ← audeering ┘
# ```
# - Mỗi wav chạy **2 forward**: SAILER (EMOS+CAT) + audeering (VAD). KHÔNG train.
# - So với exp03 (VAD từ SAILER: VAL 0.341 / ARO 0.712 / DOM 0.630) → nộp để A/B từng cột.
#
# **Cách chạy Kaggle:** GPU **T4** + Internet **On** → + Add Input dataset Track 2 (có `sets/dev.scp`,
# `metadata.csv`) → sửa `DATA_ROOT` → lần đầu `LIMIT = 20` kiểm tra VAD ra 1–5 hợp lý → rồi `None`.
#
# ⚠️ License **SAILER = Open RAIL** · **audeering = CC BY-NC-SA 4.0** (đều phi thương mại) → khai báo `docs/12_`.

# %% [markdown]
# ## 0. Cấu hình — SỬA Ở ĐÂY

# %%
import os

DATA_ROOT    = "/kaggle/input/vmc2026-track2-full/vmc2026-track2"   # << SỬA slug cho khớp Add Input
WAV_DIR      = f"{DATA_ROOT}/wav"
METADATA_CSV = f"{DATA_ROOT}/metadata.csv"     # wavID|emotion|transcript → target emotion (cho EMOS)
DEV_SCP      = f"{DATA_ROOT}/sets/dev.scp"     # danh sách wav tập DEV

OUT_DIR = "/kaggle/working"

DEVICE      = "cuda"
MAX_SECONDS = 15
SR          = 16000
LIMIT       = None          # đặt 20 để chạy thử nhanh; None = full DEV

EMOTIONS5 = ["angry", "happy", "neutral", "sad", "surprised"]
SAILER9 = ["Anger", "Contempt", "Disgust", "Fear", "Happiness", "Neutral", "Sadness", "Surprise", "Other"]
EMO2SAILER = {"angry": 0, "happy": 4, "neutral": 5, "sad": 6, "surprised": 7}

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

def stem(path_or_name):
    return os.path.splitext(os.path.basename(str(path_or_name)))[0]

print("DATA_ROOT:", DATA_ROOT)
for p in [WAV_DIR, METADATA_CSV, DEV_SCP]:
    print(("  ✅ " if os.path.exists(p) else "  ❌ THIẾU ") + p)

# %% [markdown]
# ## 1. Cài đặt + tải code SAILER (clone + sys.path, KHÔNG pip install -e .)

# %%
import sys, subprocess

def pip_install(*pkgs):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], check=True)

REPO_DIR = "/kaggle/working/vox-profile-release"
if not os.path.exists(REPO_DIR):
    subprocess.run(["git", "clone", "--depth", "1",
                    "https://github.com/tiantiaf0627/vox-profile-release.git", REPO_DIR], check=True)

pip_install("loralib", "speechbrain", "speechmos", "librosa", "soundfile", "scipy", "tqdm")

if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

# %% [markdown]
# ## 2. Nạp model SAILER (cho EMOS + CAT)

# %%
import torch
import torch.nn.functional as F

device = DEVICE if torch.cuda.is_available() else "cpu"
print("Device:", device)
if device == "cuda":
    print("  ✅ GPU:", torch.cuda.get_device_name(0))
else:
    print("  ⚠️ KHÔNG thấy GPU → Settings → Accelerator = GPU T4 rồi chạy lại.")

from src.model.emotion.wavlm_emotion import WavLMWrapper   # noqa: E402

sailer = WavLMWrapper.from_pretrained("tiantiaf/wavlm-large-categorical-emotion").to(device)
sailer.eval()
print("✅ Đã nạp SAILER (wavlm-large-categorical-emotion)")

# %% [markdown]
# ## 2b. Nạp model VAD chuyên: audeering wav2vec2 MSP-dim
# ⚠️ Kế thừa `Wav2Vec2PreTrainedModel` (theo model card) hay dính lỗi version transformers
# (thiếu `__file__` / `all_tied_weights_keys`...). Cách dứt điểm: CHỈ dùng `Wav2Vec2Model` (backbone
# được hỗ trợ tốt) + **tự nạp tay** trọng số regression head từ checkpoint → không đụng tie-weights/experts.
# ⚠️ Model xuất thứ tự **[arousal, dominance, valence]** ∈ [0,1] → đổi về [VAL,ARO,DOM] thang 1–5 khi ghi.

# %%
import torch.nn as nn
from transformers import Wav2Vec2Model, Wav2Vec2Config, Wav2Vec2Processor
from huggingface_hub import hf_hub_download

AUD_NAME = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
aud_proc = Wav2Vec2Processor.from_pretrained(AUD_NAME)

# 1) backbone wav2vec2 (load chuẩn, không subclass)
aud_cfg = Wav2Vec2Config.from_pretrained(AUD_NAME)
aud_backbone = Wav2Vec2Model(aud_cfg)

# 2) tải state_dict gốc của checkpoint (ưu tiên safetensors)
try:
    _sd = __import__("safetensors.torch", fromlist=["load_file"]).load_file(
        hf_hub_download(AUD_NAME, "model.safetensors"))
except Exception:
    _sd = torch.load(hf_hub_download(AUD_NAME, "pytorch_model.bin"), map_location="cpu")

# 3) nạp phần backbone (key có tiền tố "wav2vec2.") vào Wav2Vec2Model
bb_sd = {k[len("wav2vec2."):]: v for k, v in _sd.items() if k.startswith("wav2vec2.")}
missing, unexpected = aud_backbone.load_state_dict(bb_sd, strict=False)
print(f"  backbone: thiếu {len(missing)} key, dư {len(unexpected)} key (strict=False)")

# 4) dựng regression head theo đúng shape trong checkpoint rồi nạp trọng số "classifier.*"
_hid = _sd["classifier.dense.weight"].shape[0]
_out = _sd["classifier.out_proj.weight"].shape[0]    # = 3 (arousal, dominance, valence)
aud_head = nn.Sequential(nn.Linear(_hid, _hid), nn.Tanh(), nn.Linear(_hid, _out))
aud_head[0].weight.data.copy_(_sd["classifier.dense.weight"])
aud_head[0].bias.data.copy_(_sd["classifier.dense.bias"])
aud_head[2].weight.data.copy_(_sd["classifier.out_proj.weight"])
aud_head[2].bias.data.copy_(_sd["classifier.out_proj.bias"])

aud_backbone = aud_backbone.to(device).eval()
aud_head = aud_head.to(device).eval()
print(f"✅ Đã nạp audeering MSP-dim (backbone + head {_hid}→{_out}) — model VAD chuyên")

# %% [markdown]
# ## 3. Đọc cảm xúc target cho mỗi wav (cho EMOS của SAILER)

# %%
import numpy as np
import librosa

def load_target_emotions():
    tgt = {}
    with open(METADATA_CSV, encoding="utf-8") as f:
        for ln in f:
            parts = ln.strip().split("|")
            if len(parts) < 2:
                continue
            tgt[stem(parts[0])] = norm_emotion(parts[1])
    return tgt

target_map = load_target_emotions()
print(f"Target emotions: {len(target_map)} wav | ví dụ:", dict(list(target_map.items())[:3]))

# %% [markdown]
# ## 4. Hàm chấm: SAILER (EMOS+CAT) + audeering (VAD)

# %%
@torch.no_grad()
def sailer_probs(wav_path):
    """→ probs9 (float32[9]); None nếu thiếu/lỗi. Chỉ lấy 9 lớp (EMOS+CAT), bỏ VAD của SAILER."""
    if not os.path.exists(wav_path):
        return None
    wave, _ = librosa.load(wav_path, sr=SR, mono=True)
    wave = wave[: MAX_SECONDS * SR]
    data = torch.from_numpy(wave).float().unsqueeze(0).to(device)
    logits, _feat, _det, _aro, _val, _dom = sailer(data, return_feature=True)
    return F.softmax(logits, dim=1)[0].detach().cpu().numpy().astype(np.float32)

def emos_from_probs(probs9, target):
    if target is None or target not in EMO2SAILER:
        return None
    return 1.0 + 4.0 * float(probs9[EMO2SAILER[target]])

def cat5_from_probs(probs9):
    v = np.array([probs9[EMO2SAILER[e]] for e in EMOTIONS5], dtype=np.float32)
    s = v.sum()
    return v / s if s > 0 else np.full(5, 0.2, dtype=np.float32)

@torch.no_grad()
def audeering_vad(wav_path):
    """VAD bằng audeering → [VAL, ARO, DOM] thang 1–5; None nếu thiếu/lỗi.
    Model xuất [arousal, dominance, valence] ∈ [0,1]."""
    if not os.path.exists(wav_path):
        return None
    wave, _ = librosa.load(wav_path, sr=SR, mono=True)
    wave = wave[: MAX_SECONDS * SR]
    x = aud_proc(wave, sampling_rate=SR).input_values[0]
    x = torch.from_numpy(np.asarray(x, dtype=np.float32)).unsqueeze(0).to(device)
    h = aud_backbone(x)[0].mean(dim=1)                       # mean-pool theo thời gian
    out = aud_head(h)[0].detach().cpu().numpy()              # [arousal, dominance, valence]
    aro, dom, val = float(out[0]), float(out[1]), float(out[2])
    return np.array([1 + 4 * val, 1 + 4 * aro, 1 + 4 * dom], dtype=np.float32)   # [VAL,ARO,DOM]

# %% [markdown]
# ## 5. QMOS = SpeechMOS (UTMOS) — bắt buộc cho answer.txt

# %%
@torch.no_grad()
def run_qmos(names):
    predictor = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True).to(device).eval()
    from tqdm.auto import tqdm
    out = {}
    for n in tqdm(names, desc="QMOS"):
        p = os.path.join(WAV_DIR, n)
        if not os.path.exists(p):
            continue
        wave, _ = librosa.load(p, sr=SR, mono=True)
        x = torch.from_numpy(wave).unsqueeze(0).to(device)
        out[n] = float(predictor(x, sr=SR).mean().item())
    return out

# %% [markdown]
# ## 6. Chạy trên DEV → `answer.txt` (QMOS, EMOS, CAT ← SAILER/UTMOS · VAL,ARO,DOM ← audeering)

# %%
def list_dev():
    with open(DEV_SCP) as f:
        return [ln.strip() for ln in f if ln.strip()]

dev_names = list_dev()
if LIMIT:
    dev_names = dev_names[:LIMIT]
print("DEV:", len(dev_names), "mẫu")

qmos_scores = run_qmos(dev_names)

def fmt_cat(probs5):
    return "|".join(f"{e}:{probs5[i]:.6g}" for i, e in enumerate(EMOTIONS5))

def build_answer(out_path):
    from tqdm.auto import tqdm
    n_emos = n_default = n_vad_def = 0
    with open(out_path, "w") as f:
        f.write("wav,QMOS,EMOS,CAT,VAL,ARO,DOM\n")
        for name in tqdm(dev_names, desc="EMOS/CAT(SAILER)+VAD(audeering)"):
            sid = stem(name)
            wav = os.path.join(WAV_DIR, name)
            # EMOS + CAT từ SAILER
            probs9 = sailer_probs(wav)
            if probs9 is None:
                emos, cat5 = 3.0, np.full(5, 0.2, dtype=np.float32); n_default += 1
            else:
                emos = emos_from_probs(probs9, target_map.get(sid))
                if emos is None:
                    emos = 3.0; n_default += 1
                else:
                    n_emos += 1
                cat5 = cat5_from_probs(probs9)
            # VAD từ audeering
            vad3 = audeering_vad(wav)
            if vad3 is None:
                vad3 = np.array([3.0, 3.0, 3.0], dtype=np.float32); n_vad_def += 1
            qmos = qmos_scores.get(name, 3.0)
            f.write(f"{name},{qmos:.6g},{emos:.6g},{fmt_cat(cat5)},"
                    f"{vad3[0]:.6g},{vad3[1]:.6g},{vad3[2]:.6g}\n")
    print(f"Ghi {len(dev_names)} dòng → {out_path} | EMOS thật {n_emos}, mặc định {n_default} | VAD mặc định {n_vad_def}")

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
os.system(f"cd {OUT_DIR} && zip -j submission_track2_exp05_vad-audeering.zip answer.txt && unzip -l submission_track2_exp05_vad-audeering.zip")
print("Sẵn sàng nộp:", os.path.join(OUT_DIR, "submission_track2_exp05_vad-audeering.zip"))

# %% [markdown]
# ## Ghi chú
# - **Quan hệ với exp03:** exp03 = SAILER lo cả EMOS+CAT+VAD (giữ nguyên, file `exp03_emos_sailer`).
#   exp05 (file này) chỉ **đổi VAD sang audeering**, EMOS/CAT vẫn SAILER → nộp 2 bản để A/B từng cột VAD.
# - **Lần đầu** đặt `LIMIT = 20`, kiểm tra VAL/ARO/DOM ∈ [1,5] hợp lý (không toàn 3 / không âm).
#   Nếu giá trị lệch → có thể sai thứ tự arousal/dominance/valence, báo lại để chỉnh.
# - Khi chạy để ý dòng `backbone: thiếu N key, dư M key`: thiếu/dư vài key phụ là bình thường;
#   thiếu hàng trăm key = sai tiền tố → báo lại.
# - Nếu audeering thắng VAL nhưng thua ARO/DOM so SAILER → bản tối ưu = trộn cột
#   (VAL từ audeering, ARO/DOM từ exp03). Ghi kết quả vào `docs/04_experiments_log.md` (exp05).
