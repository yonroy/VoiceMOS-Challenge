# Track 1 — URGENT-MOS (ACR + CCR). Lazy-load: nạp model ở request đầu tiên.
import os, sys, subprocess
import torch

SR = 16000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODELS_DIR = os.environ.get("MODELS_DIR", "/tmp/models")
REPO = os.path.join(MODELS_DIR, "URGENT-MOS")
CKPT = os.environ.get("T1_CKPT", "urgent-challenge/urgent-mos-f1c1m5dcorpus")  # tự tải từ HuggingFace

_M = {}


def _load():
    if "m" in _M:
        return _M["m"]
    os.makedirs(MODELS_DIR, exist_ok=True)
    if not os.path.isdir(REPO):
        subprocess.run(f"git clone -q https://github.com/vvwangvv/URGENT-MOS.git {REPO}",
                       shell=True, check=True)
    if REPO not in sys.path:
        sys.path.insert(0, REPO)
    import importlib
    importlib.invalidate_caches()
    try:
        importlib.import_module("urgent_mos.api.infer")
    except Exception:
        subprocess.run(f"{sys.executable} -m pip install -q -e {REPO}", shell=True, check=False)
        importlib.invalidate_caches()
    from urgent_mos.utils import load_model_from_checkpoint
    m = load_model_from_checkpoint(CKPT, DEVICE)
    m.eval()
    _M["m"] = m
    return m


def predict(wave_a, wave_b=None):
    """ACR cho A (1..5); nếu có B → ACR(B) + CCR (A so với B, -3..+3, >0 = A tốt hơn)."""
    m = _load()
    from urgent_mos.api.infer import infer, infer_pairs
    wa = torch.from_numpy(wave_a).float()
    acr_a = max(1.0, min(5.0, float(
        infer(m, [wa], sample_rate=[SR], batch_frames=None, num_workers=0)[0]["mos_overall"])))
    out = {"acr_a": round(acr_a, 4)}
    if wave_b is not None:
        wb = torch.from_numpy(wave_b).float()
        acr_b = max(1.0, min(5.0, float(
            infer(m, [wb], sample_rate=[SR], batch_frames=None, num_workers=0)[0]["mos_overall"])))
        ccr = max(-3.0, min(3.0, float(
            infer_pairs(m, [(wa, wb)], sample_rate=[(SR, SR)], batch_frames=None,
                        num_workers=0)[0]["mos_overall"])))
        out["acr_b"] = round(acr_b, 4)
        out["ccr"] = round(ccr, 4)
    return out
