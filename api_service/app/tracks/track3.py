# Track 3 — ECAPA-TDNN fine-tuned (spk_sim + acc_sim). Cần 2 file: test + reference.
import os, sys, subprocess
import torch

SR = 16000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODELS_DIR = os.environ.get("MODELS_DIR", "/tmp/models")
ROOT = os.path.join(MODELS_DIR, "vmc2026-baselines")
T3 = os.path.join(ROOT, "track3")
CKPT_SPK = f"{T3}/official-egs/spk_sim_adamw_lr1e-3/model_spk_sim_step20000.pt"
CKPT_ACC = f"{T3}/official-egs/acc_sim_adamw_lr1e-3/model_acc_sim_step20000.pt"

_M = {}


def _load():
    if "spk" in _M:
        return _M
    os.makedirs(MODELS_DIR, exist_ok=True)
    if not os.path.isdir(ROOT):
        subprocess.run(f"git clone -q https://github.com/voicemos-challenge/vmc2026-baselines.git {ROOT}",
                       shell=True, check=True)
    if T3 not in sys.path:
        sys.path.insert(0, T3)
    from model import Model  # track3/model.py
    spk = Model(mlp_heads=["spk_sim"])
    spk.load_state_dict(torch.load(CKPT_SPK, map_location="cpu"))
    acc = Model(mlp_heads=["acc_sim"])
    acc.load_state_dict(torch.load(CKPT_ACC, map_location="cpu"))
    _M.update(spk=spk.to(DEVICE).eval(), acc=acc.to(DEVICE).eval())
    return _M


def predict(wave_test, wave_ref):
    """spk_sim & acc_sim (1..5) cho cặp (test, reference) + cosine zero-shot tham khảo."""
    M = _load()
    ta = torch.from_numpy(wave_test).float().unsqueeze(0).to(DEVICE)
    tb = torch.from_numpy(wave_ref).float().unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        o_spk = M["spk"](ta, tb)
        spk = float(o_spk["spk_sim"].item())
        cos = float(o_spk["cos_sim"].item())
        acc = float(M["acc"](ta, tb)["acc_sim"].item())
    return {"spk_sim": round(spk, 4), "acc_sim": round(acc, 4), "cosine": round(cos, 4)}
