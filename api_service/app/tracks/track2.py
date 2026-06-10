# Track 2 — Emotional TTS evaluator, đủ 6 cột:
#   QMOS  ← UTMOS (SpeechMOS, torch.hub)         [chất lượng]
#   EMOS/CAT/VAD ← exp08 (WavLM fine-tune + audeering frozen → trunk → 3 head)
# Hằng kiến trúc PHẢI khớp exp08 (checkpoint không lưu các số này).
import os, sys, subprocess
import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download

SR = 16000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EMO_MAX_SEC, TRUNK_HIDDEN, HEAD_HIDDEN, DROPOUT, USE_AMP = 8, 512, 128, 0.3, True
EMOTIONS5 = ["angry", "happy", "neutral", "sad", "surprised"]
_EMO_ALIAS = {
    "angry": "angry", "anger": "angry", "happy": "happy", "happiness": "happy", "joy": "happy",
    "neutral": "neutral", "calm": "neutral", "sad": "sad", "sadness": "sad",
    "surprise": "surprised", "surprised": "surprised", "surprising": "surprised",
}
# Checkpoint exp08 trên Hugging Face (đổi qua biến môi trường nếu cần)
T2_HF_REPO = os.environ.get("T2_HF_REPO", "tranminhtoan140601/voicemos2026-track2-emotion")
T2_HF_CKPT = os.environ.get("T2_HF_CKPT", "ft_emotion_full_20epoch.pt")
MODELS_DIR = os.environ.get("MODELS_DIR", "/tmp/models")

_EMO = {}
_QMOS = {}


def norm_emotion(label):
    key = str(label).strip().lower()
    return _EMO_ALIAS.get(key, key if key in EMOTIONS5 else None)


def _qmos_predict(wave):
    """UTMOS (SpeechMOS) → QMOS 1..5."""
    if "m" not in _QMOS:
        _QMOS["m"] = torch.hub.load("tarepan/SpeechMOS", "utmos22_strong", trust_repo=True).to(DEVICE).eval()
    m = _QMOS["m"]
    with torch.no_grad():
        x = torch.from_numpy(wave.astype(np.float32)).unsqueeze(0).to(DEVICE)
        return float(m(x, SR).item())


def _emo_load():
    if "infer" in _EMO:
        return _EMO["infer"]
    os.makedirs(MODELS_DIR, exist_ok=True)
    ckpt_path = hf_hub_download(T2_HF_REPO, T2_HF_CKPT)

    # code SAILER để dựng backbone WavLM
    repo = os.path.join(MODELS_DIR, "vox-profile-release")
    if not os.path.exists(repo):
        subprocess.run(["git", "clone", "--depth", "1",
                        "https://github.com/tiantiaf0627/vox-profile-release.git", repo], check=True)
    if repo not in sys.path:
        sys.path.insert(0, repo)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert "wavlm" in ckpt and "heads" in ckpt, "Checkpoint thiếu 'wavlm'/'heads' (cần bản đủ ft_emotion_full_20epoch.pt)."
    AUD_DIM = int(ckpt.get("AUD_DIM", 0))
    USE_AUDEERING = AUD_DIM > 0

    def find_hf_backbone(module):
        cands = []
        for name, m in module.named_modules():
            enc = getattr(m, "encoder", None)
            if getattr(m, "feature_extractor", None) is not None and enc is not None \
                    and getattr(enc, "layers", None) is not None:
                cands.append((name, m))
        if not cands:
            return None, None
        cands.sort(key=lambda nm: sum(p.numel() for p in nm[1].parameters()), reverse=True)
        return cands[0]

    wavlm = None
    try:
        from src.model.emotion.wavlm_emotion import WavLMWrapper
        _wrapper = WavLMWrapper.from_pretrained("tiantiaf/wavlm-large-categorical-emotion")
        _name, wavlm = find_hf_backbone(_wrapper)
    except Exception as e:
        print("⚠️ SAILER wrapper lỗi:", repr(e), "→ fallback WavLM trắng.")
    if wavlm is None:
        from transformers import WavLMModel
        wavlm = WavLMModel.from_pretrained("microsoft/wavlm-large")
    wavlm = wavlm.to(DEVICE).eval()
    WAVLM_DIM = int(wavlm.config.hidden_size)
    wavlm.config.layerdrop = 0.0
    wavlm.load_state_dict(ckpt["wavlm"], strict=False)

    def masked_mean(hidden, attn_mask):
        if attn_mask is None:
            return hidden.mean(dim=1)
        try:
            fm = wavlm._get_feature_vector_attention_mask(hidden.shape[1], attn_mask)
        except Exception:
            return hidden.mean(dim=1)
        fm = fm.unsqueeze(-1).to(hidden.dtype)
        return (hidden * fm).sum(1) / fm.sum(1).clamp(min=1e-6)

    aud_backbone = aud_head = aud_proc = None
    if USE_AUDEERING:
        from transformers import Wav2Vec2Model, Wav2Vec2Config, Wav2Vec2Processor
        AUD_NAME = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
        aud_proc = Wav2Vec2Processor.from_pretrained(AUD_NAME)
        aud_backbone = Wav2Vec2Model(Wav2Vec2Config.from_pretrained(AUD_NAME))
        try:
            _sd = __import__("safetensors.torch", fromlist=["load_file"]).load_file(
                hf_hub_download(AUD_NAME, "model.safetensors"))
        except Exception:
            _sd = torch.load(hf_hub_download(AUD_NAME, "pytorch_model.bin"), map_location="cpu")
        bb_sd = {k[len("wav2vec2."):]: v for k, v in _sd.items() if k.startswith("wav2vec2.")}
        aud_backbone.load_state_dict(bb_sd, strict=False)
        _hid = _sd["classifier.dense.weight"].shape[0]
        aud_head = nn.Sequential(nn.Linear(_hid, _hid), nn.Tanh(),
                                 nn.Linear(_hid, _sd["classifier.out_proj.weight"].shape[0]))
        aud_head[0].weight.data.copy_(_sd["classifier.dense.weight"]); aud_head[0].bias.data.copy_(_sd["classifier.dense.bias"])
        aud_head[2].weight.data.copy_(_sd["classifier.out_proj.weight"]); aud_head[2].bias.data.copy_(_sd["classifier.out_proj.bias"])
        aud_backbone = aud_backbone.to(DEVICE).eval(); aud_head = aud_head.to(DEVICE).eval()

    @torch.no_grad()
    def audeering_feat(wave):
        x = aud_proc(wave, sampling_rate=SR).input_values[0]
        x = torch.from_numpy(np.asarray(x, dtype=np.float32)).unsqueeze(0).to(DEVICE)
        h = aud_backbone(x)[0].mean(dim=1)
        out = aud_head(h)[0].cpu().numpy()
        vad = np.array([1 + 4 * out[2], 1 + 4 * out[0], 1 + 4 * out[1]], dtype=np.float32)
        return np.concatenate([h[0].cpu().numpy(), vad]).astype(np.float32)

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

    heads = EmoHeads(TRUNK_IN, TRUNK_HIDDEN, HEAD_HIDDEN, DROPOUT, N_EMO).to(DEVICE).eval()
    heads.load_state_dict(ckpt["heads"], strict=False)
    emos_mu, emos_sd = float(ckpt["emos_mu"]), float(ckpt["emos_sd"])
    vad_mu = np.asarray(ckpt["vad_mu"], dtype=np.float32)
    vad_sd = np.asarray(ckpt["vad_sd"], dtype=np.float32)

    def onehot_target(tgt):
        v = np.zeros(N_EMO, dtype=np.float32)
        if tgt in EMOTIONS5:
            v[EMOTIONS5.index(tgt)] = 1.0
        return v

    @torch.no_grad()
    def infer_wave(wave, target_emotion):
        wave = wave[: EMO_MAX_SEC * SR].astype(np.float32)
        iv = torch.from_numpy(wave).unsqueeze(0).to(DEVICE)
        am = torch.ones((1, len(wave)), dtype=torch.long, device=DEVICE)
        tgt = torch.from_numpy(onehot_target(norm_emotion(target_emotion) if target_emotion else None)).unsqueeze(0).to(DEVICE)
        with torch.cuda.amp.autocast(enabled=USE_AMP and DEVICE == "cuda"):
            fw = wavlm(iv, attention_mask=am).last_hidden_state
            fw = masked_mean(fw, am)
            if USE_AUDEERING:
                fw = torch.cat([fw, torch.from_numpy(audeering_feat(wave)).unsqueeze(0).to(DEVICE)], dim=1)
            emos_p, cat_l, vad_p = heads(fw, tgt)
        emos = float(emos_p.item()) * emos_sd + emos_mu
        cat5 = torch.softmax(cat_l, 1)[0].float().cpu().numpy()
        vad3 = vad_p[0].float().cpu().numpy() * vad_sd + vad_mu
        return emos, cat5, vad3

    print(f"✅ Track 2 exp08 nạp xong (audeering {'ON' if USE_AUDEERING else 'OFF'}) từ {ckpt_path}")
    _EMO["infer"] = infer_wave
    return infer_wave


def predict(wave, target_emotion=None):
    """Trả 6 cột: qmos, (emos nếu có target), cat{5 lớp}, vad{valence,arousal,dominance}."""
    qmos = _qmos_predict(wave)
    infer_wave = _emo_load()
    emos, cat5, vad3 = infer_wave(wave, target_emotion)
    cat = {e: round(float(cat5[i]), 4) for i, e in enumerate(EMOTIONS5)}
    out = {
        "qmos": round(qmos, 4),
        "cat": cat,
        "perceived_emotion": EMOTIONS5[int(np.argmax(cat5))],
        "vad": {
            "valence": round(float(vad3[0]), 4),
            "arousal": round(float(vad3[1]), 4),
            "dominance": round(float(vad3[2]), 4),
        },
    }
    if target_emotion:
        out["emos"] = round(float(emos), 4)
        out["target_emotion"] = norm_emotion(target_emotion)
        out["emos_match"] = (out["perceived_emotion"] == out["target_emotion"])
    return out
