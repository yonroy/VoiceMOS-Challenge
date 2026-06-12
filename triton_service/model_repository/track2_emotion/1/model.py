# ─────────────────────────────────────────────────────────────────────────────
# Triton Python backend — Track 2 (Emotional TTS), đủ 6 cột.
#   QMOS        ← UTMOS (SpeechMOS, torch.hub)                  [chất lượng]
#   EMOS/CAT/VAD ← exp08 (WavLM-large fine-tune + audeering frozen → trunk → 3 head)
#
# Triton gọi:
#   - initialize(): 1 LẦN khi model load → nạp toàn bộ checkpoint vào RAM/VRAM.
#   - execute():    MỖI nhóm request → decode audio + suy luận → trả JSON.
#
# Hằng kiến trúc PHẢI khớp exp08 (checkpoint không lưu các số này).
# Phần lớn logic bê nguyên từ api_service/app/tracks/track2.py để 2 bản cho cùng kết quả.
# ─────────────────────────────────────────────────────────────────────────────
import io
import json
import os
import subprocess
import sys

import numpy as np
import torch
import torch.nn as nn
import triton_python_backend_utils as pb_utils

SR = 16000
EMO_MAX_SEC, TRUNK_HIDDEN, HEAD_HIDDEN, DROPOUT, USE_AMP = 8, 512, 128, 0.3, True
EMOTIONS5 = ["angry", "happy", "neutral", "sad", "surprised"]
_EMO_ALIAS = {
    "angry": "angry", "anger": "angry", "happy": "happy", "happiness": "happy", "joy": "happy",
    "neutral": "neutral", "calm": "neutral", "sad": "sad", "sadness": "sad",
    "surprise": "surprised", "surprised": "surprised", "surprising": "surprised",
}

# Có thể đổi qua biến môi trường khi `docker run -e ...`
T2_HF_REPO = os.environ.get("T2_HF_REPO", "tranminhtoan140601/voicemos2026-track2-emotion")
T2_HF_CKPT = os.environ.get("T2_HF_CKPT", "ft_emotion_full_20epoch.pt")
MODELS_DIR = os.environ.get("MODELS_DIR", "/tmp/models")


def norm_emotion(label):
    key = str(label).strip().lower()
    return _EMO_ALIAS.get(key, key if key in EMOTIONS5 else None)


class TritonPythonModel:
    # ───────────────────────── nạp model 1 lần ─────────────────────────
    def initialize(self, args):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        os.makedirs(MODELS_DIR, exist_ok=True)
        self._load_qmos()
        self._load_emotion()
        pb_utils.Logger.log_info(
            f"[track2_emotion] sẵn sàng trên {self.device} "
            f"(audeering {'ON' if self.use_audeering else 'OFF'})")

    def _load_qmos(self):
        self.qmos = torch.hub.load(
            "tarepan/SpeechMOS", "utmos22_strong", trust_repo=True).to(self.device).eval()

    def _load_emotion(self):
        from huggingface_hub import hf_hub_download
        ckpt_path = hf_hub_download(T2_HF_REPO, T2_HF_CKPT)

        # code SAILER để dựng backbone WavLM
        repo = os.path.join(MODELS_DIR, "vox-profile-release")
        if not os.path.exists(repo):
            subprocess.run(["git", "clone", "--depth", "1",
                            "https://github.com/tiantiaf0627/vox-profile-release.git", repo], check=True)
        if repo not in sys.path:
            sys.path.insert(0, repo)

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        assert "wavlm" in ckpt and "heads" in ckpt, \
            "Checkpoint thiếu 'wavlm'/'heads' (cần bản đủ ft_emotion_full_20epoch.pt)."
        AUD_DIM = int(ckpt.get("AUD_DIM", 0))
        self.use_audeering = AUD_DIM > 0

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
            pb_utils.Logger.log_warn(f"[track2_emotion] SAILER wrapper lỗi: {e!r} → fallback WavLM trắng.")
        if wavlm is None:
            from transformers import WavLMModel
            wavlm = WavLMModel.from_pretrained("microsoft/wavlm-large")
        wavlm = wavlm.to(self.device).eval()
        wavlm.config.layerdrop = 0.0
        wavlm.load_state_dict(ckpt["wavlm"], strict=False)
        self.wavlm = wavlm
        WAVLM_DIM = int(wavlm.config.hidden_size)

        # audeering frozen branch (fusion) nếu checkpoint có AUD_DIM > 0
        self.aud_backbone = self.aud_head = self.aud_proc = None
        if self.use_audeering:
            from transformers import Wav2Vec2Model, Wav2Vec2Config, Wav2Vec2Processor
            AUD_NAME = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
            self.aud_proc = Wav2Vec2Processor.from_pretrained(AUD_NAME)
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
            self.aud_backbone = aud_backbone.to(self.device).eval()
            self.aud_head = aud_head.to(self.device).eval()

        N_EMO = len(EMOTIONS5)
        TRUNK_IN = WAVLM_DIM + (AUD_DIM if self.use_audeering else 0)

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

        heads = EmoHeads(TRUNK_IN, TRUNK_HIDDEN, HEAD_HIDDEN, DROPOUT, N_EMO).to(self.device).eval()
        heads.load_state_dict(ckpt["heads"], strict=False)
        self.heads = heads
        self.emos_mu, self.emos_sd = float(ckpt["emos_mu"]), float(ckpt["emos_sd"])
        self.vad_mu = np.asarray(ckpt["vad_mu"], dtype=np.float32)
        self.vad_sd = np.asarray(ckpt["vad_sd"], dtype=np.float32)

    # ───────────────────────── helpers suy luận ─────────────────────────
    def _masked_mean(self, hidden, attn_mask):
        if attn_mask is None:
            return hidden.mean(dim=1)
        try:
            fm = self.wavlm._get_feature_vector_attention_mask(hidden.shape[1], attn_mask)
        except Exception:
            return hidden.mean(dim=1)
        fm = fm.unsqueeze(-1).to(hidden.dtype)
        return (hidden * fm).sum(1) / fm.sum(1).clamp(min=1e-6)

    @torch.no_grad()
    def _audeering_feat(self, wave):
        x = self.aud_proc(wave, sampling_rate=SR).input_values[0]
        x = torch.from_numpy(np.asarray(x, dtype=np.float32)).unsqueeze(0).to(self.device)
        h = self.aud_backbone(x)[0].mean(dim=1)
        out = self.aud_head(h)[0].cpu().numpy()
        vad = np.array([1 + 4 * out[2], 1 + 4 * out[0], 1 + 4 * out[1]], dtype=np.float32)
        return np.concatenate([h[0].cpu().numpy(), vad]).astype(np.float32)

    def _onehot_target(self, tgt):
        v = np.zeros(len(EMOTIONS5), dtype=np.float32)
        if tgt in EMOTIONS5:
            v[EMOTIONS5.index(tgt)] = 1.0
        return v

    @torch.no_grad()
    def _infer_emotion(self, wave, target_emotion):
        wave = wave[: EMO_MAX_SEC * SR].astype(np.float32)
        iv = torch.from_numpy(wave).unsqueeze(0).to(self.device)
        am = torch.ones((1, len(wave)), dtype=torch.long, device=self.device)
        tgt = torch.from_numpy(
            self._onehot_target(norm_emotion(target_emotion) if target_emotion else None)
        ).unsqueeze(0).to(self.device)
        with torch.cuda.amp.autocast(enabled=USE_AMP and self.device == "cuda"):
            fw = self.wavlm(iv, attention_mask=am).last_hidden_state
            fw = self._masked_mean(fw, am)
            if self.use_audeering:
                fw = torch.cat([fw, torch.from_numpy(self._audeering_feat(wave)).unsqueeze(0).to(self.device)], dim=1)
            emos_p, cat_l, vad_p = self.heads(fw, tgt)
        emos = float(emos_p.item()) * self.emos_sd + self.emos_mu
        cat5 = torch.softmax(cat_l, 1)[0].float().cpu().numpy()
        vad3 = vad_p[0].float().cpu().numpy() * self.vad_sd + self.vad_mu
        return emos, cat5, vad3

    @torch.no_grad()
    def _qmos_predict(self, wave):
        x = torch.from_numpy(wave.astype(np.float32)).unsqueeze(0).to(self.device)
        return float(self.qmos(x, SR).item())

    def _decode_audio(self, raw_bytes):
        """bytes file audio → float32 mono 16k."""
        import soundfile as sf
        import librosa
        wave, sr = sf.read(io.BytesIO(raw_bytes), dtype="float32", always_2d=False)
        if wave.ndim > 1:
            wave = wave.mean(axis=1)
        if sr != SR:
            wave = librosa.resample(wave.astype(np.float32), orig_sr=sr, target_sr=SR)
        return np.ascontiguousarray(wave, dtype=np.float32)

    def _predict_one(self, raw_bytes, target_emotion):
        wave = self._decode_audio(raw_bytes)
        if wave.size == 0:
            raise ValueError("Audio rỗng/không đọc được.")
        qmos = self._qmos_predict(wave)
        emos, cat5, vad3 = self._infer_emotion(wave, target_emotion)
        out = {
            "qmos": round(qmos, 4),
            "cat": {e: round(float(cat5[i]), 4) for i, e in enumerate(EMOTIONS5)},
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

    # ───────────────────────── Triton gọi mỗi batch ─────────────────────────
    def execute(self, requests):
        # ── Bước 1: thu thập input của toàn batch ──────────────────────────
        all_raw = []
        all_tgt = []
        for req in requests:
            audio_t = pb_utils.get_input_tensor_by_name(req, "AUDIO_BYTES")
            raw = audio_t.as_numpy().reshape(-1)[0]
            if isinstance(raw, bytes):
                raw_bytes = raw
            else:
                raw_bytes = raw.encode("latin1") if isinstance(raw, str) else bytes(raw)
            all_raw.append(raw_bytes)

            tgt = ""
            tgt_t = pb_utils.get_input_tensor_by_name(req, "TARGET_EMOTION")
            if tgt_t is not None:
                v = tgt_t.as_numpy().reshape(-1)
                if v.size:
                    rv = v[0]
                    tgt = rv.decode("utf-8") if isinstance(rv, bytes) else str(rv)
            all_tgt.append(tgt.strip() or None)

        # ── Bước 2: decode audio; lưu lỗi để trả error riêng từng request ─
        waves = []
        decode_errs = []
        for raw in all_raw:
            try:
                w = self._decode_audio(raw)
                waves.append(w)
                decode_errs.append(None)
            except Exception as e:
                waves.append(None)
                decode_errs.append(e)

        # ── Bước 3: WavLM forward GỘP (pad + attention_mask) ───────────────
        # Đây là phần tốn GPU nhất → gộp 1 lần thay vì N lần riêng lẻ.
        wavlm_feats = [None] * len(waves)
        valid_idx = [i for i, w in enumerate(waves) if w is not None]
        if valid_idx:
            clipped = [waves[i][: EMO_MAX_SEC * SR] for i in valid_idx]
            max_len = max(len(w) for w in clipped)
            padded = np.zeros((len(clipped), max_len), dtype=np.float32)
            attn = np.zeros((len(clipped), max_len), dtype=np.int64)
            for j, w in enumerate(clipped):
                padded[j, : len(w)] = w
                attn[j, : len(w)] = 1
            iv = torch.from_numpy(padded).to(self.device)
            am = torch.from_numpy(attn).to(self.device)
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=USE_AMP and self.device == "cuda"):
                    hidden = self.wavlm(iv, attention_mask=am).last_hidden_state
                    feats = self._masked_mean(hidden, am)  # [N, wavlm_dim]
            for j, i in enumerate(valid_idx):
                wavlm_feats[i] = (feats[j], clipped[j])

        # ── Bước 4: audeering + heads + QMOS — vẫn loop (nhẹ hơn WavLM) ──
        responses = []
        for i, req in enumerate(requests):
            if decode_errs[i] is not None:
                responses.append(pb_utils.InferenceResponse(
                    output_tensors=[],
                    error=pb_utils.TritonError(f"DecodeError: {decode_errs[i]}")))
                continue
            try:
                fw, w_clip = wavlm_feats[i]
                tgt = all_tgt[i]
                tgt_norm = norm_emotion(tgt) if tgt else None

                if self.use_audeering:
                    aud = torch.from_numpy(
                        self._audeering_feat(waves[i])
                    ).unsqueeze(0).to(self.device)
                    fw_full = torch.cat([fw.unsqueeze(0), aud], dim=1)
                else:
                    fw_full = fw.unsqueeze(0)

                tgt_v = torch.from_numpy(
                    self._onehot_target(tgt_norm)
                ).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    emos_p, cat_l, vad_p = self.heads(fw_full, tgt_v)

                emos = float(emos_p.item()) * self.emos_sd + self.emos_mu
                cat5 = torch.softmax(cat_l, 1)[0].float().cpu().numpy()
                vad3 = vad_p[0].float().cpu().numpy() * self.vad_sd + self.vad_mu
                qmos = self._qmos_predict(waves[i])

                out = {
                    "qmos": round(qmos, 4),
                    "cat": {e: round(float(cat5[k]), 4) for k, e in enumerate(EMOTIONS5)},
                    "perceived_emotion": EMOTIONS5[int(np.argmax(cat5))],
                    "vad": {
                        "valence": round(float(vad3[0]), 4),
                        "arousal": round(float(vad3[1]), 4),
                        "dominance": round(float(vad3[2]), 4),
                    },
                }
                if tgt:
                    out["emos"] = round(float(emos), 4)
                    out["target_emotion"] = tgt_norm
                    out["emos_match"] = (out["perceived_emotion"] == tgt_norm)

                payload = json.dumps(out, ensure_ascii=False)
                out_t = pb_utils.Tensor("RESULT_JSON", np.array([[payload]], dtype=object))
                responses.append(pb_utils.InferenceResponse(output_tensors=[out_t]))
            except Exception as e:
                responses.append(pb_utils.InferenceResponse(
                    output_tensors=[],
                    error=pb_utils.TritonError(f"{type(e).__name__}: {e}")))
        return responses

    def finalize(self):
        pb_utils.Logger.log_info("[track2_emotion] dọn dẹp xong.")
