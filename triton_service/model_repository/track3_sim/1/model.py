import io
import json
import os
import subprocess
import sys

import numpy as np
import triton_python_backend_utils as pb_utils

SR = 16000
MODELS_DIR = os.environ.get("MODELS_DIR", "/tmp/models")
ROOT = os.path.join(MODELS_DIR, "vmc2026-baselines")
T3 = os.path.join(ROOT, "track3")
CKPT_SPK = os.environ.get("T3_CKPT_SPK",
    f"{T3}/official-egs/spk_sim_adamw_lr1e-3/model_spk_sim_step20000.pt")
CKPT_ACC = os.environ.get("T3_CKPT_ACC",
    f"{T3}/official-egs/acc_sim_adamw_lr1e-3/model_acc_sim_step20000.pt")


def _decode_audio(raw_bytes):
    import soundfile as sf
    import librosa
    wave, sr = sf.read(io.BytesIO(raw_bytes), dtype="float32", always_2d=False)
    if wave.ndim > 1:
        wave = wave.mean(axis=1)
    if sr != SR:
        wave = librosa.resample(wave.astype(np.float32), orig_sr=sr, target_sr=SR)
    return np.ascontiguousarray(wave, dtype=np.float32)


class TritonPythonModel:
    def initialize(self, args):
        import torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        os.makedirs(MODELS_DIR, exist_ok=True)
        if not os.path.isdir(ROOT):
            subprocess.run(
                ["git", "clone", "-q",
                 "https://github.com/voicemos-challenge/vmc2026-baselines.git", ROOT],
                check=True)
        if T3 not in sys.path:
            sys.path.insert(0, T3)
        from model import Model
        self.spk = Model(mlp_heads=["spk_sim"])
        self.spk.load_state_dict(torch.load(CKPT_SPK, map_location="cpu"))
        self.spk = self.spk.to(self.device).eval()
        self.acc = Model(mlp_heads=["acc_sim"])
        self.acc.load_state_dict(torch.load(CKPT_ACC, map_location="cpu"))
        self.acc = self.acc.to(self.device).eval()
        pb_utils.Logger.log_info(f"[track3_sim] sẵn sàng trên {self.device}")

    def execute(self, requests):
        import torch

        # Batch WavLM (ECAPA) forward: pad cặp test+ref theo chiều dài tối đa
        all_test, all_ref, decode_errs = [], [], []
        for req in requests:
            try:
                def _get(name):
                    t = pb_utils.get_input_tensor_by_name(req, name)
                    raw = t.as_numpy().reshape(-1)[0]
                    if not isinstance(raw, bytes):
                        raw = raw.encode("latin1") if isinstance(raw, str) else bytes(raw)
                    return _decode_audio(raw)
                wt = _get("AUDIO_BYTES_TEST")
                wr = _get("AUDIO_BYTES_REF")
                all_test.append(wt); all_ref.append(wr); decode_errs.append(None)
            except Exception as e:
                all_test.append(None); all_ref.append(None); decode_errs.append(e)

        def _pad_batch(waves):
            max_len = max(len(w) for w in waves)
            out = np.zeros((len(waves), max_len), dtype=np.float32)
            for i, w in enumerate(waves):
                out[i, :len(w)] = w
            return torch.from_numpy(out)

        valid_idx = [i for i, e in enumerate(decode_errs) if e is None]
        spk_results = [None] * len(requests)
        acc_results = [None] * len(requests)
        cos_results = [None] * len(requests)

        if valid_idx:
            t_batch = _pad_batch([all_test[i] for i in valid_idx]).to(self.device)
            r_batch = _pad_batch([all_ref[i] for i in valid_idx]).to(self.device)
            with torch.no_grad():
                o_spk = self.spk(t_batch, r_batch)
                o_acc = self.acc(t_batch, r_batch)
            for j, i in enumerate(valid_idx):
                spk_results[i] = float(o_spk["spk_sim"][j].item())
                acc_results[i] = float(o_acc["acc_sim"][j].item())
                cos_results[i] = float(o_spk["cos_sim"][j].item())

        responses = []
        for i in range(len(requests)):
            if decode_errs[i] is not None:
                responses.append(pb_utils.InferenceResponse(
                    output_tensors=[],
                    error=pb_utils.TritonError(f"DecodeError: {decode_errs[i]}")))
                continue
            out = {
                "spk_sim": round(spk_results[i], 4),
                "acc_sim": round(acc_results[i], 4),
                "cosine": round(cos_results[i], 4),
            }
            payload = json.dumps(out, ensure_ascii=False)
            out_t = pb_utils.Tensor("RESULT_JSON", np.array([[payload]], dtype=object))
            responses.append(pb_utils.InferenceResponse(output_tensors=[out_t]))
        return responses

    def finalize(self):
        pb_utils.Logger.log_info("[track3_sim] dọn dẹp xong.")
