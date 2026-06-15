import io
import json
import os
import subprocess
import sys

import numpy as np
import triton_python_backend_utils as pb_utils

SR = 16000
MODELS_DIR = os.environ.get("MODELS_DIR", "/tmp/models")
REPO = os.path.join(MODELS_DIR, "URGENT-MOS")
CKPT = os.environ.get("T1_CKPT", "urgent-challenge/urgent-mos-f1c1m5dcorpus")


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
        if not os.path.isdir(REPO):
            subprocess.run(
                ["git", "clone", "-q", "https://github.com/vvwangvv/URGENT-MOS.git", REPO],
                check=True)
        if REPO not in sys.path:
            sys.path.insert(0, REPO)
        # urgent_mos/utils.py import `from torchcodec.decoders import AudioDecoder` ở top-level,
        # nhưng track1 TỰ decode bằng soundfile rồi truyền waveform → AudioDecoder KHÔNG dùng.
        # torchcodec bản khớp torch 2.4 chưa có AudioDecoder → chèn stub cho import qua.
        try:
            import torchcodec.decoders as _tcd
            if not hasattr(_tcd, "AudioDecoder"):
                class _AudioDecoderStub:  # không dùng trên đường suy luận (đã truyền waveform)
                    def __init__(self, *a, **k):
                        raise RuntimeError("AudioDecoder stub: track1 dùng waveform, không decode file.")
                _tcd.AudioDecoder = _AudioDecoderStub
        except Exception as _e:
            pb_utils.Logger.log_warn(f"[track1_acr] không stub được torchcodec: {_e!r}")
        try:
            import importlib; importlib.import_module("urgent_mos.api.infer")
        except Exception:
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", REPO],
                           check=False)
        from urgent_mos.utils import load_model_from_checkpoint
        self.model = load_model_from_checkpoint(CKPT, self.device)
        self.model.eval()
        pb_utils.Logger.log_info(f"[track1_acr] sẵn sàng trên {self.device}")

    def execute(self, requests):
        import torch
        from urgent_mos.api.infer import infer, infer_pairs

        responses = []
        for req in requests:
            try:
                def _get_wave(name):
                    t = pb_utils.get_input_tensor_by_name(req, name)
                    if t is None:
                        return None
                    raw = t.as_numpy().reshape(-1)[0]
                    if not isinstance(raw, bytes):
                        raw = raw.encode("latin1") if isinstance(raw, str) else bytes(raw)
                    if not raw:
                        return None
                    return _decode_audio(raw)

                wa = _get_wave("AUDIO_BYTES_A")
                if wa is None or wa.size == 0:
                    raise ValueError("AUDIO_BYTES_A rỗng")
                wb = _get_wave("AUDIO_BYTES_B")

                ta = torch.from_numpy(wa).float()
                acr_a = max(1.0, min(5.0, float(
                    infer(self.model, [ta], sample_rate=[SR],
                          batch_frames=None, num_workers=0)[0]["mos_overall"])))
                out = {"acr_a": round(acr_a, 4)}

                if wb is not None and wb.size > 0:
                    tb = torch.from_numpy(wb).float()
                    acr_b = max(1.0, min(5.0, float(
                        infer(self.model, [tb], sample_rate=[SR],
                              batch_frames=None, num_workers=0)[0]["mos_overall"])))
                    ccr = max(-3.0, min(3.0, float(
                        infer_pairs(self.model, [(ta, tb)], sample_rate=[(SR, SR)],
                                    batch_frames=None, num_workers=0)[0]["mos_overall"])))
                    out["acr_b"] = round(acr_b, 4)
                    out["ccr"] = round(ccr, 4)

                payload = json.dumps(out, ensure_ascii=False)
                out_t = pb_utils.Tensor("RESULT_JSON", np.array([[payload]], dtype=object))
                responses.append(pb_utils.InferenceResponse(output_tensors=[out_t]))
            except Exception as e:
                responses.append(pb_utils.InferenceResponse(
                    output_tensors=[],
                    error=pb_utils.TritonError(f"{type(e).__name__}: {e}")))
        return responses

    def finalize(self):
        pb_utils.Logger.log_info("[track1_acr] dọn dẹp xong.")
