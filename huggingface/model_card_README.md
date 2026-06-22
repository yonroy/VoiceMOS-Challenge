---
license: cc-by-nc-sa-4.0
language:
  - en
  - zh
tags:
  - speech
  - emotion-recognition
  - mos-prediction
  - voicemos
  - wavlm
  - audio
pipeline_tag: audio-classification
---

# VoiceMOS Challenge 2026 — Track 2 (Emotional TTS) checkpoints

Bộ checkpoint cho **VoiceMOS Challenge 2026 Track 2** — dự đoán điểm MOS cho giọng nói cảm xúc
(EMOS / EmoCat / VAD) và chất lượng (QMOS). Đây là các model train trong dự án; dùng kèm code ở
[demo Space](https://huggingface.co/spaces/yonroy/voicemos2026-demo).

## 📊 KẾT QUẢ — checkpoint `ft_emotion_full_20epoch.pt` (exp08, DEV, UTT-SRCC)

> Điểm của **đúng checkpoint cảm xúc trong repo này** trên tập DEV. `↑` cao hơn = tốt hơn · `↓` thấp hơn = tốt hơn.

| Cột | Điểm (DEV) |
|---|---|
| **EMOS** ↑ | **0.811** |
| **EmoCat-err** ↓ | **0.133** |
| **Valence** ↑ | **0.659** |
| **Arousal** ↑ | **0.793** |
| **Dominance** ↑ | **0.751** |

> ℹ️ Checkpoint này **chuyên cảm xúc** (5 cột trên). **QMOS** (chất lượng giọng) dùng model riêng (`ft_qmos_utmos.pt`), không lấy từ ckpt này. Số liệu nguồn: `docs/18_leaderboard_history.md` (hàng exp08, 5/6).

## Checkpoint trong repo

| File | Experiment | Mô tả | Điểm DEV (UTT-SRCC) |
|---|---|---|---|
| `ft_emotion_full_20epoch.pt` | exp08 | **TỐT NHẤT cảm xúc.** WavLM-large fine-tune (warm-start SAILER) + audeering frozen → trunk → 3 head (EMOS/CAT/VAD) | EMOS **0.811** · CAT-err **0.133** · VAD **0.659/0.793/0.751** |
| `ft_qmos_utmos.pt` | exp13 | Fine-tune UTMOS cho QMOS (chất lượng giọng) | QMOS (exp07 mốc 0.548) |
| `ft_joint_full.pt` | exp11 | Fine-tune đồng thời WavLM + audeering, fusion 1 model | val nội bộ ~0.83 (nghi overfit) |

> **Hệ 6 cột đã NỘP (`exp_mix`)** = trộn cột: 5 cảm xúc ← `ft_emotion_full_20epoch.pt` (bảng trên) + QMOS ← exp07 →
> QMOS 0.548 · EMOS 0.811 · CAT 0.133 · VAD 0.659/0.793/0.751.

## Kiến trúc & hằng số (PHẢI khớp khi nạp `ft_emotion_full_20epoch.pt`)

Checkpoint **không lưu** các hằng kiến trúc → khi nạp phải đặt đúng:

```
TRUNK_HIDDEN = 512 · HEAD_HIDDEN = 128 · EMO_MAX_SEC = 8 · SR = 16000
EMOTIONS5 = ["angry", "happy", "neutral", "sad", "surprised"]
```

Key trong ckpt: `wavlm` (state_dict backbone), `heads` (trunk + 3 head), `emos_mu/emos_sd`,
`vad_mu/vad_sd` (chuẩn hóa nhãn), `AUD_DIM` (>0 = có audeering). Nạp bằng `torch.load(..., weights_only=False)`.

Code nạp đầy đủ: xem `app.py` của Space hoặc `kaggle_baseline/track2/exp08_finetune_emotion_pipeline.py`.

## License — ⚠️ phi thương mại

Checkpoint kế thừa từ nhiều nguồn → tuân theo ràng buộc **nghiêm ngặt nhất**:

| Thành phần | License |
|---|---|
| WavLM (microsoft/wavlm-large) | MIT |
| SAILER (tiantiaf/wavlm-large-categorical-emotion) | Open RAIL |
| audeering wav2vec2 MSP-dim | CC BY-NC-SA 4.0 (**non-commercial**) |

→ Repo này để **CC BY-NC-SA 4.0** (chỉ dùng phi thương mại, ghi nguồn, chia sẻ tương tự).
Data train (BTC VoiceMOS 2026 + ESD + DailyTalk) có license riêng — **không** đóng gói trong repo này.

## Trích dẫn

Dùng cho VoiceMOS Challenge 2026 (Track 2 — Emotional TTS). Paper ICASSP 2027 (in progress).
