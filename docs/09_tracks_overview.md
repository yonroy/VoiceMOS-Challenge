# 09 — Đặc tả 3 Track (chính thức)

> Nguồn: trang mô tả các track + Submission guideline trên CodaBench (lưu tại `reference/content_btc/`). Cập nhật: 3/6/2026.
> Track 2 là track chính của dự án — chi tiết đầy đủ ở `08_track2_spec.md`. File này để nắm nhanh & so sánh cả 3.

---

## So sánh nhanh

| | Track 1 | **Track 2 ⭐** | Track 3 |
|---|---|---|---|
| **Chủ đề** | Speech enhancement | Emotional TTS | Codec-based synthesis |
| **Dự đoán** | ACR + CCR | QMOS + EMOS (+ CAT, VAD) | Speaker sim + Accent sim |
| **Dataset nền** | URGENT Challenge 2026 | ESD + DailyTalk + 13 TTS | CodecMOS-Accent (nền VCTK) |
| **Train / Val / Eval** | — / 1008+2520 / 4032+10080 | 12,746 / 2,730 / 2,730 | 2,800 / 600 / 600 |
| **Cần reference?** | Không | Không | **Có** (cặp wav_a/wav_b) |
| **File nộp** | `predictions.csv` | `answer.txt` | `answer.txt` |
| **Metric LB** | UTT-SRCC (ACR, CCR) | UTT-SRCC ×5 + categorical error | UTT-SRCC (spk, acc) |
| **Lấy data** | Tự do (HuggingFace) | License form → Erica Cooper ✅ đã nhận (3/6) | License form → Erica Cooper ✅ đã nhận (3/6) |

> Metric chính toàn challenge: **SRCC** (utterance-level = UTT-SRCC). Summary paper thêm MSE/LCC/KTAU ở utterance & system level.

---

## Track 1 — Speech Enhancement (ACR + CCR)
- **Nhiệm vụ:** dự đoán Absolute Category Rating (ACR) và Comparative Category Rating (CCR) cho speech từ hệ thống speech enhancement.
- **Dataset:** từ listening test ICASSP 2026 **URGENT Challenge** — top 6 hệ thống, 840 utterance đa ngôn ngữ (9 ngôn ngữ).
  - Validation: 1008 ACR + 2520 CCR · Evaluation: 4032 ACR + 10080 CCR.
  - **Không có training data chính thức.** Dev set (không kèm label) trên HuggingFace: `urgent-challenge/vmc2026-track1-dev`. Nộp lên CodaBench để lấy UTT-SRCC trên dev; label dev công bố sau khi training phase kết thúc.
- **Metric LB:** UTT-SRCC cho ACR và CCR.
- **Format nộp — `predictions.csv`:**
  ```
  sample_id,pred_score
  vmc2026-track1-test-acr_4588,3.42
  vmc2026-track1-test-ccr_3061,-0.15
  ```
  - `sample_id` là ID, **không** phải path. File chứa đủ cả ACR & CCR.
  - ACR ∈ [1, 5]; CCR ∈ [-3, +3]; số thực OK.
  - Nén: `zip -j <tên>.zip predictions.csv`.
- **Baseline:** URGENT-MOS (arXiv 2601.18438).

---

## Track 2 — Emotional TTS ⭐ (track chính)
> Đầy đủ ở **`08_track2_spec.md`**. Tóm tắt:
- **Nhiệm vụ:** dự đoán (1) **QMOS** chất lượng, (2) **EMOS** độ giống cảm xúc target (bắt buộc); tùy chọn (3) **CAT** phân bố cảm xúc, (4) **VAD** valence/arousal/dominance.
- **Dataset:** thu mới trên nền **ESD + DailyTalk** + mẫu từ **13 hệ thống TTS**; 5 cảm xúc (neutral/happy/angry/sad/surprised). Train 12,746 · Val 2,730 · Eval 2,730.
- **Metric LB (6):** UTT-SRCC cho QMOS, EMOS, Valence, Arousal, Dominance + **categorical error** cho CAT (L1 trung bình giữa vector tỉ lệ vote gt và vector xác suất dự đoán, càng thấp càng tốt).
- **Format nộp — `answer.txt`:** header `wav,QMOS,EMOS,CAT,VAL,ARO,DOM` (QMOS+EMOS bắt buộc, còn lại tùy chọn/tập con). CAT dạng `angry:p|happy:p|neutral:p|sad:p|surprised:p`. Nén `zip -j <tên>.zip answer.txt`.
- **Baseline:** UTMOS (QMOS) · Emotion2vec (CAT) · Gemini LLM-as-judge (EMOS + VAD).

---

## Track 3 — Codec-based Speech Synthesis (speaker + accent similarity)
- **Nhiệm vụ:** cho input speech + **reference speech**, dự đoán điểm tương đồng **speaker** và **accent** (thang 1–5).
- **Dataset:** **CodecMOS-Accent**, nền **VCTK**, 16kHz. Hai loại mẫu: resynthesis (9 hệ thống) + voice-cloned TTS (15 hệ thống) = 24 hệ thống (đều open-source, loại commercial). Có cả mẫu tự nhiên VCTK (tính là 1 system + dùng làm reference).
  - Train 2,800 · Val 600 · Eval 600.
- **Phải tải 2 phân phối** (do license): `..._vctk` (mẫu tự nhiên VCTK) + `..._syn` (mẫu tổng hợp). Copy file `.wav` từ `_vctk/wav/` sang `_syn/wav/`.
  - `sets/train.csv`: `system_id,utterance_id,listener_id,wav_a_path,wav_b_path,spk_sim,acc_sim` (điểm theo từng listener → nhiều dòng cùng cặp mẫu).
  - `sets/dev.csv`: `system_id,utterance_id,wav_a_path,wav_b_path`.
- **Metric LB:** UTT-SRCC cho speaker similarity và accent similarity.
- **Format nộp — `answer.txt`:**
  ```
  system_id,utterance_id,wav_a_path,wav_b_path,pred_acc_sim,pred_spk_sim
  sys003,utt010,wav/vmc2026-track3-sys003-utt010.wav,wav/vmc2026-track3-sys019-utt009.wav,0.7627,0.7627
  ```
  - Được nộp chỉ `pred_acc_sim`, chỉ `pred_spk_sim`, hoặc cả hai. Số thực OK. Nén `zip -j <tên>.zip answer.txt`.
- **Baseline:** phương pháp dựa trên speaker embedding.

---

## Quy trình & luật nộp (chung)
- **Mỗi lần chỉ nộp 1 track**; `answer.txt`/`predictions.csv` chỉ chứa mẫu của track đó. Trên CodaBench (tab My Submissions) chọn đúng track ở **Selected Tracks** và **bỏ chọn** track khác (không bỏ → track thừa bị gán điểm rác MSE=100, SRCC=0; bỏ đúng → n/a).
- **Training phase:** bắt buộc nộp **≥1 lần**, tối đa **30 lần/ngày**.
- **Evaluation phase:** tối đa **3 lần**, chọn **1** làm đáp án cuối.
- Bắt buộc nộp **system description** (kiến trúc + training + data, có hình; khuyến khích arXiv + open code). Được dùng mọi **public dataset** (phải khai báo); cấm proprietary/tự thu MOS trừ khi đã public.
- FAQ nộp lần đầu: https://voicemos-challenge-2023.github.io/codalab.html

## Liên kết
- CodaBench competition: https://www.codabench.org/competitions/16419/?secret_key=<XEM_EMAIL_BTC>
- Baseline toolkit: https://github.com/voicemos-challenge/vmc2026-baselines
- Track 1 dev set: https://huggingface.co/datasets/urgent-challenge/vmc2026-track1-dev
