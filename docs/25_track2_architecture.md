# 25 — Kiến trúc hệ thống Track 2 (Emotional TTS MOS prediction)

> Tài liệu mô tả kiến trúc **hệ thống Track 2** — bản tổng hợp để đưa vào slide / paper / system description.
> Nguồn sự thật: [04_experiments_log.md](04_experiments_log.md) + [12_system_description.md](12_system_description.md). Cập nhật ngày: 16/6/2026.
>
> 👤 Người đọc mới về ML/speech → mọi thuật ngữ đều có giải thích ngắn ngay lần đầu dùng.

---

## 0. Bài toán Track 2 (nhắc lại nhanh)

Cho **1 đoạn giọng nói cảm xúc** (do TTS sinh hoặc người thật nói), hệ thống dự đoán **6 con số**:

| Cột | Ý nghĩa | Thang |
|---|---|---|
| **QMOS** | Quality MOS — chất lượng giọng (sạch/tự nhiên, ít artifact) | 1–5 |
| **EMOS** | Emotion MOS — độ khớp với cảm xúc mục tiêu (target emotion) | 1–5 |
| **CAT** | Tỉ lệ vote 5 cảm xúc (Neutral/Happy/Sad/Angry/Surprise) | vector 5, tổng = 1 |
| **VAL / ARO / DOM** | Valence / Arousal / Dominance — 3 trục mô tả cảm xúc | mỗi cái 1–5 |

- **Metric chính:** UTT-SRCC (Spearman — đo **thứ hạng**, không đo giá trị tuyệt đối) cho QMOS/EMOS/VAD; CAT dùng **categorical error**.
- **Đầu ra nộp:** file `answer.txt` mỗi dòng `wav,QMOS,EMOS,CAT,VAL,ARO,DOM`.
- **Dữ liệu:** train 12.746 · val 2.730 · eval 2.730 (ráp từ gói BTC + ESD + DailyTalk, chuẩn hóa âm lượng sv56).

> **Vì SRCC chỉ chấm thứ hạng** → ta được phép **ghép cột từ nhiều model khác nhau** miễn mỗi cột xếp hạng tốt. Đây là nền tảng của chiến lược "trộn cột" bên dưới.

---

## 1. Ý tưởng cốt lõi: "trộn cột" (column-mixing ensemble)

Grader chấm **từng cột độc lập** trên `answer.txt`. Mỗi nhánh model giỏi ở một nhóm cột khác nhau → ta lấy cột tốt nhất của từng nhánh ghép lại thành 1 đáp án.

```
                          ┌──────────────────────────────┐
   1 file audio cảm xúc ──┤   3 NHÁNH chạy độc lập        │── ghép cột → answer.txt (6 cột)
        (.wav 16kHz)      └──────────────────────────────┘
                                          │
        ┌─────────────────────────────────┼─────────────────────────────────┐
        ▼                                 ▼                                 ▼
 ┌────────────────┐               ┌────────────────┐               ┌────────────────┐
 │  NHÁNH A       │               │  NHÁNH B v2    │               │  NHÁNH C       │
 │  exp08         │               │  exp13         │               │  exp15 (Mamba) │
 │  WavLM         │               │  UTMOS         │               │  WavLM ft +    │
 │  fine-tune     │               │  fine-tune     │               │  Mamba head    │
 └───────┬────────┘               └───────┬────────┘               └───────┬────────┘
         │                                │                                │
   EMOS · CAT · VAL · DOM               QMOS                             ARO
   (4 cột cảm xúc)                    0.6296 🏆                        0.7978 🏆
         │                                │                                │
         └────────────────┬───────────────┴────────────────────────────────┘
                          ▼
       answer.txt:  wav , QMOS , EMOS , CAT , VAL , ARO , DOM
                            └B┘   └─────A─────┘  └A┘  └C┘  └A┘
```

**Nguyên lý chung quán xuyến cả 3 nhánh:** *fine-tune về đúng domain thắng frozen ở cả hai nhóm cột.*
- Nhóm cảm xúc: fine-tune WavLM (exp08) **thắng** fusion đóng băng (exp04/exp07).
- Chất lượng: fine-tune UTMOS (exp13 = 0.63) **thắng** head đóng băng + neo UTMOS (exp07 = 0.548) **thắng** UTMOS zero-shot (0.414).

> **Thuật ngữ:**
> - **frozen (đóng băng):** giữ nguyên trọng số model gốc, chỉ dùng nó như bộ trích đặc trưng → nhanh, tiết kiệm GPU, nhưng không "lái" được về domain mới.
> - **fine-tune:** cho model học tiếp trên dữ liệu Track 2 → bám đúng domain cảm xúc TTS → điểm cao hơn nhưng tốn GPU.
> - **domain:** "miền dữ liệu" — vd giọng TTS cảm xúc khác giọng đọc tin tức; model học đúng domain mới điểm cao.

---

## 2. 🅰️ NHÁNH A — exp08: 5 cột cảm xúc (nhánh quan trọng nhất)

WavLM fine-tune + audeering đóng băng → trunk chung → 3 head (EMOS/CAT/VAD).

```
                              wav (16kHz)
                                  │
              ┌───────────────────┴────────────────────┐
              ▼                                         ▼
   [A1] WavLM-large                          [A2] audeering wav2vec2-large
   (SAILER warm-start;                       (ĐÓNG BĂNG hoàn toàn, cache .npz)
    FINE-TUNE 6 lớp Transformer trên cùng)        │
              │ mean-pool theo thời gian            │ → [emb 1024 | VAD 3] = 1027-D
              ▼ 1024-D                              ▼
              └───────────────── concat ────────────┘
                                  │ 2051-D
                                  ▼
                     [A3] TRUNK chung (MLP) ──► 512-D
                          (multi-task + uncertainty weighting)
                                  │
                ┌─────────────────┼──────────────────┐
                ▼                 ▼                  ▼
        [A4a] EMOS head     [A4b] CAT head    [A4c] VAD head
        [trunk 512 |        512 → 128 → 5     512 → 128 → 3
         one-hot target 5]  → softmax         → giải z-score
        517 → 128 → 1       (tỉ lệ 5 cảm xúc) (VAL/ARO/DOM 1–5)
        → giải z-score
                │                 │                  │
              EMOS               CAT            VAL · ARO · DOM
```

**Giải thích từng khối:**
- **[A1] WavLM-large** — mạng SSL (self-supervised learning: học không cần nhãn từ lượng audio khổng lồ) biến wav → chuỗi vector ngữ nghĩa.
  - **SAILER warm-start:** khởi tạo bằng trọng số model SAILER (`tiantiaf/wavlm-large-categorical-emotion`, vô địch SER IS2025) thay vì trọng số gốc → bắt đầu đã "biết cảm xúc".
  - **Fine-tune 6 lớp trên cùng:** chỉ mở khóa học tiếp 6 lớp Transformer trên cùng (phần dưới đóng băng) → vừa đủ học, vừa tiết kiệm GPU T4.
- **mean-pool:** gộp chuỗi nhiều frame thành 1 vector 1024 chiều bằng trung bình (tôn trọng độ dài thật qua attention-mask).
- **[A2] audeering** — model VAD chuyên, đóng băng, trích 1 lần rồi **cache** (lưu ra `.npz`) để khỏi chạy lại.
- **[A3] trunk** — "thân" MLP dùng chung cho mọi head → các task **đồng học** (multi-task), chia sẻ biểu diễn.
  - **uncertainty weighting:** tự động cân trọng số loss giữa các task (task khó được nhường), không phải chỉnh tay.
- **head:** lớp MLP nhỏ ở cuối, mỗi cái lo 1 nhiệm vụ. EMOS cần biết **target emotion** nên ghép thêm one-hot 5 chiều.
- **z-score:** chuẩn hóa nhãn về trung bình 0 / lệch chuẩn 1 lúc train; lúc dự đoán **giải ngược** về thang 1–5.

**Điểm (DEV):** EMOS **0.8116** 🏆 · CAT-err **0.1331** 🏆 · VAL **0.6605** 🏆 · ARO 0.7904 · DOM **0.7539** 🏆.

---

## 3. 🅱️ NHÁNH B v2 — exp13: cột QMOS (kỷ lục 0.6296)

```
   wav ──► [B'] UTMOS22_strong  (SpeechMOS, tải qua torch.hub)
              · TRAINABLE end-to-end (fine-tune cả mạng, warm-start trọng số gốc)
              · Train trên: nhãn qMOS THẬT của Track 2
              · Loss: MSE (RANK_LAMBDA tùy chọn cho ranking loss)
                       │
                       ▼
                     QMOS (1–5)
```

- **UTMOS22_strong:** model dự đoán MOS chất lượng giọng, vốn là baseline QMOS.
- **Tiến hóa của cột QMOS:**
  - UTMOS **zero-shot** (chưa train) → 0.414
  - exp07: UTMOS đóng băng làm "neo" + head riêng → 0.548
  - **exp13: fine-tune thẳng UTMOS** về domain Track 2 → **0.6296** 🏆 (cái "neo" giờ nằm sẵn trong trọng số, bỏ head riêng).
- **MSE vs SRCC:** train bằng MSE (sai số bình phương) nhưng chấm bằng SRCC (thứ hạng) → có lệch; `RANK_LAMBDA` thêm ranking loss để tối ưu thẳng thứ hạng (đang là hướng ablation).

---

## 4. 🅲 NHÁNH C — exp15: cột ARO (kỷ lục 0.7978)

Giống nhánh A (WavLM fine-tune cho cảm xúc) nhưng **thay `mean-pool` bằng Mamba head**:

```
   WavLM-large (fine-tune) ──► frame-level features (chuỗi theo thời gian)
                                        │
                                        ▼
                              [Mamba head 2 chiều]   ◄── thay cho mean-pool
                                        │
                                        ▼
                                  heads cảm xúc → (EMOS/CAT/VAD)
```

- **Mamba:** kiến trúc **state-space** xử lý chuỗi với chi phí **O(n)** (Transformer là O(n²)) → "đọc" diễn biến cảm xúc theo thời gian thay vì lấy trung bình thô.
- **Kết quả ablation:** Mamba ≈ mean-pool ở 4 cột, **chỉ thắng rõ ở ARO** (0.7933 → **0.7978**) → một ablation đẹp cho paper ("pooling theo thời gian giúp cột arousal").

---

## 5. So sánh các phiên bản (DEV)

| Hệ | QMOS | EMOS | CAT-err | VAL / ARO / DOM | Ghi chú |
|---|---|---|---|---|---|
| Baseline (3/6) | 0.414 | 0.194 | 0.193 | — | UTMOS + Gemini |
| exp01 emotion2vec | 0.414 | 0.637 | 0.193 | — | EMOS vượt SAILER |
| exp03 SAILER | 0.414 | 0.562 | 0.190 | 0.341/0.712/0.630 | mở 3 cột VAD |
| exp04 fusion frozen | 0.414 | 0.788 | 0.145 | 0.578/0.754/0.706 | trunk multi-task |
| **exp07 fusion+QMOS** | **0.548** | 0.795 | 0.153 | 0.581/0.752/0.705 | head QMOS thứ 4 |
| **exp08 fine-tune WavLM** | 0.414 | **0.811** 🏆 | **0.133** 🏆 | **0.659**🏆/0.793🏆/**0.751**🏆 | **nhánh A** |
| exp08b resume | 0.417 | 0.8116 | 0.1331 | 0.6605/0.7904/0.7539 | hội tụ |
| **exp13 fine-tune UTMOS** | **0.6296** 🏆 | — | — | — | **nhánh B** |
| **exp15 Mamba (QMOS←exp13)** | 0.6296 | 0.8070 | 0.1349 | 0.6545/**0.7978**🏆/0.7506 | **nhánh C** |

> **Hệ 6 cột mạnh nhất hiện tại (best-per-column, 10/6):**
> QMOS **0.6296** (exp13) · EMOS **0.8144** (exp18 cross-attn) · CAT **0.1331** (exp08) · VAL **0.6605** (exp08b) · ARO **0.7978** (exp15) · DOM **0.7539** (exp08b).
>
> 🆕 **Biến thể exp18 (cross-attention fusion, 16/6):** thay concat `[WavLM | audeering]` của Nhánh A bằng **cross-attention** (WavLM *query* ⟷ audeering *key/value*), backbone **đóng băng + cache frame-level**, chỉ train fusion+heads (~1.7M tham số) → EMOS **0.8144** 🏆 (kỷ lục cột). Cùng điểm với fine-tune exp08 mà rẻ hơn nhiều → góc "hiệu quả" cho paper. Code: `kaggle_baseline/track2/exp18_crossattn_emotion_pipeline.py`.
> ⚠️ Bản TRỘN CỘT mới (QMOS←exp13 + ARO←exp15 + còn lại←exp08) **chưa nộp**; bản đã nộp 9/6 (`exp_mix`) dùng QMOS←exp07 (0.548).

---

## 6. Chi tiết huấn luyện (training details)

| Nhánh | Backbone | Cách train | Hyperparameter chính |
|---|---|---|---|
| **A (exp08)** | WavLM-large (SAILER warm-start) | fine-tune 6 lớp trên + trunk/head | LR backbone 1e-5, trunk/head 1e-3; BATCH 4 × ACCUM 8 (effective 32); AMP + gradient checkpointing; EPOCHS ≤12, early-stop theo SRCC val nội bộ; checkpoint lưu **cả backbone + heads** mỗi best |
| **B (exp13)** | UTMOS22_strong | fine-tune end-to-end | LR 1e-5, BATCH 1 × ACCUM 16 (UTMOS không có attention-mask), loss MSE (+RANK_LAMBDA tùy chọn) |
| **C (exp15)** | WavLM-large (ft) + Mamba | fine-tune + Mamba head thay mean-pool | cờ `USE_MAMBA` để ablation; ranking loss `RANK_LAMBDA=0.3` |

- **AMP (mixed precision):** tính bằng số thực 16-bit để tiết kiệm bộ nhớ/nhanh hơn trên T4.
- **gradient checkpointing:** đánh đổi tính lại để giảm VRAM (chạy được model lớn trên 16GB).
- **gradient accumulation (ACCUM):** cộng dồn gradient nhiều batch nhỏ rồi mới cập nhật → giả lập batch lớn khi GPU yếu.
- **early-stop:** dừng khi điểm val nội bộ không cải thiện → tránh overfit.

**Checkpoint:** đã lưu lên Kaggle Dataset `toanminh222/cache-exp8`
(`ft_qmos_utmos.pt` = exp13 · `ft_mamba_emotion_full.pt` = exp15 · `archive/ft_emotion_full_20epoch.pt` = exp08 TỐT NHẤT + cache audeering `.npz`).

---

## 7. Tài nguyên ngoài & license (phải khai báo khi nộp)

| Model | Vai trò trong hệ | License |
|---|---|---|
| `tiantiaf/wavlm-large-categorical-emotion` (SAILER, WavLM-large) | backbone nhánh A (fine-tune) + feature nhánh B | **Open RAIL** (phi thương mại) |
| `audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim` | feature VAD frozen (nhánh A) | **CC BY-NC-SA** (phi thương mại) |
| `iic/emotion2vec_plus_large` (funasr) | feature cảm xúc frozen (nhánh B cũ exp07) | — |
| UTMOS22_strong (SpeechMOS, torch.hub) | nền cột QMOS (nhánh B) | — |

> ⚠️ SAILER (Open RAIL) + audeering (CC BY-NC-SA) đều **phi thương mại** → bắt buộc khai báo rõ trong system description nộp BTC.

---

## 8. Liên kết

- Chi tiết thí nghiệm từng exp: [04_experiments_log.md](04_experiments_log.md)
- Mô tả hệ thống + bảng điểm đầy đủ: [12_system_description.md](12_system_description.md)
- Bảng trạng thái nhanh các exp: [20_experiments_overview.md](20_experiments_overview.md)
- Code: `kaggle_baseline/exp08_*`, `exp13_*`, `exp15_*`
- Cách tính metric: [14_leaderboard_metrics.md](14_leaderboard_metrics.md)
