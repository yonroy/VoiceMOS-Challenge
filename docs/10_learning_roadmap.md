# 10 — Lộ trình học cho người mới (gắn với dự án)

> Dành cho người mới bắt đầu mảng ML/speech. Triết lý: **học bằng cách làm** — lấy chính dự án VoiceMOS làm giáo trình. Không cần học hết lý thuyết rồi mới làm; vừa làm vừa lấp kiến thức khi cần.

---

## 0. Bức tranh lớn: mình đang làm gì?

Bài toán của mình là **MOS prediction** (dự đoán điểm đánh giá chất lượng giọng nói).

- **MOS = Mean Opinion Score:** người ta cho nhiều người nghe 1 đoạn audio rồi chấm điểm 1–5, lấy trung bình. Đây là "chuẩn vàng" để đánh giá chất lượng giọng (TTS, enhancement...).
- **Vấn đề:** thuê người chấm rất **tốn tiền + chậm**.
- **Giải pháp (việc của mình):** huấn luyện một mô hình AI **tự chấm thay con người** — đầu vào là audio, đầu ra là điểm số. Đo độ tốt bằng **SRCC** (model xếp hạng các mẫu có giống thứ tự người chấm không).

→ Đây là giao điểm của 3 mảng: **xử lý tín hiệu âm thanh** + **deep learning** + **đánh giá/đo lường**.

---

## 0.5. Kim tự tháp kiến thức (cơ bản → nâng cao)

Toàn bộ kiến thức cần cho dự án, xếp từ nền móng lên đỉnh. **Học từ dưới lên**, nhưng không cần hoàn hảo tầng dưới mới lên tầng trên — đủ dùng là tiến.

```
                              ▲
                             ╱ ╲
                            ╱ 5 ╲          TẦNG 5 — NÂNG CAO / NOVELTY
                           ╱─────╲         (cái làm bạn vượt baseline)
                          ╱ Multi- ╲
                         ╱ task,SSL  ╲
                        ╱ fusion,bias  ╲
                       ╱  correction    ╲
                      ╱───────────────────╲
                     ╱       TẦNG 4         ╲   MÔ HÌNH MOS PREDICTION
                    ╱   SSL backbone (WavLM)  ╲  (trái tim của bài toán)
                   ╱  + pooling + head + loss   ╲
                  ╱───────────────────────────────╲
                 ╱            TẦNG 3               ╲  ĐÁNH GIÁ & METRIC
                ╱   MOS là gì, SRCC/MSE, train/val   ╲ (hiểu mình tối ưu cái gì)
               ╱      /eval split, format nộp          ╲
              ╱─────────────────────────────────────────╲
             ╱                  TẦNG 2                    ╲  XỬ LÝ ÂM THANH + ML
            ╱   waveform, 16kHz, spectrogram, PyTorch,      ╲ (công cụ hằng ngày)
           ╱      tensor, training loop, GPU/Kaggle           ╲
          ╱───────────────────────────────────────────────────╲
         ╱                      TẦNG 1                          ╲  NỀN TẢNG
        ╱   Python, numpy, đọc paper, Git, audio cơ bản           ╲ (không có là không đi được)
       ╱─────────────────────────────────────────────────────────╲
```

### 🟫 TẦNG 1 — Nền tảng (học trước tiên)
- **Python + numpy/pandas:** thao tác mảng số, đọc/ghi file.
- **Âm thanh cơ bản:** âm thanh = sóng → số (sampling rate, mono/stereo). Hiểu vì sao mọi thứ resample về **16 kHz**.
- **Git + đọc paper:** sẽ đọc UTMOS, WavLM — học cách lướt paper (abstract → hình → kết quả).

### 🟫 TẦNG 2 — Xử lý âm thanh + ML cơ bản
- **Waveform vs Spectrogram/Mel:** 2 cách biểu diễn âm thanh máy "nhìn" được.
- **PyTorch:** tensor, `Dataset`/`DataLoader`, vòng lặp training (forward → loss → backward → step).
- **GPU & Kaggle T4:** chạy notebook, quản lý bộ nhớ (đã chốt dùng Kaggle T4).

### 🟫 TẦNG 3 — Đánh giá & Metric (rất quan trọng — đừng bỏ qua)
- **MOS là gì:** điểm trung bình người nghe chấm chất lượng giọng (1–5). Bài toán = **đoán điểm đó tự động**.
- **QMOS / EMOS / VAD / CAT:** các loại điểm Track 2 (chất lượng / cảm xúc / valence-arousal-dominance / loại cảm xúc).
- **SRCC (Spearman):** metric chính — đo *thứ hạng* dự đoán có khớp người thật không, không phải sai số tuyệt đối.
- **Train/val/eval split + format `answer.txt`:** luật nộp bài.

### 🟦 TẦNG 4 — Mô hình MOS Prediction (trái tim dự án)
- **SSL backbone (WavLM / HuBERT / Wav2Vec2):** model tự học trên cực nhiều giọng nói → tạo "đặc trưng" (feature) giàu thông tin. (Câu hỏi định hướng backbone trong `11_progress_reports.md`.)
- **Pooling → Head → Score:** từ chuỗi feature → gộp lại → 1 lớp dự đoán → ra điểm.
- **Loss (MSE)** + optimizer/LR/epochs: cách dạy mô hình.
- Bắt đầu từ **baseline SpeechMOS/UTMOS** rồi cải tiến.

### 🟪 TẦNG 5 — Nâng cao / Novelty (giúp vượt baseline)
- **Multi-task learning:** cùng lúc đoán QMOS + EMOS.
- **Fusion:** ghép SSL backbone + **emotion embedding** (emotion2vec) cho điểm cảm xúc.
- **Bias correction / ensemble:** xử lý lệch điểm giữa người chấm, gộp nhiều mô hình.
- **Fine-tune vs freeze backbone:** chiến lược train nâng cao.

> **Lộ trình nhanh:** bạn không cần học hết Tầng 1–2 mới làm (pipeline baseline đã chạy). Củng cố **Tầng 3** ngay (SRCC + MOS) → đào sâu **Tầng 4** qua 2 paper UTMOS & WavLM → khi data Track 2 về thì thí nghiệm **Tầng 5**.

---

## 1. Nền tảng cần có (học tới đâu dùng tới đó)

Đừng cố học hết trước. Đây là thứ tự ưu tiên:

| Mức | Kỹ năng | Tại sao cần | Học nhanh ở đâu |
|---|---|---|---|
| Bắt buộc | **Python** (numpy, pandas) | Mọi code đều là Python | Tự tin đọc/sửa code là đủ |
| Bắt buộc | **Cơ bản ML** (train/val/test, overfit, loss, metric) | Hiểu mình đang tối ưu cái gì | Andrew Ng — Machine Learning (Coursera) |
| Quan trọng | **Deep learning + PyTorch** | Model MOS là mạng neural | fast.ai hoặc "Deep Learning with PyTorch" |
| Quan trọng | **Audio cơ bản** (waveform, sample rate, spectrogram) | Đầu vào là audio | Bài blog + thử với librosa |
| Khi cần | **SSL speech models** (wav2vec2, HuBERT, WavLM) | Backbone của model hiện đại | Đọc khi tới bước fine-tune |

> Thuật ngữ sẽ gặp (sẽ giải thích dần): *embedding* (vector biểu diễn audio), *SSL/self-supervised* (model học từ audio không nhãn), *fine-tune* (lấy model có sẵn rồi huấn luyện thêm cho task mình), *backbone* (phần trích đặc trưng).

---

## 2. Lộ trình theo giai đoạn (gắn mốc dự án)

### Giai đoạn A — Hiểu & chạy được (tuần 1–2) ⬅ ĐANG Ở ĐÂY
**Mục tiêu:** chạy được baseline, hiểu luồng dữ liệu vào → điểm ra.
- [ ] Chạy `track1_baseline.ipynb` trên Kaggle → nộp thử (xem `kaggle_baseline/`).
- [ ] Mở `predictions.csv` xem: mỗi dòng = 1 audio + 1 điểm. **Đây chính là output mình phải tạo.**
- [ ] Đọc lại `08_track2_spec.md` + `09_tracks_overview.md` đến khi hiểu rõ input/output.
- 💡 *Học được:* pipeline inference là gì, format nộp, cách CodaBench chấm.

### Giai đoạn B — Hiểu baseline hoạt động (tuần 2–3)
**Mục tiêu:** không chỉ chạy mà hiểu *tại sao* nó cho ra điểm đó.
- [ ] UTMOS/SpeechMOS: audio → SSL backbone (wav2vec2) → embedding → lớp dự đoán → điểm. Đọc paper UTMOS (Interspeech 2022).
- [ ] emotion2vec: audio → embedding cảm xúc → xác suất 5 lớp. Đọc paper emotion2vec (ACL 2024).
- [ ] Ghi tóm tắt mỗi paper vào `03_literature_notes.md` (3–5 dòng: làm gì, ý chính, kết quả).
- 💡 *Học được:* kiến trúc SSL + cách biến embedding thành điểm số.

### Giai đoạn C — Cải tiến nhỏ, đo đạc (tuần 4–6)
**Mục tiêu:** thử thay đổi và *đo* xem tốt hơn không — đây là tinh thần nghiên cứu.
- [ ] Fine-tune một SSL backbone trên data Track 2, so SRCC với baseline.
- [ ] Thử multi-task (dự đoán QMOS + EMOS cùng lúc).
- [ ] **Ghi MỌI thí nghiệm** vào `04_experiments_log.md` (config → kết quả → nhận xét).
- 💡 *Học được:* vòng lặp nghiên cứu = giả thuyết → thí nghiệm → đo → kết luận.

### Giai đoạn D — Chốt model + viết (tuần 7–10)
- [ ] Chọn model tốt nhất, chạy trên eval data, nộp.
- [ ] Viết system description / draft paper từ `04_experiments_log.md`.
- 💡 *Học được:* cách trình bày kết quả nghiên cứu.

---

## 3. Cách đọc paper (cho người mới)

Đừng đọc tuần tự từ đầu đến cuối. Đọc theo thứ tự:
1. **Abstract** → họ làm gì, kết quả chính.
2. **Hình/bảng** → kiến trúc model + bảng kết quả (thường hiểu được 50% bài).
3. **Introduction** → vấn đề & đóng góp.
4. **Method** → đọc kỹ phần này khi muốn cài lại.
5. Bỏ qua phần toán nặng ở lần đọc đầu — quay lại khi cần.

> Mẹo: đọc paper **đi kèm code**. Đọc 1 đoạn method → tìm đoạn code tương ứng → khớp lại. Hiểu nhanh gấp đôi.

---

## 4. Thói quen tích lũy kiến thức

- **Reproduce trước, sáng tạo sau:** chạy lại được của người khác đã là 70% kỹ năng.
- **Ghi chép ngay:** mỗi khi hiểu một khái niệm mới, viết 2–3 dòng bằng lời của mình vào `03_literature_notes.md`. Kiến thức không ghi sẽ quên.
- **Một câu hỏi mỗi ngày:** gặp thuật ngữ lạ → hỏi (mình, mentor, hoặc search). Đừng để dồn.
- **Học từ lỗi:** mỗi lỗi môi trường/bug ghi vào mục "Lỗi & bài học" của `04_experiments_log.md`.
- **Không sợ hỏi "ngớ ngẩn":** mảng này thuật ngữ nhiều; hỏi sớm tiết kiệm hàng giờ.

---

## 5. Tài nguyên gợi ý

| Chủ đề | Tài nguyên |
|---|---|
| ML cơ bản | Andrew Ng — Machine Learning Specialization (Coursera) |
| Deep learning thực hành | fast.ai *Practical Deep Learning*; *Dive into Deep Learning* (d2l.ai, free) |
| PyTorch | pytorch.org/tutorials (60-min blitz) |
| Audio/speech | HuggingFace **Audio Course** (free, rất hợp dự án này) |
| SSL speech | Đọc paper wav2vec2, HuBERT, WavLM (khi tới giai đoạn C) |
| MOS prediction | Paper UTMOS, MOSNet; repo `sheet` (speech quality assessment) |
| Cộng đồng | HuggingFace forum, papers-with-code, các repo baseline của challenge |

---

## 6. Tâm thế

- **10 tuần không biến người mới thành chuyên gia — và không sao cả.** Mục tiêu thực tế: chạy được baseline, hiểu nó, thử 1–2 cải tiến, ghi lại tử tế. Như vậy đã là một vòng nghiên cứu hoàn chỉnh.
- **Có baseline trên leaderboard > model hoàn hảo chưa chạy.** Cứ nộp baseline trước, cải tiến sau.
- **Mentor là tài nguyên quý** — dùng `02_mentor_questions.md` để hỏi đúng trọng tâm.

> Mỗi khi thấy quá tải: quay lại Giai đoạn đang đứng (mục 2), làm đúng checklist của giai đoạn đó, bỏ qua phần xa hơn.
