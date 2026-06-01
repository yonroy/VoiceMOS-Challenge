# 01 — Kế hoạch nghiên cứu

## Phạm vi (Scope) — ✅ ĐÃ CHỐT (1/6/2026)
**Tham gia cả 3 track, nhưng dồn sức vào Track 2:**
- **Track 2 (Emotional TTS):** track chính — xây hệ thống riêng, tập trung **QMOS + EMOS** (bắt buộc), thêm CAT/VAD nếu còn thời gian.
- **Track 1 & Track 3:** chỉ **chạy baseline** để có mặt trên leaderboard + system description, không đầu tư sâu.

> Lý do: nộp baseline cho track 1&3 chi phí thấp mà vẫn tăng độ phủ; novelty/paper dồn vào track 2.
> ⚠️ Vẫn nên xác nhận hướng **novelty** của track 2 với mentor. Xem `02_mentor_questions.md`.

---

## Hướng tiếp cận (cập nhật sau khi bàn với mentor)

### Ý tưởng đang cân nhắc
- [ ] Fine-tune SSL backbone (WavLM / HuBERT / Wav2Vec2) cho MOS prediction
- [ ] Multi-task learning: dự đoán QMOS + EMOS đồng thời
- [ ] Kết hợp acoustic features + emotional embeddings (Emotion2vec)
- [ ] Cải tiến LLM-as-judge (prompt engineering trên Gemini/baseline)
- [ ] Khai thác phần optional (VAD) — ít người làm, dễ tạo novelty

### Điểm novelty dự kiến
> [Điền sau khi chốt hướng — đây là phần quyết định chất lượng paper]

---

## Milestone (10 tuần, tính từ 1/6 → 7/8)

| Tuần | Thời gian | Mục tiêu | Trạng thái |
|---|---|---|---|
| 1 | 1/6 – 7/6 | ✅ Đăng ký · ✅ join CodaBench · ✅ gửi license Track 2/3 · ✅ chốt GPU=Kaggle · ✅ chốt scope · ⬜ chạy baseline | 🔄 |
| 2 | 8/6 – 14/6 | Hiểu data, reproduce baseline scores | ⬜ |
| 3 | 15/6 – 21/6 | Đọc literature, chốt hướng với mentor | ⬜ |
| 4 | 22/6 – 28/6 | Implement model v1 | ⬜ |
| 5 | 29/6 – 5/7 | Train v1, đánh giá vs baseline | ⬜ |
| 6 | 6/7 – 12/7 | Cải tiến (ablation), model v2 | ⬜ |
| 7 | 13/7 – 19/7 | Tối ưu hyperparameter | ⬜ |
| 8 | 20/7 – 26/7 | Hoàn thiện model cuối, viết draft paper | ⬜ |
| 9 | 27/7 – 31/7 | Nhận eval data, chạy inference | ⬜ |
| 10 | 1/8 – 7/8 | **Nộp kết quả CodaBench** | ⬜ |
| — | 8/8 – 16/9 | Viết & nộp paper ICASSP 2027 | ⬜ |

---

## Sản phẩm cần nộp
1. 🔴 File điểm dự đoán (CodaBench) — hạn 7/8
2. 🔴 System description — sau 31/8
3. 🟡 Paper ICASSP 2027 — hạn 16/9
4. 🟡 Code + model weights (GitHub) — tăng impact

---

## Rủi ro & phương án
| Rủi ro | Phương án |
|---|---|
| ~~Chưa có GPU~~ | ✅ Dùng Kaggle T4 (30h/tuần) — cần verify phone để bật GPU+internet |
| ~~Scope quá rộng~~ | ✅ Đã chốt: Track 2 làm sâu, Track 1&3 chỉ chạy baseline |
| Kaggle 30h/tuần không đủ train | Ưu tiên track 2; cân nhắc Colab Pro / cloud nếu thiếu |
| Hết thời gian | Ưu tiên QMOS + EMOS, bỏ phần optional |
| Baseline khó reproduce | Liên hệ BTC sớm, hỏi cộng đồng |

---

## Câu hỏi mở
- [ ] Dùng public dataset nào để augment **ngoài** ESD + DailyTalk (đã là nền của data Track 2)? (IEMOCAP, MSP-Podcast...?)
- [ ] Một paper chung hay tách theo track?
- [ ] Mentor có co-author không?

> Lưu ý format đầu ra: hệ thống phải sinh `answer.txt` (`wav,QMOS,EMOS,CAT,VAL,ARO,DOM`) — chi tiết `08_track2_spec.md`.
