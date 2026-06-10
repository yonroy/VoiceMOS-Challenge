# 02 — Câu hỏi & trao đổi với Mentor

Ghi lại mọi câu hỏi đặt cho mentor và câu trả lời, để theo dõi các quyết định quan trọng.

---

## 🟠 Câu hỏi gửi mentor (8/6/2026 — Phiên 11)

### Kinh nghiệm fine-tune với data nhỏ
**Câu hỏi:**
> Với data nhỏ (~12k mẫu), thầy/cô thường mở băng (unfreeze) mấy lớp trên của WavLM, chọn learning rate backbone thế nào, và có mẹo gì để tránh overfit / thu hẹp khoảng lệch dev→eval ạ? Hiện em mở 6 lớp trên (LR backbone 1e-5, head 1e-3); fine-tune giúp cảm xúc nhưng làm rớt QMOS nên em đang phải "trộn cột".

**Trả lời mentor:**
> *(chờ trả lời)*

### Novelty cho ICASSP 2027
**Câu hỏi:**
> Em định theo novelty "first systematic study of EMOS prediction" + phát hiện model SER cũ (emotion2vec) vượt SOTA do metric chấm theo ranking. Góc này có đứng được không ạ? Và em xin trao đổi về việc đồng tác giả paper.

**Trả lời mentor:**
> *(chờ trả lời)*

---

## 🔴 Ưu tiên cao — cần hỏi NGAY

### Về scope (3 track vs 1 track)
**Câu hỏi:**
> Em đã xem yêu cầu của cả 3 track. Mỗi track có domain rất khác nhau (enhancement / emotion / accent). Nếu làm song song cả 3 một mình thì em lo không đủ sâu để ra paper tốt. Thầy/cô nghĩ chiến lược nên thế nào?

**Trả lời mentor:**
> Tập trung vào track 2, 2 track còn lai kéo source code làm demo UI

---

### Về GPU / compute
**Câu hỏi:**
> Em hiện chưa có GPU. Lab/trường mình có resource hỗ trợ không ạ? Nếu không thì em cần tính phương án (Kaggle, cloud credit) và có thể phải giảm scope.

**Trả lời mentor:**
> Data đơn giản không cần GPU xịn, dùng kaggle

---

## 🟡 Về định hướng nghiên cứu

### Hướng novelty
**Câu hỏi:**
> Theo thầy/cô, hướng nào cho Track 2 có tiềm năng novelty cao nhất hiện tại?

**Trả lời:**
> [Điền]

### Điểm yếu baseline
**Câu hỏi:**
> Baseline UTMOS và Gemini LLM-as-judge còn điểm yếu gì mình có thể khai thác?

**Trả lời:**
> [Điền]

### Cách tiếp cận EMOS hiện tại (bổ sung 3/6)
**Câu hỏi:**
> Em đang train một MLP head dự đoán EMOS trên đặc trưng emotion2vec **đóng băng** (không fine-tune backbone), feed thêm one-hot cảm xúc target. Hướng này có đủ "mới" để viết paper không, hay em nên **fine-tune cả backbone** / đổi sang **WavLM/wav2vec2** để mạnh hơn?

**Trả lời:**
> [Điền]

### Focus QMOS hay EMOS
**Câu hỏi:**
> Em nên tập trung dự đoán QMOS (chất lượng), EMOS (độ giống cảm xúc), hay cả hai để có contribution rõ ràng hơn?

**Trả lời:**
> [Điền]

---

## 🟢 Về dataset & kỹ thuật

### Public dataset bổ sung
**Câu hỏi:**
> Ngoài dataset của challenge, thầy/cô gợi ý dùng thêm public emotional speech dataset nào (IEMOCAP, ESD, MSP-Podcast)?

**Trả lời:**
> [Điền]

### Backbone model
**Câu hỏi:**
> Em nên dùng backbone nào — WavLM, HuBERT hay Wav2Vec2 — cho bài toán emotion MOS?

**Trả lời:**
> [Điền]

### Multi-task learning
**Câu hỏi:**
> Multi-task (QMOS + EMOS cùng lúc) có đáng thử không hay quá phức tạp?

**Trả lời:**
> [Điền]

---

## 📝 Về paper

### Contribution tối thiểu
**Câu hỏi:**
> Contribution tối thiểu cần có để submit ICASSP 2027 được chấp nhận là gì?

**Trả lời:**
> [Điền]

### Thời điểm viết
**Câu hỏi:**
> Em nên bắt đầu viết paper từ lúc nào trong quá trình làm?

**Trả lời:**
> [Điền]

### Co-author
**Câu hỏi:**
> Thầy/cô có muốn co-author không ạ?

**Trả lời:**
> [Điền]

---

## ⏱️ Về tiến độ
**Câu hỏi:**
> Thầy/cô muốn em báo cáo tiến độ theo tuần hay theo milestone?

**Trả lời:**
> [Điền]

---

## 📋 Quyết định đã chốt
> Tóm tắt các quyết định quan trọng sau mỗi buổi gặp
- [ ] Scope: ...
- [ ] GPU: ...
- [ ] Hướng nghiên cứu: ...
- [ ] Co-author: ...
