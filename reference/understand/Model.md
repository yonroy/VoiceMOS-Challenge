## Mỗi cột leaderboard ← model nào (baseline hiện tại)


| Track | Cột leaderboard | Model/Phương pháp đang dùng | Điểm hiện có |
|---|---|---|---|
| 🟦 1 | ACR UTT-SRCC | URGENT-MOS (1 model ra cả 2 cột) | 0.662 |
| 🟦 1 | CCR UTT-SRCC | ↳ cũng từ URGENT-MOS | 0.411 |
| 🟥 2 | QMOS UTT-SRCC | UTMOS / SpeechMOS | 0.414 |
| 🟥 2 | EMOS UTT-SRCC | Gemini (LLM-as-judge) | 0.194 (một phần) |
| 🟥 2 | CAT error | emotion2vec | 0.193 ⬇️|
| 🟥 2 | Valence UTT-SRCC | Gemini (1 model ra cả 3 cột VAD) | chưa chạy đủ|
| 🟥 2 | Arousal UTT-SRCC | ↳ cũng từ Gemini | chưa chạy đủ|
| 🟥 2 | Dominance UTT-SRCC | ↳ cũng từ Gemini | chưa chạy đủ|
| 🟩 3 | Speaker UTT-SRCC | ECAPA-TDNN (1 model ra cả 2 cột) | 0.451|
| 🟩 3 | Accent UTT-SRCC | ↳ cũng từ ECAPA-TDNN | 0.440|



Tôi hiểu UTT và SRCC như này

- số lẻ trên một audio là từ nhiều người chấm
- chấm theo xếp thứ hạng của model không phân biệt  giá trị chỉ phân biệt thứ hạng trong audio 
- utt-srcc chấm thứ hạng trên từng câu 
- Thang đo SRCC: −1 → 1, và 0 = đoán bừa. Gần 1 là xếp giống người, 0 là chẳng liên quan, âm là xếp ngược. Vậy nên baseline EMOS 0.194 nghĩa là "chỉ nhỉnh hơn đoán bừa một chút" — và 0.81 của exp08 là bước nhảy thực sự.

- Riêng cột CAT chấm bằng ERR, ngược chiều — sai số phân bố cảm xúc, càng THẤP càng tốt (0.133 của exp08 là tốt). 9 cột còn lại đều SRCC càng cao càng tốt. Đừng nhầm chiều khi đọc leaderboard.

- SRCC chỉ so sánh được trên cùng một bộ audio (bạn đã hỏi ở câu A/B). Hệ quả thực dụng: điểm DEV và điểm EVAL sắp tới là 2 bộ khác nhau — điểm eval có thể lệch khỏi 0.63/0.81 hiện tại, đó là bình thường.

- Loss khi train ≠ metric khi chấm. Mình train bằng MSE (tối ưu giá trị) nhưng bị chấm bằng SRCC (tối ưu thứ tự) — lệch mục tiêu này là lý do exp13/exp15 thêm ranking loss để tối ưu thẳng vào thứ hạng.
viết lại này và lưu vào 