### Hành trình đầy đủ của 1 audio qua exp08 — từng layer, từng kích thước
Lấy 1 batch B = 4 audio, mỗi cái cắt 8 giây @ 16kHz. Tensor vào: [4, 128000] (4 audio × 128.000 mẫu sóng).

#### TẦNG 1 — Backbone WavLM-large (nhánh chính, warm-start SAILER)
##### 1a. CNN Feature Extractor — 7 lớp Conv1D ❄ đóng băng

Lớp	Việc làm	Sau lớp này
Conv 1 (kernel 10, stride 5)	quét sóng thô, bắt mẫu hình ~0.6ms	bắt đầu nén thời gian
Conv 2–5 (kernel 3, stride 2) ×4	mỗi lớp nén đôi, ghép mẫu hình nhỏ thành to	nén dần
Conv 6–7 (kernel 2, stride 2) ×2	nén tiếp	tổng nén 320×
→ [4, 128000] thành [4, 399, 512] — mỗi audio giờ là 399 khung (frame), mỗi khung 1 vector 512-D đại diện ~20ms âm thanh. Mỗi conv kèm GELU + chuẩn hóa. Vai trò: đổi "sóng vật lý" thành "chuỗi viên gạch âm học".

#### 1b. Feature Projection — Linear 512→1024 + LayerNorm → [4, 399, 1024] (nắn về đúng chiều Transformer).

1c. 24 lớp Transformer encoder — 18 dưới ❄, 6 trên 🔥 (LR 1e-5)

Mỗi lớp giống hệt nhau về cấu trúc, lặp 24 lần:

Khối con	Công thức	Vai trò
Multi-head self-attention (16 đầu)	softmax(QKᵀ/√64)·V	mỗi khung "nhìn" mọi khung khác — khung 50 biết khung 300 đang lên giọng
Residual + LayerNorm	x + attn(x) rồi chuẩn hóa	giữ thông tin cũ, ổn định train
FFN	Linear 1024→4096 → GELU → Linear 4096→1024	"tiêu hóa" thông tin vừa gom, từng khung riêng
Residual + LayerNorm	như trên	—
Shape không đổi suốt 24 lớp: [4, 399, 1024]. Càng lên cao càng trừu tượng: lớp dưới mã hóa âm học (cao độ, formant — phổ quát, nên đóng băng), lớp trên mã hóa ngữ nghĩa/cảm xúc (đặc thù domain, nên mở 6 lớp cuối cho học).

#### 1d. Mean-pool thời gian — e_w = trung bình theo trục 399 khung (có attention-mask để bỏ phần pad) → [4, 1024]. Cả câu nén thành 1 vector.

TẦNG 2 — Backbone audeering (nhánh phụ, ❄ hoàn toàn, chạy trước & cache .npz)
wav2vec2-large-robust fine-tune trên MSP-Podcast (cấu trúc giống 1a–1d): wav → pooled emb [4, 1024] + head VAD gốc của nó ra 3 số (scale 1+4x về thang 1–5) → ghép thành [4, 1027]. Vai trò: "chuyên gia VAD độc lập" — góc nhìn thứ hai, vì frozen nên trích 1 lần lưu aud_*.npz, train không tốn thêm GPU.

TẦNG 3 — Concat 2 tai
[4, 1024] | [4, 1027] → [4, 2051]. Chỉ là nối đuôi, không có tham số.

TẦNG 4 — TRUNK (não chung) ✅ LR 1e-3

Linear 2051→512  →  ReLU  →  Dropout 0.3      [4, 2051] → [4, 512]
Linear  512→512  →  ReLU  →  Dropout 0.3      [4, 512]  → [4, 512]
Linear 1: nén 4× — ép trộn thông tin 2 backbone thành biểu diễn chung (~1 triệu tham số, lớp tự-train to nhất).
ReLU: bẻ phi tuyến (không có nó 2 Linear sập thành 1).
Dropout 0.3: mỗi bước train tắt ngẫu nhiên 30% neuron — chống học vẹt 12.7k mẫu.
Vector 512-D ra khỏi đây là thứ cả 3 head dùng chung — nơi multi-task chia sẻ kiến thức.
TẦNG 5 — Ba head ✅ LR 1e-3
EMOS head	CAT head	VAD head
Đầu vào	[trunk 512 | one-hot target 5] = 517	trunk 512	trunk 512
Lớp 1	Linear 517→128 → ReLU → Dropout	Linear 512→128 → ReLU → Dropout	Linear 512→128 → ReLU → Dropout
Lớp 2	Linear 128→1	Linear 128→5	Linear 128→3
Hậu xử lý	×σ + μ (giải z-score) → 1–5	softmax → phân bố 5 cảm xúc	×σ + μ → V/A/D 1–5
Shape ra	[4, 1]	[4, 5]	[4, 3]
Câu nó trả lời	"khớp cảm xúc được yêu cầu chưa?"	"người nghe sẽ vote ra sao?"	"tọa độ cảm xúc ở đâu?"
TẦNG 6 — Loss (chỉ lúc train)
EMOS: MSE trên nhãn z-scored · CAT: soft cross-entropy với phân bố vote · VAD: MSE từng trục (z-scored).
5 loss (emos, cat, val, aro, dom) cân tự động bằng uncertainty weighting: L = Σ exp(−sₜ)·Lₜ + sₜ với 5 tham số log_var học được.
backward(): gradient từ 3 head hội tụ về trunk, chảy tiếp lên 6 lớp WavLM trên cùng rồi dừng (18 lớp dưới + CNN + audeering không nhận gradient).

#### Toàn cảnh một dòng shape

[4,128000] ─CNN×7─► [4,399,512] ─proj─► [4,399,1024] ─Transformer×24─► [4,399,1024]
─mean-pool─► [4,1024] ─concat aud 1027─► [4,2051] ─trunk─► [4,512]
            ├─ +one-hot → EMOS [4,1]   (×σ+μ)
            ├─────────── CAT  [4,5]   (softmax)
            └─────────── VAD  [4,3]   (×σ+μ)


Tỉ lệ tham số đáng nhớ: WavLM ~315M (trong đó chỉ ~75M ở 6 lớp mở được train) · audeering ~315M (0 train) · trunk + 3 head ~1.3M. Phần bạn thực sự "dạy từ đầu" chỉ là cái đuôi 1.3M — toàn bộ sức mạnh còn lại là mượn và tinh chỉnh.