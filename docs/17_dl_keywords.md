# 17 — Từ điển keyword Deep Learning (cho người mới)

> Mỗi keyword: **định nghĩa ngắn → công thức → sơ đồ/trực giác → ví dụ trong dự án**.
> Đọc dần; ⭐ = thứ dùng trực tiếp trong code Track 2. Cập nhật ngày: 5/6/2026.

---

## PHẦN A — Khối nền tảng

### A1. Neuron / Linear layer (Fully-Connected) ⭐
- **Là gì:** phép biến đổi tuyến tính: nhân ma trận trọng số + cộng bias.
- **Công thức:** `y = W·x + b` (W = trọng số học được, b = bias).
- **Ví dụ:** `nn.Linear(1024, 512)` trong trunk exp08: vector 1024 chiều → 512 chiều.

### A2. Activation (hàm kích hoạt) ⭐
- **Là gì:** hàm **phi tuyến** chèn giữa các Linear → cho mạng học quan hệ phức tạp (không có nó, chồng bao nhiêu Linear cũng chỉ = 1 Linear).
- **Các hàm hay dùng:**
  | Hàm | Công thức | Dùng khi |
  |---|---|---|
  | **ReLU** | `max(0, x)` | mặc định lớp ẩn |
  | **GELU** | `x·Φ(x)` (mượt hơn ReLU) | trong Transformer |
  | **Sigmoid** | `1/(1+e^-x)` → (0,1) | xác suất nhị phân |
  | **Tanh** | `(e^x-e^-x)/(e^x+e^-x)` → (-1,1) | head audeering |
  | **Softmax** | `e^xi / Σe^xj` | xác suất nhiều lớp (CAT) |
- **Sơ đồ:** `Linear → ReLU → Dropout → Linear` (đúng cấu trúc head trong code).

### A3. Loss function (hàm mất mát) ⭐
- **Là gì:** đo model **sai bao nhiêu** so nhãn → con số để tối thiểu hóa.
- **Công thức:**
  - **MSE** (hồi quy, vd EMOS/VAD): `L = (1/N)·Σ(ŷ - y)²`
  - **Cross-Entropy** (phân loại): `L = -Σ y·log(ŷ)`
  - **Soft Cross-Entropy** ⭐ (CAT — nhãn là *phân phối* vote): `L = -Σ p_target·log(softmax(logits))`
- **Ví dụ:** exp08 dùng MSE cho EMOS/VAD, soft-CE cho CAT (mục 6 code).

### A4. Gradient Descent + Backpropagation ⭐
- **Là gì:** cách model **học** — tính đạo hàm của loss theo từng trọng số (backprop) rồi **đi ngược chiều dốc** để giảm loss.
- **Công thức cập nhật:** `W ← W - lr · ∂L/∂W`
- **Sơ đồ:**
  ```
  forward:  x → model → ŷ → loss
  backward: loss → ∂L/∂W (lan ngược) → cập nhật W
  ```
- **Ví dụ:** `loss.backward()` (tính gradient) + `opt.step()` (cập nhật) trong vòng train exp08.

### A5. Learning rate (LR) ⭐
- **Là gì:** **độ lớn mỗi bước** đi xuống dốc. Lớn quá → nhảy loạn/phân kỳ; nhỏ quá → học chậm.
- **Ví dụ:** exp08 dùng **2 LR**: backbone `1e-5` (fine-tune nhẹ), head `1e-3` (train mạnh).

### A6. Optimizer ⭐
- **Là gì:** thuật toán quyết định *cách* cập nhật W từ gradient.
- **Hay dùng:** **SGD** (cơ bản) · **Adam/AdamW** ⭐ (tự điều chỉnh LR mỗi tham số + momentum — mặc định hiện nay).
- **Ví dụ:** `torch.optim.AdamW(...)` trong exp08.

### A7. Epoch / Batch / Iteration ⭐
- **Batch:** một nhóm mẫu xử lý cùng lúc (exp08: `BATCH=4`).
- **Iteration (step):** 1 lần cập nhật W (1 batch).
- **Epoch:** đi hết **toàn bộ** tập train 1 lượt (exp08: `EPOCHS=12`).
- **Gradient accumulation** ⭐: cộng dồn gradient nhiều batch nhỏ rồi mới cập nhật → giả lập batch lớn (exp08: `ACCUM=8` → batch hiệu dụng 32). Dùng khi VRAM ít.

---

## PHẦN B — Chống overfit & ổn định

### B1. Overfitting ⭐
- **Là gì:** model **học vẹt** tập train (thuộc lòng) → kém trên dữ liệu mới. Dấu hiệu: train tốt, val tệ.
- **Trong dự án:** lý do exp08 chỉ mở băng 6 lớp + ít epoch (data chỉ 12.7k).

### B2. Dropout ⭐
- **Là gì:** **tắt ngẫu nhiên** một tỉ lệ neuron khi train → buộc mạng không phụ thuộc 1 neuron → bớt overfit.
- **Công thức:** mỗi neuron giữ lại với xác suất `1-p`.
- **Ví dụ:** `nn.Dropout(0.3)` (tắt 30%) trong head exp08.

### B3. Weight decay (L2) ⭐
- **Là gì:** phạt trọng số to → giữ W nhỏ gọn → bớt overfit.
- **Công thức:** thêm `λ·‖W‖²` vào loss.
- **Ví dụ:** `weight_decay=1e-5` trong AdamW exp08.

### B4. Early stopping ⭐
- **Là gì:** **dừng train** khi điểm validation không cải thiện sau `patience` epoch; giữ lại bản tốt nhất.
- **Ví dụ:** exp08 `PATIENCE=3`, lưu `best_state` theo val SRCC.

### B5. Normalization ⭐
- **BatchNorm / LayerNorm:** chuẩn hóa giá trị giữa các lớp → train ổn định, nhanh hội tụ. (LayerNorm có trong mỗi lớp Transformer.)
- **Z-score (chuẩn hóa nhãn)** ⭐: `z = (x - μ)/σ` → đưa nhãn về cùng thang để các MSE so được. exp08 z-score EMOS/VAD (mục 5).

---

## PHẦN C — Kiến trúc

### C1. CNN (Convolutional Neural Network) ⭐
- **Là gì:** quét **bộ lọc (kernel)** trượt trên dữ liệu → bắt **mẫu cục bộ**, chia sẻ trọng số.
- **Sơ đồ:** `input → [conv → activation → pool] × N → đặc trưng`
- **Ví dụ:** **feature-extractor của WavLM** là 7 lớp CNN (waveform thô → khung 20ms).

### C2. RNN / LSTM / GRU
- **Là gì:** xử lý chuỗi tuần tự, mang "trí nhớ" qua từng bước. **LSTM/GRU** có cổng chống quên.
- **Hạn chế:** chậm (tuần tự), khó nhớ xa → **đã bị Transformer thay thế** ở hầu hết task.

### C3. Attention / Self-attention ⭐
- **Là gì:** mỗi phần tử **"chú ý" có trọng số** tới mọi phần tử khác → nắm ngữ cảnh toàn chuỗi song song.
- **Công thức:** `Attention(Q,K,V) = softmax(Q·Kᵀ / √d)·V`
  - **Q** (query): "tôi đang tìm gì", **K** (key): "tôi chứa gì", **V** (value): "thông tin của tôi".
- **Sơ đồ:**
  ```
  mỗi khung → tạo Q,K,V
  điểm khớp = Q·Kᵀ → softmax → trọng số → gộp V → đầu ra giàu ngữ cảnh
  ```
- **Ví dụ:** trái tim mỗi lớp Transformer của WavLM — cho mỗi khung âm "nhìn" cả câu.

### C4. Transformer block ⭐
- **Là gì:** khối lặp = **Self-Attention + MLP**, kèm LayerNorm + residual.
- **Sơ đồ 1 lớp:**
  ```
  x → [LayerNorm → Self-Attention → +x] → [LayerNorm → MLP → +x] → ra
            (residual)                          (residual)
  ```
- **Ví dụ:** WavLM-large = **24 lớp** Transformer chồng lên nhau (hidden 1024).

### C5. Residual / Skip connection ⭐
- **Là gì:** cộng thẳng đầu vào vào đầu ra (`y = f(x) + x`) → giúp mạng **rất sâu** vẫn train được (gradient không tắt).
- **Ví dụ:** trong mỗi Transformer block; ResNet là nơi ý tưởng này nổi tiếng.

### C6. Embedding ⭐
- **Là gì:** **vector số** đại diện cho 1 input/đối tượng; gần nhau = giống nhau về nghĩa.
- **Ví dụ:** WavLM biến 1 wav → **embedding 1024 chiều** (sau pooling) — đầu vào trunk exp08.

### C7. Encoder / Decoder
- **Encoder:** input → biểu diễn (hiểu). **Decoder:** biểu diễn → output (sinh).
- **Ví dụ:** WavLM/BERT = encoder; GPT = decoder; Whisper = encoder-decoder.

### C8. Pooling ⭐
- **Là gì:** gộp chuỗi `[T khung × D]` → **1 vector D** đại diện cả câu.
- **Kiểu:** **mean pooling** (trung bình) ⭐, max, **attentive pooling** (trung bình có trọng số học được).
- **Ví dụ:** `masked_mean` trong exp08 = mean pooling bỏ phần pad.

---

## PHẦN D — Cách huấn luyện

### D1. SSL (Self-Supervised Learning) ⭐
- **Là gì:** học từ data **không nhãn** bằng cách tự tạo bài tập (che → đoán lại).
- **Ví dụ:** WavLM/wav2vec2/emotion2vec đều pretrain bằng SSL. Xem [16_model_architectures.md](16_model_architectures.md).

### D2. Pretrain → Fine-tune ⭐
- **Pretrain:** học nền tổng quát (SSL, data khổng lồ, từ trọng số ngẫu nhiên).
- **Fine-tune:** chỉnh model pretrain cho task cụ thể (data nhỏ có nhãn, LR nhỏ).
- **Ví dụ:** WavLM pretrain (Microsoft) → SAILER fine-tune cảm xúc → **exp08 fine-tune tiếp trên VoiceMOS**.

### D3. Logits → Softmax → Probability ⭐
- **Logits:** điểm thô (chưa chuẩn hóa) model xuất ra mỗi lớp.
- **Softmax:** biến logits → xác suất (cộng = 1).
- **Ví dụ:** CAT head xuất 5 logits → softmax → tỉ lệ 5 cảm xúc.

### D4. Multi-task learning (MTL) ⭐
- **Là gì:** 1 model học **nhiều task cùng lúc** (EMOS+CAT+VAD) → chia sẻ biểu diễn → mạnh hơn học rời.
- **Khó:** cân các loss khác thang.

### D5. Uncertainty weighting ⭐
- **Là gì:** tự học trọng số mỗi loss thay vì chỉnh tay (mỗi task 1 tham số `log σ²`).
- **Công thức:** `L = Σ [ exp(-log σ²ₜ)·Lₜ + log σ²ₜ ]`
- **Ví dụ:** `log_var` trong exp08 (mục 6). Nguồn: Kendall 2018, arXiv:1705.07115.

---

## PHẦN E — Đánh giá

### E1. SRCC (Spearman Rank Correlation) ⭐
- **Là gì:** đo **tương quan thứ hạng** giữa dự đoán và nhãn (xếp hàng đúng thứ tự là được, không cần đúng giá trị). Metric chính của challenge.
- **Khoảng:** -1 → 1 (càng gần 1 càng tốt). Chi tiết: [14_leaderboard_metrics.md](14_leaderboard_metrics.md).

### E2. Train / Validation / Test ⭐
- **Train:** dạy model. **Validation:** chỉnh siêu tham số + early-stop (không train trên đó). **Test/Eval:** chấm cuối, model chưa từng thấy.
- **Ví dụ:** exp08 tách 10% train làm validation nội bộ (early-stop), DEV của challenge = test.

---

## Lộ trình đọc gợi ý
A4 (backprop) → A2/A3 (activation/loss) → B (overfit) → C3/C4 (attention/Transformer) → D1/D2 (SSL/fine-tune).
→ Nắm **Transformer (C3,C4)** + **Pretrain/Fine-tune (D2)** là đủ hiểu 90% việc đang làm.

> Liên kết: kiến trúc model cụ thể → [16_model_architectures.md](16_model_architectures.md) · metric → [14_leaderboard_metrics.md](14_leaderboard_metrics.md) · lộ trình học → [10_learning_roadmap.md](10_learning_roadmap.md).
