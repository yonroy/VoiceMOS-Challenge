# 🍱 Thực đơn lắp ráp model — "đề bài có tính chất gì → bốc linh kiện nào"

> Cách dùng: khi xây model mới, dò từng dòng; đề bài khớp dòng nào thì bốc linh kiện dòng đó, ghép lại là ra kiến trúc.
> Cột "Đã gặp ở" trỏ về ví dụ thật trong dự án VoiceMOS 2026 — xem chi tiết tại `docs/04_experiments_log.md` (mục exp_mix) + `docs/22_slides_v2_paper_style.md`.
> Tạo ngày 11/6/2026 (buổi học Phiên 22).

---

## Nguyên tắc gốc: kiến trúc là HỆ QUẢ, không phải khởi đầu

```
① NHIỆM VỤ (vào gì → ra gì? metric nào?)
② DỮ LIỆU CÓ GÌ (nhãn loại nào, bao nhiêu?)
③ TÀI NGUYÊN (GPU gì, bao nhiêu giờ?)
        ↓
④ KIẾN TRÚC — "tự rơi ra" từ 3 câu trên
```

5 câu hỏi checklist trước khi xây:
1. **Vào gì, ra gì?** → quyết định encoder + hình dạng head (số neuron cuối).
2. **Nhãn ở đâu, bao nhiêu, dạng gì?** → scratch / pretrain / fine-tune + loss.
3. **Bài toán cần "giác quan" gì?** → tra thực đơn dưới đây (inductive bias).
4. **Metric chấm là gì?** → loss có khớp metric không (MSE ≠ SRCC!).
5. **GPU chịu được bao nhiêu?** → kích thước, freeze, ACCUM, cache.

---

## 🍱 NHÓM 1 — "Giác quan" đầu vào (encoder)

| Đề bài có tính chất... | → Bốc linh kiện | Đã gặp ở |
|---|---|---|
| Mẫu hình **cục bộ** lặp lại (vài chục ms, vài pixel) | **Conv** (CNN/TDNN) | CNN ×7 của WavLM |
| Cần ngữ cảnh cục bộ **rộng dần mà rẻ** | Conv **giãn nở** (dilated): `h[t]=Σw[k]·x[t+d·k]` | TDNN ECAPA (Track 3) |
| Nghĩa của một mảnh **phụ thuộc toàn cục** (xì = /s/ hay nhiễu?) | **Self-attention / Transformer**: `softmax(QKᵀ/√d)·V` | 24 lớp WavLM |
| **Trật tự & động lực thời gian** quan trọng (trồi sụt, diễn tiến) | **Mamba/SSM, RNN** | exp15 thắng ARO 0.7978 |
| Dữ liệu ít → cần "tai" có sẵn | **Pretrained backbone** (frozen/fine-tune) | mọi exp Track 2 |
| Một nguồn không đủ góc nhìn | **Fusion nhiều encoder** khác xuất thân (khác "bài tập thời học nghề") | exp07 (e2v+SAILER); URGENT-MOS ×4 |

## 🍱 NHÓM 2 — Khớp nối giữa thân

| Đề bài... | → Linh kiện | Đã gặp ở |
|---|---|---|
| Chuỗi dài **tùy ý** → cần 1 vector cố định | **Pooling**: mean (đơn giản) / attention-pool (khung quan trọng nói to) / **[mean‖std]** (giữ "độ rung") / Mamba (giữ trật tự) | mean exp08 · stat-pool ECAPA · Mamba exp15 |
| Không biết **tầng nào** chứa thông tin cần | **Trộn lớp** `H=Σαₗ·H⁽ˡ⁾` (αₗ học được, softmax) | URGENT-MOS Track 1 |
| Kênh nào quan trọng **tùy từng input** | **SE-block / gating** (squeeze→excite→scale) | ECAPA Track 3 |
| Mạng sâu khó train | **Residual + LayerNorm**: `x + f(x)` | mọi Transformer |
| **Nhiều task liên quan** cùng học | **Trunk chung + head riêng** (multi-task, chia sẻ kiến thức qua trunk) | exp04→exp15 |
| Đầu ra phụ thuộc một **điều kiện cho trước** (target) | **Concat điều kiện** (one-hot) vào head | EMOS head [512\|5]=517 |
| **So sánh 2 input** | **Siamese** (chung trọng số — "cùng một cây thước") + cosine (zero-shot) / interaction `[a;b;\|a−b\|;a⊙b]` + MLP (có nhãn) / hiệu 2 nhánh `g(A)−g(B)` (điểm so sánh, phản đối xứng) | Track 3 ECAPA · NCPM Track 1 |
| Đã có sẵn một dự đoán khá tốt | **Neo/residual feature** (nối nó vào head, chỉ học chỉnh sửa quanh nó) | UTMOS trong QMOS head exp07 (513) |

## 🍱 NHÓM 3 — Cửa ra (head + hậu xử lý)

| Đáp án có dạng... | → Linh kiện | Đã gặp ở |
|---|---|---|
| 1 số liên tục | Linear → 1 (hồi quy) | QMOS/EMOS head |
| k số độc lập | Linear → k (k neuron = k bộ trọng số riêng đọc chung 1 đầu vào) | VAD head (3) |
| **Phân bố xác suất** | Linear → k + **softmax** `eᶻⁱ/Σeᶻʲ` | CAT head (5) |
| Bị **chặn trong khoảng** [a,b] | **Tanh×scale+shift** (vd `Tanh×2+3`→[1,5]) hoặc train trên **z-score rồi ×σ+μ** | Track 3 head · VAD/EMOS exp08 |
| 1 nhãn trong k lớp | softmax + argmax | CAT khi cần nhãn cứng (⚠️ neutral-bias) |

## 🍱 NHÓM 4 — Cách dạy (loss & chiến lược train)

| Tình huống... | → Linh kiện | Đã gặp ở |
|---|---|---|
| Metric là giá trị (MSE/MAE) | **MSE/L1 loss** | exp13 (MSE thuần → QMOS 0.6296) |
| Metric là **thứ hạng** (SRCC) | + **pairwise ranking loss** `relu(−sign·diff)` — cần ≥2 mẫu "sống" trong 1 lần backward (batch to, hoặc mẹo cửa sổ ACCUM của exp13) | exp13 (λ=0, có sẵn) · exp15 (λ=0.3) |
| Nhãn là phân bố | **soft cross-entropy** `−Σ p·log softmax(z)` | CAT exp08 |
| **Nhiều loss** lệch thang/nhiễu | **uncertainty weighting** `L=Σexp(−sₜ)Lₜ+sₜ` (sₜ=log σₜ² học được) | exp08 (5 tham số) |
| Nhãn ít + backbone to | **freeze phần dưới, fine-tune phần trên, LR 2 tốc độ** (1e-5 / 1e-3) | exp08 (6/24 lớp) |
| Nhãn lệch thang/phân bố | **z-score nhãn** `z=(y−μ)/σ` — ⚠️ lưu μ/σ vào checkpoint! | VAD/EMOS exp08 |
| VRAM nhỏ, cần batch to | **gradient ACCUM** (gom gradient nhiều lượt mới step) + AMP + grad-checkpoint | exp08 (4×8) · exp13 (1×16) |
| Sợ overfit | Dropout, early-stop, val tách riêng | mọi exp |
| Sợ train hỏng mất trắng | **zero-shot làm sàn** (đo trước khi train, lưu ckpt sàn, chỉ ghi đè khi vượt) + Save Version ngay | exp13 |

---

## Ví dụ dùng thực đơn trong 60 giây

> Đề bài giả định: *"chấm độ giống cảm xúc giữa audio sinh ra và audio mẫu, thang 1–5, có 5k nhãn, chạy T4."*

Dò menu:
- so 2 input → **Siamese**
- cảm xúc + ít nhãn → **backbone SAILER frozen** (cache đặc trưng)
- có nhãn, cần hơn cosine → **interaction vector + MLP head**
- đáp án chặn [1,5] → **Tanh×2+3**
- metric SRCC → **MSE + ranking loss** (batch to vì head frozen rẻ)
- T4 → cache .npz, không cần ACCUM

→ Kiến trúc "tự lắp xong": `2 wav → SAILER ❄ (cache) → [a;b;|a−b|;a⊙b] → MLP → Tanh×2+3`, loss MSE+rank, zero-shot cosine làm sàn đối chứng.

---

## 3 bài học đính kèm (rút từ chính dự án)

1. **Loss phải khớp metric** — MSE tối ưu giá trị, SRCC chấm thứ hạng; CAT là ERR (giá trị) thì ranking vô nghĩa.
2. **Mỗi cột điểm một "khẩu vị"** — fine-tune kéo biểu diễn về 1 hướng, cột ngược hướng sẽ thiệt (exp08 giỏi cảm xúc, QMOS xẹp) → trộn cột/ensemble là cách rẻ nhất hợp nhất ưu điểm.
3. **Muốn biết model giỏi gì, xem nó từng làm "bài tập" gì** — WavLM (che-đoán giữa nhiễu) thính âm học; Kimi-Audio (nghe-rồi-viết) thính ngữ nghĩa; ECAPA (gọi tên 7k người) thính danh tính — chọn encoder = chọn xuất thân khớp đề bài.
