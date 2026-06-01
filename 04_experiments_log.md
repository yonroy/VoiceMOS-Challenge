# 04 — Nhật ký Thí nghiệm

Ghi lại MỌI thí nghiệm. Đây là tài liệu quan trọng nhất khi viết paper — ablation study và bảng kết quả đều lấy từ đây.

> Quy tắc: mỗi experiment ghi đủ **config → kết quả → nhận xét**. Không bao giờ chạy mà không ghi.

---

## Bảng tổng hợp kết quả

| Exp ID | Mô tả | Backbone | QMOS SRCC | EMOS SRCC | Note |
|---|---|---|---|---|---|
| baseline | UTMOS gốc | wav2vec2 | — | — | reproduce |
| exp01 | | | | | |
| exp02 | | | | | |
| exp03 | | | | | |

> SRCC = Spearman's Rank Correlation Coefficient (càng cao càng tốt). Metric chính của challenge: UTT-SRCC.

---

## Chi tiết từng thí nghiệm

### baseline — Reproduce UTMOS
- **Ngày:** [ ]
- **Mục tiêu:** Reproduce điểm baseline để có mốc so sánh
- **Config:**
  - Model: UTMOS
  - Data: [ ]
  - Hyperparameters: [ ]
- **Kết quả:**
  - QMOS UTT-SRCC: [ ]
  - EMOS UTT-SRCC: [ ]
- **Nhận xét:** [ ]

---

### exp01 — [Tên thí nghiệm]
- **Ngày:** [ ]
- **Giả thuyết:** [Mình kỳ vọng cải tiến này giúp gì?]
- **Thay đổi so với baseline:** [ ]
- **Config:**
  - Backbone: [ ]
  - Learning rate: [ ]
  - Batch size: [ ]
  - Epochs: [ ]
  - Loss: [ ]
- **Kết quả:**
  - QMOS UTT-SRCC: [ ]
  - EMOS UTT-SRCC: [ ]
- **Nhận xét:** [Có đúng giả thuyết không? Tại sao?]
- **Bước tiếp theo:** [ ]

---

### exp02 — [Tên]
- **Ngày:** [ ]
- **Giả thuyết:** [ ]
- **Thay đổi:** [ ]
- **Config:** [ ]
- **Kết quả:** [ ]
- **Nhận xét:** [ ]

---

> Copy block "exp" ở trên cho mỗi thí nghiệm mới.

---

## Ablation study (cho paper)
> Tổng hợp các thí nghiệm chứng minh từng thành phần đóng góp ra sao

| Cấu hình | QMOS SRCC | EMOS SRCC | Δ |
|---|---|---|---|
| Full model | | | — |
| − component A | | | |
| − component B | | | |

---

## Lỗi & bài học
| Ngày | Lỗi gặp phải | Cách khắc phục |
|---|---|---|
| | | |
