# 05 — Setup Môi trường

Hướng dẫn dựng môi trường để reproduce được toàn bộ project.

---

## ⚠️ Việc cần làm NGAY (Tuần 1)
- [ ] Đăng ký challenge: https://forms.gle/L6YdkUf1PJdSSwLU7 (dùng email tổ chức)
- [ ] Xác nhận đã nhận email hướng dẫn từ BTC
- [ ] **Giải quyết vấn đề GPU** (xem mục bên dưới)
- [ ] Clone baseline repo
- [ ] Tải training data từ CodaBench

---

## Vấn đề GPU (chưa có — cần giải quyết)

### Các phương án theo thứ tự ưu tiên
1. **GPU lab/trường** — hỏi mentor trước tiên (tốt nhất, miễn phí, mạnh)
2. **Kaggle Notebooks** — 30h GPU/tuần miễn phí (T4/P100)
3. **Google Colab Pro** — ~10 USD/tháng, GPU ổn định hơn free
4. **Cloud credit nghiên cứu** — AWS/GCP/Azure thường có chương trình credit cho academic
5. **Vast.ai / RunPod** — thuê GPU theo giờ, rẻ

> Ước lượng: fine-tune SSL model cho MOS cần ít nhất GPU 16GB VRAM (T4 trở lên).

---

## Setup môi trường Python

```bash
# Tạo môi trường conda
conda create -n vmc2026 python=3.10 -y
conda activate vmc2026

# Clone baseline
git clone https://github.com/voicemos-challenge/vmc2026-baselines.git
cd vmc2026-baselines

# Cài dependencies (theo README của repo)
pip install -r requirements.txt
```

> Cập nhật chính xác các bước sau khi đọc README của baseline repo.

---

## Thư viện thường dùng
| Thư viện | Mục đích |
|---|---|
| torch / torchaudio | Deep learning, xử lý audio |
| transformers (HuggingFace) | SSL models (WavLM, HuBERT, Wav2Vec2) |
| s3prl | SSL speech toolkit |
| scipy / numpy | Tính SRCC, xử lý số |
| pandas | Xử lý kết quả, file submission |
| librosa | Trích xuất acoustic features |
| funasr / emotion2vec | Emotion embeddings |

---

## Cấu trúc thư mục dự kiến
```
vmc2026/
├── data/                 # Dataset (không commit lên git)
│   ├── train/
│   └── eval/
├── baselines/            # Clone từ repo BTC
├── src/                  # Code của mình
│   ├── models/
│   ├── train.py
│   └── inference.py
├── checkpoints/          # Model weights
├── results/              # File submission
├── docs/                 # Các file .md này
└── README.md
```

---

## Cách tính metric (UTT-SRCC)
```python
from scipy.stats import spearmanr

# pred: điểm model dự đoán, true: điểm người chấm
srcc, _ = spearmanr(pred, true)
print(f"UTT-SRCC: {srcc:.4f}")
```

---

## Format file submission
> Cập nhật chính xác theo hướng dẫn CodaBench sau khi nhận được
- Thường là CSV: `filename, predicted_qmos, predicted_emos`
- Kiểm tra kỹ format trước khi nộp (sai format = lỗi submission)

---

## Checklist trước khi nộp (7/8)
- [ ] Đã chạy inference trên toàn bộ eval set
- [ ] File submission đúng format
- [ ] Đã test submit thử trên CodaBench (nếu cho phép)
- [ ] Backup code + model weights
- [ ] Nộp trước deadline (không để phút chót)
