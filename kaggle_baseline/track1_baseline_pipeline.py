# %% [markdown]
# # VMC2026 Track 1 — Baseline Pipeline (Kaggle)
#
# Baseline = **URGENT-MOS** (checkpoint pre-trained tự tải từ HuggingFace).
# Repo có sẵn script xuất đúng format nộp → gần như không phải code gì.
#
# **Không bị chặn bởi license:** dev data Track 1 công khai trên HuggingFace
# (`urgent-challenge/vmc2026-track1-dev`, configs `acr` + `ccr`).
#
# **Cách dùng:** Notebook → GPU T4 + Internet On → chạy lần lượt các cell.
# Output: `predictions_dev.csv` (`sample_id,pred_score`) → zip → nộp Track 1.

# %% [markdown]
# ## 1. Cài đặt URGENT-MOS

# %%
# !git clone -q https://github.com/vvwangvv/URGENT-MOS.git /kaggle/working/URGENT-MOS
# !pip install -q -e /kaggle/working/URGENT-MOS

# %% [markdown]
# ## 2. Smoke test (vài mẫu) — kiểm tra môi trường + tải checkpoint
# Checkpoint `urgent-challenge/urgent-mos-f1c1m5dcorpus` tự tải từ HF lần chạy đầu.

# %%
# !cd /kaggle/working/URGENT-MOS && python scripts/infer_vmc2026_track1.py \
#     --split dev --limit 10 --output /kaggle/working/predictions_smoke.csv
# import pandas as pd; pd.read_csv("/kaggle/working/predictions_smoke.csv").head()

# %% [markdown]
# ## 3. Inference đầy đủ trên dev set (ACR + CCR)
# Script tự tải dataset từ HF, chạy cả ACR (1008) + CCR (2520) → 1 file predictions.

# %%
# !cd /kaggle/working/URGENT-MOS && python scripts/infer_vmc2026_track1.py \
#     --split dev --output /kaggle/working/predictions_dev.csv
# Nếu OOM: thêm --batch-frames <N> để giảm bộ nhớ.

# %% [markdown]
# ## 4. Validate + đóng zip nộp

# %%
import pandas as pd

PRED = "/kaggle/working/predictions_dev.csv"
df = pd.read_csv(PRED)
assert list(df.columns) == ["sample_id", "pred_score"], f"Header sai: {df.columns.tolist()}"

acr = df[df["sample_id"].str.contains("-acr_")]
ccr = df[df["sample_id"].str.contains("-ccr_")]
print(f"Tổng {len(df)} dòng | ACR {len(acr)} | CCR {len(ccr)}")
print("ACR range:", acr["pred_score"].min(), "→", acr["pred_score"].max(), "(cần [1,5])")
print("CCR range:", ccr["pred_score"].min(), "→", ccr["pred_score"].max(), "(cần [-3,+3])")
# Kỳ vọng dev: ACR=1008, CCR=2520

# %%
# !cd /kaggle/working && zip -j submission_track1.zip predictions_dev.csv && unzip -l submission_track1.zip

# %% [markdown]
# ## Ghi chú
# - Nộp: My Submissions → chọn **Track 1**, **bỏ chọn** track khác → upload `submission_track1.zip`.
# - File nộp Track 1 tên **`predictions.csv`** (KHÁC Track 2/3 dùng `answer.txt`). Script đã xuất đúng cột `sample_id,pred_score`.
# - Eval phase: đổi `--split test` (sau khi eval data ra 31/7).
# - GPU khuyến nghị; chỉ inference nên nhẹ, fit T4 16GB.
