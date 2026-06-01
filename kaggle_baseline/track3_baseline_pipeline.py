# %% [markdown]
# # VMC2026 Track 3 — Baseline Pipeline (Kaggle)
#
# Baseline = **ECAPA-TDNN** speaker/accent similarity. Code + **checkpoint pre-trained**
# đã có sẵn trong `vmc2026-baselines/track3/` (Baseline 2 fine-tuned).
#
# ⏳ **CẦN DATA Track 3** (đang chờ license). Khi có: trỏ `DATA_ROOT` rồi chạy.
#
# Điểm baseline tham khảo (dev): spk SRCC ~0.45, acc SRCC ~0.44 (Baseline 2 fine-tuned).
#
# Format nộp `answer.txt`: `system_id,utterance_id,wav_a_path,wav_b_path,pred_acc_sim,pred_spk_sim`.

# %% [markdown]
# ## 0. Cấu hình — SỬA khi có data

# %%
import os

DATA_ROOT = "/kaggle/input/<track3-data>"      # << thư mục giải nén có sets/ + wav/
DEV_CSV = f"{DATA_ROOT}/sets/dev.csv"
T3 = "/kaggle/working/vmc2026-baselines/track3"
OUT_DIR = "/kaggle/working"

# Checkpoint pre-trained có sẵn trong repo (Baseline 2):
CKPT_SPK = f"{T3}/official-egs/spk_sim_adamw_lr1e-3/model_spk_sim_step20000.pt"
CKPT_ACC = f"{T3}/official-egs/acc_sim_adamw_lr1e-3/model_acc_sim_step20000.pt"

# %% [markdown]
# ## 1. Cài đặt
# Repo gốc dùng `uv`. Trên Kaggle đơn giản hơn: cài speechbrain + chạy python trực tiếp.
# Nếu import lỗi, đối chiếu `track3/pyproject.toml` để bổ sung dep.

# %%
# !git clone -q https://github.com/voicemos-challenge/vmc2026-baselines.git /kaggle/working/vmc2026-baselines
# !pip install -q speechbrain torchaudio pandas

# %% [markdown]
# ## 2. Inference — speaker similarity + accent similarity
# Dùng checkpoint fine-tuned có sẵn (Baseline 2). Mỗi metric ra 1 csv.
# (Baseline 1 zero-shot: bỏ `--checkpoint` để dùng ECAPA gốc, không cần train.)

# %%
# !cd {T3} && python inference.py --data-root {DATA_ROOT} --csv-path {DEV_CSV} \
#     --checkpoint {CKPT_SPK} --out {OUT_DIR}/spk_dev.csv
# !cd {T3} && python inference.py --data-root {DATA_ROOT} --csv-path {DEV_CSV} \
#     --checkpoint {CKPT_ACC} --out {OUT_DIR}/acc_dev.csv

# %% [markdown]
# ## 3. Gộp spk + acc → answer.txt
# inference.py xuất csv theo từng metric; cần gộp 2 file theo cặp (system_id, utterance_id).
# ⚠️ Tên cột điểm trong output tùy script — kiểm tra rồi chỉnh `SPK_COL`/`ACC_COL`.

# %%
import pandas as pd

SPK_COL = "pred_spk_sim"   # << chỉnh theo cột thực trong spk_dev.csv
ACC_COL = "pred_acc_sim"   # << chỉnh theo cột thực trong acc_dev.csv
KEYS = ["system_id", "utterance_id", "wav_a_path", "wav_b_path"]


def build_answer():
    spk = pd.read_csv(f"{OUT_DIR}/spk_dev.csv")
    acc = pd.read_csv(f"{OUT_DIR}/acc_dev.csv")
    # đổi tên cột điểm về chuẩn nộp
    spk = spk.rename(columns={SPK_COL: "pred_spk_sim"})
    acc = acc.rename(columns={ACC_COL: "pred_acc_sim"})
    merged = spk.merge(acc[KEYS + ["pred_acc_sim"]], on=KEYS, how="outer")
    cols = KEYS + ["pred_acc_sim", "pred_spk_sim"]
    merged[cols].to_csv(f"{OUT_DIR}/answer.txt", index=False)
    print(f"Ghi {len(merged)} dòng → answer.txt")
    return merged[cols]


df = build_answer()
df.head()

# %% [markdown]
# ## 4. Đóng zip nộp

# %%
# !cd {OUT_DIR} && zip -j submission_track3.zip answer.txt && unzip -l submission_track3.zip

# %% [markdown]
# ## Ghi chú
# - Nộp: My Submissions → chọn **Track 3**, **bỏ chọn** track khác → upload zip.
# - Được nộp chỉ `pred_spk_sim` HOẶC chỉ `pred_acc_sim` HOẶC cả hai.
# - Repo có sẵn dev csv mẫu trong `official-egs/` để đối chiếu format output của inference.py.
# - Lưu ý dữ liệu: cần copy wav từ phân phối `_vctk` sang `_syn/wav/` (xem README track3 + `09_tracks_overview.md`).
