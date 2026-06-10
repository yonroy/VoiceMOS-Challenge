# %% [markdown]
# # VMC2026 Track 2 — Chuẩn bị data (gộp ESD + DailyTalk) trên Kaggle
#
# Gói Track 2 thiếu **1.417 mẫu giọng thật** (license tách ra):
# - **sys006** = ESD (1.379 file) · **sys001** = DailyTalk (38 file)
#
# Notebook này: cài **SoX** + build **sv56** → gom đúng utterance từ ESD/DailyTalk →
# **chuẩn hóa âm lượng** (giống mẫu TTS) → ráp vào `wav/` đủ **15.477 file**.
#
# ### Cách dùng
# 1. Settings → **Internet = On** (cần tải/biên dịch sv56). GPU không bắt buộc.
# 2. **+ Add Input** 3 dataset:
#    - Gói Track 2 (`vmc2026_track2_..._v3.tar.gz` — Kaggle tự giải nén ra `vmc2026-track2/`).
#    - ESD: `Emotional Speech Dataset (ESD).zip` (Kaggle tự giải nén ra `Emotion Speech Dataset/`).
#    - DailyTalk: `dailytalk.zip` (giải nén ra `dailytalk/data/...`).
# 3. **Run All**. Xong → **Save Version** (Commit) để lưu `wav/` ra output.
# 4. Từ output đó → **Create Dataset** → dùng làm input cho notebook train/baseline.
#
# > Notebook tự dò vị trí ESD/DailyTalk dù Kaggle giải nén ra thư mục tên gì.

# %% [markdown]
# ## 0. Tìm gói Track 2 + copy ra thư mục ghi được

# %%
import os, glob, shutil, subprocess

# Tự dò thư mục vmc2026-track2 trong mọi input đã add.
_cands = glob.glob("/kaggle/input/*/vmc2026-track2") + glob.glob("/kaggle/input/**/vmc2026-track2", recursive=True)
TRACK2_SRC = _cands[0] if _cands else None
assert TRACK2_SRC, "Không thấy thư mục vmc2026-track2 — đã Add Input gói Track 2 chưa?"

WORK = "/kaggle/working/vmc2026-track2"     # bản ghi được (input là read-only)
print("Track2 source :", TRACK2_SRC)

# Copy toàn bộ gói ra working (gồm wav/ 14060 file + scripts + csv). Mất vài phút.
if not os.path.exists(WORK):
    print("Đang copy gói Track 2 ra working (vài phút)...")
    shutil.copytree(TRACK2_SRC, WORK)
print("Số wav hiện có:", len(os.listdir(f"{WORK}/wav")))

# %% [markdown]
# ## 1. Cài SoX + build sv56 (cần Internet = On)
# sv56 = công cụ chuẩn hóa âm lượng của ITU-T, build từ source openitu/STL.

# %%
def sh(cmd):
    print("$", cmd)
    print(subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout[-2000:])

# SoX + trình biên dịch (để build sv56)
sh("apt-get -qq update && apt-get -qq install -y sox make gcc")
sh("which sox && sox --version")

# sv56demo
SV56_DIR = "/kaggle/working/STL-2009"
SV56_BIN_DIR = f"{SV56_DIR}/src/sv56"
if not os.path.exists(f"{SV56_BIN_DIR}/sv56demo"):
    sh("cd /kaggle/working && wget -q https://github.com/openitu/STL/archive/refs/tags/v2009.tar.gz")
    sh("cd /kaggle/working && tar -xf v2009.tar.gz")
    sh(f"cd {SV56_BIN_DIR} && make -f makefile.unx")
assert os.path.exists(f"{SV56_BIN_DIR}/sv56demo"), "Build sv56 thất bại — kiểm tra Internet=On + log make."

# Đưa cả sox và sv56demo vào PATH cho các script .sh dùng được
os.environ["PATH"] = SV56_BIN_DIR + ":" + os.environ["PATH"]
sh("which sv56demo")

# %% [markdown]
# ## 2. Dò vị trí ESD + DailyTalk (tự tìm dù tên thư mục khác nhau)

# %%
def find_root(rel_path):
    """Tìm thư mục ROOT trong /kaggle/input sao cho ROOT/rel_path tồn tại."""
    base = os.path.basename(rel_path)
    for hit in glob.glob(f"/kaggle/input/**/{base}", recursive=True):
        if hit.endswith(rel_path.replace("/", os.sep)) or hit.endswith(rel_path):
            return hit[: -len(rel_path)].rstrip("/")
    return None

# ESD: dòng CSV "0014/Angry/000381.wav" → file thật là "0014/Angry/0014_000381.wav"
_esd_first = open(f"{WORK}/ESD_utts_train_dev.csv").readline().strip().split(",")[0]
_p = _esd_first.split("/")                       # [spk, emo, uttID.wav]
ESD_REL = f"{_p[0]}/{_p[1]}/{_p[0]}_{_p[2]}"
ESD_ROOT = find_root(ESD_REL)

# DailyTalk: dòng CSV "1020/0_1_d1020.wav" → file thật ".../data/1020/0_1_d1020.wav"
_dt_first = open(f"{WORK}/DT_utts_train_dev.csv").readline().strip().split(",")[0]
DT_ROOT = find_root(_dt_first)                   # ROOT sao cho ROOT/1020/0_1_d1020.wav tồn tại

print("ESD_REL  :", ESD_REL)
print("ESD_ROOT :", ESD_ROOT)
print("DT_ROOT  :", DT_ROOT)
assert ESD_ROOT, "Không thấy ESD — đã Add Input 'Emotional Speech Dataset (ESD).zip' chưa?"
assert DT_ROOT, "Không thấy DailyTalk — đã Add Input 'dailytalk.zip' chưa?"

# %% [markdown]
# ## 3. Gom các utterance cần dùng → thư mục gathered/

# %%
GATHERED = f"{WORK}/gathered"
os.makedirs(GATHERED, exist_ok=True)

# ESD: copy ROOT/spk/emo/spk_uttID  →  gathered/<tên vmc2026>
n_esd = 0
for line in open(f"{WORK}/ESD_utts_train_dev.csv"):
    src_rel, dst = line.strip().split(",")
    p = src_rel.split("/")
    src = f"{ESD_ROOT}/{p[0]}/{p[1]}/{p[0]}_{p[2]}"
    if os.path.exists(src):
        shutil.copy(src, f"{GATHERED}/{dst}")
        n_esd += 1
    else:
        print("ESD thiếu:", src)

# DailyTalk: copy ROOT/parts[0]  →  gathered/<tên vmc2026>
n_dt = 0
for line in open(f"{WORK}/DT_utts_train_dev.csv"):
    src_rel, dst = line.strip().split(",")
    src = f"{DT_ROOT}/{src_rel}"
    if os.path.exists(src):
        shutil.copy(src, f"{GATHERED}/{dst}")
        n_dt += 1
    else:
        print("DailyTalk thiếu:", src)

print(f"Đã gom: ESD {n_esd}/1379 · DailyTalk {n_dt}/38 · tổng {len(os.listdir(GATHERED))}")

# %% [markdown]
# ## 4. Chuẩn hóa âm lượng bằng sv56 (mức -26 dB, giữ nguyên sample rate)

# %%
# Dùng chính script gốc trong gói: batch_normRMSE.sh → tạo *_norm.wav trong gathered/
sh(f"bash {WORK}/sv56scripts/batch_normRMSE.sh {GATHERED}")

# Move các file đã chuẩn hóa vào wav/ với đúng tên (bỏ hậu tố _norm)
moved = 0
for n in os.listdir(GATHERED):
    if n.endswith("_norm.wav"):
        final = "_".join(n.split("_")[:-1]) + ".wav"   # bỏ "_norm"
        shutil.move(f"{GATHERED}/{n}", f"{WORK}/wav/{final}")
        moved += 1
print("Đã chuẩn hóa & move vào wav/:", moved, "file")

# %% [markdown]
# ## 5. Kiểm tra đủ 15.477 + dọn rác

# %%
total = len(glob.glob(f"{WORK}/wav/*.wav"))
print("Tổng wav trong wav/:", total)
if total == 15477:
    shutil.rmtree(GATHERED, ignore_errors=True)
    # Dọn artifact build để dataset lưu ra GỌN (chỉ giữ vmc2026-track2/)
    shutil.rmtree(SV56_DIR, ignore_errors=True)
    for f in glob.glob("/kaggle/working/v2009.tar.gz"):
        os.remove(f)
    print("✅ ĐỦ 15.477 file. Sẵn sàng Save Version → tạo Dataset.")
    print("   Dùng dataset này cho notebook baseline: DATA_ROOT = '/kaggle/input/<dataset-mới>/vmc2026-track2'")
else:
    print(f"⚠️ Chưa đủ (đang {total}). Kiểm tra log bước 3-4 xem ESD/DailyTalk có thiếu file nào.")

# %% [markdown]
# ## Ghi chú
# - Output nặng (~2-3GB do 15.477 wav). `wav/` đã gồm cả train+dev nên dùng được cho cả fine-tune lẫn inference.
# - sv56 chuẩn hóa để mẫu ESD/DailyTalk cùng mức âm lượng với mẫu TTS → tránh model bị nhiễu bởi độ to.
# - Nếu Internet Off: SoX có thể có sẵn nhưng KHÔNG build được sv56 → bắt buộc bật Internet.
