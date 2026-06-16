# 11 — Báo cáo tiến độ (gửi mentor)

> Nhật ký báo cáo tiến độ theo ngày. Báo cáo mới nhất ở trên cùng.

---

## Báo cáo ngày 15/6/2026 (Phiên 25) — CHẠY THẬT Triton serving lần đầu (CPU mode) trên server Linux + đưa được 2/3 → 3/3 track lên + viết lại guide deploy

**Người thực hiện:** Tran Minh Toan · **Nội dung:** GPU server đang đầy → dựng chế độ **CPU test** cho `triton_service/` và chạy thật end-to-end. Đây là lần đầu hệ Triton (dựng ở Phiên 24, mới pass syntax) **hoạt động thật**. Không chạy thí nghiệm, leaderboard không đổi.

### 1. ⚙️ CPU test mode (cấu hình + cách đảo lại GPU)
- 3 `config.pbtxt`: `KIND_GPU → KIND_CPU`; `docker-compose.yml`: comment khối GPU `deploy`, command thêm `--model-control-mode=explicit --load-model=...` (đầu chỉ Track 2, sau nạp đủ 3). Model code đã CPU-safe sẵn (`torch.cuda.is_available()`).
- Cổng gateway 8080 (server dùng chung đã chiếm) → đổi mặc định **18080** (`GATEWAY_PORT`). UI cổng đọc từ `GRADIO_SERVER_PORT`.
- Cache model để **named volume** (hf/torch/model) → tải UTMOS/WavLM/audeering 1 lần, restart không tải lại.

### 2. 🔧 Chuỗi ~11 lỗi đã xử lý (theo thứ tự gặp)
1. **transformers ≥4.50 cấm `torch.load`** (CVE-2025-32434, cần torch≥2.6) mà `microsoft/wavlm-large` chỉ có `.bin` → pin `transformers<4.50` (giữ torch 2.4.1 khớp libtorch Triton 24.08).
2. **`failed to execute bake`** (bug compose) sau khi build xong → `COMPOSE_BAKE=false` trong `run_server.sh`.
3. **`.pt` cụt do đứt mạng** (`PytorchStreamReader ... central directory`) → named volume + tải lại sạch.
4. **Port 8080 đã chiếm** → gateway 18080.
5. **Gradio 7860 đã chiếm** + code hard-code port → đọc `GRADIO_SERVER_PORT`.
6. **Track 3** `No module named speechbrain` → thêm `speechbrain` vào `model_requirements.txt` → **READY**.
7. **Track 1** torchcodec mới đòi `libnvrtc.so.13` (CUDA 13) → pin `torchcodec==0.1.0` (khớp torch 2.4).
8. **Track 1** torchcodec 0.1.0 lại **thiếu `AudioDecoder`** (urgent_mos import top-level) — kẹt version thật (AudioDecoder cần torchcodec≥0.4 → torch≥2.7). Vì track1 **tự decode bằng soundfile + truyền waveform** (không dùng decoder) → **chèn stub `AudioDecoder`** vào torchcodec trước khi import urgent_mos (chỉ restart, không build lại vì model.py mount volume).
9. (vận hành) `curl` thiếu `@` trước path → gửi chuỗi thay vì file.

### 3. ✅ Kết quả
- **Track 2 READY** trên CPU, `curl /track2` trả JSON 6 cột đúng. **Track 3 READY** (sau fix speechbrain). **Track 1** vừa vá stub (chờ log xác nhận READY).
- UI Gradio gọi gateway 18080, ≥2/3 tab chạy. Đo thử batch trên CPU: 100 file/workers=16 → 0.47 audio/s, latency ~31s (CHỈ chứng minh đúng, **không phải số hiệu năng** — batching chỉ lợi trên GPU).
- Bài học: workers (client, song song) ≠ batch (server, max_batch_size=8); workers phải ≥8 mới nuôi đầy batch.

### 4. 📚 Tài liệu
- Viết lại `docs/24_server_deploy_guide.md`: **Phần A** (CPU test Track 2) + **Phần B** (GPU đầy đủ + B0 đảo cấu hình) + bảng lỗi mở rộng (đủ các lỗi gặp trong phiên).

### 5. Việc tiếp theo
- 🔴 **Xác nhận Track 1 READY** sau stub; nếu `infer` còn chạm AudioDecoder thì patch sâu hơn.
- 🟠 **Đối chiếu điểm CPU vs `api_service`** (cùng ckpt phải khớp) → chứng minh port Triton đúng.
- 🟠 Khi **GPU rảnh**: đảo về GPU (B0), chạy **Locust** lấy số throughput + so batch 1 vs 8 thật (mentor yêu cầu) → số cho slide/paper.
- 🟡 Giải quyết triệt để torchcodec Track 1 (nâng torch 2.7 cho python backend? hay giữ stub).
- 🔒 (vẫn nợ) revoke token HF lộ; ablation ranking exp13; nộp bản trộn cột mới.

### Đã commit (origin/main)
`5f76065` CPU mode + guide · `b1ec65a` transformers<4.50 · `86030ca` COMPOSE_BAKE · `c17a7c4` named volume · `b449a2a` port 18080 · `0d00cca` UI port · `ed5240c` nạp 3 track · `684b30b` speechbrain+torchcodec · `2da27df` stub AudioDecoder.

---

## Báo cáo ngày 12/6/2026 (Phiên 24) — hoàn thiện Triton serving cho CẢ 3 TRACK (theo mentor) + gateway API + dynamic batching thật + Locust loadtest + UI 3 track + 2 tài liệu hướng dẫn

**Người thực hiện:** Tran Minh Toan · **Nội dung:** dựng trọn `triton_service/` theo đúng yêu cầu mentor (server Linux 1×GPU 12GB nội bộ): Triton serving 3 track, API "upload audio → trả score", xử lý dynamic batching cho audio khác độ dài, loadtest Locust. Không chạy thí nghiệm/đổi leaderboard — đây là phần **triển khai serving**.

### 1. 🏗️ Kiến trúc đích (mentor giao)
```
client --(POST multipart .wav)--> FastAPI gateway :8080 --(tritonclient)--> Triton :8000
                                                                              ├ track2_emotion (py backend, dyn-batch)
                                                                              ├ track1_acr
                                                                              └ track3_sim
        <--------------- JSON score (6 cột / ACR-CCR / spk-acc-cos) ---------
Triton metrics :8002 · docker-compose · mạng nội bộ
```
- Bỏ hướng Kaggle/PyTriton (xóa `triton_service/kaggle/`) — chuyển hẳn sang Docker trên server Linux.

### 2. ⚡ Dynamic batching THẬT (phần lõi mentor nhấn mạnh)
- Input audio để kiểu **TYPE_STRING shape `[1]`** → mỗi request đóng 1 blob bytes **cố định shape** bất kể audio dài/ngắn → Triton gom batch được. `config.pbtxt`: `max_batch_size: 8` + `dynamic_batching { max_queue_delay_microseconds: 5000 }`.
- `execute()` Track 2 viết lại theo batch thật: gom N request → decode N audio (độ dài khác nhau) → **pad tới audio dài nhất + dựng attention_mask** → **1 forward WavLM gộp** (masked-mean tôn trọng độ dài thật) → heads → tách kết quả. Audeering/QMOS vẫn loop (nhẹ hơn WavLM). Track 3 cũng pad cặp test+ref → 1 forward; Track 1 dùng `infer(list)` batch tự nhiên.

### 3. 📦 File đã tạo trong `triton_service/`
- **3 model Triton:** `model_repository/{track2_emotion,track1_acr,track3_sim}/{config.pbtxt,1/model.py}` (Track2 port từ exp08; Track1 từ URGENT-MOS; Track3 từ ECAPA-TDNN spk+acc).
- **Gateway:** `gateway/{app.py,Dockerfile,requirements.txt}` — FastAPI `/track1 /track2 /track3 /health`, nhận multipart → gọi Triton → JSON.
- **Loadtest:** `loadtest/{locustfile.py,requirements.txt}` — Locust HttpUser ramp user, đo RPS + latency cả 3 track; có kịch bản so `max_batch_size 1 vs 8`.
- **Hạ tầng:** `docker-compose.yml` (triton + gateway, GPU passthrough, healthcheck) + `run_server.sh` (Linux); giữ `run_local.ps1` tham khảo Windows.
- **UI:** viết lại `ui/app_ui.py` thành **4 tab** (3 track + batch đo throughput), gọi **gateway :8080** bằng `requests` (bỏ `tritonclient`); ô "Gateway URL" + nút kiểm tra `/health`.

### 4. 📚 2 tài liệu hướng dẫn (FILE MỚI trong `docs/`)
- `docs/23_triton_system_overview.md` — "những điểm cần hiểu để nắm hệ thống": luồng dữ liệu, vì sao 2 tầng gateway+Triton, model nạp khi nào (initialize 1 lần / execute mỗi request), dynamic batching, 3 model trong 12GB VRAM, Locust, bảng file.
- `docs/24_server_deploy_guide.md` — guide 9 bước chạy thật trên server: SSH → git pull → kiểm Docker/GPU → HF_TOKEN → **upload wav từ máy local bằng `scp`** (4A) → `bash run_server.sh` → curl test từng track → Locust 20 user → so batch 1 vs 8; kèm bảng xử lý lỗi.

### 5. 🧠 Buổi học (người mới) — đã giải thích trong session
- Dynamic batching = gom nhiều audio xử lý 1 lần thay vì tuần tự; pad + attention_mask để batch audio khác độ dài; **workers (client) ≠ batch (server)** — workers phải ≥ max_batch_size để "nuôi" batch.
- 3 cổng phân biệt: **7860** UI Gradio · **8089** Locust dashboard · **8080** gateway API (cả UI lẫn Locust gọi vào). Tab batch UI = "nếm thử", Locust = "đo chính thức" (mentor yêu cầu).
- Gateway vs Triton: gateway nói HTTP/multipart với người dùng, Triton chỉ nhận tensor → gateway là người dịch wav→tensor.

### 6. Việc tiếp theo
- 🟠 **Chạy thật trên server** (`/home/nhandt23/project/VoiceMOS`): build image, `bash run_server.sh`, curl test 3 track, đối chiếu điểm Track 2 với api_service cũ (phải khớp vì cùng ckpt exp08).
- 🟠 **Locust loadtest**: upload vài wav mẫu, ramp 20 user, ghi RPS/p95; so `max_batch_size 1 vs 8` lấy số cho slide/paper.
- 🟢 Verify VRAM 3 model < 12GB (`nvidia-smi`); nếu căng giảm `count` hoặc lazy-load.
- 🔒 (vẫn nợ từ phiên trước) revoke token HF lộ; ablation ranking exp13; nộp bản trộn cột mới.

---

## Báo cáo ngày 11/6/2026 (Phiên 23) — slide v2 paper-style 36 slide + kịch bản thuyết trình + buổi học metric/layer 3 track + phát hiện ranking loss trong code

**Người thực hiện:** Tran Minh Toan · **Nội dung:** hoàn thiện bộ tài liệu present (deck v2 + script + thực đơn thiết kế model); buổi học rất dài đào sâu cách chấm + từng layer cả 3 track; soi code phát hiện trạng thái thật của ranking loss exp13/exp15. Không chạy thí nghiệm mới, leaderboard không đổi.

### 1. 🎞️ Slide v2 paper-style (FILE MỚI `docs/22_slides_v2_paper_style.md` → `slide/voicemos2026_slides_v2.html`)
- **36 slide** theo mạch paper (Intro → Metrics → T1 → T3 → T2 đầy đủ → Conclusion), theme Apple-clean giữ từ v1, 5 hình SVG render OK (254KB).
- Nội dung MỚI so với v1: cách chấm có **ví dụ tính tay** (2 tầng điểm, SRCC bảng 5 audio, công thức CAT-ERR chính thức từ `reference/content_btc/track2.txt`); **bảng từng layer cả 3 track** (Vào→Ra→Train?→Vai trò, đúng số chiều từ code); slide "giải phẫu 3 head" (one-hot 517 / softmax / ×σ+μ); training details (uncertainty weighting, ACCUM, AMP); exp13 thành Method 3 riêng; ablation Mamba; **số liệu 10/6** (best-per-column QMOS 0.6296 · ARO 0.7978).
- v1 (`21_`) cũng được bổ sung 2 slide giải nghĩa metric + render lại HTML. Cập nhật tham chiếu trong `README.md` + `CLAUDE.md`.

### 2. 🎤 Kịch bản thuyết trình (FILE MỚI `slide/voicemos2026_v2_script.md`)
- Lời thoại từng slide cho cả 36 slide (NÓI/NHẤN/⏱), tổng ~33 phút, Track 2 chiếm 15 phút; kèm **5 câu Q&A dự phòng** + mẹo rút xuống 20 phút.

### 3. 🍱 "Thực đơn lắp ráp model" (FILE MỚI `slide/model_design_menu.md`)
- Bảng tra "tính chất đề bài → linh kiện kiến trúc" 4 nhóm (giác quan đầu vào / khớp nối / cửa ra / cách dạy), mỗi dòng kèm công thức + ví dụ thật trong dự án; checklist 5 câu hỏi trước khi xây model; ví dụ lắp ráp 60 giây.

### 4. 🔍 Phát hiện khi soi code (quan trọng cho ablation)
- **exp13 kỷ lục QMOS 0.6296 dùng MSE THUẦN** (`RANK_LAMBDA=0`) — ranking loss có sẵn (mẹo cửa sổ ACCUM=16 → 120 cặp/cửa sổ dù BATCH=1) nhưng TẮT; đính chính nhận định trước: cải thiện QMOS đến từ fine-tune đúng domain, KHÔNG phải ranking.
- **exp15 ĐANG BẬT `RANK_LAMBDA=0.3`** cho 4 cột SRCC (CAT giữ soft-CE — đúng vì CAT chấm ERR) nhưng ghép cặp theo batch, BATCH=2 → **chỉ 1 cặp/forward, tín hiệu yếu** (code tự ghi chú).
- **exp13 CHƯA có cơ chế resume** (luôn train từ UTMOS zero-shot). Chốt 2 phương án ablation ranking: **A** = chỉ đổi `RANK_LAMBDA=0.3` (so sạch MSE vs MSE+rank cùng xuất phát, cho paper); **B** = vá ~8 dòng resume từ ckpt 0.6296 + giảm LR (săn kỷ lục). Khuyên A trước; ranking mạnh nhất ở exp06/07 (BATCH=64 → 2.016 cặp, cache sẵn).

### 5. 📚 Buổi học (người mới) — cô đọng vào `03_`
- 2 tầng điểm (audio vs model) · SRCC tính tay + **phạt theo d², không đếm số câu sai** (cùng "đúng 3/5" có thể ra 0.9 / 0.6 / −0.6) · CAT-ERR = MAE bảng N×5 · ranking loss = đổi xếp-hạng-toàn-cục thành so-cặp · gradient ACCUM ("cầm lên ghi sổ thả xuống, chân đứng yên") · từng layer Track 1 (URGENT-MOS: trộn lớp αₗ, AMPM/NCPM hiệu 2 nhánh, thang CCR ±3 chuẩn ITU) + Track 3 (đọc code `model.py`: TDNN giãn nở, SE-block, stat-pool [μ‖σ], interaction vector, bẫy projection chưa-train) · CNN→Transformer ví dụ toán tính tay (contextual embedding: cùng vector [1,0] ra 2 nghĩa khác nhau theo ngữ cảnh) · WavLM masked prediction (che-đoán cụm k-means + cố tình phá nhiễu) · pretrain vs fine-tune vs scratch (SAILER = intermediate fine-tuning, đóng vai pretrained ckpt với mình) · zero-shot trong exp13 = đối chứng + sàn an toàn + khám pipeline.

### 6. Việc tiếp theo
- 🟢 **Export PDF/PPTX slide v2** gửi mentor (thêm `--allow-local-files`); present theo script.
- 🟠 **Ablation ranking exp13** (phương án A trước): đổi `RANK_LAMBDA=0.3`, nếu OOM giảm ACCUM 16→8 hoặc MAX_SEC 12→8; ghi config→kết quả→nhận xét vào `04_`.
- 🔴 (kế thừa P21) RESUME exp15 + **nộp bản trộn cột mới** (QMOS←exp13 + ARO←exp15 + còn lại←exp08).
- 🔒 (vẫn nợ) revoke token HF lộ; smoke test exp16.

---

## Báo cáo ngày 11/6/2026 (Phiên 22) — EMOS thật cho 100 audio tiếng Việt (qua API) + buổi học SRCC thực tế · calibration · CAT-ERR

**Người thực hiện:** Tran Minh Toan · **Nội dung:** hoàn thiện vòng đánh giá VoxCPM2 tiếng Việt — chấm lại 100 audio qua API HF Space để lấy **cột EMOS thật** (lần trước thiếu vì không truyền target); buổi học về ý nghĩa SRCC khi triển khai thực tế.

### 1. ⭐ Chấm lại 100 audio với `target_emotion=happy` → EMOS thật
- Phát hiện `100audio_emotion_scores.csv` (Phiên 20–21) **không có cột `emos`** — head EMOS cần one-hot target mà lần chấm trước không truyền. Dùng lại `score_100audio.py --target happy --out 100audio_emotion_scores_happy.csv` (file MỚI — nếu ghi vào file cũ, resume sẽ bỏ qua cả 100 file đã chấm).
- Chạy 100/100 OK qua `https://tranminhtoan140601-voicemos2026-api.hf.space` (~8–10s/file, nhanh hơn dự kiến vì Space đã nạp model). Gotcha Windows: `UnicodeEncodeError` cp1252 khi print tiếng Việt → fix `PYTHONIOENCODING=utf-8`.

### 2. 🔬 Kết quả EMOS (target happy, thang 1–5)
- **Dải 1.872 → 3.364, mean 2.745** — value lệch xuống do domain tiếng Việt (DEV tiếng Anh thường 4+), đúng pattern "chỉ tin thứ hạng".
- **Top 5 tốt nhất:** sample_010 (3.364) · sample_024 (3.358) · sample_085 (3.339) · sample_046 (3.190) · sample_091 (3.190). **Top 5 tệ nhất:** sample_083 (1.872) · sample_094 (2.088) · sample_069 (2.126) · sample_028 (2.174) · sample_008 (2.189).
- **Tốt nhất toàn diện = sample_024** (EMOS hạng 2 + QMOS hạng 7/100 — ngoại lệ hiếm phá trade-off "biểu cảm cao thì QMOS tụt"; top 5 QMOS còn lại toàn neutral/arousal thấp). Khớp cảm xúc nhất = sample_010.
- **SRCC(EMOS thật, proxy cat_happy) = 0.9496 · SRCC(EMOS, valence) = 0.9661** → proxy cat_happy là xấp xỉ rất tốt khi không truyền target; head hồi quy EMOS phân tách tốt/tệ rõ dù argmax vẫn neutral-bias (32/100 happy).

### 3. 🧠 Buổi học (ghi cô đọng vào `03_`)
- SRCC vs giá trị tuyệt đối khi triển khai thực tế: việc nào chỉ cần ranking (so model, reranking, regression test) vs việc nào cần value thật (báo MOS khách hàng, ngưỡng release); **calibration** (linear/isotonic/z-score) chữa thang đo mà không đổi SRCC.
- So 2 model TTS: trung bình **điểm** theo hệ thống (system-level, như SYS-SRCC của BTC), nên dùng cùng câu (paired, Wilcoxon); 10 mẫu/model là ít nếu điểm sát nhau.
- Công thức **CAT-ERR** = MAE trên phân bố vote 5 cảm xúc, dải [0, 0.4]; khác SRCC ở chỗ value tuyệt đối CÓ nghĩa → softmax "mềm" là lợi thế.

### 4. Việc tiếp theo (giữ nguyên từ Phiên 21)
- 🔴 RESUME train exp15 (`RESUME_LR_SCALE=0.5`, dừng sớm nếu 2–3 epoch không cải thiện) → predict + nộp.
- 🔴 Nộp bản trộn cột mới: QMOS←exp13 + ARO←exp15 + EMOS/CAT/VAL/DOM←exp08.
- 🔒 (vẫn nợ) revoke token HF lộ; smoke test exp16; (mới) nghe tai 3 file sample_024/010/099 để kiểm chứng ranking bộ chấm trên tiếng Việt.

---

## Báo cáo ngày 10/6/2026 (Phiên 21) — 🚀 2 KỶ LỤC CỘT/NGÀY: exp13 QMOS 0.6296 + exp15 Mamba ARO 0.7978 + vẽ kiến trúc + buổi học layer

**Người thực hiện:** Tran Minh Toan · **Nội dung:** ngày bứt phá leaderboard — 2 bản nộp DEV lập 2 kỷ lục cột; hoàn thiện tài liệu kiến trúc cho system description/paper; buổi học sâu kiến trúc từng layer.

### 1. 🏆 exp13 NỘP — phá trần QMOS sau 6 ngày đứng yên
- Fine-tune thẳng UTMOS (`utmos22_strong`) trên nhãn `qMOS` thật → **QMOS 0.548 → 0.6296** (+0.082). Xác nhận kép: ảnh leaderboard `benchmark/final.png` (0.63) + `scores.json` bản nộp exp15 (0.6296, cùng ckpt).
- Ý nghĩa: giả thuyết "UTMOS lệch domain giọng cảm xúc" đúng — fine-tune về domain thắng head frozen + neo (exp07 0.548) thắng zero-shot (0.414).

### 2. 🏆 exp15 NỘP (bản `exp15_predict` nạp 2 ckpt) — Mamba head có điểm thật
- Sửa `exp15_predict` nạp **2 checkpoint**: cảm xúc←`ft_mamba_emotion_full.pt` (exp15) + QMOS←`ft_qmos_utmos.pt` (exp13); ưu tiên QMOS: exp13 > exp07 answer > UTMOSv2; đường dẫn mặc định trỏ Kaggle Dataset `cache-exp8` (giờ chứa đủ: 2 ckpt mới ở gốc + ckpt exp08/exp11 + cache audeering trong `archive/`).
- Điểm DEV (`submissions/Track2/exp15_predict/scores/scores.json`): QMOS **0.6296** · EMOS 0.8070 · CAT 0.1349 · VAL 0.6545 · **ARO 0.7978 🏆 (kỷ lục, vượt exp08 0.7933)** · DOM 0.7506.
- **Kết luận ablation Mamba vs mean-pool: GẦN HÒA** — thua sát nút 3 cột, thắng đúng Arousal (cột biến thiên theo thời gian rõ nhất) → dòng ablation giá trị cho paper. Ckpt exp15 (8/6) hóa ra KHÔNG phải smoke-test.
- Best-per-column mới: QMOS 0.6296 (exp13) · EMOS 0.8116 (exp08b) · CAT 0.1331 (exp08) · VAL 0.6605 (exp08b) · **ARO 0.7978 (exp15)** · DOM 0.7539 (exp08b).

### 3. Vẽ kiến trúc từng layer hệ tốt nhất (cho system description + paper)
- Mục mới **exp_mix** trong `04_`: sơ đồ ASCII 2 nhánh (exp08: WavLM ft 6 lớp + audeering 1027-D → concat 2051 → trunk 512 → 3 head; exp07: e2v 1029 + SAILER 1036 + UTMOS → trunk → 4 head) + bảng vai trò từng layer đúng số chiều từ code.
- `12_system_description.md` Track 2: điền mục 1–4 (tổng quan exp_mix, HÌNH 2a–2d gồm cả nhánh B v2 = exp13, external resources + license phi thương mại, chiến lược training).

### 4. Buổi học (người mới) — ghi cô đọng vào `03_`
- Kiến trúc exp08 "2 tai → 1 não → 3 miệng" từng layer; vì sao head EMOS cần one-hot target.
- Mamba vs mean-pool (selective SSM, attention-pooling, O(n)) — và kết quả thật xác nhận "gần hòa, thắng ARO".
- z-score μ/σ lưu trong ckpt; `load_state_dict` thiếu/dư key; vì sao khâu load không ăn GPU (nghẽn mạng/disk; HF_TOKEN phải attach + set env trước `from_pretrained`).

### 5. Việc khác
- Nhận xét `100audio_emotion_scores.csv` (100/100): neutral-bias rõ (68 neutral/32 happy; khử neutral → 97 happy), QMOS thấp đồng loạt (lệch domain — chỉ dùng xếp hạng), VAD dải hẹp nhưng đúng hướng.
- README: nhúng ảnh leaderboard `benchmark/final.png` + mục slide (đổi tên `voicemos2026_final (1).html` → `voicemos2026_slides.html`, sửa link chết) + bảng điểm 4 chữ số.
- Git: 2 commit push (`967061d`, `41665df`); remote đổi sang repo mới `yonroy/VoiceMOS-Challenge` (repo được đổi tên).

### 6. Việc tiếp theo
- 🔴 **RESUME train exp15** (kế hoạch user): Add Input `cache-exp8` → tự dò ckpt → `RESUME_LR_SCALE=0.5`; dừng sớm nếu 2–3 epoch không cải thiện (bài học exp08b). Sau đó predict + nộp = kết quả cuối exp15.
- 🔴 **Nộp bản trộn cột thế hệ mới**: QMOS←exp13 + ARO←exp15 + EMOS/CAT/VAL/DOM←exp08 (0 giờ GPU, chốt hệ 6 cột mạnh nhất; khuyên làm TRƯỚC resume).
- 🔒 (vẫn nợ) revoke các token HF đã lộ (Phiên 16/19); smoke test exp16 (LLM-judge); VoxCPM2 Bước 7→8b (Phiên 20).

---

## Báo cáo ngày 10/6/2026 (Phiên 20) — client Kaggle gọi API 3 track + notebook VoxCPM2 "sinh emotion → chấm điểm" (vòng lặp emotional ruler) + sửa metric khử neutral-bias 🎯

**Người thực hiện:** Tran Minh Toan · **Nội dung:** dựng notebook đánh giá **TTS cảm xúc** bằng chính bộ chấm Track 2 (đúng góc "emotional ruler" của paper §1); phát hiện + xử lý **neutral-bias trên tiếng Việt** bằng metric ranking.

### 1. Client Kaggle gọi API 3 track (FILE MỚI)
- [kaggle_baseline/demo_api_client_kaggle.ipynb](kaggle_baseline/demo_api_client_kaggle.ipynb) + `demo_api_client_kaggle_pipeline.py`: gọi API HF Space cho cả Track 1/2/3 (urllib thuần, **resume**, xuất CSV; Track 2 xuất nháp `answer.txt`). Mục đích: chấm hàng loạt từ Kaggle (Internet ON) mà không cần GPU.
- ⚠️ Hạn chế: HF **free CPU chậm** (~vài chục giây/file) → không hợp chấm cả set eval 2.730 file qua đường này.

### 2. ⭐ Notebook VoxCPM2 sinh emotion + chấm điểm — `D:\VFS\Tuần 1\VoxCPM2\VoxCPM2_Emotion_Generate_and_Score.ipynb`
- Dựa theo `VoxCPM2_Vietnamese_Baseline_Eval.ipynb` (sinh giọng tiếng Việt theo style/emotion) nhưng **thay phần eval WER+SpeakerSim bằng chấm CẢM XÚC** (QMOS/EMOS/CAT/VAD).
- Tiến hóa trong phiên: Colab → **Kaggle GPU + chấm LOCAL** (nạp exp08 `ft_emotion_full_20epoch.pt` + UTMOS thẳng GPU, bê nguyên logic `api_service/app/tracks/track2.py`, resample 16k bằng librosa) → upload ref đổi từ `ipywidgets.FileUpload` sang **trỏ `/kaggle/input` + tự dò `.wav`**.
- Sinh 7 emotion (neutral/happy/sad/angry/surprised/calm/excited) với `cfg=2.8`, `timesteps=20`.

### 3. 🔬 Phát hiện quan trọng — neutral-bias trên tiếng Việt
- Chạy lần đầu: **mọi audio bị argmax → `neutral`** → control accuracy 20% (1/5). QMOS ~2.5, EMOS ~2.4 (thấp).
- **NHƯNG VAD đúng hướng:** arousal angry 3.71 / surprised 3.66 (cao) · sad 3.33 (thấp nhất); valence happy 3.20 (cao nhất). → scorer **THẬT SỰ cảm nhận được cảm xúc**, chỉ có đầu phân loại (argmax CAT) bị kéo về neutral. Khẳng định lại neutral-bias (Phiên 14) nặng thêm vì domain tiếng Anh (ESD/DailyTalk).

### 4. Sửa metric (theo lựa chọn): ranking VAD/EMOS + khử neutral-bias
- Thêm `perceived_nn` = argmax CAT **bỏ neutral**; `cat_target_rank` (scorer xếp target hạng mấy/5); in **2 accuracy** (argmax gốc vs khử neutral).
- Thêm **Bước 8b — SRCC theo VAD**: so VAD dự đoán với prototype kỳ vọng mỗi emotion (đúng tinh thần SRCC của challenge, miễn nhiễm neutral-bias tuyệt đối). Kỳ vọng Arousal/Valence SRCC > 0.

### 5. README viết lại + commit/push GitHub
- **Viết lại [README.md](README.md)**: thêm bảng điểm best-per-column 3 track, bảng docs đầy đủ 00→21, mục Demo UI (Gradio Space `voicemos2026-demo`) + API service + checkpoint HF, trỏ quy trình "đọc"/"xong". Thêm link Demo UI vào `07_`.
- **`.gitignore`**: chặn `cache/` (4.7GB), `*.pt`, `*.npz`, `100audio/`, `*.wav` (trước đó chưa loại → suýt commit nhầm vài GB).
- **Commit + push** `8790aff` lên `origin/main` (yonroy/VoiceMOS-Challenge-2026): **127 file** (đợt tái cấu trúc docs/ + kaggle_baseline/track* + api_service/ + demo client). Đã kiểm: **không lộ token** (toàn placeholder), file lớn nhất chỉ 125KB.

### 6. Việc tiếp theo
- 🟢 Chạy lại Bước 7→8b, gửi số mới: nếu **SRCC VAD > 0** → xác nhận hướng đo đúng; chốt phần cần tấn công tiếp = **TTS prosody**.
- 🟠 Sửa phía TTS: dùng **reference audio CÓ cảm xúc** (clone bám prosody ref → ref neutral làm output neutral), tăng cfg/timesteps, thử prompt tiếng Việt.
- 🟠 (tùy) thêm **emotion2vec** (đa ngôn ngữ) cho cột CAT tiếng Việt — giải gốc domain.
- 🔴 (vẫn nợ phiên trước) revoke token HF đã lộ; smoke test exp16 (LLM-judge) + exp13 (QMOS).

---

## Báo cáo ngày 10/6/2026 (Phiên 19) — slide Apple-clean + render HTML + DEPLOY API service 3 track lên HF Space 🚀

**Người thực hiện:** Tran Minh Toan · **Nội dung:** hoàn thiện slide (giải thích từng layer có toán cho cả 3 track + đổi giao diện Apple-clean + xuất HTML); **xây + deploy API service REST cho 3 track lên Hugging Face Space**; chạy chấm emotion cho 100 audio qua API.

### 1. Slide — thêm "giải thích từng layer có toán" cho cả 3 track (`docs/21_slides_3_tracks.md`)
- **Track 1:** hình kiến trúc URGENT-MOS MỚI (SVG) theo luồng forward có công thức từng tầng: CNN `h[t]=Σw[k]x[st+k]` → Transformer `softmax(QKᵀ/√d)·V` → trộn lớp `H=ΣαₗH⁽ˡ⁾` → mean-pool → fusion → 2 head AMPM(ACR)/NCPM(CCR).
- **Track 2:** thêm dòng "🧮 Toán từng tầng" cho cả 2 hình (Fusion C2 + Fine-tune C3): pooling/concat/trunk/head + loss uncertainty `L=Σ(1/2σₜ²)Lₜ+logσₜ`.
- **Track 3:** hình kiến trúc MỚI (lấy đúng từ code `track3/model.py`): ECAPA-TDNN **Siamese ❄** → L2-normalize `ê=e/‖e‖₂` → **cosine** `Σᵢ êₐ[i]ê_b[i]`; ghi rõ bản nộp = zero-shot cosine (spk=acc=cos) + hướng fine-tune (interaction vector → MLP).

### 2. Slide — redesign **Apple-clean** + render HTML
- Nhúng **CSS theme** đầu file: nền trắng, font SF/Helvetica, tiêu đề lớn–thanh (letter-spacing âm), bảng hairline, bullet chấm accent xanh `#0071e3`, blockquote thẻ bo góc `#f5f5f7`; title + divider Track 2 dùng layout `lead`.
- **Render HTML:** `npx @marp-team/marp-cli ... --html --no-stdin` → **`slide/voicemos2026_slides.html`** (4 hình SVG hiện đủ). Gotcha: marp-cli treo nếu thiếu `--no-stdin`.

### 3. ⭐ API SERVICE 3 TRACK — XÂY + DEPLOY HF SPACE (folder MỚI `api_service/`)
- **FastAPI REST (JSON)** đóng **Docker** → Hugging Face Space. Tái dùng nguyên code inference của `demo_all_tracks_gradio_pipeline.py`, **lazy-load** mỗi track. Track 2 **thêm cột QMOS** (UTMOS/SpeechMOS) → đủ 6 cột.
- Endpoint: `/track1` (ACR+CCR) · `/track2` (QMOS+EMOS+CAT+VAD) · `/track3` (spk+acc) · `/health` · `/docs`.
- File: `app/main.py`, `app/tracks/track{1,2,3}.py`, `Dockerfile`, `requirements.txt`, `README.md`, `push_to_hf_space.py`, `tests/smoke_test.py`.
- **ĐÃ PUSH + BUILD + RUNNING** trên **HF free CPU**: Space `tranminhtoan140601/voicemos2026-api`. Verified `/health` → `{"status":"ok"}`, `/docs` HTTP 200, `/openapi.json` đủ 3 route. **Predict thật OK** (sample_001 ~54s gồm tải model: happy, VAD 3.49/3.60/2.89, QMOS 1.63).
- Gotcha: marp/push in `→` lỗi cp1252 Windows → cần `PYTHONUTF8=1`; `python` trên máy lúc venv lúc global Python312.

### 4. Đính chính quan trọng (trung thực)
- Mình từng cảnh báo "Track 1 URGENT-MOS quá nặng, không fit free CPU" → **SAI**. Người dùng xác nhận demo hôm qua chạy **cả 3 track trên HF free CPU bình thường**. Checkpoint baseline `urgent-mos-f1c1m5dcorpus` thực tế **nhẹ, chạy được 16GB CPU**. Đã sửa cảnh báo trong `api_service/README.md`.

### 5. Chấm emotion 100 audio qua API (`score_100audio.py`)
- Script zero-dependency (urllib) + **resume**: gửi từng file `100audio/*.wav` → `/track2` → gom CSV `100audio_emotion_scores.csv` (qmos·perceived·cat5·vad3). Không có nhãn target → bỏ EMOS (chỉ CAT+VAD+QMOS); có `--target` để bật EMOS chung.
- **Đang chạy nền** (~43/100 lúc viết báo cáo). Free CPU ~5–20s/file.

### 6. Việc tiếp theo
- 🔒 **REVOKE token HF** `hf_FZbh…` (lộ trong chat phiên này) + token cũ Phiên 16 → tạo token mới.
- 🟢 Đợi 100audio chạy xong → đọc CSV (phân bố cảm xúc, QMOS cao/thấp nhất).
- 🟠 (tùy) thêm Track 1/Track 3 vào script batch; viết notebook Kaggle GPU nếu cần chấm nhanh số lượng lớn.
- 🟠 Export slide PDF/PPTX (bật HTML) gửi mentor.
- 🔴 (vẫn nợ) smoke test exp16 (LLM-judge) + exp13 (QMOS).

---

## Báo cáo ngày 10/6/2026 (Phiên 18) — làm slide present 3 track (mentor giao) + thêm hình kiến trúc SVG

**Người thực hiện:** Tran Minh Toan · **Nội dung:** hoàn thành nhiệm vụ mentor giao "làm slide thuyết trình cả 3 track"; deck Marp tiếng Việt theo mạch một bài báo rút gọn; thêm 3 hình kiến trúc vẽ trực tiếp bằng SVG inline.

### 1. Tạo deck slide (FILE MỚI) — `docs/21_slides_3_tracks.md`
- **Định dạng:** Marp markdown (mỗi `---` = 1 slide; export PDF/PPTX/HTML). **Ngôn ngữ:** tiếng Việt, giữ thuật ngữ kỹ thuật tiếng Anh. ~21 slide.
- **Cấu trúc như bài báo rút gọn:** Title → Mục lục → Bối cảnh (MOS + metric SRCC + so sánh 3 track) → **Track 1** (2 slide, baseline URGENT-MOS, 0.662/0.411) → **Track 3** (2 slide, baseline speaker-embedding, 0.451/0.440) → **Track 2 (trọng tâm)**: Động lực ("emotional ruler") → Bài toán 6 cột → Mục tiêu (overview) → Baseline & điểm yếu → Phát hiện C1 (e2v↔SAILER bổ sung) → Method 1 fusion (C2) → Method 2 fine-tune (C3) → Kết quả (bảng tiến hóa) → Phân tích/SRCC → Hướng mở rộng (Mamba/LLM-judge) → Timeline → Đóng góp C1–C3 → Liên kết/Q&A.
- Số liệu đối chiếu khớp `04_experiments_log.md` (best-per-column).

### 2. Thêm 3 hình kiến trúc (SVG inline trong chính file md — KHÔNG tạo file ảnh rời)
- **Overview:** 1 câu nói + cảm xúc target → [Hệ thống MOS cảm xúc] → 6 chip điểm (vàng = bắt buộc QMOS/EMOS · xanh = tùy chọn CAT/VAD).
- **Fusion (C2):** wav → 2 encoder ❄đóng băng (emotion2vec · SAILER/WavLM) → ⊕concat → TRUNK chung → 4 head (QMOS có neo UTMOS, EMOS có one-hot target, CAT, VAD).
- **Fine-tune (C3):** stack WavLM (6 lớp trên 🔥mở băng · lớp dưới ❄đóng băng), warm-start từ SAILER + audeering frozen → trunk → 3 head cảm xúc.
- Bảng màu thống nhất: xanh dương = frozen · cam = train · tím = trunk · xanh lá = head cảm xúc · vàng = QMOS.

### 3. Lưu ý kỹ thuật (đã ghi vào file)
- SVG inline = HTML thô → **Marp phải bật HTML** mới hiện hình: VS Code đặt `markdown.marp.enableHtml = true`; CLI thêm cờ `--html`. Hướng dẫn render ghi sẵn ở khối comment cuối file slide.
- Chưa chạy lệnh render (theo yêu cầu user "chỉ viết file md"); chưa export PDF/PPTX.

### 4. Việc tiếp theo
- 🟠 Mở Marp preview (bật enableHtml) kiểm 3 hình hiển thị ổn → export PDF/PPTX gửi mentor.
- 🔴 (vẫn nợ) Smoke test exp16 (LLM-judge) + exp13 (QMOS); 🔒 revoke token HF đã lộ (Phiên 16).

---

## Báo cáo ngày 9/6/2026 (Phiên 17) — xác nhận điểm thật exp_mix + bổ sung motivation cho paper §1

**Người thực hiện:** Tran Minh Toan · **Nội dung:** rà soát "dự án còn thiếu gì"; xác nhận điểm 6 cột của bản trộn cột từ file chấm; làm rõ mục đích/ứng dụng Track 2 và bổ sung vào Introduction của paper.

> ⚠️ **Lưu ý quy trình:** session này khởi đầu trên **snapshot docs cũ (trạng thái Phiên 13)** trong khi dự án thực tế đã ở Phiên 16. Ban đầu tôi tưởng "nộp exp_mix" và "upload checkpoint" là việc chưa làm → hóa ra đã xong từ Phiên 14 (nộp + Kaggle `cache-exp8`) và Phiên 16 (đẩy Hugging Face). Đã đối chiếu lại, không ghi đè nội dung Phiên 14/15/16.

### 1. Xác nhận điểm thật exp_mix (giá trị mới)
- Đọc `submissions/Track2/exp_mix_q07_emo08/scoring_result (2).zip` → điểm chính xác 6 cột: **QMOS 0.5480 · EMOS 0.8111 · CAT err 0.1331 · VAL 0.6590 · ARO 0.7933 · DOM 0.7509** = khớp best-per-column. Trước đây docs chỉ ghi "đã nộp" mà thiếu con số → nay đã có đủ trong `04_/12_/18_`.

### 2. Bổ sung motivation cho paper §1 (giá trị mới)
- Thêm 1 đoạn vào Introduction `19_`: predictor cảm xúc không chỉ để **đo** mà là **tín hiệu phản hồi** để xây TTS biểu cảm (so checkpoint, model selection, reward cho RLHF) — ẩn dụ "emotional ruler": muốn AI *sinh* cảm xúc thì trước hết phải *đo* được. Mở rộng danh sách ứng dụng (dubbing, customer service, companion/mental-health robots).
- Giải thích cho user (người mới) mục đích/ứng dụng Track 2 + vẽ **flowchart exp16** (Audio-LLM-as-Judge) bám đúng code.

### 3. Việc tiếp theo (gộp từ Phiên 14–16, chưa làm)
- 🔴 Smoke test **exp16** (novelty paper) + **exp13** (phá trần QMOS).
- 🟠 Điều tra **neutral-bias** (bảng CAT-top1 vs nhãn thật); exp14 ablation Mamba; viết exp17 data ngoài.
- 🔒 **REVOKE token HF** đã lộ (Phiên 16) + làm **slide 3 track** (mentor giao).

---

## Báo cáo ngày 9/6/2026 (Phiên 16) — đẩy TẤT CẢ lên Hugging Face + demo Gradio 3 track (UI clean+Plotly)

**Người thực hiện:** Tran Minh Toan · **Nội dung:** thực hiện 2 yêu cầu mentor (slide 3 track + đẩy hết lên HF); xây demo Gradio gộp 3 track + nâng UI ấn tượng.

### 1. Đẩy toàn bộ lên Hugging Face (tài khoản `tranminhtoan140601`) — XONG
3 repo (đều CC BY-NC-SA 4.0, KHÔNG kèm data thô):
- **Checkpoint** (Models) `voicemos2026-track2-emotion`: `ft_emotion_full_20epoch.pt` (1.27GB, exp08 cảm xúc), `ft_qmos_utmos.pt` (411MB, exp13 QMOS), `ft_joint_full.pt` (1.9GB, exp11) + model card.
- **UI demo** (Space Gradio) `voicemos2026-demo`: app 3 tab, tải ckpt từ Models repo.
- **Code pipeline** (Models) `voicemos2026-code`: toàn bộ `kaggle_baseline/` (55 file).
- Upload bằng `huggingface_hub` + `hf_transfer` (file đầu bị treo mạng → chuyển hf_transfer mới xong). Gotcha: `HF_TOKEN` env + `PYTHONUTF8=1` (Windows cp1252). ⚠️ **Token write đã lộ trong chat → cần REVOKE.**

### 2. Demo Gradio gộp 3 track (FILE MỚI)
- `kaggle_baseline/demo_all_tracks_gradio.{py,ipynb}`: 1 app 3 tab, **lazy-load** mỗi track (vừa RAM); Track 2 dùng **exp08** (EMOS/CAT/VAD, KHÔNG có QMOS — exp08 chuyên cảm xúc).
- `kaggle_baseline/demo_run_from_hf.{py,ipynb}` (FILE MỚI): **kéo app.py từ Space HF về chạy trên Kaggle GPU** → link `gradio.live`. Chiến lược chốt với user: **HF = chứa code/host; Kaggle = chạy GPU free**. Sau user đổi ý: Space cũng chạy luôn trên **HF free CPU** (chậm nhưng được).
- `app.py` tự nhận môi trường (`SPACE_ID`): Kaggle→`share=True`, Space→bind 7860.

### 3. Fix Space chạy được + nâng UI (clean light + Plotly)
- Sửa loạt lỗi gradio để Space RUNNING: `HfFolder` (hub 1.x bỏ → ghim <1.0 rồi **nâng gradio 4.44→6.17.3**), `bool is not iterable` schema, `show_api` không còn ở gradio 6, `launch` thiếu `server_name/port`.
- **UI mới (đã RUNNING):** theme Soft indigo + Inter, hero banner, **badge verdict màu**, **Plotly**: gauge cho ACR/CCR/EMOS/spk/acc, **bar** CAT, **radar** VAD; footer link 3 repo. Giữ nguyên lõi inference + lazy-load. `requirements.txt` thêm `plotly`.

### 4. Giải thích cho user (buổi học lồng trong phiên)
- Đọc output demo 1 file (sad ✅ đúng, không neutral-bias); bảng metric tab2 (val nội bộ > DEV = bẫy overfit; CAT-err nội bộ ≠ CodaBench); khái niệm **UTT** (chấm từng câu) vs **SRCC** (chấm thứ hạng); GPU trên HF (free=CPU; GPU=trả phí/ZeroGPU cần PRO).

### 5. Việc tiếp theo
- 🟠 **Làm slide present 3 track** (mentor giao, CHƯA làm).
- 🔒 **Revoke token HF** đã lộ.
- (tùy) thêm cột QMOS vào tab Track 2 demo (ghép exp07/exp13) nếu muốn demo đủ 6 cột.

---

## Báo cáo ngày 9/6/2026 (Phiên 15) — buổi học: đọc kết quả demo + hiểu metric UTT-SRCC

**Người thực hiện:** Tran Minh Toan · **Nội dung:** phiên giải thích (không chạy thí nghiệm) — đọc hiểu output demo cảm xúc exp08 + nắm chắc cách challenge chấm điểm. Có 1 điểm dữ liệu mới cho điều tra neutral-bias.

### 1. Đọc output demo trên 1 file (target = sad)
- Model chấm: **sad đứng đầu (CAT 39%) → KHỚP target sad** · EMOS 3.40/5 (khá) · VAD Valence **2.41** (thấp = tiêu cực, khớp "buồn") · Arousal 3.80 (hơi cao → giải thích angry lẫn 20%) · Dominance 3.24.
- 📌 **Mẫu sad này KHÔNG dính neutral-bias** (neutral chỉ 15%, sad top1) — khác hẳn mẫu surprised Phiên 14 (neutral 63%). → 1 điểm dữ liệu tốt cho bảng "CAT-top1 vs nhãn thật": **sad → đoán đúng**.

### 2. Đọc bảng metric tab 2 (val nội bộ vs mốc exp08 DEV)
- Val nội bộ (10% train.csv, seed 42): EMOS 0.8330 · VAL 0.7938 · ARO 0.8810 · DOM 0.8342 · CAT-err nội bộ 0.5066.
- **2 bẫy đã làm rõ:** (a) val nội bộ LUÔN cao hơn DEV (VAL nội bộ 0.79 vs DEV 0.66) vì cùng phân phối train → **không tin val nội bộ làm điểm thật**, chỉ dùng so tương đối; (b) **CAT-err nội bộ (0.5066) ≠ CAT-err CodaBench (0.133)** — khác công thức (L1 phân phối vs công thức BTC), không đặt cạnh nhau.
- 💡 Cách dùng đúng tab 2: train model mới → chạy lại → nếu 4 cột SRCC nội bộ **vượt 0.833/0.794/0.881/0.834** thì mới đáng nộp DEV kiểm chứng.

### 3. Hiểu metric UTT-SRCC (nền cho paper + đọc điểm)
- **UTT** = utterance-level = chấm **từng file** (2.730 điểm lẻ), khó/chi tiết hơn system-level (gộp TB theo hệ TTS).
- **SRCC** = chấm **THỨ HẠNG** cao–thấp, KHÔNG chấm giá trị tuyệt đối. Đúng thứ tự = điểm cao dù số sai. → Giải thích vì sao: VAD "nén" 2.5–3.6 vẫn điểm tốt; emotion2vec "dồn 2 cực" vẫn thắng SAILER; và vì sao có ý tưởng **ranking loss** (train MSE nhưng chấm SRCC → lệch → tối ưu thẳng thứ hạng).
- (Track 3 dùng **LCC** = Pearson, chấm cả giá trị — khác Track 2.)

### 4. Việc tiếp theo (không đổi)
- Tiếp tục điều tra **neutral-bias**: chạy demo thêm ~3 mẫu/cảm xúc đã biết nhãn → lập bảng CAT-top1 vs nhãn thật (đã có: sad ✅ đúng · surprised ❌ thành neutral).
- Viết **exp17** (data cảm xúc ngoài) cho CAT/VAD.

---

## Báo cáo ngày 9/6/2026 (Phiên 14) — demo cảm xúc (exp08), resume/predict exp15, exp14 ipynb, Related Work + Introduction, đổi focus sang cảm xúc

**Người thực hiện:** Tran Minh Toan · **Nội dung:** loạt việc code + viết paper; chuyển trọng tâm sang **5 cột cảm xúc**; phát hiện **neutral-bias** khi chạy demo thật.

### 1. Code pipeline
- **exp15 — thêm RESUME:** tự dò `ft_mamba_emotion_full*.pt` (khớp cả hậu tố `(2)`) trong /kaggle/input & working → có ckpt thì train tiếp (nạp backbone+Mamba+heads + chuẩn hóa TỪ ckpt, `best` init = điểm ckpt, `RESUME_LR_SCALE`); không có → train mới. Sửa `CACHE_INPUT`→`/kaggle/input/cache-exp8` + copy cache **đệ quy** (`**/aud_*.npz`, vì file nằm trong `archive/`).
- **exp15_predict** (FILE MỚI): predict-only — nạp ckpt → chấm DEV → answer.txt, KHÔNG train, không cần train.csv. Tự đọc `USE_MAMBA/Z_DIM/AUD_DIM` từ ckpt.
- **exp14 — tạo `.ipynb`** (trước chỉ có .py): "thêm Mamba head vào C2" = nhánh WavLM frame-level (đóng băng) → MambaEncoder → concat vào fusion exp07 6 cột; cờ `USE_MAMBA` ablation. Compile OK.
- **exp13 — thêm auto-dò `DATA_ROOT`** ở cell 0 (quét sets/train.csv) + đồng bộ chú thích sang `ft_emotion_full_20epoch.pt` (vốn đã trỏ đúng).

### 2. Demo Gradio cảm xúc (FILE MỚI) — `demo_track2_emotion_gradio`
- Chạy bằng **checkpoint exp08** (`ft_emotion_full_20epoch.pt`): chấm 5 cột cảm xúc (EMOS/CAT/VAD), KHÔNG cần API. 2 tab: (1) chấm 1 file TTS + verdict KHỚP/LỆCH target; (2) metric UTT-SRCC/CAT-err trên val nội bộ (10% train.csv, seed 42) so mốc exp08. UI thiết kế lại (2 cột, EMOS/VAD ô số riêng, header nêu "5 output = định nghĩa expressive emotion").
- **ĐÃ CHẠY THẬT trên Kaggle** → model nạp OK. **Phát hiện neutral-bias:** mẫu target=surprised → CAT cảm nhận neutral 63% (surprised 8%), EMOS 1.76/5, arousal 2.96 (thấp). 3 tín hiệu nhất quán nhưng nghi model thiên neutral / lớp surprised yếu. VAD bị "nén" 2.6–3.0 (đúng hiện tượng cũ) → đọc theo thứ hạng, không đọc trị tuyệt đối.

### 3. Paper
- **Related Work §2** — viết đầy đủ 7 đoạn (EN ở `19_`, mirror VI ở `15_`) + danh mục ref kèm arXiv ID (⚠️ cần kiểm lại ID). Định vị novelty: task EMOS mới + phát hiện e2v↔SAILER bổ sung + multi-task; KHÔNG claim "fusion là mới".
- **Introduction §1** — viết lại đoạn mở đầu theo mạch "ý nghĩa/động lực": TTS ở khắp nơi → biên giới = cảm xúc → nghẽn ở đánh giá → giám khảo tự động chưa biết chấm cảm xúc → Track 2.
- Làm rõ khái niệm cho mentor: **"5 output CHÍNH LÀ định nghĩa expressive emotion" của Track 2** — không cần chế metric riêng; metric chính = UTT-SRCC + CAT-err.

### 4. Đổi trọng tâm + tổ chức file
- **Chốt focus: 5 cột cảm xúc** (gác QMOS, vốn đã ổn 0.548). Chọn hướng tiếp theo = **thêm data cảm xúc ngoài** (ESD full / MSP-Podcast / IEMOCAP) → sẽ viết **exp17** phiên sau (data ngoài giúp CAT/VAD, KHÔNG cấp nhãn EMOS trực tiếp; lưu ý lệch miền + khai báo license).
- Chuyển `ft_qmos_utmos (1).pt` → **`cache/ft_qmos_utmos.pt`** (411 MB, ckpt QMOS fine-tune exp13).

### 5. Câu hỏi cho mentor (mới)
- "Expressiveness của TTS = EMOS hay tổ hợp EMOS+CAT+VAD? Cần 1 chỉ số tổng hợp hay báo riêng 5 cột là đủ? Verdict KHỚP/LỆCH (CAT-accuracy) có nên là metric phụ?"

### 6. Việc tiếp theo
- ✅ Bản trộn cột exp07+exp08 **đã nộp 9/6** (xem báo cáo Phiên trước / memory exp-mix) — KHÔNG còn nợ.
- Điều tra **neutral-bias**: test demo với ~3 mẫu/cảm xúc đã biết nhãn → lập bảng CAT-top1 vs nhãn thật; nếu xác nhận → class-weight/oversample lớp hiếm (vật liệu paper).
- Viết **exp17** (data cảm xúc ngoài) ở phiên mới.

---

## Báo cáo ngày 9/6/2026 (Phiên 13) — hoàn thiện exp13 (sửa lỗi ranking loss) + phân tích cải thiện QMOS

**Người thực hiện:** Tran Minh Toan · **Nội dung:** rà lại hướng cải thiện cột QMOS (yếu nhất, 0.548); **hoàn thiện exp13** bằng cách sửa lỗi tiềm ẩn của ranking loss; làm rõ "fine-tune ở dưới" trong exp14 thực chất là hướng exp15.

### 1. Phát hiện khi "đọc": checkpoint CÒN trên máy local
- Docs nhiều phiên ghi "backbone MẤT (kernel chết)", nhưng thực tế các file vẫn nằm ở `cache/`: **`ft_emotion_full_20epoch.pt`** (1.27 GB, bản cảm xúc TỐT NHẤT), `ft_emotion_full.pt`, và `ft_joint_full.pt` (1.9 GB, exp11) ở thư mục gốc. → Không cần train lại từ đầu; chỉ cần **upload lên Kaggle thành Dataset** (cho exp13 + phase Evaluation).

### 2. Phân tích cải thiện QMOS (0.548 — cột yếu nhất)
- **Vì sao thấp:** (a) UTMOS được train trên TTS *thường* → **lệch domain** giọng cảm xúc (giọng giận/run dễ bị hiểu nhầm là artifact); (b) metric là **SRCC (thứ hạng)** nhưng đang train **MSE (giá trị)**.
- **4 hướng theo ROI:** ① exp09a probe UTMOSv2 (rẻ, không tốn lượt nộp) → ② exp13 fine-tune thẳng UTMOS (phá trần) → ③ ranking loss (tối ưu thẳng SRCC) → ④ ensemble rank-average nhiều nguồn QMOS.
- ⚠️ Kỳ vọng thực tế: QMOS vốn khó đẩy (nhãn người chấm chất lượng trên giọng cảm xúc nhiễu) → ưu tiên "giữ 0.548 ổn định + thử exp13/UTMOSv2 để có bảng ablation cho paper".

### 3. 🔧 HOÀN THIỆN exp13 — sửa lỗi ranking loss
- **Lỗi tiềm ẩn (chưa lộ vì cờ mặc định tắt):** vòng train cũ gọi `loss.backward()` cho MSE **từng bước** → PyTorch giải phóng đồ thị của `pred`; tới mốc ACCUM mới gọi `rank_loss.backward()` trên chính các `pred` đó → sẽ lỗi **"backward through the graph a second time"**.
- **Cách sửa (đã code):** 2 chế độ rõ ràng — `RANK_LAMBDA=0` backward ngay từng mẫu (VRAM thấp, như cũ); `RANK_LAMBDA>0` **gom MSE (`win_loss`) + pred (`buf_p`) cả cửa sổ ACCUM rồi backward MỘT lần** (`MSE_mean + λ·pairwise_rank`). Đổi mốc cửa sổ sang **đếm mẫu hợp lệ** (`micro`) + **flush phần dư cuối epoch**. Thêm cảnh báo: bật ranking tốn VRAM hơn (giữ đồ thị cả cửa sổ) → OOM thì giảm `ACCUM`/`QMOS_MAX_SEC`.
- **Đính chính nhận định cũ:** vì exp13 gom qua cả cửa sổ ACCUM=16, ranking so **~16 câu/lần (≈120 cặp)**, KHÔNG yếu như lo ngại "BATCH=1 → 1 cặp" (cái đó chỉ đúng cho exp15 BATCH=2).
- **Kiểm:** `py_compile` OK · đồng bộ `.ipynb` (jupytext) JSON hợp lệ. CHƯA chạy thật (cần Kaggle + ckpt).

### 4. "Fine-tune ở dưới" trong exp14 = chính là hướng exp15
- exp14: WavLM frame-level **đóng băng** (`requires_grad=False`, cache fp16) → chỉ train đầu Mamba nhỏ. Mở băng WavLM ("ở dưới") = **mất cache** + WavLM-large backprop + Mamba thuần PyTorch backprop-through-time → **OOM/rất chậm trên T4**.
- Việc đó **chính là exp15** (WavLM fine-tune + Mamba head). → Khuyến nghị: chạy **exp14 đóng băng trước** (ablation rẻ: Mamba có giúp không?); chỉ khi có tín hiệu mới fine-tune qua **exp15**; fine-tune cả backbone + Mamba cùng lúc là nặng nhất → để sau cùng.

### 5. Việc tiếp theo
- 🔴 (vẫn nợ) **NỘP bản trộn cột** `exp_mix_q07_emo08/submission.zip` — ROI cao nhất.
- 🔴 **Upload `ft_emotion_full_20epoch.pt` + cache audeering lên Kaggle thành Dataset** (cần cho exp13 PHẦN B + phase Evaluation).
- 🔴 Smoke test exp13 (`LIMIT_TRAIN=300, RANK_LAMBDA=0`) → so val SRCC vs UTMOS zero-shot; nếu chưa vượt 0.548 thử `RANK_LAMBDA=0.3` (giờ đã chạy được).
- 🟠 Chạy exp14 đóng băng (USE_MAMBA on/off) → ablation Mamba; có tín hiệu → exp15.

---

## Báo cáo ngày 8/6/2026 (Phiên 12) — hướng mới: Audio-LLM-as-Judge (exp16) + buổi học train/fine-tune

**Người thực hiện:** Tran Minh Toan · **Nội dung:** buổi học kinh nghiệm train/fine-tune (8 bài rút từ chính exp08/11/12) + làm rõ trunk/fusion/pooling; chốt hướng "thêm SOTA LLM" theo cách **API audio-LLM-as-judge** (mục tiêu novelty cho paper); code experiment mới **exp16**.

### 1. Buổi học train/fine-tune (ghi `03_`)
- 8 bài kinh nghiệm rút từ chính dự án: (1) fine-tune > freeze nhưng có trần; (2) **val nội bộ đẹp = bẫy overfit** (exp11 val 0.83 vs DEV 0.66); (3) warm-start đã đỉnh → train thêm vô ích (exp08b≈exp08); (4) checkpoint phải lưu ĐỦ+mỗi best+Save Version (sự cố mất backbone exp08); (5) data nhỏ 12k → đừng from-scratch (exp12); (6) loss khớp metric (MSE vs SRCC → ranking loss); (7) mẹo T4 (AMP/grad-ckpt/grad-accum/2 LR); (8) fusion ≠ ensemble.
- Làm rõ 3 khái niệm hay lẫn: **fusion** (nối nguồn, concat) ≠ **trunk** (thân chung MLP multi-task) ≠ **pooling** (gộp chuỗi frame→1 vector). Mamba (exp15) thay **pooling** chứ không thay trunk; exp14 Mamba là nhánh **cộng thêm**.

### 2. Chốt hướng "thêm GPT/LLM" → exp16 Audio-LLM-as-Judge (API)
- Hỏi rõ user: chọn **gọi API audio-LLM** (không phải fine-tune LLM nặng trên T4), mục tiêu = **novelty cho paper** (không bắt buộc phá trần). Né được nỗi đau Phiên 10 ("LLM quá nặng cho T4") vì chỉ inference qua mạng.
- Câu chuyện paper: **"khảo sát có hệ thống audio-LLM-as-judge cho MOS cảm xúc"** — so audio-LLM zero/few-shot với hệ SSL đã train (exp07/exp08), phân tích LLM mạnh ở đâu (EMOS/CAT) yếu ở đâu (QMOS).

### 3. Code exp16 (`exp16_llm_judge_pipeline.py` + `.ipynb`)
- Thuần API, **không cần GPU**. Mỗi wav DEV → gửi audio + prompt có cấu trúc → LLM trả JSON 6 cột → ráp `answer.txt` (đúng format exp07) → validate → zip.
- **Cờ `PROVIDER`:** `gemini` (mặc định, đã có billing) / `openai` (GPT-4o-audio) → bảng so 2 audio-LLM. **Cờ `SHOT_MODE`:** zero/few-shot (few-shot nhét K audio ví dụ có nhãn từ train.csv).
- **Cache + resume bắt buộc** (`.jsonl` mỗi stem) để KHÔNG trả tiền 2 lần; parse JSON chịu lỗi (regex `{...}` + clamp [1,5] + retry); temperature=0 để tái lập.
- Tái dùng `load_target_emotions()`/`norm_emotion()`/format `answer.txt` của baseline+exp07. Kèm hàm `ensemble_rank_average` (tùy chọn) trộn rank LLM + hệ trained.
- **Kiểm:** syntax OK (py_compile) + `.ipynb` JSON hợp lệ. **CHƯA chạy** (cần API key + audio Kaggle) → chưa có điểm.

### 4. Lưu ý trước khi chạy
- ⚠️ **Xác nhận model ID nhận audio** (`GEMINI_MODEL` mặc định `gemini-2.5-flash`, `OPENAI_MODEL` `gpt-4o-audio-preview` — knowledge tới 1/2026, có thể đổi; baseline từng dùng họ `gemini-3-flash-preview`).
- Smoke test `LIMIT=20` trước → full 2730. Secrets: `GEMINI_API_KEY` (+ `OPENAI_API_KEY` nếu openai), Internet On, GPU không cần.

### 5. Việc tiếp theo
- 🔴 (vẫn nợ từ Phiên 7) **nộp DEV bản trộn cột** exp07(QMOS)+exp08(cảm xúc) — vẫn là việc ROI cao nhất.
- 🔴 Smoke test + chạy exp16 (Gemini zero-shot) → nộp → đọc SRCC → điền Bảng A (LLM vs exp07/exp08) cho paper.
- 🟠 Chạy thêm exp16 few-shot và/hoặc OpenAI → Bảng B; tùy chọn ensemble rank-average.
- 🟡 exp13/exp15 vẫn chờ chạy thật.

---

## Báo cáo ngày 8/6/2026 (Phiên 11) — buổi học củng cố hệ thống + soạn báo cáo mentor

**Người thực hiện:** Tran Minh Toan · **Nội dung:** buổi học (không chạy thí nghiệm) — hệ thống hóa cách hoạt động toàn bộ pipeline để chuẩn bị viết phần Method cho paper; soạn báo cáo gửi mentor.

### 1. Buổi học "5 mắt xích" (ghi vào `03_literature_notes.md`)
Đi từ trực giác → ví dụ → code thật, qua 5 mắt xích: (1) WavLM/SSL backbone + self-attention; (2) feature→điểm (pooling + head); (3) fusion + multi-task + uncertainty weighting (= exp07, class `FusionMTL6`); (4) freeze vs fine-tune (`requires_grad`, unfreeze 6 lớp, 2 LR — exp08); (5) MSE-vs-SRCC + ranking loss (exp15). Thêm buổi đào sâu **toán self-attention** (`softmax(QKᵀ/√d)·V`) và **mắt xích data** (nhãn listener-wise → MOS = `groupby.mean()`, EMOS cần target ở `metadata.csv`, train/val/eval split).

### 2. Báo cáo gửi mentor (bản ngắn đã gửi)
- **Điểm DEV tốt nhất từng cột:** QMOS 0.548 (exp07) · EMOS 0.811 · CAT err 0.133 · VAD 0.659/0.793/0.751 (exp08). Hệ mạnh nhất = "trộn cột" exp07(QMOS)+exp08(cảm xúc).
- **Việc hôm nay:** hệ thống hóa kiến trúc cho paper; đã code hướng Mamba (exp14/15), đang chuẩn bị chạy thử so exp08.
- **Câu hỏi cho mentor:** (a) kinh nghiệm fine-tune với data nhỏ (~12k) — mở mấy lớp, LR backbone, mẹo tránh overfit & thu hẹp gap dev→eval; (b) novelty "first systematic study of EMOS prediction" + phát hiện emotion2vec vượt SOTA do metric ranking có đứng được không; (c) đồng tác giả paper. → đã thêm vào `02_mentor_questions.md`.

### 3. Trạng thái việc tồn đọng (không đổi so Phiên 10)
🔴 Vẫn **chưa nộp DEV bản trộn cột** (món nợ ưu tiên #1). exp13/14/15 vẫn ở trạng thái đã code, chưa chạy thật.

---

## Báo cáo ngày 8/6/2026 (Phiên 10) — áp dụng SOTA mới (Mamba) theo gợi ý mentor: code exp13 + exp14 + exp15

**Người thực hiện:** Tran Minh Toan · **Nội dung:** mentor gợi ý đọc/áp dụng SOTA mới (LLM-based / Mamba) để paper xịn hơn → khảo sát 2 hướng, chọn **Mamba** (khả thi T4), code 2 pipeline mới.

### 1. Khảo sát SOTA (ghi `03_`) — chọn hướng
- **LLM-based** (audio-LLM chấm MOS): ALLD (ICLR 2025, MSE 0.17, vượt wav2vec2/WavLM), SpeechQualityLLM. → **mạnh cho paper nhưng quá nặng để train trên T4** → để dành làm section khảo sát / zero-shot sau.
- **Mamba** (State Space Model, độ phức tạp tuyến tính): MambaRate, HighRateMOS (AudioMOS 2025). → **khả thi trên T4** (chỉ thay phần head/encoder) → **chọn làm trước**.

### 2. exp14 — MAMBA cộng vào fusion 6 cột (frozen WavLM frame-level)
- Nhánh Mamba 2 chiều trên WavLM frame-level (đóng băng) **cộng thêm** vào fusion exp07 (e2v+SAILER pooled) → 6 cột. Cờ `USE_MAMBA` False=exp07/True=+Mamba = ablation. Cache frame fp16. CHƯA chạy.

### 2b. exp13 — FINE-TUNE thẳng UTMOS cho QMOS (đánh thẳng cột chất lượng)
- Song song hướng Mamba (cảm xúc), code thêm hướng phá trần **QMOS 0.548 (exp07)** = **fine-tune trực tiếp UTMOS** trên nhãn `qMOS` thật, sau đó **ghép 5 cột cảm xúc từ ckpt exp08 20ep** (`ft_emotion_full_20epoch.pt`) → `answer.txt` 6 cột.
- **Vì sao UTMOS chứ không UTMOSv2:** UTMOS = 1 `nn.Module` chuẩn, backprop được toàn model; UTMOSv2 = ensemble nhiều fold + 2 luồng, khó train. Khi fine-tune chính UTMOS thì "neo" UTMOS nằm sẵn trong trọng số warm-start → bỏ neo ngoài.
- **Config (đã code):** LR 1e-5 · BATCH=1 + ACCUM 16 (UTMOS không có attention-mask → batch>1 lệch pooling do pad) · MAX_SECONDS 12 · `FREEZE_FEAT_EXT=True` · `RANK_LAMBDA=0` (tùy chọn pairwise ranking loss để tối ưu thẳng SRCC) · EPOCHS 10 · PATIENCE 3 · AMP. Lưu `ft_qmos_utmos.pt` mỗi best.
- **Lưới an toàn:** chỉ nộp khi SRCC val nội bộ UTMOS-ft > UTMOS zero-shot (mục A in cả 2 số). CHƯA chạy. File `exp13_finetune_qmos_pipeline.py` + `.ipynb`.

### 3. exp15 — WavLM FINE-TUNE + MAMBA head cho 5 cột cảm xúc ⭐ (yêu cầu chính của user)
- User chốt: **Mamba head TRÊN WavLM (fine-tune)** predict cả 5 cột cảm xúc, SAILER warm-start. = exp08 đổi đúng 1 chỗ: thay **mean-pool** bằng **MambaEncoder** (proj 1024→256, Mamba 2 lớp 2 chiều, attentive-pool có mask) trước các head.
- **Giả thuyết:** mean-pool vứt bỏ động lực thời gian (lên/xuống giọng, ngắt quãng, run giọng) → Mamba nắm được → kỳ vọng vượt exp08 (EMOS 0.811/VAD 0.659·0.793·0.751).
- Cờ `USE_MAMBA` False=exp08/True=Mamba = **ablation "Mamba vs mean-pool"** cho paper. Tái dùng toàn bộ scaffolding exp08.
- **Gotcha đã phòng:** layerdrop=0 (CheckpointError), Mamba thuần PyTorch (fp32) + thêm dòng cài `mamba-ssm causal-conv1d` (bọc try/except, tự dùng kernel CUDA nếu được), checkpoint lưu cả backbone+Mamba+heads, không đụng numpy.
- ⚠️ **Rủi ro chính = tốc độ:** Mamba thuần PyTorch khi fine-tune (backprop-through-time) rất chậm trên full → cap MAX_SECONDS=6/BATCH=2; nếu cài được kernel CUDA thì nhanh + nhẹ hơn nhiều. CHƯA chạy.

### 4. Việc tiếp theo
- 🔴 **Smoke test exp15** (LIMIT 300/20, USE_MAMBA=True) trên Kaggle → kiểm OOM/CheckpointError/tốc độ.
- 🔴 **Smoke test exp13** (LIMIT_TRAIN=300, LIMIT_DEV=20) → kiểm UTMOS-ft chạy ổn (đặc biệt BATCH=1) + đo VAL nội bộ vs UTMOS zero-shot.
- 🟠 Chạy thật exp15 2 lần (Mamba on/off) → bảng ablation; nếu thắng exp08 → ráp answer.txt nộp DEV.
- 🟠 Chạy thật exp13 (full) → so QMOS với exp07 (0.548); nếu vượt → exp13 thay exp07 cho cột QMOS trong hệ trộn cột.
- 🟡 Vẫn nợ từ Phiên 9: nộp DEV bản trộn cột mạnh nhất (exp08 cảm xúc + exp07 QMOS); chạy đủ exp12 3 mode.

### 5. (CẬP NHẬT cuối Phiên 10) — tạo .ipynb, ghép bản trộn cột, nâng cấp exp15, smoke test
- **Tạo notebook:** `exp13_finetune_qmos.ipynb` (từ .py) + `exp15_wavlm_mamba_emotion.ipynb` (.py do user viết). exp14 hiện **chỉ có .py** (chưa convert).
- 🎯 **GHÉP BẢN TRỘN CỘT (làm được ngay, không cần GPU):** QMOS←exp07 (0.548) + 5 cột cảm xúc←exp08 → `submissions/Track2/exp_mix_q07_emo08/{answer.txt, submission.zip}` (2730 dòng, validate OK). **SẴN SÀNG NỘP** — đây là hệ 6 cột mạnh nhất, vẫn là món nợ từ Phiên 8/9. Chỉ cần upload CodaBench.
- **Nâng cấp exp15:**
  - **Ranking loss:** thêm `RANK_LAMBDA=0.3` + hàm `pairwise_rank_loss` cho 4 cột SRCC (emos/val/aro/dom); CAT giữ soft-CE. Lý do: metric là **UTT-SRCC (thứ hạng)** nhưng đang train **MSE (giá trị)** → ranking khớp metric hơn. ⚠️ điểm yếu: ranking tính cặp TRONG mini-batch, `BATCH=2` → chỉ 1 cặp/forward → tín hiệu yếu; mạnh nhất ở exp06/07 (frozen, BATCH=64).
  - **Tự dò DATA_ROOT:** hàm `find_data_root()` quét `/kaggle/input` tìm thư mục đủ `sets/train.csv`+`wav/`+`metadata.csv` → khỏi sửa slug tay. `DATA_ROOT` exp13 đặt `/kaggle/input/datasets/minhtoan2`.
  - **Sửa cài mamba-ssm:** thêm `--no-build-isolation` cho cả `causal-conv1d` + cài `ninja`.
- **Smoke test exp15 (Kaggle):** `mamba-ssm` build **vẫn fail** → tự fallback **Mamba thuần PyTorch** (chạy được nhưng chậm khi full). Bài học mới: ⚠️ **Internet phải BẬT On** (lần đầu tắt → pip fail `loralib` "name resolution").
- **Tài liệu:** tạo `docs/20_experiments_overview.md` (bảng trạng thái nhanh exp: đã nộp/đã chạy/mới code) + thêm vào bản đồ CLAUDE.md.
- **Đã giải thích cho user (ghi `03_`):** batch & gradient accumulation, LoRA/QLoRA vs partial fine-tune, UTT vs system-level, ranking loss vs MSE (SRCC không khả vi → dùng pairwise hinge), loss vốn đã per-utterance.

### 6. Việc tiếp theo (ưu tiên cập nhật)
- 🔴 **NỘP bản trộn cột** `exp_mix_q07_emo08/submission.zip` lên CodaBench → chốt điểm hệ 6 cột mạnh nhất (đã sẵn, chỉ thiếu upload).
- 🔴 Smoke test rồi chạy thật **exp13** (UTMOS-ft) + **exp15** (Mamba on/off) → bảng ablation.
- 🟠 Nếu mamba-ssm fail + muốn chạy thật exp15: hạ tải (MAX_SECONDS 5/LAYERS 1) hoặc đổi temporal head sang BiGRU/Transformer (nhanh, không cần biên dịch).

---

## Báo cáo ngày 8/6/2026 (Phiên 9) — exp08b điểm thật · exp11 fusion 2 backbone · exp12 ablation khởi tạo · fix layerdrop/numpy

**Người thực hiện:** Tran Minh Toan · **Nội dung:** đọc điểm exp08b (resume) đã nộp; code 2 hướng mới (fusion fine-tune CẢ 2 backbone + ablation from-scratch theo gợi ý mentor); xử lý 2 lỗi runtime Kaggle.

### 1. exp08b (RESUME exp08) — điểm THẬT trên CodaBench
| Cột | exp08b | exp08 gốc |
|---|---|---|
| MOS/QMOS | 0.4167 | 0.4139 |
| EMOS | 0.8116 | 0.811 |
| CAT ERR | 0.1331 | 0.133 |
| VAL/ARO/DOM | 0.6605/0.7904/0.7539 | 0.659/0.793/0.751 |
- Resume train tiếp từ checkpoint → điểm **gần như TRÙNG** exp08 → **xác nhận checkpoint đã hội tụ**, train thêm trên cùng data ~không đổi. Submission lưu `submissions/Track2/exp08b_finetune_resume/`.

### 2. exp11 — FINE-TUNE ĐỒNG THỜI WavLM + audeering, FUSION 1 model (code MỚI + đã chạy)
- Khác exp08 (audeering frozen) & exp10 (ensemble 2 model riêng): exp11 **mở băng CẢ 2 backbone**, fuse đặc trưng **trong 1 model**, warm-start WavLM+heads từ `ft_emotion_full_20epoch.pt`. Có cờ tự **RESUME** đủ (nạp cả `aud`/`aud_head`) khi trỏ vào `ft_joint_full.pt`. Chống OOM: BATCH=1+ACCUM, grad-ckpt cả 2, AMP, MAX_SECONDS=6, mở băng 4 lớp/backbone.
- **Kết quả CHẠY (8/6, VAL nội bộ):** warm-start mean SRCC **0.8298** (EMOS 0.835/VAL 0.803/ARO 0.874/DOM 0.808) → 4 epoch sau đều thấp hơn → **early stop, KHÔNG cải thiện**. ⚠️ Bài học: warm-start đã ở đỉnh + resume LR nhỏ → khó vượt; **VAL nội bộ ≠ DEV** (VAD nội bộ 0.80/0.87/0.80 >> DEV exp08 0.66/0.79/0.75 = overfit/lệch). CHƯA nộp DEV.

### 3. exp12 — Ablation KHỞI TẠO WavLM (trả lời mentor "from-scratch vs fine-tune")
- Mentor gợi ý: "12k data có thể train scratch tốt hơn". Code 1 notebook cờ `INIT_MODE`: `scratch` (random init, mở băng TOÀN BỘ, LR 1e-4) · `base` (WavLM-large pretrain SSL) · `sailer` (warm-start cảm xúc). CHỈ WavLM (bỏ audeering) để cô lập biến khởi tạo. Chạy 3 lần → bảng ablation.
- **Quan điểm đã trao đổi:** với 12k mẫu, fine-tune pretrained gần như chắc chắn > from-scratch (SSL pretrain ~94k giờ; 12k quá ít để dạy từ đầu) → exp12 để **chứng minh bằng số**. Chưa có đủ số 3 mode (đang chạy).

### 4. Hai lỗi runtime Kaggle + cách fix (ghi `04_` mục Lỗi & bài học)
- **CheckpointError** (tensor lệch khi backward): do **layerdrop** đụng gradient-checkpointing → fix `config.layerdrop=0.0` cho cả WavLM + audeering.
- **SystemError "bad call flags"** khi import torch: `pip install` đổi **numpy** → lệch ABI torch → fix **Restart Session** + khóa `numpy==<bản gốc>` mỗi lần cài, chỉ cài gói thiếu.

### 5. Khái niệm làm rõ cho người dùng (ghi `03_`)
- **Ensemble ≠ Fusion:** fusion = nối đặc trưng TRONG 1 model (exp04/07/08/11); ensemble = trung bình KẾT QUẢ của nhiều model riêng (exp10). Hệ hiện tại đã có fusion, **chưa có ensemble thật**.

### 6. Dọn file
- Dời `submission_track2_exp08_resume.zip` + `scoring_result.zip` → `submissions/Track2/exp08b_finetune_resume/` (đổi tên `submission.zip`). Dời `ft_emotion_full.pt` + `ft_emotion_full_20epoch.pt` → `cache/`. `ft_joint_full.pt` (exp11) còn ở gốc.

### 7. Việc tiếp theo
- 🔴 Nộp DEV: hệ trộn cột mạnh nhất (5 cảm xúc exp08/exp08b + QMOS exp07 0.548) — vẫn CHƯA nộp bản trộn này.
- 🟠 Chạy đủ exp12 (3 mode) → bảng ablation trả lời mentor.
- 🟠 Cân nhắc nộp exp11 DEV để biết fusion-2-backbone có hơn exp08 thật không (đừng tin VAL nội bộ).
- 🟡 Hướng tăng điểm thật (thay vì vắt internal val): **ranking loss** (tối ưu thẳng SRCC) hoặc **ensemble** (exp10) — ROI cao hơn vặn hyperparameter.

---

## Báo cáo ngày 5/6/2026 (Phiên 8) — 🧹 dọn & chuẩn hóa tên file/folder

**Người thực hiện:** Tran Minh Toan · **Nội dung:** chuẩn hóa tên lộn xộn (sai chính tả, không có số exp) trong `kaggle_baseline/track2/` và `submissions/Track2/` → dễ tra cứu, map file ↔ experiment rõ ràng.

### 1. `kaggle_baseline/track2/` — đổi tên 10 cặp `.py`+`.ipynb` (bỏ tiền tố `track2_`, thêm `expNN`)
| Cũ | Mới |
|---|---|
| track2_train_emos | **exp02**_train_emos |
| track2_emos_sailer | **exp03**_emos_sailer |
| track2_fusion | **exp04**_fusion |
| track2_vad_audeering | **exp05**_vad_audeering |
| track2_qmos_train | **exp06**_qmos_train |
| track2_fusion_qmos | **exp07**_fusion_qmos |
| track2_finetune_emotion | **exp08**_finetune_emotion |
| track2_finetune_emotion_resume | **exp08b**_finetune_resume |
| track2_qmos_utmosv2_probe | **exp09a**_qmos_utmosv2_probe |
| track2_finetune_audeering | **exp10**_finetune_audeering |
- **Giữ nguyên** (không phải experiment đơn): `track2_baseline`, `track2_prepare_data`, `demo_track2_gradio`. Đã xóa `__pycache__` cũ.

### 2. `submissions/Track2/` — chuẩn hóa folder + file
- Folder: `emotion2vec→exp01_emotion2vec`, `exp3_sailer→exp03_sailer`, `exp4fussion→exp04_fusion`, `exp7qmosfussion→exp07_fusion_qmos`, `exp8_emtion_finetune→exp08_finetune_emotion` (sửa lỗi "fussion"/"emtion").
- File trong mỗi folder: `scoring_result (N).zip→scoring_result.zip`, `submission_track2_*.zip→submission.zip`, `_extracted`/`scoring_result_5→scores/`.

### 3. Cập nhật tham chiếu (không gãy link)
- Thay tên cũ trong **60 file** (docs, README, 6 file memory, các notebook). Grep tên cũ = **0**; notebook vẫn JSON hợp lệ; xác nhận không file pipeline nào import lẫn nhau.
- Bổ sung quy ước đặt tên `expNN_tên` vào `CLAUDE.md` (mục 4).

### 4. Quy ước "dọn dẹp" + dọn docs
- Thêm mục **2B vào CLAUDE.md**: khi user gõ "dọn dẹp" → rà soát `docs/` tìm mâu thuẫn số liệu / lỗi thời / link hỏng / trùng lặp / ngày sai → báo cáo → sửa lỗi rõ ràng (nguồn chuẩn `04_`/`12_`), giữ lịch sử.
- Chạy "dọn dẹp" lần đầu: đồng bộ "tốt nhất" về exp08 (header `07_`, gỡ 🏆 lỗi thời ở bảng `12_`, sửa "TỐT NHẤT hiện tại" exp07 ở `04_`/`12_`, sửa ngày 4/6→5/6).

### 5. File mới + cập nhật paper
- Tạo **`18_leaderboard_history.md`**: bảng leaderboard theo ngày (best-per-column + từng bản nộp + Track 1/3).
- Tạo **`19_paper_v1_en.md`**: bản paper v1 TIẾNG ANH (start version, đủ 6 mục + abstract + ablation + refs placeholder).
- Cập nhật **`15_paper_draft.md`** theo kết quả tốt nhất: abstract số mới (EMOS 0.811…), C1–C4 (thêm fine-tune), Method 3.2 (exp07 4-head) + 3.4 (exp08 fine-tune), bảng Results + ablation, mốc ICASSP xác nhận.

### 6. Checklist Evaluation Phase (ghim đầu `13_`)
- Soạn checklist 4 mốc cho hạn **7/8/2026**: chuẩn bị trước 31/7 (lưu checkpoint, script trộn cột + validate test trên DEV) · khi eval thả · nộp (cả T1/T2/T3 + system description) · sau nộp (fallback exp07).

### 7. Nghiên cứu (lưu vào `03_`): 2024/2025, dev↔eval, ensemble, ICASSP
- **2024 winners:** UTT-SRCC vô địch chỉ ~0.62–0.68 (utt-level vốn khó); system-level cao hơn nhiều.
- **dev↔eval:** thường tụt, UTT-level nhạy; exp08 fine-tune rủi ro overfit hơn exp07.
- **Ensemble (AudioMOS 2025):** đội đầu gộp **5–9 model**; **cùng kiến trúc khác seed + trung bình = ensemble hợp lệ**; phân biệt *feature fusion* (1 model, đang có) vs *ensemble* (nhiều model, chưa có).
- **ICASSP 2027:** hạn 16/9/2026, hội nghị 16–21/5/2027 Toronto; ~45% nhận; tham gia challenge được trích dẫn trong paper tổng kết BTC.

### 8. Việc tiếp theo
- 🔴 Train lại exp08 (`exp08_finetune_emotion`) → có lại `ft_emotion_full.pt` → Save Version ngay.
- 🔴 Trộn cột: 5 cảm xúc exp08 + QMOS exp07 (0.548) → nộp hệ mạnh nhất 6 cột.
- 🟠 **Ensemble:** exp08 train 3 seed → trung bình + gộp exp07 (đa dạng kiến trúc) → giảm gap dev↔eval.
- ⚠️ Chưa `git commit` (chờ user). Các file đổi tên đều untracked nên git rủi ro thấp.

---

## Báo cáo ngày 5/6/2026 (Phiên 7) — exp08 NỘP (điểm thật) · UTMOSv2 · mất backbone & vá checkpoint · exp10 audeering ensemble

**Người thực hiện:** Tran Minh Toan · **Nội dung:** đọc điểm exp08 đã nộp; research model QMOS mới hơn UTMOS; xử lý sự cố mất backbone fine-tune; tạo 3 notebook mới (UTMOSv2 probe, resume, audeering ensemble).

### 1. 🏆 exp08 (fine-tune WavLM) — điểm THẬT trên CodaBench
| Cột | exp08 | exp07 | |
|---|---|---|---|
| QMOS | 0.4139 | 0.548 | 🔻 −0.134 (rớt về UTMOS — KHÔNG mượn exp07) |
| EMOS | **0.811** | 0.795 | 🚀 +0.016 |
| CAT ERR | **0.133** | 0.153 | ✅ −0.020 |
| VAL | **0.659** | 0.581 | 🚀 +0.078 |
| ARO | **0.793** | 0.752 | 🚀 +0.041 |
| DOM | **0.751** | 0.705 | ✅ +0.046 |
- Fine-tune **THẮNG cả 5 cột cảm xúc** → bộ cảm xúc tốt nhất từ trước tới nay. Nhưng QMOS rớt 0.414 do bản nộp **không có** answer.txt exp07 (rơi vào fallback UTMOS).
- ➡️ **Việc chốt:** TRỘN CỘT = 5 cột cảm xúc exp08 + QMOS exp07 (0.548) → hệ thống mạnh nhất 6 cột.

### 2. Research model QMOS mới hơn UTMOS → **UTMOSv2 (T05, vô địch VMC2024 Track 1)**
- UTMOSv2 (sarulab-speech, **MIT**) = bản kế nhiệm trực tiếp UTMOS 2022, mạnh hơn rõ. Lưu ý: vẫn train trên giọng *không* cảm xúc → có thể lệch domain.
- Tạo **probe A/B không tốn lượt nộp** `exp09a_qmos_utmosv2_probe`: chấm train (nhãn qMOS thật) bằng UTMOS vs UTMOSv2 → so SRCC. **CHƯA chạy.**
- Đổi fallback QMOS trong exp08 (UTMOS→UTMOSv2).

### 3. 🔴 SỰ CỐ: mất backbone fine-tune exp08
- `ft_emotion_meta.pt` (bản gốc) **chỉ lưu `heads`, KHÔNG lưu WavLM** → khi kernel chết, backbone fine-tune **mất vĩnh viễn** (xác nhận keys: không có `'wavlm'`). Điểm exp08 vẫn trên leaderboard nhưng **trọng số mất** → phải train lại.
- **Đã vá:** exp08 giờ lưu **`ft_emotion_full.pt`** (đủ wavlm+heads) **mỗi khi đạt best** (không chỉ cuối) → kernel chết giữa chừng vẫn còn. Notebook gốc + tự copy cache audeering (`CACHE_INPUT`).
- Tạo `exp08b_finetune_resume`: resume từ `ft_emotion_full.pt` + cache (gotcha: `weights_only=False` cho torch 2.6; slug Kaggle đổi `_`→`-`).

### 4. exp10 — fine-tune AUDEERING riêng + ensemble VAD (Hướng A cho T4)
- "Fine-tune 2 model" → chốt KHÔNG nhồi 2 backbone 1 model (T4 OOM). Thay vào: fine-tune audeering **riêng** (1 backbone) → **ensemble cột VAD** (trung bình) với exp08.
- Tạo `exp10_finetune_audeering` (lưu `ft_audeering_full.pt` mỗi best; mục 7 ensemble VAL/ARO/DOM). **CHƯA chạy.**

### 5. Việc tiếp theo
- 🔴 **Train lại exp08** (tái dùng cache audeering, lưu full) → có lại backbone → **Save Version NGAY**.
- 🔴 **Trộn cột** exp08(5 cảm xúc)+exp07(QMOS 0.548) → nộp hệ thống mạnh nhất 6 cột.
- 🟠 Chạy exp10 audeering → đọc VAD có ≥ exp08 → ensemble → nộp so sánh.
- 🟡 Chạy probe UTMOSv2; nếu thắng → exp09 fine-tune QMOS dùng UTMOSv2 làm neo.

---

## Báo cáo ngày 5/6/2026 (Phiên 6) — exp08 FINE-TUNE WavLM cho cảm xúc (theo gợi ý mentor) + tài liệu nền

**Người thực hiện:** Tran Minh Toan · **Nội dung:** theo gợi ý mentor "fine-tune để cải thiện", chốt hướng + code thí nghiệm **fine-tune đầu tiên** của dự án (exp08); viết 2 file tài liệu nền DL.

### 1. Chốt hướng fine-tune (khác mọi exp trước = đều freeze)
- **Vì sao mentor đúng:** "freeze + head" có trần (backbone học cho task khác); fine-tune phá trần + **nặng ký hơn cho paper** (mô hình hóa thật, không chỉ ghép model người khác).
- **Chốt phạm vi:** fine-tune **cho 5 cột cảm xúc** (không đụng QMOS — giữ exp07). Backbone = **WavLM-large warm-start từ SAILER** (đã giỏi cảm xúc) thay vì WavLM trắng. Bỏ emotion2vec (funasr khó fine-tune), thay bằng **audeering MSP-dim** làm nhánh phụ **frozen** (dimensional, bổ trợ valence).

### 2. Code exp08 (`exp08_finetune_emotion_pipeline.py` + `.ipynb`)
- **Kiến trúc:** WavLM (lôi backbone HF bên trong wrapper SAILER ra, có fallback WavLM trắng) — **mở băng 6 lớp Transformer trên**, đóng băng feature-extractor + lớp dưới. + audeering frozen (cache `aud_*.npz`, `[emb 1024|vad3]`) → trunk → 3 head (EMOS+target/CAT/VAD), uncertainty weighting. QMOS mượn answer.txt exp07 (hoặc UTMOS).
- **Kỹ thuật T4:** AMP fp16 · gradient checkpointing · `BATCH 4 × ACCUM 8` (hiệu dụng 32) · `MAX_SECONDS 8` · LR backbone 1e-5 / head 1e-3 · early-stop theo TB SRCC val.
- **Đổi `EPOCHS` 8→12** (trần; early-stop quyết số epoch thật — 8 hơi thấp, dễ dừng non).
- **3 kiểu xử lý trong 1 cell:** 🔥 fine-tune (6 lớp WavLM) + 🆕 train (trunk+head) + ❄️ lấy embedding (audeering) — tất cả trong **1 vòng lặp, 1 backward** (đặc thù end-to-end → mất cache, chậm).

### 3. Kết quả đang chạy (full, mới epoch 2 — val ~1.1k đáng tin)
| Cột | exp08 (epoch 2) | exp07 | |
|---|---|---|---|
| EMOS | 0.752 | 0.795 | 🔻 −0.043 (head đang ấm, có thể leo) |
| VAL | **0.747** | 0.581 | 🚀 +0.166 |
| ARO | **0.857** | 0.752 | 🚀 +0.105 |
| DOM | **0.783** | 0.705 | 🚀 +0.078 |
- **TB 4 cột:** exp08 0.785 vs exp07 0.708 → **+0.077** ngay từ epoch 2. Fine-tune ăn đậm ở **VAD** (đúng kỳ vọng). EMOS tụt nhẹ vì **bỏ emotion2vec** (vô địch EMOS lẻ).
- **Lưu ý:** `cat_err 0.582` là metric *nội bộ tạm* (L1 phân phối), KHÔNG so được CAT-ERR CodaBench (0.153).

### 4. Tài liệu nền (cho người mới + paper)
- Tạo [16_model_architectures.md](16_model_architectures.md): kiến trúc 8 model/khái niệm (WavLM/SAILER/emotion2vec/audeering/UTMOS/wav2vec2/uncertainty/fusion) — mỗi cái: kiến trúc → vai trò → exp → **arXiv** → license.
- Tạo [17_dl_keywords.md](17_dl_keywords.md): ~30 keyword DL (định nghĩa → công thức → sơ đồ → ví dụ trong code).
- Đã web-verify SAILER = **arXiv:2505.22133** (Interspeech 2025 SER winner) + Vox-Profile **arXiv:2505.14648**.

### 5. Việc tiếp theo
- 🔴 Chờ exp08 chạy **full xong** → đọc khối `✅ VAL` cuối: EMOS có leo qua 0.795 không? → quyết **nộp nguyên exp08** (thắng cả 5 cột) hay **trộn cột** (EMOS←exp07/e2v · VAD←exp08).
- 🟡 Ablation cho paper: `UNFREEZE_TOP_LAYERS=0` (head-only) vs `=6` (fine-tune) → bảng "frozen vs fine-tuned"; `USE_AUDEERING=False` đo đóng góp nhánh phụ.
- 🟢 Khai báo license exp08 (WavLM MIT · SAILER Open RAIL · audeering CC BY-NC-SA) trong `12_`.

---

## Báo cáo ngày 4/6/2026 (Phiên 5) — code exp06 + exp07 (tấn công QMOS) + cập nhật paper

**Người thực hiện:** Tran Minh Toan · **Nội dung:** chốt QMOS là đòn bẩy lớn nhất còn lại (cột duy nhất chưa train), lên kế hoạch + code 2 hướng cải thiện; cập nhật khung paper theo exp04.

### 1. Hiểu rõ bài toán QMOS
- **Hiện trạng:** QMOS = UTMOS (SpeechMOS) **zero-shot, không train** → kẹt 0.414. UTMOS được train trên MOS *naturalness* mùa 2022 (giọng không cảm xúc) → **lệch domain**; chưa dùng nhãn `qMOS` có sẵn trong `train.csv`.
- **QMOS đo gì:** độ "sạch & giống người" của audio (artifact/méo/robot) → **phần lớn trực giao với cảm xúc** (giọng sạch nhưng vô cảm: QMOS cao/EMOS thấp). → cảnh báo: backbone cảm xúc (e2v/SAILER) chưa chắc bắt tốt lỗi chất lượng.
- Ngoài SRCC, leaderboard chỉ còn **CAT-ERR**; mọi cột ở mức **UTT** (chấm từng câu, khó hơn system-level).

### 2. Code 2 thí nghiệm (chưa chạy)
- **exp06** — `exp06_qmos_train_pipeline.py` + `.ipynb`: train **head QMOS riêng** trên đặc trưng cache (e2v+SAILER) **+ điểm UTMOS làm 1 đầu vào** (neo residual quanh 0.414). Có val nội bộ in **head SRCC vs UTMOS SRCC** (khỏi đốt lượt nộp). Cuối: **ghép** QMOS mới vào answer.txt exp04 (giữ 5 cột cảm xúc đang thắng).
- **exp07** — `exp07_fusion_qmos_pipeline.py` + `.ipynb`: **gộp QMOS vào trunk fusion** (head thứ 4, đầu vào `[trunk|UTMOS]`), 6 task uncertainty weighting. Kiểm chứng giả thuyết user "chất giọng tự nhiên liên quan cảm nhận cảm xúc". In cảnh báo **negative transfer** (EMOS/VAD tụt so exp04) + cờ ablation `USE_UTMOS_FEAT`.
- Cả 2 **giữ exp04 nguyên** (file riêng, đúng cách tách exp05).

### 3. Cache tái dùng
- Folder `track2cachingcheckpoint/` = cache exp04 (`e2v_train/dev.npz`, `sailer_train/dev.npz`, `fusion_mtl.pt`, ~90 MB) → **tái dùng cho exp06/07** (đỡ ~15' trích). Tên khớp pipeline.
- Workflow Kaggle: upload thành Dataset → Add Input → **copy npz sang `/kaggle/working/fusion_cache`** (vì `/kaggle/input` read-only, pipeline cần ghi `utmos_*.npz`). UTMOS sẽ tự chấm lần đầu (chưa có cache).
- Logic cache theo **từng stem**: đủ thì không trích lại, thiếu file nào trích file đó (resume).

### 4. Cập nhật paper `15_`
- Hệ thống đề xuất chính = **exp04 fusion**; điền Method (2 backbone đóng băng→trunk→3 head, uncertainty weighting), bảng kết quả thật exp01/03/04, khung Abstract 5 câu, bảng ablation 4 cờ (còn `[ ]` chờ chạy).

### 5. Việc tiếp theo
- 🔴 Chạy **exp06** (LIMIT nhỏ→full) → xem head có vượt 0.414 → nộp bản ghép.
- 🟠 Chạy **exp07** → đọc 2 câu hỏi: QMOS vượt UTMOS? cảm xúc có tụt? → quyết định nộp bản hợp nhất hay ghép cột.
- 🟡 Ablation `USE_UTMOS_FEAT=False` (exp07) đo thẳng giả thuyết "chất lượng từ biểu diễn cảm xúc".

---

## Báo cáo ngày 4/6/2026 (Phiên 4) — 🏆 exp04 FUSION NỘP: thắng cả 5 cột cảm xúc

**Người thực hiện:** Tran Minh Toan · **Nội dung:** chạy full + nộp exp04 (fusion multi-task) → đọc `scores.json`.

### Kết quả exp04 (DEV, CodaBench 4/6) — bước nhảy lớn nhất Track 2
| Metric | Tốt nhất trước | **exp04 fusion** |
|---|---|---|
| QMOS SRCC | 0.414 | 0.4139 (giữ) |
| **EMOS SRCC** | 0.637 (exp01) | **0.7878** 🚀 |
| **CAT ERR** | 0.190 (exp03) | **0.1454** ✅ |
| **VAL SRCC** | 0.341 (exp03) | **0.5782** 🚀 |
| **ARO SRCC** | 0.712 (exp03) | **0.7544** ✅ |
| **DOM SRCC** | 0.630 (exp03) | **0.7061** ✅ |

- **Fusion thắng TẤT CẢ 5 cột cảm xúc** so với mọi model lẻ → xác nhận giả thuyết "emotion2vec + SAILER bổ sung nhau". Đây là **hệ thống chính cho paper**.
- **Bài học:** lo ngại "VAD nén chặt quanh 2.5–3.6" khi nhìn answer.txt là **sai** — SRCC chấm **thứ hạng**, giá trị nén không hại nếu thứ tự đúng (VAL 0.341→0.578).
- File điểm: `submissions/Track2/exp04_fusion/scoring_result.zip`.

### Việc tiếp theo
- 🔴 Cải thiện **QMOS** (vẫn 0.414 — cột DUY NHẤT chưa cải thiện): train head chất lượng riêng / fine-tune SSL.
- 🟡 Ablation exp04 (tắt từng nhánh) điền bảng cho paper.
- 🟢 Chạy exp05 (audeering) xem có đẩy VAL > 0.578 nữa không.

---

## Báo cáo ngày 4/6/2026 (Phiên 3) — code exp04 FUSION multi-task + exp05 audeering VAD (chưa chạy xong)

**Người thực hiện:** Tran Minh Toan · **Nội dung:** code hệ thống fusion multi-task (hướng chính cho paper) + thêm nhánh VAD chuyên đẩy VAL; xử lý loạt lỗi môi trường khi chạy Kaggle.

### 1. exp04 — FUSION multi-task ("QMOS riêng + 5 cảm xúc chung")
- Viết `kaggle_baseline/track2/exp04_fusion_pipeline.py` + `.ipynb`: gộp 2 backbone **đóng băng** emotion2vec + SAILER → **trunk chung** → 3 head (EMOS / CAT / VAD); QMOS để riêng (SpeechMOS).
- EMOS head nhận thêm **one-hot target** (EMOS phụ thuộc target); CAT = **soft cross-entropy** với phân phối vote; VAD = MSE. **Cân loss = uncertainty weighting** (log σ² học được cho 5 task, có cờ `USE_UNCERTAINTY=False` để dùng trọng số tay khi debug).
- Cache đặc trưng `.npz` **riêng từng backbone** (resume mỗi 500 file); **4 cờ ablation** (USE_E2V / USE_SAILER / USE_UNCERTAINTY / USE_CLASSPROB) cho paper.
- Gộp nhãn theo `wavID`: EMOS = TB `eMOS` · VAD = TB `val/aro/dom` · CAT = **tỉ lệ vote 5 lớp** của `emoCat`.

### 2. Lỗi đã xử khi chạy Kaggle (exp04)
- 🔴 **`train.csv` KHÔNG phải CSV dấu phẩy** → phân tách bằng **`|`**; cột `emoCat` **đa nhãn** dùng `,` bên trong (vd `Angry,Surprised`). Header: `lisID|wavID|qMOS|emoCat|eMOS|val|dom|aro`. Fix: `pd.read_csv(sep="|")` (hàm `parse_emocat_votes` tách `,` cho CAT).
- 🟢 **Ép GPU cho emotion2vec** (`device=device`) + **tắt log funasr** ồn (`disable_pbar/log/update` + logging level) → hết "dòng đỏ" rác in mỗi file.
- ℹ️ Trích đặc trưng train ~17 it/s (~12–15 phút, **chạy 1 lần rồi cache**, nhớ Save Version). RAM ~10.7/30 GiB → an toàn (dữ liệu tích lũy chỉ ~0,2 GiB).

### 3. exp05 — VAD bằng audeering MSP-dim (đẩy VAL)
- Theo todo "🟠 đẩy VAL (0.341 — thấp nhất)". **Tách file riêng** `exp05_vad_audeering` (giữ exp03 `exp03_emos_sailer` **nguyên bản cũ**): SAILER lo EMOS+CAT, **audeering lo cả 3 VAD**, QMOS=SpeechMOS.
- 🔴 audeering (model card cũ) kế thừa `Wav2Vec2PreTrainedModel` → lỗi version transformers mới (`module '__main__' has no attribute '__file__'`, rồi `all_tied_weights_keys`). **Fix dứt điểm:** bỏ subclass — chỉ dùng `Wav2Vec2Model` (backbone) + **tự nạp tay** trọng số regression head từ checkpoint → không đụng tie-weights/experts.
- ⚠️ audeering xuất thứ tự **[arousal, dominance, valence]** → đã đổi về [VAL,ARO,DOM] thang 1–5. License **CC BY-NC-SA 4.0** (phi thương mại) → khai báo `12_`.

### 4. Còn dang dở
- exp04: chưa train head xong → **CHƯA có VAL SRCC nội bộ + chưa nộp** (đang/đã trích đặc trưng).
- exp05: **chưa chạy thật** → chưa biết VAD audeering có hơn SAILER không.

### 5. Việc tiếp
- Chạy full exp04 → train head → nộp → ghi điểm (so mốc EMOS 0.637 / ARO 0.712).
- Chạy exp05 (LIMIT=20 → None) → nộp → A/B 3 cột VAD với exp03; nếu audeering chỉ thắng VAL thì trộn cột.

---

## Báo cáo ngày 4/6/2026 (Phiên 2) — exp01 emotion2vec EMOS = 0.637 (VƯỢT SAILER!) + chốt hướng fusion multi-task

**Người thực hiện:** Tran Minh Toan · **Nội dung:** chạy + nộp exp01 (EMOS bằng emotion2vec), phân tích kết quả, học sâu metric/data, chốt hướng nghiên cứu fusion.

### 1. Kết quả exp01 — bước nhảy EMOS lớn thứ hai
- Chạy `track2_baseline.ipynb` với `EMOS_METHOD="emotion2vec"` (EMOS = 1+4·P(target) từ emotion2vec, offline) → nộp CodaBench.
- **Kết quả (DEV):** QMOS **0.4139** · **EMOS 0.6365** · CAT err **0.1933**.
- 🎉 **Bất ngờ lớn:** emotion2vec EMOS (**0.637**) **VƯỢT cả SAILER (0.562)** — dù emotion2vec bị "dồn 2 cực" (overconfident). Lý do: SRCC chấm **thứ hạng**, mà thứ hạng emotion2vec khớp người chấm tốt hơn; ties ở cực không hại nhiều.
- → **EMOS tốt nhất hiện tại = emotion2vec (0.637)**, không phải SAILER. Nhưng SAILER vẫn giữ VAD (ARO 0.712/DOM 0.630) mà emotion2vec không có.

### 2. Học nền (cho người mới + cho paper)
- **SRCC** = tương quan thứ hạng (xếp hàng từng câu theo độ khớp cảm xúc); khoảng cách điểm không quan trọng, chỉ thứ tự.
- **Data thật:** `train.csv` 91.121 lượt chấm / 12.746 câu (TB **7,15 người/câu**, min 4 max 9); nhãn vàng = **TB eMOS theo wav**. Người chấm bất đồng nhiều (1↔5 cho cùng câu) → trần khó của bài.
- **2024:** đội thắng dùng **fusion 2–3 luồng đặc trưng** (T05: SSL+spectrogram-as-image; PS-SQA: SSL+pitch+codec+ensemble), **không phải multi-task**. Metric chuẩn VoiceMOS = MSE/LCC/SRCC/KTAU (utt+system).
- **Cảm xúc là track MỚI 2026** (2022–2024 chỉ có chất lượng/naturalness/singing) → ít prior art → lợi thế novelty.

### 3. Chốt hướng nghiên cứu — FUSION multi-task (ghi `03_literature_notes.md`)
- **Bằng chứng để fusion:** emotion2vec thắng EMOS, SAILER thắng VAD → 2 model **bổ sung nhau** → gộp sẽ mạnh hơn.
- **Thiết kế CHỐT:** **"QMOS riêng + 5 cảm xúc chung"** — QMOS tách nhánh chất lượng (giữ SpeechMOS), EMOS/CAT/VAL/ARO/DOM dùng chung backbone fusion (emotion2vec+SAILER+tùy chọn WavLM) → head multi-task.
- **Khả thi T4:** ✅ nhờ freeze backbone + cache embedding (.npz ~150MB) + train head nhỏ (vài phút). Né fine-tune end-to-end lúc đầu.
- **Khó nhất = cân loss** (6 loss khác loại/thang). **Lưới an toàn:** answer.txt không cần 6 cột từ 1 model → worst case dùng bản lai.

### 4. Prior art cần xác minh (việc tiếp)
- Fusion cho **nhận diện** cảm xúc đã nhiều (multimodal A+T+V, multi-feature SER) → ý tưởng kỹ thuật không mới.
- Fusion cho **EMOS** (điểm MOS cảm xúc) gần như chưa ai làm → **cần web-search xác minh** trước khi khẳng định novelty.

### 5. Việc tiếp theo
- (1) Nộp **bản lai** (EMOS←emotion2vec, VAD←SAILER, QMOS←SpeechMOS) lấy điểm tổng tốt nhất.
- (2) Web-search prior art fusion-cho-EMOS.
- (3) Code fusion multi-task (5 cảm xúc) ở session mới (context sạch) → so mốc EMOS 0.637.

---

## Báo cáo ngày 4/6/2026 — exp03 SAILER: EMOS 0.194 → 0.562 🎉 + mở 3 cột VAD

**Người thực hiện:** Tran Minh Toan · **Nội dung:** khảo sát model thay emotion2vec (gợi ý mentor), viết + chạy + nộp exp03 dùng **SAILER**.

### 1. Khảo sát theo gợi ý mentor (emotion2vec đã cũ?)
- Đối chiếu **SenseVoice** (mentor đề xuất): chỉ ra **nhãn**, không ra xác suất, **không có VAD** → không hợp metric EMOS (cần điểm liên tục cho SRCC).
- Khảo sát model SER 2025–2026 (ghi vào `03_literature_notes.md`): trên IEMOCAP emotion2vec vẫn ngang/hơn WavLM-large ở cảm xúc rời rạc → "mới ≠ chắc tốt"; **khoảng trống thật là VAD**.
- Chốt: dùng **SAILER** (`tiantiaf/wavlm-large-categorical-emotion`, WavLM-large, **vô địch Interspeech 2025 SER**) — xuất xác suất 9 lớp (đủ 5 lớp challenge kể cả Surprise) + **xuất sẵn VAD**.

### 2. Viết pipeline exp03 + xử lý lỗi
- Tạo `kaggle_baseline/track2/exp03_emos_sailer.ipynb` (+ `_pipeline.py`). 1 model SAILER lo **EMOS + CAT + VAD**; QMOS giữ SpeechMOS.
- Lỗi đã fix: (a) `pip install -e .` của repo build wheel hỏng → đổi sang **clone + sys.path** + cài deps (`loralib, speechbrain`); (b) `forward(return_feature=True)` trả **6 giá trị** (không phải 2 như model card) → unpack đúng, đồng thời **lấy được VAD miễn phí**; (c) QMOS/SAILER đẩy lên **GPU T4** cho nhanh.
- Cũng đã **chia `kaggle_baseline/` theo track** (track1/ track2/ track3/) + cập nhật README.

### 3. Kết quả nộp CodaBench (DEV, 4/6) — bước nhảy lớn nhất Track 2
| Metric | Baseline (cũ) | **exp03 SAILER** |
|---|---|---|
| QMOS SRCC | 0.414 | 0.414 (giữ) |
| **EMOS SRCC** | 0.194 | **0.562** 🚀 |
| CAT err | 0.193 | 0.190 |
| VAL / ARO / DOM SRCC | — (bỏ) | **0.341 / 0.712 / 0.630** |

- EMOS tăng gần ×3; ARO/DOM mạnh; **VAL thấp nhất** (đúng literature — valence khó nhất với acoustic-only).

### 4. Lưu ý & việc tiếp
- ⚠️ **License SAILER = Open RAIL (phi thương mại)** → đã khai báo `12_`; phải nhắc trong system description cuối.
- Việc tiếp: (1) đẩy **VAL** bằng model VAD chuyên (audeering / tiantiaf MSP-dim); (2) thử **Cách B** train head trên SAILER embedding (vượt 0.562?); (3) cải thiện **QMOS** (vẫn 0.414).

---

## Báo cáo ngày 3/6/2026 — Phiên đêm (code train EMOS exp02 + cập nhật CLAUDE.md)

**Người thực hiện:** Tran Minh Toan · **Nội dung:** chốt thiết kế + viết code train EMOS có giám sát (exp02, giai đoạn EMOS-only).

### 1. Chốt thiết kế exp02 (giai đoạn EMOS-only)
- **Phạm vi:** train riêng **EMOS** trước (điểm yếu nhất hiện tại = 0.194), chưa làm multi-task — để dễ debug, nhanh có kết quả.
- **Kiến trúc:** backbone **emotion2vec ĐÓNG BĂNG** (chỉ trích đặc trưng, không train lại) + **MLP head nhỏ** train được.
  - Feature mỗi wav = `[embedding ~D | xác suất 5 lớp emotion2vec | one-hot target emotion]`.
  - Nhãn vàng = **trung bình `eMOS` mọi listener trên cùng wav** (gộp theo `wavID` từ `sets/train.csv`).
  - Vì EMOS phụ thuộc **cả audio LẪN cảm xúc target** → bắt buộc feed thêm one-hot target (đọc từ `metadata.csv`).
- **Môi trường:** Kaggle T4.

### 2. Viết code (chạy được, CHƯA chạy thật)
- Tạo `kaggle_baseline/exp02_train_emos_pipeline.py` + convert ra `kaggle_baseline/exp02_train_emos.ipynb` (jupytext). Cú pháp Python OK ✅.
- 7 mục: cấu hình → đọc/gộp nhãn → trích đặc trưng emotion2vec (**có cache .npz**) → dựng feature → train MLP (đo **SRCC** trên 10% validation nội bộ, early-stopping, lưu `emos_head.pt`) → dự đoán DEV ra `answer.txt` đầy đủ (QMOS=SpeechMOS, CAT=emotion2vec, EMOS=head) → validate + zip.
- **Chưa chạy trên Kaggle** → chưa có VAL SRCC / điểm leaderboard. Việc tiếp theo: Run All trên Kaggle (thử `LIMIT_TRAIN=300` trước, rồi `None`).

### 3. Cập nhật quy ước
- Sửa `CLAUDE.md` mục 3: ghi rõ model **Opus 4.8** (`claude-opus-4-8`) / cửa sổ **~1M token**, thêm cột mốc token tuyệt đối (🟢<500k · 🟡500–700k · 🟠700–850k · 🔴>850k), sửa lỗi format khối code.

### 4. Câu hỏi mentor (đã tổng hợp — xem `02_mentor_questions.md`)
- 🔴 Hướng **novelty** Track 2 + **contribution tối thiểu** cho ICASSP 2027.
- 🟡 Train head trên emotion2vec đóng băng có đủ mới không, hay nên **fine-tune backbone / đổi WavLM**? · EMOS-only hay **multi-task** ngay? · Có nên dùng **public data** (IEMOCAP/MSP-Podcast)?
- 🟢 Co-author? · Báo cáo theo tuần hay milestone?

### 5. Kế hoạch session sau
- Chạy `exp02_train_emos.ipynb` trên Kaggle → ghi VAL SRCC + nộp `answer.txt` → cập nhật điểm EMOS mới vào `04_` + `12_`.
- Nếu khá → mở rộng thành **multi-task** đầy đủ (thêm head QMOS/CAT/VAD dùng chung backbone).

---

## Báo cáo ngày 3/6/2026 — Phiên tối (sắp xếp dự án + exp01 EMOS + mở khóa train)

**Người thực hiện:** Tran Minh Toan · **Nội dung:** dọn cấu trúc dự án, khởi động viết paper, làm exp01 (EMOS offline), chốt chiến lược 3 track.

### 1. Sắp xếp lại toàn bộ cấu trúc thư mục
- Gom: tài liệu → `docs/` (00..15), repo baseline → `baselines/`, tham khảo → `reference/` (`content_btc/` + `understand/`), `Submission/` → `submissions/`. `CLAUDE.md` + `README.md` giữ ở gốc.
- Cập nhật theo: `.gitignore` (`baselines/`), `CLAUDE.md` (link `docs/` + bản đồ file mới), `README.md` (sơ đồ thư mục), và các sơ đồ/đường dẫn nội bộ trong docs. **Chưa commit** (chờ user).

### 2. Bắt đầu viết paper — tạo `docs/15_paper_draft.md`
- Khung ICASSP 2027 đủ 6 mục, rót sẵn data/metric/kết quả baseline. Câu chuyện trung tâm: "baseline chắp vá 3 model → thay bằng 1 model multi-task". Quy ước: chạy experiment tới đâu đổ kết quả vào đây tới đó.

### 3. exp01 — EMOS offline bằng emotion2vec (không train)
- Sửa **cả pipeline `.py` lẫn notebook** `track2_baseline.ipynb`: thêm công tắc `EMOS_METHOD="emotion2vec"` (mặc định). EMOS = `1 + 4·P(cảm xúc target)` lấy từ `emocat_probs` (emotion2vec đã tính cho CAT) → offline, miễn phí, chấm đủ 2.730 mẫu (trước 82% mặc định = 3 kéo EMOS xuống 0.194). Giữ nhánh Gemini làm tùy chọn.
- **Chạy thử LIMIT=20 OK:** kiểm chứng công thức đúng (EMOS khớp `1+4·P(target)`), tất cả mẫu có điểm thật. Quan sát: emotion2vec rất "chắc nịch" → EMOS dồn 2 cực ~1 và ~5 (có thể giới hạn SRCC, nhưng đúng chiều → kỳ vọng > 0.194). **Chờ chạy full + nộp.**

### 4. Chốt chiến lược "train qua data" cho 3 track
- **Track 1:** ❌ **KHÔNG có official training data** (dev set còn giấu nhãn) → không thể train in-domain; muốn cải thiện phải dùng public data ngoài (NISQA/URGENT...). Loại khỏi hướng "train".
- **Track 3:** ✅ có train data (2.800 cặp) + sẵn `finetune.py` → chỗ tập train tốt, nhưng chỉ chạy mặc định ≈ tái tạo 0.451/0.440 (điểm hiện tại là dùng checkpoint cho sẵn).
- **Track 2:** ✅ nơi "train qua data" phát huy đúng nhất (track chính + novelty).

### 5. 🎉 Mở khóa exp02 — `sets/train.csv` ĐỦ NHÃN
- Xác nhận `/kaggle/input/datasets/minhtoan2/vmc2026-track2-full/sets/train.csv` có cột: `lisID, wavID, qMOS, emoCat, eMOS, val, dom, aro` → train được **multi-task** (QMOS/EMOS/CAT/VAD). Điểm theo từng listener (phải gộp TB theo wav như Track 3); `emoCat` đa nhãn → CAT = tỉ lệ vote 5 lớp; target lấy từ `metadata.csv`.

### 6. Kế hoạch session sau
- Chạy full exp01 (`LIMIT=None`) → nộp Track 2 → ghi điểm EMOS mới.
- Xây **exp02** (multi-task có train) trong session mới (cần context sạch).

---

## Báo cáo ngày 3/6/2026 — Phiên chiều (định hướng kỹ thuật, chưa chạy code)

**Người thực hiện:** Tran Minh Toan · **Nội dung:** rà soát kiến trúc baseline + chốt hướng cải tiến Track 2.

### 1. Hiểu rõ "mỗi cột leaderboard ← model nào" (baseline hiện tại)
| Cột | Model đang dùng | Điểm |
|---|---|---|
| T1 ACR + CCR | **URGENT-MOS** (1 model → 2 cột) | 0.662 / 0.411 |
| T2 QMOS | UTMOS/SpeechMOS | 0.414 |
| T2 EMOS + VAD | **Gemini** (1 hộp đen → 4 cột) | EMOS 0.194; VAD chưa chạy đủ |
| T2 CAT | emotion2vec | err 0.193 |
| T3 SPK + ACC | **ECAPA-TDNN** (1 model → 2 cột) | 0.451 / 0.440 |
→ Baseline Track 2 là kiểu **chắp vá 3 model rời**; 4/6 cột dồn vào Gemini (tốn API, zero-shot → yếu).

### 2. Chốt hướng cải tiến Track 2 (điểm yếu nhất = EMOS 0.194)
- **Cách A (làm ngay, miễn phí, offline):** bỏ Gemini cho EMOS → lấy **xác suất lớp cảm xúc target từ emotion2vec** (model đã dùng cho CAT) làm điểm EMOS. Không train, không API.
- **Cách B (mạnh nhất, để viết paper):** **train regressor có giám sát** trên **12.746 mẫu có nhãn EMOS** — kiến trúc **multi-task: 1 backbone chung (wav2vec2/WavLM/emotion2vec) + nhiều head** (QMOS/EMOS/CAT/VAD). Một model thay được cụm Gemini 4 cột.
- Làm rõ khái niệm: **không train emotion2vec để "nhận cảm xúc"** (nó biết sẵn); cái train là **head quy đặc trưng → điểm cảm nhận của người**. EMOS = mức độ khớp (regression), ≠ phân loại cảm xúc.

### 3. Việc dọn dẹp & công cụ
- Phát hiện 2 folder **`UTMOS22/` + `emotion2vec/`** (≈23 MB) **dư cho workflow Kaggle** (pipeline tải model qua torch.hub/funasr, không đọc bản local; đã trong `.gitignore`, 0 file tracked). → **chờ xác nhận xóa.**
- Đã tạo **`CLAUDE.md`** ở gốc dự án: quy ước "đọc" đầu phiên + "xong" cuối phiên (tự cập nhật md) + cảnh báo token cuối mỗi câu trả lời.
- Xác nhận: **challenge KHÔNG bắt buộc dùng baseline** — grader chỉ chấm `answer.txt`; tự do tự phát triển hệ thống (chỉ cần khai báo public data đã dùng).

### 4. Việc tiếp theo (đề xuất)
1. Sửa pipeline Track 2 → **EMOS = emotion2vec target-prob** (cách A) → nộp lại, kỳ vọng EMOS > 0.194.
2. Lên khung **exp01 multi-task** (xem `04_experiments_log.md`).

---

## Báo cáo ngày 3/6/2026

**Người thực hiện:** Tran Minh Toan
**Track:** Track 2 — Emotional TTS (chính); Track 1 & 3 — baseline/demo

### 1. Data — đã nhận đủ cả 3 track ✅
- ✅ **Track 2:** BTC xác nhận license → nhận gói data; cộng ESD + DailyTalk (Kaggle dataset) → ráp đủ **15.477 wav** qua `track2_prepare_data.ipynb`.
- ✅ **Track 3:** nhận nốt gói **VCTK** (mảnh cuối, tách riêng do license VCTK). Ghép `_vctk` (296 wav: sys008+sys019) vào `_syn` (3.252 wav) → **3.548 wav**. Train 2.800 cặp / 13.687 rating / 25 listener · dev 600 cặp.
- → **Hết bị chặn về data**; cả 3 track sẵn sàng chạy baseline thật.

### 2. Đã chạy & nộp baseline — có điểm trên leaderboard ✅
- ✅ **Track 1** (URGENT-MOS): nộp xong.
- ✅ **Track 2** (QMOS=SpeechMOS · CAT=emotion2vec · EMOS=Gemini): tối ưu pipeline (chạy GPU, lọc metadata về DEV để tiết kiệm credit) → `answer.txt` → nộp.
- ✅ **Track 3** (ECAPA-TDNN fine-tuned): ráp 3.548 wav (gộp VCTK) → inference 600 cặp dev → nộp.
- ✅ Đã ghi điểm vào `04_experiments_log.md` + `12_system_description.md`.

| Track | Metric | Điểm (DEV) |
|---|---|---|
| Track 1 | ACR / CCR UTT-SRCC | **0.662** / **0.411** |
| Track 2 | QMOS UTT-SRCC | **0.414** |
| Track 2 | EMOS UTT-SRCC | **0.194** (một phần) |
| Track 2 | CAT UTT-ERR | **0.193** (thấp = tốt) |
| Track 3 | SPK / ACC UTT-SRCC | **0.451** / **0.440** |

→ **Cả 3 track đã có điểm trên leaderboard.**

### 3. Khó khăn / cần hỗ trợ
- **Gemini free tier = `limit: 0`** → buộc bật **billing trả phí** (prepaid). Để tiết kiệm, dừng sớm ở **496/2.730 mẫu EMOS** → EMOS bị kéo thấp. Sẽ `--resume` chấm nốt (ước ~$0,5–1,5 cả tập) rồi nộp lại.
- ⬜ Hướng **novelty Track 2** vẫn chờ trao đổi mentor (xem `02_mentor_questions.md`).

### 4. Kế hoạch tuần tới
- Hoàn thiện EMOS Track 2 (chấm đủ 2.730) → nộp lại.
- Đọc UTMOS + WavLM → chốt hướng cải tiến QMOS vượt baseline 0.414.
- Trao đổi mentor về hướng novelty Track 2.

---

## Báo cáo ngày 1/6/2026

**Người thực hiện:** Tran Minh Toan
**Track:** Track 2 — Emotional TTS (chính); Track 1 & 3 — baseline/demo

### 1. Thủ tục & hạ tầng — đã xong
- ✅ Đăng ký challenge + **join CodaBench** (competition 16419).
- ✅ Chốt compute: dùng **Kaggle T4** (theo gợi ý của thầy/cô).
- 🔄 Đã gửi **license form Track 2 & 3** cho BTC (Erica Cooper). đang chờ xác nhận để nhận link data.

### 2. Nghiên cứu yêu cầu đề bài — đã xong
- ✅ Đọc & hệ thống hóa đặc tả **cả 3 track**: dataset, kích thước (Track 2: train 12.746 / val 2.730 / eval 2.730), 6 metric (QMOS/EMOS/VAD SRCC + categorical error), format nộp (`answer.txt`).
- ✅ Xác định dataset nền Track 2 = **ESD + DailyTalk** + 13 hệ thống TTS; đã có sẵn 2 bộ này qua Kaggle Dataset.

### 3. Baseline — đã dựng pipeline
- ✅ Viết pipeline chạy trên Kaggle cho **cả 3 track** (notebook sẵn sàng).
- ✅ **Track 1** (URGENT-MOS): chạy được ngay (data công khai, checkpoint pre-trained) → chuẩn bị nộp lần đầu (thỏa luật "nộp ≥1 lần trong training phase").
- ✅ **Track 3** (ECAPA): có code + checkpoint train sẵn → sẵn sàng cho mục đích demo UI.
- ✅ **Track 2**: pipeline QMOS (SpeechMOS) + EmoCat (emotion2vec) test được trên ESD ngay; EMOS/VAD (Gemini) chờ data.

### 4. Đang làm / chờ
- ⏳ Chờ data Track 2/3 (license).
- 🔄 Bắt đầu đọc literature (UTMOS, WavLM) + xây nền kiến thức.

### 5. Câu hỏi cần định hướng
- **Hướng novelty Track 2** nên đi theo đâu? (cân nhắc: fusion SSL backbone + emotion embedding, hoặc multi-task QMOS+EMOS)
- **Backbone** nên chọn WavLM hay HuBERT/Wav2Vec2 cho emotion MOS?
- Thầy/cô muốn báo cáo **theo tuần hay theo milestone**?

---

<!-- Mẫu cho báo cáo sau — copy block dưới:

## Báo cáo ngày DD/MM/YYYY
### Đã làm tuần này
-
### Kết quả / số liệu
-
### Khó khăn / cần hỗ trợ
-
### Kế hoạch tuần tới
-
-->
