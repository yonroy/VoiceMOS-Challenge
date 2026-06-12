# CLAUDE.md — Quy ước làm việc cho dự án VoiceMOS Challenge 2026

> File này là **luật** cho Claude trong mọi session của dự án. Đọc kỹ và tuân thủ.
> Người dùng là **người mới (beginner)** về ML/speech → giải thích đơn giản, định nghĩa thuật ngữ, gắn lý thuyết vào dự án thực tế.

---

## 0. Ngôn ngữ & phong cách
- Trả lời bằng **tiếng Việt**.
- Giải thích thuật ngữ ngắn gọn khi dùng lần đầu (vì người dùng mới học).
- Khi tham chiếu file, dùng link Markdown bấm được, ví dụ [13_daily_todo.md](docs/13_daily_todo.md).

---

## 1. 🟢 KHI BẮT ĐẦU SESSION — "đọc" để hiểu dự án đang ở đâu

**Mỗi khi mở session mới, TRƯỚC KHI làm bất cứ việc gì, Claude phải đọc theo thứ tự sau** rồi tóm tắt lại cho người dùng "dự án đang ở đâu":

| Thứ tự | File | Đọc để biết |
|---|---|---|
| 1 | [07_project_summary.md](docs/07_project_summary.md) | Bức tranh tổng thể: dự án là gì, 3 track, timeline, deadline, điểm leaderboard mới nhất |
| 2 | [11_progress_reports.md](docs/11_progress_reports.md) | **Báo cáo mới nhất nằm trên cùng** → biết phiên gần nhất đã làm gì |
| 3 | [13_daily_todo.md](docs/13_daily_todo.md) | Việc của hôm nay: cái nào `[x]` xong, cái nào `[ ]` còn dang dở |
| 4 | [04_experiments_log.md](docs/04_experiments_log.md) | Bảng kết quả thí nghiệm + điểm hiện có (nguồn để viết paper) |
| 5 | [12_system_description.md](docs/12_system_description.md) | Mô tả hệ thống + bảng điểm từng track (bản nộp leaderboard) |
| 6 | [14_leaderboard_metrics.md](docs/14_leaderboard_metrics.md) | Cách tính metric (UTT-SRCC...) để hiểu điểm |

**Khi đọc xong, Claude phải in ra một khối tóm tắt** dạng:

```
📌 DỰ ÁN ĐANG Ở ĐÂU (cập nhật từ các file md)
- Track trọng tâm: Track 2 (Emotional TTS)
- Điểm leaderboard mới nhất: T1 …/… · T2 QMOS … · T3 …/…
- Deadline nộp kết quả: 7/8/2026 (còn ~X tuần)
- Phiên trước làm gì: <lấy từ báo cáo mới nhất trong 11_>
- Việc dang dở hôm nay: <các mục [ ] trong 13_>
- Đề xuất việc tiếp theo: …
```

> Nếu người dùng nói **"đọc"** hoặc **"dự án đang ở đâu"** → chạy lại đúng quy trình mục 1 này.

---

## 2. 🔴 KHI KẾT THÚC SESSION — người dùng gõ "xong"

Khi người dùng gõ **"xong"** (hoặc "kết thúc", "đóng session", "cập nhật md"), Claude **TỰ ĐỘNG** cập nhật toàn bộ thông tin của session vào các file md. Làm **đầy đủ** các bước sau, không bỏ bước nào:

### Bước 1 — Tổng hợp session
Tự rà lại toàn bộ cuộc hội thoại của session này và liệt kê (ngắn gọn) cho người dùng xác nhận:
- Đã làm gì (việc, lệnh, file đã sửa)
- Kết quả/điểm số mới (nếu có)
- Vấn đề gặp phải + cách xử lý
- Việc còn dang dở

### Bước 2 — Ghi vào các file md tương ứng

| Nếu trong session có… | → Cập nhật file | Cách ghi |
|---|---|---|
| Bất kỳ tiến độ nào (luôn luôn) | [11_progress_reports.md](docs/11_progress_reports.md) | Thêm **báo cáo mới ở TRÊN CÙNG** (dưới dòng `---` đầu), tiêu đề `## Báo cáo ngày D/M/2026`, theo đúng format các báo cáo cũ |
| Chạy thí nghiệm / có điểm mới | [04_experiments_log.md](docs/04_experiments_log.md) | Thêm/điền hàng vào **Bảng tổng hợp** + thêm mục chi tiết `### expNN` (config → kết quả → nhận xét) |
| Điểm leaderboard thay đổi | [12_system_description.md](docs/12_system_description.md) **và** dòng trạng thái đầu [07_project_summary.md](docs/07_project_summary.md) | Cập nhật bảng điểm từng track |
| Hoàn thành / phát sinh việc | [13_daily_todo.md](docs/13_daily_todo.md) | Tick `[x]` việc đã xong; thêm việc mới phát sinh; nếu sang ngày mới thì tạo mục todo ngày mới |
| Học được kiến thức/đọc paper | [03_literature_notes.md](docs/03_literature_notes.md) hoặc [10_learning_roadmap.md](docs/10_learning_roadmap.md) | Ghi 3–5 dòng ghi chú |
| Đổi kế hoạch/chiến lược | [01_research_plan.md](docs/01_research_plan.md) | Cập nhật mục liên quan |

### Bước 3 — Cập nhật ngày tháng
- Đổi dòng "Cập nhật ngày: …" ở đầu file đã sửa sang ngày hiện tại.
- Ngày hiện tại lấy từ context (`currentDate`). Quy đổi ngày tương đối ("hôm nay", "mai") thành ngày tuyệt đối.

### Bước 4 — Lưu memory (nếu có thông tin bền vững)
Nếu trong session phát sinh **sự thật mới đáng nhớ qua nhiều phiên** (vd: dataset mới upload, điểm mốc quan trọng, thay đổi hướng đi) → ghi 1 file vào thư mục memory và thêm 1 dòng vào `MEMORY.md`. Không ghi lại thứ đã có trong code/git.

### Bước 5 — Báo cáo lại
In ra **danh sách file đã cập nhật** + tóm tắt 1 dòng mỗi file. KHÔNG tự `git commit`/`git push` trừ khi người dùng yêu cầu rõ.

> **Nguyên tắc:** thà ghi dư còn hơn mất thông tin. Mọi điểm số, lệnh đã chạy, lỗi đã gặp đều phải vào md để phiên sau "đọc" lại được.

---

## 2B. 🧹 KHI NGƯỜI DÙNG GÕ "dọn dẹp" — rà soát & làm sạch thư mục `docs/`

Khi người dùng gõ **"dọn dẹp"** (hoặc "rà soát docs", "làm sạch tài liệu"), Claude **rà soát toàn bộ thư mục `docs/`** để tìm và xử lý những thông tin **gây nhiễu/bối rối cho phiên sau**. Mục tiêu: giữ `docs/` là **nguồn sự thật nhất quán** để quy trình "đọc" (mục 1) không bị hiểu sai.

### Bước 1 — Rà soát (chỉ đọc, chưa sửa)
Quét các file `docs/` và liệt kê các vấn đề theo nhóm:
- **Mâu thuẫn số liệu:** cùng một điểm (QMOS/EMOS/VAD...) nhưng khác nhau giữa các file (vd `04_` vs `07_` vs `12_`), hoặc "điểm tốt nhất hiện tại" không khớp nhau.
- **Thông tin lỗi thời:** trạng thái cũ chưa cập nhật (vd "chờ chạy" nhưng đã chạy xong), deadline/đếm tuần sai so với `currentDate`, "việc tiếp theo" đã làm xong nhưng vẫn để ngỏ.
- **Tham chiếu hỏng:** tên file/folder/đường dẫn không còn tồn tại (vd tên cũ sau khi đổi tên), link Markdown chết.
- **Trùng lặp/chồng chéo:** cùng một nội dung lặp ở nhiều file dễ lệch nhau khi sửa.
- **Ngày tháng sai:** dòng "Cập nhật ngày: …" cũ hơn nội dung thực; ngày tương đối chưa quy đổi.

### Bước 2 — Báo cáo trước khi sửa
In ra danh sách phát hiện (file → vấn đề → đề xuất sửa). **Tuân thủ mục 5:** trước khi xóa/ghi đè nội dung md, đọc kỹ; nếu nội dung **mâu thuẫn thật** (không rõ bản nào đúng) thì **hỏi người dùng** thay vì tự ý ghi đè/xóa.

### Bước 3 — Sửa các lỗi rõ ràng
Tự sửa những lỗi **không cần phán đoán** (đồng bộ số liệu theo nguồn chuẩn = `04_experiments_log.md`, sửa tham chiếu hỏng, cập nhật ngày, tick todo đã xong). Với thay đổi có rủi ro mất thông tin → chờ người dùng xác nhận.

### Bước 4 — Báo cáo lại
In danh sách file đã sửa + tóm tắt 1 dòng mỗi file. **KHÔNG** tự `git commit` trừ khi được yêu cầu.

> **Nguyên tắc:** "dọn dẹp" là **làm sạch để nhất quán**, KHÔNG phải xóa lịch sử. Báo cáo tiến độ cũ (`11_`) và nhật ký thí nghiệm (`04_`) giữ nguyên dòng thời gian; chỉ sửa chỗ sai/mâu thuẫn, không cô đọng mất dữ liệu.

---

## 3. ⚠️ CUỐI MỖI CÂU TRẢ LỜI — cảnh báo token context

> **Model & cửa sổ context của dự án:** Claude **Opus 4.8** (`claude-opus-4-8`) — cửa sổ context **~1.000.000 (1M) token**. Mọi ước lượng % và mốc token dưới đây tính trên mẫu số 1M này.

**Mọi câu trả lời** (không chỉ khi "xong") đều phải **kết thúc bằng một dòng cảnh báo mức dùng context**, theo format:

```
🧮 Context: ~XX% (≈ N nghìn / 1M token) — <trạng thái>
```

Quy ước trạng thái (cửa sổ context của Opus 4.8 = ~1M token):

| Mức dùng | Mốc token (trên 1M) | Biểu tượng | Trạng thái & hành động |
|---|---|---|---|
| < 50% | < 500 nghìn | 🟢 | An toàn, cứ làm tiếp |
| 50–70% | 500–700 nghìn | 🟡 | Bắt đầu chú ý, cân nhắc gói gọn việc |
| 70–85% | 700–850 nghìn | 🟠 | **Nên sớm gõ "xong"** để Claude lưu md trước khi mất ngữ cảnh |
| > 85% | > 850 nghìn | 🔴 | **CẢNH BÁO: nên đóng cửa sổ ngữ cảnh ngay.** Gõ "xong" để lưu, rồi mở session mới |

**Lưu ý trung thực:** Claude **không đo được chính xác** số token đang dùng → đây là **ước lượng** dựa trên độ dài hội thoại (số lượt, độ dài file đã đọc, output đã sinh). Luôn ghi kèm dấu `~`/`≈` để rõ là ước lượng, và khi đã chạm 🟠/🔴 thì chủ động nhắc người dùng gõ "xong".

---

## 4. Bản đồ file dự án (tham chiếu nhanh)

> 📁 **Cấu trúc thư mục (sắp xếp lại 3/6/2026):** tài liệu nằm trong `docs/`, code trong `kaggle_baseline/`, repo baseline trong `baselines/`, dữ liệu trong `data/`, file nộp trong `submissions/`, tài liệu tham khảo trong `reference/`. **`CLAUDE.md` và `README.md` ở thư mục gốc.**

| File / thư mục | Vai trò |
|---|---|
| `docs/00_challenge_overview.md` | Tổng quan challenge |
| `docs/01_research_plan.md` | Kế hoạch nghiên cứu |
| `docs/02_mentor_questions.md` | Câu hỏi cho mentor |
| `docs/03_literature_notes.md` | Ghi chú paper đã đọc |
| `docs/04_experiments_log.md` | **Nhật ký thí nghiệm** (nguồn cho paper) |
| `docs/05_setup_environment.md` | Cài đặt môi trường |
| `docs/06_baseline_repos.md` | Các repo baseline đã clone |
| `docs/07_project_summary.md` | **Tóm tắt tổng thể** (đọc đầu tiên) |
| `docs/08_track2_spec.md` | Đặc tả Track 2 |
| `docs/09_tracks_overview.md` | Tổng quan 3 track |
| `docs/10_learning_roadmap.md` | Lộ trình học |
| `docs/11_progress_reports.md` | **Báo cáo tiến độ** (mới nhất trên cùng) |
| `docs/12_system_description.md` | Mô tả hệ thống + bảng điểm |
| `docs/13_daily_todo.md` | **Todo hằng ngày** |
| `docs/14_leaderboard_metrics.md` | Cách tính metric |
| `docs/15_paper_draft.md` | **Bản nháp paper ICASSP 2027** |
| `docs/18_leaderboard_history.md` | **Lịch sử leaderboard qua các ngày** (best-per-column + từng bản nộp) |
| `docs/19_paper_v1_en.md` | **Bản paper v1 (TIẾNG ANH)** — start version để nộp ICASSP (15_ là nháp tư duy tiếng Việt) |
| `docs/20_experiments_overview.md` | **Bảng trạng thái nhanh các exp** (đã nộp / đã chạy / mới code) — nhìn 1 phát biết đã làm gì |
| `docs/21_slides_3_tracks.md` | **Slide present 3 track** (mentor giao) — Marp markdown tiếng Việt + 3 hình kiến trúc SVG inline; render cần bật HTML (`enableHtml`/`--html`) — bản v1 ngắn |
| `docs/22_slides_v2_paper_style.md` | **Slide v2 paper-style (~36 slide)** — bản đầy đủ: bài toán + cách chấm (SRCC/CAT-ERR ví dụ tính tay) + bảng từng layer cả 3 track + training details + ablation + số liệu 10/6; render như 21_ → `slide/voicemos2026_slides_v2.html` |
| `kaggle_baseline/` | Notebook + pipeline chạy baseline 3 track + demo Gradio. **Track 2 đặt tên theo experiment: `expNN_tên.{ipynb,py}`** (vd `exp04_fusion`, `exp07_fusion_qmos`); `track2_baseline`/`track2_prepare_data`/`demo_*` giữ nguyên (không phải experiment đơn) |
| `baselines/` | Repo baseline clone: `vmc2026-baselines/` (gốc BTC), `UTMOS22/`, `emotion2vec/` |
| `data/` | Dữ liệu thô (không commit; upload lên Kaggle) |
| `submissions/` | File nộp + kết quả chấm (Track1/2/3) |
| `reference/` | Tài liệu tham khảo: `content_btc/` (text BTC), `understand/` (ghi chú) |

---

## 5. Quy ước khác
- Người dùng chạy nặng trên **Kaggle** (GPU T4), không chạy local → khi hướng dẫn thao tác, mô tả theo môi trường Kaggle (Add Input, Internet On/Off, Save Version...).
- Mỗi thí nghiệm: **luôn ghi config → kết quả → nhận xét** vào `04_`. "Không bao giờ chạy mà không ghi."
- Trước khi xóa/ghi đè file md có nội dung, đọc nó trước; nếu nội dung mâu thuẫn với mô tả thì báo lại thay vì ghi đè.
