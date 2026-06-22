# Script thuyết trình — VoiceMOS Challenge 2026 (deck Swiss Modern)

> Lời nói tự nhiên, dễ hiểu cho từng slide. Bám đúng nội dung deck `voicemos2026_swiss.html` (23 slide, thứ tự **Track 1 → Track 3 → Track 2**).
> Mỗi slide: **(nói gì)** + *[mẹo]*. Bấm **← →** (hoặc Space) chuyển slide, **F** toàn màn hình. Tổng ~11–13 phút.
>
> ⚠️ **CHÍNH XÁC — đọc trước:** hệ thực nộp **chỉ fine-tune WavLM**; **audeering + UTMOS ĐÓNG BĂNG** (chỉ trích đặc trưng, cache sẵn). Slide *Training details* ghi đúng điều này. Nhưng slide **"Kiến trúc Track 2"** và **"Per-layer Track 2"** đang tô **audeering 🔥 (như đang fine-tune)** → **SAI, cần sửa hình** (đổi 🔥→❄, bỏ chữ "FINE-TUNE" ở audeering, sửa legend "Cả 2 encoder fine-tune" → "WavLM fine-tune, audeering đóng băng"). Khi nói, cứ theo bản đúng dưới đây.

---

## Slide 1 — Cover ("Machine MOS")
"Em chào anh chị và mọi người. Em là Toàn. Hôm nay em xin trình bày dự án **VoiceMOS Challenge 2026** — một cuộc thi quốc tế về việc **dạy máy tính tự chấm điểm giọng nói thay cho con người**. Cuộc thi có 3 phần thi; em sẽ đi lần lượt và dừng lâu nhất ở phần 2 vì đó là trọng tâm của em."

*[Mẹo: nói chậm, tự tin. Một câu giới thiệu là đủ rồi sang slide. Tiêu đề lớn 'Machine MOS' chính là ý này.]*

---

## Slide 2 — Tổng quan 3 track
"Đầu tiên là bức tranh tổng thể. Cả 3 phần thi đều chung một mục tiêu — **đoán trước điểm mà người nghe sẽ chấm cho một giọng nói** — chỉ khác nhau ở loại giọng:
- **Track 1**: giọng đã *khử nhiễu*, chấm xem nghe có sạch không — đạt ACR 0.662.
- **Track 2** ⭐: giọng *do máy tạo kèm cảm xúc*, chấm cả chất lượng lẫn cảm xúc — EMOS 0.811. Đây là phần chính.
- **Track 3**: chấm xem giọng máy có *giống giọng gốc* không — quanh 0.45.

Cột bên phải là điểm tốt nhất của em ở mỗi phần."

*[Mẹo: chỉ tay theo từng dòng. Chưa giải thích con số vội — hai slide sau sẽ nói 'điểm' nghĩa là gì.]*

---

## Slide 3 — Cách chấm (1): MOS
"Vậy 'điểm' ở đây là gì? Khái niệm đầu tiên là **MOS — điểm trung bình ý kiến**. Quy trình ở hình bên phải: cho **nhiều người cùng nghe một audio**, mỗi người chấm từ 1 đến 5 theo cảm nhận; rồi **lấy trung bình** ra một con số — ví dụ ở đây là 4.25. Việc của em là **huấn luyện máy đoán trước con số đó**, để khỏi phải tổ chức cho người nghe lại — vừa nhanh, vừa rẻ."

*[Mẹo: chỉ vào hình '4 người → trung bình → MOS 4.25'. Một câu chốt: 'máy đoán thay con người'.]*

---

## Slide 4 — Cách chấm (2): SRCC
"Khái niệm thứ hai là **SRCC** — cách ban tổ chức chấm bài của em, nằm trong khoảng từ âm 1 đến 1, càng gần 1 càng tốt. Điểm này rộng lượng ở chỗ: **không bắt máy đoán đúng y con số**, mà chỉ cần **xếp đúng thứ tự** — giọng nào hay hơn giọng nào.

Nhìn bảng: máy đoán 4.4 còn người chấm 4.6 — lệch một chút, nhưng **thứ tự audio 1 hơn 2 hơn 3 hơn 4 vẫn đúng**, nên SRCC xấp xỉ 1, tức rất tốt."

*[Mẹo: slide 'chìa khóa' — các slide sau toàn nhắc SRCC. Nhấn mạnh: 'chỉ cần xếp đúng thứ tự, không cần đúng con số'.]*

---

## Slide 5 — Cách chấm (3): UTT-SRCC vs SYS-SRCC (ví dụ tính tay)
"Còn một chi tiết — có **hai mức** chấm SRCC, và cuộc thi dùng mức khó. Em lấy ví dụ tính tay bằng công thức Spearman ở bên phải.
- **UTT-SRCC** chấm **từng audio một** — ở đây 6 audio, có vài chỗ máy xếp sai thứ tự, nên ra **0.886**.
- **SYS-SRCC** chấm **trung bình mỗi hệ thống** trước rồi mới so — chỉ còn 3 con số, các lỗi nhỏ bị trung bình hóa mất, nên ra **1.000**.

Tức là **UTT phạt nặng hơn** vì xét tới từng câu. Cuộc thi dùng UTT, nên đó là cột chính em phải tối ưu."

*[Mẹo: KHÔNG cần đọc hết phép tính. Chỉ cần ý: 'UTT chấm từng câu nên khó hơn, và đó là cột chấm chính'. Ai thích toán thì chỉ vào công thức.]*

---

## Slide 6 — Divider Track 1
"Mình bắt đầu với **Track 1 — chấm chất lượng giọng đã khử nhiễu**, hệ thống tên URGENT-MOS."

*[Mẹo: slide chuyển cảnh, nói một câu rồi sang luôn.]*

---

## Slide 7 — Track 1: Bài toán & dữ liệu
"Track 1 yêu cầu hai việc:
- **ACR**: chấm chất lượng tuyệt đối của *một* đoạn, thang 1 đến 5.
- **CCR**: *so sánh hai* đoạn xem cái nào tốt hơn, thang từ âm 3 đến cộng 3.

Cái khó là dữ liệu trải tới 9 ngôn ngữ và **hoàn toàn không có dữ liệu để huấn luyện**. Nên ở phần này em không train gì cả — em dùng thẳng mô hình có sẵn URGENT-MOS và chỉ chạy dự đoán."

*[Mẹo: nhấn 'không có data train' — đó là lý do điểm Track 1 khiêm tốn hơn Track 2.]*

---

## Slide 8 — Track 1: Kiến trúc (URGENT-MOS)
"Cách hệ thống làm việc, đọc từ trái sang phải: sóng âm đi vào, qua **CNN** để cắt thành các mẩu nhỏ; rồi qua **Transformer** để mỗi mẩu 'nghe' được cả câu, hiểu ngữ cảnh; sau đó một bước **trộn lớp và gộp** thành một vector đại diện cho cả câu. Thực ra URGENT-MOS chạy **bốn 'đôi tai' như vậy song song** rồi hợp nhất ở dấu cộng. Cuối cùng tách ra hai đầu ra: **ACR** — ô đỏ, mạnh nhất, 0.662 — và **CCR**, 0.411."

*[Mẹo: đừng giảng sâu CNN/Transformer — chỉ cần 'cắt mẩu', 'nghe cả câu', 'gộp 4 đôi tai → 2 điểm'. Ô đỏ = điểm chính.]*

---

## Slide 9 — Track 1: Per-layer
"Bảng này là chi tiết từng tầng cho ai muốn xem kỹ: CNN cắt mẩu, Transformer trộn ngữ cảnh, trộn lớp tự chọn tầng hữu ích, gộp lại, rồi hai đầu ra ACR và CCR. Em không đi sâu, mọi người xem bảng là đủ."

*[Mẹo: slide tham khảo — lướt nhanh 15 giây.]*

---

## Slide 10 — Divider Track 3
"Tiếp theo em xin nhảy sang **Track 3 — đo độ giống giọng**. Em cố tình để Track 2 ở cuối vì đó là phần đáng nói nhất."

*[Mẹo: chuyển cảnh, một câu.]*

---

## Slide 11 — Track 3: Bài toán
"Track 3 đặt câu hỏi: giọng máy tạo ra có **giống giọng gốc** không? Xét hai khía cạnh — giống về **người nói**, và giống về **chất giọng vùng miền** (accent). Cách làm của em là *zero-shot*, tức dùng luôn mô hình có sẵn, không train, rồi đo độ giống bằng một phép toán đơn giản. Một hạn chế em thấy rõ: mô hình này giỏi nhận *người nói* nhưng chưa tách riêng *accent* — đó là hướng cải tiến sau."

*[Mẹo: nói rõ 'zero-shot = dùng luôn, không train'. Câu hạn chế thể hiện mình hiểu sâu.]*

---

## Slide 12 — Track 3: Kiến trúc (Siamese ECAPA)
"Đây gọi là kiến trúc **song sinh** (Siamese): hai đoạn audio — một là giọng gốc tham chiếu, một là giọng máy tạo — cùng đi qua **một** mô hình ECAPA *y hệt nhau, dùng chung trọng số*. Mỗi đoạn biến thành một 'dấu vân tay giọng nói' — một dãy 192 con số. Cuối cùng em đem hai dấu vân tay so với nhau bằng **cosine**: càng giống thì điểm càng cao. Vì là zero-shot nên hai cột speaker và accent tạm dùng chung một cách tính, ra 0.451 và 0.440."

*[Mẹo: hình ảnh 'dấu vân tay giọng nói' rất dễ nhớ — dùng nó. 'Dùng chung trọng số' = chữ ký Siamese.]*

---

## Slide 13 — Track 3: Per-layer
"Bảng chi tiết các tầng của ECAPA: từ trích đặc trưng giọng, chú ý theo kênh, gộp thời gian, nén thành 192 chiều, chuẩn hóa, rồi cosine. Em lướt nhanh."

*[Mẹo: tham khảo, 15 giây.]*

---

## Slide 14 — Divider Track 2 ⭐
"Và đây là phần chính — **Track 2: chấm giọng máy tạo ra có cảm xúc**. Em dành nhiều công sức nhất ở đây."

*[Mẹo: lên giọng một chút để báo hiệu phần quan trọng.]*

---

## Slide 15 — Track 2: 6 cột & cách chấm
"Track 2 khó vì phải chấm tới **6 cột cùng lúc** cho mỗi đoạn:
- **QMOS**: chất lượng giọng nghe có tự nhiên không.
- **EMOS**: giọng có truyền *đúng cảm xúc target* hay không.
- **CAT**: nhận đúng *loại* cảm xúc — vui, buồn, giận, trung tính, ngạc nhiên — chấm bằng tỉ lệ lỗi, càng thấp càng tốt.
- **VAD**: ba trục cảm xúc liên tục — tích cực hay tiêu cực, mức kích thích, và mức chi phối.

Mỗi cột chấm riêng bằng SRCC. Dữ liệu gộp từ ba nguồn, khoảng 12 nghìn mẫu để em huấn luyện."

*[Mẹo: đọc gọn 4 ý. CAT nhớ nói 'càng thấp càng tốt' vì nó là điểm lỗi. VAD một câu là đủ.]*

---

## Slide 16 — Track 2: Kiến trúc (hệ tốt nhất)
"Đây là hệ tốt nhất của em. Ý tưởng là **kết hợp hai chuyên gia rồi cho học chung**:
- **WavLM** — chuyên đặc trưng âm thanh, khởi tạo từ một mô hình tên SAILER nên đã 'biết' sẵn về cảm xúc. **Đây là phần em tinh chỉnh**: chỉ mở 6 lớp trên cùng cho học theo dữ liệu cảm xúc, các lớp dưới giữ nguyên.
- **audeering** — chuyên gia về ba trục cảm xúc VAD. Cái này em **giữ nguyên, đóng băng**, chỉ lấy đặc trưng làm sẵn.

Hai nguồn gộp ở dấu cộng, đi qua một **bộ não chung** (TRUNK) nén lại còn 512 chiều, rồi tỏa ra 6 cột điểm. Ngoài ra có **UTMOS đóng băng làm điểm neo** cho cột chất lượng QMOS. Kết quả: EMOS 0.811, QMOS 0.548."

*[Mẹo: ⚠️ NÓI ĐÚNG — **chỉ WavLM fine-tune**, audeering + UTMOS đóng băng. Slide đang tô audeering 🔥 như đang train là LỖI HÌNH (nên sửa 🔥→❄); nếu chưa kịp sửa thì đừng chỉ vào chữ đó, nói theo lời này. Slide 'Training details' phía sau ghi đúng.]*

---

## Slide 17 — Track 2: Per-layer
"Bảng chi tiết cho thấy rõ ai train ai không: **WavLM** tách làm hai — phần dưới (CNN và 18 lớp) giữ nguyên, chỉ **6 lớp trên cùng được tinh chỉnh** (biểu tượng lửa). Còn **audeering** và **UTMOS** thì **đóng băng** (bông tuyết) — audeering lo ba trục cảm xúc, UTMOS làm neo chất lượng. Cuối cùng gộp lại, qua trunk, ra 6 head. Nhờ chỉ train một phần nhỏ nên chạy được trên GPU miễn phí của Kaggle."

*[Mẹo: ⚠️ Trong bảng, dòng audeering đang để 🔥 'top-6 fine-tune' — đó cũng là LỖI, đúng phải là ❄ đóng băng (khớp slide Training details). Nói: 'chỉ WavLM 6 lớp trên được train'.]*

---

## Slide 18 — Track 2: Leaderboard (tiến triển)
"Đây là quá trình điểm tăng dần theo ngày, từ mùng 3 tới 16 tháng 6. QMOS từ 0.414 leo lên **0.6296**; EMOS từ 0.194 vọt lên **0.8144**; arousal đạt **0.7978** — các ô có biểu tượng tên lửa là kỷ lục của từng cột.

Một điều em xin nói thẳng: bản em *đã nộp chính thức* dùng bộ điểm an toàn (0.548 và 0.811); còn các kỷ lục mới nhất từng cột thì em **chưa gộp hết vào cùng một bản nộp**."

*[Mẹo: câu cuối thể hiện sự trung thực — luôn được đánh giá cao.]*

---

## Slide 19 — Kết quả 3 track
"Tóm lại điểm cả ba phần trong một bảng: Track 1 đạt ACR 0.662; Track 2 nổi bật nhất với EMOS 0.811 và đủ 6 cột; Track 3 quanh 0.45. Track 2 là nơi hệ thống của em mạnh nhất và cũng là nơi em đầu tư nhiều nhất."

*[Mẹo: slide tổng kết điểm — nói gọn, dẫn sang phần kỹ thuật/triển khai.]*

---

## Slide 20 — Training details
"Vài chi tiết kỹ thuật, và lưu ý chỉ Track 2 mới thật sự train. Nói gọn: em **chỉ fine-tune WavLM** — mở 6 lớp trên cùng, tốc độ học rất nhỏ; còn **audeering và UTMOS để nguyên, đóng băng và cache sẵn**. Em dùng thêm một mẹo tên *uncertainty weighting* để máy **tự cân** xem trong 6 cột cột nào khó thì ưu tiên, khỏi dò tay. Tất cả vừa đúng một GPU T4 16GB miễn phí. Track 1 thì chỉ inference, Track 3 thì zero-shot — không train."

*[Mẹo: slide này GHI ĐÚNG (audeering ❄). Đây là chỗ chốt lại thông điệp 'chỉ WavLM train'. Ai hỏi sâu mới mở rộng.]*

---

## Slide 21 — Deploy / Triton serving
"Ngoài độ chính xác, em còn **đóng gói hệ thống để dùng thật**. Người dùng gửi một file audio tới **Gateway** (cổng FastAPI); cổng chuyển vào **máy chủ Triton** đang chạy **cả ba model cùng lúc trên một GPU** — track 1, track 2, track 3 — rồi trả về điểm dạng JSON. Em làm thêm phần **gom nhiều audio xử lý một lượt** cho nhanh, và công cụ Locust để **đo tải**. Nói cách khác, đây không chỉ là bài tập mà chạy được như một dịch vụ thật."

*[Mẹo: nhấn 'ba model một GPU, chạy như dịch vụ thật' — điểm cộng lớn. Ô đỏ track2 = model có fine-tune.]*

---

## Slide 22 — Demo (video)
"Và đây là demo thực tế. Em xin bật để mọi người xem hệ thống chấm điểm trực tiếp."

*[Mẹo: BẤM nút ▶ trên video — đừng bấm Space vì Space sẽ chuyển slide (deck tự dừng video khi đổi slide). Nói trước một câu video sẽ cho thấy gì. Phòng máy yếu: mở thử video trước khi trình bày.]*

---

## Slide 23 — Closing
"Đó là toàn bộ phần trình bày của em. Tóm lại: ba phần thi, trọng tâm là Track 2 — kết hợp WavLM tinh chỉnh với audeering đóng băng, đạt EMOS 0.811 — và toàn hệ thống đã đóng gói chạy thật. Em xin cảm ơn anh chị và mọi người, rất mong nhận được góp ý ạ."

*[Mẹo: dừng lại, mỉm cười, mời câu hỏi. Slide có email liên hệ.]*

---

### Mẹo chung khi trình bày
- **Mạch chính**: bài toán → cách chấm (MOS → SRCC → UTT/SYS) → Track 1 → Track 3 → Track 2 → kết quả → triển khai → demo. Bám mạch này nếu lỡ quên.
- Câu cửa miệng nên thuộc: *"SRCC chỉ cần xếp đúng thứ tự, không cần đúng con số."*
- Các slide bảng per-layer (Track 1/3/2) là **tham khảo** — lướt nhanh, đừng đọc từng dòng.
- **Nhớ đúng về fine-tune**: chỉ **WavLM** được train; **audeering + UTMOS đóng băng**. ⚠️ Hai slide *Kiến trúc T2* và *Per-layer T2* đang vẽ audeering 🔥 (sai) — nên sửa hình; khi nói cứ theo slide *Training details* (ghi đúng).
- Bị hỏi sâu phần không chắc → trả lời ngắn rồi "em xin phép ghi nhận và tìm hiểu thêm".
