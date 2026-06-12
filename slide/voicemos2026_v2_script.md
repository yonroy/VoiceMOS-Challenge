# 🎤 Kịch bản thuyết trình — VoiceMOS 2026 slide v2 (36 slide, ~30–35 phút)

> Dùng kèm `slide/voicemos2026_v2 (1).html` (nguồn: `docs/22_slides_v2_paper_style.md`).
> Mỗi slide: **NÓI** = lời thoại gợi ý · **NHẤN** = ý phải chốt được · **⏱** = thời lượng gợi ý.
> Tổng ~33 phút nói + 5–10 phút Q&A. Soạn ngày 11/6/2026.

---

## PHẦN I — MỞ ĐẦU (slide 1–8, ~8 phút)

### Slide 1 — Title ⏱ 30s
**NÓI:** "Em chào thầy/anh chị. Hôm nay em trình bày dự án VoiceMOS Challenge 2026 — dạy máy chấm điểm giọng nói như người nghe. Em tham gia cả 3 track, trong đó dồn toàn lực vào Track 2 về giọng nói cảm xúc, hướng tới một bài báo ICASSP 2027."
**NHẤN:** 3 track nhưng trọng tâm = Track 2.

### Slide 2 — Nội dung ⏱ 40s
**NÓI:** "Em trình bày theo mạch một bài báo: trước hết là bài toán và cách chấm điểm — phần này em làm kỹ vì mọi quyết định thiết kế phía sau đều bắt nguồn từ cách chấm. Sau đó Track 1 và Track 3 em đi nhanh vì dùng baseline, rồi dành phần lớn thời gian cho Track 2 — hệ thống em tự phát triển — từ động lực, phương pháp, kết quả đến phân tích."
**NHẤN:** "mỗi track là một paper rút gọn: động lực → bài toán → cách làm → kết quả".

### Slide 3 — Introduction: MOS là gì ⏱ 1 phút
**NÓI:** "MOS là điểm trung bình người nghe chấm chất lượng giọng, thang 1–5 — tiêu chuẩn vàng để đánh giá TTS. Vấn đề: thuê người nghe hàng nghìn câu thì chậm và đắt, không thể lặp lại mỗi lần đổi model. Nên cần một 'giám khảo máy' dự đoán MOS thay người. Và biên giới mới của TTS là nói CÓ CẢM XÚC — nhưng các giám khảo máy hiện nay chỉ biết chấm chất lượng, chưa biết chấm cảm xúc."
**NHẤN:** câu thần chú của dự án: *"Muốn AI sinh ra cảm xúc, trước hết phải ĐO được cảm xúc — cần một emotional ruler."*

### Slide 4 — So sánh 3 track ⏱ 1 phút
**NÓI:** "Bảng này so 3 track. Track 1 chấm giọng đã khử nhiễu, Track 3 chấm độ giống giọng của audio codec — cả hai em chạy baseline chính thức để có mặt trên leaderboard. Track 2 chấm TTS cảm xúc với 6 cột điểm, có 12.746 câu nhãn train — đây là nơi duy nhất có data để học, và là phần đóng góp khoa học của em. Deadline nộp kết quả là 7/8/2026."
**NHẤN:** Track 2 có train data → có đất diễn; Track 1 không có → chỉ inference được.

### Slide 5 — Cách chấm (1/3): hai tầng điểm ⏱ 1 phút
**NÓI:** "Trước khi vào model, cần hiểu cách chấm — vì đây là chỗ dễ nhầm nhất. Có HAI tầng điểm. Tầng 1: mỗi audio một điểm — là trung bình của nhiều người nghe, nên mới có số lẻ như 4.8. Tầng 2: điểm trên leaderboard KHÔNG phải điểm của audio nào, mà là điểm của GIÁM KHẢO — đo xem model chấm có giống người không, trên toàn bộ 2.730 câu. Và challenge chấm ở mức utterance — xếp đúng từng câu lẻ — khó hơn nhiều so với chỉ xếp đúng 13 hệ TTS."
**NHẤN:** "số trên leaderboard là điểm CỦA MODEL, không phải của audio".

### Slide 6 — Cách chấm (2/3): SRCC ví dụ tính tay ⏱ 1.5 phút
**NÓI:** "SRCC đo tương quan THỨ HẠNG. Ví dụ 5 audio này: đổi điểm hai bên thành hạng, model xếp đúng A, B, C nhưng đảo D và E — chênh hạng d=1 ở hai chỗ, thay vào công thức ra 0.9. Điểm mấu chốt: model chấm thấp hơn người cả 1 điểm ở mọi câu mà KHÔNG bị phạt — SRCC chỉ cần đúng thứ tự. Hệ quả quan trọng cho training: loss MSE tối ưu giá trị, nhưng metric chấm thứ hạng — lệch nhau, nên về sau em có thử ranking loss."
**NHẤN:** "đúng thứ tự là được, không cần đúng giá trị" — sẽ dùng lại ý này ở slide phân tích VAD.

### Slide 7 — Cách chấm (3/3): 10 cột metric ⏱ 1 phút
**NÓI:** "Toàn cuộc thi có 10 cột: Track 1 hai cột ACR–CCR, Track 3 hai cột speaker–accent, Track 2 sáu cột — trong đó QMOS và EMOS bắt buộc. Lưu ý quan trọng nhất bảng này: 9 cột là SRCC càng cao càng tốt, RIÊNG cột CAT là sai số — càng THẤP càng tốt. Đừng đọc nhầm chiều."
**NHẤN:** ⚠️ CAT ngược chiều.

### Slide 8 — CAT-ERR công thức ⏱ 1.5 phút
**NÓI:** "Cột CAT đặc biệt nên em dành một slide. Nhãn của nó là TỈ LỆ VOTE — 10 người nghe thì 6 bảo happy, 3 neutral, 1 surprised; sự bất đồng này chính là thông tin. Grader tính sai số tuyệt đối trung bình trên từng ô của bảng N×5 — như ví dụ này, một audio đóng góp tổng lệch 0.40. Hệ của em đạt 0.1331 — tức trung bình mỗi ô lệch 13 điểm phần trăm, tốt hơn baseline 31%."
**NHẤN:** nhãn là phân bố → model phải ra phân bố (sẽ nối vào thiết kế CAT head).

---

## PHẦN II — TRACK 1 (slide 9–12, ~4 phút)

### Slide 9 — T1 bài toán ⏱ 45s
**NÓI:** "Track 1 chấm giọng đã qua khử nhiễu: ACR là chấm tuyệt đối một audio, CCR là so sánh cặp hai audio. Điểm then chốt: track này KHÔNG có training data chính thức — không thể fine-tune — nên em dùng baseline URGENT-MOS, chỉ inference."
**NHẤN:** không có data ⇒ chiến lược bắt buộc khác Track 2.

### Slide 10 — T1 kiến trúc (hình) ⏱ 1 phút
**NÓI:** "Đây là URGENT-MOS: 4 encoder đóng băng chạy song song — WavLM, Kimi-Audio, Qwen3-Omni, Audio-Flamingo. Mỗi encoder như một giám khảo có chuyên môn riêng: WavLM là kỹ sư âm thanh thính với nhiễu, các audio-LLM thiên về ngữ nghĩa và độ tự nhiên. Bốn nhận xét được fusion lại rồi đi vào 2 head."
**NHẤN:** vì không train được nên phải đa dạng hóa "đôi tai" — lỗi của 4 model ít tương quan, bù khuyết điểm cho nhau.

### Slide 11 — T1 từng layer ⏱ 1.5 phút
**NÓI:** "Đi nhanh từng tầng: CNN 7 lớp thái sóng thành khung 20ms, nén 320 lần. Transformer 24 lớp cho mỗi khung 'nhìn' toàn câu — quan trọng vì một tiếng 'xẹt' ở giây 3 kéo tụt cảm nhận cả câu. Tầng thú vị nhất là TRỘN LỚP: giữ đầu ra cả 24 lớp rồi cộng có trọng số học được — vì lớp dưới chứa chi tiết âm học, lớp trên chứa ngữ nghĩa, model tự học cần tầng nào. Cuối cùng AMPM hồi quy điểm ACR, còn NCPM lấy HIỆU hai nhánh g(A)−g(B) — đảo A và B thì điểm tự đổi dấu, đúng bản chất câu hỏi so sánh."
**NHẤN:** trộn lớp αₗ = cách chọn tầng khi-không-được-fine-tune.

### Slide 12 — T1 kết quả ⏱ 30s
**NÓI:** "Kết quả ACR 0.662, CCR 0.411 — khớp mức baseline công bố, xác nhận pipeline chạy đúng. Vai trò của track này là phủ leaderboard, không phải hướng nghiên cứu chính."

---

## PHẦN III — TRACK 3 (slide 13–16, ~3.5 phút)

### Slide 13 — T3 bài toán ⏱ 45s
**NÓI:** "Track 3 là track duy nhất cần CẶP audio: một audio sinh ra và một audio tham chiếu, dự đoán độ giống về NGƯỜI NÓI và về ACCENT. Em dùng baseline speaker embedding, cũng chỉ inference."

### Slide 14 — T3 kiến trúc (hình) ⏱ 45s
**NÓI:** "Hai audio đi qua CÙNG MỘT encoder ECAPA-TDNN — gọi là Siamese, chung trọng số. Logic rất tự nhiên: muốn so hai giọng thì phải đo bằng cùng một cây thước. Ra hai vector 192 chiều, chuẩn hóa, đo cosine."

### Slide 15 — T3 từng layer ⏱ 1.5 phút
**NÓI:** "ECAPA khác hai track kia: thuần CNN, không có Transformer. TDNN là conv GIÃN NỞ — cùng 3 trọng số nhưng xòe rộng tầm nhìn dần. SE-block là 'chú ý theo kênh': nghe toàn câu rồi quyết định máy dò nào đáng vặn to. Đáng chú ý nhất là attentive stat-pooling: thay vì chỉ lấy trung bình, nó giữ thêm ĐỘ LỆCH CHUẨN — hai giọng cùng cao độ trung bình nhưng một giọng đều, một giọng rung sẽ phân biệt được; độ rung chính là vân tay giọng. Bản nộp zero-shot dùng thẳng cosine cho CẢ HAI cột — đây là điểm yếu lộ rõ: encoder học nhận dạng người nói trên VoxCeleb, chưa từng học accent."
**NHẤN:** spk = acc = cùng 1 số → hướng nâng cấp là interaction vector + 2 head riêng.

### Slide 16 — T3 kết quả ⏱ 30s
**NÓI:** "Kết quả 0.451 và 0.440 — khớp baseline. Hai cột xấp xỉ nhau phản ánh đúng việc chúng dùng chung một con số cosine."

---

## PHẦN IV — TRACK 2 ⭐ (slide 17–34, ~15 phút — TRỌNG TÂM)

### Slide 17 — Divider ⏱ 15s
**NÓI:** "Giờ vào phần chính — Track 2, hệ thống em tự phát triển. Em trình bày đầy đủ theo mạch một bài báo."

### Slide 18 — Motivation ⏱ 45s
**NÓI:** "Vì sao cần thước đo cảm xúc? TTS đã ở khắp nơi; cái thiếu không phải nói GÌ mà là nói với CẢM XÚC nào. Nghẽn nằm ở khâu đánh giá. Một predictor cảm xúc đáng tin sẽ là tín hiệu phản hồi để xây TTS cảm xúc: so checkpoint không cần thuê người, làm reward model cho RLHF — và em đã tự kiểm chứng vòng lặp này bằng cách chấm TTS tiếng Việt, sẽ nói ở phần phân tích."

### Slide 19 — Bài toán 6 cột ⏱ 1 phút
**NÓI:** "Một câu nói vào, 6 cột điểm ra: QMOS chất lượng, EMOS độ khớp cảm xúc target — hai cột bắt buộc; CAT phân bố 5 cảm xúc, và 3 trục VAD. Data 12.746 câu train có nhãn đầy đủ. Để ý format nộp: cột CAT phải nộp nguyên phân bố xác suất 5 lớp."
**NHẤN:** 6 cột = 6 "đề thi" tính chất khác nhau trên cùng 1 audio.

### Slide 20 — Mục tiêu: 1 model 6 đầu ra (hình) ⏱ 30s
**NÓI:** "Mục tiêu thiết kế: MỘT model duy nhất ra cả 6 cột — backbone học chung để các nhiệm vụ cộng hưởng, thay vì 6 model rời."

### Slide 21 — Baseline & 3 điểm yếu ⏱ 45s
**NÓI:** "Baseline của ban tổ chức ghép 3 model zero-shot: UTMOS cho QMOS, emotion2vec cho CAT, Gemini cho EMOS và VAD. Ba điểm yếu: EMOS chỉ 0.19 — gần mức đoán bừa; gọi API tốn phí; và bỏ phí toàn bộ 12.746 nhãn người chấm. Cơ hội của em nằm đúng ở đó: thay ráp zero-shot bằng một model đa nhiệm CÓ HUẤN LUYỆN."

### Slide 22 — Phát hiện C1 ⏱ 1 phút
**NÓI:** "Phát hiện đầu tiên — em đo từng encoder cảm xúc một: emotion2vec thắng EMOS 0.637, SAILER thắng VAD 0.712. KHÔNG model nào thắng mọi cột. Kết luận: thay vì chọn một, hãy GỘP cả hai — chúng bổ sung nhau. Phát hiện này dẫn đường cho toàn bộ method phía sau."
**NHẤN:** C1 = nền móng; nguyên lý "muốn biết model giỏi gì, xem nó từng học bài tập gì".

### Slide 23 — Method C2: fusion (hình) ⏱ 45s
**NÓI:** "Method 1: hai encoder ĐÓNG BĂNG trích đặc trưng — vì đóng băng nên trích một lần rồi cache, mỗi epoch chỉ chạy MLP nhỏ, lặp thí nghiệm rất nhanh trên Kaggle T4. Đặc trưng concat lại, qua trunk chung, tỏa ra 4 head."

### Slide 24 — C2 từng layer ⏱ 1.5 phút
**NÓI:** "Cụ thể từng tầng: emotion2vec cho 1029 chiều, SAILER 1036 chiều, concat 2065 → trunk nén về 512. Tầng đáng nói nhất là QMOS head: nó nhận thêm điểm UTMOS làm NEO — chỉ học chỉnh sửa quanh một dự đoán đã tốt sẵn, sàn là 0.414 nên khó tệ hơn. Và nhờ trunk được 5 task cảm xúc 'nuôi', QMOS hưởng ké biểu diễn cảm xúc — kết quả 0.414 lên 0.548, lần đầu vượt UTMOS zero-shot mà không kéo tụt cột nào."
**NHẤN:** neo UTMOS + multi-task = combo thắng QMOS.

### Slide 25 — Giải phẫu 3 head ⏱ 1.5 phút
**NÓI:** "Slide này trả lời 'vì sao mỗi head một kiểu'. EMOS hỏi 'khớp cảm xúc ĐƯỢC YÊU CẦU không' — câu hỏi so khớp hai thứ, nên phải nối one-hot target vào: cùng một audio, đổi target thì điểm phải đổi — không có target thì bài toán vô nghiệm. CAT có nhãn là phân bố vote nên đầu ra phải là softmax — phân bố so với phân bố. VAD train trên nhãn đã z-score hóa cho dễ hội tụ, lúc dự đoán nhân ngược σ cộng μ về thang 1–5 — và bài học xương máu: μ/σ là MỘT PHẦN của model, phải lưu trong checkpoint."
**NHẤN:** nguyên tắc "đầu ra và loss phải mô phỏng đúng bản chất nhãn".

### Slide 26 — Method C3: fine-tune (hình) ⏱ 45s
**NÓI:** "Method 2: đặc trưng frozen chỉ cho ta thứ 'như nó vốn có' — head nhỏ không vặn lại được khi domain lệch. Giải pháp: mở băng 6 lớp Transformer TRÊN CÙNG của WavLM, warm-start từ SAILER, cho chính biểu diễn xoay về domain dữ liệu cảm xúc của challenge."

### Slide 27 — C3 từng layer ⏱ 1.5 phút
**NÓI:** "Vì sao chỉ 6 lớp trên? Ba lý do chồng nhau: lớp dưới học đặc trưng âm học phổ quát — domain nào cũng cần, không nên phá; lớp trên mang ngữ nghĩa cảm xúc — đúng chỗ cần chỉnh; và T4 16GB không gánh nổi backward cả 24 lớp. LR hai tốc độ: 1e-5 cho phần mượn — vặn rón rén để không quên kiến thức cũ; 1e-3 cho phần mới tinh. Nhìn tỉ lệ tham số: phần thật sự train từ đầu chỉ 1.3 triệu trên tổng 630 triệu — sức mạnh là MƯỢN và TINH CHỈNH."
**NHẤN:** câu "phần dạy từ đầu chỉ là cái đuôi 1.3M".

### Slide 28 — Training details ⏱ 1.5 phút
**NÓI:** "Vài quyết định training ăn điểm. Một: 5 loss lệch thang được cân TỰ ĐỘNG bằng uncertainty weighting — mỗi task một tham số học được, task nhiễu tự giảm trọng số nhưng phải trả phí log sigma nên không trốn việc được — chỉ 5 tham số thay cả quá trình mò tay. Hai: để vừa T4, dùng gradient accumulation — gom gradient 8 lượt nhỏ rồi mới cập nhật, mượt như batch 32 nhưng VRAM như batch 4 — cộng AMP và gradient checkpointing. Ba: early-stop theo SRCC validation và LƯU CHECKPOINT MỖI BEST — bài học từ lần mất backbone vì kernel chết."

### Slide 29 — exp13: fine-tune đúng model ⏱ 1.5 phút
**NÓI:** "Đến đây có một nghịch lý đáng kể: exp08 fine-tune mạnh thế mà QMOS lại XẸP về 0.417 — biểu diễn bị kéo nghiêng hết về cảm xúc. Giả thuyết của em: UTMOS kẹt 0.414 không phải vì kiến trúc yếu mà vì LỆCH DOMAIN — nó được train trên giọng không-cảm-xúc từ 2022. Giải pháp exp13: fine-tune thẳng CHÍNH UTMOS trên nhãn qMOS thật, LR nhỏ, có lưới an toàn chỉ nộp nếu vượt zero-shot. Kết quả: 0.548 lên 0.6296 — kỷ lục cột, xác nhận giả thuyết. Bài học lớn: MỖI CỘT MỘT KHẨU VỊ — cảm xúc cần fine-tune encoder cảm xúc, chất lượng cần fine-tune encoder chất lượng."
**NHẤN:** đây là phát hiện "ăn paper" thứ hai sau C1.

### Slide 30 — Kết quả tiến hóa ⏱ 1.5 phút
**NÓI:** "Bảng tiến hóa toàn cảnh: baseline EMOS 0.19 → fusion 0.79 → fine-tune 0.81; QMOS 0.41 → neo UTMOS 0.55 → fine-tune UTMOS 0.63. Hàng cuối là best-per-column hiện tại — so với baseline: EMOS tăng hơn 4 lần, QMOS tăng 52%."
**NHẤN:** đọc chậm 2 con số: EMOS 0.19→0.81, QMOS 0.41→0.63.

### Slide 31 — Trộn cột exp_mix ⏱ 1 phút
**NÓI:** "Grader chấm answer.txt từng cột ĐỘC LẬP — nghĩa là được phép ghép cột từ nhiều model. Em đã nộp bản trộn và điểm thật khớp đúng best-per-column, xác nhận chiến lược hợp lệ. Bản đã nộp giờ là fallback an toàn cho phase Evaluation; thế hệ mới — QMOS từ exp13, ARO từ exp15, còn lại từ exp08 — sẵn sàng nộp với 0 giờ GPU."
**NHẤN:** trộn cột = ensemble rẻ nhất, sinh ra từ bài học "mỗi cột một khẩu vị".

### Slide 32 — Ablation ⏱ 1.5 phút
**NÓI:** "Hai ablation cho paper. Một: thay mean-pool bằng Mamba head — kết quả GẦN HÒA ở 4 cột nhưng thắng đúng Arousal 0.7978, kỷ lục cột. Điều này khớp lý thuyết: mean-pool xóa trật tự thời gian, mà arousal là cột duy nhất biến thiên theo thời gian rõ — câu 'đều đều rồi bùng nổ' khác câu 'lăn tăn suốt' dù trung bình bằng nhau. Hai: ranking loss để khớp metric SRCC — đã code, đang là hướng mở; kỷ lục hiện tại chưa cần bật."
**NHẤN:** "ablation gần hòa cũng là kết quả khoa học: nó chỉ ra ĐÚNG chỗ temporal information có giá trị".

### Slide 33 — Phân tích ⏱ 1.5 phút
**NÓI:** "Hai insight. Một: model dự đoán VAD bị nén trong dải hẹp 2.5–3.6 nhưng SRCC vẫn 0.79 — vì thứ tự đúng; hiểu metric giúp không hoảng trước hiện tượng tưởng là lỗi. Hai: em đem chính hệ này chấm TTS tiếng Việt — phát hiện neutral-bias: đầu phân loại luôn argmax ra neutral, NHƯNG 3 trục VAD vẫn đúng hướng: angry arousal cao nhất, sad thấp nhất. Tức scorer thật sự CẢM được cảm xúc, chỉ đầu phân loại bị kéo về neutral do lệch domain ngôn ngữ. Em khắc phục phía đo bằng metric ranking — đúng tinh thần emotional ruler."
**NHẤN:** case study tiếng Việt = bằng chứng vòng lặp "sinh → đo" hoạt động.

### Slide 34 — Hướng mở rộng ⏱ 45s
**NÓI:** "Việc tiếp theo: resume exp15 chốt kết quả Mamba; nộp bản trộn cột thế hệ mới; ablation ranking loss; thử Audio-LLM-as-Judge làm góc novelty; và thêm data cảm xúc ngoài để giải neutral-bias tận gốc."

---

## PHẦN V — KẾT (slide 35–36, ~2 phút)

### Slide 35 — Timeline & chiến lược Evaluation ⏱ 1 phút
**NÓI:** "Eval set thả 31/7, hạn nộp 7/8 — cửa sổ chỉ MỘT TUẦN. Chiến lược của em: mọi thứ chuẩn bị TRƯỚC — pipeline inference đóng băng, script trộn cột và validate format test sẵn trên DEV, checkpoint đã lưu chắc trên Kaggle, và luôn có bản fallback đã nộp thành công. Đến ngày eval chỉ việc đổi đường dẫn input."
**NHẤN:** rủi ro lớn nhất là dồn vào tuần cuối → đã có checklist.

### Slide 36 — Đóng góp & Kết luận + Q&A ⏱ 1 phút
**NÓI:** "Ba đóng góp: C1 — phát hiện hai encoder cảm xúc bổ sung nhau; C2 — một model đa nhiệm 6 cột với neo UTMOS, không negative transfer; C3 — fine-tune đúng domain phá trần, dẫn tới chiến lược trộn cột. Kết luận một câu: fusion biểu diễn bổ sung cộng fine-tune có giám sát đúng domain vượt xa việc ráp model zero-shot — EMOS 0.19 lên 0.81, QMOS 0.41 lên 0.63. Toàn bộ tài nguyên đã mở: checkpoint, demo và API service trên Hugging Face. Em xin hết — mời thầy/anh chị đặt câu hỏi."

---

## 🛡️ Dự phòng Q&A (5 câu dễ bị hỏi nhất)

1. **"SRCC 0.63 nghĩa là gì, tốt chưa?"** → Xếp hạng 2.730 câu lẻ giống người ở mức tương quan 0.63; baseline 0.414; mức >0.6 ở utterance-level với cột chủ quan như chất lượng là mạnh (so các kỳ VMC trước).
2. **"Vì sao không fine-tune cả 24 lớp / cả emotion2vec?"** → VRAM T4 + nguy cơ catastrophic forgetting + lớp dưới là đặc trưng phổ quát; emotion2vec qua funasr khó backprop — đó là lý do chọn WavLM/SAILER làm nhánh fine-tune.
3. **"Trộn cột có hợp lệ không?"** → Grader chấm từng cột độc lập trên answer.txt; đã xác nhận bằng bản nộp thật 9/6, điểm khớp đúng best-per-column. Sẽ khai báo rõ trong system description.
4. **"Mamba gần hòa thì có đáng không?"** → Đáng: nó thắng đúng cột lý thuyết dự đoán (ARO — temporal), thành ablation có câu chuyện; chi phí thêm chấp nhận được; chỉ dùng cột nó thắng nhờ trộn cột.
5. **"Neutral-bias xử lý tận gốc thế nào?"** → Ngắn hạn: metric ranking (khử neutral, SRCC theo prototype VAD). Dài hạn: thêm data cảm xúc đa ngôn ngữ (exp17) hoặc encoder đa ngôn ngữ cho CAT.

## ⏱ Phân bổ thời gian tổng

| Phần | Slide | Phút |
|---|---|---|
| Mở đầu + cách chấm | 1–8 | ~8 |
| Track 1 | 9–12 | ~4 |
| Track 3 | 13–16 | ~3.5 |
| **Track 2 ⭐** | 17–34 | **~15** |
| Kết + Q&A mồi | 35–36 | ~2 |
| **Tổng nói** | 36 slide | **~33 phút** |

> Mẹo trình bày: nếu bị giới hạn 20 phút → cắt slide 11, 15 (bảng layer T1/T3) xuống 30s mỗi slide ("chi tiết từng tầng em để trong slide, anh chị xem sau"), giữ nguyên phần Track 2.
