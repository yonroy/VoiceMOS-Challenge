
Giải thích Batch 

### Batch = 1: đi 4 bước, mỗi bước nghe 1 ý kiến

w = 1.000
bước 1 (A): w = 1.000 − 0.01×(+8) = 0.920   ← giật mạnh xuống
bước 2 (B): w = 0.920 − 0.01×(−2) = 0.940   ← quay đầu đi ngược!
bước 3 (C): w = 0.940 − 0.01×(+5) = 0.890
bước 4 (D): w = 0.890 − 0.01×(+1) = 0.880
Đường đi zigzag: xuống – lên – xuống – xuống. Mỗi bước bị đúng 1 audio chi phối, audio nhiễu (A) tác động full lực.

### Batch = 4: nghe cả 4 ý kiến, lấy trung bình, đi 1 bước

gradient batch = (8 − 2 + 5 + 1) / 4 = +3      ← TRUNG BÌNH
w = 1.000 − 0.01×(+3) = 0.970
Một bước thẳng và điềm tĩnh: ý kiến ngược chiều của B bị 3 phiếu kia trung hòa, audio dị A bị pha loãng còn 1/4 sức.


### Accum sẽ giúp có độ mượt của gradient nhưng đổi lại tốc độ chậm
- Kiểu như nó sẽ cầm lên ghi kết quả và thả xuống chậm nhưng vẫn đảm bảo độ mượt


# layer Emos

[trunk 512 | one-hot target 5] = 517 nghĩa là: nối (concatenate) 2 vector lại thành 1 vector dài 517 số rồi mới đưa vào head EMOS. Dấu | là phép nối đuôi.

Mảnh 1 — trunk 512: model "nghe" được gì
Đây là đầu ra của trunk — vector 512 số tóm tắt mọi thứ nghe được từ audio (giọng vui hay buồn, năng lượng cao thấp, chất giọng...). Nó đến từ tai + não của model.

Mảnh 2 — one-hot target 5: đề bài YÊU CẦU cảm xúc gì
One-hot = cách mã hóa 1 nhãn rời rạc thành vector toàn 0 với đúng một số 1. Track 2 có 5 cảm xúc target, nên:

Target	Vector one-hot
neutral	[1, 0, 0, 0, 0]
happy	[0, 1, 0, 0, 0]
sad	[0, 0, 1, 0, 0]
angry	[0, 0, 0, 1, 0]
surprised	[0, 0, 0, 0, 1]
Thông tin này lấy từ metadata của dataset (mỗi audio được giao 1 cảm xúc phải thể hiện), không phải từ audio.

Nối lại: 512 + 5 = 517

[0.23, -1.07, 0.88, ..., 0.41 | 0, 1, 0, 0, 0]
 └────── nghe được gì ──────┘  └─ yêu cầu gì ─┘
        512 số                     5 số
Head EMOS là MLP Linear 517→128 → ReLU → Linear 128→1 đọc cả 2 mảnh cùng lúc.

VÌ SAO bắt buộc phải nối target vào?
Nhớ lại định nghĩa EMOS: "giọng này thể hiện đúng cảm xúc được yêu cầu tới đâu" — đây là câu hỏi so khớp giữa 2 thứ, không phải câu hỏi về riêng audio.

Thí nghiệm tưởng tượng với 1 audio giọng rất vui:

Audio (trunk 512 giống hệt)	Target nối vào	EMOS đúng phải là
giọng vui	happy [0,1,0,0,0]	cao (~4.8) — khớp ✅
giọng vui	sad [0,0,1,0,0]	thấp (~1.5) — lệch hẳn ❌
Cùng một audio, 2 điểm EMOS khác nhau — chỉ vì target khác. Nếu head chỉ nhận trunk 512 (không có target), đầu vào 2 trường hợp giống hệt nhau → model toán học không thể cho ra 2 đáp án khác nhau → bài toán vô nghiệm. Nối one-hot vào là cách "đưa đề bài" cho model: nó học được quy tắc "mảnh nghe-được khớp với mảnh yêu-cầu → điểm cao, lệch → điểm thấp".



 ### Emos
- EMOS head nhận [trunk 512 | one-hot target 5] = 517 — vì câu hỏi của nó là "khớp cảm xúc được yêu cầu không", thiếu target thì câu hỏi vô nghĩa.
- Trunk = phần thân chung của mạng, nằm giữa backbone và các head, được MỌI nhiệm vụ dùng chung
- hai lớp sập thành một lớp Linear duy nhất, thêm lớp là vô ích. Hàm phi tuyến (ReLU) chen giữa là thứ "bẻ gãy" phép nhân ma trận, cho phép model học quan hệ cong/phức tạp — ví dụ quy tắc "khớp target thì điểm cao" là quan hệ AND giữa 2 mảnh vector, Linear thuần không biểu diễn nổi.
- Lũy thừa của 2 (64/128/256/512): thuần quy ước + GPU xử lý kích thước này hiệu quả.

## Softmax
- Softmax là máy biến 5 số thô (logits, chạy từ −∞ đến +∞) thành 5 số thỏa 2 điều kiện của một phân bố — đều dương, cộng = 1
$$\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}$$

Ví dụ: logits [0.2, 2.1, 1.0, -0.5, -1.3] → softmax → [0.09, 0.61, 0.21, 0.05, 0.02] — đọc được ngay là "61% happy, 21% neutral...".

Loss trong code là soft cross-entropy (dòng 455-456): −Σ target_dist · log_softmax(logits) — giống cross-entropy phân loại thường, nhưng target là phân bố mềm thay vì one-hot. Model bị phạt khi đặt xác suất thấp vào chỗ người nghe vote nhiều.

## VAD head — z-score và phép ×σ + μ
Z-score (điểm chuẩn hóa) = cách đổi một con số sang đơn vị mới: "nó cách trung bình bao nhiêu lần độ lệch chuẩn?"

$$z = \frac{y - \mu}{\sigma}$$

μ (mu) = trung bình của cả tập số liệu
σ (sigma) = độ lệch chuẩn — thước đo "dữ liệu thường dao động quanh trung bình rộng cỡ nào"

Vấn đề: nhãn VAD nằm thang 1–5, tập trung quanh ~3. Nếu bắt head dự đoán thẳng giá trị này thì lúc khởi tạo (output quanh 0) loss MSE ≈ 3² = 9, rất to so với các task khác → gradient giai đoạn đầu toàn bị "kéo về trung bình 3" thay vì học điều tinh tế.

Giải pháp — chuẩn hóa z-score trước khi train: với mỗi trục (V/A/D), tính trung bình μ và độ lệch chuẩn σ trên tập train, rồi đổi nhãn:

$$z = \frac{y - \mu}{\sigma} \quad\text{(ví dụ: } y = 3.8,\ \mu = 3.1,\ \sigma = 0.6 \Rightarrow z = +1.17\text{)}$$

Nhãn mới có trung bình 0, độ lệch 1 — đúng "vùng thoải mái" của mạng neural (khớp khởi tạo, gradient cân với task khác). Head học dự đoán z, không phải điểm thật.

Hệ quả bắt buộc lúc dự đoán: đầu ra head là z-space → phải giải ngược về thang 1–5: y = z·σ + μ (chính là dòng 620: vad_p * vad_sd + vad_mu; EMOS cũng vậy ở dòng 618).

Và đây là chỗ sinh ra bài học xương máu: μ/σ là một phần của model. Nếu checkpoint chỉ lưu weight mà quên μ/σ, lúc load lại bạn có head dự đoán z hoàn hảo nhưng không biết đổi về điểm thật — dự đoán toàn số quanh 0, vô nghĩa. Nên ckpt của mình lưu kèm vad_mu/vad_sd/emos_mu/emos_sd, và khi load_state_dict báo "thiếu/dư key" thì phải hiểu là đang load lệch cấu trúc chứ không phải lỗi vặt bỏ qua được.

Nhãn VAD nằm thang 1–5, tụ quanh ~3 → đổi sang z-score (μ, σ tính trên 12.7k nhãn train) để head học trên dải số quanh 0, đúng "vùng thoải mái" của mạng → hội tụ nhanh, gradient cân với task khác. Lúc dự đoán phải ×σ + μ giải ngược về thang 1–5 — đó chính là dòng vad_p * vad_sd + vad_mu trong code, và lý do μ/σ phải được lưu trong checkpoint như một phần của model: mất chúng là mất chìa khóa dịch ngược, dự đoán chỉ còn là số z vô nghĩa với grader.

Tóm 1 dòng: z-score = "số bậc lệch khỏi trung bình" — đưa mọi thang đo về cùng một ngôn ngữ (trung bình 0, lệch 1), và luôn dịch ngược được nếu giữ μ, σ.

# Uncertainty weighting — cuộc "đấu giá" giữa 5 task
Vấn đề: loss tổng = L_emos + L_cat + L_val + L_aro + L_dom. Nhưng 5 loss này thang khác nhau, độ nhiễu khác nhau (CAT vote phân tán nhiễu hơn EMOS chẳng hạn). Cộng thô thì task loss to chiếm sóng gradient. Chỉnh tay trọng số λ cho từng task = mò 5 chiều, tốn vô số lần chạy.

Giải pháp (Kendall 2018): cho mỗi task một tham số học được σₜ — diễn giải là "độ nhiễu của task đó" — và viết loss:

$$L = \sum_t \frac{1}{2\sigma_t^2} L_t + \log\sigma_t$$

Đọc cơ chế như một sự đánh đổi mà model tự cân:

Task nhiễu, loss giảm mãi không xuống → model tăng σₜ → trọng số 1/2σₜ² giảm → task đó bớt chi phối gradient ("đừng cố vắt nước từ đá").
Nhưng tăng σ phải trả phí +log σₜ → không thể tăng vô hạn để trốn việc (nếu không có phí này, model sẽ đặt σ = ∞ cho mọi task và loss về 0 một cách gian lận).
Điểm cân bằng tối ưu rơi đúng ở σₜ² ≈ mức nhiễu thật của task → trọng số tự khớp với độ tin cậy của từng nhãn.