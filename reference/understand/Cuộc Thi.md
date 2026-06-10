Cuộc thi voice mos này dùng ai chấm điểm của audio thay cho người phải chấm đội nào tạo được 3 track giống nhất được giải.mình dạy một con robot biết nghe âm thanh rồi cho điểm giống như người làm giám khảo — càng giống người chấm thì càng thắng. 🏆

## MOS" nghĩa là gì?

- MOS = Mean Opinion Score = điểm ý kiến trung bình. Nhiều người nghe (listener) cùng chấm 1 audio theo thang 1–5, rồi lấy trung bình ra điểm cuối. Vì là trung bình của nhiều người nên điểm thật là số lẻ (vd 3.7, 4.2…), không chỉ là số nguyên.

## Bảng tóm tắt siêu gọn
|Track|Nghe cái gì|Robot chấm điều gì|Nộp file|
|---|---|---|---|
|Track 1🟦|Tiếng đã lau sạch ồn|Sạch & rõ tới đâu|predictions.csv|
|Track 2🟥 ⭐|Giọng máy có cảm xúc|Hay không + đúng cảm xúc không|answer.txt|
|Track 3🟩|Máy bắt chước giọng|Giống người + giống giọng vùng không|answer.txt|


## Track 1 làm sạch tiếng ồn trong audio. 
- ACR — "Nghe sạch và rõ mức nào?" (cho điểm 1 đến 5 ⭐)
- CCR — "So 2 đoạn, đoạn nào nghe đã tai hơn?" (−3 đến +3)

Nộp bài: một tờ giấy tên predictions.csv
## Track 2 Giọng máy biết bộc lộ cảm xúc
- QMOS — "Giọng nghe hay và tự nhiên không, hay nghe như người máy?" (1–5 ⭐)
- EMOS — "Bảo nó nói vui, vậy nó nghe vui thật không?" (khớp cảm xúc tới đâu, 1–5 ⭐)
- CAT đoán người nghe cảm nhận ra cảm xúc gì, dưới dạng tỉ lệ phần trăm trên 5 cảm xúc: neutral, happy, angry, sad, surprised.(neutral:0.98 | sad:0.018 | happy:0.00014 | angry:~0 | surprised:~0)

- VAD là Các cảm xúc được dùng khi tính điểm: (không bắt buộc)
    - V: Valence (hóa trị) Dễ chịu hay khó chịu? khó chịu/tiêu cực ↔ dễ chịu/tích cực (1–5 ⭐)
    - A: Arousal (kích thích) Bình tĩnh hay sôi nổi? êm dịu/buồn ngủ ↔ hưng phấn/gấp gáp (1–5 ⭐)
    - D: Dominance (chi phối) Yếu thế hay làm chủ? bị động/nhỏ bé ↔ mạnh mẽ/áp đảo (1–5 ⭐)

## Track 3 — "Máy bắt chước giọng người"
- Speaker — "Giọng bắt chước nghe có giống đúng người đó không?" (1–5 ⭐)
- Accent — "Có nói đúng giọng vùng miền của người đó không?" (1–5 ⭐)


|Chặng|Như ở trường|Được nộp?|Tính hạng?|
|---|---|---|---|
|🟢 Training|Làm bài tập ở nhà|Thoải mái (30/ngày)|❌ tập thôi|
|😴 Break|Giờ ra chơi, chờ thi|❌ không|—|
|🔴 Evaluation|Thi thật trong phòng|Chỉ 3 lần, chọn 1|✅ tính hạng|
|🏁 Post-eval|Trả bài, dán điểm|Nộp chơi được|❌ không đổi hạng|