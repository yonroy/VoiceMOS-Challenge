## Mỗi cột leaderboard ← model nào (baseline hiện tại)


| Track | Cột leaderboard | Model/Phương pháp đang dùng | Điểm hiện có |
|---|---|---|---|
| 🟦 1 | ACR UTT-SRCC | URGENT-MOS (1 model ra cả 2 cột) | 0.662 |
| 🟦 1 | CCR UTT-SRCC | ↳ cũng từ URGENT-MOS | 0.411 |
| 🟥 2 | QMOS UTT-SRCC | UTMOS / SpeechMOS | 0.414 |
| 🟥 2 | EMOS UTT-SRCC | Gemini (LLM-as-judge) | 0.194 (một phần) |
| 🟥 2 | CAT error | emotion2vec | 0.193 ⬇️|
| 🟥 2 | Valence UTT-SRCC | Gemini (1 model ra cả 3 cột VAD) | chưa chạy đủ|
| 🟥 2 | Arousal UTT-SRCC | ↳ cũng từ Gemini | chưa chạy đủ|
| 🟥 2 | Dominance UTT-SRCC | ↳ cũng từ Gemini | chưa chạy đủ|
| 🟩 3 | Speaker UTT-SRCC | ECAPA-TDNN (1 model ra cả 2 cột) | 0.451|
| 🟩 3 | Accent UTT-SRCC | ↳ cũng từ ECAPA-TDNN | 0.440|