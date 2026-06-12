---
marp: true
theme: default
paginate: true
header: 'VoiceMOS Challenge 2026 — slide v2'
footer: 'Tran Minh Toan · 11/6/2026'
---

<style>
/* ===== Apple-clean Keynote theme ===== */
:root {
  --bg: #ffffff;
  --ink: #1d1d1f;     /* Apple near-black text */
  --muted: #86868b;   /* Apple gray */
  --soft: #424245;    /* secondary ink */
  --accent: #0071e3;  /* Apple blue */
  --card: #f5f5f7;    /* Apple light gray */
  --line: #e8e8ed;    /* hairline */
}
section {
  background: var(--bg);
  color: var(--ink);
  font-family: -apple-system, "SF Pro Display", "SF Pro Text", "Helvetica Neue", "Segoe UI", Arial, sans-serif;
  font-size: 20.5px;
  line-height: 1.45;
  letter-spacing: -0.01em;
  padding: 56px 72px;
  -webkit-font-smoothing: antialiased;
}
h1 { font-size: 44px; font-weight: 700; letter-spacing: -0.03em; line-height: 1.06; margin: 0 0 .25em; }
h2 {
  font-size: 33px; font-weight: 600; letter-spacing: -0.022em; line-height: 1.12;
  margin: 0 0 .55em; padding-bottom: .28em; border-bottom: 1px solid var(--line);
}
h3 { font-size: 24px; font-weight: 600; letter-spacing: -0.018em; margin: .6em 0 .3em; }
strong { font-weight: 700; color: var(--ink); }
em { color: var(--soft); font-style: italic; }
a { color: var(--accent); text-decoration: none; }

/* lists: minimal, accent dot */
ul { list-style: none; padding-left: 0; margin: .2em 0; }
ul > li { position: relative; padding-left: 1.2em; margin: .32em 0; }
ul > li::before {
  content: ""; position: absolute; left: .05em; top: .62em;
  width: 6px; height: 6px; border-radius: 50%; background: var(--accent);
}
ol { padding-left: 1.3em; margin: .2em 0; }
ol > li { margin: .32em 0; }
li::marker { color: var(--muted); }

/* tables: borderless, hairline rows (Apple style) */
table { border-collapse: collapse; width: 100%; font-size: 19px; margin: .3em 0; }
thead th {
  color: var(--muted); font-weight: 600; text-align: left;
  border: none; border-bottom: 2px solid var(--ink); padding: 12px 16px;
}
td { border: none; border-bottom: 1px solid var(--line); padding: 12px 16px; }
tbody tr:last-child td { border-bottom: none; }

/* blockquote: rounded soft card */
blockquote {
  background: var(--card); border: none; border-left: 3px solid var(--accent);
  border-radius: 14px; padding: 16px 22px; margin: .6em 0;
  color: var(--soft); font-size: 19px;
}
blockquote p { margin: .25em 0; }

/* inline code: subtle pill */
code {
  font-family: "SF Mono", ui-monospace, "Cascadia Code", Consolas, monospace;
  font-size: .88em; background: var(--card); color: var(--ink);
  padding: 2px 7px; border-radius: 6px;
}

/* compact table variant for dense layer tables */
section.dense table { font-size: 16px; }
section.dense thead th, section.dense td { padding: 8px 10px; }
section.dense { font-size: 18px; }

/* chrome: tiny + muted */
header { color: var(--muted); font-size: 13px; font-weight: 500; letter-spacing: 0; }
footer { color: var(--muted); font-size: 13px; }
section::after { color: var(--muted); font-size: 13px; font-weight: 500; }

/* ===== lead / title slides: big, centered, airy ===== */
section.lead {
  display: flex; flex-direction: column; justify-content: center;
  text-align: center; padding: 72px 96px;
}
section.lead h1 { font-size: 72px; line-height: 1.04; margin-bottom: .15em; }
section.lead h2 {
  font-size: 30px; font-weight: 500; color: var(--muted);
  border: none; padding: 0; margin: 0 0 1.1em; letter-spacing: -0.01em;
}
section.lead strong { color: var(--accent); }
section.lead p { color: var(--soft); font-size: 21px; line-height: 1.6; margin: .1em 0; }
</style>

<!-- _class: lead -->
<!-- _paginate: false -->
<!-- _header: '' -->
<!-- _footer: '' -->

# VoiceMOS Challenge 2026
## Dạy máy chấm điểm giọng nói như người nghe — bài toán · cách chấm · kiến trúc từng layer

**Tham gia cả 3 track** — trọng tâm **Track 2 (Emotional TTS)**

Tran Minh Toan · tranminhtoan140601@gmail.com
Slide v2 (theo mạch paper) · Hướng tới ICASSP 2027

---

## Nội dung — trình bày theo mạch một bài báo

1. **Introduction** — MOS là gì, vì sao cần model chấm tự động, "emotional ruler"
2. **Cách chấm điểm (Metrics)** — hai tầng điểm · SRCC ví dụ tính tay · 10 cột · CAT-ERR
3. **Track 1 — Speech Enhancement** *(baseline)*: bài toán → kiến trúc từng layer → kết quả
4. **Track 3 — Codec Synthesis** *(baseline)*: bài toán → kiến trúc từng layer → kết quả
5. **Track 2 ⭐ — Emotional TTS** *(hệ tự phát triển)*:
   Motivation → Bài toán → Baseline → Phát hiện C1 → Method C2/C3 + exp13 → Training → Kết quả → Ablation → Phân tích
6. **Conclusion** — timeline, đóng góp C1–C3, tài nguyên mở

> Mỗi track là một "paper rút gọn": *động lực → bài toán → cách làm → kết quả → phân tích*.

---

## 1. Introduction — bài toán MOS prediction

- **MOS (Mean Opinion Score)** = điểm người nghe chấm chất lượng giọng (thang 1–5). Là "tiêu chuẩn vàng" đánh giá Text-to-Speech (TTS), speech enhancement, codec.
- **Vấn đề:** thu MOS bằng người **chậm + tốn kém** (thuê hàng chục người nghe nghìn câu) → không thể lặp lại mỗi khi đổi checkpoint.
- **Bài toán:** xây **model dự đoán MOS** thay người chấm — một "giám khảo máy".
- Biên giới mới của TTS là **nói có cảm xúc** → cần giám khảo biết chấm cả cảm xúc:

> 💡 Muốn AI **sinh** ra cảm xúc, trước hết phải **đo** được cảm xúc — cần một **"emotional ruler"**. Một predictor đáng tin = tín hiệu phản hồi để xây TTS cảm xúc (so checkpoint, model selection, reward cho RLHF).

- **VoiceMOS Challenge** = benchmark chuẩn hóa cho bài toán này; bản **2026** có **3 track**.

---

## VoiceMOS 2026 — so sánh nhanh 3 track

| | Track 1 | **Track 2 ⭐** | Track 3 |
|---|---|---|---|
| **Chủ đề** | Speech enhancement | Emotional TTS | Codec synthesis |
| **Dự đoán** | ACR + CCR | QMOS + EMOS (+CAT, VAD) | Speaker + Accent sim |
| **Cần reference?** | Không | Không | **Có** (cặp wav) |
| **Train data?** | Không có | **12.746 câu có nhãn** | 2.800 cặp |
| **Metric** | UTT-SRCC ×2 | UTT-SRCC ×5 + CAT-err | UTT-SRCC ×2 |
| **Cách của mình** | Baseline | **Hệ tự phát triển** | Baseline |

> Track 1 & 3: chạy baseline chính thức để phủ đủ leaderboard. Track 2: dồn toàn lực — phần đóng góp khoa học. Hạn nộp eval: **7/8/2026**.

---

## 2. Cách chấm (1/3) — hai tầng điểm, đừng lẫn!

**Tầng 1 — điểm của mỗi AUDIO** (nhãn + dự đoán):

- Mỗi audio được **nhiều người nghe** chấm 1–5 → nhãn = **trung bình** (vì vậy có số lẻ: (5+5+4+5+5)/5 = 4.8).
- Model của mình cũng chấm mỗi audio 1 điểm (đầu ra hồi quy, liên tục).

**Tầng 2 — điểm của MODEL trên leaderboard:**

- Leaderboard **không** hỏi "audio này mấy điểm" mà hỏi "**giám khảo máy chấm có GIỐNG NGƯỜI không**" → so 2 danh sách điểm trên toàn tập → ra **1 số SRCC duy nhất cho cả model**.

| Mức so sánh | Cách tính | Độ khó |
|---|---|---|
| **UTT-level** (challenge dùng) | so thứ hạng **từng câu** trên cả 2.730 câu trộn chung | khó — phải xếp đúng từng câu lẻ |
| System-level | gộp trung bình theo **từng hệ TTS** (13 hệ → 13 điểm) rồi so | dễ hơn nhiều |

---

## Cách chấm (2/3) — SRCC: ví dụ tính tay

**SRCC** (Spearman) = tương quan **thứ hạng**: đổi điểm 2 bên thành hạng, đo độ khớp. Ví dụ 5 audio:

| Audio | Người chấm → hạng | Model → hạng | chênh d | d² |
|---|---|---|---|---|
| A | 4.8 → 1 | 3.9 → 1 | 0 | 0 |
| B | 4.2 → 2 | 3.5 → 2 | 0 | 0 |
| C | 3.5 → 3 | 3.1 → 3 | 0 | 0 |
| D | 2.9 → 4 | 2.2 → **5** | 1 | 1 |
| E | 2.1 → 5 | 2.4 → **4** | 1 | 1 |

$$SRCC = 1 - \frac{6\sum d_i^2}{n(n^2-1)} = 1 - \frac{6 \times 2}{5 \times 24} = 0.9$$

> 💡 Model chấm **thấp hơn người ~1 điểm ở mọi câu** — SRCC không phạt! Chỉ cần đúng **thứ tự** cao–thấp. Thang đo: 1 = trùng khớp · 0 = đoán bừa · −1 = xếp ngược. Hệ quả khi train: MSE (tối ưu giá trị) ≠ SRCC (tối ưu thứ tự) → có thể thêm *ranking loss*.

---

## Cách chấm (3/3) — 10 cột metric trên leaderboard

| Track | Cột | Là gì | Tốt khi |
|---|---|---|---|
| 1 | **ACR** | chất lượng tuyệt đối 1 audio (1–5) | SRCC ⬆ |
| 1 | **CCR** | so sánh cặp 2 audio (−3 → +3) | SRCC ⬆ |
| 2 | **QMOS** ✅ | chất lượng giọng (tự nhiên, sạch) | SRCC ⬆ |
| 2 | **EMOS** ✅ | độ khớp **cảm xúc target** | SRCC ⬆ |
| 2 | **CAT** | sai lệch **phân bố 5 cảm xúc** cảm nhận | **ERR ⬇** ⚠️ |
| 2 | **VAL / ARO / DOM** | 3 trục cảm xúc: tích cực · năng lượng · chi phối | SRCC ⬆ |
| 3 | **SPK / ACC** | giống người nói / giống accent so với reference | SRCC ⬆ |

> ⚠️ **9 cột SRCC cao = tốt; riêng CAT là ERR thấp = tốt** — đừng nhầm chiều. ✅ = bắt buộc (Track 2).

---

## CAT-ERR — công thức chính thức (từ BTC)

Nhãn CAT của mỗi audio = **tỉ lệ vote** 5 cảm xúc của người nghe (họ thường *không thống nhất* — đó là thông tin!). Model nộp 5 xác suất. Grader tính **MAE từng ô** trên bảng N×5:

$$\text{CAT-ERR} = \frac{1}{N \times 5} \sum_{i=1}^{N}\sum_{c=1}^{5} \left| \hat p_i(c) - p_i(c) \right|$$

Ví dụ 1 audio (10 người nghe: 6 happy · 3 neutral · 1 surprised):

| | angry | happy | neutral | sad | surprised |
|---|---|---|---|---|---|
| Người vote `p` | 0.0 | 0.6 | 0.3 | 0.0 | 0.1 |
| Model `p̂` | 0.05 | 0.45 | 0.40 | 0.05 | 0.05 |
| \|chênh\| | 0.05 | 0.15 | 0.10 | 0.05 | 0.05 → **tổng 0.40** |

> Mốc đọc số: đoán bừa ~0.25–0.30 · baseline 0.193 · **hệ của mình 0.1331** (lệch trung bình ~13 điểm % mỗi ô).

---

<!-- _header: 'Track 1 — Speech Enhancement' -->

# Track 1 — Speech Enhancement

## Bài toán & dữ liệu

- **Nhiệm vụ:** dự đoán 2 điểm cho giọng đã qua khử nhiễu/tăng cường:
  - **ACR** (Absolute Category Rating) — người nghe chấm chất lượng tuyệt đối 1 audio (1–5).
  - **CCR** (Comparative Category Rating) — người nghe so sánh 2 audio (−3 → +3).
- **Dữ liệu:** listening test của **URGENT Challenge** (ICASSP 2026), 9 ngôn ngữ. *Không có training data chính thức* → không thể train in-domain.
- **Cách tiếp cận:** dùng baseline **URGENT-MOS** (multi-encoder fusion). **Chỉ inference, không train** — mục tiêu là phủ leaderboard.

---

<!-- _header: 'Track 1 — Speech Enhancement' -->

## Kiến trúc URGENT-MOS — luồng tính toán

<div style="text-align:center">
<svg viewBox="0 0 1040 330" style="width:97%;height:auto" font-family="Arial, sans-serif">
  <defs>
    <marker id="arrU" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto">
      <path d="M0,0 L8,3 L0,6 Z" fill="#475569"/>
    </marker>
  </defs>

  <!-- khung "mỗi encoder ×4" -->
  <rect x="96" y="96" width="604" height="206" rx="14" fill="none" stroke="#3b82f6" stroke-width="2" stroke-dasharray="7 5"/>
  <text x="398" y="88" text-anchor="middle" font-size="15" font-weight="bold" fill="#1e3a8a">Mỗi encoder ❄ (×4): WavLM · Kimi-Audio · Qwen3-Omni · Audio-Flamingo</text>

  <!-- wav -->
  <rect x="10" y="113" width="74" height="74" rx="10" fill="#f1f5f9" stroke="#475569" stroke-width="2"/>
  <text x="47" y="148" text-anchor="middle" font-size="18">🎙</text>
  <text x="47" y="170" text-anchor="middle" font-size="13">wav</text>
  <text x="47" y="210" text-anchor="middle" font-size="12" fill="#475569">x ∈ ℝ^T</text>
  <text x="47" y="227" text-anchor="middle" font-size="12" fill="#475569">16 kHz</text>

  <!-- CNN -->
  <rect x="104" y="113" width="132" height="74" rx="10" fill="#dbeafe" stroke="#3b82f6" stroke-width="2"/>
  <text x="170" y="146" text-anchor="middle" font-size="16" font-weight="bold">CNN ×7</text>
  <text x="170" y="168" text-anchor="middle" font-size="13" fill="#1e3a8a">feature extractor</text>
  <text x="170" y="208" text-anchor="middle" font-size="12" fill="#475569">h[t]=Σₖ w[k]·x[s·t+k]</text>
  <text x="170" y="226" text-anchor="middle" font-size="12" fill="#475569">nén ~320× → Z∈ℝ^{L×d}</text>

  <!-- Transformer -->
  <rect x="258" y="113" width="158" height="74" rx="10" fill="#dbeafe" stroke="#3b82f6" stroke-width="2"/>
  <text x="337" y="146" text-anchor="middle" font-size="16" font-weight="bold">Transformer ×24</text>
  <text x="337" y="168" text-anchor="middle" font-size="13" fill="#1e3a8a">self-attention</text>
  <text x="337" y="208" text-anchor="middle" font-size="13" fill="#475569" font-weight="bold">softmax(QKᵀ/√d)·V</text>
  <text x="337" y="226" text-anchor="middle" font-size="12" fill="#475569">trộn ngữ cảnh toàn câu</text>

  <!-- layer-weighted sum -->
  <rect x="438" y="113" width="120" height="74" rx="10" fill="#dbeafe" stroke="#3b82f6" stroke-width="2"/>
  <text x="498" y="146" text-anchor="middle" font-size="15" font-weight="bold">Trộn lớp</text>
  <text x="498" y="168" text-anchor="middle" font-size="13" fill="#1e3a8a">αₗ học được</text>
  <text x="498" y="208" text-anchor="middle" font-size="12" fill="#475569">H=Σₗ αₗ·H⁽ˡ⁾</text>
  <text x="498" y="226" text-anchor="middle" font-size="12" fill="#475569">Σαₗ=1 (softmax)</text>

  <!-- pooling -->
  <rect x="580" y="113" width="120" height="74" rx="10" fill="#dbeafe" stroke="#3b82f6" stroke-width="2"/>
  <text x="640" y="146" text-anchor="middle" font-size="15" font-weight="bold">Mean pool</text>
  <text x="640" y="168" text-anchor="middle" font-size="13" fill="#1e3a8a">gộp thời gian</text>
  <text x="640" y="208" text-anchor="middle" font-size="12" fill="#475569">e=(1/L)Σₜ H[t]</text>
  <text x="640" y="226" text-anchor="middle" font-size="12" fill="#475569">→ eᵢ ∈ ℝ^dᵢ (×4)</text>

  <!-- fusion -->
  <rect x="722" y="100" width="150" height="100" rx="12" fill="#ede9fe" stroke="#7c3aed" stroke-width="2.5"/>
  <text x="797" y="134" text-anchor="middle" font-size="18" font-weight="bold">⊕ FUSION</text>
  <text x="797" y="158" text-anchor="middle" font-size="13" fill="#5b21b6">align + concat</text>
  <text x="797" y="182" text-anchor="middle" font-size="12" fill="#5b21b6">f=[ẽ₁;…;ẽ₄]∈ℝ^{4d_f}</text>

  <!-- head ACR -->
  <rect x="894" y="40" width="142" height="86" rx="10" fill="#fef3c7" stroke="#d97706" stroke-width="2"/>
  <text x="965" y="68" text-anchor="middle" font-size="16" font-weight="bold">AMPM → ACR</text>
  <text x="965" y="90" text-anchor="middle" font-size="11" fill="#92400e">ŷ=w·ReLU(Wf+b)</text>
  <text x="965" y="110" text-anchor="middle" font-size="12" fill="#92400e" font-weight="bold">ACR ∈ [1, 5]</text>

  <!-- head CCR -->
  <rect x="894" y="174" width="142" height="86" rx="10" fill="#dcfce7" stroke="#16a34a" stroke-width="2"/>
  <text x="965" y="202" text-anchor="middle" font-size="16" font-weight="bold">NCPM → CCR</text>
  <text x="965" y="224" text-anchor="middle" font-size="11" fill="#166534">ŷ=g(f_A)−g(f_B)</text>
  <text x="965" y="244" text-anchor="middle" font-size="12" fill="#166534" font-weight="bold">CCR ∈ [−3, +3]</text>

  <!-- arrows -->
  <g stroke="#475569" stroke-width="2" fill="none" marker-end="url(#arrU)">
    <path d="M84,150 L102,150"/>
    <path d="M236,150 L256,150"/>
    <path d="M416,150 L436,150"/>
    <path d="M558,150 L578,150"/>
    <path d="M700,150 L720,150"/>
    <path d="M872,140 L892,92"/>
    <path d="M872,160 L892,208"/>
  </g>
</svg>
</div>

- **4 encoder ❄ đóng băng** chạy song song → mỗi cái 1 vector câu `eᵢ`; **fusion** ghép thành `f` → 2 head.

---

<!-- _header: 'Track 1 — Speech Enhancement' -->
<!-- _class: dense -->

## Track 1 — từng layer làm gì

| # | Layer | Vào → Ra | Vai trò |
|---|---|---|---|
| 1 | **CNN ×7** (feature extractor) | sóng `x∈ℝ^T` → khung `Z∈ℝ^{L×d}` (nén ~320×) | đổi "sóng vật lý" thành chuỗi viên gạch âm học ~20ms; `h[t]=Σₖw[k]·x[s·t+k]` |
| 2 | **Transformer ×24** (self-attention) | `L×d` → `L×d` (shape giữ nguyên) | mỗi khung "nhìn" mọi khung khác — `softmax(QKᵀ/√d)·V` trộn ngữ cảnh toàn câu |
| 3 | **Trộn lớp** `H=Σₗ αₗH⁽ˡ⁾` | 24 đầu ra lớp → 1 | mỗi lớp giỏi 1 khía cạnh (âm học ↔ ngữ nghĩa); αₗ học được chọn "tầng nào hữu ích" |
| 4 | **Mean-pool** `e=(1/L)ΣₜH[t]` | `L×d` → `d` | cả câu nén thành 1 vector |
| 5 | **Fusion** ⊕ | 4 vector (4 encoder) → `f` | 4 "đôi tai" khác nhau bù khuyết điểm cho nhau |
| 6 | **AMPM head** | `f` → ACR ∈ [1,5] | hồi quy điểm tuyệt đối: `ŷ=w·ReLU(Wf+b)` |
| 7 | **NCPM head** | `f_A, f_B` → CCR ∈ [−3,+3] | điểm so sánh = **hiệu 2 nhánh** `g(f_A)−g(f_B)` |

> Train gốc bằng MSE; leaderboard chấm SRCC. Ta **chỉ inference** — checkpoint baseline đủ nhẹ chạy cả CPU.

---

<!-- _header: 'Track 1 — Speech Enhancement' -->

## Track 1 — Kết quả (DEV)

| Cột | UTT-SRCC |
|---|---|
| **ACR** | **0.662** |
| **CCR** | **0.411** |

- Khớp mức baseline công bố → pipeline chạy đúng.
- Vai trò: **đảm bảo có mặt trên leaderboard** cả 3 track; không phải hướng nghiên cứu chính.

---

<!-- _header: 'Track 3 — Codec Synthesis' -->

# Track 3 — Codec-based Synthesis

## Bài toán & dữ liệu

- **Nhiệm vụ:** cho 1 audio + **audio tham chiếu (reference)**, dự đoán độ tương đồng:
  - **Speaker similarity** — có giống *người nói* gốc không.
  - **Accent similarity** — có giống *giọng vùng miền* gốc không.
- **Dữ liệu:** **CodecMOS-Accent** (nền VCTK), 24 hệ tổng hợp (resynthesis + voice-clone). Train 2.800 · Val 600 · Eval 600.
- **Cách tiếp cận:** baseline **speaker embedding** (ECAPA-TDNN), so khớp cặp `wav_a`/`wav_b` bằng cosine. Chỉ inference.

---

<!-- _header: 'Track 3 — Codec Synthesis' -->

## Kiến trúc baseline Track 3 — luồng tính toán

<div style="text-align:center">
<svg viewBox="0 0 1040 300" style="width:97%;height:auto" font-family="Arial, sans-serif">
  <defs>
    <marker id="arrT3" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto">
      <path d="M0,0 L8,3 L0,6 Z" fill="#475569"/>
    </marker>
  </defs>

  <!-- wav_a / wav_b -->
  <rect x="12" y="46" width="92" height="56" rx="10" fill="#f1f5f9" stroke="#475569" stroke-width="2"/>
  <text x="58" y="80" text-anchor="middle" font-size="16">🎙 wav_a</text>
  <rect x="12" y="196" width="92" height="56" rx="10" fill="#f1f5f9" stroke="#475569" stroke-width="2"/>
  <text x="58" y="230" text-anchor="middle" font-size="16">🎙 wav_b</text>

  <!-- ECAPA encoders (Siamese, frozen) -->
  <rect x="150" y="42" width="216" height="64" rx="10" fill="#dbeafe" stroke="#3b82f6" stroke-width="2"/>
  <text x="258" y="70" text-anchor="middle" font-size="16" font-weight="bold">ECAPA-TDNN ❄</text>
  <text x="258" y="92" text-anchor="middle" font-size="13" fill="#1e3a8a">speaker embedding</text>
  <rect x="150" y="192" width="216" height="64" rx="10" fill="#dbeafe" stroke="#3b82f6" stroke-width="2"/>
  <text x="258" y="220" text-anchor="middle" font-size="16" font-weight="bold">ECAPA-TDNN ❄</text>
  <text x="258" y="242" text-anchor="middle" font-size="13" fill="#1e3a8a">speaker embedding</text>
  <!-- shared-weights note -->
  <text x="258" y="132" text-anchor="middle" font-size="12" fill="#475569">TDNN + SE-block + attentive stat-pool</text>
  <text x="258" y="150" text-anchor="middle" font-size="12" fill="#475569">→ e ∈ ℝ^192</text>
  <text x="258" y="170" text-anchor="middle" font-size="12" fill="#7c3aed" font-weight="bold">Siamese: CÙNG trọng số ❄</text>

  <!-- L2 normalize -->
  <rect x="404" y="46" width="120" height="56" rx="10" fill="#dbeafe" stroke="#3b82f6" stroke-width="2"/>
  <text x="464" y="72" text-anchor="middle" font-size="15">L2 normalize</text>
  <text x="464" y="92" text-anchor="middle" font-size="12" fill="#1e3a8a">ê=e/‖e‖₂</text>
  <rect x="404" y="196" width="120" height="56" rx="10" fill="#dbeafe" stroke="#3b82f6" stroke-width="2"/>
  <text x="464" y="222" text-anchor="middle" font-size="15">L2 normalize</text>
  <text x="464" y="242" text-anchor="middle" font-size="12" fill="#1e3a8a">ê=e/‖e‖₂</text>

  <!-- cosine -->
  <rect x="566" y="108" width="150" height="84" rx="12" fill="#ede9fe" stroke="#7c3aed" stroke-width="2.5"/>
  <text x="641" y="142" text-anchor="middle" font-size="18" font-weight="bold">COSINE</text>
  <text x="641" y="166" text-anchor="middle" font-size="13" fill="#5b21b6">ê_a · ê_b</text>
  <text x="641" y="214" text-anchor="middle" font-size="12" fill="#475569">cos=Σᵢ êₐ[i]·ê_b[i] ∈ [−1,1]</text>

  <!-- outputs -->
  <rect x="766" y="84" width="184" height="58" rx="10" fill="#fef3c7" stroke="#d97706" stroke-width="2"/>
  <text x="858" y="119" text-anchor="middle" font-size="16" font-weight="bold">Speaker sim</text>
  <rect x="766" y="176" width="184" height="58" rx="10" fill="#dcfce7" stroke="#16a34a" stroke-width="2"/>
  <text x="858" y="211" text-anchor="middle" font-size="16" font-weight="bold">Accent sim</text>
  <text x="858" y="258" text-anchor="middle" font-size="12" fill="#475569">zero-shot: spk = acc = cos</text>

  <!-- arrows -->
  <g stroke="#475569" stroke-width="2" fill="none" marker-end="url(#arrT3)">
    <path d="M104,74 L148,74"/>
    <path d="M104,224 L148,224"/>
    <path d="M366,74 L402,74"/>
    <path d="M366,224 L402,224"/>
    <path d="M524,74 L564,128"/>
    <path d="M524,224 L564,172"/>
    <path d="M716,138 L764,116"/>
    <path d="M716,160 L764,200"/>
  </g>
</svg>
</div>

- 2 audio qua **cùng một** encoder (Siamese — chia sẻ trọng số ❄) → 2 embedding → cosine.

---

<!-- _header: 'Track 3 — Codec Synthesis' -->
<!-- _class: dense -->

## Track 3 — từng layer làm gì

| # | Layer | Vào → Ra | Vai trò |
|---|---|---|---|
| 1 | **TDNN** (dilated Conv1D nhiều tầng) | wav → chuỗi khung | conv giãn nở — mỗi khung gom ngữ cảnh nhiều khung lân cận (đặc trưng giọng nói) |
| 2 | **SE-block** (Squeeze-Excitation) | khung → khung (có trọng số kênh) | "chú ý theo kênh": tự học kênh đặc trưng nào quan trọng thì khuếch đại |
| 3 | **Attentive stat-pooling** | chuỗi khung → 1 vector | gộp thời gian bằng **trung bình + độ lệch chuẩn có trọng số attention** (khung nào "đặc trưng giọng" hơn được chú ý hơn) |
| 4 | **Linear** → `e ∈ ℝ¹⁹²` | pooled → 192-D | nén thành **danh tính giọng nói** (speaker embedding) |
| 5 | **L2-normalize** `ê = e/‖e‖₂` | 192-D → 192-D (độ dài 1) | đưa mọi embedding lên mặt cầu đơn vị → tích vô hướng = cosine |
| 6 | **Cosine** `cos = Σᵢ êₐ[i]·ê_b[i]` | 2 vector → 1 số ∈ [−1,1] | độ giống nhau về hướng = độ giống giọng |

- **Zero-shot (bản nộp):** dùng thẳng cosine cho **cả 2 cột** → `spk = acc = cos`. Không train.
- Hướng nâng cấp: vector tương tác `g=[eₐ; e_b; |eₐ−e_b|; eₐ⊙e_b]` → MLP → 2 head spk/acc riêng.

---

<!-- _header: 'Track 3 — Codec Synthesis' -->

## Track 3 — Kết quả (DEV)

| Cột | UTT-SRCC |
|---|---|
| **Speaker sim** | **0.451** |
| **Accent sim** | **0.440** |

- Khớp mức baseline (~0.45/0.44).
- Hạn chế tự nhiên của zero-shot: ECAPA học **danh tính người nói**, không học **accent** → 2 cột dùng chung 1 điểm cosine; muốn tách phải fine-tune 2 head riêng (hướng mở).

---

<!-- _header: 'Track 2 ⭐ — Emotional TTS' -->
<!-- _class: lead -->

# Track 2 — Emotional TTS ⭐
## Track chính của dự án

> Từ đây trình bày đầy đủ theo mạch một bài báo:
> **Motivation → Bài toán → Baseline → Phát hiện C1 → Method C2 · C3 · exp13 → Training → Kết quả → Ablation → Phân tích**

---

<!-- _header: 'Track 2 ⭐ — Emotional TTS' -->

## Motivation: vì sao cần "thước đo cảm xúc"

- TTS đã ở khắp nơi (trợ lý ảo, sách nói, lồng tiếng, robot đồng hành…).
- Biên giới mới không còn là *nói gì* mà là **nói với cảm xúc nào** → "expressive TTS".
- **Nghẽn nằm ở khâu đánh giá:** giám khảo tự động hiện nay **chỉ chấm chất lượng**, *chưa biết chấm cảm xúc*.
- Một predictor cảm xúc đáng tin = **tín hiệu phản hồi** để *xây* TTS cảm xúc:
  - so sánh checkpoint / model selection không cần thuê người nghe;
  - làm **reward model** cho RLHF;
  - đã tự kiểm chứng: dùng chính hệ của mình chấm TTS tiếng Việt (VoxCPM2) → phần Phân tích.

---

<!-- _header: 'Track 2 ⭐ — Emotional TTS' -->

## Bài toán: 1 câu nói → 6 cột điểm

| Cột | Ý nghĩa | Bắt buộc | Metric |
|---|---|---|---|
| **QMOS** | Chất lượng giọng (1–5) | ✅ | SRCC ⬆ |
| **EMOS** | Độ khớp cảm xúc *target* (1–5) | ✅ | SRCC ⬆ |
| **CAT** | Phân bố 5 cảm xúc người nghe cảm nhận | ⬜ | ERR ⬇ |
| **VAL / ARO / DOM** | 3 trục Valence / Arousal / Dominance | ⬜ | SRCC ⬆ |

- **Dữ liệu:** ESD + DailyTalk + 13 hệ TTS · 5 cảm xúc · Train **12.746** (có nhãn từng người nghe) · Dev 2.730 · Eval 2.730.
- Format nộp — chú ý CAT là **phân bố xác suất**:

```
wav,QMOS,EMOS,CAT,VAL,ARO,DOM
sys012-utt041.wav,3.59,1,angry:0.00|happy:0.0001|neutral:0.98|sad:0.018|surprised:0.00,1,5,5
```

---

<!-- _header: 'Track 2 ⭐ — Emotional TTS' -->

## Mục tiêu: một model → sáu đầu ra

<div style="text-align:center">
<svg viewBox="0 0 1000 220" style="width:96%;height:auto" font-family="Arial, sans-serif">
  <defs>
    <marker id="arrC" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto">
      <path d="M0,0 L8,3 L0,6 Z" fill="#475569"/>
    </marker>
  </defs>
  <!-- input -->
  <rect x="18" y="70" width="256" height="84" rx="12" fill="#f1f5f9" stroke="#475569" stroke-width="2"/>
  <text x="146" y="104" text-anchor="middle" font-size="17">🎙 1 câu nói (wav)</text>
  <text x="146" y="130" text-anchor="middle" font-size="15" fill="#475569">+ cảm xúc target</text>
  <!-- system -->
  <rect x="352" y="56" width="248" height="108" rx="14" fill="#e0e7ff" stroke="#4338ca" stroke-width="2.5"/>
  <text x="476" y="100" text-anchor="middle" font-size="19" font-weight="bold" fill="#312e81">HỆ THỐNG</text>
  <text x="476" y="126" text-anchor="middle" font-size="16" fill="#3730a3">MOS cảm xúc</text>
  <text x="476" y="148" text-anchor="middle" font-size="12" fill="#4f46e5">(1 model · 6 đầu ra)</text>
  <!-- chips -->
  <g font-size="15" font-weight="bold" text-anchor="middle">
    <rect x="660" y="52" width="104" height="46" rx="8" fill="#fef3c7" stroke="#d97706" stroke-width="2"/>
    <text x="712" y="81" fill="#92400e">QMOS</text>
    <rect x="772" y="52" width="104" height="46" rx="8" fill="#fef3c7" stroke="#d97706" stroke-width="2"/>
    <text x="824" y="81" fill="#92400e">EMOS</text>
    <rect x="884" y="52" width="104" height="46" rx="8" fill="#dcfce7" stroke="#16a34a" stroke-width="2"/>
    <text x="936" y="81" fill="#166534">CAT</text>
    <rect x="660" y="120" width="104" height="46" rx="8" fill="#dcfce7" stroke="#16a34a" stroke-width="2"/>
    <text x="712" y="149" fill="#166534">VAL</text>
    <rect x="772" y="120" width="104" height="46" rx="8" fill="#dcfce7" stroke="#16a34a" stroke-width="2"/>
    <text x="824" y="149" fill="#166534">ARO</text>
    <rect x="884" y="120" width="104" height="46" rx="8" fill="#dcfce7" stroke="#16a34a" stroke-width="2"/>
    <text x="936" y="149" fill="#166534">DOM</text>
  </g>
  <!-- arrows -->
  <g stroke="#475569" stroke-width="2.5" fill="none" marker-end="url(#arrC)">
    <path d="M274,112 L350,110"/>
    <path d="M600,110 L654,100"/>
    <path d="M600,112 L654,140"/>
  </g>
</svg>
</div>

<span style="font-size:15px">🟨 Vàng = **bắt buộc** (QMOS, EMOS) · 🟢 Xanh = **tùy chọn** (CAT, VAD). Cùng một backbone học chung → tiết kiệm + cộng hưởng giữa các nhiệm vụ (multi-task).</span>

---

<!-- _header: 'Track 2 ⭐ — Emotional TTS' -->

## Baseline chính thức & 3 điểm yếu

**Baseline = 3 model rời ghép lại (đều zero-shot):**
- UTMOS → QMOS · emotion2vec → CAT · **LLM (Gemini)** → EMOS + VAD

**3 điểm yếu:**
1. 🔴 Dự đoán cảm xúc **rất yếu** — EMOS SRCC chỉ ~**0.19** (gần mức đoán bừa).
2. 💸 LLM gọi API **tốn phí**, không chấm hết được toàn tập.
3. 🗑️ **Bỏ phí 12.746 nhãn người chấm** có sẵn (không train gì).

> → Cơ hội: thay "ráp model zero-shot" bằng **một model đa nhiệm có huấn luyện**.

---

<!-- _header: 'Track 2 ⭐ — Emotional TTS' -->

## Phát hiện (C1): hai encoder cảm xúc *bổ sung nhau*

So sánh từng model SSL cảm xúc trên DEV:

| Model | EMOS | VAD (Arousal) |
|---|---|---|
| **emotion2vec** | **0.637** 🏆 | (không có VAD) |
| **SAILER** (WavLM SER) | 0.562 | **0.712** 🏆 |

- emotion2vec **thắng EMOS**, SAILER **thắng VAD** — *không model nào thắng mọi cột*.
- → Thay vì chọn 1 model tốt nhất, **gộp (fusion) cả hai** sẽ mạnh hơn từng cái lẻ.
- Đây là phát hiện dẫn đường cho toàn bộ method phía sau.

---

<!-- _header: 'Track 2 ⭐ — Emotional TTS' -->

## Method 1 (C2): Fusion đa nhiệm 6 cột — sơ đồ

<div style="text-align:center">
<svg viewBox="0 0 1000 470" style="width:88%;height:auto" font-family="Arial, sans-serif">
  <defs>
    <marker id="arrF" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto">
      <path d="M0,0 L8,3 L0,6 Z" fill="#475569"/>
    </marker>
  </defs>
  <!-- wav -->
  <rect x="18" y="205" width="92" height="60" rx="10" fill="#f1f5f9" stroke="#475569" stroke-width="2"/>
  <text x="64" y="242" text-anchor="middle" font-size="20">🎙 wav</text>
  <!-- encoders (frozen) -->
  <rect x="158" y="118" width="224" height="74" rx="10" fill="#dbeafe" stroke="#3b82f6" stroke-width="2"/>
  <text x="270" y="148" text-anchor="middle" font-size="19" font-weight="bold">emotion2vec ❄</text>
  <text x="270" y="173" text-anchor="middle" font-size="14" fill="#1e3a8a">emb + 5 xác suất cảm xúc</text>
  <rect x="158" y="278" width="224" height="74" rx="10" fill="#dbeafe" stroke="#3b82f6" stroke-width="2"/>
  <text x="270" y="308" text-anchor="middle" font-size="19" font-weight="bold">SAILER / WavLM ❄</text>
  <text x="270" y="333" text-anchor="middle" font-size="14" fill="#1e3a8a">emb + xác suất + VAD</text>
  <!-- concat -->
  <rect x="428" y="205" width="64" height="60" rx="10" fill="#ffffff" stroke="#475569" stroke-width="2"/>
  <text x="460" y="244" text-anchor="middle" font-size="24">⊕</text>
  <!-- trunk -->
  <rect x="540" y="178" width="168" height="114" rx="12" fill="#ede9fe" stroke="#7c3aed" stroke-width="2.5"/>
  <text x="624" y="224" text-anchor="middle" font-size="20" font-weight="bold">TRUNK chung</text>
  <text x="624" y="252" text-anchor="middle" font-size="14" fill="#5b21b6">Linear→ReLU ×2</text>
  <!-- heads -->
  <rect x="766" y="28" width="216" height="82" rx="10" fill="#fef3c7" stroke="#d97706" stroke-width="2"/>
  <text x="874" y="60" text-anchor="middle" font-size="19" font-weight="bold">QMOS</text>
  <text x="874" y="86" text-anchor="middle" font-size="14" fill="#92400e">+ điểm UTMOS làm neo</text>
  <rect x="766" y="138" width="216" height="82" rx="10" fill="#dcfce7" stroke="#16a34a" stroke-width="2"/>
  <text x="874" y="170" text-anchor="middle" font-size="19" font-weight="bold">EMOS</text>
  <text x="874" y="196" text-anchor="middle" font-size="14" fill="#166534">+ one-hot cảm xúc target</text>
  <rect x="766" y="248" width="216" height="82" rx="10" fill="#dcfce7" stroke="#16a34a" stroke-width="2"/>
  <text x="874" y="280" text-anchor="middle" font-size="19" font-weight="bold">CAT</text>
  <text x="874" y="306" text-anchor="middle" font-size="14" fill="#166534">softmax 5 lớp cảm xúc</text>
  <rect x="766" y="358" width="216" height="82" rx="10" fill="#dcfce7" stroke="#16a34a" stroke-width="2"/>
  <text x="874" y="390" text-anchor="middle" font-size="19" font-weight="bold">VAD</text>
  <text x="874" y="416" text-anchor="middle" font-size="14" fill="#166534">Valence · Arousal · Dominance</text>
  <!-- arrows -->
  <g stroke="#475569" stroke-width="2" fill="none" marker-end="url(#arrF)">
    <path d="M110,228 L156,160"/>
    <path d="M110,242 L156,312"/>
    <path d="M382,155 L426,224"/>
    <path d="M382,315 L426,246"/>
    <path d="M492,235 L538,235"/>
    <path d="M708,232 L762,70"/>
    <path d="M708,234 L762,180"/>
    <path d="M708,236 L762,290"/>
    <path d="M708,238 L762,398"/>
  </g>
</svg>
</div>

- 2 encoder **❄ frozen** → đặc trưng trích 1 lần, **cache .npz** → mỗi epoch chỉ chạy MLP nhỏ (rẻ, lặp nhanh).

---

<!-- _header: 'Track 2 ⭐ — Emotional TTS' -->
<!-- _class: dense -->

## C2 (exp07) — từng layer làm gì

| # | Layer | Vào → Ra | Train? | Vai trò |
|---|---|---|---|---|
| B1 | emotion2vec+ large | wav → emb 1024 + 5 prob = **1029** | ❄ cache | backbone cảm xúc #1 (một mình đạt EMOS 0.637) |
| B2 | SAILER WavLM-large | wav → emb 1024 + 9 prob + VAD 3 = **1036** | ❄ cache | backbone cảm xúc #2 — góc nhìn khác cho fusion |
| B3 | UTMOS22_strong | wav → **1** điểm MOS | ❄ cache | **neo chất lượng** cho QMOS head |
| B4 | concat + z-score → **TRUNK** MLP ×2 (ReLU, Dropout 0.3) | 1029+1036 = **2065 → 512** | ✅ LR 1e-3 | não chung — nén 2 nguồn thành biểu diễn multi-task |
| B5a | **QMOS head** `[trunk 512 \| UTMOS 1]` | **513** → 128 → 1 | ✅ | chỉ học *chỉnh sửa quanh* UTMOS (residual) — sàn = 0.414, khó tệ hơn |
| B5b | EMOS head `[trunk \| target one-hot 5]` | **517** → 128 → 1 | ✅ | so khớp cảm xúc *được yêu cầu* |
| B5c–d | CAT head · VAD head | 512 → 128 → 5/3 | ✅ | phân bố vote · 3 trục cảm xúc |

> **Kết quả then chốt:** QMOS head hưởng ké trunk được 5 task cảm xúc "nuôi" → QMOS **0.414 → 0.548** — lần đầu vượt UTMOS zero-shot, *không kéo tụt cột cảm xúc*.

---

<!-- _header: 'Track 2 ⭐ — Emotional TTS' -->

## Giải phẫu 3 head — đầu ra phải mô phỏng đúng bản chất nhãn

| | EMOS head | CAT head | VAD head |
|---|---|---|---|
| **Đầu vào** | `[trunk 512 \| one-hot target 5]` = 517 | trunk 512 | trunk 512 |
| **Đầu ra** | 1 số → **×σ+μ** | 5 số → **softmax** | 3 số → **×σ+μ** |
| **Vì sao** | câu hỏi **so khớp 2 thứ** — cùng audio, đổi target thì điểm phải đổi → bắt buộc nối target vào | nhãn = **phân bố vote** (người nghe bất đồng!) → ra phân bố, không phải 1 nhãn cứng | nhãn z-score hóa khi train (trung bình 0, lệch 1 — dễ hội tụ) → dự đoán phải giải ngược về thang 1–5 |
| **Loss** | MSE | **soft cross-entropy** với phân bố vote | MSE từng trục |

> ⚠️ Bài học thực chiến: **μ/σ là một phần của model** — phải lưu trong checkpoint, mất là dự đoán thành số vô nghĩa. One-hot = vector 5 số toàn 0 trừ đúng 1 số 1 đánh dấu cảm xúc target.

---

<!-- _header: 'Track 2 ⭐ — Emotional TTS' -->

## Method 2 (C3): Fine-tune phá trần — sơ đồ

<div style="text-align:center">
<svg viewBox="0 0 1000 340" style="width:86%;height:auto" font-family="Arial, sans-serif">
  <defs>
    <marker id="arrB" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto">
      <path d="M0,0 L8,3 L0,6 Z" fill="#475569"/>
    </marker>
  </defs>
  <!-- WavLM stack -->
  <text x="135" y="28" text-anchor="middle" font-size="16" font-weight="bold" fill="#334155">WavLM-large</text>
  <rect x="40" y="38" width="190" height="96" rx="8" fill="#ffedd5" stroke="#ea580c" stroke-width="2.5"/>
  <text x="135" y="80" text-anchor="middle" font-size="16" font-weight="bold" fill="#9a3412">6 lớp trên 🔥</text>
  <text x="135" y="104" text-anchor="middle" font-size="14" fill="#9a3412">mở băng (train)</text>
  <rect x="40" y="134" width="190" height="120" rx="8" fill="#e2e8f0" stroke="#94a3b8" stroke-width="2"/>
  <text x="135" y="186" text-anchor="middle" font-size="15" fill="#475569">lớp dưới + CNN ❄</text>
  <text x="135" y="208" text-anchor="middle" font-size="14" fill="#475569">đóng băng</text>
  <!-- warm-start -->
  <rect x="40" y="282" width="190" height="40" rx="8" fill="#fae8ff" stroke="#c026d3" stroke-width="2"/>
  <text x="135" y="307" text-anchor="middle" font-size="14" fill="#86198f">SAILER (warm-start)</text>
  <!-- mean-pool -->
  <rect x="288" y="58" width="116" height="58" rx="10" fill="#f1f5f9" stroke="#475569" stroke-width="2"/>
  <text x="346" y="86" text-anchor="middle" font-size="15">mean-pool</text>
  <text x="346" y="104" text-anchor="middle" font-size="12" fill="#64748b">(B,T,1024)→1024</text>
  <!-- audeering frozen -->
  <rect x="268" y="186" width="156" height="72" rx="10" fill="#dbeafe" stroke="#3b82f6" stroke-width="2"/>
  <text x="346" y="216" text-anchor="middle" font-size="15" font-weight="bold" fill="#1e3a8a">audeering ❄</text>
  <text x="346" y="240" text-anchor="middle" font-size="13" fill="#1e3a8a">[emb | VAD]</text>
  <!-- concat -->
  <rect x="470" y="112" width="58" height="56" rx="10" fill="#ffffff" stroke="#475569" stroke-width="2"/>
  <text x="499" y="148" text-anchor="middle" font-size="22">⊕</text>
  <!-- trunk -->
  <rect x="572" y="92" width="152" height="96" rx="12" fill="#ede9fe" stroke="#7c3aed" stroke-width="2.5"/>
  <text x="648" y="135" text-anchor="middle" font-size="18" font-weight="bold">TRUNK</text>
  <text x="648" y="160" text-anchor="middle" font-size="13" fill="#5b21b6">chung</text>
  <!-- heads -->
  <rect x="772" y="40" width="182" height="68" rx="10" fill="#dcfce7" stroke="#16a34a" stroke-width="2"/>
  <text x="863" y="80" text-anchor="middle" font-size="17" font-weight="bold">EMOS</text>
  <rect x="772" y="128" width="182" height="68" rx="10" fill="#dcfce7" stroke="#16a34a" stroke-width="2"/>
  <text x="863" y="168" text-anchor="middle" font-size="17" font-weight="bold">CAT</text>
  <rect x="772" y="216" width="182" height="68" rx="10" fill="#dcfce7" stroke="#16a34a" stroke-width="2"/>
  <text x="863" y="256" text-anchor="middle" font-size="17" font-weight="bold">VAD</text>
  <!-- arrows -->
  <g stroke="#475569" stroke-width="2" fill="none" marker-end="url(#arrB)">
    <path d="M230,86 L286,86"/>
    <path d="M346,116 L346,184" stroke-dasharray="0"/>
    <path d="M404,90 L468,130"/>
    <path d="M424,218 L468,150"/>
    <path d="M528,140 L570,140"/>
    <path d="M724,138 L770,76"/>
    <path d="M724,140 L770,162"/>
    <path d="M724,142 L770,248"/>
  </g>
  <!-- warm-start arrow (dashed) -->
  <path d="M135,282 L135,138" stroke="#c026d3" stroke-width="2" fill="none" stroke-dasharray="6 4" marker-end="url(#arrB)"/>
</svg>
</div>

> **Ý chính:** đặc trưng frozen chỉ cho ta thứ "như nó vốn có" — head nhỏ không vặn lại được. Mở băng 6 lớp trên = cho phép **chính biểu diễn xoay về domain** ESD/DailyTalk.

---

<!-- _header: 'Track 2 ⭐ — Emotional TTS' -->
<!-- _class: dense -->

## C3 (exp08) — từng layer làm gì

| # | Layer | Vào → Ra | Train? | Vai trò |
|---|---|---|---|---|
| A1a | WavLM **CNN ×7** | wav 8s → 399 khung × 512 (nén 320×) | ❄ | sóng → "viên gạch âm học" ~20ms — phổ quát, không nên phá |
| A1b | WavLM **Transformer 18 lớp dưới** | 399×1024 → 399×1024 | ❄ | đặc trưng âm học chung (pitch, formant) |
| A1c | WavLM **Transformer 6 lớp trên** | 399×1024 → 399×1024 | 🔥 **LR 1e-5** | tầng ngữ nghĩa/cảm xúc — chỉnh nhẹ quanh warm-start SAILER, đủ khớp domain mà không quên kiến thức cũ |
| A1d | mean-pool (attention-mask) | 399×1024 → **1024** | — | cả câu nén 1 vector |
| A2 | audeering wav2vec2 + head VAD gốc | wav → emb 1024 + VAD 3 = **1027** | ❄ cache | "chuyên gia VAD" độc lập — góc nhìn thứ hai |
| A3 | concat → **TRUNK** MLP ×2 | 1024+1027 = **2051 → 512** | ✅ LR 1e-3 | não chung multi-task |
| A4 | 3 head: EMOS (517) · CAT (512→5) · VAD (512→3) | 512 → 128 → ra | ✅ | như slide "giải phẫu head" |

> Tỉ lệ tham số: WavLM 315M (chỉ ~75M ở 6 lớp mở) · audeering 315M (0 train) · trunk+head **~1.3M**. Phần "dạy từ đầu" chỉ là cái đuôi — sức mạnh còn lại là mượn + tinh chỉnh.

---

<!-- _header: 'Track 2 ⭐ — Emotional TTS' -->

## Training details — các quyết định ăn điểm

**Loss đa nhiệm + uncertainty weighting (Kendall 2018):**

$$L = \sum_{t}\, e^{-s_t} L_t + s_t \qquad (s_t = \log\sigma_t^2,\ \text{5 tham số HỌC ĐƯỢC})$$

- Task nhiễu → model tự tăng σₜ → giảm trọng số (nhưng trả phí +sₜ, không trốn việc được) → **khỏi mò trọng số tay**. Loss thành phần: MSE (EMOS/VAD, nhãn z-scored) + soft-CE (CAT).

**Chạy vừa Kaggle T4 16GB:**

| Kỹ thuật | Tác dụng |
|---|---|
| **BATCH 4 × ACCUM 8** (effective 32) | gradient accumulation: gom gradient 8 lượt mới step 1 lần — *mượt như batch 32, VRAM như batch 4* |
| AMP (mixed precision) + grad-checkpointing | giảm ~nửa bộ nhớ activation |
| Cắt audio 8s · layerdrop=0 | chặn OOM · tái lập được |
| Early-stop theo SRCC val + **lưu ckpt mỗi best** | bài học mất backbone exp08: Save Version ngay |

---

<!-- _header: 'Track 2 ⭐ — Emotional TTS' -->

## Method 3 — exp13: QMOS cần fine-tune ĐÚNG model

- **Quan sát:** exp08 fine-tune cảm xúc rất mạnh nhưng QMOS chỉ 0.417 — biểu diễn bị kéo nghiêng về cảm xúc, tín hiệu *chất lượng* không được chăm.
- **Giả thuyết:** UTMOS kẹt 0.414 vì **lệch domain** (train trên giọng không-cảm-xúc 2022) — không phải vì kiến trúc yếu.
- **Cách làm (exp13):** fine-tune **chính UTMOS** (`utmos22_strong`) trên nhãn `qMOS` thật:
  - LR 1e-5 (warm-start đã tốt, chỉ chỉnh nhẹ) · BATCH 1 × ACCUM 16 · cắt 12s · đóng băng CNN;
  - loss MSE thuần (`RANK_LAMBDA=0`; pairwise ranking loss đã code sẵn — hướng mở);
  - lưới an toàn: chỉ nộp nếu SRCC val > zero-shot.

> 🏆 **QMOS 0.548 → 0.6296** — kỷ lục cột, xác nhận giả thuyết domain. Bài học ghép với exp08: **mỗi cột điểm một "khẩu vị"** — cảm xúc cần fine-tune encoder cảm xúc, chất lượng cần fine-tune encoder chất lượng.

---

<!-- _header: 'Track 2 ⭐ — Emotional TTS' -->

## Kết quả: tiến hóa qua các bước (DEV, UTT-level)

| Hệ thống | QMOS | EMOS | CAT ⬇ | VAL / ARO / DOM |
|---|---|---|---|---|
| Baseline BTC | 0.414 | 0.194 | 0.193 | — |
| exp04 fusion frozen | 0.414 | 0.788 | 0.145 | 0.58 / 0.75 / 0.71 |
| exp07 + QMOS head | 0.548 | 0.795 | 0.153 | 0.58 / 0.75 / 0.70 |
| exp08 fine-tune WavLM | 0.417 | 0.811 | **0.133** | 0.66 / 0.79 / 0.75 |
| exp13 fine-tune UTMOS | **0.6296** | — | — | — |
| exp15 Mamba head (2 ckpt) | 0.6296 | 0.807 | 0.135 | 0.65 / **0.798** / 0.75 |
| **Best-per-column (10/6)** | **0.6296** | **0.8116** | **0.1331** | **0.6605 / 0.7978 / 0.7539** |

- 🚀 So với baseline: EMOS **0.19 → 0.81** · QMOS **0.41 → 0.63**.

---

<!-- _header: 'Track 2 ⭐ — Emotional TTS' -->

## Trộn cột (exp_mix): lấy cái tốt nhất của từng hệ

- Grader CodaBench chấm `answer.txt` **từng cột độc lập** → được phép **ghép cột từ nhiều model** (đã xác nhận bằng điểm thật 9/6: khớp đúng best-per-column).
- Bản đã nộp: QMOS←exp07 + 5 cột cảm xúc←exp08 → **fallback an toàn** cho phase Evaluation.
- Thế hệ mới (sẵn sàng nộp): QMOS←**exp13** + ARO←**exp15** + EMOS/CAT/VAL/DOM←exp08 — 0 giờ GPU.

> **Bài học kiến trúc:** fine-tune thắng frozen ở cảm xúc, nhưng QMOS cần neo + đúng domain — hai nhu cầu kéo về hai model khác nhau → trộn cột là cách *rẻ nhất* hợp nhất ưu điểm. Đây cũng là chiến lược ensemble tự nhiên cho eval.

---

<!-- _header: 'Track 2 ⭐ — Emotional TTS' -->

## Ablation — chất liệu cho paper

**1. Mamba temporal head vs mean-pool (exp15, có điểm thật 10/6):**

| Pooling | QMOS | EMOS | CAT ⬇ | VAL | ARO | DOM |
|---|---|---|---|---|---|---|
| mean-pool (exp08) | — | **0.811** | **0.133** | **0.659** | 0.793 | **0.751** |
| **Mamba** (exp15) | — | 0.807 | 0.135 | 0.654 | **0.7978** 🏆 | 0.7506 |

- **Gần hòa** — Mamba thắng đúng **Arousal** (cột biến thiên theo thời gian rõ nhất: selective SSM bắt động lực mà mean-pool xóa mất). Đúng như lý thuyết dự đoán.

**2. Ranking loss (pairwise hinge):** loss `relu(−sign·diff)` trên ~120 cặp/cửa sổ ACCUM — đã code trong exp13/exp15, kỷ lục hiện tại **chưa cần bật** (MSE thuần đã vượt) → ablation MSE vs MSE+rank là bước rẻ tiếp theo.

**3. Khởi tạo WavLM (exp12, đang dở):** scratch vs base-SSL vs SAILER warm-start.

---

<!-- _header: 'Track 2 ⭐ — Emotional TTS' -->

## Phân tích — 2 insight chính

**1. SRCC "cứu" VAD bị nén dải:**
- Model dự đoán VAD dồn trong dải hẹp 2.5–3.6 (thay vì 1–5) — nhưng **thứ tự đúng** → SRCC vẫn 0.79. Nếu chấm MSE thì thảm họa; hiểu metric giúp chọn đúng thứ để tối ưu.

**2. Neutral-bias — case study TTS tiếng Việt (VoxCPM2):**
- Dùng chính hệ exp08 chấm audio tiếng Việt sinh theo 7 cảm xúc: argmax CAT **luôn ra neutral** (accuracy 20%)…
- …**nhưng VAD đúng hướng**: arousal angry 3.71 / surprised 3.66 cao, sad 3.33 thấp nhất; valence happy cao nhất → scorer *thật sự cảm nhận* cảm xúc, chỉ đầu phân loại bị kéo về neutral (lệch domain tiếng Anh).
- Khắc phục phía đo: metric ranking (khử neutral khi argmax + SRCC theo prototype VAD) — đúng tinh thần "emotional ruler" của paper.

---

<!-- _header: 'Track 2 ⭐ — Emotional TTS' -->

## Hướng mở rộng (future work)

- 🔁 **Resume exp15** (Mamba) với LR giảm — chốt kết quả cuối; nộp **bản trộn cột thế hệ mới** (QMOS←exp13 + ARO←exp15 + còn lại←exp08).
- 📐 **Ranking loss ablation**: bật `RANK_LAMBDA=0.3` cho exp13 — tối ưu thẳng thứ hạng (metric là SRCC).
- 🤖 **Audio-LLM-as-Judge (exp16)**: khảo sát LLM nghe audio chấm 6 cột — góc *novelty*, đã code, chờ smoke test.
- 📈 **Data cảm xúc ngoài (exp17)** cho CAT/VAD; giải neutral-bias tận gốc (emotion2vec đa ngôn ngữ cho tiếng Việt).

---

<!-- _header: 'Tổng kết' -->

## Timeline & chiến lược phase Evaluation

| Mốc | Ngày |
|---|---|
| Training phase (đang mở) | đến 31/7/2026 |
| Eval set release (2.730 câu) | 31/7/2026 |
| 🔴 **Hạn nộp kết quả** | **7/8/2026** |
| Công bố kết quả | 31/8/2026 |
| Hạn nộp paper ICASSP 2027 | 16/9/2026 |

> Cửa sổ eval chỉ ~1 tuần → chuẩn bị TRƯỚC: đóng băng pipeline inference từng cột · script **trộn cột + validate format** test sẵn trên DEV · checkpoint đã lưu chắc trên Kaggle Dataset · **fallback an toàn** = exp_mix đã nộp.

---

<!-- _header: 'Tổng kết' -->

## Đóng góp & Kết luận

**Đóng góp chính (Track 2):**
- **(C1)** Phát hiện 2 SSL encoder cảm xúc **bổ sung nhau** (emotion2vec ↔ SAILER).
- **(C2)** **Một model đa nhiệm 6 cột** — fusion + neo UTMOS cho QMOS, multi-task không negative transfer.
- **(C3)** **Fine-tune đúng domain phá trần**: WavLM cho 5 cột cảm xúc, UTMOS cho QMOS — "mỗi cột một khẩu vị" → trộn cột.

**Kết luận:** *Fusion biểu diễn bổ sung + fine-tune có giám sát đúng domain* **vượt xa** việc ráp các model zero-shot: EMOS 0.19→0.81, QMOS 0.41→0.63.

**Tài nguyên mở:** 3 repo Hugging Face (checkpoint · demo Gradio · code) + **API service REST 3 track** (HF Space, free CPU) — chạy offline, không tốn API trả phí.

---

<!-- _header: '' -->
<!-- _footer: '' -->

## Liên kết & Q&A

- **CodaBench (competition):** codabench.org/competitions/16419
- **Baseline chính thức:** github.com/voicemos-challenge/vmc2026-baselines
- **Hugging Face (`tranminhtoan140601`):** checkpoint · demo Space `voicemos2026-demo` · API `voicemos2026-api`
- Website: sites.google.com/view/voicemos-challenge

# Cảm ơn — Q&A 🎤

<!--
Hướng dẫn render slide này (Marp):

⚠️ QUAN TRỌNG: slide có HÌNH KIẾN TRÚC vẽ bằng SVG inline → PHẢI bật HTML,
   không bật thì 5 hình sẽ bị ẩn.

1) VS Code: cài extension "Marp for VS Code"
   → Settings → "Marp: Enable HTML" → BẬT (markdown.marp.enableHtml = true)
   → mở file → preview → "..." → Export Slide Deck → PDF/PPTX/HTML.

2) CLI (cần Node.js) — NHỚ cờ --html (+ --no-stdin nếu treo):
   npx @marp-team/marp-cli docs/22_slides_v2_paper_style.md --html --no-stdin -o slide/voicemos2026_slides_v2.html
   npx @marp-team/marp-cli docs/22_slides_v2_paper_style.md --html --allow-local-files --no-stdin -o slides_v2.pdf

Số liệu khớp docs/04_experiments_log.md (best-per-column cập nhật 10/6/2026):
QMOS 0.6296 (exp13) · EMOS 0.8116 (exp08b) · CAT 0.1331 (exp08) · VAL 0.6605 (exp08b) · ARO 0.7978 (exp15) · DOM 0.7539 (exp08b).
Đây là slide v2 (paper-style, ~36 slide); bản v1 ngắn hơn ở docs/21_slides_3_tracks.md.
-->
