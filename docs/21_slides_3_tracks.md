---
marp: true
theme: default
paginate: true
header: 'VoiceMOS Challenge 2026'
footer: 'Tran Minh Toan · 10/6/2026'
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
## Tự động dự đoán điểm MOS cho giọng nói

**Tham gia cả 3 track** — trọng tâm **Track 2 (Emotional TTS)**

Tran Minh Toan · tranminhtoan14062001@gmail.com
Hướng tới paper ICASSP 2027

---

## Nội dung trình bày

1. **Bối cảnh** — VoiceMOS là gì? Metric SRCC. So sánh 3 track
2. **Track 1** — Speech Enhancement *(dùng baseline)*
3. **Track 3** — Codec synthesis *(dùng baseline)*
4. **Track 2 ⭐ — Emotional TTS** *(hệ thống tự phát triển)*
   - Động lực → Bài toán → Phương pháp → Kết quả → Phân tích
5. **Timeline · Đóng góp · Kết luận**

> Mỗi track trình bày theo mạch một bài báo (rút gọn): *động lực → bài toán → cách làm → kết quả*.

---

## Bối cảnh: MOS và VoiceMOS Challenge

- **MOS (Mean Opinion Score)** = điểm người nghe chấm chất lượng giọng (1–5). Là "tiêu chuẩn vàng" để đánh giá Text-to-Speech (TTS).
- Vấn đề: thu MOS bằng người **chậm, tốn kém** → cần **model tự động dự đoán MOS** thay người chấm.
- **VoiceMOS Challenge** = cuộc thi chuẩn hóa benchmark cho bài toán này. Bản **2026** có **3 track** với các loại giọng khác nhau.

**Metric chính: UTT-SRCC**
- **UTT** = chấm từng câu (utterance-level), không gộp theo hệ thống.
- **SRCC** = tương quan **thứ hạng** (Spearman). Đoán đúng *thứ tự* cao–thấp là được, không cần đúng giá trị tuyệt đối. **Càng gần 1 càng tốt.**

---

## So sánh nhanh 3 track

| | Track 1 | **Track 2 ⭐** | Track 3 |
|---|---|---|---|
| **Chủ đề** | Speech enhancement | Emotional TTS | Codec synthesis |
| **Dự đoán** | ACR + CCR | QMOS + EMOS (+CAT, VAD) | Speaker + Accent sim |
| **Cần reference?** | Không | Không | **Có** (cặp wav) |
| **Metric** | UTT-SRCC ×2 | UTT-SRCC ×5 + CAT-err | UTT-SRCC ×2 |
| **Cách của mình** | Baseline | **Hệ tự phát triển** | Baseline |

> Track 1 & 3: chạy baseline chính thức để có mặt trên leaderboard.
> Track 2: dồn toàn lực — đây là phần đóng góp khoa học.

---

<!-- _header: 'Track 1 — Speech Enhancement' -->

# Track 1 — Speech Enhancement

## Bài toán & cách tiếp cận

- **Nhiệm vụ:** dự đoán 2 điểm cho giọng đã qua khử nhiễu/tăng cường:
  - **ACR** (Absolute Category Rating) — chấm chất lượng tuyệt đối 1 audio.
  - **CCR** (Comparative Category Rating) — so sánh 2 audio.
- **Dữ liệu:** listening test của **URGENT Challenge** (ICASSP 2026), 9 ngôn ngữ. *Không có training data chính thức.*
- **Cách làm:** dùng baseline **URGENT-MOS** (multi-encoder: WavLM, Kimi-Audio, Qwen3-Omni, Audio Flamingo → fusion). **Chỉ inference, không train.**

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

- **4 encoder ❄ đóng băng** chạy song song → mỗi cái cho 1 vector câu `eᵢ`; **fusion** ghép lại thành `f`.
- 2 head riêng: **AMPM** hồi quy điểm tuyệt đối (ACR), **NCPM** ra điểm so sánh (CCR = hiệu 2 nhánh).
- Train gốc bằng **MSE** `L=(1/N)Σ(ŷ−y)²`, nhưng LB chấm **SRCC** (thứ hạng) → ta **chỉ inference**, không train lại.

---

<!-- _header: 'Track 1 — Speech Enhancement' -->

## Track 1 — Kết quả (DEV)

| Cột | UTT-SRCC |
|---|---|
| **ACR** | **0.662** |
| **CCR** | **0.411** |

- Khớp mức baseline công bố → pipeline chạy đúng.
- Vai trò: **đảm bảo có mặt trên leaderboard** cả 3 track, không phải hướng nghiên cứu chính.

---

<!-- _header: 'Track 3 — Codec Synthesis' -->

# Track 3 — Codec-based Synthesis

## Bài toán & cách tiếp cận

- **Nhiệm vụ:** cho 1 audio + **audio tham chiếu (reference)**, dự đoán độ tương đồng:
  - **Speaker similarity** — có giống *người nói* gốc không.
  - **Accent similarity** — có giống *giọng vùng miền* gốc không.
- **Dữ liệu:** **CodecMOS-Accent** (nền VCTK), 24 hệ thống tổng hợp (resynthesis + voice-clone). Train 2.800 · Val 600 · Eval 600.
- **Cách làm:** baseline dựa trên **speaker embedding** (ECAPA), so khớp cặp `wav_a` / `wav_b`. Chỉ inference.

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

- 2 audio qua **cùng một** encoder ECAPA-TDNN ❄ (Siamese — chia sẻ trọng số) → 2 embedding giọng.
- 🧮 Mỗi audio: `e∈ℝ^192` → chuẩn hóa `ê=e/‖e‖₂` → **cosine** `cos(êₐ,ê_b)=Σᵢ êₐ[i]·ê_b[i] ∈[−1,1]` (vì đã chuẩn hóa → tích vô hướng = cosine).
- **Zero-shot (bản ta nộp):** dùng thẳng cosine cho **cả 2 cột** → `spk_sim = acc_sim = cos`. Chỉ inference, không train.
- Hướng nâng cấp (fine-tune): vector tương tác `g=[eₐ ; e_b ; |eₐ−e_b| ; eₐ⊙e_b]` → MLP → `Tanh·2+3 ∈[1,5]`, **tách 2 head** spk/acc riêng.

---

<!-- _header: 'Track 3 — Codec Synthesis' -->

## Track 3 — Kết quả (DEV)

| Cột | UTT-SRCC |
|---|---|
| **Speaker sim** | **0.451** |
| **Accent sim** | **0.440** |

- Khớp mức baseline (~0.45/0.44).
- Như Track 1: **dùng baseline để phủ đủ 3 track**, không phát triển hệ riêng.

---

<!-- _header: 'Track 2 ⭐ — Emotional TTS' -->
<!-- _class: lead -->

# Track 2 — Emotional TTS ⭐
## Track chính của dự án

> Từ đây trình bày đầy đủ theo mạch một bài báo:
> **Động lực → Bài toán → Baseline → Phát hiện → Phương pháp → Kết quả → Phân tích**

---

<!-- _header: 'Track 2 ⭐ — Emotional TTS' -->

## Động lực: vì sao cần "thước đo cảm xúc"

- TTS đã ở khắp nơi (trợ lý ảo, sách nói, lồng tiếng, robot đồng hành…).
- Biên giới mới của TTS không còn là *nói gì* mà là **nói với cảm xúc nào** → "expressive TTS".
- **Nghẽn nằm ở khâu đánh giá:** giám khảo tự động hiện nay **chỉ chấm chất lượng**, *chưa biết chấm cảm xúc*.
- Một predictor cảm xúc đáng tin = **tín hiệu phản hồi** để *xây* TTS cảm xúc (so checkpoint, model selection, reward cho RLHF).

> 💡 Muốn AI **sinh** ra cảm xúc, trước hết phải **đo** được cảm xúc — cần một "emotional ruler".

---

<!-- _header: 'Track 2 ⭐ — Emotional TTS' -->

## Bài toán: dự đoán 6 cột cho mỗi câu

| Cột | Ý nghĩa | Bắt buộc | Metric |
|---|---|---|---|
| **QMOS** | Chất lượng giọng (1–5) | ✅ | SRCC ⬆ |
| **EMOS** | Độ khớp cảm xúc *target* (1–5) | ✅ | SRCC ⬆ |
| **CAT** | Phân bố 5 cảm xúc (vote) | ⬜ | error ⬇ |
| **VAD** | Valence / Arousal / Dominance | ⬜ | SRCC ⬆ |

- **Dữ liệu:** ESD + DailyTalk + 13 hệ TTS; 5 cảm xúc. Train **12.746** · Val 2.730 · Eval 2.730.
- Khó hơn MOS truyền thống: 1 câu phải chấm đồng thời **6 trục** chất lượng + cảm xúc.

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

<span style="font-size:15px">🟨 Vàng = **bắt buộc** (QMOS, EMOS) · 🟢 Xanh = **tùy chọn** (CAT, VAD). Cùng một backbone học chung → tiết kiệm + cộng hưởng giữa các nhiệm vụ.</span>

---

<!-- _header: 'Track 2 ⭐ — Emotional TTS' -->

## Baseline & điểm yếu

**Baseline = 3 model rời ghép lại:**
- UTMOS → QMOS · emotion2vec → CAT · **LLM zero-shot (Gemini)** → EMOS + VAD

**3 điểm yếu:**
1. 🔴 Dự đoán cảm xúc **rất yếu** — EMOS SRCC chỉ ~**0.19**.
2. 💸 LLM gọi API **tốn phí**, không chấm hết được toàn tập.
3. 🗑️ **Bỏ phí 12.746 nhãn người chấm** có sẵn (không train gì).

> → Cơ hội: thay "ráp model zero-shot" bằng **một model đa nhiệm có huấn luyện**.

---

<!-- _header: 'Track 2 ⭐ — Emotional TTS' -->

## Phát hiện (C1): hai encoder *bổ sung nhau*

So sánh từng model cảm xúc trên DEV:

| Model | EMOS | VAD (Arousal) |
|---|---|---|
| **emotion2vec** | **0.637** 🏆 | (không có VAD) |
| **SAILER** (WavLM SER) | 0.562 | **0.712** 🏆 |

- emotion2vec **thắng EMOS**, SAILER **thắng VAD** — *không model nào thắng mọi cột*.
- → Thay vì chọn 1 model tốt nhất, **gộp (fusion) cả hai** sẽ mạnh hơn.

---

<!-- _header: 'Track 2 ⭐ — Emotional TTS' -->

## Phương pháp 1 (C2): Fusion đa nhiệm 6 cột

<div style="text-align:center">
<svg viewBox="0 0 1000 470" style="width:96%;height:auto" font-family="Arial, sans-serif">
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

- 2 encoder **đóng băng** (❄) → trích đặc trưng → **trunk chung** → **4 head**.
- QMOS dùng thêm **điểm UTMOS làm neo** → lần đầu QMOS cải thiện (0.414 → **0.548**), *không kéo tụt cảm xúc*.
- Cân bằng 6 loss bằng **uncertainty weighting** (tự học trọng số mỗi task).
- 🧮 **Toán từng tầng:** pooling mỗi encoder `eᵢ=(1/L)Σₜ Hᵢ[t]` → concat `f=[e_e2v ; e_SAILER]` → trunk `z=ReLU(W₂·ReLU(W₁f+b₁)+b₂)` → head tuyến tính `ŷₜ=wₜ·z+bₜ` (QMOS: `ŷ=wq·[z ; UTMOS]`; CAT: `softmax`). Loss đa nhiệm + uncertainty: `L=Σₜ (1/2σₜ²)·Lₜ + log σₜ`.

---

<!-- _header: 'Track 2 ⭐ — Emotional TTS' -->

## Phương pháp 2 (C3): Fine-tune phá trần

<div style="text-align:center">
<svg viewBox="0 0 1000 340" style="width:92%;height:auto" font-family="Arial, sans-serif">
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

> **Ý chính:** chỉ **mở băng 6 lớp Transformer trên** của WavLM (warm-start từ SAILER) + audeering đóng băng → đủ học domain cảm xúc mà không OOM trên T4 → **vượt mọi cấu hình đóng băng ở cả 5 cột cảm xúc**.

- 🧮 **Toán từng tầng:** WavLM mở băng 6 lớp trên `H_top=WavLM_{6 lớp 🔥}(wav)` → mean-pool `e_w=(1/T)Σₜ H_top[t]` → concat audeering `f=[e_w ; e_aud]` → trunk → 3 head cảm xúc. Gradient **chỉ chảy về 6 lớp trên** (lớp dưới + CNN ❄). Loss như C2 (+ tùy chọn **ranking loss** tối ưu thẳng SRCC).

| Cột | Fusion đóng băng (exp07) | Fine-tune (exp08) |
|---|---|---|
| EMOS | 0.795 | **0.811** |
| VAL / ARO / DOM | 0.581 / 0.752 / 0.705 | **0.659 / 0.793 / 0.751** |
| CAT err ⬇ | 0.153 | **0.133** |

---

<!-- _header: 'Track 2 ⭐ — Emotional TTS' -->

## Kết quả Track 2: tiến hóa qua các bước

| Hệ thống | QMOS | EMOS | CAT err ⬇ | VAD (V/A/D) |
|---|---|---|---|---|
| Baseline | 0.414 | 0.194 | 0.193 | — |
| exp04 (fusion) | 0.414 | 0.788 | 0.145 | 0.58/0.75/0.71 |
| exp07 (+QMOS head) | **0.548** | 0.795 | 0.153 | 0.58/0.75/0.70 |
| exp08 (fine-tune) | 0.417 | **0.811** | **0.133** | 0.66/0.79/0.75 |
| **exp_mix (đã nộp)** | **0.548** | **0.811** | **0.133** | **0.659/0.793/0.751** |

- **exp_mix** = trộn cột: QMOS ← exp07, cảm xúc ← exp08 → **hệ 6 cột mạnh nhất** (best-per-column).
- 🚀 EMOS **0.19 → 0.81** · QMOS **0.41 → 0.55** so với baseline.

---

<!-- _header: 'Track 2 ⭐ — Emotional TTS' -->

## Phân tích & hướng mở rộng

**Phân tích:**
- SRCC chấm **thứ hạng** → dù VAD bị "nén" quanh 2.5–3.6, **thứ tự vẫn đúng** → điểm vẫn cao.
- Fusion **thắng mọi model lẻ** ở cả 5 cột cảm xúc → khẳng định giả thuyết "bổ sung nhau".

**Đang làm (future work — chất liệu cho paper):**
- 🐍 **Mamba temporal head** (exp14/15): thay mean-pool bằng SSM để bắt động lực thời gian.
- 🤖 **Audio-LLM-as-Judge** (exp16): khảo sát LLM nghe audio chấm cảm xúc — góc **novelty**.
- 📈 Thêm data cảm xúc ngoài (exp17) cho CAT/VAD; điều tra *neutral-bias*.

---

<!-- _header: 'Tổng kết' -->

## Timeline

| Mốc | Ngày |
|---|---|
| Training phase (đang mở) | đến 31/7/2026 |
| Eval set release | 31/7/2026 |
| 🔴 **Hạn nộp kết quả** | **7/8/2026** |
| Công bố kết quả | 31/8/2026 |
| Hạn nộp paper ICASSP 2027 | 16/9/2026 |

> Chiến lược phase Evaluation: đóng băng pipeline + script trộn cột + **fallback an toàn** = exp_mix.

---

<!-- _header: 'Tổng kết' -->

## Đóng góp & Kết luận

**Đóng góp chính (Track 2):**
- **(C1)** Phát hiện 2 SSL encoder cảm xúc **bổ sung nhau** (emotion2vec ↔ SAILER).
- **(C2)** **Một model đa nhiệm 6 cột** — fusion + neo UTMOS cho QMOS, không negative transfer.
- **(C3)** **Fine-tune** phá trần "đóng băng" ở cả 5 cột cảm xúc.

**Kết luận:** *Fusion biểu diễn bổ sung + fine-tune có giám sát* **vượt xa** việc ráp các model zero-shot cho MOS cảm xúc.

**Tài nguyên mở:** 3 repo Hugging Face (checkpoint · demo Gradio · code) — chạy offline, không tốn API.

---

<!-- _header: '' -->
<!-- _footer: '' -->

## Liên kết & Q&A

- **CodaBench (competition):** codabench.org/competitions/16419
- **Baseline chính thức:** github.com/voicemos-challenge/vmc2026-baselines
- **Hugging Face (`tranminhtoan140601`):** checkpoint · demo Space · code
- Website: sites.google.com/view/voicemos-challenge

# Cảm ơn — Q&A 🎤

<!--
Hướng dẫn render slide này (Marp):

⚠️ QUAN TRỌNG: slide có HÌNH KIẾN TRÚC vẽ bằng SVG inline → PHẢI bật HTML,
   không bật thì 3 hình (overview / fusion / fine-tune) sẽ bị ẩn.

1) VS Code: cài extension "Marp for VS Code"
   → Settings → tìm "Marp: Enable HTML" → BẬT (markdown.marp.enableHtml = true)
   → mở file → bấm preview (góc trên phải) → "..." → Export Slide Deck → PDF/PPTX/HTML.

2) CLI (cần Node.js) — NHỚ thêm cờ --html:
   npx @marp-team/marp-cli docs/21_slides_3_tracks.md --html -o slides.pptx
   npx @marp-team/marp-cli docs/21_slides_3_tracks.md --html --allow-local-files -o slides.pdf

Số liệu trong slide khớp docs/04_experiments_log.md (best-per-column, cập nhật 9/6/2026).
-->
