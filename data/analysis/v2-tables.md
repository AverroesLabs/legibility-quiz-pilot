### All core papers pooled (12 papers, 716 questions, 3 samples)

| Reader | No paper | Original | Obfuscated (Δ) | Compressed (Δ) | Compressed + obfuscated (Δ) |
|---|---|---|---|---|---|
| GPT-5.6-sol (smart) | 0.35 [0.34, 0.37] | 0.80 [0.78, 0.82] | 0.79 (−0.01 [−0.02, +0.00]) | 0.79 (−0.00 [−0.02, +0.01]) | 0.77 (**−0.03 [−0.04, −0.02]**) |
| Gemma-4-31B | 0.10 [0.09, 0.11] | 0.51 [0.49, 0.54] | 0.47 (**−0.05 [−0.06, −0.03]**) | 0.49 (**−0.02 [−0.04, −0.01]**) | 0.42 (**−0.09 [−0.11, −0.07]**) |
| Ministral-8B | 0.17 [0.16, 0.18] | 0.35 [0.34, 0.37] | 0.32 (**−0.03 [−0.05, −0.02]**) | 0.33 (**−0.02 [−0.04, −0.01]**) | 0.30 (**−0.05 [−0.07, −0.04]**) |
| Llama-3.2-3B | 0.10 [0.09, 0.11] | 0.15 [0.14, 0.16] | 0.14 (**−0.02 [−0.03, −0.00]**) | 0.13 (**−0.02 [−0.03, −0.01]**) | 0.13 (**−0.02 [−0.03, −0.01]**) |

Score [95% interval]; Δ is the change from the original, paired by question; bold where its interval excludes 0. Intervals resample questions within each paper, so papers keep their weight. All versions included, whether or not they pass a paper's information check. From [`analyze_v2.py`](../scripts/quiz/analyze_v2.py).

### 2608.25490v1 (95 questions)

| Reader | No paper | Original | Obfuscated (Δ) | Compressed (Δ) | Compressed + obfuscated (Δ) |
|---|---|---|---|---|---|
| GPT-5.6-sol (smart) | 0.33 [0.28, 0.37] | 0.85 [0.80, 0.89] | 0.84 (−0.00 [−0.03, +0.02]) | 0.84 (−0.01 [−0.04, +0.02]) | 0.82 (−0.03 [−0.06, +0.01]) |
| Gemma-4-31B | 0.01 [0.00, 0.01] | 0.57 [0.50, 0.64] | 0.51 (**−0.06 [−0.10, −0.02]**) | 0.56 (−0.01 [−0.06, +0.03]) | 0.48 (**−0.09 [−0.13, −0.05]**) |
| Ministral-8B | 0.16 [0.12, 0.20] | 0.35 [0.30, 0.41] | 0.29 (**−0.06 [−0.10, −0.02]**) | 0.36 (+0.01 [−0.04, +0.05]) | 0.33 (−0.03 [−0.07, +0.02]) |
| Llama-3.2-3B | 0.07 [0.04, 0.10] | 0.13 [0.10, 0.16] | 0.14 (+0.01 [−0.02, +0.05]) | 0.13 (−0.00 [−0.03, +0.03]) | 0.13 (+0.00 [−0.03, +0.04]) |

Information check (smart reader's score / original): obf 1.00, comp 0.99, compobf 0.97. Lengths (bytes): comp 0.37x, compobf 0.39x. Failed grades 0, unanswered 28. Data: [`takes-summary.json`](../data/runs/v2-2608.25490v1/takes-summary.json).

### 2608.30724v1 (85 questions)

| Reader | No paper | Original | Obfuscated (Δ) | Compressed (Δ) | Compressed + obfuscated (Δ) |
|---|---|---|---|---|---|
| GPT-5.6-sol (smart) | 0.34 [0.30, 0.39] | 0.82 [0.77, 0.86] | 0.80 (−0.02 [−0.06, +0.01]) | 0.79 (−0.03 [−0.06, +0.01]) | 0.80 (−0.01 [−0.06, +0.03]) |
| Gemma-4-31B | 0.07 [0.05, 0.10] | 0.44 [0.38, 0.51] | 0.44 (−0.01 [−0.05, +0.04]) | 0.47 (+0.03 [−0.02, +0.08]) | 0.44 (−0.01 [−0.05, +0.03]) |
| Ministral-8B | 0.19 [0.15, 0.24] | 0.37 [0.32, 0.43] | 0.31 (**−0.06 [−0.11, −0.02]**) | 0.34 (−0.03 [−0.08, +0.01]) | 0.35 (−0.02 [−0.08, +0.02]) |
| Llama-3.2-3B | 0.12 [0.09, 0.16] | 0.18 [0.14, 0.22] | 0.14 (**−0.04 [−0.07, −0.00]**) | 0.15 (**−0.03 [−0.05, −0.00]**) | 0.16 (−0.02 [−0.05, +0.01]) |

Information check (smart reader's score / original): obf 0.97, comp 0.97, compobf 0.98. Lengths (bytes): comp 0.49x, compobf 0.52x. Failed grades 0, unanswered 5. Data: [`takes-summary.json`](../data/runs/v2-2608.30724v1/takes-summary.json).

### 2609.00755v1 (80 questions)

| Reader | No paper | Original | Obfuscated (Δ) | Compressed (Δ) | Compressed + obfuscated (Δ) |
|---|---|---|---|---|---|
| GPT-5.6-sol (smart) | 0.30 [0.25, 0.35] | 0.78 [0.72, 0.83] | 0.72 (**−0.05 [−0.09, −0.01]**) | 0.77 (−0.01 [−0.05, +0.04]) | 0.73 (−0.05 [−0.10, +0.00]) |
| Gemma-4-31B | 0.16 [0.12, 0.21] | 0.42 [0.36, 0.48] | 0.34 (**−0.08 [−0.12, −0.03]**) | 0.44 (+0.02 [−0.03, +0.07]) | 0.33 (**−0.08 [−0.15, −0.02]**) |
| Ministral-8B | 0.18 [0.14, 0.23] | 0.31 [0.25, 0.36] | 0.25 (**−0.05 [−0.10, −0.00]**) | 0.31 (−0.00 [−0.05, +0.04]) | 0.25 (**−0.06 [−0.11, −0.01]**) |
| Llama-3.2-3B | 0.10 [0.07, 0.14] | 0.18 [0.14, 0.22] | 0.15 (−0.03 [−0.06, +0.00]) | 0.16 (−0.02 [−0.05, +0.01]) | 0.18 (−0.00 [−0.03, +0.03]) |

Information check (smart reader's score / original): obf 0.93, comp 0.99, compobf 0.94. Lengths (bytes): comp 0.37x, compobf 0.40x. Failed grades 0, unanswered 1. Data: [`takes-summary.json`](../data/runs/v2-2609.00755v1/takes-summary.json).

### 2609.30094v2 (73 questions)

| Reader | No paper | Original | Obfuscated (Δ) | Compressed (Δ) | Compressed + obfuscated (Δ) |
|---|---|---|---|---|---|
| GPT-5.6-sol (smart) | 0.33 [0.27, 0.39] | 0.83 [0.79, 0.88] | 0.83 (−0.00 [−0.03, +0.03]) | 0.83 (+0.00 [−0.03, +0.03]) | 0.80 (−0.03 [−0.07, +0.00]) |
| Gemma-4-31B | 0.03 [0.01, 0.05] | 0.58 [0.51, 0.65] | 0.50 (**−0.08 [−0.15, −0.01]**) | 0.46 (**−0.12 [−0.18, −0.07]**) | 0.42 (**−0.16 [−0.23, −0.10]**) |
| Ministral-8B | 0.12 [0.08, 0.16] | 0.21 [0.15, 0.26] | 0.26 (**+0.05 [+0.00, +0.11]**) | 0.24 (+0.03 [−0.02, +0.08]) | 0.24 (+0.03 [−0.02, +0.08]) |
| Llama-3.2-3B | 0.06 [0.04, 0.09] | 0.08 [0.05, 0.11] | 0.06 (−0.02 [−0.04, +0.01]) | 0.11 (+0.03 [−0.00, +0.06]) | 0.09 (+0.01 [−0.02, +0.04]) |

Information check (smart reader's score / original): obf 1.00, comp 1.00, compobf 0.96. Lengths (bytes): comp 0.41x, compobf 0.43x. Failed grades 0, unanswered 97. Data: [`takes-summary.json`](../data/runs/v2-2609.30094v2/takes-summary.json).

### 2607.28685v1 (71 questions)

| Reader | No paper | Original | Obfuscated (Δ) | Compressed (Δ) | Compressed + obfuscated (Δ) |
|---|---|---|---|---|---|
| GPT-5.6-sol (smart) | 0.31 [0.26, 0.36] | 0.83 [0.78, 0.88] | 0.83 (−0.01 [−0.05, +0.03]) | 0.84 (+0.01 [−0.03, +0.05]) | 0.79 (**−0.04 [−0.08, −0.00]**) |
| Gemma-4-31B | 0.08 [0.05, 0.11] | 0.58 [0.51, 0.65] | 0.51 (**−0.08 [−0.12, −0.03]**) | 0.55 (−0.03 [−0.09, +0.03]) | 0.48 (**−0.11 [−0.17, −0.04]**) |
| Ministral-8B | 0.13 [0.10, 0.17] | 0.35 [0.29, 0.41] | 0.35 (−0.00 [−0.04, +0.04]) | 0.29 (**−0.06 [−0.11, −0.01]**) | 0.25 (**−0.10 [−0.15, −0.04]**) |
| Llama-3.2-3B | 0.08 [0.05, 0.11] | 0.11 [0.08, 0.15] | 0.12 (+0.01 [−0.03, +0.04]) | 0.08 (**−0.03 [−0.06, −0.00]**) | 0.07 (**−0.04 [−0.07, −0.01]**) |

Information check (smart reader's score / original): obf 0.99, comp 1.01, compobf 0.95. Lengths (bytes): comp 0.40x, compobf 0.40x. Failed grades 0, unanswered 6. Data: [`takes-summary.json`](../data/runs/v2-2607.28685v1/takes-summary.json).

### 2607.20379v2 (63 questions)

| Reader | No paper | Original | Obfuscated (Δ) | Compressed (Δ) | Compressed + obfuscated (Δ) |
|---|---|---|---|---|---|
| GPT-5.6-sol (smart) | 0.38 [0.33, 0.43] | 0.75 [0.69, 0.81] | 0.73 (−0.02 [−0.07, +0.02]) | 0.77 (+0.01 [−0.04, +0.07]) | 0.76 (+0.00 [−0.05, +0.06]) |
| Gemma-4-31B | 0.21 [0.17, 0.26] | 0.56 [0.49, 0.63] | 0.51 (**−0.05 [−0.08, −0.01]**) | 0.54 (−0.02 [−0.07, +0.03]) | 0.43 (**−0.13 [−0.20, −0.06]**) |
| Ministral-8B | 0.19 [0.14, 0.23] | 0.40 [0.35, 0.46] | 0.34 (**−0.06 [−0.11, −0.01]**) | 0.36 (−0.04 [−0.10, +0.02]) | 0.34 (**−0.06 [−0.12, −0.01]**) |
| Llama-3.2-3B | 0.10 [0.07, 0.14] | 0.15 [0.10, 0.19] | 0.12 (−0.03 [−0.06, +0.01]) | 0.13 (−0.02 [−0.06, +0.02]) | 0.11 (**−0.04 [−0.07, −0.01]**) |

Information check (smart reader's score / original): obf 0.97, comp 1.01, compobf 1.00. Lengths (bytes): comp 0.46x, compobf 0.48x. Failed grades 0, unanswered 3. Data: [`takes-summary.json`](../data/runs/v2-2607.20379v2/takes-summary.json).

### 2608.13754v1 (61 questions)

| Reader | No paper | Original | Obfuscated (Δ) | Compressed (Δ) | Compressed + obfuscated (Δ) |
|---|---|---|---|---|---|
| GPT-5.6-sol (smart) | 0.34 [0.28, 0.40] | 0.78 [0.72, 0.83] | 0.75 (−0.02 [−0.07, +0.02]) | 0.76 (−0.02 [−0.07, +0.02]) | 0.74 (−0.04 [−0.08, −0.00]) |
| Gemma-4-31B | 0.04 [0.02, 0.07] | 0.51 [0.43, 0.60] | 0.46 (**−0.06 [−0.11, −0.00]**) | 0.40 (**−0.11 [−0.15, −0.07]**) | 0.34 (**−0.18 [−0.25, −0.11]**) |
| Ministral-8B | 0.13 [0.08, 0.17] | 0.34 [0.27, 0.41] | 0.28 (**−0.06 [−0.11, −0.02]**) | 0.28 (**−0.06 [−0.11, −0.01]**) | 0.22 (**−0.12 [−0.18, −0.07]**) |
| Llama-3.2-3B | 0.10 [0.06, 0.14] | 0.12 [0.08, 0.17] | 0.11 (−0.01 [−0.05, +0.03]) | 0.09 (−0.03 [−0.06, +0.01]) | 0.11 (−0.01 [−0.04, +0.02]) |

Information check (smart reader's score / original): obf 0.97, comp 0.97, compobf 0.95. Lengths (bytes): comp 0.43x, compobf 0.46x. Failed grades 0, unanswered 3. Data: [`takes-summary.json`](../data/runs/v2-2608.13754v1/takes-summary.json).

### 2609.02893v1 (53 questions)

| Reader | No paper | Original | Obfuscated (Δ) | Compressed (Δ) | Compressed + obfuscated (Δ) |
|---|---|---|---|---|---|
| GPT-5.6-sol (smart) | 0.39 [0.33, 0.44] | 0.75 [0.68, 0.81] | 0.80 (**+0.06 [+0.02, +0.10]**) | 0.76 (+0.01 [−0.03, +0.06]) | 0.73 (−0.01 [−0.05, +0.03]) |
| Gemma-4-31B | 0.13 [0.09, 0.18] | 0.40 [0.31, 0.50] | 0.41 (+0.00 [−0.06, +0.06]) | 0.44 (+0.04 [−0.02, +0.10]) | 0.44 (+0.03 [−0.05, +0.11]) |
| Ministral-8B | 0.19 [0.14, 0.25] | 0.35 [0.27, 0.43] | 0.33 (−0.02 [−0.05, +0.02]) | 0.33 (−0.02 [−0.07, +0.03]) | 0.31 (−0.04 [−0.10, +0.01]) |
| Llama-3.2-3B | 0.13 [0.09, 0.18] | 0.24 [0.19, 0.30] | 0.18 (**−0.07 [−0.11, −0.03]**) | 0.20 (**−0.04 [−0.07, −0.01]**) | 0.19 (**−0.05 [−0.09, −0.01]**) |

Information check (smart reader's score / original): obf 1.08, comp 1.02, compobf 0.99. Lengths (bytes): comp 0.37x, compobf 0.40x. Failed grades 0, unanswered 3. Data: [`takes-summary.json`](../data/runs/v2-2609.02893v1/takes-summary.json).

### 2609.34091v1 (44 questions)

| Reader | No paper | Original | Obfuscated (Δ) | Compressed (Δ) | Compressed + obfuscated (Δ) |
|---|---|---|---|---|---|
| GPT-5.6-sol (smart) | 0.43 [0.35, 0.50] | 0.78 [0.68, 0.86] | 0.79 (+0.02 [−0.04, +0.06]) | 0.77 (−0.00 [−0.06, +0.06]) | 0.72 (**−0.06 [−0.11, −0.01]**) |
| Gemma-4-31B | 0.15 [0.10, 0.19] | 0.42 [0.32, 0.51] | 0.37 (−0.05 [−0.11, +0.01]) | 0.38 (−0.03 [−0.11, +0.04]) | 0.32 (**−0.10 [−0.18, −0.02]**) |
| Ministral-8B | 0.21 [0.15, 0.27] | 0.31 [0.24, 0.38] | 0.27 (−0.04 [−0.10, +0.02]) | 0.30 (−0.01 [−0.06, +0.03]) | 0.28 (−0.03 [−0.08, +0.01]) |
| Llama-3.2-3B | 0.09 [0.05, 0.14] | 0.12 [0.08, 0.17] | 0.14 (+0.02 [−0.02, +0.06]) | 0.12 (−0.00 [−0.05, +0.04]) | 0.14 (+0.02 [−0.02, +0.06]) |

Information check (smart reader's score / original): obf 1.02, comp 1.00, compobf 0.92. Lengths (bytes): comp 0.38x, compobf 0.40x. Failed grades 0, unanswered 1. Data: [`takes-summary.json`](../data/runs/v2-2609.34091v1/takes-summary.json).

### 2608.19216v1 (42 questions)

| Reader | No paper | Original | Obfuscated (Δ) | Compressed (Δ) | Compressed + obfuscated (Δ) |
|---|---|---|---|---|---|
| GPT-5.6-sol (smart) | 0.44 [0.35, 0.54] | 0.75 [0.67, 0.82] | 0.76 (+0.01 [−0.03, +0.05]) | 0.77 (+0.02 [−0.03, +0.07]) | 0.75 (+0.00 [−0.05, +0.04]) |
| Gemma-4-31B | 0.13 [0.09, 0.18] | 0.62 [0.52, 0.71] | 0.61 (−0.01 [−0.05, +0.03]) | 0.64 (+0.02 [−0.04, +0.08]) | 0.60 (−0.02 [−0.08, +0.04]) |
| Ministral-8B | 0.23 [0.17, 0.29] | 0.56 [0.47, 0.63] | 0.50 (−0.06 [−0.13, +0.01]) | 0.48 (−0.07 [−0.16, +0.01]) | 0.41 (**−0.15 [−0.23, −0.07]**) |
| Llama-3.2-3B | 0.14 [0.08, 0.20] | 0.25 [0.18, 0.32] | 0.26 (+0.01 [−0.05, +0.07]) | 0.17 (**−0.08 [−0.15, −0.01]**) | 0.20 (−0.05 [−0.11, +0.01]) |

Information check (smart reader's score / original): obf 1.01, comp 1.03, compobf 1.00. Lengths (bytes): comp 0.41x, compobf 0.43x. Failed grades 0, unanswered 2. Data: [`takes-summary.json`](../data/runs/v2-2608.19216v1/takes-summary.json).

### 2607.17117v2 (38 questions)

| Reader | No paper | Original | Obfuscated (Δ) | Compressed (Δ) | Compressed + obfuscated (Δ) |
|---|---|---|---|---|---|
| GPT-5.6-sol (smart) | 0.42 [0.34, 0.50] | 0.79 [0.72, 0.86] | 0.76 (−0.03 [−0.10, +0.03]) | 0.79 (−0.00 [−0.07, +0.06]) | 0.71 (−0.08 [−0.18, +0.00]) |
| Gemma-4-31B | 0.16 [0.10, 0.23] | 0.49 [0.37, 0.61] | 0.49 (−0.00 [−0.06, +0.06]) | 0.51 (+0.02 [−0.04, +0.08]) | 0.39 (**−0.10 [−0.17, −0.04]**) |
| Ministral-8B | 0.17 [0.11, 0.23] | 0.39 [0.29, 0.50] | 0.40 (+0.01 [−0.05, +0.07]) | 0.39 (−0.00 [−0.06, +0.06]) | 0.38 (−0.02 [−0.09, +0.05]) |
| Llama-3.2-3B | 0.10 [0.05, 0.15] | 0.14 [0.08, 0.19] | 0.11 (−0.03 [−0.09, +0.03]) | 0.09 (−0.04 [−0.09, +0.00]) | 0.09 (−0.05 [−0.10, +0.00]) |

Information check (smart reader's score / original): obf 0.96, comp 0.99, compobf 0.89 (fails, below 0.9). Lengths (bytes): comp 0.35x, compobf 0.38x. Failed grades 0, unanswered 3. Data: [`takes-summary.json`](../data/runs/v2-2607.17117v2/takes-summary.json).

### 2607.10112v1 (11 questions)

| Reader | No paper | Original | Obfuscated (Δ) | Compressed (Δ) | Compressed + obfuscated (Δ) |
|---|---|---|---|---|---|
| GPT-5.6-sol (smart) | 0.44 [0.27, 0.61] | 0.89 [0.77, 1.00] | 0.83 (−0.06 [−0.17, +0.02]) | 0.80 (−0.09 [−0.21, +0.00]) | 0.83 (−0.06 [−0.15, +0.02]) |
| Gemma-4-31B | 0.14 [0.05, 0.24] | 0.74 [0.52, 0.94] | 0.62 (−0.12 [−0.30, +0.09]) | 0.59 (−0.15 [−0.33, +0.03]) | 0.45 (**−0.29 [−0.52, −0.09]**) |
| Ministral-8B | 0.20 [0.08, 0.32] | 0.67 [0.48, 0.82] | 0.59 (−0.08 [−0.21, +0.06]) | 0.64 (−0.03 [−0.12, +0.06]) | 0.58 (−0.09 [−0.30, +0.08]) |
| Llama-3.2-3B | 0.11 [0.00, 0.23] | 0.23 [0.08, 0.39] | 0.15 (−0.08 [−0.21, +0.08]) | 0.23 (−0.00 [−0.06, +0.06]) | 0.20 (−0.03 [−0.17, +0.09]) |

Information check (smart reader's score / original): obf 0.93, comp 0.90 (fails, below 0.9), compobf 0.93. Lengths (bytes): comp 0.45x, compobf 0.48x. Failed grades 0, unanswered 0. Data: [`takes-summary.json`](../data/runs/v2-2607.10112v1/takes-summary.json).
