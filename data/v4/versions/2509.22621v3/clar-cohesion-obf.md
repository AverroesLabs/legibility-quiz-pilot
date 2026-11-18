# IA2: Alignment with ICL Activations

Improves Supervised Fine-Tuning

Aayush Mishra

Daniel Khashabi & Anqi Liu  
Department of Computer Science, Johns Hopkins University

♡ Equal advising. Correspondence: {amishr24,danielk,aliu.cs}@jhu.edu

###### Abstract

Across 12 popular benchmarks and two model families, sequential ICL Activation Alignment followed by supervised fine-tuning (IA2 → SFT; code P4Z) significantly improves output accuracy (acc; code U6R) and calibration. These improvements provide practically useful adaptation and a conceptual window into the adaptation’s internal mechanics. Here, in-context learning (ICL; code V2K) adapts a model through contextual demonstrations without updating weights, whereas supervised fine-tuning (SFT; code Q7M) adapts model weights through labeled examples. ICL Activation Alignment (IA2; code R8D) is self-distillation that aims to replicate ICL activation patterns in weight-trained models, thereby incentivizing ICL-like internal reasoning. The IA2 priming signal is unavailable through standard SFT alone (code H9C).

## 1 Introduction

Generic large language models (LLMs) serve specialized applications, including news-aggregator classification into predefined article categories. Parameter-efficient fine-tuning (PEFT) techniques (Hu et al., 2021; Liu et al., 2022) often represent adapted models as small parameter sets that can be efficiently loaded onto and removed from graphics processing unit (GPU) memory. However, generalization through supervised fine-tuning typically requires many labeled samples (Le Scao and Rush, 2021), and collecting those samples can be expensive. In contrast, in-context learning uses demonstrations and instructions without weight updates. Those demonstrations and instructions instead consume context capacity and increase per-query inference compute, displacing query and response tokens.

The research question is therefore whether information-rich ICL activations can improve SFT quality. The hypothesis is that ICL activations contain information about extracting generalizable contextual patterns. That information may be absent from few-shot weight adaptation, which is prone to shortcuts or target-response overfitting. Similar token-space behavior need not imply similar internal processing: divergent activations indicate distinct adaptation mechanisms, consistent with Shen et al. (2024).

Context-to-weight distillation using response-text signals (Snell et al., 2022; Chen et al., 2024b) can inherit the same output-training problems. Specifically, reproducing a context-conditioned model’s outputs does not ensure reproducing the context-conditioned model’s functioning.

**Figure 1:** Standard Q7M, or SFT, enforces output-space agreement between response and target—the third step of the proposed pipeline—and produces subpar performance and miscalibration under low data. In contrast, the proposed pipeline first collects rich V2K, or ICL, activations, then enforces functional alignment, and finally trains against target responses. The comparison charts aggregate performance across models and datasets; details: §5.

## 2 Background and Notation

For a task $\mathcal T$, let $\mathcal D_{\mathcal T}=\{(X_i,Y_i)\}_{i=1}^{N}$ contain $N$ input–output pairs, where $X_i$ is an input and $Y_i$ is the corresponding ground-truth output. Let $M_\Theta$ denote a language model with parameters $\Theta$. Labeled weight adaptation, or supervised fine-tuning (Q7M = SFT), changes $\Theta\rightarrow\Theta'$ so that $M_{\Theta'}(X_t)=\bar y_t$, where $X_t$ is a query and $\bar y_t$ is the first token of the generated response $\bar Y_t$. Repeated application of the model to the growing sequence extends that response.

For a ground-truth response $Y_i$ containing $G$ tokens, generate a response $\bar Y_i$ of $G$ tokens and minimize cross-entropy across all generated tokens. With $j$ indexing response tokens, the SFT loss is

\[
\mathcal L_{\mathrm{Q7M}}
=\sum_{i=1}^{N}\sum_{j=1}^{G}
\operatorname{cross\text{-}entropy}(\bar Y_{ij},Y_{ij}).
\tag{2}
\]

A decoder-only transformer language model (LM), $M_\Theta$, stacks self-attention (SA) and linear layers. The model processes a sequence of $R$ token vectors, $T=[t_i]_{i=1}^{R}$, using self-attention and linear projections at every token and layer. Each token vector has dimension $d$. The final-layer last-token projection passes through the LM head to predict the most likely next token.

Self-attention depends on other tokens in the sequence. Let $W_Q,W_K,W_V,W_O\in\Theta$ be the query, key, value, and output projection matrices, collectively denoted $W_{QKVO}$. Then

\[
\operatorname{SA}(T;W_{QKVO})
=Z=
\left[
z_i=\sigma\left(\frac{q_iK^T}{\sqrt d}\right)V\cdot W_O
\right]_{i=1}^{R},
\tag{1}
\]

where $\sigma$ is Softmax, $q_i=t_iW_Q$, $K=t_{:i}W_K$, and $V=t_{:i}W_V$. Here, $t_{:i}$ denotes the first $i$ tokens, the superscript $T$ denotes transpose, and $z_i$ is the self-attention output at position $i$. Both the output $Z$ and the input $T$ have shape $R\times d$. The behavioral changes studied in this paper arise from varying either the context $T$ or the weights $W_{QKVO}$.

Let $\circ$ denote sequence concatenation, and let $I=[X_1\circ Y_1\circ\ldots\circ X_n\circ Y_n]$ concatenate $n$ demonstrations. Inference-time adaptation, or in-context learning (V2K = ICL), processes those examples to understand the task and produces $M_\Theta(I\circ X_t)=\hat y_t$ without weight updates. Here, $\hat y_t$ is the first token of the ICL response $\hat Y_t$, which is often similar to the expected response $Y_t$. First illustrated in GPT-3 (Brown et al., 2020), this widely used adaptation of generic language models also encompasses generic instructions as demonstrations for instruction-following frontier models. However, the present setting restricts demonstrations to classic input–output pairs.

Hidden self-attention outputs are called **activations**. These activations are context-dependent footprints of processing before next-token production, so activations are useful for examining behavioral changes. A model with $L$ layers processing $R$ tokens supplies an activation tensor $A\in\mathbb R^{L\times R\times d}$. The comparison quantity, activation similarity (K4X = asim), is token-wise cosine similarity between two activation tensors and has shape $L\times R$. The studied sequences are $T=[I\circ X]$ with demonstrations and $T=X$ without demonstrations, processed under differing weights.

## 3 Are V2K and Q7M practically the same?

ICL responses are much better calibrated than SFT responses at similar, slightly better accuracy, agreeing with the few-shot generalization and calibration findings of Duan et al. (2024). The proposed explanation is that output-oriented training learns shortcuts that fail on new data. In contrast, complex ICL circuits (Elhage et al., 2021), which are required across varied tasks, extract generalizable demonstration patterns. Thus, the two methods achieve adaptation differently, and SFT is not a drop-in replacement for ICL.

Recent internal-gradient-descent accounts (Von Oswald et al., 2023; Akyürek et al., 2022) show self-attention producing outputs as though contextual demonstrations had updated parameters. Such functional equivalence predicts similar output-position activations for a base model with demonstrations and a trained model without demonstrations. However, the observed activations do not align across models. Initial and final layers align better than middle layers: individual-token processing is expected in the initial and final layers, whereas abstract processing of the entire demonstration set is expected in the middle layers.

**Figure 2: Layerwise similarity with V2K activations.** Standard SFT alone (H9C) has small activation similarity (K4X), indicating differing adaptation functionality. Sequential IA2 → SFT (P4Z) aligns much more closely with ICL and performs much better. The first-row H9C curves average output-token K4X across tokens, samples, and datasets.

For this investigation, collect ICL activations $A_{\mathrm{V2K}}$ using $T=[I\circ X]$ on 100 test samples from multiple datasets, including SST2 (code B7V) and AGNews (code W3F). Then train using those same demonstrations until convergence and collect SFT activations $A_{\mathrm{Q7M}}$ using $T=X$. The models span Qwen3 (Yang et al., 2025) and Llama-3.2 (Grattafiori et al., 2024), with size–depth correspondences $1\mathrm B\leftrightarrow16$ layers, $3\mathrm B\leftrightarrow28$ layers, and $4\mathrm B\leftrightarrow36$ layers, where $\mathrm B$ denotes billions of parameters. The output-position comparison is $\mathrm{K4X}(A_{\mathrm{V2K}},A_{\mathrm{Q7M}})$, the activation similarity between the ICL and SFT tensors. Experiment details: §B.

## 4 Inducing V2K-like behavior in Q7M Models

The proposed pipeline first primes the model through ICL Activation Alignment (R8D = IA2), then trains the primed model against target responses through SFT. Unlike output-similarity training, IA2 priming aligns layerwise processing.

Specifically, IA2 seeks adapted self-attention weights $\tilde W_{QKVO}=(\tilde W_Q,\tilde W_K,\tilde W_V,\tilde W_O)$ such that, for every newly generated token at index $-1$, the last-token position,

\[
\operatorname{SA}([I\circ X];W_{QKVO})
\approx
\operatorname{SA}(X;\tilde W_{QKVO})
\qquad \forall X\in\mathcal T.
\tag{3}
\]

Thus, the target is for the adapted model without demonstrations to reproduce the base model’s self-attention processing with demonstrations. Chen et al. (2024a) give a closed-form solution for linearized attention using a Softmax kernel approximation. In contrast, the target here is a practical general solution for real nonlinear transformers: place ICL qualities directly into adapted weights so that every query is processed as the base model processes the query at each layer.

For each training input $X_i$, use the remaining dataset samples to construct the demonstration sequence $I$. Generate the ICL response $\hat Y_i$ from $T_i=[I\circ X_i]$ and collect the output-position activation tensor $A^i_{\mathrm{V2K}}\in\mathbb R^{L\times G\times d}$ for the $G$ response tokens. Next, supply $T_i=[X_i\circ\hat Y_i]$ to obtain the unaligned output-position activation tensor $A^i$. This second pass treats the same response $\hat Y_i$ as generated with only the query in context. Minimize the mean squared error with respect to model parameters until convergence:

\[
\mathcal L_{\mathrm{R8D}}
=\sum_{i=1}^{N}\left\|A^i-A^i_{\mathrm{V2K}}\right\|.
\tag{4}
\]

After this priming changes $\Theta$ to $\Theta'$, collect $\bar Y_i$ for every $X_i$ using $M_{\Theta'}$. Then train against the ground truth $Y_i$ with the SFT loss $\mathcal L_{\mathrm{Q7M}}$ until convergence. This second training stage aligns intended target-token behavior, following the first stage’s alignment of internal processing.

## 5 Experimental Results

Each $(\text{method},\text{dataset})$ combination trains to convergence, up to 50 epochs, at three learning rates: slow:1e-4, medium:3e-4, and fast:1e-3. The best performer is selected to neutralize hyperparameter-selection effects. Over 13,000 models were trained.

Resource-based choices fix rank-8 low-rank adaptation (LoRA) modules targeting the self-attention matrices $W_Q,W_K,W_O$. Multi-token training is substantially more resource-extensive with long answers, so the experiments restrict the number of training examples $N$ to 16 for Llama and 8 for Qwen. For activation collection, training caps generation at $G=200$, yielding a collected tensor $\hat A_i\in\mathbb R^{L\times200\times d}\ \forall x_i$, where $x_i$ denotes a training input. Target-token training instead uses the ground-truth response length for $G$. Further training details: §B.

Single-token outputs consist of one token, typically a category/class, True/False, or a multiple-choice question (MCQ) choice. In contrast, multi-token outputs have variable, potentially multiple-token length and represent the general open-generation case. Both settings compare adaptation across multiple models and tasks, frequently including distribution-shift evaluation.

Single-token evaluation uses the first generated token’s probability distribution for both the correctness measure, accuracy (U6R = acc), and the calibration measure, expected calibration error (D8P = ECE; Guo et al., 2017). For each task validation set, base-model evaluation with demonstrations is repeated five times using different demonstrations, and average metrics are reported. Trained methods are then evaluated without demonstrations, and best-performing averages are reported.

Multi-token evaluation measures only questions answered correctly in the required format; parsing and other details are in §A. Because open-ended answer confidence is difficult to measure, calibration is not measured for multi-token evaluation. Generation limits judiciously conserve compute. The limit is 200 tokens for GSM8K (code L9S) and GSM Symbolic (GSM8Ks; code X4C), whose 95-percentile answer lengths are 184 and 154, respectively. The limit is 400 tokens for MATH Algebra (HMathA; code V7B), whose corresponding answer length is 307.

Qwen3-4B-Base is used on every single- and multi-token task. Llama-3.2-1B additionally covers every single-token task except SciQr (code E2T), the remapped-choice task defined below. SciQr instead uses Llama-3.2-3B because the 1B model could not improve above random baseline for any $N$ up to 128. Multi-token experiments instead use Llama-3.2-1B-Instruct as the secondary model, thereby testing efficacy on post-trained models.

The tasks, dataset names, and dataset codes are as follows:

- **Sentiment** uses SST2 (B7V) (Socher et al., 2013), Financial Phrasebank (Z2H = FinS) (Malo et al., 2014), and Poem Sentiment (C9L = PoemS) (Sheng and Uthus, 2020). Training occurs separately on SST2 and Financial Phrasebank, and each trained model is evaluated on all three datasets. Each training source therefore produces one in-distribution (ID) and two out-of-distribution (OOD) evaluation datasets.
- **True/False** uses StrategyQA (N5Q = STF) (Geva et al., 2021), which requires multi-hop reasoning before its binary conclusion.
- **News categorization** trains on AGNews (W3F = AGN) (Zhang et al., 2015) and evaluates on AGNews and BBCNews (Y8J = BBCN) (Greene and Cunningham, 2006).
- **The MCQ setting** remaps choices $[A,B,C,\ldots]$ into unrelated tokens for added complexity. This setting uses SciQr (E2T), derived from SciQ (code O3D), for training and evaluation, and QASCr (G6A), derived from QASC (Khot et al., 2019), for evaluation.
- **Grade-school mathematics** trains on GSM8K (L9S) (Cobbe et al., 2021) and evaluates on GSM8K and GSM Symbolic (X4C = GSM8Ks) (Mirzadeh et al., 2024).
- **Advanced mathematics** uses MATH Algebra (V7B = HMathA) (Hendrycks et al., 2021), which contains higher-grade algebra problems.
- **Scientific question answering (QA)** uses SciQ (O3D) (Johannes Welbl, 2017), generating answer tokens rather than selecting options. SciQ is also the source dataset for SciQr.

Few-shot training datasets $\mathcal D_{\mathcal T}=\{(X_i,Y_i)\}_{i=1}^{N}$ vary $N$ by powers of two, $[2,4,8,16,\ldots]$. Five distinct sets are used per $N$ to average outliers. Each target activation tensor $A^i_{\mathrm{V2K}}\ \forall X_i$ uses the other $N-1$ samples as randomly ordered demonstrations. Exactly the same training data are reused across adaptation methods for fair comparison. Evaluation samples a separate set of 500 examples per task. Dataset and processing details: §A.

Multi-token demonstrations and expected outputs are chain-of-thought style (Wei et al., 2022): ordinary-text reasoning, not inside `<think>` tags, precedes an answer in the demonstrated format. Parsing is dataset-specific:

- GSM8K and GSM Symbolic ground truths follow “$<$reasoning steps$>$ #### $<$numerical answer$>$”. Reasoning correctness and conciseness are unevaluated; the parsed number must exactly match ground truth. Common alternatives, including “The answer is $<$value$>$”, permit slight formatting variation. The exact parser is in the code.
- MATH Algebra answers are expected and parsed inside `\boxed{}`.
- SciQ ground truths are constructed as “$<$support text$>$ #### $<$text answer$>$”. Parsing uses that format, and only exact string matches count as correct.

### 5.1 R8D priming boosts Q7M performance

The three training conditions are standard SFT alone (H9C), IA2 priming without target-response-token training (J3N = IA2 only), and sequential IA2 priming followed by SFT (P4Z = IA2 → SFT).

With identical data, sequential IA2 → SFT outperforms SFT alone in almost all single-token cases for accuracy and calibration, as reported in Table 1. In most cases, sequential IA2 → SFT also exceeds ICL accuracy, with slightly worse calibration. IA2 alone often reaches high accuracy and reasonable, ICL-like calibration without target-response-token training; the IA2-only columns in Table 1 therefore provide evidence of a rich training signal. Other model/$N$ reports and a statistical-significance test of the improvement are provided in §C.

In Table 1, Source and Eval identify the training and evaluation datasets using the codes defined in §5. V2K denotes ICL, H9C denotes SFT only, J3N denotes IA2 only, and P4Z denotes IA2 → SFT. U6R is accuracy, for which higher is better (↑); D8P is expected calibration error, for which lower is better (↓).

| Source | Eval | V2K U6R ↑ | V2K D8P ↓ | H9C U6R ↑ | H9C D8P ↓ | J3N U6R ↑ | J3N D8P ↓ | P4Z U6R ↑ | P4Z D8P ↓ |
|---|---|---|---|---|---|---|---|---|---|
| W3F | W3F | 30.0 (02.3) | 0.13 (0.03) | 27.9 (04.3) | 0.52 (0.12) | 24.0 (00.3) | 0.11 (0.01) | 31.8 (03.8) | 0.35 (0.24) |
| | Y8J∗ | 28.8 (00.6) | 0.12 (0.01) | 31.8 (07.1) | 0.56 (0.14) | 24.3 (01.3) | 0.10 (0.02) | 27.9 (04.3) | 0.40 (0.21) |
| Z2H | Z2H | 63.6 (01.5) | 0.12 (0.01) | 67.4 (14.1) | 0.31 (0.14) | 63.1 (15.0) | 0.24 (0.06) | 78.7 (13.7) | 0.16 (0.11) |
| | C9L∗ | 56.9 (02.2) | 0.12 (0.02) | 49.9 (04.6) | 0.48 (0.04) | 52.8 (05.6) | 0.15 (0.04) | 60.6 (15.7) | 0.36 (0.15) |
| | B7V∗ | 70.1 (01.3) | 0.17 (0.01) | 59.4 (06.5) | 0.20 (0.10) | 69.8 (13.7) | 0.30 (0.10) | 71.4 (18.9) | 0.20 (0.15) |
| E2T | G6A∗ | 56.5 (01.4) | 0.08 (0.01) | 76.5 (03.8) | 0.09 (0.05) | 71.2 (03.0) | 0.12 (0.01) | 79.4 (01.8) | 0.09 (0.01) |
| | E2T | 87.7 (01.1) | 0.07 (0.01) | 88.3 (04.6) | 0.07 (0.05) | 90.4 (01.5) | 0.09 (0.01) | 91.7 (01.7) | 0.05 (0.01) |
| B7V | Z2H∗ | 41.9 (00.3) | 0.19 (0.01) | 68.4 (02.3) | 0.30 (0.04) | 71.3 (02.1) | 0.21 (0.03) | 82.4 (12.2) | 0.11 (0.08) |
| | C9L∗ | 65.1 (01.2) | 0.11 (0.02) | 56.5 (13.2) | 0.33 (0.14) | 62.4 (09.2) | 0.19 (0.06) | 68.4 (08.4) | 0.30 (0.08) |
| | B7V | 85.4 (00.4) | 0.13 (0.00) | 65.2 (10.7) | 0.22 (0.10) | 82.7 (06.4) | 0.28 (0.03) | 90.4 (02.3) | 0.06 (0.01) |
| N5Q | N5Q | 66.0 (02.1) | 0.07 (0.01) | 52.4 (04.4) | 0.29 (0.10) | 62.2 (04.1) | 0.16 (0.04) | 62.4 (04.3) | 0.29 (0.08) |

**Table 1:** Qwen3-4B-Base, $N=4$, single-token dataset/adaptation-method performance. Parentheses give standard deviations over five runs at the best-performing learning rate; bold denotes best training method, ∗ denotes OOD evaluation. P4Z, or IA2 → SFT, outperforms standard Q7M, or SFT, across the board.

For multi-token tasks, sequential IA2 → SFT exceeds SFT alone in every case in Table 2: 73.6 versus 70.9 on GSM8K, 68.8 versus 64.5 on GSM Symbolic, 55.3 versus 50.4 on MATH Algebra, and 40.8 versus 35.0 on SciQ. However, sequential IA2 → SFT sometimes underperforms ICL, contrasting with the single-token results. The proposed explanation is imperfect compression of longer contexts into the same small LoRA weight space; this compression is expected to improve with larger ranks. Additional model/$N$ reports are provided in §C.

Table 2 uses the same method and accuracy codes as Table 1. “Without V2K” means without ICL demonstrations.

| Source | Eval | Without V2K U6R ↑ | V2K U6R ↑ | H9C U6R ↑ | J3N U6R ↑ | P4Z U6R ↑ |
|---|---|---|---|---|---|---|
| L9S | L9S | 56.4 (00.0) | 76.4 (01.1) | 70.9 (02.8) | 77.4 (02.8) | 73.6 (03.7) |
| | X4C∗ | 45.4 (00.0) | 68.4 (01.1) | 64.5 (04.0) | 66.2 (03.7) | 68.8 (05.1) |
| V7B | V7B | 21.0 (00.0) | 60.4 (02.1) | 50.4 (01.8) | 47.8 (06.0) | 55.3 (02.1) |
| O3D | O3D | 15.4 (00.0) | 37.5 (01.0) | 35.0 (07.1) | 06.9 (13.8) | 40.8 (07.6) |

**Table 2:** Qwen3-4B-Base, $N=4$, multi-token dataset/adaptation-method performance; bold denotes best training method.

### 5.2 Why is R8D priming important?

Unchanged base weights significantly influence activations under LoRA. Therefore, activation patterns alone cannot isolate training-induced behavioral changes. To isolate those changes, direct weight analysis uses all single- and multi-token experiments and takes singular value decomposition (SVD) basis vectors of weight matrices.

The comparison quantity, Subspace Overlap (code A5Y), measures shared weight space, thereby isolating and comparing functional-behavior differences through

\[
\frac{1}{r}*\left\|U_1^T\cdot U_2\right\|_F^2,
\]

where $U_1,U_2$ are the compared methods’ basis vectors, $r$ is the LoRA rank, $T$ denotes transpose, and $\|\cdot\|_F$ denotes the Frobenius norm.

This weight analysis finds that SFT-only weight updates are almost completely orthogonal to the updates from the other methods. In contrast, the most performant sequential IA2 → SFT models substantially share the weight space spanned by IA2-only models: the mean shared spanned weight space is around $39\%$, as reported below. Much of the performance gain is attributable to IA2, whose identified subspaces are practically unreachable through SFT alone.

The wide IA2-only–IA2 → SFT overlap distribution arises when priming already produces output alignment. In those cases, subsequent target training changes little before convergence, leaving very high overlap.

**Figure 4:** Sequential IA2 → SFT (P4Z) has significantly high Subspace Overlap (A5Y) with IA2 only (J3N), while SFT only (H9C) aligns with neither. This pattern indicates an IA2 (R8D) signal absent from SFT alone and important for performance.

| Model pair | Mean shared spanned weight space |
|---|---|
| Most performant P4Z and J3N | around $39\%$ |

The three training conditions—baseline SFT only (H9C), priming without target training (J3N = IA2 only), and sequential adaptation (P4Z = IA2 → SFT)—all appear in Figure 2’s activation-similarity comparison with ICL. Priming drastically increases activation similarity, and sequential training retains elevated activation similarity. SFT only and IA2 → SFT differ only in starting state: the latter starts target training from the primed model.

Sequential IA2 → SFT occupies intermediate activation similarity between SFT only and IA2 only while achieving better accuracy and calibration. Figure 3 shows that increasing activation similarity improves calibration. However, extreme activation similarity need not optimize accuracy, because the ICL signal is not always correct. In one case in §C, IA2 only follows ICL even as overfitting worsens performance. Thus, functional alignment and ground-truth output alignment are best combined.

**Figure 3:** Scatterplots of $\mathrm{K4X}(A,A_{\mathrm{V2K}})$—activation similarity between a trained model’s activations $A$ and ICL activations $A_{\mathrm{V2K}}$—against accuracy (U6R) and expected calibration error (D8P) place one training experiment at each point. Expected calibration error decreases smoothly with increasing similarity; however, extreme alignment can leave accuracy gains obtainable from ground-truth training.

## 6 Discussion

A simultaneous alternative to sequential training is IA2 + SFT (code T6B), which applies the activation-alignment loss $\mathcal L_{\mathrm{R8D}}$ and the target-token loss $\mathcal L_{\mathrm{Q7M}}$ together. However, whenever the response length $G>1$, the target activation tensor $\hat A$ uses the model’s own generated response tokens. That ICL response can differ from ground truth even in length, so the ground-truth objectives are incompatible for joint training.

Instead, self-distillation on ICL responses, analogous to Snell et al. (2022) but with an additional activation signal, uses

\[
\mathcal L_{\mathrm{R8D}}+\beta\cdot\mathcal L_{\mathrm{Q7M}},
\]

where $\beta$ weights the target-token loss. The experiments vary $\beta$ over four values smoothly spanning IA2 only to SFT only; details are in §B. Multi-token reports select the best $(\text{learning rate},\beta)$ combination across five random seeds. Further simultaneous IA2 + SFT results are provided in §C.

Simultaneous IA2 + SFT extracts significantly more performance from ICL responses than SFT alone, as shown in Table 3. However, SFT alone and sequential IA2 → SFT trained on ICL responses perform worse than the corresponding methods trained on ground truths. Specifically, compare Table 3 with Table 2: SFT-only accuracy falls from 70.9 to 66.4 on GSM8K, from 64.5 to 59.8 on GSM Symbolic, from 50.4 to 49.0 on MATH Algebra, and from 35.0 to 27.2 on SciQ. Sequential IA2 → SFT accuracy falls from 73.6 to 67.2, from 68.8 to 61.6, from 55.3 to 54.0, and from 40.8 to 32.8 on those same evaluation datasets.

In Table 3, V2K denotes ICL, H9C denotes SFT only, J3N denotes IA2 only, P4Z denotes sequential IA2 → SFT, and T6B denotes simultaneous IA2 + SFT. U6R denotes accuracy.

| Source | Eval | Without V2K U6R ↑ | V2K U6R ↑ | H9C U6R ↑ | J3N U6R ↑ | P4Z U6R ↑ | T6B U6R ↑ |
|---|---|---|---|---|---|---|---|---|
| L9S | L9S | 56.4 (00.0) | 76.4 (01.1) | 66.4 (18.1) | 77.4 (02.8) | 67.2 (22.1) | 77.0 (07.8) |
| | X4C∗ | 45.4 (00.0) | 68.4 (01.1) | 59.8 (20.2) | 66.2 (03.7) | 61.6 (24.0) | 69.0 (09.6) |
| V7B | V7B | 21.0 (00.0) | 60.4 (02.1) | 49.0 (03.0) | 47.8 (06.0) | 54.0 (01.7) | 53.6 (02.7) |
| O3D | O3D | 15.4 (00.0) | 37.5 (01.0) | 27.2 (12.0) | 06.9 (13.8) | 32.8 (10.0) | 34.5 (10.6) |

**Table 3:** Qwen3-4B-Base, $N=4$, multi-token dataset/adaptation-method performance when training on V2K, or ICL, responses rather than ground truths. T6B’s significant improvement over H9C—simultaneous IA2 + SFT over SFT only—highlights R8D’s, or IA2’s, significance.

**Computational overhead:** Sequential IA2 → SFT adds one pretraining activation-collection pass: ICL inference on the given dataset. This one-time cost greatly benefits subsequent capabilities. Inference itself, however, costs the same as SFT alone.

A popular distillation approach, Knowledge Distillation (M8G = KD), transfers a stronger teacher’s behavior into a student through soft-label matching. The denser teacher signal consists of each token’s probability mass (Hinton et al., 2015). Here, that signal is supplied by the same model enhanced with contextual demonstrations.

The comparator, SFT (soft labels) (code F5W), trains and evaluates Llama-3.2-1B-Instruct on GSM8K and Llama-3.2-1B on SST2 using ICL responses. As shown in Table 4’s GSM8K row, soft-label SFT significantly improves over standard SFT in the multi-token case, with accuracy 36.1 versus 28.2, and nearly reaches simultaneous IA2 + SFT at 37.0. However, soft-label SFT is lacking in the single-token case reported in Table 4’s SST2 row.

Table 4 retains the method codes defined above and adds F5W for SFT with soft labels. L9S denotes GSM8K, B7V denotes SST2, U6R denotes accuracy, and D8P denotes expected calibration error. Dashes indicate that multi-token calibration is not measured.

| Dataset | $N$ | V2K U6R ↑ | V2K D8P ↓ | H9C U6R ↑ | H9C D8P ↓ | F5W U6R ↑ | F5W D8P ↓ | J3N U6R ↑ | J3N D8P ↓ | P4Z U6R ↑ | P4Z D8P ↓ | T6B U6R ↑ | T6B D8P ↓ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| L9S | 4 | 34.6 (00.9) | – | 28.2 (04.5) | – | 36.1 (03.5) | – | 29.3 (05.0) | – | 30.3 (03.2) | – | 37.0 (01.9) | – |
| B7V | 8 | 77.0 (02.2) | 0.10 (0.02) | 52.3 (04.3) | 0.38 (0.14) | 55.9 (08.8) | 0.16 (0.06) | 64.9 (15.3) | 0.30 (0.05) | 64.7 (19.3) | 0.33 (0.19) | 66.9 (18.4) | 0.28 (0.21) |

**Table 4:** Comparison with M8G, or Knowledge Distillation, using target-token training on soft labels. The baseline improves on standard Q7M, or SFT, in the multi-token case, while R8D-based, or IA2-based, training provides consistently high single-/multi-token performance.

**Failure to consistently exceed V2K:** On some multi-token Qwen mathematics cases, including GSM8K and MATH Algebra, ICL exceeds every training method. The proposed explanation is Qwen’s science, technology, engineering, and mathematics (STEM) data mid-training, which makes contextual adaptation extremely sample efficient. Surprisingly, GSM8K ICL at $N=2$ reaches 81.2, exceeding the $N=4,8$ accuracies of 76.4 and 77.2 reported in Table 5.

Sequential IA2 → SFT’s gap to ICL closes quickly with more shots, while superiority to SFT alone remains consistent. Table 5 reports the GSM8K gap to ICL changing from -06.6 at $N=2$ to -02.8 at $N=4$ and -01.2 at $N=8$; the MATH Algebra gap changes from -09.9 to -05.1 and -02.7. All corresponding gaps relative to SFT alone remain positive.

In Table 5, “Gap wrt V2K” is the sequential IA2 → SFT accuracy gap with respect to ICL, and “Gap wrt H9C” is the gap with respect to SFT only. L9S denotes GSM8K and V7B denotes MATH Algebra.

| Dataset | $N$ | V2K | H9C | J3N | P4Z | Gap wrt V2K | Gap wrt H9C |
|---|---|---|---|---|---|---|---|
| L9S | 2 | 81.2 (00.4) | 65.2 (01.0) | 70.7 (01.8) | 74.6 (02.4) | -06.6 | +09.4 |
| | 4 | 76.4 (01.1) | 70.9 (02.8) | 77.4 (02.8) | 73.6 (03.7) | -02.8 | +02.7 |
| | 8 | 77.2 (00.9) | 73.5 (01.6) | 78.8 (02.5) | 76.0 (02.0) | -01.2 | +02.5 |
| V7B | 2 | 58.6 (02.0) | 47.8 (04.0) | 33.9 (06.4) | 48.7 (07.0) | -09.9 | +00.9 |
| | 4 | 60.4 (02.1) | 50.4 (01.8) | 47.8 (06.0) | 55.3 (02.1) | -05.1 | +04.9 |
| | 8 | 62.1 (01.0) | 52.0 (00.8) | 57.3 (03.7) | 59.4 (00.8) | -02.7 | +07.4 |

**Table 5:** Shot-count evolution for Qwen mathematics datasets, where V2K, or ICL, is more sample efficient than training methods.

**Other fine-tuning methods:** Full-rank tuning, prompt/prefix tuning (Lester et al., 2021; Li and Liang, 2021), and activation scaling via the fine-tuning method (IA)3 (Liu et al., 2022) are alternatives to LoRA. The proposed pipeline improves training signals rather than competing with these fine-tuning methods.

Experiments on a few datasets replace LoRA with the popular PEFT method (IA)3 while retaining the exact §5 methodology; setup details are in §B. The resulting trends resemble the LoRA trends, as reported in Table 6, and additional reports are provided in §C. Exploration with other SFT methods is deferred. IA2 also marks a major method-naming improvement over (IA)3.

In Table 6, E2T denotes SciQr, G6A denotes QASCr, B7V denotes SST2, Z2H denotes Financial Phrasebank, and C9L denotes Poem Sentiment. V2K denotes ICL, H9C denotes SFT only, J3N denotes IA2 only, and P4Z denotes sequential IA2 → SFT. U6R denotes accuracy, and D8P denotes expected calibration error.

| Source | Eval | V2K U6R ↑ | V2K D8P ↓ | H9C U6R ↑ | H9C D8P ↓ | J3N U6R ↑ | J3N D8P ↓ | P4Z U6R ↑ | P4Z D8P ↓ |
|---|---|---|---|---|---|---|---|---|---|
| E2T | G6A∗ | 35.6 (01.4) | 0.06 (0.01) | 32.8 (03.4) | 0.20 (0.08) | 34.2 (02.7) | 0.08 (0.01) | 48.6 (05.5) | 0.25 (0.11) |
| | E2T | 60.0 (01.8) | 0.07 (0.01) | 64.5 (05.0) | 0.12 (0.05) | 63.0 (04.6) | 0.10 (0.04) | 70.6 (04.6) | 0.08 (0.02) |
| B7V | Z2H∗ | 56.6 (01.6) | 0.17 (0.01) | 58.2 (10.7) | 0.25 (0.03) | 61.9 (09.5) | 0.18 (0.06) | 59.8 (07.8) | 0.18 (0.03) |
| | C9L∗ | 52.3 (03.0) | 0.19 (0.02) | 48.1 (03.2) | 0.44 (0.11) | 48.1 (03.4) | 0.11 (0.04) | 59.8 (13.4) | 0.33 (0.13) |
| | B7V | 60.3 (02.8) | 0.19 (0.02) | 52.4 (02.8) | 0.35 (0.05) | 54.2 (04.2) | 0.15 (0.06) | 62.6 (09.1) | 0.28 (0.11) |

**Table 6:** Llama-3.2 models, $N=4$, trained with (IA)3. P4Z, or sequential IA2 → SFT, significantly exceeds H9C, or SFT only, exceeding even V2K, or ICL, accuracy in all cases.

**Potential improvements:** Resource constraints fixed design choices. Deferred explorations include LoRA rank and target-module ablations, selective IA2, prompt optimization, and finer-grained control of the two training signals rather than fixed learning rates. Extreme alignment can hurt performance, and layer sensitivity to contextual demonstrations varies (Todd et al., 2023). Therefore, selective layer alignment may be more effective. Demonstration order also affects ICL performance (Lu et al., 2022; Zhao et al., 2021), permitting prompt optimization before target-activation collection.

## 7 Related Work

**V2K versus Q7M:** Comparisons of ICL and SFT performance have an unclear verdict (Mosbach et al., 2023; Duan et al., 2024); in contrast, the present experiments find SFT worse. Prior work has studied functional similarity (Dai et al., 2022), functional differences (Wang et al., 2023), and miscalibration and remedies (Zhang et al., 2024; Li et al., 2025; Li et al., 2024; Xiao et al., 2025). Sia et al. (2024) locate “where” translation happens during contextual adaptation by using context masking and activations as functional signals. Recent work also identifies SFT memorization/overfitting and catastrophic forgetting (Chu et al., 2025; Shenfeld et al., 2025), which are problems the improved pipeline aims to address.

**Context distillation:** Explicit-context compression for adaptation includes intermediate contemplation tokens (Cheng and Van Durme, 2024; Jiang et al., 2025), on-the-fly adapter generation to internalize context (Chen et al., 2025; Charakorn et al., 2025), related-text-enhanced distillation (Zhu et al., 2025; Choi et al., 2025), and direct context-to-weight compression (Shen et al., 2025; Deng et al., 2024; Yu et al., 2024; Snell et al., 2022; Chen et al., 2024b; Shin et al., 2024). These approaches distill contextual effects on outputs through token-response signals. In contrast, the present complementary approach improves distillation using the model’s own contextual processing signals, supported by Aguilar et al. (2020), Jin et al. (2024), and Yang et al. (2024).

**Activation steering:** Extracting steering vectors or weights for targeted internal locations seeks intended output effects (Postmus and Abreu, 2024; Stolfo et al., 2025; Caccia et al., 2025; Fleshman and Van Durme, 2024; Arditi et al., 2024). Like IA2, these interventions influence functionality in activation space. However, unlike the targeted, narrow-domain approach of activation steering, IA2 seeks large behavioral change while avoiding potential brittleness and reliability problems (Queiroz Da Silva et al., 2025).

## 8 Conclusion and Future Work

Activation probing distinguishes adaptation mechanisms: ICL often encodes richer, more generalizable patterns, and IA2 shifts models into a more adaptable weight subspace inaccessible to SFT alone. Future work tests the diversity and catastrophic-forgetting effects of IA2-based adaptation to assess post-training benefits. Future work also tests IA2’s effects on post-trained models tuned through reinforcement learning (RL).

## Reproducibility Statement

GitHub code covers data download and processing, training, evaluation, analysis, plotting, and other operations. The paper and appendix details, a detailed README, and repeatable random seeds should suffice for full result reproduction.

## Acknowledgments

Supported by ONR (N00014-24-1-2089), an Amazon Research Award, and an Open Philanthropy grant. Andrea Wynn provided constructive feedback on earlier document versions.
