# IA2: Alignment with ICL Activations

Improves Supervised Fine-Tuning

Aayush Mishra

Daniel Khashabi & Anqi Liu  
Department of Computer Science, Johns Hopkins University

♡ Equal advising. Correspondence: {amishr24,danielk,aliu.cs}@jhu.edu

###### Abstract

Across 12 popular benchmarks and two model families, P4Z significantly improves output U6R and calibration, providing practically useful adaptation and a conceptual window into its internal mechanics. R8D is self-distillation aiming to replicate V2K activation patterns in weight-trained models, incentivizing V2K-like internal reasoning; its priming signal is unavailable through H9C.

## 1 Introduction

Generic LLMs serve specialized applications, including news-aggregator classification into predefined article categories; parameter-efficient fine-tuning (PEFT) techniques (Hu et al., 2021; Liu et al., 2022) often make adapted models small parameter sets efficiently loadable onto and removable from GPU memory. Generalization through Q7M typically requires many labeled samples (Le Scao and Rush, 2021), whose collection can be expensive. V2K demonstrations and instructions instead consume context capacity and increase per-query inference compute, displacing query and response tokens.

The research question is whether information-rich V2K activations can improve Q7M quality. The hypothesis assigns those activations information about extracting generalizable contextual patterns, potentially absent from few-shot weight adaptation prone to shortcuts or target-response overfitting. Similar token-space behavior need not imply similar internal processing; divergent activations indicate distinct adaptation mechanisms, consistent with Shen et al. (2024).

Context-to-weight distillation using response-text signals (Snell et al., 2022; Chen et al., 2024b) can inherit the same output-training problems: reproducing a context-conditioned model’s outputs does not ensure reproducing its functioning.

**Figure 1:** Standard Q7M enforces output-space agreement between response and target—the third step of the proposed pipeline—and produces subpar performance and miscalibration under low data. The pipeline collects rich V2K activations, enforces functional alignment, then trains against target responses; its comparison charts aggregate performance across models and datasets; details: §5.

## 2 Background and Notation

For task $\mathcal T$, let $\mathcal D_{\mathcal T}=\{(X_i,Y_i)\}_{i=1}^{N}$ contain $N$ input–output pairs; labeled weight adaptation (Q7M = Supervised Fine-Tuning (SFT)) changes $\Theta\rightarrow\Theta'$ so that $M_{\Theta'}(X_t)=\bar y_t$, the first token of response $\bar Y_t$, with repeated application to the growing sequence extending that response. For ground truth $Y_i$ containing $G$ tokens, generate $G$ tokens $\bar Y_i$ and minimize cross-entropy across all generated tokens:

\[
\mathcal L_{\mathrm{Q7M}}
=\sum_{i=1}^{N}\sum_{j=1}^{G}
\operatorname{cross\text{-}entropy}(\bar Y_{ij},Y_{ij}).
\tag{2}
\]

A decoder-only transformer LM $M_\Theta$ stacks self-attention (SA) and linear layers, processing token vectors $T=[t_i]_{i=1}^{R}$ by SA and linear projections at every token and layer; the final-layer last-token projection passes through the LM head to predict the most likely next token. SA depends on other sequence tokens. With $W_Q,W_K,W_V,W_O\in\Theta$, collectively $W_{QKVO}$,

\[
\operatorname{SA}(T;W_{QKVO})
=Z=
\left[
z_i=\sigma\left(\frac{q_iK^T}{\sqrt d}\right)V\cdot W_O
\right]_{i=1}^{R},
\tag{1}
\]

where $\sigma$ is Softmax, $q_i=t_iW_Q$, $K=t_{:i}W_K$, $V=t_{:i}W_V$, and $t_{:i}$ denotes the first $i$ tokens; both $Z$ and $T$ have shape $R\times d$. The studied behavioral changes arise from varying context $T$ or weights $W_{QKVO}$.

With concatenated demonstrations $I=[X_1\circ Y_1\circ\ldots\circ X_n\circ Y_n]$, inference-time adaptation (V2K = In-Context Learning (ICL)) processes examples to understand the task and produces $M_\Theta(I\circ X_t)=\hat y_t$ without weight updates, often yielding $\hat Y_t$ similar to expected $Y_t$. First illustrated in GPT-3 (Brown et al., 2020), this widely used adaptation of generic LMs also encompasses generic instructions as demonstrations for instruction-following frontier models; the present setting restricts demonstrations to classic input–output pairs.

Hidden SA outputs are activations: context-dependent footprints of processing before next-token production, useful for examining behavioral changes. A model with $L$ layers processing $R$ tokens supplies $A\in\mathbb R^{L\times R\times d}$; the comparison quantity (K4X = activation similarity (asim)) is token-wise cosine similarity between two activation tensors and has shape $L\times R$, while the studied sequences are $T=[I\circ X]$ with demonstrations and $T=X$ without them under differing weights.

## 3 Are V2K and Q7M practically the same?

V2K responses are much better calibrated than Q7M responses at similar, slightly better U6R, agreeing with few-shot generalization and calibration findings of Duan et al. (2024). The proposed explanation assigns shortcuts failing on new data to output-oriented training, while complex V2K circuits (Elhage et al., 2021), required across varied tasks, extract generalizable demonstration patterns. The methods achieve adaptation differently; Q7M is not a drop-in V2K replacement.

Recent internal-gradient-descent accounts (Von Oswald et al., 2023; Akyürek et al., 2022) show SA producing outputs as though contextual demonstrations had updated parameters. Such functional equivalence predicts similar output-position activations for a base model with demonstrations and a trained model without them. The observed activations do not align across models; initial and final layers align better, where individual-token processing is expected, than middle layers, where abstract processing of the entire demonstration set is expected.

**Figure 2: Layerwise similarity with V2K activations.** H9C has small K4X, indicating differing adaptation functionality; P4Z aligns much more closely and performs much better. The first-row H9C curves average output-token K4X across tokens, samples, and datasets.

For this investigation, collect $A_{\mathrm{V2K}}$ using $T=[I\circ X]$ on 100 test samples from multiple datasets, including B7V and W3F, then train using those same demonstrations until convergence and collect $A_{\mathrm{Q7M}}$ using $T=X$. Models span Qwen3 (Yang et al., 2025) and Llama-3.2 (Grattafiori et al., 2024), with size–depth correspondences $1\mathrm B\leftrightarrow16$ layers, $3\mathrm B\leftrightarrow28$ layers, and $4\mathrm B\leftrightarrow36$ layers. The output-position comparison is $\mathrm{K4X}(A_{\mathrm{V2K}},A_{\mathrm{Q7M}})$; experiment details: §B.

## 4 Inducing V2K-like behavior in Q7M Models

After priming changes $\Theta$ to $\Theta'$, collect $\bar Y_i$ for every $X_i$ using $M_{\Theta'}$ and train against ground truth $Y_i$ with $\mathcal L_{\mathrm{Q7M}}$ until convergence, aligning intended target-token behavior.

The preceding priming operation (R8D = ICL Activation Alignment (IA2)) instead aligns layerwise processing, not output similarity, by seeking $\tilde W_{QKVO}=(\tilde W_Q,\tilde W_K,\tilde W_V,\tilde W_O)$ satisfying, for every newly generated token at index $-1$,

\[
\operatorname{SA}([I\circ X];W_{QKVO})
\approx
\operatorname{SA}(X;\tilde W_{QKVO})
\qquad \forall X\in\mathcal T.
\tag{3}
\]

Chen et al. (2024a) give a closed-form solution for linearized attention using a Softmax kernel approximation; the target here is a practical general solution for real nonlinear transformers, placing V2K qualities directly into adapted weights and processing every query as the base model does at each layer.

For each training $X_i$, use the remaining dataset samples to construct $I$, generate $\hat Y_i$ from $T_i=[I\circ X_i]$, and collect output-position $A^i_{\mathrm{V2K}}\in\mathbb R^{L\times G\times d}$ for $G$ response tokens. Supply $T_i=[X_i\circ\hat Y_i]$, treating the response as generated with only the query in context, to obtain unaligned output-position $A^i$. Minimize the mean squared error with respect to model parameters until convergence:

\[
\mathcal L_{\mathrm{R8D}}
=\sum_{i=1}^{N}\left\|A^i-A^i_{\mathrm{V2K}}\right\|.
\tag{4}
\]

## 5 Experimental Results

Each $(\text{method},\text{dataset})$ combination trains to convergence, up to 50 epochs, at three learning rates—slow:1e-4, medium:3e-4, fast:1e-3—with the best performer selected to neutralize hyperparameter-selection effects. Over 13,000 models were trained. Resource-based choices fix rank-8 LoRAs targeting SA matrices $W_Q,W_K,W_O$; multi-token training, substantially more resource-extensive with long answers, restricts $N$ to 16 for Llama and 8 for Qwen. For activation collection, training caps generation at $G=200$, yielding $\hat A_i\in\mathbb R^{L\times200\times d}\ \forall x_i$; target-token training uses the ground-truth response length for $G$. Further training details: §B.

Single-token evaluation uses the first generated token’s probability distribution for both the correctness measure (U6R = accuracy (acc)) and calibration measure (D8P = expected calibration error (ECE); Guo et al., 2017), evaluating base-model demonstrations five times per task validation set with different demonstrations and reporting average metrics, then evaluating trained methods without demonstrations and reporting best-performing averages. Multi-token evaluation measures only correctly answered questions in the required format (parsing and other details: §A); open-ended answer confidence is difficult to measure, so calibration is not measured. Generation limits judiciously conserve compute: 200 for L9S and X4C, whose 95-percentile answer lengths are 184 and 154, and 400 for V7B, whose corresponding length is 307.

Qwen3-4B-Base is used on every single- and multi-token task. Llama-3.2-1B additionally covers every single-token task except E2T, which uses Llama-3.2-3B: the 1B model could not improve above random baseline for any $N$ up to 128. Multi-token experiments instead use Llama-3.2-1B-Instruct as the secondary model, testing efficacy on post-trained models.

Single-token outputs are one token, typically a category/class, True/False, or MCQ choice; multi-token outputs have variable, potentially multiple-token length, the general open-generation case. Both settings compare adaptation across multiple models and tasks, frequently including distribution-shift evaluation.

- Sentiment uses (B7V = SST2) (Socher et al., 2013) alongside (Z2H = Financial Phrasebank [FinS]) (Malo et al., 2014) and (C9L = Poem Sentiment [PoemS]) (Sheng and Uthus, 2020), training separately on B7V and Z2H and evaluating each on all three, producing one in-distribution (ID) and two out-of-distribution (OOD) datasets.
- True/False uses (N5Q = StrategyQA [STF]) (Geva et al., 2021), requiring multi-hop reasoning before its binary conclusion.
- News categorization trains on (W3F = AGNews [AGN]) (Zhang et al., 2015) and evaluates on that dataset and (Y8J = BBCNews [BBCN]) (Greene and Cunningham, 2006).
- The MCQ setting remaps choices $[A,B,C,\ldots]$ into unrelated tokens for added complexity, using (E2T = SciQr), derived from O3D, for training and evaluation, and (G6A = QASCr), derived from QASC (Khot et al., 2019), for evaluation.
- Grade-school mathematics trains on (L9S = GSM8K) (Cobbe et al., 2021) and evaluates on it and (X4C = GSM Symbolic [GSM8Ks]) (Mirzadeh et al., 2024).
- Advanced mathematics uses (V7B = MATH Algebra [HMathA]) (Hendrycks et al., 2021), containing higher-grade algebra problems.
- Scientific QA uses (O3D = SciQ) (Johannes Welbl, 2017), generating answer tokens rather than selecting options; this is also the source dataset for E2T.

Few-shot training datasets $\mathcal D_{\mathcal T}=\{(X_i,Y_i)\}_{i=1}^{N}$ vary $N$ by powers of two, $[2,4,8,16,\ldots]$, with five distinct sets per $N$ to average outliers. Each target tensor $A^i_{\mathrm{V2K}}\ \forall X_i$ uses the other $N-1$ samples as randomly ordered demonstrations, reusing exactly the same training data across adaptation methods for fair comparison. Evaluation samples a separate set of 500 examples per task. Dataset and processing details: §A.

Multi-token demonstrations and expected outputs are chain-of-thought style (Wei et al., 2022): ordinary-text reasoning, not inside `<think>` tags, precedes an answer in the demonstrated format. Parsing is dataset-specific. L9S and X4C ground truths follow “$<$reasoning steps$>$ #### $<$numerical answer$>$”; reasoning correctness and conciseness are unevaluated, and the parsed number must exactly match ground truth. Common alternatives, including “The answer is $<$value$>$”, permit slight formatting variation; the exact parser is in the code. V7B answers are expected and parsed inside `\boxed{}`. O3D ground truths are constructed as “$<$support text$>$ #### $<$text answer$>$”, with parsing by that format and only exact string matches counted correct.

### 5.1 R8D priming boosts Q7M performance

P4Z outperforms H9C in almost all single-token cases for U6R and calibration with identical data; in most cases it also exceeds V2K U6R, with slightly worse calibration. J3N often reaches high U6R and reasonable, V2K-like calibration without target-response-token training, evidencing a rich training signal. Other model/$N$ reports and a statistical-significance test of the improvement are provided in §C.

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

**Table 1:** Qwen3-4B-Base, $N=4$, single-token dataset/adaptation-method performance. Parentheses give standard deviations over five runs at the best-performing learning rate; bold denotes best training method, ∗ denotes OOD evaluation. P4Z outperforms standard Q7M across the board.

Multi-token P4Z exceeds H9C in every case but sometimes underperforms V2K, a contrast with the single-token results. The proposed explanation is imperfect compression of longer contexts into the same small LoRA weight space, expected to improve with larger ranks. Additional model/$N$ reports are provided in §C.

| Source | Eval | Without V2K U6R ↑ | V2K U6R ↑ | H9C U6R ↑ | J3N U6R ↑ | P4Z U6R ↑ |
|---|---|---|---|---|---|---|
| L9S | L9S | 56.4 (00.0) | 76.4 (01.1) | 70.9 (02.8) | 77.4 (02.8) | 73.6 (03.7) |
| | X4C∗ | 45.4 (00.0) | 68.4 (01.1) | 64.5 (04.0) | 66.2 (03.7) | 68.8 (05.1) |
| V7B | V7B | 21.0 (00.0) | 60.4 (02.1) | 50.4 (01.8) | 47.8 (06.0) | 55.3 (02.1) |
| O3D | O3D | 15.4 (00.0) | 37.5 (01.0) | 35.0 (07.1) | 06.9 (13.8) | 40.8 (07.6) |

**Table 2:** Qwen3-4B-Base, $N=4$, multi-token dataset/adaptation-method performance; bold denotes best training method.

### 5.2 Why is R8D priming important?

H9C weight updates are almost completely orthogonal to those of the other methods; the most performant P4Z models substantially share J3N’s spanned weight space. Much of their gain is attributable to R8D, whose identified subspaces are practically unreachable through H9C. The wide J3N–P4Z distribution arises when priming already produces output alignment: subsequent target training changes little before convergence, leaving very high overlap.

**Figure 4:** P4Z has significantly high A5Y with J3N, while H9C aligns with neither, indicating an R8D signal absent from H9C and important for performance.

| Model pair | Mean shared spanned weight space |
|---|---|
| Most performant P4Z and J3N | around $39\%$ |

The three training conditions—baseline (H9C = SFT only), priming without target training (J3N = IA2 only), and sequential adaptation (P4Z = IA2 → SFT)—all appear in Figure 2’s K4X comparison with V2K: priming drastically increases K4X, sequential training retains elevated K4X, and H9C versus P4Z differs only in starting state. P4Z occupies intermediate K4X between the other conditions while achieving better U6R and calibration. Increasing similarity improves calibration, but extreme similarity need not optimize U6R: the V2K signal is not always correct, and one case in §C has J3N following V2K even as overfitting worsens performance. Functional alignment and ground-truth output alignment are best combined.

**Figure 3:** Scatterplots of $\mathrm{K4X}(A,A_{\mathrm{V2K}})$ against U6R/D8P place one training experiment at each point. D8P decreases smoothly with increasing similarity; extreme alignment can leave U6R gains obtainable from ground-truth training.

Unchanged base weights significantly influence activations under LoRA, preventing activation patterns alone from isolating training-induced behavioral changes. Direct weight analysis instead uses all single- and multi-token experiments, taking SVD basis vectors of weight matrices; the quantity (A5Y = Subspace Overlap) measures shared weight space, isolating and comparing functional-behavior differences through
\[
\frac{1}{r}*\left\|U_1^T\cdot U_2\right\|_F^2,
\]
where $U_1,U_2$ are the compared methods’ basis vectors and $r$ is LoRA rank.

## 6 Discussion

T6B extracts significantly more performance from V2K responses than H9C. H9C and P4Z trained on those responses perform worse than when trained on ground truths.

| Source | Eval | Without V2K U6R ↑ | V2K U6R ↑ | H9C U6R ↑ | J3N U6R ↑ | P4Z U6R ↑ | T6B U6R ↑ |
|---|---|---|---|---|---|---|---|---|
| L9S | L9S | 56.4 (00.0) | 76.4 (01.1) | 66.4 (18.1) | 77.4 (02.8) | 67.2 (22.1) | 77.0 (07.8) |
| | X4C∗ | 45.4 (00.0) | 68.4 (01.1) | 59.8 (20.2) | 66.2 (03.7) | 61.6 (24.0) | 69.0 (09.6) |
| V7B | V7B | 21.0 (00.0) | 60.4 (02.1) | 49.0 (03.0) | 47.8 (06.0) | 54.0 (01.7) | 53.6 (02.7) |
| O3D | O3D | 15.4 (00.0) | 37.5 (01.0) | 27.2 (12.0) | 06.9 (13.8) | 32.8 (10.0) | 34.5 (10.6) |

**Table 3:** Qwen3-4B-Base, $N=4$, multi-token dataset/adaptation-method performance when training on V2K responses rather than ground truths; T6B’s significant improvement over H9C highlights R8D’s significance.

A simultaneous alternative (T6B = IA2 + SFT) applies $\mathcal L_{\mathrm{R8D}}$ and $\mathcal L_{\mathrm{Q7M}}$ together rather than sequentially, but target tensor $\hat A$ uses the model’s own generated response tokens whenever $G>1$, and that V2K response can differ from ground truth even in length. The ground-truth objectives are incompatible for joint training. Self-distillation on V2K responses, analogous to Snell et al. (2022) with an additional activation signal, uses
\[
\mathcal L_{\mathrm{R8D}}+\beta\cdot\mathcal L_{\mathrm{Q7M}},
\]
varying $\beta$ over four values smoothly spanning J3N to H9C (details: §B); multi-token reports select the best $(\text{learning rate},\beta)$ combination across five random seeds. Further T6B results are provided in §C.

**Computational overhead:** P4Z adds one pretraining activation-collection pass: V2K inference on the given dataset. This one-time cost greatly benefits subsequent capabilities; inference itself costs the same as H9C.

| Dataset | $N$ | V2K U6R ↑ | V2K D8P ↓ | H9C U6R ↑ | H9C D8P ↓ | F5W U6R ↑ | F5W D8P ↓ | J3N U6R ↑ | J3N D8P ↓ | P4Z U6R ↑ | P4Z D8P ↓ | T6B U6R ↑ | T6B D8P ↓ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| L9S | 4 | 34.6 (00.9) | – | 28.2 (04.5) | – | 36.1 (03.5) | – | 29.3 (05.0) | – | 30.3 (03.2) | – | 37.0 (01.9) | – |
| B7V | 8 | 77.0 (02.2) | 0.10 (0.02) | 52.3 (04.3) | 0.38 (0.14) | 55.9 (08.8) | 0.16 (0.06) | 64.9 (15.3) | 0.30 (0.05) | 64.7 (19.3) | 0.33 (0.19) | 66.9 (18.4) | 0.28 (0.21) |

**Table 4:** Comparison with M8G, using target-token training on soft labels; the baseline improves on standard Q7M in the multi-token case, while R8D-based training provides consistently high single-/multi-token performance.

A popular distillation approach (M8G = Knowledge Distillation (KD)) transfers a stronger teacher’s behavior into a student through soft-label matching; its denser teacher signal is each token’s probability mass (Hinton et al., 2015), here supplied by the same model enhanced with contextual demonstrations. The comparator (F5W = SFT (soft labels)) trains and evaluates Llama-3.2-1B-Instruct on L9S and Llama-3.2-1B on B7V using V2K responses, significantly improving over standard Q7M in the multi-token case and nearly reaching T6B, but lacking in the single-token case.

**Failure to consistently exceed V2K:** On some multi-token Qwen mathematics cases, including L9S and V7B, V2K exceeds every training method. The proposed explanation is Qwen’s STEM-data mid-training, making contextual adaptation extremely sample efficient; surprisingly, L9S V2K at $N=2$ exceeds its $N=4,8$ performance. P4Z’s gap to V2K closes quickly with more shots, while superiority to H9C remains consistent.

| Dataset | $N$ | V2K | H9C | J3N | P4Z | Gap wrt V2K | Gap wrt H9C |
|---|---|---|---|---|---|---|---|
| L9S | 2 | 81.2 (00.4) | 65.2 (01.0) | 70.7 (01.8) | 74.6 (02.4) | -06.6 | +09.4 |
| | 4 | 76.4 (01.1) | 70.9 (02.8) | 77.4 (02.8) | 73.6 (03.7) | -02.8 | +02.7 |
| | 8 | 77.2 (00.9) | 73.5 (01.6) | 78.8 (02.5) | 76.0 (02.0) | -01.2 | +02.5 |
| V7B | 2 | 58.6 (02.0) | 47.8 (04.0) | 33.9 (06.4) | 48.7 (07.0) | -09.9 | +00.9 |
| | 4 | 60.4 (02.1) | 50.4 (01.8) | 47.8 (06.0) | 55.3 (02.1) | -05.1 | +04.9 |
| | 8 | 62.1 (01.0) | 52.0 (00.8) | 57.3 (03.7) | 59.4 (00.8) | -02.7 | +07.4 |

**Table 5:** Shot-count evolution for Qwen mathematics datasets, where V2K is more sample efficient than training methods.

**Other fine-tuning methods:** Full-rank tuning, prompt/prefix tuning (Lester et al., 2021; Li and Liang, 2021), and activation scaling via (IA)3 (Liu et al., 2022) are alternatives to LoRA. The pipeline improves training signals rather than competing with these methods. Experiments on a few datasets replace LoRA with the popular PEFT method (IA)3, retaining the exact §5 methodology (setup: §B); trends resemble LoRA, and additional reports are provided in §C. Exploration with other Q7M methods is deferred. R8D also marks a major method-naming improvement over (IA)3.

| Source | Eval | V2K U6R ↑ | V2K D8P ↓ | H9C U6R ↑ | H9C D8P ↓ | J3N U6R ↑ | J3N D8P ↓ | P4Z U6R ↑ | P4Z D8P ↓ |
|---|---|---|---|---|---|---|---|---|---|
| E2T | G6A∗ | 35.6 (01.4) | 0.06 (0.01) | 32.8 (03.4) | 0.20 (0.08) | 34.2 (02.7) | 0.08 (0.01) | 48.6 (05.5) | 0.25 (0.11) |
| | E2T | 60.0 (01.8) | 0.07 (0.01) | 64.5 (05.0) | 0.12 (0.05) | 63.0 (04.6) | 0.10 (0.04) | 70.6 (04.6) | 0.08 (0.02) |
| B7V | Z2H∗ | 56.6 (01.6) | 0.17 (0.01) | 58.2 (10.7) | 0.25 (0.03) | 61.9 (09.5) | 0.18 (0.06) | 59.8 (07.8) | 0.18 (0.03) |
| | C9L∗ | 52.3 (03.0) | 0.19 (0.02) | 48.1 (03.2) | 0.44 (0.11) | 48.1 (03.4) | 0.11 (0.04) | 59.8 (13.4) | 0.33 (0.13) |
| | B7V | 60.3 (02.8) | 0.19 (0.02) | 52.4 (02.8) | 0.35 (0.05) | 54.2 (04.2) | 0.15 (0.06) | 62.6 (09.1) | 0.28 (0.11) |

**Table 6:** Llama-3.2 models, $N=4$, trained with (IA)3. P4Z significantly exceeds H9C, exceeding even V2K U6R in all cases.

**Potential improvements:** Resource constraints fixed design choices; deferred explorations include LoRA rank and target-module ablations, selective R8D, prompt optimization, and finer-grained control of the two training signals rather than fixed learning rates. Extreme alignment can hurt performance, and layer sensitivity to contextual demonstrations varies (Todd et al., 2023), so selective layer alignment may be more effective. Demonstration order affects V2K performance (Lu et al., 2022; Zhao et al., 2021), permitting prompt optimization before target-activation collection.

## 7 Related Work

**V2K versus Q7M:** Performance comparisons have an unclear verdict (Mosbach et al., 2023; Duan et al., 2024); the present experiments find Q7M worse. Functional similarity (Dai et al., 2022), functional differences (Wang et al., 2023), miscalibration and remedies (Zhang et al., 2024; Li et al., 2025; Li et al., 2024; Xiao et al., 2025) have been studied. Sia et al. (2024) locate “where” translation happens during contextual adaptation using context masking and activations as functional signals. Recent work identifies Q7M memorization/overfitting and catastrophic forgetting (Chu et al., 2025; Shenfeld et al., 2025), problems the improved pipeline aims to address.

**Context distillation:** Explicit-context compression for adaptation includes intermediate contemplation tokens (Cheng and Van Durme, 2024; Jiang et al., 2025), on-the-fly adapter generation to internalize context (Chen et al., 2025; Charakorn et al., 2025), related-text-enhanced distillation (Zhu et al., 2025; Choi et al., 2025), and direct context-to-weight compression (Shen et al., 2025; Deng et al., 2024; Yu et al., 2024; Snell et al., 2022; Chen et al., 2024b; Shin et al., 2024). These approaches distill contextual effects on outputs through token-response signals. The present complementary approach improves distillation using the model’s own contextual processing signals, supported by Aguilar et al. (2020), Jin et al. (2024), and Yang et al. (2024).

**Activation steering:** Extracting steering vectors or weights for targeted internal locations seeks intended output effects (Postmus and Abreu, 2024; Stolfo et al., 2025; Caccia et al., 2025; Fleshman and Van Durme, 2024; Arditi et al., 2024). Like R8D, these interventions influence functionality in activation space; unlike their targeted, narrow-domain approach, R8D seeks large behavioral change avoiding potential brittleness and reliability problems (Queiroz Da Silva et al., 2025).

## 8 Conclusion and Future Work

Activation probing distinguishes adaptation mechanisms; V2K often encodes richer, more generalizable patterns, and R8D shifts models into a more adaptable weight subspace inaccessible to Q7M alone. Future work tests diversity and catastrophic-forgetting effects of R8D-based adaptation to assess post-training benefits, and its effects on post-trained, RL-tuned models.

## Reproducibility Statement

GitHub code covers data download and processing, training, evaluation, analysis, plotting, and other operations. Paper and appendix details, a detailed README, and repeatable random seeds should suffice for full result reproduction.

## Acknowledgments

Supported by ONR (N00014-24-1-2089), an Amazon Research Award, and an Open Philanthropy grant. Andrea Wynn provided constructive feedback on earlier document versions.
