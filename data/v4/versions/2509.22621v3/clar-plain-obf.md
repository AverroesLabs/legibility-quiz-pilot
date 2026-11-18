# IA2: Alignment with ICL Activations

Improves Supervised Fine-Tuning

Aayush Mishra

Daniel Khashabi & Anqi Liu  
Department of Computer Science, Johns Hopkins University

♡ Equal advising. Correspondence: {amishr24,danielk,aliu.cs}@jhu.edu

###### Abstract

Across 12 popular benchmarks and two model families, P4Z significantly improves output accuracy and calibration, or agreement between confidence and correctness. P4Z applies ICL Activation Alignment (IA2), then Supervised Fine-Tuning (SFT), which trains model weights on labeled examples. This sequence provides useful adaptation and a way to examine its internal workings. The paper denotes accuracy as U6R.

R8D, the paper’s label for IA2, uses self-distillation: the model learns from its own signals. It aims to reproduce the activation patterns of In-Context Learning (ICL), denoted V2K, in models trained through weight updates. Activations are internal outputs produced as a model processes text. ICL adapts a model through examples in its context, without changing its weights. R8D encourages ICL-like internal reasoning. Its priming signal is unavailable through H9C, the SFT-only baseline.

## 1 Introduction

Generic large language models (LLMs), which process and generate text, serve specialized applications. For example, a news aggregator may classify articles into predefined categories. Parameter-efficient fine-tuning (PEFT) methods adapt models by training small parameter sets (Hu et al., 2021; Liu et al., 2022). These sets can often be loaded onto, and removed from, graphics processing unit (GPU) memory efficiently.

Generalization through Q7M—the paper’s label for SFT—typically requires many labeled samples (Le Scao and Rush, 2021). Collecting these samples can be expensive. V2K demonstrations and instructions instead use context capacity and increase computation for each query during inference, when the model generates an answer. They leave less space for query and response tokens, the units of text that the model processes.

We ask whether information-rich V2K activations can improve Q7M quality. We hypothesize that these activations contain information about how to extract contextual patterns that generalize. Few-shot weight adaptation, which trains on a small set of examples, may lack this information because it can learn shortcuts or overfit target responses. Similar behavior in token space need not mean similar internal processing. Different activations indicate different adaptation mechanisms, consistent with Shen et al. (2024).

Context-to-weight distillation transfers context-conditioned behavior into model weights. Methods that use response-text signals (Snell et al., 2022; Chen et al., 2024b) can inherit the same output-training problems. Reproducing a context-conditioned model’s outputs does not ensure that the trained model works in the same way.

**Figure 1:** Standard Q7M enforces agreement between model responses and targets in output space. This is the third step of the proposed pipeline. With little data, standard Q7M produces subpar performance and miscalibration. The pipeline collects rich V2K activations, enforces alignment in how the model works, and then trains against target responses. The comparison charts aggregate performance across models and datasets. Details: §5.

## 2 Background and Notation

For task $\mathcal T$, let $\mathcal D_{\mathcal T}=\{(X_i,Y_i)\}_{i=1}^{N}$ contain $N$ input–output pairs. Labeled weight adaptation, Q7M = Supervised Fine-Tuning (SFT), changes model parameters from $\Theta$ to $\Theta'$. The adapted model produces $M_{\Theta'}(X_t)=\bar y_t$, the first token of response $\bar Y_t$. Repeated application to the growing sequence extends the response.

For ground truth $Y_i$ with $G$ tokens, generate $G$ tokens $\bar Y_i$. Minimize cross-entropy, a loss that measures disagreement between predictions and target tokens, across all generated tokens:

\[
\mathcal L_{\mathrm{Q7M}}
=\sum_{i=1}^{N}\sum_{j=1}^{G}
\operatorname{cross\text{-}entropy}(\bar Y_{ij},Y_{ij}).
\tag{2}
\]

A decoder-only transformer language model (LM), $M_\Theta$, stacks self-attention (SA) and linear layers. Self-attention lets a token’s processing depend on other tokens in the sequence. The model processes token vectors $T=[t_i]_{i=1}^{R}$ through self-attention and linear projections at every token and layer. A linear projection transforms a vector using a weight matrix. The final layer’s last-token projection passes through the LM head, which predicts the most likely next token.

Let $W_Q,W_K,W_V,W_O\in\Theta$ denote the attention weight matrices, collectively $W_{QKVO}$. Then

\[
\operatorname{SA}(T;W_{QKVO})
=Z=
\left[
z_i=\sigma\left(\frac{q_iK^T}{\sqrt d}\right)V\cdot W_O
\right]_{i=1}^{R},
\tag{1}
\]

where $\sigma$ is Softmax, which converts scores into normalized weights. Here, $q_i=t_iW_Q$, $K=t_{:i}W_K$, and $V=t_{:i}W_V$ are the query, key, and value representations. The notation $t_{:i}$ denotes the first $i$ tokens, and the superscript $T$ denotes a matrix transpose. Both $Z$ and $T$ have shape $R\times d$: $R$ token vectors, each with $d$ components. We study behavioral changes caused by varying the context $T$ or the weights $W_{QKVO}$.

Let $I=[X_1\circ Y_1\circ\ldots\circ X_n\circ Y_n]$ concatenate demonstrations, where $\circ$ means concatenation. Inference-time adaptation, V2K = In-Context Learning (ICL), processes these examples to understand the task. It produces $M_\Theta(I\circ X_t)=\hat y_t$ without weight updates, often yielding a response $\hat Y_t$ similar to the expected $Y_t$.

First illustrated in GPT-3 (Brown et al., 2020), this widely used way to adapt generic language models also includes using generic instructions as demonstrations for instruction-following frontier models. Here, we restrict demonstrations to classic input–output pairs.

Hidden self-attention outputs are activations. They record context-dependent processing before the model produces the next token, so they help us examine behavioral changes. A model with $L$ layers processing $R$ tokens supplies an activation tensor $A\in\mathbb R^{L\times R\times d}$, a multidimensional array.

We compare activations with K4X = activation similarity (asim). This quantity uses token-wise cosine similarity, which compares vector directions, between two activation tensors. It has shape $L\times R$. We study sequences $T=[I\circ X]$ with demonstrations and $T=X$ without demonstrations, under different weights.

## 3 Are V2K and Q7M practically the same?

V2K responses have much better calibration than Q7M responses, with similar and slightly better U6R. This agrees with the few-shot generalization and calibration findings of Duan et al. (2024). Our proposed explanation is that output-oriented training learns shortcuts that fail on new data. By contrast, complex V2K circuits—internal processing structures described by Elhage et al. (2021)—must work across varied tasks and extract demonstration patterns that generalize. The methods adapt models differently. Q7M is not a drop-in replacement for V2K.

Recent accounts describe V2K as internal gradient descent, a process resembling parameter updates that reduce a loss (Von Oswald et al., 2023; Akyürek et al., 2022). They show self-attention producing outputs as though contextual demonstrations had updated the parameters. This functional equivalence predicts similar output-position activations for a base model with demonstrations and a trained model without them.

The observed activations do not align across models. Initial and final layers align better than middle layers. We expect the initial and final layers to process individual tokens, while middle layers perform abstract processing of the full demonstration set.

**Figure 2: Layerwise similarity with V2K activations.** H9C has small K4X, indicating that its adaptation works differently. P4Z aligns much more closely and performs much better. The first-row H9C curves average output-token K4X across tokens, samples, and datasets.

For this investigation, we collect $A_{\mathrm{V2K}}$ using $T=[I\circ X]$ on 100 test samples from multiple datasets, including B7V (SST2) and W3F (AGNews). We then train with those same demonstrations until convergence, when training settles, and collect $A_{\mathrm{Q7M}}$ using $T=X$.

The models span Qwen3 (Yang et al., 2025) and Llama-3.2 (Grattafiori et al., 2024). Their size–depth correspondences are $1\mathrm B\leftrightarrow16$ layers, $3\mathrm B\leftrightarrow28$ layers, and $4\mathrm B\leftrightarrow36$ layers, where B denotes billion parameters. We compare output positions using $\mathrm{K4X}(A_{\mathrm{V2K}},A_{\mathrm{Q7M}})$. Experiment details: §B.

## 4 Inducing V2K-like behavior in Q7M Models

After priming changes $\Theta$ to $\Theta'$, collect $\bar Y_i$ for every $X_i$ using $M_{\Theta'}$. Then train against ground truth $Y_i$ with $\mathcal L_{\mathrm{Q7M}}$ until convergence. This aligns the model with the intended target-token behavior.

The preceding priming operation, R8D = ICL Activation Alignment (IA2), aligns processing layer by layer rather than matching outputs. It seeks $\tilde W_{QKVO}=(\tilde W_Q,\tilde W_K,\tilde W_V,\tilde W_O)$ such that, for every newly generated token at index $-1$,

\[
\operatorname{SA}([I\circ X];W_{QKVO})
\approx
\operatorname{SA}(X;\tilde W_{QKVO})
\qquad \forall X\in\mathcal T.
\tag{3}
\]

Here, $\approx$ means approximately equal, and $\forall$ means “for every.”

Chen et al. (2024a) give a closed-form solution—one expressed directly rather than found through iterative training—for linearized attention using a Softmax kernel approximation. We instead seek a practical, general solution for real nonlinear transformers. The aim is to place V2K qualities directly into adapted weights, so the adapted model processes every query as the base model does at each layer.

For each training input $X_i$, construct $I$ from the remaining dataset samples. Generate $\hat Y_i$ from $T_i=[I\circ X_i]$, and collect output-position activations $A^i_{\mathrm{V2K}}\in\mathbb R^{L\times G\times d}$ for $G$ response tokens. Next, supply $T_i=[X_i\circ\hat Y_i]$ to obtain unaligned output-position activations $A^i$. This treats the response as if the model had generated it with only the query in context.

Minimize the mean squared error, a measure of differences between activation values, with respect to model parameters until convergence:

\[
\mathcal L_{\mathrm{R8D}}
=\sum_{i=1}^{N}\left\|A^i-A^i_{\mathrm{V2K}}\right\|.
\tag{4}
\]

## 5 Experimental Results

We train each $(\text{method},\text{dataset})$ combination to convergence, for up to 50 epochs, or passes through the training data. We use three learning rates, which control the size of training updates: slow:1e-4, medium:3e-4, and fast:1e-3. We select the best performer to neutralize the effects of hyperparameter selection, or the choice of training settings. We trained over 13,000 models.

Resource limits led us to fix rank-8 LoRAs targeting the self-attention matrices $W_Q,W_K,W_O$. LoRA means low-rank adaptation: it trains weight updates represented by lower-rank matrices. Rank controls the size of that representation. Multi-token training requires substantially more resources for long answers, so we restrict $N$ to 16 for Llama and 8 for Qwen.

For activation collection, training caps generation at $G=200$. This yields $\hat A_i\in\mathbb R^{L\times200\times d}\ \forall x_i$. Target-token training instead sets $G$ to the ground-truth response length. Further training details: §B.

Single-token evaluation uses the first generated token’s probability distribution for both U6R = accuracy (acc), the correctness measure, and D8P = expected calibration error (ECE), the calibration measure (Guo et al., 2017). Expected calibration error measures disagreement between confidence and correctness. For each task validation set, we evaluate the base model five times with different demonstrations and report average metrics. We then evaluate trained methods without demonstrations and report best-performing averages.

Multi-token evaluation counts only questions answered correctly in the required format. Parsing and other details appear in §A. We do not measure calibration because confidence in open-ended answers is difficult to measure.

We set generation limits to conserve computation: 200 for L9S (GSM8K) and X4C (GSM Symbolic, or GSM8Ks), whose 95-percentile answer lengths are 184 and 154; and 400 for V7B (MATH Algebra, or HMathA), whose corresponding length is 307.

We use Qwen3-4B-Base on every single- and multi-token task. Llama-3.2-1B also covers every single-token task except E2T (SciQr), which uses Llama-3.2-3B. The 1B model could not improve above the random baseline for any $N$ up to 128. Multi-token experiments instead use Llama-3.2-1B-Instruct as the secondary model to test efficacy on post-trained models, which have undergone further training after base-model training.

Single-token outputs contain one token, typically a category or class, True/False, or a multiple-choice question (MCQ) option. Multi-token outputs have variable, potentially multiple-token length. This is the general open-generation setting. Both settings compare adaptation across multiple models and tasks, often including evaluation under distribution shift, where evaluation data differ from training data.

- **Sentiment:** We use B7V = SST2 (Socher et al., 2013), Z2H = Financial Phrasebank [FinS] (Malo et al., 2014), and C9L = Poem Sentiment [PoemS] (Sheng and Uthus, 2020). We train separately on B7V and Z2H and evaluate each on all three datasets. This produces one in-distribution (ID) dataset, matching the training source, and two out-of-distribution (OOD) datasets, from other sources.
- **True/False:** We use N5Q = StrategyQA [STF] (Geva et al., 2021). It requires multi-hop reasoning—reasoning through several linked steps—before a binary conclusion.
- **News categorization:** We train on W3F = AGNews [AGN] (Zhang et al., 2015) and evaluate on it and Y8J = BBCNews [BBCN] (Greene and Cunningham, 2006).
- **MCQ:** We remap choices $[A,B,C,\ldots]$ into unrelated tokens to add complexity. We train and evaluate on E2T = SciQr, derived from O3D = SciQ. We also evaluate on G6A = QASCr, derived from QASC (Khot et al., 2019).
- **Grade-school mathematics:** We train on L9S = GSM8K (Cobbe et al., 2021) and evaluate on it and X4C = GSM Symbolic [GSM8Ks] (Mirzadeh et al., 2024).
- **Advanced mathematics:** We use V7B = MATH Algebra [HMathA] (Hendrycks et al., 2021), which contains higher-grade algebra problems.
- **Scientific question answering (QA):** We use O3D = SciQ (Johannes Welbl, 2017). The model generates answer tokens rather than selecting options. This dataset is also the source for E2T.

The few-shot training datasets $\mathcal D_{\mathcal T}=\{(X_i,Y_i)\}_{i=1}^{N}$ vary $N$ by powers of two, $[2,4,8,16,\ldots]$. We use five distinct sets per $N$ to average outliers. For each $X_i$, the target tensor $A^i_{\mathrm{V2K}}$ uses the other $N-1$ samples as randomly ordered demonstrations. We reuse exactly the same training data across adaptation methods for a fair comparison. Evaluation samples a separate set of 500 examples per task. Dataset and processing details: §A.

Multi-token demonstrations and expected outputs use a chain-of-thought style (Wei et al., 2022). Ordinary-text reasoning, outside `<think>` tags, precedes an answer in the demonstrated format. Parsing, which extracts the answer from the generated text, is dataset-specific.

L9S and X4C ground truths follow “$<$reasoning steps$>$ #### $<$numerical answer$>$”. We do not evaluate whether the reasoning is correct or concise. The parsed number must exactly match the ground truth. Common alternatives, including “The answer is $<$value$>$”, allow slight formatting variation. The code contains the exact parser.

V7B answers must appear inside `\boxed{}`, where the parser extracts them. We construct O3D ground truths as “$<$support text$>$ #### $<$text answer$>$”. We parse that format and count only exact string matches as correct.

### 5.1 R8D priming boosts Q7M performance

P4Z outperforms H9C in almost all single-token cases for U6R and calibration, using identical data. In most cases, it also exceeds V2K U6R, with slightly worse calibration. J3N, which uses IA2 only, often reaches high U6R and reasonable, V2K-like calibration without target-response-token training. This shows that the training signal is rich. Other model/$N$ reports and a statistical-significance test of the improvement appear in §C.

In the tables, ↑ means higher is better, and ↓ means lower is better.

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

**Table 1:** Qwen3-4B-Base, $N=4$, performance by single-token dataset and adaptation method. Parentheses give standard deviations, a measure of variation, over five runs at the best-performing learning rate. Bold denotes the best training method; ∗ denotes OOD evaluation. P4Z outperforms standard Q7M across the board.

Multi-token P4Z exceeds H9C in every case but sometimes underperforms V2K, unlike the single-token results. Our proposed explanation is that longer contexts compress imperfectly into the same small LoRA weight space. We expect larger ranks to improve this compression. Additional model/$N$ reports appear in §C.

| Source | Eval | Without V2K U6R ↑ | V2K U6R ↑ | H9C U6R ↑ | J3N U6R ↑ | P4Z U6R ↑ |
|---|---|---|---|---|---|---|
| L9S | L9S | 56.4 (00.0) | 76.4 (01.1) | 70.9 (02.8) | 77.4 (02.8) | 73.6 (03.7) |
| | X4C∗ | 45.4 (00.0) | 68.4 (01.1) | 64.5 (04.0) | 66.2 (03.7) | 68.8 (05.1) |
| V7B | V7B | 21.0 (00.0) | 60.4 (02.1) | 50.4 (01.8) | 47.8 (06.0) | 55.3 (02.1) |
| O3D | O3D | 15.4 (00.0) | 37.5 (01.0) | 35.0 (07.1) | 06.9 (13.8) | 40.8 (07.6) |

**Table 2:** Qwen3-4B-Base, $N=4$, performance by multi-token dataset and adaptation method. Bold denotes the best training method.

### 5.2 Why is R8D priming important?

H9C weight updates are almost completely orthogonal to those of the other methods: their directions have almost no alignment. The best-performing P4Z models share a substantial part of J3N’s spanned weight space, the space described by its weight directions. Much of their gain is attributable to R8D. H9C can practically never reach the subspaces identified by R8D.

The wide J3N–P4Z distribution arises when priming already aligns outputs. In these cases, subsequent target training changes little before convergence, leaving very high overlap.

**Figure 4:** P4Z has significantly high A5Y, or Subspace Overlap, with J3N. This quantity measures shared weight space. H9C aligns with neither method, indicating that R8D supplies a signal absent from H9C and important for performance.

| Model pair | Mean shared spanned weight space |
|---|---|
| Most performant P4Z and J3N | around $39\%$ |

All three training conditions appear in Figure 2’s K4X comparison with V2K:

- Baseline: H9C = SFT only.
- Priming without target training: J3N = IA2 only.
- Sequential adaptation: P4Z = IA2 → SFT, where the arrow means “followed by.”

Priming drastically increases K4X, and sequential training retains elevated K4X. H9C and P4Z differ only in their starting state. P4Z has intermediate K4X between the other conditions while achieving better U6R and calibration.

Increasing similarity improves calibration, but extreme similarity need not optimize U6R. The V2K signal is not always correct. In one case in §C, J3N follows V2K even as overfitting worsens performance. Combining functional alignment with ground-truth output alignment works best.

**Figure 3:** Scatterplots compare $\mathrm{K4X}(A,A_{\mathrm{V2K}})$ with U6R/D8P. Each point represents one training experiment. D8P decreases smoothly as similarity increases. With extreme alignment, ground-truth training can still provide U6R gains.

Under LoRA, unchanged base weights strongly influence activations. Activation patterns alone therefore cannot isolate behavioral changes caused by training. Instead, we analyze weights directly across all single- and multi-token experiments.

We take basis vectors—directions that describe a matrix’s spanned space—from the singular value decomposition (SVD) of weight matrices. A5Y = Subspace Overlap measures shared weight space. It isolates and compares differences in functional behavior through

\[
\frac{1}{r}*\left\|U_1^T\cdot U_2\right\|_F^2,
\]

where $U_1,U_2$ are the compared methods’ basis vectors and $r$ is the LoRA rank. The symbol $*$ denotes multiplication, and $\|\cdot\|_F$ denotes the Frobenius norm, the square root of the sum of squared matrix entries.

## 6 Discussion

T6B, which applies IA2 and SFT simultaneously, extracts significantly more performance from V2K responses than H9C. H9C and P4Z perform worse when trained on those responses than when trained on ground truths.

| Source | Eval | Without V2K U6R ↑ | V2K U6R ↑ | H9C U6R ↑ | J3N U6R ↑ | P4Z U6R ↑ | T6B U6R ↑ |
|---|---|---|---|---|---|---|---|
| L9S | L9S | 56.4 (00.0) | 76.4 (01.1) | 66.4 (18.1) | 77.4 (02.8) | 67.2 (22.1) | 77.0 (07.8) |
| | X4C∗ | 45.4 (00.0) | 68.4 (01.1) | 59.8 (20.2) | 66.2 (03.7) | 61.6 (24.0) | 69.0 (09.6) |
| V7B | V7B | 21.0 (00.0) | 60.4 (02.1) | 49.0 (03.0) | 47.8 (06.0) | 54.0 (01.7) | 53.6 (02.7) |
| O3D | O3D | 15.4 (00.0) | 37.5 (01.0) | 27.2 (12.0) | 06.9 (13.8) | 32.8 (10.0) | 34.5 (10.6) |

**Table 3:** Qwen3-4B-Base, $N=4$, performance by multi-token dataset and adaptation method when training on V2K responses rather than ground truths. T6B’s significant improvement over H9C highlights R8D’s importance.

The simultaneous alternative, T6B = IA2 + SFT, applies $\mathcal L_{\mathrm{R8D}}$ and $\mathcal L_{\mathrm{Q7M}}$ together rather than sequentially. However, whenever $G>1$, the target tensor $\hat A$ uses the model’s own generated response tokens. That V2K response can differ from the ground truth, even in length. The ground-truth objectives are incompatible for joint training.

Self-distillation on V2K responses, analogous to Snell et al. (2022) but with an added activation signal, uses

\[
\mathcal L_{\mathrm{R8D}}+\beta\cdot\mathcal L_{\mathrm{Q7M}},
\]

where $\beta$ weights the target-token loss. We vary $\beta$ over four values that smoothly span J3N to H9C. Details appear in §B. Multi-token reports select the best $(\text{learning rate},\beta)$ combination across five random seeds, the settings that control randomized operations. Further T6B results appear in §C.

**Computational overhead:** P4Z adds one activation-collection pass before training: V2K inference on the given dataset. This one-time cost greatly benefits subsequent capabilities. Inference itself costs the same as H9C.

The next comparison includes F5W = SFT (soft labels). Soft labels give token probability distributions rather than a single target token.

| Dataset | $N$ | V2K U6R ↑ | V2K D8P ↓ | H9C U6R ↑ | H9C D8P ↓ | F5W U6R ↑ | F5W D8P ↓ | J3N U6R ↑ | J3N D8P ↓ | P4Z U6R ↑ | P4Z D8P ↓ | T6B U6R ↑ | T6B D8P ↓ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| L9S | 4 | 34.6 (00.9) | – | 28.2 (04.5) | – | 36.1 (03.5) | – | 29.3 (05.0) | – | 30.3 (03.2) | – | 37.0 (01.9) | – |
| B7V | 8 | 77.0 (02.2) | 0.10 (0.02) | 52.3 (04.3) | 0.38 (0.14) | 55.9 (08.8) | 0.16 (0.06) | 64.9 (15.3) | 0.30 (0.05) | 64.7 (19.3) | 0.33 (0.19) | 66.9 (18.4) | 0.28 (0.21) |

**Table 4:** Comparison with M8G = Knowledge Distillation (KD), which transfers a teacher model’s behavior to a student, using target-token training on soft labels. The baseline improves on standard Q7M in the multi-token case. R8D-based training provides consistently high single- and multi-token performance.

M8G is a popular distillation approach that transfers a stronger teacher’s behavior into a student through soft-label matching. Its denser teacher signal is each token’s probability mass (Hinton et al., 2015). Here, the same model enhanced with contextual demonstrations supplies that signal.

For the comparator F5W = SFT (soft labels), we train and evaluate Llama-3.2-1B-Instruct on L9S and Llama-3.2-1B on B7V using V2K responses. It significantly improves over standard Q7M in the multi-token case and nearly reaches T6B, but falls short in the single-token case.

**Failure to consistently exceed V2K:** In some multi-token Qwen mathematics cases, including L9S and V7B, V2K exceeds every training method. Our proposed explanation is Qwen’s mid-training on science, technology, engineering, and mathematics (STEM) data. This makes contextual adaptation extremely sample efficient, or effective with few examples. Surprisingly, L9S V2K at $N=2$ exceeds its performance at $N=4,8$. P4Z’s gap to V2K closes quickly with more shots, or examples, while its superiority to H9C remains consistent.

In the following table, “wrt” means “with respect to.”

| Dataset | $N$ | V2K | H9C | J3N | P4Z | Gap wrt V2K | Gap wrt H9C |
|---|---|---|---|---|---|---|---|
| L9S | 2 | 81.2 (00.4) | 65.2 (01.0) | 70.7 (01.8) | 74.6 (02.4) | -06.6 | +09.4 |
| | 4 | 76.4 (01.1) | 70.9 (02.8) | 77.4 (02.8) | 73.6 (03.7) | -02.8 | +02.7 |
| | 8 | 77.2 (00.9) | 73.5 (01.6) | 78.8 (02.5) | 76.0 (02.0) | -01.2 | +02.5 |
| V7B | 2 | 58.6 (02.0) | 47.8 (04.0) | 33.9 (06.4) | 48.7 (07.0) | -09.9 | +00.9 |
| | 4 | 60.4 (02.1) | 50.4 (01.8) | 47.8 (06.0) | 55.3 (02.1) | -05.1 | +04.9 |
| | 8 | 62.1 (01.0) | 52.0 (00.8) | 57.3 (03.7) | 59.4 (00.8) | -02.7 | +07.4 |

**Table 5:** Changes with shot count for Qwen mathematics datasets, where V2K is more sample efficient than training methods.

**Other fine-tuning methods:** Alternatives to LoRA include full-rank tuning, which does not restrict updates to low rank; prompt/prefix tuning, which trains prompt or prefix representations (Lester et al., 2021; Li and Liang, 2021); and activation scaling, which changes activation magnitudes, through (IA)3 (Liu et al., 2022). The pipeline improves training signals rather than competing with these methods.

On a few datasets, we replace LoRA with the popular PEFT method (IA)3 while keeping the exact §5 methodology. Setup details appear in §B. The trends resemble those for LoRA, and additional reports appear in §C. We defer exploration with other Q7M methods. R8D also marks a major improvement in method naming over (IA)3.

| Source | Eval | V2K U6R ↑ | V2K D8P ↓ | H9C U6R ↑ | H9C D8P ↓ | J3N U6R ↑ | J3N D8P ↓ | P4Z U6R ↑ | P4Z D8P ↓ |
|---|---|---|---|---|---|---|---|---|---|
| E2T | G6A∗ | 35.6 (01.4) | 0.06 (0.01) | 32.8 (03.4) | 0.20 (0.08) | 34.2 (02.7) | 0.08 (0.01) | 48.6 (05.5) | 0.25 (0.11) |
| | E2T | 60.0 (01.8) | 0.07 (0.01) | 64.5 (05.0) | 0.12 (0.05) | 63.0 (04.6) | 0.10 (0.04) | 70.6 (04.6) | 0.08 (0.02) |
| B7V | Z2H∗ | 56.6 (01.6) | 0.17 (0.01) | 58.2 (10.7) | 0.25 (0.03) | 61.9 (09.5) | 0.18 (0.06) | 59.8 (07.8) | 0.18 (0.03) |
| | C9L∗ | 52.3 (03.0) | 0.19 (0.02) | 48.1 (03.2) | 0.44 (0.11) | 48.1 (03.4) | 0.11 (0.04) | 59.8 (13.4) | 0.33 (0.13) |
| | B7V | 60.3 (02.8) | 0.19 (0.02) | 52.4 (02.8) | 0.35 (0.05) | 54.2 (04.2) | 0.15 (0.06) | 62.6 (09.1) | 0.28 (0.11) |

**Table 6:** Llama-3.2 models, $N=4$, trained with (IA)3. P4Z significantly exceeds H9C and exceeds even V2K U6R in all cases.

**Potential improvements:** Resource constraints fixed our design choices. We defer studies of LoRA rank and target modules, selective R8D, prompt optimization, and finer-grained control of the two training signals instead of fixed learning rates. Rank and target-module ablations would vary those choices to examine their effects.

Extreme alignment can hurt performance. Layers also vary in their sensitivity to contextual demonstrations (Todd et al., 2023), so aligning selected layers may work better. Demonstration order affects V2K performance (Lu et al., 2022; Zhao et al., 2021). This allows prompt optimization before collecting target activations.

## 7 Related Work

**V2K versus Q7M:** Performance comparisons give no clear verdict (Mosbach et al., 2023; Duan et al., 2024). Our experiments find Q7M worse. Prior work studies functional similarity (Dai et al., 2022), functional differences (Wang et al., 2023), and miscalibration and its remedies (Zhang et al., 2024; Li et al., 2025; Li et al., 2024; Xiao et al., 2025).

Sia et al. (2024) locate “where” translation happens during contextual adaptation. They use context masking, which hides parts of the context, and activations as signals of model function. Recent work identifies Q7M memorization and overfitting, in which models learn training examples rather than generalizable patterns, and catastrophic forgetting, in which new training erases earlier capabilities (Chu et al., 2025; Shenfeld et al., 2025). Our improved pipeline aims to address these problems.

**Context distillation:** Approaches that compress explicit context for adaptation include:

- Intermediate contemplation tokens, which provide intermediate processing in token form (Cheng and Van Durme, 2024; Jiang et al., 2025).
- On-the-fly generation of adapters, small adaptation components, to internalize context (Chen et al., 2025; Charakorn et al., 2025).
- Distillation enhanced with related text (Zhu et al., 2025; Choi et al., 2025).
- Direct context-to-weight compression (Shen et al., 2025; Deng et al., 2024; Yu et al., 2024; Snell et al., 2022; Chen et al., 2024b; Shin et al., 2024).

These approaches distill the effects of context on outputs through token-response signals. Our complementary approach improves distillation through the model’s own contextual processing signals. Aguilar et al. (2020), Jin et al. (2024), and Yang et al. (2024) support this approach.

**Activation steering:** These methods extract steering vectors or weights to change internal processing at selected locations and produce intended output effects (Postmus and Abreu, 2024; Stolfo et al., 2025; Caccia et al., 2025; Fleshman and Van Durme, 2024; Arditi et al., 2024). Like R8D, they change model function in activation space. Unlike their targeted, narrow-domain approach, R8D seeks large behavioral change while avoiding potential brittleness and reliability problems (Queiroz Da Silva et al., 2025).

## 8 Conclusion and Future Work

Activation probing, or examining internal activations, distinguishes adaptation mechanisms. V2K often encodes richer patterns that generalize better. R8D shifts models into a more adaptable weight subspace that Q7M alone cannot access.

Future work will test how R8D-based adaptation affects diversity and catastrophic forgetting to assess its post-training benefits. It will also test effects on post-trained models tuned through reinforcement learning (RL), which trains behavior using rewards.

## Reproducibility Statement

The GitHub code covers data download and processing, training, evaluation, analysis, plotting, and other operations. The paper and appendix details, a detailed README, and repeatable random seeds should suffice to reproduce all results.

## Acknowledgments

ONR (N00014-24-1-2089), an Amazon Research Award, and an Open Philanthropy grant supported this work. Andrea Wynn provided constructive feedback on earlier versions of the document.
