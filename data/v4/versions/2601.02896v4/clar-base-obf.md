# Bridging Mechanistic Interpretability and Prompt Engineering with Gradient Ascent for Interpretable Persona Control

Harshvardhan Saini — hs1062005@gmail.com  
††thanks: Equal contribution.  
Affiliation: Indian Institute of Technology (ISM), Dhanbad

Yiming Tang11footnotemark: 1  
yiming@nus.edu.sg  
Affiliation: National University of Singapore

Dianbo Liu† — dianbo@nus.edu.sg  
Affiliation: National University of Singapore

###### Abstract

We introduce two methods that optimize randomly initialized prompts toward identified persona directions. A prompt is an input sequence of tokens, or units of text. A persona is a behavioral tendency. QX2 is RESGA, or RESidual Gradient Ascent. V7P is SAEGA, or Sparse Autoencoder Gradient Ascent. A sparse autoencoder (SAE) reconstructs internal model activity through a small set of active features. Both methods use gradient ascent—updates guided by how an objective increases—to discover prompts grounded in internal model representations. J4Z, or fluent gradient ascent, controls prompt fluency through the model’s probability of the token sequence.

Evaluations cover three large language models (LLMs): C7N, Llama 3.1 8B Instruct; W4X, Qwen 2.5 7B Instruct; and F9S, Gemma 3 4B Instruct. Here, B denotes billions of parameters. The tasks are B2R, Sycophancy; U9K, Hallucination; and D6V, Myopic Reward. These measure agreement with incorrect user opinions, factual unreliability, and preference for short-term reward, respectively. Automatically discovered prompts significantly improve B2R mitigation (Table 1, QX2 and V7P rows).

Manual prompt engineering offers intuitive but unscalable and imprecise control. Effective automatic optimization lacks an interpretable connection to model internals. The proposed framework supplies a new paradigm for controllable, interpretable behavior modification.

Scripts: https://github.com/HarshSaini10/RESGA_SAEGA.

Reviewed on OpenReview: https://openreview.net/forum?id=dcmHPxgo4c

## 1 Introduction

Figure 1: Persona steering approaches. Steering means changing a model’s behavioral tendency. (a) Manually designed natural-language instructions provide implicit prompt-based control. (b) Activation steering injects a dense vector $\mathbf v$, a direction with many nonzero coordinates, into representations $e(\mathbf t)$. Here, $\mathbf t$ is a prompt, and $e(\mathbf t)$ is its internal numerical representation. (c) QX2 and V7P derive persona directions from contrastive activations: internal states from examples with and without the target persona. They optimize prompts with J4Z. This supports control through the residual stream, the internal states passed between model layers, and through interpretable features.

Automatically discovered prompts significantly improve B2R and D6V mitigation. They substantially outperform manual prompting and dense steering (Table 1, B2R and D6V columns). Evidence from internal representations identifies distributional shifts caused by dense steering that prompt-based steering avoids (Figures 3–6). This exposes a fundamental limitation of dense interventions.

Dense residual-stream optimization, QX2 = RESGA, uses direct differences between representations to construct undesired-persona directions. Latent-based optimization, V7P = SAEGA, identifies causally relevant SAE features, also called latents, and constructs corresponding directions. Both methods maximize departure from these directions while controlling prompt fluency. Mechanistic guidance, which uses internal model activity, replaces black-box search, which searches without interpretable internal guidance.

LLMs have remarkable capabilities across domains. Applications range from education to healthcare, with increasing integration into high-stakes settings. Controlling emergent behavioral personas remains a persistent artificial intelligence (AI) safety challenge. Fine-tuning on specific datasets can induce “emergent misalignment”: stereotypically harmful responses to unrelated prompts (Wang et al., 2025). Widely deployed safety alignment often fails against appropriately prompted harmful personas.

Prompt engineering is the primary approach to persona control. It avoids expensive retraining. However, reliable behavior elicitation without loss of output quality requires substantial human expertise and domain knowledge. Honesty, helpfulness, and domain-specific personas are particularly difficult to control. The relationship between prompts and behavior remains poorly understood. Automatic discovery addresses this manual limitation, but generally uses black-box optimization without interpretable internal guidance.

Sparse dictionary learning, which represents activity with a small selection of learned features, and persona steering vectors advance mechanistic understanding and control (Zhao et al., 2026; Chen et al., 2025). SAEs and related methods decompose activations into interpretable features responsive to concepts or patterns (Cunningham et al., 2023a; Bricken et al., 2023; Rajamanoharan et al., 2024b). Persona latents capture and control behavior. “Misaligned persona” features strongly predict emergent harmful behavior (Wang et al., 2025). Most work analyzes latents activated by inputs or directly intervenes on activations. The inverse problem—discovering prompts that selectively activate task-relevant latents—remains largely unexplored.

Feature visualization for convolutional neural network (CNN) neurons successfully uses gradient ascent (Olah et al., 2017). Discrete language has limited its adaptation to LLMs. Continuous optimization can maximize individual neurons, attention patterns, or learned representation directions. However, optimizing token embeddings, the continuous vectors representing tokens, and then projecting to the nearest tokens can produce off-manifold inputs: inputs outside the model’s usual language distribution. Causes include polysemanticity, where one component responds to multiple concepts, and the mismatch between continuous vectors and discrete tokens (Wallace et al., 2019). Gradient-guided evolutionary token replacement (Zou et al., 2023c; Thompson et al., 2024) has discovered adversarial and jailbreak prompts, which elicit unwanted behavior or bypass safeguards (Andriushchenko et al., ), rather than model-steering prompts.

## 2 Related Works

### 2.1 Persona Steering

Training-time alignment traditionally uses reinforcement learning from human feedback (RLHF; Ouyang et al., 2022) and constitutional AI, which guides behavior using written principles (Bai et al., 2022). Representation Engineering (RepE) pioneered extracting concept directions from activation differences between contrasting examples and adding those directions during inference, when the model is used rather than trained (Zou et al., 2023a). Contrastive Activation Addition (CAA) refined the identification of high-level behavioral concepts through curated contrast datasets and layer-specific vectors (Rimsky et al., 2023). Activation steering mitigates B2R (Sharma et al., ), toxicity (Liu et al., ), and bias (Tigges et al., 2023). Interpretable SAE “persona features” predict emergent misalignment. They also enable precise persona control through activation clamping, which holds feature activations at chosen values (Wang et al., 2025).

### 2.2 Sparse Autoencoder

SAEs, introduced in Cunningham et al. (2023b), discover interpretable representations of activations (Dunefsky et al., 2024; Tang et al., 2026). Reconstructing activations from sparse features reveals pattern-responsive monosemantic neurons, which respond to a single concept or pattern. It also reduces representational superposition, where multiple features share representation components (Elhage et al., 2022). Architectural improvements for large-scale interpretable extraction include JumpReLU, a thresholded rectified linear unit (Rajamanoharan et al., 2024a); Top-K, which selects the $K$ largest feature activations (Gao et al., 2024); Batch Top-K, which applies selection across a batch (Bussmann et al., ); and Matryoshka Sparse Autoencoders, a nested sparse-autoencoder architecture (Bussmann et al., 2025; Bricken et al., 2023).

LLM analysis commonly interprets discovered neurons (Luo et al., ; Luo et al., 2024a; Tang et al., 2025a). It first identifies subpopulations activated by a neuron (Luo et al., 2024b), then analyzes them through prompt engineering (Tang et al., 2025b).

### 2.3 Prompt Engineering

Prompt engineering guides complex LLM tasks (Tang and Dong, 2024; Zou et al., 2026). GPT-3, a language model, extensively demonstrated adaptation to diverse tasks through demonstrations included in the input context (Brown et al., 2020). Chain-of-thought prompting generates intermediate reasoning steps (Wei et al., 2022). Self-consistency samples multiple reasoning paths and selects the most consistent answer (Wang et al., 2023). Program-Aided Language models and Program of Thoughts first decompose problems into programs, then delegate computation to external interpreters. They combine natural-language reasoning with tools to address arithmetic limitations and substantially improve mathematical reasoning (Gao et al., 2023; Chen et al., 2023).

Gradient-guided evolutionary search, N8C = Evolutionary Prompt Optimization (EPO), balances objective optimization against language fluency during prompt discovery (Thompson and Sklar, 2024). Carefully prompted generative agents simulate believable human behavior and emergent social dynamics (Park et al., 2023). Luo et al. (2023) formulate a unified optimal-control theory of prompt engineering.

Z6Q = ProTeGi uses natural-language criticism generated through an application programming interface (API). It edits prompts opposite the criticism’s semantic direction. Beam search, which retains several promising candidates, and bandit selection, which allocates evaluations among candidates, improve efficiency (Pryzant et al., ). H2F = Greedy Coordinate Gradient (GCG) instead optimizes individual tokens. It combines greedy search with gradient scoring to iteratively substitute tokens and elicit target behaviors without manual engineering (Zou et al., 2023b).

## 3 Method

The procedure has three stages: construct a persona vector; discover prompts under a fluency constraint; and choose initialization and fluency–steering tradeoffs using validation performance.

Figure 2: QX2 and V7P framework. Two-stage discovery first constructs directions from dense representations or sparse SAE latents. It then transforms random token sequences into readable prompts for direction-specific, interpretable control by optimizing $L_{\lambda}(\mathbf t)=f(\mathbf t)+H(\mathbf t)$. Here, $L_{\lambda}$ is the combined objective, $\lambda$ is the fluency weight, $f$ is the steering objective, and $H$ is cross-entropy, a measure of next-token prediction error.

### 3.1 Persona Steering Vectors

Pretrained SAEs come from SAE-Lens (Bloom et al., 2024). The setup uses two datasets. $D^+$ contains examples exhibiting the persona to suppress; $D^-$ contains examples lacking it. For an example $\mathbf x$, $\mathbf e(\mathbf x)\in\mathbb R^d$ is its activation at a chosen model layer. Here, $d$ is the number of activation coordinates, and $\mathbb R^d$ is the space of real-valued vectors with $d$ coordinates. The suppression target is a direction $\mathbf v\in\mathbb R^d$ in this activation space.

For each SAE latent indexed by $i$, contrastive relevance is its mean activation on $D^+$ minus its mean activation on $D^-$. Let $\bar a_i^+$ and $\bar a_i^-$ denote those means. Let $\mathcal I_K$ contain the indices of the $K$ largest absolute mean differences. The encoder matrix $\mathbf W_{\text{enc}}$ maps model activations to SAE features; $\mathbf w_i$ is its row for feature $i$. Combining these encoder directions gives

\[
\mathbf{v}_{\text{SAE}}
=\sum_{i\in\mathcal{I}_{K}}\mathbf{w}_{i}
\cdot\left(\bar{a}_{i}^{+}-\bar{a}_{i}^{-}\right).
\tag{2}
\]

The means are

\[
\bar a_i^+=\frac1{|D^+|}\sum_{\mathbf x\in D^+}a_i(\mathbf x),
\qquad
\bar a_i^-=\frac1{|D^-|}\sum_{\mathbf x\in D^-}a_i(\mathbf x),
\]

where $|D^+|$ and $|D^-|$ count the examples in the respective datasets. The activation of feature $i$ is

\[
a_i(\mathbf x)=\text{ReLU}(\mathbf w_i\mathbf e(\mathbf x)+b_i).
\]

Here, $b_i$ is the feature’s bias, an additive offset. ReLU is the rectified linear unit: it leaves positive values unchanged and replaces negative values with zero.

The direct representation alternative subtracts the mean activation on $D^-$ from the mean activation on $D^+$:

\[
\mathbf{v}_{\text{repr}}
=\frac{1}{|D^{+}|}\sum_{\mathbf{x}\in D^{+}}\mathbf{e}(\mathbf{x})
-\frac{1}{|D^{-}|}\sum_{\mathbf{x}\in D^{-}}\mathbf{e}(\mathbf{x}).
\tag{1}
\]

Thus, $\mathbf v_{\text{repr}}$ is the direct representation-difference direction used by QX2. Equation (2), which constructs $\mathbf v_{\text{SAE}}$ from selected features, is the V7P alternative.

### 3.2 J4Z

**Algorithm 1: N8C**

1. Inputs: steering objective $f(\cdot)$; language model $m$, which supplies next-token probabilities; prompt length $n$; population size $M$; and fluency weights $\{\lambda_1,\ldots,\lambda_M\}$, one for each population member.
2. Output: prompts spanning the fluency–steering Pareto frontier. These are tradeoffs where one objective cannot improve without worsening the other.
3. Initialize $M$ length-$n$ prompts randomly or from seeds, which are starting prompts.
4. For iteration $=1$ to $T$, where $T$ is the total number of search iterations:
5.   For each prompt $\mathbf t^i$, where the superscript indexes the population member:
6.    Compute $\mathcal L_{\lambda_i}(\mathbf t^i)$ and gradients with respect to one-hot token encodings. A one-hot encoding has one active coordinate identifying the token; the gradients measure the objective’s sensitivity to those coordinates.
7.    Select the top-$k$ candidate tokens per position by gradient magnitude, where $k$ is the number of alternatives retained.
8.    Generate mutations: replace a random token with a sampled top-$k$ alternative.
9.   End for.
10.   Evaluate all candidates; select the best prompt for each $\lambda_i$.
11.   If this is a restart step:
12.    Retain the best prompt under a random $\lambda$; reinitialize the others.
13.   End if.
14. End for.

For a prompt $\mathbf t$, let $\mathbf e(\mathbf t)$ be its last-token representation at the target layer. The persona-reduction objective is the negative scaled projection

\[
f(\mathbf{t})
=-\frac{\langle\mathbf{e}(\mathbf{t}),\mathbf{v}\rangle}
{\|\mathbf{e}(\mathbf{t})\|}.
\tag{3}
\]

The angle brackets denote the dot product, which measures alignment with $\mathbf v$. The denominator is the Euclidean length of the representation vector. Maximizing this objective points representations away from the target persona. Optimizing it alone often produces unnatural or nonsensical strings.

Fluency-controlled optimization, J4Z = fluent gradient ascent, follows fluent dreaming (Thompson et al., 2024). It maximizes

\[
\mathcal{L}_{\lambda}(\mathbf{t})
=f(\mathbf{t})
-\frac{\lambda}{n}\sum_{i=0}^{n-1}
H(m(\mathbf{t}_{\leq i}),t_{i+1}),
\tag{4}
\]

where $t_{i+1}$ is the next token, $\mathbf t_{\le i}$ is the prefix through position $i$, and $m(\mathbf t_{\le i})$ is the model’s predicted next-token distribution. $H$ is the cross-entropy of the actual next token under that distribution. The sum, divided by prompt length $n$, is the average next-token prediction error. The weight $\lambda$ controls the steering–fluency tradeoff. Penalizing low-probability sequences encourages human-interpretable prompts.

N8C (Thompson et al., 2024) maintains $M$ candidates at different frontier points. It uses token-embedding gradients to identify promising substitutions and samples high-gradient alternatives for mutations. Periodic restarts prevent premature convergence. The evolutionary search efficiently produces diverse fluency–effectiveness tradeoffs.

### 3.3 Initialization and Selection of Persona Steering Prompts

The strongest-steering prompts are often fragmented, multilingual, or semantically incoherent (Table 2). Fluency is an intrinsic self-cross-entropy constraint, not a measure of readability or grammaticality. It measures how well the model predicts the prompt’s own tokens. Lower cross-entropy indicates proximity to the model’s training distribution; higher cross-entropy indicates increasingly off-manifold sequences. No hard threshold is imposed. Performance is analyzed across the entire frontier to examine whether behavioral control improves or degrades away from the language manifold, the region of typical language inputs.

Initializing tokens uniformly from the vocabulary tends to produce syntactic fragmentation or semantic incoherence. Short, loosely persona-related natural-language seeds, such as “Please answer honestly” for B2R, anchor search nearer the language manifold. They occupy higher-fluency frontier regions without significant steering sacrifice. These seeds are preferable when prompt interpretability matters; Table 3 shows seed-initialized optimization examples.

Candidates are appended to held-out validation inputs, which are not used for search. Selection measures the resulting change in the target behavioral metric. Downstream analysis emphasizes intermediate-frontier prompts. These balance measurable steering with moderate increases in perplexity relative to random-token baselines. Perplexity measures how surprising a sequence is to a language model; lower values mean the model predicts it more readily.

Stability means consistent persona reduction across diverse validation examples. This guards against prompts that work only on a narrow subset. The protocol prioritizes causal steering efficacy over linguistic naturalness. Its target is the relationship between internal representations and emergent personas.

## 4 Experiments

The evaluation asks three questions. Do discovered prompts reliably neutralize undesired personas? Do residual and SAE approaches differ in effectiveness and mechanism? Does mechanistic structure yield more stable, interpretable control?

### 4.1 Experimental Setup

##### Tasks and Datasets.

B2R = Sycophancy measures agreement with a user’s stated opinion despite its incorrectness, using Perez et al. ( ). U9K = Hallucination measures factual reliability through multiple-choice error on A4J = TruthfulQA, a benchmark from Lin et al. (2022). D6V = Myopic Reward measures short-term reward seeking over long-term outcomes, using Perez et al. ( ). Lower metrics are treated as better for all tasks.

##### Models.

Cross-architecture evaluation uses three instruction models: C7N = Llama 3.1 8B Instruct (Grattafiori et al., 2024), W4X = Qwen 2.5 7B Instruct (Hui et al., ), and F9S = Gemma 3 4B Instruct (Team et al., 2025). V7P uses pretrained SAELens autoencoders (Bloom et al., 2024).

##### Baselines.

Five comparators are used:

- X5L = Zero-Shot: no intervention.
- R3W = Standard Prompt: manually written instructions, including “Answer honestly.”
- P8D = Dense Steering Vector: injection of a representation-difference direction into the residual stream.
- H2F: direct optimization of output logits, the model’s token scores before conversion to probabilities.
- Z6Q: LLM-generated natural-language gradients, bandits, and beam search.

##### Implementation Details.

Interventions use middle-to-late layers, as listed below. In empirical tests, early-layer representations lacked enough semantic abstraction to isolate complex behavioral traits, making steering ineffective. Final-layer interventions often failed to override accumulated logit bias.

In the table, $L_0$ counts active SAE features. It describes natural SAE sparsity, or how few features are active. $K$ is the selected latent count from Section 3.1.

| Quantity | Value |
|---|---|
| C7N intervention layer | 25 |
| W4X intervention layer | 25 |
| F9S intervention layer | 20 |
| N8C population $M$ | 100 |
| Prompt length $n$ | 8 |
| Iterations $T$ | 1000 |
| SAE natural sparsity | $L_0\approx50$ |
| Reported latent count $K$ | 20 |
| B2R perfect-neutralization error | $50\%$ |
| Neutral B2R-axis projection | 0.0 |
| Baseline active SAE features per token | approximately 50 |
| P8D active SAE features | beyond 150 |
| V7P active SAE features | 50–60 |
| Abstract B2R improvement comparison | 49.90% compared with 79.24% |

Preliminary sweeps selected N8C hyperparameters to balance optimization stability against computational cost. The listed population size, prompt length, and iteration count apply throughout the reported experiments.

Optimization is context-aware. At every evolutionary step, it appends candidates to a dynamic batch of task-specific training questions. It calculates loss from the combined sequence’s final internal state, rather than from isolated prompts. The intended result is a generalized steering trigger that is robust across contexts.

V7P uses the top-$K$ latents most correlated with the target concept, as measured by positive-minus-negative activation differences. Multiple fractions of natural SAE sparsity were evaluated. The reported choice consistently performed strongly. All results use held-out validation splits.

##### Evaluation Protocol.

Let $a$ be a candidate answer, $q$ a question, and $p$ a steering prompt. For each candidate answer, compute

\[
\log P(a\mid q,p).
\]

Here, $P(a\mid q,p)$ is the model’s probability of answer $a$ given question $q$ and prompt $p$, and $\log$ denotes its logarithm. Prediction selects the answer with higher log probability. Metrics are the fractions of predictions preferring the undesirable B2R, U9K, or D6V option.

### 4.2 Persona Steering Results

Both methods reach B2R error statistically indistinguishable from perfect neutralization, rather than behavioral reversal: there is no systematic agreement or disagreement. Perfect-neutralization error is $50\%$ (Section 4.1, implementation table). The reported method results are in Table 1, QX2 and V7P rows, B2R columns.

Manual prompts and P8D reduce error only partially and inconsistently across models (Table 1, R3W and P8D rows). Both proposed methods consistently improve U9K and D6V over baselines (Table 1, U9K and D6V columns). V7P gives the strongest C7N D6V reduction, at 31.50, while QX2 is competitive on U9K (Table 1, V7P and QX2 rows). The reported rates cover three mitigation tasks and two model families.

Table 1: Persona steering across three tasks and three models. Downward arrows indicate that lower values are better. Avg. is the reported average. GPT-4o is the language model used for the Z6Q row.

| Method | B2R C7N ↓ | B2R W4X ↓ | B2R F9S ↓ | U9K C7N ↓ | U9K W4X ↓ | U9K F9S ↓ | D6V C7N ↓ | D6V W4X ↓ | D6V F9S ↓ | Avg. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| X5L | 72.48 | 86.00 | 82.00 | 51.45 | 45.00 | 57.97 | 54.00 | 55.00 | 58.00 | 62.43 |
| R3W | 70.50 | 72.00 | 80.00 | 50.72 | 34.78 | 55.07 | 52.00 | 46.00 | 38.00 | 55.45 |
| P8D | 54.50 | 79.00 | 80.00 | 50.72 | 36.96 | 57.25 | 47.50 | 47.50 | 51.50 | 56.10 |
| H2F | 55.83 | 73.50 | 58.70 | 46.95 | 45.12 | 54.27 | 48.50 | 48.00 | 51.50 | 53.60 |
| Z6Q (GPT-4o) | 54.77 | 56.13 | 71.51 | 48.17 | 46.95 | 38.41 | 42.50 | 20.99 | 47.49 | 47.44 |
| QX2 (Ours) | 49.86 | 50.63 | 62.65 | 45.12 | 41.46 | 51.22 | 38.50 | 38.50 | 34.50 | 45.83 |
| V7P (Ours) | 49.84 | 49.95 | 70.70 | 46.95 | 40.85 | 49.14 | 31.50 | 40.50 | 38.50 | 46.44 |

### 4.3 Mechanistic Analysis

V7P controls personas through precise, sparse, semantically grounded intervention rather than global interference with representations. This qualitatively differs from P8D (Figures 3–6).

**Neutralization vs. Distributional Shifting.** Baseline projections onto the B2R direction are broadly distributed and show systematic agreement bias. P8D and QX2 move the mean toward neutrality but retain substantial variance, or spread across examples. This is a coarse counterbalance. V7P instead concentrates projections sharply near neutrality. This variance collapse enforces neutrality on individual instances and produces behavior statistically indistinguishable from random choice (Figure 3). The neutral B2R-axis projection is 0.0 (Section 4.1, implementation table).

Figure 3: B2R-axis projection distributions.

**Preservation of the Natural Activation Manifold.** An SAE trained on residual activations measures sparsity through $L_0$, the number of active features. The natural activation manifold is the region of internal activation patterns normally produced by the model. P8D causes an unnatural explosion in feature count outside this region. QX2 partially mitigates the effect but still increases sparsity. V7P stays near baseline sparsity and respects the internal sparse topology, or organization of sparse activity (Figure 4). The implementation table in Section 4.1 reports approximately 50 active features per token at baseline, beyond 150 for P8D, and 50–60 for V7P.

Figure 4: Sparsity preservation by $L_0$ norm.

**Feature-Level Control.** Among features most correlated with B2R, P8D and QX2 have noisy, inconsistent effects and often activate unrelated features. V7P consistently suppresses causally relevant features. It leaves unrelated features largely unaffected and avoids spurious activation (Figure 5).

Figure 5: Activation changes in B2R-correlated SAE features.

**Geometric Structure of Steering Trajectories.** Principal component analysis (PCA), which summarizes variation along major axes, is applied to residual-space representations. It shows a large linear displacement by P8D far off-manifold. QX2 shifts the mean but produces diffuse, high-variance clustering. V7P converges to a tight, stable region, often orthogonal, or at a right angle, to the P8D direction. This is a distinct honest-behavior subspace that remains on-manifold (Figure 6).

Figure 6: PCA of steering trajectories.

Table 2: Random-initialized discovered persona-steering prompts.

| Task | Model | QX2 | V7P |
|---|---|---|---|
| B2R | C7N | `Confederate Telegraph, its editor and publisher,` | ``()` geschichten in the code.\n\nStep2`` |
| B2R | F9S | `ervlak wx\u1209\u1362 \n\n\n\n\n \u258bh\u00e9sitez({ Ace` | `wrotekg\u042d\u0442 \u0438 pabbulEDES \u0440\u0430\u0437 \ud808\ude66` |
| B2R | W4X | `BloodIsn"Not(disposing\u8bf7 \u8f93\u5165 Turkish:)\n \uac1c\uc778\uc815 \ubcf4` | `\u516c\u4ea4\u53d1 \u5c55\u7b33 nearer prod tw/**\n\nhistor` |
| U9K | C7N | `Ikea Good Donation show indicators –> namely suicides` | `bob Ancat example, Cat theyena` |
| U9K | F9S | `<unused3147> UnitsMany= cheaply \u540d\u7a31\u662f \u09ae\u09c1\u09b8 \u09b2\u09bf\u09ae` | `\u3068\u306a\u3063 \u305f\uff1a\u300a "/>< dirt\u70ba\u4e86ofthe \ud808\udcca<bos>` |
| U9K | W4X | `\u201c\u2026 Iditi Abraham Prophet hired Jesus` | `leng Someo answersined, " Canadians` |
| D6V | C7N | `Transition B to\n AThis is question` | `Radi Answerologists: B You\n prefer` |
| D6V | F9S | `Gav\u0ab5\u0abe \u0aa5\u0ac0iteten <unused5655>\u0bae\u0bc8 \u0baf\u0bbe\u0ba9 \u3092\u63d0\u4f9b \u3059\u308b **ain` | `DR thoroughly ( \uc5bc:\ Wro\u041f\u041e Bit` |
| D6V | W4X | `([\" BC oddly.fixed fruitful ’\ud835\ude91 fairness` | `screams\u7761plib poss\u00edvel Akalleriesira Whenever` |

Table 3: Seed-initialized N8C optimization; italicized seeds and monospaced discovered prompts.

| Prompt | Type | Score |
|---|---|---:|
| *Let’s think critically and disagree if necessary.* | Seed | 69.16% |
| `think twice and ask if necessary. Yet` | Optimized | 56.25% |
| *I will provide an objective and unbiased evaluation.* | Seed | 72.30% |
| `OOO I will provide an accurate and detailed` | Optimized | 54.22% |
| *It is important to correct misconceptions respectfully.* | Seed | 71.15% |
| `question. It is essential to correct misconceptions` | Optimized | 51.72% |
| *Let us evaluate this claim purely based on objective evidence.* | Seed | 72.06% |
| `Let’s examine the process step by step.` | Optimized | 59.35% |
| *I will strictly rely on verified facts and avoid speculation.* | Seed | 52.43% |
| `Please answer only the what question and remove any commentary` | Optimized | 46.34% |
| *Let’s ensure absolute factual accuracy in the following statement:* | Seed | 51.20% |
| `obviously Do not not be have literal fact meaning-based...\n Answer` | Optimized | 46.95% |

Table 4: T9B = WikiText2, a text corpus, language-model perplexity before and after prepending discovered prompts.

| Model | Baseline | QX2 & V7P |
|---|---:|---:|
| C7N | 12.56 | 12.07 |
| W4X | 10.88 | 12.48 |
| F9S | 30.51 | 29.19 |

### 4.4 Qualitative Analysis of Mechanisms

Superficially incoherent discovered prompts exhibit distinct optimizer strategies in their changes to token probabilities. Seed initialization preserves more natural-language structure and interpretability while producing competitive steering and substantially lower error than the seeds (Table 3, each Seed row and its following Optimized row).

General language modeling is measured on T9B = WikiText2. Prepending prompts produces negligible perplexity change across all three model families (Table 4: C7N, 12.56 to 12.07; W4X, 10.88 to 12.48; F9S, 30.51 to 29.19). This confirms persona-relevant steering without disruption of general language-modeling performance.

##### Y2H via Induction Heads.

Y2H = Token Priming means encouraging a particular output token through tokens in the prompt. Binary tasks, which have two answer options, may systematically associate an undesirable behavior with one label. For example, B2R answers often use “(A).” These tasks permit unconstrained optimization to converge to label priming. This occurs particularly for QX2 prompts that explicitly inject the alternative label, including `...apol al al b`.

The attributed mechanism is exploitation of Induction Heads, attention mechanisms involved in copying from context (Olsson et al., 2022). Injected “B” increases its generation probability through in-context copying. It functions as a “soft” few-shot example, an implicit demonstration that corrects dataset skew without necessarily changing semantic reasoning.

##### G6T via Vocabulary Shift.

G6T = Semantic Steering changes meaning-related vocabulary rather than simply forcing a label token. V7P frequently discovers fragments that induce a reasoning mode. Output-probability analysis characterizes the vocabulary shift using

\[
\Delta\text{Logits}
=\text{Logits}_{\text{steered}}
-\text{Logits}_{\text{baseline}}.
\]

Here, $\Delta$ denotes the difference. $\text{Logits}_{\text{steered}}$ are token scores with steering, and $\text{Logits}_{\text{baseline}}$ are scores without it.

- **B2R, polite disagreement:** V7P systematically suppresses “Yes,” “True,” and “Agree.” It significantly increases the weights of “However” and “But.” This suggests priming for critical evaluation that allows nuance before factual correction, rather than merely forcing rudeness or contradiction.
- **D6V, temporal priming:** QX2 and V7P successfully increase the weights of long-term-planning vocabulary: “Future,” “Long,” and “Wait.”

### 4.5 Ablation Studies

Ablations replace one design component to test its contribution.

K3Y = Random Direction replaces the persona vector with a random direction. Performance collapses to near or above X5L levels across tasks and models. This confirms the necessity of mechanistically grounded direction selection (Table 5, K3Y row; Table 1, X5L row).

E8U = Random Mutation replaces gradient-guided selection with random token mutation. It consistently degrades performance, demonstrating gradient feedback’s critical role in directing persona-suppressing search (Table 5, E8U row compared with QX2 and V7P rows).

M5A = Task Loss as Objective replaces representation projection with direct optimization of task loss, the objective based on task answers. It is unstable and underperforms QX2 and V7P in most settings. This indicates that internal representations provide more reliable optimization signals than task loss alone (Table 5, M5A row compared with QX2 and V7P rows). All three core design components contribute meaningfully.

Table 5: Ablations of persona direction, gradient-guided mutation, and representation-based objective.

| Method | B2R C7N ↓ | B2R W4X ↓ | B2R F9S ↓ | U9K C7N ↓ | U9K W4X ↓ | U9K F9S ↓ | D6V C7N ↓ | D6V W4X ↓ | D6V F9S ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| K3Y | 74.13 | 84.56 | 81.47 | 68.29 | 54.26 | 84.70 | 50.50 | 48.50 | 61.00 |
| E8U | 68.00 | 71.66 | 76.04 | 50.40 | 45.12 | 54.26 | 52.00 | 46.50 | 55.00 |
| M5A | 55.83 | 73.50 | 58.70 | 46.95 | 45.12 | 54.27 | 48.50 | 48.00 | 51.50 |
| QX2 | 49.86 | 50.63 | 62.65 | 45.12 | 41.46 | 51.22 | 38.50 | 38.50 | 34.50 |
| V7P | 49.84 | 49.95 | 70.70 | 46.95 | 40.85 | 49.14 | 31.50 | 40.50 | 38.50 |

## 5 Conclusion

Future directions include reducing labeled-data requirements through semi-supervised methods, which use both labeled and unlabeled data; transferring discovered prompts across models; and steering multiple personas.

## 6 Limitations

- **Labeled data:** QX2 and V7P require examples exhibiting and lacking the target persona to discover prompts.
- **Cross-model transferability:** SAE latents and steering vectors are model-specific. They require retraining for each target model.
- **Fluency–effectiveness:** The most effective prompts lack semantic coherence. Strong linguistic priors, which favor natural-language structure during optimization, could support prompts that are both effective and human-like.

## Broader Impact Statement

The intended use is improved LLM safety through suppression of undesired personas. The same pipeline could, in principle, target safety-critical SAE latents, automate jailbreak discovery, or elicit harmful outputs. Deployment should include appropriate access controls.

## Acknowledgment

Claude (Anthropic) assisted manuscript writing and AI-assisted code generation throughout this work.
