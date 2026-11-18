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

QX2 (RESGA, or RESidual Gradient Ascent) and V7P (SAEGA, or Sparse Autoencoder Gradient Ascent) optimize randomly initialized prompts toward identified persona directions. A prompt is input text, and a persona is a pattern of model behavior. A persona direction is a direction in the model’s internal representations linked to that behavior. Both methods adapt gradient ascent—iterative changes that increase an objective—to targeted prompt discovery grounded in model mechanisms. V7P uses a sparse autoencoder (SAE), which reconstructs internal model values through a small set of active features. J4Z (fluent gradient ascent) controls the fluency of discovered prompts.

Evaluations cover C7N (Llama 3.1 8B Instruct), W4X (Qwen 2.5 7B Instruct), and F9S (Gemma 3 4B Instruct). They test B2R (Sycophancy, or agreement with an incorrect user opinion), U9K (Hallucination, assessed through factual reliability), and D6V (Myopic Reward, or seeking short-term rewards over long-term outcomes). Automatically discovered prompts significantly improve B2R mitigation.

Manual prompt engineering provides intuitive control, but it is imprecise and does not scale. Effective automatic optimization lacks an interpretable link to model internals. The proposed framework offers a new paradigm for controllable, interpretable behavior change. Scripts: https://github.com/HarshSaini10/RESGA_SAEGA.

Reviewed on OpenReview: https://openreview.net/forum?id=dcmHPxgo4c

## 1 Introduction

Figure 1: Persona steering approaches. (a) Manually designed natural-language instructions provide implicit prompt-based control. (b) Activation steering changes internal model values by injecting a dense vector $\mathbf v$ into representations $e(\mathbf t)$. A dense vector can use many representation dimensions. (c) QX2 and V7P derive persona directions from contrastive activations—internal values compared across contrasting examples—and optimize prompts with J4Z. They support control through the residual stream, the model’s running internal representation, and through interpretable features.

Automatically discovered prompts significantly improve B2R and D6V mitigation. They substantially outperform manual prompting and dense steering. Evidence from internal representations identifies distributional shifts caused by dense steering that prompt-based steering avoids. This exposes a fundamental limitation of dense interventions.

QX2 uses dense residual-stream optimization. It constructs undesired-persona directions directly from representation differences. V7P instead uses latents, the internal features learned by an SAE. It identifies causally relevant SAE features and constructs directions from them. Both methods maximize departure from these directions while allowing control over prompt fluency. Guidance from model mechanisms replaces black-box search, which uses outcomes without interpretable internal guidance, with targeted, interpretable persona control.

Large language models (LLMs) show remarkable abilities across domains. Their uses range from education to healthcare, and their integration into high-stakes settings continues to grow. Controlling emergent behavioral personas remains a persistent artificial intelligence (AI) safety challenge. Fine-tuning, or further training, on a specific dataset can induce “emergent misalignment”: stereotypically harmful responses to unrelated prompts (Wang et al., 2025). Widely deployed safety alignment often fails against harmful personas elicited by suitable prompts.

Prompt engineering is the main approach to persona control. It avoids expensive retraining, but reliably eliciting behavior without reducing output quality requires substantial human expertise and domain knowledge. Honesty, helpfulness, and domain-specific personas are particularly difficult to control. The links between prompts and behavior remain poorly understood. Automatic discovery addresses the limits of manual prompting, but it generally uses black-box optimization without interpretable internal guidance.

Sparse dictionary learning, which represents activations through a small set of learned components, and persona steering vectors advance mechanistic understanding and control (Zhao et al., 2026; Chen et al., 2025). SAEs and their variants break activations into interpretable features that respond to concepts or patterns (Cunningham et al., 2023a; Bricken et al., 2023; Rajamanoharan et al., 2024b). Persona latents capture and control behavior. “Misaligned persona” features strongly predict emergent harmful behavior (Wang et al., 2025). Most work analyzes latents activated by inputs or directly intervenes on activations. The inverse problem—finding prompts that selectively activate task-relevant latents—remains largely unexplored.

Feature visualization for convolutional neural network (CNN) neurons successfully uses gradient ascent to find inputs that activate neurons (Olah et al., 2017). Discrete language, which consists of separate tokens or text units, has limited its adaptation to LLMs. Continuous optimization can maximize individual neurons, attention patterns, or learned representation directions. However, optimizing embeddings—continuous vector representations of tokens—and then projecting them to the nearest tokens produces off-manifold inputs, outside the model’s usual input distribution. This results from polysemanticity, where one component represents several meanings, and the mismatch between continuous vectors and discrete tokens (Wallace et al., 2019). Gradient-guided evolutionary token replacement uses derivatives to guide candidate changes and selection. It has discovered adversarial and jailbreak prompts, which induce unwanted behavior or bypass safeguards, rather than model-steering prompts (Zou et al., 2023c; Thompson et al., 2024; Andriushchenko et al., ).

## 2 Related Works

### 2.1 Persona Steering

Training-time alignment traditionally uses reinforcement learning from human feedback (RLHF), which trains from human evaluations (Ouyang et al., 2022), and constitutional AI, which guides behavior through stated principles (Bai et al., 2022). Representation Engineering (RepE) pioneered extracting concept directions from activation differences between contrasting example pairs and adding those directions during inference, when the model produces answers (Zou et al., 2023a). Contrastive Activation Addition (CAA) refined the identification of high-level behavioral concepts through curated contrast datasets and layer-specific vectors (Rimsky et al., 2023). Activation steering mitigates B2R (Sharma et al., ), toxicity (Liu et al., ), and bias (Tigges et al., 2023). Interpretable SAE “persona features” predict emergent misalignment. They also enable precise persona control through activation clamping, which holds activations at specified values (Wang et al., 2025).

### 2.2 Sparse Autoencoder

SAEs, introduced in Cunningham et al. (2023b), discover interpretable representations of activations (Dunefsky et al., 2024; Tang et al., 2026). Reconstruction with sparse features uncovers pattern-responsive monosemantic neurons, which respond to a single meaning. It also reduces representational superposition, where multiple features share representation dimensions (Elhage et al., 2022). JumpReLU (Rajamanoharan et al., 2024a), Top-K (Gao et al., 2024), Batch Top-K (Bussmann et al., ), and Matryoshka Sparse Autoencoders (Bussmann et al., 2025) improve SAE architectures for large-scale extraction of interpretable features (Bricken et al., 2023).

LLM analysis commonly interprets discovered neurons (Luo et al., ; Luo et al., 2024a; Tang et al., 2025a). Researchers identify subpopulations activated by a neuron (Luo et al., 2024b), then analyze them through prompt engineering (Tang et al., 2025b).

### 2.3 Prompt Engineering

Prompt engineering guides complex LLM tasks (Tang and Dong, 2024; Zou et al., 2026). GPT-3 extensively demonstrated adaptation to diverse tasks through in-context demonstrations: examples included in the prompt (Brown et al., 2020). Chain-of-thought prompting generates intermediate reasoning steps (Wei et al., 2022). Self-consistency samples multiple reasoning paths and selects the most consistent answer (Wang et al., 2023). Program-Aided Language models and Program of Thoughts break problems into programmatic steps and delegate computation to external interpreters. They combine natural-language reasoning with tools to address arithmetic limitations and substantially improve mathematical reasoning (Gao et al., 2023; Chen et al., 2023).

Gradient-guided evolutionary search, N8C (Evolutionary Prompt Optimization, or EPO), balances objective optimization against language fluency during prompt discovery (Thompson and Sklar, 2024). Carefully prompted generative agents simulate believable human behavior and emergent social dynamics (Park et al., 2023). Luo et al. (2023) formulate a unified optimal-control theory of prompt engineering.

Z6Q (ProTeGi) uses natural-language criticism generated through an application programming interface (API). It edits prompts opposite to the criticism’s semantic direction. For efficiency, it uses beam search, which retains several promising candidates, and bandits, which allocate search effort among alternatives (Pryzant et al., ). H2F (Greedy Coordinate Gradient, or GCG) instead optimizes individual tokens. It combines greedy search and gradient scoring to make iterative substitutions that elicit target behaviors without manual engineering (Zou et al., 2023b).

## 3 Method

The procedure has three stages: construct persona vectors, discover prompts under a fluency constraint, and choose initializations and points on the fluency–steering frontier using validation performance.

Figure 2: QX2 and V7P framework. Two-stage discovery first constructs directions from dense representations or sparse SAE latents. Optimization with $L_{\lambda}(\mathbf t)=f(\mathbf t)+H(\mathbf t)$ then transforms random token sequences into readable, direction-specific prompts for interpretable control.

### 3.1 Persona Steering Vectors

We use pretrained SAEs from SAE-Lens (Bloom et al., 2024). For each latent, contrastive relevance is its mean activation on persona-positive examples minus its mean activation on persona-negative examples. The index set $\mathcal I_K$ selects the $K$ largest absolute differences. Combining the corresponding encoder directions gives

\[
\mathbf{v}_{\text{SAE}}
=\sum_{i\in\mathcal{I}_{K}}\mathbf{w}_{i}
\cdot\left(\bar{a}_{i}^{+}-\bar{a}_{i}^{-}\right).
\tag{2}
\]

Here, $\mathbf w_i$ is row $i$ of the encoder matrix $\mathbf W_{\text{enc}}$. The two mean activations are

\[
\bar a_i^+=\frac1{|D^+|}\sum_{\mathbf x\in D^+}a_i(\mathbf x),
\qquad
\bar a_i^-=\frac1{|D^-|}\sum_{\mathbf x\in D^-}a_i(\mathbf x),
\]

and the activation of latent $i$ is

\[
a_i(\mathbf x)=\text{ReLU}(\mathbf w_i\mathbf e(\mathbf x)+b_i).
\]

Here, $b_i$ is the latent’s bias. ReLU, the rectified linear unit, sets negative inputs to zero and leaves nonnegative inputs unchanged.

The direct representation alternative is

\[
\mathbf{v}_{\text{repr}}
=\frac{1}{|D^{+}|}\sum_{\mathbf{x}\in D^{+}}\mathbf{e}(\mathbf{x})
-\frac{1}{|D^{-}|}\sum_{\mathbf{x}\in D^{-}}\mathbf{e}(\mathbf{x}).
\tag{1}
\]

The persona to suppress has direction $\mathbf v\in\mathbb R^d$ in a $d$-dimensional activation space. The dataset $D^+$ exhibits that persona, while $D^-$ lacks it. The notation $|D|$ gives the number of examples in a dataset. The vector $\mathbf e(\mathbf x)\in\mathbb R^d$ is the activation for input $\mathbf x$ at a chosen layer. Equation (2) is V7P’s alternative to direct representation differences.

### 3.2 J4Z

**Algorithm 1: N8C**

1. Inputs: steering objective $f(\cdot)$, language model $m$, prompt length $n$, population size $M$, fluency weights $\{\lambda_1,\ldots,\lambda_M\}$.
2. Output: prompts spanning the fluency–steering Pareto frontier, where improving one objective requires worsening the other.
3. Initialize $M$ length-$n$ prompts randomly or from seeds.
4. For iteration $=1$ to $T$:
5.   For each prompt $\mathbf t^i$:
6.    Compute $\mathcal L_{\lambda_i}(\mathbf t^i)$ and gradients with respect to one-hot token encodings, which mark each token with one selected vector entry.
7.    Select the top-$k$ candidate tokens per position by gradient magnitude.
8.    Generate mutations: replace a random token with an alternative sampled from the top-$k$ candidates.
9.   End for.
10.   Evaluate all candidates; select the best prompt for each $\lambda_i$.
11.   If this is a restart step:
12.    Retain the best prompt under a random $\lambda$; reinitialize the others.
13.   End if.
14. End for.

The persona-reduction objective is the negative scaled projection

\[
f(\mathbf{t})
=-\frac{\langle\mathbf{e}(\mathbf{t}),\mathbf{v}\rangle}
{\|\mathbf{e}(\mathbf{t})\|},
\tag{3}
\]

where $\mathbf t$ is a prompt and $\mathbf e(\mathbf t)$ is its last-token representation at the target layer. The inner product in the numerator measures alignment with the persona direction. The denominator is the representation’s norm, or length. Maximizing this objective points representations away from the target persona. Optimizing it alone often produces unnatural or nonsensical strings.

J4Z controls fluency by following fluent dreaming (Thompson et al., 2024) and maximizing

\[
\mathcal{L}_{\lambda}(\mathbf{t})
=f(\mathbf{t})
-\frac{\lambda}{n}\sum_{i=0}^{n-1}
H(m(\mathbf{t}_{\leq i}),t_{i+1}),
\tag{4}
\]

where $H$ is cross-entropy, a loss that penalizes low probability for the next token $t_{i+1}$. The model $m$ predicts that token from the prefix $\mathbf t_{\le i}$. Prompt length is $n$, and $\lambda$ controls the tradeoff between steering and fluency. Penalizing low-probability sequences encourages human-interpretable prompts.

N8C (Thompson et al., 2024) maintains $M$ candidates at different frontier points. It uses token-embedding gradients to identify promising substitutions and samples high-gradient alternatives for mutation. Periodic restarts prevent premature convergence. This evolutionary search efficiently produces diverse tradeoffs between fluency and effectiveness.

### 3.3 Initialization and Selection of Persona Steering Prompts

The strongest-steering prompts are often fragmented, multilingual, or semantically incoherent. Here, fluency is an intrinsic self-cross-entropy constraint, not a measure of readability or grammar. It measures how well the model predicts a prompt’s own tokens. Lower values indicate greater proximity to the model’s training distribution. Higher values indicate sequences increasingly outside the language manifold, the region occupied by typical language inputs. We impose no hard threshold. We analyze performance across the entire frontier to examine whether behavioral control improves or degrades away from that manifold.

Initializing tokens uniformly from the vocabulary tends to produce syntactic fragmentation or semantic incoherence. Short natural-language seeds loosely related to the persona—such as “Please answer honestly” for B2R—anchor the search nearer the language manifold. They occupy higher-fluency frontier regions without a significant loss of steering. They are preferable when prompt interpretability matters.

We append candidates to held-out validation inputs and select them by the change in the target behavioral metric. Downstream analysis emphasizes intermediate-frontier prompts. These balance measurable steering with moderate increases in perplexity, a measure of token unpredictability, relative to random-token baselines. Stability means consistent persona reduction across diverse validation examples. It guards against degenerate behavior limited to a narrow subset. The protocol prioritizes causal steering efficacy over linguistic naturalness. It targets the relationship between internal representations and emergent personas.

## 4 Experiments

The evaluation asks whether discovered prompts reliably neutralize undesired personas. It also asks whether residual and SAE approaches differ in effectiveness and mechanism, and whether mechanistic structure yields more stable, interpretable control.

### 4.1 Experimental Setup

##### Tasks and Datasets.

B2R measures agreement with a user’s stated opinion despite that opinion being incorrect, using Perez et al. ( ). U9K measures factual reliability through multiple-choice error on A4J (TruthfulQA), a benchmark from Lin et al. (2022). D6V measures short-term reward seeking over long-term outcomes, using Perez et al. ( ). We treat lower metrics as better for all tasks.

##### Models.

Cross-architecture evaluation uses three instruction models: C7N from Grattafiori et al. (2024), W4X from Hui et al. ( ), and F9S from Team et al. (2025). V7P uses pretrained SAELens autoencoders (Bloom et al., 2024).

##### Baselines.

We use five comparators:

- X5L (Zero-Shot): no intervention.
- R3W (Standard Prompt): manually written instructions, including “Answer honestly.”
- P8D (Dense Steering Vector): injection of a representation-difference direction into the residual stream.
- H2F: direct optimization of output logits, the model’s raw output scores.
- Z6Q: LLM-generated natural-language gradients, bandits, and beam search.

##### Implementation Details.

We intervene in middle-to-late layers, using the assignments below. In empirical tests, early-layer representations lacked enough semantic abstraction to isolate complex behavioral traits, making steering ineffective. Final-layer interventions often failed to override accumulated logit bias. In the table, $L_0$ denotes the number of active SAE features, a measure of sparsity.

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

Preliminary sweeps selected N8C hyperparameters, the settings that govern optimization, to balance stability against computational cost. The listed population size, prompt length, and iteration count apply throughout the reported experiments.

Context-aware optimization appends candidates to a changing batch of task-specific training questions at every evolutionary step. It calculates loss from the combined sequence’s final internal state rather than from isolated prompts. The intended result is a generalized steering trigger that remains robust across contexts.

V7P uses the top-$K$ latents most correlated with the target concept, measured by positive/negative activation differences. We evaluated multiple fractions of natural SAE sparsity. The reported choice consistently performed strongly. All results use held-out validation splits.

##### Evaluation Protocol.

For each candidate answer $a$, question $q$, and steering prompt $p$, compute

\[
\log P(a\mid q,p).
\]

This is the logarithm of the answer’s probability given the question and prompt. Prediction selects the answer with the higher log probability. Metrics are the fractions of predictions that prefer the undesirable B2R, U9K, or D6V option.

### 4.2 Persona Steering Results

Both methods reach B2R error statistically indistinguishable from perfect neutralization, rather than behavioral reversal. This means no systematic agreement or disagreement. Manual prompts and P8D reduce error only partially and inconsistently across models. Both methods consistently improve U9K and D6V over baselines. V7P gives the strongest C7N D6V reduction, while QX2 is competitive on U9K. The reported rates cover three mitigation tasks and two model families.

Table 1: Persona steering across three tasks and three models. Downward arrows indicate that lower values are better; “Avg.” denotes the average.

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

V7P controls personas through precise, sparse, semantically grounded intervention rather than global interference with representations. Its mechanism differs qualitatively from P8D.

**Neutralization vs. Distributional Shifting.** Baseline projections onto the B2R axis are broadly distributed and show systematic agreement bias. P8D and QX2 move the mean toward neutrality but retain substantial variance, providing a coarse counterbalance. V7P instead sharply concentrates projections near neutrality. This collapse in variance enforces neutrality for individual instances and produces behavior statistically indistinguishable from random choice.

Figure 3: B2R-axis projection distributions.

**Preservation of the Natural Activation Manifold.** An SAE trained on residual activations measures sparsity through $L_0$, the number of active features. P8D causes an unnatural explosion in feature count, moving activations off the natural activation manifold—the region occupied by typical internal activations. QX2 partly mitigates this effect but still increases sparsity. V7P stays near baseline sparsity and respects the internal sparse topology, or structure of sparse feature activity.

Figure 4: Sparsity preservation by $L_0$ norm.

**Feature-Level Control.** Among the features most correlated with B2R, P8D and QX2 produce noisy, inconsistent effects and often activate unrelated features. V7P consistently suppresses causally relevant features. It leaves unrelated features largely unaffected and avoids spurious activation.

Figure 5: Activation changes in B2R-correlated SAE features.

**Geometric Structure of Steering Trajectories.** Principal component analysis (PCA), which summarizes variation along a smaller set of axes, shows P8D’s large linear displacement far off-manifold in residual space. It shows QX2’s mean shift with diffuse, high-variance clustering. V7P converges to a tight, stable region, often orthogonal, or perpendicular, to the P8D direction. This is a distinct honest-behavior subspace that remains on-manifold.

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

Table 4: T9B (WikiText2) language-model perplexity before and after prepending discovered prompts.

| Model | Baseline | QX2 & V7P |
|---|---:|---:|
| C7N | 12.56 | 12.07 |
| W4X | 10.88 | 12.48 |
| F9S | 30.51 | 29.19 |

### 4.4 Qualitative Analysis of Mechanisms

Token-probability shifts reveal distinct optimizer strategies in discovered prompts that appear incoherent. Seed initialization preserves more natural-language structure and interpretability. It also produces competitive steering and substantially lower error than the seeds. General language modeling on the T9B corpus shows negligible perplexity changes across all three model families after prepending prompts. This confirms persona-relevant steering without disrupting general language-modeling performance.

##### Y2H via Induction Heads.

Binary tasks that systematically associate an undesirable behavior with one label permit unconstrained convergence to label priming, Y2H (Token Priming). For example, B2R answers often use “(A).” This effect occurs particularly with QX2 prompts that explicitly insert the alternative label, including `...apol al al b`.

The attributed mechanism exploits Induction Heads, model components that copy patterns from context (Olsson et al., 2022). Injected “B” increases its generation probability through in-context copying. It functions as a “soft” few-shot example, a prompt-based demonstration that corrects dataset skew without necessarily changing semantic reasoning.

##### G6T via Vocabulary Shift.

V7P frequently discovers reasoning-mode fragments rather than simple token forcing. We call this G6T (Semantic Steering). Output-probability analysis characterizes the vocabulary shift through changes in logits:

\[
\Delta\text{Logits}=\text{Logits}_{\text{steered}}-\text{Logits}_{\text{baseline}}.
\]

- **B2R, polite disagreement:** V7P systematically suppresses “Yes,” “True,” and “Agree,” while significantly increasing the weights of “However” and “But.” This suggests priming for critical evaluation that allows nuance before factual correction, rather than merely forcing rudeness or contradiction.
- **D6V, temporal priming:** QX2 and V7P successfully increase the weights of long-term-planning vocabulary: “Future,” “Long,” and “Wait.”

### 4.5 Ablation Studies

Ablation studies replace individual design components to test their contribution. Replacing the persona vector with a random direction, K3Y (Random Direction), collapses performance to near or above X5L levels across tasks and models. This confirms the need to select directions grounded in model mechanisms.

Replacing gradient-guided selection with random token mutation, E8U (Random Mutation), consistently degrades performance. This demonstrates the critical role of gradient feedback in directing the search toward persona suppression.

Replacing representation projection with direct task-loss optimization, M5A (Task Loss as Objective), is unstable. It underperforms QX2 and V7P in most settings. This indicates that internal representations provide more reliable optimization signals than task loss alone. All three core design components contribute meaningfully.

Table 5: Ablations of persona direction, gradient-guided mutation, and representation-based objective.

| Method | B2R C7N ↓ | B2R W4X ↓ | B2R F9S ↓ | U9K C7N ↓ | U9K W4X ↓ | U9K F9S ↓ | D6V C7N ↓ | D6V W4X ↓ | D6V F9S ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| K3Y | 74.13 | 84.56 | 81.47 | 68.29 | 54.26 | 84.70 | 50.50 | 48.50 | 61.00 |
| E8U | 68.00 | 71.66 | 76.04 | 50.40 | 45.12 | 54.26 | 52.00 | 46.50 | 55.00 |
| M5A | 55.83 | 73.50 | 58.70 | 46.95 | 45.12 | 54.27 | 48.50 | 48.00 | 51.50 |
| QX2 | 49.86 | 50.63 | 62.65 | 45.12 | 41.46 | 51.22 | 38.50 | 38.50 | 34.50 |
| V7P | 49.84 | 49.95 | 70.70 | 46.95 | 40.85 | 49.14 | 31.50 | 40.50 | 38.50 |

## 5 Conclusion

Future directions include semi-supervised methods, which combine labeled and unlabeled data, to reduce labeled-data requirements. Other directions include transferring discovered prompts across models and steering multiple personas.

## 6 Limitations

- **Labeled data:** QX2 and V7P require examples that exhibit and lack the target persona to discover prompts.
- **Cross-model transferability:** SAE latents and steering vectors are model-specific, so each target model requires retraining.
- **Fluency–effectiveness:** The most effective prompts lack semantic coherence. Strong linguistic priors, which favor natural-language structure during search, could support prompts that are both effective and human-like.

## Broader Impact Statement

The intended use is to improve LLM safety by suppressing undesired personas. In principle, the same pipeline could target safety-critical SAE latents, automate jailbreak discovery, or elicit harmful outputs. Deployment should include appropriate access controls.

## Acknowledgment

Claude (Anthropic) assisted with manuscript writing and AI-assisted code generation throughout this work.
