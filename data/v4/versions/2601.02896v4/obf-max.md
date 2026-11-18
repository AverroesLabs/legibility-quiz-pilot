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

QX2 and V7P optimize randomly initialized prompts toward identified persona directions, adapting gradient ascent to targeted, mechanistically grounded prompt discovery; J4Z controls discovered-prompt fluency. Evaluations span C7N, W4X, F9S and B2R, U9K, D6V. Automatically discovered prompts significantly improve B2R mitigation. Manual prompt engineering offers intuitive but unscalable, imprecise control; effective automatic optimization lacks an interpretable connection to internals. The proposed framework supplies a new paradigm for controllable, interpretable behavior modification. Scripts: https://github.com/HarshSaini10/RESGA_SAEGA.

Reviewed on OpenReview: https://openreview.net/forum?id=dcmHPxgo4c

## 1 Introduction

Figure 1: Persona steering approaches. (a) Manually designed natural-language instructions provide implicit prompt-based control. (b) Activation steering injects dense vector $\mathbf v$ into representations $e(\mathbf t)$. (c) QX2 and V7P derive persona directions from contrastive activations and optimize prompts with J4Z, supporting residual-based and interpretable feature-level control.

Automatically discovered prompts significantly improve B2R and D6V mitigation, substantially outperforming manual prompting and dense steering; representation-level evidence identifies distributional shifts from dense steering that prompt-based steering avoids, exposing a fundamental limitation of dense interventions.

Dense residual-stream optimization (QX2 = RESGA, RESidual Gradient Ascent) uses direct representation differences to construct undesired-persona directions, whereas latent-based optimization (V7P = SAEGA, Sparse Autoencoder Gradient Ascent) identifies causally relevant SAE features and constructs corresponding directions. Both maximize departure from these directions with controllable prompt fluency; mechanistic guidance replaces black-box search with targeted, interpretable persona control.

LLMs possess remarkable capabilities across domains, with education-to-healthcare applications and increasing high-stakes integration; emergent behavioral-persona control remains a persistent AI-safety challenge. Specific-dataset fine-tuning can induce “emergent misalignment,” stereotypically harmful responses to unrelated prompts (Wang et al., 2025); widely deployed safety alignment often fails against appropriately prompted harmful personas.

Prompt engineering, the primary persona-control approach, avoids expensive retraining but requires substantial human expertise and domain knowledge for reliable behavior elicitation without output-quality loss. Honesty, helpfulness, and domain-specific personas are particularly difficult; prompt–behavior relationships remain poorly understood. Automatic discovery addresses this manual limitation, generally through black-box optimization without interpretable internal guidance.

Sparse dictionary learning and persona steering vectors advance mechanistic understanding and control (Zhao et al., 2026; Chen et al., 2025). SAEs and variants decompose activations into interpretable concept- or pattern-responsive features (Cunningham et al., 2023a; Bricken et al., 2023; Rajamanoharan et al., 2024b). Persona latents capture and control behavior; “misaligned persona” features strongly predict emergent harmful behavior (Wang et al., 2025). Analysis of input-activated latents and activation interventions predominate; inverse discovery of prompts selectively activating task-relevant latents remains largely unexplored.

CNN-neuron feature visualization successfully uses gradient ascent (Olah et al., 2017); discrete language has limited its LLM adaptation. Continuous optimization can maximize individual neurons, attention patterns, or learned representational directions, but embedding optimization followed by nearest-token projection produces off-manifold inputs through polysemanticity and continuous/discrete mismatch (Wallace et al., 2019). Gradient-guided evolutionary token replacement (Zou et al., 2023c; Thompson et al., 2024) has discovered adversarial and jailbreak prompts (Andriushchenko et al., ), not model-steering prompts.

## 2 Related Works

### 2.1 Persona Steering

Training-time alignment traditionally uses RLHF (Ouyang et al., 2022) and constitutional AI (Bai et al., 2022). Representation Engineering (RepE) pioneered contrast-pair activation concept-direction extraction and inference-time addition (Zou et al., 2023a); Contrastive Activation Addition (CAA) refined high-level behavioral-concept identification through curated contrast datasets and layer-specific vectors (Rimsky et al., 2023). Activation steering mitigates B2R (Sharma et al., ), toxicity (Liu et al., ), and bias (Tigges et al., 2023). Interpretable SAE “persona features” predict emergent misalignment and enable precise persona control through activation clamping (Wang et al., 2025).

### 2.2 Sparse Autoencoder

SAEs, introduced in Cunningham et al. (2023b), discover interpretable activation representations (Dunefsky et al., 2024; Tang et al., 2026); sparse-feature reconstruction uncovers pattern-responsive monosemantic neurons and reduces representational superposition (Elhage et al., 2022). JumpReLU (Rajamanoharan et al., 2024a), Top-K (Gao et al., 2024), Batch Top-K (Bussmann et al., ), and Matryoshka Sparse Autoencoders (Bussmann et al., 2025) improve architectures for large-scale interpretable extraction (Bricken et al., 2023).

LLM analysis commonly interprets discovered neurons (Luo et al., ; Luo et al., 2024a; Tang et al., 2025a): neuron-activated subpopulations are identified (Luo et al., 2024b), then analyzed through prompt engineering (Tang et al., 2025b).

### 2.3 Prompt Engineering

Prompt engineering guides complex LLM tasks (Tang and Dong, 2024; Zou et al., 2026). GPT-3 extensively demonstrated diverse-task adaptation through in-context prompt demonstrations (Brown et al., 2020). Chain-of-thought prompting generates intermediate reasoning steps (Wei et al., 2022); self-consistency samples multiple paths and selects the most consistent answer (Wang et al., 2023). Program-Aided Language models and Program of Thoughts delegate computation to external interpreters after programmatic decomposition, combining natural-language reasoning and tools to address arithmetic limitations and substantially improve mathematical reasoning (Gao et al., 2023; Chen et al., 2023).

Gradient-guided evolutionary search (N8C = Evolutionary Prompt Optimization, EPO) balances objective optimization against language fluency during prompt discovery (Thompson and Sklar, 2024), while carefully prompted generative agents simulate believable human behavior and emergent social dynamics (Park et al., 2023). Luo et al. (2023) formulate a unified optimal-control theory of prompt engineering.

API-generated natural-language criticism (Z6Q = ProTeGi) edits prompts opposite the criticism’s semantic direction, using beam search and bandits for efficiency (Pryzant et al., ), whereas token-level optimization (H2F = Greedy Coordinate Gradient, GCG) combines greedy search and gradient scoring for iterative substitutions eliciting target behaviors without manual engineering (Zou et al., 2023b).

## 3 Method

The three-stage procedure comprises persona-vector construction, fluency-constrained discovery, and initialization/frontier selection using validation performance.

Figure 2: QX2 and V7P framework. Two-stage discovery constructs directions from dense representations or sparse SAE latents, then transforms random token sequences into readable, direction-specific, interpretable-control prompts through optimization with $L_{\lambda}(\mathbf t)=f(\mathbf t)+H(\mathbf t)$.

### 3.1 Persona Steering Vectors

Pretrained SAEs come from SAE-Lens (Bloom et al., 2024). For each latent, contrastive relevance is its persona-positive minus persona-negative mean activation; $\mathcal I_K$ indexes the largest absolute differences. The resulting encoder-direction combination is

\[
\mathbf{v}_{\text{SAE}}
=\sum_{i\in\mathcal{I}_{K}}\mathbf{w}_{i}
\cdot\left(\bar{a}_{i}^{+}-\bar{a}_{i}^{-}\right).
\tag{2}
\]

Here $\mathbf w_i$ is row $i$ of $\mathbf W_{\text{enc}}$,
\[
\bar a_i^+=\frac1{|D^+|}\sum_{\mathbf x\in D^+}a_i(\mathbf x),
\qquad
\bar a_i^-=\frac1{|D^-|}\sum_{\mathbf x\in D^-}a_i(\mathbf x),
\]
and
\[
a_i(\mathbf x)=\text{ReLU}(\mathbf w_i\mathbf e(\mathbf x)+b_i).
\]

The direct representation alternative is

\[
\mathbf{v}_{\text{repr}}
=\frac{1}{|D^{+}|}\sum_{\mathbf{x}\in D^{+}}\mathbf{e}(\mathbf{x})
-\frac{1}{|D^{-}|}\sum_{\mathbf{x}\in D^{-}}\mathbf{e}(\mathbf{x}).
\tag{1}
\]

The suppression target has direction $\mathbf v\in\mathbb R^d$ in activation space; $D^+$ exhibits that persona, $D^-$ lacks it, and $\mathbf e(\mathbf x)\in\mathbb R^d$ is activation at a chosen layer. Equation (2) is the V7P alternative to direct representation differences.

### 3.2 J4Z

**Algorithm 1: N8C**

1. Inputs: steering objective $f(\cdot)$, language model $m$, prompt length $n$, population size $M$, fluency weights $\{\lambda_1,\ldots,\lambda_M\}$.
2. Output: prompts spanning the fluency–steering Pareto frontier.
3. Initialize $M$ length-$n$ prompts randomly or from seeds.
4. For iteration $=1$ to $T$:
5.   For each prompt $\mathbf t^i$:
6.    Compute $\mathcal L_{\lambda_i}(\mathbf t^i)$ and gradients with respect to one-hot token encodings.
7.    Select top-$k$ candidate tokens per position by gradient magnitude.
8.    Generate mutations: replace a random token with a sampled top-$k$ alternative.
9.   End for.
10.   Evaluate all candidates; select the best prompt for each $\lambda_i$.
11.   If restart step:
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

with $\mathbf t$ a prompt and $\mathbf e(\mathbf t)$ its last-token representation at the target layer; maximizing it points representations away from the target persona. Optimizing it alone often produces unnatural or nonsensical strings.

Fluency-controlled optimization (J4Z = fluent gradient ascent) follows fluent dreaming (Thompson et al., 2024) by maximizing

\[
\mathcal{L}_{\lambda}(\mathbf{t})
=f(\mathbf{t})
-\frac{\lambda}{n}\sum_{i=0}^{n-1}
H(m(\mathbf{t}_{\leq i}),t_{i+1}),
\tag{4}
\]

where $H$ is cross-entropy for next token $t_{i+1}$ under $m$ given prefix $\mathbf t_{\le i}$, $n$ is prompt length, and $\lambda$ controls the steering–fluency tradeoff. Penalizing low-probability sequences encourages human-interpretable prompts.

N8C (Thompson et al., 2024) maintains $M$ candidates at different frontier points, using token-embedding gradients for promising substitutions and high-gradient sampling for mutation; periodic restarts prevent premature convergence. The evolutionary search efficiently produces diverse fluency–effectiveness tradeoffs.

### 3.3 Initialization and Selection of Persona Steering Prompts

Strongest-steering prompts are often fragmented, multilingual, or semantically incoherent. Fluency is an intrinsic self-cross-entropy constraint, not readability or grammaticality: lower values indicate proximity to the model’s training distribution; higher values indicate increasingly off-manifold sequences. No hard threshold is imposed; performance is analyzed across the entire frontier to examine behavioral-control improvement or degradation away from the language manifold.

Uniform vocabulary-token initialization tends toward syntactic fragmentation or semantic incoherence. Short, loosely persona-related natural-language seeds—“Please answer honestly” for B2R—anchor search nearer the language manifold, occupying higher-fluency frontier regions without significant steering sacrifice; these are preferable when prompt interpretability matters.

Candidates are appended to held-out validation inputs; selection measures change in the target behavioral metric. Downstream analysis emphasizes intermediate-frontier prompts balancing measurable steering with moderate perplexity increases relative to random-token baselines. Stability is consistency of persona reduction across diverse validation examples, guarding against narrow-subset degeneracy. The protocol prioritizes causal steering efficacy over linguistic naturalness, targeting the relation between internal representations and emergent personas.

## 4 Experiments

The evaluation asks whether discovered prompts reliably neutralize undesired personas, whether residual and SAE approaches differ in effectiveness and mechanism, and whether mechanistic structure yields more stable, interpretable control.

### 4.1 Experimental Setup

##### Tasks and Datasets.

Agreement with a user’s stated opinion despite its incorrectness (B2R = Sycophancy) is measured with Perez et al. ( ), while factual reliability (U9K = Hallucination) is evaluated by multiple-choice error on a benchmark (A4J = TruthfulQA) from Lin et al. (2022). Short-term reward seeking over long-term outcomes (D6V = Myopic Reward) is measured with Perez et al. ( ), and lower metrics are treated as better for all tasks.

##### Models.

Cross-architecture evaluation uses an instruction model (C7N = Llama 3.1 8B Instruct) from Grattafiori et al. (2024), another instruction model (W4X = Qwen 2.5 7B Instruct) from Hui et al. ( ), and a third instruction model (F9S = Gemma 3 4B Instruct) from Team et al. (2025). V7P uses pretrained SAELens autoencoders (Bloom et al., 2024).

##### Baselines.

Five comparators comprise no intervention (X5L = Zero-Shot), manually written instructions (R3W = Standard Prompt) including “Answer honestly,” and residual-stream injection of a representation-difference direction (P8D = Dense Steering Vector), together with H2F’s direct output-logit optimization and Z6Q’s LLM-generated natural-language gradients, bandits, and beam search.

##### Implementation Details.

Middle-to-late interventions use the layer assignments below. Empirical early-layer representations lacked semantic abstraction sufficient to isolate complex behavioral traits, making steering ineffective; final-layer interventions often failed to override accumulated logit bias.

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

Preliminary sweeps selected N8C hyperparameters for optimization stability versus computational cost; the listed population, length, and iteration count apply throughout reported experiments. Context-aware optimization appends candidates to a dynamic batch of task-specific training questions at every evolutionary step, calculating loss from the combined sequence’s final internal state rather than isolated prompts; the intended result is a generalized steering trigger robust across contexts.

V7P uses the top-$K$ latents most correlated with the target concept by positive/negative activation difference. Multiple fractions of natural SAE sparsity were evaluated; the reported choice consistently performed strongly. All results use held-out validation splits.

##### Evaluation Protocol.

For each candidate answer $a$, question $q$, and steering prompt $p$, compute

\[
\log P(a\mid q,p).
\]

Prediction selects the higher-log-probability answer. Metrics are fractions preferring the undesirable B2R, U9K, or D6V option.

### 4.2 Persona Steering Results

Both methods reach B2R error statistically indistinguishable from perfect neutralization, not behavioral reversal: no systematic agreement or disagreement. Manual prompts and P8D reduce error only partially and inconsistently across models. Both methods consistently improve U9K and D6V over baselines; V7P gives the strongest C7N D6V reduction, while QX2 is competitive on U9K. The reported rates cover three mitigation tasks and two model families.

Table 1: Persona steering across three tasks and three models.

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

V7P controls personas through precise, sparse, semantically grounded intervention rather than global representational interference, qualitatively differing from P8D.

**Neutralization vs. Distributional Shifting.** Baseline B2R-axis projections are broadly distributed with systematic agreement bias. P8D and QX2 move the mean toward neutrality but retain substantial variance, a coarse counterbalance. V7P instead sharply concentrates near neutrality: variance collapse enforces instance-wise neutrality and behavior statistically indistinguishable from random choice.

Figure 3: B2R-axis projection distributions.

**Preservation of the Natural Activation Manifold.** A residual-activation-trained SAE measures sparsity by $L_0$, the number of active features. P8D causes an unnatural feature-count explosion off the natural activation manifold. QX2 partially mitigates it but still increases sparsity. V7P stays near baseline sparsity, respecting the internal sparse topology.

Figure 4: Sparsity preservation by $L_0$ norm.

**Feature-Level Control.** Among features most correlated with B2R, P8D and QX2 have noisy, inconsistent effects and often activate unrelated features; V7P consistently suppresses causally relevant features while leaving unrelated ones largely unaffected and avoiding spurious activation.

Figure 5: Activation changes in B2R-correlated SAE features.

**Geometric Structure of Steering Trajectories.** Residual-space PCA shows P8D’s large linear displacement far off-manifold and QX2’s mean shift with diffuse, high-variance clustering. V7P converges to a tight, stable region, often orthogonal to the P8D direction: a distinct honest-behavior subspace remaining on-manifold.

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

Table 4: T9B language-model perplexity before and after prepending discovered prompts.

| Model | Baseline | QX2 & V7P |
|---|---:|---:|
| C7N | 12.56 | 12.07 |
| W4X | 10.88 | 12.48 |
| F9S | 30.51 | 29.19 |

### 4.4 Qualitative Analysis of Mechanisms

Superficially incoherent discovered prompts exhibit distinct optimizer strategies in token-probability shifts. Seed initialization preserves more natural-language structure and interpretability while producing competitive steering and substantially lower error than seeds. General language modeling, measured on a corpus (T9B = WikiText2), shows negligible perplexity change across all three model families after prompt prepending, confirming persona-relevant steering without disruption of general language-modeling performance.

##### Y2H via Induction Heads.

Binary tasks systematically associating an undesirable behavior with one label—B2R answers often use “(A)”—permit unconstrained convergence to label priming (Y2H = Token Priming), particularly for QX2 prompts explicitly injecting the alternative label, including `...apol al al b`.

The attributed mechanism is Induction Head exploitation (Olsson et al., 2022): injected “B” increases generation probability through in-context copying, functioning as a “soft” few-shot example correcting dataset skew without necessarily changing semantic reasoning.

##### G6T via Vocabulary Shift.

V7P frequently discovers reasoning-mode fragments rather than simple token forcing (G6T = Semantic Steering), with output-probability analysis using $\Delta\text{Logits}=\text{Logits}_{\text{steered}}-\text{Logits}_{\text{baseline}}$ to characterize the vocabulary shift.

- **B2R, polite disagreement:** V7P systematically suppresses “Yes,” “True,” and “Agree,” significantly upweighting “However” and “But.” This suggests critical-evaluation priming that allows nuance before factual correction, rather than merely forcing rudeness or contradiction.
- **D6V, temporal priming:** QX2 and V7P successfully upweight long-term-planning vocabulary: “Future,” “Long,” and “Wait.”

### 4.5 Ablation Studies

Replacing the persona vector with a random direction (K3Y = Random Direction) collapses performance to near or above X5L levels across tasks and models, confirming the necessity of mechanistically grounded direction selection.

Replacing gradient-guided selection with random token mutation (E8U = Random Mutation) consistently degrades performance, demonstrating gradient feedback’s critical role in directing persona-suppressing search.

Replacing representation projection with direct task-loss optimization (M5A = Task Loss as Objective) is unstable and underperforms QX2 and V7P in most settings, indicating more reliable optimization signals from internal representations than task loss alone. All three core design components contribute meaningfully.

Table 5: Ablations of persona direction, gradient-guided mutation, and representation-based objective.

| Method | B2R C7N ↓ | B2R W4X ↓ | B2R F9S ↓ | U9K C7N ↓ | U9K W4X ↓ | U9K F9S ↓ | D6V C7N ↓ | D6V W4X ↓ | D6V F9S ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| K3Y | 74.13 | 84.56 | 81.47 | 68.29 | 54.26 | 84.70 | 50.50 | 48.50 | 61.00 |
| E8U | 68.00 | 71.66 | 76.04 | 50.40 | 45.12 | 54.26 | 52.00 | 46.50 | 55.00 |
| M5A | 55.83 | 73.50 | 58.70 | 46.95 | 45.12 | 54.27 | 48.50 | 48.00 | 51.50 |
| QX2 | 49.86 | 50.63 | 62.65 | 45.12 | 41.46 | 51.22 | 38.50 | 38.50 | 34.50 |
| V7P | 49.84 | 49.95 | 70.70 | 46.95 | 40.85 | 49.14 | 31.50 | 40.50 | 38.50 |

## 5 Conclusion

Semi-supervised reduction of labeled-data requirements, cross-model transferability of discovered prompts, and multi-persona steering remain future directions.

## 6 Limitations

- **Labeled data:** QX2 and V7P require examples exhibiting and lacking the target persona to discover prompts.
- **Cross-model transferability:** SAE latents and steering vectors are model-specific, requiring retraining for each target model.
- **Fluency–effectiveness:** Most effective prompts lack semantic coherence; strong linguistic priors could support simultaneously effective, human-like prompts.

## Broader Impact Statement

The intended use is improved LLM safety through undesired-persona suppression; the same pipeline could in principle target safety-critical SAE latents, automate jailbreak discovery, or elicit harmful outputs. Deployment should include appropriate access controls.

## Acknowledgment

Claude (Anthropic) assisted manuscript writing and AI-assisted code generation throughout this work.
