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

QX2, also called RESGA (RESidual Gradient Ascent), and V7P, also called SAEGA (Sparse Autoencoder Gradient Ascent), optimize randomly initialized prompts toward identified persona directions. A persona is a recurring behavioral tendency. A persona direction is a numerical direction in the model’s internal representations associated with that tendency. Gradient ascent uses derivatives to search for inputs that increase an objective. QX2 uses dense internal representations. V7P uses sparse autoencoders (SAEs), which reconstruct internal representations using a small set of active, interpretable features. Both methods adapt gradient ascent to targeted, mechanistically grounded prompt discovery: internal model measurements guide the search. J4Z, or fluent gradient ascent, controls discovered-prompt fluency.

Evaluations span three instruction-following models: C7N (Llama 3.1 8B Instruct), W4X (Qwen 2.5 7B Instruct), and F9S (Gemma 3 4B Instruct). Here, B in a model name denotes billions of parameters. The tasks are B2R (Sycophancy: agreeing with a user’s incorrect opinion), U9K (Hallucination: factual unreliability), and D6V (Myopic Reward: preferring short-term rewards over long-term outcomes). Automatically discovered prompts significantly improve B2R mitigation; Table 1 reports the QX2 and V7P results alongside the comparison methods.

Manual prompt engineering offers intuitive but unscalable, imprecise control. Effective automatic optimization lacks an interpretable connection to model internals. The proposed framework supplies a new paradigm for controllable, interpretable behavior modification. Scripts: https://github.com/HarshSaini10/RESGA_SAEGA.

Reviewed on OpenReview: https://openreview.net/forum?id=dcmHPxgo4c

## 1 Introduction

A prompt is a sequence of tokens, the text units processed by a model. In Figure 1, \(\mathbf t\) denotes that sequence, \(e(\mathbf t)\) its internal numerical representation, and \(\mathbf v\) a steering vector: a direction used to change the representation. A dense vector uses the full representation space rather than selecting a small set of sparse features.

Figure 1: Persona steering approaches. (a) Manually designed natural-language instructions provide implicit prompt-based control. (b) Activation steering injects dense vector $\mathbf v$ into representations $e(\mathbf t)$. (c) QX2 and V7P derive persona directions from contrastive activations and optimize prompts with J4Z, supporting residual-based and interpretable feature-level control.

Automatically discovered prompts significantly improve B2R and D6V mitigation, substantially outperforming manual prompting and dense steering. Table 1 gives the comparisons in the QX2 and V7P rows against the manual-instruction and dense-steering rows. Representation-level evidence identifies distributional shifts from dense steering that prompt-based steering avoids, exposing a fundamental limitation of dense interventions. Sections 4.3 and Figures 3–6 present this evidence.

QX2 performs dense residual-stream optimization. The residual stream is the internal representation carried forward and updated through model layers. QX2 constructs an undesired-persona direction from direct differences between representations of examples that exhibit and lack the persona. V7P performs latent-based optimization. A latent is a learned SAE feature. V7P identifies causally relevant SAE features and constructs corresponding directions. Both methods maximize departure from these directions while controlling prompt fluency. Mechanistic guidance replaces black-box search—search without interpretable internal guidance—with targeted, interpretable persona control.

Large language models (LLMs) possess remarkable capabilities across domains, with applications from education to healthcare and increasing integration into high-stakes settings. Control of emergent behavioral personas remains a persistent artificial intelligence (AI) safety challenge. Fine-tuning, or further training on a specific dataset, can induce “emergent misalignment”: stereotypically harmful responses to unrelated prompts (Wang et al., 2025). Widely deployed safety alignment often fails against appropriately prompted harmful personas.

Prompt engineering guides behavior through input instructions rather than expensive retraining. It is the primary persona-control approach, but reliable behavior elicitation without output-quality loss requires substantial human expertise and domain knowledge. Honesty, or truthful responding; helpfulness, or useful assistance; and domain-specific personas are particularly difficult to control. Prompt–behavior relationships remain poorly understood. Automatic discovery addresses this manual limitation, but generally uses black-box optimization without interpretable internal guidance.

Sparse dictionary learning represents activations using a small selection of learned components. This approach and persona steering vectors advance mechanistic understanding and control (Zhao et al., 2026; Chen et al., 2025). SAEs and their variants decompose activations—the model’s internal numerical responses—into interpretable features responsive to concepts or patterns (Cunningham et al., 2023a; Bricken et al., 2023; Rajamanoharan et al., 2024b). Persona latents capture and control behavior. “Misaligned persona” features strongly predict emergent harmful behavior (Wang et al., 2025). Existing work predominantly analyzes latents activated by inputs or intervenes directly on activations. The inverse problem—discovering prompts that selectively activate task-relevant latents—remains largely unexplored.

Feature visualization for convolutional neural network (CNN) neurons successfully uses gradient ascent to find inputs that increase neuron activation (Olah et al., 2017). Discrete language has limited adaptation of this approach to LLMs. Continuous optimization can maximize individual neurons, attention patterns, or learned representational directions. Attention patterns describe how model processing relates tokens to one another. However, optimizing continuous token embeddings and then projecting them to the nearest tokens can produce off-manifold inputs: inputs outside the model’s usual language patterns. Token embeddings are numerical token representations; nearest-token projection replaces an optimized representation with its closest available token. Polysemanticity, where one component responds to multiple meanings, and the mismatch between continuous representations and discrete tokens contribute to this problem (Wallace et al., 2019). Gradient-guided evolutionary token replacement has discovered adversarial and jailbreak prompts—prompts that elicit unwanted behavior or bypass safeguards—not model-steering prompts (Zou et al., 2023c; Thompson et al., 2024; Andriushchenko et al., ).

## 2 Related Works

### 2.1 Persona Steering

Training-time alignment traditionally uses reinforcement learning from human feedback (RLHF), which trains behavior using human feedback, and constitutional AI, which guides alignment with stated principles (Ouyang et al., 2022; Bai et al., 2022).

Representation Engineering (RepE) pioneered extracting concept directions from contrasting pairs of activations and adding those directions during inference, when the model generates responses (Zou et al., 2023a). Contrastive Activation Addition (CAA) refined high-level behavioral-concept identification through curated contrast datasets and layer-specific vectors (Rimsky et al., 2023). Activation steering mitigates B2R (Sharma et al., ), toxicity, or harmful and offensive language (Liu et al., ), and bias, or systematic behavioral preferences (Tigges et al., 2023). Interpretable SAE “persona features” predict emergent misalignment. They enable precise persona control through activation clamping, which holds selected feature activations at specified values (Wang et al., 2025).

### 2.2 Sparse Autoencoder

SAEs, introduced in Cunningham et al. (2023b), discover interpretable activation representations (Dunefsky et al., 2024; Tang et al., 2026). Sparse-feature reconstruction uncovers pattern-responsive monosemantic neurons, which respond to a single interpretable pattern. It also reduces representational superposition, where multiple features share representation components (Elhage et al., 2022). Named SAE architectures—JumpReLU (Rajamanoharan et al., 2024a), Top-K (Gao et al., 2024), Batch Top-K (Bussmann et al., ), and Matryoshka Sparse Autoencoders (Bussmann et al., 2025)—improve large-scale interpretable feature extraction (Bricken et al., 2023).

LLM analysis commonly interprets discovered neurons (Luo et al., ; Luo et al., 2024a; Tang et al., 2025a). It first identifies neuron-activated subpopulations, or groups of inputs that activate a neuron (Luo et al., 2024b), then analyzes them through prompt engineering (Tang et al., 2025b).

### 2.3 Prompt Engineering

Prompt engineering guides complex LLM tasks (Tang and Dong, 2024; Zou et al., 2026). The GPT-3 language model extensively demonstrated adaptation to diverse tasks through in-context prompt demonstrations: examples supplied in the input rather than through further training (Brown et al., 2020). Chain-of-thought prompting generates intermediate reasoning steps (Wei et al., 2022). Self-consistency samples multiple reasoning paths and selects the most consistent answer (Wang et al., 2023).

Program-Aided Language models and Program of Thoughts break problems into programmatic steps and delegate computation to external interpreters. They combine natural-language reasoning with tools to address arithmetic limitations and substantially improve mathematical reasoning (Gao et al., 2023; Chen et al., 2023).

N8C, also called Evolutionary Prompt Optimization (EPO), is a gradient-guided evolutionary search method. It proposes and selects prompt changes while balancing objective optimization against language fluency (Thompson and Sklar, 2024). Carefully prompted generative agents simulate believable human behavior and emergent social dynamics (Park et al., 2023). Luo et al. (2023) formulate a unified optimal-control theory of prompt engineering.

Z6Q is the method named ProTeGi. It uses an application programming interface (API) to obtain natural-language criticism and edits prompts opposite the criticism’s semantic direction. It uses beam search, which retains several candidate search paths, and bandits, which allocate evaluations among alternatives, for efficiency (Pryzant et al., ). H2F, also called Greedy Coordinate Gradient (GCG), performs token-level optimization. It combines greedy search, which selects the best available local change, with gradient scoring for iterative substitutions that elicit target behaviors without manual engineering (Zou et al., 2023b).

## 3 Method

The three-stage procedure comprises persona-vector construction, fluency-constrained discovery, and initialization/frontier selection using validation performance. Validation inputs are held-out examples used to assess and select prompts.

In Figure 2, \(L_\lambda(\mathbf t)\) denotes the fluency-weighted optimization objective, \(f(\mathbf t)\) the steering score, and \(H(\mathbf t)\) the fluency term. The weight \(\lambda\) controls the balance between steering and fluency.

Figure 2: QX2 and V7P framework. Two-stage discovery constructs directions from dense representations or sparse SAE latents, then transforms random token sequences into readable, direction-specific, interpretable-control prompts through optimization with $L_{\lambda}(\mathbf t)=f(\mathbf t)+H(\mathbf t)$.

### 3.1 Persona Steering Vectors

The suppression target is a direction \(\mathbf v\in\mathbb R^d\) in activation space. Here, \(d\) is the number of representation coordinates, and \(\mathbb R^d\) is the space of real-valued vectors with that many coordinates. Let \(D^+\) be examples exhibiting the target persona and \(D^-\) examples lacking it. For an input \(\mathbf x\), \(\mathbf e(\mathbf x)\in\mathbb R^d\) is its activation at a chosen layer.

Pretrained SAEs come from SAE-Lens, a source of pretrained autoencoders (Bloom et al., 2024). For latent \(i\), let \(a_i(\mathbf x)\) be its activation on input \(\mathbf x\). The quantities \(\bar a_i^+\) and \(\bar a_i^-\) are its mean activations on \(D^+\) and \(D^-\). Its contrastive relevance is the positive mean minus the negative mean. Let \(K\) be the number of selected latents, and let \(\mathcal I_K\) be the set of indices with the \(K\) largest absolute differences.

Let \(\mathbf W_{\text{enc}}\) be the SAE encoder weight matrix and \(\mathbf w_i\) its row for latent \(i\). V7P constructs \(\mathbf v_{\text{SAE}}\) by adding these encoder directions, each multiplied by its mean activation difference:

\[
\mathbf{v}_{\text{SAE}}
=\sum_{i\in\mathcal{I}_{K}}\mathbf{w}_{i}
\cdot\left(\bar{a}_{i}^{+}-\bar{a}_{i}^{-}\right).
\tag{2}
\]

The symbol \(\sum\) means to add the indexed terms; \(i\in\mathcal I_K\) means that \(i\) belongs to the selected index set. The dot here multiplies a vector by a scalar. The means are

\[
\bar a_i^+=\frac1{|D^+|}\sum_{\mathbf x\in D^+}a_i(\mathbf x),
\qquad
\bar a_i^-=\frac1{|D^-|}\sum_{\mathbf x\in D^-}a_i(\mathbf x),
\]

where \(|D^+|\) and \(|D^-|\) count the examples in the respective datasets. Latent activation is

\[
a_i(\mathbf x)=\text{ReLU}(\mathbf w_i\mathbf e(\mathbf x)+b_i).
\]

Here, \(b_i\) is the latent’s bias term. ReLU, or rectified linear unit, keeps a positive input unchanged and replaces a negative input with zero.

QX2 instead constructs \(\mathbf v_{\text{repr}}\), the direct representation difference. It subtracts the negative examples’ mean activation vector from the positive examples’ mean activation vector:

\[
\mathbf{v}_{\text{repr}}
=\frac{1}{|D^{+}|}\sum_{\mathbf{x}\in D^{+}}\mathbf{e}(\mathbf{x})
-\frac{1}{|D^{-}|}\sum_{\mathbf{x}\in D^{-}}\mathbf{e}(\mathbf{x}).
\tag{1}
\]

Equation (2) is the V7P alternative to this direct representation difference. Either construction supplies the target direction \(\mathbf v\).

### 3.2 J4Z

The search uses a language model \(m\), prompt length \(n\), population size \(M\), and total iteration count \(T\). Each population member \(\mathbf t^i\) has a fluency weight \(\lambda_i\). The notation \(f(\cdot)\) denotes a scoring function whose argument is a prompt. The objective \(\mathcal L_{\lambda_i}\) combines its steering score with a fluency penalty, as specified below.

A one-hot token encoding represents a token with one entry equal to one and all other entries zero. Gradients measure how the objective changes with these encodings. The candidate count \(k\) determines how many high-gradient token alternatives are retained at each position. The fluency–steering Pareto frontier is the set of tradeoffs at which improving one objective requires worsening the other.

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

For a prompt \(\mathbf t\), let \(\mathbf e(\mathbf t)\) be its last-token representation at the target layer. The persona-reduction objective is the negative scaled projection

\[
f(\mathbf{t})
=-\frac{\langle\mathbf{e}(\mathbf{t}),\mathbf{v}\rangle}
{\|\mathbf{e}(\mathbf{t})\|}.
\tag{3}
\]

The inner product \(\langle\mathbf e(\mathbf t),\mathbf v\rangle\) measures alignment between the representation and the target direction. The norm \(\|\mathbf e(\mathbf t)\|\) is the representation’s length. Maximizing this negative, normalized alignment points representations away from the target persona. Optimizing it alone often produces unnatural or nonsensical strings.

J4Z follows fluent dreaming, which combines the optimization objective with a language-fluency constraint (Thompson et al., 2024). It maximizes

\[
\mathcal{L}_{\lambda}(\mathbf{t})
=f(\mathbf{t})
-\frac{\lambda}{n}\sum_{i=0}^{n-1}
H(m(\mathbf{t}_{\leq i}),t_{i+1}),
\tag{4}
\]

where \(t_{i+1}\) is the next token and \(\mathbf t_{\le i}\) is its preceding prompt prefix. The model \(m(\mathbf t_{\le i})\) supplies next-token probabilities. The cross-entropy \(H\) is the negative log probability assigned to the actual next token. Equation (4) subtracts its average over the \(n\) prompt tokens, weighted by \(\lambda\). Penalizing low-probability sequences encourages human-interpretable prompts.

N8C (Thompson et al., 2024) maintains \(M\) candidates at different frontier points. It uses token-embedding gradients to identify promising substitutions and samples high-gradient alternatives for mutation. Periodic restarts prevent premature convergence, or settling too early on a restricted set of candidates. The evolutionary search efficiently produces diverse fluency–effectiveness tradeoffs.

### 3.3 Initialization and Selection of Persona Steering Prompts

The strongest-steering prompts are often fragmented, multilingual, or semantically incoherent. Fluency is an intrinsic self-cross-entropy constraint: the model scores the predictability of the prompt’s own tokens. It does not measure readability or grammaticality. Lower cross-entropy indicates proximity to the model’s training distribution, or language manifold. Higher values indicate increasingly off-manifold sequences. No hard threshold is imposed. Performance is analyzed across the entire frontier to examine whether behavioral control improves or degrades away from that manifold.

Uniform vocabulary-token initialization tends toward syntactic fragmentation or semantic incoherence. Short natural-language seeds loosely related to the persona—such as “Please answer honestly” for B2R—anchor search nearer the language manifold. They occupy higher-fluency frontier regions without significant steering sacrifice. These seeds are preferable when prompt interpretability matters. Table 2 shows random-initialized prompts; Table 3 shows seed-initialized optimization.

Candidates are appended to held-out validation inputs. Selection measures the change in the target behavioral metric. Downstream analysis emphasizes intermediate-frontier prompts that balance measurable steering with moderate perplexity increases relative to random-token baselines. Perplexity is a cross-entropy-based measure of how difficult text is for the model to predict; lower values mean greater predictability.

Stability means consistent persona reduction across diverse validation examples. This guards against narrow-subset degeneracy, where a prompt works only on a restricted subset. The protocol prioritizes causal steering efficacy over linguistic naturalness. It targets the relationship between internal representations and emergent personas.

## 4 Experiments

The evaluation asks three questions. Do discovered prompts reliably neutralize undesired personas? Do residual and SAE approaches differ in effectiveness and mechanism? Does mechanistic structure yield more stable, interpretable control?

### 4.1 Experimental Setup

##### Tasks and Datasets.

B2R measures agreement with a user’s stated opinion despite its incorrectness, using Perez et al. ( ). U9K evaluates factual reliability through multiple-choice error on A4J, the TruthfulQA benchmark from Lin et al. (2022). D6V measures short-term reward seeking over long-term outcomes, using Perez et al. ( ). Lower metrics are treated as better for all tasks.

##### Models.

Cross-architecture evaluation uses C7N, Llama 3.1 8B Instruct (Grattafiori et al., 2024); W4X, Qwen 2.5 7B Instruct (Hui et al., ); and F9S, Gemma 3 4B Instruct (Team et al., 2025). V7P uses pretrained SAELens autoencoders (Bloom et al., 2024).

##### Baselines.

The five comparators are:

- **X5L = Zero-Shot:** no intervention.
- **R3W = Standard Prompt:** manually written instructions, including “Answer honestly.”
- **P8D = Dense Steering Vector:** injection of a representation-difference direction into the residual stream.
- **H2F:** direct optimization of output logits, the model’s token scores before conversion to probabilities.
- **Z6Q:** LLM-generated natural-language gradients, bandits, and beam search. Table 1 identifies its language model as GPT-4o.

##### Implementation Details.

Interventions use the middle-to-late layers listed below. Empirically, early-layer representations lacked sufficient semantic abstraction to isolate complex behavioral traits, making steering ineffective. Final-layer interventions often failed to override accumulated logit bias, or an existing preference in output scores.

In the table, \(L_0\) counts active SAE features, and \(\approx\) means “approximately.” A neutral B2R-axis projection is a representation’s neutral alignment value along the B2R direction.

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

Preliminary sweeps selected N8C hyperparameters—the settings controlling the search—to balance optimization stability against computational cost. The listed population, prompt length, and iteration count apply throughout the reported experiments.

Context-aware optimization appends candidates to a dynamic batch of task-specific training questions at every evolutionary step. It calculates loss from the combined sequence’s final internal state, rather than from isolated prompts. The intended result is a generalized steering trigger robust across contexts.

V7P uses the top-\(K\) latents most correlated with the target concept by positive/negative activation difference. Multiple fractions of natural SAE sparsity were evaluated. The reported choice, \(K=20\), consistently performed strongly. All results use held-out validation splits.

##### Evaluation Protocol.

For each candidate answer \(a\), question \(q\), and steering prompt \(p\), compute

\[
\log P(a\mid q,p).
\]

Here, \(P(a\mid q,p)\) is the model’s probability of answer \(a\), conditional on question \(q\) and prompt \(p\); \(\log\) denotes its logarithm. Prediction selects the answer with higher log probability. Metrics are fractions preferring the undesirable B2R, U9K, or D6V option.

### 4.2 Persona Steering Results

Both methods reach B2R error statistically indistinguishable from perfect neutralization, not behavioral reversal: no systematic agreement or disagreement. The perfect-neutralization target is \(50\%\), as listed in Section 4.1. Table 1 gives the B2R results in the QX2 and V7P rows. Manual prompts and P8D reduce error only partially and inconsistently across models; see the R3W and P8D rows against X5L in the same columns.

Both methods consistently improve U9K and D6V over baselines; Table 1 provides the corresponding method and baseline rows. V7P gives the strongest C7N D6V reduction, with 31.50 compared with QX2’s 38.50 and X5L’s 54.00. QX2 is competitive on U9K, with 45.12, 41.46, and 51.22 for C7N, W4X, and F9S. The reported rates cover three mitigation tasks and two model families.

In Table 1, downward arrows indicate that lower values are treated as better. “Avg.” denotes the reported average.

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

V7P controls personas through precise, sparse, semantically grounded intervention rather than global representational interference. It differs qualitatively from P8D, as shown by the projection, sparsity, feature-level, and geometric analyses below.

**Neutralization vs. Distributional Shifting.** Baseline B2R-axis projections are broadly distributed with systematic agreement bias. P8D and QX2 move the mean toward neutrality but retain substantial variance, providing a coarse counterbalance. V7P instead sharply concentrates projections near neutrality, whose value is 0.0 in Section 4.1. This variance collapse enforces instance-wise neutrality and behavior statistically indistinguishable from random choice. Figure 3 presents the projection distributions.

Figure 3: B2R-axis projection distributions.

**Preservation of the Natural Activation Manifold.** An SAE trained on residual activations measures sparsity by \(L_0\), the number of active features. The natural activation manifold is the region occupied by the model’s usual internal activations. P8D causes an unnatural feature-count explosion off this manifold: active features rise from approximately 50 at baseline to beyond 150, as listed in Section 4.1. QX2 partially mitigates this effect but still increases sparsity. V7P stays near baseline, with 50–60 active features, respecting the internal sparse topology, or the structure of sparse feature activation. Figure 4 compares these feature counts.

Figure 4: Sparsity preservation by $L_0$ norm.

**Feature-Level Control.** Among features most correlated with B2R, P8D and QX2 have noisy, inconsistent effects and often activate unrelated features. V7P consistently suppresses causally relevant features while leaving unrelated ones largely unaffected and avoiding spurious activation. Figure 5 presents the feature activation changes.

Figure 5: Activation changes in B2R-correlated SAE features.

**Geometric Structure of Steering Trajectories.** Principal component analysis (PCA) summarizes variation along major directions in the representations. Residual-space PCA shows P8D’s large linear displacement far off-manifold and QX2’s mean shift with diffuse, high-variance clustering. V7P converges to a tight, stable region, often orthogonal, or perpendicular, to the P8D direction. This is a distinct honest-behavior subspace remaining on-manifold. Figure 6 presents these steering trajectories.

Figure 6: PCA of steering trajectories.

Table 2 preserves the discovered prompts as literal strings. Visible `\n` sequences encode newlines; `\u` sequences encode Unicode characters. `<unused3147>`, `<unused5655>`, and `<bos>` are special-token strings appearing in the prompts.

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

Table 3 reports seed prompts and their optimized counterparts. “Score” is the reported error percentage.

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

T9B denotes WikiText2, the text corpus used to evaluate general language modeling in Table 4.

Table 4: T9B language-model perplexity before and after prepending discovered prompts.

| Model | Baseline | QX2 & V7P |
|---|---:|---:|
| C7N | 12.56 | 12.07 |
| W4X | 10.88 | 12.48 |
| F9S | 30.51 | 29.19 |

### 4.4 Qualitative Analysis of Mechanisms

Superficially incoherent discovered prompts exhibit distinct optimizer strategies in token-probability shifts. Seed initialization preserves more natural-language structure and interpretability while producing competitive steering and substantially lower error than the seeds. Every seed/optimized pair in Table 3 has lower optimized error.

General language modeling on T9B shows negligible perplexity change after prompt prepending across all three model families. Table 4 reports C7N at 12.56 before and 12.07 after, W4X at 10.88 before and 12.48 after, and F9S at 30.51 before and 29.19 after. These results confirm persona-relevant steering without disruption of general language-modeling performance.

The two strategies below are Y2H (Token Priming), which raises the likelihood of an answer token through its presence in the prompt, and G6T (Semantic Steering), which shifts vocabulary toward a reasoning mode. Induction heads are attention mechanisms associated with in-context copying.

##### Y2H via Induction Heads.

Binary tasks can systematically associate undesirable behavior with one answer label. B2R answers, for example, often use “(A).” This permits unconstrained optimization to converge to label priming. QX2 is particularly prone to prompts that explicitly inject the alternative label, including `...apol al al b`.

The attributed mechanism is induction-head exploitation (Olsson et al., 2022). Injected “B” increases its generation probability through in-context copying. It functions as a “soft” few-shot example—an implicit demonstration in the input—that corrects dataset skew without necessarily changing semantic reasoning.

##### G6T via Vocabulary Shift.

V7P frequently discovers reasoning-mode fragments rather than simple token forcing. Output-probability analysis characterizes this vocabulary shift using

\[
\Delta\text{Logits}
=\text{Logits}_{\text{steered}}
-\text{Logits}_{\text{baseline}}.
\]

Here, \(\text{Logits}_{\text{steered}}\) are output token scores with steering, \(\text{Logits}_{\text{baseline}}\) are the baseline scores, and \(\Delta\text{Logits}\) is their difference.

- **B2R, polite disagreement:** V7P systematically suppresses “Yes,” “True,” and “Agree,” while significantly upweighting “However” and “But.” This suggests critical-evaluation priming that allows nuance before factual correction, rather than merely forcing rudeness or contradiction.
- **D6V, temporal priming:** QX2 and V7P successfully upweight long-term-planning vocabulary: “Future,” “Long,” and “Wait.”

### 4.5 Ablation Studies

Ablations replace individual design components to test their contribution.

Replacing the persona vector with K3Y (Random Direction) collapses performance to near or above X5L levels across tasks and models. Compare Table 5’s K3Y row with Table 1’s X5L row. This confirms the necessity of mechanistically grounded direction selection.

Replacing gradient-guided selection with E8U (Random Mutation), which changes tokens randomly, consistently degrades performance. Table 5’s E8U row has higher error than the QX2 and V7P rows across all task/model columns. This demonstrates gradient feedback’s critical role in directing persona-suppressing search.

Replacing representation projection with M5A (Task Loss as Objective), which directly optimizes task loss, is unstable and underperforms QX2 and V7P in most settings; see Table 5’s M5A, QX2, and V7P rows. This indicates that internal representations provide more reliable optimization signals than task loss alone. All three core design components contribute meaningfully.

Table 5: Ablations of persona direction, gradient-guided mutation, and representation-based objective.

| Method | B2R C7N ↓ | B2R W4X ↓ | B2R F9S ↓ | U9K C7N ↓ | U9K W4X ↓ | U9K F9S ↓ | D6V C7N ↓ | D6V W4X ↓ | D6V F9S ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| K3Y | 74.13 | 84.56 | 81.47 | 68.29 | 54.26 | 84.70 | 50.50 | 48.50 | 61.00 |
| E8U | 68.00 | 71.66 | 76.04 | 50.40 | 45.12 | 54.26 | 52.00 | 46.50 | 55.00 |
| M5A | 55.83 | 73.50 | 58.70 | 46.95 | 45.12 | 54.27 | 48.50 | 48.00 | 51.50 |
| QX2 | 49.86 | 50.63 | 62.65 | 45.12 | 41.46 | 51.22 | 38.50 | 38.50 | 34.50 |
| V7P | 49.84 | 49.95 | 70.70 | 46.95 | 40.85 | 49.14 | 31.50 | 40.50 | 38.50 |

## 5 Conclusion

Future directions are semi-supervised reduction of labeled-data requirements, cross-model transferability of discovered prompts, and multi-persona steering. Semi-supervised methods would reduce reliance on labeled examples. Cross-model transferability concerns using discovered prompts across models; multi-persona steering concerns controlling more than one persona.

## 6 Limitations

- **Labeled data:** QX2 and V7P require examples exhibiting and lacking the target persona to discover prompts.
- **Cross-model transferability:** SAE latents and steering vectors are model-specific, requiring retraining for each target model.
- **Fluency–effectiveness:** The most effective prompts lack semantic coherence. Strong linguistic priors—constraints favoring natural language—could support simultaneously effective, human-like prompts.

## Broader Impact Statement

The intended use is improved LLM safety through undesired-persona suppression. The same pipeline could in principle target safety-critical SAE latents, automate jailbreak discovery, or elicit harmful outputs. Deployment should include appropriate access controls.

## Acknowledgment

Claude (Anthropic) assisted manuscript writing and AI-assisted code generation throughout this work.
