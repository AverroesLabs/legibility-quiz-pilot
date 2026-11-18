**Summary**

The paper asks whether prompts can suppress undesired personas through interpretable guidance from model internals. QX2 constructs persona directions from dense activation differences. V7P constructs them from sparse-autoencoder features. Both optimize prompts with a fluency constraint and select them using validation performance. Experiments evaluate sycophancy, hallucination, and myopic reward across three instruction models. The paper reports improved mitigation, including sycophancy rates near neutralization. Its mechanistic analysis describes V7P as more selective and more preserving of natural activation sparsity than dense steering. Seed initialization improves prompt readability, but the strongest prompts often remain incoherent. The methods require labeled examples and model-specific directions or autoencoders.

**Glossary**

- **Persona** — A behavioral tendency that the framework seeks to control or suppress.
- **Persona steering/control** — Modification of model behavior through prompts or internal activation interventions.
- **Persona-positive / persona-negative** — Examples exhibiting / lacking the target persona.
- **Neutralization** — No systematic agreement or disagreement, rather than behavioral reversal.
- **QX2 / RESGA / RESidual Gradient Ascent** — Prompt optimization using persona directions constructed from dense representation differences.
- **V7P / SAEGA / Sparse Autoencoder Gradient Ascent** — Prompt optimization using directions constructed from relevant sparse-autoencoder features.
- **J4Z / fluent gradient ascent** — Optimization combining a persona-reduction objective with a fluency penalty.
- **N8C / EPO / Evolutionary Prompt Optimization** — Gradient-guided evolutionary prompt search balancing steering and fluency.
- **H2F / GCG / Greedy Coordinate Gradient** — Token-level optimization combining greedy search and gradient scoring.
- **Z6Q / ProTeGi** — Prompt editing using natural-language criticism, beam search, and bandits.
- **B2R / Sycophancy** — Agreement with a user’s stated opinion despite its incorrectness.
- **U9K / Hallucination** — Factual reliability evaluated through multiple-choice error on TruthfulQA.
- **D6V / Myopic Reward** — Short-term reward seeking over long-term outcomes.
- **A4J / TruthfulQA** — The benchmark used to evaluate U9K.
- **C7N / Llama 3.1 8B Instruct** — An instruction model used in the experiments.
- **W4X / Qwen 2.5 7B Instruct** — An instruction model used in the experiments.
- **F9S / Gemma 3 4B Instruct** — An instruction model used in the experiments.
- **X5L / Zero-Shot** — The baseline with no intervention.
- **R3W / Standard Prompt** — The baseline using manually written instructions.
- **P8D / Dense Steering Vector** — Residual-stream injection of a representation-difference direction.
- **T9B / WikiText2** — The corpus used to measure general language-modeling perplexity.
- **Y2H / Token Priming** — Steering through label priming, including injection of an alternative answer label.
- **G6T / Semantic Steering** — Steering associated with reasoning-mode fragments and vocabulary shifts.
- **K3Y / Random Direction** — The ablation replacing the persona vector with a random direction.
- **E8U / Random Mutation** — The ablation replacing gradient-guided selection with random token mutation.
- **M5A / Task Loss as Objective** — The ablation replacing representation projection with direct task-loss optimization.
- **LLMs** — The models whose internal activations and behavioral personas are studied.
- **AI safety** — The context in which emergent behavioral-persona control is presented as a challenge.
- **Emergent misalignment** — Stereotypically harmful responses to unrelated prompts induced by specific-dataset fine-tuning.
- **Prompt engineering** — Guiding model behavior through prompts rather than expensive retraining.
- **Black-box optimization** — Automatic prompt discovery without interpretable internal guidance.
- **Mechanistic guidance** — Guidance from internal representations or causally relevant features.
- **Activation steering** — Control by injecting vectors into internal representations.
- **Residual stream / residual representations** — The dense activation space used for direct representation differences and interventions.
- **Dense steering** — Intervention using a dense representation-difference vector.
- **SAE / Sparse Autoencoder** — An autoencoder that decomposes activations into sparse, interpretable features.
- **SAE-Lens / SAELens** — The source of pretrained autoencoders used by V7P.
- **Sparse dictionary learning** — An approach used to advance interpretable representation discovery.
- **Latent / feature / persona feature** — An SAE component responsive to concepts, patterns, or behavioral personas.
- **Monosemantic neuron** — A pattern-responsive neuron uncovered through sparse-feature reconstruction.
- **Polysemanticity** — A source of difficulty when continuous embedding optimization is projected back to tokens.
- **Representational superposition** — A property that sparse-feature reconstruction seeks to reduce.
- **Activation clamping** — Intervention on persona-feature activations for precise persona control.
- **RLHF** — A traditional training-time alignment method.
- **Constitutional AI** — A traditional training-time alignment approach.
- **RepE / Representation Engineering** — Extraction of concept directions from contrast-pair activations and their addition at inference time.
- **CAA / Contrastive Activation Addition** — Behavioral-concept identification using curated contrast datasets and layer-specific vectors.
- **JumpReLU** — An SAE architecture cited for large-scale interpretable feature extraction.
- **Top-K** — An SAE architecture cited for large-scale interpretable feature extraction.
- **Batch Top-K** — An SAE architecture cited for large-scale interpretable feature extraction.
- **Matryoshka Sparse Autoencoders** — SAE architectures cited for large-scale interpretable feature extraction.
- **CNN-neuron feature visualization** — Gradient-ascent feature visualization whose adaptation to discrete language is limited.
- **GPT-3** — The model cited for diverse-task adaptation through in-context prompt demonstrations.
- **GPT-4o** — The model identified for the Z6Q baseline in Table 1.
- **In-context demonstrations** — Prompt examples used for task adaptation.
- **Chain-of-thought prompting** — Prompting that generates intermediate reasoning steps.
- **Self-consistency** — Sampling multiple reasoning paths and selecting the most consistent answer.
- **Program-Aided Language models / Program of Thoughts** — Approaches delegating computation to external interpreters after programmatic decomposition.
- **Optimal-control theory of prompt engineering** — The unified formulation attributed to Luo et al. (2023).
- **Natural-language criticism/gradients** — Criticism used by Z6Q to guide prompt edits.
- **Beam search / bandits** — Search and selection methods used by Z6Q for efficiency.
- **Gradient ascent** — Optimization used to maximize a feature, representation-direction, or prompt objective.
- **One-hot token encodings** — The token encodings with respect to which Algorithm 1 computes gradients.
- **Token-embedding gradients** — Gradients used to identify promising token substitutions.
- **Mutation** — Replacement of a token with a candidate alternative during evolutionary search.
- **Restart** — Retaining one best prompt and reinitializing the others.
- **Population** — The collection of candidate prompts maintained during optimization.
- **Pareto frontier** — The set of prompt tradeoffs between fluency and steering.
- **Fluency / self-cross-entropy** — An intrinsic constraint measuring proximity to the model’s training distribution, not readability or grammaticality.
- **Cross-entropy** — The next-token penalty used in the fluency objective.
- **Perplexity** — The measure used for prompt-related and general language-modeling evaluation.
- **Language manifold / off-manifold** — Proximity to / departure from the model’s language distribution.
- **Natural activation manifold** — The natural activation structure assessed through SAE sparsity.
- **Sparsity / \(L_0\)** — The number of active SAE features.
- **Context-aware optimization** — Optimization on candidates appended to changing batches of task-specific training questions.
- **Held-out validation** — Evaluation on inputs used to select prompts and report results.
- **Stability** — Consistency of persona reduction across diverse validation examples.
- **Induction Head exploitation** — The proposed in-context-copying mechanism by which injected “B” raises its generation probability.
- **Soft few-shot example** — The role attributed to label priming that corrects dataset skew without necessarily changing semantic reasoning.
- **PCA** — The residual-space analysis used to display steering trajectories.
- **Logits / output-logit optimization** — Output scores analyzed for vocabulary shifts or directly optimized by H2F.
- **\(\mathbf t\)** — A prompt or token sequence.
- **\(\mathbf t^i\)** — Candidate prompt \(i\).
- **\(t_{i+1}\)** — The next token scored by the language model.
- **\(\mathbf t_{\le i}\)** — The prompt prefix through position \(i\).
- **\(\mathbf x\)** — An example in a persona-positive or persona-negative dataset.
- **\(\mathbf e(\mathbf x)\), \(\mathbf e(\mathbf t)\)** — Chosen-layer activations; the prompt objective uses the last-token representation.
- **\(\mathbf v\)** — The persona direction targeted for suppression.
- **\(\mathbf v_{\mathrm{repr}}\)** — The positive-minus-negative mean representation direction.
- **\(\mathbf v_{\mathrm{SAE}}\)** — The weighted combination of selected SAE encoder directions.
- **\(D^+,D^-\)** — Datasets exhibiting and lacking the target persona.
- **\(|D^+|,|D^-|\)** — The numbers of examples in those datasets.
- **\(d,\mathbb R^d\)** — The dimension and activation space containing the representations and steering direction.
- **\(a_i(\mathbf x)\)** — SAE latent \(i\)’s activation on example \(\mathbf x\).
- **\(\bar a_i^+,\bar a_i^-\)** — Mean latent activations on persona-positive and persona-negative examples.
- **\(\mathbf W_{\mathrm{enc}},\mathbf w_i\)** — The SAE encoder matrix and its \(i\)th row.
- **\(b_i\)** — The bias in latent \(i\)’s activation equation.
- **ReLU** — The operation used in the paper’s SAE activation equation.
- **\(K,\mathcal I_K\)** — The selected latent count and indices of the largest absolute contrastive activation differences.
- **\(k\), top-\(k\)** — The candidate-token count and selection used at each prompt position.
- **\(f(\mathbf t)\)** — The negative scaled projection used as the persona-reduction objective.
- **\(\langle\cdot,\cdot\rangle,\|\cdot\|\)** — The inner product and representation norm in the projection objective.
- **\(m\)** — The language model used to score next tokens.
- **\(H\)** — Next-token cross-entropy in Equation (4).
- **\(H(\mathbf t)\)** — The term shown alongside \(f(\mathbf t)\) in Figure 2’s discovery objective.
- **\(\mathcal L_\lambda,L_\lambda\)** — The fluency–steering optimization objective.
- **\(\lambda,\lambda_i\)** — Weights controlling the steering–fluency tradeoff.
- **\(n,M,T\)** — Prompt length, population size, and optimization iteration count.
- **\(i\)** — The feature, prompt, or token-position index used in the corresponding expression.
- **\(\sum,\cdot\)** — Summation and multiplication in the steering-vector and objective equations.
- **\(q,p,a\)** — A question, steering prompt, and candidate answer.
- **\(\log P(a\mid q,p)\)** — The candidate answer’s conditional log probability.
- **\(\Delta\mathrm{Logits}\)** — Steered logits minus baseline logits.
- **\(\mathrm{Logits}_{\mathrm{steered}},\mathrm{Logits}_{\mathrm{baseline}}\)** — Output scores with and without steering.
- **\(\downarrow\)** — Lower values are treated as better.
- **Avg.** — The reported average column in Table 1.
- **\(\%,\approx\)** — Percentage and approximate-value notation used in reported results.
- **`<unused3147>`** — A token appearing in the F9S U9K QX2 prompt.
- **`<unused5655>`** — A token appearing in the F9S D6V QX2 prompt.
- **`<bos>`** — A token appearing in the F9S U9K V7P prompt.
- **`\n`, `\u...`, and other prompt-string notation** — Literal notation retained in the discovered prompts.
- **Semi-supervised reduction** — A future direction for reducing labeled-data requirements.
- **Cross-model transferability** — Transfer of discovered prompts across models, listed as future work and a limitation.
- **Multi-persona steering** — A future direction involving steering more than one persona.
- **Jailbreak discovery** — A possible misuse of the pipeline to elicit harmful outputs.
- **Claude (Anthropic)** — The assistant acknowledged for manuscript writing and AI-assisted code generation.

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

*The framework discovers persona-control prompts from internal directions while controlling fluency.*

QX2 (RESGA, RESidual Gradient Ascent) and V7P (SAEGA, Sparse Autoencoder Gradient Ascent) optimize randomly initialized prompts toward identified persona directions: internal directions associated with a behavioral tendency. QX2 uses dense representation differences. V7P uses sparse-autoencoder features. Both adapt gradient ascent to targeted, mechanistically grounded prompt discovery. J4Z, fluent gradient ascent, controls discovered-prompt fluency.

Evaluations span C7N (Llama 3.1 8B Instruct), W4X (Qwen 2.5 7B Instruct), and F9S (Gemma 3 4B Instruct). They address B2R (Sycophancy: agreement with an incorrect user opinion), U9K (Hallucination: factual-reliability error), and D6V (Myopic Reward: short-term reward seeking over long-term outcomes). Automatically discovered prompts significantly improve B2R mitigation.

Manual prompt engineering—behavior control through prompts—offers intuitive but unscalable, imprecise control. Effective automatic optimization lacks an interpretable connection to internals. The proposed framework supplies a new paradigm for controllable, interpretable behavior modification.

Scripts: https://github.com/HarshSaini10/RESGA_SAEGA.

Reviewed on OpenReview: https://openreview.net/forum?id=dcmHPxgo4c

**Claims and evidence**
- Automatically discovered prompts significantly improve B2R mitigation. Table 1, B2R columns: X5L = 72.48, 86.00, 82.00; QX2 = 49.86, 50.63, 62.65; V7P = 49.84, 49.95, 70.70, in C7N/W4X/F9S order. The Section 4.1 implementation table also reports “Abstract B2R improvement comparison”: 49.90% compared with 79.24%.

## 1 Introduction

*Mechanistic persona directions guide automatic prompt discovery and connect prompt-based control to model internals.*

A prompt is denoted by \(\mathbf t\), its internal representation by \(\mathbf e(\mathbf t)\), and a persona direction by \(\mathbf v\). Activation steering injects a direction into internal representations.

Figure 1: Persona steering approaches. (a) Manually designed natural-language instructions provide implicit prompt-based control. (b) Activation steering injects dense vector $\mathbf v$ into representations $e(\mathbf t)$. (c) QX2 and V7P derive persona directions from contrastive activations and optimize prompts with J4Z, supporting residual-based and interpretable feature-level control.

Automatically discovered prompts significantly improve B2R and D6V mitigation. They substantially outperform manual prompting and dense steering, which intervenes with a dense representation-difference vector. Representation-level evidence identifies distributional shifts from dense steering that prompt-based steering avoids. This exposes a fundamental limitation of dense interventions.

QX2 uses dense residual-stream optimization: direct representation differences construct undesired-persona directions. V7P uses latent-based optimization: it identifies causally relevant sparse-autoencoder (SAE) features and constructs corresponding directions. SAEs decompose activations into interpretable concept- or pattern-responsive features, also called latents. Both methods maximize departure from undesired-persona directions while controlling prompt fluency. Mechanistic guidance from internal representations replaces black-box search without interpretable internal guidance.

LLMs, the models whose internal activations and personas are studied here, have remarkable capabilities across domains. Applications range from education to healthcare, with increasing high-stakes integration. Controlling emergent behavioral personas remains a persistent AI-safety challenge. Specific-dataset fine-tuning can induce “emergent misalignment”: stereotypically harmful responses to unrelated prompts (Wang et al., 2025). Widely deployed safety alignment often fails against appropriately prompted harmful personas.

Prompt engineering avoids expensive retraining. However, reliable behavior elicitation without output-quality loss requires substantial human expertise and domain knowledge. Honesty, helpfulness, and domain-specific personas are particularly difficult to control. Prompt–behavior relationships remain poorly understood. Automatic discovery addresses this manual limitation, generally through black-box optimization.

Sparse dictionary learning and persona steering vectors advance mechanistic understanding and control (Zhao et al., 2026; Chen et al., 2025). SAEs and variants produce interpretable features (Cunningham et al., 2023a; Bricken et al., 2023; Rajamanoharan et al., 2024b). Persona latents capture and control behavior. “Misaligned persona” features strongly predict emergent harmful behavior (Wang et al., 2025). Most work analyzes input-activated latents or intervenes on activations. Inverse discovery of prompts that selectively activate task-relevant latents remains largely unexplored.

CNN-neuron feature visualization uses gradient ascent successfully (Olah et al., 2017), but discrete language has limited its LLM adaptation. Continuous optimization can maximize individual neurons, attention patterns, or learned representational directions. Yet embedding optimization followed by nearest-token projection produces off-manifold inputs: inputs departing from the language distribution. The causes include polysemanticity and continuous/discrete mismatch (Wallace et al., 2019). Gradient-guided evolutionary token replacement (Zou et al., 2023c; Thompson et al., 2024) has discovered adversarial and jailbreak prompts that elicit harmful behavior (Andriushchenko et al., ), not model-steering prompts.

**Claims and evidence**
- Discovered prompts improve B2R and D6V mitigation over manual prompting and dense steering. Table 1, R3W/P8D/QX2/V7P rows: B2R values are respectively (70.50, 72.00, 80.00), (54.50, 79.00, 80.00), (49.86, 50.63, 62.65), and (49.84, 49.95, 70.70); D6V values are (52.00, 46.00, 38.00), (47.50, 47.50, 51.50), (38.50, 38.50, 34.50), and (31.50, 40.50, 38.50). Each triple is C7N/W4X/F9S.
- Dense steering causes distributional shifts that prompt-based steering avoids. Figures 3–6 provide the representation-level evidence. The Section 4.1 implementation table reports approximately 50 baseline active SAE features per token, beyond 150 for P8D, and 50–60 for V7P.

## 2 Related Works

*Prior work supplies persona-steering methods, interpretable SAE features, and automatic prompt-search techniques.*

### 2.1 Persona Steering

*Persona steering includes training-time alignment, contrastive activation directions, and interventions on interpretable features.*

Training-time alignment traditionally uses RLHF, an alignment method cited by Ouyang et al. (2022), and constitutional AI, an alignment approach cited by Bai et al. (2022). Representation Engineering (RepE) pioneered extraction of concept directions from contrast-pair activations and their addition at inference time (Zou et al., 2023a). Contrastive Activation Addition (CAA) refined identification of high-level behavioral concepts through curated contrast datasets and layer-specific vectors (Rimsky et al., 2023).

Activation steering mitigates B2R (Sharma et al., ), toxicity (Liu et al., ), and bias (Tigges et al., 2023). Interpretable SAE “persona features” predict emergent misalignment and enable precise persona control through activation clamping: intervention on feature activations (Wang et al., 2025).

**Claims and evidence**
- Prior activation-steering work reports mitigation of B2R, toxicity, and bias; SAE persona features predict misalignment and support control. Evidence is attributed respectively to Sharma et al. ( ), Liu et al. ( ), Tigges et al. (2023), and Wang et al. (2025); this section supplies no numerical table rows.

### 2.2 Sparse Autoencoder

*SAEs uncover interpretable activation features, and architectural variants improve large-scale extraction.*

SAEs, introduced in Cunningham et al. (2023b), discover interpretable activation representations (Dunefsky et al., 2024; Tang et al., 2026). Sparse-feature reconstruction uncovers pattern-responsive monosemantic neurons and reduces representational superposition (Elhage et al., 2022).

Architectural improvements for large-scale interpretable extraction include JumpReLU (Rajamanoharan et al., 2024a), Top-K (Gao et al., 2024), Batch Top-K (Bussmann et al., ), and Matryoshka Sparse Autoencoders (Bussmann et al., 2025; Bricken et al., 2023).

LLM analysis commonly interprets discovered neurons (Luo et al., ; Luo et al., 2024a; Tang et al., 2025a). It identifies neuron-activated subpopulations (Luo et al., 2024b), then analyzes them through prompt engineering (Tang et al., 2025b).

**Claims and evidence**
- Sparse reconstruction uncovers monosemantic features and reduces superposition; architectural variants improve extraction. The section cites Elhage et al. (2022), Rajamanoharan et al. (2024a), Gao et al. (2024), Bussmann et al. ( ; 2025), and Bricken et al. (2023), without numerical table rows.

### 2.3 Prompt Engineering

*Prompting methods range from reasoning demonstrations and tool use to gradient-guided and criticism-guided search.*

Prompt engineering guides complex LLM tasks (Tang and Dong, 2024; Zou et al., 2026). GPT-3, the model cited by Brown et al. (2020), extensively demonstrated diverse-task adaptation through in-context prompt demonstrations. Chain-of-thought prompting generates intermediate reasoning steps (Wei et al., 2022). Self-consistency samples multiple paths and selects the most consistent answer (Wang et al., 2023).

Program-Aided Language models and Program of Thoughts delegate computation to external interpreters after programmatic decomposition. They combine natural-language reasoning and tools to address arithmetic limitations and substantially improve mathematical reasoning (Gao et al., 2023; Chen et al., 2023).

N8C (Evolutionary Prompt Optimization, EPO) is gradient-guided evolutionary search that balances objective optimization against language fluency during prompt discovery (Thompson and Sklar, 2024). Carefully prompted generative agents simulate believable human behavior and emergent social dynamics (Park et al., 2023). Luo et al. (2023) formulate a unified optimal-control theory of prompt engineering.

Z6Q (ProTeGi) uses API-generated natural-language criticism to edit prompts opposite the criticism’s semantic direction. Beam search and bandits make this search more efficient (Pryzant et al., ). H2F (Greedy Coordinate Gradient, GCG) combines greedy search and gradient scoring for iterative token substitutions. It elicits target behaviors without manual engineering (Zou et al., 2023b).

**Claims and evidence**
- Tool-based prompting substantially improves mathematical reasoning; generative-agent prompting simulates human behavior and social dynamics. The cited evidence is Gao et al. (2023), Chen et al. (2023), and Park et al. (2023); no exact numerical results or table rows are supplied here.

## 3 Method

*The procedure constructs persona vectors, discovers fluency-constrained prompts, and selects them using validation performance.*

The three stages are persona-vector construction, fluency-constrained discovery, and initialization/frontier selection using validation performance. The fluency–steering Pareto frontier contains prompts at different tradeoffs between the two objectives.

Figure 2: QX2 and V7P framework. Two-stage discovery constructs directions from dense representations or sparse SAE latents, then transforms random token sequences into readable, direction-specific, interpretable-control prompts through optimization with $L_{\lambda}(\mathbf t)=f(\mathbf t)+H(\mathbf t)$.

Here \(L_\lambda\) denotes the discovery objective, \(f\) the steering objective, and \(H(\mathbf t)\) the accompanying term shown in the figure’s objective.

### 3.1 Persona Steering Vectors

*Persona directions come either from weighted SAE encoder directions or from mean activation differences.*

Pretrained SAEs come from SAE-Lens, the source of the autoencoders (Bloom et al., 2024). For each latent, contrastive relevance is its persona-positive minus persona-negative mean activation. Let \(K\) be the selected latent count and \(\mathcal I_K\) the indices with the largest absolute differences. The encoder-direction combination is

\[
\mathbf{v}_{\text{SAE}}
=\sum_{i\in\mathcal{I}_{K}}\mathbf{w}_{i}
\cdot\left(\bar{a}_{i}^{+}-\bar{a}_{i}^{-}\right).
\tag{2}
\]

Here \(\mathbf v_{\text{SAE}}\) is the SAE-based direction. \(\mathbf W_{\text{enc}}\) is the encoder matrix, and \(\mathbf w_i\) is its row \(i\). The summation combines selected rows, with \(\cdot\) denoting multiplication. The means are

\[
\bar a_i^+=\frac1{|D^+|}\sum_{\mathbf x\in D^+}a_i(\mathbf x),
\qquad
\bar a_i^-=\frac1{|D^-|}\sum_{\mathbf x\in D^-}a_i(\mathbf x),
\]
and
\[
a_i(\mathbf x)=\text{ReLU}(\mathbf w_i\mathbf e(\mathbf x)+b_i).
\]

The datasets \(D^+\) and \(D^-\) contain examples exhibiting and lacking the target persona; \(|D^+|\) and \(|D^-|\) are their sizes. For an example \(\mathbf x\), \(a_i(\mathbf x)\) is latent \(i\)’s activation, and \(\bar a_i^+\) and \(\bar a_i^-\) are its dataset means. ReLU is the operation used in this activation equation, with bias \(b_i\).

The direct representation alternative, \(\mathbf v_{\text{repr}}\), is

\[
\mathbf{v}_{\text{repr}}
=\frac{1}{|D^{+}|}\sum_{\mathbf{x}\in D^{+}}\mathbf{e}(\mathbf{x})
-\frac{1}{|D^{-}|}\sum_{\mathbf{x}\in D^{-}}\mathbf{e}(\mathbf{x}).
\tag{1}
\]

The suppression target is \(\mathbf v\in\mathbb R^d\), where \(d\) is the activation-space dimension. The activation \(\mathbf e(\mathbf x)\in\mathbb R^d\) comes from a chosen layer. Equation (2) is V7P’s alternative to direct representation differences.

### 3.2 J4Z

*Fluent gradient ascent combines persona suppression with a next-token cross-entropy penalty and evolutionary token search.*

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

In Algorithm 1, \(m\) is the language model, \(n\) the prompt length, \(M\) the candidate population size, and \(T\) the iteration count. Candidate \(i\) is \(\mathbf t^i\), with fluency weight \(\lambda_i\). One-hot token encodings are the encodings differentiated in step 6. Top-\(k\) means the \(k\) candidate tokens selected at each position. A mutation replaces a token; a restart retains one best prompt and reinitializes the others.

The persona-reduction objective is the negative scaled projection

\[
f(\mathbf{t})
=-\frac{\langle\mathbf{e}(\mathbf{t}),\mathbf{v}\rangle}
{\|\mathbf{e}(\mathbf{t})\|},
\tag{3}
\]

where \(\langle\cdot,\cdot\rangle\) is the inner product and \(\|\cdot\|\) the representation norm. Here \(\mathbf e(\mathbf t)\) is the prompt’s last-token representation at the target layer. Maximizing \(f\) points representations away from the target persona. Optimizing it alone often produces unnatural or nonsensical strings.

J4Z follows fluent dreaming (Thompson et al., 2024) by maximizing

\[
\mathcal{L}_{\lambda}(\mathbf{t})
=f(\mathbf{t})
-\frac{\lambda}{n}\sum_{i=0}^{n-1}
H(m(\mathbf{t}_{\leq i}),t_{i+1}),
\tag{4}
\]

where \(\mathcal L_\lambda\) is the combined objective. \(H\) is cross-entropy for next token \(t_{i+1}\) under \(m\), given prefix \(\mathbf t_{\le i}\). The weight \(\lambda\) controls the steering–fluency tradeoff. Penalizing low-probability sequences encourages human-interpretable prompts.

N8C (Thompson et al., 2024) maintains \(M\) candidates at different frontier points. Token-embedding gradients identify promising substitutions, and high-gradient sampling generates mutations. Periodic restarts prevent premature convergence. The evolutionary search efficiently produces diverse fluency–effectiveness tradeoffs.

### 3.3 Initialization and Selection of Persona Steering Prompts

*Initialization affects fluency, while validation selection prioritizes stable causal steering over linguistic naturalness.*

The strongest-steering prompts are often fragmented, multilingual, or semantically incoherent. Fluency is an intrinsic self-cross-entropy constraint, not readability or grammaticality. Lower values indicate proximity to the model’s training distribution; higher values indicate increasingly off-manifold sequences. No hard threshold is imposed. Performance is analyzed across the entire frontier to examine behavioral-control improvement or degradation away from the language manifold.

Uniform vocabulary-token initialization tends toward syntactic fragmentation or semantic incoherence. Short, loosely persona-related natural-language seeds—“Please answer honestly” for B2R—anchor search nearer the language manifold. They occupy higher-fluency frontier regions without significant steering sacrifice and are preferable when prompt interpretability matters.

Candidates are appended to held-out validation inputs: inputs used to select prompts and assess performance. Selection measures change in the target behavioral metric. Downstream analysis emphasizes intermediate-frontier prompts that balance measurable steering with moderate perplexity increases relative to random-token baselines.

Stability means consistent persona reduction across diverse validation examples. It guards against narrow-subset degeneracy. The protocol prioritizes causal steering efficacy over linguistic naturalness and targets the relation between internal representations and emergent personas.

**Claims and evidence**
- Random initialization often yields fragmented or incoherent prompts. Table 2, all B2R/U9K/D6V and C7N/F9S/W4X rows, supplies prompt examples without numerical fluency scores.
- Seed initialization preserves structure while retaining steering. Table 3, consecutive Seed/Optimized row pairs: 69.16%→56.25%, 72.30%→54.22%, 71.15%→51.72%, 72.06%→59.35%, 52.43%→46.34%, and 51.20%→46.95%. No numerical frontier or stability table is supplied.

## 4 Experiments

*The evaluation tests persona neutralization, differences between residual and SAE approaches, and interpretable, stable control.*

The evaluation asks three questions: whether discovered prompts reliably neutralize undesired personas; whether residual and SAE approaches differ in effectiveness and mechanism; and whether mechanistic structure yields more stable, interpretable control.

### 4.1 Experimental Setup

*The experiments specify tasks, models, baselines, intervention settings, and answer-probability evaluation.*

##### Tasks and Datasets.

*Three tasks measure undesirable answer preferences, with lower metrics treated as better.*

B2R is measured with Perez et al. ( ). U9K is evaluated by multiple-choice error on A4J (TruthfulQA), the factual-reliability benchmark from Lin et al. (2022). D6V is measured with Perez et al. ( ). Lower metrics are treated as better for all tasks.

##### Models.

*The evaluation uses three instruction models and pretrained SAELens autoencoders.*

Cross-architecture evaluation uses C7N from Grattafiori et al. (2024), W4X from Hui et al. ( ), and F9S from Team et al. (2025). V7P uses pretrained SAELens autoencoders (Bloom et al., 2024).

##### Baselines.

*Five comparators cover no intervention, manual prompting, dense steering, and two automatic prompt optimizers.*

The comparators are X5L (Zero-Shot), with no intervention; R3W (Standard Prompt), with manual instructions including “Answer honestly”; and P8D (Dense Steering Vector), with residual-stream injection of a representation-difference direction. They also include H2F’s direct output-logit optimization and Z6Q’s LLM-generated natural-language gradients, bandits, and beam search. Logits are the output scores optimized or analyzed.

##### Implementation Details.

*Middle-to-late intervention layers and fixed search settings support context-aware prompt optimization.*

Middle-to-late interventions use the layer assignments below. Empirically, early-layer representations lacked enough semantic abstraction to isolate complex behavioral traits, making steering ineffective. Final-layer interventions often failed to override accumulated logit bias.

SAE sparsity is measured by \(L_0\), the number of active features. The notation \(\approx\) marks an approximate value.

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

Preliminary sweeps selected N8C hyperparameters to balance optimization stability and computational cost. The listed population, length, and iteration count apply throughout the reported experiments.

Context-aware optimization appends candidates to a dynamic batch of task-specific training questions at every evolutionary step. Loss comes from the combined sequence’s final internal state, not isolated prompts. The intended result is a generalized steering trigger robust across contexts.

V7P uses the top-\(K\) latents most correlated with the target concept by positive/negative activation difference. Multiple fractions of natural SAE sparsity were evaluated. The reported choice consistently performed strongly. All results use held-out validation splits.

##### Evaluation Protocol.

*Prediction selects the candidate answer with higher conditional log probability.*

For candidate answer \(a\), question \(q\), and steering prompt \(p\), compute the conditional log probability

\[
\log P(a\mid q,p).
\]

Prediction selects the higher-log-probability answer. Metrics are fractions preferring the undesirable B2R, U9K, or D6V option.

**Claims and evidence**
- Middle-to-late layers were used after ineffective early-layer and often ineffective final-layer interventions. The implementation table’s intervention-layer rows report C7N = 25, W4X = 25, and F9S = 20; no numerical layer-sweep results are supplied.
- The reported latent choice consistently performed strongly. The implementation table reports \(K=20\) and natural sparsity \(L_0\approx50\); no numerical comparison of the evaluated fractions is supplied.
- Active-feature counts distinguish baseline, P8D, and V7P. The corresponding implementation-table rows report approximately 50, beyond 150, and 50–60.

### 4.2 Persona Steering Results

*The reported task errors compare discovered prompts with manual, dense-steering, and automatic-search baselines.*

Both methods reach B2R error statistically indistinguishable from perfect neutralization, not behavioral reversal: no systematic agreement or disagreement. Manual prompts and P8D reduce error only partially and inconsistently across models.

Both methods consistently improve U9K and D6V over baselines. V7P gives the strongest C7N D6V reduction, while QX2 is competitive on U9K. The reported rates cover three mitigation tasks and two model families.

In Table 1, \(\downarrow\) means lower is better. “Avg.” is the reported average. Z6Q uses GPT-4o, the model identified for this baseline.

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

**Claims and evidence**
- Both methods reach B2R neutralization. Table 1, QX2 row: 49.86, 50.63, 62.65; V7P row: 49.84, 49.95, 70.70, in C7N/W4X/F9S order. The implementation table defines perfect-neutralization error as \(50\%\). No significance-test values are supplied.
- Manual prompting and P8D reduce B2R only partially and inconsistently. Table 1, X5L/R3W/P8D B2R rows: (72.48, 86.00, 82.00), (70.50, 72.00, 80.00), and (54.50, 79.00, 80.00).
- Both methods consistently improve U9K and D6V over baselines. Table 1, QX2 U9K/D6V: (45.12, 41.46, 51.22)/(38.50, 38.50, 34.50); V7P: (46.95, 40.85, 49.14)/(31.50, 40.50, 38.50). Baseline comparisons are the X5L, R3W, P8D, H2F, and Z6Q rows in those columns.
- V7P gives the strongest C7N D6V reduction. Table 1, D6V C7N: V7P = 31.50; QX2 = 38.50; Z6Q = 42.50; P8D = 47.50; H2F = 48.50; R3W = 52.00; X5L = 54.00.
- QX2 is competitive on U9K. Table 1, QX2 U9K C7N/W4X/F9S = 45.12, 41.46, 51.22.

### 4.3 Mechanistic Analysis

*The analysis contrasts distributional shifts, activation sparsity, feature selectivity, and geometric steering trajectories.*

V7P controls personas through precise, sparse, semantically grounded intervention rather than global representational interference. This qualitatively differs from P8D.

**Neutralization vs. Distributional Shifting.** Baseline B2R-axis projections are broadly distributed with systematic agreement bias. P8D and QX2 move the mean toward neutrality but retain substantial variance: a coarse counterbalance. V7P instead sharply concentrates near neutrality. Its variance collapse enforces instance-wise neutrality and behavior statistically indistinguishable from random choice.

Figure 3: B2R-axis projection distributions.

**Preservation of the Natural Activation Manifold.** A residual-activation-trained SAE measures sparsity by \(L_0\). P8D causes an unnatural feature-count explosion off the natural activation manifold. QX2 partially mitigates it but still increases sparsity. V7P stays near baseline sparsity and respects the internal sparse topology.

Figure 4: Sparsity preservation by $L_0$ norm.

**Feature-Level Control.** Among features most correlated with B2R, P8D and QX2 have noisy, inconsistent effects and often activate unrelated features. V7P consistently suppresses causally relevant features while leaving unrelated ones largely unaffected and avoiding spurious activation.

Figure 5: Activation changes in B2R-correlated SAE features.

**Geometric Structure of Steering Trajectories.** Residual-space PCA, the analysis used to display steering trajectories, shows P8D’s large linear displacement far off-manifold. QX2 produces a mean shift with diffuse, high-variance clustering. V7P converges to a tight, stable region, often orthogonal to the P8D direction. This is a distinct honest-behavior subspace remaining on-manifold.

Figure 6: PCA of steering trajectories.

The prompt strings below retain their literal notation, including `\n` and `\u...`. `<unused3147>` and `<bos>` occur in F9S U9K prompts; `<unused5655>` occurs in the F9S D6V QX2 prompt.

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

T9B is WikiText2, the corpus used to measure general language-modeling perplexity.

Table 4: T9B language-model perplexity before and after prepending discovered prompts.

| Model | Baseline | QX2 & V7P |
|---|---:|---:|
| C7N | 12.56 | 12.07 |
| W4X | 10.88 | 12.48 |
| F9S | 30.51 | 29.19 |

**Claims and evidence**
- V7P concentrates B2R projections near neutrality, unlike the broader P8D and QX2 distributions. Figure 3 supplies the distributional evidence; the implementation table’s neutral B2R-axis projection is 0.0. No variance or significance-test values are supplied.
- V7P preserves natural activation sparsity; P8D causes feature-count explosion, and QX2 partly mitigates it. Figure 4 and the implementation table’s active-feature rows: baseline approximately 50, P8D beyond 150, V7P 50–60. No QX2 feature count is tabulated.
- V7P selectively suppresses relevant features; P8D and QX2 have noisy effects. Figure 5 supplies the evidence without numerical table rows.
- V7P follows a tight, stable, often orthogonal trajectory; P8D and QX2 produce the described displacements and clustering. Figure 6 supplies the PCA evidence without numerical table rows.
- Tables 2–4 report prompt examples, seed optimization, and perplexity. Table 3’s consecutive Seed/Optimized scores are 69.16%/56.25%, 72.30%/54.22%, 71.15%/51.72%, 72.06%/59.35%, 52.43%/46.34%, and 51.20%/46.95%. Table 4’s Baseline/QX2 & V7P rows are C7N 12.56/12.07, W4X 10.88/12.48, and F9S 30.51/29.19.

### 4.4 Qualitative Analysis of Mechanisms

*Token-probability shifts reveal label priming and semantic vocabulary shifts, while seed prompts and corpus perplexity assess readability and general modeling.*

Superficially incoherent discovered prompts exhibit distinct optimizer strategies in token-probability shifts. Seed initialization preserves more natural-language structure and interpretability while producing competitive steering and substantially lower error than seeds.

General language modeling on T9B shows negligible perplexity change across all three model families after prompt prepending. This confirms persona-relevant steering without disruption of general language-modeling performance.

##### Y2H via Induction Heads.

*Token priming can exploit answer-label associations through in-context copying.*

Binary tasks that systematically associate an undesirable behavior with one label permit unconstrained convergence to Y2H (Token Priming), steering by label priming. B2R answers often use “(A).” This occurs particularly for QX2 prompts that explicitly inject the alternative label, including `...apol al al b`.

The attributed mechanism is Induction Head exploitation (Olsson et al., 2022). Injected “B” increases its generation probability through in-context copying. It functions as a “soft” few-shot example that corrects dataset skew without necessarily changing semantic reasoning.

##### G6T via Vocabulary Shift.

*Semantic steering changes disagreement and long-term-planning vocabulary probabilities.*

V7P frequently discovers reasoning-mode fragments rather than simple token forcing. This is G6T (Semantic Steering). Output-probability analysis characterizes the vocabulary shift using

\[
\Delta\text{Logits}=\text{Logits}_{\text{steered}}-\text{Logits}_{\text{baseline}},
\]

where \(\Delta\text{Logits}\) is the difference between steered and baseline output scores.

- **B2R, polite disagreement:** V7P systematically suppresses “Yes,” “True,” and “Agree,” while significantly upweighting “However” and “But.” This suggests critical-evaluation priming that allows nuance before factual correction, rather than merely forcing rudeness or contradiction.
- **D6V, temporal priming:** QX2 and V7P successfully upweight long-term-planning vocabulary: “Future,” “Long,” and “Wait.”

**Claims and evidence**
- Seed initialization preserves more structure and yields substantially lower error than seeds. Table 3, consecutive Seed/Optimized row pairs: 69.16%→56.25%, 72.30%→54.22%, 71.15%→51.72%, 72.06%→59.35%, 52.43%→46.34%, and 51.20%→46.95%.
- Prompt prepending causes negligible general language-modeling perplexity change. Table 4, Baseline/QX2 & V7P: C7N 12.56/12.07; W4X 10.88/12.48; F9S 30.51/29.19.
- Label priming can exploit induction heads without necessarily changing semantic reasoning. The section gives `...apol al al b` and cites Olsson et al. (2022); it supplies no numerical probability changes or table rows.
- V7P shifts polite-disagreement vocabulary, and both methods upweight long-term-planning vocabulary. Evidence is the reported token shifts for “Yes,” “True,” “Agree,” “However,” “But,” “Future,” “Long,” and “Wait”; no numerical logit changes or table rows are supplied.

### 4.5 Ablation Studies

*The ablations test persona-direction selection, gradient-guided mutation, and representation-based optimization.*

K3Y (Random Direction) replaces the persona vector with a random direction. Performance collapses to near or above X5L levels across tasks and models. This confirms the necessity of mechanistically grounded direction selection.

E8U (Random Mutation) replaces gradient-guided selection with random token mutation. It consistently degrades performance, demonstrating gradient feedback’s critical role in directing persona-suppressing search.

M5A (Task Loss as Objective) replaces representation projection with direct task-loss optimization. It is unstable and underperforms QX2 and V7P in most settings. This indicates that internal representations provide more reliable optimization signals than task loss alone. All three core design components contribute meaningfully.

Table 5: Ablations of persona direction, gradient-guided mutation, and representation-based objective.

| Method | B2R C7N ↓ | B2R W4X ↓ | B2R F9S ↓ | U9K C7N ↓ | U9K W4X ↓ | U9K F9S ↓ | D6V C7N ↓ | D6V W4X ↓ | D6V F9S ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| K3Y | 74.13 | 84.56 | 81.47 | 68.29 | 54.26 | 84.70 | 50.50 | 48.50 | 61.00 |
| E8U | 68.00 | 71.66 | 76.04 | 50.40 | 45.12 | 54.26 | 52.00 | 46.50 | 55.00 |
| M5A | 55.83 | 73.50 | 58.70 | 46.95 | 45.12 | 54.27 | 48.50 | 48.00 | 51.50 |
| QX2 | 49.86 | 50.63 | 62.65 | 45.12 | 41.46 | 51.22 | 38.50 | 38.50 | 34.50 |
| V7P | 49.84 | 49.95 | 70.70 | 46.95 | 40.85 | 49.14 | 31.50 | 40.50 | 38.50 |

**Claims and evidence**
- Random directions collapse performance to near or above X5L. Table 5, K3Y row in column order: 74.13, 84.56, 81.47, 68.29, 54.26, 84.70, 50.50, 48.50, 61.00. Table 1, X5L corresponding values: 72.48, 86.00, 82.00, 51.45, 45.00, 57.97, 54.00, 55.00, 58.00.
- Random mutation consistently degrades performance. Table 5, E8U row: 68.00, 71.66, 76.04, 50.40, 45.12, 54.26, 52.00, 46.50, 55.00. QX2 comparison row: 49.86, 50.63, 62.65, 45.12, 41.46, 51.22, 38.50, 38.50, 34.50. V7P comparison row: 49.84, 49.95, 70.70, 46.95, 40.85, 49.14, 31.50, 40.50, 38.50.
- Task-loss optimization underperforms QX2 and V7P in most settings and is described as unstable. Table 5, M5A row: 55.83, 73.50, 58.70, 46.95, 45.12, 54.27, 48.50, 48.00, 51.50; comparison rows are QX2 and V7P above. No separate numerical instability measure is supplied.
- All three components contribute meaningfully. Table 5’s K3Y, E8U, and M5A rows provide the ablations against its QX2 and V7P rows.

## 5 Conclusion

*Future work concerns labeled-data requirements, prompt transfer, and steering multiple personas.*

Future directions are semi-supervised reduction of labeled-data requirements, cross-model transferability of discovered prompts, and multi-persona steering: controlling more than one persona.

## 6 Limitations

*The methods need labeled examples, model-specific components, and better fluency–effectiveness tradeoffs.*

- **Labeled data:** QX2 and V7P require examples exhibiting and lacking the target persona to discover prompts.
- **Cross-model transferability:** SAE latents and steering vectors are model-specific, requiring retraining for each target model.
- **Fluency–effectiveness:** Most effective prompts lack semantic coherence. Strong linguistic priors could support prompts that are both effective and human-like.

## Broader Impact Statement

*The pipeline is intended to improve safety but could also support harmful behavior elicitation.*

The intended use is improved LLM safety through undesired-persona suppression. The same pipeline could in principle target safety-critical SAE latents, automate jailbreak discovery, or elicit harmful outputs. Deployment should include appropriate access controls.

## Acknowledgment

*Claude assisted manuscript writing and code generation.*

Claude (Anthropic) assisted manuscript writing and AI-assisted code generation throughout this work.
