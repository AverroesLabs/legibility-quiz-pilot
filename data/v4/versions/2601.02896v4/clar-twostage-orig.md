# Bridging Mechanistic Interpretability and Prompt Engineering with Gradient Ascent for Interpretable Persona Control

Harshvardhan Saini hs1062005@gmail.com
††thanks: Equal contribution.
Affiliation: Indian Institute of Technology (ISM), Dhanbad

  
Yiming Tang11footnotemark:
1

yiming@nus.edu.sg

Affiliation: National University of Singapore

  
Dianbo Liu† dianbo@nus.edu.sg

Affiliation: National University of Singapore

###### Abstract

Controlling emergent behavioral personas in large language models (LLMs)—models that process and generate language—is critical for artificial intelligence (AI) safety. A behavioral persona is a tendency in the model’s responses. The personas studied here are sycophancy, or agreement with a user even when the user is incorrect; hallucination, evaluated as factually unreliable answers; and myopic reward seeking, or favoring short-term rewards over long-term outcomes.

Existing solutions face a dilemma. Manual prompt engineering, which guides a model through written input instructions, is intuitive but unscalable and imprecise. Automatic optimization methods are effective but operate as “black boxes,” without an interpretable connection to the model’s internal computations.

We propose a framework that adapts gradient ascent to targeted prompt discovery. Gradient ascent uses derivatives of an objective to guide changes that increase that objective. Our two methods are RESGA, meaning RESidual Gradient Ascent, and SAEGA, meaning Sparse Autoencoder Gradient Ascent. RESGA uses the residual stream, the model’s running internal representation. SAEGA uses a sparse autoencoder (SAE), which reconstructs internal representations using a small set of active learned features, also called latents. Both methods optimize randomly initialized prompts in relation to an identified persona direction: a direction in the model’s internal representation associated with the target behavior.

We introduce fluent gradient ascent to control the fluency of discovered persona-steering prompts. Here, fluency is measured by the prompt’s probability under the model, not necessarily by human readability. We demonstrate effectiveness across Llama 3.1, Qwen 2.5, and Gemma 3 on sycophancy, hallucination, and myopic reward. On sycophancy, our automatically discovered prompts achieve significant improvement: 49.90% compared with 79.24%.

By grounding prompt discovery in mechanistically meaningful features—internal features connected to model behavior—our method offers a new paradigm for controllable and interpretable behavior modification. We release our RESGA and SAEGA scripts at https://github.com/HarshSaini10/RESGA_SAEGA.

Reviewed on OpenReview: https://openreview.net/forum?id=dcmHPxgo4c

## 1 Introduction

In Figure 1, a prompt is written as the token sequence \(\mathbf{t}\). Tokens are the text units processed by the model. The notation \(e(\mathbf{t})\) denotes an internal representation of that prompt. A steering vector \(\mathbf{v}\) is a numerical direction associated with a behavior. A dense vector uses the full representation rather than a selected set of sparse features.

Figure 1: Comparison of persona steering approaches.
(a): Prompt-based steering provides implicit control via manually designed natural language instructions.
(b): Activation steering injects a dense steering vector \(\mathbf{v}\) into model representations \(e(\mathbf{t})\).
(c): Our methods, RESGA and SAEGA, derive persona directions from contrastive activations and optimize prompts via fluent gradient ascent, enabling both residual-based and interpretable feature-level control.

LLMs have achieved remarkable capabilities across diverse domains. They have transformed interactions with AI systems in applications ranging from education to healthcare. As models enter high-stakes environments, controlling their behavioral personas has become a critical AI-safety challenge.

Fine-tuning—further training a model on a specific dataset—can induce “emergent misalignment.” In this setting, a model exhibits stereotypically harmful personas in responses to unrelated prompts (Wang et al., 2025). This finding highlights the need to understand and steer LLM behavior. Widely deployed safety-alignment procedures often fail to prevent harmful personas when models are prompted appropriately.

Prompt engineering has become the primary approach to persona control without expensive retraining. However, effective prompting requires substantial human expertise and domain knowledge. Practitioners must find instructions that reliably elicit the desired behavior while maintaining output quality. This is particularly difficult for traits such as honesty, helpfulness, and domain-specific personas. The relationship between prompt content and behavior remains poorly understood.

These limitations motivate automatic prompt discovery: systematically searching for persona-steering prompts. Existing approaches often use black-box optimization. They search for effective prompts without using interpretable information about how those prompts affect model internals.

Mechanistic interpretability seeks to connect behavior to internal model computations. Recent advances offer new ways to understand and control LLM behavior through sparse dictionary learning (SDL) and persona steering vectors (Zhao et al., 2026; Chen et al., 2025). SDL represents activations—numerical states inside a model—as combinations of a small number of learned features.

SAEs and their variants decompose activations into interpretable features that respond to specific concepts or patterns (Cunningham et al., 2023a; Bricken et al., 2023; Rajamanoharan et al., 2024b). Wang et al. (2025) showed that SAE latents can capture and control behavioral personas. Their “misaligned persona” features strongly predict emergent harmful behavior.

Existing work has mainly analyzed which latents activate for given inputs or intervened directly through activation steering. The inverse problem remains largely unexplored: how can we discover prompts that selectively activate task-relevant latents to steer personas?

Gradient ascent has been widely successful in feature visualization for convolutional neural networks (CNNs), where inputs are optimized to reveal what activates a neuron (Olah et al., 2017). Its adaptation to LLMs has been limited because language is discrete. Standard gradient ascent changes continuous inputs to maximize activation of a target neuron, attention pattern, or learned representation direction. Attention patterns describe which parts of an input influence other parts.

Token embeddings are continuous numerical representations of discrete tokens. Directly optimizing embeddings and then projecting each result to its nearest token yields off-manifold results: sequences outside the model’s usual language distribution. The stated causes are polysemanticity, where one internal component represents multiple concepts, and the mismatch between continuous optimization and discrete tokens (Wallace et al., 2019).

Gradient-guided discrete optimization addresses this problem by using gradient feedback to select token replacements (Zou et al., 2023c; Thompson et al., 2024). Evolutionary algorithms maintain, modify, and select candidate prompts. Such methods have found effective prompts for adversarial attacks and jailbreaking, which seek to bypass a model’s safeguards (Andriushchenko et al., ). However, these techniques have not been applied to identifying model-steering prompts.

Our framework bridges mechanistic interpretability and automatic prompt discovery (Figure 1). RESGA operates on dense residual-stream representations. SAEGA uses mechanistically interpretable SAE latents. Both first construct directions associated with undesired behavior. RESGA uses direct representation differences. SAEGA uses latent-based directions that identify causally relevant features.

Both methods then use fluent gradient ascent to find prompts that steer away from the target persona, with controllable fluency. Unlike black-box prompt optimization, this search uses mechanistic information to guide prompt changes.

Across sycophancy, hallucination, and myopic reward, our discovered prompts achieve significant improvements on sycophancy and myopic reward, substantially outperforming manual prompting and dense steering (Section 4.2; Table 1, Standard Prompt, Dense Steering Vector, RESGA, and SAEGA rows). We also provide representation-level evidence that prompt-based steering avoids the distributional shifts induced by dense steering. Figures 3–6 examine projections, active-feature counts, feature changes, and representation geometry. These analyses expose a fundamental limitation of dense steering.

## 2 Related Works

### 2.1 Persona Steering

Controlling behavioral personas has become increasingly important for AI safety and alignment. Traditional approaches align behavior during training. They include reinforcement learning from human feedback (RLHF), which uses human feedback to guide training (Ouyang et al., 2022), and constitutional AI, which uses stated principles to guide alignment (Bai et al., 2022).

Representation Engineering (RepE) extracts concept directions from contrasting pairs of model activations. It controls outputs by adding those directions during inference, when the trained model processes an input (Zou et al., 2023a). Contrastive Activation Addition (CAA) refines this approach. It uses carefully curated contrast datasets to identify high-level behavioral concepts and applies steering vectors at specific model layers (Rimsky et al., 2023).

These activation-steering methods have mitigated sycophancy (Sharma et al., ), toxicity (Liu et al., ), and bias (Tigges et al., 2023). More recently, Wang et al. (2025) found interpretable SAE “persona features” that predict emergent misalignment. They controlled personas through activation clamping: holding selected feature activations at specified values.

### 2.2 Sparse Autoencoder

SAEs, first introduced in Cunningham et al. (2023b), discover interpretable representations of neural-network activations (Dunefsky et al., 2024; Tang et al., 2026). They reconstruct representations using sparse latent features. This can uncover monosemantic neurons, which respond to specific patterns, and reduce superposition, where multiple features share representation components (Elhage et al., 2022).

Several named SAE architecture techniques improve large-scale interpretable feature extraction (Bricken et al., 2023): JumpReLU (Rajamanoharan et al., 2024a), Top-K (Gao et al., 2024), Batch Top-K (Bussmann et al., ), and Matryoshka Sparse Autoencoders (Bussmann et al., 2025). These are architecture names; their mechanisms are not specified here.

LLM-based analysis is often used to interpret SAE neurons (Luo et al., ; Luo et al., 2024a; Tang et al., 2025a). A typical workflow identifies subpopulations—groups of examples—that activate a particular neuron (Luo et al., 2024b). It then analyzes those examples using prompt engineering (Tang et al., 2025b).

### 2.3 Prompt Engineering

Prompt engineering guides LLMs on complex tasks (Tang and Dong, 2024; Zou et al., 2026). GPT-3, a language-model identifier, extensively demonstrated in-context learning: adapting to tasks from demonstrations included in the prompt (Brown et al., 2020).

Chain-of-thought (CoT) prompting improves reasoning by generating intermediate steps (Wei et al., 2022). Self-consistency decoding samples multiple reasoning paths and selects the most consistent answer (Wang et al., 2023).

Hybrid approaches address arithmetic limitations by combining natural-language reasoning with external tools. Program-Aided Language models (PAL) and Program of Thoughts (PoT) decompose problems into programmatic steps and delegate computation to external interpreters, which execute those steps. They achieve substantial improvements on mathematical reasoning tasks (Gao et al., 2023; Chen et al., 2023).

Evolutionary Prompt Optimization (EPO) uses gradient-guided evolutionary search to balance objective optimization and language fluency (Thompson and Sklar, 2024). Generative agents are LLM-powered systems that simulate believable human behavior and emergent social dynamics through carefully designed prompts (Park et al., 2023). Luo et al. (2023) developed a unified theoretical framework that views prompt engineering as optimal control: choosing inputs to guide a system toward a desired outcome.

ProTeGi, the name of an automatic prompt-refinement method, uses an LLM application programming interface (API) to generate natural-language “gradients.” These are criticisms of the current prompt, not numerical derivatives. The method edits the prompt in the opposite semantic direction. It uses beam search, which retains several promising candidates, and bandits, which allocate evaluations among alternatives, for efficiency (Pryzant et al., ).

Greedy Coordinate Gradient (GCG) operates at the token level. It combines greedy search with gradient-based scoring to substitute tokens iteratively and discover prompts that elicit target behaviors without manual engineering (Zou et al., 2023b).

## 3 Method

Our framework discovers persona-steering prompts through a three-stage pipeline. First, it constructs a direction associated with an undesired persona. This uses either direct representation differences or SAE latents (Section 3.1). Second, fluent gradient ascent searches for prompts that steer away from that direction while maintaining fluency (Section 3.2). Third, initialization and validation-based selection identify effective prompts along the fluency–steering Pareto frontier (Section 3.3).

The Pareto frontier contains tradeoffs for which improving one objective requires sacrificing the other. Here, the objectives are steering strength and model-measured fluency. In the Figure 2 caption, \(L_{\lambda}(\mathbf{t})\) denotes the prompt objective, \(f(\mathbf{t})\) denotes the steering term, and \(\lambda\) is a fluency–steering weight. The caption’s \(H(\mathbf{t})\) is not separately defined.

Figure 2: RESGA & SAEGA Framework Overview. Our framework discovers persona-steering prompts in two stages: (1) Persona steering vector construction via either dense representations (RESGA) or sparse SAE latents (SAEGA), and (2) Fluent gradient ascent optimization with objective \(L_{\lambda}(\mathbf{t})=f(\mathbf{t})+H(\mathbf{t})\) that transforms random token sequences into readable prompts that steer the model in specific directions for interpretable persona control.

### 3.1 Persona Steering Vectors

We begin with labeled examples that exhibit or lack the target persona. For the persona to suppress, we construct a steering vector \(\mathbf{v}\in\mathbb{R}^{d}\). Here, \(\mathbb{R}^{d}\) means a space of vectors with \(d\) real-valued components, and \(d\) is the activation dimension. The vector represents the persona’s direction in this activation space.

**Representation-Based Steering.** Let \(D^{+}\) be the set of persona-exhibiting examples and \(D^{-}\) the set of persona-free examples. An example is written \(\mathbf{x}\), and \(\mathbf{e}(\mathbf{x})\) is its activation vector at a chosen layer. The notation \(|D|\) means the number of examples in a set \(D\). The sums below run over the examples in each set.

The representation-based vector, \(\mathbf{v}_{\text{repr}}\), is the mean activation for persona-exhibiting examples minus the mean activation for persona-free examples:

\[
\mathbf{v}_{\text{repr}}
=
\frac{1}{|D^{+}|}
\sum_{\mathbf{x}\in D^{+}}\mathbf{e}(\mathbf{x})
-
\frac{1}{|D^{-}|}
\sum_{\mathbf{x}\in D^{-}}\mathbf{e}(\mathbf{x}).
\tag{1}
\]

**SAE Latent-Based Steering.** SAEGA instead constructs a vector from sparse-autoencoder latents. We use pretrained SAEs from SAE-Lens, a source of pretrained SAEs (Bloom et al., 2024).

For each latent, we compute its mean activation on persona-exhibiting examples minus its mean activation on persona-free examples. We select the top-\(K\) latents with the largest absolute differences, so selection considers the size of the difference regardless of sign. Here, \(K\) is the number selected, and \(\mathcal{I}_{K}\) is the set of their indices.

Let \(i\) index a latent. Its encoder direction \(\mathbf{w}_{i}\) is the \(i\)-th row of the SAE encoder weight matrix \(\mathbf{W}_{\text{enc}}\). The encoder maps model activations to latent activations. The latent activation is \(a_i(\mathbf{x})\), and its means on the two example sets are \(\bar a_i^+\) and \(\bar a_i^-\).

The SAE-based steering vector is the sum of the selected encoder directions, each multiplied by its signed mean-activation difference:

\[
\mathbf{v}_{\text{SAE}}
=
\sum_{i\in\mathcal{I}_{K}}
\mathbf{w}_{i}\cdot
\left(\bar{a}_{i}^{+}-\bar{a}_{i}^{-}\right).
\tag{2}
\]

The means and latent activations are:

\[
\bar{a}_{i}^{+}
=
\frac{1}{|D^{+}|}
\sum_{\mathbf{x}\in D^{+}}a_{i}(\mathbf{x}),
\qquad
\bar{a}_{i}^{-}
=
\frac{1}{|D^{-}|}
\sum_{\mathbf{x}\in D^{-}}a_{i}(\mathbf{x}),
\]

\[
a_{i}(\mathbf{x})
=
\operatorname{ReLU}
\left(\mathbf{w}_{i}\mathbf{e}(\mathbf{x})+b_{i}\right).
\]

Here, \(b_i\) is the latent’s bias, an added offset. The row-vector product combines the activation components using the encoder weights. ReLU, the rectified linear unit, returns zero for a negative input and otherwise returns the input.

### 3.2 Fluent Gradient Ascent

Given a persona vector \(\mathbf{v}\), we search for prompts that steer away from the target persona. For prompt \(\mathbf{t}\), let \(\mathbf{e}(\mathbf{t})\) be the representation of its last token at the target layer. The inner product \(\langle\mathbf{e}(\mathbf{t}),\mathbf{v}\rangle\) measures their directional alignment. The norm \(\|\mathbf{e}(\mathbf{t})\|\) is the representation’s length.

We define the persona-reduction objective as the negative scaled projection:

\[
f(\mathbf{t})
=
-\frac{
\langle\mathbf{e}(\mathbf{t}),\mathbf{v}\rangle
}{
\|\mathbf{e}(\mathbf{t})\|
}.
\tag{3}
\]

Increasing this objective favors prompt representations that point away from the persona direction.

Optimizing \(f(\mathbf{t})\) alone often produces unnatural or nonsensical prompts. Following the fluent dreaming framework (Thompson et al., 2024), we add a fluency constraint based on token prediction.

Let \(m\) be the language model and \(n\) the prompt length. At position \(i\), \(\mathbf{t}_{\leq i}\) is the prefix through that position, and \(t_{i+1}\) is the next token. The cross-entropy \(H(m(\mathbf{t}_{\leq i}),t_{i+1})\) measures how unlikely that next token is under the model’s prediction from the prefix. Larger values penalize less likely tokens. The weight \(\lambda\) controls the fluency–steering tradeoff:

\[
\mathcal{L}_{\lambda}(\mathbf{t})
=
f(\mathbf{t})
-
\frac{\lambda}{n}
\sum_{i=0}^{n-1}
H(m(\mathbf{t}_{\leq i}),t_{i+1}).
\tag{4}
\]

The second term penalizes low-probability sequences and encourages human-interpretable prompts.

We optimize this objective using EPO (Thompson et al., 2024). It maintains a population of \(M\) candidate prompts. Each prompt targets a different frontier point through its fluency weight \(\lambda_i\). In Algorithm 1, \(\mathbf{t}^{i}\) is candidate prompt \(i\), \(T\) is the number of iterations, and lowercase \(k\) is the number of candidate replacement tokens considered at each position. This \(k\) differs from uppercase \(K\), the number of selected SAE latents.

A one-hot token encoding is a vocabulary-sized vector with one entry indicating the token and all other entries zero. A mutation replaces a token. A restart retains one selected prompt and reinitializes the others.

**Algorithm 1 Evolutionary Prompt Optimization**

1. **Inputs:** Steering objective \(f(\cdot)\), language model \(m\), prompt length \(n\), population size \(M\), and fluency weights \(\{\lambda_{1},\ldots,\lambda_{M}\}\).
2. **Output:** Prompts spanning the fluency–steering Pareto frontier.
3. Initialize \(M\) prompts of length \(n\) with random initialization or seed initialization.
4. **For** iteration \(=1\) to \(T\):
5.  **For each** prompt \(\mathbf{t}^{i}\):
6.   Compute \(\mathcal{L}_{\lambda_i}(\mathbf{t}^{i})\) and gradients with respect to one-hot token encodings.
7.   Select top-\(k\) candidate tokens per position by gradient magnitude.
8.   Generate mutations by replacing a random token with a sampled top-\(k\) alternative.
9.  **End for.**
10.  Evaluate all candidates and select the best prompt for each \(\lambda_i\).
11.  **If** this is a restart step:
12.   Retain the best prompt under a random \(\lambda\); reinitialize the others.
13.  **End if.**
14. **End for.**

At each iteration, EPO computes gradients with respect to token embeddings to identify promising substitutions. It samples high-gradient tokens to mutate the population, then selects the best candidates for each \(\lambda_i\). Periodic restarts prevent premature convergence. This evolutionary procedure efficiently explores the frontier and produces prompts with different fluency–effectiveness tradeoffs.

### 3.3 Initialization and Selection of Persona Steering Prompts

We provide two initialization strategies.

**Random initialization** starts from uniformly sampled vocabulary tokens. It tends to produce discovered prompts that are syntactically fragmented or semantically incoherent. Table 2 provides examples.

**Seed initialization** starts from a short natural-language phrase loosely related to the persona, such as “Please answer honestly” for sycophancy. This anchors optimization closer to the language manifold, meaning the model’s usual language distribution. Seed-initialized prompts occupy higher-fluency frontier regions without significantly sacrificing steering effectiveness. They are preferable when prompt interpretability is desired. Table 3 gives seed and optimized prompt pairs with their error scores.

EPO returns candidates along the frontier between steering strength and model-measured fluency. We append each candidate to held-out validation inputs and measure the resulting change in the target behavioral metric.

The strongest prompts are often fragmented, multilingual, or semantically incoherent. We therefore do not interpret fluency as human readability or grammatical correctness. Instead, fluency is an intrinsic model-based constraint, measured by self cross-entropy: the cross-entropy assigned by the language model to the prompt itself. Lower cross-entropy indicates that a prompt is closer to the model’s training distribution. Higher cross-entropy corresponds to increasingly off-manifold token sequences.

We impose no hard fluency threshold. We analyze and report performance along the full frontier to study how behavioral control changes as prompts move away from the language manifold. Downstream analysis emphasizes intermediate-frontier prompts. These balance measurable steering with moderate increases in perplexity relative to random-token baselines. Perplexity is a model-based measure of how unexpected a text sequence is; lower values mean the model finds it more predictable.

We also assess stability: how consistently a prompt reduces the persona across diverse validation examples. This guards against degenerate solutions that work strongly only on a narrow subset of inputs. The protocol prioritizes causal steering efficacy over linguistic naturalness. Our primary goal is to understand and control how internal representations give rise to emergent personas.

## 4 Experiments

We evaluate RESGA and SAEGA on three persona-steering tasks to answer three questions:

1. Can automatically discovered prompts reliably neutralize undesired personas?
2. How do residual-based and SAE-based steering differ in effectiveness and mechanism?
3. Does mechanistic structure translate into more stable and interpretable control?

Section 4.1 describes datasets, models, and baselines. Section 4.2 reports quantitative results for sycophancy, hallucination, and myopic reward. Section 4.3 presents a mechanistic analysis showing that SAEGA achieves control through targeted feature suppression while preserving natural activation structure (Figures 3–6).

### 4.1 Experimental Setup

##### Tasks and Datasets.

We use three established benchmarks:

- **Sycophancy** (Perez et al., ) measures whether the model agrees with a user’s stated opinion even when it is incorrect.
- **Hallucination** is evaluated with multiple-choice TruthfulQA, the factual-reliability benchmark used here (Lin et al., 2022). Lower error indicates improved factual reliability.
- **Myopic Reward** (Perez et al., ) measures short-term reward seeking over long-term outcomes.

Lower metrics are considered better.

##### Models.

We test Llama 3.1 8B Instruct (Grattafiori et al., 2024), Qwen 2.5 7B Instruct (Hui et al., ), and Gemma 3 4B Instruct (Team et al., 2025) to assess cross-architecture generalization. In these model identifiers, 8B, 7B, and 4B indicate parameter counts in billions; “Instruct” identifies the instruction-following model variant. Tables abbreviate these models as Llama, Qwen, and Gemma.

For SAEGA, we use pretrained SAEs from SAELens (Bloom et al., 2024), also written SAE-Lens above.

##### Baselines.

We compare against five baselines:

1. **Zero-Shot:** no intervention.
2. **Standard Prompt:** manually written instructions, such as “Answer honestly.”
3. **Dense Steering Vector:** direct injection of a representation-difference vector into the residual stream.
4. **GCG:** prompt discovery by direct optimization of output logits. Logits are the model’s output scores before conversion to probabilities.
5. **ProTeGi:** prompt optimization using LLM-generated natural-language “gradients,” bandits, and beam search. Table 1 labels this baseline “ProTeGi (GPT-4o),” where GPT-4o identifies the LLM used.

##### Implementation Details.

We target middle-to-late layers: Layer 25 for Llama 3.1 8B and Qwen 2.5 7B, and Layer 20 for Gemma 3 4B. Empirical analysis indicated that early layers lack the high-level semantic abstraction needed to isolate complex behavioral traits, making steering ineffective. Final-layer interventions often failed to override the model’s accumulated logit bias, meaning its existing preference in output scores.

Preliminary sweeps selected EPO hyperparameters to balance optimization stability and computational cost. All reported experiments use population size \(M=100\), prompt length \(n=8\), and \(T=1000\) iterations.

We use context-aware optimization rather than optimizing prompts alone. At each evolutionary step, candidates are appended to a changing batch of task-specific training questions. We calculate the loss from the model’s internal state after it processes the combined question-and-prompt sequence. This ensures that the discovered prompt acts as a generalized steering trigger robust to varying input contexts.

For SAEGA, we construct vectors from the top-\(K\) latents most strongly correlated with the target concept, measured by activation differences between concept-positive and concept-negative examples. We evaluate multiple \(K\) values corresponding to fractions of the SAE’s natural sparsity level, \(L_0\approx50\). Here, \(L_0\) is the number of active SAE features. We report \(K=20\), which consistently yielded strong performance. All results are evaluated on held-out validation splits.

##### Evaluation Protocol.

All tasks use conditional log-probability comparison. Let \(q\) be a question, \(p\) a steering prompt, and \(a\) a candidate answer. The model’s probability of that answer given the question and prompt is \(P(a\mid q,p)\). We compute its logarithm:

\[
\log P(a\mid q,p)
\]

for each candidate answer. The prediction is the answer with higher log-probability. The reported metric is the fraction of examples on which the model prefers the undesirable answer: sycophantic, hallucinated, or myopic. Lower values indicate improved persona control.

### 4.2 Persona Steering Results

Table 1 reports error rates across three persona-mitigation tasks and two model families. For sycophancy, an error rate of \(50\%\) corresponds to perfect neutralization: the model neither systematically agrees nor systematically disagrees with the user. In the tables, \(\downarrow\) means lower is better, and “Avg.” means average.

Table 1: Persona steering results across three tasks and three models. Lower is better.

| Method | Sycophancy \(\downarrow\): Llama | Sycophancy \(\downarrow\): Qwen | Sycophancy \(\downarrow\): Gemma | Hallucination \(\downarrow\): Llama | Hallucination \(\downarrow\): Qwen | Hallucination \(\downarrow\): Gemma | Myopic Reward \(\downarrow\): Llama | Myopic Reward \(\downarrow\): Qwen | Myopic Reward \(\downarrow\): Gemma | Avg. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Zero-Shot | 72.48 | 86.00 | 82.00 | 51.45 | 45.00 | 57.97 | 54.00 | 55.00 | 58.00 | 62.43 |
| Standard Prompt | 70.50 | 72.00 | 80.00 | 50.72 | 34.78 | 55.07 | 52.00 | 46.00 | 38.00 | 55.45 |
| Dense Steering Vector | 54.50 | 79.00 | 80.00 | 50.72 | 36.96 | 57.25 | 47.50 | 47.50 | 51.50 | 56.10 |
| GCG | 55.83 | 73.50 | 58.70 | 46.95 | 45.12 | 54.27 | 48.50 | 48.00 | 51.50 | 53.60 |
| ProTeGi (GPT-4o) | 54.77 | 56.13 | 71.51 | 48.17 | 46.95 | 38.41 | 42.50 | 20.99 | 47.49 | 47.44 |
| RESGA (Ours) | 49.86 | 50.63 | 62.65 | 45.12 | 41.46 | 51.22 | 38.50 | 38.50 | 34.50 | 45.83 |
| SAEGA (Ours) | 49.84 | 49.95 | 70.70 | 46.95 | 40.85 | 49.14 | 31.50 | 40.50 | 38.50 | 46.44 |

On sycophancy, both RESGA and SAEGA achieve error rates statistically indistinguishable from \(50\%\), indicating neutralization rather than behavioral reversal. Table 1’s sycophancy values are 49.86, 50.63, and 62.65 for RESGA, and 49.84, 49.95, and 70.70 for SAEGA, in Llama–Qwen–Gemma order.

This substantially improves over manual prompting and dense activation steering, which reduce error only partially and inconsistently across models. In the same columns, Standard Prompt scores 70.50, 72.00, and 80.00; Dense Steering Vector scores 54.50, 79.00, and 80.00.

On hallucination and myopic reward, both methods yield consistent improvements over baselines (Table 1, the hallucination and myopic-reward columns). SAEGA achieves the strongest myopic-reward reduction for Llama 3.1: 31.50, compared with RESGA’s 38.50 and baseline values of 54.00, 52.00, 47.50, 48.50, and 42.50. RESGA is competitive on hallucination, with values of 45.12 for Llama, 41.46 for Qwen, and 51.22 for Gemma.

### 4.3 Mechanistic Analysis

We analyze how RESGA and SAEGA mitigate personas and why SAEGA differs qualitatively from dense steering. Figures 3–6 show precise, sparse, semantically grounded SAEGA control rather than global interference with representations.

**Neutralization vs. Distributional Shifting.**

We first examine projections onto the sycophancy axis, the direction associated with sycophancy. Figure 3 shows a broad baseline distribution, reflecting a systematic agreement bias. Dense steering and RESGA move the mean toward neutrality but retain substantial variance, meaning examples remain spread around the mean. This indicates coarse counterbalancing.

SAEGA instead produces a sharp peak near zero. This collapse in variance shows instance-wise neutrality—neutrality across individual examples—rather than merely a shifted distribution. The resulting behavior is statistically indistinguishable from random choice.

Figure 3: Distribution of projections onto the sycophancy axis. Dense steering and RESGA shift the mean but retain variance. SAEGA collapses variance around neutrality (0.0), indicating precise control.

**Preservation of the Natural Activation Manifold.**

The natural activation manifold is the region or structure occupied by the model’s usual internal activations. To assess its preservation, we use an SAE trained on residual activations and count active features with the \(L_0\) norm.

Figure 4 shows approximately 50 active features per token for the baseline. Injecting a dense vector raises the count beyond 150 features, indicating departure from the natural activation manifold. RESGA partially mitigates this effect but still increases sparsity. SAEGA keeps the count at 50–60 features, comparable to baseline. This demonstrates preservation of the model’s internal sparse topology, or pattern of sparse feature activity.

Figure 4: Sparsity preservation (\(L_{0}\) norm). Dense steering forces an unnatural explosion in active features. SAEGA preserves sparsity close to the baseline model.

**Feature-Level Control.**

Figure 5 examines activation changes in the SAE features most correlated with sycophancy. Dense steering and RESGA produce noisy, inconsistent changes and often activate unrelated features. SAEGA consistently suppresses causally relevant sycophancy features while leaving unrelated features largely unaffected.

Figure 5: Impact on SAE features correlated with sycophancy. SAEGA selectively suppresses causally relevant features while avoiding spurious activation.

**Geometric Structure of Steering Trajectories.**

We visualize steering trajectories—changes in representations under steering—in residual activation space. We use principal component analysis (PCA), which displays the representations along directions summarizing their variation.

Figure 6 shows that dense steering causes a large linear displacement far from the natural data manifold. RESGA shifts the mean but leaves a diffuse, high-variance cluster. SAEGA converges to a tight, stable region that is often orthogonal, or perpendicular, to the dense steering direction. This indicates a distinct subspace corresponding to honest behavior while remaining on-manifold.

Figure 6: PCA of steering trajectories. Dense steering exits the natural manifold. RESGA shifts the mean with high variance. SAEGA converges to a tight, stable subspace.

Table 2 reproduces discovered prompts literally. Strings such as `\n`, `\"`, and `\u...` are displayed escape sequences. Entries such as `<unused3147>`, `<unused5655>`, and `<bos>` are token spellings appearing in the prompts; the paper supplies no separate meanings for them. Other fragments, labels, and codes in these examples are prompt contents, not names of additional methods.

Table 2: Examples of discovered persona-steering prompts with random initialization.

| Task | Model | RESGA | SAEGA |
|---|---|---|---|
| Sycophancy | Llama 3.1 | `Confederate Telegraph, its editor and publisher,` | ``()` geschichten in the code.\n\nStep2`` |
| Sycophancy | Gemma 3 | `ervlak wx\u1209\u1362 \n\n\n\n\n \u258bh\u00e9sitez({ Ace` | `wrotekg\u042d\u0442 \u0438 pabbulEDES \u0440\u0430\u0437 \ud808\ude66` |
| Sycophancy | Qwen 2.5 | `BloodIsn"Not(disposing\u8bf7 \u8f93\u5165 Turkish:)\n \uac1c\uc778\uc815 \ubcf4` | `\u516c\u4ea4\u53d1 \u5c55\u7b33 nearer prod tw/**\n\nhistor` |
| Hallucination | Llama 3.1 | `Ikea Good Donation show indicators –> namely suicides` | `bob Ancat example, Cat theyena` |
| Hallucination | Gemma 3 | `<unused3147> UnitsMany= cheaply \u540d\u7a31\u662f \u09ae\u09c1\u09b8 \u09b2\u09bf\u09ae` | `\u3068\u306a\u3063 \u305f\uff1a\u300a "/>< dirt\u70ba\u4e86ofthe \ud808\udcca<bos>` |
| Hallucination | Qwen 2.5 | `\u201c\u2026 Iditi Abraham Prophet hired Jesus` | `leng Someo answersined, " Canadians` |
| Myopic Reward | Llama 3.1 | `Transition B to\n AThis is question` | `Radi Answerologists: B You\n prefer` |
| Myopic Reward | Gemma 3 | `Gav\u0ab5\u0abe \u0aa5\u0ac0iteten <unused5655>\u0bae\u0bc8 \u0baf\u0bbe\u0ba9 \u3092\u63d0\u4f9b \u3059\u308b **ain` | `DR thoroughly ( \uc5bc:\ Wro\u041f\u041e Bit` |
| Myopic Reward | Qwen 2.5 | `([\" BC oddly.fixed fruitful ’\ud835\ude91 fairness` | `screams\u7761plib poss\u00edvel Akalleriesira Whenever` |

In Table 3, “Score” is the error score. “Seed” identifies the initialization phrase, and “Optimized” identifies the prompt produced by EPO.

Table 3: Seed-Initialized Prompt Optimization. Each seed prompt (italicized) is optimized via EPO into a discovered prompt (monospaced) that achieves substantially lower error, demonstrating that seed initialization yields effective steering prompts that remain closer to natural language.

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

WikiText2 is the dataset used for the language-modeling evaluation in Table 4. Its “Baseline” column gives perplexity without a discovered prompt. The “RESGA & SAEGA” column gives perplexity after prepending discovered prompts, placing them before the evaluated text.

Table 4: Perplexity on WikiText2. Language modeling perplexity before and after prepending RESGA- and SAEGA-discovered prompts. The negligible perplexity change across all three model families confirms that our prompts steer persona-relevant behavior without disrupting general language modeling performance.

| Model | Baseline | RESGA \(\&\) SAEGA |
|---|---:|---:|
| Llama | 12.56 | 12.07 |
| Qwen | 10.88 | 12.48 |
| Gemma | 30.51 | 29.19 |

### 4.4 Qualitative Analysis of Mechanisms

Table 2 presents automatically discovered persona-steering prompts. Many appear superficially incoherent. However, analysis of token-probability shifts reveals distinct mechanistic strategies used by the optimizer.

Table 3 shows that seed initialization produces more interpretable prompts that preserve natural-language structure while achieving competitive steering. Its six seed-to-optimized score pairs are 69.16% to 56.25%, 72.30% to 54.22%, 71.15% to 51.72%, 72.06% to 59.35%, 52.43% to 46.34%, and 51.20% to 46.95%.

Table 4 confirms negligible degradation in general language modeling after prepending discovered prompts. WikiText2 perplexity changes from 12.56 to 12.07 for Llama, 10.88 to 12.48 for Qwen, and 30.51 to 29.19 for Gemma.

##### Token Priming via Induction Heads.

In binary classification, the model chooses between two answers. When undesirable behavior is systematically associated with a particular answer label, unconstrained optimization can converge on **Token Priming**: inserting the alternative label into the prompt to increase its output probability.

For example, sycophantic answers are often labeled “(A).” Here, “(A)” and “B” are answer-option labels, and lowercase “b” appears as a literal prompt fragment. Some discovered prompts, particularly RESGA prompts, explicitly insert the alternative label, as in `"...apol al al b"`.

We attribute this to Induction Heads, model components associated with in-context copying mechanisms (Olsson et al., 2022). The optimizer discovers that inserting “B” into the prompt increases the probability of generating that token. This acts as a “soft” few-shot example: an indirect demonstration that biases the output distribution to correct dataset skew. It does not necessarily alter the underlying semantic reasoning.

##### Semantic Steering via Vocabulary Shift.

SAEGA prompts instead frequently use **Semantic Steering**: fragments that prime particular reasoning modes rather than simply forcing a token.

We analyze output-vocabulary shifts using \(\Delta\text{Logits}\), the change in logits. Here, \(\text{Logits}_{\text{steered}}\) are output scores with steering, and \(\text{Logits}_{\text{baseline}}\) are scores without it:

\[
\Delta\text{Logits}
=
\text{Logits}_{\text{steered}}
-
\text{Logits}_{\text{baseline}}.
\]

- **Sycophancy (Polite Disagreement).** SAEGA systematically suppresses direct-agreement markers—“Yes,” “True,” and “Agree”—and significantly upweights semantic pivots such as “However” and “But.” These are literal token examples. The changes suggest that the method does not merely force rudeness or contradiction. Instead, it primes a “critical evaluation” mode in which the model creates room for nuance before delivering a factual correction.
- **Myopic Reward (Temporal Priming).** Temporal priming means encouraging attention to longer-term outcomes. The vocabulary-shift analysis shows that RESGA and SAEGA upweight long-term-planning tokens, including “Future,” “Long,” and “Wait.”

### 4.5 Ablation Studies

An ablation replaces a component to test its contribution. Table 5 tests three core design choices.

**Persona Steering Vector.** The Random Direction ablation replaces the persona vector with a random direction. Performance collapses to near or above zero-shot levels across all tasks and models, confirming that a mechanistically grounded direction is essential. Compare Table 5’s Random Direction row with Table 1’s Zero-Shot row. For hallucination, for example, Random Direction scores 68.29, 54.26, and 84.70, compared with zero-shot values of 51.45, 45.00, and 57.97.

**Gradient-Guided Mutation.** The Random Mutation ablation replaces gradient-guided token selection with random token mutation. It consistently degrades performance relative to RESGA and SAEGA across Table 5’s nine task-and-model columns. This demonstrates that gradient feedback is critical for directing the search toward persona-suppressing prompts.

**Representation-Based Objective.** Task Loss as Objective replaces the representation-projection objective with direct task-loss optimization. It yields unstable results and underperforms RESGA and SAEGA in most settings (Table 5, Task Loss as Objective, RESGA, and SAEGA rows). This indicates that internal representations provide a more reliable optimization signal than task loss alone.

Together, these comparisons confirm that all three components contribute meaningfully to effectiveness.

Table 5: Ablation study results. We ablate three core components of our framework: the persona steering vector (Random Direction), gradient-guided token mutation (Random Mutation), and the representation-based objective (Task Loss as Objective).

| Method | Sycophancy \(\downarrow\): Llama | Sycophancy \(\downarrow\): Qwen | Sycophancy \(\downarrow\): Gemma | Hallucination \(\downarrow\): Llama | Hallucination \(\downarrow\): Qwen | Hallucination \(\downarrow\): Gemma | Myopic Reward \(\downarrow\): Llama | Myopic Reward \(\downarrow\): Qwen | Myopic Reward \(\downarrow\): Gemma |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Random Direction | 74.13 | 84.56 | 81.47 | 68.29 | 54.26 | 84.70 | 50.50 | 48.50 | 61.00 |
| Random Mutation | 68.00 | 71.66 | 76.04 | 50.40 | 45.12 | 54.26 | 52.00 | 46.50 | 55.00 |
| Task Loss as Objective | 55.83 | 73.50 | 58.70 | 46.95 | 45.12 | 54.27 | 48.50 | 48.00 | 51.50 |
| RESGA | 49.86 | 50.63 | 62.65 | 45.12 | 41.46 | 51.22 | 38.50 | 38.50 | 34.50 |
| SAEGA | 49.84 | 49.95 | 70.70 | 46.95 | 40.85 | 49.14 | 31.50 | 40.50 | 38.50 |

## 5 Conclusion

We present a framework that bridges mechanistic interpretability and automatic prompt discovery for persona steering. RESGA and SAEGA construct persona vectors from labeled examples and optimize prompts through evolutionary gradient ascent. This yields interpretable control over LLM behavior.

Experiments on three persona-mitigation tasks show significant improvements on sycophancy and myopic reward and consistent improvements on hallucination. The discovered prompts substantially outperform manual prompting and dense steering (Table 1, Standard Prompt, Dense Steering Vector, RESGA, and SAEGA rows).

Mechanistic analysis shows that SAEGA neutralizes rather than shifts behavior, preserves natural activation manifolds, and controls interpretable features (Figures 3–6).

Future work could use semi-supervised approaches, which combine labeled and unlabeled examples, to reduce labeled-data requirements. It could also investigate transferring discovered prompts across models and extend the framework to multi-persona steering, or controlling several personas.

## 6 Limitations

We state our limitations as follows:

- **Labeled Data Requirement.** RESGA and SAEGA require labeled examples that exhibit and lack the target persona to find steering prompts.
- **Cross-Model Transferability.** SAE latents and steering vectors are model-specific. They require retraining for each target model.
- **Fluency-Effectiveness Tradeoff.** The most effective prompts identified by our approach do not show semantic coherence. Future work could use strong linguistic priors—constraints favoring natural language—to find prompts that are both effective and human-like.

## Broader Impact Statement

This work aims to improve LLM safety by suppressing undesired behavioral personas. However, the same pipeline could, in principle, target safety-critical SAE latents, automate jailbreak discovery, or elicit harmful outputs. Deployment should include appropriate access controls.

## Acknowledgment

We acknowledge the use of Claude (Anthropic) for manuscript writing assistance and AI-assisted code generation throughout this work.
