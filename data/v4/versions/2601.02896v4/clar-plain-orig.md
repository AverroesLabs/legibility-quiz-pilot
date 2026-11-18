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

Controlling emergent behavioral personas in large language models (LLMs) remains a persistent challenge for artificial intelligence (AI) safety. These personas include sycophancy, or agreeing with users even when they are wrong, and hallucination, or producing false information. Existing methods face a dilemma. Manual prompt engineering is intuitive but hard to scale and imprecise. Automatic optimization methods work well but act as “black boxes,” with no interpretable link to the model’s internal operations.

We propose a framework that adapts gradient ascent—search guided by changes that increase an objective—to discover targeted prompts for LLMs. Our two methods, RESGA and SAEGA, optimize randomly initialized prompts so that their internal representations align better with an identified persona direction. We introduce fluent gradient ascent to control the fluency of the discovered persona-steering prompts. We demonstrate both methods on Llama 3.1, Qwen 2.5, and Gemma 3 for three personas: sycophancy, hallucination, and myopic reward, or seeking short-term rewards over long-term outcomes. On sycophancy, our automatically discovered prompts achieve a significant improvement: 49.90% compared with 79.24%. By grounding prompt discovery in features with meaningful roles inside the model, our method offers a new approach to controllable and interpretable behavior modification. We release our RESGA and SAEGA scripts at https://github.com/HarshSaini10/RESGA_SAEGA.

Reviewed on OpenReview: https://openreview.net/forum?id=dcmHPxgo4c

## 1 Introduction

Figure 1: Comparison of persona steering approaches.
(a): Prompt-based steering provides indirect control through manually written natural language instructions.
(b): Activation steering adds a dense steering vector, $\mathbf{v}$, to the model’s representations, $e(\mathbf{t})$. A steering vector is a numerical direction used to change model behavior; a dense vector is not restricted to a few nonzero components.
(c): Our methods, RESGA and SAEGA, derive persona directions by contrasting internal activations, the numerical signals inside the model. They then optimize prompts through fluent gradient ascent. This supports control through both residual representations and interpretable features.

Large language models have gained strong capabilities across many fields. They have changed how people interact with AI systems in applications ranging from education to healthcare. As these models enter high-stakes settings, controlling their behavioral personas has become a critical AI safety challenge. Recent work shows that fine-tuning on specific datasets can cause “emergent misalignment”: models adopt stereotypically harmful personas when answering unrelated prompts (Wang et al., 2025). This finding highlights the need to understand and steer LLM behavior. Widely used safety alignment procedures often fail to stop models from adopting harmful personas when prompted appropriately. This failure underscores the need for robust persona control.

Prompt engineering—writing instructions to guide model outputs—has become the main way to control LLM personas without expensive retraining. However, effective prompt engineering requires substantial human expertise and domain knowledge. Practitioners must find prompts that reliably produce the desired behavior while preserving output quality. This manual process becomes especially difficult for complex traits such as honesty, helpfulness, or domain-specific personas. The link between prompt content and model behavior remains poorly understood. These limits have motivated automatic methods for discovering persona-steering prompts. Yet existing methods often use black-box optimization without interpretable insight into how prompts affect the model’s internal operations.

Recent advances in mechanistic interpretability, which studies how a model’s internal parts produce its behavior, offer new ways to understand and control LLMs. These include sparse dictionary learning (SDL), which represents activations using a small number of learned features, and persona steering vectors (Zhao et al., 2026; Chen et al., 2025). Sparse autoencoders (SAEs) reconstruct model activations through a small set of active learned features, called latents. SAEs and their variants have shown strong results in separating activations into interpretable features that respond to specific concepts or patterns (Cunningham et al., 2023a; Bricken et al., 2023; Rajamanoharan et al., 2024b). Recent work shows that SAE-discovered latents can capture and control behavioral personas (Wang et al., 2025). In particular, “misaligned persona” features strongly predict emergent harmful behaviors. However, most existing work studies which latents activate for given inputs or intervenes directly through activation steering. It largely leaves the inverse problem unexplored: how can we systematically find prompts that selectively activate task-relevant latents to steer model personas?

Gradient ascent has proved useful for interpreting neurons in convolutional neural networks (CNNs) through feature visualization, which finds inputs that strongly activate a target neuron (Olah et al., 2017). Its use in LLMs remains limited because language consists of discrete tokens rather than continuously adjustable inputs. Standard gradient ascent adjusts continuous inputs to maximize activation of target model internals. These targets can include individual neurons, attention patterns, or learned directions in representation space.

Directly optimizing token embeddings—the numerical vectors that represent tokens—and then choosing the nearest tokens produces off-manifold results. These results lie outside the patterns typical of the model’s data. This problem arises from polysemanticity, where one feature represents multiple concepts, and from the mismatch between continuous optimization and discrete tokens (Wallace et al., 2019). Recent gradient-guided discrete optimization methods address this problem with evolutionary algorithms. These algorithms use gradients to guide repeated token replacements and candidate selection (Zou et al., 2023c; Thompson et al., 2024). They have found effective prompts for tasks ranging from adversarial attacks to jailbreaking, which bypasses model safeguards (Andriushchenko et al., ). However, researchers have not applied these methods to discover model-steering prompts.

We propose a framework that connects mechanistic interpretability with automatic prompt discovery for persona steering (Figure 1). It introduces two complementary algorithms:

- **RESGA**, or RESidual Gradient Ascent, operates on dense representations in the residual stream, the internal representation passed through the model’s layers.
- **SAEGA**, or Sparse Autoencoder Gradient Ascent, uses mechanistically interpretable SAE latents.

Both algorithms first build persona steering vectors that capture undesired behavioral traits in the model’s representation space. RESGA uses direct differences between representations. SAEGA uses SAE latents to identify causally relevant features. Both then use fluent gradient ascent to discover prompts that steer maximally away from target personas while allowing control over fluency.

Unlike black-box prompt optimization, our framework uses mechanistic insights to guide the search. This enables more targeted and interpretable persona control. We demonstrate both methods on sycophancy, hallucination, and myopic reward. Automatically discovered prompts achieve significant improvements on sycophancy and myopic reward (see Section 4.2), substantially outperforming manual prompting and dense steering. We also provide representation-level evidence that prompt-based steering avoids the distributional shifts caused by dense steering. This exposes a fundamental limitation of dense steering methods.

## 2 Related Works

### 2.1 Persona Steering

Controlling LLM behavioral personas has become increasingly important for AI safety and alignment. Traditional methods align model behavior during training through reinforcement learning from human feedback (RLHF), which uses human feedback to guide learning (Ouyang et al., 2022), and constitutional AI, which guides behavior with stated principles (Bai et al., 2022).

Representation Engineering (RepE) introduced the extraction of concept directions from contrasting pairs of model activations. Adding these directions during inference, when the model processes inputs, enables control over outputs (Zou et al., 2023a). Contrastive Activation Addition (CAA) refined this approach. It identifies high-level behavioral concepts from carefully curated contrast datasets and applies steering vectors at selected layers (Rimsky et al., 2023). These activation-steering methods have reduced undesired behaviors, including sycophancy (Sharma et al., ), toxicity (Liu et al., ), and bias (Tigges et al., 2023).

More recently, Wang et al. (2025) found that SAE latents can capture interpretable “persona features” that predict emergent misalignment. These SAE-identified features support precise persona control through activation clamping, which fixes selected activations at chosen values.

### 2.2 Sparse Autoencoder

Sparse autoencoders, first introduced in Cunningham et al. (2023b), help discover interpretable representations of neural network activations (Dunefsky et al., 2024; Tang et al., 2026). They reconstruct model representations using sparse latent features. This can uncover monosemantic neurons, which respond to specific patterns, and reduce superposition, where multiple features share representation dimensions (Elhage et al., 2022).

Several techniques have improved SAE architectures for large-scale interpretable feature extraction (Bricken et al., 2023). These include JumpReLU (Rajamanoharan et al., 2024a), Top-K (Gao et al., 2024), Batch Top-K (Bussmann et al., ), and Matryoshka Sparse Autoencoders (Bussmann et al., 2025).

Researchers often use LLM-based analysis to interpret neurons discovered by SAEs (Luo et al., ; Luo et al., 2024a; Tang et al., 2025a). A typical workflow identifies subpopulations of samples that activate specific neurons (Luo et al., 2024b). It then analyzes those samples with prompt engineering (Tang et al., 2025b).

### 2.3 Prompt Engineering

Prompt engineering has become a key technique for guiding LLMs through complex tasks (Tang and Dong, 2024; Zou et al., 2026). GPT-3 first demonstrated this approach extensively (Brown et al., 2020). It showed that language models could adapt to varied tasks through in-context learning, using demonstrations supplied in the prompt.

Chain-of-thought (CoT) prompting improves LLM reasoning by generating intermediate reasoning steps (Wei et al., 2022). Self-consistency decoding improves performance by sampling several reasoning paths and choosing the most consistent answer (Wang et al., 2023).

To address limits in arithmetic computation, researchers combined natural language reasoning with external tools. Program-Aided Language models (PAL) (Gao et al., 2023) and Program of Thoughts (PoT) (Chen et al., 2023) divide problems into programmatic steps and delegate computation to external interpreters. These approaches substantially improve mathematical reasoning performance.

Evolutionary prompt optimization (EPO) uses gradient-guided evolutionary algorithms to discover effective prompts while balancing the target objective against language fluency (Thompson and Sklar, 2024). Generative agents show that carefully prompted LLM systems can simulate believable human behavior and emergent social dynamics (Park et al., 2023). Luo et al. (2023) recently developed a unified theory that treats prompt engineering as an optimal control problem.

ProTeGi automates prompt refinement (Pryzant et al., ). It uses an LLM application programming interface (API) to generate natural language “gradients” that criticize the current prompt. It then edits the prompt in the opposite semantic direction. Beam search, which retains several promising candidates, and bandits, which guide candidate evaluation, improve efficiency. Greedy Coordinate Gradient (GCG) combines greedy search with gradient-based scoring at the token level. It repeatedly replaces tokens to discover prompts that produce target behaviors without manual engineering (Zou et al., 2023b).

## 3 Method

Our framework discovers persona-steering prompts in three stages. First, we construct steering vectors that capture undesired behavioral directions in the model’s representation space. We use either direct representation differences or SAE latents (Section 3.1). Second, we use fluent gradient ascent to find prompts that steer maximally away from the target persona while maintaining fluency (Section 3.2). Finally, we initialize and select effective prompts from the Pareto frontier based on validation performance (Section 3.3). The Pareto frontier contains tradeoffs for which improving fluency or steering effectiveness requires sacrificing the other.

Figure 2: RESGA & SAEGA Framework Overview. Our framework discovers persona-steering prompts in two stages: (1) construct persona steering vectors from either dense representations (RESGA) or sparse SAE latents (SAEGA), and (2) optimize through fluent gradient ascent with objective $L_{\lambda}(\mathbf{t})=f(\mathbf{t})+H(\mathbf{t})$. This transforms random token sequences into readable prompts that steer the model in specific directions for interpretable persona control.

### 3.1 Persona Steering Vectors

For a target persona to suppress, we construct a steering vector $\mathbf{v}\in\mathbb{R}^{d}$. Here, $\mathbb{R}^{d}$ denotes a space of real-valued vectors with $d$ components. The vector represents the persona’s direction in the model’s activation space. We consider two complementary approaches.

**Representation-Based Steering.** The most direct approach computes the mean representation of persona-exhibiting examples minus the mean representation of persona-free examples:

$$
\mathbf{v}_{\text{repr}}=
\frac{1}{|D^{+}|}\sum_{\mathbf{x}\in D^{+}}\mathbf{e}(\mathbf{x})
-
\frac{1}{|D^{-}|}\sum_{\mathbf{x}\in D^{-}}\mathbf{e}(\mathbf{x})
\tag{1}
$$

Here, $D^{+}$ contains examples that exhibit the target persona, and $D^{-}$ contains persona-free examples. The notation $|D|$ gives the number of examples in a dataset. The vector $\mathbf{e}(\mathbf{x})\in\mathbb{R}^{d}$ is the activation for input $\mathbf{x}$ at a chosen layer.

**SAE Latent-Based Steering.** For SAEGA, we instead construct steering vectors from sparse autoencoder latents. We use pretrained SAEs from SAE-Lens (Bloom et al., 2024).

We identify persona-relevant latents by contrasting activations. For each latent $i$, we subtract its mean activation on persona-free examples from its mean activation on persona-exhibiting examples. We select the top $K$ latents with the largest absolute activation differences. Their indices form the set $\mathcal{I}_{K}$.

We compute the SAE steering vector from the encoder directions of these top $K$ latents:

$$
\mathbf{v}_{\text{SAE}}=
\sum_{i\in\mathcal{I}_{K}}\mathbf{w}_{i}\cdot
\left(\bar{a}_{i}^{+}-\bar{a}_{i}^{-}\right)
\tag{2}
$$

Here, $\mathbf{w}_{i}$ is row $i$ of the encoder weight matrix $\mathbf{W}_{\text{enc}}$. The mean activations of latent $i$ on persona-exhibiting and persona-free examples are, respectively,

$$
\bar{a}_{i}^{+}=
\frac{1}{|D^{+}|}\sum_{\mathbf{x}\in D^{+}}a_{i}(\mathbf{x}),
\qquad
\bar{a}_{i}^{-}=
\frac{1}{|D^{-}|}\sum_{\mathbf{x}\in D^{-}}a_{i}(\mathbf{x}),
$$

with

$$
a_{i}(\mathbf{x})=
\text{ReLU}(\mathbf{w}_{i}\mathbf{e}(\mathbf{x})+b_{i}).
$$

The bias is $b_i$. The rectified linear unit, ReLU, returns zero for negative inputs and leaves nonnegative inputs unchanged.

### 3.2 Fluent Gradient Ascent

Given a persona steering vector $\mathbf{v}$, we seek prompts that steer the model maximally away from that persona. We define the persona reduction objective as the negative scaled projection of the prompt representation onto the steering vector:

$$
f(\mathbf{t})=
-\frac{\langle\mathbf{e}(\mathbf{t}),\mathbf{v}\rangle}
{\|\mathbf{e}(\mathbf{t})\|}
\tag{3}
$$

Here, $\mathbf{t}$ is a prompt, and $\mathbf{e}(\mathbf{t})$ is its last token’s representation at the target layer. The angle brackets denote a dot product, and the double bars denote vector length. Maximizing this objective finds prompts whose representations point away from the persona direction.

Optimizing $f(\mathbf{t})$ alone often produces unnatural or nonsensical prompts. Following fluent dreaming (Thompson et al., 2024), we balance persona steering against prompt fluency:

$$
\mathcal{L}_{\lambda}(\mathbf{t})=
f(\mathbf{t})
-\frac{\lambda}{n}
\sum_{i=0}^{n-1}H(m(\mathbf{t}_{\leq i}),t_{i+1})
\tag{4}
$$

Here, $H$ is cross-entropy, a measure of how unlikely token $t_{i+1}$ is under model $m$ given the preceding tokens $\mathbf{t}_{\leq i}$. The weight $\lambda$ controls the fluency–steering tradeoff, and $n$ is the prompt length. The second term penalizes low-probability token sequences to encourage human-interpretable prompts.

We optimize this objective with Evolutionary Prompt Optimization (EPO) (Thompson et al., 2024). EPO maintains $M$ candidate prompts. Each targets a different point on the Pareto frontier between fluency and steering effectiveness. Algorithm 1 gives the procedure.

**Algorithm 1 Evolutionary Prompt Optimization**

1. **Inputs:** Steering objective $f(\cdot)$, language model $m$, prompt length $n$, population size $M$, and fluency weights $\{\lambda_{1},\ldots,\lambda_{M}\}$.
2. **Outputs:** Prompts spanning the fluency–steering Pareto frontier.
3. Initialize $M$ prompts of length $n$ using random initialization or seed initialization.
4. **For** iteration $=1$ to $T$:
5.  **For each** prompt $\mathbf{t}^{i}$:
6.   Compute $\mathcal{L}_{\lambda_{i}}(\mathbf{t}^{i})$ and gradients with respect to one-hot token encodings, which mark the selected token with a one and all others with zeros.
7.   Select the top $k$ candidate tokens at each position by gradient magnitude.
8.   Generate mutations by replacing a random token with a sampled top-$k$ alternative.
9.  **End for.**
10.  Evaluate all candidates and select the best prompt for each $\lambda_{i}$.
11.  **If** this is a restart step:
12.   Retain the best prompt under a random $\lambda$ and reinitialize the others.
13.  **End if.**
14. **End for.**

At each iteration, EPO computes gradients with respect to token embeddings to find promising substitutions. It mutates the population by sampling high-gradient tokens, then selects the best candidate for each fluency weight $\lambda_i$. Periodic restarts prevent premature convergence. This evolutionary search efficiently explores the Pareto frontier and produces varied prompts with different fluency–effectiveness tradeoffs.

### 3.3 Initialization and Selection of Persona Steering Prompts

We provide two initialization strategies for EPO. Random initialization starts prompts with tokens sampled uniformly from the vocabulary. It tends to produce syntactically fragmented or semantically incoherent prompts. Seed initialization starts with a short natural language phrase loosely related to the target persona, such as “Please answer honestly” for sycophancy. This anchors optimization closer to the language manifold, the patterns typical of natural language under the model. Seed-initialized prompts therefore occupy higher-fluency parts of the Pareto frontier without significantly sacrificing steering effectiveness. They are preferable when the discovered prompt should be interpretable.

EPO produces candidate prompts across the frontier between persona-steering strength and model-based fluency. We append each prompt to task inputs in held-out validation data and measure the change in the target behavioral metric.

In practice, the strongest steering prompts are often fragmented, multilingual, or semantically incoherent. We therefore do not treat fluency as human readability or grammatical correctness. Instead, we treat it as a model-based constraint, measured by the prompt’s self cross-entropy under the language model. Lower cross-entropy indicates that the prompt lies closer to the model’s training distribution. Higher cross-entropy indicates increasingly off-manifold token sequences.

We do not impose a hard fluency threshold. Instead, we analyze and report steering performance along the full Pareto frontier. This lets us study how behavioral control improves or degrades as prompts move farther from the language manifold. In downstream analysis, we emphasize prompts in intermediate parts of the frontier. These balance measurable steering effectiveness with moderate increases in perplexity relative to random-token baselines. Perplexity measures how difficult a token sequence is for the model to predict.

Finally, we assess prompt stability by measuring how consistently a prompt reduces the target persona across varied validation examples. This guards against degenerate solutions that steer strongly only on a narrow subset of inputs. Overall, our protocol prioritizes causal steering effectiveness over linguistic naturalness. Our primary goal is to understand and control how internal representations produce emergent personas.

## 4 Experiments

We evaluate RESGA and SAEGA on three persona-steering tasks to answer three questions:

1. Can automatically discovered prompts reliably neutralize undesired personas?
2. How do residual-based and SAE-based steering differ in effectiveness and mechanism?
3. Does mechanistic structure lead to more stable and interpretable control?

We first describe the datasets, models, and baselines (Section 4.1). We then report quantitative results for sycophancy, hallucination, and myopic reward (Section 4.2). Finally, our mechanistic analysis shows that SAEGA controls personas through targeted feature suppression while preserving natural activation structure (Section 4.3).

### 4.1 Experimental Setup

##### Tasks and Datasets.

We evaluate persona steering on three established benchmarks.

- **Sycophancy** (Perez et al., ) measures whether a model agrees with a user’s stated opinion even when it is incorrect.
- **Hallucination** uses multiple-choice TruthfulQA (Lin et al., 2022). Lower error indicates greater factual reliability.
- **Myopic Reward** (Perez et al., ) measures short-term reward seeking over long-term outcomes.

Lower metrics are better.

##### Models.

We test Llama 3.1 8B Instruct (Grattafiori et al., 2024), Qwen 2.5 7B Instruct (Hui et al., ), and Gemma 3 4B Instruct (Team et al., 2025) to assess generalization across architectures. In these model names, “B” denotes billions of parameters. For SAEGA, we use pretrained sparse autoencoders from SAELens (Bloom et al., 2024).

##### Baselines.

We compare against five baselines:

1. **Zero-Shot:** no intervention.
2. **Standard Prompt:** manually written instructions, such as “Answer honestly.”
3. **Dense Steering Vector:** direct addition of a representation-difference steering vector to the residual stream.
4. **Greedy Coordinate Gradient (GCG):** direct optimization of output logits, the model’s answer scores before conversion to probabilities, to discover prompts.
5. **ProTeGi:** prompt optimization using LLM-generated natural language “gradients,” bandits, and beam search.

##### Implementation Details.

We target middle-to-late layers: Layer 25 for Llama 3.1 8B and Qwen 2.5 7B, and Layer 20 for Gemma 3 4B. Our empirical analysis indicated that early-layer representations lack the high-level semantic abstraction needed to isolate complex behavioral traits, making steering ineffective. Final-layer interventions often failed to override the model’s accumulated logit bias.

We selected EPO hyperparameters through preliminary sweeps that balanced optimization stability against computational cost. All reported experiments use population size $M=100$, prompt length $n=8$, and $T=1000$ iterations.

Crucially, we use context-aware optimization rather than optimizing prompts alone. At each evolutionary step, we append candidate prompts to a changing batch of task-specific training questions. We calculate the loss from the model’s internal state after it processes the combined sequence. This ensures that the discovered prompt acts as a general steering trigger that remains robust across input contexts.

For SAEGA, we construct steering vectors from the top $K$ SAE latents most strongly correlated with the target concept. We measure this correlation through activation differences between concept-positive and concept-negative examples. We evaluate several values of $K$ that correspond to different fractions of the SAE’s natural sparsity level, $L_{0}\approx 50$. The $L_0$ measure counts active features. We report results for $K=20$, which consistently performed strongly. We evaluate all results on held-out validation splits.

##### Evaluation Protocol.

We evaluate all tasks by comparing conditional log-probabilities. For a question $q$, steering prompt $p$, and each candidate answer $a$, we compute:

$$
\log P(a\mid q,p)
$$

This is the logarithm of the answer’s probability given the question and prompt. The model’s prediction is the answer with the higher log-probability. Each metric reports the fraction of examples for which the model prefers the undesirable option: sycophantic, hallucinated, or myopic. Lower values indicate better persona control.

### 4.2 Persona Steering Results

Table 1 reports error rates for three persona-mitigation tasks and two model families. For sycophancy, an error rate of $50\%$ represents perfect neutralization: the model neither systematically agrees nor systematically disagrees with the user.

Table 1: Persona steering results across three tasks and three models. Lower is better.

| Method | Sycophancy: Llama ↓ | Sycophancy: Qwen ↓ | Sycophancy: Gemma ↓ | Hallucination: Llama ↓ | Hallucination: Qwen ↓ | Hallucination: Gemma ↓ | Myopic Reward: Llama ↓ | Myopic Reward: Qwen ↓ | Myopic Reward: Gemma ↓ | Avg. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Zero-Shot | 72.48 | 86.00 | 82.00 | 51.45 | 45.00 | 57.97 | 54.00 | 55.00 | 58.00 | 62.43 |
| Standard Prompt | 70.50 | 72.00 | 80.00 | 50.72 | 34.78 | 55.07 | 52.00 | 46.00 | 38.00 | 55.45 |
| Dense Steering Vector | 54.50 | 79.00 | 80.00 | 50.72 | 36.96 | 57.25 | 47.50 | 47.50 | 51.50 | 56.10 |
| GCG | 55.83 | 73.50 | 58.70 | 46.95 | 45.12 | 54.27 | 48.50 | 48.00 | 51.50 | 53.60 |
| ProTeGi (GPT-4o) | 54.77 | 56.13 | 71.51 | 48.17 | 46.95 | 38.41 | 42.50 | 20.99 | 47.49 | 47.44 |
| RESGA (Ours) | 49.86 | 50.63 | 62.65 | 45.12 | 41.46 | 51.22 | 38.50 | 38.50 | 34.50 | 45.83 |
| SAEGA (Ours) | 49.84 | 49.95 | 70.70 | 46.95 | 40.85 | 49.14 | 31.50 | 40.50 | 38.50 | 46.44 |

On sycophancy, both RESGA and SAEGA achieve error rates statistically indistinguishable from $50\%$. This indicates effective neutralization rather than behavioral reversal. The result substantially improves on manual prompting and dense activation steering, which reduce error only partly and inconsistently across models.

Both methods consistently improve on baselines for hallucination and myopic reward. SAEGA achieves the strongest reduction in myopic reward for Llama 3.1. RESGA performs competitively on hallucination.

### 4.3 Mechanistic Analysis

We analyze how RESGA and SAEGA mitigate personas and why SAEGA differs qualitatively from dense steering. Our analysis shows that SAEGA provides precise, sparse, semantically grounded control rather than broad interference with representations.

**Neutralization vs. Distributional Shifting.** We first examine projections onto the sycophancy axis. As Figure 3 shows, the baseline model has a broad distribution that reflects a systematic bias toward agreement. Dense steering and RESGA move the mean toward neutrality but retain substantial variance. This indicates coarse counterbalancing.

SAEGA instead produces a sharp peak near zero. This collapse in variance shows that SAEGA enforces neutrality for individual examples rather than merely shifting the distribution. It produces behavior statistically indistinguishable from random choice.

Figure 3: Distribution of projections onto the sycophancy axis. Dense steering and RESGA shift the mean but retain variance. SAEGA collapses variance around neutrality (0.0), indicating precise control.

**Preservation of the Natural Activation Manifold.** To test whether steering preserves natural internal structure, we measure sparsity with an SAE trained on residual activations. Figure 4 reports the $L_0$ norm, the number of active SAE features. The baseline model activates approximately 50 features per token.

Adding a dense steering vector drives this number beyond 150 features. This indicates a departure from the natural activation manifold. RESGA partly reduces this effect but still increases sparsity. SAEGA keeps sparsity close to the baseline, at 50–60 features. This demonstrates that it respects the model’s internal sparse topology, or pattern of sparse feature activity.

Figure 4: Sparsity preservation ($L_{0}$ norm). Dense steering forces an unnatural explosion in active features. SAEGA preserves sparsity close to the baseline model.

**Feature-Level Control.** Figure 5 examines activation changes in the SAE features most correlated with sycophancy. Dense steering and RESGA produce noisy, inconsistent effects and often activate unrelated features. SAEGA consistently suppresses causally relevant sycophancy features while leaving unrelated features largely unchanged.

Figure 5: Impact on SAE features correlated with sycophancy. SAEGA selectively suppresses causally relevant features while avoiding spurious activation.

**Geometric Structure of Steering Trajectories.** We visualize steering trajectories in residual activation space with principal component analysis (PCA), which projects representations onto directions that capture their variation. As Figure 6 shows, dense steering causes a large linear displacement far from the natural data manifold. RESGA shifts the mean but produces a diffuse cluster with high variance.

SAEGA converges to a tight, stable region that is often orthogonal, or at right angles, to the dense steering direction. This indicates that it finds a distinct subspace corresponding to honest behavior while remaining on-manifold.

Figure 6: PCA of steering trajectories. Dense steering exits the natural manifold. RESGA shifts the mean with high variance. SAEGA converges to a tight, stable subspace.

Table 2: Examples of discovered persona-steering prompts with random initialization.

| Task | Model | RESGA | SAEGA |
|---|---|---|---|
| Sycophancy | Llama 3.1 | `Confederate Telegraph, its editor and publisher,` | ``()` geschichten in the code.\n\nStep2`` |
| | Gemma 3 | `ervlak wx\u1209\u1362 \n\n\n\n\n \u258bh\u00e9sitez({ Ace` | `wrotekg\u042d\u0442 \u0438 pabbulEDES \u0440\u0430\u0437 \ud808\ude66` |
| | Qwen 2.5 | `BloodIsn"Not(disposing\u8bf7 \u8f93\u5165 Turkish:)\n \uac1c\uc778\uc815 \ubcf4` | `\u516c\u4ea4\u53d1 \u5c55\u7b33 nearer prod tw/**\n\nhistor` |
| Hallucination | Llama 3.1 | `Ikea Good Donation show indicators –> namely suicides` | `bob Ancat example, Cat theyena` |
| | Gemma 3 | `<unused3147> UnitsMany= cheaply \u540d\u7a31\u662f \u09ae\u09c1\u09b8 \u09b2\u09bf\u09ae` | `\u3068\u306a\u3063 \u305f\uff1a\u300a "/>< dirt\u70ba\u4e86ofthe \ud808\udcca<bos>` |
| | Qwen 2.5 | `\u201c\u2026 Iditi Abraham Prophet hired Jesus` | `leng Someo answersined, " Canadians` |
| Myopic Reward | Llama 3.1 | `Transition B to\n AThis is question` | `Radi Answerologists: B You\n prefer` |
| | Gemma 3 | `Gav\u0ab5\u0abe \u0aa5\u0ac0iteten <unused5655>\u0bae\u0bc8 \u0baf\u0bbe\u0ba9 \u3092\u63d0\u4f9b \u3059\u308b **ain` | `DR thoroughly ( \uc5bc:\ Wro\u041f\u041e Bit` |
| | Qwen 2.5 | `([\" BC oddly.fixed fruitful ’\ud835\ude91 fairness` | `screams\u7761plib poss\u00edvel Akalleriesira Whenever` |

Table 3: Seed-Initialized Prompt Optimization. EPO turns each seed prompt (italicized) into a discovered prompt (monospaced) with substantially lower error. This demonstrates that seed initialization produces effective steering prompts that remain closer to natural language.

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

Table 4: Perplexity on WikiText2. Language modeling perplexity before and after adding RESGA- and SAEGA-discovered prompts at the beginning. The negligible perplexity change across all three model families confirms that our prompts steer persona-relevant behavior without disrupting general language modeling performance.

| Model | Baseline | RESGA & SAEGA |
|---|---:|---:|
| Llama | 12.56 | 12.07 |
| Qwen | 10.88 | 12.48 |
| Gemma | 30.51 | 29.19 |

### 4.4 Qualitative Analysis of Mechanisms

Table 2 gives examples of persona-steering prompts that RESGA and SAEGA discovered automatically. Many appear incoherent. However, our analysis of changes in token probabilities reveals distinct mechanisms that the optimizer uses to control behavior.

Table 3 also shows that seed initialization yields more interpretable prompts. These preserve natural language structure while achieving competitive steering performance. Table 4 confirms that adding the discovered prompts at the beginning causes negligible degradation in general language modeling ability, as measured by WikiText2 perplexity.

##### Token Priming via Induction Heads.

In binary classification tasks, the undesirable behavior can consistently correspond to one label. For example, sycophantic answers in Sycophancy are often labeled “(A).” We observed that unconstrained optimization can converge on Token Priming: putting a token into the context to encourage its later use. Some discovered prompts, particularly RESGA prompts, explicitly insert the alternative label, as in "...apol al al b".

We attribute this effect to Induction Heads, attention mechanisms that support copying patterns from context (Olsson et al., 2022). The optimizer finds that inserting the target token "B" into the prompt increases its probability during generation through in-context copying. This acts as a “soft” few-shot example. It biases the output distribution to correct for dataset skew without necessarily changing the underlying semantic reasoning.

##### Semantic Steering via Vocabulary Shift.

SAEGA prompts, by contrast, often use Semantic Steering. They discover fragments that prime specific reasoning modes rather than simply forcing tokens. To study this, we analyzed prompt-induced changes in output token probabilities through the logit difference:

$$
\Delta\text{Logits}
=
\text{Logits}_{\text{steered}}
-
\text{Logits}_{\text{baseline}}.
$$

- **Sycophancy (Polite Disagreement):** SAEGA systematically suppresses direct agreement markers, such as "Yes", "True", and "Agree". It also significantly increases scores for semantic pivots, such as "However" and "But". This suggests that the method does not simply force rude or contradictory behavior. Instead, it primes a “critical evaluation” mode in which the model allows nuance before giving a factual correction.
- **Myopic Reward (Temporal Priming):** Both RESGA and SAEGA successfully increased scores for tokens associated with long-term planning, including "Future", "Long", and "Wait".

### 4.5 Ablation Studies

We test the framework’s three core design choices through ablation studies, which replace individual components to assess their contribution (Table 5).

**Persona Steering Vector.** Replacing the persona steering vector with a random direction causes performance to collapse to near or above zero-shot levels across all tasks and models. This confirms that a mechanistically grounded steering direction is essential.

**Gradient-Guided Mutation.** Replacing gradient-guided token selection with random token mutation consistently degrades performance. This shows that gradient feedback is critical for guiding the search toward persona-suppressing prompts.

**Representation-Based Objective.** Replacing the representation projection objective with direct task-loss optimization produces unstable results. It underperforms RESGA and SAEGA in most settings. This indicates that an objective grounded in internal model representations provides a more reliable optimization signal than task loss alone.

Together, these results confirm that all three components contribute meaningfully to the framework’s effectiveness.

Table 5: Ablation study results. We ablate three core components of our framework: the persona steering vector (Random Direction), gradient-guided token mutation (Random Mutation), and the representation-based objective (Task Loss as Objective).

| Method | Sycophancy: Llama ↓ | Sycophancy: Qwen ↓ | Sycophancy: Gemma ↓ | Hallucination: Llama ↓ | Hallucination: Qwen ↓ | Hallucination: Gemma ↓ | Myopic Reward: Llama ↓ | Myopic Reward: Qwen ↓ | Myopic Reward: Gemma ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Random Direction | 74.13 | 84.56 | 81.47 | 68.29 | 54.26 | 84.70 | 50.50 | 48.50 | 61.00 |
| Random Mutation | 68.00 | 71.66 | 76.04 | 50.40 | 45.12 | 54.26 | 52.00 | 46.50 | 55.00 |
| Task Loss as Objective | 55.83 | 73.50 | 58.70 | 46.95 | 45.12 | 54.27 | 48.50 | 48.00 | 51.50 |
| RESGA | 49.86 | 50.63 | 62.65 | 45.12 | 41.46 | 51.22 | 38.50 | 38.50 | 34.50 |
| SAEGA | 49.84 | 49.95 | 70.70 | 46.95 | 40.85 | 49.14 | 31.50 | 40.50 | 38.50 |

## 5 Conclusion

We present a framework that connects mechanistic interpretability with automatic prompt discovery for persona steering. RESGA and SAEGA construct persona steering vectors from labeled examples and optimize prompts through evolutionary gradient ascent. This achieves interpretable control over LLM behavior.

Experiments on three persona-mitigation tasks show significant improvements on sycophancy and myopic reward, along with consistent improvements on hallucination. The discovered prompts substantially outperform manual prompting and dense steering. Our mechanistic analysis shows that SAEGA succeeds by neutralizing rather than shifting behavior. It also preserves natural activation manifolds and controls interpretable features.

Future work could explore semi-supervised approaches, which combine labeled and unlabeled data, to reduce labeled-data requirements. It could also test whether discovered prompts transfer across models and extend the framework to steer multiple personas.

## 6 Limitations

Our framework has the following limitations:

- **Labeled Data Requirement.** RESGA and SAEGA need labeled examples that exhibit and lack the target persona to find steering prompts.
- **Cross-Model Transferability.** SAE latents and steering vectors are model-specific and require retraining for each target model.
- **Fluency-Effectiveness Tradeoff.** The most effective prompts found by our approach lack semantic coherence. Future work could use strong linguistic priors—constraints that favor natural language—to find prompts that are both effective and human-like.

## Broader Impact Statement

This work aims to improve LLM safety by suppressing undesired behavioral personas. However, the same pipeline could, in principle, target safety-critical SAE latents. It could then automate jailbreak discovery or elicit harmful outputs. Deployment should include appropriate access controls.

## Acknowledgment

We acknowledge the use of Claude (Anthropic) for manuscript writing assistance and AI-assisted code generation throughout this work.
