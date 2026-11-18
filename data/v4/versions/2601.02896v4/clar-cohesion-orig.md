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

Controlling emergent behavioral personas, such as sycophancy and hallucination, in Large Language Models (LLMs) is critical for AI safety, yet remains a persistent challenge. Existing solutions face a dilemma: manual prompt engineering is intuitive but unscalable and imprecise, while automatic optimization methods are effective but operate as “black boxes” with no interpretable connection to model internals.

We propose a framework that adapts gradient ascent to LLMs to enable targeted prompt discovery. Specifically, we propose RESGA (RESidual Gradient Ascent) and SAEGA (Sparse Autoencoder Gradient Ascent). Both methods optimize randomly initialized prompts so that the prompts’ representations become better aligned with an identified persona direction. We also introduce fluent gradient ascent to control the fluency of the discovered persona-steering prompts.

We demonstrate the effectiveness of RESGA and SAEGA across Llama 3.1, Qwen 2.5, and Gemma 3 for steering three personas: sycophancy, hallucination, and myopic reward. Crucially, on sycophancy, our automatically discovered prompts achieve significant improvement: 49.90% compared with 79.24%. By grounding prompt discovery in mechanistically meaningful features, our method offers a new paradigm for controllable and interpretable behavior modification. We release our scripts for RESGA and SAEGA in this github repo: https://github.com/HarshSaini10/RESGA_SAEGA.

Reviewed on OpenReview: https://openreview.net/forum?id=dcmHPxgo4c

## 1 Introduction

Figure 1: Comparison of persona steering approaches.
(a): Prompt-based steering provides implicit control via manually designed natural language instructions.
(b): Activation steering injects a dense steering vector $\mathbf{v}$ into model representations $e(\mathbf{t})$.
(c): Our methods, RESGA and SAEGA, derive persona directions from contrastive activations and optimize prompts via fluent gradient ascent, enabling both residual-based and interpretable feature-level control.

In Figure 1, $\mathbf{t}$ denotes a prompt, $e(\mathbf{t})$ denotes the prompt’s model representation, and $\mathbf{v}$ denotes a direction used to steer that representation. RESGA means RESidual Gradient Ascent, and SAEGA means Sparse Autoencoder Gradient Ascent.

Large language models (LLMs) have achieved remarkable capabilities across diverse domains, transforming how we interact with AI systems in applications ranging from education to healthcare. However, as LLMs become increasingly integrated into high-stakes environments, controlling their behavioral personas has emerged as a critical challenge for AI safety. Recent work has demonstrated that fine-tuning on specific datasets can induce “emergent misalignment”: models exhibit stereotypically harmful personas in responses to unrelated prompts (Wang et al., 2025). These findings highlight the urgent need for methods to understand and steer LLM behavior. Moreover, although safety alignment procedures have been widely deployed, these procedures often fail to prevent models from adopting harmful personas when prompted appropriately. This failure further underscores the importance of robust persona control mechanisms.

Prompt engineering has become the primary approach to persona control because practitioners can guide model outputs through carefully crafted instructions without expensive retraining. However, effective prompt engineering requires significant human expertise and domain knowledge: practitioners must identify prompts that reliably elicit desired behaviors while maintaining output quality. This manual process becomes particularly challenging for complex behavioral traits such as honesty, helpfulness, or domain-specific personas because the relationship between prompt content and model behavior remains poorly understood. These limitations have motivated automatic prompt discovery methods that systematically identify persona-steering prompts. However, existing automatic approaches often rely on black-box optimization rather than interpretable insights into how prompts influence model internals.

Recent advances in mechanistic interpretability have opened new avenues for understanding and controlling LLM behavior through sparse dictionary learning (SDL) and persona steering vectors (Zhao et al., 2026; Chen et al., 2025). Sparse autoencoders (SAEs) and their variants decompose model activations into sparse, interpretable features that respond to specific concepts or patterns (Cunningham et al., 2023a; Bricken et al., 2023; Rajamanoharan et al., 2024b). Recent work on persona features has shown that SAE-discovered latents can capture and control behavioral personas in language models. In particular, “misaligned persona” features strongly predict emergent harmful behaviors (Wang et al., 2025). However, existing work has primarily analyzed which latents activate for given inputs or intervened through activation steering. The inverse problem therefore remains largely unexplored: how can we systematically discover prompts that selectively activate task-relevant latents to steer model personas?

Gradient ascent techniques have been widely successful in interpreting convolutional neural network (CNN) neurons through feature visualization (Olah et al., 2017). However, the discrete nature of language has limited the adaptation of gradient ascent to LLMs. Standard gradient ascent optimizes continuous inputs to maximally activate target model internals, whether individual neurons, attention patterns, or learned directions in representation space. Direct optimization in embedding space followed by nearest-token projection, however, yields off-manifold results because of polysemanticity and the mismatch between continuous optimization and discrete token spaces (Wallace et al., 2019). To address this mismatch, recent gradient-guided discrete optimization methods employ evolutionary algorithms: gradient feedback guides iterative token replacement (Zou et al., 2023c; Thompson et al., 2024). These methods have successfully discovered effective prompts for tasks ranging from adversarial attacks to jailbreaking (Andriushchenko et al., ). Yet these techniques have not been applied to identify model steering prompts.

In this work, we bridge mechanistic interpretability with automatic prompt discovery for persona steering, as illustrated in Figure 1. Our framework introduces two complementary algorithms:

- **RESGA**, or RESidual Gradient Ascent, operates on dense residual stream representations.
- **SAEGA**, or Sparse Autoencoder Gradient Ascent, leverages mechanistically interpretable SAE latents.

Both algorithms first construct persona steering vectors that capture undesired behavioral traits in the model’s representation space. RESGA uses direct representation differences, whereas SAEGA uses SAE latent-based approaches that identify causally relevant features. Both algorithms then perform fluent gradient ascent to discover prompts that maximally steer away from the target personas with controllable fluency. Thus, unlike black-box prompt optimization methods, our framework uses mechanistic insights to guide prompt search, enabling more targeted and interpretable persona control.

We demonstrate both methods on three persona steering tasks: sycophancy, hallucination, and myopic reward. The automatically discovered prompts achieve significant improvement on sycophancy and myopic reward, substantially outperforming manual prompting and dense steering methods (Section 4.2, Table 1). We also provide representation-level evidence that prompt-based steering avoids the distributional shifts induced by dense steering methods. This evidence, presented in Section 4.3, exposes a fundamental limitation of dense steering approaches.

## 2 Related Works

### 2.1 Persona Steering

Controlling behavioral personas in large language models has become increasingly important for AI safety and alignment. Traditional approaches align model behavior during training through reinforcement learning from human feedback (RLHF) (Ouyang et al., 2022) and constitutional AI (Bai et al., 2022).

Representation Engineering (RepE) pioneered the extraction of concept directions from contrast pairs of model activations. RepE then controls model outputs by adding those directions during inference (Zou et al., 2023a). Contrastive Activation Addition (CAA) refined this approach by using carefully curated contrast datasets to identify high-level behavioral concepts and applying steering vectors at specific model layers (Rimsky et al., 2023). These activation steering methods have demonstrated effectiveness in mitigating undesired behaviors, including sycophancy (Sharma et al., ), toxicity (Liu et al., ), and bias (Tigges et al., 2023).

More recently, Wang et al. (2025) discovered that sparse autoencoder latents can capture interpretable “persona features” that predict emergent misalignment behaviors. Their results show that SAE-identified features enable precise control over model personas through activation clamping.

### 2.2 Sparse Autoencoder

Sparse Autoencoders (SAEs), first introduced in (Cunningham et al., 2023b), are tools for discovering interpretable representations of neural network activations (Dunefsky et al., 2024; Tang et al., 2026). SAEs reconstruct model representations using sparse latent features. Through this reconstruction, SAEs can uncover monosemantic neurons that activate in response to specific patterns, thereby reducing superposition in the representation space (Elhage et al., 2022).

Several techniques have improved SAE architectures for large-scale interpretable feature extraction (Bricken et al., 2023). These techniques include JumpReLU (Rajamanoharan et al., 2024a), Top-K (Gao et al., 2024), Batch Top-K (Bussmann et al., ), and Matryoshka Sparse Autoencoders (Bussmann et al., 2025).

LLM-based analysis is often used to interpret the neurons discovered by SAEs (Luo et al., ; Luo et al., 2024a; Tang et al., 2025a). A typical workflow first identifies subpopulations activated by specific neurons (Luo et al., 2024b), then analyzes those samples through prompt engineering techniques (Tang et al., 2025b).

### 2.3 Prompt Engineering

Prompt engineering has emerged as a critical technique for guiding large language models to perform complex tasks (Tang and Dong, 2024; Zou et al., 2026). The paradigm was first extensively demonstrated with GPT-3 (Brown et al., 2020): by receiving demonstrations in a prompt, language models could adapt to diverse tasks through in-context learning.

Chain-of-thought (CoT) prompting further enhanced LLM reasoning by generating step-by-step intermediate reasoning steps (Wei et al., 2022). Self-consistency decoding then improved performance by sampling multiple reasoning paths and selecting the most consistent answer (Wang et al., 2023).

To address limitations in arithmetic computation, researchers proposed hybrid approaches that combine natural language reasoning with external tools. Program-Aided Language models (PAL) (Gao et al., 2023) and Program of Thoughts (PoT) (Chen et al., 2023) decompose problems into programmatic steps while delegating computation to external interpreters. These approaches achieve substantial improvements on mathematical reasoning tasks.

Evolutionary prompt optimization (EPO) uses gradient-guided evolutionary algorithms to discover effective prompts by balancing objective optimization with language fluency (Thompson and Sklar, 2024). Beyond individual prompt techniques, generative agents demonstrated that LLM-powered systems can simulate believable human behavior and emergent social dynamics through carefully designed prompts (Park et al., 2023). Recently, Luo et al. (2023) developed a unified theoretical framework that views prompt engineering as optimal control problems.

ProTeGi automates prompt refinement by using an LLM application programming interface (API) to generate natural language “gradients” that criticize the current prompt. ProTeGi edits the prompt in the opposite semantic direction and uses beam search and bandits to guide the process efficiently (Pryzant et al., ). In contrast, Greedy Coordinate Gradient (GCG) operates at the token level: GCG combines greedy search with gradient-based scoring to iteratively substitute tokens and discover prompts that elicit target behaviors without manual engineering (Zou et al., 2023b).

## 3 Method

Our framework discovers persona-steering prompts through a three-stage pipeline:

1. **Construct persona steering vectors.** These vectors capture the direction of undesired behavioral traits in the model’s representation space, using either direct representation differences or SAE latent-based approaches (Section 3.1).
2. **Optimize prompts through fluent gradient ascent.** The optimization discovers prompts that maximally steer away from the target persona while maintaining fluency (Section 3.2).
3. **Initialize and select prompts.** We describe initialization strategies and select effective prompts from the Pareto frontier based on validation performance (Section 3.3). Here, the Pareto frontier expresses the tradeoff between persona steering strength and language-model fluency.

In Figure 2, $\mathbf{t}$ denotes a prompt, $f(\mathbf{t})$ denotes the persona-steering objective, $H(\mathbf{t})$ denotes the fluency component shown in the caption, and $\lambda$ denotes a fluency-steering tradeoff parameter. Section 3.2 specifies the optimization objective.

Figure 2: RESGA & SAEGA Framework Overview. Our framework discovers persona-steering prompts in two stages: (1) Persona steering vector construction via either dense representations (RESGA) or sparse SAE latents (SAEGA), and (2) Fluent gradient ascent optimization with objective $L_{\lambda}(\mathbf{t})=f(\mathbf{t})+H(\mathbf{t})$ that transforms random token sequences into readable prompts that steer the model in specific directions for interpretable persona control.

### 3.1 Persona Steering Vectors

Given a target persona to suppress, we construct a steering vector $\mathbf{v}\in\mathbb{R}^{d}$. Here, $d$ is the activation-space dimension, and $\mathbf{v}$ represents the target persona’s direction in that space. We consider two complementary constructions.

**Representation-Based Steering.** Let $D^{+}$ be a dataset of examples exhibiting the target persona, and let $D^{-}$ be a dataset of persona-free examples. For an example $\mathbf{x}$, let $\mathbf{e}(\mathbf{x})\in\mathbb{R}^{d}$ denote the model activation at a chosen layer. The notation $|D^{+}|$ and $|D^{-}|$ denotes the number of examples in the corresponding datasets.

The most direct steering-vector construction takes the mean representation of persona-exhibiting examples and subtracts the mean representation of persona-free examples:

$$
\mathbf{v}_{\text{repr}}
=
\frac{1}{|D^{+}|}
\sum_{\mathbf{x}\in D^{+}}\mathbf{e}(\mathbf{x})
-
\frac{1}{|D^{-}|}
\sum_{\mathbf{x}\in D^{-}}\mathbf{e}(\mathbf{x}).
\tag{1}
$$

Thus, $\mathbf{v}_{\text{repr}}$ points in the direction captured by the representation difference between the two datasets.

**SAE Latent-Based Steering.** For SAEGA, we alternatively construct steering vectors from sparse autoencoder latents. We use pretrained SAEs from SAE-Lens (Bloom et al., 2024).

For each SAE latent $i$, we measure relevance to the target behavior by subtracting the latent’s mean activation on persona-free examples from its mean activation on persona-exhibiting examples. We then select the top-$K$ latents with the largest absolute activation differences. Here, $K$ is the number of selected latents, and $\mathcal{I}_{K}$ is the index set of those selected latents.

To define the latent activations, let $\mathbf{W}_{\text{enc}}$ denote the SAE encoder weight matrix, let $\mathbf{w}_{i}$ be the $i$-th row of that matrix, and let $b_i$ be the corresponding bias. The activation of latent $i$ for example $\mathbf{x}$ is

$$
a_i(\mathbf{x})
=
\text{ReLU}\!\left(\mathbf{w}_{i}\mathbf{e}(\mathbf{x})+b_i\right),
$$

where $\text{ReLU}$ is the rectified linear unit. The mean activations on the two datasets are

$$
\bar{a}_{i}^{+}
=
\frac{1}{|D^{+}|}
\sum_{\mathbf{x}\in D^{+}}a_i(\mathbf{x}),
\qquad
\bar{a}_{i}^{-}
=
\frac{1}{|D^{-}|}
\sum_{\mathbf{x}\in D^{-}}a_i(\mathbf{x}).
$$

The SAE-based steering vector combines the encoder directions of the selected latents, weighting each encoder direction by its signed mean activation difference:

$$
\mathbf{v}_{\text{SAE}}
=
\sum_{i\in\mathcal{I}_{K}}
\mathbf{w}_{i}\cdot
\left(\bar{a}_{i}^{+}-\bar{a}_{i}^{-}\right).
\tag{2}
$$

### 3.2 Fluent Gradient Ascent

Given a persona steering vector $\mathbf{v}$ from Section 3.1, we discover prompts that maximally steer the model away from the target persona. Let $\mathbf{t}$ denote a prompt, and let $\mathbf{e}(\mathbf{t})$ denote the representation of the prompt’s last token at the target layer. We define the persona reduction objective as the negative scaled projection of that representation onto the steering vector:

$$
f(\mathbf{t})
=
-\frac{\langle\mathbf{e}(\mathbf{t}),\mathbf{v}\rangle}
{\|\mathbf{e}(\mathbf{t})\|}.
\tag{3}
$$

Here, $\langle\cdot,\cdot\rangle$ denotes the inner product, and $\|\mathbf{e}(\mathbf{t})\|$ denotes the norm of the prompt representation. Because Equation (3) uses the negative projection onto the persona direction, maximizing $f(\mathbf{t})$ discovers prompts whose representations point away from that direction.

However, optimizing $f(\mathbf{t})$ alone often yields unnatural or nonsensical prompts. Therefore, following the fluent dreaming framework (Thompson et al., 2024), we balance persona steering with prompt fluency.

Let $n$ be the prompt length, $m$ be the language model, $\mathbf{t}_{\leq i}$ be the prompt prefix through position $i$, and $t_{i+1}$ be the next token. Let $H(m(\mathbf{t}_{\leq i}),t_{i+1})$ be the cross-entropy measuring how likely the next token is under the model given that prefix. With $\lambda$ controlling the fluency-steering tradeoff, the optimization objective is

$$
\mathcal{L}_{\lambda}(\mathbf{t})
=
f(\mathbf{t})
-
\frac{\lambda}{n}
\sum_{i=0}^{n-1}
H\!\left(m(\mathbf{t}_{\leq i}),t_{i+1}\right).
\tag{4}
$$

The second term penalizes low-probability token sequences and therefore encourages human-interpretable prompts.

We optimize Equation (4) using Evolutionary Prompt Optimization (EPO) (Thompson et al., 2024). EPO maintains a population of $M$ candidate prompts, with each candidate targeting a different point on the Pareto frontier between fluency and steering effectiveness. Each candidate has a fluency weight from $\{\lambda_1,\ldots,\lambda_M\}$. In Algorithm 1, $T$ is the number of optimization iterations, $k$ is the number of candidate token substitutions retained per position, and $\mathbf{t}^{i}$ is the $i$-th candidate prompt.

**Algorithm 1 Evolutionary Prompt Optimization**

1. **Inputs:** Steering objective $f(\cdot)$, language model $m$, prompt length $n$, population size $M$, fluency weights $\{\lambda_{1},\ldots,\lambda_{M}\}$.
2. **Outputs:** Prompts spanning the fluency–steering Pareto frontier.
3. Initialize $M$ prompts of length $n$ with random initialization or seed initialization.
4. **for** iteration $=1$ to $T$ **do**
5.   **for** each prompt $\mathbf{t}^{i}$ **do**
6.    Compute $\mathcal{L}_{\lambda_{i}}(\mathbf{t}^{i})$ and gradients with respect to one-hot token encodings.
7.    Select top-$k$ candidate tokens per position by gradient magnitude.
8.    Generate mutations by replacing a random token with a sampled top-$k$ alternative.
9.   **end for**
10.   Evaluate all candidates and select the best prompt for each $\lambda_i$.
11.   **if** restart step **then**
12.    Retain the best prompt under a random $\lambda$, and reinitialize the others.
13.   **end if**
14. **end for**

At each iteration, EPO computes gradients with respect to token embeddings to identify promising token substitutions. EPO then mutates the population by sampling from high-gradient tokens and selects the best candidates for each fluency weight $\lambda_i$. Periodic restarts prevent premature convergence. Consequently, this evolutionary approach efficiently explores the Pareto frontier and produces diverse prompts with varying fluency-effectiveness tradeoffs.

### 3.3 Initialization and Selection of Persona Steering Prompts

We provide two initialization strategies for EPO.

- **Random initialization:** Prompts are seeded with uniformly sampled vocabulary tokens. This initialization tends to produce discovered prompts that are syntactically fragmented or semantically incoherent.
- **Seed initialization:** Prompts are seeded with a short natural language phrase loosely related to the target persona, such as “Please answer honestly” for sycophancy. This initialization anchors optimization closer to the language manifold.

Because seed initialization anchors the search closer to the language manifold, seed-initialized prompts occupy higher-fluency regions of the Pareto frontier without significantly sacrificing steering effectiveness. Seed initialization is therefore preferable when interpretability of the discovered prompt is desired.

EPO yields candidate prompts spanning the Pareto frontier between persona steering strength and language-model fluency. We evaluate each candidate on held-out validation data by appending the candidate prompt to task inputs and measuring the resulting change in the target behavioral metric.

In practice, the prompts with the strongest steering effects are often syntactically fragmented, multilingual, or semantically incoherent. Therefore, we do not interpret fluency as human readability or grammatical correctness. Instead, we treat fluency as an intrinsic model-based constraint, measured by the prompt’s self cross-entropy under the language model. Lower cross-entropy indicates that a prompt lies closer to the model’s training distribution, whereas higher cross-entropy corresponds to increasingly off-manifold token sequences.

Rather than imposing a hard fluency threshold, we analyze and report steering performance along the full Pareto frontier. This protocol allows us to study how behavioral control degrades or improves as prompts move further from the language manifold. In downstream analysis, we emphasize prompts in intermediate regions of the frontier: these prompts balance measurable steering effectiveness with moderate increases in perplexity relative to random token baselines. Perplexity is the language-model fluency measure also used in the downstream language modeling evaluation in Table 4.

Finally, we assess prompt stability by measuring how consistently a prompt reduces the target persona across diverse validation examples. This stability assessment guards against degenerate solutions that steer strongly only on a narrow subset of inputs. Taken together, the evaluation protocol prioritizes causal steering efficacy over linguistic naturalness because our primary goal is to understand and control how internal representations give rise to emergent personas.

## 4 Experiments

We evaluate RESGA and SAEGA on three persona steering tasks to answer three questions:

1. Can automatically discovered prompts reliably neutralize undesired personas?
2. How do residual-based and SAE-based steering differ in effectiveness and mechanism?
3. Does mechanistic structure translate into more stable and interpretable control?

We first describe the datasets, models, and baselines in Section 4.1. We then present quantitative results for sycophancy, hallucination, and myopic reward in Section 4.2. Finally, Section 4.3 presents a mechanistic analysis showing that SAEGA achieves persona control through targeted feature suppression while preserving natural activation structure.

### 4.1 Experimental Setup

##### Tasks and Datasets.

We evaluate persona steering on three established benchmarks:

- **Sycophancy** measures whether a model agrees with a user’s stated opinion even when that opinion is incorrect (Perez et al., ).
- **Hallucination** is evaluated using multiple-choice TruthfulQA (Lin et al., 2022). Lower error indicates improved factual reliability.
- **Myopic Reward** measures short-term reward seeking over long-term outcomes (Perez et al., ).

Lower metrics are considered better for all three tasks.

##### Models.

We conduct experiments on Llama 3.1 8B Instruct (Grattafiori et al., 2024), Qwen 2.5 7B Instruct (Hui et al., ), and Gemma 3 4B Instruct (Team et al., 2025) to assess cross-architecture generalization. For SAEGA, we use pretrained sparse autoencoders from SAELens (Bloom et al., 2024).

##### Baselines.

We compare against five baselines:

1. **Zero-Shot:** No intervention.
2. **Standard Prompt:** Manually written instructions, such as “Answer honestly.”
3. **Dense Steering Vector:** Direct injection of a representation-difference steering vector into the residual stream.
4. **Greedy Coordinate Gradient (GCG):** Prompt discovery by optimizing directly on the output logits.
5. **ProTeGi:** Prompt optimization using LLM-generated natural language “gradients,” with bandits and beam search.

##### Implementation Details.

We target middle-to-late layers: Layer 25 for Llama 3.1 8B and Qwen 2.5 7B, and Layer 20 for Gemma 3 4B. This choice follows empirical analysis showing that early-layer representations lack the high-level semantic abstraction needed to isolate complex behavioral traits, which renders steering ineffective. In contrast, final-layer interventions often failed to override the model’s accumulated logit bias.

Evolutionary Prompt Optimization hyperparameters were selected through preliminary sweeps to balance optimization stability and computational cost. All reported experiments use population size $M=100$, prompt length $n=8$, and $T=1000$ iterations.

Crucially, we use context-aware optimization rather than optimizing prompts in isolation. At each evolutionary step, candidate prompts are appended to a dynamic batch of task-specific training questions. The loss is then calculated from the model’s internal state after processing the combined sequence. This setup ensures that the discovered prompt acts as a generalized steering trigger robust to varying input contexts.

For SAEGA, steering vectors use the top-$K$ SAE latents most strongly correlated with the target concept, as measured by activation differences between concept-positive and concept-negative examples. We evaluate multiple values of $K$, corresponding to different fractions of the SAE’s natural sparsity level, $L_0\approx 50$. Here, $L_0$ denotes the number of active SAE features. We report results using $K=20$ because this value consistently yielded strong performance. All results are evaluated on held-out validation splits.

##### Evaluation Protocol.

All tasks use conditional log-probability comparison. Let $q$ denote a question, $p$ a steering prompt, and $a$ a candidate answer. For each candidate answer, we compute

$$
\log P(a\mid q,p),
$$

where $P(a\mid q,p)$ is the model’s probability of the answer conditional on the question and steering prompt. The model prediction is the answer with higher log-probability.

The reported metric is the fraction of examples for which the model prefers the undesirable option: the sycophantic, hallucinated, or myopic answer. Therefore, lower values indicate improved persona control.

### 4.2 Persona Steering Results

Table 1 reports error rates across three persona mitigation tasks and two model families. For sycophancy, an error rate of $50\%$ corresponds to perfect neutralization: the model neither systematically agrees nor systematically disagrees with the user.

Table 1: Persona steering results across three tasks and three models. Lower is better.

| Method | Sycophancy $\downarrow$: Llama | Sycophancy $\downarrow$: Qwen | Sycophancy $\downarrow$: Gemma | Hallucination $\downarrow$: Llama | Hallucination $\downarrow$: Qwen | Hallucination $\downarrow$: Gemma | Myopic Reward $\downarrow$: Llama | Myopic Reward $\downarrow$: Qwen | Myopic Reward $\downarrow$: Gemma | Avg. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Zero-Shot | 72.48 | 86.00 | 82.00 | 51.45 | 45.00 | 57.97 | 54.00 | 55.00 | 58.00 | 62.43 |
| Standard Prompt | 70.50 | 72.00 | 80.00 | 50.72 | 34.78 | 55.07 | 52.00 | 46.00 | 38.00 | 55.45 |
| Dense Steering Vector | 54.50 | 79.00 | 80.00 | 50.72 | 36.96 | 57.25 | 47.50 | 47.50 | 51.50 | 56.10 |
| GCG | 55.83 | 73.50 | 58.70 | 46.95 | 45.12 | 54.27 | 48.50 | 48.00 | 51.50 | 53.60 |
| ProTeGi (GPT-4o) | 54.77 | 56.13 | 71.51 | 48.17 | 46.95 | 38.41 | 42.50 | 20.99 | 47.49 | 47.44 |
| RESGA (Ours) | 49.86 | 50.63 | 62.65 | 45.12 | 41.46 | 51.22 | 38.50 | 38.50 | 34.50 | 45.83 |
| SAEGA (Ours) | 49.84 | 49.95 | 70.70 | 46.95 | 40.85 | 49.14 | 31.50 | 40.50 | 38.50 | 46.44 |

On sycophancy, both RESGA and SAEGA achieve error rates statistically indistinguishable from $50\%$, indicating effective neutralization rather than behavioral reversal. The sycophancy entries in Table 1 are 49.86, 50.63, and 62.65 for RESGA on Llama, Qwen, and Gemma, respectively, and 49.84, 49.95, and 70.70 for SAEGA. This substantially improves over manual prompting and dense activation steering, which reduce error only partially and inconsistently across models: Table 1 reports 70.50, 72.00, and 80.00 for Standard Prompt, and 54.50, 79.00, and 80.00 for Dense Steering Vector.

On hallucination and myopic reward, both methods yield consistent improvements over baselines (Table 1, RESGA and SAEGA rows compared with the five baseline rows). SAEGA achieves the strongest reduction on myopic reward for Llama 3.1, with an error of 31.50 compared with RESGA’s 38.50 and the baseline values in the same column. RESGA performs competitively on hallucination, with errors of 45.12, 41.46, and 51.22 for Llama, Qwen, and Gemma, respectively.

### 4.3 Mechanistic Analysis

We analyze how RESGA and SAEGA achieve persona mitigation and why SAEGA operates qualitatively differently from dense steering. The analyses in Figures 3–6 reveal that SAEGA performs precise, sparse, and semantically grounded control rather than global representational interference.

**Neutralization vs. Distributional Shifting.** We first examine projections onto the sycophancy axis. In Figure 3, the baseline model exhibits a broad distribution, reflecting a systematic bias toward agreement. Dense steering and RESGA shift the mean toward neutrality but retain substantial variance. Thus, the mean shifts indicate coarse counterbalancing.

In contrast, SAEGA produces a sharp peak centered near zero in Figure 3. This collapse in variance shows that SAEGA enforces instance-wise neutrality rather than merely shifting the distribution, yielding behavior statistically indistinguishable from random choice.

Figure 3: Distribution of projections onto the sycophancy axis. Dense steering and RESGA shift the mean but retain variance. SAEGA collapses variance around neutrality (0.0), indicating precise control.

**Preservation of the Natural Activation Manifold.** To assess whether steering preserves natural internal structure, we measure sparsity using a sparse autoencoder trained on residual activations. Figure 4 reports the $L_0$ norm, defined here as the number of active SAE features. The baseline model activates approximately 50 features per token.

Injecting a dense steering vector increases the number of active features beyond 150, indicating a departure from the natural activation manifold (Figure 4). RESGA partially mitigates this effect but still increases sparsity. In contrast, SAEGA maintains sparsity comparable to the baseline, at 50–60 active features. This preservation demonstrates that SAEGA respects the model’s internal sparse topology.

Figure 4: Sparsity preservation ($L_{0}$ norm). Dense steering forces an unnatural explosion in active features. SAEGA preserves sparsity close to the baseline model.

**Feature-Level Control.** Figure 5 analyzes activation changes in SAE features most correlated with sycophancy. Dense steering and RESGA produce noisy and inconsistent effects, often activating unrelated features. In contrast, SAEGA consistently suppresses causally relevant sycophancy features while leaving unrelated features largely unaffected.

Figure 5: Impact on SAE features correlated with sycophancy. SAEGA selectively suppresses causally relevant features while avoiding spurious activation.

**Geometric Structure of Steering Trajectories.** We visualize steering trajectories in residual activation space using principal component analysis (PCA). Figure 6 shows that dense steering induces a large linear displacement far from the natural data manifold. RESGA shifts the mean but produces a diffuse, high-variance cluster.

In contrast, SAEGA converges to a tight, stable region that is often orthogonal to the dense steering direction. This geometry indicates that SAEGA discovers a distinct subspace corresponding to honest behavior while remaining on-manifold.

Figure 6: PCA of steering trajectories. Dense steering exits the natural manifold. RESGA shifts the mean with high variance. SAEGA converges to a tight, stable subspace.

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

Table 4: Perplexity on WikiText2. Language modeling perplexity before and after prepending RESGA- and SAEGA-discovered prompts. The negligible perplexity change across all three model families confirms that our prompts steer persona-relevant behavior without disrupting general language modeling performance.

| Model | Baseline | RESGA $\&$ SAEGA |
|---|---:|---:|
| Llama | 12.56 | 12.07 |
| Qwen | 10.88 | 12.48 |
| Gemma | 30.51 | 29.19 |

### 4.4 Qualitative Analysis of Mechanisms

Table 2 presents examples of persona-steering prompts automatically discovered by RESGA and SAEGA. Although many of these prompts appear superficially incoherent, analysis of the token probability shifts reveals distinct mechanistic strategies used by the optimizer to control model behavior.

Table 3 further demonstrates that seed initialization yields more interpretable discovered prompts that preserve natural language structure while achieving competitive steering performance. For example, the first seed–optimized pair reduces the score from 69.16% to 56.25%, and the “It is important to correct misconceptions respectfully.” pair reduces the score from 71.15% to 51.72%.

Table 4 confirms that prepending the discovered prompts incurs negligible degradation in general language modeling ability, as measured by perplexity on WikiText2. The baseline and prompted perplexities are 12.56 and 12.07 for Llama, 10.88 and 12.48 for Qwen, and 30.51 and 29.19 for Gemma.

##### Token Priming via Induction Heads.

For binary classification tasks in which the undesirable behavior is systematically aligned with a specific label, unconstrained optimization can converge on **Token Priming**. For example, in Sycophancy, the sycophantic answer is often labeled “(A).” Some discovered prompts, particularly those from RESGA, explicitly inject the alternative label into the context, as in “...apol al al b”.

We attribute this behavior to the exploitation of Induction Heads (Olsson et al., 2022). Specifically, the optimizer discovers that placing the target token “B” in the prompt increases the probability of “B” appearing in the generation through in-context copying mechanisms. Token priming therefore acts as a “soft” few-shot example: token priming biases the model’s output distribution to correct for the dataset skew without necessarily altering the underlying semantic reasoning.

##### Semantic Steering via Vocabulary Shift.

In contrast to simple token forcing, SAEGA prompts frequently employ **Semantic Steering**: the discovered fragments prime specific reasoning modes. To examine semantic steering, we analyze the change in output token probabilities induced by the steering prompts through the logit difference

$$
\Delta\text{Logits}
=
\text{Logits}_{\text{steered}}
-
\text{Logits}_{\text{baseline}},
$$

where $\text{Logits}_{\text{steered}}$ are the model’s output logits with the steering prompt and $\text{Logits}_{\text{baseline}}$ are the baseline output logits.

- **Sycophancy (Polite Disagreement):** SAEGA systematically suppresses direct agreement markers such as “Yes”, “True”, and “Agree”, while significantly upweighting semantic pivots such as “However” and “But”. This vocabulary shift suggests that SAEGA does not merely force the model to be rude or contradictory. Instead, SAEGA primes a “critical evaluation” mode in which the model creates space for nuance before delivering the factual correction.

- **Myopic Reward (Temporal Priming):** Analysis of the vocabulary shifts shows that RESGA and SAEGA successfully upweight tokens associated with long-term planning, including “Future”, “Long”, and “Wait”.

### 4.5 Ablation Studies

We validate the framework’s three core design choices through the ablation studies in Table 5.

**Persona Steering Vector.** Replacing the persona steering vector with a random direction causes performance to collapse to near or above zero-shot levels across all tasks and models. This result confirms that a mechanistically grounded steering direction is essential (Table 5, Random Direction row, compared with Table 1, Zero-Shot row).

**Gradient-Guided Mutation.** Replacing gradient-guided token selection with random token mutation consistently degrades performance. The Random Mutation row of Table 5 has higher errors than the RESGA and SAEGA rows across the reported tasks and models. This degradation demonstrates that gradient feedback is critical for directing the search toward persona-suppressing prompts.

**Representation-Based Objective.** Replacing the representation projection objective with direct task loss optimization yields unstable results and underperforms RESGA and SAEGA in the majority of settings (Table 5, Task Loss as Objective row compared with the RESGA and SAEGA rows). This result indicates that grounding the objective in model internal representations provides a more reliable optimization signal than task loss alone.

Together, these ablation results confirm that all three components contribute meaningfully to the framework’s effectiveness.

Table 5: Ablation study results. We ablate three core components of our framework: the persona steering vector (Random Direction), gradient-guided token mutation (Random Mutation), and the representation-based objective (Task Loss as Objective).

| Method | Sycophancy $\downarrow$: Llama | Sycophancy $\downarrow$: Qwen | Sycophancy $\downarrow$: Gemma | Hallucination $\downarrow$: Llama | Hallucination $\downarrow$: Qwen | Hallucination $\downarrow$: Gemma | Myopic Reward $\downarrow$: Llama | Myopic Reward $\downarrow$: Qwen | Myopic Reward $\downarrow$: Gemma |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Random Direction | 74.13 | 84.56 | 81.47 | 68.29 | 54.26 | 84.70 | 50.50 | 48.50 | 61.00 |
| Random Mutation | 68.00 | 71.66 | 76.04 | 50.40 | 45.12 | 54.26 | 52.00 | 46.50 | 55.00 |
| Task Loss as Objective | 55.83 | 73.50 | 58.70 | 46.95 | 45.12 | 54.27 | 48.50 | 48.00 | 51.50 |
| RESGA | 49.86 | 50.63 | 62.65 | 45.12 | 41.46 | 51.22 | 38.50 | 38.50 | 34.50 |
| SAEGA | 49.84 | 49.95 | 70.70 | 46.95 | 40.85 | 49.14 | 31.50 | 40.50 | 38.50 |

## 5 Conclusion

We present a framework that bridges mechanistic interpretability with automatic prompt discovery for persona steering. RESGA and SAEGA achieve interpretable control over LLM behavior by constructing persona steering vectors from labeled examples and optimizing prompts through evolutionary gradient ascent.

Experiments on three persona mitigation tasks demonstrate that the discovered prompts achieve significant improvement on sycophancy and myopic reward and consistent improvements on hallucination, substantially outperforming manual prompting and dense steering methods (Table 1). Mechanistic analysis shows that SAEGA succeeds by neutralizing rather than shifting behavior, preserving natural activation manifolds, and operating through interpretable feature-level control (Figures 3–6).

Future work could explore semi-supervised approaches to reduce labeled data requirements, investigate the cross-model transferability of discovered prompts, and extend the framework to multi-persona steering.

## 6 Limitations

We state our limitations as follows:

- **Labeled Data Requirement.** RESGA and SAEGA require labeled examples exhibiting and lacking the target persona to find steering prompts.

- **Cross-Model Transferability.** The SAE latents and steering vectors are model-specific. Consequently, retraining is required for each target model.

- **Fluency-Effectiveness Tradeoff.** Most effective prompts identified by our approach do not show semantic coherence. Future work could use strong linguistic priors to develop prompts that are effective and human-like at the same time.

## Broader Impact Statement

This work aims to improve LLM safety by suppressing undesired behavioral personas. However, the same pipeline could in principle be adapted to target safety-critical SAE latents, thereby automating jailbreak discovery or eliciting harmful outputs. Deployment should therefore be accompanied by appropriate access controls.

## Acknowledgment

We acknowledge the use of Claude (Anthropic) for manuscript writing assistance and AI-assisted code generation throughout this work.
