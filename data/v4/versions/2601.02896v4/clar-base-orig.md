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

Controlling emergent behavioral personas in Large Language Models (LLMs) is critical for artificial intelligence (AI) safety. A persona is a recurring behavioral tendency. Examples include sycophancy, or agreeing with a user even when the user is wrong, and hallucination, or producing factually unreliable answers. Existing solutions face a dilemma. Manual prompt engineering—writing instructions to guide a model—is intuitive but difficult to scale and imprecise. Automatic optimization methods are effective but operate as “black boxes,” without an interpretable connection to model internals.

We propose a framework that adapts gradient ascent to LLMs for targeted prompt discovery. Gradient ascent uses the direction of increasing objective value to guide optimization. Our two methods are RESGA (RESidual Gradient Ascent) and SAEGA (Sparse Autoencoder Gradient Ascent). A sparse autoencoder (SAE) reconstructs model activations using a small set of active internal features. Both methods optimize randomly initialized prompts so that their internal representations align better with an identified persona direction. We introduce fluent gradient ascent to control the fluency of the discovered steering prompts.

We demonstrate RESGA and SAEGA across Llama 3.1, Qwen 2.5, and Gemma 3. We steer three personas: sycophancy, hallucination, and myopic reward, meaning preference for short-term reward over long-term outcomes. On sycophancy, our automatically discovered prompts achieve significant improvement: 49.90% compared with 79.24%. By grounding prompt discovery in features tied to model mechanisms, our method offers a new paradigm for controllable and interpretable behavior modification. We release our scripts for RESGA and SAEGA in this github repo: https://github.com/HarshSaini10/RESGA_SAEGA.

Reviewed on OpenReview: https://openreview.net/forum?id=dcmHPxgo4c

## 1 Introduction

Figure 1: Comparison of persona steering approaches.  
(a): Prompt-based steering provides implicit control through manually designed natural-language instructions.  
(b): Activation steering injects a dense steering vector $\mathbf{v}$ into model representations $e(\mathbf{t})$. Here, $\mathbf{t}$ is a prompt, $e(\mathbf{t})$ is its internal representation, and $\mathbf{v}$ is a numerical direction used to change that representation. “Dense” means that the vector is not restricted to a small set of active features.  
(c): Our methods, RESGA and SAEGA, derive persona directions from contrastive activations: internal model responses to examples that exhibit or lack a persona. They optimize prompts through fluent gradient ascent. This enables control through either residual-stream representations—the internal states passed between model layers—or interpretable features.

LLMs have achieved remarkable capabilities across diverse domains. They are changing interactions with AI systems in applications from education to healthcare. As these models enter high-stakes environments, controlling their behavioral personas has become a critical AI safety challenge. Recent work shows that fine-tuning—additional training on a specific dataset—can induce “emergent misalignment.” In this setting, models exhibit stereotypically harmful personas when answering unrelated prompts (Wang et al., 2025). This finding highlights the need to understand and steer LLM behavior. Although safety alignment procedures are widely deployed, they often fail to prevent harmful personas when models are prompted appropriately.

Prompt engineering is the primary approach to persona control. Practitioners guide outputs with carefully written instructions, without expensive retraining. However, effective prompts require substantial human expertise and domain knowledge. They must reliably elicit the desired behavior while maintaining output quality. This is especially difficult for traits such as honesty, helpfulness, or domain-specific personas. The relationship between prompt content and behavior remains poorly understood. These limitations motivate automatic prompt discovery. Yet existing methods often use black-box optimization without interpretable insight into how prompts affect model internals.

Mechanistic interpretability seeks to explain behavior through a model’s internal computations. Recent advances include sparse dictionary learning (SDL), which represents activations through a collection of features with few active at once, and persona steering vectors (Zhao et al., 2026; Chen et al., 2025). SAEs and their variants can decompose model activations into interpretable features that respond to specific concepts or patterns (Cunningham et al., 2023a; Bricken et al., 2023; Rajamanoharan et al., 2024b). These internal feature variables are also called *latents*. Work on persona features shows that SAE-discovered latents can capture and control behavioral personas (Wang et al., 2025). In particular, “misaligned persona” features strongly predict emergent harmful behavior. Existing work mainly identifies which latents activate for an input or directly changes activations. It leaves an inverse problem largely unexplored: how can we discover prompts that selectively activate task-relevant latents to steer a persona?

Gradient ascent has been widely used to interpret neurons in convolutional neural networks (CNNs) through feature visualization (Olah et al., 2017). Its adaptation to LLMs has been limited because language consists of discrete tokens, the units of text processed by the model. Standard gradient ascent changes continuous inputs to maximize selected internal responses. These may be individual neurons, attention patterns, or learned directions in representation space. A token embedding is the continuous vector representing a token. Directly optimizing embeddings and then replacing them with their nearest tokens produces off-manifold results: representations outside the model’s usual region of language representations. The causes include polysemanticity, where one internal unit responds to multiple concepts, and the mismatch between continuous optimization and discrete tokens (Wallace et al., 2019).

Recent gradient-guided discrete optimization methods instead use evolutionary algorithms. These algorithms maintain candidates and repeatedly modify and select them. Gradient feedback guides token replacements (Zou et al., 2023c; Thompson et al., 2024). Such methods discover effective prompts for adversarial attacks and jailbreaking, meaning attempts to bypass model safeguards (Andriushchenko et al., ). However, these techniques have not been applied to identifying model-steering prompts.

We propose a framework connecting mechanistic interpretability to automatic persona-steering prompt discovery (Figure 1). RESGA operates on dense residual-stream representations. SAEGA uses mechanistically interpretable SAE latents. Both first construct vectors representing undesired behavioral traits. RESGA uses direct differences between representations. SAEGA uses latent-based approaches that identify causally relevant features. Both then use fluent gradient ascent to discover prompts that maximally steer away from the target persona, with controllable fluency.

Unlike black-box prompt optimization, our framework uses mechanistic insights to guide the search. This enables more targeted and interpretable persona control. Across sycophancy, hallucination, and myopic reward, the discovered prompts significantly improve sycophancy and myopic reward, substantially outperforming manual prompting and dense steering (Section 4.2, Table 1, RESGA and SAEGA rows). We also provide representation-level evidence that prompt-based steering avoids the distributional shifts induced by dense steering. This exposes a fundamental limitation of dense steering approaches (Section 4.3, Figures 3–6).

## 2 Related Works

### 2.1 Persona Steering

Controlling LLM personas is increasingly important for AI safety and alignment. Traditional approaches align behavior during training. They include reinforcement learning from human feedback (RLHF), which uses human feedback as a training signal (Ouyang et al., 2022), and constitutional AI, which uses stated behavioral principles (Bai et al., 2022).

Representation Engineering (RepE) extracts concept directions from contrasting pairs of model activations. Adding these directions during inference—the model’s processing of a new input—controls outputs (Zou et al., 2023a). Contrastive Activation Addition (CAA) refines this approach. It identifies high-level behavioral concepts from curated contrast datasets and applies steering vectors at particular model layers (Rimsky et al., 2023).

These activation-steering methods mitigate undesired behaviors, including sycophancy (Sharma et al., ), toxicity, meaning harmful or abusive language (Liu et al., ), and bias, meaning systematic skew in responses (Tigges et al., 2023). Wang et al. (2025) more recently identified interpretable SAE “persona features” that predict emergent misalignment. They showed that these features enable precise persona control through activation clamping: holding selected feature activations at chosen values.

### 2.2 Sparse Autoencoder

SAEs, first introduced in Cunningham et al. (2023b), discover interpretable representations of neural-network activations (Dunefsky et al., 2024; Tang et al., 2026). They reconstruct model representations using sparse latent features. This can uncover monosemantic neurons, which respond to specific patterns rather than multiple distinct concepts. It thereby reduces superposition, where multiple features share overlapping representation directions (Elhage et al., 2022).

Architectural techniques for large-scale interpretable feature extraction include JumpReLU, a thresholded activation method (Rajamanoharan et al., 2024a); Top-K, which selects a specified number of the strongest activations (Gao et al., 2024); Batch Top-K, which applies such selection across a batch of examples (Bussmann et al., ); and Matryoshka Sparse Autoencoders, which use a nested feature structure (Bussmann et al., 2025). These techniques have improved SAE architectures for large-scale interpretable feature extraction (Bricken et al., 2023).

LLM-based analysis is often used to interpret SAE-discovered neurons (Luo et al., ; Luo et al., 2024a; Tang et al., 2025a). A typical workflow identifies subpopulations—subsets of examples—activated by specific neurons (Luo et al., 2024b). It then analyzes those examples through prompt engineering (Tang et al., 2025b).

### 2.3 Prompt Engineering

Prompt engineering guides LLMs on complex tasks (Tang and Dong, 2024; Zou et al., 2026). GPT-3 first extensively demonstrated this paradigm (Brown et al., 2020). It showed that models could adapt to diverse tasks through in-context learning: using demonstrations supplied in the prompt.

Chain-of-thought (CoT) prompting improves reasoning by generating intermediate, step-by-step reasoning (Wei et al., 2022). Self-consistency decoding samples multiple reasoning paths and selects the most consistent answer (Wang et al., 2023).

Hybrid approaches address arithmetic limitations by combining natural-language reasoning with external tools. Program-Aided Language models (PAL) (Gao et al., 2023) and Program of Thoughts (PoT) (Chen et al., 2023) break problems into programmatic steps. External interpreters execute the computations. These approaches substantially improve mathematical reasoning.

Evolutionary prompt optimization (EPO) uses gradient-guided evolutionary algorithms to discover prompts while balancing the optimization objective against language fluency (Thompson and Sklar, 2024). Generative agents show that LLM-powered systems can simulate believable human behavior and emergent social dynamics through carefully designed prompts (Park et al., 2023). Luo et al. (2023) developed a unified theoretical framework treating prompt engineering as optimal control: choosing inputs to guide a system toward a desired outcome.

ProTeGi automates prompt refinement through an LLM application programming interface (API). It generates natural-language “gradients” that criticize the current prompt, then edits in the opposite semantic direction. It uses beam search, which retains several promising candidates, and bandit-based candidate allocation for efficiency (Pryzant et al., ).

Greedy Coordinate Gradient (GCG) operates at the token level. It combines greedy search—selecting the best available local change—with gradient-based scoring. It repeatedly substitutes tokens to discover prompts that elicit target behaviors without manual engineering (Zou et al., 2023b).

## 3 Method

Our framework discovers persona-steering prompts through three stages:

1. Construct a direction representing the undesired persona. Use either direct representation differences or SAE latents (Section 3.1).
2. Use fluent gradient ascent to find prompts that maximally steer away from that persona while maintaining fluency (Section 3.2).
3. Initialize candidates and select effective prompts using validation performance (Section 3.3).

Candidate selection considers the Pareto frontier between fluency and steering. This frontier contains tradeoffs where improving one objective requires sacrificing the other.

Figure 2: RESGA & SAEGA Framework Overview. Our framework discovers persona-steering prompts in two stages: (1) persona steering vector construction through either dense representations (RESGA) or sparse SAE latents (SAEGA), and (2) fluent gradient ascent optimization with objective $L_{\lambda}(\mathbf{t})=f(\mathbf{t})+H(\mathbf{t})$. Here, $L_{\lambda}$ denotes the combined objective, $\lambda$ controls the fluency tradeoff, $f$ is the steering objective, and $H$ denotes the cross-entropy fluency quantity. The process transforms random token sequences into readable prompts that steer the model in specific directions for interpretable persona control. Section 3.2 specifies the optimization objective.

### 3.1 Persona Steering Vectors

Given a target persona to suppress, we construct a steering vector $\mathbf{v}\in\mathbb{R}^{d}$. Here, $\mathbb{R}^{d}$ is the space of real-valued vectors with $d$ components, and $d$ is the activation dimension at the chosen layer. The vector represents the target persona’s direction in activation space. We consider two approaches.

**Representation-Based Steering.** We compute the mean representation of persona-exhibiting examples and subtract the mean representation of persona-free examples:

$$
\mathbf{v}_{\text{repr}}
=
\frac{1}{|D^{+}|}\sum_{\mathbf{x}\in D^{+}}\mathbf{e}(\mathbf{x})
-
\frac{1}{|D^{-}|}\sum_{\mathbf{x}\in D^{-}}\mathbf{e}(\mathbf{x}).
\tag{1}
$$

Here:

- $\mathbf{v}_{\text{repr}}$ is the representation-based steering vector.
- $D^{+}$ is the set of examples exhibiting the target persona.
- $D^{-}$ is the set of persona-free examples.
- $\mathbf{x}$ is an example.
- $|D^{+}|$ and $|D^{-}|$ are the numbers of examples in the respective sets.
- $\sum$ means to sum over the indicated examples.
- $\mathbf{e}(\mathbf{x})\in\mathbb{R}^{d}$ is the example’s activation at the chosen layer.

Thus, the two normalized sums are the two groups’ mean activations. Their difference points from the persona-free mean toward the persona-exhibiting mean.

**SAE Latent-Based Steering.** For SAEGA, we instead construct steering vectors from pretrained SAEs available through SAE-Lens (Bloom et al., 2024).

We identify persona-relevant latents through contrastive activation analysis. For each latent, indexed by $i$, we compute its mean activation on persona-exhibiting examples minus its mean activation on persona-free examples. We select the top-$K$ latents with the largest absolute differences. Here, $K$ is the number selected, and $\mathcal{I}_{K}$ is the set of their indices.

We combine their encoder directions as follows:

$$
\mathbf{v}_{\text{SAE}}
=
\sum_{i\in\mathcal{I}_{K}}
\mathbf{w}_{i}\cdot
\left(\bar{a}_{i}^{+}-\bar{a}_{i}^{-}\right).
\tag{2}
$$

Here, $\mathbf{v}_{\text{SAE}}$ is the SAE-based steering vector. The vector $\mathbf{w}_{i}$ is row $i$ of the SAE encoder weight matrix $\mathbf{W}_{\text{enc}}$. The dot in Equation (2) scales this vector by the scalar activation difference. The mean activations are

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

The activation of latent $i$ on example $\mathbf{x}$ is

$$
a_i(\mathbf{x})
=
\operatorname{ReLU}
\left(\mathbf{w}_{i}\mathbf{e}(\mathbf{x})+b_i\right).
$$

Here, $b_i$ is the encoder bias for latent $i$. ReLU means rectified linear unit: it returns the input when positive and zero otherwise.

### 3.2 Fluent Gradient Ascent

Given a persona vector $\mathbf{v}$, we search for prompts that maximally steer away from the target persona. We define the persona-reduction objective as the negative scaled projection onto that vector:

$$
f(\mathbf{t})
=
-\frac{\langle\mathbf{e}(\mathbf{t}),\mathbf{v}\rangle}
{\|\mathbf{e}(\mathbf{t})\|}.
\tag{3}
$$

Here, $\mathbf{t}$ is the prompt, and $\mathbf{e}(\mathbf{t})$ is the representation of its last token at the target layer. The angle brackets denote the inner product, which measures alignment between two vectors. The norm $\|\mathbf{e}(\mathbf{t})\|$ is the representation’s length. Maximizing the negative projection favors prompt representations pointing away from the persona direction.

Optimizing $f(\mathbf{t})$ alone often produces unnatural or nonsensical prompts. Following fluent dreaming (Thompson et al., 2024), we add a fluency penalty:

$$
\mathcal{L}_{\lambda}(\mathbf{t})
=
f(\mathbf{t})
-
\frac{\lambda}{n}
\sum_{i=0}^{n-1}
H\left(m(\mathbf{t}_{\leq i}),t_{i+1}\right).
\tag{4}
$$

The symbols mean:

- $\mathcal{L}_{\lambda}$ is the combined objective to maximize.
- $n$ is the number of prompt tokens.
- $i$ indexes token positions in the sum.
- $\mathbf{t}_{\leq i}$ is the prompt prefix through position $i$.
- $t_{i+1}$ is the next token.
- $m$ is the language model, and $m(\mathbf{t}_{\leq i})$ gives its next-token predictions.
- $H$ is cross-entropy: the penalty for how unlikely the actual next token is under those predictions.
- $\lambda$ is the weight controlling the fluency–steering tradeoff.

The second term subtracts a length-averaged penalty for low-probability sequences. It encourages human-interpretable prompts.

We optimize Equation (4) with EPO (Thompson et al., 2024). It maintains $M$ candidate prompts, where $M$ is the population size. Each candidate targets a different fluency–steering tradeoff through its weight $\lambda_i$. Algorithm 1 gives the procedure. In the algorithm, $T$ is the number of iterations, and $k$ is the number of candidate replacement tokens retained at each position.

**Algorithm 1 Evolutionary Prompt Optimization**

1. **Inputs:** Steering objective $f(\cdot)$, language model $m$, prompt length $n$, population size $M$, and fluency weights $\{\lambda_1,\ldots,\lambda_M\}$.
2. **Outputs:** Prompts spanning the fluency–steering Pareto frontier.
3. Initialize $M$ prompts of length $n$ using random initialization or seed initialization.
4. **For** iteration $=1$ to $T$:
5.   **For each** prompt $\mathbf{t}^{i}$, where superscript $i$ indexes a population member:
6.    Compute $\mathcal{L}_{\lambda_i}(\mathbf{t}^{i})$ and its gradients with respect to one-hot token encodings. A one-hot encoding has one entry for each vocabulary token, with only the selected token’s entry set to one.
7.    Select the top-$k$ candidate tokens at each position by gradient magnitude.
8.    Generate mutations by replacing a random token with a sampled top-$k$ alternative.
9.   **End for.**
10.   Evaluate all candidates and select the best prompt for each $\lambda_i$.
11.   **If** this is a restart step:
12.    Retain the best prompt under a randomly selected $\lambda$ and reinitialize the others.
13.   **End if.**
14. **End for.**

At each iteration, EPO computes gradients with respect to token embeddings to identify promising substitutions. It mutates candidates by sampling high-gradient tokens, then selects the best candidates for each fluency weight. Periodic restarts prevent premature convergence to a limited set of candidates. This evolutionary search efficiently explores the Pareto frontier and produces prompts with different fluency–effectiveness tradeoffs.

### 3.3 Initialization and Selection of Persona Steering Prompts

We provide two initialization strategies.

**Random initialization** samples prompt tokens uniformly from the vocabulary. It tends to produce discovered prompts that are syntactically fragmented or semantically incoherent.

**Seed initialization** starts from a short natural-language phrase loosely related to the persona. For example, “Please answer honestly” can seed sycophancy optimization. This anchors optimization closer to the language manifold—the region of sequences typical of the model’s language distribution. Seed-initialized prompts consequently occupy higher-fluency regions of the Pareto frontier without significantly sacrificing steering effectiveness. They are preferable when prompt interpretability is desired; Table 3 gives seed and optimized examples.

EPO produces candidate prompts across the fluency–steering frontier. We evaluate each prompt on held-out validation data: examples not used to optimize the prompts. We append the prompt to task inputs and measure the resulting change in the target behavioral metric.

Prompts with the strongest steering effects are often fragmented, multilingual, or semantically incoherent. We therefore do not interpret fluency as human readability or grammatical correctness. Instead, fluency is an intrinsic, model-based constraint measured by the prompt’s self cross-entropy under the language model. Lower cross-entropy indicates greater proximity to the training distribution. Higher cross-entropy indicates increasingly off-manifold sequences.

We do not impose a hard fluency threshold. We analyze and report steering across the full Pareto frontier to study how control changes as prompts move away from the language manifold. In downstream analysis, we emphasize intermediate-frontier prompts. These balance measurable steering with moderate increases in perplexity relative to random-token baselines. Perplexity is a measure of how difficult a sequence is for a language model to predict; lower values mean greater model-assigned predictability.

Finally, we assess stability: how consistently a prompt reduces the target persona across diverse validation examples. This guards against degenerate solutions that work strongly on only a narrow subset of inputs. Overall, the protocol prioritizes causal steering efficacy over linguistic naturalness. Our primary goal is to understand and control how internal representations produce emergent personas.

## 4 Experiments

We evaluate RESGA and SAEGA on three persona-steering tasks. We ask:

1. Can automatically discovered prompts reliably neutralize undesired personas?
2. How do residual-based and SAE-based steering differ in effectiveness and mechanism?
3. Does mechanistic structure produce more stable and interpretable control?

Section 4.1 describes the datasets, models, and baselines. Section 4.2 reports quantitative results for sycophancy, hallucination, and myopic reward. Section 4.3 analyzes how SAEGA controls personas through targeted feature suppression while preserving natural activation structure.

### 4.1 Experimental Setup

##### Tasks and Datasets.

We use three established benchmarks:

- **Sycophancy** measures whether a model agrees with a user’s stated opinion even when it is incorrect (Perez et al., ).
- **Hallucination** uses multiple-choice TruthfulQA (Lin et al., 2022). Lower error indicates improved factual reliability.
- **Myopic Reward** measures short-term reward seeking over long-term outcomes (Perez et al., ).

Lower metrics are considered better.

##### Models.

We test Llama 3.1 8B Instruct (Grattafiori et al., 2024), Qwen 2.5 7B Instruct (Hui et al., ), and Gemma 3 4B Instruct (Team et al., 2025) to assess generalization across model architectures. In these model names, B denotes billions of parameters, and Instruct identifies the instruction-oriented version. SAEGA uses pretrained SAEs from SAELens (Bloom et al., 2024).

##### Baselines.

We compare against five baselines:

1. **Zero-Shot:** no intervention.
2. **Standard Prompt:** manually written instructions, such as “Answer honestly.”
3. **Dense Steering Vector:** direct injection of a representation-difference vector into the residual stream.
4. **GCG:** prompt discovery by directly optimizing output logits, the model’s numerical token scores before conversion to probabilities.
5. **ProTeGi:** prompt optimization using LLM-generated natural-language “gradients,” bandits, and beam search.

##### Implementation Details.

We intervene at middle-to-late layers: Layer 25 for Llama 3.1 8B and Qwen 2.5 7B, and Layer 20 for Gemma 3 4B. Empirical analysis indicated that early-layer representations lack the high-level semantic abstraction needed to isolate complex behavioral traits, making steering ineffective. Final-layer interventions often failed to override the model’s accumulated logit bias.

We selected EPO hyperparameters—settings fixed for the optimization procedure—through preliminary sweeps balancing stability and computational cost. All reported experiments use population size $M=100$, prompt length $n=8$, and $T=1000$ iterations.

Crucially, optimization is context-aware. We do not optimize prompts in isolation. At each evolutionary step, we append candidates to a dynamic batch of task-specific training questions. We calculate the loss from the model’s internal state after it processes the combined sequence. This ensures that the discovered prompt acts as a generalized steering trigger robust to different input contexts.

SAEGA constructs vectors from the top-$K$ latents most strongly correlated with the target concept. Correlation is measured through activation differences between concept-positive and concept-negative examples. We evaluate several values of $K$, corresponding to fractions of the SAE’s natural sparsity level, $L_0\approx 50$. Here, $L_0$ is the number of active, nonzero SAE features. We report $K=20$, which consistently yielded strong performance. All results use held-out validation splits.

##### Evaluation Protocol.

All tasks use conditional log-probability comparison. For question $q$, steering prompt $p$, and each candidate answer $a$, we compute

$$
\log P(a\mid q,p).
$$

Here, $P(a\mid q,p)$ is the model-assigned probability of answer $a$ given the question and prompt; $\log$ denotes its logarithm. The prediction is the answer with higher log-probability.

Each metric reports the fraction of examples where the model prefers the undesirable answer: sycophantic, hallucinated, or myopic. Lower values indicate improved persona control.

### 4.2 Persona Steering Results

Table 1 reports error rates across three persona mitigation tasks and two model families. For sycophancy, an error rate of $50\%$ represents perfect neutralization: the model neither systematically agrees nor systematically disagrees with the user.

Table 1: Persona steering results across three tasks and three models. Lower is better. The downward arrows indicate this preference; Avg. denotes the reported average.

| Method | Sycophancy: Llama $\downarrow$ | Sycophancy: Qwen $\downarrow$ | Sycophancy: Gemma $\downarrow$ | Hallucination: Llama $\downarrow$ | Hallucination: Qwen $\downarrow$ | Hallucination: Gemma $\downarrow$ | Myopic Reward: Llama $\downarrow$ | Myopic Reward: Qwen $\downarrow$ | Myopic Reward: Gemma $\downarrow$ | Avg. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Zero-Shot | 72.48 | 86.00 | 82.00 | 51.45 | 45.00 | 57.97 | 54.00 | 55.00 | 58.00 | 62.43 |
| Standard Prompt | 70.50 | 72.00 | 80.00 | 50.72 | 34.78 | 55.07 | 52.00 | 46.00 | 38.00 | 55.45 |
| Dense Steering Vector | 54.50 | 79.00 | 80.00 | 50.72 | 36.96 | 57.25 | 47.50 | 47.50 | 51.50 | 56.10 |
| GCG | 55.83 | 73.50 | 58.70 | 46.95 | 45.12 | 54.27 | 48.50 | 48.00 | 51.50 | 53.60 |
| ProTeGi (GPT-4o) | 54.77 | 56.13 | 71.51 | 48.17 | 46.95 | 38.41 | 42.50 | 20.99 | 47.49 | 47.44 |
| RESGA (Ours) | 49.86 | 50.63 | 62.65 | 45.12 | 41.46 | 51.22 | 38.50 | 38.50 | 34.50 | 45.83 |
| SAEGA (Ours) | 49.84 | 49.95 | 70.70 | 46.95 | 40.85 | 49.14 | 31.50 | 40.50 | 38.50 | 46.44 |

On sycophancy, both RESGA and SAEGA achieve error rates statistically indistinguishable from $50\%$. This indicates effective neutralization rather than behavioral reversal (Table 1, RESGA and SAEGA rows, Sycophancy columns). This substantially improves over manual prompting and dense activation steering, which reduce error only partially and inconsistently across models (Table 1, Standard Prompt and Dense Steering Vector rows).

On hallucination and myopic reward, both methods yield consistent improvements over baselines (Table 1, Hallucination and Myopic Reward columns). SAEGA achieves the strongest myopic-reward reduction for Llama 3.1, with 31.50 (Table 1, SAEGA row, Myopic Reward: Llama). RESGA performs competitively on hallucination (Table 1, RESGA row, Hallucination columns).

### 4.3 Mechanistic Analysis

We analyze how RESGA and SAEGA mitigate personas and why SAEGA differs qualitatively from dense steering. The analysis reveals precise, sparse, semantically grounded control by SAEGA rather than global interference with representations (Figures 3–6).

**Neutralization vs. Distributional Shifting.** We first examine projections onto the sycophancy axis. Figure 3 shows a broad baseline distribution, reflecting a systematic agreement bias. Dense steering and RESGA shift the mean toward neutrality but retain substantial variance—the spread of projections around their mean. This indicates coarse counterbalancing.

SAEGA instead produces a sharp peak near zero. The collapse in variance shows instance-wise neutrality rather than merely a shifted distribution. It yields behavior statistically indistinguishable from random choice (Figure 3).

Figure 3: Distribution of projections onto the sycophancy axis. Dense steering and RESGA shift the mean but retain variance. SAEGA collapses variance around neutrality (0.0), indicating precise control.

**Preservation of the Natural Activation Manifold.** We use an SAE trained on residual activations to assess whether steering preserves natural internal structure. Figure 4 reports the $L_0$ norm, or number of active SAE features. The baseline activates approximately 50 features per token.

Dense-vector injection raises this number beyond 150, indicating departure from the natural activation manifold. RESGA partly mitigates this effect but still increases sparsity. SAEGA maintains sparsity comparable to baseline, at 50–60 features. This demonstrates respect for the model’s internal sparse topology, meaning its pattern of sparse feature activity (Figure 4).

Figure 4: Sparsity preservation ($L_0$ norm). Dense steering forces an unnatural explosion in active features. SAEGA preserves sparsity close to the baseline model.

**Feature-Level Control.** Figure 5 examines activation changes in the SAE features most correlated with sycophancy. Dense steering and RESGA have noisy, inconsistent effects and often activate unrelated features. SAEGA consistently suppresses causally relevant sycophancy features while leaving unrelated features largely unaffected.

Figure 5: Impact on SAE features correlated with sycophancy. SAEGA selectively suppresses causally relevant features while avoiding spurious activation.

**Geometric Structure of Steering Trajectories.** We visualize steering trajectories—changes in residual activations—using principal component analysis (PCA). PCA represents variation along a smaller set of directions. Figure 6 shows that dense steering creates a large linear displacement far from the natural data manifold. RESGA shifts the mean but produces a diffuse, high-variance cluster.

SAEGA converges to a tight, stable region. This region is often orthogonal, or perpendicular, to the dense steering direction. It indicates discovery of a distinct subspace corresponding to honest behavior while remaining on-manifold (Figure 6).

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

Table 3: Seed-Initialized Prompt Optimization. Each seed prompt (italicized) is optimized through EPO into a discovered prompt (monospaced) with substantially lower error. This demonstrates that seed initialization yields effective steering prompts that remain closer to natural language. Score is the reported error percentage.

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

Table 4: Perplexity on WikiText2. Language modeling perplexity before and after prepending RESGA- and SAEGA-discovered prompts. WikiText2 is the text dataset used for this evaluation. Prepending means placing a prompt before the evaluated text. The negligible perplexity change across all three model families confirms that our prompts steer persona-relevant behavior without disrupting general language modeling performance.

| Model | Baseline | RESGA $\&$ SAEGA |
|---|---:|---:|
| Llama | 12.56 | 12.07 |
| Qwen | 10.88 | 12.48 |
| Gemma | 30.51 | 29.19 |

### 4.4 Qualitative Analysis of Mechanisms

Table 2 gives automatically discovered prompts. Many appear superficially incoherent. However, analysis of their effects on token probabilities reveals distinct strategies for controlling behavior.

Table 3 shows that seed initialization produces more interpretable prompts. They preserve natural-language structure while achieving competitive steering. Each Optimized row has lower reported error than its preceding Seed row. Table 4 confirms negligible degradation in general language modeling when these prompts are prepended, as measured by WikiText2 perplexity.

##### Token Priming via Induction Heads.

In some binary classification tasks, the undesirable behavior is systematically associated with a particular answer label. For example, sycophantic answers are often labeled “(A).” We observed that unconstrained optimization can then converge on *Token Priming*: placing a token in the context to make the model more likely to output it. Some discovered prompts, particularly RESGA prompts, explicitly insert the alternative label, as in “...apol al al b”.

We attribute this to Induction Heads, attention mechanisms associated with copying patterns from context (Olsson et al., 2022). The optimizer discovers that inserting “B” raises the probability of generating “B” through in-context copying. This acts as a “soft” few-shot example: an indirect contextual demonstration rather than an explicit set of task examples. It biases outputs to correct the dataset’s label skew without necessarily changing the underlying semantic reasoning.

##### Semantic Steering via Vocabulary Shift.

SAEGA prompts, in contrast, frequently use *Semantic Steering*. They discover fragments that prime particular reasoning modes rather than simply forcing tokens. We analyze this through changes in output token probabilities, expressed as logit differences:

$$
\Delta\text{Logits}
=
\text{Logits}_{\text{steered}}
-
\text{Logits}_{\text{baseline}}.
$$

Here, $\Delta$ denotes a change. $\text{Logits}_{\text{steered}}$ are token scores with the steering prompt, and $\text{Logits}_{\text{baseline}}$ are scores without it.

- **Sycophancy (Polite Disagreement):** SAEGA systematically suppresses direct agreement markers, including “Yes,” “True,” and “Agree.” It significantly increases the scores of semantic pivots such as “However” and “But.” This suggests that it does not merely force rudeness or contradiction. It primes a “critical evaluation” mode, creating room for nuance before a factual correction.
- **Myopic Reward (Temporal Priming):** The vocabulary-shift analysis shows that RESGA and SAEGA successfully increase scores for tokens associated with long-term planning, including “Future,” “Long,” and “Wait.”

### 4.5 Ablation Studies

We test the framework’s three core design choices through ablations: experiments that replace one component to assess its contribution (Table 5).

**Persona Steering Vector.** Replacing the persona vector with a random direction causes performance to collapse to near or above zero-shot levels across all tasks and models. This confirms that a mechanistically grounded direction is essential (Table 5, Random Direction row; Table 1, Zero-Shot row).

**Gradient-Guided Mutation.** Replacing gradient-guided token selection with random mutation consistently degrades performance. This demonstrates that gradients are critical for directing the search toward persona-suppressing prompts (Table 5, Random Mutation row compared with RESGA and SAEGA rows).

**Representation-Based Objective.** Replacing the representation-projection objective with direct task-loss optimization produces unstable results. Task loss measures the objective on task answers rather than on internal projections. It underperforms RESGA and SAEGA in the majority of settings. This indicates that internal representations provide a more reliable optimization signal than task loss alone (Table 5, Task Loss as Objective row compared with RESGA and SAEGA rows).

Together, these results confirm that all three components contribute meaningfully to effectiveness.

Table 5: Ablation study results. We ablate three core components: the persona steering vector (Random Direction), gradient-guided token mutation (Random Mutation), and the representation-based objective (Task Loss as Objective).

| Method | Sycophancy: Llama $\downarrow$ | Sycophancy: Qwen $\downarrow$ | Sycophancy: Gemma $\downarrow$ | Hallucination: Llama $\downarrow$ | Hallucination: Qwen $\downarrow$ | Hallucination: Gemma $\downarrow$ | Myopic Reward: Llama $\downarrow$ | Myopic Reward: Qwen $\downarrow$ | Myopic Reward: Gemma $\downarrow$ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Random Direction | 74.13 | 84.56 | 81.47 | 68.29 | 54.26 | 84.70 | 50.50 | 48.50 | 61.00 |
| Random Mutation | 68.00 | 71.66 | 76.04 | 50.40 | 45.12 | 54.26 | 52.00 | 46.50 | 55.00 |
| Task Loss as Objective | 55.83 | 73.50 | 58.70 | 46.95 | 45.12 | 54.27 | 48.50 | 48.00 | 51.50 |
| RESGA | 49.86 | 50.63 | 62.65 | 45.12 | 41.46 | 51.22 | 38.50 | 38.50 | 34.50 |
| SAEGA | 49.84 | 49.95 | 70.70 | 46.95 | 40.85 | 49.14 | 31.50 | 40.50 | 38.50 |

## 5 Conclusion

We present a framework connecting mechanistic interpretability with automatic prompt discovery for persona steering. RESGA and SAEGA construct persona vectors from labeled examples and optimize prompts through evolutionary gradient ascent. This achieves interpretable control over LLM behavior.

Experiments on three mitigation tasks show significant improvement on sycophancy and myopic reward, and consistent improvement on hallucination. The discovered prompts substantially outperform manual prompting and dense steering (Table 1, RESGA and SAEGA rows compared with Standard Prompt and Dense Steering Vector rows).

Mechanistic analysis shows that SAEGA succeeds by neutralizing behavior rather than shifting it, preserving natural activation manifolds, and controlling interpretable features (Section 4.3, Figures 3–6). Future work could explore semi-supervised approaches, which combine labeled and unlabeled examples, to reduce labeling requirements. It could also investigate transfer of discovered prompts across models and steering multiple personas.

## 6 Limitations

We state our limitations as follows:

- **Labeled Data Requirement.** RESGA and SAEGA need labeled examples that exhibit and lack the target persona to discover steering prompts.
- **Cross-Model Transferability.** SAE latents and steering vectors are model-specific. They require retraining for each target model.
- **Fluency-Effectiveness Tradeoff.** The most effective prompts identified by our approach do not show semantic coherence. Future work could use strong linguistic priors—constraints favoring natural language—to produce prompts that are both effective and human-like.

## Broader Impact Statement

This work aims to improve LLM safety by suppressing undesired behavioral personas. However, the same pipeline could in principle target safety-critical SAE latents. It could automate jailbreak discovery or elicit harmful outputs. Deployment should therefore include appropriate access controls.

## Acknowledgment

We acknowledge the use of Claude (Anthropic) for manuscript writing assistance and AI-assisted code generation throughout this work.
