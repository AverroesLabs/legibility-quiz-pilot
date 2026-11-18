**Summary**

The paper studies automatic prompt discovery for controlling behavioral personas in large language models (LLMs). It introduces RESGA (RESidual Gradient Ascent), which uses dense residual representations, and SAEGA (Sparse Autoencoder Gradient Ascent), which uses sparse autoencoder features. Both construct persona directions from contrasting examples and optimize prompts with a fluency constraint. Experiments cover sycophancy—agreement with an incorrect user opinion—hallucination, evaluated through TruthfulQA error, and myopic reward, or short-term reward seeking over long-term outcomes. The paper reports improvements across Llama 3.1, Qwen 2.5, and Gemma 3. Its abstract reports sycophancy improvement of 49.90% compared with 79.24%. Mechanistic analyses report that SAEGA suppresses relevant features, concentrates behavior near neutrality, and preserves natural activation sparsity. The paper also identifies limitations involving labeled data, model-specific steering directions, and prompt coherence.

**Glossary**

- **AI** — The systems and safety context in which the paper studies persona control.
- **Large language model (LLM)** — The language models whose behavioral personas the paper seeks to control.
- **Behavioral persona** — A behavioral trait, such as sycophancy, hallucination, or myopic reward, targeted by steering.
- **Persona control / persona steering** — Modification of model behavior toward or away from a target persona.
- **Persona mitigation / suppression / reduction** — Steering a model away from an undesired persona.
- **Sycophancy** — Agreement with a user’s stated opinion even when that opinion is incorrect.
- **Hallucination** — The behavior evaluated using multiple-choice TruthfulQA, where lower error indicates greater factual reliability.
- **Myopic reward** — Short-term reward seeking over long-term outcomes.
- **Emergent misalignment** — Harmful personas appearing on unrelated prompts after fine-tuning on specific datasets.
- **Safety alignment** — Procedures intended to prevent harmful model behavior.
- **Prompt engineering / manual prompting** — Guiding model outputs through carefully written instructions without retraining.
- **Automatic prompt discovery / prompt optimization** — Systematic search for prompts that elicit desired behavior.
- **Black-box optimization** — Optimization without an interpretable connection to model internals.
- **Mechanistic interpretability** — Understanding and controlling behavior through internal representations and features.
- **Sparse dictionary learning (SDL)** — An approach discussed as a route to interpretable model features.
- **Sparse autoencoder (SAE)** — A model that reconstructs neural activations using sparse latent features.
- **Latent / SAE feature** — A sparse autoencoder component that responds to particular concepts or patterns.
- **Persona feature / misaligned persona feature** — An SAE latent that captures a behavioral persona or predicts harmful behavior.
- **Monosemantic neuron** — A neuron that activates in response to a specific pattern.
- **Superposition** — A property of representation space that SAE feature decomposition seeks to reduce.
- **Polysemanticity** — A factor the paper associates with off-manifold results from embedding-space optimization.
- **Activation / representation** — A model’s internal state at a chosen layer.
- **Residual stream / residual activation** — The internal representation space used by RESGA and dense steering.
- **Dense representation** — The residual representation used directly rather than through sparse SAE latents.
- **Steering vector / persona direction** — A direction in activation space representing a target persona.
- **Contrastive activation analysis** — Comparing activations on persona-exhibiting and persona-free examples.
- **Representation difference** — The difference between mean representations of the two example groups.
- **Causally relevant feature** — A feature described as relevant to controlling the target behavior.
- **Activation steering / dense steering** — Directly adding a steering vector to model representations.
- **Activation clamping** — The intervention used in cited work to control SAE-identified persona features.
- **RESGA** — RESidual Gradient Ascent, the proposed method using dense residual representations.
- **SAEGA** — Sparse Autoencoder Gradient Ascent, the proposed method using SAE-based steering directions.
- **Gradient ascent** — Input optimization that increases activation of a target internal component or direction.
- **Fluent gradient ascent / fluent dreaming** — Optimization balancing a steering objective with a language-model fluency term.
- **CNN** — The model class discussed in connection with neuron feature visualization; the abbreviation is not expanded in the paper.
- **Feature visualization** — The cited use of gradient ascent to interpret CNN neurons.
- **Embedding space / token embeddings** — The continuous token representations used to compute optimization gradients.
- **Nearest-token projection** — Mapping an optimized continuous input back to tokens.
- **Discrete token space** — The token-valued space in which language prompts must be expressed.
- **Language manifold / natural data manifold** — The region associated with natural language or natural model activations.
- **Off-manifold** — The paper’s description of token sequences or activations departing from that natural region.
- **Gradient-guided discrete optimization** — Iterative token replacement guided by gradient feedback.
- **Evolutionary algorithm** — A search procedure that generates mutations and selects candidates.
- **Evolutionary Prompt Optimization (EPO)** — The gradient-guided evolutionary procedure used to optimize prompt steering and fluency.
- **Mutation** — Replacement of a prompt token with a sampled candidate token.
- **Gradient-guided mutation** — Token replacement using alternatives selected by gradient magnitude.
- **Random mutation** — The ablation that replaces gradient-guided selection with random token mutation.
- **Population** — The candidate prompts maintained during EPO.
- **Pareto frontier** — The set of prompts spanning fluency–steering tradeoffs.
- **Restart** — Retaining the best prompt under a random fluency weight and reinitializing the others.
- **Random initialization** — Starting prompts with uniformly sampled vocabulary tokens.
- **Seed initialization** — Starting prompts with a short natural-language phrase loosely related to the persona.
- **Fluency** — The model-based constraint measured by prompt self cross-entropy, not necessarily readability or grammatical correctness.
- **Cross-entropy / self cross-entropy** — The model-based measure of how likely a prompt’s tokens are under their prefixes.
- **Perplexity** — The language-modeling measure reported for WikiText2 and used when discussing prompt fluency.
- **Fluency–effectiveness tradeoff** — The balance between model-based prompt fluency and persona-steering strength.
- **Validation data / held-out validation split** — Data used to evaluate discovered prompts outside their optimization examples.
- **Prompt stability** — Consistency in reducing a target persona across diverse validation examples.
- **Degenerate solution** — A prompt with strong steering on only a narrow subset of inputs.
- **Context-aware optimization** — Optimizing prompts while appending them to dynamic batches of task-specific training questions.
- **Reinforcement learning from human feedback (RLHF)** — A training-time approach cited for aligning model behavior.
- **Constitutional AI** — Another cited training-time alignment approach.
- **Representation Engineering (RepE)** — Extraction of concept directions from contrasting model activations for inference-time control.
- **Contrastive Activation Addition (CAA)** — Steering through directions identified from curated contrast datasets and applied at specific layers.
- **JumpReLU** — A cited SAE architecture technique.
- **Top-K** — A cited SAE architecture technique; separately, top-\(K\) denotes selection of the most relevant latents.
- **Batch Top-K** — A cited technique for SAE feature extraction.
- **Matryoshka Sparse Autoencoders** — A cited SAE architecture for interpretable feature extraction.
- **LLM-based analysis** — Use of language models to interpret SAE-discovered neurons.
- **Subpopulation** — A set of samples activated by a particular neuron in the cited interpretation workflow.
- **In-context learning** — Task adaptation through demonstrations supplied in a prompt.
- **GPT-3** — The model cited as demonstrating broad task adaptation through in-context learning.
- **Chain-of-thought (CoT) prompting** — Prompting that generates step-by-step intermediate reasoning.
- **Self-consistency decoding** — Sampling several reasoning paths and selecting the most consistent answer.
- **Program-Aided Language models (PAL)** — Methods that use programmatic steps and external interpreters for computation.
- **Program of Thoughts (PoT)** — Methods that delegate computation to external interpreters after programmatic decomposition.
- **External interpreter / external tool** — The computation mechanism used by the cited hybrid reasoning approaches.
- **Generative agents** — LLM-powered systems cited as simulating human behavior and emergent social dynamics through prompts.
- **Optimal control problem** — The theoretical perspective on prompt engineering cited from Luo et al. (2023).
- **ProTeGi** — Prompt refinement using LLM-generated natural-language “gradients,” beam search, and bandits.
- **API** — The interface through which ProTeGi obtains LLM-generated prompt criticism.
- **Natural-language “gradient”** — A criticism of the current prompt used to guide semantic editing.
- **Beam search / bandits** — The search and efficiency mechanisms used by ProTeGi.
- **Greedy Coordinate Gradient (GCG)** — Greedy, gradient-scored token substitution for discovering behavior-eliciting prompts.
- **Output logits / logits** — The model outputs directly optimized by GCG and compared in the vocabulary-shift analysis.
- **Logit bias** — The accumulated model bias that final-layer interventions often failed to override.
- **SAE-Lens / SAELens** — The source of pretrained SAEs used by SAEGA.
- **Llama 3.1 8B Instruct** — One experimental model; also abbreviated as Llama or Llama 3.1.
- **Qwen 2.5 7B Instruct** — One experimental model; also abbreviated as Qwen or Qwen 2.5.
- **Gemma 3 4B Instruct** — One experimental model; also abbreviated as Gemma or Gemma 3.
- **8B / 7B / 4B / Instruct** — Codes appearing in the experimental model names.
- **Cross-architecture generalization** — Evaluation across the three model architectures.
- **TruthfulQA** — The multiple-choice benchmark used to evaluate hallucination.
- **Zero-Shot** — The baseline with no intervention.
- **Standard Prompt** — The baseline using manually written instructions.
- **Dense Steering Vector** — The baseline that directly injects a representation-difference vector into the residual stream.
- **GPT-4o** — The model named for the ProTeGi baseline in Table 1.
- **Random Direction** — The ablation replacing the persona steering vector with a random direction.
- **Task Loss as Objective** — The ablation replacing representation projection with direct task-loss optimization.
- **Conditional log-probability comparison** — Selecting the candidate answer with the higher log-probability given a question and steering prompt.
- **Error rate / score / behavioral metric** — The fraction of examples on which the model prefers the undesirable option.
- **Neutralization** — For sycophancy, a \(50\%\) error rate with neither systematic agreement nor disagreement.
- **Behavioral reversal** — Steering beyond neutrality toward the opposite behavior.
- **Distributional shifting / coarse counterbalancing** — Moving the mean persona projection while retaining substantial variance.
- **Instance-wise neutrality** — The near-zero, low-variance behavior attributed to SAEGA.
- **Sycophancy axis** — The persona direction onto which activations are projected in Figure 3.
- **Sparsity / \(L_0\) norm** — The number of active SAE features.
- **Natural sparse topology** — The internal sparse activation structure that SAEGA is said to preserve.
- **PCA** — The visualization used for steering trajectories in residual activation space.
- **Steering trajectory** — The path of internal representations under steering.
- **Orthogonal subspace** — The distinct region, often orthogonal to dense steering, described for SAEGA.
- **Token Priming** — Injecting a label token into a prompt to bias its subsequent generation.
- **Induction Heads** — The cited mechanism to which the paper attributes in-context token copying.
- **In-context copying** — Increased generation probability for a token injected into the prompt.
- **“Soft” few-shot example** — The paper’s description of label-token priming without necessarily changing semantic reasoning.
- **Dataset skew** — Systematic alignment between an undesirable behavior and a label.
- **Semantic Steering / vocabulary shift** — Priming reasoning modes by changing output-token probabilities.
- **Polite Disagreement / critical evaluation** — The sycophancy-related mode associated with suppressing agreement markers and upweighting pivots.
- **Temporal Priming** — Upweighting tokens associated with long-term planning.
- **WikiText2** — The dataset used for the language-modeling perplexity evaluation.
- **Ablation study** — Replacement of a framework component to assess its contribution.
- **Semi-supervised approach** — A future direction proposed to reduce labeled-data requirements.
- **Cross-model transferability** — Transfer of discovered prompts or steering mechanisms between models.
- **Multi-persona steering** — A future extension to steering more than one persona.
- **Linguistic prior** — A proposed future constraint for obtaining effective, human-like prompts.
- **Jailbreak discovery** — A potential misuse of the pipeline to elicit harmful outputs.
- **Access controls** — The deployment safeguard recommended in the impact statement.
- **Claude (Anthropic)** — The tool acknowledged for manuscript writing assistance and AI-assisted code generation.
- **OpenReview** — The review site linked in the abstract.
- **GitHub repository** — The linked release location for RESGA and SAEGA scripts.
- **\(\mathbf v\in\mathbb R^d\)** — A persona steering vector in the model’s \(d\)-dimensional activation space.
- **\(\mathbb R^d\), \(d\)** — The activation space and its dimension.
- **\(\mathbf v_{\mathrm{repr}}\)** — The steering vector formed from mean representation differences.
- **\(\mathbf v_{\mathrm{SAE}}\)** — The steering vector formed from selected SAE encoder directions.
- **\(D^+\), \(D^-\)** — Persona-exhibiting and persona-free example sets, respectively.
- **\(|D^+|\), \(|D^-|\)** — The sizes of those example sets.
- **\(\mathbf x\)** — An example in a contrastive dataset.
- **\(\mathbf e(\mathbf x)\), \(\mathbf e(\mathbf t)\), \(e(\mathbf t)\)** — A chosen-layer activation; for a prompt, the representation of its last token.
- **\(K\)** — The number of SAE latents selected by absolute activation difference.
- **\(\mathcal I_K\)** — The index set of the selected top-\(K\) latents.
- **\(i\)** — A latent, token-position, or prompt index, depending on the expression.
- **\(\mathbf W_{\mathrm{enc}}\)** — The SAE encoder matrix.
- **\(\mathbf w_i\)** — Row \(i\) of the SAE encoder matrix.
- **\(a_i(\mathbf x)\)** — Activation of SAE latent \(i\) on example \(\mathbf x\).
- **\(\bar a_i^+\), \(\bar a_i^-\)** — Mean latent activations on persona-exhibiting and persona-free examples.
- **\(b_i\)** — The bias term in the formula for latent \(i\)’s activation.
- **ReLU** — The activation function appearing in the SAE latent formula.
- **\(\mathbf t\), \(\mathbf t^i\)** — A prompt and the \(i\)-th candidate prompt.
- **\(t_{i+1}\), \(\mathbf t_{\leq i}\)** — The next token and its preceding prompt prefix.
- **\(f(\mathbf t)\), \(f(\cdot)\)** — The persona-reduction objective: negative scaled projection onto the steering vector.
- **\(\langle\cdot,\cdot\rangle\)** — The projection numerator used in Equation (3).
- **\(\|\mathbf e(\mathbf t)\|\)** — The representation norm used to scale that projection.
- **\(\mathcal L_\lambda(\mathbf t)\), \(L_\lambda(\mathbf t)\)** — The combined steering-and-fluency objective.
- **\(H\)** — Cross-entropy; Figure 2 also uses this symbol in its displayed objective.
- **\(m\)** — The language model used to evaluate prefix-conditioned token likelihood.
- **\(\lambda\), \(\lambda_i\), \(\{\lambda_1,\ldots,\lambda_M\}\)** — Fluency weights controlling the steering–fluency tradeoff.
- **\(n\)** — Prompt length.
- **\(M\)** — Population size.
- **\(T\)** — Number of optimization iterations.
- **\(k\)** — Number of candidate replacement tokens selected per position by gradient magnitude.
- **One-hot token encodings** — The token representation with respect to which Algorithm 1 computes gradients.
- **w.r.t.** — With respect to.
- **\(q\), \(p\), \(a\)** — An evaluation question, steering prompt, and candidate answer.
- **\(\log P(a\mid q,p)\)** — Conditional log-probability of a candidate answer given its question and steering prompt.
- **\(\Delta\mathrm{Logits}\)** — Steered logits minus baseline logits.
- **\(\mathrm{Logits}_{\mathrm{steered}}\), \(\mathrm{Logits}_{\mathrm{baseline}}\)** — Output logits with steering and without steering.
- **\(\sum\), \(\in\), \(\cdot\)** — Summation, set membership, and multiplication in the paper’s formulas.
- **\(\approx\)** — The approximate relation used for the SAE’s natural sparsity level.
- **\(\downarrow\)** — Lower is better.
- **Avg.** — The average reported in Table 1.
- **Seed / Optimized** — The original and EPO-discovered prompt types in Table 3.
- **RESGA \(\&\) SAEGA** — The combined prompt-intervention column label in Table 4.
- **“(A),” “B,” and label-token examples** — Answer labels used to explain Token Priming.
- **“Yes,” “True,” “Agree”** — Agreement markers reported as suppressed by SAEGA.
- **“However,” “But”** — Semantic pivots reported as upweighted by SAEGA.
- **“Future,” “Long,” “Wait”** — Long-term-planning tokens reported as upweighted by RESGA and SAEGA.
- **Literal prompt strings in Tables 2–3** — The reported random-initialized, seed, and optimized prompts.
- **`\n`, `\u…`, `<unused3147>`, `<unused5655>`, `<bos>`** — Literal escape sequences and special-token strings printed in the prompt examples.
- **ISM / `11footnotemark: 1` / \(\dagger\)** — Affiliation and author-footnote notation reproduced from the author list; the supplied author note says “Equal contribution.”

# Bridging Mechanistic Interpretability and Prompt Engineering with Gradient Ascent for Interpretable Persona Control

Harshvardhan Saini hs1062005@gmail.com
††thanks: Equal contribution.
Affiliation: Indian Institute of Technology (ISM), Dhanbad

  
Yiming Tang11footnotemark:
1

yiming@nus.edu.sg

Affiliation: National University of Singapore

  
Dianbo Liu† dianbo@nus.edu.sg

Affiliation: National University of Singapore

###### Abstract

*The paper introduces two representation-guided prompt-search methods and reports persona-steering results across three model families.*

Controlling emergent behavioral personas—traits such as sycophancy and hallucination—in Large Language Models (LLMs) is important for AI safety but remains difficult. Sycophancy means agreeing with a user’s opinion even when it is incorrect. Hallucination is evaluated here through multiple-choice TruthfulQA error. Manual prompt engineering, or writing instructions to guide outputs, is intuitive but unscalable and imprecise. Automatic optimization is effective but operates as a “black box,” without an interpretable connection to model internals.

We propose a framework that adapts gradient ascent—optimization toward a target internal activation or direction—to targeted prompt discovery. RESGA (RESidual Gradient Ascent) uses dense residual representations. SAEGA (Sparse Autoencoder Gradient Ascent) uses features from sparse autoencoders (SAEs), which reconstruct model activations with sparse latent features. Both optimize randomly initialized prompts to align their representations better with an identified persona direction.

We introduce fluent gradient ascent to control the discovered prompts’ fluency. We demonstrate both methods across Llama 3.1, Qwen 2.5, and Gemma 3 on sycophancy, hallucination, and myopic reward. Myopic reward means seeking short-term reward over long-term outcomes. On sycophancy, automatically discovered prompts achieve significant improvement: 49.90% compared with 79.24%.

Grounding prompt discovery in mechanistically meaningful features offers a new paradigm for controllable and interpretable behavior modification. We release RESGA and SAEGA scripts at https://github.com/HarshSaini10/RESGA_SAEGA.

Reviewed on OpenReview: https://openreview.net/forum?id=dcmHPxgo4c

**Claims and evidence**

- **The methods improve sycophancy steering.** This abstract reports **49.90% compared with 79.24%**.
- **The evaluation covers three personas and three model families.** Table 1, RESGA and SAEGA rows, reports sycophancy, hallucination, and myopic reward for Llama, Qwen, and Gemma.

## 1 Introduction

*This section motivates representation-guided prompt discovery and introduces RESGA and SAEGA.*

Figure 1: Comparison of persona steering approaches.  
(a): Prompt-based steering provides implicit control via manually designed natural language instructions.  
(b): Activation steering injects a dense steering vector \(\mathbf{v}\) into model representations \(e(\mathbf{t})\).  
(c): Our methods, RESGA and SAEGA, derive persona directions from contrastive activations and optimize prompts via fluent gradient ascent, enabling both residual-based and interpretable feature-level control.

LLMs have achieved capabilities across domains, changing AI applications from education to healthcare. Their growing use in high-stakes settings makes behavioral persona control an important AI-safety challenge. Fine-tuning on specific datasets can induce “emergent misalignment”: harmful personas appear in responses to unrelated prompts (Wang et al., 2025). Widely deployed safety-alignment procedures still often fail to prevent harmful personas when models are prompted appropriately.

Prompt engineering guides outputs through instructions without expensive retraining. Effective instructions require substantial human expertise and domain knowledge. They must elicit the desired behavior while preserving output quality. This is especially difficult for complex traits such as honesty, helpfulness, and domain-specific personas, because the connection between prompt content and behavior remains poorly understood. Automatic prompt discovery addresses this manual burden, but existing methods often use black-box optimization without interpretable insights into model internals.

Mechanistic interpretability studies behavior through internal representations and features. Recent work on sparse dictionary learning (SDL) and persona steering vectors has opened new routes to understanding and controlling LLMs (Zhao et al., 2026; Chen et al., 2025). A steering vector is a direction in representation space associated with a target concept or behavior. SAEs and their variants decompose activations into interpretable features that respond to particular concepts or patterns (Cunningham et al., 2023a; Bricken et al., 2023; Rajamanoharan et al., 2024b).

Persona-feature work shows that SAE latents—individual sparse features—can capture and control behavioral personas. “Misaligned persona” features strongly predict emergent harmful behavior (Wang et al., 2025). Existing work mainly identifies which latents activate for an input or intervenes directly through activation steering, meaning changes to internal activations. The inverse problem remains largely unexplored: how can prompts be discovered that selectively activate task-relevant latents and steer personas?

Gradient ascent has been successful in feature visualization, which interprets CNN neurons by optimizing inputs to activate them (Olah et al., 2017). Its use in LLMs is limited by language’s discrete nature. Standard gradient ascent optimizes continuous inputs to activate neurons, attention patterns, or learned representation-space directions. Directly optimizing token embeddings—continuous token representations—and then projecting to the nearest token yields off-manifold results. The paper attributes this to polysemanticity and the mismatch between continuous optimization and discrete token space (Wallace et al., 2019).

Gradient-guided discrete optimization instead uses gradient feedback to replace tokens iteratively through evolutionary algorithms (Zou et al., 2023c; Thompson et al., 2024). These methods have discovered prompts for adversarial attacks and jailbreaking (Andriushchenko et al., ), but have not been applied to identifying model-steering prompts.

Our framework connects mechanistic interpretability with automatic prompt discovery for persona steering (Figure 1). RESGA uses dense residual-stream representations, the model’s internal residual activations. SAEGA uses mechanistically interpretable SAE latents. Both first construct directions capturing undesired behavioral traits. RESGA uses direct representation differences. SAEGA uses latent-based approaches that identify causally relevant features. Fluent gradient ascent then discovers prompts that maximally steer away from target personas while allowing control over fluency.

Unlike black-box prompt optimization, our search uses mechanistic insights for targeted, interpretable control. Across sycophancy, hallucination, and myopic reward, discovered prompts significantly improve sycophancy and myopic reward performance (Section 4.2), substantially outperforming manual prompting and dense steering. Representation-level evidence further shows that prompt steering avoids the distributional shifts induced by dense steering, exposing a fundamental limitation of dense steering.

**Claims and evidence**

- **Discovered prompts substantially outperform manual prompting and dense steering.** Table 1, Avg. column: Standard Prompt **55.45**, Dense Steering Vector **56.10**, RESGA **45.83**, SAEGA **46.44**.
- **The methods improve sycophancy and myopic reward.** Table 1, RESGA row: sycophancy **49.86, 50.63, 62.65** and myopic reward **38.50, 38.50, 34.50** for Llama, Qwen, Gemma. SAEGA row: **49.84, 49.95, 70.70** and **31.50, 40.50, 38.50**.
- **Prompt steering avoids dense steering’s distributional disruption.** Figures 3–6; Section 4.3 reports approximately **50** baseline active features, **beyond 150** under dense steering, and **50–60** under SAEGA.

## 2 Related Works

*This section reviews persona steering, sparse autoencoders, and prompt-engineering methods.*

### 2.1 Persona Steering

*This subsection reviews training-time alignment and inference-time steering through dense directions or SAE features.*

Persona control is increasingly important for LLM safety and alignment. Traditional training-time approaches include reinforcement learning from human feedback (RLHF) (Ouyang et al., 2022) and constitutional AI (Bai et al., 2022).

Representation Engineering (RepE) extracts concept directions from contrasting pairs of activations and adds them during inference to control outputs (Zou et al., 2023a). Contrastive Activation Addition (CAA) refines this approach with curated contrast datasets and steering vectors applied at particular layers (Rimsky et al., 2023). Activation steering has mitigated sycophancy (Sharma et al., ), toxicity (Liu et al., ), and bias (Tigges et al., 2023).

Wang et al. (2025) found interpretable SAE “persona features” that predict emergent misalignment. Their work shows precise persona control through activation clamping, an intervention on SAE-identified features.

### 2.2 Sparse Autoencoder

*This subsection reviews sparse feature reconstruction, SAE architectures, and interpretation workflows.*

SAEs, first introduced in Cunningham et al. (2023b), discover interpretable representations of neural-network activations (Dunefsky et al., 2024; Tang et al., 2026). They reconstruct representations using sparse latent features. This can reveal monosemantic neurons—neurons responding to specific patterns—and reduce superposition in representation space (Elhage et al., 2022).

SAE architecture techniques include JumpReLU (Rajamanoharan et al., 2024a), Top-K (Gao et al., 2024), Batch Top-K (Bussmann et al., ), and Matryoshka Sparse Autoencoders (Bussmann et al., 2025). These have improved large-scale interpretable feature extraction (Bricken et al., 2023).

LLM-based analysis is often used to interpret SAE-discovered neurons (Luo et al., ; Luo et al., 2024a; Tang et al., 2025a). A typical workflow finds subpopulations—samples activated by particular neurons—and analyzes those samples using prompt engineering (Luo et al., 2024b; Tang et al., 2025b).

### 2.3 Prompt Engineering

*This subsection reviews prompting, reasoning, tool use, and automatic prompt optimization.*

Prompt engineering guides LLMs through complex tasks (Tang and Dong, 2024; Zou et al., 2026). GPT-3 provided the first extensive demonstration of this paradigm through in-context learning: models adapt to tasks through demonstrations included in a prompt (Brown et al., 2020).

Chain-of-thought (CoT) prompting further enhances LLM reasoning by generating step-by-step intermediate reasoning steps (Wei et al., 2022). Self-consistency decoding improves performance by sampling several reasoning paths and selecting the most consistent answer (Wang et al., 2023). Hybrid methods address arithmetic limitations by combining natural-language reasoning with external computation. Program-Aided Language models (PAL) (Gao et al., 2023) and Program of Thoughts (PoT) (Chen et al., 2023) decompose problems into programmatic steps and delegate computation to external interpreters. They substantially improve mathematical reasoning.

Evolutionary prompt optimization (EPO) uses gradient-guided evolutionary algorithms to balance an objective with language fluency (Thompson and Sklar, 2024). Generative agents show that carefully prompted LLM systems can simulate believable human behavior and emergent social dynamics (Park et al., 2023). Luo et al. (2023) provide a unified theoretical framework treating prompt engineering as an optimal control problem.

ProTeGi automates refinement through an LLM API. It generates natural-language “gradients”—criticisms of a prompt—and edits in the opposite semantic direction. Beam search and bandits guide the process efficiently (Pryzant et al., ). Greedy Coordinate Gradient (GCG) combines greedy search with gradient-based token scoring and iterative substitution to elicit target behaviors without manual engineering (Zou et al., 2023b).

## 3 Method

*This section describes steering-vector construction, fluent prompt optimization, and prompt initialization and selection.*

Our framework has three stages:

1. Construct a persona steering vector from direct representation differences or SAE latents (Section 3.1).
2. Use fluent gradient ascent to discover prompts that steer away from the persona while maintaining fluency (Section 3.2).
3. Initialize and select prompts from the Pareto frontier—the set spanning fluency–steering tradeoffs—using validation performance (Section 3.3).

Figure 2: RESGA & SAEGA Framework Overview. Our framework discovers persona-steering prompts in two stages: (1) Persona steering vector construction via either dense representations (RESGA) or sparse SAE latents (SAEGA), and (2) Fluent gradient ascent optimization with objective \(L_{\lambda}(\mathbf{t})=f(\mathbf{t})+H(\mathbf{t})\) that transforms random token sequences into readable prompts that steer the model in specific directions for interpretable persona control.

### 3.1 Persona Steering Vectors

*This subsection constructs persona directions from mean residual representations or selected SAE encoder directions.*

For a persona to suppress, we construct \(\mathbf v\in\mathbb R^d\), a steering vector in the model’s \(d\)-dimensional activation space.

**Representation-Based Steering.** We compute the difference between mean representations of persona-exhibiting and persona-free examples:

\[
\mathbf{v}_{\text{repr}}
=
\frac{1}{|D^{+}|}\sum_{\mathbf{x}\in D^{+}}\mathbf{e}(\mathbf{x})
-
\frac{1}{|D^{-}|}\sum_{\mathbf{x}\in D^{-}}\mathbf{e}(\mathbf{x}).
\tag{1}
\]

Here, \(D^+\) is the persona-exhibiting example set and \(D^-\) is the persona-free set. The expressions \(|D^+|\) and \(|D^-|\) are their sizes. An example is denoted by \(\mathbf x\); \(\mathbf e(\mathbf x)\in\mathbb R^d\) is its activation at the chosen layer. The sums average activations over each set.

**SAE Latent-Based Steering.** SAEGA alternatively constructs directions from pretrained SAEs provided by SAE-Lens (Bloom et al., 2024).

We identify relevant latents through contrastive activation analysis: for each latent \(i\), compare its mean activation on persona-exhibiting and persona-free examples. Select the top-\(K\) latents with the largest absolute differences. Here, \(K\) is the number selected, and \(\mathcal I_K\) is their index set.

The steering vector uses their encoder directions:

\[
\mathbf{v}_{\text{SAE}}
=
\sum_{i\in\mathcal{I}_{K}}
\mathbf{w}_{i}\cdot
\left(\bar{a}_{i}^{+}-\bar{a}_{i}^{-}\right).
\tag{2}
\]

The vector \(\mathbf w_i\) is row \(i\) of the SAE encoder matrix \(\mathbf W_{\mathrm{enc}}\). The mean activations are

\[
\bar{a}_{i}^{+}
=
\frac{1}{|D^{+}|}
\sum_{\mathbf{x}\in D^{+}}a_{i}(\mathbf{x}),
\qquad
\bar{a}_{i}^{-}
=
\frac{1}{|D^{-}|}
\sum_{\mathbf{x}\in D^{-}}a_{i}(\mathbf{x}).
\]

Latent activation is

\[
a_i(\mathbf x)
=
\text{ReLU}\!\left(\mathbf w_i\mathbf e(\mathbf x)+b_i\right),
\]

where \(b_i\) is its bias term and ReLU is the activation function in this expression.

### 3.2 Fluent Gradient Ascent

*This subsection combines persona reduction with a fluency penalty and optimizes the result through EPO.*

Given \(\mathbf v\), we search for prompts that maximally steer away from the persona. For prompt \(\mathbf t\), define the persona-reduction objective as its negative scaled projection onto \(\mathbf v\):

\[
f(\mathbf{t})
=
-\frac{\langle\mathbf{e}(\mathbf{t}),\mathbf{v}\rangle}
{\|\mathbf{e}(\mathbf{t})\|}.
\tag{3}
\]

Here, \(\mathbf e(\mathbf t)\) is the last token’s representation at the target layer. The numerator is the projection’s inner-product term, and the denominator is the representation norm. Maximizing \(f\) finds representations pointing away from the persona direction.

Optimizing \(f\) alone often produces unnatural or nonsensical prompts. Following fluent dreaming (Thompson et al., 2024), we balance steering with fluency:

\[
\mathcal{L}_{\lambda}(\mathbf{t})
=
f(\mathbf{t})
-
\frac{\lambda}{n}
\sum_{i=0}^{n-1}
H\!\left(m(\mathbf{t}_{\leq i}),t_{i+1}\right).
\tag{4}
\]

Here:

- \(n\) is prompt length.
- \(m\) is the language model.
- \(\mathbf t_{\leq i}\) is the prefix before next token \(t_{i+1}\).
- \(H\) is cross-entropy, measuring that next token’s likelihood under the model.
- \(\lambda\) controls the fluency–steering tradeoff.

The second term penalizes low-probability sequences and encourages human-interpretable prompts.

We optimize with EPO (Thompson et al., 2024). It maintains \(M\) candidate prompts, where \(M\) is population size. Each prompt targets a different fluency–steering tradeoff.

**Algorithm 1 Evolutionary Prompt Optimization**

1. **Inputs:** Steering objective \(f(\cdot)\), language model \(m\), prompt length \(n\), population size \(M\), fluency weights \(\{\lambda_1,\ldots,\lambda_M\}\).
2. **Output:** Prompts spanning the fluency–steering Pareto frontier.
3. Initialize \(M\) prompts of length \(n\) with random initialization or seed initialization.
4. **for** iteration \(=1\) to \(T\) **do**
5.   **for each** prompt \(\mathbf t^i\) **do**
6.     Compute \(\mathcal L_{\lambda_i}(\mathbf t^i)\) and gradients w.r.t. one-hot token encodings.
7.     Select top-\(k\) candidate tokens per position by gradient magnitude.
8.     Generate mutations by replacing a random token with a sampled top-\(k\) alternative.
9.   **end for**
10.   Evaluate all candidates and select the best prompt for each \(\lambda_i\).
11.   **if** restart step **then**
12.     Retain best prompt under a random \(\lambda\), reinitialize others.
13.   **end if**
14. **end for**

Here, \(T\) is the iteration count; \(\mathbf t^i\) is candidate prompt \(i\); and \(k\) is the number of candidate replacement tokens per position. “w.r.t.” means “with respect to”; one-hot token encodings are the token representations used for gradients in the algorithm.

At each iteration, EPO computes gradients with respect to token embeddings, samples high-gradient substitutions, and selects the best candidates for each \(\lambda_i\). Periodic restarts prevent premature convergence by retaining one best prompt and reinitializing the rest. This efficiently explores the Pareto frontier and produces diverse prompts with different fluency–effectiveness tradeoffs.

### 3.3 Initialization and Selection of Persona Steering Prompts

*This subsection describes random and seeded starts, validation-based selection, model-based fluency, and stability evaluation.*

We use two EPO initialization strategies.

**Random initialization** uniformly samples vocabulary tokens. It tends to produce syntactically fragmented or semantically incoherent discovered prompts.

**Seed initialization** starts with a short natural-language phrase loosely related to the persona, such as “Please answer honestly” for sycophancy. It anchors optimization closer to the language manifold—the region associated here with natural language. Seed-initialized prompts occupy higher-fluency parts of the Pareto frontier without significantly sacrificing steering effectiveness. They are preferable when prompt interpretability is desired.

EPO produces candidates spanning persona-steering strength and language-model fluency. We append each prompt to held-out validation inputs and measure the change in the target behavioral metric. The strongest prompts are often fragmented, multilingual, or semantically incoherent.

We therefore do not equate fluency with readability or grammatical correctness. Fluency is an intrinsic model-based constraint measured by the prompt’s self cross-entropy. Lower cross-entropy places a prompt closer to the model’s training distribution. Higher cross-entropy corresponds to increasingly off-manifold token sequences.

Rather than impose a hard fluency threshold, we report performance across the full Pareto frontier. This shows how behavioral control changes as prompts move farther from the language manifold. Downstream analyses emphasize intermediate prompts that combine measurable steering with moderate perplexity increases relative to random-token baselines. Perplexity is the language-modeling measure used here to discuss these changes.

We also assess prompt stability: how consistently a prompt reduces the persona across diverse validation examples. This guards against degenerate solutions that work strongly on only a narrow input subset. The protocol prioritizes causal steering efficacy over linguistic naturalness, reflecting our goal of understanding how internal representations produce emergent personas.

**Claims and evidence**

- **Seed initialization yields effective prompts closer to natural language.** Table 3, Seed → Optimized row pairs: **69.16% → 56.25%**, **72.30% → 54.22%**, **71.15% → 51.72%**, **72.06% → 59.35%**, **52.43% → 46.34%**, and **51.20% → 46.95%**.
- **Strong prompts can be fragmented or incoherent.** Table 2, random-initialized RESGA and SAEGA prompt examples across all task/model rows.

## 4 Experiments

*This section evaluates persona mitigation, compares the two methods, and analyzes stability and mechanism.*

We evaluate RESGA and SAEGA on three persona-steering tasks:

1. Can discovered prompts reliably neutralize undesired personas?
2. How do residual-based and SAE-based steering differ in effectiveness and mechanism?
3. Does mechanistic structure produce more stable and interpretable control?

Section 4.1 describes datasets, models, and baselines. Section 4.2 reports sycophancy, hallucination, and myopic reward results. Section 4.3 analyzes how SAEGA controls personas through targeted feature suppression while preserving natural activation structure.

### 4.1 Experimental Setup

*This subsection specifies the tasks, models, baselines, optimization settings, and answer-selection protocol.*

##### Tasks and Datasets.

*The evaluation uses three benchmarks, with lower metrics treated as better.*

We evaluate three established benchmarks:

- **Sycophancy** (Perez et al., ): whether a model agrees with a user’s stated opinion even when it is incorrect.
- **Hallucination:** multiple-choice TruthfulQA (Lin et al., 2022), where lower error means improved factual reliability.
- **Myopic Reward** (Perez et al., ): short-term reward seeking over long-term outcomes.

Lower metrics are considered better.

##### Models.

*The evaluation uses three model architectures and pretrained SAEs for SAEGA.*

We use Llama 3.1 8B Instruct (Grattafiori et al., 2024), Qwen 2.5 7B Instruct (Hui et al., ), and Gemma 3 4B Instruct (Team et al., 2025) to assess cross-architecture generalization. These are abbreviated as Llama, Qwen, and Gemma in tables; 8B, 7B, 4B, and Instruct are parts of their model names. SAEGA uses pretrained sparse autoencoders from SAELens (Bloom et al., 2024).

##### Baselines.

*Five baselines cover no intervention, manual prompting, direct activation injection, and two automatic prompt optimizers.*

We compare against:

1. **Zero-Shot:** no intervention.
2. **Standard Prompt:** manually written instructions, such as “Answer honestly.”
3. **Dense Steering Vector:** direct injection of a representation-difference steering vector into the residual stream.
4. **GCG:** prompt discovery by directly optimizing output logits.
5. **ProTeGi:** prompt optimization using LLM-generated natural-language “gradients,” bandits, and beam search.

##### Implementation Details.

*Optimization targets middle-to-late layers, uses context-aware EPO, and reports SAEGA with \(K=20\).*

We target Layer 25 in Llama 3.1 8B and Qwen 2.5 7B, and Layer 20 in Gemma 3 4B. Empirical analysis indicated that early layers lack the high-level semantic abstraction needed to isolate complex behavioral traits, making steering ineffective. Final-layer interventions often failed to override accumulated logit bias, the model bias described at those layers.

Preliminary EPO sweeps balanced stability and computational cost. Every reported experiment uses population size \(M=100\), prompt length \(n=8\), and \(T=1000\) iterations.

We use context-aware optimization. At each evolutionary step, candidate prompts are appended to a dynamic batch of task-specific training questions. Loss is calculated from the internal state after the combined sequence. This ensures that the prompt acts as a generalized steering trigger robust to varying contexts, rather than being optimized in isolation.

SAEGA selects the top-\(K\) latents most strongly correlated with the concept, using activation differences between concept-positive and concept-negative examples. We evaluate multiple \(K\) values corresponding to fractions of the SAE’s natural sparsity level, \(L_0\approx50\). Here, \(L_0\) counts active SAE features. We report \(K=20\), which consistently gave strong performance. All results use held-out validation splits.

**Claims and evidence**

- **Middle-to-late layers were effective targets; early and final layers had the stated shortcomings.** Implementation Details reports **Layer 25** for Llama and Qwen and **Layer 20** for Gemma, together with the empirical layer-selection observations.
- **The selected latent count consistently performed strongly.** Implementation Details reports **\(K=20\)** against natural sparsity **\(L_0\approx50\)**.

##### Evaluation Protocol.

*Predictions are selected by conditional answer log-probability, and metrics count undesirable preferences.*

All tasks use conditional log-probability comparison. For question \(q\), steering prompt \(p\), and candidate answer \(a\), compute

\[
\log P(a\mid q,p).
\]

The prediction is the answer with the higher conditional log-probability. The metric is the fraction of examples where the model prefers the undesirable option: sycophantic, hallucinated, or myopic. Lower values mean improved persona control.

### 4.2 Persona Steering Results

*This subsection reports task error rates and claims effective sycophancy neutralization and improvements on the other personas.*

Table 1 reports error rates across three persona-mitigation tasks and two model families. For sycophancy, \(50\%\) error means perfect neutralization: neither systematic agreement nor systematic disagreement with the user. Neutralization differs from behavioral reversal, or steering toward the opposite behavior.

Table 1: Persona steering results across three tasks and three models. Lower is better.

| Method | Sycophancy ↓ Llama | Sycophancy ↓ Qwen | Sycophancy ↓ Gemma | Hallucination ↓ Llama | Hallucination ↓ Qwen | Hallucination ↓ Gemma | Myopic Reward ↓ Llama | Myopic Reward ↓ Qwen | Myopic Reward ↓ Gemma | Avg. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Zero-Shot | 72.48 | 86.00 | 82.00 | 51.45 | 45.00 | 57.97 | 54.00 | 55.00 | 58.00 | 62.43 |
| Standard Prompt | 70.50 | 72.00 | 80.00 | 50.72 | 34.78 | 55.07 | 52.00 | 46.00 | 38.00 | 55.45 |
| Dense Steering Vector | 54.50 | 79.00 | 80.00 | 50.72 | 36.96 | 57.25 | 47.50 | 47.50 | 51.50 | 56.10 |
| GCG | 55.83 | 73.50 | 58.70 | 46.95 | 45.12 | 54.27 | 48.50 | 48.00 | 51.50 | 53.60 |
| ProTeGi (GPT-4o) | 54.77 | 56.13 | 71.51 | 48.17 | 46.95 | 38.41 | 42.50 | 20.99 | 47.49 | 47.44 |
| RESGA (Ours) | 49.86 | 50.63 | 62.65 | 45.12 | 41.46 | 51.22 | 38.50 | 38.50 | 34.50 | 45.83 |
| SAEGA (Ours) | 49.84 | 49.95 | 70.70 | 46.95 | 40.85 | 49.14 | 31.50 | 40.50 | 38.50 | 46.44 |

The downward arrows mean lower is better; Avg. is the reported average. GPT-4o names the model used for the ProTeGi row.

On sycophancy, both RESGA and SAEGA achieve error rates statistically indistinguishable from \(50\%\). This indicates effective neutralization rather than reversal. It substantially improves over manual prompting and dense activation steering, which reduce error only partially and inconsistently across models.

On hallucination and myopic reward, both methods consistently improve over baselines. SAEGA gives the strongest myopic-reward reduction for Llama 3.1. RESGA performs competitively on hallucination.

**Claims and evidence**

- **Both methods neutralize sycophancy at rates statistically indistinguishable from \(50\%\).** Table 1, Sycophancy columns: RESGA **49.86, 50.63, 62.65**; SAEGA **49.84, 49.95, 70.70**, in Llama–Qwen–Gemma order.
- **Manual prompting and dense steering give partial, inconsistent sycophancy reductions.** Table 1, Sycophancy columns: Zero-Shot **72.48, 86.00, 82.00**; Standard Prompt **70.50, 72.00, 80.00**; Dense Steering Vector **54.50, 79.00, 80.00**.
- **Both methods consistently improve hallucination and myopic reward over baselines.** Table 1: RESGA Hallucination **45.12, 41.46, 51.22**, Myopic Reward **38.50, 38.50, 34.50**; SAEGA Hallucination **46.95, 40.85, 49.14**, Myopic Reward **31.50, 40.50, 38.50**.
- **SAEGA gives the strongest Llama myopic-reward reduction.** Table 1, Myopic Reward–Llama: SAEGA **31.50**; RESGA **38.50**; ProTeGi **42.50**; Dense Steering Vector **47.50**; GCG **48.50**; Standard Prompt **52.00**; Zero-Shot **54.00**.
- **RESGA is competitive on hallucination.** Table 1, RESGA Hallucination cells: **45.12, 41.46, 51.22**.

### 4.3 Mechanistic Analysis

*This subsection attributes SAEGA’s control to concentrated neutrality, preserved sparsity, selective feature suppression, and distinct steering geometry.*

We analyze how RESGA and SAEGA mitigate personas and why SAEGA differs qualitatively from dense steering. The analysis reveals precise, sparse, semantically grounded control rather than global representational interference.

**Neutralization vs. Distributional Shifting.** We first examine activation projections onto the sycophancy axis, the direction representing sycophancy. Figure 3 shows a broad baseline distribution reflecting systematic agreement bias. Dense steering and RESGA move the mean toward neutrality but retain substantial variance. This indicates coarse counterbalancing: shifting the distribution rather than tightly controlling individual instances.

SAEGA instead produces a sharp peak near zero. The variance collapse shows instance-wise neutrality rather than a mean shift, yielding behavior statistically indistinguishable from random choice.

Figure 3: Distribution of projections onto the sycophancy axis. Dense steering and RESGA shift the mean but retain variance. SAEGA collapses variance around neutrality (0.0), indicating precise control.

**Preservation of the Natural Activation Manifold.** We assess internal structure using an SAE trained on residual activations. Figure 4 reports the \(L_0\) norm, or active-feature count. The baseline activates approximately 50 features per token.

Dense-vector injection increases this beyond 150 features, indicating departure from the natural activation manifold. RESGA partly mitigates the effect but still increases sparsity. SAEGA maintains baseline-comparable sparsity of 50–60 features. This shows preservation of the model’s internal sparse topology, or natural sparse activation structure.

Figure 4: Sparsity preservation (\(L_{0}\) norm). Dense steering forces an unnatural explosion in active features. SAEGA preserves sparsity close to the baseline model.

**Feature-Level Control.** Figure 5 examines activation changes in SAE features most correlated with sycophancy. Dense steering and RESGA have noisy, inconsistent effects and often activate unrelated features. SAEGA consistently suppresses causally relevant sycophancy features while leaving unrelated features largely unchanged.

Figure 5: Impact on SAE features correlated with sycophancy. SAEGA selectively suppresses causally relevant features while avoiding spurious activation.

**Geometric Structure of Steering Trajectories.** We visualize steering trajectories—changes in residual representations—using PCA, the visualization method used here. Figure 6 shows dense steering causing a large linear displacement far from the natural data manifold. RESGA moves the mean but produces a diffuse, high-variance cluster.

SAEGA converges to a tight, stable region, often orthogonal to the dense-steering direction. This indicates a distinct subspace corresponding to honest behavior while remaining on-manifold.

Figure 6: PCA of steering trajectories. Dense steering exits the natural manifold. RESGA shifts the mean with high variance. SAEGA converges to a tight, stable subspace.

Table 2: Examples of discovered persona-steering prompts with random initialization.

The following strings are the reported discovered prompts. Escape sequences and special-token strings are preserved literally.

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

“Seed” identifies the starting phrase; “Optimized” identifies the EPO-discovered prompt. Score is the reported error.

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

WikiText2 is the language-modeling evaluation dataset. “RESGA \(\&\) SAEGA” is the reported prompt-intervention column.

| Model | Baseline | RESGA \(\&\) SAEGA |
|---|---:|---:|
| Llama | 12.56 | 12.07 |
| Qwen | 10.88 | 12.48 |
| Gemma | 30.51 | 29.19 |

**Claims and evidence**

- **Dense steering and RESGA shift the sycophancy mean while retaining variance; SAEGA enforces instance-wise neutrality.** Figure 3 and its caption: SAEGA concentrates near **0.0**.
- **SAEGA preserves natural activation sparsity, unlike dense steering.** Figure 4 and accompanying text: baseline approximately **50** active features per token; dense steering **beyond 150**; SAEGA **50–60**. RESGA partly mitigates the increase.
- **SAEGA selectively suppresses relevant features while avoiding unrelated activation.** Figure 5 and its caption.
- **SAEGA reaches a tight, stable, often orthogonal subspace while remaining on-manifold.** Figure 6 and accompanying trajectory analysis.
- **Random initialization discovers the displayed steering prompts.** Table 2, all Sycophancy, Hallucination, and Myopic Reward model rows.
- **Seed optimization substantially lowers error while retaining closer natural-language structure.** Table 3, paired Seed → Optimized rows: **69.16% → 56.25%**, **72.30% → 54.22%**, **71.15% → 51.72%**, **72.06% → 59.35%**, **52.43% → 46.34%**, **51.20% → 46.95%**.
- **Prompt prepending causes negligible general language-modeling disruption.** Table 4, Baseline → RESGA \(\&\) SAEGA: Llama **12.56 → 12.07**; Qwen **10.88 → 12.48**; Gemma **30.51 → 29.19**.

### 4.4 Qualitative Analysis of Mechanisms

*This subsection distinguishes label-token priming from semantic vocabulary shifts and relates prompt examples to interpretability and perplexity.*

Table 2 shows automatically discovered prompts. Many appear superficially incoherent, but token-probability shifts reveal distinct optimizer strategies. Table 3 shows that seed initialization produces more interpretable prompts that preserve natural-language structure while achieving competitive steering. Table 4 confirms negligible degradation in general language modeling, measured by WikiText2 perplexity.

##### Token Priming via Induction Heads.

*Some prompts bias answer labels through in-context copying rather than necessarily changing semantic reasoning.*

In binary classification, undesirable behavior may be systematically associated with a label. For example, the sycophantic answer in Sycophancy is often labeled “(A).” Unconstrained optimization can then converge on **Token Priming**: injecting an alternative answer-label token into the context. Some prompts, particularly RESGA prompts, explicitly include the alternative label, as in `"...apol al al b"`.

We attribute this to **Induction Heads** (Olsson et al., 2022), the mechanism cited here for in-context copying. Injecting target token “B” increases the probability that the model generates it. This acts as a “soft” few-shot example: it corrects dataset skew in output probabilities without necessarily changing underlying semantic reasoning. Dataset skew here means systematic alignment between an undesirable behavior and a label.

**Claims and evidence**

- **Some prompts, particularly RESGA prompts, exploit label-token priming.** This subsection reports the example **`"...apol al al b"`**, the sycophantic label **“(A)”**, and target token **“B”**.
- **The mechanism is attributed to in-context copying rather than necessarily changed reasoning.** The subsection’s Induction Heads analysis cites Olsson et al. (2022).

##### Semantic Steering via Vocabulary Shift.

*Other prompts change semantic token preferences, favoring critical evaluation or long-term planning.*

SAEGA prompts frequently use **Semantic Steering**: fragments that prime reasoning modes rather than simply force tokens. We analyze the induced output-logit change, where logits are the model outputs compared before and after steering:

\[
\Delta\text{Logits}
=
\text{Logits}_{\text{steered}}
-
\text{Logits}_{\text{baseline}}.
\]

Here, \(\Delta\text{Logits}\) is the steered-minus-baseline vocabulary shift.

- **Sycophancy (Polite Disagreement).** SAEGA systematically suppresses agreement markers “Yes,” “True,” and “Agree.” It significantly upweights semantic pivots “However” and “But.” This suggests a “critical evaluation” mode that creates room for nuance before factual correction, rather than merely making the model rude or contradictory.
- **Myopic Reward (Temporal Priming).** RESGA and SAEGA upweight long-term-planning tokens, including “Future,” “Long,” and “Wait.” This vocabulary shift is described as Temporal Priming.

**Claims and evidence**

- **SAEGA primes polite disagreement or critical evaluation.** The reported vocabulary analysis suppresses **“Yes,” “True,” “Agree”** and upweights **“However,” “But.”**
- **RESGA and SAEGA prime long-term planning.** The reported Myopic Reward analysis upweights **“Future,” “Long,” “Wait.”**
- **Seed initialization preserves more natural-language structure with competitive steering.** Table 3, Seed → Optimized scores: **69.16% → 56.25%**, **72.30% → 54.22%**, **71.15% → 51.72%**, **72.06% → 59.35%**, **52.43% → 46.34%**, **51.20% → 46.95%**.
- **General language-modeling degradation is negligible.** Table 4: Llama **12.56 → 12.07**, Qwen **10.88 → 12.48**, Gemma **30.51 → 29.19**.

### 4.5 Ablation Studies

*This subsection tests the steering direction, gradient-guided mutation, and representation-based objective.*

Ablation studies replace framework components to assess their contributions. Table 5 tests three core choices.

**Persona Steering Vector.** Replacing the persona direction with a random direction makes performance collapse to near or above zero-shot levels across tasks and models. This confirms the need for a mechanistically grounded direction.

**Gradient-Guided Mutation.** Replacing gradient-guided token selection with random token mutation consistently worsens performance. Gradient feedback is therefore critical for finding persona-suppressing prompts.

**Representation-Based Objective.** Replacing representation projection with direct task-loss optimization produces unstable results. It underperforms RESGA and SAEGA in most settings. Grounding the objective in internal representations therefore supplies a more reliable optimization signal than task loss alone.

Together, the results confirm meaningful contributions from all three components.

Table 5: Ablation study results. We ablate three core components of our framework: the persona steering vector (Random Direction), gradient-guided token mutation (Random Mutation), and the representation-based objective (Task Loss as Objective).

| Method | Sycophancy ↓ Llama | Sycophancy ↓ Qwen | Sycophancy ↓ Gemma | Hallucination ↓ Llama | Hallucination ↓ Qwen | Hallucination ↓ Gemma | Myopic Reward ↓ Llama | Myopic Reward ↓ Qwen | Myopic Reward ↓ Gemma |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Random Direction | 74.13 | 84.56 | 81.47 | 68.29 | 54.26 | 84.70 | 50.50 | 48.50 | 61.00 |
| Random Mutation | 68.00 | 71.66 | 76.04 | 50.40 | 45.12 | 54.26 | 52.00 | 46.50 | 55.00 |
| Task Loss as Objective | 55.83 | 73.50 | 58.70 | 46.95 | 45.12 | 54.27 | 48.50 | 48.00 | 51.50 |
| RESGA | 49.86 | 50.63 | 62.65 | 45.12 | 41.46 | 51.22 | 38.50 | 38.50 | 34.50 |
| SAEGA | 49.84 | 49.95 | 70.70 | 46.95 | 40.85 | 49.14 | 31.50 | 40.50 | 38.50 |

**Claims and evidence**

Values below follow Table 5’s order: Sycophancy Llama–Qwen–Gemma; Hallucination Llama–Qwen–Gemma; Myopic Reward Llama–Qwen–Gemma.

- **A meaningful persona direction is essential.** Table 5, Random Direction: **74.13, 84.56, 81.47; 68.29, 54.26, 84.70; 50.50, 48.50, 61.00**. Table 1, Zero-Shot: **72.48, 86.00, 82.00; 51.45, 45.00, 57.97; 54.00, 55.00, 58.00**.
- **Random mutation consistently degrades performance.** Table 5, Random Mutation: **68.00, 71.66, 76.04; 50.40, 45.12, 54.26; 52.00, 46.50, 55.00**. RESGA: **49.86, 50.63, 62.65; 45.12, 41.46, 51.22; 38.50, 38.50, 34.50**. SAEGA: **49.84, 49.95, 70.70; 46.95, 40.85, 49.14; 31.50, 40.50, 38.50**.
- **Direct task loss is unstable and worse in most settings.** Table 5, Task Loss as Objective: **55.83, 73.50, 58.70; 46.95, 45.12, 54.27; 48.50, 48.00, 51.50**, compared with the RESGA and SAEGA rows above.
- **All three components contribute to effectiveness.** Table 5, the three ablation rows compared with RESGA and SAEGA.

## 5 Conclusion

*This section summarizes the framework and findings and proposes lower-data, cross-model, and multi-persona extensions.*

We present a framework connecting mechanistic interpretability with automatic persona-steering prompt discovery. RESGA and SAEGA construct steering vectors from labeled examples and optimize prompts through evolutionary gradient ascent to achieve interpretable control over LLM behavior.

Across three persona-mitigation tasks, discovered prompts significantly improve sycophancy and myopic reward and consistently improve hallucination. They substantially outperform manual prompting and dense steering. Mechanistic analysis shows that SAEGA neutralizes rather than shifts behavior, preserves natural activation manifolds, and controls interpretable features.

Future work could use semi-supervised approaches to reduce labeled-data requirements, investigate cross-model transferability of discovered prompts, and extend to multi-persona steering.

**Claims and evidence**

- **The methods improve persona steering and outperform manual prompting and dense steering.** Table 1, Avg. column: RESGA **45.83**, SAEGA **46.44**, Standard Prompt **55.45**, Dense Steering Vector **56.10**. The task-specific RESGA and SAEGA rows are reproduced in full in Table 1.
- **SAEGA neutralizes behavior, preserves natural activations, and controls relevant features.** Figures 3–6: neutrality near **0.0**; approximately **50** baseline active features, **beyond 150** for dense steering, and **50–60** for SAEGA; selective suppression in Figure 5 and stable trajectory geometry in Figure 6.

## 6 Limitations

*This section identifies labeled-data needs, model-specific steering, and the conflict between effectiveness and semantic coherence.*

- **Labeled Data Requirement.** RESGA and SAEGA require labeled examples exhibiting and lacking the target persona to find steering prompts.
- **Cross-Model Transferability.** SAE latents and steering vectors are model-specific and require retraining for each target model.
- **Fluency-Effectiveness Tradeoff.** Most effective discovered prompts lack semantic coherence. Strong linguistic priors—constraints proposed to favor natural language—could help produce prompts that are both effective and human-like.

## Broader Impact Statement

*This section describes the safety aim, possible misuse for harmful-output discovery, and the need for access controls.*

The work aims to improve LLM safety by suppressing undesired personas. The same pipeline could, in principle, target safety-critical SAE latents and automate jailbreak discovery or harmful-output elicitation. Deployment should include appropriate access controls.

## Acknowledgment

*This section acknowledges AI assistance with manuscript writing and code generation.*

We acknowledge the use of Claude (Anthropic) for manuscript writing assistance and AI-assisted code generation throughout this work.
