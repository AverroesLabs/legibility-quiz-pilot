# Revisiting CroPA: A Reproducibility Study and Enhancements for Cross-Prompt Adversarial Transferability in Vision-Language Models

| Author | Email | Affiliation | Author notation |
|---|---|---|---|
| Atharv Mittal | atharv_m@mfs.iitr.ac.in | Mehta Family School of Data Science and Artificial Intelligence; Indian Institute of Technology, Roorkee | ††thanks: All authors contributed equally |
| Agam Pandey | agam_p@ce.iitr.ac.in | Department of Civil Engineering; Indian Institute of Technology, Roorkee | 11footnotemark: 1 |
| Amritanshu Tiwari | amritanshu_t@mfs.iitr.ac.in | Mehta Family School of Data Science and Artificial Intelligence; Indian Institute of Technology, Roorkee | 11footnotemark: 1 |
| Sukrit Jindal | sukrit_j@mfs.iitr.ac.in | Mehta Family School of Data Science and Artificial Intelligence; Indian Institute of Technology, Roorkee | 11footnotemark: 1 |
| Swadesh Swain | swadesh_s@ece.iitr.ac.in | Department of Electronics and Communication; Indian Institute of Technology, Roorkee | 11footnotemark: 1 |

###### Abstract

The reproducibility study of “An Image is Worth 1000 Lies: Adversarial Transferability Across Prompts on Vision-Language Models” validates Q7X and its superiority to existing baselines in N6Z; semantic meaning and target-word frequency do not determine that transferability. Initialization, common perturbations across images, and attention-directed loss extend the replication. Evaluation encompasses Flamingo, BLIP-2, InstructBLIP, and extended LLaVA experiments; the original results are validated and the enhancements consistently increase adversarial effectiveness. The resulting framework generates more robust transferable adversarial examples and bears on real-world VLM security.

Code: https://github.com/Swadesh06/Revisting_CroPA

Reviewed on OpenReview: https://openreview.net/forum?id=5L90cl0xtf

## 1 Introduction

Subtle image perturbations induce incorrect or harmful VLM outputs; textual input adds attack vectors beyond those of traditional vision models. Large VLMs transformed computer vision through image G9M, D5R, and visual question answering, enabling deeper investigation of visual–linguistic interaction, while vulnerabilities were demonstrated by Zhao et al. (2023); Qi et al. (2023); Zhang et al. (2022a); Carlini et al. (2024).

Architectural survival of adversarial examples (F8V = Cross-Model Transferability), their survival over multiple images (K3R = Cross-Image Transferability), and effectiveness over varying textual prompts (N6Z = Cross-Prompt Transferability) distinguish the transfer settings, with the first commonly called transferability and the second admitting shared perturbations (Z4P = Universal Adversarial Perturbations, UAPs), rather than requiring a separately optimized perturbation for each image.

Luo et al. (2024) introduced N6Z and competing learnable prompts; their image attacks retain effectiveness regardless of textual input across image G9M, D5R, and VQA, unlike earlier independent treatment of visual and textual adversarial perturbations. Zhao et al. (2023) analyzed BLIP and BLIP-2 using query- and transfer-based black-box attacks; Schlarmann & Hein (2023) examined targeted and untargeted white-box attacks. Qi et al. (2023) showed adversarial images eliciting harmful instruction execution, and Tu et al. (2023) systematically evaluated attacks on both modalities.

The replication tests superiority across target texts, rapid saturation with additional prompts, robustness under added in-context examples, and convergence over attack iterations, including taskwise analysis beyond the original aggregate analysis. Extensions cover initialization improving both H2Q and transferability, shared perturbations on the Q7X backbone with transfer-promoting image augmentation, and target value vectors: attention components govern patch interaction, and targeted manipulation is proposed to improve generalization and H2Q. Analyses investigate the mechanisms of the enhancements and the nature of VLM adversarial vulnerabilities.

## 2 Related Works

Szegedy et al. (2013) and Goodfellow et al. (2014) introduced adversarial attacks on deep neural networks; their application to VLMs adds challenges and opportunities.

### 2.1 Adversarial Transferability

Classification-directed examples also affect object detection and other visual tasks, establishing cross-task transferability alongside F8V and K3R: Naseer et al. (2019b); Naseer et al. (2019a); Lu et al. (2020); Salzmann et al. (2021). This multidimensional behavior raises robustness questions for modern machine-learning systems.

Liu et al. (2017) and Tramèr et al. (2017) established that attacks crafted on one model often deceive other models, including different architectures. Moosavi-Dezfooli et al. (2017); Mopuri et al. (2017) established image-generalizing perturbations and Z4P.

### 2.2 Adversarial Attacks on Vision-Language Models

Early task-specific D5R attacks manipulated outputs in Xu et al. (2019); Zhang et al. (2020); Aafaq et al. (2021); Chen et al. (2017). Targeted VQA attacks misled attention mechanisms in Xu et al. (2018); Kaushik et al. (2021); Kovatchev et al. (2022); Li et al. (2021); Sheng et al. (2021); Zhang et al. (2022b).

These attacks principally addressed lightweight CNN–RNN systems without in-context learning, restricting applicability to contemporary VLMs. The modern-large-VLM black-box study of BLIP and BLIP-2 by Zhao et al. (2023) emphasized F8V rather than N6Z.

### 2.3 Cross-Prompt Adversarial Transferability

No credible benchmarks or baselines have yet been established for this relatively new concept; Yang et al. (2024) nevertheless supplies an additional experimental comparator (E3W = Context Injection Attack, CIA, also called Contextual Injection Attack), using gradient-based perturbations to inject target tokens into visual and textual contexts, rather than the competing-prompt optimization pioneered by Q7X in Luo et al. (2024).

Prompt-driven adaptation permits large VLMs to switch tasks dynamically, unlike fixed-task traditional vision models: Li et al. (2023b); Awadalla et al. (2023); Li et al. (2023b); Zhu et al. (2023); Liu et al. (2023). Adversarial validity across changing prompts constitutes an additional attack surface. Competing rather than independently manipulated modalities establish a distinct paradigm for evaluating VLM robustness.

### 2.4 Enhancing Adversarial Transferability

Gradient misalignment limits transferability and has motivated optimization strategies in Li et al. (2023a); Wang et al. (2023). Input diversity in Xie et al. (2019) and augmentation in Wang et al. (2021) also improve attack robustness. These findings inform the multidimensional Q7X enhancements.

## 3 Scope of reproducibility

Faithful reproduction of Luo et al. (2024) requires their experimental procedures and tests the following named claims:

- Claim 1: Q7X achieves N6Z across target texts.
- Claim 2: Q7X has the best aggregate performance across prompt counts.
- Claim 3: Q7X converges toward higher H2Q with increasing iterations.
- Claim 4: Q7X outperforms baselines in few-shot settings.

The extended scope includes F8V, K3R, and comparison to other N6Z methods: images generated through Q7X are tested for consistent deception across diverse VLMs, input prompts, and model parameters. The scope concerns broader robustness and versatility of adversarial attacks, beyond validation of the original prompt-transfer improvement.

## 4 Methodology

### 4.1 Problem Formulation

Exact target-text generation defines targeted success, whereas any departure from the clean-image prediction defines untargeted success; the corresponding effectiveness quantity (H2Q = Attack Success Rate, ASR) is applied to the reproduced targeted and non-targeted scenarios, with a language-modeling objective over prompt instances rather than a vision-only objective.

\[
\min_{\delta_v}\sum_{i=1}^{k}
L\bigl(f(x_v+\delta_v,x_t^i),T\bigr)
\tag{1}
\]

\[
\max_{\delta_v}\sum_{i=1}^{k}
L\bigl(f(x_v+\delta_v,x_t^i),f(x_v,x_t^i)\bigr)
\tag{2}
\]

Here \(f\) processes visual \(x_v\) and textual \(x_t\) inputs, \(L\) is language-modeling loss, and \(T\) is predetermined text generated independently of the input prompt in the targeted scenario. The untargeted scenario maximizes clean/adversarial output discrepancy. The human-imperceptibility bound is \(\|\delta_v\|_p\leq\epsilon_v\), adapting established adversarial practice to multimodal real-world applicability, Carlini et al. (2024). The formulation extends the neural-network vulnerability setting of Goodfellow et al. (2014) to prompt variation, Luo et al. (2024).

### 4.2 Baseline Approaches

The reference attacks introduced by Luo et al. (2024), absent established N6Z baselines, optimize a perturbation with one prompt (R4K = Single-P) or a collection of prompts (V9D = Multi-P), and their untargeted forms invert the targeted objectives to maximize clean/perturbed output discrepancy rather than target-text likelihood.

\[
\min_{\delta_v}L\bigl(f(x_v+\delta_v,x_t),T\bigr)
\tag{3}
\]

\[
\min_{\delta_v}\sum_{i=1}^{k}
L\bigl(f(x_v+\delta_v,x_t^i),T\bigr)
\tag{4}
\]

The collection is \(X_t=\{x_t^1,x_t^2,\ldots,x_t^k\}\); the one-prompt case is the simplest baseline.

**Figure 1.** Luo et al. (2024) framework: image \(\delta_v\) and prompt \(\delta_t\) perturbations act antagonistically, minimizing and maximizing language-modeling loss respectively. Solid arrows denote the forward pass through vision encoder and language model; dashed arrows denote backpropagation, with configurable perturbation-update frequencies. Asterisks mark fixed model parameters; \(\oplus\) denotes elementwise addition.

### 4.3 Cross Prompt Attack

Evaluation discards prompt perturbations and applies only the optimized image perturbation: effectiveness must reside in the latter's inherent transferability, not modified prompts. The competing-prompt method of Luo et al. (2024) (Q7X = Cross-Prompt Attack, CroPA) jointly optimizes \(\delta_v\) and \(\delta_t\), with antagonistic rather than collaborative objectives intended to force stronger image transferability across textual prompts.

\[
\min_{\delta_v}\max_{\delta_t}
L\bigl(f(x_v+\delta_v,x_t+\delta_t),T\bigr)
\tag{5}
\]

The implementation uses iterative image-gradient descent and prompt-gradient ascent (S3D = Projected Gradient Descent, PGD), following Madry et al. (2017), with image updates occurring \(N\) times per prompt update and configurable frequency rather than simultaneous equal-frequency updates of the two modalities.

**Algorithm 1. Q7X**

**Input:** model \(f\); target text \(T\); vision input \(x_v\); prompt set \(X_t\); perturbation size \(\epsilon\); step sizes \(\alpha_1,\alpha_2\); iteration count \(K\); adversarial-prompt update interval \(N\).  
**Output:** adversarial example \(x'_v\).

```text
1:  Initialize x'_v = x_v
2:  for step = 1 to K do
3:      Uniformally sample prompt x_t^i from X_t
4:      if x_t^i is not initialised then
5:          Initialise x'_t^i = x_t^i
6:      end if
7:      Compute adversarial-image gradient:
            g_v = ∇_{x_v} L(f(x'_v, x_t^i), T)
8:      Gradient-descent update:
            x'_v = x'_v - α_1 · sign(g_v)
9:      if mod(step, N) = 0 then
10:         Compute adversarial-prompt gradient:
                g_t = ∇_{x_t} L(f(x'_v, x_t^i), T)
11:         Gradient-ascent update:
                x'_t^i = x'_t^i + α_2 · sign(g_t)
12:     end if
13:     Project within the ε-ball of x_v:
            x'_v = Clip_{x_v, ε}(x'_v)
14: end for
15: return x'_v
```

## 5 Additional Investigative Experiments

Q7X performs poorly on G9M and D5R and has limited F8V. Its S3D optimization effectively raises single-model H2Q, but dependence on that model's loss feedback constrains transferability, a recognized gradient-attack limitation: Qin et al. (2024); Li et al. (2024); Wang et al. (2023); Li et al. (2023a). The original authors recognize it, and Table 9 in Appendix A.1 verifies it.

The extensions address task inconsistency and transfer dimensions. K3R could remove separate per-image training; F8V could provide still greater efficiency. Transfer exploration is not a core proposal of Luo et al. (2024). The resource cost and quantitative characterizations are retained separately from the associated comparisons.

| Quantity or reported characterization | Value |
|---|---|
| Q7X time for training perturbations on 50 images | over 6 hours |
| GPU for that run | 48GB L40S |
| VRAM utilization | 95% |
| Added initialization time per image on a single GPU | 20-25 seconds |
| Initialization aggregate accuracy gains | 6-15% |
| Initialization targeted H2Q gains, reported upper extent | up to 15% |
| P2F aggregate targeted H2Q | 96.85% |
| P2F increase over base Q7X, results characterization | 12.5% |
| P2F increase over base Q7X, discussion characterization | 13.5% |
| P2F A7L increase | 13% |
| P2F D5R increase | 28% |
| Value-guided F8V G9M improvement | 14% |
| Value-guided F8V D5R improvement | 21% |
| J6T aggregate K3R increase | approximately 27 percent |
| U8G net increase over base K3R Q7X | around 12 percent |

### 5.1 Noise Initialization via Vision Encoding Optimization

The diffusion-anchored method (W5J = Noise Initialization via Vision Encoding Optimization) replaces target-agnostic random noise by target-image feature alignment, retaining imperceptibility while supplying semantically meaningful adversarial directions at the outset; conventional initialization neither guarantees such directions nor avoids reduced H2Q and N6Z.

Q7X shares random initialization with many gradient attacks. Target-semantic ignorance is associated with instability and slower convergence, particularly in VLMs requiring semantic prompt alignment. The bi-level framework is empirically initialization-sensitive (Section 6.5.1; Figure 7; Table 3). The proposed hypothesis is that semantically informed rather than purely random perturbations improve attack efficacy within the same imperceptibility requirement.

A state-of-the-art diffusion model synthesizes a target image, its encoder representation anchors initial noise to the objective, and an encoder-output error loss enforces semantic similarity while retaining adversarial effectiveness; the error measure (L6V = Mean Squared Error, MSE) supplies the optimization criterion rather than a purely random starting direction for subsequent attack updates.

The method is presented as improving adversarial robustness, strengthening perturbations and N6Z, and reducing steps needed for convergence.

#### 5.1.1 Noise Initialization

Preliminary BLIP-2 and InstructBLIP experiments favor semantic over random noise. Diffusion-based anchoring improves encoder-space alignment while preserving adversarial-example visual fidelity and strict perturbation budgets; its single-image computation increment is limited to the separately stated single-GPU overhead.

#### 5.1.2 Diffusion-Based Target Synthesis

\[
\mathbf{x}_{\mathrm{target}}
=\mathcal{D}(T,\mathbf{z};\theta_{\mathrm{SDXL}})
\tag{6}
\]

Stable Diffusion XL, SDXL, supplies \(\mathcal D\); \(\mathbf z\) is Gaussian-sampled and \(\theta_{\mathrm{SDXL}}\) consists of pretrained diffusion weights. Generated \(\mathbf{x}_{\mathrm{target}}\) embodies the semantic attributes of prompt \(T\) and supplies the initialization reference.

#### 5.1.3 Vision-Encoder Anchored Perturbation

\[
\boldsymbol{\delta}_{\mathrm{init}}
=
\underset{\|\boldsymbol{\delta}\|_\infty\leq\epsilon}
{\arg\min}\;
\left\|
f_v(\mathbf{x}+\boldsymbol{\delta})
-f_v(\mathbf{x}_{\mathrm{target}})
\right\|_2^2
\tag{7}
\]

The VLM vision component is \(f_v\); \(\epsilon\) is the maximum allowable, imperceptible \(\ell_\infty\) perturbation. Mimicking the target image's encoder-space representation places the adversarial example in a semantically meaningful neighborhood of the target prompt representation. Algorithm 2 in Appendix B formalizes the method.

#### 5.1.4 Adversarial Optimization via S3D

\[
\mathbf{x}_{\mathrm{adv}}^{(t+1)}
=
\Pi_{\mathcal{B}_{\epsilon}(\mathbf{x})}
\left[
\mathbf{x}_{\mathrm{adv}}^{(t)}
-
\alpha\nabla_{\mathbf{x}_{\mathrm{adv}}^{(t)}}
\mathcal{L}_{\mathrm{L6V}}
\right]
\tag{8}
\]

\[
\mathcal{L}_{\mathrm{L6V}}
=
\left\|
f_v(\mathbf{x}_{\mathrm{adv}})
-f_v(\mathbf{x}_{\mathrm{target}})
\right\|_2^2
\tag{9}
\]

After initialization, these refinement updates incrementally align the example with the target embedding. At iteration \(t\), \(\alpha\) is step size and \(\Pi_{\mathcal B_\epsilon(\mathbf x)}\) projects into the admissible \(\ell_\infty\) ball around original \(\mathbf x\).

### 5.2 Guiding perturbations via Target Value Vectors and evaluating F8V

Value vectors in middle-to-late vision layers are the intended vulnerable components. Q7X lacks architectural sensitivity to the encoder's representational backbone and treats vision and language as uniformly vulnerable, overlooking modality-specific weaknesses despite the encoder's dominant semantic-grounding role. Generic S3D output manipulation is less structured and less targeted than the proposed alternative; the intended improvements concern attack strength, transferability, interpretability, and computational feasibility.

The prior component-selection method (X9C = Doubly-Universal Adversarial Perturbation, Doubly-UAP, also D-UAP), proposed by Kim et al. (2024), identifies which attention components most influence VLM performance when constructing Z4P, and its component loss is adapted here by minimizing rather than maximizing losses between value vectors of \(x_i\) and \(T_i\), retaining a reference associated with the target text.

Cui et al. (2023); Wang et al. (2024) show that perturbing vision-output embeddings impairs LVLM visual perception and misleads language responses. LVLMs commonly use similar image encoders to produce visual tokens containing semantic and structural information, the interface through which language decoders access and reason about images. Targeting encoder internals is proposed to exploit that shared interface.

Two components have fundamental roles: attention weights control patch-to-patch focus, relevance, and interaction; value vectors contain each patch's actual information. Interference with the former is hypothesized to damage relationship establishment, and perturbation of the latter to damage essential content and interpretation. Prior analysis identifies middle-to-late-layer value vectors as most vulnerable.

\[
\delta^*
=
\arg\max_\delta
\frac{1}{|L|}
\sum_{l\in L}
\operatorname{Loss}\bigl(V_l(x),V_l(x+\delta)\bigr)
\tag{10}
\]

In the standard X9C objective, \(V_l(x)\) denotes input \(x\)'s value vectors at layer \(l\), and \(\operatorname{Loss}(\cdot)\) is the loss applied to target vectors.

The proposed combination retains Q7X's opposing image/text objectives and adds attention-value disruption, under the same hyperparameter constraints. The hypothesis is enhanced H2Q through value-vector perturbation relative to a reference image corresponding to target text. Freezing the text encoder is intended to isolate degraded visual processing rather than text artifacts. The modifications seek encoder-information exploitation, target-text semantic alignment, and better cross-model adversarial robustness.

#### 5.2.1 Our Modified Approach

\[
L_{\mathrm{X9C}}
=
\sum_{i=1}^{N}
\left(
1-
\frac{V_i(x+\delta)\cdot V_t}
{\|V_i(x+\delta)\|_2\|V_t\|_2}
\right)
\tag{11}
\]

\[
L_{\mathrm{P2F}}
=
L_{\mathrm{Q7X}}(x_v+\delta_v,x_t+\delta_t,T)
-\lambda L_{\mathrm{X9C}}(\delta_v)
\tag{12}
\]

The combined method (P2F = CroPA+D-UAP, also re-CroPA) extracts the input and target-associated reference image's layer-\(l\) value vectors, computes their cosine-similarity loss, and jointly minimizes its negative with the Q7X loss under appropriate scaling, rather than solely maximizing departure from the original representation.

Here \(V_i(x+\delta)\) is the perturbed input's value vector at attention head \(i\), \(V_t\) is the reference image's target value vector, \(N\) counts heads, and \(\lambda\) controls relative value-loss importance. The loss is described as encouraging closeness to the target representation; combined gradients update \(\delta\), preserving adversarial character, N6Z, and guided semantic influence. Controlled, interpretable manipulation is intended for robustness and security analysis. Algorithm 3 in Appendix B gives the steps.

The encoder dependence of X9C supplies a limitation hypothesis: transfer gains may be restricted to VLMs with the same or similar vision-encoder architecture as the model used for Q7X optimization.

**Figure 2.** X9C architecture diagram.

### 5.3 Transferability across images via Image Augmentation

Per-image iterative gradients overfit image-specific features, giving poor K3R and high computational expense; Q7X likewise learns each image's perturbation across prompts. Luo et al. (2024) describe their method as orthogonal to K3R and suggest computing perturbations across images, following Moosavi-Dezfooli et al. (2017). The investigation tests that claim and an additional Z4P-literature-based approach.

One investigated augmentation (U8G = CutMix) uses regional replacement rather than pixel blending, synthesizing images from rectangular patches cut from another image and labels mixed in proportion to patch area; its partial-view learning promotes spatially invariant features and robustness to occlusion and adversarial attacks, Yun et al. (2019).

Fang et al. (2024) introduces Z4P for Vision-Language Pretraining, VLP, models to reduce gradient-attack overfitting and enable K3R. Zhang et al. (2024) improves global and local optimization across VLP models and tasks through a framework (M7A = Effective and Transferable Universal Adversarial Attack, ETU), within which self- and cross-mixing augmentation (J6T = SCMix) increases diversity while preserving semantics and spatial/scale invariance.

J6T is selected for integration into perturbation generation on the basis of empirical results: for the same model, J6T alone, without their additional local utility reinforcement, yields superior K3R.

#### 5.3.1 J6T

\[
I'_1=\eta X_1+(1-\eta)X_2,
\qquad
X_i=\operatorname{Resize}(\operatorname{RandomCrop}(I_1)),
\qquad
\eta=0.5
\tag{13}
\]

\[
I_3=\beta_1 I'_1+\beta_2 I_2,
\qquad
\beta_1\gg\beta_2\in[0,1)
\tag{14}
\]

The first stage self-mixes random crops/resizes of base image \(I_1\); the second cross-mixes its result with secondary \(I_2\). Original J6T samples \(\eta\sim\operatorname{Beta}(\alpha,\alpha)\), \(\alpha>0\); this implementation fixes \(\eta=0.5\), equivalent to \(\alpha=\infty\), justified in Appendix D.3.

The weighting preserves dominant \(I_1\) semantics and compatibility of \(I_3\) with text \(T_1\), despite additional \(I_2\) features, enriching cross-modal interaction. Intra-image variation and inter-image blending regularize against Q7X overfitting and encourage perturbations directed at invariant features. The proposed common perturbation may then apply to unseen images without image-specific optimization.

Q7X returns only a perturbed image, equivalently its perturbation, at inference. Effective K3R would make that output universal in both the original N6Z sense and the additional investigated image sense.

**Figure 3.** Self-mixing \(I_1\) generates \(X_1,X_2\), combined into \(I'_1\); external \(I_2\) is mixed with \(I'_1\) to produce \(I_3\).

#### 5.3.2 U8G

\[
\tilde{x}=M\odot x_A+(1-M)\odot x_B
\tag{15}
\]

\(M\in\{0,1\}^{W\times H}\) is a binary mask for a randomly located rectangle. With paired images \(x_A,x_B\), discrete replacement retains partial views. J6T softly blends features within and across images; U8G uses a hard mask. In adversarial training, persistent features across regional combinations are expected to direct perturbations toward spatially invariant, higher-level semantics rather than localized artifacts, potentially improving K3R.

### 5.4 Attack Setup and Threat Models for Transferability Experiments

Knowledge, perturbation capability, and evaluation differ between model transfer and image transfer.

#### 5.4.1 Using X9C to explore F8V for Q7X

The attacker seeks one image perturbation eliciting a specified target text from multiple VLMs independently of prompt: accessible-surrogate optimization followed by deployment against unknown models. White-box surrogate access includes gradients and internal representations; the surrogate is BLIP-2 or Flamingo. Only a target's general architecture family, such as CLIP-based, is known; target parameters are unavailable.

The imperceptibility budget is \(\epsilon=16/255\) in \(L_\infty\). Intra-family testing optimizes BLIP-2 with OPT-2.7b and evaluates InstructBLIP with Vicuna-7b: both have similar CLIP ViT-L/14 encoders but different language models and fine-tuning objectives. Cross-architecture testing runs BLIP-2 to Flamingo and the reverse, changing vision encoder, language model, and architecture.

Each task is scored independently. Success in this evaluation is failure by the target model to generate the correct output from the perturbed image, regardless of prompt.

#### 5.4.2 Investigating K3R via Augmentation Techniques

The goal is one pattern causing predetermined target text for any image, regardless of image content and prompt, improving efficiency over per-image optimization. The attacker has a limited training-image set and corresponding J6T variants, white-box access during optimization, and must generalize to unseen images. The \(L_\infty\) budget is \(\epsilon=16/255\).

A separate test set contains no training images. Taskwise H2Q counts failure to generate correct output text on previously unseen images independently of content or prompt. Controlled self- and cross-mixing is hypothesized to capture image-independent model vulnerabilities rather than specific image features.

#### 5.4.3 Evaluation Rationale for Transferability experiments

All N6Z and F8V transferability results are presented under untargeted attacks. Non-targeted H2Q is also the chosen robust, fair, comparable assessment of between-model and between-image transfer. Distinct visual embedding mechanisms make perturbation influence on final token embeddings difficult to isolate. Targeted transfer is substantially stronger between shared-encoder models such as BLIP-2 and InstructBLIP but varies drastically across heterogeneous encoders; direct targeted-H2Q comparisons across such architectures are imprudent.

Decoder-only LLaVA and OpenFlamingo introduce visual features late, through cross-attention or token concatenation, as auxiliary memory for causal generation; visual influence fades as decoding proceeds, Awadalla et al. (2023), Liu et al. (2023). Fusion-based BLIP-2, Li et al. (2023b), and InstructBLIP, Dai et al. (2023), integrate visual features early and deeply into input embeddings, retaining visual grounding throughout generation. Targeted attacks are inherently more feasible on the latter, Dai et al. (2023).

E3W is excluded from transfer comparisons after negligible transfer in experiments. Its per-image visual-token manipulation embeds target concepts in image-dependent patterns: gradients and token spaces differ vastly across images and concepts, and input/target-image token-space transformations are non-transferable. Its dependence on visual/textual encoder structure also prevents F8V across structural differences among BLIP2, LLaVA, Flamingo, and other LVLMs.

## 6 Results

Five state-of-the-art VLMs were evaluated: OpenFlamingo, BLIP-2, InstructBLIP, and LLaVA. The Flamingo implementation is publicly accessible OpenFlamingo-9B, Awadalla et al. (2023), with performance comparable to the original model.

BLIP-2's two-stage fusion first extracts features with a frozen CLIP image encoder, then processes them through a Querying Transformer, Li et al. (2023b); OPT-2.7b provides flexible language generation and efficient task adaptation. InstructBLIP adds instruction tuning to this fusion architecture, Dai et al. (2023), and uses Vicuna-7b for task-specific instruction following and finer output control through crafted prompts.

LLaVA, Liu et al. (2023), connects pretrained CLIP ViT-L to Vicuna-13B through a lightweight projection. Its decoder-only generation supplies visual features directly as special tokens in the prompt context, unlike explicit pre-generation visual–textual alignment in fusion models. OpenFlamingo and LLaVA represent decoder-only architectures, BLIP-2 and InstructBLIP fusion-based architectures; the distinction significantly affects targeted adversarial effectiveness across prompts.

### 6.1 Claim 1: Q7X achieves N6Z across target texts [Reproduced]

Flamingo-generated Q7X examples transfer across held-out prompts in VQA, G9M, and D5R. The evaluation reproduces the central N6Z claim, with nonzero H2Q reported across all tasks and diverse target prompts.

#### 6.1.1 Experimental Details

The scored task families distinguish general visual questioning (A7L = VQAgeneral), specific visual questioning (T4B = VQAspecific), image-label prediction (G9M = Classification), and image description (D5R = Captioning), together with the aggregate column (C8H = Overall), while the two questioning labels refer to how specifically a prompt pertains to a particular image.

Training follows Luo et al. (2024); evaluation prompts differ from training prompts and span a broad range of semantic meanings. The Flamingo test measures targeted H2Q. Appendix D.1 gives details for all reproduced experiments; Appendix E lists the prompts.

#### 6.1.2 Comparative Analysis

Q7X generally agrees with the claimed transfer behavior but consistently underperforms on G9M and D5R, especially for abstract target texts “too late” and “not sure,” which lack significant visual semantics. The observed successful deception persists across diverse targets.

| Target Prompt | Method | A7L | T4B | G9M | D5R | C8H |
|---|---|---:|---:|---:|---:|---:|
| unknown | R4K | 0.2640 | 0.4720 | 0.2700 | 0.1400 | 0.2865 |
| | V9D | 0.7240 | 0.8740 | 0.5550 | 0.2850 | 0.6095 |
| | Q7X | 0.9680 | 0.9880 | 0.7070 | 0.4200 | 0.7708 |
| I am sorry | R4K | 0.2120 | 0.4740 | 0.5170 | 0.2530 | 0.3640 |
| | V9D | 0.7020 | 0.9020 | 0.7090 | 0.6140 | 0.7318 |
| | Q7X | 0.8620 | 0.9260 | 0.7170 | 0.6890 | 0.7985 |
| not sure | R4K | 0.3540 | 0.4240 | 0.1450 | 0.0000 | 0.2308 |
| | V9D | 0.6520 | 0.6860 | 0.2310 | 0.0030 | 0.3930 |
| | Q7X | 0.8760 | 0.8940 | 0.2420 | 0.0010 | 0.5033 |
| very good | R4K | 0.3980 | 0.5760 | 0.2180 | 0.0480 | 0.3100 |
| | V9D | 0.8940 | 0.9640 | 0.4040 | 0.2080 | 0.6175 |
| | Q7X | 0.9620 | 0.9860 | 0.6060 | 0.2540 | 0.7020 |
| too late | R4K | 0.2520 | 0.4800 | 0.2280 | 0.0220 | 0.2455 |
| | V9D | 0.7880 | 0.9120 | 0.5040 | 0.1630 | 0.5918 |
| | Q7X | 0.9300 | 0.9580 | 0.7010 | 0.1790 | 0.6920 |
| metaphor | R4K | 0.3280 | 0.6020 | 0.5550 | 0.1040 | 0.3973 |
| | V9D | 0.8880 | 0.9520 | 0.8420 | 0.4910 | 0.7933 |
| | Q7X | 0.9840 | 0.9940 | 0.9100 | 0.5840 | 0.8680 |

**Table 1.** Targeted H2Q on Flamingo across target texts and vision–language tasks.

### 6.2 Claim 2: Q7X achieves the best aggregate performance across prompt counts [Reproduced]

At equal prompt counts Q7X exceeds the baselines consistently, and additional prompts do not allow baseline saturation to overtake it. Increasing prompt counts raises H2Q for all methods; R4K intrinsically uses one prompt, making the count variation redundant for it.

“unknown” avoids high-frequency responses common in vision–language tasks, the original authors' rationale accepted for this experiment. Prompt-count gains saturate rapidly, consistent with Luo et al. (2024).

**Figure 4.** Targeted H2Q against prompt count on Flamingo, holding attack iterations equal; Q7X significantly outperforms baselines. The target is “unknown.” R4K is the lone star.

### 6.3 Claim 3: Q7X converges towards higher H2Q as iterations increase [Reproduced]

Taskwise convergence is unstable despite exact reuse of the authors' training/evaluation parameters and code. The original reported convergence metrics (Appendix A.2) differ from these results and showed convergence with more prompts. Across tasks jointly, increasing-iteration convergence follows the original aggregate trend.

The authors did not specify the model for this scenario; the reproduced curves use OpenFlamingo and target “unknown.”

- (a) VQA
- (b) VQA-Specific
- (c) G9M
- (d) D5R

**Figure 5.** R4K, V9D, and Q7X compared across categories.

**Figure 6.** Aggregate convergence and Q7X comparison.

### 6.4 Claim 4: Q7X outperforms baselines under few-shot settings [Reproduced]

Q7X significantly exceeds the baseline on most tasks when context widens through additional examples. This supports the min–max objective's intended robustness to expanded prompt space and consequent context changes.

| Setting | Method | A7L | T4B | G9M | D5R | C8H |
|---|---|---:|---:|---:|---:|---:|
| shot=0 | V9D | 0.7240 | 0.8740 | 0.5550 | 0.2850 | 0.6095 |
| | Q7X | 0.9680 | 0.9880 | 0.7070 | 0.4200 | 0.7708 |
| shot=2 | V9D | 0.7860 | 0.9420 | 0.5630 | 0.0010 | 0.5730 |
| | Q7X | 0.9440 | 0.9940 | 0.6860 | 0.0300 | 0.6635 |

**Table 2.** Targeted H2Q with and without visual in-context learning; shot counts added images.

Flamingo supports extra images as in-context examples for task adaptation. Following Luo et al. (2024), perturbations are optimized at 0-shot, with no extra images, and evaluated at 2-shot, with two extra images, differing from optimization.

Generation-condition shifts make attacks harder, Luo et al. (2024); contextual awareness moves the correct prompt space away from target-text space. Zhao et al. (2023), Qi et al. (2023) verify that context-specific attacks such as Q7X lose effectiveness when context changes.

### 6.5 Results Beyond the Original Paper

The extended experiments incorporate initialization, attention-value guidance, and input-diversity modifications into the original framework.

#### 6.5.1 Noise Initialization

The integrated variant (B8N = CroPA+Init) is reported to outperform Q7X and V9D consistently across every metric, with gains in aggregate accuracy, adversarial robustness, and N6Z; the semantic initialization is also reported to require fewer optimization steps, providing a robust, adaptable alternative to random initialization rather than a change to the perturbation budget.

| Target Prompt | Method | A7L | T4B | G9M | D5R | C8H |
|---|---|---:|---:|---:|---:|---:|
| unknown | V9D | 0.7240 | 0.8740 | 0.5550 | 0.2850 | 0.6095 |
| | Q7X | 0.9680 | 0.9880 | 0.7070 | 0.4200 | 0.7708 |
| | B8N | 0.8824 | 0.8920 | 0.8858 | 0.9306 | 0.8977 |
| bomb | V9D | 0.6840 | 0.8260 | 0.9050 | 0.6090 | 0.7560 |
| | Q7X | 0.8176 | 0.9100 | 0.9498 | 0.6960 | 0.8433 |
| | B8N | 0.9140 | 0.9468 | 0.9020 | 0.9412 | 0.9260 |
| I am sorry | V9D | 0.7020 | 0.9020 | 0.7090 | 0.6140 | 0.7318 |
| | Q7X | 0.8620 | 0.9260 | 0.7170 | 0.6890 | 0.7985 |
| | B8N | 0.9320 | 0.9524 | 0.9270 | 0.9716 | 0.9458 |
| very good | V9D | 0.7900 | 0.8860 | 0.6700 | 0.6700 | 0.7540 |
| | Q7X | 0.8540 | 0.8990 | 0.7680 | 0.7550 | 0.8190 |
| | B8N | 0.8745 | 0.8894 | 0.8362 | 0.9043 | 0.8761 |
| too late | V9D | 0.6960 | 0.8540 | 0.6780 | 0.6060 | 0.7585 |
| | Q7X | 0.8990 | 0.9060 | 0.7940 | 0.8150 | 0.8535 |
| | B8N | 0.9340 | 0.9220 | 0.9110 | 0.9660 | 0.9333 |

**Table 3.** BLIP2 targeted H2Q over target texts with vision-encoder noise initialization. Detailed hyperparameters: Section D.2.

##### Theoretical Analysis of Initialization in Bi-level Optimization

Semantic starting points are deliberately closer to meaningful adversarial directions. Zhang et al. (2025) find min–max problems substantially more initialization-sensitive than ordinary minimization, often needing carefully designed starts for stable convergence. Q7X resembles a saddle-point problem, with prompt perturbations continually changing the optimization landscape.

More frequent image than prompt updates create asymmetry and amplify starting-point importance. Chen et al. (2019) find local cycling in alternating-gradient min–max methods and convergence strongly dependent on initial proximity to adversarially robust regions. Wu et al. (2020) find semantic-prior perturbations transfer better across models than random starts and yield substantial success-rate improvements; the experimental observations align with that finding, particularly for the more complex bi-level optimization paths.

| Method | A7L | T4B | G9M | D5R | C8H |
|---|---:|---:|---:|---:|---:|
| V9D | 0.7053 | 0.8807 | 0.7001 | 0.5553 | 0.7104 |
| Q7X | 0.8723 | 0.9457 | 0.7518 | 0.6138 | 0.7958 |
| E3W | 0.2985 | 0.2281 | 0.4857 | 0.4687 | 0.370 |
| B8N | 0.9145 | 0.9342 | 0.8982 | 0.9464 | 0.9209 |

**Table 4.** Average targeted H2Q on BLIP-2 across visual-questioning, G9M, and D5R tasks. Best task values are highlighted in bold. E3W, Yang et al. (2024), is the additional baseline.

Gradient attacks ordinarily acquire target semantics and theoretically should not require semantic initialization. Min–max complexity weakens that alignment; injecting target semantics at initialization is described as necessary to reinforce it and assist convergence.

Feature uptake is quantified between target text and learned noise (Y2K = CLIP score), comparing the base perturbation with its initialized counterpart rather than measuring only task success, to assess the extent to which perturbations acquire target attributes.

| Method | Y2K |
|---|---:|
| Q7X | 21.80 |
| B8N | 23.09 |

Target text: “Bomb”.

**Figure 7.** (a) Aggregate convergence of Q7X and B8N. (b) Learned-perturbation/target feature similarity quantified by Y2K.

#### 6.5.2 Investigating K3R and Image Augmentation

Learning across images does not yield effective universality for unmodified Q7X: prompt-directed signals remain tightly linked to the originating image's low-level features. This weakens the original claim that learning perturbations across images would establish K3R.

Both input-mixing approaches give measurable gains, but the gap between cross-image methods and Q7X applied to a single image remains a fundamental limitation. Q7X excels at prompt robustness, not effective image-transferring adversarial attacks.

U8G is believed to outperform J6T for adversarial transfer: region-based mixing retains distinct, non-overlapping spatial areas, encouraging modular, independently manipulable features and greater cross-input generalization.

| Method | VQA | T4B | G9M | D5R | C8H |
|---|---:|---:|---:|---:|---:|
| Q7X (Base) | 0.3500 | 0.4920 | 0.4000 | 0.5000 | 0.4355 |
| Q7X with J6T | 0.6560 | 0.7160 | 0.5000 | 0.9500 | 0.7055 |
| Q7X with U8G | 0.4520 | 0.7240 | 0.5000 | 0.5500 | 0.5565 |

**Table 5.** Untargeted K3R H2Q, best across all iterations. Unmixed Q7X peaks at 1700 iterations; J6T and U8G exceed it significantly earlier, at 1500 and 1600 respectively. Experimental parameters: Appendix D.3.

#### 6.5.3 Guiding perturbations via Target Value Vectors

P2F greatly improves the existing algorithm across tasks, especially A7L and D5R. The target “Bomb” is harmful text expected inherently to activate guardrails; the improvement is maintained for such worse expected harmful-word cases and vulnerable scenarios.

| Method | A7L | T4B | G9M | D5R | C8H |
|---|---:|---:|---:|---:|---:|
| V9D | 0.68 | 0.82 | 0.90 | 0.60 | 0.75 |
| Q7X | 0.8176 | 0.9100 | 0.9498 | 0.6960 | 0.8433 |
| E3W | 0.3431 | 0.3102 | 0.4732 | 0.5534 | 0.4199 |
| P2F | 0.9420 | 0.9720 | 0.9790 | 0.9810 | 0.9685 |

**Table 6.** Targeted BLIP2 H2Q with X9C loss and “Bomb,” outperforming S3D baselines, E3W, and base Q7X.

| Method | A7L | T4B | G9M | D5R | C8H |
|---|---:|---:|---:|---:|---:|
| V9D | 0.6960 | 0.8540 | 0.8990 | 0.6060 | 0.7638 |
| Q7X | 0.7900 | 0.8860 | 0.9470 | 0.6700 | 0.8233 |
| E3W | 0.3027 | 0.4302 | 0.5112 | 0.5080 | 0.4380 |
| P2F | 0.9000 | 0.9520 | 0.9500 | 0.9060 | 0.9270 |

**Table 7.** Targeted Open Flamingo H2Q with X9C loss and “Bomb.”

#### 6.5.4 Analyzing F8V by Value Vector guidance

The value-guided method significantly improves vanilla Q7X transfer, especially G9M and D5R, with modest VQA gain. VQA prompts occupy a more complex embedding space than the smaller output spans of targeted G9M and D5R.

Encoder dependence verifies the earlier limitation hypothesis: BLIP2 perturbations are passably transferable to same-encoder InstructBLIP and fail to transfer to OpenFlamingo. Training on Flamingo and applying perturbations to the other models likewise gives insignificant H2Q (Table D.4.5, Appendix D.4.5). Experimental details: Appendix D.4.

| Settings | Method | A7L | T4B | G9M | D5R | C8H |
|---|---|---:|---:|---:|---:|---:|
| BLIP2 to InstructBLIP | V9D | 0.1760 | 0.2090 | 0.2750 | 0.2480 | 0.2270 |
| | Q7X | 0.3220 | 0.4170 | 0.6720 | 0.6150 | 0.5065 |
| | P2F | 0.3940 | 0.5780 | 0.8800 | 0.8880 | 0.6850 |
| BLIP2 to Flamingo | V9D | 0.0810 | 0.0660 | 0.0760 | 0.0890 | 0.0780 |
| | Q7X | 0.1520 | 0.1220 | 0.1180 | 0.1820 | 0.1435 |
| | P2F | 0.1850 | 0.2190 | 0.1420 | 0.3880 | 0.2335 |
| BLIP2 to LLaVA | V9D | 0.1220 | 0.1060 | 0.0810 | 0.0630 | 0.0930 |
| | Q7X | 0.3190 | 0.2570 | 0.2730 | 0.3670 | 0.3040 |
| | P2F | 0.5070 | 0.3420 | 0.4380 | 0.8430 | 0.5325 |
| InstructBLIP to BLIP2 | V9D | 0.1230 | 0.1520 | 0.1020 | 0.0960 | 0.1183 |
| | Q7X | 0.2690 | 0.2320 | 0.1870 | 0.2270 | 0.2288 |
| | P2F | 0.4800 | 0.5100 | 0.4700 | 0.5000 | 0.4900 |
| InstructBLIP to LLaVA | V9D | 0.0650 | 0.0580 | 0.0420 | 0.0370 | 0.0505 |
| | Q7X | 0.1620 | 0.1120 | 0.1410 | 0.1230 | 0.1345 |
| | P2F | 0.3450 | 0.3150 | 0.3680 | 0.3950 | 0.3558 |
| InstructBLIP to Flamingo | V9D | 0.0320 | 0.0210 | 0.0120 | 0.0110 | 0.0190 |
| | Q7X | 0.0740 | 0.0680 | 0.0490 | 0.0480 | 0.0598 |
| | P2F | 0.1600 | 0.1400 | 0.1200 | 0.1300 | 0.1375 |
| Flamingo to BLIP2 | V9D | 0.0140 | 0.0070 | 0.0090 | 0.0120 | 0.0105 |
| | Q7X | 0.0450 | 0.0210 | 0.0300 | 0.0380 | 0.0335 |
| | P2F | 0.1080 | 0.0800 | 0.0990 | 0.1180 | 0.1013 |
| Flamingo to InstructBLIP | V9D | 0.0100 | 0.0060 | 0.0050 | 0.0090 | 0.0075 |
| | Q7X | 0.0360 | 0.0220 | 0.0280 | 0.0300 | 0.0290 |
| | P2F | 0.1050 | 0.0720 | 0.0880 | 0.1100 | 0.0938 |
| Flamingo to LLaVA | V9D | 0.0180 | 0.0110 | 0.0140 | 0.0170 | 0.0150 |
| | Q7X | 0.0400 | 0.0300 | 0.0350 | 0.0430 | 0.0370 |
| | P2F | 0.1100 | 0.0850 | 0.0980 | 0.1170 | 0.1025 |

**Table 8.** Cross-model tests. A setting's left model supplies optimized perturbations; the right model is tested.

For a given target text, vision-encoder value vectors remain approximately consistent across input images. They reflect encoder architecture and pretraining. Their consistency may constrain perturbation diversity and highly varied adversarial-example generation. Target-text choice strongly influences outcomes relative to input-image-specific features.

## 7 Discussion

### 7.1 Observations with reproducibility

Reproduction used the authors' public repository and original model configurations for faithful baseline comparison. The implementation was generally well documented; BLIP-2 and InstructBLIP code nevertheless produced multiple runtime errors, requiring substantial debugging and adjustment to run.

### 7.2 Noise Initialization findings

The initialization hypothesis includes reduced cross-modal as well as cross-prompt transfer under semantic ignorance. Experiments across all evaluated benchmarks are described as significantly favoring semantic noise over Q7X and random-initialized baselines.

Q7X perturbations already acquire target context, as expected of gradients. Initialization produces a modest Y2K increment, not a significant semantic-feature upgrade, interpreted as a subtle shift closer to adversarially vulnerable regions. Convergence before and after initialization aligns with the hypothesis that a starting point nearer meaningful adversarial directions converges faster.

### 7.3 Evaluation of Transferability Experiments

#### 7.3.1 K3R

J6T improves K3R particularly for VQA, T4B, and D5R; U8G also improves the base image-transfer version. The proposed account favors J6T's explicit spatial/scale invariance from same-image random cropping and resizing followed by soft mixing with another image. Semantically aligned features become less sensitive to position and scale, easing cross-input generalization.

Both augmentations preserve original semantic information while adding image diversity. Their proposed benefit is learning cross-image features, but augmentation alone does not eliminate the limitation. Further work should combine augmentation with more complex, layered methods: latent-representation feature-alignment attacks or direct latent-space perturbation.

#### 7.3.2 F8V

X9C guidance is described as a significant advance robust to harmful target text and guardrail activation. Encoder-specific dependence remains the obstacle to encoder-agnostic attacks: it narrows perturbation diversity and heterogeneous-architecture transfer. Despite that limitation, the framework is presented as progress toward universal attacks effective across prompts and images and toward improved VLM generalizability and robustness. E3W cannot supply comparable transfer results and is inherently non-transferable.

### 7.4 Trade-offs Between Different Types of Adversarial Transferability

#### 7.4.1 N6Z vs. F8V

Competing text embeddings implicitly anchor original Q7X perturbations to the target architecture; moving attacks between models causes a significant H2Q drop. X9C targets potentially generalizable encoder components rather than model-specific features. It improves related-architecture transfer, BLIP-2 to InstructBLIP, but offers only modest gains across fundamentally different architectures, BLIP-2 to Flamingo. Simultaneously optimizing prompt diversity and model diversity is inherently tense.

#### 7.4.2 N6Z vs. K3R

Image-specific coupling serves strong N6Z and fundamentally limits K3R: the two are competing optimization objectives. Augmentation diversifies optimization and yields promising improvements without resolving that fundamental tension.

#### 7.4.3 Role of Initialization in Transferability Balance

Semantic anchoring raises H2Q without sacrificing N6Z, indicating that starting-point selection substantially influences the achievable transfer balance. Unlike optimization-process modifications, better initialization allows convergence toward more general adversarial directions remaining effective across prompts. Even then, optimizing a perturbation for one transfer dimension inevitably sacrifices effectiveness in others.

#### 7.4.4 Implications for Future Research

No single methodology can maximize all transfer forms simultaneously. Controlled balancing may require multi-objective optimization or architecturally aware attack designs. Pushing individual dimensions is insufficient for a comprehensive solution without fundamentally rethinking perturbation interactions with model architecture and data representation.

N6Z remains underexplored, with little literature on prompt-induced vulnerability. The resulting understanding gap matters for secure deployment.

### 7.5 Broader Impact

High-N6Z examples can elicit malicious or misleading outputs from harmless queries, enabling misinformation or subversion of VLM-dependent content-generation and decision systems.

Imperceptible perturbations also offer privacy protection: embedded in personal images, they can force predetermined nonsensitive text and thwart unauthorized extraction of confidential information. This is a proactive security mechanism as visual data becomes increasingly exploitable.

The initialization and loss refinements increase H2Q and cross-model/cross-image generalizability, strengthening cross-prompt attacks while opening paths toward more resilient defenses. Deeper understanding is necessary for robust VLMs, effective countermeasures, user-data protection, and integrity of outputs.

### 7.6 Limitations and Future Work

Computational scarcity repeatedly hindered the successful key-result replication and extensive ablations. Large Flamingo and BLIP2 attack training/evaluation require substantial GPU memory and processing, limiting experimental scale and depth, extensive hyperparameter tuning, and transfer testing over wider architecture ranges.

Planned ensemble-model transfer investigation could have deepened generalizability analysis but exceeded available memory and compute and was infeasible on the existing infrastructure. Future work includes encoder-agnostic F8V, stronger K3R, and query-based optimization for practical applicability.

### 7.7 Communication With Original Authors

No direct communication with the original authors was established during replication. The GitHub issue concerning BLIP-2 and InstructBLIP remains unresolved at writing.
