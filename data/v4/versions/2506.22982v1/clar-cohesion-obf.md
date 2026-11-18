# Revisiting CroPA: A Reproducibility Study and Enhancements for Cross-Prompt Adversarial Transferability in Vision-Language Models

| Author | Email | Affiliation | Author notation |
|---|---|---|---|
| Atharv Mittal | atharv_m@mfs.iitr.ac.in | Mehta Family School of Data Science and Artificial Intelligence; Indian Institute of Technology, Roorkee | ††thanks: All authors contributed equally |
| Agam Pandey | agam_p@ce.iitr.ac.in | Department of Civil Engineering; Indian Institute of Technology, Roorkee | 11footnotemark: 1 |
| Amritanshu Tiwari | amritanshu_t@mfs.iitr.ac.in | Mehta Family School of Data Science and Artificial Intelligence; Indian Institute of Technology, Roorkee | 11footnotemark: 1 |
| Sukrit Jindal | sukrit_j@mfs.iitr.ac.in | Mehta Family School of Data Science and Artificial Intelligence; Indian Institute of Technology, Roorkee | 11footnotemark: 1 |
| Swadesh Swain | swadesh_s@ece.iitr.ac.in | Department of Electronics and Communication; Indian Institute of Technology, Roorkee | 11footnotemark: 1 |

###### Abstract

This reproducibility study of “An Image is Worth 1000 Lies: Adversarial Transferability Across Prompts on Vision-Language Models” validates the Cross-Prompt Attack method (CroPA, coded Q7X) and its superiority to existing baselines in Cross-Prompt Transferability (coded N6Z): the effectiveness of an adversarial image across varying textual prompts. The study finds that semantic meaning and target-word frequency do not determine that transferability. The replication is extended through perturbation initialization, common perturbations across images, and attention-directed loss. Evaluation encompasses the vision-language models (VLMs) Flamingo, BLIP-2, and InstructBLIP, together with extended LLaVA experiments. The original results are validated, and the enhancements consistently increase adversarial effectiveness. The resulting framework generates more robust transferable adversarial examples and therefore bears on real-world VLM security.

Code: https://github.com/Swadesh06/Revisting_CroPA

Reviewed on OpenReview: https://openreview.net/forum?id=5L90cl0xtf

## 1 Introduction

Subtle image perturbations induce incorrect or harmful VLM outputs. Because VLMs also accept textual input, textual input adds attack vectors beyond those of traditional vision models. Large VLMs transformed computer vision through image Classification (coded G9M), Captioning (coded D5R), and visual question answering (VQA), enabling deeper investigation of visual–linguistic interaction. However, vulnerabilities were demonstrated by Zhao et al. (2023); Qi et al. (2023); Zhang et al. (2022a); Carlini et al. (2024).

Three transfer settings distinguish where adversarial examples remain effective:

- **Cross-Model Transferability (coded F8V):** adversarial examples survive changes in model architecture. This setting is commonly called transferability.
- **Cross-Image Transferability (coded K3R):** perturbations remain effective over multiple images. This setting admits shared perturbations, called Universal Adversarial Perturbations (UAPs, coded Z4P), rather than requiring a separately optimized perturbation for each image.
- **Cross-Prompt Transferability (N6Z):** adversarial examples remain effective over varying textual prompts.

Luo et al. (2024) introduced cross-prompt transferability and competing learnable prompts. Their image attacks retain effectiveness regardless of textual input across Classification, Captioning, and VQA. This differs from earlier work that treated visual and textual adversarial perturbations independently. Zhao et al. (2023) analyzed BLIP and BLIP-2 using query- and transfer-based black-box attacks, whereas Schlarmann & Hein (2023) examined targeted and untargeted white-box attacks. Qi et al. (2023) showed that adversarial images elicit harmful instruction execution, and Tu et al. (2023) systematically evaluated attacks on both modalities.

The replication tests CroPA's superiority across target texts, rapid saturation as prompts are added, robustness under added in-context examples, and convergence over attack iterations. The replication also adds taskwise analysis beyond the original aggregate analysis. The effectiveness measure is Attack Success Rate (ASR, coded H2Q), with targeted and untargeted success defined in Section 4.1.

The extensions investigate three changes: initialization intended to improve both attack success rate and transferability; shared perturbations on the CroPA backbone, using transfer-promoting image augmentation; and target value vectors. Because attention components govern patch interaction, targeted manipulation of those components is proposed to improve generalization and attack success rate. The analyses investigate both the mechanisms of these enhancements and the nature of VLM adversarial vulnerabilities.

## 2 Related Works

Szegedy et al. (2013) and Goodfellow et al. (2014) introduced adversarial attacks on deep neural networks. Applying adversarial attacks to VLMs adds challenges and opportunities.

### 2.1 Adversarial Transferability

Examples directed at classification also affect object detection and other visual tasks. This establishes cross-task transferability alongside cross-model and cross-image transferability: Naseer et al. (2019b); Naseer et al. (2019a); Lu et al. (2020); Salzmann et al. (2021). Because transferability spans these different dimensions, this behavior raises robustness questions for modern machine-learning systems.

Liu et al. (2017) and Tramèr et al. (2017) established that attacks crafted on one model often deceive other models, including models with different architectures. In contrast, Moosavi-Dezfooli et al. (2017); Mopuri et al. (2017) established perturbations that generalize across images, including Universal Adversarial Perturbations.

### 2.2 Adversarial Attacks on Vision-Language Models

Early task-specific Captioning attacks manipulated outputs in Xu et al. (2019); Zhang et al. (2020); Aafaq et al. (2021); Chen et al. (2017). Targeted VQA attacks misled attention mechanisms in Xu et al. (2018); Kaushik et al. (2021); Kovatchev et al. (2022); Li et al. (2021); Sheng et al. (2021); Zhang et al. (2022b).

These attacks principally addressed lightweight convolutional-neural-network–recurrent-neural-network (CNN–RNN) systems without in-context learning. Consequently, their applicability to contemporary VLMs is restricted. The modern-large-VLM black-box study of BLIP and BLIP-2 by Zhao et al. (2023) emphasized cross-model rather than cross-prompt transferability.

### 2.3 Cross-Prompt Adversarial Transferability

No credible benchmarks or baselines have yet been established for this relatively new concept. Nevertheless, Yang et al. (2024) supplies an additional experimental comparator: Context Injection Attack (CIA, coded E3W), also called Contextual Injection Attack. Context Injection Attack uses gradient-based perturbations to inject target tokens into visual and textual contexts, rather than using the competing-prompt optimization pioneered by CroPA in Luo et al. (2024).

Unlike fixed-task traditional vision models, large VLMs can switch tasks dynamically through prompt-driven adaptation: Li et al. (2023b); Awadalla et al. (2023); Li et al. (2023b); Zhu et al. (2023); Liu et al. (2023). Therefore, adversarial validity across changing prompts constitutes an additional attack surface. Making the modalities compete, rather than manipulating the modalities independently, establishes a distinct paradigm for evaluating VLM robustness.

### 2.4 Enhancing Adversarial Transferability

Gradient misalignment limits transferability and has therefore motivated optimization strategies in Li et al. (2023a); Wang et al. (2023). Input diversity in Xie et al. (2019) and augmentation in Wang et al. (2021) also improve attack robustness. These findings inform the multidimensional CroPA enhancements investigated here.

## 3 Scope of reproducibility

Faithful reproduction of Luo et al. (2024) requires following their experimental procedures. The reproduction tests the following named claims:

- Claim 1: CroPA achieves cross-prompt transferability across target texts.
- Claim 2: CroPA has the best aggregate performance across prompt counts.
- Claim 3: CroPA converges toward higher attack success rate with increasing iterations.
- Claim 4: CroPA outperforms baselines in few-shot settings.

The extended scope includes cross-model transferability, cross-image transferability, and comparison to other cross-prompt methods. Specifically, images generated through CroPA are tested for consistent deception across diverse VLMs, input prompts, and model parameters. Thus, the extended scope concerns the broader robustness and versatility of adversarial attacks, beyond validation of the original prompt-transfer improvement.

## 4 Methodology

### 4.1 Problem Formulation

Targeted success requires generation of the exact target text. In contrast, untargeted success requires any departure from the clean-image prediction. Attack Success Rate (ASR; H2Q) is the corresponding effectiveness quantity for the reproduced targeted and non-targeted scenarios.

The formulation uses a language-modeling objective over prompt instances rather than a vision-only objective. Let \(f\) be the model that processes a visual input \(x_v\) and a textual input \(x_t\). Let \(\delta_v\) be an image perturbation, \(x_t^i\) the \(i\)-th textual prompt among \(k\) prompts, and \(L\) the language-modeling loss. In the targeted scenario, \(T\) is predetermined text to be generated independently of the input prompt. The targeted objective is:

\[
\min_{\delta_v}\sum_{i=1}^{k}
L\bigl(f(x_v+\delta_v,x_t^i),T\bigr)
\tag{1}
\]

In contrast, the untargeted objective maximizes discrepancy between the clean-image output \(f(x_v,x_t^i)\) and the adversarial-image output:

\[
\max_{\delta_v}\sum_{i=1}^{k}
L\bigl(f(x_v+\delta_v,x_t^i),f(x_v,x_t^i)\bigr)
\tag{2}
\]

The human-imperceptibility bound is \(\|\delta_v\|_p\leq\epsilon_v\), where \(\|\cdot\|_p\) is the perturbation's \(p\)-norm and \(\epsilon_v\) is the allowed perturbation bound. This bound adapts established adversarial practice to multimodal real-world applicability, Carlini et al. (2024). The formulation extends the neural-network vulnerability setting of Goodfellow et al. (2014) to prompt variation, Luo et al. (2024).

### 4.2 Baseline Approaches

Because established cross-prompt baselines were absent, Luo et al. (2024) introduced two reference attacks. Single-P (coded R4K) optimizes a perturbation with one prompt \(x_t\):

\[
\min_{\delta_v}L\bigl(f(x_v+\delta_v,x_t),T\bigr)
\tag{3}
\]

Multi-P (coded V9D) instead optimizes over a collection of prompts, \(X_t=\{x_t^1,x_t^2,\ldots,x_t^k\}\):

\[
\min_{\delta_v}\sum_{i=1}^{k}
L\bigl(f(x_v+\delta_v,x_t^i),T\bigr)
\tag{4}
\]

The one-prompt case is the simplest baseline. The untargeted forms of both baselines invert the targeted objectives: rather than increasing target-text likelihood, the untargeted forms maximize clean/perturbed output discrepancy.

In Figure 1, \(\delta_t\) denotes a prompt perturbation, complementing the image perturbation \(\delta_v\).

**Figure 1.** Luo et al. (2024) framework: image \(\delta_v\) and prompt \(\delta_t\) perturbations act antagonistically, minimizing and maximizing language-modeling loss respectively. Solid arrows denote the forward pass through vision encoder and language model; dashed arrows denote backpropagation, with configurable perturbation-update frequencies. Asterisks mark fixed model parameters; \(\oplus\) denotes elementwise addition.

### 4.3 Cross Prompt Attack

Cross-Prompt Attack (CroPA; Q7X), the competing-prompt method of Luo et al. (2024), jointly optimizes the image perturbation \(\delta_v\) and prompt perturbation \(\delta_t\). The objectives are antagonistic rather than collaborative, with the intention of forcing stronger image transferability across textual prompts:

\[
\min_{\delta_v}\max_{\delta_t}
L\bigl(f(x_v+\delta_v,x_t+\delta_t),T\bigr)
\tag{5}
\]

During evaluation, the prompt perturbations are discarded, and only the optimized image perturbation is applied. Therefore, evaluation effectiveness must reside in the image perturbation's inherent transferability, not in modified prompts.

The implementation uses iterative image-gradient descent and prompt-gradient ascent, following Projected Gradient Descent (PGD, coded S3D), Madry et al. (2017). Image updates occur \(N\) times per prompt update. Thus, the two modalities have configurable update frequencies rather than simultaneous, equal-frequency updates.

Algorithm 1 uses \(\epsilon\) for the perturbation size, \(\alpha_1\) and \(\alpha_2\) for the image and prompt step sizes, \(K\) for the iteration count, and \(N\) for the adversarial-prompt update interval. The adversarial image is \(x'_v\), and the initialized adversarial version of prompt \(x_t^i\) is \(x'_t{}^i\). The gradients \(g_v\) and \(g_t\) are the image and prompt loss gradients. The operator \(\nabla\) denotes a gradient, \(\operatorname{sign}\) takes its elementwise sign, \(\operatorname{mod}\) gives the remainder, and \(\operatorname{Clip}_{x_v,\epsilon}\) projects the image within the \(\epsilon\)-ball around \(x_v\).

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

CroPA performs poorly on Classification and Captioning and has limited cross-model transferability. Although its Projected Gradient Descent optimization effectively raises single-model attack success rate, dependence on that model's loss feedback constrains transferability. This is a recognized limitation of gradient attacks: Qin et al. (2024); Li et al. (2024); Wang et al. (2023); Li et al. (2023a). The original authors recognize the limitation, and Table 9 in Appendix A.1 verifies the limitation.

The extensions address both task inconsistency and transfer dimensions. Cross-image transferability could remove separate per-image training, and cross-model transferability could provide still greater efficiency. However, transfer exploration is not a core proposal of Luo et al. (2024).

The following table retains resource costs and quantitative characterizations separately from the associated comparisons. For this table, GPU means graphics processing unit, and VRAM means video random-access memory. CroPA+D-UAP (P2F), also called re-CroPA, is the value-guided method detailed in Section 5.2; SCMix (J6T) and CutMix (U8G) are the augmentations detailed in Section 5.3. VQAgeneral (A7L) denotes general visual questioning. The remaining codes retain their meanings: Q7X is CroPA, H2Q is attack success rate, D5R is Captioning, G9M is Classification, F8V is cross-model transferability, and K3R is cross-image transferability.

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

Noise Initialization via Vision Encoding Optimization (coded W5J) is a diffusion-anchored method that replaces target-agnostic random noise with target-image feature alignment. The method retains imperceptibility while supplying semantically meaningful adversarial directions at the outset. In contrast, conventional initialization neither guarantees such directions nor avoids reduced attack success rate and cross-prompt transferability.

CroPA shares random initialization with many gradient attacks. Ignorance of target semantics is associated with instability and slower convergence, particularly in VLMs requiring semantic prompt alignment. Empirically, the bi-level framework is initialization-sensitive, as examined in Section 6.5.1, Figure 7, and Table 3. The proposed hypothesis is therefore that semantically informed, rather than purely random, perturbations improve attack efficacy within the same imperceptibility requirement.

A state-of-the-art diffusion model synthesizes a target image. The target image's encoder representation then anchors the initial noise to the objective. An encoder-output error loss enforces semantic similarity while retaining adversarial effectiveness. Specifically, Mean Squared Error (MSE, coded L6V) supplies the optimization criterion, rather than a purely random starting direction for subsequent attack updates.

The method is presented as improving adversarial robustness, strengthening perturbations and cross-prompt transferability, and reducing the steps needed for convergence.

#### 5.1.1 Noise Initialization

Preliminary BLIP-2 and InstructBLIP experiments favor semantic over random noise. Diffusion-based anchoring improves encoder-space alignment while preserving the visual fidelity of adversarial examples and strict perturbation budgets. The additional computation for a single image is limited to the single-GPU overhead of 20-25 seconds reported in the Section 5 resource table.

#### 5.1.2 Diffusion-Based Target Synthesis

Let \(\mathcal D\) be Stable Diffusion XL (SDXL), \(T\) the target text, \(\mathbf z\) Gaussian-sampled noise, and \(\theta_{\mathrm{SDXL}}\) the pretrained diffusion weights. The synthesized target image is \(\mathbf{x}_{\mathrm{target}}\):

\[
\mathbf{x}_{\mathrm{target}}
=\mathcal{D}(T,\mathbf{z};\theta_{\mathrm{SDXL}})
\tag{6}
\]

The generated target image embodies the semantic attributes of prompt \(T\) and therefore supplies the initialization reference.

#### 5.1.3 Vision-Encoder Anchored Perturbation

Let \(f_v\) be the VLM's vision component, \(\mathbf x\) the original image, and \(\boldsymbol{\delta}\) a candidate perturbation. The maximum allowable, imperceptible \(\ell_\infty\) perturbation is \(\epsilon\). The initialization perturbation \(\boldsymbol{\delta}_{\mathrm{init}}\) minimizes the squared encoder-representation distance:

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

Here \(\arg\min\) selects the perturbation that minimizes the objective, and \(\|\cdot\|_2^2\) is squared Euclidean distance. Mimicking the target image's encoder-space representation places the adversarial example in a semantically meaningful neighborhood of the target prompt representation. Algorithm 2 in Appendix B formalizes the method.

#### 5.1.4 Adversarial Optimization via S3D

Projected Gradient Descent (S3D) supplies refinement updates after initialization. Let \(\mathbf{x}_{\mathrm{adv}}^{(t)}\) be the adversarial example at iteration \(t\), \(\alpha\) the step size, and \(\Pi_{\mathcal B_\epsilon(\mathbf x)}\) projection into the admissible \(\ell_\infty\) ball around the original image \(\mathbf x\). The loss \(\mathcal L_{\mathrm{L6V}}\) is the encoder-output Mean Squared Error, defined in Equation (9):

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

These refinement updates incrementally align the adversarial example with the target embedding while projection retains the perturbation within the admissible ball.

### 5.2 Guiding perturbations via Target Value Vectors and evaluating F8V

This extension targets cross-model transferability (F8V) through value vectors in middle-to-late vision layers, which are the intended vulnerable components. CroPA lacks architectural sensitivity to the encoder's representational backbone and treats vision and language as uniformly vulnerable. Consequently, CroPA overlooks modality-specific weaknesses despite the encoder's dominant role in semantic grounding. Generic Projected Gradient Descent output manipulation is less structured and less targeted than the proposed alternative. The intended improvements concern attack strength, transferability, interpretability, and computational feasibility.

The prior component-selection method is Doubly-Universal Adversarial Perturbation (Doubly-UAP, also D-UAP, coded X9C), proposed by Kim et al. (2024). Doubly-UAP identifies the attention components that most influence VLM performance when constructing Universal Adversarial Perturbations. Its component loss is adapted here by minimizing, rather than maximizing, losses between value vectors of input image \(x_i\) and target-associated reference \(T_i\), thereby retaining a reference associated with the target text.

Cui et al. (2023); Wang et al. (2024) show that perturbing vision-output embeddings impairs the visual perception of large vision-language models (LVLMs) and misleads their language responses. LVLMs commonly use similar image encoders to produce visual tokens containing semantic and structural information. Because these tokens are the interface through which language decoders access and reason about images, targeting encoder internals is proposed to exploit that shared interface.

Two attention components have fundamental roles. Attention weights control patch-to-patch focus, relevance, and interaction, whereas value vectors contain each patch's actual information. Therefore, interference with attention weights is hypothesized to damage relationship establishment, while perturbation of value vectors is hypothesized to damage essential content and interpretation. Prior analysis identifies middle-to-late-layer value vectors as the most vulnerable components.

In the standard Doubly-UAP objective below, \(L\) is the set of selected layers, \(|L|\) is the number of selected layers, \(l\) indexes a layer, and \(V_l(x)\) denotes input image \(x\)'s value vectors at layer \(l\). The perturbation is \(\delta\), \(\delta^*\) is the maximizing perturbation, and \(\operatorname{Loss}(\cdot)\) is the loss applied to target vectors:

\[
\delta^*
=
\arg\max_\delta
\frac{1}{|L|}
\sum_{l\in L}
\operatorname{Loss}\bigl(V_l(x),V_l(x+\delta)\bigr)
\tag{10}
\]

The proposed combination retains CroPA's opposing image/text objectives and adds attention-value disruption under the same hyperparameter constraints. The hypothesis is that perturbing value vectors relative to a reference image corresponding to the target text enhances attack success rate. Freezing the text encoder is intended to isolate degraded visual processing rather than text artifacts. Thus, the modifications seek encoder-information exploitation, target-text semantic alignment, and better cross-model adversarial robustness.

#### 5.2.1 Our Modified Approach

The combined method is CroPA+D-UAP (P2F), also called re-CroPA. For its value-vector loss, \(V_i(x+\delta)\) is the perturbed input's value vector at attention head \(i\), \(V_t\) is the reference image's target value vector, and \(N\) counts heads. The dot denotes an inner product, and \(\|\cdot\|_2\) denotes a Euclidean norm. The loss \(L_{\mathrm{X9C}}\) computes one minus cosine similarity:

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

Let \(L_{\mathrm{Q7X}}\) be the CroPA loss over perturbed visual input \(x_v+\delta_v\), perturbed textual input \(x_t+\delta_t\), and target text \(T\). The scaling parameter \(\lambda\) controls the relative importance of the value loss. The combined loss is:

\[
L_{\mathrm{P2F}}
=
L_{\mathrm{Q7X}}(x_v+\delta_v,x_t+\delta_t,T)
-\lambda L_{\mathrm{X9C}}(\delta_v)
\tag{12}
\]

CroPA+D-UAP extracts the input image's and target-associated reference image's layer-\(l\) value vectors, computes their cosine-similarity loss, and jointly minimizes the negative of that loss with the CroPA loss under appropriate scaling. Thus, the method does not solely maximize departure from the original representation.

The loss is described as encouraging closeness to the target representation. Combined gradients update \(\delta\), preserving adversarial character, cross-prompt transferability, and guided semantic influence. The intended controlled, interpretable manipulation supports robustness and security analysis. Algorithm 3 in Appendix B gives the steps.

However, the encoder dependence of Doubly-UAP supplies a limitation hypothesis: transfer gains may be restricted to VLMs with the same or similar vision-encoder architecture as the model used for CroPA optimization.

**Figure 2.** X9C architecture diagram.

### 5.3 Transferability across images via Image Augmentation

Per-image iterative gradients overfit image-specific features, producing poor cross-image transferability and high computational expense. CroPA likewise learns each image's perturbation across prompts. Luo et al. (2024) describe CroPA as orthogonal to cross-image transferability and suggest computing perturbations across images, following Moosavi-Dezfooli et al. (2017). This investigation tests that claim and an additional approach based on Universal Adversarial Perturbation literature.

One investigated augmentation is CutMix (U8G). Rather than blending pixels, CutMix synthesizes images by replacing rectangular regions with patches cut from another image. Labels are mixed in proportion to patch area. This partial-view learning promotes spatially invariant features and robustness to occlusion and adversarial attacks, Yun et al. (2019).

Fang et al. (2024) introduces Universal Adversarial Perturbations for Vision-Language Pretraining (VLP) models to reduce gradient-attack overfitting and enable cross-image transferability. Zhang et al. (2024) improves global and local optimization across VLP models and tasks through Effective and Transferable Universal Adversarial Attack (ETU, coded M7A). Within that framework, self- and cross-mixing augmentation (SCMix, J6T) increases diversity while preserving semantics and spatial/scale invariance.

SCMix is selected for integration into perturbation generation on the basis of empirical results: for the same model, SCMix alone, without their additional local utility reinforcement, yields superior cross-image transferability.

#### 5.3.1 J6T

SCMix (J6T) has two stages. In the first stage, \(I_1\) is the base image, and \(X_1,X_2\) are random crops of \(I_1\) resized by \(\operatorname{Resize}\). The mixing weight is \(\eta\), and the self-mixed image is \(I'_1\):

\[
I'_1=\eta X_1+(1-\eta)X_2,
\qquad
X_i=\operatorname{Resize}(\operatorname{RandomCrop}(I_1)),
\qquad
\eta=0.5
\tag{13}
\]

In the second stage, the self-mixed image is cross-mixed with secondary image \(I_2\), producing \(I_3\). The weights \(\beta_1\) and \(\beta_2\) give much greater weight to the base-image contribution, as denoted by \(\gg\):

\[
I_3=\beta_1 I'_1+\beta_2 I_2,
\qquad
\beta_1\gg\beta_2\in[0,1)
\tag{14}
\]

Original SCMix samples \(\eta\sim\operatorname{Beta}(\alpha,\alpha)\), where \(\operatorname{Beta}\) is the beta distribution and \(\alpha>0\) is its parameter. In contrast, this implementation fixes \(\eta=0.5\), equivalent to \(\alpha=\infty\), as justified in Appendix D.3.

The weighting preserves dominant \(I_1\) semantics and the compatibility of \(I_3\) with the base image's text \(T_1\), despite the additional \(I_2\) features. Consequently, the mixed image enriches cross-modal interaction. Intra-image variation and inter-image blending regularize against CroPA overfitting and encourage perturbations directed at invariant features. The proposed common perturbation may then apply to unseen images without image-specific optimization.

CroPA returns only a perturbed image, equivalently its perturbation, at inference. Effective cross-image transferability would therefore make that output universal in both the original cross-prompt sense and the additional investigated cross-image sense.

**Figure 3.** Self-mixing \(I_1\) generates \(X_1,X_2\), combined into \(I'_1\); external \(I_2\) is mixed with \(I'_1\) to produce \(I_3\).

#### 5.3.2 U8G

CutMix (U8G) instead uses discrete regional replacement. Let \(x_A,x_B\) be paired images and \(M\in\{0,1\}^{W\times H}\) a binary mask for a randomly located rectangle, with image width \(W\) and height \(H\). The operator \(\odot\) denotes elementwise multiplication, and the mixed image is \(\tilde x\):

\[
\tilde{x}=M\odot x_A+(1-M)\odot x_B
\tag{15}
\]

Discrete replacement retains partial views. Thus, whereas SCMix softly blends features within and across images, CutMix uses a hard mask. In adversarial training, features that persist across regional combinations are expected to direct perturbations toward spatially invariant, higher-level semantics rather than localized artifacts, potentially improving cross-image transferability.

### 5.4 Attack Setup and Threat Models for Transferability Experiments

Attacker knowledge, perturbation capability, and evaluation differ between the model-transfer and image-transfer experiments.

#### 5.4.1 Using X9C to explore F8V for Q7X

This experiment uses Doubly-UAP (X9C) guidance to explore cross-model transferability (F8V) for CroPA (Q7X). The attacker seeks one image perturbation that elicits a specified target text from multiple VLMs independently of prompt. The attacker optimizes against an accessible surrogate and then deploys the perturbation against unknown models.

White-box surrogate access includes gradients and internal representations; the surrogate is BLIP-2 or Flamingo. Only a target's general architecture family, such as a CLIP-based family, is known, and target parameters are unavailable. CLIP denotes Contrastive Language–Image Pretraining.

The imperceptibility budget is \(\epsilon=16/255\) in \(L_\infty\). Intra-family testing optimizes BLIP-2 with OPT-2.7b and evaluates InstructBLIP with Vicuna-7b. Both have similar CLIP ViT-L/14 encoders, where ViT denotes Vision Transformer, but have different language models and fine-tuning objectives. Cross-architecture testing runs BLIP-2 to Flamingo and the reverse, thereby changing the vision encoder, language model, and architecture.

Each task is scored independently. In this evaluation, success means that the target model fails to generate the correct output from the perturbed image, regardless of prompt.

#### 5.4.2 Investigating K3R via Augmentation Techniques

The cross-image transferability (K3R) experiment seeks one pattern that causes predetermined target text for any image, regardless of image content and prompt. Such a pattern would improve efficiency over per-image optimization.

The attacker has a limited training-image set and corresponding SCMix variants, has white-box access during optimization, and must generalize to unseen images. The \(L_\infty\) budget is \(\epsilon=16/255\).

A separate test set contains no training images. Taskwise attack success rate counts failure to generate correct output text on previously unseen images, independently of content or prompt. Controlled self- and cross-mixing is hypothesized to capture image-independent model vulnerabilities rather than specific image features.

#### 5.4.3 Evaluation Rationale for Transferability experiments

All cross-prompt and cross-model transferability results are presented under untargeted attacks. Non-targeted attack success rate is also the chosen robust, fair, comparable assessment of between-model and between-image transfer.

Distinct visual embedding mechanisms make the influence of perturbations on final token embeddings difficult to isolate. Targeted transfer is substantially stronger between shared-encoder models such as BLIP-2 and InstructBLIP, but varies drastically across heterogeneous encoders. Therefore, direct targeted-success-rate comparisons across such architectures are imprudent.

Decoder-only LLaVA and OpenFlamingo introduce visual features late, through cross-attention or token concatenation, as auxiliary memory for causal generation. As a result, visual influence fades as decoding proceeds, Awadalla et al. (2023), Liu et al. (2023). In contrast, fusion-based BLIP-2, Li et al. (2023b), and InstructBLIP, Dai et al. (2023), integrate visual features early and deeply into input embeddings, retaining visual grounding throughout generation. Targeted attacks are therefore inherently more feasible on the latter models, Dai et al. (2023).

Context Injection Attack is excluded from transfer comparisons after exhibiting negligible transfer in experiments. Its per-image visual-token manipulation embeds target concepts in image-dependent patterns. Because gradients and token spaces differ vastly across images and concepts, the input/target-image token-space transformations are non-transferable. Context Injection Attack's dependence on visual/textual encoder structure also prevents cross-model transfer across the structural differences among BLIP2, LLaVA, Flamingo, and other LVLMs.

## 6 Results

Five state-of-the-art VLMs were evaluated: OpenFlamingo, BLIP-2, InstructBLIP, and LLaVA. The Flamingo implementation is publicly accessible OpenFlamingo-9B, Awadalla et al. (2023), whose performance is comparable to the original model.

BLIP-2's two-stage fusion first extracts features with a frozen CLIP image encoder, then processes those features through a Querying Transformer, Li et al. (2023b). OPT-2.7b provides flexible language generation and efficient task adaptation. InstructBLIP adds instruction tuning to this fusion architecture, Dai et al. (2023), and uses Vicuna-7b for task-specific instruction following and finer output control through crafted prompts.

LLaVA, Liu et al. (2023), connects pretrained CLIP ViT-L to Vicuna-13B through a lightweight projection. Its decoder-only generation supplies visual features directly as special tokens in the prompt context, unlike the explicit pre-generation visual–textual alignment of fusion models. Thus, OpenFlamingo and LLaVA represent decoder-only architectures, whereas BLIP-2 and InstructBLIP represent fusion-based architectures. This distinction significantly affects targeted adversarial effectiveness across prompts.

### 6.1 Claim 1: Q7X achieves N6Z across target texts [Reproduced]

Flamingo-generated CroPA examples transfer across held-out prompts in VQA, Classification, and Captioning. The evaluation reproduces the central cross-prompt transferability claim: Table 1's CroPA rows report nonzero attack success rate across all tasks and diverse target prompts.

#### 6.1.1 Experimental Details

The scored task families are general visual questioning (VQAgeneral, A7L), specific visual questioning (VQAspecific, T4B), image-label prediction (Classification, G9M), and image description (Captioning, D5R). The two questioning labels distinguish how specifically a prompt pertains to a particular image. The aggregate column is Overall (C8H).

Training follows Luo et al. (2024). Evaluation prompts differ from training prompts and span a broad range of semantic meanings, so the Flamingo test measures targeted attack success rate on held-out prompts. Appendix D.1 gives details for all reproduced experiments, and Appendix E lists the prompts.

#### 6.1.2 Comparative Analysis

CroPA generally agrees with the claimed transfer behavior. However, CroPA consistently underperforms on Classification and Captioning, especially for the abstract target texts “too late” and “not sure,” which lack significant visual semantics. For example, Table 1's CroPA row for “not sure” reports Classification success of 0.2420 and Captioning success of 0.0010, while the CroPA row for “too late” reports 0.7010 and 0.1790, respectively. Nevertheless, successful deception persists across diverse targets, as shown by the nonzero values in all CroPA rows.

In Table 1, R4K is Single-P, V9D is Multi-P, and Q7X is CroPA. The task columns are VQAgeneral (A7L), VQAspecific (T4B), Classification (G9M), Captioning (D5R), and Overall (C8H).

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

At equal prompt counts, CroPA consistently exceeds the baselines, as shown in Figure 4. Additional prompts do not allow baseline saturation to overtake CroPA. Increasing prompt counts raises attack success rate for all methods; however, Single-P intrinsically uses one prompt, so count variation is redundant for Single-P.

The target “unknown” is chosen because it avoids high-frequency responses common in vision–language tasks. This experiment accepts that rationale from the original authors. Prompt-count gains saturate rapidly, consistent with Luo et al. (2024).

**Figure 4.** Targeted H2Q against prompt count on Flamingo, holding attack iterations equal; Q7X significantly outperforms baselines. The target is “unknown.” R4K is the lone star.

### 6.3 Claim 3: Q7X converges towards higher H2Q as iterations increase [Reproduced]

Taskwise convergence is unstable despite exact reuse of the authors' training/evaluation parameters and code, as shown in Figure 5. The original reported convergence metrics, reproduced in Appendix A.2, differ from these results and showed convergence with more prompts. However, across tasks jointly, convergence with increasing iterations follows the original aggregate trend, as shown in Figure 6.

The original authors did not specify the model for this scenario. The reproduced curves use OpenFlamingo and target “unknown.”

- (a) VQA
- (b) VQA-Specific
- (c) G9M
- (d) D5R

Here VQA-Specific denotes specific visual questioning, G9M denotes Classification, and D5R denotes Captioning. The methods compared are Single-P (R4K), Multi-P (V9D), and CroPA (Q7X).

**Figure 5.** R4K, V9D, and Q7X compared across categories.

**Figure 6.** Aggregate convergence and Q7X comparison.

### 6.4 Claim 4: Q7X outperforms baselines under few-shot settings [Reproduced]

Flamingo supports extra images as in-context examples for task adaptation. Following Luo et al. (2024), perturbations are optimized at 0-shot, with no extra images, and evaluated at 2-shot, with two extra images. Thus, the evaluation context differs from the optimization context.

When context widens through additional examples, CroPA significantly exceeds the baseline on most tasks. In Table 2's 2-shot rows, CroPA has Overall success of 0.6635, compared with Multi-P's 0.5730. These results support the min–max objective's intended robustness to an expanded prompt space and the consequent context changes.

In Table 2, V9D is Multi-P and Q7X is CroPA. The task columns are VQAgeneral (A7L), VQAspecific (T4B), Classification (G9M), Captioning (D5R), and Overall (C8H).

| Setting | Method | A7L | T4B | G9M | D5R | C8H |
|---|---|---:|---:|---:|---:|---:|
| shot=0 | V9D | 0.7240 | 0.8740 | 0.5550 | 0.2850 | 0.6095 |
| | Q7X | 0.9680 | 0.9880 | 0.7070 | 0.4200 | 0.7708 |
| shot=2 | V9D | 0.7860 | 0.9420 | 0.5630 | 0.0010 | 0.5730 |
| | Q7X | 0.9440 | 0.9940 | 0.6860 | 0.0300 | 0.6635 |

**Table 2.** Targeted H2Q with and without visual in-context learning; shot counts added images.

Generation-condition shifts make attacks harder, Luo et al. (2024), because contextual awareness moves the correct prompt space away from target-text space. Zhao et al. (2023), Qi et al. (2023) verify that context-specific attacks such as CroPA lose effectiveness when context changes.

### 6.5 Results Beyond the Original Paper

The extended experiments incorporate initialization, attention-value guidance, and input-diversity modifications into the original framework.

#### 6.5.1 Noise Initialization

The integrated initialization variant is CroPA+Init (coded B8N). CroPA+Init is reported to outperform CroPA and Multi-P consistently across every metric, with gains in aggregate accuracy, adversarial robustness, and cross-prompt transferability. The targetwise measurements appear in Table 3, and the average measurements appear in Table 4. Semantic initialization is also reported to require fewer optimization steps, as examined in Figure 7. Thus, semantic initialization is presented as a robust, adaptable alternative to random initialization rather than a change to the perturbation budget.

In Table 3, V9D is Multi-P, Q7X is CroPA, and B8N is CroPA+Init. The task columns are VQAgeneral (A7L), VQAspecific (T4B), Classification (G9M), Captioning (D5R), and Overall (C8H).

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

Semantic starting points are deliberately closer to meaningful adversarial directions. Zhang et al. (2025) find that min–max problems are substantially more initialization-sensitive than ordinary minimization and often need carefully designed starts for stable convergence. This finding is relevant because CroPA resembles a saddle-point problem: prompt perturbations continually change the optimization landscape.

More frequent image updates than prompt updates create asymmetry and amplify the importance of the starting point. Chen et al. (2019) find local cycling in alternating-gradient min–max methods and find that convergence depends strongly on initial proximity to adversarially robust regions. Wu et al. (2020) find that semantic-prior perturbations transfer better across models than random starts and yield substantial success-rate improvements. The experimental observations align with that finding, particularly for the more complex bi-level optimization paths.

In Table 4, V9D is Multi-P, Q7X is CroPA, E3W is Context Injection Attack, and B8N is CroPA+Init. The task columns are VQAgeneral (A7L), VQAspecific (T4B), Classification (G9M), Captioning (D5R), and Overall (C8H).

| Method | A7L | T4B | G9M | D5R | C8H |
|---|---:|---:|---:|---:|---:|
| V9D | 0.7053 | 0.8807 | 0.7001 | 0.5553 | 0.7104 |
| Q7X | 0.8723 | 0.9457 | 0.7518 | 0.6138 | 0.7958 |
| E3W | 0.2985 | 0.2281 | 0.4857 | 0.4687 | 0.370 |
| B8N | 0.9145 | 0.9342 | 0.8982 | 0.9464 | 0.9209 |

**Table 4.** Average targeted H2Q on BLIP-2 across visual-questioning, G9M, and D5R tasks. Best task values are highlighted in bold. E3W, Yang et al. (2024), is the additional baseline.

Gradient attacks ordinarily acquire target semantics and theoretically should not require semantic initialization. However, min–max complexity weakens that alignment. Injecting target semantics at initialization is therefore described as necessary to reinforce alignment and assist convergence.

Feature uptake between target text and learned noise is quantified using the CLIP score (coded Y2K). This comparison assesses the extent to which perturbations acquire target attributes, rather than measuring only task success. For target text “Bomb,” the base CroPA perturbation has a CLIP score of 21.80, while the CroPA+Init perturbation has a score of 23.09.

| Method | Y2K |
|---|---:|
| Q7X | 21.80 |
| B8N | 23.09 |

Target text: “Bomb”.

**Figure 7.** (a) Aggregate convergence of Q7X and B8N. (b) Learned-perturbation/target feature similarity quantified by Y2K.

#### 6.5.2 Investigating K3R and Image Augmentation

Learning across images does not yield effective universality for unmodified CroPA, because prompt-directed signals remain tightly linked to the originating image's low-level features. Table 5's unmodified CroPA row reports Overall untargeted cross-image attack success of 0.4355. This result weakens the original claim that learning perturbations across images would establish cross-image transferability.

Both input-mixing approaches give measurable gains: Table 5 reports Overall success of 0.7055 for CroPA with SCMix and 0.5565 for CroPA with CutMix, compared with 0.4355 for unmodified CroPA. Nevertheless, the gap between cross-image methods and CroPA applied to a single image remains a fundamental limitation. CroPA excels at prompt robustness, not effective image-transferring adversarial attacks.

CutMix is believed to outperform SCMix for adversarial transfer because region-based mixing retains distinct, non-overlapping spatial areas. Such regions are expected to encourage modular, independently manipulable features and greater cross-input generalization.

In Table 5, Q7X is CroPA, J6T is SCMix, and U8G is CutMix. VQA denotes visual question answering; the other columns are VQAspecific (T4B), Classification (G9M), Captioning (D5R), and Overall (C8H).

| Method | VQA | T4B | G9M | D5R | C8H |
|---|---:|---:|---:|---:|---:|
| Q7X (Base) | 0.3500 | 0.4920 | 0.4000 | 0.5000 | 0.4355 |
| Q7X with J6T | 0.6560 | 0.7160 | 0.5000 | 0.9500 | 0.7055 |
| Q7X with U8G | 0.4520 | 0.7240 | 0.5000 | 0.5500 | 0.5565 |

**Table 5.** Untargeted K3R H2Q, best across all iterations. Unmixed Q7X peaks at 1700 iterations; J6T and U8G exceed it significantly earlier, at 1500 and 1600 respectively. Experimental parameters: Appendix D.3.

#### 6.5.3 Guiding perturbations via Target Value Vectors

CroPA+D-UAP greatly improves the existing algorithm across tasks, especially general visual questioning and Captioning. In Table 6, general-questioning success rises from CroPA's 0.8176 to CroPA+D-UAP's 0.9420, while Captioning success rises from 0.6960 to 0.9810.

The target “Bomb” is harmful text expected inherently to activate guardrails. The improvement is maintained for such worse expected harmful-word cases and vulnerable scenarios, as shown by the “Bomb” results in Tables 6 and 7.

In Tables 6 and 7, V9D is Multi-P, Q7X is CroPA, E3W is Context Injection Attack, and P2F is CroPA+D-UAP. The task columns are VQAgeneral (A7L), VQAspecific (T4B), Classification (G9M), Captioning (D5R), and Overall (C8H). In the captions, X9C denotes Doubly-UAP guidance, S3D denotes Projected Gradient Descent, and H2Q denotes attack success rate.

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

Value-vector guidance significantly improves vanilla CroPA cross-model transfer, especially for Classification and Captioning, with modest VQA gain. For example, Table 8's BLIP2-to-InstructBLIP rows report Classification success of 0.6720 for CroPA and 0.8800 for CroPA+D-UAP, and Captioning success of 0.6150 and 0.8880, respectively. VQA prompts occupy a more complex embedding space than the smaller output spans of targeted Classification and Captioning.

Encoder dependence verifies the earlier limitation hypothesis: BLIP2 perturbations are passably transferable to same-encoder InstructBLIP and fail to transfer to OpenFlamingo. The corresponding Overall CroPA+D-UAP values in Table 8 are 0.6850 for BLIP2 to InstructBLIP and 0.2335 for BLIP2 to Flamingo. Training on Flamingo and applying the perturbations to the other models likewise gives insignificant attack success rate; see Table D.4.5 in Appendix D.4.5. Appendix D.4 gives the experimental details.

In Table 8, V9D is Multi-P, Q7X is CroPA, and P2F is CroPA+D-UAP. The task columns are VQAgeneral (A7L), VQAspecific (T4B), Classification (G9M), Captioning (D5R), and Overall (C8H).

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

For a given target text, vision-encoder value vectors remain approximately consistent across input images because the value vectors reflect encoder architecture and pretraining. This consistency may constrain perturbation diversity and the generation of highly varied adversarial examples. Consequently, target-text choice strongly influences outcomes relative to input-image-specific features.

## 7 Discussion

### 7.1 Observations with reproducibility

Reproduction used the authors' public repository and original model configurations to support faithful baseline comparison. The implementation was generally well documented. Nevertheless, the BLIP-2 and InstructBLIP code produced multiple runtime errors, requiring substantial debugging and adjustment before the code would run.

### 7.2 Noise Initialization findings

The initialization hypothesis includes reduced cross-modal as well as cross-prompt transfer under semantic ignorance. Experiments across all evaluated benchmarks are described as significantly favoring semantic noise over CroPA and random-initialized baselines; Tables 3 and 4 report the targetwise and average attack success rates.

As expected of gradients, CroPA perturbations already acquire target context. Initialization produces a modest CLIP-score increment, from 21.80 for CroPA to 23.09 for CroPA+Init in Figure 7(b), rather than a significant semantic-feature upgrade. This modest increment is interpreted as a subtle shift closer to adversarially vulnerable regions. The convergence comparison before and after initialization in Figure 7(a) aligns with the hypothesis that a starting point nearer meaningful adversarial directions converges faster.

### 7.3 Evaluation of Transferability Experiments

#### 7.3.1 K3R

SCMix improves cross-image transferability particularly for VQA, specific visual questioning, and Captioning. Table 5 reports increases from unmodified CroPA's 0.3500, 0.4920, and 0.5000 to SCMix's 0.6560, 0.7160, and 0.9500, respectively. CutMix also improves the base image-transfer version, with Overall success of 0.5565 compared with unmodified CroPA's 0.4355.

The proposed account favors SCMix's explicit spatial/scale invariance, produced by random cropping and resizing of the same image followed by soft mixing with another image. Under this account, semantically aligned features become less sensitive to position and scale, easing cross-input generalization.

Both augmentations preserve original semantic information while adding image diversity. Their proposed benefit is learning cross-image features. However, augmentation alone does not eliminate the cross-image limitation. Therefore, further work should combine augmentation with more complex, layered methods, including latent-representation feature-alignment attacks or direct latent-space perturbation.

#### 7.3.2 F8V

Doubly-UAP guidance is described as a significant advance that is robust to harmful target text and guardrail activation, as examined with “Bomb” in Tables 6 and 7. However, encoder-specific dependence remains the obstacle to encoder-agnostic attacks because encoder dependence narrows perturbation diversity and transfer across heterogeneous architectures.

Despite that limitation, the framework is presented as progress toward universal attacks effective across prompts and images and toward improved VLM generalizability and robustness. Context Injection Attack cannot supply comparable transfer results and is inherently non-transferable, for the image-dependent and encoder-dependent reasons given in Section 5.4.3.

### 7.4 Trade-offs Between Different Types of Adversarial Transferability

#### 7.4.1 N6Z vs. F8V

Cross-prompt and cross-model transferability are in tension. Competing text embeddings implicitly anchor original CroPA perturbations to the target architecture; consequently, moving attacks between models causes a significant attack-success-rate drop.

Doubly-UAP targets potentially generalizable encoder components rather than model-specific features. Therefore, Doubly-UAP guidance improves related-architecture transfer, such as BLIP-2 to InstructBLIP, but offers only modest gains across fundamentally different architectures, such as BLIP-2 to Flamingo. Table 8's Overall results rise from 0.5065 to 0.6850 for BLIP2 to InstructBLIP and from 0.1435 to 0.2335 for BLIP2 to Flamingo. Thus, simultaneously optimizing prompt diversity and model diversity is inherently tense.

#### 7.4.2 N6Z vs. K3R

Image-specific coupling serves strong cross-prompt transferability but fundamentally limits cross-image transferability. Therefore, cross-prompt and cross-image transferability are competing optimization objectives. Augmentation diversifies optimization and yields promising improvements, as reported in Table 5, without resolving that fundamental tension.

#### 7.4.3 Role of Initialization in Transferability Balance

Semantic anchoring raises attack success rate without sacrificing cross-prompt transferability, indicating that starting-point selection substantially influences the achievable transfer balance. For example, Table 4 reports Overall targeted success of 0.7958 for CroPA and 0.9209 for CroPA+Init.

Unlike modifications to the optimization process, better initialization allows convergence toward more general adversarial directions that remain effective across prompts. Even then, optimizing a perturbation for one transfer dimension inevitably sacrifices effectiveness in other dimensions.

#### 7.4.4 Implications for Future Research

No single methodology can maximize all forms of transfer simultaneously. Therefore, controlled balancing may require multi-objective optimization or architecturally aware attack designs. Pushing individual dimensions is insufficient for a comprehensive solution without fundamentally rethinking perturbation interactions with model architecture and data representation.

Cross-prompt transferability remains underexplored, with little literature on prompt-induced vulnerability. The resulting understanding gap matters for secure deployment.

### 7.5 Broader Impact

Examples with high cross-prompt transferability can elicit malicious or misleading outputs from harmless queries. Consequently, these examples can enable misinformation or subversion of VLM-dependent content-generation and decision systems.

In contrast, imperceptible perturbations also offer privacy protection. Embedded in personal images, perturbations can force predetermined nonsensitive text and thwart unauthorized extraction of confidential information. Thus, perturbations can provide a proactive security mechanism as visual data becomes increasingly exploitable.

The initialization and loss refinements increase attack success rate and cross-model/cross-image generalizability. These refinements therefore strengthen cross-prompt attacks while opening paths toward more resilient defenses. Deeper understanding is necessary for robust VLMs, effective countermeasures, user-data protection, and the integrity of outputs.

### 7.6 Limitations and Future Work

Computational scarcity repeatedly hindered successful key-result replication and extensive ablations. Attack training and evaluation on large Flamingo and BLIP2 models require substantial GPU memory and processing. Consequently, available resources limited experimental scale and depth, extensive hyperparameter tuning, and transfer testing over wider architecture ranges.

A planned ensemble-model transfer investigation could have deepened the generalizability analysis. However, the ensemble investigation exceeded available memory and compute and was infeasible on the existing infrastructure. Future work includes encoder-agnostic cross-model transferability, stronger cross-image transferability, and query-based optimization for practical applicability.

### 7.7 Communication With Original Authors

No direct communication with the original authors was established during replication. The GitHub issue concerning BLIP-2 and InstructBLIP remains unresolved at writing.
