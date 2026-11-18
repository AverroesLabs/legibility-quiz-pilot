# Revisiting CroPA: A Reproducibility Study and Enhancements for Cross-Prompt Adversarial Transferability in Vision-Language Models

| Author | Email | Affiliation | Author notation |
|---|---|---|---|
| Atharv Mittal | atharv_m@mfs.iitr.ac.in | Mehta Family School of Data Science and Artificial Intelligence; Indian Institute of Technology, Roorkee | ††thanks: All authors contributed equally |
| Agam Pandey | agam_p@ce.iitr.ac.in | Department of Civil Engineering; Indian Institute of Technology, Roorkee | 11footnotemark: 1 |
| Amritanshu Tiwari | amritanshu_t@mfs.iitr.ac.in | Mehta Family School of Data Science and Artificial Intelligence; Indian Institute of Technology, Roorkee | 11footnotemark: 1 |
| Sukrit Jindal | sukrit_j@mfs.iitr.ac.in | Mehta Family School of Data Science and Artificial Intelligence; Indian Institute of Technology, Roorkee | 11footnotemark: 1 |
| Swadesh Swain | swadesh_s@ece.iitr.ac.in | Department of Electronics and Communication; Indian Institute of Technology, Roorkee | 11footnotemark: 1 |

###### Abstract

This study reproduces “An Image is Worth 1000 Lies: Adversarial Transferability Across Prompts on Vision-Language Models.” It validates the Cross-Prompt Attack, also called CroPA and labeled Q7X here. Q7X outperforms existing baselines in cross-prompt transferability, labeled N6Z: an attack remains effective when the text prompt changes. Semantic meaning and target-word frequency do not determine this transferability.

We extend the replication through initialization, shared perturbations across images, and an attention-directed loss. Initialization sets the attack's starting point. Perturbations are small changes to an input, and a loss measures the objective that the attack optimizes. Attention-directed loss targets the model components that control how image information interacts.

We evaluate Flamingo, BLIP-2, InstructBLIP, and extended LLaVA experiments. These vision-language models, or VLMs, process images and text. The evaluation validates the original results, and the enhancements consistently increase attack effectiveness. The resulting framework generates more robust, transferable adversarial examples: modified inputs designed to mislead a model. These findings bear on real-world VLM security.

Code: https://github.com/Swadesh06/Revisting_CroPA

Reviewed on OpenReview: https://openreview.net/forum?id=5L90cl0xtf

## 1 Introduction

Subtle image perturbations can make VLMs produce incorrect or harmful outputs. Text input adds attack routes beyond those of traditional vision models. Large VLMs transformed computer vision through image classification, labeled G9M; image captioning, labeled D5R; and visual question answering, or VQA. Classification predicts an image label, captioning describes an image, and VQA answers questions about an image. These capabilities enabled deeper study of how visual and language information interact. Zhao et al. (2023), Qi et al. (2023), Zhang et al. (2022a), and Carlini et al. (2024) demonstrated vulnerabilities in these models.

Three settings distinguish adversarial transfer:

- **Cross-model transferability, labeled F8V:** adversarial examples remain effective across model architectures. Researchers commonly call this simply transferability.
- **Cross-image transferability, labeled K3R:** a perturbation remains effective across multiple images.
- **Cross-prompt transferability, labeled N6Z:** a perturbation remains effective across different text prompts.

Cross-image transfer allows shared perturbations, called Universal Adversarial Perturbations, or UAPs, and labeled Z4P. It does not require a separately optimized perturbation for each image.

Luo et al. (2024) introduced N6Z and competing learnable prompts: prompts whose perturbations are optimized against the image attack. Their image attacks remain effective regardless of text input across G9M, D5R, and VQA. Earlier work treated visual and textual adversarial perturbations independently.

Zhao et al. (2023) studied BLIP and BLIP-2 through query-based and transfer-based black-box attacks. Black-box attacks do not access the target model's internal parameters. Schlarmann & Hein (2023) studied targeted and untargeted white-box attacks, which do access model internals. Targeted attacks seek a specified output; untargeted attacks seek an incorrect output. Qi et al. (2023) showed that adversarial images can elicit harmful instruction execution. Tu et al. (2023) systematically evaluated attacks on both image and text inputs.

Our replication tests four properties: superiority across target texts, rapid saturation as prompts are added, robustness to added in-context examples, and convergence over attack iterations. In-context examples are extra examples supplied in the input. Convergence describes how optimization settles as updates continue. We also analyze each task separately, beyond the original aggregate analysis.

The extensions examine initialization that improves both Attack Success Rate, or ASR, labeled H2Q, and transferability. They also examine shared perturbations built on Q7X, with image augmentation to promote transfer. Image augmentation creates modified versions of input images. A further extension targets value vectors, the attention components that carry image-patch information. Attention components govern interactions between patches. We propose that targeting them can improve generalization and H2Q. Our analyses investigate how these enhancements work and what they reveal about VLM adversarial vulnerabilities.

## 2 Related Works

Szegedy et al. (2013) and Goodfellow et al. (2014) introduced adversarial attacks on deep neural networks. Applying these attacks to VLMs creates further challenges and opportunities.

### 2.1 Adversarial Transferability

Examples designed to attack classification also affect object detection and other visual tasks. This behavior establishes cross-task transferability alongside F8V and K3R: Naseer et al. (2019b); Naseer et al. (2019a); Lu et al. (2020); Salzmann et al. (2021). These multiple forms of transfer raise questions about the robustness of modern machine-learning systems.

Liu et al. (2017) and Tramèr et al. (2017) established that attacks crafted on one model often deceive other models, including models with different architectures. Moosavi-Dezfooli et al. (2017) and Mopuri et al. (2017) established perturbations that generalize across images and introduced Z4P.

### 2.2 Adversarial Attacks on Vision-Language Models

Early task-specific D5R attacks manipulated model outputs: Xu et al. (2019); Zhang et al. (2020); Aafaq et al. (2021); Chen et al. (2017). Targeted VQA attacks misled attention mechanisms: Xu et al. (2018); Kaushik et al. (2021); Kovatchev et al. (2022); Li et al. (2021); Sheng et al. (2021); Zhang et al. (2022b).

These attacks mainly addressed lightweight systems that combined convolutional neural networks, or CNNs, with recurrent neural networks, or RNNs. CNNs process spatial patterns, while RNNs process sequences. These systems lacked in-context learning, which limits the attacks' relevance to contemporary VLMs. Zhao et al. (2023) studied black-box attacks on modern large VLMs, BLIP and BLIP-2, but emphasized F8V rather than N6Z.

### 2.3 Cross-Prompt Adversarial Transferability

Researchers have not yet established credible benchmarks or baselines for this relatively new concept. Yang et al. (2024) nevertheless provides an additional experimental comparator: the Context Injection Attack, or CIA, labeled E3W and also called Contextual Injection Attack. E3W uses gradient-based perturbations to inject target tokens into visual and textual contexts. A gradient indicates how a loss changes with an input; tokens are the units of text or image representation that a model processes. E3W differs from the competing-prompt optimization that Q7X pioneered in Luo et al. (2024).

Prompts let large VLMs switch tasks dynamically, unlike traditional vision models with fixed tasks: Li et al. (2023b); Awadalla et al. (2023); Li et al. (2023b); Zhu et al. (2023); Liu et al. (2023). Attacks that remain valid as prompts change create another attack surface. Making the two modalities compete, rather than manipulating them independently, establishes a distinct way to evaluate VLM robustness.

### 2.4 Enhancing Adversarial Transferability

Gradient misalignment—differences in gradient directions—limits transferability. This problem has motivated optimization strategies in Li et al. (2023a) and Wang et al. (2023). Input diversity in Xie et al. (2019) and augmentation in Wang et al. (2021) also improve attack robustness. These findings inform the Q7X enhancements across multiple transfer dimensions.

## 3 Scope of reproducibility

We follow the experimental procedures of Luo et al. (2024) to reproduce their work faithfully. We test these named claims:

- Claim 1: Q7X achieves N6Z across target texts.
- Claim 2: Q7X has the best aggregate performance across prompt counts.
- Claim 3: Q7X converges toward higher H2Q with increasing iterations.
- Claim 4: Q7X outperforms baselines in few-shot settings.

The extended scope includes F8V, K3R, and comparisons with other N6Z methods. We test whether Q7X-generated images consistently deceive different VLMs under varied input prompts and model parameters. This scope addresses the broader robustness and versatility of adversarial attacks, beyond validating the original improvement in prompt transfer.

## 4 Methodology

### 4.1 Problem Formulation

A targeted attack succeeds only when the model generates the exact target text. An untargeted attack succeeds when the output differs from the clean-image prediction. H2Q measures effectiveness in the reproduced targeted and non-targeted scenarios. The attacks use a language-modeling objective over prompt instances, rather than a vision-only objective. Language-modeling loss measures the model's fit to an output text.

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

Here, \(f\) processes visual input \(x_v\) and text input \(x_t\). The image perturbation is \(\delta_v\), and \(L\) is the language-modeling loss. The sum covers \(k\) prompts. In the targeted scenario, \(T\) is predetermined text generated independently of the input prompt. The untargeted scenario maximizes the difference between clean and adversarial outputs.

The human-imperceptibility bound is \(\|\delta_v\|_p\leq\epsilon_v\). The \(p\)-norm measures perturbation size, and \(\epsilon_v\) sets its upper limit. This bound adapts established adversarial practice to real-world multimodal use, Carlini et al. (2024). The formulation extends the neural-network vulnerability setting of Goodfellow et al. (2014) to changing prompts, Luo et al. (2024).

### 4.2 Baseline Approaches

Because established N6Z baselines were absent, Luo et al. (2024) introduced two reference attacks. Single-P, labeled R4K, optimizes a perturbation with one prompt. Multi-P, labeled V9D, uses a collection of prompts. Their untargeted forms reverse the targeted objectives: they maximize the difference between clean and perturbed outputs instead of target-text likelihood.

\[
\min_{\delta_v}L\bigl(f(x_v+\delta_v,x_t),T\bigr)
\tag{3}
\]

\[
\min_{\delta_v}\sum_{i=1}^{k}
L\bigl(f(x_v+\delta_v,x_t^i),T\bigr)
\tag{4}
\]

The prompt collection is \(X_t=\{x_t^1,x_t^2,\ldots,x_t^k\}\). The one-prompt case is the simplest baseline.

**Figure 1.** Framework of Luo et al. (2024). Image perturbations \(\delta_v\) and prompt perturbations \(\delta_t\) oppose each other: they minimize and maximize language-modeling loss, respectively. Solid arrows show the forward pass through the vision encoder and language model. The vision encoder converts images into model representations. Dashed arrows show backpropagation, which computes gradients, with configurable perturbation-update frequencies. Asterisks mark fixed model parameters; \(\oplus\) means elementwise addition.

### 4.3 Cross Prompt Attack

Evaluation discards prompt perturbations and applies only the optimized image perturbation. The image perturbation must therefore carry the transferability itself; modified prompts cannot supply it.

The competing-prompt method of Luo et al. (2024), Q7X, jointly optimizes \(\delta_v\) and \(\delta_t\). Their objectives oppose rather than support each other. This competition aims to force stronger image transferability across text prompts.

\[
\min_{\delta_v}\max_{\delta_t}
L\bigl(f(x_v+\delta_v,x_t+\delta_t),T\bigr)
\tag{5}
\]

The implementation uses iterative image-gradient descent and prompt-gradient ascent. Descent reduces a loss, while ascent increases it. The method uses Projected Gradient Descent, or PGD, labeled S3D, following Madry et al. (2017). Projection keeps updates within the allowed perturbation region. Image updates occur \(N\) times per prompt update. Thus, the two modalities use a configurable update frequency rather than simultaneous updates at equal frequency.

**Algorithm 1. Q7X**

**Input:** model \(f\); target text \(T\); vision input \(x_v\); prompt set \(X_t\); perturbation size \(\epsilon\); step sizes \(\alpha_1,\alpha_2\); iteration count \(K\); adversarial-prompt update interval \(N\).  
**Output:** adversarial example \(x'_v\).

```text
1:  Initialize x'_v = x_v
2:  for step = 1 to K do
3:      Uniformly sample prompt x_t^i from X_t
4:      if x_t^i is not initialized then
5:          Initialize x'_t^i = x_t^i
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

Here, `sign` takes each gradient component's sign, `mod` gives the remainder after division, and `Clip` enforces the perturbation bound.

## 5 Additional Investigative Experiments

Q7X performs poorly on G9M and D5R and has limited F8V. Its S3D optimization effectively raises H2Q on a single model. However, reliance on that model's loss feedback limits transferability. This is a recognized limitation of gradient attacks: Qin et al. (2024); Li et al. (2024); Wang et al. (2023); Li et al. (2023a). The original authors acknowledge it, and Table 9 in Appendix A.1 verifies it.

Our extensions address inconsistent performance across tasks and different transfer dimensions. K3R could remove the need to train separately for each image. F8V could provide even greater efficiency. Transfer exploration is not a core proposal of Luo et al. (2024). The table below records resource costs and quantitative characterizations separately from the related comparisons. A graphics processing unit, or GPU, performs the computation; video random-access memory, or VRAM, is its working memory.

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

In this table, P2F denotes CroPA+D-UAP, also called re-CroPA, the value-guided method introduced in Section 5.2.1. A7L denotes general visual questioning. J6T denotes SCMix, a self- and cross-mixing augmentation. U8G denotes CutMix, an augmentation that replaces image regions. Sections 5.3 and 6.1.1 describe these terms further.

### 5.1 Noise Initialization via Vision Encoding Optimization

Noise Initialization via Vision Encoding Optimization, labeled W5J, replaces random noise that ignores the target with target-image feature alignment. The method uses a diffusion model, which synthesizes an image, to anchor the starting perturbation. It preserves imperceptibility while supplying adversarial directions with semantic meaning from the outset. Conventional initialization neither guarantees such directions nor avoids reduced H2Q and N6Z.

Like many gradient attacks, Q7X uses random initialization. Ignoring target meaning is associated with instability and slower convergence, especially in VLMs that require alignment with the meaning of a prompt. Q7X uses bi-level optimization: it optimizes image and prompt perturbations through opposing objectives. This framework is empirically sensitive to initialization, as Section 6.5.1, Figure 7, and Table 3 show. We hypothesize that semantically informed perturbations improve attack effectiveness over purely random perturbations under the same imperceptibility requirement.

A state-of-the-art diffusion model synthesizes a target image. The image's encoder representation anchors the initial noise to the objective. An encoder-output error loss enforces semantic similarity while preserving adversarial effectiveness. This error measure is Mean Squared Error, or MSE, labeled L6V: the mean squared difference between representations. It supplies an optimization criterion instead of a purely random starting direction for later attack updates.

We present the method as improving adversarial robustness, strengthening perturbations and N6Z, and reducing the steps needed for convergence.

#### 5.1.1 Noise Initialization

Preliminary experiments on BLIP-2 and InstructBLIP favor semantic noise over random noise. Diffusion-based anchoring improves alignment in the encoder's representation space while preserving the adversarial image's visual fidelity and strict perturbation budget. Its added computation per image is limited to the single-GPU overhead stated separately above.

#### 5.1.2 Diffusion-Based Target Synthesis

\[
\mathbf{x}_{\mathrm{target}}
=\mathcal{D}(T,\mathbf{z};\theta_{\mathrm{SDXL}})
\tag{6}
\]

Stable Diffusion XL, abbreviated SDXL, supplies \(\mathcal D\). The method samples \(\mathbf z\) from a Gaussian distribution, and \(\theta_{\mathrm{SDXL}}\) contains pretrained diffusion weights. The generated image, \(\mathbf{x}_{\mathrm{target}}\), embodies the semantic attributes of prompt \(T\) and serves as the initialization reference.

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

Here, \(f_v\) is the VLM's vision component. The bound \(\epsilon\) sets the maximum allowable, imperceptible \(\ell_\infty\) perturbation. The \(\ell_\infty\) norm measures the largest absolute component of the perturbation; the squared \(\ell_2\) expression measures squared distance between representations.

Matching the target image's encoder representation places the adversarial example in a semantically meaningful neighborhood of the target prompt representation. Algorithm 2 in Appendix B formalizes the method.

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

After initialization, these refinement updates gradually align the example with the target embedding, or numerical representation. At iteration \(t\), \(\alpha\) is the step size. The operator \(\Pi_{\mathcal B_\epsilon(\mathbf x)}\) projects the update into the allowed \(\ell_\infty\) ball around the original image \(\mathbf x\). This ball is the set of inputs within the perturbation bound.

### 5.2 Guiding perturbations via Target Value Vectors and evaluating F8V

We target vulnerable value vectors in the middle-to-late vision layers. Q7X does not account for the architecture of the encoder's core representation system. It treats vision and language as equally vulnerable, overlooking modality-specific weaknesses despite the encoder's dominant role in grounding meaning in images.

Generic S3D output manipulation is less structured and less targeted than the proposed alternative. We seek improvements in attack strength, transferability, interpretability, and computational feasibility.

Kim et al. (2024) proposed Doubly-Universal Adversarial Perturbation, also called Doubly-UAP or D-UAP and labeled X9C. This component-selection method identifies the attention components that most affect VLM performance when constructing Z4P. We adapt its component loss by minimizing, rather than maximizing, losses between the value vectors of \(x_i\) and \(T_i\). This retains a reference associated with the target text.

Cui et al. (2023) and Wang et al. (2024) show that perturbing vision-output embeddings impairs visual perception in large vision-language models, or LVLMs, and misleads their language responses. LVLMs commonly use similar image encoders to produce visual tokens that contain semantic and structural information. These tokens form the interface through which language decoders access and reason about images. A decoder generates the model's output text. We propose targeting encoder internals to exploit this shared interface.

Two components play fundamental roles. Attention weights control focus, relevance, and interaction between patches. Value vectors contain each patch's actual information. We hypothesize that disrupting attention weights damages the model's ability to establish relationships. Perturbing value vectors should damage essential content and its interpretation. Prior analysis identifies middle-to-late-layer value vectors as the most vulnerable.

\[
\delta^*
=
\arg\max_\delta
\frac{1}{|L|}
\sum_{l\in L}
\operatorname{Loss}\bigl(V_l(x),V_l(x+\delta)\bigr)
\tag{10}
\]

In the standard X9C objective, \(L\) is the set of layers, and \(|L|\) counts them. The term \(V_l(x)\) denotes the value vectors for input \(x\) at layer \(l\). The function \(\operatorname{Loss}(\cdot)\) is the loss applied to target vectors.

Our combination retains Q7X's opposing image and text objectives and adds attention-value disruption under the same hyperparameter constraints. Hyperparameters are settings that control optimization. We hypothesize that perturbing value vectors relative to a reference image associated with the target text will improve H2Q. Freezing the text encoder—keeping its parameters fixed—aims to isolate degraded visual processing rather than text artifacts. These changes seek to exploit encoder information, align with target-text meaning, and improve cross-model adversarial robustness.

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

The combined method, P2F, extracts layer-\(l\) value vectors from the input image and the target-associated reference image. It computes their cosine-similarity loss, which compares vector directions. It then jointly minimizes the negative of this loss and the Q7X loss, with suitable scaling. This differs from solely maximizing departure from the original representation.

Here, \(V_i(x+\delta)\) is the perturbed input's value vector at attention head \(i\). Attention heads are separate attention components. The vector \(V_t\) is the reference image's target value vector, \(N\) counts heads, and \(\lambda\) controls the relative importance of the value loss.

We describe the loss as encouraging closeness to the target representation. Combined gradients update \(\delta\), preserving its adversarial character, N6Z, and guided semantic influence. This controlled, interpretable manipulation aims to support robustness and security analysis. Algorithm 3 in Appendix B gives the steps.

X9C's encoder dependence suggests a limitation: transfer gains may be restricted to VLMs whose vision encoders have the same or similar architecture as the model used for Q7X optimization.

**Figure 2.** X9C architecture diagram.

### 5.3 Transferability across images via Image Augmentation

Iterative gradients for each image overfit image-specific features: they adapt too closely to those features. This produces poor K3R and high computational cost. Q7X likewise learns each image's perturbation across prompts.

Luo et al. (2024) describe their method as orthogonal to K3R, meaning that image transfer is a separate concern. They suggest computing perturbations across images, following Moosavi-Dezfooli et al. (2017). We test that claim and another approach drawn from the Z4P literature.

One augmentation, U8G, uses regional replacement rather than pixel blending. It creates an image by inserting rectangular patches cut from another image. It mixes labels in proportion to patch area. Learning from these partial views promotes spatially invariant features—features less dependent on position—and robustness to occlusion and adversarial attacks, Yun et al. (2019).

Fang et al. (2024) introduces Z4P for Vision-Language Pretraining, or VLP, models. This work seeks to reduce gradient-attack overfitting and enable K3R. Zhang et al. (2024) improves global and local optimization across VLP models and tasks through Effective and Transferable Universal Adversarial Attack, or ETU, labeled M7A. Within that framework, self- and cross-mixing augmentation, J6T, increases diversity while preserving semantics and spatial and scale invariance. Scale invariance means reduced sensitivity to image size.

We select J6T for perturbation generation based on empirical results. For the same model, J6T alone yields superior K3R without the authors' additional local utility reinforcement.

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

The first stage self-mixes randomly cropped and resized versions of the base image \(I_1\). The second stage cross-mixes that result with a secondary image, \(I_2\). In Equation 14, \(\beta_1\) is much larger than \(\beta_2\).

Original J6T samples the mixing weight as \(\eta\sim\operatorname{Beta}(\alpha,\alpha)\), with \(\alpha>0\). This implementation fixes \(\eta=0.5\), equivalent to \(\alpha=\infty\). Appendix D.3 justifies this choice.

The weighting keeps \(I_1\)'s semantics dominant. It also keeps \(I_3\) compatible with text \(T_1\), despite the added features from \(I_2\), enriching cross-modal interaction. Variation within an image and blending between images act as regularization against Q7X overfitting. They encourage perturbations aimed at invariant features. The resulting shared perturbation may then apply to unseen images without image-specific optimization.

At inference, when the trained attack is applied, Q7X returns only a perturbed image or, equivalently, its perturbation. Effective K3R would make that output universal both across prompts, in the original N6Z sense, and across images, in the additional sense investigated here.

**Figure 3.** Self-mixing \(I_1\) generates \(X_1,X_2\), which combine into \(I'_1\). External image \(I_2\) mixes with \(I'_1\) to produce \(I_3\).

#### 5.3.2 U8G

\[
\tilde{x}=M\odot x_A+(1-M)\odot x_B
\tag{15}
\]

Here, \(M\in\{0,1\}^{W\times H}\) is a binary mask for a randomly located rectangle in an image of width \(W\) and height \(H\). The symbol \(\odot\) denotes elementwise multiplication. For paired images \(x_A,x_B\), discrete replacement preserves partial views.

J6T softly blends features within and across images; U8G uses a hard mask. During adversarial training, features that persist across regional combinations are expected to direct perturbations toward spatially invariant, higher-level semantics rather than local artifacts. This may improve K3R.

### 5.4 Attack Setup and Threat Models for Transferability Experiments

Model-transfer and image-transfer experiments differ in attacker knowledge, perturbation capability, and evaluation. These conditions define the threat models.

#### 5.4.1 Using X9C to explore F8V for Q7X

The attacker seeks one image perturbation that elicits a specified target text from multiple VLMs, independently of the prompt. The attacker optimizes it on an accessible surrogate model, then deploys it against unknown models.

The attacker has white-box access to the surrogate's gradients and internal representations. The surrogate is BLIP-2 or Flamingo. For the target, the attacker knows only the general architecture family, such as CLIP-based; target parameters are unavailable. CLIP is the shared image-encoder family considered here.

The imperceptibility budget is \(\epsilon=16/255\) in \(L_\infty\). Intra-family testing optimizes on BLIP-2 with OPT-2.7b and evaluates on InstructBLIP with Vicuna-7b. Both use similar CLIP ViT-L/14 encoders, but their language models and fine-tuning objectives differ. Fine-tuning adapts a pretrained model to an objective.

Cross-architecture testing transfers from BLIP-2 to Flamingo and in the reverse direction. This changes the vision encoder, language model, and architecture.

We score each task independently. Success in this evaluation means that the target model fails to generate the correct output from the perturbed image, regardless of the prompt.

#### 5.4.2 Investigating K3R via Augmentation Techniques

The goal is one pattern that causes predetermined target text for any image, regardless of image content or prompt. This would improve efficiency over per-image optimization.

The attacker has a limited training-image set and its corresponding J6T variants. The attacker has white-box access during optimization, but the perturbation must generalize to unseen images. The \(L_\infty\) budget is \(\epsilon=16/255\).

A separate test set contains no training images. Taskwise H2Q counts failures to generate the correct output text on previously unseen images, independently of content or prompt. We hypothesize that controlled self- and cross-mixing captures image-independent model vulnerabilities rather than specific image features.

#### 5.4.3 Evaluation Rationale for Transferability experiments

We present all N6Z and F8V transferability results under untargeted attacks. We also choose non-targeted H2Q as a robust, fair, and comparable measure of transfer between models and between images.

Different visual embedding mechanisms make it hard to isolate a perturbation's effect on final token embeddings. Targeted transfer is substantially stronger between models with shared encoders, such as BLIP-2 and InstructBLIP. However, it varies drastically across heterogeneous encoders. Direct comparisons of targeted H2Q across such architectures are therefore imprudent.

Decoder-only LLaVA and OpenFlamingo introduce visual features late, through cross-attention or token concatenation. Cross-attention lets text processing attend to visual features, while concatenation joins visual and text tokens. These features act as auxiliary memory for causal generation, which generates text from preceding context. Their visual influence fades as decoding continues, Awadalla et al. (2023), Liu et al. (2023).

Fusion-based BLIP-2, Li et al. (2023b), and InstructBLIP, Dai et al. (2023), combine visual features with input embeddings early and deeply. They retain visual grounding throughout generation. Targeted attacks are inherently more feasible on these models, Dai et al. (2023).

We exclude E3W from transfer comparisons because it showed negligible transfer in experiments. Its per-image manipulation of visual tokens embeds target concepts in image-dependent patterns. Gradients and token spaces differ greatly across images and concepts, and the transformations between input-image and target-image token spaces do not transfer. E3W also depends on visual and textual encoder structure. This dependence prevents F8V across the structural differences among BLIP2, LLaVA, Flamingo, and other LVLMs.

## 6 Results

We evaluated five state-of-the-art VLMs: OpenFlamingo, BLIP-2, InstructBLIP, and LLaVA. The Flamingo implementation is the publicly available OpenFlamingo-9B, Awadalla et al. (2023). Its performance is comparable to that of the original model.

BLIP-2 uses two-stage fusion. A frozen CLIP image encoder first extracts features. A Querying Transformer then processes them, Li et al. (2023b). OPT-2.7b provides flexible language generation and efficient task adaptation.

InstructBLIP adds instruction tuning to this fusion architecture, Dai et al. (2023). Instruction tuning trains the model to follow instructions. InstructBLIP uses Vicuna-7b for task-specific instruction following and finer output control through crafted prompts.

LLaVA, Liu et al. (2023), connects pretrained CLIP ViT-L to Vicuna-13B through a lightweight projection that maps between their representations. Its decoder-only generation supplies visual features directly as special tokens in the prompt context. This differs from the explicit visual–textual alignment that fusion models perform before generation.

OpenFlamingo and LLaVA represent decoder-only architectures. BLIP-2 and InstructBLIP represent fusion-based architectures. This distinction significantly affects targeted adversarial effectiveness across prompts.

### 6.1 Claim 1: Q7X achieves N6Z across target texts [Reproduced]

Q7X examples generated on Flamingo transfer across held-out prompts in VQA, G9M, and D5R. Held-out prompts do not appear in training. This evaluation reproduces the central N6Z claim. It reports nonzero H2Q across all tasks and diverse target prompts.

#### 6.1.1 Experimental Details

The scored task families are:

- **A7L = VQAgeneral:** general visual questioning.
- **T4B = VQAspecific:** specific visual questioning.
- **G9M = Classification:** image-label prediction.
- **D5R = Captioning:** image description.
- **C8H = Overall:** aggregate performance.

The two questioning labels distinguish how specifically a prompt relates to a particular image.

Training follows Luo et al. (2024). Evaluation prompts differ from training prompts and cover a broad range of meanings. The Flamingo test measures targeted H2Q. Appendix D.1 gives details for all reproduced experiments, and Appendix E lists the prompts.

#### 6.1.2 Comparative Analysis

Q7X generally matches the claimed transfer behavior. However, it consistently underperforms on G9M and D5R, especially for the abstract target texts “too late” and “not sure,” which lack significant visual meaning. Successful deception still occurs across diverse targets.

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

At equal prompt counts, Q7X consistently exceeds the baselines. Adding prompts does not let the baselines overtake Q7X as their performance saturates. Increasing prompt counts raises H2Q for all methods. R4K inherently uses one prompt, so varying the prompt count is redundant for it.

The original authors chose “unknown” to avoid high-frequency responses common in vision–language tasks. We accept that rationale for this experiment. Gains from adding prompts saturate rapidly, consistent with Luo et al. (2024).

**Figure 4.** Targeted H2Q against prompt count on Flamingo, with equal attack iterations. Q7X significantly outperforms the baselines. The target is “unknown.” R4K is the lone star.

### 6.3 Claim 3: Q7X converges towards higher H2Q as iterations increase [Reproduced]

Taskwise convergence is unstable even though we reuse the authors' exact training and evaluation parameters and code. The original convergence metrics in Appendix A.2 differ from these results and showed convergence with more prompts. When we combine tasks, convergence as iterations increase follows the original aggregate trend.

The authors did not specify the model for this scenario. Our reproduced curves use OpenFlamingo and the target “unknown.”

- (a) VQA
- (b) VQA-Specific
- (c) G9M
- (d) D5R

**Figure 5.** R4K, V9D, and Q7X compared across categories.

**Figure 6.** Aggregate convergence and Q7X comparison.

### 6.4 Claim 4: Q7X outperforms baselines under few-shot settings [Reproduced]

Q7X significantly exceeds the baseline on most tasks when added examples expand the context. This supports the intended robustness of the min–max objective to a larger prompt space and the resulting context changes. The min–max objective minimizes the image loss while maximizing the prompt loss.

| Setting | Method | A7L | T4B | G9M | D5R | C8H |
|---|---|---:|---:|---:|---:|---:|
| shot=0 | V9D | 0.7240 | 0.8740 | 0.5550 | 0.2850 | 0.6095 |
| | Q7X | 0.9680 | 0.9880 | 0.7070 | 0.4200 | 0.7708 |
| shot=2 | V9D | 0.7860 | 0.9420 | 0.5630 | 0.0010 | 0.5730 |
| | Q7X | 0.9440 | 0.9940 | 0.6860 | 0.0300 | 0.6635 |

**Table 2.** Targeted H2Q with and without visual in-context learning. The shot count is the number of added images.

Flamingo supports extra images as in-context examples for task adaptation. Following Luo et al. (2024), we optimize perturbations at 0-shot, with no extra images. We evaluate at 2-shot, with two extra images, so evaluation differs from optimization.

Changes in generation conditions make attacks harder, Luo et al. (2024). Contextual awareness moves the correct prompt space away from the target-text space. Zhao et al. (2023) and Qi et al. (2023) verify that context-specific attacks such as Q7X lose effectiveness when context changes.

### 6.5 Results Beyond the Original Paper

The extended experiments add initialization, attention-value guidance, and input-diversity changes to the original framework.

#### 6.5.1 Noise Initialization

The integrated variant, CroPA+Init, labeled B8N, is reported to outperform Q7X and V9D consistently across every metric. Reported gains cover aggregate accuracy, adversarial robustness, and N6Z. Semantic initialization is also reported to require fewer optimization steps. It provides a robust, adaptable alternative to random initialization without changing the perturbation budget.

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

**Table 3.** BLIP2 targeted H2Q across target texts with vision-encoder noise initialization. Section D.2 gives detailed hyperparameters.

##### Theoretical Analysis of Initialization in Bi-level Optimization

Semantic starting points deliberately lie closer to meaningful adversarial directions. Zhang et al. (2025) find that min–max problems are substantially more sensitive to initialization than ordinary minimization. They often need carefully designed starting points for stable convergence. Q7X resembles a saddle-point problem, in which competing minimization and maximization objectives interact. Its prompt perturbations continually change the optimization landscape.

More frequent image updates than prompt updates create asymmetry and increase the importance of the starting point. Chen et al. (2019) find local cycling in alternating-gradient min–max methods, where updates repeatedly move around a region. They also find that convergence strongly depends on starting close to adversarially robust regions.

Wu et al. (2020) find that perturbations with semantic prior information transfer better across models than random starts and substantially improve success rates. Our observations align with that finding, especially for the more complex paths of bi-level optimization.

| Method | A7L | T4B | G9M | D5R | C8H |
|---|---:|---:|---:|---:|---:|
| V9D | 0.7053 | 0.8807 | 0.7001 | 0.5553 | 0.7104 |
| Q7X | 0.8723 | 0.9457 | 0.7518 | 0.6138 | 0.7958 |
| E3W | 0.2985 | 0.2281 | 0.4857 | 0.4687 | 0.370 |
| B8N | 0.9145 | 0.9342 | 0.8982 | 0.9464 | 0.9209 |

**Table 4.** Average targeted H2Q on BLIP-2 across visual-questioning, G9M, and D5R tasks. Best task values are highlighted in bold. E3W, Yang et al. (2024), is the additional baseline.

Gradient attacks ordinarily acquire target semantics and, in theory, should not need semantic initialization. The complexity of min–max optimization weakens that alignment. We describe injecting target semantics at initialization as necessary to reinforce alignment and aid convergence.

We quantify feature uptake between the target text and learned noise with the CLIP score, labeled Y2K. This score measures their feature similarity. We compare the base perturbation with its initialized counterpart, rather than measuring task success alone, to assess how much the perturbations acquire target attributes.

| Method | Y2K |
|---|---:|
| Q7X | 21.80 |
| B8N | 23.09 |

Target text: “Bomb”.

**Figure 7.** (a) Aggregate convergence of Q7X and B8N. (b) Feature similarity between learned perturbations and the target, quantified by Y2K.

#### 6.5.2 Investigating K3R and Image Augmentation

Learning across images does not make unmodified Q7X effectively universal. Prompt-directed signals remain tightly tied to the originating image's low-level features. This weakens the original claim that learning perturbations across images would establish K3R.

Both input-mixing approaches produce measurable gains. However, the gap between cross-image methods and Q7X applied to a single image remains a fundamental limitation. Q7X excels at prompt robustness, not at effective adversarial attacks that transfer across images.

U8G is believed to outperform J6T for adversarial transfer. Its region-based mixing preserves distinct, non-overlapping spatial areas. This encourages modular features that can be manipulated independently and greater generalization across inputs.

| Method | VQA | T4B | G9M | D5R | C8H |
|---|---:|---:|---:|---:|---:|
| Q7X (Base) | 0.3500 | 0.4920 | 0.4000 | 0.5000 | 0.4355 |
| Q7X with J6T | 0.6560 | 0.7160 | 0.5000 | 0.9500 | 0.7055 |
| Q7X with U8G | 0.4520 | 0.7240 | 0.5000 | 0.5500 | 0.5565 |

**Table 5.** Untargeted K3R H2Q, best across all iterations. Unmixed Q7X peaks at 1700 iterations. J6T and U8G significantly exceed it earlier, at 1500 and 1600 iterations, respectively. Appendix D.3 gives experimental parameters.

#### 6.5.3 Guiding perturbations via Target Value Vectors

P2F greatly improves the existing algorithm across tasks, especially A7L and D5R. The target “Bomb” is harmful text expected inherently to activate guardrails, the model's safeguards. The improvement persists in such harmful-word cases, where worse performance is expected, and in vulnerable scenarios.

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

The value-guided method significantly improves transfer over unmodified Q7X, especially for G9M and D5R. VQA gains are modest. VQA prompts occupy a more complex embedding space than the smaller output spans of targeted G9M and D5R.

Encoder dependence verifies the earlier limitation hypothesis. BLIP2 perturbations transfer passably to InstructBLIP, which uses the same encoder, and fail to transfer to OpenFlamingo. Training on Flamingo and applying the perturbations to the other models likewise produces insignificant H2Q, as Table D.4.5 in Appendix D.4.5 shows. Appendix D.4 gives experimental details.

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

**Table 8.** Cross-model tests. In each setting, the left model supplies the optimized perturbations, and the right model is tested.

For a given target text, vision-encoder value vectors remain approximately consistent across input images. They reflect encoder architecture and pretraining. This consistency may limit perturbation diversity and the generation of highly varied adversarial examples. Target-text choice strongly influences outcomes relative to input-image-specific features.

## 7 Discussion

### 7.1 Observations with reproducibility

We used the authors' public repository and original model configurations to compare baselines faithfully. The implementation was generally well documented. However, the BLIP-2 and InstructBLIP code produced multiple runtime errors. Running it required substantial debugging and adjustment.

### 7.2 Noise Initialization findings

The initialization hypothesis includes reduced cross-modal and cross-prompt transfer when initialization ignores meaning. Cross-modal transfer concerns transfer between image and text processing. We describe experiments across all evaluated benchmarks as significantly favoring semantic noise over Q7X and random-initialized baselines.

As expected from gradient optimization, Q7X perturbations already acquire target context. Initialization produces a modest Y2K increase, not a significant upgrade in semantic features. We interpret this as a subtle move closer to adversarially vulnerable regions. Convergence before and after initialization aligns with the hypothesis that starting nearer meaningful adversarial directions leads to faster convergence.

### 7.3 Evaluation of Transferability Experiments

#### 7.3.1 K3R

J6T improves K3R, particularly for VQA, T4B, and D5R. U8G also improves the base image-transfer version.

Our proposed explanation favors J6T's explicit spatial and scale invariance. J6T randomly crops and resizes the same image, then softly mixes the result with another image. Semantically aligned features become less sensitive to position and scale, making generalization across inputs easier.

Both augmentations preserve original semantic information while adding image diversity. Their proposed benefit is learning features shared across images. However, augmentation alone does not remove the limitation. Further work should combine augmentation with more complex, layered methods. These could include feature-alignment attacks on latent representations—the model's internal representations—or direct perturbation in latent space.

#### 7.3.2 F8V

We describe X9C guidance as a significant advance that remains robust to harmful target text and guardrail activation. Encoder-specific dependence remains the obstacle to encoder-agnostic attacks, which would not depend on a particular encoder. This dependence limits perturbation diversity and transfer across heterogeneous architectures.

Despite this limitation, we present the framework as progress toward universal attacks that work across prompts and images. It also represents progress toward improved VLM generalizability and robustness. E3W cannot provide comparable transfer results and is inherently non-transferable.

### 7.4 Trade-offs Between Different Types of Adversarial Transferability

#### 7.4.1 N6Z vs. F8V

Competing text embeddings implicitly tie original Q7X perturbations to the target architecture. Moving attacks between models causes a significant H2Q drop.

X9C targets encoder components that may generalize, rather than model-specific features. It improves transfer between related architectures, from BLIP-2 to InstructBLIP. However, it offers only modest gains across fundamentally different architectures, from BLIP-2 to Flamingo. Optimizing prompt diversity and model diversity at the same time creates an inherent tension.

#### 7.4.2 N6Z vs. K3R

Image-specific coupling supports strong N6Z but fundamentally limits K3R. The two are competing optimization objectives. Augmentation diversifies optimization and produces promising improvements, but it does not resolve this fundamental tension.

#### 7.4.3 Role of Initialization in Transferability Balance

Semantic anchoring raises H2Q without sacrificing N6Z. This indicates that starting-point selection substantially influences the transfer balance that an attack can achieve.

Unlike changes to the optimization process, better initialization allows convergence toward more general adversarial directions that remain effective across prompts. Even so, optimizing a perturbation for one transfer dimension inevitably sacrifices effectiveness in others.

#### 7.4.4 Implications for Future Research

No single method can maximize all forms of transfer at once. Controlled balancing may require multi-objective optimization, which optimizes several objectives together, or attack designs that account for architecture.

Improving individual dimensions is not enough for a comprehensive solution. Such a solution requires a fundamental rethink of how perturbations interact with model architecture and data representation.

N6Z remains underexplored, with little literature on prompt-induced vulnerability. This gap in understanding matters for secure deployment.

### 7.5 Broader Impact

Examples with high N6Z can elicit malicious or misleading outputs from harmless queries. They can enable misinformation or subvert VLM-dependent systems for content generation and decision-making.

Imperceptible perturbations can also protect privacy. When embedded in personal images, they can force predetermined, nonsensitive text and block unauthorized extraction of confidential information. This provides a proactive security mechanism as visual data becomes increasingly exploitable.

The initialization and loss refinements raise H2Q and improve generalization across models and images. They strengthen cross-prompt attacks while opening paths toward more resilient defenses. Deeper understanding is necessary for robust VLMs, effective countermeasures, protection of user data, and output integrity.

### 7.6 Limitations and Future Work

Limited computation repeatedly hindered successful replication of key results and extensive ablations, which test the effects of individual components. Attack training and evaluation on large Flamingo and BLIP2 models require substantial GPU memory and processing. These demands limited experimental scale and depth, extensive hyperparameter tuning, and transfer testing across wider ranges of architectures.

We planned to investigate transfer using an ensemble of models, meaning several models together. This could have deepened the generalizability analysis. However, it exceeded available memory and computation and was infeasible on the existing infrastructure.

Future work includes encoder-agnostic F8V, stronger K3R, and query-based optimization for practical use.

### 7.7 Communication With Original Authors

We did not establish direct communication with the original authors during replication. The GitHub issue concerning BLIP-2 and InstructBLIP remains unresolved at the time of writing.
