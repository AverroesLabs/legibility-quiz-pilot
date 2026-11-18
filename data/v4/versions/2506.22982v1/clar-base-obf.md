# Revisiting CroPA: A Reproducibility Study and Enhancements for Cross-Prompt Adversarial Transferability in Vision-Language Models

| Author | Email | Affiliation | Author notation |
|---|---|---|---|
| Atharv Mittal | atharv_m@mfs.iitr.ac.in | Mehta Family School of Data Science and Artificial Intelligence; Indian Institute of Technology, Roorkee | ††thanks: All authors contributed equally |
| Agam Pandey | agam_p@ce.iitr.ac.in | Department of Civil Engineering; Indian Institute of Technology, Roorkee | 11footnotemark: 1 |
| Amritanshu Tiwari | amritanshu_t@mfs.iitr.ac.in | Mehta Family School of Data Science and Artificial Intelligence; Indian Institute of Technology, Roorkee | 11footnotemark: 1 |
| Sukrit Jindal | sukrit_j@mfs.iitr.ac.in | Mehta Family School of Data Science and Artificial Intelligence; Indian Institute of Technology, Roorkee | 11footnotemark: 1 |
| Swadesh Swain | swadesh_s@ece.iitr.ac.in | Department of Electronics and Communication; Indian Institute of Technology, Roorkee | 11footnotemark: 1 |

###### Abstract

This study reproduces “An Image is Worth 1000 Lies: Adversarial Transferability Across Prompts on Vision-Language Models.” A vision-language model (VLM) processes images and text. The study validates Q7X, the Cross-Prompt Attack (CroPA), and its superiority to existing baselines in N6Z, or Cross-Prompt Transferability. N6Z means that an image attack remains effective when the textual prompt changes. Semantic meaning and target-word frequency do not determine this transferability.

The study extends the replication through perturbation initialization, perturbations shared across images, and an attention-directed loss, an optimization objective that acts on internal attention components. Evaluation covers the Flamingo, BLIP-2, InstructBLIP, and LLaVA model families, with extended experiments on LLaVA. The original results are validated, and the enhancements consistently increase adversarial effectiveness. The resulting framework generates more robust transferable adversarial examples: modified inputs designed to induce incorrect model outputs. These findings bear on real-world VLM security.

Code: https://github.com/Swadesh06/Revisting_CroPA

Reviewed on OpenReview: https://openreview.net/forum?id=5L90cl0xtf

## 1 Introduction

Subtle image perturbations—small changes to an image—can induce incorrect or harmful VLM outputs. Textual input adds attack opportunities beyond those in traditional vision models. Large VLMs transformed computer vision through image G9M, or Classification, which predicts image labels; D5R, or Captioning, which produces image descriptions; and visual question answering (VQA), which answers questions about images. These capabilities enable deeper investigation of visual–linguistic interaction. Zhao et al. (2023), Qi et al. (2023), Zhang et al. (2022a), and Carlini et al. (2024) demonstrated vulnerabilities in these systems.

Three transfer settings are distinct:

- **F8V = Cross-Model Transferability:** an adversarial example remains effective on different model architectures. This is commonly called transferability.
- **K3R = Cross-Image Transferability:** a perturbation remains effective across multiple images.
- **N6Z = Cross-Prompt Transferability:** an image attack remains effective across different textual prompts.

K3R permits Z4P, or Universal Adversarial Perturbations (UAPs). These are shared perturbations applied to multiple images, rather than perturbations optimized separately for each image.

Luo et al. (2024) introduced N6Z and competing learnable prompts. Their image attacks remain effective regardless of textual input across G9M, D5R, and VQA. Earlier work treated visual and textual adversarial perturbations independently. Zhao et al. (2023) analyzed the BLIP and BLIP-2 VLMs using black-box attacks, which do not require access to target-model internals. Their attacks used either model queries or transfer from another model. Schlarmann & Hein (2023) examined targeted and untargeted white-box attacks, which have access to model internals. A targeted attack seeks a specified output; an untargeted attack seeks a departure from the model's original output. Qi et al. (2023) showed that adversarial images can elicit harmful instruction execution. Tu et al. (2023) systematically evaluated attacks on both image and text inputs.

This replication tests four aspects of Q7X: superiority across target texts, rapid saturation as training prompts are added, robustness when in-context examples are added, and convergence as attack iterations increase. In-context examples are examples included in the model's input to guide its response. The replication also examines individual tasks, beyond the original aggregate analysis.

The extensions address three areas. First, initialization aims to improve both H2Q, or Attack Success Rate (ASR), and transferability. H2Q measures how often an attack meets its success criterion. Second, shared perturbations use the Q7X framework with image augmentation: transformations or combinations of training images intended to promote transfer. Third, target value vectors guide perturbations. In attention mechanisms, which control interactions among image patches, value vectors carry patch information. The proposed manipulation aims to improve generalization and H2Q. The analyses examine the mechanisms behind these enhancements and the nature of VLM adversarial vulnerabilities.

## 2 Related Works

Szegedy et al. (2013) and Goodfellow et al. (2014) introduced adversarial attacks on deep neural networks. Applying these attacks to VLMs introduces additional challenges and opportunities.

### 2.1 Adversarial Transferability

Examples designed to attack classification also affect object detection and other visual tasks. This establishes cross-task transferability alongside F8V and K3R: Naseer et al. (2019b); Naseer et al. (2019a); Lu et al. (2020); Salzmann et al. (2021). Transfer across these dimensions raises robustness questions for modern machine-learning systems.

Liu et al. (2017) and Tramèr et al. (2017) established that attacks crafted on one model often deceive other models, including models with different architectures. Moosavi-Dezfooli et al. (2017) and Mopuri et al. (2017) established perturbations that generalize across images, including Z4P.

### 2.2 Adversarial Attacks on Vision-Language Models

Early task-specific D5R attacks manipulated captioning outputs: Xu et al. (2019); Zhang et al. (2020); Aafaq et al. (2021); Chen et al. (2017). Targeted VQA attacks misled attention mechanisms: Xu et al. (2018); Kaushik et al. (2021); Kovatchev et al. (2022); Li et al. (2021); Sheng et al. (2021); Zhang et al. (2022b).

These attacks mainly addressed lightweight CNN–RNN systems. Here, CNN means convolutional neural network, a network that extracts image features, and RNN means recurrent neural network, a network that processes sequences. These systems lacked in-context learning, which restricts the attacks' applicability to contemporary VLMs. Zhao et al. (2023) studied modern large VLMs, specifically BLIP and BLIP-2, in a black-box setting. Their emphasis was F8V rather than N6Z.

### 2.3 Cross-Prompt Adversarial Transferability

No credible benchmarks or baselines have yet been established for this relatively new concept. Yang et al. (2024) nevertheless provides an additional experimental comparator: E3W, the Context Injection Attack (CIA), also called Contextual Injection Attack. E3W uses gradient-based perturbations, which use derivatives of a loss to choose input changes. It injects target tokens—units of text processed by the model—into visual and textual contexts. This differs from the competing-prompt optimization introduced by Q7X in Luo et al. (2024).

Prompts allow large VLMs to switch tasks dynamically, unlike fixed-task traditional vision models: Li et al. (2023b); Awadalla et al. (2023); Li et al. (2023b); Zhu et al. (2023); Liu et al. (2023). An attack that remains valid as prompts change creates an additional attack surface. Making the image and text perturbations compete, rather than manipulating them independently, establishes a distinct way to evaluate VLM robustness.

### 2.4 Enhancing Adversarial Transferability

Gradient misalignment means that attack-update directions do not align across the settings to which an attack must transfer. It limits transferability and has motivated optimization strategies in Li et al. (2023a) and Wang et al. (2023). Input diversity in Xie et al. (2019) and augmentation in Wang et al. (2021) also improve attack robustness. These findings inform the Q7X enhancements across multiple transfer dimensions.

## 3 Scope of reproducibility

Faithful reproduction of Luo et al. (2024) requires following their experimental procedures. This study tests the following named claims:

- Claim 1: Q7X achieves N6Z across target texts.
- Claim 2: Q7X has the best aggregate performance across prompt counts.
- Claim 3: Q7X converges toward higher H2Q with increasing iterations.
- Claim 4: Q7X outperforms baselines in few-shot settings. Here, few-shot settings provide a small number of in-context examples.

The extended scope includes F8V, K3R, and comparison with other N6Z methods. Images generated through Q7X are tested for consistent deception across different VLMs, input prompts, and model parameters. This extends the investigation from the original prompt-transfer improvement to the broader robustness and versatility of adversarial attacks.

## 4 Methodology

### 4.1 Problem Formulation

Targeted success requires generation of the exact target text. Untargeted success requires any departure from the clean-image prediction, meaning the model's prediction on the unmodified image. H2Q applies the corresponding criterion in each scenario.

The objective uses language-modeling loss, which measures output-text error, across prompt instances. It is not a vision-only objective. Let:

- \(f\) be the VLM;
- \(x_v\) be the visual input;
- \(x_t^i\) be textual prompt \(i\), where \(i\) indexes the \(k\) prompts;
- \(\delta_v\) be the image perturbation;
- \(L\) be the language-modeling loss;
- \(T\) be predetermined target text, chosen independently of the input prompt.

The targeted objective chooses \(\delta_v\) to minimize the summed loss against \(T\):

\[
\min_{\delta_v}\sum_{i=1}^{k}
L\bigl(f(x_v+\delta_v,x_t^i),T\bigr)
\tag{1}
\]

The untargeted objective chooses \(\delta_v\) to maximize the summed discrepancy from clean-image outputs:

\[
\max_{\delta_v}\sum_{i=1}^{k}
L\bigl(f(x_v+\delta_v,x_t^i),f(x_v,x_t^i)\bigr)
\tag{2}
\]

Here, \(\min\) and \(\max\) mean minimization and maximization, and \(\sum\) adds the losses over prompts. The human-imperceptibility constraint is \(\|\delta_v\|_p\leq\epsilon_v\). The notation \(\|\cdot\|_p\) denotes a \(p\)-norm, a measure of perturbation size, and \(\epsilon_v\) is its allowed upper bound.

This formulation adapts established adversarial practice to multimodal real-world applicability, Carlini et al. (2024). It extends the neural-network vulnerability setting of Goodfellow et al. (2014) to prompt variation, Luo et al. (2024).

### 4.2 Baseline Approaches

Because established N6Z baselines were absent, Luo et al. (2024) introduced two reference attacks:

- **R4K = Single-P:** optimize a perturbation using one prompt.
- **V9D = Multi-P:** optimize a perturbation using a collection of prompts.

Their targeted objectives are:

\[
\min_{\delta_v}L\bigl(f(x_v+\delta_v,x_t),T\bigr)
\tag{3}
\]

\[
\min_{\delta_v}\sum_{i=1}^{k}
L\bigl(f(x_v+\delta_v,x_t^i),T\bigr)
\tag{4}
\]

The prompt collection is \(X_t=\{x_t^1,x_t^2,\ldots,x_t^k\}\). The one-prompt case is the simplest baseline. The untargeted forms invert the targeted objectives: they maximize clean/perturbed output discrepancy rather than target-text likelihood.

For Figure 1, \(\delta_t\) denotes a prompt perturbation. A vision encoder converts an image into internal features; a language model uses its inputs to generate text. A forward pass computes the output, and backpropagation computes derivatives used to update perturbations.

**Figure 1.** Luo et al. (2024) framework: image \(\delta_v\) and prompt \(\delta_t\) perturbations act antagonistically, minimizing and maximizing language-modeling loss respectively. Solid arrows denote the forward pass through vision encoder and language model; dashed arrows denote backpropagation, with configurable perturbation-update frequencies. Asterisks mark fixed model parameters; \(\oplus\) denotes elementwise addition.

### 4.3 Cross Prompt Attack

Q7X, the Cross-Prompt Attack or CroPA of Luo et al. (2024), jointly optimizes \(\delta_v\) and \(\delta_t\). Their objectives oppose each other rather than cooperate. The image perturbation minimizes the target-text loss, while the prompt perturbation maximizes it:

\[
\min_{\delta_v}\max_{\delta_t}
L\bigl(f(x_v+\delta_v,x_t+\delta_t),T\bigr)
\tag{5}
\]

This competition is intended to force stronger image transferability across textual prompts. Evaluation discards the prompt perturbations and applies only the optimized image perturbation. Effectiveness must therefore come from the image perturbation's own transferability, not from modified evaluation prompts.

The implementation uses iterative image-gradient descent and prompt-gradient ascent. Descent reduces the loss; ascent increases it. S3D denotes Projected Gradient Descent (PGD), following Madry et al. (2017). Projection returns an updated image to the permitted perturbation region. Image updates occur \(N\) times per prompt update. The two modalities therefore have configurable update frequencies, not necessarily simultaneous, equal-frequency updates.

In Algorithm 1, \(x'_v\) and \(x'_t{}^i\) denote perturbed image and prompt versions. The step sizes \(\alpha_1,\alpha_2\) control update magnitudes; \(K\) is the iteration count; and \(\epsilon\) is the perturbation bound. The gradients \(g_v,g_t\) are derivatives of the loss with respect to the indicated inputs. The operator \(\nabla\) denotes a gradient, \(\operatorname{sign}\) takes each component's sign, and \(\operatorname{mod}\) gives the division remainder. Thus, \(\operatorname{mod}(\text{step},N)=0\) triggers a prompt update every \(N\) iterations. \(\operatorname{Clip}_{x_v,\epsilon}\) enforces the permitted distance from \(x_v\).

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

Q7X performs poorly on G9M and D5R and has limited F8V. S3D effectively raises H2Q on a single model. However, dependence on that model's loss feedback constrains transferability. This is a recognized limitation of gradient attacks: Qin et al. (2024); Li et al. (2024); Wang et al. (2023); Li et al. (2023a). The original authors recognize it, and Table 9 in Appendix A.1 verifies it.

The extensions address task inconsistency and different transfer dimensions. K3R could remove the need for separate training on each image. F8V could provide still greater efficiency. Transfer exploration is not a core proposal of Luo et al. (2024).

The following table retains the resource costs and reported quantitative characterizations separately from the comparisons below. Its additional method labels are:

- **P2F = CroPA+D-UAP**, also called **re-CroPA**, the value-guided method developed in Section 5.2.
- **A7L = VQAgeneral**, general visual questioning. **T4B = VQAspecific** denotes questions more specific to a particular image.
- **J6T = SCMix**, self- and cross-mixing augmentation, developed in Section 5.3.
- **U8G = CutMix**, regional image-patch replacement, also developed in Section 5.3.

A GPU is a graphics processing unit. VRAM is the GPU's memory, and GB denotes gigabytes.

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

W5J denotes Noise Initialization via Vision Encoding Optimization. This diffusion-anchored method replaces target-agnostic random noise with alignment to a target image's features. A diffusion model generates that target image from the target text. Its encoded representation then guides the initial perturbation.

The method retains imperceptibility while providing semantically meaningful adversarial directions at the start. Conventional initialization neither guarantees these directions nor avoids reduced H2Q and N6Z.

Q7X shares random initialization with many gradient attacks. Ignoring target semantics is associated with instability and slower convergence, particularly in VLMs that require semantic alignment with prompts. Q7X's bi-level framework has interacting image and prompt optimization levels. It is empirically sensitive to initialization; see Section 6.5.1, Figure 7, and Table 3. The hypothesis is that semantically informed perturbations improve attack effectiveness within the same imperceptibility requirement.

A state-of-the-art diffusion model synthesizes a target image. Its encoder representation anchors the initial noise to the objective. An encoder-output error loss enforces semantic similarity while retaining adversarial effectiveness. L6V denotes Mean Squared Error (MSE), a squared-error measure used here as the optimization criterion. This replaces a purely random starting direction for later attack updates.

The method is presented as improving adversarial robustness, strengthening perturbations and N6Z, and reducing the steps needed for convergence.

#### 5.1.1 Noise Initialization

Preliminary BLIP-2 and InstructBLIP experiments favor semantic noise over random noise. Diffusion-based anchoring improves alignment in encoder space, the space of representations produced by the image encoder. It also preserves visual fidelity and strict perturbation budgets. Its additional single-image computation is limited to the single-GPU overhead reported above: 20-25 seconds per image.

#### 5.1.2 Diffusion-Based Target Synthesis

Let \(\mathbf{x}_{\mathrm{target}}\) be the generated target image and \(\mathcal D\) the image-generating diffusion model:

\[
\mathbf{x}_{\mathrm{target}}
=\mathcal{D}(T,\mathbf{z};\theta_{\mathrm{SDXL}})
\tag{6}
\]

Stable Diffusion XL (SDXL) supplies \(\mathcal D\). The vector \(\mathbf z\) is sampled from a Gaussian distribution, and \(\theta_{\mathrm{SDXL}}\) denotes the pretrained diffusion-model weights. The generated image embodies the semantic attributes of target prompt \(T\) and provides the initialization reference.

#### 5.1.3 Vision-Encoder Anchored Perturbation

Let \(\mathbf x\) be the original image, \(f_v\) the VLM's vision component, and \(\boldsymbol\delta_{\mathrm{init}}\) the initialized perturbation:

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

Here, \(\arg\min\) selects the perturbation that minimizes the objective. The \(\ell_\infty\) norm, written \(\|\cdot\|_\infty\), measures the largest absolute component change. The bound \(\epsilon\) is the maximum allowable, imperceptible perturbation. The squared \(\ell_2\) norm, \(\|\cdot\|_2^2\), sums squared component differences.

The objective makes the perturbed image's encoder representation mimic the target image's representation. This places the adversarial example in a semantically meaningful neighborhood of the target prompt representation. Algorithm 2 in Appendix B formalizes the method.

#### 5.1.4 Adversarial Optimization via S3D

Let \(\mathbf{x}_{\mathrm{adv}}^{(t)}\) be the adversarial image at iteration \(t\), \(\alpha\) the step size, and \(\mathcal L_{\mathrm{L6V}}\) the encoder-error loss:

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

After initialization, these updates incrementally align the example with the target embedding, meaning its encoded representation. The set \(\mathcal B_\epsilon(\mathbf x)\) is the admissible \(\ell_\infty\) ball around the original image. The operator \(\Pi_{\mathcal B_\epsilon(\mathbf x)}\) projects the updated image into that set.

### 5.2 Guiding perturbations via Target Value Vectors and evaluating F8V

The intended vulnerable components are value vectors in middle-to-late vision layers. Q7X lacks sensitivity to the encoder's architecture and representational structure. It treats vision and language as uniformly vulnerable. This overlooks modality-specific weaknesses, despite the encoder's dominant role in grounding the output in image meaning.

Generic S3D output manipulation is less structured and less targeted than the proposed alternative. The intended improvements concern attack strength, transferability, interpretability, and computational feasibility.

The prior component-selection method is X9C: Doubly-Universal Adversarial Perturbation, also called Doubly-UAP or D-UAP. Kim et al. (2024) proposed it to identify which attention components most influence VLM performance when constructing Z4P. Here, its component loss is adapted by minimizing rather than maximizing losses between the value vectors of input \(x_i\) and target-associated reference \(T_i\). The subscript \(i\) indexes the corresponding input and reference. The method retains a reference associated with the target text.

Cui et al. (2023) and Wang et al. (2024) show that perturbing vision-output embeddings impairs visual perception and misleads language responses in large vision-language models (LVLMs). LVLMs commonly use similar image encoders to produce visual tokens, the internal units representing image content. These tokens contain semantic and structural information. They are the interface through which language decoders access and reason about images. A decoder is the component that generates output text. Targeting encoder internals is proposed to exploit this shared interface.

Two attention components have fundamental roles:

- **Attention weights** control patch-to-patch focus, relevance, and interaction.
- **Value vectors** contain each patch's actual information.

Interfering with attention weights is hypothesized to damage relationship establishment. Perturbing value vectors is hypothesized to damage essential content and interpretation. Prior analysis identifies middle-to-late-layer value vectors as most vulnerable.

The standard X9C objective is:

\[
\delta^*
=
\arg\max_\delta
\frac{1}{|L|}
\sum_{l\in L}
\operatorname{Loss}\bigl(V_l(x),V_l(x+\delta)\bigr)
\tag{10}
\]

In this equation, \(x\) is an input image, \(\delta\) is its perturbation, and \(\delta^*\) is the selected perturbation. The operator \(\arg\max\) selects the perturbation that maximizes the objective. Here, \(L\) denotes the set of selected layers, not the language-modeling loss; \(|L|\) is the number of layers in that set. The index \(l\) identifies a layer, \(V_l(x)\) gives its value vectors for input \(x\), and \(\operatorname{Loss}\) is the loss applied to the vectors.

The proposed combination keeps Q7X's opposing image/text objectives and adds attention-value disruption under the same hyperparameter constraints. Hyperparameters are the configured settings controlling the attack. The hypothesis is that value-vector perturbation improves H2Q when guided by a reference image corresponding to the target text.

Freezing the text encoder—keeping its parameters fixed—is intended to isolate degraded visual processing rather than text artifacts. The modifications seek to exploit encoder information, align with target-text semantics, and improve adversarial robustness across models.

#### 5.2.1 Our Modified Approach

The modified value loss and combined loss are:

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

Here:

- \(L_{\mathrm{X9C}}\) is the value-vector loss.
- \(V_i(x+\delta)\) is the perturbed input's value vector at attention head \(i\). An attention head is one attention component within a layer.
- \(V_t\) is the reference image's target value vector.
- \(N\) counts attention heads in Equation (11), rather than the update interval used in Algorithm 1.
- The dot \(\cdot\) is a vector dot product. Dividing it by the two vector lengths gives cosine similarity, a measure of directional agreement.
- \(L_{\mathrm{Q7X}}\) is the Q7X loss, and \(L_{\mathrm{P2F}}\) is the combined loss.
- \(\lambda\) controls the relative importance of the value loss.

P2F, CroPA+D-UAP or re-CroPA, extracts layer-\(l\) value vectors from the input and the target-associated reference image. It computes their cosine-similarity loss and jointly minimizes its negative with the Q7X loss, under appropriate scaling. This differs from solely maximizing departure from the original representation.

The loss is described as encouraging closeness to the target representation. Combined gradients update \(\delta\), preserving adversarial character, N6Z, and guided semantic influence. The intended use is controlled, interpretable manipulation for robustness and security analysis. Algorithm 3 in Appendix B gives the steps.

X9C depends on the encoder. This motivates a limitation hypothesis: transfer gains may be restricted to VLMs whose vision encoders have the same or similar architecture as the model used for Q7X optimization.

**Figure 2.** X9C architecture diagram.

### 5.3 Transferability across images via Image Augmentation

Per-image iterative gradients overfit image-specific features: they specialize too closely to the training image. This gives poor K3R and high computational cost. Q7X similarly learns a perturbation for each image across prompts.

Luo et al. (2024) describe their method as orthogonal to K3R, meaning that it addresses a separate transfer dimension. They suggest computing perturbations across images, following Moosavi-Dezfooli et al. (2017). This investigation tests that claim and an additional approach based on the Z4P literature.

U8G, or CutMix, uses regional replacement rather than pixel blending. It creates images by replacing rectangular regions with patches from another image. Labels are mixed in proportion to patch area. Learning from partial views promotes spatially invariant features—features that remain useful across locations—and robustness to occlusion and adversarial attacks, Yun et al. (2019). Occlusion means that part of an image is hidden or replaced.

Fang et al. (2024) introduces Z4P for Vision-Language Pretraining (VLP) models. VLP trains models to connect visual and textual information. Its Z4P approach aims to reduce gradient-attack overfitting and enable K3R.

Zhang et al. (2024) improves global and local optimization across VLP models and tasks through M7A, the Effective and Transferable Universal Adversarial Attack (ETU) framework. Within that framework, J6T, or SCMix, increases diversity through self- and cross-mixing while preserving semantics and invariance to position and scale.

J6T is selected for integration into perturbation generation based on empirical results. For the same model, J6T alone yields superior K3R without the authors' additional local utility reinforcement.

#### 5.3.1 J6T

Let \(I_1\) be the base image. Self-mixing first generates two cropped and resized versions, \(X_1\) and \(X_2\), and combines them into \(I'_1\):

\[
I'_1=\eta X_1+(1-\eta)X_2,
\qquad
X_i=\operatorname{Resize}(\operatorname{RandomCrop}(I_1)),
\qquad
\eta=0.5
\tag{13}
\]

Here, \(\operatorname{RandomCrop}\) selects a random image region, \(\operatorname{Resize}\) changes its size, and \(\eta\) is the mixing weight. Cross-mixing then combines \(I'_1\) with a secondary image \(I_2\), producing \(I_3\):

\[
I_3=\beta_1 I'_1+\beta_2 I_2,
\qquad
\beta_1\gg\beta_2\in[0,1)
\tag{14}
\]

The weights are \(\beta_1,\beta_2\). The symbol \(\gg\) means “much greater than,” and \([0,1)\) includes 0 but excludes 1.

Original J6T samples \(\eta\sim\operatorname{Beta}(\alpha,\alpha)\), where \(\sim\) means “is sampled from,” \(\operatorname{Beta}\) is the beta distribution, and \(\alpha>0\) controls that distribution. This implementation fixes \(\eta=0.5\), equivalent to \(\alpha=\infty\). Appendix D.3 provides the justification.

The weighting preserves the dominant semantics of \(I_1\). It also preserves compatibility between \(I_3\) and \(T_1\), the text corresponding to \(I_1\), despite adding features from \(I_2\). This enriches cross-modal interaction.

Variation within one image and blending across images regularize against Q7X overfitting. They encourage perturbations aimed at invariant features. The resulting common perturbation may then apply to unseen images without image-specific optimization.

At inference, when the optimized attack is applied, Q7X returns only a perturbed image, equivalently its perturbation. Effective K3R would make that output universal both across prompts, in the original N6Z sense, and across images, in the additional investigated sense.

**Figure 3.** Self-mixing \(I_1\) generates \(X_1,X_2\), combined into \(I'_1\); external \(I_2\) is mixed with \(I'_1\) to produce \(I_3\).

#### 5.3.2 U8G

U8G constructs the mixed image \(\tilde{x}\) from paired images \(x_A,x_B\):

\[
\tilde{x}=M\odot x_A+(1-M)\odot x_B
\tag{15}
\]

The mask \(M\in\{0,1\}^{W\times H}\) has binary entries, 0 or 1, across image width \(W\) and height \(H\). It selects a randomly located rectangle. The operator \(\odot\) denotes elementwise multiplication.

Discrete regional replacement retains partial views. J6T softly blends features within and across images; U8G uses a hard mask that selects one image or the other at each location. During adversarial training, features that persist across regional combinations are expected to guide perturbations toward spatially invariant, higher-level semantics rather than localized artifacts. This may improve K3R.

### 5.4 Attack Setup and Threat Models for Transferability Experiments

A threat model specifies the attacker's knowledge, permitted changes, and goal. These differ between model-transfer and image-transfer experiments.

#### 5.4.1 Using X9C to explore F8V for Q7X

The attacker seeks one image perturbation that elicits a specified target text from multiple VLMs, independently of the prompt. Optimization uses an accessible surrogate model, which stands in for the unknown models later attacked.

White-box surrogate access includes gradients and internal representations. The surrogate is BLIP-2 or Flamingo. Only the target's general architecture family, such as CLIP-based, is known. CLIP is the visual-encoding model family referred to here. Target-model parameters are unavailable.

The imperceptibility budget is \(\epsilon=16/255\) in \(L_\infty\), the maximum-component perturbation measure. Intra-family testing optimizes on BLIP-2 with the OPT-2.7b language model and evaluates on InstructBLIP with the Vicuna-7b language model. Both have similar CLIP ViT-L/14 encoders. ViT denotes a Vision Transformer image encoder; ViT-L/14 identifies the encoder configuration. The two VLMs differ in language models and fine-tuning objectives, the objectives used for further task adaptation.

Cross-architecture testing transfers from BLIP-2 to Flamingo and in the reverse direction. This changes the vision encoder, language model, and overall architecture.

Each task is scored independently. Success in this evaluation means that the target model fails to generate the correct output from the perturbed image, regardless of prompt.

#### 5.4.2 Investigating K3R via Augmentation Techniques

The goal is one perturbation pattern that causes predetermined target text for any image, regardless of image content or prompt. This aims to improve efficiency over per-image optimization.

The attacker has a limited training-image set, its corresponding J6T variants, and white-box model access during optimization. The perturbation must generalize to unseen images. The \(L_\infty\) budget is \(\epsilon=16/255\).

A separate test set contains no training images. Taskwise H2Q counts failures to generate correct output text on previously unseen images, independently of content or prompt. Controlled self- and cross-mixing is hypothesized to capture image-independent model vulnerabilities rather than specific image features.

#### 5.4.3 Evaluation Rationale for Transferability experiments

All N6Z and F8V transferability results are presented under untargeted attacks. Non-targeted H2Q is also the chosen robust, fair, and comparable assessment for transfer between models and between images.

Different visual-embedding mechanisms make it difficult to isolate how perturbations influence final token embeddings. Targeted transfer is substantially stronger between models with shared encoders, such as BLIP-2 and InstructBLIP. It varies drastically across heterogeneous encoders. Direct targeted-H2Q comparisons across such architectures are therefore imprudent.

Decoder-only LLaVA and OpenFlamingo introduce visual features late, through cross-attention or token concatenation. Cross-attention lets generation attend to visual information; concatenation places visual and textual tokens together. Visual features serve as auxiliary memory for causal generation, which generates text from preceding context. Their influence fades as decoding proceeds, Awadalla et al. (2023), Liu et al. (2023).

Fusion-based BLIP-2, Li et al. (2023b), and InstructBLIP, Dai et al. (2023), integrate visual features early and deeply into input embeddings. They retain visual grounding throughout generation. Targeted attacks are inherently more feasible on these models, Dai et al. (2023).

E3W is excluded from transfer comparisons after showing negligible transfer in experiments. Its per-image visual-token manipulation embeds target concepts in image-dependent patterns. Gradients and token spaces differ greatly across images and concepts, and transformations between input-image and target-image token spaces are non-transferable. Its dependence on visual/textual encoder structure also prevents F8V across structural differences among BLIP2, LLaVA, Flamingo, and other LVLMs.

## 6 Results

Five state-of-the-art VLMs were evaluated: OpenFlamingo, BLIP-2, InstructBLIP, and LLaVA. The Flamingo implementation is the publicly accessible OpenFlamingo-9B, Awadalla et al. (2023). Its performance is comparable to the original model.

BLIP-2 uses two-stage fusion. A frozen CLIP image encoder first extracts features. A Querying Transformer then processes them, Li et al. (2023b). OPT-2.7b provides flexible language generation and efficient task adaptation.

InstructBLIP adds instruction tuning to this fusion architecture, Dai et al. (2023). Instruction tuning adapts the model to follow instructions. It uses Vicuna-7b for task-specific instruction following and finer output control through crafted prompts.

LLaVA, Liu et al. (2023), connects pretrained CLIP ViT-L to Vicuna-13B through a lightweight projection, a mapping between their representations. Its decoder-only generation receives visual features directly as special tokens in the prompt context. This differs from fusion models' explicit alignment of visual and textual information before generation.

OpenFlamingo and LLaVA represent decoder-only architectures. BLIP-2 and InstructBLIP represent fusion-based architectures. This distinction significantly affects targeted adversarial effectiveness across prompts.

### 6.1 Claim 1: Q7X achieves N6Z across target texts [Reproduced]

Q7X examples generated on Flamingo transfer across held-out prompts: evaluation prompts not used in training. This holds across VQA, G9M, and D5R. Table 1 reports nonzero Q7X H2Q for every task and target text, reproducing the central N6Z claim.

#### 6.1.1 Experimental Details

The scored task families are:

- **A7L = VQAgeneral:** general visual questioning.
- **T4B = VQAspecific:** visual questioning specific to a particular image.
- **G9M = Classification:** image-label prediction.
- **D5R = Captioning:** image description.
- **C8H = Overall:** the aggregate result.

The two questioning labels distinguish how specifically a prompt pertains to a particular image.

Training follows Luo et al. (2024). Evaluation prompts differ from training prompts and cover a broad range of semantic meanings. The Flamingo test measures targeted H2Q. Appendix D.1 gives details for all reproduced experiments. Appendix E lists the prompts.

#### 6.1.2 Comparative Analysis

Q7X generally agrees with the claimed transfer behavior. However, it consistently underperforms on G9M and D5R. This is especially evident for the abstract target texts “too late” and “not sure,” which lack significant visual semantics; see their Q7X rows in Table 1. Successful deception nevertheless persists across diverse targets.

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

At equal prompt counts, Q7X consistently exceeds the baselines; see Figure 4. Additional prompts do not allow the baselines' saturated performance to overtake Q7X. Increasing prompt counts raises H2Q for all methods. R4K intrinsically uses one prompt, so varying the count is redundant for it.

The target “unknown” avoids high-frequency responses common in vision–language tasks. This experiment accepts the original authors' rationale for choosing it. Gains from adding prompts saturate rapidly, consistent with Luo et al. (2024) and the trend in Figure 4.

**Figure 4.** Targeted H2Q against prompt count on Flamingo, holding attack iterations equal; Q7X significantly outperforms baselines. The target is “unknown.” R4K is the lone star.

### 6.3 Claim 3: Q7X converges towards higher H2Q as iterations increase [Reproduced]

Taskwise convergence is unstable despite exact reuse of the authors' training/evaluation parameters and code; see Figure 5. Convergence here means the behavior of attack success as optimization proceeds. The originally reported convergence metrics in Appendix A.2 differ from these results and showed convergence with more prompts.

When tasks are combined, convergence as iterations increase follows the original aggregate trend; see Figure 6. The authors did not specify the model for this scenario. The reproduced curves use OpenFlamingo and target “unknown.”

- (a) VQA
- (b) VQA-Specific
- (c) G9M
- (d) D5R

**Figure 5.** R4K, V9D, and Q7X compared across categories.

**Figure 6.** Aggregate convergence and Q7X comparison.

### 6.4 Claim 4: Q7X outperforms baselines under few-shot settings [Reproduced]

Q7X significantly exceeds the baseline on most tasks when additional examples expand the context; see Table 2, shot=2 rows. This supports the intended robustness of the min–max objective, which minimizes over image perturbations while maximizing over prompt perturbations, to expanded prompt space and the resulting context changes.

| Setting | Method | A7L | T4B | G9M | D5R | C8H |
|---|---|---:|---:|---:|---:|---:|
| shot=0 | V9D | 0.7240 | 0.8740 | 0.5550 | 0.2850 | 0.6095 |
| | Q7X | 0.9680 | 0.9880 | 0.7070 | 0.4200 | 0.7708 |
| shot=2 | V9D | 0.7860 | 0.9420 | 0.5630 | 0.0010 | 0.5730 |
| | Q7X | 0.9440 | 0.9940 | 0.6860 | 0.0300 | 0.6635 |

**Table 2.** Targeted H2Q with and without visual in-context learning; shot counts added images.

Flamingo supports extra images as in-context examples for task adaptation. Following Luo et al. (2024), perturbations are optimized at 0-shot, with no extra images. They are evaluated at 2-shot, with two extra images. Evaluation conditions therefore differ from optimization conditions.

Changes in generation conditions make attacks harder, Luo et al. (2024). Contextual awareness moves the correct prompt space away from the target-text space. Zhao et al. (2023) and Qi et al. (2023) verify that context-specific attacks such as Q7X lose effectiveness when context changes.

### 6.5 Results Beyond the Original Paper

The extended experiments add initialization, attention-value guidance, and input-diversity modifications to the original framework.

#### 6.5.1 Noise Initialization

B8N denotes CroPA+Init, the Q7X variant with semantic initialization. It is reported to outperform Q7X and V9D consistently across every metric, with gains in aggregate accuracy, adversarial robustness, and N6Z; the taskwise results are in Table 3, and averages are in Table 4.

Semantic initialization is also reported to require fewer optimization steps; see Figure 7(a). It provides a robust, adaptable alternative to random initialization without changing the perturbation budget.

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

Semantic starting points are deliberately closer to meaningful adversarial directions. Zhang et al. (2025) find that min–max problems are substantially more initialization-sensitive than ordinary minimization. They often need carefully designed starting points for stable convergence.

Q7X resembles a saddle-point problem: its image and prompt variables seek opposing changes in the objective. Prompt perturbations continually change the optimization landscape, meaning the loss values and directions encountered by the image updates.

Updating images more frequently than prompts creates asymmetry and amplifies the importance of the starting point. Chen et al. (2019) find local cycling, or repeated movement around a region, in alternating-gradient min–max methods. They also find that convergence strongly depends on initial proximity to adversarially robust regions.

Wu et al. (2020) find that perturbations with semantic prior information transfer better across models than random starts and yield substantial success-rate improvements. The experimental observations align with that finding, particularly for more complex bi-level optimization paths.

| Method | A7L | T4B | G9M | D5R | C8H |
|---|---:|---:|---:|---:|---:|
| V9D | 0.7053 | 0.8807 | 0.7001 | 0.5553 | 0.7104 |
| Q7X | 0.8723 | 0.9457 | 0.7518 | 0.6138 | 0.7958 |
| E3W | 0.2985 | 0.2281 | 0.4857 | 0.4687 | 0.370 |
| B8N | 0.9145 | 0.9342 | 0.8982 | 0.9464 | 0.9209 |

**Table 4.** Average targeted H2Q on BLIP-2 across visual-questioning, G9M, and D5R tasks. Best task values are highlighted in bold. E3W, Yang et al. (2024), is the additional baseline.

Gradient attacks ordinarily acquire target semantics. In theory, they should not require semantic initialization. Min–max complexity weakens that alignment. Injecting target semantics at initialization is described as necessary to reinforce it and assist convergence.

Feature uptake is quantified using Y2K, or CLIP score, a measure of feature similarity between target text and learned noise. The comparison uses the base perturbation and its initialized counterpart. It assesses how much the perturbations acquire target attributes, rather than measuring only task success.

| Method | Y2K |
|---|---:|
| Q7X | 21.80 |
| B8N | 23.09 |

Target text: “Bomb”.

**Figure 7.** (a) Aggregate convergence of Q7X and B8N. (b) Learned-perturbation/target feature similarity quantified by Y2K.

#### 6.5.2 Investigating K3R and Image Augmentation

Learning across images does not yield effective universality for unmodified Q7X. Prompt-directed signals remain tightly linked to the originating image's low-level features. This weakens the original claim that learning perturbations across images would establish K3R. Table 5 gives the base cross-image results and the augmentation comparisons.

Both input-mixing approaches give measurable gains; compare the J6T and U8G rows with Q7X (Base) in Table 5. However, the gap between cross-image methods and Q7X applied to a single image remains a fundamental limitation. Q7X excels at prompt robustness, not effective attacks that transfer across images.

U8G is believed to outperform J6T for adversarial transfer. Regional mixing retains distinct, non-overlapping spatial areas. This is expected to encourage modular, independently manipulable features and greater generalization across inputs.

| Method | VQA | T4B | G9M | D5R | C8H |
|---|---:|---:|---:|---:|---:|
| Q7X (Base) | 0.3500 | 0.4920 | 0.4000 | 0.5000 | 0.4355 |
| Q7X with J6T | 0.6560 | 0.7160 | 0.5000 | 0.9500 | 0.7055 |
| Q7X with U8G | 0.4520 | 0.7240 | 0.5000 | 0.5500 | 0.5565 |

**Table 5.** Untargeted K3R H2Q, best across all iterations. Unmixed Q7X peaks at 1700 iterations; J6T and U8G exceed it significantly earlier, at 1500 and 1600 respectively. Experimental parameters: Appendix D.3.

#### 6.5.3 Guiding perturbations via Target Value Vectors

P2F greatly improves the existing algorithm across tasks, especially A7L and D5R; compare the P2F and Q7X rows in Tables 6 and 7. The target “Bomb” is harmful text expected inherently to activate guardrails, the model's mechanisms for restricting harmful outputs. The improvement is maintained for such harmful-word cases with worse expected performance and vulnerable scenarios.

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

The value-guided method significantly improves vanilla, or unmodified, Q7X transfer. Gains are especially evident for G9M and D5R, with modest VQA gains; see the Q7X and P2F rows in Table 8. VQA prompts occupy a more complex embedding space than the smaller output spans of targeted G9M and D5R.

Encoder dependence verifies the earlier limitation hypothesis. BLIP2 perturbations transfer passably to same-encoder InstructBLIP and fail to transfer to OpenFlamingo; see the corresponding settings in Table 8. Training on Flamingo and applying perturbations to the other models likewise gives insignificant H2Q; see Table D.4.5 in Appendix D.4.5. Appendix D.4 gives experimental details.

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

For a given target text, vision-encoder value vectors remain approximately consistent across input images. They reflect encoder architecture and pretraining. This consistency may constrain perturbation diversity and the generation of highly varied adversarial examples. Target-text choice strongly influences outcomes relative to input-image-specific features.

## 7 Discussion

### 7.1 Observations with reproducibility

Reproduction used the authors' public repository and original model configurations to support faithful baseline comparison. The implementation was generally well documented. However, the BLIP-2 and InstructBLIP code produced multiple runtime errors. Substantial debugging and adjustment were needed to run it.

### 7.2 Noise Initialization findings

The initialization hypothesis includes reduced cross-modal transfer, meaning transfer involving visual and textual information, as well as reduced cross-prompt transfer when target semantics are ignored. Experiments across all evaluated benchmarks are described as significantly favoring semantic noise over Q7X and random-initialized baselines; see Tables 3 and 4.

Q7X perturbations already acquire target context, as expected from gradient updates. Initialization gives a modest Y2K increase, from 21.80 to 23.09 in Figure 7(b), rather than a significant semantic-feature upgrade. This is interpreted as a subtle shift closer to adversarially vulnerable regions.

The convergence comparison before and after initialization aligns with the hypothesis that starting nearer meaningful adversarial directions leads to faster convergence; see Figure 7(a).

### 7.3 Evaluation of Transferability Experiments

#### 7.3.1 K3R

J6T improves K3R particularly for VQA, T4B, and D5R. U8G also improves the base image-transfer version; see Table 5.

The proposed explanation favors J6T's explicit spatial and scale invariance. Random cropping and resizing of the same image are followed by soft mixing with another image. Semantically aligned features become less sensitive to position and scale, making generalization across inputs easier.

Both augmentations preserve original semantic information while adding image diversity. Their proposed benefit is learning features shared across images. Augmentation alone does not eliminate the limitation, however. Further work should combine augmentation with more complex, layered methods. Examples include feature-alignment attacks on latent representations, the model's internal representations, or direct perturbation in that latent space.

#### 7.3.2 F8V

X9C guidance is described as a significant advance that remains robust to harmful target text and guardrail activation; see the “Bomb” results in Tables 6 and 7. Encoder-specific dependence remains the obstacle to encoder-agnostic attacks, which would not depend on one encoder architecture. It narrows perturbation diversity and transfer across heterogeneous architectures; see Table 8.

Despite this limitation, the framework is presented as progress toward universal attacks effective across prompts and images. It is also presented as progress toward improved VLM generalizability and robustness. E3W cannot provide comparable transfer results and is inherently non-transferable.

### 7.4 Trade-offs Between Different Types of Adversarial Transferability

#### 7.4.1 N6Z vs. F8V

Competing text embeddings implicitly anchor original Q7X perturbations to the target architecture. Moving attacks between models causes a significant H2Q drop; see the Q7X transfer rows in Table 8.

X9C targets potentially generalizable encoder components rather than model-specific features. It improves transfer between related architectures, such as BLIP-2 to InstructBLIP. It provides only modest gains across fundamentally different architectures, such as BLIP-2 to Flamingo; compare the Q7X and P2F rows for those settings in Table 8. Simultaneously optimizing prompt diversity and model diversity creates an inherent tension.

#### 7.4.2 N6Z vs. K3R

Coupling perturbations to image-specific features supports strong N6Z but fundamentally limits K3R. The two are competing optimization objectives. Augmentation diversifies optimization and gives promising improvements, as Table 5 shows, without resolving this fundamental tension.

#### 7.4.3 Role of Initialization in Transferability Balance

Semantic anchoring raises H2Q without sacrificing N6Z; see the initialization results in Tables 3 and 4. This indicates that starting-point selection substantially influences the achievable transfer balance.

Unlike changes to the optimization process, better initialization allows convergence toward more general adversarial directions that remain effective across prompts. Even so, optimizing a perturbation for one transfer dimension inevitably sacrifices effectiveness in others.

#### 7.4.4 Implications for Future Research

No single methodology can maximize all forms of transfer simultaneously. Controlled balancing may require multi-objective optimization, which jointly handles several objectives, or architecturally aware attack designs.

Improving individual dimensions is insufficient for a comprehensive solution without fundamentally rethinking how perturbations interact with model architecture and data representation.

N6Z remains underexplored. Little literature addresses prompt-induced vulnerability. This gap in understanding matters for secure deployment.

### 7.5 Broader Impact

Examples with high N6Z can elicit malicious or misleading outputs from harmless queries. They can enable misinformation or subvert VLM-dependent content-generation and decision systems.

Imperceptible perturbations also offer privacy protection. Embedded in personal images, they can force predetermined nonsensitive text and prevent unauthorized extraction of confidential information. This provides a proactive security mechanism as visual data becomes increasingly exploitable.

The initialization and loss refinements increase H2Q and generalizability across models and images; see Tables 3–8. They strengthen cross-prompt attacks while opening paths toward more resilient defenses. Deeper understanding is needed for robust VLMs, effective countermeasures, user-data protection, and output integrity.

### 7.6 Limitations and Future Work

Scarce computational resources repeatedly hindered successful replication of key results and extensive ablations, the experiments used to isolate individual components' effects. Attack training and evaluation on large Flamingo and BLIP2 models require substantial GPU memory and processing.

These demands limited experimental scale and depth. They also limited extensive hyperparameter tuning and transfer testing across a wider range of architectures.

A planned investigation of ensemble-model transfer, using multiple models together, could have deepened the generalizability analysis. It exceeded available memory and computation and was infeasible on the existing infrastructure.

Future work includes encoder-agnostic F8V, stronger K3R, and query-based optimization, which uses model queries rather than internal gradients, for practical applicability.

### 7.7 Communication With Original Authors

No direct communication with the original authors was established during replication. The GitHub issue concerning BLIP-2 and InstructBLIP remains unresolved at writing.
