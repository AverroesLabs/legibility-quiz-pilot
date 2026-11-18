# Revisiting CroPA: A Reproducibility Study and Enhancements for Cross-Prompt Adversarial Transferability in Vision-Language Models

| Author | Email | Affiliation | Author notation |
|---|---|---|---|
| Atharv Mittal | atharv_m@mfs.iitr.ac.in | Mehta Family School of Data Science and Artificial Intelligence; Indian Institute of Technology, Roorkee | ††thanks: All authors contributed equally |
| Agam Pandey | agam_p@ce.iitr.ac.in | Department of Civil Engineering; Indian Institute of Technology, Roorkee | 11footnotemark: 1 |
| Amritanshu Tiwari | amritanshu_t@mfs.iitr.ac.in | Mehta Family School of Data Science and Artificial Intelligence; Indian Institute of Technology, Roorkee | 11footnotemark: 1 |
| Sukrit Jindal | sukrit_j@mfs.iitr.ac.in | Mehta Family School of Data Science and Artificial Intelligence; Indian Institute of Technology, Roorkee | 11footnotemark: 1 |
| Swadesh Swain | swadesh_s@ece.iitr.ac.in | Department of Electronics and Communication; Indian Institute of Technology, Roorkee | 11footnotemark: 1 |

###### Abstract

This study reproduces “An Image is Worth 1000 Lies: Adversarial Transferability Across Prompts on Vision-Language Models.” Vision-language models (VLMs) process images and text. The reproduced method, Cross-Prompt Attack (CroPA; code Q7X), optimizes image perturbations against competing learnable prompt perturbations. The study validates its superiority to existing baselines in Cross-Prompt Transferability (code N6Z): an adversarial image remains effective when the textual prompt changes. Semantic meaning and target-word frequency do not determine that transferability. The reproduced comparisons appear in Sections 6.1–6.4.

The extensions investigate initialization, perturbations shared across images, and attention-directed loss. Evaluation encompasses Flamingo, BLIP-2, InstructBLIP, and extended LLaVA experiments. The original results are validated, and the enhancements consistently increase adversarial effectiveness (Tables 3–8). The resulting framework generates more robust transferable adversarial examples—perturbed inputs that induce incorrect model outputs—and bears on real-world VLM security.

Code: https://github.com/Swadesh06/Revisting_CroPA

Reviewed on OpenReview: https://openreview.net/forum?id=5L90cl0xtf

## 1 Introduction

Subtle image perturbations can induce incorrect or harmful VLM outputs. Textual input adds attack vectors beyond those of traditional vision models. Large VLMs transformed image-label prediction, called Classification (code G9M); image description, called Captioning (code D5R); and visual question answering (VQA). These capabilities enabled deeper investigation of visual–linguistic interaction. Zhao et al. (2023), Qi et al. (2023), Zhang et al. (2022a), and Carlini et al. (2024) demonstrated vulnerabilities.

Three settings distinguish where an adversarial example remains effective:

- **Cross-Model Transferability (F8V):** effectiveness survives a change of model architecture. This setting is commonly called transferability.
- **Cross-Image Transferability (K3R):** a perturbation remains effective over multiple images.
- **Cross-Prompt Transferability (N6Z):** an adversarial image remains effective over varying textual prompts.

Cross-image transfer permits shared perturbations, called Universal Adversarial Perturbations (UAPs; code Z4P). It therefore need not require a separately optimized perturbation for each image.

Luo et al. (2024) introduced cross-prompt transferability and competing learnable prompts. Their image attacks retain effectiveness regardless of textual input across classification, captioning, and VQA. Earlier work treated visual and textual adversarial perturbations independently. Zhao et al. (2023) analyzed the BLIP and BLIP-2 VLMs using query- and transfer-based black-box attacks. Such attacks use model queries or attacks learned on another model rather than the target model’s internal gradients. Schlarmann & Hein (2023) examined targeted and untargeted white-box attacks, which use internal model access. Qi et al. (2023) showed adversarial images eliciting harmful instruction execution. Tu et al. (2023) systematically evaluated attacks on both modalities: images and text.

The replication tests superiority across target texts, rapid saturation as training prompts are added, robustness to additional in-context examples, and convergence over attack iterations. It also analyzes individual tasks, beyond the original aggregate analysis. Attack Success Rate (ASR; code H2Q) measures effectiveness. Targeted success requires exact target-text generation; untargeted success means departure from the clean-image prediction, as formalized in Section 4.1.

The extensions examine three changes. First, initialization is intended to improve both attack success rate and transferability. Second, shared perturbations use CroPA as their backbone, with image augmentation intended to promote transfer. Third, target value vectors guide perturbations. Attention components govern interactions between image patches; value vectors carry patch information. Their targeted manipulation is proposed to improve generalization and attack success rate. The analyses investigate these enhancements and the nature of VLM adversarial vulnerabilities.

## 2 Related Works

Szegedy et al. (2013) and Goodfellow et al. (2014) introduced adversarial attacks on deep neural networks. Applying these attacks to VLMs adds challenges and opportunities.

### 2.1 Adversarial Transferability

Examples directed at classification also affect object detection and other visual tasks. This establishes cross-task transferability alongside cross-model and cross-image transferability: Naseer et al. (2019b); Naseer et al. (2019a); Lu et al. (2020); Salzmann et al. (2021). These multiple transfer dimensions raise robustness questions for modern machine-learning systems.

Liu et al. (2017) and Tramèr et al. (2017) established that attacks crafted on one model often deceive other models, including different architectures. Moosavi-Dezfooli et al. (2017) and Mopuri et al. (2017) established perturbations that generalize across images, including universal adversarial perturbations.

### 2.2 Adversarial Attacks on Vision-Language Models

Early task-specific captioning attacks manipulated outputs in Xu et al. (2019), Zhang et al. (2020), Aafaq et al. (2021), and Chen et al. (2017). Targeted VQA attacks misled attention mechanisms in Xu et al. (2018), Kaushik et al. (2021), Kovatchev et al. (2022), Li et al. (2021), Sheng et al. (2021), and Zhang et al. (2022b).

These attacks principally addressed lightweight systems combining convolutional neural networks and recurrent neural networks (CNN–RNN systems). They did not support in-context learning, in which examples supplied with the input help specify the task. This restricts their applicability to contemporary VLMs. Zhao et al.’s (2023) black-box study of modern large VLMs, BLIP and BLIP-2, emphasized cross-model rather than cross-prompt transferability.

### 2.3 Cross-Prompt Adversarial Transferability

No credible benchmarks or baselines have yet been established for this relatively new concept. Yang et al. (2024) nevertheless supplies an additional experimental comparator: Context Injection Attack (CIA; code E3W), also called Contextual Injection Attack. It uses gradient-based perturbations to inject target tokens into visual and textual contexts. Tokens are the units represented and processed by the model. This differs from the competing-prompt optimization pioneered by CroPA in Luo et al. (2024).

Prompt-driven adaptation lets large VLMs switch tasks dynamically, unlike fixed-task traditional vision models: Li et al. (2023b); Awadalla et al. (2023); Li et al. (2023b); Zhu et al. (2023); Liu et al. (2023). An attack that remains valid as prompts change therefore exposes an additional attack surface. Making the modalities compete, rather than manipulating them independently, establishes a distinct paradigm for evaluating VLM robustness.

### 2.4 Enhancing Adversarial Transferability

Gradient misalignment—differences between the optimization directions provided by different models—limits transferability. It has motivated optimization strategies in Li et al. (2023a) and Wang et al. (2023). Input diversity in Xie et al. (2019) and augmentation in Wang et al. (2021) also improve attack robustness. These findings inform the multidimensional CroPA enhancements.

## 3 Scope of reproducibility

Faithful reproduction of Luo et al. (2024) requires their experimental procedures. This study tests four named claims:

- Claim 1: Q7X achieves N6Z across target texts.
- Claim 2: Q7X has the best aggregate performance across prompt counts.
- Claim 3: Q7X converges toward higher H2Q with increasing iterations.
- Claim 4: Q7X outperforms baselines in few-shot settings.

Here Q7X is CroPA, N6Z is cross-prompt transferability, and H2Q is attack success rate. Few-shot settings supply additional examples in the input context.

The extended scope includes cross-model transferability, cross-image transferability, and comparison with other cross-prompt methods. Images generated through CroPA are tested for consistent deception across diverse VLMs, input prompts, and model parameters. This examines broader attack robustness and versatility, beyond validating the original prompt-transfer improvement.

## 4 Methodology

### 4.1 Problem Formulation

A model \(f\) processes a visual input \(x_v\) and a textual input \(x_t\). The image perturbation is \(\delta_v\). A set of \(k\) prompts contains prompts \(x_t^i\), where \(i\) indexes the prompts. The language-modeling loss \(L\) measures the model output against a specified reference output. In the targeted setting, \(T\) is predetermined target text, generated independently of the input prompt.

Targeted success requires exact generation of \(T\). Untargeted success is any departure from the clean-image prediction. The corresponding effectiveness measure is attack success rate (H2Q). The reproduced targeted and non-targeted scenarios use language-modeling objectives over prompt instances, rather than vision-only objectives.

The targeted objective minimizes the summed loss against \(T\):

\[
\min_{\delta_v}\sum_{i=1}^{k}
L\bigl(f(x_v+\delta_v,x_t^i),T\bigr)
\tag{1}
\]

The untargeted objective maximizes discrepancy between adversarial and clean outputs:

\[
\max_{\delta_v}\sum_{i=1}^{k}
L\bigl(f(x_v+\delta_v,x_t^i),f(x_v,x_t^i)\bigr)
\tag{2}
\]

Here \(\min\) and \(\max\) mean minimize and maximize over the indicated perturbation; \(\sum\) adds the losses over prompts. The clean output is \(f(x_v,x_t^i)\), and the adversarial output uses \(x_v+\delta_v\).

The human-imperceptibility bound is \(\|\delta_v\|_p\leq\epsilon_v\). The \(p\)-norm \(\|\cdot\|_p\) measures perturbation size, and \(\epsilon_v\) is its allowed maximum. This adapts established adversarial practice to multimodal real-world applicability, Carlini et al. (2024). The formulation extends the neural-network vulnerability setting of Goodfellow et al. (2014) to prompt variation, Luo et al. (2024).

### 4.2 Baseline Approaches

In the absence of established cross-prompt baselines, Luo et al. (2024) introduced two reference attacks. Single-P (code R4K) optimizes an image perturbation using one prompt. Multi-P (code V9D) uses a collection of prompts,
\(X_t=\{x_t^1,x_t^2,\ldots,x_t^k\}\).

Single-P minimizes:

\[
\min_{\delta_v}L\bigl(f(x_v+\delta_v,x_t),T\bigr)
\tag{3}
\]

Multi-P minimizes:

\[
\min_{\delta_v}\sum_{i=1}^{k}
L\bigl(f(x_v+\delta_v,x_t^i),T\bigr)
\tag{4}
\]

The one-prompt case is the simplest baseline. Untargeted versions invert the targeted objectives: they maximize clean/perturbed output discrepancy rather than target-text likelihood.

**Figure 1.** Luo et al. (2024) framework: image \(\delta_v\) and prompt \(\delta_t\) perturbations act antagonistically, minimizing and maximizing language-modeling loss respectively. Solid arrows denote the forward pass through the vision encoder, which maps images to representations, and the language model; dashed arrows denote backpropagation, which computes gradients, with configurable perturbation-update frequencies. Asterisks mark fixed model parameters; \(\oplus\) denotes elementwise addition.

### 4.3 Cross Prompt Attack

CroPA (Q7X) jointly optimizes image perturbation \(\delta_v\) and prompt perturbation \(\delta_t\). Their objectives oppose one another. The image perturbation minimizes target-text loss, while the prompt perturbation maximizes it:

\[
\min_{\delta_v}\max_{\delta_t}
L\bigl(f(x_v+\delta_v,x_t+\delta_t),T\bigr)
\tag{5}
\]

These antagonistic, rather than collaborative, objectives are intended to force stronger image transferability across textual prompts. Evaluation discards prompt perturbations and applies only the optimized image perturbation. Thus, effectiveness must reside in the image perturbation’s inherent transferability, not in modified evaluation prompts.

The implementation uses iterative image-gradient descent and prompt-gradient ascent with Projected Gradient Descent (PGD; code S3D), following Madry et al. (2017). Projection keeps the image within the allowed perturbation region. Image updates occur \(N\) times per prompt update. The update frequency is configurable; the modalities are not updated simultaneously at equal frequency.

In Algorithm 1, \(x'_v\) is the adversarial image and \(x'^i_t\) is the stored perturbed version of prompt \(i\). The image and prompt step sizes are \(\alpha_1\) and \(\alpha_2\). The iteration count is \(K\), and \(\epsilon\) is the perturbation-size limit. Gradients \(g_v\) and \(g_t\) give the local loss-change directions; \(\nabla\) denotes a gradient. The function \(\operatorname{sign}\) takes each gradient component’s sign. The function \(\operatorname{mod}\) gives the division remainder, so \(\operatorname{mod}(\text{step},N)=0\) selects every \(N\)th step. The clipping operator \(\operatorname{Clip}_{x_v,\epsilon}\) projects into the \(\epsilon\)-ball around the original image.

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

CroPA performs poorly on classification and captioning and has limited cross-model transferability. Its PGD optimization effectively raises single-model attack success rate. However, dependence on that model’s loss feedback constrains transferability. This is a recognized limitation of gradient attacks: Qin et al. (2024); Li et al. (2024); Wang et al. (2023); Li et al. (2023a). The original authors recognize it, and Table 9 in Appendix A.1 verifies it. Related results appear in Table 1’s classification and captioning columns and Table 8’s cross-model rows.

The extensions address task inconsistency and multiple transfer dimensions. Cross-image transfer could remove separate per-image training. Cross-model transfer could provide still greater efficiency. Transfer exploration is not a core proposal of Luo et al. (2024).

The following table retains resource costs and reported quantitative characterizations separately from the comparisons. Its additional method codes are:

- **X9C:** Doubly-Universal Adversarial Perturbation, also called Doubly-UAP or D-UAP.
- **P2F:** CroPA+D-UAP, also called re-CroPA, which adds target-value-vector guidance.
- **J6T:** SCMix, an augmentation that mixes an image with itself and then with another image.
- **U8G:** CutMix, an augmentation that replaces rectangular image regions.
- **A7L:** VQAgeneral, general visual questioning.

The graphics processing unit (GPU) for the reported run is an L40S with 48GB, or gigabytes, of memory. VRAM means the GPU’s video memory.

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

Noise Initialization via Vision Encoding Optimization (code W5J) replaces target-agnostic random noise with alignment to a target image’s features. A diffusion image-generation model supplies the target image. This retains imperceptibility while supplying semantically meaningful adversarial directions at the outset. Conventional initialization neither guarantees such directions nor avoids reduced attack success rate and cross-prompt transferability.

CroPA shares random initialization with many gradient attacks. Ignoring target semantics is associated with instability and slower convergence, particularly in VLMs that require semantic alignment with prompts. CroPA’s bi-level framework—optimization with opposing image and prompt objectives—is empirically sensitive to initialization (Section 6.5.1; Figure 7; Table 3). The hypothesis is that semantically informed perturbations improve attack efficacy within the same imperceptibility requirement.

A state-of-the-art diffusion model synthesizes a target image. Its encoder representation anchors the initial noise to the attack objective. An encoder-output error loss enforces semantic similarity while retaining adversarial effectiveness. The criterion is Mean Squared Error (MSE; code L6V), which measures squared differences between representations, rather than a purely random starting direction.

The method is presented as improving adversarial robustness, strengthening perturbations and cross-prompt transferability, and reducing the steps needed for convergence. Tables 3–4 and Figure 7 provide the corresponding evaluation.

#### 5.1.1 Noise Initialization

Preliminary BLIP-2 and InstructBLIP experiments favor semantic over random noise. Diffusion-based anchoring improves encoder-space alignment while preserving visual fidelity and strict perturbation budgets. Its added single-image computation is limited to the single-GPU overhead reported above: 20-25 seconds per image.

#### 5.1.2 Diffusion-Based Target Synthesis

Let \(\mathcal D\) be the Stable Diffusion XL (SDXL) image generator. It takes target text \(T\), Gaussian-sampled random noise \(\mathbf z\), and pretrained diffusion weights \(\theta_{\mathrm{SDXL}}\). It generates the target image \(\mathbf{x}_{\mathrm{target}}\):

\[
\mathbf{x}_{\mathrm{target}}
=\mathcal{D}(T,\mathbf{z};\theta_{\mathrm{SDXL}})
\tag{6}
\]

The generated image embodies the semantic attributes of \(T\) and supplies the initialization reference.

#### 5.1.3 Vision-Encoder Anchored Perturbation

Let \(\mathbf x\) be the original image, \(\boldsymbol\delta\) a candidate perturbation, and \(f_v\) the VLM’s vision encoder. The initialization \(\boldsymbol\delta_{\mathrm{init}}\) is chosen to minimize the squared distance from the target image’s encoder representation:

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

Here \(\arg\min\) selects the perturbation that minimizes the objective. The \(\ell_\infty\) norm, \(\|\cdot\|_\infty\), measures the largest absolute component. Its bound \(\epsilon\) is the maximum allowable, imperceptible perturbation. The squared Euclidean norm \(\|\cdot\|_2^2\) sums squared representation differences.

Mimicking the target image’s encoder-space representation places the adversarial example in a semantically meaningful neighborhood of the target prompt representation. Algorithm 2 in Appendix B formalizes the method.

#### 5.1.4 Adversarial Optimization via S3D

Let \(\mathbf{x}_{\mathrm{adv}}^{(t)}\) be the adversarial image at iteration \(t\), and let \(\alpha\) be the step size. The projection \(\Pi_{\mathcal B_\epsilon(\mathbf x)}\) keeps the updated image within the admissible \(\ell_\infty\) ball \(\mathcal B_\epsilon(\mathbf x)\) around original image \(\mathbf x\):

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

The encoder-alignment loss is:

\[
\mathcal{L}_{\mathrm{L6V}}
=
\left\|
f_v(\mathbf{x}_{\mathrm{adv}})
-f_v(\mathbf{x}_{\mathrm{target}})
\right\|_2^2
\tag{9}
\]

After initialization, these refinement updates incrementally align the adversarial example with the target embedding, or encoder representation.

### 5.2 Guiding perturbations via Target Value Vectors and evaluating F8V

The intended vulnerable components are value vectors in middle-to-late vision layers. CroPA lacks architectural sensitivity to the encoder’s representational backbone. It treats vision and language as uniformly vulnerable, overlooking modality-specific weaknesses despite the encoder’s dominant role in grounding language in image content. Generic PGD output manipulation is less structured and less targeted than the proposed alternative. The intended improvements concern attack strength, transferability, interpretability, and computational feasibility.

Kim et al. (2024) proposed Doubly-UAP (X9C). It identifies which attention components most influence VLM performance when constructing universal adversarial perturbations. Here, its component loss is adapted by minimizing rather than maximizing losses between the value vectors of input \(x_i\) and target-associated reference \(T_i\). The reference retains an association with the target text.

Cui et al. (2023) and Wang et al. (2024) show that perturbing vision-output embeddings impairs visual perception and misleads language responses in large vision-language models (LVLMs). LVLMs commonly use similar image encoders. These encoders produce visual tokens containing semantic and structural information. The tokens form the interface through which language decoders access and reason about images. Targeting encoder internals is proposed to exploit this shared interface.

Two attention components have fundamental roles. Attention weights control patch-to-patch focus, relevance, and interaction. Value vectors contain each patch’s actual information. Interfering with attention weights is hypothesized to damage the establishment of relationships. Perturbing value vectors is hypothesized to damage essential content and its interpretation. Prior analysis identifies middle-to-late-layer value vectors as most vulnerable.

For the standard Doubly-UAP objective below, \(L\) now denotes a set of selected layers, not the language-modeling loss. The number of layers is \(|L|\), and \(l\) indexes them. The function \(V_l(x)\) gives input \(x\)’s value vectors at layer \(l\). The function \(\operatorname{Loss}\) is the loss applied to the vectors. The perturbation \(\delta^*\) is selected by \(\arg\max\), which returns a maximizing perturbation:

\[
\delta^*
=
\arg\max_\delta
\frac{1}{|L|}
\sum_{l\in L}
\operatorname{Loss}\bigl(V_l(x),V_l(x+\delta)\bigr)
\tag{10}
\]

The proposed combination retains CroPA’s opposing image/text objectives and adds attention-value disruption under the same hyperparameter constraints. The hypothesis is that value-vector perturbation relative to a target-associated reference image enhances attack success rate. Freezing the text encoder is intended to isolate degraded visual processing rather than text artifacts. The modifications seek better exploitation of encoder information, target-text semantic alignment, and cross-model adversarial robustness.

#### 5.2.1 Our Modified Approach

The combined method is CroPA+D-UAP, also called re-CroPA (P2F). It extracts value vectors from the input and a reference image associated with the target text.

In Equation 11, \(V_i(x+\delta)\) is the perturbed input’s value vector at attention head \(i\), where a head is an attention component. The target value vector from the reference image is \(V_t\). Here \(N\) counts heads; it is not Algorithm 1’s prompt-update interval. The dot product is denoted by \(\cdot\). Dividing it by the vectors’ Euclidean norms gives cosine similarity:

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

The combined loss uses \(L_{\mathrm{Q7X}}\), the CroPA loss, and \(\lambda\), a weight controlling the value loss’s relative importance:

\[
L_{\mathrm{P2F}}
=
L_{\mathrm{Q7X}}(x_v+\delta_v,x_t+\delta_t,T)
-\lambda L_{\mathrm{X9C}}(\delta_v)
\tag{12}
\]

The method computes the layer-\(l\) value vectors’ cosine-similarity loss. It jointly minimizes the negative of that loss with the CroPA loss under appropriate scaling, rather than solely maximizing departure from the original representation.

The loss is described as encouraging closeness to the target representation. Combined gradients update \(\delta\), preserving adversarial character, cross-prompt transferability, and guided semantic influence. Controlled, interpretable manipulation is intended for robustness and security analysis. Algorithm 3 in Appendix B gives the steps.

Doubly-UAP’s encoder dependence supplies a limitation hypothesis: transfer gains may be restricted to VLMs with the same or similar vision-encoder architecture as the model used for CroPA optimization. Section 6.5.4 and Table 8 examine this limitation.

**Figure 2.** X9C architecture diagram.

### 5.3 Transferability across images via Image Augmentation

Per-image iterative gradients overfit image-specific features. This produces poor cross-image transferability and high computational expense. CroPA likewise learns each image’s perturbation across prompts. Luo et al. (2024) describe their method as orthogonal to cross-image transferability and suggest computing perturbations across images, following Moosavi-Dezfooli et al. (2017). This investigation tests that claim and an additional approach drawn from universal-perturbation literature.

CutMix (U8G) uses regional replacement rather than pixel blending. It synthesizes images using rectangular patches cut from another image and mixes labels in proportion to patch area. Its partial-view learning promotes spatially invariant features and robustness to occlusion and adversarial attacks, Yun et al. (2019).

Fang et al. (2024) introduces universal adversarial perturbations for Vision-Language Pretraining (VLP) models to reduce gradient-attack overfitting and enable cross-image transfer. Zhang et al. (2024) improves global and local optimization across VLP models and tasks through Effective and Transferable Universal Adversarial Attack (ETU; code M7A). Within that framework, SCMix (J6T) increases diversity through self- and cross-mixing while preserving semantics and spatial/scale invariance.

SCMix is selected for integration into perturbation generation on the basis of empirical results. For the same model, SCMix alone, without their additional local utility reinforcement, yields superior cross-image transferability.

#### 5.3.1 J6T

SCMix first mixes two cropped and resized versions, \(X_1\) and \(X_2\), of base image \(I_1\). The operations \(\operatorname{RandomCrop}\) and \(\operatorname{Resize}\) select a random crop and resize it. The mixing weight is \(\eta\), and the resulting self-mixed image is \(I'_1\):

\[
I'_1=\eta X_1+(1-\eta)X_2,
\qquad
X_i=\operatorname{Resize}(\operatorname{RandomCrop}(I_1)),
\qquad
\eta=0.5
\tag{13}
\]

The second stage mixes \(I'_1\) with secondary image \(I_2\), producing \(I_3\). Weights \(\beta_1\) and \(\beta_2\) control their contributions; \(\gg\) means much greater than:

\[
I_3=\beta_1 I'_1+\beta_2 I_2,
\qquad
\beta_1\gg\beta_2\in[0,1)
\tag{14}
\]

Original SCMix samples \(\eta\sim\operatorname{Beta}(\alpha,\alpha)\), meaning that \(\eta\) follows a beta distribution with shape parameter \(\alpha>0\). This \(\alpha\) is unrelated to the optimization step size. This implementation fixes \(\eta=0.5\), equivalent to \(\alpha=\infty\), as justified in Appendix D.3.

The weighting preserves dominant \(I_1\) semantics and keeps \(I_3\) compatible with \(I_1\)’s text \(T_1\), despite the added \(I_2\) features. This enriches cross-modal interaction. Variation within images and blending between images regularize against CroPA overfitting and encourage perturbations directed at invariant features. The proposed common perturbation may then apply to unseen images without image-specific optimization.

At inference, CroPA returns only a perturbed image, equivalently its perturbation. Effective cross-image transfer would make that output universal in both the original cross-prompt sense and the additional investigated image sense.

**Figure 3.** Self-mixing \(I_1\) generates \(X_1,X_2\), combined into \(I'_1\); external \(I_2\) is mixed with \(I'_1\) to produce \(I_3\).

#### 5.3.2 U8G

CutMix combines paired images \(x_A\) and \(x_B\) using binary mask \(M\), producing \(\tilde x\). The symbol \(\odot\) denotes elementwise multiplication:

\[
\tilde{x}=M\odot x_A+(1-M)\odot x_B
\tag{15}
\]

The mask \(M\in\{0,1\}^{W\times H}\) marks a randomly located rectangle in an image of width \(W\) and height \(H\). Discrete replacement retains partial views of the paired images. SCMix softly blends features within and across images; CutMix uses a hard mask.

In adversarial training, features that persist across regional combinations are expected to direct perturbations toward spatially invariant, higher-level semantics rather than localized artifacts. This may improve cross-image transferability.

### 5.4 Attack Setup and Threat Models for Transferability Experiments

The attacker’s knowledge, perturbation capability, and evaluation differ between model transfer and image transfer.

#### 5.4.1 Using X9C to explore F8V for Q7X

The attacker seeks one image perturbation that elicits a specified target text from multiple VLMs independently of the prompt. Optimization uses an accessible surrogate model, and deployment targets unknown models. White-box surrogate access includes gradients and internal representations. The surrogate is BLIP-2 or Flamingo.

Only the target’s general architecture family, such as use of a CLIP image encoder, is known. Target parameters are unavailable. CLIP names the image-encoder family discussed here.

The imperceptibility budget is \(\epsilon=16/255\) in \(L_\infty\), the largest-component perturbation norm. Intra-family testing optimizes BLIP-2 with its OPT-2.7b language model and evaluates InstructBLIP with its Vicuna-7b language model. Both use similar CLIP ViT-L/14 vision encoders, but their language models and fine-tuning objectives differ. ViT-L/14 identifies the vision-encoder configuration; OPT-2.7b and Vicuna-7b identify the language-model configurations.

Cross-architecture testing runs BLIP-2 to Flamingo and the reverse. These tests change the vision encoder, language model, and architecture.

Each task is scored independently. Success in this evaluation means that the target model fails to generate the correct output from the perturbed image, regardless of prompt.

#### 5.4.2 Investigating K3R via Augmentation Techniques

The goal is one pattern that causes predetermined target text for any image, regardless of content and prompt. This would improve efficiency over per-image optimization. The attacker has a limited training-image set and corresponding SCMix variants, plus white-box access during optimization. The perturbation must generalize to unseen images. The \(L_\infty\) budget is \(\epsilon=16/255\).

A separate test set contains no training images. Taskwise attack success rate counts failure to generate correct output text on previously unseen images, independently of content or prompt. Controlled self- and cross-mixing is hypothesized to capture image-independent model vulnerabilities rather than image-specific features.

#### 5.4.3 Evaluation Rationale for Transferability experiments

All cross-prompt and cross-model transferability results are presented under untargeted attacks. Non-targeted attack success rate is also the chosen robust, fair, comparable assessment of between-model and between-image transfer.

Distinct visual embedding mechanisms make a perturbation’s influence on final token embeddings difficult to isolate. Targeted transfer is substantially stronger between shared-encoder models such as BLIP-2 and InstructBLIP, but varies drastically across heterogeneous encoders. Direct targeted-success-rate comparisons across such architectures are therefore imprudent.

Decoder-only LLaVA and OpenFlamingo introduce visual features late, through cross-attention or token concatenation. These mechanisms respectively make visual information accessible through attention or join visual tokens to the input sequence. Visual features serve as auxiliary memory for causal generation, and their influence fades as decoding proceeds, Awadalla et al. (2023), Liu et al. (2023). Fusion-based BLIP-2, Li et al. (2023b), and InstructBLIP, Dai et al. (2023), integrate visual features early and deeply into input embeddings. This retains visual grounding throughout generation. Targeted attacks are inherently more feasible on the latter, Dai et al. (2023).

Context Injection Attack (E3W) is excluded from transfer comparisons after negligible experimental transfer. Its per-image visual-token manipulation embeds target concepts in image-dependent patterns. Gradients and token spaces differ vastly across images and concepts, and input/target-image token-space transformations are non-transferable. Its dependence on visual/textual encoder structure also prevents cross-model transfer across structural differences among BLIP2, LLaVA, Flamingo, and other LVLMs.

## 6 Results

Five state-of-the-art VLMs were evaluated: OpenFlamingo, BLIP-2, InstructBLIP, and LLaVA. The Flamingo implementation is publicly accessible OpenFlamingo-9B, Awadalla et al. (2023), with performance comparable to the original model.

BLIP-2, also written BLIP2, uses two-stage fusion. A frozen CLIP image encoder first extracts features. A Querying Transformer then processes them, Li et al. (2023b). The OPT-2.7b language model provides flexible generation and efficient task adaptation. InstructBLIP adds instruction tuning to this fusion architecture, Dai et al. (2023). It uses Vicuna-7b for task-specific instruction following and finer output control through crafted prompts.

LLaVA, Liu et al. (2023), connects a pretrained CLIP ViT-L vision encoder to the Vicuna-13B language model through a lightweight projection. Its decoder-only generation receives visual features directly as special tokens in the prompt context. This differs from fusion models’ explicit visual–textual alignment before generation.

OpenFlamingo and LLaVA represent decoder-only architectures. BLIP-2 and InstructBLIP represent fusion-based architectures. This distinction significantly affects targeted adversarial effectiveness across prompts.

### 6.1 Claim 1: Q7X achieves N6Z across target texts [Reproduced]

Flamingo-generated CroPA examples transfer across held-out prompts in VQA, classification, and captioning. Table 1 reports nonzero CroPA attack success rates across all tasks and diverse target texts, reproducing the central cross-prompt-transfer claim.

#### 6.1.1 Experimental Details

The scored task families and table codes are:

- **A7L = VQAgeneral:** general visual questioning.
- **T4B = VQAspecific:** specific visual questioning.
- **G9M = Classification:** image-label prediction.
- **D5R = Captioning:** image description.
- **C8H = Overall:** the aggregate column.

The two questioning labels distinguish how specifically a prompt pertains to a particular image.

Training follows Luo et al. (2024). Evaluation prompts differ from training prompts and span a broad range of semantic meanings. The Flamingo test measures targeted attack success rate, requiring exact target text. Appendix D.1 gives details for all reproduced experiments; Appendix E lists the prompts.

#### 6.1.2 Comparative Analysis

CroPA generally agrees with the claimed transfer behavior. However, it consistently underperforms on classification and captioning, especially for abstract target texts “too late” and “not sure,” which lack significant visual semantics. In Table 1, the CroPA “too late” row reports G9M 0.7010 and D5R 0.1790; the “not sure” row reports G9M 0.2420 and D5R 0.0010. Successful deception nevertheless persists across diverse targets, as shown by the nonzero CroPA entries.

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

**Table 1.** Targeted H2Q on Flamingo across target texts and vision–language tasks. R4K is Single-P, V9D is Multi-P, and Q7X is CroPA.

### 6.2 Claim 2: Q7X achieves the best aggregate performance across prompt counts [Reproduced]

At equal prompt counts, CroPA consistently exceeds the baselines. Additional prompts do not allow baseline saturation to overtake it (Figure 4). Increasing prompt counts raises attack success rate for all methods. Single-P intrinsically uses one prompt, so count variation is redundant for it.

The experiment accepts the original authors’ rationale for choosing “unknown”: it avoids high-frequency responses common in vision–language tasks. Prompt-count gains saturate rapidly, consistent with Luo et al. (2024) and the comparison in Figure 4.

**Figure 4.** Targeted H2Q against prompt count on Flamingo, holding attack iterations equal; Q7X significantly outperforms baselines. The target is “unknown.” R4K is the lone star.

### 6.3 Claim 3: Q7X converges towards higher H2Q as iterations increase [Reproduced]

Taskwise convergence is unstable despite exact reuse of the authors’ training/evaluation parameters and code (Figure 5). The original reported convergence metrics in Appendix A.2 differ from these results and showed convergence with more prompts. Across tasks jointly, increasing-iteration convergence follows the original aggregate trend (Figure 6).

The authors did not specify the model for this scenario. The reproduced curves use OpenFlamingo and target “unknown.”

- (a) VQA
- (b) VQA-Specific
- (c) G9M
- (d) D5R

**Figure 5.** R4K, V9D, and Q7X compared across categories.

**Figure 6.** Aggregate convergence and Q7X comparison.

### 6.4 Claim 4: Q7X outperforms baselines under few-shot settings [Reproduced]

Flamingo supports additional images as in-context examples for task adaptation. Following Luo et al. (2024), perturbations are optimized at 0-shot, with no extra images, and evaluated at 2-shot, with two extra images. Evaluation context therefore differs from optimization context.

CroPA significantly exceeds the baseline on most tasks as this context widens. In Table 2’s 2-shot rows, CroPA’s aggregate success rate is 0.6635, versus Multi-P’s 0.5730. This supports the min–max objective’s intended robustness to expanded prompt space and consequent context changes.

| Setting | Method | A7L | T4B | G9M | D5R | C8H |
|---|---|---:|---:|---:|---:|---:|
| shot=0 | V9D | 0.7240 | 0.8740 | 0.5550 | 0.2850 | 0.6095 |
| | Q7X | 0.9680 | 0.9880 | 0.7070 | 0.4200 | 0.7708 |
| shot=2 | V9D | 0.7860 | 0.9420 | 0.5630 | 0.0010 | 0.5730 |
| | Q7X | 0.9440 | 0.9940 | 0.6860 | 0.0300 | 0.6635 |

**Table 2.** Targeted H2Q with and without visual in-context learning; shot counts added images.

Changes in generation conditions make attacks harder, Luo et al. (2024). Contextual awareness moves the correct prompt space away from target-text space. Zhao et al. (2023) and Qi et al. (2023) verify that context-specific attacks such as CroPA lose effectiveness when context changes. Table 2 shows CroPA’s aggregate success rate changing from 0.7708 at 0-shot to 0.6635 at 2-shot, and captioning from 0.4200 to 0.0300.

### 6.5 Results Beyond the Original Paper

The extended experiments incorporate initialization, attention-value guidance, and input-diversity modifications into the original framework.

#### 6.5.1 Noise Initialization

The integrated initialization variant is CroPA+Init (code B8N). It is reported to outperform CroPA and Multi-P consistently across every metric, with gains in aggregate accuracy, adversarial robustness, and cross-prompt transferability (Tables 3–4). Semantic initialization is also reported to require fewer optimization steps (Figure 7). It provides a robust, adaptable alternative to random initialization rather than changing the perturbation budget.

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

Semantic starting points are deliberately closer to meaningful adversarial directions. Zhang et al. (2025) find min–max problems substantially more initialization-sensitive than ordinary minimization. They often need carefully designed starts for stable convergence. CroPA resembles a saddle-point problem, in which one variable is minimized while another is maximized. Prompt perturbations continually change its optimization landscape.

More frequent image than prompt updates create asymmetry and amplify the starting point’s importance. Chen et al. (2019) find local cycling—repeated movement around a region—in alternating-gradient min–max methods. They also find that convergence strongly depends on initial proximity to adversarially robust regions. Wu et al. (2020) find that semantic-prior perturbations transfer better across models than random starts and yield substantial success-rate improvements. The experimental observations align with that finding, particularly for the more complex bi-level optimization paths.

| Method | A7L | T4B | G9M | D5R | C8H |
|---|---:|---:|---:|---:|---:|
| V9D | 0.7053 | 0.8807 | 0.7001 | 0.5553 | 0.7104 |
| Q7X | 0.8723 | 0.9457 | 0.7518 | 0.6138 | 0.7958 |
| E3W | 0.2985 | 0.2281 | 0.4857 | 0.4687 | 0.370 |
| B8N | 0.9145 | 0.9342 | 0.8982 | 0.9464 | 0.9209 |

**Table 4.** Average targeted H2Q on BLIP-2 across visual-questioning, G9M, and D5R tasks. Best task values are highlighted in bold. E3W, Yang et al. (2024), is the additional baseline.

Gradient attacks ordinarily acquire target semantics and theoretically should not require semantic initialization. Min–max complexity weakens that alignment. Injecting target semantics at initialization is described as necessary to reinforce alignment and assist convergence. Table 4 reports aggregate targeted success rates of 0.7958 for CroPA and 0.9209 for CroPA+Init.

Feature uptake is quantified between target text and learned noise using a CLIP score (code Y2K). This score measures their feature similarity. Comparing the base perturbation with its initialized counterpart assesses how much target information perturbations acquire, rather than measuring only task success.

| Method | Y2K |
|---|---:|
| Q7X | 21.80 |
| B8N | 23.09 |

Target text: “Bomb”.

**Figure 7.** (a) Aggregate convergence of Q7X and B8N. (b) Learned-perturbation/target feature similarity quantified by Y2K.

#### 6.5.2 Investigating K3R and Image Augmentation

Learning across images does not yield effective universality for unmodified CroPA. Prompt-directed signals remain tightly linked to the originating image’s low-level features. Table 5 reports an aggregate untargeted cross-image success rate of 0.4355 for base CroPA. This weakens the original claim that learning perturbations across images would establish cross-image transferability.

Both input-mixing approaches give measurable gains: Table 5 reports aggregate rates of 0.7055 for SCMix and 0.5565 for CutMix. However, the gap between cross-image methods and CroPA applied to a single image remains a fundamental limitation. CroPA excels at prompt robustness, not effective image-transferring adversarial attacks.

CutMix is believed to outperform SCMix for adversarial transfer. Region-based mixing retains distinct, non-overlapping spatial areas. This encourages modular, independently manipulable features and greater cross-input generalization. Table 5 gives the comparison.

| Method | VQA | T4B | G9M | D5R | C8H |
|---|---:|---:|---:|---:|---:|
| Q7X (Base) | 0.3500 | 0.4920 | 0.4000 | 0.5000 | 0.4355 |
| Q7X with J6T | 0.6560 | 0.7160 | 0.5000 | 0.9500 | 0.7055 |
| Q7X with U8G | 0.4520 | 0.7240 | 0.5000 | 0.5500 | 0.5565 |

**Table 5.** Untargeted K3R H2Q, best across all iterations. Unmixed Q7X peaks at 1700 iterations; J6T and U8G exceed it significantly earlier, at 1500 and 1600 respectively. Experimental parameters: Appendix D.3.

#### 6.5.3 Guiding perturbations via Target Value Vectors

CroPA+D-UAP (P2F) greatly improves the existing algorithm across tasks, especially general VQA and captioning. In Table 6, general-VQA success is 0.9420 for P2F versus 0.8176 for CroPA; captioning success is 0.9810 versus 0.6960.

The target “Bomb” is harmful text expected inherently to activate guardrails, or mechanisms intended to prevent harmful outputs. The improvement is maintained for such worse expected harmful-word cases and vulnerable scenarios. Tables 6–7 report the targeted results.

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

Value guidance significantly improves vanilla CroPA transfer, especially classification and captioning, with modest VQA gain. Table 8’s BLIP2-to-InstructBLIP rows report classification rates of 0.6720 for CroPA and 0.8800 for P2F, and captioning rates of 0.6150 and 0.8880. VQA prompts occupy a more complex embedding space than the smaller output spans of targeted classification and captioning.

Encoder dependence verifies the earlier limitation hypothesis. BLIP2 perturbations are passably transferable to same-encoder InstructBLIP and fail to transfer to OpenFlamingo. The corresponding P2F aggregate values in Table 8 are 0.6850 and 0.2335. Training on Flamingo and applying perturbations to the other models likewise gives insignificant attack success rates (Table D.4.5, Appendix D.4.5; also Table 8’s Flamingo-origin rows). Experimental details appear in Appendix D.4.

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

**Table 8.** Cross-model tests. A setting’s left model supplies optimized perturbations; the right model is tested.

For a given target text, vision-encoder value vectors remain approximately consistent across input images. They reflect encoder architecture and pretraining. This consistency may constrain perturbation diversity and the generation of highly varied adversarial examples. Target-text choice strongly influences outcomes relative to input-image-specific features.

## 7 Discussion

### 7.1 Observations with reproducibility

Reproduction used the authors’ public repository and original model configurations for faithful baseline comparison. The implementation was generally well documented. Nevertheless, BLIP-2 and InstructBLIP code produced multiple runtime errors. Substantial debugging and adjustment were required to run it.

### 7.2 Noise Initialization findings

The initialization hypothesis includes reduced cross-modal as well as cross-prompt transfer when initialization ignores semantics. Experiments across all evaluated benchmarks are described as significantly favoring semantic noise over CroPA and random-initialized baselines (Tables 3–4; Figure 7).

CroPA perturbations already acquire target context, as expected of gradients. Initialization produces a modest CLIP-score increment, from 21.80 to 23.09 for “Bomb” in Figure 7’s associated table. This is not a significant semantic-feature upgrade. It is interpreted as a subtle shift closer to adversarially vulnerable regions. Convergence before and after initialization in Figure 7 aligns with the hypothesis that a starting point nearer meaningful adversarial directions converges faster.

### 7.3 Evaluation of Transferability Experiments

#### 7.3.1 K3R

SCMix improves cross-image transferability particularly for VQA, specific VQA, and captioning. Table 5 reports 0.6560, 0.7160, and 0.9500 for SCMix, compared with base values of 0.3500, 0.4920, and 0.5000. CutMix also improves the base image-transfer version, with aggregate success of 0.5565 versus 0.4355.

The proposed account favors SCMix’s explicit spatial/scale invariance. Random cropping and resizing vary the same image before soft mixing with another image. Semantically aligned features become less sensitive to position and scale, easing cross-input generalization.

Both augmentations preserve original semantic information while adding image diversity. Their proposed benefit is learning cross-image features, but augmentation alone does not eliminate the limitation. Further work should combine augmentation with more complex, layered methods. These include attacks that align features in latent representations—the model’s internal representation space—or perturb that space directly.

#### 7.3.2 F8V

Doubly-UAP guidance is described as a significant advance robust to harmful target text and guardrail activation, as evaluated with “Bomb” in Tables 6–7. Encoder-specific dependence remains the obstacle to encoder-agnostic attacks. It narrows perturbation diversity and transfer across heterogeneous architectures (Table 8).

Despite that limitation, the framework is presented as progress toward universal attacks effective across prompts and images, and toward improved VLM generalizability and robustness. Context Injection Attack cannot supply comparable transfer results and is inherently non-transferable, as discussed in Section 5.4.3.

### 7.4 Trade-offs Between Different Types of Adversarial Transferability

#### 7.4.1 N6Z vs. F8V

Competing text embeddings implicitly anchor original CroPA perturbations to the target architecture. Moving attacks between models causes a significant attack-success-rate drop, as reflected in Table 8.

Doubly-UAP targets potentially generalizable encoder components rather than model-specific features. It improves related-architecture transfer: Table 8’s BLIP-2-to-InstructBLIP aggregate rates are 0.5065 for CroPA and 0.6850 for P2F. Across fundamentally different architectures, BLIP-2 to Flamingo, it offers only modest gains: 0.1435 and 0.2335. Simultaneously optimizing prompt diversity and model diversity is inherently tense.

#### 7.4.2 N6Z vs. K3R

Image-specific coupling supports strong cross-prompt transferability and fundamentally limits cross-image transferability. The two are competing optimization objectives. Augmentation diversifies optimization and yields promising improvements, as in Table 5, without resolving that fundamental tension.

#### 7.4.3 Role of Initialization in Transferability Balance

Semantic anchoring raises attack success rate without sacrificing cross-prompt transferability. Tables 3–4 and Figure 7 indicate that starting-point selection substantially influences the achievable transfer balance.

Unlike modifications to the optimization process, better initialization allows convergence toward more general adversarial directions that remain effective across prompts. Even then, optimizing a perturbation for one transfer dimension inevitably sacrifices effectiveness in others.

#### 7.4.4 Implications for Future Research

No single methodology can maximize all transfer forms simultaneously. Controlled balancing may require multi-objective optimization, which optimizes several goals together, or architecturally aware attack designs. Pushing individual dimensions is insufficient for a comprehensive solution without fundamentally rethinking perturbation interactions with model architecture and data representation.

Cross-prompt transferability remains underexplored, with little literature on prompt-induced vulnerability. The resulting understanding gap matters for secure deployment.

### 7.5 Broader Impact

Examples with high cross-prompt transferability can elicit malicious or misleading outputs from harmless queries. This enables misinformation or subversion of VLM-dependent content-generation and decision systems.

Imperceptible perturbations also offer privacy protection. Embedded in personal images, they can force predetermined nonsensitive text and thwart unauthorized extraction of confidential information. This is a proactive security mechanism as visual data becomes increasingly exploitable.

The initialization and loss refinements increase attack success rate and cross-model/cross-image generalizability, as evaluated in Tables 3–8. They strengthen cross-prompt attacks while opening paths toward more resilient defenses. Deeper understanding is necessary for robust VLMs, effective countermeasures, user-data protection, and output integrity.

### 7.6 Limitations and Future Work

Computational scarcity repeatedly hindered successful key-result replication and extensive ablations, or experiments isolating individual method components. Large Flamingo and BLIP2 attack training and evaluation require substantial GPU memory and processing. For example, the reported 50-image run took over 6 hours on a 48GB L40S with 95% VRAM utilization. These requirements limited experimental scale and depth, extensive hyperparameter tuning, and transfer testing over wider architecture ranges.

A planned ensemble-model transfer investigation, using multiple models, could have deepened generalizability analysis. It exceeded available memory and compute and was infeasible on the existing infrastructure. Future work includes encoder-agnostic cross-model transferability, stronger cross-image transferability, and query-based optimization for practical applicability.

### 7.7 Communication With Original Authors

No direct communication with the original authors was established during replication. The GitHub issue concerning BLIP-2 and InstructBLIP remains unresolved at writing.
