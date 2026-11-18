**Summary.** This study reproduces “An Image is Worth 1000 Lies: Adversarial Transferability Across Prompts on Vision-Language Models.” It tests whether an image attack remains effective when the text prompt changes. The study reproduces Cross-Prompt Attack results across target texts, prompt counts, attack iterations, and few-shot settings. It also tests semantic noise initialization, target-value-vector guidance, and shared perturbations learned with image augmentation. The paper reports improved attack success from these extensions. It also reports unstable taskwise convergence, limited transfer across images, and dependence of cross-model transfer on vision-encoder architecture. Evaluation includes Flamingo, BLIP-2, InstructBLIP, and extended LLaVA experiments.

**Glossary.** Definitions below follow the paper's usage. Symbols with different meanings in different equations are identified separately.

- **Adversarial attack / adversarial example** — An image perturbation, or an image containing it, intended to induce an incorrect, harmful, or specified model output.
- **Perturbation / adversarial noise** — A modification added to an image or text input during attack optimization.
- **Targeted attack / targeted success** — An attack whose success requires generation of the exact predetermined target text.
- **Untargeted / non-targeted attack or success** — An attack evaluated by departure from the clean-image prediction or failure to generate the correct output.
- **White-box access** — Access to model gradients and internal representations during optimization.
- **Black-box attack** — An attack studied without access to the target model's parameters, including query- and transfer-based attacks.
- **Surrogate model** — An accessible model used to optimize perturbations before testing them on another model.
- **Transferability** — Continued attack effectiveness across models, images, tasks, or textual prompts.
- **F8V / Cross-Model Transferability** — Survival of adversarial examples across model architectures.
- **K3R / Cross-Image Transferability** — Survival of adversarial effectiveness across multiple images.
- **N6Z / Cross-Prompt Transferability** — Adversarial effectiveness across varying textual prompts.
- **Cross-task transferability** — Effectiveness on visual tasks other than the task used to craft an attack.
- **Z4P / Universal Adversarial Perturbations / UAPs** — Shared perturbations that generalize across images instead of being separately optimized for each image.
- **Q7X / Cross-Prompt Attack / CroPA** — The method that antagonistically optimizes image and prompt perturbations to promote image-attack transfer across prompts.
- **R4K / Single-P** — The baseline that optimizes an image perturbation using one prompt.
- **V9D / Multi-P** — The baseline that optimizes an image perturbation using a collection of prompts.
- **E3W / Context Injection Attack / Contextual Injection Attack / CIA** — A gradient-based attack that injects target tokens into visual and textual contexts.
- **S3D / Projected Gradient Descent / PGD** — The iterative gradient-based optimization used with projection into the allowed perturbation region.
- **W5J / Noise Initialization via Vision Encoding Optimization** — Initialization that aligns image-encoder features with a diffusion-generated target image.
- **L6V / Mean Squared Error / MSE** — The encoder-output error criterion used for target-image feature alignment.
- **B8N / CroPA+Init** — Q7X combined with vision-encoder-based noise initialization.
- **X9C / Doubly-Universal Adversarial Perturbation / Doubly-UAP / D-UAP** — The component-selection method whose attention-component loss is adapted for value-vector guidance.
- **P2F / CroPA+D-UAP / re-CroPA** — Q7X combined with the adapted value-vector loss.
- **U8G / CutMix** — Image augmentation that replaces a rectangular region with a patch from another image.
- **J6T / SCMix** — Self- and cross-mixing augmentation using same-image crops and mixing with another image.
- **M7A / Effective and Transferable Universal Adversarial Attack / ETU** — The framework containing SCMix and additional local utility reinforcement.
- **H2Q / Attack Success Rate / ASR** — The effectiveness quantity applied to targeted and untargeted attacks.
- **G9M / Classification** — Image-label prediction.
- **D5R / Captioning** — Image description.
- **VQA / visual question answering** — Answering questions about visual input.
- **A7L / VQAgeneral** — General visual questioning.
- **T4B / VQAspecific / VQA-Specific** — Visual questioning with prompts specific to a particular image.
- **C8H / Overall** — The aggregate result column.
- **Y2K / CLIP score** — The feature-similarity measure between target text and learned perturbation.
- **VLM / vision-language model** — A model that processes visual and textual inputs.
- **LVLM / large VLM** — A large vision-language model whose visual processing and language responses are studied.
- **VLP / Vision-Language Pretraining** — The model setting used in the cited universal-perturbation studies.
- **Flamingo / OpenFlamingo / OpenFlamingo-9B** — The Flamingo model and its publicly accessible implementation used in the experiments.
- **BLIP / BLIP-2 / BLIP2** — Vision-language models studied in prior attacks; BLIP-2 uses a frozen image encoder and a Querying Transformer.
- **InstructBLIP** — An instruction-tuned fusion architecture based on BLIP-2.
- **LLaVA** — A model connecting pretrained CLIP ViT-L to Vicuna-13B through a lightweight projection.
- **CLIP / CLIP ViT-L / CLIP ViT-L/14** — The image-encoder names used in the described model configurations and feature comparisons.
- **OPT-2.7b** — The language model used with BLIP-2.
- **Vicuna-7b / Vicuna-13B** — The language models used with InstructBLIP and LLaVA, respectively.
- **CNN–RNN systems** — The lightweight systems without in-context learning addressed by earlier attacks.
- **Vision encoder / \(f_v\)** — The VLM component that produces image representations.
- **Text encoder** — The component frozen in the value-guided approach to isolate visual-processing effects.
- **Language model / decoder** — The component that generates text from the model's input representations.
- **Querying Transformer** — The component that processes extracted image features in BLIP-2's two-stage fusion.
- **Fusion-based architecture** — An architecture described as integrating visual features early and deeply into input embeddings.
- **Decoder-only architecture** — An architecture described as supplying visual features through cross-attention or prompt tokens for causal generation.
- **Visual tokens / embeddings / representations** — Encoded visual information used by language components to access and reason about images.
- **Attention weights** — Components controlling patch-to-patch focus, relevance, and interaction.
- **Value vectors** — Attention components containing each image patch's actual information.
- **Attention head** — The indexed attention component supplying a value vector in Equation (11).
- **Semantic alignment / anchoring** — Alignment of perturbation or image features with the target text or its reference image.
- **Bi-level / min–max optimization** — The opposing image-minimization and prompt-maximization problem used by Q7X.
- **In-context learning / few-shot settings** — Task adaptation using additional examples in the input context.
- **0-shot / 2-shot / shot** — No added images, two added images, or the number of added images, respectively.
- **Input diversity / image augmentation** — Image transformations or mixing used to reduce image-specific overfitting.
- **Self-mixing / cross-mixing** — Mixing transformed views of one image / mixing that result with another image.
- **Local utility reinforcement** — The additional ETU component omitted when SCMix alone is compared.
- **Guardrails** — Model protections expected to activate for harmful target text such as “Bomb.”
- **Stable Diffusion XL / SDXL / \(\mathcal D\)** — The pretrained diffusion model used to synthesize a target image from text.
- **GPU / L40S / VRAM** — The computing device, the device named for the reported run, and its reported memory utilization.
- **\(f\)** — The model processing visual and textual inputs.
- **\(x_v,x_t,x_t^i\)** — Visual input, textual input, and the \(i\)-th textual prompt.
- **\(X_t=\{x_t^1,\ldots,x_t^k\}\)** — The collection of \(k\) prompts.
- **\(T\)** — Predetermined target text, generated independently of the input prompt in the targeted scenario.
- **\(L\)** — Language-modeling loss in the attack objectives; in Equation (10), the set of selected layers.
- **\(\operatorname{Loss}\)** — The loss applied to value vectors in Equation (10).
- **\(\delta_v,\delta_t,\delta\)** — Image perturbation, prompt perturbation, and a perturbation written without a modality subscript.
- **\(x'_v,x_t'^i\)** — The adversarial image and the initialized or updated adversarial prompt in Algorithm 1.
- **\(g_v,g_t\)** — Image and prompt gradients in Algorithm 1.
- **\(\epsilon_v,\epsilon\)** — The human-imperceptibility bound or maximum allowed perturbation size.
- **\(\|\cdot\|_p,\ell_\infty,L_\infty,\|\cdot\|_\infty\)** — Norm notation used to constrain perturbations; the infinity-norm budget is used in the experiments.
- **\(\|\cdot\|_2^2\)** — Squared encoder-representation error in the initialization objective.
- **\(K,\mathrm{step}\)** — The attack iteration count and the current iteration in Algorithm 1.
- **\(N\)** — The adversarial-prompt update interval in Algorithm 1; the number of heads in Equation (11).
- **\(\alpha_1,\alpha_2\)** — The image and prompt step sizes in Algorithm 1.
- **\(\alpha\)** — The refinement step size in Equation (8); the Beta-distribution parameter in the original SCMix sampling rule.
- **\(\nabla,\operatorname{sign},\operatorname{mod}\)** — The gradient, gradient-sign operation, and interval test used in Algorithm 1.
- **\(\operatorname{Clip}_{x_v,\epsilon}\)** — Projection within the \(\epsilon\)-ball around the original visual input.
- **\(\min,\max,\arg\min,\arg\max\)** — The minimization, maximization, and optimizer notation in the attack objectives.
- **\(\sum,i,k\)** — Summation and the indices or counts used for prompts or attention heads.
- **\(\oplus\), asterisks, solid arrows, dashed arrows** — Elementwise addition, fixed parameters, forward passes, and backpropagation in Figure 1.
- **\(\mathbf{x}_{\mathrm{target}},\mathbf z,\theta_{\mathrm{SDXL}}\)** — The synthesized target image, Gaussian-sampled input, and pretrained diffusion weights.
- **\(\mathbf x,\boldsymbol\delta_{\mathrm{init}}\)** — The original image and its optimized initialization perturbation.
- **\(\mathbf{x}_{\mathrm{adv}}^{(t)},t\)** — The adversarial image at refinement iteration \(t\).
- **\(\mathcal L_{\mathrm{L6V}}\)** — The squared error between adversarial-image and target-image encoder representations.
- **\(\Pi_{\mathcal B_\epsilon(\mathbf x)}\)** — Projection into the admissible \(\ell_\infty\) ball around the original image.
- **\(\delta^*,V_l(x),l,|L|\)** — The optimized perturbation, input value vectors at layer \(l\), a layer index, and the number of selected layers in Equation (10).
- **\(x_i,T_i\)** — The input and target-associated references named in describing the adapted component loss.
- **\(V_i(x+\delta),V_t\)** — The perturbed input's value vector at head \(i\) and the reference image's target value vector.
- **\(L_{\mathrm{X9C}},L_{\mathrm{Q7X}},L_{\mathrm{P2F}},\lambda\)** — The value-vector loss, Q7X loss, combined loss, and relative value-loss weight.
- **Cosine-similarity loss / \(\cdot\)** — The normalized value-vector comparison in Equation (11), using a vector dot product.
- **\(I_1,I_2,I'_1,I_3\)** — The base image, secondary image, self-mixed image, and final cross-mixed image.
- **\(X_1,X_2,X_i\)** — Cropped and resized views of \(I_1\) in Equation (13).
- **\(\operatorname{RandomCrop},\operatorname{Resize}\)** — The crop and resize transformations used for self-mixing.
- **\(\eta,\operatorname{Beta}(\alpha,\alpha)\)** — The self-mixing weight and its original sampling distribution; this implementation fixes \(\eta=0.5\).
- **\(\beta_1,\beta_2,\gg,[0,1)\)** — Cross-mixing weights, their stated dominance relation, and the stated interval in Equation (14).
- **\(T_1\)** — The text associated with the dominant base-image semantics.
- **\(\tilde x,x_A,x_B\)** — The CutMix output and its paired source images.
- **\(M,\{0,1\}^{W\times H},W,H,\odot\)** — The binary rectangular mask, its stated shape, its dimensions, and the masked elementwise multiplication in Equation (15).
- **\(\infty\)** — The value assigned to \(\alpha\) when fixed \(\eta=0.5\) is described as equivalent to the original sampling rule.

# Revisiting CroPA: A Reproducibility Study and Enhancements for Cross-Prompt Adversarial Transferability in Vision-Language Models

| Author | Email | Affiliation | Author notation |
|---|---|---|---|
| Atharv Mittal | atharv_m@mfs.iitr.ac.in | Mehta Family School of Data Science and Artificial Intelligence; Indian Institute of Technology, Roorkee | ††thanks: All authors contributed equally |
| Agam Pandey | agam_p@ce.iitr.ac.in | Department of Civil Engineering; Indian Institute of Technology, Roorkee | 11footnotemark: 1 |
| Amritanshu Tiwari | amritanshu_t@mfs.iitr.ac.in | Mehta Family School of Data Science and Artificial Intelligence; Indian Institute of Technology, Roorkee | 11footnotemark: 1 |
| Sukrit Jindal | sukrit_j@mfs.iitr.ac.in | Mehta Family School of Data Science and Artificial Intelligence; Indian Institute of Technology, Roorkee | 11footnotemark: 1 |
| Swadesh Swain | swadesh_s@ece.iitr.ac.in | Department of Electronics and Communication; Indian Institute of Technology, Roorkee | 11footnotemark: 1 |

###### Abstract

*The study validates the original cross-prompt attack results and reports extensions using initialization, shared perturbations, and attention-directed loss.*

This reproducibility study examines “An Image is Worth 1000 Lies: Adversarial Transferability Across Prompts on Vision-Language Models.” It validates Q7X (Cross-Prompt Attack, CroPA), which antagonistically optimizes image and prompt perturbations. It also validates Q7X's superiority to existing baselines in N6Z (Cross-Prompt Transferability), meaning effectiveness across varying textual prompts. Semantic meaning and target-word frequency do not determine that transferability.

The extensions test initialization, common perturbations across images, and attention-directed loss. Evaluation covers Flamingo, BLIP-2, InstructBLIP, and extended LLaVA experiments. These are VLMs (vision-language models), which process visual and textual inputs. The original results are validated, and the enhancements consistently increase adversarial effectiveness. The resulting framework generates more robust transferable adversarial examples: perturbed images intended to induce incorrect or specified outputs. It bears on real-world VLM security.

Code: https://github.com/Swadesh06/Revisting_CroPA

Reviewed on OpenReview: https://openreview.net/forum?id=5L90cl0xtf

**Claims and evidence**
- The study validates Q7X's cross-prompt results: Tables 1–2 and Figures 4–6; detailed evidence appears in Sections 6.1–6.4.
- The extensions improve adversarial effectiveness: Tables 3–8; detailed evidence appears in Section 6.5.
- Semantic meaning and target-word frequency do not determine transferability: this is stated in the abstract; no separate numerical frequency comparison is supplied.

## 1 Introduction

*The paper distinguishes transfer settings and introduces the replication tests and extensions.*

Subtle image perturbations—small modifications added to an image—can induce incorrect or harmful VLM outputs. Textual input adds attack vectors beyond those of traditional vision models. Large VLMs transformed computer vision through G9M (Classification, or image-label prediction), D5R (Captioning, or image description), and VQA (visual question answering). They enabled deeper investigation of visual–linguistic interaction. Zhao et al. (2023), Qi et al. (2023), Zhang et al. (2022a), and Carlini et al. (2024) demonstrated vulnerabilities.

The paper distinguishes three transfer settings:

- **F8V (Cross-Model Transferability):** adversarial examples survive across model architectures. This setting is commonly called transferability.
- **K3R (Cross-Image Transferability):** adversarial effectiveness survives across multiple images.
- **N6Z:** adversarial effectiveness survives across varying textual prompts.

K3R admits Z4P (Universal Adversarial Perturbations, UAPs): shared perturbations rather than a separately optimized perturbation for each image.

Luo et al. (2024) introduced N6Z and competing learnable prompts. Their image attacks retain effectiveness regardless of textual input across G9M, D5R, and VQA. Earlier work treated visual and textual adversarial perturbations independently. Zhao et al. (2023) analyzed BLIP and BLIP-2 with query- and transfer-based black-box attacks, which do not use target parameters. Schlarmann & Hein (2023) examined targeted and untargeted white-box attacks, which use model access. Qi et al. (2023) showed adversarial images eliciting harmful instruction execution. Tu et al. (2023) systematically evaluated attacks on both modalities.

The replication tests superiority across target texts, rapid saturation as prompts are added, robustness to added in-context examples, and convergence over attack iterations. In-context examples are additional examples supplied in the input context. The replication also analyzes individual tasks beyond the original aggregate analysis.

The extensions cover:

- Initialization intended to improve H2Q (Attack Success Rate, ASR), the measure of attack effectiveness, and transferability.
- Shared perturbations on the Q7X backbone, with image augmentation to promote transfer.
- Target value vectors, the attention components containing each image patch's information.

Attention components govern patch interaction. Targeted manipulation is proposed to improve generalization and H2Q. The analyses investigate both the mechanisms of these enhancements and the nature of VLM adversarial vulnerabilities.

## 2 Related Works

*Prior work establishes adversarial attacks and transferability, while VLMs introduce additional challenges and opportunities.*

Szegedy et al. (2013) and Goodfellow et al. (2014) introduced adversarial attacks on deep neural networks. Applying these attacks to VLMs adds challenges and opportunities.

### 2.1 Adversarial Transferability

*Adversarial examples can transfer across visual tasks, model architectures, and images.*

Examples crafted for classification also affect object detection and other visual tasks. This establishes cross-task transferability: effectiveness on tasks other than the one used to craft an attack. It accompanies F8V and K3R in Naseer et al. (2019b), Naseer et al. (2019a), Lu et al. (2020), and Salzmann et al. (2021). This multidimensional behavior raises robustness questions for modern machine-learning systems.

Liu et al. (2017) and Tramèr et al. (2017) established that attacks crafted on one model often deceive other models, including different architectures. Moosavi-Dezfooli et al. (2017) and Mopuri et al. (2017) established image-generalizing perturbations and Z4P.

### 2.2 Adversarial Attacks on Vision-Language Models

*Earlier attacks targeted captioning and visual questioning, while more recent work examined large VLMs and model transfer.*

Early task-specific D5R attacks manipulated outputs in Xu et al. (2019), Zhang et al. (2020), Aafaq et al. (2021), and Chen et al. (2017). Targeted VQA attacks misled attention mechanisms in Xu et al. (2018), Kaushik et al. (2021), Kovatchev et al. (2022), Li et al. (2021), Sheng et al. (2021), and Zhang et al. (2022b).

These attacks mainly addressed lightweight CNN–RNN systems without in-context learning. This restricted their applicability to contemporary VLMs. Zhao et al. (2023) studied modern large VLMs, BLIP and BLIP-2, in a black-box setting. Their emphasis was F8V rather than N6Z.

### 2.3 Cross-Prompt Adversarial Transferability

*Changing prompts create an additional attack surface, and competing-prompt optimization differs from context injection.*

No credible benchmarks or baselines have yet been established for this relatively new concept. Yang et al. (2024) nevertheless provides an additional experimental comparator: E3W (Context Injection Attack, CIA, also called Contextual Injection Attack). E3W uses gradient-based perturbations to inject target tokens into visual and textual contexts. This differs from the competing-prompt optimization pioneered by Q7X in Luo et al. (2024).

Prompt-driven adaptation lets large VLMs switch tasks dynamically, unlike fixed-task traditional vision models: Li et al. (2023b), Awadalla et al. (2023), Li et al. (2023b), Zhu et al. (2023), and Liu et al. (2023). Attack validity across changing prompts is an additional attack surface. Making the modalities compete, rather than manipulating them independently, establishes a distinct paradigm for evaluating VLM robustness.

### 2.4 Enhancing Adversarial Transferability

*Gradient-based optimization and input augmentation inform the proposed Q7X enhancements.*

Gradient misalignment limits transferability. It has motivated optimization strategies in Li et al. (2023a) and Wang et al. (2023). Input diversity in Xie et al. (2019) and augmentation in Wang et al. (2021) also improve attack robustness. These findings inform the multidimensional Q7X enhancements.

## 3 Scope of reproducibility

*The study tests four original claims and extends evaluation to transfer across models and images.*

Faithful reproduction of Luo et al. (2024) requires their experimental procedures. The study tests these named claims:

- Claim 1: Q7X achieves N6Z across target texts.
- Claim 2: Q7X has the best aggregate performance across prompt counts.
- Claim 3: Q7X converges toward higher H2Q with increasing iterations.
- Claim 4: Q7X outperforms baselines in few-shot settings, which add examples to the input context.

The extended scope includes F8V, K3R, and comparison with other N6Z methods. Images generated through Q7X are tested for consistent deception across diverse VLMs, input prompts, and model parameters. The scope therefore concerns broader attack robustness and versatility, beyond validating the original prompt-transfer improvement.

## 4 Methodology

*The method formulates targeted and untargeted attacks, defines the baselines, and describes Q7X's competing image and prompt updates.*

### 4.1 Problem Formulation

*Targeted attacks seek exact target text, while untargeted attacks seek a change from the clean-image prediction.*

Targeted success requires exact target-text generation. Untargeted success requires any departure from the clean-image prediction. H2Q is applied to both reproduced scenarios. The objective is language modeling over prompt instances rather than a vision-only objective.

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

Here:

- \(f\) is the model processing visual input \(x_v\) and textual input \(x_t\).
- \(\delta_v\) is the image perturbation.
- \(x_t^i\) is the \(i\)-th prompt, and \(k\) is the number of prompt instances summed.
- \(L\) is language-modeling loss.
- \(T\) is predetermined target text, generated independently of the input prompt in the targeted scenario.
- \(\min\) and \(\max\) specify minimization and maximization.

The untargeted objective maximizes discrepancy between clean and adversarial outputs. The human-imperceptibility bound is \(\|\delta_v\|_p\leq\epsilon_v\), where the norm bounds the image perturbation and \(\epsilon_v\) is its allowed size. This adapts established adversarial practice to multimodal real-world applicability, Carlini et al. (2024). The formulation extends the neural-network vulnerability setting of Goodfellow et al. (2014) to prompt variation, Luo et al. (2024).

### 4.2 Baseline Approaches

*The baselines optimize an image perturbation using either one prompt or a collection of prompts.*

Luo et al. (2024) introduced reference attacks in the absence of established N6Z baselines. R4K (Single-P) optimizes with one prompt. V9D (Multi-P) optimizes with a collection of prompts.

\[
\min_{\delta_v}L\bigl(f(x_v+\delta_v,x_t),T\bigr)
\tag{3}
\]

\[
\min_{\delta_v}\sum_{i=1}^{k}
L\bigl(f(x_v+\delta_v,x_t^i),T\bigr)
\tag{4}
\]

The collection is \(X_t=\{x_t^1,x_t^2,\ldots,x_t^k\}\). The one-prompt case is the simplest baseline. Untargeted forms invert the targeted objectives. They maximize clean/perturbed output discrepancy rather than target-text likelihood.

In Figure 1, \(\delta_t\) is a prompt perturbation. The vision encoder produces image representations, and the language model generates text from the processed inputs.

**Figure 1.** Luo et al. (2024) framework: image \(\delta_v\) and prompt \(\delta_t\) perturbations act antagonistically, minimizing and maximizing language-modeling loss respectively. Solid arrows denote the forward pass through vision encoder and language model; dashed arrows denote backpropagation, with configurable perturbation-update frequencies. Asterisks mark fixed model parameters; \(\oplus\) denotes elementwise addition.

### 4.3 Cross Prompt Attack

*Q7X optimizes opposing image and prompt perturbations but retains only the image perturbation for evaluation.*

Evaluation discards prompt perturbations. It applies only the optimized image perturbation. Effectiveness must therefore reside in the image perturbation's inherent transferability, not in modified prompts.

Q7X jointly optimizes \(\delta_v\) and \(\delta_t\). Their objectives are antagonistic rather than collaborative. The aim is to force stronger image transferability across textual prompts.

\[
\min_{\delta_v}\max_{\delta_t}
L\bigl(f(x_v+\delta_v,x_t+\delta_t),T\bigr)
\tag{5}
\]

The implementation uses iterative image-gradient descent and prompt-gradient ascent, with S3D (Projected Gradient Descent, PGD), following Madry et al. (2017). S3D projects updates into the permitted perturbation region. Image updates occur \(N\) times per prompt update, where \(N\) is the adversarial-prompt update interval. The frequency is configurable; the two modalities are not updated simultaneously at equal frequency.

In Algorithm 1, \(x'_v\) is the adversarial image, \(x_t'^i\) is an adversarial prompt, \(K\) is the iteration count, and \(\alpha_1,\alpha_2\) are the image and prompt step sizes. The gradients are \(g_v,g_t\); \(\nabla\) computes a gradient, \(\operatorname{sign}\) takes its sign, and \(\operatorname{mod}\) tests the update interval. \(\operatorname{Clip}_{x_v,\epsilon}\) projects within the allowed \(\epsilon\)-ball around \(x_v\).

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

*The extensions address task inconsistency and transfer limitations through initialization, value-vector guidance, and image augmentation.*

Q7X performs poorly on G9M and D5R and has limited F8V. Its S3D optimization effectively raises single-model H2Q. However, dependence on that model's loss feedback constrains transferability. This is a recognized gradient-attack limitation: Qin et al. (2024), Li et al. (2024), Wang et al. (2023), and Li et al. (2023a). The original authors recognize it, and Table 9 in Appendix A.1 verifies it.

The extensions address task inconsistency and transfer dimensions. K3R could remove separate per-image training. F8V could provide still greater efficiency. Transfer exploration is not a core proposal of Luo et al. (2024).

The table below retains resource costs and reported quantitative characterizations separately from their comparisons. GPU denotes the computing device; L40S is the named device for the run, and VRAM is its reported memory utilization. P2F (CroPA+D-UAP, also re-CroPA) combines Q7X with value-vector guidance. A7L (VQAgeneral) means general visual questioning. J6T (SCMix) is self- and cross-mixing augmentation. U8G (CutMix) replaces image regions with patches from another image.

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

*The proposed initialization aligns image features with a diffusion-generated target image before attack optimization.*

W5J (Noise Initialization via Vision Encoding Optimization) is a diffusion-anchored method. It replaces target-agnostic random noise with target-image feature alignment. It retains imperceptibility while supplying semantically meaningful adversarial directions at the outset. Conventional initialization neither guarantees such directions nor avoids reduced H2Q and N6Z.

Q7X shares random initialization with many gradient attacks. Ignorance of target semantics is associated with instability and slower convergence, particularly when VLMs require semantic prompt alignment. The bi-level framework—the opposing image and prompt optimization—is empirically sensitive to initialization (Section 6.5.1; Figure 7; Table 3). The hypothesis is that semantically informed perturbations improve attack efficacy over purely random perturbations under the same imperceptibility requirement.

A state-of-the-art diffusion model synthesizes a target image. Its encoder representation anchors initial noise to the objective. An encoder-output error loss enforces semantic similarity while retaining adversarial effectiveness. L6V (Mean Squared Error, MSE) is this optimization criterion.

The method is presented as improving adversarial robustness, strengthening perturbations and N6Z, and reducing the steps needed for convergence.

#### 5.1.1 Noise Initialization

*Preliminary experiments favor semantic initialization while retaining visual fidelity and strict perturbation budgets.*

Preliminary BLIP-2 and InstructBLIP experiments favor semantic over random noise. Diffusion-based anchoring improves encoder-space alignment while preserving adversarial-example visual fidelity and strict perturbation budgets. Its extra single-image computation is limited to the separately stated single-GPU overhead.

**Claims and evidence**
- Preliminary BLIP-2 and InstructBLIP experiments favor semantic noise: the section reports this result without separate numerical rows for the preliminary experiments; BLIP-2 comparisons appear in Table 3.
- The stated extra computation is limited: Section 5's characterization table, “Added initialization time per image on a single GPU” row, **20-25 seconds**.

#### 5.1.2 Diffusion-Based Target Synthesis

*Stable Diffusion XL creates the target image used as the initialization reference.*

\[
\mathbf{x}_{\mathrm{target}}
=\mathcal{D}(T,\mathbf{z};\theta_{\mathrm{SDXL}})
\tag{6}
\]

Stable Diffusion XL (SDXL) supplies the diffusion model \(\mathcal D\). The input \(\mathbf z\) is Gaussian-sampled, and \(\theta_{\mathrm{SDXL}}\) contains pretrained diffusion weights. The generated target image, \(\mathbf{x}_{\mathrm{target}}\), embodies the semantic attributes of \(T\). It supplies the initialization reference.

#### 5.1.3 Vision-Encoder Anchored Perturbation

*The initialization minimizes the difference between perturbed-image and target-image encoder representations.*

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

Here \(f_v\) is the VLM's vision component, \(\mathbf x\) is the original image, and \(\boldsymbol\delta_{\mathrm{init}}\) is the optimized initial perturbation. The \(\arg\min\) selects the perturbation minimizing squared encoder-representation error, \(\|\cdot\|_2^2\). The bound \(\epsilon\) is the maximum allowable, imperceptible \(\ell_\infty\) perturbation.

Mimicking the target image's encoder-space representation places the adversarial example in a semantically meaningful neighborhood of the target prompt representation. Algorithm 2 in Appendix B formalizes the method.

#### 5.1.4 Adversarial Optimization via S3D

*Projected refinement updates incrementally align the initialized example with the target embedding.*

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

After initialization, these updates align the adversarial example with the target embedding. At iteration \(t\), \(\mathbf{x}_{\mathrm{adv}}^{(t)}\) is the adversarial image, \(\alpha\) is the step size, and \(\mathcal L_{\mathrm{L6V}}\) is the squared encoder-output error. The operator \(\Pi_{\mathcal B_\epsilon(\mathbf x)}\) projects into the admissible \(\ell_\infty\) ball around original image \(\mathbf x\).

**Claims and evidence — Section 5.1**
- Initialization sensitivity and improved attack efficacy are reported: Table 3, “unknown” Q7X/B8N rows, C8H **0.7708 / 0.8977**; Figure 7 supplies the convergence comparison.
- Reduced convergence steps are reported: Figure 7(a); no numerical step counts are supplied.

### 5.2 Guiding perturbations via Target Value Vectors and evaluating F8V

*The proposed guidance targets vulnerable vision-encoder value vectors while retaining Q7X's opposing image and text objectives.*

The intended vulnerable components are value vectors in middle-to-late vision layers. Q7X lacks architectural sensitivity to the encoder's representational backbone. It treats vision and language as uniformly vulnerable, overlooking modality-specific weaknesses despite the encoder's dominant role in semantic grounding. Generic S3D output manipulation is less structured and less targeted than the proposed alternative. The intended improvements concern attack strength, transferability, interpretability, and computational feasibility.

Kim et al. (2024) proposed X9C (Doubly-Universal Adversarial Perturbation, Doubly-UAP, also D-UAP). This component-selection method identifies which attention components most influence VLM performance when constructing Z4P. Its component loss is adapted here by minimizing, rather than maximizing, losses between value vectors of \(x_i\) and \(T_i\), the named input and target-associated references. The adaptation retains a reference associated with the target text.

Cui et al. (2023) and Wang et al. (2024) show that perturbing vision-output embeddings impairs visual perception and misleads language responses in LVLMs (large VLMs). LVLMs commonly use similar image encoders. Their visual tokens—encoded semantic and structural image information—form the interface through which language decoders access and reason about images. Targeting encoder internals is proposed to exploit this shared interface.

Two attention components have fundamental roles:

- **Attention weights** control patch-to-patch focus, relevance, and interaction. Interfering with them is hypothesized to damage relationship establishment.
- **Value vectors** contain each patch's actual information. Perturbing them is hypothesized to damage essential content and interpretation.

Prior analysis identifies middle-to-late-layer value vectors as most vulnerable.

\[
\delta^*
=
\arg\max_\delta
\frac{1}{|L|}
\sum_{l\in L}
\operatorname{Loss}\bigl(V_l(x),V_l(x+\delta)\bigr)
\tag{10}
\]

In this standard X9C objective, \(\delta^*\) is the maximizing perturbation. \(L\) is the selected layer set, \(|L|\) is its size, and \(l\) indexes a layer. \(V_l(x)\) denotes input \(x\)'s value vectors at layer \(l\). \(\operatorname{Loss}(\cdot)\) is the loss applied to target vectors.

The proposed combination retains Q7X's opposing image/text objectives and adds attention-value disruption under the same hyperparameter constraints. The hypothesis is that value-vector perturbation relative to an image corresponding to the target text improves H2Q. Freezing the text encoder—the text-processing component—is intended to isolate degraded visual processing rather than text artifacts.

The modifications seek to exploit encoder information, align with target-text semantics, and improve cross-model adversarial robustness.

#### 5.2.1 Our Modified Approach

*P2F combines Q7X loss with a scaled negative value-vector cosine-similarity loss.*

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

P2F extracts layer-\(l\) value vectors from the input and the target-associated reference image. It computes their cosine-similarity loss, a normalized vector comparison using the dot product \(\cdot\). It jointly minimizes the negative of that loss with Q7X loss under appropriate scaling, rather than solely maximizing departure from the original representation.

Here:

- \(V_i(x+\delta)\) is the perturbed input's value vector at attention head \(i\).
- \(V_t\) is the reference image's target value vector.
- \(N\) counts attention heads.
- \(L_{\mathrm{X9C}}\), \(L_{\mathrm{Q7X}}\), and \(L_{\mathrm{P2F}}\) are the value-vector, Q7X, and combined losses.
- \(\lambda\) controls the relative importance of value loss.

The loss is described as encouraging closeness to the target representation. Combined gradients update \(\delta\), preserving adversarial character, N6Z, and guided semantic influence. Controlled, interpretable manipulation is intended for robustness and security analysis. Algorithm 3 in Appendix B gives the steps.

X9C's encoder dependence supplies a limitation hypothesis: transfer gains may be restricted to VLMs with the same or similar vision-encoder architecture as the model used for Q7X optimization.

**Figure 2.** X9C architecture diagram.

### 5.3 Transferability across images via Image Augmentation

*The study tests whether shared optimization and image mixing improve Q7X transfer to unseen images.*

Per-image iterative gradients overfit image-specific features. This produces poor K3R and high computational expense. Q7X likewise learns each image's perturbation across prompts. Luo et al. (2024) describe their method as orthogonal to K3R. They suggest computing perturbations across images, following Moosavi-Dezfooli et al. (2017). This investigation tests that claim and an additional approach based on the Z4P literature.

U8G uses regional replacement rather than pixel blending. It synthesizes images from rectangular patches cut from another image, with labels mixed in proportion to patch area. Its partial-view learning promotes spatially invariant features and robustness to occlusion and adversarial attacks, Yun et al. (2019).

Fang et al. (2024) introduces Z4P for VLP (Vision-Language Pretraining) models. The aim is to reduce gradient-attack overfitting and enable K3R. Zhang et al. (2024) improves global and local optimization across VLP models and tasks through M7A (Effective and Transferable Universal Adversarial Attack, ETU). Within ETU, J6T increases diversity while preserving semantics and spatial/scale invariance.

J6T is selected for perturbation generation on empirical grounds. For the same model, J6T alone, without ETU's additional local utility reinforcement, yields superior K3R.

#### 5.3.1 J6T

*SCMix first mixes transformed views of one image, then softly mixes the result with another image.*

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

The first stage self-mixes random crops and resizes of base image \(I_1\). The resulting views are \(X_1,X_2\), produced by \(\operatorname{RandomCrop}\) and \(\operatorname{Resize}\). Their mixture is \(I'_1\). The second stage cross-mixes \(I'_1\) with secondary image \(I_2\), producing \(I_3\).

The original J6T samples self-mixing weight \(\eta\sim\operatorname{Beta}(\alpha,\alpha)\), with \(\alpha>0\). This implementation fixes \(\eta=0.5\), described as equivalent to \(\alpha=\infty\). Appendix D.3 justifies this choice. The weights \(\beta_1,\beta_2\) satisfy the stated dominance relation \(\beta_1\gg\beta_2\in[0,1)\).

The weighting preserves dominant \(I_1\) semantics and compatibility of \(I_3\) with associated text \(T_1\), despite additional \(I_2\) features. It enriches cross-modal interaction. Intra-image variation and inter-image blending regularize against Q7X overfitting. They encourage perturbations directed at invariant features. The proposed common perturbation may then apply to unseen images without image-specific optimization.

At inference, Q7X returns only a perturbed image, equivalently its perturbation. Effective K3R would make that output universal in both the original N6Z sense and the additional image sense investigated here.

**Figure 3.** Self-mixing \(I_1\) generates \(X_1,X_2\), combined into \(I'_1\); external \(I_2\) is mixed with \(I'_1\) to produce \(I_3\).

#### 5.3.2 U8G

*CutMix uses a hard rectangular mask to retain partial views from paired images.*

\[
\tilde{x}=M\odot x_A+(1-M)\odot x_B
\tag{15}
\]

Here \(x_A,x_B\) are paired images, \(\tilde x\) is the mixed output, and \(\odot\) is the masked elementwise multiplication. \(M\in\{0,1\}^{W\times H}\) is a binary mask for a randomly located rectangle; \(W,H\) specify its dimensions.

Discrete replacement retains partial views. J6T softly blends features within and across images; U8G uses a hard mask. In adversarial training, persistent features across regional combinations are expected to direct perturbations toward spatially invariant, higher-level semantics rather than localized artifacts. This may improve K3R.

### 5.4 Attack Setup and Threat Models for Transferability Experiments

*Model-transfer and image-transfer experiments differ in attacker knowledge, perturbation capability, and evaluation.*

Knowledge, perturbation capability, and evaluation differ between transfer across models and transfer across images.

#### 5.4.1 Using X9C to explore F8V for Q7X

*The attacker optimizes on an accessible surrogate and tests the perturbation on models with unavailable parameters.*

The attacker seeks one image perturbation that elicits specified target text from multiple VLMs independently of prompt. Optimization uses an accessible surrogate model, meaning the model available to the attacker. Deployment targets unknown models.

White-box surrogate access includes gradients and internal representations. The surrogate is BLIP-2 or Flamingo. Only the target's general architecture family, such as CLIP-based, is known. Target parameters are unavailable.

The imperceptibility budget is \(\epsilon=16/255\) in \(L_\infty\). Intra-family testing optimizes BLIP-2 with OPT-2.7b, its language model, and evaluates InstructBLIP with Vicuna-7b. Both have similar CLIP ViT-L/14 image encoders but different language models and fine-tuning objectives.

Cross-architecture testing runs BLIP-2 to Flamingo and the reverse. This changes the vision encoder, language model, and architecture.

Each task is scored independently. Success in this evaluation means that the target model fails to generate the correct output from the perturbed image, regardless of prompt.

#### 5.4.2 Investigating K3R via Augmentation Techniques

*The attacker learns one perturbation on a limited image set and evaluates it on a separate unseen-image set.*

The goal is one pattern that causes predetermined target text for any image, regardless of image content and prompt. This would improve efficiency over per-image optimization.

The attacker has a limited training-image set and corresponding J6T variants, plus white-box access during optimization. The perturbation must generalize to unseen images. The \(L_\infty\) budget is \(\epsilon=16/255\).

A separate test set contains no training images. Taskwise H2Q counts failures to generate correct output text on previously unseen images, independently of content or prompt. Controlled self- and cross-mixing is hypothesized to capture image-independent model vulnerabilities rather than specific image features.

#### 5.4.3 Evaluation Rationale for Transferability experiments

*The transfer comparisons use untargeted success because targeted transfer depends strongly on visual-embedding architecture.*

All N6Z and F8V transferability results are presented under untargeted attacks. Non-targeted H2Q is also chosen as a robust, fair, comparable assessment of between-model and between-image transfer.

Distinct visual-embedding mechanisms make it difficult to isolate perturbation influence on final token embeddings. Targeted transfer is substantially stronger between shared-encoder models such as BLIP-2 and InstructBLIP. It varies drastically across heterogeneous encoders. Direct targeted-H2Q comparisons across such architectures are therefore imprudent.

Decoder-only architectures, as described here, introduce visual features late as auxiliary memory for causal generation. LLaVA and OpenFlamingo do this through cross-attention or token concatenation. Visual influence fades as decoding proceeds, Awadalla et al. (2023), Liu et al. (2023).

Fusion-based architectures integrate visual features early and deeply into input embeddings. BLIP-2, Li et al. (2023b), and InstructBLIP, Dai et al. (2023), retain visual grounding throughout generation. Targeted attacks are inherently more feasible on these models, Dai et al. (2023).

E3W is excluded from transfer comparisons after negligible transfer in experiments. Its per-image visual-token manipulation embeds target concepts in image-dependent patterns. Gradients and token spaces differ vastly across images and concepts. Input/target-image token-space transformations are non-transferable. Dependence on visual/textual encoder structure also prevents F8V across structural differences among BLIP2, LLaVA, Flamingo, and other LVLMs.

**Claims and evidence**
- Targeted transfer is stronger between shared-encoder models and varies across heterogeneous encoders: the section reports this rationale without numerical targeted-transfer rows.
- E3W showed negligible transfer and was excluded: no numerical E3W transfer table is supplied.

**Claims and evidence — Section 5**
- Q7X's reported resource cost is high: the characterization table rows report **over 6 hours** for **50 images**, a **48GB L40S**, and **95%** VRAM utilization.
- Initialization gains and overhead are reported separately: the same table gives **20-25 seconds** per image, **6-15%** aggregate accuracy gains, and targeted H2Q gains **up to 15%**.
- P2F's reported aggregate targeted H2Q is **96.85%**: the characterization table's P2F aggregate row; Table 6, P2F row, C8H **0.9685**.
- The separate P2F increase characterizations remain **12.5%** and **13.5%**; the characterization table also reports A7L **13%** and D5R **28%**.
- The reported transfer improvements are G9M **14%**, D5R **21%**, J6T **approximately 27 percent**, and U8G **around 12 percent**: the corresponding characterization-table rows.

## 6 Results

*The results reproduce the four original claims and evaluate the three extensions across the described VLM architectures.*

Five state-of-the-art VLMs were evaluated: OpenFlamingo, BLIP-2, InstructBLIP, and LLaVA. The Flamingo implementation is the publicly accessible OpenFlamingo-9B, Awadalla et al. (2023). Its performance is comparable to the original model.

BLIP-2 uses two-stage fusion. First, a frozen CLIP image encoder extracts features. A Querying Transformer then processes them, Li et al. (2023b). OPT-2.7b provides flexible language generation and efficient task adaptation.

InstructBLIP adds instruction tuning to this fusion architecture, Dai et al. (2023). It uses Vicuna-7b for task-specific instruction following and finer output control through crafted prompts.

LLaVA, Liu et al. (2023), connects pretrained CLIP ViT-L to Vicuna-13B through a lightweight projection. Its decoder-only generation supplies visual features directly as special tokens in the prompt context. This differs from fusion models' explicit visual–textual alignment before generation.

OpenFlamingo and LLaVA represent decoder-only architectures. BLIP-2 and InstructBLIP represent fusion-based architectures. The distinction significantly affects targeted adversarial effectiveness across prompts.

### 6.1 Claim 1: Q7X achieves N6Z across target texts [Reproduced]

*Flamingo-generated Q7X examples remain effective across held-out prompts and diverse target texts.*

Flamingo-generated Q7X examples transfer across held-out prompts in VQA, G9M, and D5R. This reproduces the central N6Z claim. Nonzero H2Q is reported across all tasks and diverse target prompts.

#### 6.1.1 Experimental Details

*Evaluation uses prompts distinct from training prompts and reports targeted success on Flamingo.*

The task families are:

- A7L: general visual questioning.
- T4B (VQAspecific): specific visual questioning.
- G9M: image-label prediction.
- D5R: image description.
- C8H (Overall): the aggregate column.

The two questioning labels distinguish how specifically a prompt pertains to a particular image.

Training follows Luo et al. (2024). Evaluation prompts differ from training prompts and span a broad range of semantic meanings. The Flamingo test measures targeted H2Q. Appendix D.1 gives details for all reproduced experiments. Appendix E lists the prompts.

#### 6.1.2 Comparative Analysis

*Q7X generally reproduces cross-prompt transfer but performs worse on classification and captioning, especially for abstract targets.*

Q7X generally agrees with the claimed transfer behavior. However, it consistently underperforms on G9M and D5R, especially for abstract target texts “too late” and “not sure,” which lack significant visual semantics. Successful deception persists across diverse targets.

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

**Claims and evidence — Sections 6.1 and 6.1.2**
- Q7X achieves nonzero success across all listed tasks and targets: Table 1, Q7X rows for “unknown,” “I am sorry,” “not sure,” “very good,” “too late,” and “metaphor”; C8H values are **0.7708, 0.7985, 0.5033, 0.7020, 0.6920, 0.8680**.
- G9M and D5R are weaker than the questioning tasks: Table 1, “not sure” Q7X row, A7L/T4B **0.8760/0.8940**, G9M/D5R **0.2420/0.0010**; “too late” Q7X row, **0.9300/0.9580** versus **0.7010/0.1790**.
- Successful deception persists across diverse targets: the six Q7X rows in Table 1 provide the taskwise values and aggregate values above.

### 6.2 Claim 2: Q7X achieves the best aggregate performance across prompt counts [Reproduced]

*Q7X remains ahead of the baselines at equal prompt counts, while gains from more prompts saturate rapidly.*

At equal prompt counts, Q7X consistently exceeds the baselines. Additional prompts do not allow baseline saturation to overtake it. Increasing prompt counts raises H2Q for all methods. R4K intrinsically uses one prompt, so count variation is redundant for it.

The original authors chose “unknown” to avoid high-frequency responses common in vision–language tasks. This experiment accepts that rationale. Prompt-count gains saturate rapidly, consistent with Luo et al. (2024).

**Figure 4.** Targeted H2Q against prompt count on Flamingo, holding attack iterations equal; Q7X significantly outperforms baselines. The target is “unknown.” R4K is the lone star.

**Claims and evidence**
- Q7X exceeds baselines at equal prompt counts and is not overtaken as counts increase: Figure 4; no numerical curve values or table rows are supplied.
- Gains rise and then saturate rapidly: Figure 4; no numerical saturation point is supplied.
- R4K uses one prompt: its definition in Section 4.2 and the lone star in Figure 4.

### 6.3 Claim 3: Q7X converges towards higher H2Q as iterations increase [Reproduced]

*Aggregate convergence follows the original trend, but individual-task convergence is unstable.*

Taskwise convergence is unstable despite exact reuse of the authors' training/evaluation parameters and code. The original convergence metrics in Appendix A.2 differ from these results. They showed convergence with more prompts. Across tasks jointly, convergence as iterations increase follows the original aggregate trend.

The authors did not specify the model for this scenario. The reproduced curves use OpenFlamingo and target “unknown.”

- (a) VQA
- (b) VQA-Specific
- (c) G9M
- (d) D5R

**Figure 5.** R4K, V9D, and Q7X compared across categories.

**Figure 6.** Aggregate convergence and Q7X comparison.

**Claims and evidence**
- Individual-task convergence is unstable: Figure 5, panels (a)–(d); no numerical curve values or table rows are supplied.
- Aggregate convergence follows the original trend: Figure 6; the original comparison is referenced in Appendix A.2, without numerical values supplied here.
- The reproduced scenario uses OpenFlamingo and “unknown”: the experimental statement above; the original authors did not specify the model.

### 6.4 Claim 4: Q7X outperforms baselines under few-shot settings [Reproduced]

*Q7X exceeds Multi-P when evaluation adds visual in-context examples, although context changes make attacks harder.*

Q7X significantly exceeds the baseline on most tasks when additional examples widen the context. This supports the min–max objective's intended robustness to expanded prompt space and resulting context changes.

| Setting | Method | A7L | T4B | G9M | D5R | C8H |
|---|---|---:|---:|---:|---:|---:|
| shot=0 | V9D | 0.7240 | 0.8740 | 0.5550 | 0.2850 | 0.6095 |
| | Q7X | 0.9680 | 0.9880 | 0.7070 | 0.4200 | 0.7708 |
| shot=2 | V9D | 0.7860 | 0.9420 | 0.5630 | 0.0010 | 0.5730 |
| | Q7X | 0.9440 | 0.9940 | 0.6860 | 0.0300 | 0.6635 |

**Table 2.** Targeted H2Q with and without visual in-context learning; shot counts added images.

Flamingo supports extra images as in-context examples for task adaptation. Following Luo et al. (2024), perturbations are optimized at 0-shot, with no extra images. They are evaluated at 2-shot, with two extra images. Evaluation therefore differs from optimization.

Generation-condition shifts make attacks harder, Luo et al. (2024). Contextual awareness moves the correct prompt space away from target-text space. Zhao et al. (2023) and Qi et al. (2023) verify that context-specific attacks such as Q7X lose effectiveness when context changes.

**Claims and evidence**
- Q7X exceeds V9D with added examples: Table 2, shot=2 Q7X/V9D rows, A7L **0.9440/0.7860**, T4B **0.9940/0.9420**, G9M **0.6860/0.5630**, D5R **0.0300/0.0010**, C8H **0.6635/0.5730**.
- Context changes reduce aggregate effectiveness: Table 2, shot=0/shot=2 rows, Q7X C8H **0.7708/0.6635** and V9D C8H **0.6095/0.5730**.
- Optimization and evaluation use different contexts: **0-shot** optimization and **2-shot** evaluation, as stated above and labeled in Table 2.

### 6.5 Results Beyond the Original Paper

*The extended experiments test initialization, attention-value guidance, and input diversity within the original framework.*

The extended experiments incorporate initialization, attention-value guidance, and input-diversity modifications into the original framework.

#### 6.5.1 Noise Initialization

*The initialization variant is reported to improve attack effectiveness and convergence, with taskwise and feature-similarity comparisons.*

B8N (CroPA+Init) is Q7X combined with the proposed initialization. It is reported to outperform Q7X and V9D consistently across every metric. Reported gains include aggregate accuracy, adversarial robustness, and N6Z. Semantic initialization is also reported to require fewer optimization steps. It is presented as a robust, adaptable alternative to random initialization, not a change to the perturbation budget.

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

*Prior min–max analyses motivate semantic starting points, while CLIP scores measure target-feature uptake.*

Semantic starting points are deliberately closer to meaningful adversarial directions. Zhang et al. (2025) find min–max problems substantially more initialization-sensitive than ordinary minimization. Stable convergence often needs carefully designed starts. Q7X resembles a saddle-point problem because prompt perturbations continually change the optimization landscape.

More frequent image updates than prompt updates create asymmetry and amplify starting-point importance. Chen et al. (2019) find local cycling in alternating-gradient min–max methods. Convergence depends strongly on initial proximity to adversarially robust regions. Wu et al. (2020) find that semantic-prior perturbations transfer better across models than random starts and yield substantial success-rate improvements. The experimental observations align with that finding, particularly for more complex bi-level optimization paths.

| Method | A7L | T4B | G9M | D5R | C8H |
|---|---:|---:|---:|---:|---:|
| V9D | 0.7053 | 0.8807 | 0.7001 | 0.5553 | 0.7104 |
| Q7X | 0.8723 | 0.9457 | 0.7518 | 0.6138 | 0.7958 |
| E3W | 0.2985 | 0.2281 | 0.4857 | 0.4687 | 0.370 |
| B8N | 0.9145 | 0.9342 | 0.8982 | 0.9464 | 0.9209 |

**Table 4.** Average targeted H2Q on BLIP-2 across visual-questioning, G9M, and D5R tasks. Best task values are highlighted in bold. E3W, Yang et al. (2024), is the additional baseline.

Gradient attacks ordinarily acquire target semantics. Theoretically, they should not require semantic initialization. Min–max complexity weakens that alignment. Injecting target semantics at initialization is described as necessary to reinforce alignment and assist convergence.

Y2K (CLIP score) quantifies feature uptake between target text and learned noise. The comparison uses the base perturbation and its initialized counterpart. It assesses acquired target attributes rather than measuring only task success.

| Method | Y2K |
|---|---:|
| Q7X | 21.80 |
| B8N | 23.09 |

Target text: “Bomb”.

**Figure 7.** (a) Aggregate convergence of Q7X and B8N. (b) Learned-perturbation/target feature similarity quantified by Y2K.

**Claims and evidence — Section 6.5.1 and its theoretical analysis**
- B8N is reported to outperform across every metric: Table 3 provides the targetwise comparisons in full. Its B8N/Q7X C8H pairs are “unknown” **0.8977/0.7708**, “bomb” **0.9260/0.8433**, “I am sorry” **0.9458/0.7985**, “very good” **0.8761/0.8190**, and “too late” **0.9333/0.8535**.
- The taskwise values accompanying that claim include: Table 3, “unknown” B8N/Q7X A7L **0.8824/0.9680** and T4B **0.8920/0.9880**; “bomb” G9M **0.9020/0.9498**; “very good” T4B **0.8894/0.8990**.
- Average targeted success improves in aggregate: Table 4, B8N/Q7X/V9D/E3W C8H **0.9209/0.7958/0.7104/0.370**. B8N/Q7X task values are A7L **0.9145/0.8723**, T4B **0.9342/0.9457**, G9M **0.8982/0.7518**, and D5R **0.9464/0.6138**.
- Initialization is reported to require fewer optimization steps: Figure 7(a); no numerical step counts are supplied.
- The initialized perturbation has a higher target-feature score: the Y2K table, B8N **23.09** versus Q7X **21.80**, with target “Bomb”; Figure 7(b).

#### 6.5.2 Investigating K3R and Image Augmentation

*Image mixing improves cross-image success, but does not remove the gap from image-specific Q7X attacks.*

Learning across images does not produce effective universality for unmodified Q7X. Prompt-directed signals remain tightly linked to the originating image's low-level features. This weakens the original claim that learning perturbations across images would establish K3R.

Both input-mixing approaches give measurable gains. However, a fundamental gap remains between cross-image methods and Q7X applied to a single image. Q7X excels at prompt robustness, not effective image-transferring adversarial attacks.

U8G is believed to outperform J6T for adversarial transfer. The proposed reason is that region-based mixing retains distinct, non-overlapping spatial areas. This encourages modular, independently manipulable features and greater cross-input generalization.

| Method | VQA | T4B | G9M | D5R | C8H |
|---|---:|---:|---:|---:|---:|
| Q7X (Base) | 0.3500 | 0.4920 | 0.4000 | 0.5000 | 0.4355 |
| Q7X with J6T | 0.6560 | 0.7160 | 0.5000 | 0.9500 | 0.7055 |
| Q7X with U8G | 0.4520 | 0.7240 | 0.5000 | 0.5500 | 0.5565 |

**Table 5.** Untargeted K3R H2Q, best across all iterations. Unmixed Q7X peaks at 1700 iterations; J6T and U8G exceed it significantly earlier, at 1500 and 1600 respectively. Experimental parameters: Appendix D.3.

**Claims and evidence**
- Unmodified Q7X has limited cross-image success: Table 5, Q7X (Base) row, VQA **0.3500**, T4B **0.4920**, G9M **0.4000**, D5R **0.5000**, C8H **0.4355**.
- Both augmentations improve aggregate K3R: Table 5, J6T/U8G/base C8H **0.7055/0.5565/0.4355**.
- U8G is believed to outperform J6T: the paired Table 5 rows report VQA **0.4520/0.6560**, T4B **0.7240/0.7160**, G9M **0.5000/0.5000**, D5R **0.5500/0.9500**, C8H **0.5565/0.7055**.
- Augmented methods exceed the base earlier: Table 5 caption, base **1700** iterations, J6T **1500**, U8G **1600**.
- A gap from single-image Q7X remains: this limitation is stated here; no matched single-image comparison row is supplied in Table 5.

#### 6.5.3 Guiding perturbations via Target Value Vectors

*Value-vector guidance improves targeted success across tasks, including the harmful target “Bomb.”*

P2F greatly improves the existing algorithm across tasks, especially A7L and D5R. The target “Bomb” is harmful text expected inherently to activate guardrails, meaning model protections against such output. The improvement is maintained for these worse expected harmful-word cases and vulnerable scenarios.

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

**Claims and evidence**
- P2F improves targeted BLIP2 success across tasks: Table 6, P2F/Q7X rows, A7L **0.9420/0.8176**, T4B **0.9720/0.9100**, G9M **0.9790/0.9498**, D5R **0.9810/0.6960**, C8H **0.9685/0.8433**.
- P2F exceeds the additional baselines in aggregate on BLIP2: Table 6, P2F/V9D/E3W C8H **0.9685/0.75/0.4199**.
- P2F improves targeted Open Flamingo success: Table 7, P2F/Q7X rows, A7L **0.9000/0.7900**, T4B **0.9520/0.8860**, G9M **0.9500/0.9470**, D5R **0.9060/0.6700**, C8H **0.9270/0.8233**.
- Improvements hold for the harmful target “Bomb”: Tables 6–7 use that target. Guardrail activation is an expectation stated in the text, not a separately measured table row.

#### 6.5.4 Analyzing F8V by Value Vector guidance

*Value guidance improves model transfer, but encoder dependence limits transfer across architectures.*

The value-guided method significantly improves vanilla Q7X transfer, especially G9M and D5R, with modest VQA gain. VQA prompts occupy a more complex embedding space than the smaller output spans of targeted G9M and D5R.

Encoder dependence verifies the earlier limitation hypothesis. BLIP2 perturbations are passably transferable to same-encoder InstructBLIP and fail to transfer to OpenFlamingo. Training on Flamingo and applying perturbations to other models likewise gives insignificant H2Q (Table D.4.5, Appendix D.4.5). Appendix D.4 gives experimental details.

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

For a given target text, vision-encoder value vectors remain approximately consistent across input images. They reflect encoder architecture and pretraining. This consistency may constrain perturbation diversity and highly varied adversarial-example generation. Target-text choice strongly influences outcomes relative to input-image-specific features.

**Claims and evidence**
- Value guidance improves BLIP2-to-InstructBLIP transfer: Table 8, P2F/Q7X rows, A7L **0.3940/0.3220**, T4B **0.5780/0.4170**, G9M **0.8800/0.6720**, D5R **0.8880/0.6150**, C8H **0.6850/0.5065**.
- Transfer depends on the encoder: Table 8, P2F C8H is **0.6850** for BLIP2 to InstructBLIP and **0.2335** for BLIP2 to Flamingo. The text characterizes the latter as failure to transfer.
- Flamingo-origin transfer is described as insignificant: Table 8, Flamingo-to-BLIP2/InstructBLIP/LLaVA P2F C8H **0.1013/0.0938/0.1025**, versus Q7X **0.0335/0.0290/0.0370**; the text also cites Table D.4.5.
- Other tested transfers improve in aggregate: Table 8, P2F/Q7X C8H is BLIP2 to LLaVA **0.5325/0.3040**, InstructBLIP to BLIP2 **0.4900/0.2288**, InstructBLIP to LLaVA **0.3558/0.1345**, and InstructBLIP to Flamingo **0.1375/0.0598**.
- Approximate value-vector consistency and target-text influence are reported: no numerical consistency or target-choice comparison table is supplied in this section.

**Claims and evidence — Section 6.5**
- Initialization improves reported aggregate success: Table 4, B8N/Q7X C8H **0.9209/0.7958**.
- Input mixing improves cross-image aggregate success: Table 5, J6T/U8G/base C8H **0.7055/0.5565/0.4355**.
- Value guidance improves targeted success: Table 6, P2F/Q7X C8H **0.9685/0.8433**; Table 7, **0.9270/0.8233**.
- Value-guided transfer remains architecture-dependent: Table 8, BLIP2-to-InstructBLIP/Flamingo P2F C8H **0.6850/0.2335**.

**Claims and evidence — Section 6**
- Cross-prompt success is reproduced across targets: Table 1, the six Q7X C8H values **0.7708, 0.7985, 0.5033, 0.7020, 0.6920, 0.8680**.
- Prompt-count superiority and aggregate convergence are reproduced: Figures 4 and 6; no numerical curve values are supplied. Figure 5 retains the taskwise-instability caveat.
- Few-shot superiority is reproduced: Table 2, shot=2 Q7X/V9D C8H **0.6635/0.5730**.
- The extensions improve reported success but retain transfer limitations: Tables 3–8 and the section-specific evidence above.
- Architecture significantly affects targeted effectiveness: the architectural comparison is stated in the section introduction; no isolated architecture-effect table is supplied.

## 7 Discussion

*The discussion reviews reproduction difficulties, proposed mechanisms, transfer trade-offs, impacts, and resource limits.*

### 7.1 Observations with reproducibility

*The public implementation supported reproduction, but BLIP-2 and InstructBLIP required substantial debugging.*

Reproduction used the authors' public repository and original model configurations for faithful baseline comparison. The implementation was generally well documented. However, BLIP-2 and InstructBLIP code produced multiple runtime errors. Substantial debugging and adjustment were required to run it.

**Claims and evidence**
- Reproduction required debugging despite generally good documentation: the implementation experience reported here; no numerical error counts or table rows are supplied.

### 7.2 Noise Initialization findings

*Semantic initialization is interpreted as a small shift toward vulnerable regions rather than a large increase in learned semantics.*

The initialization hypothesis includes reduced cross-modal as well as cross-prompt transfer when target semantics are ignored. Experiments across all evaluated benchmarks are described as significantly favoring semantic noise over Q7X and random-initialized baselines.

Q7X perturbations already acquire target context, as expected from gradients. Initialization gives a modest Y2K increment, not a significant semantic-feature upgrade. This is interpreted as a subtle shift closer to adversarially vulnerable regions. Convergence before and after initialization aligns with the hypothesis that a start nearer meaningful adversarial directions converges faster.

**Claims and evidence**
- Semantic noise is described as improving benchmark performance: Table 4, B8N/Q7X/V9D C8H **0.9209/0.7958/0.7104**; Table 3 retains the complete taskwise comparisons.
- The semantic-feature increment is modest: the Y2K table in Section 6.5.1, Q7X **21.80**, B8N **23.09**.
- Initialization is associated with faster convergence: Figure 7(a); no numerical convergence-step values are supplied. The movement toward vulnerable regions is the interpretation stated here.

### 7.3 Evaluation of Transferability Experiments

*Augmentation improves image transfer, while value guidance improves model transfer within encoder-dependent limits.*

#### 7.3.1 K3R

*Both augmentations help, but image diversity alone does not eliminate cross-image limitations.*

J6T improves K3R particularly for VQA, T4B, and D5R. U8G also improves the base image-transfer version. The proposed explanation favors J6T's explicit spatial/scale invariance: random crops and resizes of the same image are followed by soft mixing with another image. Semantically aligned features become less sensitive to position and scale, easing cross-input generalization.

Both augmentations preserve original semantic information while adding image diversity. Their proposed benefit is learning cross-image features. However, augmentation alone does not eliminate the limitation. Further work should combine augmentation with more complex, layered methods, including latent-representation feature-alignment attacks or direct latent-space perturbation.

**Claims and evidence**
- J6T particularly improves VQA, T4B, and D5R: Table 5, J6T/base values **0.6560/0.3500**, **0.7160/0.4920**, and **0.9500/0.5000**, respectively.
- U8G improves the base aggregate: Table 5, U8G/base C8H **0.5565/0.4355**.
- Augmentation does not eliminate the limitation: Section 6.5.2 reports the remaining gap; no matched single-image comparison row is supplied in Table 5.
- Spatial/scale invariance and cross-image feature learning are the proposed explanations; they are not separately quantified in the table.

#### 7.3.2 F8V

*Value guidance is presented as progress, but encoder dependence and E3W's non-transferability remain obstacles.*

X9C guidance is described as a significant advance robust to harmful target text and guardrail activation. Encoder-specific dependence remains the obstacle to encoder-agnostic attacks. It narrows perturbation diversity and transfer across heterogeneous architectures.

Despite this limitation, the framework is presented as progress toward universal attacks effective across prompts and images, and toward improved VLM generalizability and robustness. E3W cannot supply comparable transfer results and is inherently non-transferable.

**Claims and evidence**
- Guidance improves success for harmful target text: Table 6, “Bomb,” P2F/Q7X C8H **0.9685/0.8433**; Table 7, **0.9270/0.8233**. Guardrail activation is not separately measured.
- Encoder dependence remains: Table 8, P2F C8H **0.6850** for BLIP2 to InstructBLIP versus **0.2335** for BLIP2 to Flamingo.
- E3W cannot supply comparable transfer results: Section 5.4.3 reports negligible transfer and exclusion; no numerical transfer rows are supplied.

**Claims and evidence — Section 7.3**
- Image augmentation improves reported K3R: Table 5, base/J6T/U8G C8H **0.4355/0.7055/0.5565**.
- Value guidance improves reported F8V but remains encoder-dependent: Table 8, BLIP2-to-InstructBLIP P2F/Q7X C8H **0.6850/0.5065**, compared with BLIP2-to-Flamingo **0.2335/0.1435**.

### 7.4 Trade-offs Between Different Types of Adversarial Transferability

*The paper argues that prompt, model, and image transfer impose competing optimization demands.*

#### 7.4.1 N6Z vs. F8V

*Prompt competition anchors perturbations to an architecture, while value guidance helps related architectures more than different ones.*

Competing text embeddings implicitly anchor original Q7X perturbations to the target architecture. Moving attacks between models causes a significant H2Q drop.

X9C targets potentially generalizable encoder components rather than model-specific features. It improves related-architecture transfer, BLIP-2 to InstructBLIP. It offers only modest gains across fundamentally different architectures, BLIP-2 to Flamingo. Simultaneously optimizing prompt diversity and model diversity is inherently tense.

**Claims and evidence**
- X9C improves related-architecture transfer: Table 8, BLIP2-to-InstructBLIP Q7X/P2F C8H **0.5065/0.6850**.
- Gains across different architectures are described as modest: Table 8, BLIP2-to-Flamingo Q7X/P2F C8H **0.1435/0.2335**.
- Moving models causes a significant drop: Table 8 supplies transferred H2Q; no matched within-model untargeted row is supplied there.
- Architectural anchoring and tension between prompt and model diversity are the explanations stated here, not separately quantified results.

#### 7.4.2 N6Z vs. K3R

*Image-specific coupling supports prompt transfer but limits transfer across images.*

Image-specific coupling supports strong N6Z and fundamentally limits K3R. The two are competing optimization objectives. Augmentation diversifies optimization and gives promising improvements without resolving the fundamental tension.

**Claims and evidence**
- Augmentation improves K3R without resolving the reported limitation: Table 5, base/J6T/U8G C8H **0.4355/0.7055/0.5565**; Section 6.5.2 states the remaining single-image gap.
- Image-specific coupling and competing objectives are the paper's explanation; no separate numerical trade-off table is supplied.

#### 7.4.3 Role of Initialization in Transferability Balance

*Semantic anchoring improves reported success and prompt transfer, but the paper retains a broader trade-off among transfer dimensions.*

Semantic anchoring raises H2Q without sacrificing N6Z. This indicates that starting-point selection substantially influences the achievable transfer balance. Unlike changes to the optimization process, better initialization allows convergence toward more general adversarial directions that remain effective across prompts.

Even then, optimizing a perturbation for one transfer dimension inevitably sacrifices effectiveness in others.

**Claims and evidence**
- Semantic anchoring raises reported aggregate H2Q while retaining cross-prompt evaluation: Table 3, B8N/Q7X C8H pairs **0.8977/0.7708**, **0.9260/0.8433**, **0.9458/0.7985**, **0.8761/0.8190**, and **0.9333/0.8535**.
- Convergence changes with initialization: Figure 7(a); no numerical step counts are supplied.
- The inevitable sacrifice among transfer dimensions is stated here; no numerical all-dimensions comparison is supplied.

#### 7.4.4 Implications for Future Research

*The paper calls for controlled balancing of transfer objectives and more work on prompt-induced vulnerability.*

No single methodology can maximize all transfer forms simultaneously. Controlled balancing may require multi-objective optimization or architecturally aware attack designs. Improving individual dimensions is insufficient for a comprehensive solution without fundamentally rethinking perturbation interactions with model architecture and data representation.

N6Z remains underexplored. Little literature addresses prompt-induced vulnerability. This understanding gap matters for secure deployment.

**Claims and evidence — Section 7.4**
- Related-architecture and different-architecture transfer behave differently: Table 8, BLIP2-to-InstructBLIP/Flamingo P2F C8H **0.6850/0.2335**.
- Augmentation and initialization improve their reported dimensions: Table 5, J6T/base C8H **0.7055/0.4355**; Table 4, B8N/Q7X C8H **0.9209/0.7958**.
- The broader need to balance objectives is the paper's stated conclusion; no single table measures simultaneous maximization of all transfer forms.

### 7.5 Broader Impact

*The attacks pose misinformation risks, may support image privacy, and motivate more resilient defenses.*

High-N6Z examples can elicit malicious or misleading outputs from harmless queries. They can enable misinformation or subvert VLM-dependent content-generation and decision systems.

Imperceptible perturbations also offer privacy protection. Embedded in personal images, they can force predetermined nonsensitive text and thwart unauthorized extraction of confidential information. This is a proactive security mechanism as visual data becomes increasingly exploitable.

The initialization and loss refinements increase H2Q and cross-model/cross-image generalizability. They strengthen cross-prompt attacks while opening paths toward more resilient defenses. Deeper understanding is necessary for robust VLMs, effective countermeasures, user-data protection, and output integrity.

**Claims and evidence**
- The refinements improve reported attack success and transfer: Table 4, B8N/Q7X C8H **0.9209/0.7958**; Table 5, J6T/base **0.7055/0.4355**; Table 8, BLIP2-to-InstructBLIP P2F/Q7X **0.6850/0.5065**.
- Misinformation, privacy protection, and defense benefits are the broader impacts described here; no separate numerical impact evaluation is supplied.

### 7.6 Limitations and Future Work

*Compute and memory scarcity limited replication, ablations, tuning, and broader transfer experiments.*

Computational scarcity repeatedly hindered successful key-result replication and extensive ablations. Attack training and evaluation on large Flamingo and BLIP2 models require substantial GPU memory and processing. This limited experimental scale and depth, extensive hyperparameter tuning, and transfer testing across wider architecture ranges.

A planned ensemble-model transfer investigation could have deepened the generalizability analysis. It exceeded available memory and compute and was infeasible on the existing infrastructure.

Future work includes encoder-agnostic F8V, stronger K3R, and query-based optimization for practical applicability.

### 7.7 Communication With Original Authors

*The replication established no direct author communication, and the reported GitHub issue remained unresolved.*

No direct communication with the original authors was established during replication. The GitHub issue concerning BLIP-2 and InstructBLIP remains unresolved at writing.
