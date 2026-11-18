# Revisiting CroPA: A Reproducibility Study and Enhancements for Cross-Prompt Adversarial Transferability in Vision-Language Models

Atharv Mittal atharv_m@mfs.iitr.ac.in
††thanks: All authors contributed equally
Affiliation: Mehta Family School of Data Science and Artificial Intelligence

Affiliation: Indian Institute of Technology, Roorkee

  
Agam Pandey11footnotemark:
1

agam_p@ce.iitr.ac.in

Affiliation: Department of Civil Engineering

Affiliation: Indian Institute of Technology, Roorkee

  
Amritanshu Tiwari11footnotemark:
1

amritanshu_t@mfs.iitr.ac.in

Affiliation: Mehta Family School of Data Science and Artificial Intelligence

Affiliation: Indian Institute of Technology, Roorkee

  
Sukrit Jindal11footnotemark:
1

sukrit_j@mfs.iitr.ac.in

Affiliation: Mehta Family School of Data Science and Artificial Intelligence

Affiliation: Indian Institute of Technology, Roorkee

  
Swadesh Swain11footnotemark:
1

swadesh_s@ece.iitr.ac.in

Affiliation: Department of Electronics and Communication

Affiliation: Indian Institute of Technology, Roorkee

###### Abstract

Large Vision-Language Models (VLMs) process images and text. They support tasks such as image classification, which assigns an image to a category; captioning, which describes an image; and visual question answering, which answers a question about an image. However, VLMs remain highly vulnerable to adversarial attacks: deliberate input changes that cause incorrect or harmful outputs. Both images and text can be manipulated.

We reproduce “An Image is Worth 1000 Lies: Adversarial Transferability Across Prompts on Vision-Language Models.” We validate its Cross-Prompt Attack (CroPA). Cross-prompt transferability means that an adversarial image remains effective when the accompanying textual prompt changes. We confirm CroPA’s superior cross-prompt transferability over existing comparison methods.

We also propose and investigate three enhancements: (1) a new initialization strategy that significantly improves Attack Success Rate (ASR), the fraction of evaluated attacks that succeed; (2) universal perturbations, or shared input changes, for transfer across images; and (3) a new loss function, a numerical optimization objective, that targets the vision encoder’s attention mechanism to improve generalization. The vision encoder converts an image into internal representations. Its attention mechanism controls interactions among image regions.

Our evaluation uses the VLMs Flamingo, BLIP-2, and InstructBLIP, with extended experiments on LLaVA. It validates the original results and shows that our improvements consistently increase adversarial effectiveness. The targeted results appear in Tables 1–4, 6, and 7. Cross-image and cross-model results appear in Tables 5 and 8. Our work reinforces the need to study VLM vulnerabilities and provides a more robust framework for generating transferable adversarial examples. These findings have significant implications for understanding VLM security in real-world applications.

Our code is available at https://github.com/Swadesh06/Revisting_CroPA

Reviewed on OpenReview: https://openreview.net/forum?id=5L90cl0xtf

## 1 Introduction

Large Vision-Language Models (VLMs) have transformed computer vision by supporting image classification, captioning, and visual question answering. Their versatility has encouraged deeper study of interactions between visual and linguistic information.

Recent studies show that VLMs remain highly vulnerable to adversarial attacks (Zhao et al., 2023; Qi et al., 2023; Zhang et al., 2022a; Carlini et al., 2024). These attacks add subtle perturbations, or deliberate changes, to input images. The changes can cause incorrect or harmful outputs. Textual inputs provide an additional attack route beyond those available in traditional vision models. Images and text are the two input modalities, or types of information, considered here.

Several studies examine VLM adversarial robustness: the ability to maintain correct behavior under adversarial changes. Zhao et al. (2023) studied models such as BLIP and BLIP-2 under black-box attacks, where the attacker lacks access to model internals. They considered query-based attacks, which use model responses, and transfer-based attacks, which reuse attacks developed on another model. Schlarmann & Hein (2023) studied targeted and untargeted attacks under white-box access, where the attacker can inspect model internals. A targeted attack seeks a chosen output. An untargeted attack seeks an output different from the clean input’s prediction.

Later research also examined text perturbations. Qi et al. (2023) showed that adversarial images could induce VLMs to execute harmful instructions. Tu et al. (2023) systematically evaluated both visual and textual attacks.

Adversarial generalization and transferability in VLMs have traditionally been divided into two categories:

- **Cross-Model Transferability:** An adversarial example remains adversarial across different VLM architectures, or model designs. This is commonly called transferability.
- **Cross-Image Transferability:** A perturbation remains effective across multiple images. Such shared changes are often called Universal Adversarial Perturbations (UAPs).

Luo et al. (2024) introduced **Cross-Prompt Transferability**: an adversarial image remains effective across different textual prompts. Unlike work that treats visual and textual perturbations independently, their Cross-Prompt Attack (CroPA) uses learnable prompts—prompt representations adjusted during optimization—to make adversarial images effective regardless of textual input. They demonstrated CroPA on classification, captioning, and visual question answering.

Our goals are:

- **[Reproducibility Study] Reproducing the Results from the Original Paper:** We reproduce and verify four main claims. First, CroPA achieves better cross-prompt transferability than established baselines across target texts. This transferability is independent of the target text’s semantic meaning or word frequency (Section 6.1; Table 1). Here, semantic meaning refers to the concepts conveyed by the text. Second, adding prompts improves transferability, but performance stabilizes rapidly. Even with additional prompts, the baselines do not surpass CroPA (Section 6.2; Figure 4). Third, CroPA remains effective with additional in-context learning examples: examples supplied in the input to help specify the task (Section 6.4; Table 2). Fourth, we reproduce the overall ASR convergence trend as attack iterations increase. Convergence means that performance approaches a stable level. We also analyze individual tasks (Section 6.3; Figures 5 and 6).
- **[Extended Work] Better Initialization Strategy:** We propose a new starting-point selection method that substantially increases ASR and transferability (Section 6.5.1).
- **[Extended Work] Investigating Cross Image Transferability and Image Augmentation:** We use CroPA’s underlying framework to learn one perturbation for all images. We test whether transferability-promoting changes help (Section 6.5.2). Image augmentation means constructing modified training images to increase input diversity.
- **[Extended Work] Guiding perturbations via Target Value Vectors:** We propose a loss based on the vision encoder’s attention mechanism. Image patches are the image regions processed by this mechanism. Attention components control interactions among patches. Value vectors hold the information associated with each patch. We target these vectors to improve generalization and ASR (Sections 6.5.3 and 6.5.4).

We also analyze the mechanisms behind these improvements. Our work reinforces the importance of VLM adversarial vulnerabilities and provides a more robust and versatile framework for generating transferable adversarial examples.

## 2 Related Works

Adversarial attacks on deep neural networks have been studied extensively since Szegedy et al. (2013) and Goodfellow et al. (2014). Applying them to VLMs introduces distinct challenges and opportunities. This section reviews adversarial transferability and places CroPA in that context.

### 2.1 Adversarial Transferability

Adversarial transferability occurs along several dimensions. Liu et al. (2017) and Tramèr et al. (2017) showed that examples crafted for one model can often deceive other models, including models with different architectures. Moosavi-Dezfooli et al. (2017) and Mopuri et al. (2017) showed that perturbations can also generalize across images, enabling UAPs.

Transferability also extends across tasks. Examples designed for classification can affect object detection, which locates and identifies objects, and other visual tasks (Naseer et al., 2019b; Naseer et al., 2019a; Lu et al., 2020; Salzmann et al., 2021). These different forms of transferability raise questions about the robustness of modern machine learning systems.

### 2.2 Adversarial Attacks on Vision-Language Models

Early VLM robustness research focused on task-specific attacks. Xu et al. (2019), Zhang et al. (2020), Aafaq et al. (2021), and Chen et al. (2017) demonstrated ways to manipulate image captions. Visual Question Answering (VQA) systems were also vulnerable to targeted attacks that mislead attention mechanisms (Xu et al., 2018; Kaushik et al., 2021; Kovatchev et al., 2022; Li et al., 2021; Sheng et al., 2021; Zhang et al., 2022b).

These approaches mainly targeted lightweight CNN-RNN architectures: combinations of convolutional neural networks, which process local image patterns, and recurrent neural networks, which process sequences. These models lack in-context learning capabilities. This limits the applicability of those attacks to contemporary VLMs.

Zhao et al. (2023) studied modern large VLMs, including BLIP and BLIP-2, under black-box access. Their focus was cross-model transferability rather than CroPA’s cross-prompt perspective.

### 2.3 Cross-Prompt Adversarial Transferability

Large VLMs can adapt to tasks through textual prompts (Li et al., 2023b; Awadalla et al., 2023; Li et al., 2023b; Zhu et al., 2023; Liu et al., 2023). Traditional vision models generally perform specific tasks. Prompt-driven adaptation creates an attack setting in which adversarial examples must remain effective as prompts change.

CroPA introduced learnable prompts that compete with image perturbations during optimization (Luo et al., 2024). This differs from methods that treat image and text perturbations independently. It establishes a new way to evaluate VLM robustness.

Because cross-prompt transferability is relatively new, no credible benchmarks or baselines have yet been established. Yang et al. (2024) nevertheless study it through the Context Injection Attack (CIA). CIA uses gradients—directions describing how a loss changes with an input—to inject target tokens into visual and textual contexts. Tokens are the units represented and processed by the model. We include CIA as an additional comparison.

### 2.4 Enhancing Adversarial Transferability

Input diversity methods (Xie et al., 2019) and data augmentation strategies (Wang et al., 2021) have shown promise for stronger attacks. Recent studies identify gradient misalignment, where update directions do not align, as a key limit on transferability (Li et al., 2023a; Wang et al., 2023). This has motivated new optimization strategies. We build on these insights to improve CroPA across multiple transferability dimensions.

## 3 Scope of reproducibility

We examine and validate the results of Luo et al. (2024). Our main objective is to confirm that CroPA significantly improves adversarial transferability across prompts. We do so by reproducing their experimental procedures.

We aim to verify these claims:

- **Claim 1:** CroPA achieves cross-prompt transferability across various target texts.
- **Claim 2:** CroPA achieves the best overall performance across different number of prompts.
- **Claim 3:** CroPA converges towards higher ASR as number of iterations are increased.
- **Claim 4:** CroPA outperforms baselines under few-shot settings. A few-shot setting supplies a small number of examples in the input.

Beyond replication, we investigate cross-model and cross-image transferability. We also compare CroPA with other cross-prompt methods. Specifically, we test whether CroPA-generated adversarial images can consistently deceive different VLMs despite changes in prompts or model parameters, the numerical values that determine model behavior.

These experiments reproduce the original work and examine CroPA’s broader applicability to robust and versatile adversarial attacks.

## 4 Methodology

### 4.1 Problem Formulation

Neural networks’ vulnerability to adversarial attacks is well documented (Goodfellow et al., 2014). Luo et al. (2024) extend this setting to perturbations that remain effective across textual inputs.

Let \(f\) be a VLM. Its image input is \(x_v\), and its text input is \(x_t\). Let \(\delta_v\) be an image perturbation. The perturbed image is \(x_v+\delta_v\). The perturbation must satisfy

\[
\|\delta_v\|_p\leq\epsilon_v.
\]

Here, \(\|\cdot\|_p\) is a norm measuring perturbation size, with \(p\) selecting the norm. The bound \(\epsilon_v\) is the maximum allowed size. The authors use this constraint to keep changes imperceptible to humans. It follows established adversarial-learning practice in the multimodal setting (Carlini et al., 2024).

We reproduce two attack scenarios. Let \(k\) be the number of prompts, \(i\) index them, and \(x_t^i\) be prompt \(i\). Let \(L\) be the language modeling loss, which measures disagreement between the model’s generated text and a reference text.

**Targeted attack.** The attacker wants the model to produce a predetermined text \(T\), regardless of the prompt. The image perturbation minimizes the total loss over the \(k\) prompts:

\[
\min_{\delta_v}\sum_{i=1}^{k}
L\bigl(f(x_v+\delta_v,x_t^i),T\bigr).
\tag{1}
\]

**Non-targeted attack.** The attacker wants the perturbed output to differ from the clean output. Here, “clean” means unperturbed. The perturbation maximizes the loss between those outputs:

\[
\max_{\delta_v}\sum_{i=1}^{k}
L\bigl(f(x_v+\delta_v,x_t^i),f(x_v,x_t^i)\bigr).
\tag{2}
\]

We measure effectiveness using ASR. A targeted attack succeeds only if the model generates the exact target text. A non-targeted attack succeeds if the output differs in any way from the clean image’s prediction.

### 4.2 Baseline Approaches

Cross-prompt transferability is a new research direction without established baselines. We therefore use the two baselines introduced by Luo et al. (2024): Single-P and Multi-P.

**Single-P** optimizes an image perturbation with one prompt. Its targeted objective is

\[
\min_{\delta_v}L\bigl(f(x_v+\delta_v,x_t),T\bigr).
\tag{3}
\]

Figure 1 also uses \(\delta_t\) for a learnable prompt perturbation. Its \(\oplus\) symbol means element-wise addition: adding corresponding entries of two inputs. A forward pass computes model outputs. Backpropagation computes gradients through that calculation.

Figure 1: Framework overview of CroPA as presented in Luo et al. (2024). The architecture employs learnable perturbations for both image (\(\delta_v\)) and prompt (\(\delta_t\)) inputs. These perturbations operate antagonistically, while \(\delta_v\) is optimized to minimize the language modeling loss, \(\delta_t\) works to maximize it. The model’s forward pass (solid arrows) processes the perturbed inputs through the vision encoder and language model, with backpropagation (dashed arrows) updating the perturbations at configurable frequencies. The model parameters (marked with *) remain fixed during the attack. \(\oplus\) denotes element-wise addition.

**Multi-P** uses a prompt collection \(X_t=\{x_t^1,x_t^2,\ldots,x_t^k\}\). Its targeted objective is

\[
\min_{\delta_v}\sum_{i=1}^{k}
L\bigl(f(x_v+\delta_v,x_t^i),T\bigr).
\tag{4}
\]

For non-targeted attacks, these objectives are inverted to maximize disagreement between clean and perturbed outputs. Both methods provide reference points for evaluating CroPA’s cross-prompt transferability.

### 4.3 Cross Prompt Attack

CroPA uses learnable prompts to improve cross-prompt transferability (Luo et al., 2024). Its prompt perturbations compete with image perturbations rather than helping them deceive the model.

The image perturbation \(\delta_v\) minimizes the loss for generating \(T\). The prompt perturbation \(\delta_t\) maximizes that same loss. This competition forces \(\delta_v\) to develop stronger transferability across prompts. The objective is a min-max problem: one variable minimizes a quantity while the other maximizes it.

\[
\min_{\delta_v}\max_{\delta_t}
L\bigl(f(x_v+\delta_v,x_t+\delta_t),T\bigr).
\tag{5}
\]

Here, \(f\), \(T\), and \(L\) retain their definitions from Section 4.1.

The implementation uses Projected Gradient Descent (PGD), an iterative method that updates an input using a gradient and then projects it back into the permitted perturbation region (Madry et al., 2017). Image updates use gradient descent to reduce the loss. Prompt updates use gradient ascent to increase it. The parameter \(N\) sets the update schedule: there are \(N\) image updates for each prompt update. Figure 1 shows the framework.

Algorithm 1 reproduces the algorithm of Luo et al. (2024). Its additional notation is:

- \(x'_v\): the current adversarial image.
- \(x_t^{\prime i}\): the initialized or updated version of prompt \(i\).
- \(\epsilon\): the perturbation-size bound.
- \(\alpha_1,\alpha_2\): image and prompt step sizes.
- \(K\): the number of iteration steps.
- \(g_v,g_t\): image and prompt gradients.
- \(\nabla\): the gradient operator.
- \(\operatorname{sign}\): the element-wise sign of a gradient.
- \(\operatorname{mod}(\text{step},N)\): the remainder after dividing the step number by \(N\).
- \(\operatorname{Clip}_{x_v,\epsilon}\): projection into the allowed \(\epsilon\)-ball around \(x_v\), meaning the set of permitted perturbations.

**Algorithm 1 CroPA: Cross Prompt Attack**

| Step | Operation |
|---|---|
| 0: | **Inputs:** Model \(f\), Target Text \(T\), vision input \(x_v\), prompt set \(X_t\), perturbation size \(\epsilon\), step sizes \(\alpha_1\) and \(\alpha_2\), number of iteration steps \(K\), adversarial prompt update interval \(N\) |
| 0: | **Output:** Adversarial example \(x'_v\) |
| 1: | Initialize \(x'_v=x_v\) |
| 2: | **for** step = 1 to \(K\) **do** |
| 3: | Uniformly sample the prompt \(x_t^i\) from \(X_t\) |
| 4: | **if** \(x_t^i\) is not initialized **then** |
| 5: | Initialize \(x_t^{\prime i}=x_t^i\) |
| 6: | **end if** |
| 7: | Compute gradient for adversarial image: \(g_v=\nabla_{x_v}L(f(x'_v,x_t^i),T)\) |
| 8: | Update with gradient descent: \(x'_v=x'_v-\alpha_1\cdot\operatorname{sign}(g_v)\) |
| 9: | **if** \(\operatorname{mod}(\text{step},N)=0\) **then** |
| 10: | Compute gradient for adversarial prompt: \(g_t=\nabla_{x_t}L(f(x'_v,x_t^i),T)\) |
| 11: | Update with gradient ascent: \(x_t^{\prime i}=x_t^{\prime i}+\alpha_2\cdot\operatorname{sign}(g_t)\) |
| 12: | **end if** |
| 13: | Project \(x'_v\) to be within the \(\epsilon\)-ball of \(x_v\): \(x'_v=\operatorname{Clip}_{x_v,\epsilon}(x'_v)\) |
| 14: | **end for** |
| 15: | **return** \(x'_v\) |

At evaluation, only the optimized image perturbation is used. Prompt perturbations are discarded. Thus, attack effectiveness must come from the image perturbation’s transferability, not from prompt modification.

## 5 Additional Investigative Experiments

CroPA’s iterative PGD optimization effectively increases ASR on a single model. However, reliance on feedback from that model limits transferability. Similar limitations occur in other gradient-based perturbation methods (Qin et al., 2024; Li et al., 2024; Wang et al., 2023; Li et al., 2023a). The original authors recognize this limitation. Table 9 in Appendix A.1 verifies it. CroPA also performs especially poorly on some tasks, including classification and captioning (Table 1).

CroPA is computationally intensive. Training adversarial perturbations for only 50 images takes over 6 hours on a 48GB L40S GPU, a graphics processing unit, while using 95% of its VRAM, or graphics memory.

Our extensions address two goals: (i) improve CroPA’s inconsistent performance across tasks; and (ii) investigate the transferability types defined in Section 1 and add methods that improve image-perturbation transferability.

Transferability is not a core proposal of Luo et al. (2024), but it could reduce computation. Ideally, a cross-image perturbation would remove the need to optimize a separate perturbation for every image. A cross-model perturbation could offer still greater efficiency.

The following subsections explain the motivations and mechanisms of our experiments.

### 5.1 Noise Initialization via Vision Encoding Optimization

CroPA and many gradient-based attacks start from random noise. This starting point does not use the target’s semantics. It can cause instability and slower convergence. This is especially problematic for VLMs, where semantic alignment with prompts is important. Alignment means making internal representations similar.

Our evaluation of CroPA’s bi-level optimization—optimization with two interacting objectives—in Section 6.5.1 shows strong initialization sensitivity (Figure 7 and Table 3). CroPA cannot guide its initial trajectory toward semantically meaningful directions. Conventional initialization therefore produces suboptimal ASR and reduced cross-prompt transferability.

We hypothesize that semantically informed initialization can improve attacks while keeping perturbations imperceptible. We use a diffusion-based image-generation model to synthesize a target image from the desired prompt. We then align the initial perturbation with that image in the vision encoder’s feature space, the space of numerical image representations.

Our method has three steps:

- Generate a semantically relevant target image using a state-of-the-art diffusion model, such as Stable Diffusion XL.
- Align the perturbation with the target image’s vision-encoder representation.
- Optimize using a Mean Squared Error (MSE)-based loss, which measures squared differences between representations.

Replacing random initialization with this vision-guided approach improves adversarial robustness and attack efficiency. It reduces the optimization steps needed for convergence (Section 6.5.1; Figure 7). The resulting perturbations are semantically informed from the start and provide stronger attacks and improved cross-prompt transferability.

#### 5.1.1 Noise Initialization

Preliminary experiments on BLIP-2 and InstructBLIP suggest that semantically informed initialization significantly improves attack effectiveness over random initialization. It adds only 20-25 seconds of computation per image on a single GPU.

The diffusion-based approach improves feature-space alignment. It also preserves the visual fidelity of adversarial images while respecting strict perturbation budgets.

#### 5.1.2 Diffusion-Based Target Synthesis

Given target prompt \(T\), we generate a target image \(\mathbf{x}_{\text{target}}\) that expresses its semantic attributes. We use Stable Diffusion XL (SDXL):

\[
\mathbf{x}_{\text{target}}
=\mathcal{D}(T,\mathbf{z};\theta_{\text{SDXL}}).
\tag{6}
\]

Here, \(\mathcal{D}\) is the diffusion image-generation process. The vector \(\mathbf{z}\) is sampled from a Gaussian, or normal, distribution. The parameters \(\theta_{\text{SDXL}}\) are the diffusion model’s pretrained weights, learned before this attack procedure. The resulting image captures the prompt’s semantic content and provides an informative initialization target.

#### 5.1.3 Vision-Encoder Anchored Perturbation

Let \(f_v(\cdot)\) be the VLM’s vision encoder. Let \(\mathbf{x}\) be the original image and \(\bm{\delta}\) its perturbation. We seek a perturbed image \(\mathbf{x}+\bm{\delta}\) whose encoder representation resembles the target image’s representation.

The initial perturbation \(\bm{\delta}_{\text{init}}\) is defined by

\[
\bm{\delta}_{\text{init}}
=
\underset{\|\bm{\delta}\|_{\infty}\leq\epsilon}{\arg\min}
\left\|
f_v(\mathbf{x}+\bm{\delta})
-f_v(\mathbf{x}_{\text{target}})
\right\|_2^2.
\tag{7}
\]

The operator \(\arg\min\) returns the perturbation that minimizes the expression. The \(\ell_\infty\) norm, written \(\|\cdot\|_\infty\), measures the largest absolute entry. Thus, \(\epsilon\) bounds each perturbation entry. The squared \(\ell_2\) norm, \(\|\cdot\|_2^2\), sums squared differences.

This initialization places the adversarial example in a semantically meaningful neighborhood of the target prompt’s representation. Algorithm 2 in Appendix B formally describes the method.

#### 5.1.4 Adversarial Optimization via PGD

We refine the adversarial image using PGD. Let \(\mathbf{x}_{\text{adv}}^{(t)}\) be the adversarial image at iteration \(t\). The update is

\[
\mathbf{x}_{\text{adv}}^{(t+1)}
=
\Pi_{\mathcal{B}_{\epsilon}(\mathbf{x})}
\Bigl[
\mathbf{x}_{\text{adv}}^{(t)}
-\alpha
\nabla_{\mathbf{x}_{\text{adv}}^{(t)}}
\mathcal{L}_{\text{MSE}}
\Bigr].
\tag{8}
\]

Here, \(\alpha\) is the step size. The set \(\mathcal{B}_{\epsilon}(\mathbf{x})\) is the admissible \(\ell_\infty\) ball around \(\mathbf{x}\). The projection operator \(\Pi\) returns the updated image to that set.

The loss \(\mathcal{L}_{\text{MSE}}\) compares vision-encoder outputs:

\[
\mathcal{L}_{\text{MSE}}
=
\left\|
f_v(\mathbf{x}_{\text{adv}})
-f_v(\mathbf{x}_{\text{target}})
\right\|_2^2.
\tag{9}
\]

Each update therefore moves the adversarial representation toward the target’s semantic embedding, or numerical representation.

### 5.2 Guiding perturbations via Target Value Vectors and evaluating Cross-Model Tranferability

CroPA optimizes image and text inputs through PGD. However, it lacks sensitivity to architecture, particularly the internal behavior of the vision encoder. The encoder has a dominant role in semantic grounding: connecting visual content to meaning.

CroPA treats vision and language as uniformly vulnerable. It does not exploit modality-specific weaknesses or directly target the encoder’s representational backbone. This limits structured perturbations, adversarial strength, and cross-model transferability. Its image perturbation minimizes the loss for a target output, while its text perturbation maximizes that loss.

Cui et al. (2023) and Wang et al. (2024) show that perturbing vision-encoder output embeddings can substantially impair the visual perception of large vision-language models (LVLMs). This can mislead their language-model responses. Different LVLMs also commonly use similar image encoders. These encoders produce visual tokens that contain semantic and structural information. Language decoders, the components that generate text, use these tokens to access and reason about images.

We propose combining CroPA’s loss with a loss targeting the encoder’s attention mechanism, specifically its value vectors. We hypothesize that aligning the input image’s value vectors with those of a reference image for the target text can significantly increase Adversarial Success Rate (ASR). We retain CroPA’s hyperparameter constraints. Hyperparameters are settings chosen for the optimization procedure rather than learned model parameters.

Our approach is inspired by Doubly-Universal Adversarial Perturbation (Doubly-UAP). Instead of maximizing losses between the value vectors of an input image \(x_i\) and a target-associated reference image \(T_i\), we minimize them. Prior work identifies value vectors as effective components for adversarial influence. We hypothesize that targeting essential visual features in these vectors will disrupt interpretation more effectively than CroPA’s generic PGD perturbations.

We also freeze the text encoder, meaning that its parameters remain unchanged. This keeps the perturbations focused on visual processing rather than text-based artifacts.

Our modification aims to:

1. Improve attack effectiveness by exploiting information in the vision encoder.
2. Improve semantic alignment between perturbations and the target text representation.
3. Achieve better adversarial robustness across models while remaining computationally feasible.

Jointly optimizing CroPA’s loss and the modified Doubly-UAP loss creates a more structured, interpretable, and targeted attack.

The modification adapts part of Kim et al. (2024). Their Doubly-UAP method identifies the attention components with the greatest influence on VLM performance. We focus on two fundamental components:

1. **Attention Weights:** These specify how strongly each image patch focuses on other patches. They determine patch interactions and relevance. We hypothesize that targeting them interferes with the encoder’s ability to establish those relationships.
2. **Value Vectors:** These contain the information within each patch. We expect that perturbing them disrupts essential content and impairs interpretation.

We target vision encoders because they are central to visual interpretation. Based on prior analysis, we focus on value vectors in middle-to-late attention layers.

In the standard Doubly-UAP attack, the optimized perturbation \(\delta^*\) is

\[
\delta^*
=
\arg\max_{\delta}
\frac{1}{|L|}
\sum_{l\in L}
\operatorname{Loss}\bigl(V_l(x),V_l(x+\delta)\bigr).
\tag{10}
\]

Here, \(\arg\max\) returns the perturbation that maximizes the expression. In this equation, \(L\) denotes a set of selected layers, rather than the language modeling loss used earlier. The symbol \(|L|\) is the number of layers in that set. The index \(l\) identifies a layer, \(V_l(x)\) gives its value vectors for image \(x\), and \(\operatorname{Loss}\) compares two sets of vectors.

#### 5.2.1 Our Modified Approach

We introduce a target value vector from a reference image associated with desired text \(T\). Instead of only maximizing deviation from the original image’s representation, we enforce alignment with this target representation.

Figure 2 uses D-UAP as the label for the Doubly-UAP-based component.

Figure 2: D-UAP Architecture Diagram

The procedure is:

1. Extract value vectors from the \(l\)-th vision-encoder layer for the input image and the target text’s associated image.
2. Compute a cosine similarity loss between the vectors. Cosine similarity measures their directional alignment.
3. Jointly minimize the negative of this loss and the CroPA loss, with appropriate scaling for adversarial robustness and target alignment.

Let \(V_i(x+\delta)\) denote the perturbed input’s value vectors at attention head \(i\). An attention head is one component that computes attention relationships. Let \(V_t\) be the reference image’s target value vectors. Define

\[
L_{\text{d-UAP}}
=
\sum_{i=1}^{N}
\left(
1-
\frac{V_i(x+\delta)\cdot V_t}
{\|V_i(x+\delta)\|_2\|V_t\|_2}
\right).
\tag{11}
\]

Here, \(N\) is the number of attention heads, not the update interval from Section 4.3. The centered dot is a dot product, and \(\|\cdot\|_2\) is vector length. This loss encourages the perturbed image’s value vectors to align closely with the target representation.

We combine it with the CroPA loss:

\[
L_{\text{re-CroPA}}
=
L_{\text{CroPA}}(x_v+\delta_v,x_t+\delta_t,T)
-\lambda L_{\text{d-UAP}}(\delta_v).
\tag{12}
\]

The symbol \(L_{\text{re-CroPA}}\) names the combined objective. The hyperparameter \(\lambda\) controls the relative importance of the value-vector loss.

During optimization, gradients from both components update the perturbation \(\delta\). The joint objective preserves adversarial behavior and enforces target semantic alignment. It combines cross-prompt transferability with guided semantic influence. This permits more precise, controlled, and interpretable VLM manipulation for robustness and security analysis. Algorithm 3 in Appendix B gives the steps.

We propose this as a step toward stronger CroPA transferability. However, D-UAP depends on model-specific value vectors and therefore on the vision encoder. We hypothesize that transfer may be limited to models with the same or similar vision-encoder architecture as the model used for optimization.

### 5.3 Transferability across images via Image Augmentation

Traditional adversarial methods use iterative gradient descent on individual images. They can overfit, meaning that they become too specialized to those images’ features. The resulting poor cross-image transferability also makes attacks expensive.

CroPA likewise learns a perturbation for one image across multiple prompts. Luo et al. (2024) describe their method as orthogonal to cross-image approaches: it addresses a separate transferability dimension. They suggest that computing perturbations across images could improve cross-image transfer, as in Moosavi-Dezfooli et al. (2017). We test that claim and add an approach based on UAP research.

Fang et al. (2024) introduce UAPs for Vision-Language Pretraining (VLP) models to reduce overfitting and enable cross-image attacks. VLP refers to training models on visual and textual information before their later use.

CutMix provides a complementary regularization strategy, meaning a method intended to reduce overfitting (Yun et al., 2019). It cuts rectangular regions from one image and pastes them into another. It also mixes labels in proportion to patch area. Unlike pixel blending, this trains models on partial views. It encourages spatially invariant features—features useful despite changes in position—and improves robustness to adversarial attacks and occlusion, where part of an image is hidden.

Zhang et al. (2024) propose Effective and Transferable Universal Adversarial Attack (ETU). It improves UAP transfer across VLP models and tasks through global and local optimization. ETU introduces SCMix, an augmentation method combining self-mixing and cross-mixing. SCMix increases input diversity while preserving semantic integrity. It seeks both spatially invariant and scale-invariant features, which remain useful despite changes in size.

We integrate SCMix into perturbation generation. We select it because Zhang et al. (2024) report better cross-image transfer on the same model using SCMix alone, without their additional local utility reinforcement component.

#### 5.3.1 SCMix

SCMix has two stages: self-mixing and cross-mixing.

In self-mixing, \(I_1\) is the base image. Randomly cropped and resized versions \(X_1\) and \(X_2\) are blended to produce \(I'_1\):

\[
I'_1=\eta X_1+(1-\eta)X_2
\quad\text{where}\quad
X_i=\operatorname{Resize}(\operatorname{RandomCrop}(I_1)),
\quad\eta=0.5.
\tag{13}
\]

Here, \(\operatorname{RandomCrop}\) selects an image region, \(\operatorname{Resize}\) changes its size, and \(\eta\) is the blending weight.

Original SCMix samples \(\eta\sim\operatorname{Beta}(\alpha,\alpha)\), where \(\operatorname{Beta}\) is a probability distribution on blending weights and \(\alpha>0\) controls its shape. We fix \(\eta=0.5\). This is equivalent to setting \(\alpha=\infty\), as justified in Appendix D.3.

Cross-mixing adds a secondary image \(I_2\):

\[
I_3=\beta_1 I'_1+\beta_2 I_2
\quad\text{with}\quad
\beta_1\gg\beta_2\in[0,1).
\tag{14}
\]

The output is \(I_3\). The weights are \(\beta_1\) and \(\beta_2\), and \(\gg\) means “much greater than.” The interval \([0,1)\) includes 0 and excludes 1.

The weighting preserves the dominant semantics of \(I_1\) while adding diversity from \(I_2\). This lets the perturbation generator learn across diverse images. The dominance of \(I_1\) also preserves compatibility between \(I_3\) and \(T_1\), the text description of \(I_1\). The synthetic image therefore remains aligned with its description despite including features from \(I_2\).

The two stages reduce overfitting by forcing perturbations to attack features that survive within-image variation and between-image blending. This promotes universal perturbations that may work on unseen images without image-specific attack training.

At inference, or evaluation use, CroPA returns only the perturbed image, equivalently the image’s perturbation. Thus, if a cross-image perturbation can be learned, the attack can become universal in both the cross-prompt and cross-image senses.

Figure 3: Illustration of the self-mixing and cross-mixing processes. The input image \(I_1\) undergoes self-mixing to generate intermediate representations \(X_1\) and \(X_2\), which are then combined to form \(I_1'\). Additionally, an external image \(I_2\) is mixed with \(I_1'\) to produce the final output \(I_3\).

#### 5.3.2 CutMix

CutMix uses discrete regional replacement to improve transferability (Yun et al., 2019). For images \(x_A\) and \(x_B\), the composite image \(\tilde{x}\) is

\[
\tilde{x}
=
M\odot x_A+(1-M)\odot x_B.
\tag{15}
\]

Here, \(\odot\) means element-wise multiplication. The mask \(M\in\{0,1\}^{W\times H}\) has binary entries over image width \(W\) and height \(H\). It defines a randomly placed rectangular region. The complementary mask \(1-M\) selects the other image.

CutMix learns from partial views. SCMix softly blends features within and across images; CutMix instead uses hard, mask-based replacement. This discrete replacement forces perturbations to target spatially invariant features that persist across image combinations.

When integrated with adversarial training, CutMix encourages attacks on higher-level semantic patterns rather than localized artifacts. This may improve cross-image transferability.

### 5.4 Attack Setup and Threat Models for Transferability Experiments

A threat model specifies the attacker’s objective, knowledge, and capabilities. We define separate threat models and evaluation procedures for cross-model and cross-image transfer.

#### 5.4.1 Using D-UAP to explore Cross-Model Transferability for CroPA

**Adversary’s Goal:** Generate one perturbation for an image that causes multiple VLMs to output a specific target text, regardless of prompt. This is a black-box transfer setting: the attacker optimizes on an accessible model and attacks unknown models.

**Knowledge and Capabilities:** The attacker has white-box access to a surrogate model, the model used for attack optimization. In our experiments, this is BLIP-2 or Flamingo. Access includes gradients and internal representations. For target models, the attacker knows only the general architecture family, such as CLIP-based models. CLIP is the image-encoder family named in this setup. The attacker cannot access target-model parameters.

The maximum perturbation budget is \(\epsilon=16/255\) under the \(L_\infty\) norm, also written \(\ell_\infty\). This constraint is intended to ensure visual imperceptibility.

**Cross-Model Settings:**

- **Intra-family transfer:** Optimize on BLIP-2 with the OPT-2.7b language model and evaluate on InstructBLIP with the Vicuna-7b language model. These are the named language-model variants used. Both VLMs share similar CLIP ViT-L/14 vision encoders, a named Vision Transformer variant, but differ in language models and fine-tuning objectives. Fine-tuning is additional training for a particular use.
- **Cross-architecture transfer:** Transfer from BLIP-2 to Flamingo and from Flamingo to BLIP-2. These models use different vision encoders, language models, and architectural designs.

**Evaluation Setup:** Measure ASR separately for VQAgeneral, VQAspecific, Classification, and Captioning. VQAgeneral and VQAspecific distinguish how specific a question prompt is to an image. Success means that the target model fails to generate the correct output for the perturbed image, regardless of prompt.

#### 5.4.2 Investigating Cross-Image Transferability via Augmentation Techniques

**Adversary’s Goal:** Generate one perturbation pattern that causes a VLM to output predetermined text for any image, regardless of prompt or image content. This is more efficient than optimizing separately for each image.

**Knowledge and Capabilities:** The attacker has a limited training-image set and corresponding SCMix-augmented variants. The attacker has white-box access to the target model during optimization. The perturbation must generalize to unseen images and satisfy an \(L_\infty\) budget of \(\epsilon=16/255\).

**Evaluation Setup:** Test the universal perturbation on a separate image set not used in training. Measure ASR across task types. Success means that the model fails to generate the correct output text for a previously unseen image, regardless of content or prompt.

SCMix is intended to counter the tendency of gradient-based methods to overfit image-specific features. We hypothesize that controlled self-mixing and cross-mixing will reveal model vulnerabilities that do not depend on particular images.

#### 5.4.3 Evaluation Rationale for Transferability experiments

We present all cross-prompt and cross-model transferability results under the untargeted setting. There are two main reasons.

First, VLMs integrate visual information through different embedding mechanisms. This makes it difficult to determine how input perturbations affect the final output-token embeddings. Targeted transfer is notably higher between models with similar vision encoders, such as BLIP-2 and InstructBLIP. It varies sharply otherwise. Direct comparison of targeted ASRs across heterogeneous architectures is therefore imprudent.

Second, visual-textual fusion—the combination of image and text information—affects adversarial control. Decoder-only VLMs, such as LLaVA and OpenFlamingo, incorporate image features late through cross-attention or token concatenation. Cross-attention lets one representation attend to another; concatenation places tokens together in a sequence. These models treat visual information as auxiliary memory during causal language generation, which generates text from preceding context. Visual influence consequently fades as decoding proceeds (Awadalla et al., 2023; Liu et al., 2023).

Fusion-based models, including BLIP-2 and InstructBLIP, integrate visual information early and deeply into input embeddings. This maintains visual grounding throughout generation (Li et al., 2023b; Dai et al., 2023). Targeted attacks are therefore inherently more feasible in fusion-based models than in decoder-only models (Dai et al., 2023).

Given these differences, non-targeted ASR provides a more robust, fair, and comparable measure of cross-model and cross-image transfer.

We exclude CIA from transferability comparisons because our experiments found negligible transfer. Its method also has these limitations:

- **Image-Specific Semantic Embedding:** CIA embeds target concepts through per-image visual-token manipulation. Gradient directions and token spaces differ greatly across images and concepts. The resulting input-to-target token transformations depend on the image and lack cross-image consistency.
- **Model-Specific Embedding Architectures:** CIA depends strongly on the structures of visual and textual encoders. Structural differences among LVLMs, including BLIP2, LLaVA, and Flamingo, prevent demonstrated cross-model transferability.

## 6 Results

Our main reproducibility objective is to reproduce Luo et al. (2024)’s cross-prompt results for CroPA, as stated in Section 4. We compare our results with the original paper.

**Models Used:**

We evaluated five state-of-the-art VLMs: OpenFlamingo, BLIP-2, InstructBLIP and LLaVA.

For Flamingo, we used the publicly accessible OpenFlamingo-9B implementation (Awadalla et al., 2023). It provides performance comparable to the original model.

BLIP-2 uses a two-stage, fusion-based approach. A frozen CLIP image encoder first extracts visual features. A Querying Transformer then processes them (Li et al., 2023b). This architecture supports efficient adaptation to different vision-language tasks. Its language-model component is OPT-2.7b.

InstructBLIP builds on BLIP-2’s fusion-based architecture and adds instruction tuning, training intended to improve responses to task instructions (Dai et al., 2023). It uses Vicuna-7b as its language model. This supports more precise output control through prompts.

LLaVA uses decoder-only generation (Liu et al., 2023). A pretrained CLIP ViT-L vision encoder connects to Vicuna-13B through a lightweight projection layer, which maps visual features into the language-model representation. LLaVA places those features directly into the language-model context as special prompt tokens.

The models therefore represent two approaches:

- **Decoder-only:** OpenFlamingo and LLaVA use visual embeddings as generation context.
- **Fusion-based:** BLIP-2 and InstructBLIP explicitly align image and text representations before generation.

These differences significantly affect targeted adversarial effectiveness across prompts.

We used the authors’ public code and preserved the original model configurations for faithful comparison. The implementation was generally well documented. However, BLIP-2 and InstructBLIP produced multiple runtime errors. Significant debugging and adjustments were required to run them successfully.

### 6.1 Claim 1: CroPA achieves cross-prompt transferability across various target texts. [Reproduced]

We replicated Luo et al. (2024)’s setup to test CroPA’s cross-prompt transferability. We trained adversarial examples with CroPA and evaluated them on Flamingo with diverse target prompts. We measured targeted ASR on general VQA, specific VQA, classification, and captioning.

#### 6.1.1 Experimental Details

We evaluated adversarial examples with prompts distinct from those used during training. Table 1 lists the target prompts used in this evaluation. They span broad semantic meanings. Appendix D.1 provides additional details for all reproduced experiments.

#### 6.1.2 Comparative Analysis

Table 1 reports Flamingo’s targeted ASRs for different target texts. The results support CroPA’s cross-prompt transferability. ASRs are non-zero across all evaluated tasks, including with diverse prompts.

CroPA nevertheless consistently underperforms on classification and captioning. This is especially evident for abstract texts with limited visual semantics, such as “too late” and “not sure” (Table 1, corresponding CroPA rows).

| Target Prompt | Method | VQAgeneral | VQAspecific | Classification | Captioning | Overall |
|---|---|---:|---:|---:|---:|---:|
| unknown | Single-P | 0.2640 | 0.4720 | 0.2700 | 0.1400 | 0.2865 |
| unknown | Multi-P | 0.7240 | 0.8740 | 0.5550 | 0.2850 | 0.6095 |
| unknown | CroPA | 0.9680 | 0.9880 | 0.7070 | 0.4200 | 0.7708 |
| I am sorry | Single-P | 0.2120 | 0.4740 | 0.5170 | 0.2530 | 0.3640 |
| I am sorry | Multi-P | 0.7020 | 0.9020 | 0.7090 | 0.6140 | 0.7318 |
| I am sorry | CroPA | 0.8620 | 0.9260 | 0.7170 | 0.6890 | 0.7985 |
| not sure | Single-P | 0.3540 | 0.4240 | 0.1450 | 0.0000 | 0.2308 |
| not sure | Multi-P | 0.6520 | 0.6860 | 0.2310 | 0.0030 | 0.3930 |
| not sure | CroPA | 0.8760 | 0.8940 | 0.2420 | 0.0010 | 0.5033 |
| very good | Single-P | 0.3980 | 0.5760 | 0.2180 | 0.0480 | 0.3100 |
| very good | Multi-P | 0.8940 | 0.9640 | 0.4040 | 0.2080 | 0.6175 |
| very good | CroPA | 0.9620 | 0.9860 | 0.6060 | 0.2540 | 0.7020 |
| too late | Single-P | 0.2520 | 0.4800 | 0.2280 | 0.0220 | 0.2455 |
| too late | Multi-P | 0.7880 | 0.9120 | 0.5040 | 0.1630 | 0.5918 |
| too late | CroPA | 0.9300 | 0.9580 | 0.7010 | 0.1790 | 0.6920 |
| metaphor | Single-P | 0.3280 | 0.6020 | 0.5550 | 0.1040 | 0.3973 |
| metaphor | Multi-P | 0.8880 | 0.9520 | 0.8420 | 0.4910 | 0.7933 |
| metaphor | CroPA | 0.9840 | 0.9940 | 0.9100 | 0.5840 | 0.8680 |

Table 1: Targeted ASRs tested on Flamingo with different target texts, across a variety of Vision-Language tasks. Here VQAgeneral and VQAspecific, refer to the degree of specificity of prompts pertaining to a particular image. The prompts used are reported in Appendix E

These experiments reproduce the core finding that CroPA has robust cross-prompt transferability. They validate the vulnerability of VLMs to these attacks.

### 6.2 Claim 2: CroPA achieves the best overall performance across different number of prompts [Reproduced]

We compare CroPA and the baselines as the number of prompts changes (Figure 4). ASR increases with prompt count for all methods. Single-P necessarily uses one prompt, so it appears as a lone star.

At the same prompt count, CroPA consistently achieves higher ASR than the baselines. This supports its cross-prompt motivation. We use “unknown” as the target text to avoid high-frequency responses common in vision-language tasks. We consider this rationale from Luo et al. (2024) valid.

The baseline’s ASR also saturates below CroPA’s as prompts are added (Figure 4). Thus, adding prompts does not allow the baseline to surpass CroPA. This agrees with Luo et al. (2024).

Figure 4: Targeted ASR tested with different number of prompts on Flamingo. With the same number of attack iterations, our CroPA significantly outperforms baselines. "unknown" is used as the target text.

### 6.3 Claim 3: CroPA converges towards higher ASR as number of iterations increase [Reproduced]

Our convergence investigation produced results that contrast with some of the original reported convergence metrics, despite using the same training and evaluation parameters and the authors’ code. Appendix A.2 contains the authors’ results. The original metrics showed CroPA converging with a higher number of prompts.

Our overall ASR across tasks follows that trend (Figure 6), reproducing the type of figure reported in the original paper. However, individual-task convergence is unstable (Figure 5). This task-level analysis therefore reveals behavior hidden by the overall measure.

The original authors did not specify the evaluation model for this scenario. We use OpenFlamingo and the target text “unknown.”

(a) VQA

(b) VQA-Specific

(c) Classification

(d) Captioning

Figure 5: Comparison of Single-P, Multi-P, and CroPA across different categories.

Figure 6: Overall Convergence and Comparison of CroPA

### 6.4 Claim 4: CroPA outperforms baselines under few-shot settings [Reproduced]

Flamingo accepts extra images as in-context learning examples in addition to the text prompt. These examples improve task adaptation.

Luo et al. (2024) tested whether adversarial images remain effective when the number of in-context examples changes between optimization and evaluation. Optimization uses the 0-shot setting: no extra example images. Evaluation uses the 2-shot setting: two extra example images. Table 2 compares these settings.

| Setting | Method | VQAgeneral | VQAspecific | Classification | Captioning | Overall |
|---|---|---:|---:|---:|---:|---:|
| shot=0 | Multi-P | 0.7240 | 0.8740 | 0.5550 | 0.2850 | 0.6095 |
| shot=0 | CroPA | 0.9680 | 0.9880 | 0.7070 | 0.4200 | 0.7708 |
| shot=2 | Multi-P | 0.7860 | 0.9420 | 0.5630 | 0.0010 | 0.5730 |
| shot=2 | CroPA | 0.9440 | 0.9940 | 0.6860 | 0.0300 | 0.6635 |

Table 2: Targeted ASRs with and without visual in-context learning. The shot indicates the number of images added for in-context learning.

In-context examples inherently make attacks harder by shifting generation conditions away from those used in optimization (Luo et al., 2024). Extra images also make the model more aware of the correct prompt space, which differs from the target-text space. Zhao et al. (2023) and Qi et al. (2023) verify that attacks optimized for a particular context lose effectiveness when that context changes.

CroPA nevertheless significantly outperforms the baseline on most tasks (Table 2, shot=2 rows). It is more robust to context expansion through examples. This verifies the motivation for the min-max objective in Section 4.3: robustness to a wider prompt space and resulting context changes.

### 6.5 Results Beyond the Original Paper

The following experiments extend CroPA with enhancements intended to mitigate its limitations.

#### 6.5.1 Noise Initialization

Noise Initialization via Vision Encoding Optimization, described in Section 5.1, significantly improves performance across tasks. Semantically informed initialization provides superior adversarial robustness and cross-prompt transferability.

Our method, labeled **CroPA+Init**, consistently outperforms CroPA and Multi-P across all metrics, achieving 6-15% gains in overall accuracy (Table 3). The vision-encoder-anchored perturbations also reduce optimization steps (Figure 7). These results validate semantic initialization as a robust and adaptable alternative to random noise.

| Target Prompt | Method | VQAgeneral | VQAspecific | Classification | Captioning | Overall |
|---|---|---:|---:|---:|---:|---:|
| unknown | Multi-P | 0.7240 | 0.8740 | 0.5550 | 0.2850 | 0.6095 |
| unknown | CroPA | 0.9680 | 0.9880 | 0.7070 | 0.4200 | 0.7708 |
| unknown | CroPA+Init | 0.8824 | 0.8920 | 0.8858 | 0.9306 | 0.8977 |
| bomb | Multi-P | 0.6840 | 0.8260 | 0.9050 | 0.6090 | 0.7560 |
| bomb | CroPA | 0.8176 | 0.9100 | 0.9498 | 0.6960 | 0.8433 |
| bomb | CroPA+Init | 0.9140 | 0.9468 | 0.9020 | 0.9412 | 0.9260 |
| I am sorry | Multi-P | 0.7020 | 0.9020 | 0.7090 | 0.6140 | 0.7318 |
| I am sorry | CroPA | 0.8620 | 0.9260 | 0.7170 | 0.6890 | 0.7985 |
| I am sorry | CroPA+Init | 0.9320 | 0.9524 | 0.9270 | 0.9716 | 0.9458 |
| very good | Multi-P | 0.7900 | 0.8860 | 0.6700 | 0.6700 | 0.7540 |
| very good | CroPA | 0.8540 | 0.8990 | 0.7680 | 0.7550 | 0.8190 |
| very good | CroPA+Init | 0.8745 | 0.8894 | 0.8362 | 0.9043 | 0.8761 |
| too late | Multi-P | 0.6960 | 0.8540 | 0.6780 | 0.6060 | 0.7585 |
| too late | CroPA | 0.8990 | 0.9060 | 0.7940 | 0.8150 | 0.8535 |
| too late | CroPA+Init | 0.9340 | 0.9220 | 0.9110 | 0.9660 | 0.9333 |

Table 3: Targeted ASRs on Blip2 with different target texts using CroPA Vision Encoder noise initialization. The detailed hyperparameters used are discussed under section D.2.

##### Theoretical Analysis of Initialization in Bi-level Optimization

We attribute the gains from diffusion-guided initialization to three challenges in CroPA’s min-max optimization.

First, CroPA resembles a saddle-point problem: the objective is minimized in one variable and maximized in another. Initialization strongly affects stability in such problems. Zhang et al. (2025) show that min-max problems are significantly more initialization-sensitive than standard minimization problems. Carefully designed starting points are often needed for stable convergence. In CroPA, prompt perturbations continuously change the optimization landscape—the loss as a function of possible updates—making this sensitivity particularly important.

Second, CroPA updates image perturbations more often than prompt perturbations. This creates optimization asymmetry. Chen et al. (2019) show that alternating gradient methods for min-max problems often cycle locally instead of settling. Convergence depends strongly on the initial point’s proximity to adversarially robust regions. Our semantic initialization deliberately starts closer to meaningful adversarial directions.

Third, our observations align with Wu et al. (2020). They show that perturbations initialized with semantic priors—target information supplied before optimization—transfer better across models than randomly initialized perturbations. This substantially improves ASR. The effect is particularly important in bi-level settings with complex optimization paths.

Our method initializes toward semantic targets rather than random noise. This supports faster convergence and more transferable adversarial examples. Table 3 reports significant ASR increases.

Table 4 reports average targeted ASRs. It also compares CIA, called Contextual Injection Attack here.

| Method | VQAgeneral | VQAspecific | Classification | Captioning | Overall |
|---|---:|---:|---:|---:|---:|
| Multi-P | 0.7053 | 0.8807 | 0.7001 | 0.5553 | 0.7104 |
| CroPA | 0.8723 | **0.9457** | 0.7518 | 0.6138 | 0.7958 |
| CIA | 0.2985 | 0.2281 | 0.4857 | 0.4687 | 0.370 |
| CroPA+Init | **0.9145** | 0.9342 | **0.8982** | **0.9464** | **0.9209** |

Table 4: Average Targeted ASRs on BLIP-2 for different methods. Results show the impact on different vision-language tasks: Visual Question Answering (general and specific), Classification, and Captioning. The best performance values for each task are highlighted in bold. Apart from the previously mentioned baselines, we also compare our results with the Contextual Injection Attack (CIA) Yang et al. (2024) as an additional baseline.

Gradient-based attacks often acquire their targets’ semantics. In theory, they should not need semantic initialization. However, we argue that CroPA’s min-max complexities weaken gradient-produced semantic alignment. Adding target semantics at initialization is necessary to strengthen it and improve convergence. Figure 7 provides experimental support.

We measure target-feature incorporation using a **CLIP score**, a similarity score between the target text and the learned noise perturbation under CLIP representations. We compare base CroPA perturbations with those from CroPA+Init.

CroPA acquires target context, as expected for a gradient-based attack. Initialization produces a modest CLIP-score increase, from 21.80 to 23.09 for “Bomb” (Figure 7b). This suggests a subtle shift toward adversarially vulnerable regions rather than a large increase in acquired semantic features.

The convergence comparison also supports our hypothesis: placing the initial point closer to meaningful adversarial directions leads to faster convergence (Figure 7a).

| Method | CLIP Score |
|---|---:|
| CroPA | 21.80 |
| CroPA+Init | 23.09 |

Target Text: "Bomb"

(a) Overall convergence comparison

(b) CLIP-score comparison

Figure 7: (a) Overall Convergence Comparison of base CroPA and CroPA+Init. (b) Quantifying feature similarity of learned perturbation to target by CLIP score.

#### 6.5.2 Investigating Cross-Image transferability and Image Augmentation

CroPA retains adversarial effectiveness across prompts. However, its perturbations remain tightly linked to the low-level features of the image used to generate them. This prevents effective generalization to other images and makes universal cross-image attacks ineffective. It weakens the original authors’ claim that learning a perturbation across images would produce cross-image transferability.

We add SCMix and CutMix, described in Section 5.3, to increase input diversity during training. The goal is to move beyond image-specific features.

We believe CutMix outperforms SCMix in promoting adversarial transferability because it preserves distinct, non-overlapping image regions. This encourages modular, independently manipulable features that can generalize across inputs.

Table 5 shows measurable improvements from both methods. Cross-image transfer nevertheless remains difficult. The performance gap between these cross-image methods and CroPA applied to a single image reveals a fundamental limitation: CroPA excels at cross-prompt robustness but does not effectively support transferable attacks across images.

| Method | VQA | VQAspecific | Classification | Captioning | Overall |
|---|---:|---:|---:|---:|---:|
| CroPA (Base) | 0.3500 | 0.4920 | 0.4000 | 0.5000 | 0.4355 |
| CroPA with SCMix | 0.6560 | 0.7160 | 0.5000 | 0.9500 | 0.7055 |
| CroPA with CutMix | 0.4520 | 0.7240 | 0.5000 | 0.5500 | 0.5565 |

Table 5: Untargeted ASR of CroPA in cross-image settings with and without SCMix and CutMix. The results represent the best ASRs achieved across all iterations. CroPA without input mixing attained its highest ASRs at 1700 iterations, while CroPA with SCMix and CutMix outperformed it significantly earlier, at just 1500 iterations and 1600 iterations, respectively. Additional details on experimental parameters are reported in Appendix D.3.

#### 6.5.3 Guiding perturbations via Target Value Vectors

Tables 6 and 7 report results for the D-UAP-modified CroPA framework from Section 5.2.

CroPA+D-UAP substantially improves the existing algorithm across tasks. On BLIP-2, its overall ASR is 96.85%, a 12.5% increase over base CroPA (Table 6, CroPA and CroPA+D-UAP rows). VQAgeneral and Captioning improve by 13% and 28%, respectively (the same rows).

The target text is “Bomb.” This demonstrates performance with harmful text expected to trigger model guardrails, or safety mechanisms. Thus, even in the worst expected cases involving harmful words, the enhancement performs significantly better in vulnerable scenarios.

| Method | VQAgeneral | VQAspecific | Classification | Captioning | Overall |
|---|---:|---:|---:|---:|---:|
| Multi-P | 0.68 | 0.82 | 0.90 | 0.60 | 0.75 |
| CroPA | 0.8176 | 0.9100 | 0.9498 | 0.6960 | 0.8433 |
| CIA | 0.3431 | 0.3102 | 0.4732 | 0.5534 | 0.4199 |
| CroPA+D-UAP | 0.9420 | 0.9720 | 0.9790 | 0.9810 | 0.9685 |

Table 6: Targeted ASRs on Blip2 using D-UAP function and target text being "Bomb", outperforming PGD baselines and CIA, as well as the base CroPA method.

| Method | VQAgeneral | VQAspecific | Classification | Captioning | Overall |
|---|---:|---:|---:|---:|---:|
| Multi-P | 0.6960 | 0.8540 | 0.8990 | 0.6060 | 0.7638 |
| CroPA | 0.7900 | 0.8860 | 0.9470 | 0.6700 | 0.8233 |
| CIA | 0.3027 | 0.4302 | 0.5112 | 0.5080 | 0.4380 |
| CroPA+D-UAP | 0.9000 | 0.9520 | 0.9500 | 0.9060 | 0.9270 |

Table 7: Targeted ASRs on Open Flamingo using D-UAP Loss function and target text being "Bomb".

#### 6.5.4 Analyzing Cross Model Transferability by Value Vector guidance

The method also improves cross-model transferability over vanilla, or unmodified, CroPA (Table 8). Classification and Captioning show especially large improvements, 14% and 21%, respectively, with a modest VQA increase.

We attribute the task difference to prompt complexity. VQA prompts span a more complex embedding space. Classification and captioning outputs occupy much smaller spans.

A key limitation confirms our hypothesis from Section 5.2. Value vectors depend on the VLM’s specific vision encoder. This can restrict transferability to models with the same or similar encoder architecture.

In Table 8, perturbations trained on BLIP2 transfer passably to InstructBLIP, which uses the same vision encoder, but fail to transfer to OpenFlamingo. The expected pattern also occurs when training on Flamingo and applying the perturbations to other models: ASR is insignificant (Table 8, Flamingo-source settings; also Table D.4.5 in Appendix D.4.5). Appendix D.4 reports experimental details.

In Table 8, **Cls** abbreviates Classification, and **Cap** abbreviates Captioning. A setting “A to B” means that perturbations optimized on model A are evaluated on model B.

| Settings | Method | VQAgeneral | VQAspecific | Cls | Cap | Overall |
|---|---|---:|---:|---:|---:|---:|
| BLIP2 to InstructBLIP | Multi-P | 0.1760 | 0.2090 | 0.2750 | 0.2480 | 0.2270 |
| BLIP2 to InstructBLIP | CroPA | 0.3220 | 0.4170 | 0.6720 | 0.6150 | 0.5065 |
| BLIP2 to InstructBLIP | CroPA + D-UAP | 0.3940 | 0.5780 | 0.8800 | 0.8880 | 0.6850 |
| BLIP2 to Flamingo | Multi-P | 0.0810 | 0.0660 | 0.0760 | 0.0890 | 0.0780 |
| BLIP2 to Flamingo | CroPA | 0.1520 | 0.1220 | 0.1180 | 0.1820 | 0.1435 |
| BLIP2 to Flamingo | CroPA + D-UAP | 0.1850 | 0.2190 | 0.1420 | 0.3880 | 0.2335 |
| BLIP2 to LLaVA | Multi-P | 0.1220 | 0.1060 | 0.0810 | 0.0630 | 0.0930 |
| BLIP2 to LLaVA | CroPA | 0.3190 | 0.2570 | 0.2730 | 0.3670 | 0.3040 |
| BLIP2 to LLaVA | CroPA + D-UAP | 0.5070 | 0.3420 | 0.4380 | 0.8430 | 0.5325 |
| InstructBLIP to BLIP2 | Multi-P | 0.1230 | 0.1520 | 0.1020 | 0.0960 | 0.1183 |
| InstructBLIP to BLIP2 | CroPA | 0.2690 | 0.2320 | 0.1870 | 0.2270 | 0.2288 |
| InstructBLIP to BLIP2 | CroPA+D-UAP | 0.4800 | 0.5100 | 0.4700 | 0.5000 | 0.4900 |
| InstructBLIP to LLaVA | Multi-P | 0.0650 | 0.0580 | 0.0420 | 0.0370 | 0.0505 |
| InstructBLIP to LLaVA | CroPA | 0.1620 | 0.1120 | 0.1410 | 0.1230 | 0.1345 |
| InstructBLIP to LLaVA | CroPA+D-UAP | 0.3450 | 0.3150 | 0.3680 | 0.3950 | 0.3558 |
| InstructBLIP to Flamingo | Multi-P | 0.0320 | 0.0210 | 0.0120 | 0.0110 | 0.0190 |
| InstructBLIP to Flamingo | CroPA | 0.0740 | 0.0680 | 0.0490 | 0.0480 | 0.0598 |
| InstructBLIP to Flamingo | CroPA+D-UAP | 0.1600 | 0.1400 | 0.1200 | 0.1300 | 0.1375 |
| Flamingo to BLIP2 | Multi-P | 0.0140 | 0.0070 | 0.0090 | 0.0120 | 0.0105 |
| Flamingo to BLIP2 | CroPA | 0.0450 | 0.0210 | 0.0300 | 0.0380 | 0.0335 |
| Flamingo to BLIP2 | CroPA+D-UAP | 0.1080 | 0.0800 | 0.0990 | 0.1180 | 0.1013 |
| Flamingo to InstructBLIP | Multi-P | 0.0100 | 0.0060 | 0.0050 | 0.0090 | 0.0075 |
| Flamingo to InstructBLIP | CroPA | 0.0360 | 0.0220 | 0.0280 | 0.0300 | 0.0290 |
| Flamingo to InstructBLIP | CroPA+D-UAP | 0.1050 | 0.0720 | 0.0880 | 0.1100 | 0.0938 |
| Flamingo to LLaVA | Multi-P | 0.0180 | 0.0110 | 0.0140 | 0.0170 | 0.0150 |
| Flamingo to LLaVA | CroPA | 0.0400 | 0.0300 | 0.0350 | 0.0430 | 0.0370 |
| Flamingo to LLaVA | CroPA+D-UAP | 0.1100 | 0.0850 | 0.0980 | 0.1170 | 0.1025 |

Table 8: The cross model test under different settings. "BLIP2 to InstructBLIP" means the perturbations optimised on BLIP2 are tested on the InstructBLIP and similarly for "BLIP2 to Flamingo"

We also observed that, for a fixed target text, the vision encoder’s value vectors remain approximately consistent across input images. This has three implications:

1. **Vision Encoder Specificity:** The vectors reflect the encoder’s architecture and pretraining.
2. **Limited Diversity in Perturbations:** Their consistency may constrain perturbation diversity and the ability to generate highly varied adversarial examples.
3. **Target Text Dependency:** Similar vectors across images for the same target text indicate that target choice strongly influences the attack outcome, rather than the input image’s specific features.

## 7 Discussion

### 7.1 Observations with reproducibility

We reproduced key aspects of Luo et al. (2024)’s CroPA framework and substantiated its claims about improved cross-prompt adversarial transferability.

CroPA shows strong transferability across VQA, classification, and captioning, and consistently outperforms the baselines (Table 1). It also remains robust in few-shot settings (Section 6.4; Table 2). It tolerates the change in generation conditions introduced by in-context examples better than the baseline.

### 7.2 Noise Initialization findings

Section 5.1 identifies a critical limitation in CroPA and related methods: random initialization ignores the target prompt’s semantic structure. We hypothesized that this causes instability, slower convergence, and reduced cross-prompt and cross-modal transferability. Cross-modal transferability concerns transfer between the image and text modalities.

We address this with a diffusion-generated reference image and vision-encoder feature alignment. This gives perturbations a meaningful adversarial direction from the start.

Section 6.5.1 validates this hypothesis. Across all evaluated benchmarks, the method significantly outperformed CroPA and random-initialized baselines. It produced up to 15% gains in targeted ASR (Table 3) and reduced optimization steps (Figure 7a).

These results show that semantic anchoring improves attack effectiveness and efficiency. It provides a more principled and robust starting point than random noise and offers a foundation for improved VLM attack frameworks.

### 7.3 Evaluation of Transferability Experiments

#### 7.3.1 Cross-Image Transferability

The original authors suggest that computing perturbations across images can produce cross-image transferability. Our experiments instead show that CroPA’s perturbations remain tightly coupled to the low-level features of the image used to generate them, despite strong cross-prompt performance.

We integrate SCMix and CutMix to increase image diversity while preserving the original images’ semantic information (Section 5.3).

SCMix significantly improves cross-image transfer, particularly for VQA, specific VQA, and captioning. Overall ASR increases by approximately 27 percent (Section 6.5.2; Table 5, CroPA with SCMix versus CroPA (Base)). CutMix provides a net increase of around 12 percent (Table 5, CroPA with CutMix versus CroPA (Base)).

We believe SCMix helps because cropping and resizing introduce spatial and scale invariance before soft blending with another image. This encourages attacks on semantically aligned features that are less sensitive to precise position or size. Such features generalize more readily across inputs.

Both augmentation methods improve cross-image transfer within CroPA’s cross-prompt framework (Table 5). We believe they introduce cross-image features that improve perturbation generalization.

Future work should go beyond augmentation. More complex, layered methods could combine augmentation with feature-alignment attacks on latent representations, the model’s internal image representations, or perturb those latent representations directly.

#### 7.3.2 Cross-Model Transferability

The D-UAP-modified CroPA framework from Section 5.2 significantly advances CroPA’s attack method. It achieves 96.85% overall targeted ASR, a 13.5% improvement over base CroPA (Section 6.5.3; Table 6, CroPA and CroPA+D-UAP rows). VQAgeneral and Captioning improve by 13% and 28%, respectively.

These improvements occur even for challenging or harmful target text such as “Bomb,” which typically triggers guardrails. They demonstrate the framework’s robustness in bypassing safety mechanisms.

Cross-model transfer also improves, especially between models with similar vision encoders, such as BLIP2 and InstructBLIP (Table 8, BLIP2 to InstructBLIP). However, dependence on a specific encoder constrains perturbation diversity and transfer to different architectures.

This limitation highlights the need for encoder-agnostic attacks: methods that do not depend on a particular vision-encoder design. Nevertheless, D-UAP is a critical step toward universal perturbations effective across prompts and images. It supports further research on VLM generalizability and robustness.

We do not include Contextual Injection Attack in cross-model comparisons. It is inherently non-transferable and did not produce comparable results.

### 7.4 Trade-offs Between Different Types of Adversarial Transferability

Our analysis reveals fundamental trade-offs among transferability dimensions. These trade-offs matter when designing robust adversarial attacks.

#### 7.4.1 Cross-Prompt vs. Cross-Model Transferability

CroPA optimizes against competing text embeddings and achieves strong cross-prompt transfer. However, this anchors perturbations to the optimized model’s architecture. In Section A.1, ASRs drop substantially when perturbations generated on one model are applied to another.

D-UAP attempts to address this by targeting generalizable vision-encoder components rather than model-specific features (Section 5.2). It improves transfer between similar architectures, such as BLIP-2 and InstructBLIP. Improvements remain modest across fundamentally different architectures, such as BLIP-2 and Flamingo (Table 8, corresponding source-target settings).

These results suggest tension between optimizing prompt diversity and model diversity simultaneously.

#### 7.4.2 Cross-Prompt vs. Cross-Image Transferability

Cross-prompt and cross-image transfer are competing objectives. CroPA’s strong cross-prompt behavior depends on image-specific feature coupling. This coupling limits cross-image generalization.

SCMix and CutMix diversify optimization through augmentation (Section 5.3). Their improvements are promising (Table 5). However, augmentation does not fully resolve the fundamental tension between these objectives.

#### 7.4.3 Role of Initialization in Transferability Balance

Diffusion-based initialization offers another perspective on these trade-offs (Section 5.1). Semantic anchoring achieves higher ASRs without sacrificing cross-prompt transferability (Table 3). This suggests that the starting point strongly affects the attainable balance among transferability types.

Unlike methods that change the optimization process itself, better initialization lets perturbations converge toward more general adversarial directions that remain effective across prompts.

Even with better initialization, the fundamental cross-prompt versus cross-image/cross-model tension remains. Optimizing a perturbation for one dimension inevitably sacrifices effectiveness in others.

#### 7.4.4 Implications for Future Research

No single attack methodology can simultaneously maximize every form of transferability. Future frameworks should allow controlled balance among competing objectives. Possible directions include multi-objective optimization, which optimizes several goals together, and architecturally aware attack designs.

Individual techniques can improve particular dimensions. A comprehensive solution would require rethinking how perturbations interact with model architectures and data representations.

Cross-prompt transferability remains underexplored. The lack of literature leaves an important gap in understanding prompt-related VLM vulnerabilities. Addressing it is essential for secure deployment.

### 7.5 Broader Impact

From an attacker’s perspective, highly cross-prompt-transferable examples pose significant threats. They can induce malicious or misleading VLM outputs even in response to harmless queries. This could support misinformation or undermine systems that use VLMs for content generation and decision-making.

From a defensive perspective, imperceptible perturbations offer a way to protect sensitive information. Embedding them in personal images could make VLMs consistently output predetermined, non-sensitive text. This could thwart unauthorized attempts to extract confidential information. The technique offers a proactive privacy and data-security measure as visual data becomes increasingly vulnerable to exploitation.

Our improved initialization and loss function increase ASR and adversarial generalizability across models and images (Tables 3, 5, 6, 7, and 8). These advances reinforce cross-prompt attack effectiveness and may support more resilient defenses.

Our reproducibility study affirms Luo et al. (2024)’s foundational work. It also highlights the need for deeper study of cross-prompt transferability. Understanding this phenomenon is crucial for robust VLMs, effective countermeasures, user-data protection, and reliable model outputs.

### 7.6 Limitations and Future Work

We reproduced CroPA’s key results and conducted extensive ablations, experiments that compare individual method changes. However, limited computational resources repeatedly hindered progress.

Training and evaluating attacks on large VLMs such as Flamingo and BLIP2 requires substantial GPU memory and processing power. This restricted the scale and depth of some experiments. Resource availability also limited extensive hyperparameter tuning and evaluation across a wider range of model architectures.

We intended to investigate an ensemble, or combined set, of these models to study perturbation generalization further. Its memory and computation requirements were too high for our infrastructure.

Future work could improve cross-model transfer through encoder-agnostic methods and improve cross-image generalization. Query-based optimization could also increase practical applicability.

### 7.7 Communication With Original Authors

We could not establish direct communication with the original authors during replication. At the time of writing, the issue raised on their GitHub repository about BLIP-2 and InstructBLIP remains unresolved.
