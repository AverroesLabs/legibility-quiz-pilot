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

Large Vision-Language Models (VLMs) have revolutionized computer vision by enabling tasks such as image classification, captioning, and visual question answering. However, VLMs remain highly vulnerable to adversarial attacks, particularly when both visual and textual modalities can be manipulated.

In this study, we conduct a comprehensive reproducibility study of “An Image is Worth 1000 Lies: Adversarial Transferability Across Prompts on Vision-Language Models.” We validate the Cross-Prompt Attack (CroPA) and confirm that CroPA achieves superior cross-prompt transferability—the ability of adversarial images to remain effective across different textual prompts—compared to existing baselines.

Beyond replication, we propose several key improvements: (1) a novel initialization strategy that significantly improves Attack Success Rate (ASR), the measure of attack effectiveness; (2) an investigation of cross-image transferability through learned universal perturbations, which are shared across images; and (3) a novel loss function that targets vision encoder attention mechanisms to improve generalization. Our evaluation across prominent VLMs—including Flamingo, BLIP-2, and InstructBLIP, as well as extended experiments on LLaVA—validates the original results and demonstrates that our improvements consistently boost adversarial effectiveness.

Our work therefore reinforces the importance of studying adversarial vulnerabilities in VLMs and provides a more robust framework for generating transferable adversarial examples. These findings have significant implications for understanding VLM security in real-world applications. Our code is available at https://github.com/Swadesh06/Revisting_CroPA

Reviewed on OpenReview: https://openreview.net/forum?id=5L90cl0xtf

## 1 Introduction

The advent of large Vision-Language Models (VLMs) has significantly transformed computer vision by enabling a wide range of tasks, including image classification, captioning, and visual question answering. This versatility has fostered deeper exploration into visual-linguistic interactions. However, recent studies have demonstrated that VLMs remain highly vulnerable to adversarial attacks (Zhao et al. (2023); Qi et al. (2023); Zhang et al. (2022a); Carlini et al. (2024)). These attacks involve subtle perturbations to input images that lead VLMs to produce incorrect or even harmful outputs. Furthermore, because VLMs also accept textual inputs, the textual modality introduces additional attack vectors beyond those faced by traditional vision models.

Several studies have investigated VLM adversarial robustness. For example, Zhao et al. (2023) analyzed VLMs such as BLIP and BLIP-2 using query-based and transfer-based adversarial attacks in black-box settings, where the attacker does not have access to model parameters. In contrast, Schlarmann & Hein (2023) examined targeted and untargeted adversarial attacks in white-box settings, where the attacker has access to the model. While these works primarily focused on adversarial image attacks, subsequent research also explored adversarial perturbations in textual inputs. Qi et al. (2023) demonstrated that adversarial images could manipulate VLMs into executing harmful instructions, while Tu et al. (2023) systematically evaluated both visual and textual adversarial attacks.

Traditionally, the generalization and transferability of adversarial examples in VLMs have been classified into two primary categories:

- **Cross-Model Transferability:** The ability of adversarial examples to maintain their adversarial nature across different VLM architectures. This ability is commonly referred to as transferability.
- **Cross-Image Transferability:** The ability of adversarial perturbations to generate adversarial examples that generalize across multiple images. Such perturbations are often known as Universal Adversarial Perturbations (UAPs).

Luo et al. (2024) introduced a third concept, **Cross-Prompt Transferability**: the ability of adversarial images to remain effective across varying textual prompts. Unlike prior work that treated visual and textual adversarial perturbations independently, Luo et al. (2024) proposed the Cross-Prompt Attack (CroPA), which employs learnable prompts to ensure that adversarial images retain their effectiveness regardless of textual input. Luo et al. (2024) demonstrated CroPA’s efficacy across image classification, captioning, and visual question answering.

Our work addresses the following goals:

- **[Reproducibility Study] Reproducing the Results from the Original Paper:** We successfully reproduced and verified the original paper’s four main claims. First, CroPA achieves superior cross-prompt transferability compared to established baselines across various target texts, independently of the target text’s semantic meaning or word frequency; the corresponding task-level results appear in Table 1. Second, increasing the number of prompts improves transferability, but convergence occurs rapidly, and baseline methods fail to surpass CroPA even with additional prompts, as shown in Figure 4. Third, CroPA remains effective when additional in-context learning examples are provided, as shown in Table 2. Lastly, we reproduce the original findings on convergence by evaluating overall ASR across increasing attack iterations, while adding an analysis of individual-task ASR in Figures 5 and 6. ASR measures whether an attack meets its success criterion; Section 4.1 defines the targeted and non-targeted criteria.
- **[Extended Work] Better Initialization Strategy:** We propose a new initialization method that substantially increases both ASR and transferability, with targeted ASRs reported in Tables 3 and 4.
- **[Extended Work] Investigating Cross Image Transferability and Image Augmentation:** We explore learning a common perturbation for all images using CroPA’s backbone. We also investigate whether transferability-promoting enhancements help learn such a perturbation; Table 5 reports the results.
- **[Extended Work] Guiding perturbations via Target Value Vectors:** We propose a novel loss function based on the role of specific components within the vision encoder’s attention mechanism. Attention weights determine the level of interaction between image patches, while value vectors contain the information within patches. Our proposal targets the value vectors to obtain greater generalization and ASR, as evaluated in Tables 6–8.

Furthermore, we conduct in-depth analyses of the mechanisms behind these improvements to offer insights into adversarial vulnerabilities in VLMs. Our work therefore both reinforces the importance of studying these vulnerabilities and provides a more robust and versatile framework for generating transferable adversarial examples.

## 2 Related Works

Adversarial attacks on deep neural networks have been extensively studied since their introduction by Szegedy et al. (2013) and Goodfellow et al. (2014). However, applying these attacks to Vision-Language Models (VLMs) presents distinct challenges and opportunities. This section reviews the progression of adversarial transferability research and positions CroPA within that broader context.

### 2.1 Adversarial Transferability

Adversarial transferability has manifested across multiple dimensions in machine learning systems. Liu et al. (2017) and Tramèr et al. (2017) established that adversarial examples crafted for one model can often deceive other models, including models with different architectures. Beyond cross-model transferability, researchers found that perturbations can generalize across images, enabling Universal Adversarial Perturbations (UAPs) (Moosavi-Dezfooli et al. (2017); Mopuri et al. (2017)).

Transferability also extends across tasks: adversarial examples designed for classification can affect object detection and other visual tasks (Naseer et al. (2019b); Naseer et al. (2019a); Lu et al. (2020); Salzmann et al. (2021)). Because adversarial examples can transfer across models, images, and tasks, this multi-dimensional transferability raises critical questions about modern machine learning systems’ robustness.

### 2.2 Adversarial Attacks on Vision-Language Models

Early research on vision-language adversarial robustness focused on task-specific attacks. For image captioning, Xu et al. (2019), Zhang et al. (2020), Aafaq et al. (2021), and Chen et al. (2017) demonstrated methods for manipulating model outputs. Similarly, Visual Question Answering (VQA) systems proved vulnerable to targeted attacks that mislead attention mechanisms (Xu et al. (2018); Kaushik et al. (2021); Kovatchev et al. (2022); Li et al. (2021); Sheng et al. (2021); Zhang et al. (2022b)).

However, these approaches primarily targeted lightweight architectures combining convolutional neural networks and recurrent neural networks (CNN-RNN architectures). Because these architectures lacked in-context learning capabilities, the attacks had limited applicability to contemporary VLMs. More recently, Zhao et al. (2023) explored the adversarial robustness of modern large VLMs, including BLIP and BLIP-2, in black-box settings. That work focused on cross-model transferability rather than CroPA’s cross-prompt perspective.

### 2.3 Cross-Prompt Adversarial Transferability

Large VLMs use prompts to adapt to tasks, and this prompt-driven adaptation introduces another dimension of adversarial vulnerability. Unlike traditional vision models that perform specific tasks, VLMs can dynamically adapt to different tasks based on textual prompts (Li et al. (2023b); Awadalla et al. (2023); Li et al. (2023b); Zhu et al. (2023); Liu et al. (2023)). Consequently, an adversarial example intended to remain effective across those tasks must also remain effective across varying prompts.

CroPA pioneered this concept by introducing learnable prompts that compete with image perturbations during optimization (Luo et al. (2024)). This competition differs from existing methods that treat visual and textual perturbations independently, and therefore establishes a new paradigm for evaluating VLM robustness. Because cross-prompt transferability is relatively novel, no credible benchmarks or baselines have yet been established.

However, Yang et al. (2024) also explore cross-prompt transferability through the Context Injection Attack (CIA). CIA employs gradient-based perturbations to inject target tokens into both visual and textual contexts. We include CIA as an additional comparison in our experiments.

### 2.4 Enhancing Adversarial Transferability

Various techniques aim to improve adversarial transferability. Input diversity methods (Xie et al. (2019)) and data augmentation strategies (Wang et al. (2021)) have shown promise in creating more robust attacks. Recent work has also identified gradient misalignment as a key factor limiting transferability, motivating new optimization strategies (Li et al. (2023a); Wang et al. (2023)). Our work builds on these insights to enhance CroPA across multiple transferability dimensions.

## 3 Scope of reproducibility

This study examines and validates the results demonstrated by Luo et al. (2024). Our primary objective is to confirm that CroPA significantly enhances adversarial-example transferability across prompts by meticulously reproducing the original experimental procedures.

We aim to verify the following main claims:

- **Claim 1:** CroPA achieves cross-prompt transferability across various target texts.
- **Claim 2:** CroPA achieves the best overall performance across different number of prompts.
- **Claim 3:** CroPA converges towards higher ASR as number of iterations are increased.
- **Claim 4:** CroPA outperforms baselines under few-shot settings.

Beyond replication, we investigate CroPA’s efficacy in cross-model and cross-image contexts and compare CroPA with other cross-prompt transferability methods. Specifically, we assess whether adversarial images generated through CroPA can consistently deceive diverse Vision-Language Models (VLMs), regardless of the input prompt or specific model parameters.

Addressing these questions allows us both to faithfully reproduce the original experiments and to investigate CroPA’s broader applicability in enhancing the robustness and versatility of adversarial attacks on VLMs.

## 4 Methodology

### 4.1 Problem Formulation

Neural networks’ vulnerability to adversarial attacks has been well documented since Goodfellow et al. (2014). Building on that foundation, we examine Luo et al.’s (2024) formulation of cross-prompt attacks on Vision-Language Models (VLMs). The formulation asks how adversarial perturbations can remain effective across varying textual inputs.

Let \(f\) denote a VLM that processes an image input \(x_v\) and a textual input \(x_t\). Let \(\delta_v\) denote the adversarial image perturbation. To ensure real-world applicability, the authors constrain the perturbation to human-imperceptible bounds:

\[
\|\delta_v\|_p\leq\epsilon_v,
\]

where \(\|\cdot\|_p\) denotes the \(p\)-norm and \(\epsilon_v\) is the permitted perturbation bound. This constraint follows established adversarial-machine-learning practices while adapting those practices to VLMs’ multi-modal setting (Carlini et al. (2024)).

We reproduce two attack scenarios. In both scenarios, \(k\) is the number of prompt instances, \(i\) indexes those instances, \(x_t^i\) is the \(i\)-th prompt, and \(L\) is the language modeling loss.

1. **Targeted attacks:** The objective is to make the VLM generate a predetermined target text \(T\), regardless of the input prompt. Therefore, Equation 1 minimizes the language modeling loss relative to \(T\) across the \(k\) prompts:

   \[
   \min_{\delta_v}\sum_{i=1}^{k}L(f(x_v+\delta_v,x_t^i),T).
   \tag{1}
   \]

2. **Non-targeted attacks:** In contrast, the objective is to maximize the discrepancy between the model’s outputs for clean and adversarial inputs. Equation 2 therefore compares the perturbed-image output with the clean-image output under the same prompt:

   \[
   \max_{\delta_v}\sum_{i=1}^{k}L(f(x_v+\delta_v,x_t^i),f(x_v,x_t^i)).
   \tag{2}
   \]

We quantify effectiveness through Attack Success Rate (ASR). A targeted attack succeeds only if the model generates the exact target text. In contrast, a non-targeted attack succeeds if the model produces any output different from the clean image’s prediction.

### 4.2 Baseline Approaches

Because cross-prompt transferability is a novel direction in adversarial machine learning, the current literature provides no established baselines. We therefore evaluate CroPA against two baselines introduced by Luo et al. (2024): Single-P and Multi-P.

**Single-P** optimizes an image perturbation using one prompt. For a targeted attack, Single-P minimizes the language modeling loss relative to target text \(T\):

\[
\min_{\delta_v}L(f(x_v+\delta_v,x_t),T).
\tag{3}
\]

Figure 1 introduces both the image perturbation \(\delta_v\) and the prompt perturbation \(\delta_t\). CroPA uses these two perturbations with opposing objectives, as detailed in Section 4.3.

**Figure 1:** Framework overview of CroPA as presented in Luo et al. (2024). The architecture employs learnable perturbations for both image (\(\delta_v\)) and prompt (\(\delta_t\)) inputs. These perturbations operate antagonistically, while \(\delta_v\) is optimized to minimize the language modeling loss, \(\delta_t\) works to maximize it. The model’s forward pass (solid arrows) processes the perturbed inputs through the vision encoder and language model, with backpropagation (dashed arrows) updating the perturbations at configurable frequencies. The model parameters (marked with *) remain fixed during the attack. \(\oplus\) denotes element-wise addition.

**Multi-P** extends Single-P by optimizing with multiple prompts. Let \(X_t=\{x_t^1,x_t^2,\ldots,x_t^k\}\) denote the collection of \(k\) textual prompts. The targeted objective is then:

\[
\min_{\delta_v}\sum_{i=1}^{k}L(f(x_v+\delta_v,x_t^i),T).
\tag{4}
\]

For non-targeted attacks, the Single-P and Multi-P objectives are inverted to maximize the discrepancy between the model’s clean-input and perturbed-input outputs. These two approaches provide reference points for evaluating CroPA’s enhanced cross-prompt transferability.

### 4.3 Cross Prompt Attack

The Cross-Prompt Attack (CroPA), introduced by Luo et al. (2024), uses learnable prompts during optimization to enhance cross-prompt transferability. The key distinction is that prompt perturbations compete with image perturbations rather than collaborating with image perturbations to deceive the model.

Specifically, CroPA optimizes the visual perturbation \(\delta_v\) to minimize the language modeling loss for generating target text \(T\). In contrast, CroPA optimizes the prompt perturbation \(\delta_t\) to maximize that loss. Because the prompt perturbation makes the targeted objective harder for the image perturbation, this adversarial relationship forces the image perturbation to develop stronger transferability across prompts.

The resulting min-max objective is:

\[
\min_{\delta_v}\max_{\delta_t}L(f(x_v+\delta_v,x_t+\delta_t),T),
\tag{5}
\]

where \(f\) is the VLM, \(T\) is the target text, and \(L\) is the language modeling loss.

The implementation uses an iterative Projected Gradient Descent (PGD) approach (Madry et al. (2017)). Image updates use gradient descent to minimize the loss, whereas prompt updates use gradient ascent to maximize the loss. The update-interval parameter \(N\) controls the relative frequency: the image perturbation is updated \(N\) times for each prompt-perturbation update. Figure 1 depicts this framework.

Algorithm 1 reproduces the algorithm described by Luo et al. (2024). In the algorithm, \(x'_v\) denotes the adversarial image, \(x_t^{\prime i}\) denotes an initialized adversarial prompt, \(\epsilon\) is the perturbation size, \(\alpha_1\) and \(\alpha_2\) are the image and prompt step sizes, and \(K\) is the number of iteration steps. The gradients are \(g_v\) and \(g_t\); \(\nabla\) denotes differentiation, \(\operatorname{sign}\) applies the sign operation, \(\operatorname{mod}\) gives the remainder, and \(\operatorname{Clip}_{x_v,\epsilon}\) projects the image into the \(\epsilon\)-ball around \(x_v\).

**Algorithm 1 CroPA: Cross Prompt Attack**

**Input (0):** Model \(f\), Target Text \(T\), vision input \(x_v\), prompt set \(X_t\), perturbation size \(\epsilon\), step sizes \(\alpha_1\) and \(\alpha_2\), number of iteration steps \(K\), adversarial prompt update interval \(N\)

**Output (0):** Adversarial example \(x'_v\)

1. Initialize \(x'_v=x_v\).
2. **for** step \(=1\) to \(K\) **do**
3.   Uniformally sample the prompt \(x_t^i\) from \(X_t\).
4.   **if** \(x_t^i\) is not initialised **then**
5.    Initialise \(x_t^{\prime i}=x_t^i\).
6.   **end if**
7.   Compute gradient for adversarial image:
   \[
   g_v=\nabla_{x_v}L(f(x'_v,x_t^i),T).
   \]
8.   Update with gradient descent:
   \[
   x'_v=x'_v-\alpha_1\cdot\operatorname{sign}(g_v).
   \]
9.   **if** \(\operatorname{mod}(\text{step},N)=0\) **then**
10.    Compute gradient for adversarial prompt:
    \[
    g_t=\nabla_{x_t}L(f(x'_v,x_t^i),T).
    \]
11.    Update with gradient ascent:
    \[
    x_t^{\prime i}=x_t^{\prime i}+\alpha_2\cdot\operatorname{sign}(g_t).
    \]
12.   **end if**
13.   Project \(x'_v\) to be within the \(\epsilon\)-ball of \(x_v\):
    \[
    x'_v=\operatorname{Clip}_{x_v,\epsilon}(x'_v).
    \]
14. **end for**
15. **return** \(x'_v\).

During evaluation, CroPA applies only the optimized image perturbation and discards the prompt perturbations. Consequently, evaluation measures the image perturbation’s inherent transferability rather than the effect of modifying the evaluation prompts.

## 5 Additional Investigative Experiments

CroPA’s iterative PGD optimization increases attack success effectively on a single model. However, because the optimization minimizes loss using feedback from that single model, the optimization limits transferability. Other gradient-based perturbation techniques share this limitation, and several works have proposed enhancements to address the limitation (Qin et al. (2024); Li et al. (2024); Wang et al. (2023); Li et al. (2023a)). The original authors also recognize this limitation, which Table 9 in Appendix A.1 verifies. Furthermore, Table 1 shows particularly poor CroPA performance on tasks such as Classification and Captioning: for example, the CroPA row for “not sure” reports 0.2420 for Classification and 0.0010 for Captioning.

CroPA is also highly computationally intensive. Training adversarial perturbations for just 50 images takes over 6 hours on a 48GB L40S graphics processing unit (GPU), while utilizing 95% of the GPU’s video random-access memory (VRAM).

These limitations motivate our extended experiments. We build on the CroPA framework to target (i) inconsistent performance across tasks and (ii) the transferability concepts introduced in Section 1. Our enhancements aim to move the original framework toward better transferability of image perturbations.

Although transferability is not a core proposal of Luo et al. (2024), investigating transferability may also help reduce computation costs. In an ideal scenario, cross-image transferable perturbations would avoid the need to train a separate perturbation for each image, significantly increasing efficiency. Cross-model transferable perturbations could potentially yield even greater efficiency.

The following subsections explain the motivations and mechanisms of the individual enhancements.

### 5.1 Noise Initialization via Vision Encoding Optimization

A core limitation of CroPA is its random noise initialization strategy, which many gradient-based adversarial attack methods share. Although CroPA iteratively optimizes the noise, the random starting point is uninformed by the target semantics. This lack of information leads to instability and slower convergence, and is especially problematic for VLMs because semantic alignment with prompts is crucial.

Our evaluation of CroPA’s bi-level optimization framework in Section 6.5.1 empirically verifies this sensitivity to initialization; Figure 7 and Table 3 provide the corresponding evidence. The sensitivity stems from CroPA’s inability to guide the optimization trajectory toward semantically meaningful directions from the outset. Thus, conventional initialization can produce suboptimal ASRs and reduced cross-prompt transferability because the initial perturbation need not point in a meaningful adversarial direction.

We therefore hypothesize that semantically informed initialization, rather than purely random initialization, can significantly enhance attack efficacy while maintaining perturbation imperceptibility. Specifically, we propose synthesizing a target image corresponding to a desired prompt through a diffusion-based approach. We then use the vision encoder’s feature space to align the perturbation with that target image, so that optimization begins in a semantically meaningful direction.

Our **Noise Initialization via Vision Encoding Optimization** method combines semantic anchoring with adversarial optimization through three steps:

- Generate a semantically relevant target image using a state-of-the-art diffusion model, such as Stable Diffusion XL.
- Align the perturbation with the target image’s vision encoder representation, so that the initial noise conforms to the adversarial objective.
- Optimize perturbation updates using a Mean Squared Error (MSE)-based loss that enforces semantic similarity while maintaining adversarial effectiveness.

By replacing random initialization with vision-guided initialization, we improve adversarial robustness and attack efficiency, including reducing the optimization steps required for convergence. The results in Section 6.5.1 and Figure 7 support these improvements. The method therefore provides a principled alternative to conventional initialization, producing stronger perturbations and improved cross-prompt transferability by making perturbations semantically informed from the start.

#### 5.1.1 Noise Initialization

Preliminary experiments on BLIP-2 and InstructBLIP suggest that semantically informed noise significantly improves attack efficacy over traditional random initialization. The added computation time is only 20-25 seconds per image on a single GPU. In addition to improving alignment in the vision encoder’s feature space, the diffusion-based approach preserves adversarial examples’ visual fidelity while conforming to strict perturbation budgets.

#### 5.1.2 Diffusion-Based Target Synthesis

Given a target prompt \(T\), we first generate a target image \(\mathbf{x}_{\text{target}}\) that embodies the semantic attributes described by \(T\). We use Stable Diffusion XL (SDXL), with diffusion process \(\mathcal{D}\), Gaussian-sampled input \(\mathbf{z}\), and pretrained diffusion-model weights \(\theta_{\text{SDXL}}\):

\[
\mathbf{x}_{\text{target}}=\mathcal{D}(T,\mathbf{z};\theta_{\text{SDXL}}).
\tag{6}
\]

Because the resulting target image captures the prompt’s semantic essence, the target image provides an informative basis for initializing adversarial perturbations.

#### 5.1.3 Vision-Encoder Anchored Perturbation

Let \(f_v(\cdot)\) denote the VLM’s vision encoder, \(\mathbf{x}\) the original image, and \(\boldsymbol{\delta}\) the image perturbation. Our objective is to make the perturbed image \(\mathbf{x}+\boldsymbol{\delta}\) mimic the target image’s semantic representation in the vision encoder’s output space.

We obtain the initial perturbation \(\boldsymbol{\delta}_{\text{init}}\) by minimizing the squared \(\ell_2\) distance between the perturbed-image and target-image encoder outputs, subject to an \(\ell_\infty\) perturbation bound:

\[
\boldsymbol{\delta}_{\text{init}}
=
\underset{\|\boldsymbol{\delta}\|_\infty\leq\epsilon}{\arg\min}
\left\|f_v(\mathbf{x}+\boldsymbol{\delta})-f_v(\mathbf{x}_{\text{target}})\right\|_2^2.
\tag{7}
\]

Here, \(\epsilon\) is the maximum allowable perturbation, and the \(\ell_\infty\) constraint ensures imperceptibility. The initialization therefore places the adversarial example within a semantically meaningful neighborhood of the target prompt’s representation. Algorithm 2 in Appendix B formally depicts this method.

#### 5.1.4 Adversarial Optimization via PGD

After initialization, we refine the adversarial example using Projected Gradient Descent (PGD). Let \(\mathbf{x}_{\text{adv}}^{(t)}\) denote the adversarial image at iteration \(t\), \(\alpha\) the step size, and \(\mathcal{L}_{\text{MSE}}\) the MSE loss between vision encoder outputs. Let \(\mathcal{B}_\epsilon(\mathbf{x})\) be the admissible \(\ell_\infty\) ball around the original image, and let \(\Pi_{\mathcal{B}_\epsilon(\mathbf{x})}\) project onto that ball. The update is:

\[
\mathbf{x}_{\text{adv}}^{(t+1)}
=
\Pi_{\mathcal{B}_\epsilon(\mathbf{x})}
\Bigl[
\mathbf{x}_{\text{adv}}^{(t)}
-
\alpha\nabla_{\mathbf{x}_{\text{adv}}^{(t)}}\mathcal{L}_{\text{MSE}}
\Bigr].
\tag{8}
\]

The loss is:

\[
\mathcal{L}_{\text{MSE}}
=
\left\|f_v(\mathbf{x}_{\text{adv}})-f_v(\mathbf{x}_{\text{target}})\right\|_2^2.
\tag{9}
\]

Because the loss compares the two encoder outputs, each update incrementally aligns the adversarial example with the target image’s semantic embedding. The projection then keeps the updated adversarial image within the allowed perturbation budget.

### 5.2 Guiding perturbations via Target Value Vectors and evaluating Cross-Model Tranferability

CroPA uses PGD to optimize both image and text inputs, but CroPA lacks architectural sensitivity, particularly to the vision encoder’s internal dynamics. Because the vision encoder plays a dominant role in VLM semantic grounding, treating visual and language modalities as uniformly vulnerable fails to exploit modality-specific weaknesses. This oversight limits structured, transferable perturbations. More critically, the lack of targeted influence on the encoder’s representational backbone undermines both adversarial strength and cross-model transferability.

CroPA’s image and text perturbations respectively minimize and maximize losses to manipulate the model toward a specific output target. However, prior research shows that perturbing vision encoder output embeddings can significantly impair the visual perception of large vision-language models (LVLMs), thereby misleading their language-model responses (Cui et al. (2023); Wang et al. (2024)). Various LVLMs also typically use similar image encoders. These encoders convert images into visual tokens containing rich semantic and structural information, which language decoders use to access and reason about image content. Consequently, architectural details of the vision encoder offer potential targeted vulnerabilities that CroPA does not sufficiently leverage.

We therefore propose combining the CroPA loss with a perturbation loss targeting the vision encoder’s core attention mechanism, specifically the value vectors in attention layers. Value vectors hold the information within image patches. Our hypothesis is that perturbing value vectors using a reference image corresponding to a target text can significantly enhance Adversarial Success Rate (ASR) while retaining CroPA’s hyperparameter constraints.

Our approach draws on the Doubly-Universal Adversarial Perturbation method, abbreviated **Doubly-UAP** and referred to here as **D-UAP**. Rather than maximizing the loss between value vectors, our adaptation minimizes losses between value vectors of \(x_i\), an input image, and \(T_i\), the corresponding target reference. Doubly-UAP identifies which components of a vision encoder’s attention mechanism most effectively influence adversarial performance. We hypothesize that targeting middle-to-late vision-layer value vectors, which encode essential visual features, will disrupt interpretation more effectively than CroPA’s generic PGD-based perturbations. Furthermore, freezing the text encoder keeps the perturbations focused on degrading visual processing rather than introducing text-based artifacts.

Our modification therefore aims to:

1. Improve attack effectiveness by exploiting information in the vision encoder.
2. Enhance semantic alignment between adversarial perturbations and the target text representation.
3. Achieve better adversarial robustness across models while maintaining computational feasibility.

Jointly optimizing the CroPA loss and our modified Doubly-UAP loss produces a more structured, interpretable, and targeted adversarial attack framework.

This incorporation adapts a component of Kim et al.’s (2024) Doubly-UAP method. That method introduces a UAP by identifying the attention components that most effectively influence VLM performance. We focus on two components with fundamental roles:

1. **Attention Weights:** Attention weights control how much each patch focuses on other patches, and therefore determine interaction or relevance between patches. We hypothesize that targeting attention weights can interfere with the encoder’s ability to establish those relationships.
2. **Value Vectors:** Value vectors contain the information within each patch. We expect perturbing value vectors to disrupt that information and thus impair the model’s interpretative abilities.

Because vision encoders are crucial for visual interpretation, our attack targets the attention mechanism within the vision encoder. Based on prior analysis, we specifically aim at the most vulnerable components: value vectors in middle-to-late layers.

For the standard Doubly-UAP objective, let \(L\) denote the set of targeted layers and \(|L|\) the number of targeted layers; here, \(L\) is a layer set rather than the language modeling loss used earlier. Let \(l\) index a targeted layer, \(V_l(x)\) denote that layer’s value vectors for image \(x\), \(\delta\) denote an image perturbation, and \(\operatorname{Loss}(\cdot)\) denote the loss applied to the value vectors. The standard attack obtains \(\delta^*\) as:

\[
\delta^*
=
\arg\max_{\delta}
\frac{1}{|L|}
\sum_{l\in L}
\operatorname{Loss}(V_l(x),V_l(x+\delta)).
\tag{10}
\]

Thus, the standard objective maximizes the difference between clean-image and perturbed-image value vectors.

#### 5.2.1 Our Modified Approach

We instead introduce a target value vector derived from a reference image corresponding to desired target text \(T\). Rather than solely maximizing deviation from the original image’s value vectors, our modified loss enforces alignment with a predefined target representation.

**Figure 2:** D-UAP Architecture Diagram

The method proceeds as follows:

1. Extract value vectors from the \(l\)-th vision encoder layer for both the input image and the image associated with the target text.
2. Compute the cosine similarity loss between those value vectors.
3. Jointly minimize the negative of this loss along with the CroPA loss, using appropriate scaling to ensure adversarial robustness and target alignment.

Formally, let \(V_i(x+\delta)\) denote the perturbed value vectors of the \(i\)-th attention head for input image \(x\), and let \(V_t\) denote the target reference image’s value vectors. Let \(N\) denote the number of attention heads; in this subsection, \(N\) is not the prompt-update interval used in Algorithm 1. With the dot product denoted by \(\cdot\) and the \(\ell_2\) norm by \(\|\cdot\|_2\), our value vector loss is:

\[
L_{\text{d-UAP}}
=
\sum_{i=1}^{N}
\left(
1-
\frac{
V_i(x+\delta)\cdot V_t
}{
\|V_i(x+\delta)\|_2\|V_t\|_2
}
\right).
\tag{11}
\]

This loss encourages the perturbed image’s value vectors to align closely with the target text’s representation.

We combine this loss with \(L_{\text{CroPA}}\), the CroPA loss. Let \(\lambda\) control the value vector loss’s relative importance, and let \(L_{\text{re-CroPA}}\) denote the resulting final objective:

\[
L_{\text{re-CroPA}}
=
L_{\text{CroPA}}(x_v+\delta_v,x_t+\delta_t,T)
-
\lambda L_{\text{d-UAP}}(\delta_v).
\tag{12}
\]

Joint optimization preserves the perturbation’s adversarial nature while enforcing semantic alignment with the target text. Specifically, gradients from both loss components are combined to update the perturbation \(\delta\), so that the resulting adversarial example exhibits both cross-prompt transferability and guided semantic influence. This enhancement to Doubly-UAP enables more precise, controlled, and interpretable manipulation of VLMs, with applications in adversarial robustness and security analysis. Algorithm 3 in Appendix B summarizes the steps.

Although we propose this modification to increase the transferability of CroPA-generated perturbations, D-UAP depends on model-specific value vectors and therefore on the vision encoder. Consequently, we hypothesize that transferability may be limited to models whose vision encoder architectures are the same as, or similar to, the architecture of the model used for optimization.

### 5.3 Transferability across images via Image Augmentation

Traditional adversarial methods optimize perturbations by iterative gradient descent on individual images. This per-image optimization often overfits image-specific features, leading to poor cross-image transferability and high computational cost. CroPA has the same issue because CroPA learns perturbations for one image across multiple prompts.

Luo et al. (2024) state that CroPA is orthogonal to cross-image transferability approaches and that computing perturbations across images can enhance cross-image transferability, as in Moosavi-Dezfooli et al. (2017). We investigate this claim and introduce a novel approach based on other UAP literature.

Fang et al. (2024) introduce UAPs for Vision-Language Pretraining (VLP) models to mitigate gradient-based attacks’ overfitting while enabling cross-image transferability. Separately, CutMix provides regularization through regional replacement rather than pixel blending (Yun et al. (2019)). CutMix cuts rectangular patches from one image and pastes those patches into another image, while mixing labels in proportion to patch area. Because this process forces learning from partial views and prioritizes spatially invariant features, CutMix improves robustness against adversarial attacks and occlusions.

Building on UAP frameworks, Zhang et al. (2024) propose Effective and Transferable Universal Adversarial Attack (ETU). ETU enhances transferability across VLP models and tasks through global and local optimization techniques. In particular, ETU introduces **SCMix**, a data augmentation strategy combining self-mix and cross-mix operations to increase data diversity while preserving semantic integrity. SCMix aims to integrate spatially and scale invariant image features.

We incorporate SCMix into perturbation generation because Zhang et al.’s empirical results show that, for the same model, SCMix alone yields superior cross-image transferability without their additional local utility reinforcement.

#### 5.3.1 SCMix

SCMix diversifies training inputs through two stages: self-mixing followed by cross-mixing.

In self-mixing, \(I_1\) denotes a base image. Each \(X_i\) is a randomly cropped and resized version of that base image, and \(\eta\) is the blend weight. The self-mixed image \(I'_1\) is:

\[
I'_1=\eta X_1+(1-\eta)X_2
\quad\text{where}\quad
X_i=\operatorname{Resize}(\operatorname{RandomCrop}(I_1)),
\quad\eta=0.5.
\tag{13}
\]

The original SCMix method sampled \(\eta\) probabilistically from a Beta distribution, \(\eta\sim\operatorname{Beta}(\alpha,\alpha)\), with distribution parameter \(\alpha>0\). In contrast, we fix \(\eta=0.5\). This choice is equivalent to setting \(\alpha=\infty\), as justified in Appendix D.3.

The cross-mixing stage then incorporates a secondary image \(I_2\). With weights \(\beta_1\) and \(\beta_2\), the final mixed image \(I_3\) is:

\[
I_3=\beta_1I'_1+\beta_2I_2
\quad\text{with}\quad
\beta_1\gg\beta_2\in[0,1).
\tag{14}
\]

Here, \(\gg\) indicates that the base-image weight is much greater than the secondary-image weight. This weighting preserves \(I_1\)’s dominant visual semantics while adding diversity from \(I_2\), enabling the adversarial generator to learn perturbations effective across diverse images. Furthermore, because the base image remains dominant, the mixed image \(I_3\) remains compatible with \(T_1\), the text description of \(I_1\). Thus, the synthetic image supports enriched cross-modal interactions while remaining aligned with the original description despite incorporating another image’s features.

The two-stage mixing process reduces potential CroPA overfitting by forcing perturbations to attack features invariant to intra-image variation through self-mixing and inter-image blending through cross-mixing. This regularization promotes universal perturbations that may transfer to unseen images without separately training the attack for each image.

The proposal is compatible with CroPA’s inference output: CroPA returns only the perturbed image, or equivalently an appropriate image perturbation. Therefore, if effective cross-image perturbations can be learned, the attack can be universal in both senses—cross-prompt transferable, as CroPA claims, and cross-image transferable, as we investigate.

**Figure 3:** Illustration of the self-mixing and cross-mixing processes. The input image \(I_1\) undergoes self-mixing to generate intermediate representations \(X_1\) and \(X_2\), which are then combined to form \(I'_1\). Additionally, an external image \(I_2\) is mixed with \(I'_1\) to produce the final output \(I_3\).

#### 5.3.2 CutMix

CutMix enhances transferability through discrete regional feature replacement (Yun et al. (2019)). For images \(x_A\) and \(x_B\), let \(\tilde{x}\) denote the composite image and \(M\) a binary mask. Let \(W\) and \(H\) denote image width and height, and let \(\odot\) denote element-wise multiplication. The composite is:

\[
\tilde{x}=M\odot x_A+(1-M)\odot x_B,
\tag{15}
\]

where \(M\in\{0,1\}^{W\times H}\) defines a randomly placed rectangular region.

CutMix therefore enables learning from partial views of the mixed images. The main distinction from SCMix is that SCMix softly blends features within and across images, whereas CutMix replaces regions through a hard mask. This discrete replacement forces perturbations to target spatially invariant features that persist across image combinations. When integrated with adversarial training, CutMix therefore encourages perturbations that exploit higher-level semantic patterns rather than localized artifacts, potentially improving cross-image transferability.

### 5.4 Attack Setup and Threat Models for Transferability Experiments

We establish threat models for cross-model and cross-image transferability. Each threat model specifies the attacker’s goal, capabilities, knowledge, and evaluation protocol.

#### 5.4.1 Using D-UAP to explore Cross-Model Transferability for CroPA

We use the following threat model to evaluate how well perturbations generated with the D-UAP enhancement transfer between models.

**Adversary’s Goal:** The attacker aims to generate one perturbation for an image that makes multiple VLMs output a specific target text regardless of the input prompt. This represents black-box transfer: the attacker optimizes on an accessible model and deploys the perturbation against unknown models.

**Knowledge and Capabilities:** The attacker has white-box access to a surrogate model—either BLIP-2 or Flamingo in our experiments—including gradients and internal representations. For target models, the attacker knows only the general architecture family, such as CLIP-based, and cannot access parameters. The maximum perturbation is \(\epsilon=16/255\) under the \(L_\infty\) norm to ensure visual imperceptibility.

**Cross-Model Settings:** We examine two scenarios:

- **Intra-family transfer:** Perturbations optimized on BLIP-2, which uses the OPT-2.7b language model, are evaluated on InstructBLIP, which uses Vicuna-7b. The models share similar CLIP ViT-L/14 vision encoders but differ in their language models and fine-tuning objectives.
- **Cross-architecture transfer:** Perturbations are tested from BLIP-2 to Flamingo and from Flamingo to BLIP-2. These models differ fundamentally in vision encoders, language models, and architectural designs.

**Evaluation Setup:** We measure ASR independently for general VQA, specific VQA, Classification, and Captioning. The labels **VQAgeneral** and **VQAspecific** distinguish the specificity of prompts pertaining to an image. Success is measured by the target model’s failure to generate the correct output for the perturbed image, regardless of the input prompt.

#### 5.4.2 Investigating Cross-Image Transferability via Augmentation Techniques

For universal perturbations intended to work across images, we use the following threat model.

**Adversary’s Goal:** The attacker aims to generate a single perturbation pattern that causes a VLM to output a predetermined target text when applied to any image, regardless of image content or prompt. Compared with per-image optimization, this goal represents a higher level of attack efficiency.

**Knowledge and Capabilities:** The attacker has a limited set of training images and their SCMix-augmented variants. The attacker has white-box access to the target model during optimization, but the learned perturbation must generalize to unseen images. The perturbation budget is \(\epsilon=16/255\) under the \(L_\infty\) norm.

**Evaluation Setup:** We evaluate universal perturbations on a separate set of images not seen during training, measuring ASR across task types. An attack succeeds when the model fails to generate the correct output text for a previously unseen image, regardless of image content or the associated prompt.

The SCMix enhancement specifically targets gradient-based methods’ tendency to overfit image-specific features. Because self-mixing and cross-mixing introduce controlled variations during training, we hypothesize that the resulting perturbations will capture vulnerabilities that are less dependent on particular image characteristics.

#### 5.4.3 Evaluation Rationale for Transferability experiments

In our experiments, we present all cross-prompt and cross-model transferability results under the untargeted attack setting. Two factors motivate this choice.

First, different VLMs incorporate visual information through distinct embedding mechanisms. Consequently, the effect of an input perturbation on final output-token embeddings is difficult to identify. Targeted transferability is notably higher between models with similar vision encoders, such as BLIP-2 and InstructBLIP, but varies drastically otherwise. Therefore, directly comparing targeted ASRs across heterogeneous architectures is imprudent.

Second, visual-textual fusion substantially affects adversarial control. Decoder-only VLMs such as LLaVA and OpenFlamingo incorporate visual features late through cross-attention or token concatenation. These models treat visual features as auxiliary memory during causal language generation, so visual influence fades as decoding progresses (Awadalla et al. (2023); Liu et al. (2023)).

In contrast, fusion-based models such as BLIP-2 (Li et al. (2023b)) and InstructBLIP (Dai et al. (2023)) integrate visual features early and deeply into the input embeddings. This integration ensures persistent visual grounding during generation. Thus, targeted attacks are inherently more feasible for fusion-based models than for decoder-only models (Dai et al. (2023)).

Because of these architectural disparities, non-targeted ASR provides a more robust, fair, and comparable assessment of transferability between models and between images.

We also exclude CIA from transferability comparisons because experiments yielded negligible transferability. We identify two fundamental limitations in CIA’s method:

- **Image-Specific Semantic Embedding:** CIA manipulates per-image visual tokens to embed target concepts. The resulting adversarial patterns depend on the image and lack cross-image consistency because gradient directions and token spaces differ greatly between images and concepts. The token-space transformations between input-image and target-image pairs are therefore non-transferable.
- **Model-Specific Embedding Architectures:** CIA depends heavily on the structure of visual and textual encoders. Structural variation across LVLMs, including BLIP2, LLaVA, and Flamingo, therefore prevents CIA from demonstrating cross-model transferability.

## 6 Results

As stated in Section 4, a major objective is to reproduce Luo et al.’s (2024) cross-prompt transferability results for CroPA. This section describes the reproduction and compares the reproduced findings with the original paper.

**Models Used:**

In reproducing Luo et al. (2024), we evaluated five state-of-the-art Vision-Language Models (VLMs): OpenFlamingo, BLIP-2, InstructBLIP and LLaVA. For Flamingo, we used the publicly accessible OpenFlamingo-9B implementation (Awadalla et al. (2023)), which provides performance comparable to the original model.

BLIP-2 uses a two-stage fusion-based approach. A frozen CLIP image encoder first extracts visual features, and a Querying Transformer then processes those features (Li et al. (2023b)). This architecture enables efficient adaptation to diverse vision-language tasks. BLIP-2 uses OPT-2.7b as its language model, supporting flexible text generation.

InstructBLIP builds on BLIP-2’s fusion-based architecture and adds instruction tuning (Dai et al. (2023)). InstructBLIP differs notably in using the Vicuna-7b language model, which improves task-specific instruction following. Consequently, carefully crafted prompts enable more precise control over outputs.

LLaVA uses a decoder-only generation approach (Liu et al. (2023)). A pretrained CLIP ViT-L vision encoder connects to Vicuna-13B through a lightweight projection layer. Unlike fusion-based models, LLaVA passes visual features directly into the language model’s context as special tokens in the prompt sequence.

These models therefore represent two architectural approaches: decoder-only models, OpenFlamingo and LLaVA, use visual embeddings as generation context; fusion-based models, BLIP-2 and InstructBLIP, explicitly align visual and textual representations before generation. These differences significantly affect targeted-attack effectiveness across prompts, as discussed in Section 5.4.3.

We based the reproduction on the authors’ public code and retained the original model configurations to ensure faithful baseline comparisons. Although the implementation was generally well documented, successful reproduction required modifications. In particular, the BLIP-2 and InstructBLIP code triggered multiple runtime errors, requiring significant debugging and adjustments before execution was functional.

### 6.1 Claim 1: CroPA achieves cross-prompt transferability across various target texts. [Reproduced]

To validate Luo et al.’s (2024) central cross-prompt transferability claim, we replicated their experimental setup on the Flamingo VLM. We generated adversarial examples with CroPA and evaluated targeted ASR across general VQA, specific VQA, image classification, and image captioning.

#### 6.1.1 Experimental Details

Following Luo et al. (2024), we evaluated adversarial examples with prompts distinct from those used in training. Table 1 lists the target prompts used in the evaluation. These prompts span broad semantic meanings to support a rigorous cross-prompt transferability evaluation. Appendix D.1 reports additional details for all reproduced experiments.

#### 6.1.2 Comparative Analysis

Table 1 reports Flamingo targeted ASRs for each target text and method. Overall, the results support CroPA’s cross-prompt transferability: CroPA produces non-zero ASRs across every evaluated task and target text. Thus, the CroPA-generated examples mislead the VLM even under diverse prompts.

However, Classification and Captioning consistently underperform relative to the VQA tasks. This weakness is especially evident for abstract target texts without significant visual semantics. For “not sure,” the CroPA row reports Classification ASR of 0.2420 and Captioning ASR of 0.0010, compared with 0.8760 for VQAgeneral and 0.8940 for VQAspecific. For “too late,” the corresponding CroPA ASRs are 0.7010 and 0.1790 for Classification and Captioning, compared with 0.9300 and 0.9580 for the VQA tasks.

| Target Prompt | Method | VQAgeneral | VQAspecific | Classification | Captioning | Overall |
|---|---|---|---|---|---|---|
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

**Table 1:** Targeted ASRs tested on Flamingo with different target texts, across a variety of Vision-Language tasks. Here VQAgeneral and VQAspecific, refer to the degree of specificity of prompts pertaining to a particular image. The prompts used are reported in Appendix E

These experiments reproduce the core finding that CroPA exhibits robust cross-prompt transferability. For example, Table 1’s CroPA overall ASRs exceed the Single-P and Multi-P overall ASRs for every listed target text. The results therefore validate VLMs’ adversarial vulnerability to cross-prompt attacks.

### 6.2 Claim 2: CroPA achieves the best overall performance across different number of prompts [Reproduced]

Figure 4 compares baseline and CroPA performance as the number of prompts changes. ASR increases with the number of prompts for all methods. Single-P is an exception in the sense that Single-P inherently gets tested on a single prompt, so Figure 4 represents Single-P with a lone “star” rather than a multi-prompt curve.

For the same number of prompts, CroPA achieves higher ASR than the baselines and is consistently the better performer in Figure 4. This result supports the motivation for CroPA’s cross-prompt transferability. We use “unknown” as the target text to avoid high-frequency responses commonly found in vision-language tasks; we find Luo et al.’s (2024) argument for this choice valid.

Furthermore, Figure 4 shows that the baseline saturates at an ASR lower than CroPA’s as prompts are added. Thus, increasing the baseline’s prompt count does not allow the baseline to surpass CroPA, consistent with Luo et al. (2024).

**Figure 4:** Targeted ASR tested with different number of prompts on Flamingo. With the same number of attack iterations, our CroPA significantly outperforms baselines. "unknown" is used as the target text.

### 6.3 Claim 3: CroPA converges towards higher ASR as number of iterations increase [Reproduced]

Our convergence investigation produced results that contrast with some of the original reported convergence metrics, which are presented in Appendix A.2. This contrast occurred despite using the same training and evaluation parameters and the authors’ own code. The original metrics showed CroPA converging with a higher number of prompts.

The overall ASR across tasks follows the convergence trend shown in the original paper: our Figure 6 imitates the original reported figures. However, when we examine each task separately, the ASR convergence trend is prone to instability, as Figure 5 shows. Thus, overall convergence does not imply stable convergence for every individual task.

The original authors did not specify the model used for this evaluation. We therefore report results for OpenFlamingo with “unknown” as the target text.

(a) VQA

(b) VQA-Specific

(c) Classification

(d) Captioning

**Figure 5:** Comparison of Single-P, Multi-P, and CroPA across different categories.

**Figure 6:** Overall Convergence and Comparison of CroPA

### 6.4 Claim 4: CroPA outperforms baselines under few-shot settings [Reproduced]

Flamingo accepts extra images as in-context learning examples in addition to the textual prompt. These examples improve task adaptation. To investigate their effect on cross-prompt attacks, Luo et al. (2024) evaluated adversarial images with a different number of in-context examples from the number used during optimization.

Specifically, optimization uses the **0-shot setting**, meaning that no extra in-context images are provided. Evaluation then uses the **2-shot setting**, meaning that two extra images are provided. Table 2 compares these settings.

| Setting | Method | VQAgeneral | VQAspecific | Classification | Captioning | Overall |
|---|---|---|---|---|---|---|
| shot=0 | Multi-P | 0.7240 | 0.8740 | 0.5550 | 0.2850 | 0.6095 |
| shot=0 | CroPA | 0.9680 | 0.9880 | 0.7070 | 0.4200 | 0.7708 |
| shot=2 | Multi-P | 0.7860 | 0.9420 | 0.5630 | 0.0010 | 0.5730 |
| shot=2 | CroPA | 0.9440 | 0.9940 | 0.6860 | 0.0300 | 0.6635 |

**Table 2:** Targeted ASRs with and without visual in-context learning. The shot indicates the number of images added for in-context learning.

In-context examples make attack success inherently harder because the examples can shift generation conditions away from the optimization conditions (Luo et al. (2024)). Adding in-context images also makes the model more contextually aware of the correct prompt space, which diverges from the target-text space. Zhao et al. (2023) and Qi et al. (2023) verify that attacks such as CroPA, when optimized for a specific context, lose effectiveness when that context changes. In Table 2, the overall ASR falls from 0.7708 to 0.6635 for CroPA and from 0.6095 to 0.5730 for Multi-P when evaluation changes from shot=0 to shot=2.

However, CroPA performs significantly better than the baseline in most tasks. Under shot=2, for example, CroPA’s overall ASR is 0.6635 compared with Multi-P’s 0.5730, and the CroPA row exceeds the Multi-P row in all four task columns. Thus, CroPA is more robust to the widened context space introduced by in-context examples. This finding verifies the motivation for Section 4.3’s min-max objective, which was designed to improve robustness to a wider prompt space and consequent context changes.

### 6.5 Results Beyond the Original Paper

The following subsections report extended experiments that build on CroPA and incorporate enhancements intended to mitigate the corresponding CroPA limitations.

#### 6.5.1 Noise Initialization

Noise Initialization via Vision Encoding Optimization, described in Section 5.1, significantly improves performance across tasks compared with baseline methods. Semantically informed initialization yields superior adversarial robustness and cross-prompt transferability. As shown in Table 3, our integrated method, **CroPA+Init**, consistently outperforms both CroPA and Multi-P across all metrics, achieving 6-15% gains in overall accuracy. The Overall column reports, for example, 0.8977 for CroPA+Init versus 0.7708 for CroPA and 0.6095 for Multi-P on “unknown,” and 0.9458 versus 0.7985 and 0.7318 on “I am sorry.”

These results validate semantically guided initialization over random-noise approaches. The vision-encoder-anchored perturbations increase ASR and improve efficiency by reducing optimization steps, with the convergence comparison shown in Figure 7(a). Thus, semantically guided initialization provides a robust and adaptable alternative for adversarial attack initialization.

| Target Prompt | Method | VQAgeneral | VQAspecific | Classification | Captioning | Overall |
|---|---|---|---|---|---|---|
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

**Table 3:** Targeted ASRs on Blip2 with different target texts using CroPA Vision Encoder noise initialization. The detailed hyperparameters used are discussed under section D.2.

##### Theoretical Analysis of Initialization in Bi-level Optimization

We attribute the significant gains from diffusion-guided initialization to three challenges in CroPA’s min-max optimization.

First, CroPA’s adversarial formulation resembles a saddle-point problem, in which initialization heavily influences optimization stability. Zhang et al. (2025) demonstrated that min-max problems are significantly more sensitive to initialization than standard minimization problems and often require carefully designed initializations for stable convergence. This sensitivity is especially pronounced in CroPA because prompt perturbations continuously change the optimization landscape.

Second, CroPA’s alternating update schedule updates image perturbations more frequently than prompt perturbations. The resulting optimization asymmetry further increases initialization’s importance. Chen et al. (2019) found that alternating gradient-based methods in min-max problems frequently exhibit local cycling, with convergence strongly dependent on the initial point’s proximity to adversarially robust regions. Our semantic initialization uses this insight to place the starting point closer to meaningful adversarial directions.

Third, our empirical findings align with Wu et al. (2020), who showed that perturbations initialized with semantic priors consistently transfer better across models than randomly initialized perturbations and substantially improve ASR. This effect is especially pronounced in bi-level optimization settings such as CroPA, where optimization pathways are more complex.

Our diffusion-based method addresses these challenges by initializing perturbations toward semantic targets rather than with standard random noise. Consequently, the method enables faster convergence and enhances the resulting adversarial examples’ transferability. Table 3’s Overall column shows the associated ASR increases for all listed target texts.

Table 4 reports average targeted ASRs on BLIP-2. In addition to the previously described baselines, Table 4 includes the Contextual Injection Attack (CIA) of Yang et al. (2024).

| Method | VQAgeneral | VQAspecific | Classification | Captioning | Overall |
|---|---|---|---|---|---|
| Multi-P | 0.7053 | 0.8807 | 0.7001 | 0.5553 | 0.7104 |
| CroPA | 0.8723 | 0.9457 | 0.7518 | 0.6138 | 0.7958 |
| CIA | 0.2985 | 0.2281 | 0.4857 | 0.4687 | 0.370 |
| CroPA+Init | 0.9145 | 0.9342 | 0.8982 | 0.9464 | 0.9209 |

**Table 4:** Average Targeted ASRs on BLIP-2 for different methods. Results show the impact on different vision-language tasks: Visual Question Answering (general and specific), Classification, and Captioning. The best performance values for each task are highlighted in bold. Apart from the previously mentioned baselines, we also compare our results with the Contextual Injection Attack (CIA) Yang et al. (2024) as an additional baseline.

Gradient-based attacks often acquire their targets’ semantics, which theoretically should remove the need for semantically aligned initialization. However, the min-max complexities described above weaken gradient-produced semantic alignment. Injecting target semantics at initialization is therefore necessary to strengthen that alignment and improve convergence. Figure 7 provides experimental support for this claim.

To evaluate target-attribute incorporation, we calculate the CLIP score between the target text and noise perturbations learned by base CroPA and by CroPA+Init. Here, the CLIP score quantifies feature similarity between the learned perturbation and the target text. This comparison indicates how much of the target’s features the perturbations acquire.

CroPA exhibits the expected gradient-based behavior by acquiring the target’s context. However, Figure 7(b) shows a modest CLIP-score increase with initialization: CroPA scores 21.80, whereas CroPA+Init scores 23.09 for target text “Bomb.” This suggests that, although CroPA already acquires target semantics, initialization provides a subtle shift toward adversarially vulnerable regions, even if it does not significantly upgrade the acquired semantic features.

The convergence comparison in Figure 7(a) also supports our hypothesis: deliberately starting closer to meaningful adversarial directions produces faster convergence.

| Method | CLIP Score |
|---|---|
| CroPA | 21.80 |
| CroPA+Init | 23.09 |

Target Text: "Bomb"

(a) Overall convergence comparison

(b) CLIP-score comparison

**Figure 7:** (a) Overall Convergence Comparison of base CroPA and CroPA+Init. (b) Quantifying feature similarity of learned perturbation to target by CLIP score.

#### 6.5.2 Investigating Cross-Image transferability and Image Augmentation

CroPA’s effectiveness comes from retaining adversarial impact across textual prompts. However, because CroPA optimizes perturbations across prompts for a particular image, the resulting signals remain tightly coupled to that image’s low-level features. This dependency prevents generalization to other images and makes universal cross-image attacks ineffective, weakening the original authors’ claim that learning perturbations across images would lead to cross-image transferability.

To address this limitation, we integrated SCMix and CutMix, described in Section 5.3. Both methods mix image features during training to increase input diversity. The goal is therefore to encourage perturbations to generalize beyond one image.

We believe CutMix outperforms SCMix in promoting adversarial transferability because CutMix’s regional mixing preserves distinct, non-overlapping spatial areas. Such mixing encourages modular, independently manipulable features, which in turn makes perturbations more generalizable across inputs.

Table 5 shows that cross-image transferability remains challenging, but both SCMix and CutMix produce measurable improvements. The overall ASR is 0.4355 for CroPA (Base), 0.7055 for CroPA with SCMix, and 0.5565 for CroPA with CutMix. Despite these gains, the performance gap between cross-image methods and CroPA applied to one image demonstrates a fundamental limitation: CroPA excels at cross-prompt robustness but does not effectively support transferable adversarial attacks across images.

| Method | VQA | VQAspecific | Classification | Captioning | Overall |
|---|---|---|---|---|---|
| CroPA (Base) | 0.3500 | 0.4920 | 0.4000 | 0.5000 | 0.4355 |
| CroPA with SCMix | 0.6560 | 0.7160 | 0.5000 | 0.9500 | 0.7055 |
| CroPA with CutMix | 0.4520 | 0.7240 | 0.5000 | 0.5500 | 0.5565 |

**Table 5:** Untargeted ASR of CroPA in cross-image settings with and without SCMix and CutMix. The results represent the best ASRs achieved across all iterations. CroPA without input mixing attained its highest ASRs at 1700 iterations, while CroPA with SCMix and CutMix outperformed it significantly earlier, at just 1500 iterations and 1600 iterations, respectively. Additional details on experimental parameters are reported in Appendix D.3.

#### 6.5.3 Guiding perturbations via Target Value Vectors

Tables 6 and 7 report results for **CroPA+D-UAP**, the D-UAP-modified CroPA framework described in Section 5.2. The enhanced perturbations greatly improve on the existing algorithm across the tasks. In Table 6, the overall ASR is 0.9685, or 96.85%, compared with base CroPA’s 0.8433; this is an 12.5% increase. The enhancement particularly improves VQAgeneral and Captioning, with increases of 13% and 28%, respectively. The corresponding Table 6 values are 0.9420 versus 0.8176 for VQAgeneral and 0.9810 versus 0.6960 for Captioning.

These experiments use “Bomb” as the target text. The results therefore demonstrate improvement even for harmful text that is inherently set to trigger a model’s guardrails. Thus, for worse expected cases involving “harmful words,” the enhancement makes the attack perform significantly better in vulnerable scenarios.

| Method | VQAgeneral | VQAspecific | Classification | Captioning | Overall |
|---|---|---|---|---|---|
| Multi-P | 0.68 | 0.82 | 0.90 | 0.60 | 0.75 |
| CroPA | 0.8176 | 0.9100 | 0.9498 | 0.6960 | 0.8433 |
| CIA | 0.3431 | 0.3102 | 0.4732 | 0.5534 | 0.4199 |
| CroPA+D-UAP | 0.9420 | 0.9720 | 0.9790 | 0.9810 | 0.9685 |

**Table 6:** Targeted ASRs on Blip2 using D-UAP function and target text being "Bomb", outperforming PGD baselines and CIA, as well as the base CroPA method.

| Method | VQAgeneral | VQAspecific | Classification | Captioning | Overall |
|---|---|---|---|---|---|
| Multi-P | 0.6960 | 0.8540 | 0.8990 | 0.6060 | 0.7638 |
| CroPA | 0.7900 | 0.8860 | 0.9470 | 0.6700 | 0.8233 |
| CIA | 0.3027 | 0.4302 | 0.5112 | 0.5080 | 0.4380 |
| CroPA+D-UAP | 0.9000 | 0.9520 | 0.9500 | 0.9060 | 0.9270 |

**Table 7:** Targeted ASRs on Open Flamingo using D-UAP Loss function and target text being "Bomb".

#### 6.5.4 Analyzing Cross Model Transferability by Value Vector guidance

Table 8 shows that value-vector guidance also increases cross-model transferability compared with vanilla CroPA. Improvements are particularly significant for Classification and Captioning, at 14% and 21%, respectively, with a modest increase in VQA. We attribute the task discrepancy to VQA prompts spanning a much more complex embedding space than targeted tasks such as Classification and Captioning, where outputs occupy much smaller spans.

However, the vision encoder’s behavior also exposes a key limitation, verifying the hypothesis in Section 5.2. Value vectors depend on the specific vision encoder within the VLM. Consequently, the CroPA Doubly-UAP approach may transfer only to models with the same or similar vision encoder architecture.

Table 8 illustrates this limitation. Perturbations trained on BLIP2 are passably transferable to InstructBLIP, which uses the same vision encoder, but fail to transfer to OpenFlamingo. In the BLIP2-to-InstructBLIP rows, CroPA+D-UAP’s overall ASR is 0.6850, compared with CroPA’s 0.5065. In the BLIP2-to-Flamingo rows, the corresponding overall ASRs are 0.2335 and 0.1435. The expected pattern also appears when perturbations are trained on Flamingo and applied to the other models: ASR is insignificant, as reported in Table D.4.5 in Appendix D.4.5 and in the Flamingo-source rows of Table 8. Appendix D.4 reports experimental details.

In Table 8, **Cls** abbreviates Classification and **Cap** abbreviates Captioning. Each setting names the optimization model first and the evaluation model second.

| Settings | Method | VQAgeneral | VQAspecific | Cls | Cap | Overall |
|---|---|---|---|---|---|---|
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

**Table 8:** The cross model test under different settings. "BLIP2 to InstructBLIP" means the perturbations optimised on BLIP2 are tested on the InstructBLIP and similarly for "BLIP2 to Flamingo"

Our investigations also found that, for a given target text, the vision encoder’s value vectors remain approximately consistent across input images. This observation has three implications:

1. **Vision Encoder Specificity:** The produced value vectors align with the vision encoder’s architecture and pretraining.
2. **Limited Diversity in Perturbations:** Because value vectors remain consistent for a given target text, the diversity of generated perturbations may be constrained. This constraint could limit the attack’s ability to produce highly varied adversarial examples.
3. **Target Text Dependency:** Similar value vectors across different images for the same target text indicate that target-text choice strongly influences the attack outcome, rather than the input image’s specific features.

## 7 Discussion

### 7.1 Observations with reproducibility

Our study replicated key aspects and findings of Luo et al. (2024) concerning the Cross-Prompt Attack (CroPA). The experiments substantiate the original claims about enhanced cross-prompt adversarial transferability in Vision-Language Models (VLMs).

In particular, Table 1 confirms strong cross-prompt transferability across VQA, classification, and captioning, with CroPA consistently outperforming baseline methods. CroPA also remains robust under few-shot evaluation: Section 6.4 and Table 2 show that CroPA outperforms the baseline despite the shift from optimization conditions introduced by in-context examples. For shot=2, CroPA’s overall ASR is 0.6635 versus Multi-P’s 0.5730.

### 7.2 Noise Initialization findings

Section 5.1 identifies reliance on random noise initialization as a critical limitation of CroPA and related attacks. Random initialization is agnostic to the target prompt’s semantic structure. We therefore hypothesized that the lack of semantic guidance introduces instability, prolongs convergence, and reduces cross-prompt and cross-modal transferability.

To address that limitation, we proposed initialization using a diffusion-synthesized reference image and alignment in the vision encoder’s feature space. The purpose is to orient perturbations toward a meaningful adversarial direction from the outset.

Section 6.5.1 empirically validates this hypothesis: across all evaluated benchmarks, our initialization method significantly outperforms CroPA and randomly initialized baselines, yielding up to 15% gains in Targeted ASR and notably reducing optimization steps. Table 3 reports the targeted ASRs, while Figure 7(a) shows the convergence comparison. These findings confirm that semantically anchored initialization both improves adversarial efficacy and provides a principled, efficient foundation for VLM attack frameworks.

Thus, the improvement is not limited to attack effectiveness. By reducing the required optimization steps and anchoring perturbations in the vision encoder’s semantic space, the method also provides a more efficient and robust alternative to random initialization.

### 7.3 Evaluation of Transferability Experiments

#### 7.3.1 Cross-Image Transferability

The original authors claim that computing perturbations across images can achieve cross-image transferability. However, our experiments find that CroPA’s perturbations remain tightly coupled to the low-level features of the image used for generation, despite CroPA’s strong cross-prompt transferability.

We therefore integrated SCMix, described in Section 5.3, and CutMix. Both augmentation methods increase cross-image diversity while preserving semantic information from the original images.

Section 6.5.2 and Table 5 show substantial cross-image improvements with SCMix, especially for VQA, VQA-specific, and Captioning. Overall ASR increases by approximately 27 percent, from 0.4355 for base cross-image CroPA to 0.7055 with SCMix. CutMix produces a net increase of around 12 percent, reaching 0.5565.

We believe SCMix’s advantage arises because random cropping and resizing explicitly introduce spatial and scale invariance within the same image, after which soft mixing adds a second image’s features. Consequently, perturbations are encouraged to target semantically aligned features that are less sensitive to exact position or scale, making generalization across inputs easier.

Thus, Table 5 shows that both SCMix and CutMix improve cross-image transferability when applied to a cross-prompt framework such as CroPA. Because both enhancements are image augmentation methods, we believe the methods add cross-image features that support more general perturbations.

For further improvement, future research should investigate more complex, layered approaches beyond augmentation. Possibilities include combining augmentation with feature-alignment attacks based on images’ latent representations or perturbing latent space directly.

#### 7.3.2 Cross-Model Transferability

The D-UAP-modified CroPA framework in Section 5.2 significantly advances CroPA’s attack method. As reported in Section 6.5.3 and Table 6, the overall Targeted ASR reaches 96.85%, an 13.5% improvement over base CroPA. Improvements are especially notable for VQAgeneral and Captioning, with increases of 13% and 28%, respectively; Table 6 reports 0.9420 versus 0.8176 for VQAgeneral and 0.9810 versus 0.6960 for Captioning.

These gains occur even with challenging or harmful target text such as “Bomb,” which typically triggers model guardrails. The results therefore demonstrate the D-UAP framework’s robustness in bypassing safety mechanisms.

The framework also improves cross-model transferability, particularly between models with similar vision encoders. For example, Table 8’s BLIP2-to-InstructBLIP rows report overall ASR of 0.6850 for CroPA + D-UAP versus 0.5065 for CroPA. However, dependency on the particular vision encoder constrains effectiveness, limits perturbation diversity, and hinders transfer to models with different architectures.

This limitation highlights the broader challenge of encoder-agnostic adversarial methods. Nevertheless, D-UAP represents a critical step toward universal adversarial perturbations effective across both prompts and images, motivating further research on generalizability and robustness in vision-language models.

Finally, our cross-model comparisons exclude the Contextual Injection Attack because CIA is inherently non-transferable and did not produce comparable results, as explained in Section 5.4.3.

### 7.4 Trade-offs Between Different Types of Adversarial Transferability

Our analysis reveals fundamental trade-offs between transferability dimensions in VLMs. Understanding these trade-offs is essential when designing robust adversarial attacks.

#### 7.4.1 Cross-Prompt vs. Cross-Model Transferability

Original CroPA achieves strong cross-prompt transferability by optimizing perturbations against competing text embeddings. However, that optimization implicitly anchors perturbations to the target model’s architecture. Section A.1 confirms the limitation: ASR drops significantly when perturbations generated on one model are applied to another.

The D-UAP enhancement in Section 5.2 addresses this trade-off by targeting generalizable vision encoder components rather than model-specific features. Table 8 shows improved transfer between architecturally similar models, such as BLIP-2 and InstructBLIP. However, the improvement remains modest between fundamentally different architectures, such as BLIP-2 and Flamingo: overall ASR rises from 0.1435 for CroPA to 0.2335 for CroPA + D-UAP in the BLIP2-to-Flamingo rows. These results suggest an inherent tension between optimizing simultaneously for prompt diversity and model diversity.

#### 7.4.2 Cross-Prompt vs. Cross-Image Transferability

Cross-prompt and cross-image transferability are competing optimization objectives. CroPA attains strong cross-prompt transferability by tightly coupling perturbations to image-specific features. However, that same coupling fundamentally limits cross-image generalization.

We propose data augmentation to address this limitation. The SCMix and CutMix integrations in Section 5.3 diversify optimization inputs. Table 5’s overall ASRs—0.4355 for the base method, 0.7055 with SCMix, and 0.5565 with CutMix—show promising improvements for future research. However, our findings also indicate that augmentation does not fully resolve the fundamental tension between cross-prompt and cross-image objectives.

#### 7.4.3 Role of Initialization in Transferability Balance

Diffusion-based initialization, described in Section 5.1, offers another perspective on these trade-offs. Anchoring perturbations to semantically meaningful initializations achieves higher ASRs without sacrificing cross-prompt transferability. Table 3’s Overall column and Figure 7(a)’s convergence comparison support this result. Thus, the starting point in the optimization landscape significantly influences the achievable transferability balance.

Unlike methods that change optimization itself, improved initialization allows perturbations to converge toward more general adversarial directions that remain effective across prompts. However, even improved initialization does not eliminate the fundamental tension between cross-prompt and cross-image or cross-model transferability. A perturbation optimized for one dimension inevitably sacrifices effectiveness in others.

#### 7.4.4 Implications for Future Research

The observed trade-offs highlight a critical challenge: no single attack methodology can simultaneously maximize every form of VLM adversarial transferability. Future work should therefore develop frameworks that permit controlled balance among these competing objectives, potentially through multi-objective optimization or architecturally aware attack designs.

Our work shows that individual techniques can extend particular transferability dimensions. However, a comprehensive solution would require fundamentally rethinking interactions between adversarial perturbations, model architectures, and data representations.

Despite these affirmations, cross-prompt adversarial transferability remains underexplored. The considerable lack of literature leaves a gap in understanding prompt-induced VLM vulnerabilities. Addressing that gap is imperative because of its implications for secure model deployment.

### 7.5 Broader Impact

From an attacker’s perspective, highly cross-prompt transferable adversarial examples pose significant threats. Such examples can make VLMs produce malicious or misleading outputs even in response to harmless queries. Consequently, these examples could be used to disseminate false information or subvert systems that rely on VLMs for content generation and decision-making.

Conversely, imperceptible perturbations offer a defensive mechanism for protecting sensitive information. Embedding such perturbations in images can cause VLMs to consistently output predetermined, non-sensitive text. The resulting output can thwart unauthorized attempts to extract confidential information from personal images. Thus, the technique offers a proactive privacy and data-security measure when visual data is increasingly vulnerable to exploitation.

Our refinements to CroPA include improved initialization and an enhanced loss function. The corresponding ASR results appear in Tables 3, 4, 6, and 7, while Tables 5 and 8 report improvements in adversarial-example generalizability across images and models. These advances reinforce cross-prompt attacks’ efficacy while also creating opportunities for more resilient defenses.

In conclusion, the reproducibility study affirms Luo et al.’s (2024) foundational work and emphasizes the need for deeper investigation of cross-prompt adversarial transferability. A comprehensive understanding is crucial both for developing VLMs that withstand adversarial manipulation and for designing countermeasures that protect user data and preserve output integrity.

### 7.6 Limitations and Future Work

Although we reproduced CroPA’s key results and conducted extensive ablations, limited computational resources repeatedly hindered progress. Training and evaluating adversarial attacks on large-scale VLMs such as Flamingo and BLIP2 require substantial GPU memory and processing power. Consequently, resource availability limited the scale and depth of some experiments, including extensive hyperparameter tuning and cross-model transferability evaluation across a wider range of architectures.

We also intended to investigate transferability using an ensemble of these models, which would have provided deeper insight into adversarial perturbations’ generalizability. However, the ensemble’s very high memory and compute requirements made the approach infeasible with our infrastructure.

Future work could improve cross-model transferability through encoder-agnostic methods and enhance cross-image generalization. Implementing optimization with query-based strategies offers another way to improve practical applicability.

### 7.7 Communication With Original Authors

We could not establish direct communication with the original authors during replication. At the time of writing, the issue raised on their GitHub repository concerning the BLIP-2 and InstructBLIP models remains unresolved.
