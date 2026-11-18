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

Large Vision-Language Models (VLMs), which process images and text together, have revolutionized computer vision. They support image classification, image captioning, and visual question answering. However, they remain highly vulnerable to adversarial attacks: deliberate changes to inputs that cause incorrect or harmful outputs. This vulnerability is particularly important when both image and text inputs can be manipulated.

We conduct a comprehensive reproducibility study of “An Image is Worth 1000 Lies: Adversarial Transferability Across Prompts on Vision-Language Models.” We validate its Cross-Prompt Attack (CroPA). Cross-prompt transferability means that an adversarial image remains effective when the accompanying textual prompt changes. Our reproduction confirms CroPA’s superior transferability compared with the baselines used in the original work (Tables 1–2 and Figures 4–6).

We also propose three extensions: (1) a new initialization strategy that significantly improves Attack Success Rate (ASR), the rate at which an attack meets its success criterion; (2) an investigation of cross-image transferability using universal perturbations, or input changes shared across images; and (3) a new loss function that targets attention mechanisms, the processes that control interactions between image regions, within the vision encoder. The vision encoder converts images into representations used by the language model.

We evaluate Flamingo, BLIP-2, and InstructBLIP, all VLMs, and extend the experiments to the VLM LLaVA. These evaluations validate the original results and show that our improvements consistently boost adversarial effectiveness (Tables 3–8). Our work reinforces the importance of studying VLM vulnerabilities. It also provides a more robust framework for generating transferable adversarial examples, meaning perturbed inputs that remain effective beyond their optimization setting. These findings have significant implications for VLM security in real-world applications.

Our code is available at https://github.com/Swadesh06/Revisting_CroPA

Reviewed on OpenReview: https://openreview.net/forum?id=5L90cl0xtf

## 1 Introduction

Large Vision-Language Models (VLMs) have significantly transformed computer vision. They support image classification, captioning, and visual question answering. Classification assigns an image to a category. Captioning generates a textual description. Visual question answering generates an answer to a question about an image. This versatility has encouraged deeper exploration of interactions between images and language.

However, recent studies show that VLMs remain highly vulnerable to adversarial attacks: Zhao et al. (2023), Qi et al. (2023), Zhang et al. (2022a), and Carlini et al. (2024). These attacks introduce subtle perturbations, or changes, to input images. The changes can cause incorrect or harmful outputs. Text inputs introduce further attack vectors beyond those faced by traditional vision models.

Prior studies examine different access settings. In a black-box setting, the attacker does not have access to the model’s parameters or internal gradients. Query-based attacks use model responses, while transfer-based attacks reuse adversarial inputs developed elsewhere. Zhao et al. (2023) analyzed these methods on VLMs such as BLIP and BLIP-2. In a white-box setting, the attacker can access gradients and internal representations. Schlarmann & Hein (2023) examined white-box targeted attacks, which seek a specified output, and untargeted attacks, which seek a changed output.

These studies primarily examined image attacks. Later work also explored textual adversarial perturbations. Qi et al. (2023) showed that adversarial images could cause VLMs to execute harmful instructions. Tu et al. (2023) systematically evaluated both visual and textual attacks.

Adversarial generalization in VLMs has traditionally been divided into two main categories:

- **Cross-Model Transferability:** An adversarial example remains adversarial across different VLM architectures. This is commonly called transferability.
- **Cross-Image Transferability:** A perturbation remains effective across multiple images. Such shared perturbations are often called Universal Adversarial Perturbations (UAPs).

Luo et al. (2024) introduced **Cross-Prompt Transferability**: an adversarial image remains effective across different textual prompts. Earlier work treated visual and textual perturbations independently. Their Cross-Prompt Attack (CroPA) instead uses learnable prompt changes during optimization to make the adversarial image effective regardless of the text input. They demonstrated CroPA on classification, captioning, and visual question answering.

Our goals are:

- **[Reproducibility Study] Reproducing the Results from the Original Paper:** We successfully reproduced and verified four main claims. First, CroPA achieves superior cross-prompt transferability across target texts, independently of their semantic meaning or word frequency (Table 1). Second, increasing the prompt count improves transferability, but performance converges rapidly. Even with more prompts, the baselines do not surpass CroPA (Figure 4). Third, CroPA remains effective when additional in-context learning examples—examples included in the input to help the model adapt to a task—are provided (Table 2). Lastly, we reproduce the overall ASR convergence trend as attack iterations increase. We also analyze convergence separately for each task (Figures 5–6).

- **[Extended Work] Better Initialization Strategy:** We propose a new starting point for perturbation optimization. It substantially increases ASR and transferability (Tables 3–4 and Figure 7).

- **[Extended Work] Investigating Cross Image Transferability and Image Augmentation:** We use CroPA’s framework to learn one perturbation shared across images. We investigate whether image augmentation, which creates varied training inputs, improves transferability (Table 5).

- **[Extended Work] Guiding perturbations via Target Value Vectors:** We propose a new loss function based on the vision encoder’s attention mechanism. Attention operates over image patches, or image regions. Its value vectors hold information from those patches. Guiding these vectors toward a target improves generalization and ASR (Tables 6–8).

We also analyze the mechanisms behind these improvements. This analysis offers insights into VLM vulnerabilities and provides a more robust and versatile framework for transferable adversarial examples.

## 2 Related Works

Adversarial attacks on deep neural networks have been studied extensively since Szegedy et al. (2013) and Goodfellow et al. (2014). Applying them to VLMs introduces distinct challenges and opportunities. This section reviews adversarial transferability and positions CroPA within that research.

### 2.1 Adversarial Transferability

Adversarial transferability occurs along several dimensions. Liu et al. (2017) and Tramèr et al. (2017) established that examples crafted for one model can often deceive other models, including models with different architectures.

Perturbations can also generalize across images. Moosavi-Dezfooli et al. (2017) and Mopuri et al. (2017) showed how this enables UAPs.

Transferability also extends across tasks. Examples designed for classification can affect object detection and other visual tasks: Naseer et al. (2019b), Naseer et al. (2019a), Lu et al. (2020), and Salzmann et al. (2021). These different forms of transferability raise important questions about model robustness.

### 2.2 Adversarial Attacks on Vision-Language Models

Early vision-language robustness research focused on task-specific attacks. Xu et al. (2019), Zhang et al. (2020), Aafaq et al. (2021), and Chen et al. (2017) demonstrated attacks on image captioning.

Visual Question Answering (VQA) systems were also vulnerable to targeted attacks that mislead attention mechanisms. Examples include Xu et al. (2018), Kaushik et al. (2021), Kovatchev et al. (2022), Li et al. (2021), Sheng et al. (2021), and Zhang et al. (2022b).

These methods mainly targeted lightweight CNN-RNN architectures: combinations of convolutional neural networks for visual processing and recurrent neural networks for sequential processing. Those architectures lacked in-context learning. This limits the methods’ applicability to contemporary VLMs.

Zhao et al. (2023) studied modern large VLMs, including BLIP and BLIP-2, in black-box settings. Their emphasis was cross-model transferability rather than CroPA’s cross-prompt perspective.

### 2.3 Cross-Prompt Adversarial Transferability

Large VLMs can adapt to tasks through textual prompts rather than being restricted to one fixed task: Li et al. (2023b), Awadalla et al. (2023), Li et al. (2023b), Zhu et al. (2023), and Liu et al. (2023). This flexibility creates an attack surface in which adversarial examples must remain effective as prompts change.

CroPA pioneered this perspective (Luo et al., 2024). During optimization, learnable prompts compete with image perturbations. This differs from treating visual and textual attacks independently and provides a new way to evaluate VLM robustness.

Because this topic is relatively new, no credible established benchmarks or baselines yet exist. Yang et al. (2024) nevertheless also explore cross-prompt transferability. Their Context Injection Attack (CIA) uses gradient-based perturbations—updates guided by how the loss changes—to inject target tokens into visual and textual contexts. Tokens are the units processed by the model. We include CIA as an additional comparison.

### 2.4 Enhancing Adversarial Transferability

Input diversity methods (Xie et al., 2019) and data augmentation strategies (Wang et al., 2021) have shown promise for stronger transferable attacks.

Recent work identifies gradient misalignment, or differences between useful update directions, as a key transferability limitation: Li et al. (2023a) and Wang et al. (2023). This finding motivates new optimization strategies. We build on these insights to enhance CroPA across several transferability dimensions.

## 3 Scope of reproducibility

We examine and validate the results of Luo et al. (2024). Our primary objective is to confirm that CroPA significantly improves adversarial transferability across prompts by carefully reproducing their procedures.

We aim to verify four claims:

- **Claim 1:** CroPA achieves cross-prompt transferability across various target texts.
- **Claim 2:** CroPA achieves the best overall performance across different number of prompts.
- **Claim 3:** CroPA converges towards higher ASR as number of iterations are increased.
- **Claim 4:** CroPA outperforms baselines under few-shot settings.

The corresponding evidence appears in Sections 6.1–6.4.

Beyond replication, we investigate cross-model and cross-image transferability and compare CroPA with another cross-prompt method. We assess whether CroPA-generated adversarial images consistently deceive different VLMs, regardless of input prompt or specific model parameters (Sections 6.5.2–6.5.4).

These experiments both reproduce the original work and explore CroPA’s broader applicability to robust and versatile adversarial attacks on VLMs.

## 4 Methodology

### 4.1 Problem Formulation

Neural-network vulnerability to adversarial attacks has been well documented since Goodfellow et al. (2014). Luo et al. (2024) extend this formulation to cross-prompt attacks on VLMs. Their central question is whether an image perturbation remains effective across different text inputs.

Let \(f\) be a VLM that processes an image \(x_v\) and a text input \(x_t\). The subscript \(v\) denotes the visual input; \(t\) denotes the textual input. Let \(\delta_v\) be an image perturbation. The perturbed image is \(x_v+\delta_v\).

The perturbation must satisfy
\[
\|\delta_v\|_p\leq\epsilon_v.
\]
Here, \(\|\cdot\|_p\) measures perturbation size using the norm selected by \(p\), and \(\epsilon_v\) is the maximum allowed size. This constraint keeps the change within human-imperceptible bounds. It follows established adversarial-learning practice in a multimodal setting, meaning a setting with both image and text inputs (Carlini et al., 2024).

We reproduce two attack scenarios. For both, let \(k\) be the number of prompts and \(x_t^i\) the \(i\)-th prompt, where \(i\) runs from \(1\) to \(k\). Let \(L\) be the language-modeling loss, which measures the mismatch between the model’s output and a comparison text. The summation symbol \(\sum\) adds this loss across prompts.

A **targeted attack** seeks a predetermined output text \(T\), regardless of the input prompt. It chooses \(\delta_v\) to minimize, denoted by \(\min\), the summed loss:

\[
\min_{\delta_{v}}\sum_{i=1}^{k}L(f(x_{v}+\delta_{v},x_{t}^{i}),T).
\tag{1}
\]

A **non-targeted attack** instead seeks a discrepancy between the clean-image output and the perturbed-image output. It maximizes, denoted by \(\max\), the corresponding loss:

\[
\max_{\delta_{v}}\sum_{i=1}^{k}L(f(x_{v}+\delta_{v},x_{t}^{i}),f(x_{v},x_{t}^{i})).
\tag{2}
\]

ASR measures attack effectiveness. For a targeted attack, success requires the exact target text. For a non-targeted attack, success means any output different from the prediction for the clean image.

### 4.2 Baseline Approaches

Cross-prompt transferability is a new direction without established literature baselines. We therefore use the two baselines introduced by Luo et al. (2024): **Single-P**, which optimizes with one prompt, and **Multi-P**, which optimizes with multiple prompts.

Single-P uses this targeted objective:

\[
\min_{\delta_{v}}L(f(x_{v}+\delta_{v},x_{t}),T).
\tag{3}
\]

**Figure 1:** Framework overview of CroPA as presented in Luo et al. (2024). The architecture uses learnable perturbations for both the image (\(\delta_v\)) and the prompt (\(\delta_t\)). They work antagonistically: \(\delta_v\) minimizes the language-modeling loss, while \(\delta_t\) maximizes it. The forward pass, which computes the model output, is shown by solid arrows through the vision encoder and language model. Backpropagation, which computes gradients for updating the perturbations, is shown by dashed arrows. The update frequencies are configurable. Model parameters marked with \(*\) remain fixed during the attack. The symbol \(\oplus\) denotes element-wise addition.

For Multi-P, let \(X_t=\{x_t^1,x_t^2,\ldots,x_t^k\}\) be the prompt collection. Its targeted objective is:

\[
\min_{\delta_{v}}\sum_{i=1}^{k}L(f(x_{v}+\delta_{v},x^{i}_{t}),T).
\tag{4}
\]

For non-targeted attacks, these objectives are inverted to maximize the difference between clean and perturbed outputs. Single-P and Multi-P provide reference points for evaluating CroPA.

### 4.3 Cross Prompt Attack

CroPA uses learnable prompts to improve cross-prompt transferability (Luo et al., 2024). Its key feature is competition between prompt and image perturbations, rather than collaboration to deceive the model.

The image perturbation \(\delta_v\) minimizes the loss for generating target text \(T\). The prompt perturbation \(\delta_t\) maximizes that loss. The image perturbation must therefore remain effective against prompts that make its objective harder. This antagonistic relationship forces stronger transferability across prompts.

The resulting min-max objective—minimization over image changes and maximization over prompt changes—is:

\[
\min_{\delta_{v}}\max_{\delta_{t}}L(f(x_{v}+\delta_{v},x_{t}+\delta_{t}),T).
\tag{5}
\]

The implementation uses Projected Gradient Descent (PGD), an iterative method that updates an input using its loss gradient and then projects it back into the allowed perturbation region (Madry et al., 2017). Image updates use gradient descent to decrease loss. Prompt updates use gradient ascent to increase it.

The parameter \(N\) controls the update frequency: the image is updated \(N\) times for each prompt update. Figure 1 shows the framework. Algorithm 1 reproduces the procedure described by Luo et al. (2024).

In the algorithm, \(x'_v\) is the adversarial image and \(x_t^{\prime i}\) is the initialized or updated adversarial version of prompt \(i\). The budget is \(\epsilon\); \(\alpha_1\) and \(\alpha_2\) are image and prompt step sizes; and \(K\) is the iteration count. The symbol \(\nabla\) denotes a gradient with respect to the indicated input. The gradients \(g_v\) and \(g_t\) guide image and prompt updates. The function \(\operatorname{sign}\) takes each gradient component’s sign. The condition \(\operatorname{mod}(\text{step},N)=0\) means the current step is divisible by \(N\). Finally, \(\operatorname{Clip}_{x_v,\epsilon}\) projects the image into the permitted \(\epsilon\)-ball around \(x_v\), the set of inputs within budget \(\epsilon\).

**Algorithm 1 CroPA: Cross Prompt Attack**

**Inputs:** Model \(f\), target text \(T\), vision input \(x_v\), prompt set \(X_t\), perturbation size \(\epsilon\), step sizes \(\alpha_1\) and \(\alpha_2\), iteration count \(K\), adversarial prompt update interval \(N\).

**Output:** Adversarial example \(x'_v\).

1. Initialize \(x'_v=x_v\).
2. **For** step \(=1\) to \(K\):
3.   Uniformly sample prompt \(x_t^i\) from \(X_t\).
4.   **If** \(x_t^i\) is not initialized:
5.   Initialize \(x_t^{\prime i}=x_t^i\).
6.   **End if.**
7.   Compute the adversarial-image gradient:
   \[
   g_v=\nabla_{x_v}L(f(x'_v,x_t^i),T).
   \]
8.   Update by gradient descent:
   \[
   x'_v=x'_v-\alpha_1\cdot\operatorname{sign}(g_v).
   \]
9.   **If** \(\operatorname{mod}(\text{step},N)=0\):
10.   Compute the adversarial-prompt gradient:
    \[
    g_t=\nabla_{x_t}L(f(x'_v,x_t^i),T).
    \]
11.   Update by gradient ascent:
    \[
    x_t^{\prime i}=x_t^{\prime i}+\alpha_2\cdot\operatorname{sign}(g_t).
    \]
12.   **End if.**
13.   Project \(x'_v\) into the \(\epsilon\)-ball of \(x_v\):
    \[
    x'_v=\operatorname{Clip}_{x_v,\epsilon}(x'_v).
    \]
14. **End for.**
15. **Return** \(x'_v\).

At evaluation, we apply only the optimized image perturbation and discard the prompt perturbations. Thus, effectiveness must come from the image perturbation’s transferability, not from modifying the evaluation prompt.

## 5 Additional Investigative Experiments

CroPA’s PGD optimization effectively increases ASR on a single model. However, its reliance on that model’s loss feedback limits transferability. Other work identifies the same limitation in gradient-based perturbation methods and proposes enhancements to address it: Qin et al. (2024), Li et al. (2024), Wang et al. (2023), and Li et al. (2023a). The original authors also recognize it. Table 9 in Appendix A.1 verifies this limitation. Table 1 further shows weak performance on some tasks, especially Classification and Captioning.

CroPA is also computationally intensive. Optimizing perturbations for only 50 images takes over 6 hours on an L40S graphics processing unit (GPU) with 48GB—48 gigabytes—of memory, using 95% of its video random-access memory (VRAM).

These limitations motivate our extensions. We target (i) CroPA’s inconsistent performance across tasks and (ii) the transferability dimensions defined in Section 1. Our aim is to supplement the framework with more transferable image perturbations.

Transferability is not a core proposal of Luo et al. (2024), but it could also reduce computation. In an ideal scenario, cross-image transfer would avoid optimizing a separate perturbation for every image. Cross-model transfer could provide even greater efficiency.

The following subsections describe the experiments and their mechanisms.

### 5.1 Noise Initialization via Vision Encoding Optimization

A core limitation of CroPA is its random noise initialization, a strategy shared by many gradient-based attacks. Although CroPA subsequently optimizes this noise, the starting point contains no information about the target’s meaning. This produces instability and slower convergence. The problem is especially important for VLMs, where semantic alignment—matching the meaning represented by an image and a prompt—is crucial.

Our evaluation of CroPA’s bi-level optimization, in which image and prompt changes have opposing objectives, shows strong initialization sensitivity (Section 6.5.1, Table 3, and Figure 7). CroPA cannot guide the optimization toward semantically meaningful directions from the outset. Conventional initialization therefore produces suboptimal ASR and reduced cross-prompt transferability.

We hypothesize that a semantically informed initialization can improve attack efficacy while preserving perturbation imperceptibility. We first use a diffusion-based image-generation model to synthesize an image corresponding to the target prompt. We then use the vision encoder’s feature space—the representations it produces—to align the initial perturbation with that target.

Our Noise Initialization via Vision Encoding Optimization method has three steps:

- Generate a semantically relevant target image using a state-of-the-art diffusion model, such as Stable Diffusion XL.
- Align the perturbation with that image’s vision-encoder representation.
- Optimize with a Mean Squared Error (MSE) loss, a measure of squared differences between representations, to enforce semantic similarity while maintaining adversarial effectiveness.

Replacing random initialization in this way improves adversarial robustness, attack strength, cross-prompt transferability, and efficiency. It reduces the optimization steps needed for convergence (Table 3 and Figure 7). The perturbation is semantically informed from the start rather than acquiring all target information during the later attack.

#### 5.1.1 Noise Initialization

Preliminary experiments on BLIP-2 and InstructBLIP suggest that semantically informed noise substantially improves attack efficacy over random initialization. It adds only 20–25 seconds of computation per image on a single GPU.

The diffusion-based method improves alignment in the vision encoder’s feature space. It also preserves the visual fidelity of adversarial examples while meeting strict perturbation budgets.

#### 5.1.2 Diffusion-Based Target Synthesis

Given target prompt \(T\), we generate a target image \(\mathbf{x}_{\text{target}}\) that embodies its semantic attributes. We use Stable Diffusion XL (SDXL):

\[
\mathbf{x}_{\text{target}}=\mathcal{D}(T,\mathbf{z};\theta_{\text{SDXL}}).
\tag{6}
\]

Here, \(\mathcal{D}\) is the diffusion process. The input \(\mathbf{z}\) is sampled from a Gaussian, or normal, distribution. The symbol \(\theta_{\text{SDXL}}\) denotes the diffusion model’s pretrained weights.

The generated image captures the semantic essence of \(T\). It supplies an informative target for perturbation initialization.

#### 5.1.3 Vision-Encoder Anchored Perturbation

Let \(f_v(\cdot)\) be the VLM’s vision encoder. Let \(\mathbf{x}\) be the original image and \(\boldsymbol{\delta}\) a perturbation. We seek an initial perturbation \(\boldsymbol{\delta}_{\text{init}}\) whose perturbed image resembles the target image in the encoder’s output space:

\[
\boldsymbol{\delta}_{\text{init}}
=
\underset{\|\boldsymbol{\delta}\|_{\infty}\leq\epsilon}{\arg\min}
\left\|f_v(\mathbf{x}+\boldsymbol{\delta})-f_v(\mathbf{x}_{\text{target}})\right\|_2^2.
\tag{7}
\]

The operator \(\arg\min\) selects the perturbation that minimizes the expression. The squared \(2\)-norm, \(\|\cdot\|_2^2\), sums squared component differences. The infinity norm, \(\|\cdot\|_\infty\), measures the largest absolute component. Thus, the \(\ell_\infty\) constraint limits every perturbation component to the budget \(\epsilon\), ensuring imperceptibility.

This initialization places the adversarial example in a semantically meaningful neighborhood of the target representation. Algorithm 2 in Appendix B formally describes the method.

#### 5.1.4 Adversarial Optimization via PGD

After initialization, we refine the adversarial image using PGD. Let \(\mathbf{x}_{\text{adv}}^{(t)}\) be the adversarial image at iteration \(t\), and let \(\alpha\) be the step size. The update is:

\[
\mathbf{x}_{\text{adv}}^{(t+1)}
=
\Pi_{\mathcal{B}_{\epsilon}(\mathbf{x})}
\Bigl[
\mathbf{x}_{\text{adv}}^{(t)}
-\alpha\nabla_{\mathbf{x}_{\text{adv}}^{(t)}}\mathcal{L}_{\text{MSE}}
\Bigr].
\tag{8}
\]

The set \(\mathcal{B}_{\epsilon}(\mathbf{x})\) is the admissible \(\ell_\infty\) ball around the original image. The operator \(\Pi\) projects an update back into this set.

The optimization loss is the MSE between the vision-encoder outputs:

\[
\mathcal{L}_{\text{MSE}}
=
\left\|f_v(\mathbf{x}_{\text{adv}})-f_v(\mathbf{x}_{\text{target}})\right\|_2^2.
\tag{9}
\]

Each update therefore moves the adversarial image’s representation toward the target’s semantic embedding, or encoded representation.

### 5.2 Guiding perturbations via Target Value Vectors and evaluating Cross-Model Tranferability

CroPA uses PGD to optimize image and text inputs, but it lacks architectural sensitivity. In particular, it does not exploit the internal dynamics of the vision encoder, which plays a dominant role in grounding language in image content. CroPA treats image and language modalities as uniformly vulnerable. This overlooks modality-specific weaknesses and limits the strength, structure, and transferability of its perturbations.

CroPA minimizes image-perturbation loss and maximizes text-perturbation loss to guide output toward a target. It does not directly target the encoder’s representational backbone.

Cui et al. (2023) and Wang et al. (2024) show that perturbing vision-encoder output embeddings can significantly impair the perception of large vision-language models (LVLMs). This, in turn, misleads their language-model responses. Many LVLMs use similar image encoders. These encoders produce visual tokens containing semantic and structural information. The tokens form a critical interface through which the language decoder accesses and reasons over images.

We therefore combine the CroPA loss with a loss targeting the vision encoder’s attention mechanism, specifically its value vectors. Our hypothesis is that guiding an input image’s value vectors toward those of a reference image associated with the target text can improve ASR—also termed Adversarial Success Rate here—under the same hyperparameter constraints as CroPA. Hyperparameters are settings chosen for the optimization procedure.

Our approach is inspired by Doubly-Universal Adversarial Perturbation (Doubly-UAP). Instead of maximizing value-vector differences, we minimize losses between the input and target-reference value vectors, described here as those of \(x_i\) and \(T_i\). We hypothesize that targeting value vectors, which encode essential visual features, disrupts interpretation more effectively than generic PGD perturbations. We also freeze the text encoder, leaving its parameters unchanged. This focuses the perturbations on visual processing rather than text-based artifacts.

Our modification aims to:

1. Improve attack effectiveness by exploiting information in the vision encoder.
2. Align perturbations semantically with the target-text representation.
3. Improve adversarial robustness across models while remaining computationally feasible.

Jointly optimizing the CroPA loss and the modified Doubly-UAP loss gives a more structured, interpretable, and targeted attack framework.

We adapt a component of Kim et al. (2024), whose Doubly-UAP method identifies the attention components that most effectively influence VLM performance. We focus on two fundamental components:

1. **Attention Weights:** These determine how much each image patch attends to other patches. They control patch interaction and relevance. Targeting them could interfere with the encoder’s ability to establish those relationships.
2. **Value Vectors:** These hold each patch’s actual information. Perturbing them can disrupt essential patch content and impair interpretation.

We target vision encoders because they are crucial for visual interpretation. Based on prior analysis, we focus on vulnerable value vectors in middle-to-late attention layers.

In standard Doubly-UAP, the perturbation \(\delta^*\) is selected by:

\[
\delta^{*}
=
\arg\max_{\delta}
\frac{1}{|L|}
\sum_{l\in L}
\operatorname{Loss}(V_l(x),V_l(x+\delta)).
\tag{10}
\]

Here, \(x\) is an input image and \(\delta\) its perturbation. The operator \(\arg\max\) selects the perturbation maximizing the expression. In this equation, \(L\) is a set of layer indices, not the language-modeling loss used above; \(|L|\) is the number of selected layers. The index \(l\) identifies a layer, and \(V_l(x)\) gives that layer’s value vectors for image \(x\). The function \(\operatorname{Loss}\) compares the vectors.

#### 5.2.1 Our Modified Approach

Our modification introduces target value vectors from a reference image corresponding to desired target text \(T\). Instead of only maximizing deviation from the original vectors, we enforce alignment with a predefined target representation. We use **D-UAP** as the label for this modified Doubly-UAP component.

**Figure 2:** D-UAP Architecture Diagram

The steps are:

1. Extract value vectors from the \(l\)-th vision-encoder layer for both the input image and the target text’s reference image.
2. Compute the cosine similarity loss. Cosine similarity compares vector directions.
3. Jointly minimize the negative of this loss and the CroPA loss, with appropriate scaling, to ensure adversarial robustness and target alignment.

Let \(V_i(x+\delta)\) be the perturbed value vectors of attention head \(i\). An attention head is one component of the attention mechanism. Let \(V_t\) be the reference image’s target value vectors. We define:

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

Here, \(L_{\text{d-UAP}}\) is the value-vector loss. The dot denotes the vector dot product, and \(\|\cdot\|_2\) is vector length. In this equation, \(N\) means the number of attention heads, rather than the prompt-update interval used in Algorithm 1. This loss encourages the perturbed image’s vectors to align closely with the target representation.

We combine it with the CroPA loss:

\[
L_{\text{re-CroPA}}
=
L_{\text{CroPA}}(x_v+\delta_v,x_t+\delta_t,T)
-\lambda L_{\text{d-UAP}}(\delta_v).
\tag{12}
\]

The symbol \(L_{\text{CroPA}}\) denotes the CroPA loss; \(L_{\text{re-CroPA}}\) labels the combined objective. The hyperparameter \(\lambda\) controls the relative importance of the value-vector term.

Joint optimization preserves the perturbation’s adversarial nature and enforces semantic alignment. Gradients from both terms are combined to update \(\delta\). The resulting adversarial example has cross-prompt transferability and guided semantic influence. This modification supports more precise, controlled, and interpretable manipulation for robustness and security analysis. Algorithm 3 in Appendix B summarizes the method.

This is a step toward more transferable CroPA perturbations. However, the value vectors depend on the model’s vision encoder. We therefore hypothesize that transferability may remain limited to models with the same or similar encoder architectures. Table 8 evaluates this limitation.

### 5.3 Transferability across images via Image Augmentation

Traditional adversarial methods optimize each image separately through iterative gradient descent. This often overfits image-specific features, limiting cross-image transferability and increasing computation. CroPA likewise learns a perturbation for one image across multiple prompts.

Luo et al. (2024) describe CroPA as orthogonal to cross-image transferability methods: the approaches address separate aspects and can be combined. They suggest computing perturbations across images, as in Moosavi-Dezfooli et al. (2017), to improve cross-image transfer. We test this claim and introduce an approach based on UAP research.

Fang et al. (2024) introduce UAPs for Vision-Language Pretraining (VLP) models to reduce overfitting and support cross-image transfer.

CutMix provides a complementary regularization strategy, meaning a way to discourage overfitting (Yun et al., 2019). It cuts rectangular regions from one image and pastes them into another, rather than blending pixels. Labels are mixed in proportion to patch area. This makes models learn from partial views and prioritize spatially invariant features, which remain useful as their position changes. It thereby improves robustness to adversarial attacks and occlusions, where image content is hidden.

Zhang et al. (2024) propose Effective and Transferable Universal Adversarial Attack (ETU). It improves UAP transfer across VLP models and tasks using global and local optimization. ETU introduces SCMix, an augmentation method combining self-mixing and cross-mixing to increase diversity while preserving semantic integrity. SCMix incorporates spatially invariant and scale-invariant features, which remain useful under changes in position and size.

We integrate SCMix into perturbation generation. We select it because the original results show that, for the same model, SCMix alone provides superior cross-image transferability even without ETU’s additional local utility reinforcement.

#### 5.3.1 SCMix

SCMix diversifies training inputs in two stages: self-mixing and cross-mixing.

For self-mixing, let \(I_1\) be the base image. Independently crop and resize it to produce \(X_1\) and \(X_2\), then blend those versions into \(I'_1\):

\[
I'_1=\eta X_1+(1-\eta)X_2
\quad\text{where}\quad
X_i=\operatorname{Resize}(\operatorname{RandomCrop}(I_1)),
\quad\eta=0.5.
\tag{13}
\]

The operator \(\operatorname{RandomCrop}\) selects a random image region, and \(\operatorname{Resize}\) changes its size. The weight \(\eta\) controls the blend.

Original SCMix samples \(\eta\sim\operatorname{Beta}(\alpha,\alpha)\): \(\eta\) is drawn from a Beta distribution with parameter \(\alpha>0\). Here, \(\alpha\) is a distribution parameter, not the PGD step size. We instead fix \(\eta=0.5\), equivalent to setting \(\alpha=\infty\), where \(\infty\) denotes infinity. Appendix D.3 justifies this choice.

Cross-mixing introduces a secondary image \(I_2\). The mixed output \(I_3\) is:

\[
I_3=\beta_1I'_1+\beta_2I_2
\quad\text{with}\quad
\beta_1\gg\beta_2\in[0,1).
\tag{14}
\]

The symbols \(\beta_1\) and \(\beta_2\) are mixing weights. The symbol \(\gg\) means “much greater than,” and \([0,1)\) is the range from zero inclusive to one exclusive.

The weighting preserves \(I_1\)’s dominant visual meaning while adding diversity from \(I_2\). It also preserves compatibility between the mixed image \(I_3\) and \(T_1\), the text description associated with \(I_1\). Thus, the image remains aligned with its text while incorporating another image’s features.

Self-mixing forces perturbations to attack features that survive variations within an image. Cross-mixing forces them to attack features that survive blending across images. Together, these operations reduce overfitting and promote universal perturbations that may work on unseen images.

At inference, CroPA returns only the perturbed image, or equivalently its perturbation. If that perturbation transfers across images, the attack can become universal across both prompts and images.

**Figure 3:** Illustration of self-mixing and cross-mixing. Input image \(I_1\) is self-mixed to produce intermediate representations \(X_1\) and \(X_2\), which are combined into \(I'_1\). An external image \(I_2\) is then mixed with \(I'_1\) to produce final output \(I_3\).

#### 5.3.2 CutMix

CutMix enhances transferability through discrete regional replacement (Yun et al., 2019). For images \(x_A\) and \(x_B\), it produces composite image \(\tilde{x}\):

\[
\tilde{x}=M\odot x_A+(1-M)\odot x_B.
\tag{15}
\]

Here, \(M\in\{0,1\}^{W\times H}\) is a binary mask with image width \(W\) and height \(H\). Its zero and one entries define a randomly placed rectangular region. The symbol \(\odot\) means element-wise multiplication.

CutMix learns from partial views. Unlike SCMix’s soft blending within and across images, it uses hard regional replacement. This forces perturbations to target spatially invariant features that persist across image combinations. When combined with adversarial training, it encourages attacks on higher-level semantic patterns rather than localized artifacts. This may improve cross-image transferability.

### 5.4 Attack Setup and Threat Models for Transferability Experiments

We define threat models—the attacker’s goals, knowledge, capabilities, and constraints—for cross-model and cross-image transferability. We also specify each evaluation protocol.

#### 5.4.1 Using D-UAP to explore Cross-Model Transferability for CroPA

**Adversary’s Goal:** Generate one perturbation for an image that makes multiple VLMs output a specified target text, regardless of prompt. This is a practical black-box transfer scenario: the attacker optimizes on an accessible model and deploys against unknown models.

**Knowledge and Capabilities:** The attacker has white-box access to a surrogate model, the accessible model used for optimization. In our experiments, this is either BLIP-2 or Flamingo. Access includes gradients and internal representations. For target models, the attacker knows only the general architecture family, such as whether it is CLIP-based, and cannot access parameters. CLIP here names the image-encoding architecture used by several evaluated VLMs.

The maximum perturbation budget is \(\epsilon=16/255\) under the \(L_\infty\), or infinity, norm. This ensures visual imperceptibility.

**Cross-Model Settings:** We examine two scenarios:

- **Intra-family transfer:** Optimize on BLIP-2 with the OPT-2.7b language model, then evaluate on InstructBLIP with the Vicuna-7b language model. OPT-2.7b and Vicuna-7b are language-model identifiers. Both VLMs use similar CLIP ViT-L/14 vision encoders, where ViT-L/14 identifies the encoder variant. They differ in language model and fine-tuning objective.
- **Cross-architecture transfer:** Transfer between BLIP-2 and Flamingo in both directions. These models have different vision encoders, language models, and architectural designs.

**Evaluation Setup:** Measure ASR separately for VQAgeneral, VQAspecific, Classification, and Captioning. VQAgeneral and VQAspecific distinguish the specificity of image-related question prompts. A successful attack makes the target model fail to generate the correct output for the perturbed image, regardless of prompt.

#### 5.4.2 Investigating Cross-Image Transferability via Augmentation Techniques

**Adversary’s Goal:** Generate one perturbation pattern that makes a VLM output a predetermined target text for any image and prompt. This offers greater attack efficiency than optimizing each image separately.

**Knowledge and Capabilities:** The attacker has a limited training-image set and corresponding SCMix-augmented variants. The attacker has white-box access to the target model during optimization. The perturbation must nevertheless generalize to unseen images. Its \(L_\infty\) budget is \(\epsilon=16/255\).

**Evaluation Setup:** Apply the universal perturbation to a separate set of test images not used in training. Measure ASR across task types. Success means the model fails to generate the correct output text for a previously unseen image, regardless of its content or prompt.

Our SCMix enhancement aims to reduce gradient-based overfitting to individual images. We hypothesize that controlled self-mixing and cross-mixing expose vulnerabilities independent of specific image features.

#### 5.4.3 Evaluation Rationale for Transferability experiments

We present all cross-prompt and cross-model transferability results under the untargeted attack setting. Two factors motivate this decision.

First, VLMs incorporate visual information through different embedding mechanisms. This makes it difficult to determine how an input perturbation changes the final output-token embeddings. We observe higher targeted transfer between similar vision encoders, such as BLIP-2 to InstructBLIP, but sharply different behavior across other architectures. Direct targeted-ASR comparison across heterogeneous architectures is therefore imprudent.

Second, the visual-textual fusion mechanism affects adversarial control. Decoder-only VLMs such as LLaVA and OpenFlamingo incorporate visual features late through cross-attention or token concatenation. Cross-attention lets language processing attend to visual features; concatenation places tokens together in a sequence. These models treat visual features as auxiliary memory during causal language generation, which generates subsequent tokens from preceding context. Visual influence consequently fades as decoding progresses (Awadalla et al., 2023; Liu et al., 2023).

Fusion-based BLIP-2 and InstructBLIP instead integrate visual features early and deeply into the input embeddings. This maintains visual grounding throughout generation (Li et al., 2023b; Dai et al., 2023). Targeted attacks are therefore inherently more feasible in fusion-based models than in decoder-only models (Dai et al., 2023).

Given these architectural differences, non-targeted ASR provides a more robust, fair, and comparable assessment of transfer across models and images.

We exclude CIA from transferability comparisons. In our experiments, it produced negligible transferability. Its method also has two fundamental limitations:

- **Image-Specific Semantic Embedding:** CIA manipulates visual tokens for each image to embed target concepts. The resulting adversarial patterns lack cross-image consistency. Gradient directions and token spaces differ greatly across images and concepts, and the transformations between input-image and target-image token spaces are non-transferable.
- **Model-Specific Embedding Architectures:** CIA depends heavily on visual- and textual-encoder structure. Structural differences among LVLMs such as BLIP2, LLaVA, and Flamingo prevent cross-model transferability. BLIP2 is another spelling of BLIP-2 used here.

## 6 Results

As stated in Section 4, a major objective is to reproduce Luo et al. (2024)’s cross-prompt transferability results for CroPA. We compare our reproduced results with those reported in the original paper.

**Models Used:**

We evaluated five state-of-the-art VLMs: OpenFlamingo, BLIP-2, InstructBLIP, and LLaVA.

For Flamingo, we use the publicly accessible OpenFlamingo-9B implementation (Awadalla et al., 2023). OpenFlamingo-9B is the model identifier for this open-source implementation. It provides performance comparable to the original Flamingo.

BLIP-2 uses a two-stage fusion-based approach. It first extracts image features through a frozen CLIP image encoder, whose weights remain unchanged. A Querying Transformer then processes these features (Li et al., 2023b). This architecture supports efficient adaptation to different vision-language tasks. Its language-model component is OPT-2.7b.

InstructBLIP builds on BLIP-2’s fusion-based architecture and adds instruction tuning, or training to follow instructions (Dai et al., 2023). It uses Vicuna-7b as its language model. This improves task-specific instruction following and permits more precise prompt-based output control.

LLaVA uses decoder-only generation (Liu et al., 2023). A pretrained CLIP ViT-L vision encoder connects to the Vicuna-13B language model through a lightweight projection layer, which maps between representations. ViT-L and Vicuna-13B identify the encoder and language-model variants. LLaVA passes visual features directly into the language model’s context as special tokens in the prompt sequence.

These models represent two architectural approaches:

- **Decoder-only models:** OpenFlamingo and LLaVA treat visual embeddings as generation context.
- **Fusion-based models:** BLIP-2 and InstructBLIP explicitly align visual and textual representations before generation.

These differences significantly affect targeted cross-prompt attack effectiveness.

We use the original authors’ public code and retain the original model configurations for faithful comparison. The implementation was generally well documented. However, the BLIP-2 and InstructBLIP code produced multiple runtime errors. Substantial debugging and adjustments were needed to make it run.

### 6.1 Claim 1: CroPA achieves cross-prompt transferability across various target texts. [Reproduced]

We replicated Luo et al. (2024)’s setup to test CroPA’s central cross-prompt claim. We trained CroPA adversarial examples on Flamingo and measured targeted ASR for general and specific VQA, classification, and captioning.

#### 6.1.1 Experimental Details

Following the original protocol, we evaluate adversarial examples on prompts distinct from those used in training. The specific target prompts are listed in Table 1. These target strings represent a broad range of semantic meanings. Additional details for all reproduced experiments appear in Appendix D.1.

#### 6.1.2 Comparative Analysis

Table 1 reports targeted ASR on Flamingo. The “Target Prompt” column contains the desired output strings. “Overall” is the aggregate ASR reported across tasks.

The results support cross-prompt transferability. CroPA has non-zero ASR for every evaluated task and target text. Its Overall ASR exceeds both baselines for every target: for example, the “unknown” rows report 0.7708 for CroPA, 0.6095 for Multi-P, and 0.2865 for Single-P.

However, CroPA consistently underperforms on Classification and Captioning, especially for abstract texts with little visual meaning. For “not sure,” CroPA’s Classification and Captioning ASRs are 0.2420 and 0.0010. For “too late,” they are 0.7010 and 0.1790 (Table 1, corresponding CroPA rows).

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

**Table 1:** Targeted ASRs tested on Flamingo with different target texts across a variety of vision-language tasks. VQAgeneral and VQAspecific refer to the degree of specificity of prompts pertaining to a particular image. The prompts used are reported in Appendix E.

These experiments reproduce the core finding that CroPA has robust cross-prompt transferability. They thereby validate VLM vulnerability to this form of attack.

### 6.2 Claim 2: CroPA achieves the best overall performance across different number of prompts [Reproduced]

Figure 4 compares the baselines and CroPA as prompt count changes. ASR increases with prompt count for all applicable methods. Single-P is inherently tested on one prompt, so varying its prompt count is redundant and it appears as a lone star symbol.

For the same number of prompts, CroPA consistently achieves higher ASR than the baselines. The comparison uses the same number of attack iterations. This supports the motivation for learning cross-prompt transferability.

We use target text “unknown” to avoid high-frequency responses commonly found in vision-language tasks. We consider this rationale from Luo et al. (2024) valid.

The baseline saturates at an ASR below CroPA as more prompts are added. Thus, adding prompts does not allow the baseline to surpass CroPA. This trend agrees with Luo et al. (2024).

**Figure 4:** Targeted ASR tested with different numbers of prompts on Flamingo. With the same number of attack iterations, our CroPA significantly outperforms baselines. “unknown” is the target text.

### 6.3 Claim 3: CroPA converges towards higher ASR as number of iterations increase [Reproduced]

Our convergence investigation produced results that contrast with the original reported metrics as iteration count increases. This occurred despite using the same training and evaluation parameters and the original authors’ code. Appendix A.2 reports the original authors’ results, where CroPA converges with a higher number of prompts.

Overall ASR across tasks follows the original convergence trend, as Figure 6 shows. However, task-specific ASR is unstable. Figure 5 shows this separately for VQA, VQA-Specific, Classification, and Captioning.

The original authors did not specify which model they used for this convergence evaluation. We therefore report OpenFlamingo results with target text “unknown.”

(a) VQA

(b) VQA-Specific

(c) Classification

(d) Captioning

**Figure 5:** Comparison of Single-P, Multi-P, and CroPA across different categories.

**Figure 6:** Overall Convergence and Comparison of CroPA

### 6.4 Claim 4: CroPA outperforms baselines under few-shot settings [Reproduced]

Flamingo can receive extra images as in-context learning examples to improve task adaptation. Following Luo et al. (2024), we test adversarial images with a different number of these examples at evaluation than during optimization.

We optimize in the **0-shot** setting: no additional in-context images. We then evaluate in both 0-shot and **2-shot**, where two additional images are supplied. In Table 2, “shot” specifies this image count.

| Setting | Method | VQAgeneral | VQAspecific | Classification | Captioning | Overall |
|---|---|---:|---:|---:|---:|---:|
| shot=0 | Multi-P | 0.7240 | 0.8740 | 0.5550 | 0.2850 | 0.6095 |
| shot=0 | CroPA | 0.9680 | 0.9880 | 0.7070 | 0.4200 | 0.7708 |
| shot=2 | Multi-P | 0.7860 | 0.9420 | 0.5630 | 0.0010 | 0.5730 |
| shot=2 | CroPA | 0.9440 | 0.9940 | 0.6860 | 0.0300 | 0.6635 |

**Table 2:** Targeted ASRs with and without visual in-context learning. The shot indicates the number of images added for in-context learning.

In-context examples make attacks inherently harder because they shift generation conditions away from those used during optimization (Luo et al., 2024). In Table 2, CroPA’s Overall ASR falls from 0.7708 in 0-shot to 0.6635 in 2-shot. Multi-P falls from 0.6095 to 0.5730.

Extra images also make the model more aware of the correct prompt space, moving it away from the target-text space. Zhao et al. (2023) and Qi et al. (2023) verify that context-specific attacks lose effectiveness when context changes.

Nevertheless, CroPA performs significantly better than the baseline in most tasks. In 2-shot evaluation, its Overall ASR is 0.6635, compared with Multi-P’s 0.5730. This supports the min-max motivation in Section 4.3: competing prompt updates make the image perturbation more robust to an expanded prompt space and changing context.

### 6.5 Results Beyond the Original Paper

The following experiments extend CroPA with enhancements intended to mitigate its corresponding limitations.

#### 6.5.1 Noise Initialization

Our Noise Initialization via Vision Encoding Optimization method, described in Section 5.1, significantly improves performance across tasks. We denote CroPA with this initialization as **CroPA+Init**.

CroPA+Init consistently outperforms CroPA and Multi-P across all metrics, achieving 6–15% gains in overall accuracy (Table 3). Its Overall ASRs include 0.8977 for “unknown,” compared with CroPA’s 0.7708, and 0.9458 for “I am sorry,” compared with 0.7985.

These results validate semantically guided initialization over random noise. Vision-encoder-anchored perturbations increase ASR and reduce optimization steps, improving efficiency. Figure 7 presents the convergence comparison. Together, the results support a robust and adaptable alternative for attack initialization.

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

**Table 3:** Targeted ASRs on Blip2 with different target texts using CroPA Vision Encoder noise initialization. Blip2 denotes BLIP-2. Detailed hyperparameters are discussed in Section D.2.

##### Theoretical Analysis of Initialization in Bi-level Optimization

CroPA’s min-max framework has distinctive optimization challenges. Three factors explain the gains from diffusion-guided initialization.

First, CroPA resembles a saddle-point problem: one set of variables minimizes the objective while another maximizes it. Initialization strongly affects optimization stability. Zhang et al. (2025) show that min-max problems are substantially more initialization-sensitive than standard minimization and often need carefully designed starting points for stable convergence. This sensitivity is especially relevant to CroPA because prompt perturbations continuously change the optimization landscape—the loss values and update directions encountered during training.

Second, CroPA updates image perturbations more frequently than prompt perturbations. This asymmetric alternating schedule amplifies initialization’s importance. Chen et al. (2019) identify local cycling, where updates repeatedly move around nearby states, in alternating gradient methods for min-max problems. Their convergence depends strongly on whether the initial point is close to adversarially robust regions. Our semantic initialization deliberately starts closer to meaningful adversarial directions.

Third, our results agree with Wu et al. (2020), who show that perturbations initialized with semantic priors transfer better across models than randomly initialized perturbations and achieve substantially higher ASR. Semantic priors are target-related information supplied before optimization. Their effect becomes particularly pronounced in bi-level settings such as CroPA, where optimization paths are more complex.

Our diffusion-based method addresses these challenges by initializing toward a semantic target rather than with standard random noise. It enables faster convergence and improves transferability. Table 3 reports the corresponding ASR increases, and Table 4 compares average targeted ASR with additional baselines.

| Method | VQAgeneral | VQAspecific | Classification | Captioning | Overall |
|---|---:|---:|---:|---:|---:|
| Multi-P | 0.7053 | 0.8807 | 0.7001 | 0.5553 | 0.7104 |
| CroPA | 0.8723 | **0.9457** | 0.7518 | 0.6138 | 0.7958 |
| CIA | 0.2985 | 0.2281 | 0.4857 | 0.4687 | 0.370 |
| CroPA+Init | **0.9145** | 0.9342 | **0.8982** | **0.9464** | **0.9209** |

**Table 4:** Average Targeted ASRs on BLIP-2 for different methods. Results show the impact on Visual Question Answering, both general and specific, Classification, and Captioning. The best values for each task are bold. In addition to the previously mentioned baselines, we compare against the Contextual Injection Attack (CIA), the additional baseline from Yang et al. (2024).

Gradient-based attacks often acquire the semantics of their targets. In theory, this should remove the need for semantically aligned initialization. However, the min-max complexities described above weaken that alignment. Injecting target semantics at initialization is therefore necessary to strengthen it and improve convergence. Figure 7 provides experimental support.

We measure how much target information appears in the learned perturbation using the **CLIP score**, a measure of similarity between the target text and perturbation representation. Figure 7(b) compares base CroPA with our initialization enhancement.

For target text “Bomb,” base CroPA has a CLIP score of 21.80, showing that it acquires target context as expected for a gradient-based attack. CroPA+Init has a modestly higher score of 23.09. This suggests that initialization provides a subtle improvement in proximity to adversarially vulnerable regions, even if it does not significantly upgrade the semantic features acquired relative to the target.

Figure 7(a) also shows faster convergence after initialization. This agrees with our hypothesis that starting closer to meaningful adversarial directions improves optimization.

| Method | CLIP Score |
|---|---:|
| CroPA | 21.80 |
| CroPA+Init | 23.09 |

Target Text: “Bomb”

(a) Overall convergence comparison

(b) CLIP-score comparison

**Figure 7:** (a) Overall Convergence Comparison of base CroPA and CroPA+Init. (b) Quantifying feature similarity of learned perturbation to target by CLIP score.

#### 6.5.2 Investigating Cross-Image transferability and Image Augmentation

CroPA retains adversarial impact across text prompts. However, its optimization remains tightly coupled to the low-level features of the image used to generate the perturbation. This dependency prevents effective generalization to other images and makes universal cross-image attacks ineffective. It weakens the original authors’ claim that learning perturbations across images would yield cross-image transferability.

We integrate SCMix and CutMix, described in Section 5.3, to introduce image-feature diversity during training. The aim is to make perturbations generalize beyond one image.

We believe CutMix outperforms SCMix in promoting transferability because its region-based mixing preserves distinct, non-overlapping spatial areas. This encourages modular, independently manipulable features and more generalizable perturbations.

Table 5 shows measurable improvements from both methods. Overall untargeted ASR is 0.4355 for base cross-image CroPA, 0.7055 with SCMix, and 0.5565 with CutMix. Nevertheless, cross-image transfer remains difficult. The performance gap between these cross-image methods and CroPA applied to one image shows a fundamental limitation: CroPA excels at cross-prompt robustness but does not effectively support transferable attacks across images.

| Method | VQA | VQAspecific | Classification | Captioning | Overall |
|---|---:|---:|---:|---:|---:|
| CroPA (Base) | 0.3500 | 0.4920 | 0.4000 | 0.5000 | 0.4355 |
| CroPA with SCMix | 0.6560 | 0.7160 | 0.5000 | 0.9500 | 0.7055 |
| CroPA with CutMix | 0.4520 | 0.7240 | 0.5000 | 0.5500 | 0.5565 |

**Table 5:** Untargeted ASR of CroPA in cross-image settings with and without SCMix and CutMix. Values are the best ASRs across all iterations. CroPA without input mixing reaches its highest ASRs at 1700 iterations. SCMix and CutMix outperform it earlier, at 1500 and 1600 iterations, respectively. Additional experimental parameters are reported in Appendix D.3.

#### 6.5.3 Guiding perturbations via Target Value Vectors

Tables 6 and 7 evaluate the D-UAP-modified CroPA framework from Section 5.2, denoted **CroPA+D-UAP**.

The enhanced perturbations improve the existing algorithm across tasks. On BLIP-2, Overall ASR reaches 96.85%, a 12.5% increase over base CroPA. Table 6 reports Overall values of 0.9685 and 0.8433, respectively. VQAgeneral and Captioning improve by 13% and 28%, respectively; the corresponding rows report 0.8176 to 0.9420 and 0.6960 to 0.9810.

These experiments use target text “Bomb.” This is harmful text that is inherently expected to trigger guardrails, the model’s safety controls. The results therefore show improved effectiveness even in vulnerable scenarios involving these difficult harmful-word targets.

| Method | VQAgeneral | VQAspecific | Classification | Captioning | Overall |
|---|---:|---:|---:|---:|---:|
| Multi-P | 0.68 | 0.82 | 0.90 | 0.60 | 0.75 |
| CroPA | 0.8176 | 0.9100 | 0.9498 | 0.6960 | 0.8433 |
| CIA | 0.3431 | 0.3102 | 0.4732 | 0.5534 | 0.4199 |
| CroPA+D-UAP | 0.9420 | 0.9720 | 0.9790 | 0.9810 | 0.9685 |

**Table 6:** Targeted ASRs on Blip2 using the D-UAP function with target text “Bomb,” outperforming PGD baselines, CIA, and base CroPA.

| Method | VQAgeneral | VQAspecific | Classification | Captioning | Overall |
|---|---:|---:|---:|---:|---:|
| Multi-P | 0.6960 | 0.8540 | 0.8990 | 0.6060 | 0.7638 |
| CroPA | 0.7900 | 0.8860 | 0.9470 | 0.6700 | 0.8233 |
| CIA | 0.3027 | 0.4302 | 0.5112 | 0.5080 | 0.4380 |
| CroPA+D-UAP | 0.9000 | 0.9520 | 0.9500 | 0.9060 | 0.9270 |

**Table 7:** Targeted ASRs on Open Flamingo using the D-UAP loss function with target text “Bomb.”

#### 6.5.4 Analyzing Cross Model Transferability by Value Vector guidance

Table 8 also shows improved cross-model transferability over vanilla, or unmodified, CroPA. The gains are particularly strong for Classification and Captioning, at 14% and 21%, respectively, with more modest VQA improvements.

We attribute the task differences to VQA prompts spanning a more complex embedding space. Classification and Captioning have outputs limited to much smaller spans.

A key limitation concerns the vision encoder. Value vectors depend on the specific encoder used by the VLM. Transfer can therefore be limited to models with the same or similar encoder architecture, as hypothesized in Section 5.2.

Table 8 verifies this pattern. Perturbations trained on BLIP2 transfer passably to InstructBLIP, which uses the same vision encoder, but fail to transfer to OpenFlamingo. For BLIP2 to InstructBLIP, CroPA+D-UAP has Overall ASR 0.6850, compared with CroPA’s 0.5065. For BLIP2 to Flamingo, the corresponding values are 0.2335 and 0.1435.

The reverse pattern is also expected: perturbations trained on Flamingo have insignificant ASR on the other models. Table 8 reports CroPA+D-UAP Overall values of 0.1013 for Flamingo to BLIP2, 0.0938 for Flamingo to InstructBLIP, and 0.1025 for Flamingo to LLaVA. See also Table D.4.5 in Appendix D.4.5. Experimental details appear in Appendix D.4.

In Table 8, **Cls** means Classification and **Cap** means Captioning. A setting such as “BLIP2 to InstructBLIP” identifies the optimization model first and the evaluation model second.

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

**Table 8:** Cross-model tests under different settings. “BLIP2 to InstructBLIP” means that perturbations optimized on BLIP2 are tested on InstructBLIP. “BLIP2 to Flamingo” follows the same convention.

We also find that, for a given target text, the vision encoder’s value vectors remain approximately consistent across different input images. This has three implications:

1. **Vision Encoder Specificity:** The vectors align with the encoder’s architecture and pretraining.
2. **Limited Diversity in Perturbations:** Consistent vectors may constrain perturbation diversity and limit the ability to generate highly varied adversarial examples.
3. **Target Text Dependency:** Similar vectors across images for one target indicate that outcomes depend strongly on the target text rather than on specific input-image features.

## 7 Discussion

### 7.1 Observations with reproducibility

Our study replicates key aspects of Luo et al. (2024) and substantiates their claims about CroPA’s improved cross-prompt adversarial transferability.

CroPA achieves strong transferability across VQA, Classification, and Captioning and consistently outperforms the baselines (Table 1 and Figures 4–6). It also remains robust in few-shot settings. Section 6.4 and Table 2 show that CroPA outperforms the baseline despite the shift between optimization and evaluation conditions introduced by in-context examples.

### 7.2 Noise Initialization findings

Section 5.1 identifies a critical limitation of CroPA and related methods: random initialization does not account for the target prompt’s semantic structure. We hypothesize that this lack of guidance causes instability, prolongs convergence, and reduces cross-prompt and cross-modal transferability, meaning transfer involving the image and text modalities.

Our proposed initialization uses a diffusion-synthesized reference image and alignment in the vision encoder’s feature space. This orients perturbations toward a meaningful adversarial direction from the start.

Section 6.5.1 empirically validates the hypothesis. Across all evaluated benchmarks, our initialization significantly outperforms CroPA and random-initialized baselines. It yields up to 15% gains in targeted ASR and notably reduces optimization steps (Tables 3–4 and Figure 7).

Semantically anchored initialization therefore improves both attack efficacy and efficiency. It provides a more principled and robust foundation for VLM attack frameworks than random noise.

### 7.3 Evaluation of Transferability Experiments

#### 7.3.1 Cross-Image Transferability

The original authors suggest that computing perturbations across images can yield cross-image transferability. Our experiments instead show that CroPA’s perturbations remain tightly linked to the low-level features of the image on which they were generated.

We integrate SCMix and CutMix to increase cross-image diversity while preserving semantic information (Section 5.3). Both methods improve transferability (Section 6.5.2 and Table 5).

SCMix produces significant gains, especially for VQA, VQA-specific, and Captioning. Overall ASR increases by approximately 27 percent: Table 5 reports 0.4355 for base CroPA and 0.7055 with SCMix. CutMix gives a net gain of around 12 percent, reaching 0.5565.

We attribute SCMix’s advantage to its explicit spatial and scale invariance. It randomly crops and resizes one image, then softly blends it with another. This encourages perturbations to target semantically aligned features that depend less on exact position or scale.

Both augmentations add cross-image features and improve generalization when learning perturbations. Future work should explore more complex, layered approaches beyond augmentation. Possibilities include combining augmentation with feature-alignment attacks on latent representations—the internal encoded forms of images—or directly perturbing that latent space.

#### 7.3.2 Cross-Model Transferability

The D-UAP-modified CroPA framework is a significant improvement over CroPA’s attack method (Section 5.2). It achieves 96.85% targeted ASR, a 13.5% improvement over the base method, as shown in Section 6.5.3. Table 6 reports Overall ASRs of 0.9685 for CroPA+D-UAP and 0.8433 for CroPA.

The gains are particularly notable for VQAgeneral and Captioning, at 13% and 28%, respectively. They occur even for challenging harmful text such as “Bomb,” which typically triggers guardrails. This demonstrates the framework’s robustness in bypassing safety mechanisms (Tables 6–7).

Cross-model transfer also improves, especially between models with similar encoders, such as BLIP2 and InstructBLIP (Table 8). However, dependence on the specific vision encoder limits perturbation diversity and transfer to different architectures.

This limitation highlights the need for encoder-agnostic attacks: methods that do not depend on one particular vision encoder. Nevertheless, D-UAP is a critical step toward universal perturbations effective across both prompts and images. It opens directions for improved generalizability and robustness.

Our cross-model comparisons exclude the Contextual Injection Attack. CIA is inherently non-transferable and could not produce comparable results.

### 7.4 Trade-offs Between Different Types of Adversarial Transferability

Our analysis reveals fundamental trade-offs among transferability dimensions in VLMs. These trade-offs matter when designing robust adversarial attacks.

#### 7.4.1 Cross-Prompt vs. Cross-Model Transferability

CroPA achieves strong cross-prompt transfer by optimizing against competing text embeddings. However, this process implicitly anchors perturbations to the target model’s architecture. Section A.1 confirms that ASR drops significantly when perturbations move from one model to another.

Our D-UAP enhancement targets generalizable vision-encoder components rather than model-specific features (Section 5.2). This improves transfer between similar architectures. For BLIP2 to InstructBLIP, Table 8 reports Overall ASR 0.5065 for CroPA and 0.6850 for CroPA + D-UAP.

The gain remains modest between fundamentally different architectures. For BLIP2 to Flamingo, Table 8 reports 0.1435 and 0.2335, respectively. This suggests an inherent tension between optimizing for prompt diversity and model diversity.

#### 7.4.2 Cross-Prompt vs. Cross-Image Transferability

Cross-prompt and cross-image transferability are competing optimization objectives. CroPA achieves strong prompt transfer by tightly coupling perturbations to image-specific features. This coupling fundamentally limits generalization to other images.

We use augmentation to diversify optimization (Section 5.3). SCMix and CutMix produce promising gains in Table 5: Overall ASRs of 0.7055 and 0.5565, compared with 0.4355 for base cross-image CroPA.

However, augmentation does not fully resolve the fundamental tension between these objectives.

#### 7.4.3 Role of Initialization in Transferability Balance

Diffusion-based initialization offers another perspective on these trade-offs (Section 5.1). Semantically meaningful starting points produce higher ASRs without sacrificing cross-prompt transferability (Tables 3–4 and Figure 7). The starting point therefore significantly affects the achievable balance among transferability types.

Unlike methods that change the optimization process itself, better initialization guides convergence toward more general adversarial directions that remain effective across prompts.

Nevertheless, improved initialization does not remove the fundamental tension between cross-prompt and cross-image or cross-model transferability. Optimizing a perturbation for one dimension inevitably sacrifices effectiveness in others.

#### 7.4.4 Implications for Future Research

No single attack method can simultaneously maximize every form of transferability. Future frameworks should allow controlled balance among these objectives. Possible approaches include multi-objective optimization, which jointly considers several goals, and architecture-aware attack designs.

Individual techniques can improve particular transferability dimensions. A comprehensive solution, however, requires reconsidering how perturbations interact with model architectures and data representations.

Cross-prompt adversarial transferability remains underexplored. The limited literature leaves an important gap in understanding prompt-induced vulnerabilities. Addressing this gap is imperative for secure VLM deployment.

### 7.5 Broader Impact

From an attacker’s perspective, highly cross-prompt-transferable examples pose significant threats. They can cause malicious or misleading outputs even when the textual query is harmless. They could be used to spread false information or subvert VLM-based content-generation and decision-making systems.

From a defensive perspective, imperceptible perturbations could help protect sensitive information. Embedding them in personal images could make VLMs consistently output predetermined, non-sensitive text. This could obstruct unauthorized extraction of confidential image content and provide proactive privacy protection.

Our initialization and loss refinements substantially increase ASR and generalizability across models and images (Tables 3–8). These improvements reinforce the effectiveness of cross-prompt attacks and support the development of more resilient defenses.

Our reproducibility study affirms Luo et al. (2024)’s foundational work. It also highlights the need for deeper study of cross-prompt transferability. Understanding this vulnerability is crucial for robust VLMs, effective countermeasures, user-data protection, and trustworthy model outputs.

### 7.6 Limitations and Future Work

We reproduced CroPA’s key results and conducted extensive ablations, or experiments varying components of the method. However, limited computational resources repeatedly hindered progress.

Training and evaluating attacks on large VLMs such as Flamingo and BLIP2 require substantial GPU memory and processing power. Resource availability limited the scale and depth of some experiments. It also constrained extensive hyperparameter tuning and cross-model evaluation across a wider range of architectures.

We intended to investigate transferability using an ensemble, a group of models considered together. This could provide deeper insight into perturbation generalizability. Its memory and computation requirements made it infeasible with our infrastructure.

Future work could develop encoder-agnostic methods to improve cross-model transfer and further strengthen cross-image generalization. Query-based optimization, which uses model responses to guide updates, could also improve practical applicability.

### 7.7 Communication With Original Authors

We could not establish direct communication with the original authors during replication. The issue raised in their GitHub repository concerning BLIP-2 and InstructBLIP remains unresolved at the time of writing.
