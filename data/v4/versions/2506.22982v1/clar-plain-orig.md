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

Large Vision-Language Models (VLMs), which process images and text, have transformed computer vision. They support tasks such as image classification, captioning, and visual question answering. However, they remain highly vulnerable to adversarial attacks: changes to inputs that cause incorrect or harmful outputs. This risk is especially high when attackers can change both images and text.

We conduct a comprehensive reproducibility study of “An Image is Worth 1000 Lies: Adversarial Transferability Across Prompts on Vision-Language Models.” We validate its Cross-Prompt Attack (CroPA). We also confirm that CroPA transfers across prompts better than existing baselines. Cross-prompt transferability means that an adversarial image remains effective when the textual prompt changes.

Beyond replication, we propose several improvements: (1) a new initialization strategy that significantly improves Attack Success Rate (ASR), the proportion of successful attacks; (2) an investigation of cross-image transferability through universal perturbations, or shared input changes that work across images; and (3) a new loss function that targets attention mechanisms in the vision encoder to improve generalization. A loss function measures the objective being optimized. The vision encoder converts an image into internal feature representations, and attention controls interactions among those representations.

Our evaluation covers Flamingo, BLIP-2, and InstructBLIP, with extended experiments on LLaVA. It validates the original results and shows that our improvements consistently increase attack effectiveness. Our work reinforces the need to study VLM vulnerabilities. It also provides a more robust framework for generating transferable adversarial examples, with significant implications for VLM security in real-world applications.

Our code is available at https://github.com/Swadesh06/Revisting_CroPA

Reviewed on OpenReview: https://openreview.net/forum?id=5L90cl0xtf

## 1 Introduction

Large Vision-Language Models (VLMs) have transformed computer vision by supporting many tasks. These include image classification, captioning, and visual question answering. Their flexibility has encouraged deeper study of interactions between images and language.

However, recent studies show that VLMs remain highly vulnerable to adversarial attacks: Zhao et al. (2023); Qi et al. (2023); Zhang et al. (2022a); Carlini et al. (2024). These attacks add subtle perturbations, or changes, to input images. The changes cause VLMs to produce incorrect or even harmful outputs. Text inputs create further ways to attack these models, beyond those available against traditional vision models.

Several studies have examined adversarial robustness, meaning a model’s ability to withstand adversarial inputs. Zhao et al. (2023), for example, studied VLMs such as BLIP and BLIP-2 in black-box settings, where attackers cannot access the model’s internal parameters. They examined query-based attacks, which use model responses, and transfer-based attacks, which move attacks from another model. Schlarmann & Hein (2023) studied targeted and untargeted attacks in white-box settings, where attackers can access model internals. Targeted attacks seek a specified output. Untargeted attacks seek an incorrect output without requiring a particular one.

These works mainly studied attacks on images. Later research also examined adversarial changes to text inputs. Qi et al. (2023) showed that adversarial images could cause VLMs to execute harmful instructions. Tu et al. (2023) systematically evaluated attacks on both images and text.

Researchers have traditionally divided the generalization and transferability of adversarial examples in VLMs into two main categories:

- **Cross-Model Transferability:** Adversarial examples remain effective across different VLM architectures. Researchers commonly call this transferability.
- **Cross-Image Transferability:** Adversarial perturbations work across multiple images. Researchers often call these Universal Adversarial Perturbations (UAPs).

Luo et al. (2024) introduced **Cross-Prompt Transferability**: adversarial images remain effective across different textual prompts. Earlier work treated adversarial changes to images and text independently. In contrast, Luo et al. (2024) proposed the Cross-Prompt Attack (CroPA), which uses learnable prompts to keep adversarial images effective regardless of the text input. Learnable prompts are prompt representations that the attack updates during optimization. Their experiments showed that CroPA works across image classification, captioning, and visual question answering.

Our work has the following goals:

- **[Reproducibility Study] Reproducing the Results from the Original Paper:** We successfully reproduced and verified the original paper’s four main claims. First, CroPA transfers across prompts better than established baselines across several target texts. This transferability does not depend on the target text’s meaning or word frequency. Second, adding prompts improves transferability, but performance converges quickly. Even with more prompts, baseline methods do not surpass CroPA. Third, CroPA remains effective when the model receives additional in-context learning examples—examples supplied in the input to help it adapt to a task. Fourth, we reproduce CroPA’s convergence results by measuring overall ASR as attack iterations increase. We also analyze ASR for individual tasks.
- **[Extended Work] Better Initialization Strategy:** We propose a new initialization method that substantially increases ASR and transferability.
- **[Extended Work] Investigating Cross Image Transferability and Image Augmentation:** We use CroPA’s underlying framework to learn a shared perturbation for all images. We also test whether enhancements designed to promote transferability help.
- **[Extended Work] Guiding perturbations via Target Value Vectors:** We propose a new loss function based on components of the vision encoder’s attention mechanism. These components control interactions among image patches, the image regions processed by the encoder. Targeted changes to value vectors—the information carried by patches within attention—lead to greater generalization and ASR.

We also analyze the mechanisms behind our improvements to better understand VLM vulnerabilities. Our work reinforces the need to study these vulnerabilities and provides a more robust and flexible framework for generating transferable adversarial examples.

## 2 Related Works

Researchers have studied adversarial attacks on deep neural networks extensively since Szegedy et al. (2013) and Goodfellow et al. (2014) introduced them. Applying these attacks to Vision-Language Models (VLMs) creates distinct challenges and opportunities. This section reviews research on adversarial transferability and places CroPA in that context.

### 2.1 Adversarial Transferability

Adversarial attacks can transfer in several ways. Liu et al. (2017) and Tramèr et al. (2017) showed that adversarial examples made for one model can often deceive other models, even when their architectures differ. Researchers also found that perturbations can generalize across images, enabling Universal Adversarial Perturbations (UAPs): Moosavi-Dezfooli et al. (2017); Mopuri et al. (2017).

Attacks can also transfer across tasks. Adversarial examples designed for classification can affect object detection and other vision tasks: Naseer et al. (2019b); Naseer et al. (2019a); Lu et al. (2020); Salzmann et al. (2021). These forms of transferability raise important questions about the robustness of modern machine learning systems.

### 2.2 Adversarial Attacks on Vision-Language Models

Early research on the adversarial robustness of vision-language models focused on attacks for specific tasks. Xu et al. (2019); Zhang et al. (2020); Aafaq et al. (2021); Chen et al. (2017) showed how to manipulate image-captioning outputs. Visual Question Answering (VQA), in which models answer questions about images, also proved vulnerable to targeted attacks that mislead attention mechanisms: Xu et al. (2018); Kaushik et al. (2021); Kovatchev et al. (2022); Li et al. (2021); Sheng et al. (2021); Zhang et al. (2022b).

However, these approaches mainly targeted lightweight CNN-RNN architectures, which combine convolutional neural networks and recurrent neural networks. Those architectures lacked in-context learning, limiting the attacks’ relevance to current VLMs. More recently, Zhao et al. (2023) studied the robustness of large VLMs such as BLIP and BLIP-2 in black-box settings. Their work focused on cross-model transferability rather than CroPA’s cross-prompt perspective.

### 2.3 Cross-Prompt Adversarial Transferability

Large VLMs that adapt to tasks through prompts introduce a new kind of adversarial vulnerability. Traditional vision models perform specific tasks. VLMs can instead change tasks in response to text prompts: Li et al. (2023b); Awadalla et al. (2023); Li et al. (2023b); Zhu et al. (2023); Liu et al. (2023). This flexibility creates a new attack surface, or set of possible attack points, where adversarial examples must remain effective as prompts change.

CroPA pioneered this concept by using learnable prompts that compete with image perturbations during optimization (Luo et al. (2024)). Unlike methods that treat image and text perturbations independently, CroPA offers a new way to evaluate VLM robustness.

Because this concept is relatively new, researchers have not yet established credible benchmarks or baselines. However, Yang et al. (2024) study cross-prompt transferability through the Context Injection Attack (CIA). CIA uses gradient-based perturbations—changes guided by how the loss responds to an input—to insert target tokens into both image and text contexts. Tokens are the units that models process. We include CIA as an additional comparison in our experiments.

### 2.4 Enhancing Adversarial Transferability

Researchers have proposed several ways to improve adversarial transferability. Input diversity methods (Xie et al. (2019)) and data augmentation strategies (Wang et al. (2021)), which create varied training inputs, have shown promise for producing more robust attacks.

Recent work identifies gradient misalignment—differences between gradient directions—as a key limit on transferability: Li et al. (2023a); Wang et al. (2023). This finding has led to new optimization strategies. We build on these insights to improve CroPA across several forms of transferability.

## 3 Scope of reproducibility

This study examines and validates the results of Luo et al. (2024). Our main objective is to confirm that CroPA significantly improves the transferability of adversarial examples across prompts by carefully reproducing their experiments.

We aim to verify these main claims:

- **Claim 1:** CroPA achieves cross-prompt transferability across various target texts.
- **Claim 2:** CroPA achieves the best overall performance across different numbers of prompts.
- **Claim 3:** CroPA converges toward higher ASR as the number of iterations increases.
- **Claim 4:** CroPA outperforms baselines in few-shot settings, where the input supplies a small number of task examples.

Beyond replication, we test CroPA in cross-model and cross-image settings. We also compare it with other cross-prompt transferability methods. Specifically, we assess whether images generated through CroPA consistently deceive different VLMs, regardless of the input prompt or the model’s specific parameters.

These tests let us reproduce the original experiments and explore CroPA’s broader use in making adversarial attacks on VLMs more robust and flexible.

## 4 Methodology

### 4.1 Problem Formulation

Researchers have documented neural networks’ vulnerability to adversarial attacks since the foundational work of Goodfellow et al. (2014). We examine the authors’ formulation, which extends these ideas to cross-prompt attacks on VLMs. It explains how adversarial perturbations can remain effective across different text inputs (Luo et al. (2024)).

The authors represent a VLM as a function \(f\). It processes an image input \(x_v\) and a text input \(x_t\). To support real-world use, they require the image perturbation \(\delta_v\) to remain within human-imperceptible bounds:

\[
\|\delta_v\|_p \leq \epsilon_v.
\]

Here, the \(p\)-norm measures perturbation size, and \(\epsilon_v\) is its allowed limit. This constraint follows established adversarial machine learning practice while adapting it to VLMs’ image and text inputs (Carlini et al. (2024)).

The authors define two attack settings, both of which we reproduce:

1. **Targeted attacks:** The attack seeks a predetermined text \(T\), regardless of the input prompt. Equation 1 minimizes the language modeling loss \(L\), which measures how well the output matches the desired text, across \(k\) prompt instances \(x_t^i\).
2. **Non-targeted attacks:** The attack maximizes the difference between outputs for clean and adversarial inputs, as Equation 2 shows. A clean input is one without the adversarial perturbation.

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

We measure effectiveness through Attack Success Rate (ASR). A targeted attack succeeds only if the model generates the exact target text. A non-targeted attack succeeds if the output differs in any way from the clean image’s prediction.

### 4.2 Baseline Approaches

Cross-prompt transferability is a new direction in adversarial machine learning, so the literature has no established baselines. We therefore examine two baselines from Luo et al. (2024): Single-P and Multi-P.

Single-P optimizes an image perturbation with one prompt. For targeted attacks, its objective is:

\[
\min_{\delta_v}L\bigl(f(x_v+\delta_v,x_t),T\bigr)
\tag{3}
\]

**Figure 1:** Overview of the CroPA framework presented in Luo et al. (2024). The architecture uses learnable perturbations for both image inputs (\(\delta_v\)) and prompt inputs (\(\delta_t\)). These perturbations oppose each other: \(\delta_v\) minimizes the language modeling loss, while \(\delta_t\) maximizes it. The forward pass, shown by solid arrows, processes the perturbed inputs through the vision encoder and language model. Backpropagation, shown by dashed arrows, updates the perturbations at adjustable frequencies. Backpropagation computes gradients through the model. Model parameters, marked with *, remain fixed during the attack. The symbol \(\oplus\) means element-wise addition.

Multi-P extends Single-P by using several prompts during optimization. For a prompt collection \(X_t=\{x_t^1,x_t^2,\ldots,x_t^k\}\), its objective is:

\[
\min_{\delta_v}\sum_{i=1}^{k}
L\bigl(f(x_v+\delta_v,x_t^i),T\bigr)
\tag{4}
\]

For non-targeted attacks, these objectives reverse: they maximize the difference between the model’s outputs for clean and perturbed inputs. Both baselines provide reference points for evaluating CroPA’s improved cross-prompt transferability.

### 4.3 Cross Prompt Attack

The Cross-Prompt Attack (CroPA) uses learnable prompts during optimization to improve cross-prompt transferability (Luo et al. (2024)). Its key innovation is that prompt perturbations compete with image perturbations rather than help them deceive the model.

CroPA optimizes the image perturbation \(\delta_v\) and prompt perturbation \(\delta_t\) with opposing objectives. The image perturbation minimizes the language modeling loss for the target text. The prompt perturbation maximizes that loss. This competition forces the image perturbation to transfer more strongly across prompts.

The resulting min-max problem, which minimizes over one variable while maximizing over another, is:

\[
\min_{\delta_v}\max_{\delta_t}
L\bigl(f(x_v+\delta_v,x_t+\delta_t),T\bigr)
\tag{5}
\]

Here, \(f\) is the VLM, \(T\) is the target text, and \(L\) is the language modeling loss.

The implementation uses Projected Gradient Descent (PGD), an iterative method that updates inputs using gradients and projects them back within the allowed perturbation bounds (Madry et al. (2017)). Image updates use gradient descent to reduce the loss. Prompt updates use gradient ascent to increase it.

A parameter \(N\) controls the update frequency: the method updates the image perturbation \(N\) times for each prompt perturbation update. Figure 1 shows the framework. We reproduce the algorithm from Luo et al. (2024) below.

**Algorithm 1 CroPA: Cross Prompt Attack**

**0: Inputs:** Model \(f\), target text \(T\), vision input \(x_v\), prompt set \(X_t\), perturbation size \(\epsilon\), step sizes \(\alpha_1\) and \(\alpha_2\), number of iteration steps \(K\), and adversarial prompt update interval \(N\).

**0: Output:** Adversarial example \(x'_v\).

1. Initialize \(x'_v=x_v\).
2. **For** step = 1 to \(K\):
3.  Sample prompt \(x_t^i\) uniformly from \(X_t\).
4.  **If** \(x_t^i\) is not initialized:
5.   Initialize \(x_t^{\prime i}=x_t^i\).
6.  **End if.**
7.  Compute the adversarial image gradient:
   \[
   g_v=\nabla_{x_v}L(f(x'_v,x_t^i),T).
   \]
8.  Update by gradient descent:
   \[
   x'_v=x'_v-\alpha_1\cdot\operatorname{sign}(g_v).
   \]
9.  **If** \(\operatorname{mod}(\text{step},N)=0\):
10.   Compute the adversarial prompt gradient:
    \[
    g_t=\nabla_{x_t}L(f(x'_v,x_t^i),T).
    \]
11.   Update by gradient ascent:
    \[
    x_t^{\prime i}=x_t^{\prime i}+\alpha_2\cdot\operatorname{sign}(g_t).
    \]
12.  **End if.**
13.  Project \(x'_v\) into the \(\epsilon\)-ball around \(x_v\):
    \[
    x'_v=\operatorname{Clip}_{x_v,\epsilon}(x'_v).
    \]
14. **End for.**
15. **Return** \(x'_v\).

The symbol \(\nabla\) denotes a gradient, \(\operatorname{sign}\) takes each gradient element’s sign, and \(\operatorname{mod}\) gives the remainder after division. The \(\epsilon\)-ball contains inputs within the allowed distance from the original image.

At evaluation, the attack uses only the optimized image perturbation and discards prompt perturbations. Its effectiveness therefore comes from the image perturbation’s transferability, not from changes to evaluation prompts.

## 5 Additional Investigative Experiments

CroPA’s iterative PGD optimization effectively increases ASR on a single model. However, it depends on feedback from that model to minimize its loss, which limits transferability. Researchers have identified this limit in other gradient-based perturbation methods and proposed enhancements: Qin et al. (2024); Li et al. (2024); Wang et al. (2023); Li et al. (2023a). The original authors also recognize this limitation, which Table 9 in Appendix A.1 verifies. Table 1 further shows that CroPA performs especially poorly on some tasks, including Classification and Captioning.

CroPA also requires substantial computation. Training adversarial perturbations for just 50 images takes over 6 hours on a 48GB L40S graphics processing unit (GPU), using 95% of its video memory (VRAM).

These limitations motivate our extensions to CroPA. We focus on: (i) improving its inconsistent performance across tasks; and (ii) testing the transferability types described in Section 1 and adding methods that improve image-perturbation transferability.

Transferability was not a core proposal of Luo et al. (2024), but exploring it may also reduce computation costs. Ideally, cross-image transferable perturbations would avoid the need to train a separate perturbation for each image. Cross-model transferable perturbations could offer even greater efficiency.

The following subsections explain our experiments, their motivations, and their mechanisms.

### 5.1 Noise Initialization via Vision Encoding Optimization

A core limitation of CroPA is its random noise initialization, a strategy shared by many gradient-based attacks. CroPA iteratively optimizes this noise, but the starting point does not reflect the target’s meaning. This causes instability and slower convergence. The problem is especially important for VLMs, where alignment with prompt meaning is crucial.

Our evaluation of CroPA’s bi-level optimization framework, which couples two optimization objectives, shows that effectiveness depends strongly on initialization. Section 6.5.1 examines this dependence, and Figure 7 and Table 3 provide empirical evidence. CroPA cannot guide optimization toward semantically meaningful directions from the start. This highlights the need for a more principled initialization strategy. Conventional initialization also fails to establish meaningful adversarial directions at the outset, reducing ASR and cross-prompt transferability.

We hypothesize that initialization informed by meaning, rather than purely random noise, can significantly improve attack effectiveness while keeping perturbations imperceptible. We propose using a diffusion model, an image-generation model, to synthesize an image that matches a desired prompt. We then use the vision encoder’s feature space—the space of its internal image representations—to align the initial perturbation with that image.

Our Noise Initialization via Vision Encoding Optimization method combines this semantic anchoring with adversarial optimization through three steps:

- Generate a semantically relevant target image with a state-of-the-art diffusion model, such as Stable Diffusion XL.
- Align the perturbation with the target image’s vision encoder representation so that the initial noise follows the adversarial objective.
- Optimize perturbation updates with a Mean Squared Error (MSE) loss, which measures squared differences between representations, to enforce semantic similarity while preserving attack effectiveness.

Replacing random initialization with this vision-guided approach improves adversarial robustness and attack efficiency. It reduces the number of optimization steps needed for convergence. The method offers a principled alternative that produces stronger perturbations and improves cross-prompt transferability. It ensures that perturbations reflect the target’s meaning from the start.

#### 5.1.1 Noise Initialization

Preliminary experiments on BLIP-2 and InstructBLIP suggest that semantically informed noise significantly improves attack effectiveness over random initialization. It adds only 20-25 seconds of computation per image on a single GPU.

The diffusion-based method improves alignment in the vision encoder’s feature space. It also preserves the visual fidelity of adversarial examples while meeting strict perturbation budgets, or limits on perturbation size.

#### 5.1.2 Diffusion-Based Target Synthesis

For a target prompt \(T\), we first generate an image \(\mathbf{x}_{\text{target}}\) that represents the attributes described by \(T\). We use the diffusion model Stable Diffusion XL (SDXL):

\[
\mathbf{x}_{\text{target}}
=\mathcal{D}(T,\mathbf{z};\theta_{\text{SDXL}}),
\tag{6}
\]

where \(\mathcal{D}\) is the diffusion process, \(\mathbf{z}\) is sampled from a Gaussian distribution, and \(\theta_{\text{SDXL}}\) contains the diffusion model’s pretrained weights. The generated image captures the prompt’s meaning and provides an informative basis for initializing the perturbation.

#### 5.1.3 Vision-Encoder Anchored Perturbation

Let \(f_v(\cdot)\) denote the VLM’s vision encoder. We seek a perturbation \(\boldsymbol{\delta}\) that makes the perturbed image \(\mathbf{x}+\boldsymbol{\delta}\) resemble the target image in the encoder’s output space. We obtain the initial perturbation by minimizing:

\[
\boldsymbol{\delta}_{\text{init}}
=
\underset{\|\boldsymbol{\delta}\|_{\infty}\leq\epsilon}{\arg\min}
\left\|
f_v(\mathbf{x}+\boldsymbol{\delta})
-f_v(\mathbf{x}_{\text{target}})
\right\|_2^2,
\tag{7}
\]

where \(\epsilon\) is the largest allowed perturbation. The \(\ell_\infty\) constraint limits the largest absolute change to any input element and ensures imperceptibility. The squared \(\ell_2\) norm measures the squared difference between encoder outputs, and \(\arg\min\) selects the perturbation that minimizes it.

This initialization places the adversarial example in a semantically meaningful neighborhood of the target prompt’s representation. Algorithm 2 in Appendix B formally describes the method.

#### 5.1.4 Adversarial Optimization via PGD

After initialization, we refine the adversarial example with projected gradient descent (PGD). At iteration \(t\), the update is:

\[
\mathbf{x}_{\text{adv}}^{(t+1)}
=
\Pi_{\mathcal{B}_{\epsilon}(\mathbf{x})}
\Bigl[
\mathbf{x}_{\text{adv}}^{(t)}
-\alpha
\nabla_{\mathbf{x}_{\text{adv}}^{(t)}}
\mathcal{L}_{\text{MSE}}
\Bigr],
\tag{8}
\]

where \(\alpha\) is the step size. The projection \(\Pi_{\mathcal{B}_{\epsilon}(\mathbf{x})}\) returns the updated input to the allowed \(\ell_\infty\)-ball around the original image \(\mathbf{x}\).

The optimization uses a simple mean squared error (MSE) between vision encoder outputs:

\[
\mathcal{L}_{\text{MSE}}
=
\left\|
f_v(\mathbf{x}_{\text{adv}})
-f_v(\mathbf{x}_{\text{target}})
\right\|_2^2.
\tag{9}
\]

This loss makes each update move the adversarial example closer to the target’s semantic embedding, or internal representation of meaning.

### 5.2 Guiding perturbations via Target Value Vectors and evaluating Cross-Model Tranferability

CroPA improves attack effectiveness by optimizing image and text inputs through PGD. However, it does not account for architectural details, especially the internal behavior of the vision encoder, which plays a dominant role in connecting VLM outputs to visual meaning.

CroPA treats image and language inputs as equally vulnerable rather than exploiting weaknesses specific to each. This limits its ability to generate transferable, structured perturbations. More importantly, its perturbations do not target the encoder components that form visual representations. This weakens both attacks and cross-model transferability. CroPA instead minimizes image-perturbation loss and maximizes text-perturbation loss to push the output toward a target.

Cui et al. (2023) and Wang et al. (2024) show that perturbing the vision encoder’s output embeddings can significantly impair the visual perception of large vision-language models (LVLMs). This, in turn, misleads the language model’s responses. LVLMs also commonly use similar image encoders to convert images into visual tokens. These tokens encode rich semantic and structural information. They provide a crucial interface through which language decoders—the components that generate text—access and reason about image content.

CroPA does not sufficiently exploit the vision encoder’s architecture. Instead, it treats vision and language components as equally open to manipulation, potentially missing vulnerabilities specific to the vision encoder.

We therefore propose combining the CroPA loss with a perturbation loss that disrupts the encoder’s core attention mechanism. We focus on the value vectors in its attention layers. We hypothesize that perturbing these vectors between the original image and a reference image for the target text can significantly improve Adversarial Success Rate (ASR) under CroPA’s existing hyperparameter constraints. Hyperparameters are settings that control optimization.

This motivation comes from empirical findings on adversarial transferability in VLMs. We draw on Doubly-Universal Adversarial Perturbation (Doubly-UAP), a method that identifies the attention components most effective for adversarial influence. Instead of maximizing losses between value vectors of \(x_i\) and \(T_i\), we minimize them.

We hypothesize that targeting value vectors in vision layers that encode essential visual features will disrupt interpretation more effectively than CroPA’s generic PGD perturbations. We also freeze the text encoder, meaning that we hold it fixed. This ensures that perturbations specifically degrade visual processing rather than introduce text-based artifacts.

Our modification aims to:

1. Improve attack effectiveness by exploiting information in the vision encoder.
2. Improve semantic alignment between perturbations and the target text representation.
3. Improve adversarial robustness across models while keeping computation feasible.

Jointly optimizing the CroPA loss and our modified Doubly-UAP loss creates a more structured, interpretable, and targeted attack framework.

We adapt a component of Doubly-UAP from Kim et al. (2024). That method introduced a UAP by identifying the vision encoder attention components that most strongly affect VLM performance. We focus on two components with fundamental roles:

1. **Attention Weights:** These control how much each patch focuses on other patches. They determine the degree of interaction or relevance between patches. We hypothesize that targeting these weights can disrupt the encoder’s ability to establish those relationships.
2. **Value Vectors:** These contain each patch’s actual information. We expect that perturbing them will disrupt essential patch content and further impair interpretation.

We target vision encoders because they are crucial to visual interpretation. Within them, we focus on attention, the core process for interpreting visual features. Based on prior analysis, we target its most vulnerable components: value vectors in middle-to-late layers.

In the standard Doubly-UAP attack, the perturbation \(\delta^*\) is:

\[
\delta^*
=
\arg\max_{\delta}
\frac{1}{|L|}
\sum_{l\in L}
\operatorname{Loss}\bigl(V_l(x),V_l(x+\delta)\bigr),
\tag{10}
\]

where \(L\) is the set of selected layers, \(|L|\) is its size, and \(V_l(x)\) contains the value vectors for input image \(x\) at layer \(l\). The function \(\operatorname{Loss}(\cdot)\) measures the loss on those vectors, and \(\arg\max\) selects the perturbation that maximizes it.

#### 5.2.1 Our Modified Approach

We modify the loss to include a target value vector from a reference image that corresponds to the desired target text \(T\). Rather than only maximizing the difference from the original value vectors, we enforce alignment between the encoder’s output and a predefined target representation.

Our method has the following steps.

**Figure 2:** D-UAP Architecture Diagram

1. Extract value vectors from layer \(l\) of the vision encoder for both the input image and the image associated with the target text.
2. Compute their cosine similarity loss. Cosine similarity measures alignment between vector directions.
3. Jointly minimize the negative of this loss and the CroPA loss, with suitable scaling to ensure adversarial robustness and target alignment.

Let \(V_i(x+\delta)\) denote the perturbed value vectors of attention head \(i\) for input image \(x\). An attention head is one component of an attention layer. Let \(V_t\) denote the value vectors of the target text’s reference image. We define:

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
\right),
\tag{11}
\]

where \(N\) is the number of attention heads and the dot denotes a dot product. This loss encourages the perturbed image’s value vectors to align closely with the target text’s representation.

We combine this loss with the CroPA loss in Equation 12. The hyperparameter \(\lambda\) controls the value-vector loss’s relative importance:

\[
L_{\text{re-CroPA}}
=
L_{\text{CroPA}}(x_v+\delta_v,x_t+\delta_t,T)
-\lambda L_{\text{d-UAP}}(\delta_v).
\tag{12}
\]

Joint optimization preserves the perturbation’s adversarial effect while enforcing semantic alignment with the target text. Gradients from both losses combine to update \(\delta\). The resulting adversarial example therefore has both cross-prompt transferability and guided semantic influence.

This extension to Doubly-UAP enables more precise, controlled, and interpretable manipulation of VLMs. It has applications in adversarial robustness and security analysis. Algorithm 3 in Appendix B summarizes the method.

We propose this method as a step toward improving CroPA’s perturbation transferability. However, D-UAP depends on model value vectors and thus on the vision encoder. We therefore hypothesize that transferability may remain limited to models whose vision encoders have the same or similar architecture as the model used to run CroPA.

### 5.3 Transferability across images via Image Augmentation

Traditional attacks use iterative gradient descent to optimize perturbations for individual images. This often causes overfitting to image-specific features, meaning that perturbations become too specialized to those images. The resulting poor cross-image transferability also makes attacks computationally expensive.

CroPA has the same issue: it learns perturbations for one image across multiple prompts. Luo et al. (2024) state that their method is orthogonal to cross-image transferability approaches—it addresses a separate aspect. They also state that computing perturbations across images can improve cross-image transferability, as in Moosavi-Dezfooli et al. (2017). We test this claim and introduce a new approach based on other UAP research.

Fang et al. (2024) introduce UAPs for Vision-Language Pretraining (VLP) models to reduce overfitting in gradient-based attacks and enable cross-image transferability. VLP models learn image and language representations during pretraining.

CutMix, proposed by Yun et al. (2019), offers a different regularization strategy, or method for reducing overfitting. It creates training examples by replacing image regions rather than blending pixels. CutMix cuts rectangular patches from one image and pastes them into another. It mixes their labels in proportion to patch area. This forces models to learn from partial views and prioritize spatially invariant features—features that remain useful across locations. It thereby improves robustness to attacks and occlusions, where parts of an image are hidden.

Zhang et al. (2024) extend UAP frameworks with Effective and Transferable Universal Adversarial Attack (ETU). ETU improves transferability across VLP models and tasks through better global and local optimization. It introduces SCMix, an augmentation method that combines self-mixing and cross-mixing to increase input diversity while preserving meaning. SCMix aims to combine spatially invariant and scale-invariant features, which remain useful across image locations and sizes.

We integrate SCMix into perturbation generation. We select it because their experiments show that, for the same model, SCMix alone yields superior cross-image transferability without their additional local utility reinforcement.

#### 5.3.1 SCMix

SCMix diversifies training inputs through two stages: self-mixing and cross-mixing. Self-mixing creates variations of a base image \(I_1\) by blending randomly cropped and resized patches:

\[
I'_1=\eta X_1+(1-\eta)X_2
\quad\text{where}\quad
X_i=\operatorname{Resize}(\operatorname{RandomCrop}(I_1)),
\quad \eta=0.5.
\tag{13}
\]

The original SCMix samples \(\eta\) from a beta distribution:
\(\eta\sim\operatorname{Beta}(\alpha,\alpha)\), for some \(\alpha>0\).
We instead fix \(\eta=0.5\), which is equivalent to setting \(\alpha=\infty\). Appendix D.3 justifies this choice.

Cross-mixing then adds features from a second image \(I_2\) through a weighted combination:

\[
I_3=\beta_1 I'_1+\beta_2 I_2
\quad\text{with}\quad
\beta_1\gg\beta_2\in[0,1).
\tag{14}
\]

Here, \(\beta_1\gg\beta_2\) means that the first weight is much larger than the second. This weighting preserves the dominant visual meaning of \(I_1\) in \(I_3\) while adding diversity from \(I_2\). It lets the adversarial generator learn perturbations that work across varied images.

Keeping the base image dominant also ensures that \(I_3\) remains compatible with \(I_1\)’s text description \(T_1\). The synthetic image therefore supports richer image-text interactions while remaining aligned with \(T_1\), despite including features from \(I_2\).

This two-stage mixing reduces the overfitting that can occur in CroPA. It forces perturbations to attack features that remain stable under both within-image changes, through self-mixing, and between-image blending, through cross-mixing. This regularization promotes universal perturbations that may work on unseen images without training a separate attack for each one.

The approach connects targeted effectiveness with cross-image generalization. At inference, CroPA returns only the perturbed image, or equivalently a suitable image perturbation. If it learns effective cross-image perturbations, the attack can become universal in two senses: cross-prompt transferable, as CroPA claims, and cross-image transferable, as we aim to test.

**Figure 3:** Illustration of self-mixing and cross-mixing. Self-mixing of input image \(I_1\) produces intermediate representations \(X_1\) and \(X_2\), which combine into \(I'_1\). Mixing an external image \(I_2\) with \(I'_1\) produces the final image \(I_3\).

#### 5.3.2 CutMix

CutMix improves transferability through discrete replacement of regional features (Yun et al. (2019)). For images \(x_A\) and \(x_B\), it creates:

\[
\tilde{x}=M\odot x_A+(1-M)\odot x_B,
\tag{15}
\]

where \(M\in\{0,1\}^{W\times H}\) is a binary mask defining a randomly placed rectangular region. Its entries are either 0 or 1, and \(W\) and \(H\) are image width and height. The symbol \(\odot\) means element-wise multiplication.

CutMix allows learning from partial image views. Its main difference from SCMix is that SCMix softly blends features within and across images, while CutMix uses hard, mask-based replacement.

Unlike continuous blending, discrete region replacement forces perturbations to target spatially invariant features that persist across image combinations. When combined with adversarial training, this encourages perturbations to exploit higher-level semantic patterns rather than local artifacts. It may therefore improve cross-image transferability.

### 5.4 Attack Setup and Threat Models for Transferability Experiments

We define threat models—the attacker’s goals, knowledge, and capabilities—for cross-model and cross-image transferability. We also specify each evaluation protocol.

#### 5.4.1 Using D-UAP to explore Cross-Model Transferability for CroPA

We use the following threat model to test how well perturbations from our D-UAP enhancement transfer across models.

**Adversary’s Goal:** The attacker seeks one perturbation for an image that causes multiple VLMs to output a specific target text, regardless of the prompt. This represents a practical black-box transfer setting: the attacker optimizes against an accessible model and deploys the perturbation against unknown models.

**Knowledge and Capabilities:** The attacker has white-box access to a surrogate model—a model used to develop the attack. Our experiments use either BLIP-2 or Flamingo. Access includes gradients and internal representations. For target models, the attacker knows only the general architecture family, such as CLIP-based, and cannot access parameters. CLIP refers to a Contrastive Language–Image Pretraining model. The attack must stay within an \(L_\infty\) perturbation budget of \(\epsilon=16/255\) to ensure visual imperceptibility.

**Cross-Model Settings:** We examine two scenarios that reflect real-world deployment:

- **Intra-family transfer:** We optimize perturbations on BLIP-2 with the OPT-2.7b language model and evaluate them on InstructBLIP with Vicuna-7b. Both use similar CLIP ViT-L/14 vision encoders but differ in language models and fine-tuning objectives. ViT denotes a Vision Transformer.
- **Cross-architecture transfer:** We test perturbations from BLIP-2 on Flamingo and from Flamingo on BLIP-2. These models differ fundamentally in vision encoders, language models, and architecture.

**Evaluation Setup:** We measure ASR separately for VQAgeneral, VQAspecific, Classification, and Captioning. We count success when the target model fails to generate the correct output for the perturbed image, regardless of the prompt.

#### 5.4.2 Investigating Cross-Image Transferability via Augmentation Techniques

We use the following threat model to evaluate universal perturbations across images.

**Adversary’s Goal:** The attacker seeks one perturbation pattern that causes a VLM to output predetermined target text when applied to any image, regardless of its content or prompt. This is more efficient than optimizing each image separately.

**Knowledge and Capabilities:** The attacker can use a limited set of training images and their SCMix-augmented variants. The attacker has white-box access to the target model during optimization but must produce a perturbation that generalizes to unseen images. The \(L_\infty\) perturbation budget is \(\epsilon=16/255\).

**Evaluation Setup:** We test universal perturbations on separate images that were not used in training. We measure ASR across task types. An attack succeeds if it causes the model to fail to generate the correct output text for a previously unseen image, regardless of content or prompt.

Our SCMix enhancement aims to reduce gradient-based methods’ tendency to overfit to specific image features. We hypothesize that controlled self-mixing and cross-mixing variations will help perturbations capture model vulnerabilities that do not depend on particular image characteristics.

#### 5.4.3 Evaluation Rationale for Transferability experiments

We present all cross-prompt and cross-model transferability results under the untargeted attack setting. Two main factors motivate this choice.

First, VLMs use different embedding mechanisms to incorporate visual information. This makes it hard to determine how input perturbations affect final output-token embeddings. We observe notably higher targeted transferability between models with similar vision encoders, such as BLIP-2 and InstructBLIP. Targeted transferability varies sharply otherwise, making direct comparisons of targeted ASRs across different architectures imprudent.

Second, the method used to combine visual and textual information strongly affects adversarial control. Decoder-only VLMs such as LLaVA and OpenFlamingo add visual features late through cross-attention or token concatenation. Cross-attention lets one representation attend to another; token concatenation joins token sequences. These models treat visual features as auxiliary memory during causal language generation, which generates text based on preceding context. Visual influence therefore fades as decoding proceeds (Awadalla et al. (2023), Liu et al. (2023)).

In contrast, fusion-based models such as BLIP-2 (Li et al. (2023b)) and InstructBLIP (Dai et al. (2023)) integrate visual features early and deeply into input embeddings. This preserves visual grounding throughout generation. Targeted attacks are therefore inherently more feasible in fusion-based models than in decoder-only models (Dai et al. (2023)).

Given these architectural differences, non-targeted ASR provides a more robust, fair, and comparable measure of transferability across models and images.

We also exclude CIA from transferability comparisons. Our experiments found negligible transferability, and the method itself presents these limitations:

- **Image-Specific Semantic Embedding:** CIA manipulates visual tokens for each image to embed target concepts. Its adversarial patterns depend on that image and lack cross-image consistency. Gradient directions and token spaces vary greatly across images and concepts. Transformations between input and target images in token space do not transfer.
- **Model-Specific Embedding Architectures:** CIA depends strongly on visual and text encoder structure. Structural differences across LVLMs, including BLIP2, LLaVA, and Flamingo, prevent it from demonstrating cross-model transferability.

## 6 Results

As Section 4 states, a major objective is to reproduce Luo et al. (2024)’s cross-prompt transferability results for CroPA. This section describes our replication and compares its results with the original paper.

**Models Used:**

We evaluated five state-of-the-art VLMs to reproduce Luo et al. (2024): OpenFlamingo, BLIP-2, InstructBLIP and LLaVA. For Flamingo, we used the publicly available OpenFlamingo-9B implementation (Awadalla et al. (2023)). It performs comparably to the original model.

BLIP-2 uses a two-stage, fusion-based approach. It first extracts visual features with a frozen CLIP image encoder, then processes them through a Querying Transformer (Li et al. (2023b)). This architecture supports efficient adaptation to different vision-language tasks. BLIP-2 uses OPT-2.7b as its language model, supporting flexible text generation.

InstructBLIP builds on BLIP-2’s fusion architecture and adds instruction tuning, or training to follow instructions (Dai et al. (2023)). It notably uses the Vicuna-7b language model, which improves its ability to follow task-specific instructions. Carefully designed prompts can therefore control its outputs more precisely.

LLaVA uses decoder-only generation (Liu et al. (2023)). A lightweight projection layer connects a pretrained CLIP ViT-L vision encoder to Vicuna-13B. Unlike fusion-based models, LLaVA passes visual features directly into the language model’s context. It treats images as special tokens in the prompt sequence.

These models represent two architectural approaches. Decoder-only models, OpenFlamingo and LLaVA, treat visual embeddings as input context for generation. Fusion-based models, BLIP-2 and InstructBLIP, explicitly align image and text representations before generation. These differences significantly affect targeted attacks across prompts.

We based our reproduction on the authors’ public code. We kept the original model configurations to support faithful comparison with baseline results. The overall implementation was well documented, but several problems required changes. In particular, the BLIP-2 and InstructBLIP code produced multiple runtime errors. We had to make substantial debugging adjustments before it would run.

### 6.1 Claim 1: CroPA achieves cross-prompt transferability across various target texts. [Reproduced]

We carefully replicated Luo et al. (2024)’s setup to test CroPA’s central cross-prompt transferability claim. We trained adversarial examples with CroPA and tested them across varied target prompts on Flamingo. We measured targeted ASR for general and specific VQA, image classification, and image captioning.

#### 6.1.1 Experimental Details

Following Luo et al. (2024), we generated adversarial examples with CroPA and evaluated them using target prompts different from those used in training. Table 1 lists the target prompts. They cover a broad range of meanings to support a rigorous transferability evaluation. Appendix D.1 gives further details for all reproduced experiments.

#### 6.1.2 Comparative Analysis

Table 1 reports CroPA’s targeted ASRs on Flamingo for different target texts. Overall, our results support the claim that CroPA transfers across prompts. We observed non-zero ASRs for every evaluated task. Thus, CroPA’s adversarial examples misled the VLM even with varied target prompts.

However, CroPA consistently underperforms on Classification and Captioning. This is especially true for abstract texts such as “too late” and “not sure,” which lack significant visual meaning.

| Target Prompt | Method | VQAgeneral | VQAspecific | Classification | Captioning | Overall |
|---|---|---|---|---|---|---|
| unknown | Single-P | 0.2640 | 0.4720 | 0.2700 | 0.1400 | 0.2865 |
| | Multi-P | 0.7240 | 0.8740 | 0.5550 | 0.2850 | 0.6095 |
| | CroPA | 0.9680 | 0.9880 | 0.7070 | 0.4200 | 0.7708 |
| I am sorry | Single-P | 0.2120 | 0.4740 | 0.5170 | 0.2530 | 0.3640 |
| | Multi-P | 0.7020 | 0.9020 | 0.7090 | 0.6140 | 0.7318 |
| | CroPA | 0.8620 | 0.9260 | 0.7170 | 0.6890 | 0.7985 |
| not sure | Single-P | 0.3540 | 0.4240 | 0.1450 | 0.0000 | 0.2308 |
| | Multi-P | 0.6520 | 0.6860 | 0.2310 | 0.0030 | 0.3930 |
| | CroPA | 0.8760 | 0.8940 | 0.2420 | 0.0010 | 0.5033 |
| very good | Single-P | 0.3980 | 0.5760 | 0.2180 | 0.0480 | 0.3100 |
| | Multi-P | 0.8940 | 0.9640 | 0.4040 | 0.2080 | 0.6175 |
| | CroPA | 0.9620 | 0.9860 | 0.6060 | 0.2540 | 0.7020 |
| too late | Single-P | 0.2520 | 0.4800 | 0.2280 | 0.0220 | 0.2455 |
| | Multi-P | 0.7880 | 0.9120 | 0.5040 | 0.1630 | 0.5918 |
| | CroPA | 0.9300 | 0.9580 | 0.7010 | 0.1790 | 0.6920 |
| metaphor | Single-P | 0.3280 | 0.6020 | 0.5550 | 0.1040 | 0.3973 |
| | Multi-P | 0.8880 | 0.9520 | 0.8420 | 0.4910 | 0.7933 |
| | CroPA | 0.9840 | 0.9940 | 0.9100 | 0.5840 | 0.8680 |

**Table 1:** Targeted ASRs on Flamingo for different target texts across vision-language tasks. VQAgeneral and VQAspecific describe how specific a prompt is to a particular image. Appendix E lists the prompts.

Our experiments reproduce the core finding that CroPA has robust cross-prompt transferability. They thus validate VLMs’ vulnerability to these attacks.

### 6.2 Claim 2: CroPA achieves the best overall performance across different number of prompts [Reproduced]

Figure 4 compares CroPA and the baselines as the number of prompts changes. ASR increases with the number of prompts for all methods. This comparison is redundant for Single-P because it is inherently tested on a single prompt; Figure 4 therefore represents it with a single star.

For the same number of prompts, CroPA consistently achieves higher ASR than the baselines. This supports its central goal of transferring across prompts. We use “unknown” as the target text to avoid high-frequency responses common in vision-language tasks. We consider this justification from Luo et al. (2024) valid for this experiment.

As the number of prompts increases, the baseline’s ASR saturates below CroPA’s. The trend indicates that adding prompts does not let the baseline surpass CroPA, consistent with Luo et al. (2024).

**Figure 4:** Targeted ASR on Flamingo for different numbers of prompts. With the same number of attack iterations, our CroPA significantly outperforms the baselines. The target text is “unknown.”

### 6.3 Claim 3: CroPA converges towards higher ASR as number of iterations increase [Reproduced]

Our convergence results differ from the original reported metrics as the number of iterations increases. Appendix A.2 gives the authors’ results. We used the same training and evaluation parameters and the authors’ own code. The original metrics showed CroPA converging with more prompts.

We observe this trend in overall ASR across tasks, as Figure 6 shows. That figure follows the figures reported in the original paper. However, ASR for individual tasks is prone to instability, as Figure 5 shows.

The authors did not specify the model used for this evaluation. We therefore report OpenFlamingo results with “unknown” as the target text.

(a) VQA

(b) VQA-Specific

(c) Classification

(d) Captioning

**Figure 5:** Comparison of Single-P, Multi-P, and CroPA across different categories.

**Figure 6:** Overall Convergence and Comparison of CroPA

### 6.4 Claim 4: CroPA outperforms baselines under few-shot settings [Reproduced]

Flamingo accepts extra images as in-context learning examples, alongside the text prompt, to improve task adaptation. Luo et al. (2024) tested how these examples affect cross-prompt attacks. They evaluated adversarial images with a different number of in-context examples from the number used during optimization.

Optimization uses a 0-shot setting: no extra in-context images. Evaluation uses a 2-shot setting: two extra in-context images.

| Setting | Method | VQAgeneral | VQAspecific | Classification | Captioning | Overall |
|---|---|---|---|---|---|---|
| shot=0 | Multi-P | 0.7240 | 0.8740 | 0.5550 | 0.2850 | 0.6095 |
| | CroPA | 0.9680 | 0.9880 | 0.7070 | 0.4200 | 0.7708 |
| shot=2 | Multi-P | 0.7860 | 0.9420 | 0.5630 | 0.0010 | 0.5730 |
| | CroPA | 0.9440 | 0.9940 | 0.6860 | 0.0300 | 0.6635 |

**Table 2:** Targeted ASRs with and without visual in-context learning. “Shot” is the number of images added as in-context examples.

In-context examples inherently make attacks harder because they can shift generation conditions away from those used during optimization (Luo et al. (2024)). Extra images also make the model more aware of the correct prompt space, which differs from the target text space. Zhao et al. (2023) and Qi et al. (2023) verify that attacks such as CroPA, optimized for a specific context, become less effective when context changes.

However, CroPA performs significantly better than the baseline on most tasks. It is therefore more robust to the broader context introduced by in-context examples. This verifies the motivation and hypothesis behind the min-max objective in Subsection 4.3: to make CroPA more robust to a wider prompt space and resulting context changes.

### 6.5 Results Beyond the Original Paper

The following subsections describe our extensions to CroPA. They introduce enhancements that address its corresponding limitations.

#### 6.5.1 Noise Initialization

Our Noise Initialization via Vision Encoding Optimization method, described in Subsection 5.1, significantly improves performance across tasks relative to baselines. Semantically informed initialization improves adversarial robustness and cross-prompt transferability.

Table 3 shows that our integrated method, CroPA+Init, consistently outperforms CroPA and Multi-P across all metrics. It achieves 6-15% gains in overall accuracy.

These results validate semantically guided initialization over random noise. Vision-encoder-anchored perturbations improve ASR and reduce optimization steps. This supports the method’s robustness and adaptability as a principled alternative for attack initialization.

| Target Prompt | Method | VQAgeneral | VQAspecific | Classification | Captioning | Overall |
|---|---|---|---|---|---|---|
| unknown | Multi-P | 0.7240 | 0.8740 | 0.5550 | 0.2850 | 0.6095 |
| | CroPA | 0.9680 | 0.9880 | 0.7070 | 0.4200 | 0.7708 |
| | CroPA+Init | 0.8824 | 0.8920 | 0.8858 | 0.9306 | 0.8977 |
| bomb | Multi-P | 0.6840 | 0.8260 | 0.9050 | 0.6090 | 0.7560 |
| | CroPA | 0.8176 | 0.9100 | 0.9498 | 0.6960 | 0.8433 |
| | CroPA+Init | 0.9140 | 0.9468 | 0.9020 | 0.9412 | 0.9260 |
| I am sorry | Multi-P | 0.7020 | 0.9020 | 0.7090 | 0.6140 | 0.7318 |
| | CroPA | 0.8620 | 0.9260 | 0.7170 | 0.6890 | 0.7985 |
| | CroPA+Init | 0.9320 | 0.9524 | 0.9270 | 0.9716 | 0.9458 |
| very good | Multi-P | 0.7900 | 0.8860 | 0.6700 | 0.6700 | 0.7540 |
| | CroPA | 0.8540 | 0.8990 | 0.7680 | 0.7550 | 0.8190 |
| | CroPA+Init | 0.8745 | 0.8894 | 0.8362 | 0.9043 | 0.8761 |
| too late | Multi-P | 0.6960 | 0.8540 | 0.6780 | 0.6060 | 0.7585 |
| | CroPA | 0.8990 | 0.9060 | 0.7940 | 0.8150 | 0.8535 |
| | CroPA+Init | 0.9340 | 0.9220 | 0.9110 | 0.9660 | 0.9333 |

**Table 3:** Targeted ASRs on Blip2 for different target texts using CroPA Vision Encoder noise initialization. Section D.2 gives the detailed hyperparameters.

##### Theoretical Analysis of Initialization in Bi-level Optimization

The large gains from diffusion-guided initialization reflect the distinctive optimization challenges in CroPA’s min-max framework. Our analysis identifies three key factors.

First, CroPA’s adversarial formulation resembles a saddle-point problem, in which the objective is minimized in one direction and maximized in another. Initialization strongly affects optimization stability in such problems. Zhang et al. (2025) show that min-max problems are significantly more sensitive to initialization than standard minimization problems. Stable convergence often requires carefully designed starting points. This sensitivity is especially strong in CroPA because prompt perturbations continually change the optimization landscape—the loss across possible perturbations—during training.

Second, CroPA updates image perturbations more often than prompt perturbations. This alternating schedule creates optimization asymmetry that further increases the importance of initialization. Chen et al. (2019) found that alternating gradient methods for min-max problems often cycle locally rather than settle. Their convergence depends strongly on how close the initial point is to adversarially robust regions. Our semantic initialization uses this insight by placing the starting point closer to meaningful adversarial directions.

Third, our experiments align with Wu et al. (2020). They showed that perturbations initialized with semantic priors, or prior information about meaning, transfer better across models than randomly initialized perturbations. They also produce substantial ASR improvements. This effect is especially pronounced in bi-level optimization settings such as CroPA, where optimization paths are more complex.

Our diffusion-based method addresses these challenges by initializing perturbations toward semantic targets instead of using standard random noise. This principled starting point enables faster convergence and improves the resulting examples’ transferability. Table 3 shows significant ASR increases.

| Method | VQAgeneral | VQAspecific | Classification | Captioning | Overall |
|---|---|---|---|---|---|
| Multi-P | 0.7053 | 0.8807 | 0.7001 | 0.5553 | 0.7104 |
| CroPA | 0.8723 | 0.9457 | 0.7518 | 0.6138 | 0.7958 |
| CIA | 0.2985 | 0.2281 | 0.4857 | 0.4687 | 0.370 |
| CroPA+Init | 0.9145 | 0.9342 | 0.8982 | 0.9464 | 0.9209 |

**Table 4:** Average targeted ASRs on BLIP-2 for different methods. Results cover general and specific Visual Question Answering, Classification, and Captioning. The best values for each task are highlighted in bold. Alongside the previously described baselines, we compare against the Contextual Injection Attack (CIA) of Yang et al. (2024).

Gradient-based attacks often acquire their targets’ semantic features. In theory, they should therefore not need semantically aligned initialization. However, the min-max complexities described above weaken the semantic alignment produced by gradients. Injecting target meaning into initialization is necessary to strengthen that alignment and improve convergence. Figure 7 provides experimental evidence for this claim.

We assess how much target information the perturbations acquire by measuring the CLIP score between the target text and perturbations from base CroPA and CroPA+Init. This score measures their feature similarity.

CroPA behaves as expected for gradient-based methods and acquires the target’s context. Initialization produces a modest increase in CLIP score. This suggests that CroPA already acquires target meaning, while initialization makes a subtle change that places perturbations closer to adversarially vulnerable regions. It does so even without significantly improving the perturbations’ target-related semantic features.

The convergence comparison also supports our hypothesis. By deliberately placing the starting point closer to meaningful adversarial directions, semantic initialization achieves faster convergence.

| Method | CLIP Score |
|---|---|
| CroPA | 21.80 |
| CroPA+Init | 23.09 |

Target Text: “Bomb”

(a) (b)

**Figure 7:** (a) Overall convergence comparison of base CroPA and CroPA+Init. (b) CLIP scores quantify the feature similarity between learned perturbations and the target.

#### 6.5.2 Investigating Cross-Image transferability and Image Augmentation

CroPA remains effective across text prompts, but that focus inherently limits cross-image transferability. It optimizes perturbations across prompts while keeping the resulting adversarial signals tightly coupled to the source image’s low-level features. This dependence prevents generalization to other images and makes universal cross-image attacks ineffective. It weakens the authors’ claim that learning perturbations across images would produce cross-image transferability.

We therefore integrated SCMix and CutMix, described in Subsection 5.3. Both diversify training inputs by mixing image features. Our goal was to help perturbations generalize beyond one image.

We believe CutMix outperforms SCMix in promoting adversarial transferability because its region-based mixing preserves distinct, non-overlapping spatial areas. This encourages modular, independently manipulable features, which make perturbations more generalizable across inputs.

Table 5 shows that both SCMix and CutMix produce measurable improvements, although cross-image transferability remains difficult. Despite these gains, the performance gap between cross-image methods and single-image CroPA exposes a fundamental limitation: CroPA excels at cross-prompt robustness but does not effectively support transferable attacks across images.

| Method | VQA | VQAspecific | Classification | Captioning | Overall |
|---|---|---|---|---|---|
| CroPA (Base) | 0.3500 | 0.4920 | 0.4000 | 0.5000 | 0.4355 |
| CroPA with SCMix | 0.6560 | 0.7160 | 0.5000 | 0.9500 | 0.7055 |
| CroPA with CutMix | 0.4520 | 0.7240 | 0.5000 | 0.5500 | 0.5565 |

**Table 5:** Untargeted ASR in cross-image settings, with and without SCMix and CutMix. Values are the best ASRs across all iterations. CroPA without mixing reached its highest ASRs at 1700 iterations. CroPA with SCMix and CutMix significantly outperformed it earlier, at 1500 and 1600 iterations, respectively. Appendix D.3 gives further experimental parameters.

#### 6.5.3 Guiding perturbations via Target Value Vectors

We evaluate the D-UAP-modified CroPA framework from Subsection 5.2. Tables 6 and 7 show that CroPA+D-UAP is effective across vision-language tasks. Its enhanced perturbations greatly improve the existing algorithm across the board. Overall ASR reaches 96.85%, a 12.5% increase over base CroPA. VQAgeneral and Captioning improve particularly strongly, by 13% and 28%, respectively.

These experiments use “Bomb” as the target text. This demonstrates improved performance even for harmful text that is inherently set to trigger model guardrails, or safety mechanisms. Thus, in the worst expected cases involving “harmful words,” the enhancement makes the model perform significantly better in vulnerable scenarios.

| Method | VQAgeneral | VQAspecific | Classification | Captioning | Overall |
|---|---|---|---|---|---|
| Multi-P | 0.68 | 0.82 | 0.90 | 0.60 | 0.75 |
| CroPA | 0.8176 | 0.9100 | 0.9498 | 0.6960 | 0.8433 |
| CIA | 0.3431 | 0.3102 | 0.4732 | 0.5534 | 0.4199 |
| CroPA+D-UAP | 0.9420 | 0.9720 | 0.9790 | 0.9810 | 0.9685 |

**Table 6:** Targeted ASRs on Blip2 with the D-UAP function and “Bomb” as target text. The method outperforms PGD baselines, CIA, and base CroPA.

| Method | VQAgeneral | VQAspecific | Classification | Captioning | Overall |
|---|---|---|---|---|---|
| Multi-P | 0.6960 | 0.8540 | 0.8990 | 0.6060 | 0.7638 |
| CroPA | 0.7900 | 0.8860 | 0.9470 | 0.6700 | 0.8233 |
| CIA | 0.3027 | 0.4302 | 0.5112 | 0.5080 | 0.4380 |
| CroPA+D-UAP | 0.9000 | 0.9520 | 0.9500 | 0.9060 | 0.9270 |

**Table 7:** Targeted ASRs on Open Flamingo with the D-UAP loss function and “Bomb” as target text.

#### 6.5.4 Analyzing Cross Model Transferability by Value Vector guidance

Table 8 also shows a strong case for improved cross-model transferability. The method significantly improves base CroPA, especially on Classification and Captioning, by 14% and 21%, respectively. VQA gains are modest.

This difference across tasks can be attributed to the fact that VQA prompts span a much more complex embedding space. Tasks such as Classification and Captioning have outputs limited to much smaller spans.

Our analysis also reveals an important limitation in vision encoder behavior. It verifies the hypothesis in Subsection 5.2.

CroPA Doubly-UAP depends on value vectors from the VLM’s specific vision encoder. This may restrict transferability to models with the same or similar encoder architectures. Table 8 verifies this: perturbations trained on BLIP2 transfer passably to InstructBLIP, which used the same vision encoder, but fail to transfer to OpenFlamingo.

The same expected pattern occurs when training perturbations on Flamingo and applying them to other models, where ASR becomes insignificant. See Table D.4.5 in Appendix D.4.5. Appendix D.4 reports experimental details.

In Table 8, “Cls” means Classification and “Cap” means Captioning.

| Settings | Method | VQAgeneral | VQAspecific | Cls | Cap | Overall |
|---|---|---|---|---|---|---|
| BLIP2 to InstructBLIP | Multi-P | 0.1760 | 0.2090 | 0.2750 | 0.2480 | 0.2270 |
| | CroPA | 0.3220 | 0.4170 | 0.6720 | 0.6150 | 0.5065 |
| | CroPA + D-UAP | 0.3940 | 0.5780 | 0.8800 | 0.8880 | 0.6850 |
| BLIP2 to Flamingo | Multi-P | 0.0810 | 0.0660 | 0.0760 | 0.0890 | 0.0780 |
| | CroPA | 0.1520 | 0.1220 | 0.1180 | 0.1820 | 0.1435 |
| | CroPA + D-UAP | 0.1850 | 0.2190 | 0.1420 | 0.3880 | 0.2335 |
| BLIP2 to LLaVA | Multi-P | 0.1220 | 0.1060 | 0.0810 | 0.0630 | 0.0930 |
| | CroPA | 0.3190 | 0.2570 | 0.2730 | 0.3670 | 0.3040 |
| | CroPA + D-UAP | 0.5070 | 0.3420 | 0.4380 | 0.8430 | 0.5325 |
| InstructBLIP to BLIP2 | Multi-P | 0.1230 | 0.1520 | 0.1020 | 0.0960 | 0.1183 |
| | CroPA | 0.2690 | 0.2320 | 0.1870 | 0.2270 | 0.2288 |
| | CroPA+D-UAP | 0.4800 | 0.5100 | 0.4700 | 0.5000 | 0.4900 |
| InstructBLIP to LLaVA | Multi-P | 0.0650 | 0.0580 | 0.0420 | 0.0370 | 0.0505 |
| | CroPA | 0.1620 | 0.1120 | 0.1410 | 0.1230 | 0.1345 |
| | CroPA+D-UAP | 0.3450 | 0.3150 | 0.3680 | 0.3950 | 0.3558 |
| InstructBLIP to Flamingo | Multi-P | 0.0320 | 0.0210 | 0.0120 | 0.0110 | 0.0190 |
| | CroPA | 0.0740 | 0.0680 | 0.0490 | 0.0480 | 0.0598 |
| | CroPA+D-UAP | 0.1600 | 0.1400 | 0.1200 | 0.1300 | 0.1375 |
| Flamingo to BLIP2 | Multi-P | 0.0140 | 0.0070 | 0.0090 | 0.0120 | 0.0105 |
| | CroPA | 0.0450 | 0.0210 | 0.0300 | 0.0380 | 0.0335 |
| | CroPA+D-UAP | 0.1080 | 0.0800 | 0.0990 | 0.1180 | 0.1013 |
| Flamingo to InstructBLIP | Multi-P | 0.0100 | 0.0060 | 0.0050 | 0.0090 | 0.0075 |
| | CroPA | 0.0360 | 0.0220 | 0.0280 | 0.0300 | 0.0290 |
| | CroPA+D-UAP | 0.1050 | 0.0720 | 0.0880 | 0.1100 | 0.0938 |
| Flamingo to LLaVA | Multi-P | 0.0180 | 0.0110 | 0.0140 | 0.0170 | 0.0150 |
| | CroPA | 0.0400 | 0.0300 | 0.0350 | 0.0430 | 0.0370 |
| | CroPA+D-UAP | 0.1100 | 0.0850 | 0.0980 | 0.1170 | 0.1025 |

**Table 8:** Cross-model tests under different settings. “BLIP2 to InstructBLIP” means that perturbations optimized on BLIP2 are tested on InstructBLIP. “BLIP2 to Flamingo” follows the same convention.

We also found that, for a given target text, the vision encoder’s value vectors remain approximately consistent across input images. This has several implications:

1. **Vision Encoder Specificity:** Value vectors align with the vision encoder’s architecture and pretraining.
2. **Limited Diversity in Perturbations:** Consistent value vectors for a target text may constrain perturbation diversity. This could limit the attack’s ability to produce highly varied adversarial examples.
3. **Target Text Dependency:** Similar value vectors across images for the same target text indicate that target text strongly influences attack outcomes, rather than the input image’s specific features.

## 7 Discussion

### 7.1 Observations with reproducibility

We sought to replicate key aspects and findings of Luo et al. (2024)’s CroPA framework. Our study substantiates the original claims about improved cross-prompt adversarial transferability in VLMs.

Our experiments confirm strong cross-prompt transferability across VQA, image classification, and captioning. CroPA consistently outperforms the baselines. It also remains robust in few-shot settings (6.4). It handles the shift away from optimization-stage generation conditions introduced by in-context examples better than the baselines.

### 7.2 Noise Initialization findings

Subsection 5.1 identifies a critical limitation in CroPA and related attacks: random noise initialization ignores the target prompt’s semantic structure. We hypothesized that this lack of initial semantic guidance causes instability, slows convergence, and reduces cross-prompt and cross-modal transferability. Cross-modal transferability concerns transfer across different input types.

We addressed this with semantically informed initialization. A diffusion-synthesized reference image and alignment in the vision encoder’s feature space orient perturbations toward a meaningful adversarial direction from the start.

The experiments in 6.5.1 empirically validate this hypothesis. Across all evaluated benchmarks, our method significantly outperformed CroPA and randomly initialized baselines. It produced up to 15% gains in targeted ASR and notably reduced optimization steps. These findings confirm that semantic anchoring improves attack effectiveness and provides a principled, efficient foundation for VLM attacks.

The enhancement increased effectiveness while reducing the number of required optimization steps. Anchoring perturbations in the vision encoder’s semantic space produced a more principled and robust starting point than random noise. This highlights its potential for improving adversarial attack frameworks.

### 7.3 Evaluation of Transferability Experiments

#### 7.3.1 Cross-Image Transferability

The authors claim that computing perturbations across images can achieve cross-image transferability. Our experiments instead found that CroPA’s perturbations remain tightly tied to the low-level features of their source images, despite strong cross-prompt transferability.

We addressed this limitation by integrating SCMix (Subsection 5.3) and CutMix. Both augmentation methods increase cross-image diversity while preserving the original images’ semantic information.

Our results in 6.5.2 show that SCMix significantly improves cross-image transferability, especially for VQA, VQA-specific, and Captioning. Overall ASR increases by approximately 27 percent. CutMix produces a net increase of around 12 percent over base cross-image CroPA.

We believe SCMix performs this way because random cropping and resizing explicitly introduce spatial and scale invariance within an image, before soft mixing with a second image. This encourages perturbations to target semantically aligned features that depend less on exact position or scale. The perturbations can then generalize more easily across inputs.

Both SCMix and CutMix improve cross-image transferability when added to a cross-prompt framework such as CroPA. We believe these augmentation methods add cross-image features that help learned perturbations generalize.

Future work on cross-image transferability in cross-prompt frameworks should explore more complex, layered methods beyond augmentation. Possibilities include combining augmentation with feature-alignment attacks based on latent representations—the models’ internal image representations—or perturbing the latent space directly.

#### 7.3.2 Cross-Model Transferability

The D-UAP-modified CroPA framework in Subsection 5.2 significantly advances CroPA’s attack method. It achieves an overall targeted ASR of 96.85%, a 13.5% improvement over base CroPA, as shown in 6.5.3. VQAgeneral and Captioning improve particularly strongly, by 13% and 28%, respectively.

These gains also occur with challenging or harmful target text such as “Bomb,” which typically triggers model guardrails. This demonstrates the D-UAP framework’s robustness in bypassing safety mechanisms.

The framework also improves cross-model transferability, particularly between models with similar vision encoders, such as BLIP2 and InstructBLIP. However, dependence on a specific vision encoder constrains effectiveness. It limits perturbation diversity and makes transfer to different architectures harder.

This limitation highlights the broader challenge of encoder-agnostic attacks, which do not depend on a particular vision encoder. Even so, D-UAP marks an important step toward universal perturbations effective across prompts and images. It supports future research on generalization and robustness in VLMs.

Our cross-model comparisons exclude the Contextual Injection Attack because it is inherently non-transferable and could not produce comparable results.

### 7.4 Trade-offs Between Different Types of Adversarial Transferability

Our analysis reveals fundamental trade-offs among forms of adversarial transferability in VLMs. Understanding them is essential for designing robust attacks.

#### 7.4.1 Cross-Prompt vs. Cross-Model Transferability

Original CroPA achieves strong cross-prompt transferability by optimizing perturbations against competing text embeddings. However, this implicitly ties perturbations to the target model’s architecture. Our experiments in Section A.1 confirm this limit: ASRs fall significantly when perturbations generated on one model are applied to another.

Our D-UAP enhancement in Section 5.2 targets generalizable vision encoder components rather than model-specific features. It improves transfer between architecturally similar models, such as BLIP-2 and InstructBLIP. Gains remain modest between fundamentally different architectures, such as BLIP-2 and Flamingo. This suggests an inherent tension between optimizing for prompt diversity and model diversity at the same time.

#### 7.4.2 Cross-Prompt vs. Cross-Image Transferability

Cross-prompt and cross-image transferability involve competing optimization objectives. CroPA achieves strong cross-prompt transferability by closely tying perturbations to image-specific features. That same coupling fundamentally limits cross-image generalization.

We propose data augmentation to address this limitation. Our SCMix and CutMix integrations in Section 5.3 diversify optimization through augmented inputs. Table 5 shows promising improvements. However, our findings also suggest that augmentation does not fully resolve the fundamental tension between these objectives.

#### 7.4.3 Role of Initialization in Transferability Balance

Our diffusion-based initialization in Section 5.1 offers another perspective on these trade-offs. Semantically meaningful starting points increase ASR without sacrificing cross-prompt transferability. This suggests that the initial point in the optimization landscape strongly affects the achievable balance among transferability types.

Unlike methods that change optimization itself, better initialization helps perturbations converge toward more general adversarial directions that remain effective across prompts. However, the fundamental tension between cross-prompt and cross-image or cross-model transferability remains. Optimizing a perturbation for one dimension inevitably sacrifices effectiveness in others.

#### 7.4.4 Implications for Future Research

These trade-offs expose a central challenge for adversarial machine learning in VLMs: no single attack method can maximize every form of transferability at once. Future research should develop frameworks that allow controlled balances among competing objectives. Possible approaches include multi-objective optimization, which jointly handles several objectives, and architecture-aware attack designs.

Our work shows that individual techniques can advance particular transferability dimensions. A comprehensive solution would require fundamentally rethinking how perturbations interact with model architectures and data representations.

Despite these findings, cross-prompt adversarial transferability remains underexplored. The limited literature reveals a gap in our understanding of prompt-induced VLM vulnerabilities. Addressing that gap is essential because it has profound implications for secure deployment.

### 7.5 Broader Impact

From an attacker’s perspective, highly cross-prompt transferable examples pose significant threats. They can cause VLMs to produce malicious or misleading outputs even in response to harmless questions. Attackers could exploit them to spread false information or subvert systems that use VLMs for content generation and decision-making.

From a defensive perspective, imperceptible perturbations offer a new way to protect sensitive information. Adding them to images can cause VLMs to consistently output predetermined, non-sensitive text. This can block unauthorized attempts to extract confidential information from personal images. The technique offers proactive privacy and security protection as visual data becomes increasingly open to exploitation.

Our study also improves CroPA through better initialization and an enhanced loss function. These changes markedly increase ASR and adversarial examples’ generalization across models and images. They strengthen cross-prompt attacks while also supporting more resilient defenses.

In conclusion, our reproducibility study affirms Luo et al. (2024)’s foundational work and highlights the need for deeper study of cross-prompt transferability. A comprehensive understanding is crucial for building VLMs that withstand adversarial manipulation. It is also crucial for developing countermeasures that protect user data and preserve output integrity.

### 7.6 Limitations and Future Work

We reproduced CroPA’s key results and conducted extensive ablations, or tests of individual method components. However, limited computational resources repeatedly hindered our progress.

Training and evaluating attacks on large VLMs such as Flamingo and BLIP2 requires substantial GPU memory and processing power. This restricted some experiments’ scale and depth. Resource availability often limited extensive hyperparameter tuning and transferability tests across more architectures.

We also wanted to study transferability with an ensemble of these models, meaning several models used together. This could have provided deeper insight into perturbation generalization. However, its very high memory and computation requirements made it infeasible with our infrastructure.

Future work could improve cross-model transferability through encoder-agnostic methods and strengthen cross-image generalization. Implementing optimization with query-based strategies could also improve practical use.

### 7.7 Communication With Original Authors

We could not establish direct communication with the original authors during replication. At the time of writing, the issue raised on their GitHub repository about BLIP-2 and InstructBLIP remains unresolved.
