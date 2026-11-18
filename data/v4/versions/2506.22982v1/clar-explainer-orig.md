## Summary

This paper studies whether Cross-Prompt Attack (CroPA) produces adversarial images that remain effective across different textual prompts in Vision-Language Models (VLMs), which process images and text. It reproduces four claims about cross-prompt transferability, prompt counts, convergence, and few-shot evaluation. It also tests semantically guided initialization, a loss targeting vision-encoder value vectors, and image augmentation for transfer across images. The paper reports that CroPA outperforms its baselines, although task-level convergence can be unstable. Guided initialization improves overall targeted Attack Success Rate (ASR), the proportion of successful attacks. The value-vector enhancement reaches 96.85% targeted ASR on BLIP-2. Image augmentation improves cross-image results, but cross-image transfer remains limited. Cross-model transfer improves most between models with similar vision encoders.

## Glossary

### Terms, abbreviations, methods, models, and codes

- **Adversarial attack:** An attack that perturbs an input to make a model produce an incorrect, harmful, or predetermined output.
- **Adversarial example:** An input modified by an adversarial perturbation.
- **Adversarial perturbation:** A constrained change added to an image or prompt during an attack.
- **Adversarial robustness:** A model’s resistance to adversarial manipulation.
- **Adversarial transferability:** The ability of an adversarial example or perturbation to remain effective beyond the setting used to generate it.
- **ASR:** Attack Success Rate; targeted success requires the exact target text, while non-targeted success requires an output different from the clean prediction.
- **Adversarial Success Rate:** The alternative expansion of ASR used in Section 5.2.
- **Targeted attack:** An attack intended to make a VLM generate a predetermined text.
- **Non-targeted / untargeted attack:** An attack intended to make the output differ from the clean image’s prediction.
- **Cross-model transferability:** Adversarial effectiveness across different VLM architectures.
- **Cross-image transferability:** The ability of a perturbation to generalize across multiple images.
- **Cross-prompt transferability:** The ability of an adversarial image to remain effective across different textual prompts.
- **Cross-task transferability:** The ability of adversarial examples designed for one task to affect other tasks.
- **UAP:** Universal Adversarial Perturbation; a perturbation that generalizes across images.
- **VLM:** Vision-Language Model; a model that processes visual and textual inputs.
- **LVLM:** Large vision-language model.
- **VLP:** Vision-Language Pretraining; the model setting discussed in the cross-image attack literature.
- **VQA:** Visual Question Answering.
- **VQAgeneral / VQA-general:** The general-prompt VQA category.
- **VQAspecific / VQA-Specific / VQA-specific:** The image-specific-prompt VQA category.
- **Classification / Cls:** The image classification task.
- **Captioning / Cap:** The image captioning task.
- **Overall:** The aggregate ASR reported across the evaluated tasks.
- **CroPA:** Cross-Prompt Attack; an attack that optimizes image and prompt perturbations with opposing objectives.
- **Single-P:** A baseline that optimizes an image perturbation using one prompt.
- **Multi-P:** A baseline that optimizes an image perturbation using multiple prompts.
- **CroPA+Init / Init:** CroPA with Noise Initialization via Vision Encoding Optimization.
- **Noise Initialization via Vision Encoding Optimization:** Initialization that aligns a perturbed image’s vision-encoder representation with a diffusion-generated target image.
- **PGD:** Projected Gradient Descent; the iterative optimization method used for the perturbations.
- **MSE:** Mean Squared Error; the loss used to align vision-encoder outputs during initialization.
- **SDXL:** Stable Diffusion XL; the diffusion model used to synthesize a target image.
- **Diffusion-based target synthesis:** Generation of an image corresponding to a target prompt using a diffusion model.
- **Semantic alignment:** Alignment of a perturbation or representation with the target text’s semantic representation.
- **Vision encoder:** The VLM component that converts images into visual representations or tokens.
- **Text encoder:** The text-processing component frozen in the proposed value-vector approach.
- **Language model / language decoder:** The component that generates text from the model’s representations.
- **Language modeling loss:** The loss minimized for image perturbations and maximized for prompt perturbations in targeted CroPA.
- **Querying Transformer:** The BLIP-2 component that processes features extracted by its frozen image encoder.
- **Visual tokens:** Image-derived representations that encode semantic and structural information for a language decoder.
- **Attention mechanism:** The vision-encoder process through which image patches interact.
- **Attention weights:** Components that determine how much each patch focuses on other patches.
- **Value vectors:** Components that hold the information within each image patch.
- **Attention head:** The component indexed in the proposed value-vector loss.
- **Doubly-UAP:** Doubly-Universal Adversarial Perturbation; the method from which the value-vector component is adapted.
- **D-UAP / d-UAP:** The value-vector enhancement and loss adapted from Doubly-UAP.
- **CroPA+D-UAP / CroPA + D-UAP:** CroPA combined with the proposed value-vector loss.
- **re-CroPA:** The label used for the combined objective in Equation 12.
- **CIA:** Called Context Injection Attack in Section 2.3 and Contextual Injection Attack elsewhere; a method that injects target tokens into visual and textual contexts.
- **ETU:** Effective and Transferable Universal Adversarial Attack; a UAP method using global and local optimization.
- **SCMix:** An augmentation method combining self-mixing and cross-mixing.
- **Self-mixing:** Blending randomly cropped and resized patches from the same image.
- **Cross-mixing:** Blending a self-mixed image with a secondary image while preserving the base image’s dominance.
- **CutMix:** Augmentation that replaces rectangular regions with regions from another image and mixes labels by patch area.
- **Input diversity / image augmentation:** Variation of training inputs intended to reduce overfitting and improve transferability.
- **Spatial invariance:** Insensitivity to spatial changes, encouraged by the mixing methods.
- **Scale invariance:** Insensitivity to scale changes, encouraged by SCMix’s cropping and resizing.
- **White-box access:** Access to a model’s gradients and internal representations during attack optimization.
- **Black-box transfer:** Optimization on an accessible model followed by deployment against models whose parameters are unavailable.
- **Surrogate model:** The accessible model used to optimize a perturbation for transfer.
- **Target model:** The model on which the generated perturbation is evaluated.
- **Threat model:** The stated attacker goals, knowledge, capabilities, and evaluation conditions.
- **Intra-family transfer:** Transfer between models with similar vision encoders but different language models or fine-tuning objectives.
- **Cross-architecture transfer:** Transfer between models with different encoders, language models, and architectural designs.
- **In-context learning:** Task adaptation using extra example images supplied with the prompt.
- **Few-shot setting:** Evaluation with a small number of in-context examples.
- **0-shot / shot=0:** No additional in-context images.
- **2-shot / shot=2:** Two additional in-context images.
- **Decoder-only VLM:** A model that uses visual embeddings as context during causal text generation.
- **Fusion-based VLM:** A model that aligns or integrates visual and textual representations before generation.
- **Cross-attention / token concatenation:** Mechanisms described for incorporating visual features in decoder-only VLMs.
- **Bi-level / min-max optimization:** Optimization in which image and prompt perturbations pursue opposing loss objectives.
- **Saddle-point problem:** The optimization form used to explain CroPA’s sensitivity to initialization.
- **Alternating update schedule:** Updating image perturbations more frequently than prompt perturbations.
- **Local cycling:** A behavior of alternating min-max optimization discussed in the initialization analysis.
- **Gradient misalignment:** A factor identified in related work as limiting adversarial transferability.
- **Gradient descent / ascent:** Updates that minimize / maximize the loss.
- **Forward pass:** Processing perturbed inputs through the vision encoder and language model.
- **Backpropagation:** The process used to update perturbations from loss gradients.
- **Perturbation budget:** The maximum allowed perturbation magnitude.
- **Imperceptibility:** The intended human-invisibility of the constrained image perturbation.
- **Feature space / semantic embedding:** The representation space used to align images with target semantics.
- **Latent representation / latent space:** Representations proposed for future feature-alignment or perturbation approaches.
- **Guardrails:** Model mechanisms described as being triggered by harmful target text.
- **Encoder-agnostic method:** A proposed future attack method not dependent on a particular vision encoder.
- **Ensemble:** The proposed use of multiple models together to investigate transferability.
- **Query-based strategy:** A proposed future alternative for implementing attack optimization.
- **CNN-RNN:** The lightweight architecture category discussed as lacking in-context learning capabilities.
- **BLIP:** A VLM examined in prior adversarial-robustness work.
- **BLIP-2 / BLIP2 / Blip2:** The fusion-based VLM using a frozen CLIP image encoder, a Querying Transformer, and OPT-2.7b.
- **InstructBLIP:** An instruction-tuned BLIP-2-based model using Vicuna-7b.
- **Flamingo:** The VLM evaluated through the open-source OpenFlamingo implementation.
- **OpenFlamingo / Open Flamingo / OpenFlamingo-9B:** The publicly accessible Flamingo implementation used in the experiments.
- **LLaVA:** A decoder-only VLM connecting a pretrained CLIP ViT-L encoder to Vicuna-13B through a projection layer.
- **CLIP:** The image-encoder family and similarity-scoring mechanism used in the paper.
- **CLIP score:** The score used to compare learned perturbations with the target text.
- **ViT-L / ViT-L/14:** The vision-encoder architecture labels used for the evaluated models.
- **OPT-2.7b:** The language model used in BLIP-2.
- **Vicuna-7b:** The language model used in InstructBLIP.
- **Vicuna-13B:** The language model used in LLaVA.
- **GPU:** The computing hardware used to train and evaluate attacks.
- **L40S:** The 48GB GPU reported for the CroPA runtime example.
- **VRAM:** The GPU memory reported as 95% utilized in that example.
- **“unknown,” “I am sorry,” “not sure,” “very good,” “too late,” “metaphor,” “bomb,” and “Bomb”:** Target texts used in the reported experiments.
- **Star in Figure 4:** The single plotted point for Single-P.
- **Solid arrows / dashed arrows / `*`:** Forward-pass paths / backpropagation paths / fixed model parameters in Figure 1.
- **`-ve`:** The negative sign used in the description of the modified value-vector loss.

### Symbols and algorithm notation

- **\(f\):** The VLM function processing visual and textual inputs.
- **\(f_v(\cdot)\):** The VLM’s vision encoder.
- **\(x_v\):** The visual input.
- **\(x_t\):** The textual prompt input.
- **\(x_t^i\), \(x_t^k\):** Indexed prompts from a collection of prompts.
- **\(X_t=\{x_t^1,x_t^2,\ldots,x_t^k\}\):** The prompt set used for optimization.
- **\(k\):** The number of prompt instances in the summed objectives.
- **\(T\):** The predetermined target text or target prompt.
- **\(\delta_v\):** The image perturbation.
- **\(\delta_t\):** The prompt perturbation.
- **\(\delta\), \(\bm{\delta}\):** A perturbation in the initialization or value-vector formulations.
- **\(\delta_{\mathrm{init}}\):** The perturbation obtained by minimizing target-image feature discrepancy.
- **\(\delta^*\):** The perturbation obtained by the standard Doubly-UAP objective.
- **\(\epsilon_v\), \(\epsilon\):** The permitted perturbation magnitude.
- **\(\|\delta_v\|_p\):** The perturbation norm constrained in the general formulation.
- **\(L_\infty\), \(\ell_\infty\), \(\|\cdot\|_\infty\):** The norm used for the stated maximum-perturbation constraints.
- **\(\|\cdot\|_2^2\):** The squared representation difference used in the initialization loss.
- **\(\|\cdot\|_2\):** The vector norm used in the cosine-similarity denominator.
- **\(L\):** The language modeling loss; in Equation 10, \(L\) also indexes the selected layers.
- **\(l\), \(|L|\):** A selected layer index and the number of selected layers in Equation 10.
- **\(x_v'\):** The adversarial image returned by Algorithm 1.
- **\(x_t^{\prime i}\):** The initialized and updated adversarial version of prompt \(i\).
- **\(g_v\), \(g_t\):** The image and prompt loss gradients in Algorithm 1.
- **\(\nabla\):** The gradient used in the perturbation updates.
- **\(\alpha_1\), \(\alpha_2\):** The image and prompt step sizes in Algorithm 1.
- **\(K\):** The number of iteration steps in Algorithm 1.
- **\(N\):** The prompt-update interval in Algorithm 1; the number of attention heads in Equation 11.
- **`step`:** The current iteration counter in Algorithm 1.
- **\(\operatorname{sign}(\cdot)\):** The gradient-sign operation used in Algorithm 1.
- **\(\operatorname{mod}(\text{step},N)\):** The operation used to trigger a prompt update every \(N\) steps.
- **\(\operatorname{Clip}_{x_v,\epsilon}\):** Projection of the adversarial image into the \(\epsilon\)-ball around \(x_v\).
- **\(\oplus\):** Element-wise addition in Figure 1.
- **\(\mathbf{x}\), \(x\):** The original image in the initialization and value-vector formulations.
- **\(\mathbf{x}_{\mathrm{target}}\):** The diffusion-generated image corresponding to the target prompt.
- **\(\mathcal D\):** The diffusion process used to generate the target image.
- **\(\mathbf z\):** Gaussian-sampled input to the diffusion process.
- **\(\theta_{\mathrm{SDXL}}\):** The diffusion model’s pretrained weights.
- **\(\mathbf{x}_{\mathrm{adv}}\), \(\mathbf{x}_{\mathrm{adv}}^{(t)}\):** The adversarial image and its value at iteration \(t\).
- **\(t\):** The iteration index in Equation 8.
- **\(\alpha\):** The PGD step size in Equation 8; the Beta-distribution parameter in Section 5.3.1.
- **\(\mathcal B_\epsilon(\mathbf x)\):** The admissible \(\ell_\infty\)-ball around the original image.
- **\(\Pi_{\mathcal B_\epsilon(\mathbf x)}\):** Projection back into that admissible ball.
- **\(\mathcal L_{\mathrm{MSE}}\):** The squared difference between adversarial and target vision-encoder outputs.
- **\(V_l(x)\):** The value vectors at vision-encoder layer \(l\) for image \(x\).
- **\(\operatorname{Loss}(\cdot)\):** The loss applied to value vectors in the standard Doubly-UAP objective.
- **\(V_i(x+\delta)\):** The perturbed value vectors of attention head \(i\).
- **\(V_t\):** The value vectors of the target text’s reference image.
- **\(x_i\), \(T_i\):** The input and target instances named in the description of the modified Doubly-UAP approach.
- **\(L_{\mathrm{d\text{-}UAP}}\):** The proposed value-vector loss.
- **\(L_{\mathrm{CroPA}}\):** The CroPA loss.
- **\(L_{\mathrm{re\text{-}CroPA}}\):** The combined loss in Equation 12.
- **\(\lambda\):** The weight controlling the value-vector loss’s relative importance.
- **\(\cdot\):** The vector product used in Equation 11’s cosine-similarity expression.
- **\(I_1\):** The base image in SCMix.
- **\(X_1\), \(X_2\), \(X_i\):** Randomly cropped and resized versions of the base image.
- **\(I_1'\):** The self-mixed base image.
- **\(I_2\):** The secondary image used for cross-mixing.
- **\(I_3\):** The final cross-mixed image.
- **\(T_1\):** The base image’s text description, retained for the synthetic image.
- **\(\eta\):** The self-mixing weight, fixed at \(0.5\) in this study.
- **\(\operatorname{Beta}(\alpha,\alpha)\):** The distribution used to sample \(\eta\) in the original SCMix method.
- **\(\alpha=\infty\):** The setting described as equivalent to fixing \(\eta=0.5\).
- **\(\operatorname{RandomCrop}\), \(\operatorname{Resize}\):** The image operations used to form SCMix’s self-mixing inputs.
- **\(\beta_1\), \(\beta_2\):** Cross-mixing weights chosen so that the base image dominates.
- **\(\gg\):** The weight relationship expressing \(\beta_1\)’s dominance over \(\beta_2\).
- **\(x_A\), \(x_B\):** The two images combined by CutMix.
- **\(\tilde{x}\):** The composite image produced by CutMix.
- **\(M\):** CutMix’s binary rectangular-region mask.
- **\(W\), \(H\):** The mask’s image dimensions.
- **\(\odot\):** The mask-based multiplication used in the CutMix equation.
- **\(\min\), \(\max\), \(\arg\min\), \(\arg\max\):** The minimization and maximization operations defining the attack objectives.
- **\(\sum\):** Summation over the prompts, layers, or attention heads specified by an objective.

# Revisiting CroPA: A Reproducibility Study and Enhancements for Cross-Prompt Adversarial Transferability in Vision-Language Models

Atharv Mittal atharv_m@mfs.iitr.ac.in
††thanks: All authors contributed equally
Affiliation: Mehta Family School of Data Science and Artificial Intelligence

Affiliation: Indian Institute of Technology, Roorkee

  
Agam Pandey11footnotemark:
1

agam_p@ce.iitr.ac.in

Affiliation: Department of Civil Engineering

Affiliation: Indian Institute of Technology, Roorkee

  
Amritanshu Tiwari11footnotemark:
1

amritanshu_t@mfs.iitr.ac.in

Affiliation: Mehta Family School of Data Science and Artificial Intelligence

Affiliation: Indian Institute of Technology, Roorkee

  
Sukrit Jindal11footnotemark:
1

sukrit_j@mfs.iitr.ac.in

Affiliation: Mehta Family School of Data Science and Artificial Intelligence

Affiliation: Indian Institute of Technology, Roorkee

  
Swadesh Swain11footnotemark:
1

swadesh_s@ece.iitr.ac.in

Affiliation: Department of Electronics and Communication

Affiliation: Indian Institute of Technology, Roorkee

###### Abstract

*The study reproduces CroPA and proposes initialization, cross-image, and vision-attention enhancements.*

Large Vision-Language Models (VLMs), which process images and text, enable image classification, captioning, and visual question answering. They remain highly vulnerable to adversarial attacks: input manipulations that cause incorrect or harmful outputs. The vulnerability is especially important when both visual and textual inputs can be manipulated.

We reproduce “An Image is Worth 1000 Lies: Adversarial Transferability Across Prompts on Vision-Language Models.” We validate Cross-Prompt Attack (CroPA), which uses competing image and prompt perturbations, and confirm its superior cross-prompt transferability: effectiveness across different textual prompts.

We also propose or investigate three enhancements:

1. An initialization strategy that significantly improves Attack Success Rate (ASR), the proportion of successful attacks.
2. Universal perturbations to investigate cross-image transferability, or effectiveness across images.
3. A loss targeting the vision encoder’s attention mechanism to improve generalization.

Experiments on Flamingo, BLIP-2, and InstructBLIP, with extended experiments on LLaVA, validate the original results and show that our improvements consistently boost adversarial effectiveness. The work reinforces the importance of studying VLM vulnerabilities. It provides a more robust framework for transferable adversarial examples, with implications for real-world VLM security.

Code: https://github.com/Swadesh06/Revisting_CroPA

Reviewed on OpenReview: https://openreview.net/forum?id=5L90cl0xtf

**Claims and evidence**

- CroPA’s cross-prompt results are reproduced: Table 1, CroPA rows, Overall ASRs of 0.7708, 0.7985, 0.5033, 0.7020, 0.6920, and 0.8680.
- Initialization improves overall targeted ASR: Table 3, CroPA+Init rows, Overall values of 0.8977, 0.9260, 0.9458, 0.8761, and 0.9333.
- Augmentation and value-vector guidance improve attack results: Table 5, SCMix and CutMix rows, Overall 0.7055 and 0.5565; Table 6, CroPA+D-UAP row, Overall 0.9685; Table 8, cross-model comparisons.

## 1 Introduction

*The paper defines transferability settings and states its reproduction and extension goals.*

Large VLMs have transformed computer vision by supporting image classification, image captioning, and visual question answering. This versatility has encouraged research into visual-linguistic interactions. Yet Zhao et al. (2023), Qi et al. (2023), Zhang et al. (2022a), and Carlini et al. (2024) show that VLMs remain highly vulnerable to adversarial attacks. Subtle image perturbations can make them produce incorrect or harmful outputs. Text inputs introduce further attack vectors beyond those of traditional vision models.

Prior studies examine several settings. Zhao et al. (2023) analyze BLIP and BLIP-2 using query-based and transfer-based attacks in black-box settings, where model parameters are unavailable. Schlarmann & Hein (2023) study targeted and untargeted attacks in white-box settings, where internal model information is accessible. A targeted attack seeks a predetermined output; an untargeted attack seeks an output different from the clean prediction.

Later work also considers text. Qi et al. (2023) show that adversarial images can induce harmful instruction execution. Tu et al. (2023) systematically evaluate visual and textual attacks.

Adversarial generalization and transferability in VLMs have traditionally been divided into:

- **Cross-Model Transferability:** Adversarial examples remain adversarial across different VLM architectures. This is commonly called transferability.
- **Cross-Image Transferability:** A perturbation generalizes across multiple images. Such perturbations are often called Universal Adversarial Perturbations (UAPs).

Luo et al. (2024) introduce **Cross-Prompt Transferability**: adversarial images remain effective under different textual prompts. Their CroPA method uses learnable prompts to make image attacks effective regardless of textual input. Unlike work that treats visual and textual perturbations independently, CroPA couples their optimization. Luo et al. demonstrate it on classification, captioning, and visual question answering.

Our goals are:

- **[Reproducibility Study] Reproducing the Results from the Original Paper:** We reproduce and verify four main claims. CroPA outperforms established baselines across target texts, independently of target meaning or word frequency. More prompts improve transferability, but convergence is rapid and baselines still fail to surpass CroPA. CroPA remains effective with extra in-context examples. Its overall ASR converges as attack iterations increase. We also analyze convergence separately by task.
- **[Extended Work] Better Initialization Strategy:** We propose an initialization method that substantially increases ASR and transferability.
- **[Extended Work] Investigating Cross Image Transferability and Image Augmentation:** We learn a common perturbation across images using CroPA’s backbone and test transferability-promoting enhancements.
- **[Extended Work] Guiding perturbations via Target Value Vectors:** We propose a loss targeting value vectors, which hold image-patch information in the vision encoder’s attention mechanism. The proposal builds on the role of attention components in controlling patch interactions. Targeted manipulation is intended to improve generalization and ASR.

We also analyze the mechanisms behind these improvements. The work reinforces the importance of VLM vulnerabilities and offers a more robust and versatile framework for transferable adversarial examples.

**Claims and evidence**

- CroPA outperforms baselines across target texts: Table 1, each target’s Overall column; CroPA values are 0.7708, 0.7985, 0.5033, 0.7020, 0.6920, and 0.8680. The statement about meaning and word frequency is the section’s stated reproduction claim.
- Prompt-count and convergence findings: Figures 4–6; the text gives no numerical curve values.
- CroPA remains effective with in-context examples: Table 2, shot=2 rows, Overall 0.6635 for CroPA and 0.5730 for Multi-P.
- The extensions improve reported attack performance: Tables 3, 5, 6, and 8; detailed comparisons appear in Section 6.5.

## 2 Related Works

*Prior work establishes several transferability dimensions and motivates CroPA’s extensions.*

Adversarial attacks on deep neural networks have been studied since Szegedy et al. (2013) and Goodfellow et al. (2014). Their application to VLMs presents distinct challenges and opportunities. This section reviews adversarial transferability and positions CroPA within that literature.

### 2.1 Adversarial Transferability

*Adversarial examples can transfer between models, images, and tasks.*

Liu et al. (2017) and Tramèr et al. (2017) establish that examples crafted for one model can deceive other models, including models with different architectures. Moosavi-Dezfooli et al. (2017) and Mopuri et al. (2017) show that perturbations can also generalize across images, enabling UAPs.

Cross-task transferability means that examples designed for classification can affect object detection and other visual tasks. Naseer et al. (2019b), Naseer et al. (2019a), Lu et al. (2020), and Salzmann et al. (2021) study this setting. These dimensions raise questions about machine-learning robustness.

### 2.2 Adversarial Attacks on Vision-Language Models

*Earlier vision-language attacks were task-specific, while recent work examines modern large VLMs.*

Early work focuses on task-specific attacks. Xu et al. (2019), Zhang et al. (2020), Aafaq et al. (2021), and Chen et al. (2017) manipulate image-captioning outputs. Visual Question Answering (VQA), answering questions about images, is also vulnerable to targeted attacks on attention mechanisms: Xu et al. (2018), Kaushik et al. (2021), Kovatchev et al. (2022), Li et al. (2021), Sheng et al. (2021), and Zhang et al. (2022b).

These approaches mainly target lightweight CNN-RNN architectures, which lack in-context learning: adaptation through supplied examples. This limits their applicability to contemporary VLMs. Zhao et al. (2023) study modern BLIP and BLIP-2 in black-box settings, but focus on cross-model rather than cross-prompt transferability.

### 2.3 Cross-Prompt Adversarial Transferability

*Prompt-driven task adaptation creates an attack surface addressed by CroPA and CIA.*

Large VLMs adapt to tasks through textual prompts rather than performing only a fixed task: Li et al. (2023b), Awadalla et al. (2023), Li et al. (2023b), Zhu et al. (2023), and Liu et al. (2023). This creates an attack surface requiring adversarial images to remain effective across prompts.

CroPA, introduced by Luo et al. (2024), optimizes learnable prompts that compete with image perturbations. This differs from independently manipulating the two modalities. The concept is new enough that no credible benchmarks or baselines have yet been established.

Yang et al. (2024) also examine cross-prompt transferability. Their **Context Injection Attack (CIA)** uses gradient-based perturbations to inject target tokens into visual and textual contexts. We include it as an additional comparison.

### 2.4 Enhancing Adversarial Transferability

*Input diversity, augmentation, and gradient-focused optimization motivate the proposed enhancements.*

Input diversity, variation in attack inputs, and data augmentation have improved transferability in Xie et al. (2019) and Wang et al. (2021). Li et al. (2023a) and Wang et al. (2023) identify gradient misalignment as a limiting factor and propose optimization strategies. We build on these insights to extend CroPA across transferability dimensions.

## 3 Scope of reproducibility

*The study tests four CroPA claims and extends evaluation to models and images.*

We aim to validate Luo et al. (2024) by carefully reproducing their procedures. The main objective is to confirm that CroPA significantly improves adversarial transferability across prompts.

The claims are:

- **Claim 1:** CroPA achieves cross-prompt transferability across various target texts.
- **Claim 2:** CroPA achieves the best overall performance across different number of prompts.
- **Claim 3:** CroPA converges towards higher ASR as number of iterations are increased.
- **Claim 4:** CroPA outperforms baselines under few-shot settings.

Beyond replication, we investigate cross-model and cross-image effectiveness and compare CroPA with other cross-prompt methods. We ask whether its adversarial images consistently deceive diverse VLMs regardless of prompt or model parameters. These tests reproduce the original experiments and explore CroPA’s broader applicability to robust and versatile adversarial attacks.

## 4 Methodology

*The method formulates targeted and non-targeted attacks, defines baselines, and reproduces CroPA.*

### 4.1 Problem Formulation

*The attack optimizes bounded image perturbations across textual prompts.*

Following Goodfellow et al. (2014), Luo et al. (2024) extend adversarial attacks to cross-prompt VLM settings. Their VLM function \(f\) takes a visual input \(x_v\) and a textual input \(x_t\).

The visual perturbation \(\delta_v\) is constrained by
\[
\|\delta_v\|_p\leq\epsilon_v,
\]
where \(\|\cdot\|_p\) is the perturbation norm and \(\epsilon_v\) is its allowed magnitude. The authors use this bound to keep perturbations human-imperceptible, following multimodal adversarial practice discussed by Carlini et al. (2024).

We reproduce two scenarios:

1. **Targeted:** Generate a predetermined text \(T\), regardless of prompt. Equation 1 minimizes language modeling loss \(L\) over \(k\) prompt instances \(x_t^i\).
2. **Non-targeted:** Maximize the discrepancy between clean and adversarial outputs, as in Equation 2.

\[
\min_{\delta_v}\sum_{i=1}^{k}
L(f(x_v+\delta_v,x_t^i),T)
\tag{1}
\]

\[
\max_{\delta_v}\sum_{i=1}^{k}
L(f(x_v+\delta_v,x_t^i),f(x_v,x_t^i))
\tag{2}
\]

ASR measures effectiveness. A targeted attack succeeds only if the model generates the exact target text. A non-targeted attack succeeds if its output differs from the clean image’s prediction.

### 4.2 Baseline Approaches

*Single-P uses one prompt, while Multi-P optimizes over a prompt collection.*

Cross-prompt transferability has no established literature baselines. We therefore use the two baselines introduced by Luo et al. (2024).

**Single-P** optimizes an image perturbation with one prompt:

\[
\min_{\delta_v}L(f(x_v+\delta_v,x_t),T)
\tag{3}
\]

**Figure 1:** Framework overview of CroPA as presented in Luo et al. (2024). The architecture employs learnable perturbations for both image (\(\delta_v\)) and prompt (\(\delta_t\)) inputs. These perturbations operate antagonistically, while \(\delta_v\) is optimized to minimize the language modeling loss, \(\delta_t\) works to maximize it. The model’s forward pass (solid arrows) processes the perturbed inputs through the vision encoder and language model, with backpropagation (dashed arrows) updating the perturbations at configurable frequencies. The model parameters (marked with *) remain fixed during the attack. \(\oplus\) denotes element-wise addition.

Here, \(\delta_t\) is the prompt perturbation. A forward pass processes the inputs; backpropagation updates perturbations using gradients. Solid and dashed arrows distinguish these paths. The `*` marks fixed parameters, and \(\oplus\) means element-wise addition.

**Multi-P** uses a prompt collection \(X_t=\{x_t^1,x_t^2,\ldots,x_t^k\}\):

\[
\min_{\delta_v}\sum_{i=1}^{k}
L(f(x_v+\delta_v,x_t^i),T)
\tag{4}
\]

For non-targeted attacks, the objectives instead maximize clean–perturbed output discrepancy. These baselines provide the reference for CroPA’s cross-prompt performance.

### 4.3 Cross Prompt Attack

*CroPA makes image and prompt perturbations compete, then evaluates only the perturbed image.*

CroPA uses learnable prompt perturbations to improve cross-prompt transferability. Image and prompt perturbations compete rather than collaborate. The image perturbation minimizes the loss for generating the target; the prompt perturbation maximizes it. This opposition forces the image perturbation to become more transferable.

The min-max objective is:

\[
\min_{\delta_v}\max_{\delta_t}
L(f(x_v+\delta_v,x_t+\delta_t),T)
\tag{5}
\]

Here, \(f\) is the VLM, \(T\) is the target text, and \(L\) is the language modeling loss.

The implementation uses **Projected Gradient Descent (PGD)**, iterative gradient updates projected into the permitted perturbation region, following Madry et al. (2017). Images use gradient descent to minimize loss. Prompts use gradient ascent to maximize it. The interval \(N\) controls frequency: there are \(N\) image updates per prompt update. Figure 1 shows the framework.

**Algorithm 1 CroPA: Cross Prompt Attack**

**0: Inputs:** Model \(f\), Target Text \(T\), vision input \(x_v\), prompt set \(X_t\), perturbation size \(\epsilon\), step sizes \(\alpha_1\) and \(\alpha_2\), number of iteration steps \(K\), adversarial prompt update interval \(N\).

**0: Output:** Adversarial example \(x_v'\).

1. Initialize \(x_v'=x_v\).
2. **for** step = 1 to \(K\) **do**
3.   Uniformally sample the prompt \(x_t^i\) from \(X_t\).
4.   **if** \(x_t^i\) is not initialised **then**
5.     Initialise \(x_t^{\prime i}=x_t^i\).
6.   **end if**
7.   Compute gradient for adversarial image:
   \[
   g_v=\nabla_{x_v}L(f(x_v',x_t^i),T).
   \]
8.   Update with gradient descent:
   \[
   x_v'=x_v'-\alpha_1\cdot\operatorname{sign}(g_v).
   \]
9.   **if** \(\operatorname{mod}(\text{step},N)=0\) **then**
10.     Compute gradient for adversarial prompt:
    \[
    g_t=\nabla_{x_t}L(f(x_v',x_t^i),T).
    \]
11.     Update with gradient ascent:
    \[
    x_t^{\prime i}=x_t^{\prime i}+\alpha_2\cdot\operatorname{sign}(g_t).
    \]
12.   **end if**
13.   Project \(x_v'\) to be within the \(\epsilon\)-ball of \(x_v\):
    \[
    x_v'=\operatorname{Clip}_{x_v,\epsilon}(x_v').
    \]
14. **end for**
15. **return** \(x_v'\).

In the algorithm, \(x_v'\) is the adversarial image and \(x_t^{\prime i}\) is the updated prompt. The gradients are \(g_v\) and \(g_t\), with \(\nabla\) denoting differentiation. The step sizes are \(\alpha_1\) and \(\alpha_2\), and \(K\) is the iteration count. The `sign` operation uses gradient signs; `mod` triggers periodic prompt updates; `Clip` enforces the perturbation bound.

Evaluation discards prompt perturbations and applies only the optimized image perturbation. Thus, attack effectiveness must come from the image’s transferability rather than modified prompts.

## 5 Additional Investigative Experiments

*The extensions address task inconsistency, computation costs, and limited transferability.*

CroPA’s PGD optimization effectively increases ASR on a single model. However, reliance on that model’s loss feedback limits transferability. Similar limitations motivate Qin et al. (2024), Li et al. (2024), Wang et al. (2023), and Li et al. (2023a). The original authors recognize the limitation, and Table 9 in Appendix A.1 verifies it. Table 1 also shows especially poor performance on some Classification and Captioning tasks.

CroPA is computationally intensive. Training perturbations for only 50 images takes over 6 hours on a 48GB L40S GPU, the computing device used, while using 95% of VRAM, its available graphics memory.

Our extensions target:

1. Inconsistent performance across tasks.
2. Cross-image and cross-model transferability.

Although broader transferability is not Luo et al.’s core proposal, it could reduce computation. Ideally, cross-image transfer would remove the need to optimize a separate perturbation for every image. Cross-model transfer could offer further efficiency.

**Claims and evidence**

- CroPA has transferability limitations: Table 9, Appendix A.1, as cited in this section.
- Classification and Captioning can be weak: Table 1, “not sure”/CroPA row, 0.2420 and 0.0010; “too late”/CroPA row, 0.7010 and 0.1790.
- CroPA is computationally intensive: the section reports over 6 hours, 50 images, a 48GB L40S GPU, and 95% VRAM use; no table is cited for this measurement.
- Transfer could reduce repeated optimization: stated as a potential benefit, not a measured result.

### 5.1 Noise Initialization via Vision Encoding Optimization

*The proposed initialization aligns initial perturbations with a synthesized target image.*

CroPA’s random noise initialization, shared with many gradient-based attacks, does not use target semantics. Iterative optimization starts without semantic guidance, causing instability and slower convergence. This is especially problematic for VLMs, where prompt alignment matters.

Section 6.5.1, Figure 7, and Table 3 show that CroPA’s bi-level optimization—its opposing image and prompt objectives—is highly sensitive to initialization. The method cannot guide the initial trajectory toward semantically meaningful directions. Conventional initialization therefore yields suboptimal ASR and reduced cross-prompt transferability.

We hypothesize that semantically informed initialization can improve attack efficacy while preserving imperceptibility. A diffusion model synthesizes a target image corresponding to the desired prompt. The vision encoder’s feature space, its image-representation space, then provides an alignment target.

The method:

- Generates a semantically relevant image with a state-of-the-art diffusion model, such as Stable Diffusion XL.
- Aligns the perturbation with that image’s vision-encoder representation.
- Uses a **Mean Squared Error (MSE)** loss, the squared discrepancy between representations, to enforce semantic similarity while maintaining adversarial effectiveness.

Replacing random initialization with this vision-guided initialization improves adversarial robustness and attack efficiency. It reduces the steps needed for convergence and improves cross-prompt transferability. Perturbations are semantically informed from the start.

**Claims and evidence**

- Initialization affects attack performance: Table 3, CroPA and CroPA+Init rows; Overall values are 0.7708/0.8977, 0.8433/0.9260, 0.7985/0.9458, 0.8190/0.8761, and 0.8535/0.9333.
- The method improves convergence and semantic guidance: Figure 7(a), convergence comparison; Figure 7(b), CLIP scores 21.80 for CroPA and 23.09 for CroPA+Init.

#### 5.1.1 Noise Initialization

*Preliminary tests report stronger attacks with a small per-image time increase.*

Preliminary BLIP-2 and InstructBLIP experiments suggest that semantically informed noise improves attack efficacy over random initialization. It adds only 20-25 seconds per image on a single GPU. The diffusion approach improves vision-feature alignment, preserves visual fidelity, and respects strict perturbation budgets.

**Claims and evidence**

- Semantically informed initialization improves efficacy and preserves fidelity under the budget: preliminary BLIP-2 and InstructBLIP experiments described here; no numerical efficacy or fidelity values are stated in this subsection.
- Added computation is limited: 20-25 seconds for one image on one GPU; no table is cited.

#### 5.1.2 Diffusion-Based Target Synthesis

*Stable Diffusion XL generates the target image from the desired prompt.*

For target prompt \(T\), we generate an image \(\mathbf{x}_{\text{target}}\) representing its semantic attributes. We use **Stable Diffusion XL (SDXL)**:

\[
\mathbf{x}_{\text{target}}
=\mathcal D(T,\mathbf z;\theta_{\text{SDXL}}),
\tag{6}
\]

where \(\mathcal D\) is the diffusion process, \(\mathbf z\) is Gaussian-sampled input, and \(\theta_{\text{SDXL}}\) contains pretrained model weights. The target image captures the prompt’s semantic essence and provides an informative initialization basis.

#### 5.1.3 Vision-Encoder Anchored Perturbation

*Initialization minimizes the difference between perturbed-image and target-image encoder outputs.*

Let \(f_v(\cdot)\) be the VLM’s vision encoder. For original image \(\mathbf x\), we seek a perturbation \(\bm\delta\) such that \(\mathbf x+\bm\delta\) resembles the target image in the encoder’s output space:

\[
\bm{\delta}_{\text{init}}
=\underset{\|\bm{\delta}\|_\infty\leq\epsilon}{\arg\min}
\left\|f_v(\mathbf x+\bm\delta)
-f_v(\mathbf{x}_{\text{target}})\right\|_2^2.
\tag{7}
\]

Here, \(\bm{\delta}_{\text{init}}\) is the resulting initial perturbation. The maximum permitted magnitude is \(\epsilon\), enforced through the \(\ell_\infty\) constraint to ensure imperceptibility. The squared norm measures encoder-output discrepancy.

This starts the example in a semantically meaningful neighborhood of the target representation. Algorithm 2 in Appendix B formally describes the method.

#### 5.1.4 Adversarial Optimization via PGD

*Projected updates refine the initialized image using the encoder-output MSE.*

After initialization, PGD refines the adversarial image. At iteration \(t\):

\[
\mathbf{x}_{\text{adv}}^{(t+1)}
=\Pi_{\mathcal B_\epsilon(\mathbf x)}
\Bigl[
\mathbf{x}_{\text{adv}}^{(t)}
-\alpha\nabla_{\mathbf{x}_{\text{adv}}^{(t)}}
\mathcal L_{\text{MSE}}
\Bigr].
\tag{8}
\]

The image at iteration \(t\) is \(\mathbf{x}_{\text{adv}}^{(t)}\), and \(\alpha\) is the step size. The projection \(\Pi_{\mathcal B_\epsilon(\mathbf x)}\) returns the update to the admissible \(\ell_\infty\)-ball \(\mathcal B_\epsilon(\mathbf x)\) around the original image.

The MSE loss is:

\[
\mathcal L_{\text{MSE}}
=\left\|
f_v(\mathbf{x}_{\text{adv}})
-f_v(\mathbf{x}_{\text{target}})
\right\|_2^2.
\tag{9}
\]

Each update aligns the adversarial image incrementally with the target’s semantic embedding, its encoder representation.

### 5.2 Guiding perturbations via Target Value Vectors and evaluating Cross-Model Tranferability

*The proposed loss targets vision-attention value vectors to improve attack strength and transferability.*

CroPA optimizes image and text inputs with PGD but lacks sensitivity to the vision encoder’s internal dynamics. The encoder plays a dominant role in grounding VLM semantics. Treating vision and language as uniformly vulnerable overlooks modality-specific weaknesses. This limits structured, transferable perturbations and their influence on the encoder’s representational backbone.

CroPA minimizes image-perturbation loss and maximizes text-perturbation loss to steer output toward a target. Cui et al. (2023) and Wang et al. (2024) show that perturbing vision-encoder output embeddings can impair the visual perception of **large vision-language models (LVLMs)** and mislead their language outputs. LVLMs often use similar image encoders. These encoders produce **visual tokens**, representations carrying semantic and structural image information for language decoders.

We combine CroPA’s loss with a loss targeting the vision encoder’s attention mechanism. This mechanism determines interactions among image patches. We focus on **value vectors**, which hold each patch’s information. We hypothesize that perturbing value vectors between an original image and a target-text reference image improves the **Adversarial Success Rate (ASR)** under CroPA’s existing hyperparameter constraints.

The approach adapts **Doubly-Universal Adversarial Perturbation (Doubly-UAP)**. Instead of maximizing losses between value vectors, we minimize losses between those of \(x_i\) and \(T_i\), the input and target instances named in this description. Targeting vision-layer values should disrupt interpretation more effectively than generic PGD perturbations. Freezing the text encoder, the text-processing component, keeps the perturbations focused on visual processing rather than text artifacts.

The modification aims to:

1. Improve effectiveness by exploiting vision-encoder information.
2. Improve alignment with the target text representation.
3. Improve adversarial robustness across models while remaining computationally feasible.

Jointly optimizing CroPA and modified Doubly-UAP losses yields a more structured, interpretable, and targeted attack.

The component comes from Kim et al. (2024), who identify which attention components most influence VLM performance. We consider:

1. **Attention Weights:** These determine how much each patch focuses on others and thus control patch interaction or relevance. Targeting them could interfere with those relationships.
2. **Value Vectors:** These hold patch information. Perturbing them could disrupt essential content and interpretation.

Based on prior analysis, we target value vectors in middle-to-late vision-encoder layers.

The standard Doubly-UAP perturbation is:

\[
\delta^*
=\arg\max_\delta
\frac{1}{|L|}
\sum_{l\in L}
\operatorname{Loss}(V_l(x),V_l(x+\delta)).
\tag{10}
\]

Here, \(\delta^*\) is the optimized perturbation, \(x\) is the input image, and \(V_l(x)\) denotes its value vectors at layer \(l\). The sum uses selected layers \(L\), whose count is \(|L|\). The function \(\operatorname{Loss}(\cdot)\) is applied to the target vectors.

#### 5.2.1 Our Modified Approach

*The modified loss uses a target reference image and is combined with CroPA’s objective.*

We introduce target value vectors from a reference image corresponding to desired text \(T\). Rather than only maximizing deviation from original values, we enforce alignment with a predefined target representation.

**Figure 2:** D-UAP Architecture Diagram

Our steps are:

1. Extract the vision encoder’s layer-\(l\) value vectors for the input and target-text reference image.
2. Compute cosine similarity loss between them.
3. Jointly minimize -ve of this loss with CroPA’s loss, using suitable scaling for adversarial robustness and target alignment.

Here, **D-UAP** denotes the adapted value-vector method, and `-ve` denotes the negative sign in the loss description.

Let \(V_i(x+\delta)\) be the perturbed value vectors of attention head \(i\), and let \(V_t\) be those of the target reference image. An attention head is the component indexed in this loss. We define:

\[
L_{\text{d-UAP}}
=\sum_{i=1}^{N}
\left(
1-
\frac{V_i(x+\delta)\cdot V_t}
{\|V_i(x+\delta)\|_2\|V_t\|_2}
\right).
\tag{11}
\]

Here, \(N\) is the number of attention heads. The dot product and vector norms form the cosine-similarity expression. This loss encourages alignment with the target text representation.

The combined objective is:

\[
L_{\text{re-CroPA}}
=L_{\text{CroPA}}(x_v+\delta_v,x_t+\delta_t,T)
-\lambda L_{\text{d-UAP}}(\delta_v).
\tag{12}
\]

Here, \(L_{\text{CroPA}}\) is CroPA’s loss, \(L_{\text{d-UAP}}\) is the proposed value-vector loss, and \(L_{\text{re-CroPA}}\) labels their combination. The hyperparameter \(\lambda\) sets the value-vector loss’s relative importance.

Gradients from both components update \(\delta\). The objective preserves adversarial behavior while enforcing target alignment, cross-prompt transferability, and guided semantic influence. This offers more precise, controlled, and interpretable manipulation for robustness and security analysis. Algorithm 3 in Appendix B gives the steps.

The method is intended to improve transferability. However, its dependence on model value vectors and the vision encoder leads us to hypothesize that transfer may be limited to models with the same or similar vision-encoder architectures.

### 5.3 Transferability across images via Image Augmentation

*SCMix and CutMix are introduced to reduce image-specific overfitting.*

Traditional iterative gradient attacks optimize individual images and can overfit their features. Their weak cross-image transfer also makes them expensive. CroPA likewise learns a perturbation for one image across prompts.

Luo et al. (2024) describe CroPA as orthogonal to cross-image approaches. They suggest that optimizing across images, as in Moosavi-Dezfooli et al. (2017), can improve cross-image transfer. We test this claim and investigate an additional UAP-based approach.

Fang et al. (2024) introduce UAPs for **Vision-Language Pretraining (VLP)** models to reduce overfitting and enable cross-image transfer. **CutMix**, from Yun et al. (2019), instead regularizes through regional replacement. It cuts rectangular patches from one image and pastes them into another, mixing labels in proportion to patch area. Learning from partial views encourages spatially invariant features, improving robustness to attacks and occlusion.

Zhang et al. (2024) propose **Effective and Transferable Universal Adversarial Attack (ETU)**. It improves UAP transfer across VLP models and tasks through global and local optimization. Its **SCMix** augmentation combines **self-mixing**, mixing variants of one image, with **cross-mixing**, mixing that result with another image. It targets spatial and scale invariance: reduced sensitivity to position and size changes.

We integrate SCMix because ETU’s results show that, for the same model, SCMix alone improves cross-image transfer without the additional local utility reinforcement.

#### 5.3.1 SCMix

*SCMix blends cropped variants of a base image, then adds a smaller contribution from another image.*

SCMix uses two stages. Self-mixing forms variants of base image \(I_1\) from randomly cropped and resized patches:

\[
I_1'
=\eta X_1+(1-\eta)X_2
\quad\text{where}\quad
X_i=\operatorname{Resize}(\operatorname{RandomCrop}(I_1)),
\quad\eta=0.5.
\tag{13}
\]

The operations `RandomCrop` and `Resize` create variants \(X_i\); \(I_1'\) is their self-mixed result. The mixing weight is \(\eta\).

Original SCMix samples
\[
\eta\sim\operatorname{Beta}(\alpha,\alpha)
\]
for \(\alpha>0\), where \(\alpha\) controls the Beta distribution. We fix \(\eta=0.5\), described as equivalent to \(\alpha=\infty\). Appendix D.3 justifies this choice.

Cross-mixing combines \(I_1'\) with secondary image \(I_2\):

\[
I_3=\beta_1I_1'+\beta_2I_2
\quad\text{with}\quad
\beta_1\gg\beta_2\in[0,1).
\tag{14}
\]

Here, \(I_3\) is the final image. The weights \(\beta_1\) and \(\beta_2\), with \(\beta_1\) much greater than \(\beta_2\), preserve the base image’s dominant semantics while adding diversity. Thus, \(I_3\) remains compatible with \(I_1\)’s text description \(T_1\), supporting cross-modal interactions despite the added features.

Self-mixing forces attacks against features invariant to intra-image changes. Cross-mixing adds inter-image variation. Together, they reduce the overfitting possible in CroPA and promote universal perturbations for unseen images.

At inference, CroPA returns only a perturbed image, or equivalently its perturbation. If that perturbation transfers across images, the attack can become both cross-prompt and cross-image transferable.

**Figure 3:** Illustration of the self-mixing and cross-mixing processes. The input image \(I_1\) undergoes self-mixing to generate intermediate representations \(X_1\) and \(X_2\), which are then combined to form \(I_1'\). Additionally, an external image \(I_2\) is mixed with \(I_1'\) to produce the final output \(I_3\).

#### 5.3.2 CutMix

*CutMix uses hard regional replacement rather than SCMix’s continuous blending.*

For images \(x_A\) and \(x_B\), CutMix creates:

\[
\tilde{x}
=M\odot x_A+(1-M)\odot x_B.
\tag{15}
\]

Here, \(\tilde{x}\) is the composite image, \(M\in\{0,1\}^{W\times H}\) is a binary mask defining a randomly positioned rectangle, \(W\) and \(H\) are its image dimensions, and \(\odot\) denotes mask-based multiplication.

CutMix learns from partial views. Unlike SCMix’s soft blending within and across images, it uses hard mask-based replacement. This forces perturbations to target spatially invariant features that persist across image combinations. Integrated with adversarial training, it may encourage attacks on higher-level semantics rather than localized artifacts, potentially improving cross-image transfer.

### 5.4 Attack Setup and Threat Models for Transferability Experiments

*The threat models specify attacker goals, knowledge, constraints, and evaluation protocols.*

We define threat models—the assumed goals and capabilities of an attacker—for cross-model and cross-image evaluation.

#### 5.4.1 Using D-UAP to explore Cross-Model Transferability for CroPA

*The cross-model attacker optimizes with white-box access and transfers the perturbation to other models.*

**Adversary’s Goal:** Generate one perturbation for an image that makes multiple VLMs produce a specified target text regardless of prompt. This is a practical black-box transfer scenario: optimize on an accessible model, then deploy against unknown models.

**Knowledge and Capabilities:** The attacker has white-box access to a **surrogate model**, the accessible optimization model, either BLIP-2 or Flamingo. This includes gradients and internal representations. For **target models**, the evaluation models, only the general architecture family, such as CLIP-based, is known. Parameters are unavailable. The budget is \(\epsilon=16/255\) under the \(L_\infty\) norm to ensure visual imperceptibility.

**Cross-Model Settings:**

- **Intra-family transfer:** Optimize on BLIP-2 with the OPT-2.7b language model and evaluate on InstructBLIP with Vicuna-7b. Both have similar CLIP ViT-L/14 vision encoders but different language models and fine-tuning objectives. These are the model and encoder codes used for this comparison.
- **Cross-architecture transfer:** Transfer from BLIP-2 to Flamingo and vice versa. Their vision encoders, language models, and architectural designs differ.

**Evaluation Setup:** Measure ASR separately for VQAgeneral, VQAspecific, Classification, and Captioning. VQAgeneral and VQAspecific distinguish general from image-specific prompts. Success means the target model fails to generate the correct output from the perturbed image, regardless of prompt.

#### 5.4.2 Investigating Cross-Image Transferability via Augmentation Techniques

*The cross-image attacker learns one bounded perturbation from training images and tests it on unseen images.*

**Adversary’s Goal:** Generate one perturbation pattern that makes a VLM produce a predetermined target text for any image and prompt. This is more efficient than per-image optimization.

**Knowledge and Capabilities:** The attacker has a limited training-image set, corresponding SCMix variants, and white-box access to the target model during optimization. The resulting perturbation must generalize to unseen images. Its \(L_\infty\) budget is \(\epsilon=16/255\).

**Evaluation Setup:** Evaluate on separate images not used for training. Measure ASR by task. An attack succeeds when the model fails to generate the correct output text for a previously unseen image, regardless of content or prompt.

SCMix is intended to counter the image-feature overfitting of gradient methods. Controlled self- and cross-mixing should help perturbations capture vulnerabilities independent of a particular image.

#### 5.4.3 Evaluation Rationale for Transferability experiments

*Untargeted evaluation is used to compare architectures, and CIA is excluded from transfer comparisons.*

We present all cross-prompt and cross-model transferability results under the untargeted setting. Two considerations motivate this choice.

First, VLMs use different visual embedding mechanisms. This makes it difficult to identify how input perturbations affect final output-token embeddings. Targeted transfer is observed to be higher between similar encoders, such as BLIP-2 and InstructBLIP, but varies sharply across architectures. Direct targeted-ASR comparison across heterogeneous models is therefore imprudent.

Second, visual-textual fusion affects attack control. **Decoder-only VLMs**, which use visual features as context during causal language generation, include LLaVA and OpenFlamingo. They incorporate features late through cross-attention or token concatenation, mechanisms for supplying visual context. Those features serve as auxiliary memory, and their influence fades during decoding: Awadalla et al. (2023), Liu et al. (2023).

By contrast, **fusion-based models**, which integrate visual and textual representations before generation, include BLIP-2, Li et al. (2023b), and InstructBLIP, Dai et al. (2023). Their early, deep integration provides persistent visual grounding. Targeted attacks are therefore inherently more feasible in these models than in decoder-only models, Dai et al. (2023).

For these reasons, non-targeted ASR offers a more robust, fair, and comparable evaluation across models and images.

We also exclude CIA from transfer comparisons. Experiments yielded negligible transferability, and the method has these limitations:

- **Image-Specific Semantic Embedding:** CIA manipulates each image’s visual tokens to embed target concepts. Gradient directions and token spaces differ greatly across images and concepts. Input-to-target token transformations are non-transferable, making the resulting patterns image-dependent.
- **Model-Specific Embedding Architectures:** CIA depends heavily on visual and textual encoder structure. Architectural differences among LVLMs, including BLIP2, LLaVA, and Flamingo, prevent cross-model transfer.

**Claims and evidence**

- Targeted transfer is higher between similar encoders but varies across architectures: the qualitative observation stated here; Table 8 later reports cross-model results under the transfer-evaluation rationale.
- Visual fusion affects targeted feasibility: the section cites Awadalla et al. (2023), Liu et al. (2023), and Dai et al. (2023), without numerical comparisons here.
- CIA has negligible transferability: the experimental observation stated here; no numerical transfer values are reported in this subsection.

## 6 Results

*The experiments reproduce CroPA’s claims and evaluate the proposed extensions.*

A major objective is to reproduce Luo et al.’s cross-prompt results, as stated in Section 4. We compare reproduced findings with the original paper.

**Models Used:**

In reproducing Luo et al. (2024), we evaluated five state-of-the-art VLMs: OpenFlamingo, BLIP-2, InstructBLIP and LLaVA.

For Flamingo, we use **OpenFlamingo-9B**, the publicly accessible implementation from Awadalla et al. (2023). It offers performance comparable to the original model.

BLIP-2 uses a two-stage fusion-based approach. A frozen **CLIP image encoder**, the image-representation component, first extracts visual features. A **Querying Transformer**, the feature-processing component, then processes them, Li et al. (2023b). The architecture adapts efficiently to varied vision-language tasks and uses OPT-2.7b for flexible text generation.

InstructBLIP adds **instruction tuning**, training for task-specific instruction following, to BLIP-2’s fusion-based architecture, Dai et al. (2023). It uses Vicuna-7b, enabling more precise output control through prompts.

LLaVA, Liu et al. (2023), uses decoder-only generation. A pretrained CLIP ViT-L encoder connects to Vicuna-13B through a lightweight projection layer. Visual features enter the language-model context as special prompt tokens.

Thus, OpenFlamingo and LLaVA treat visual embeddings as generation context, while BLIP-2 and InstructBLIP explicitly align image and text before generation. These differences significantly affect targeted attack effectiveness across prompts.

We use the authors’ public code and preserve original model configurations for faithful comparison. Although the implementation is generally well documented, BLIP-2 and InstructBLIP caused multiple runtime errors. Significant debugging and modifications were needed to execute them.

### 6.1 Claim 1: CroPA achieves cross-prompt transferability across various target texts. [Reproduced]

*CroPA is evaluated across target texts and vision-language tasks on Flamingo.*

We replicate Luo et al.’s setup to test CroPA’s central transferability claim. We train adversarial examples and evaluate them across target prompts on Flamingo. The metric is targeted ASR for general and specific VQA, classification, and captioning.

#### 6.1.1 Experimental Details

*Evaluation uses target prompts distinct from those used during training.*

Following Luo et al. (2024), we generate CroPA adversarial examples and test target prompts different from training prompts. Table 1 lists targets selected to span different semantic meanings. Appendix D.1 gives details for all reproduced experiments.

#### 6.1.2 Comparative Analysis

*The results support cross-prompt transferability, but Classification and Captioning remain weaker.*

Table 1 reports targeted ASRs on Flamingo. The results broadly support cross-prompt transferability. Non-zero CroPA ASRs occur in every evaluated task, showing that adversarial examples can mislead the VLM under diverse target prompts.

CroPA nevertheless consistently under-performs on Classification and Captioning, especially for “too late” and “not sure.” These texts have abstract meanings and lack significant visual semantics.

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

**Table 1:** Targeted ASRs tested on Flamingo with different target texts, across a variety of Vision-Language tasks. Here VQAgeneral and VQAspecific, refer to the degree of specificity of prompts pertaining to a particular image. The prompts used are reported in Appendix E

The experiments reproduce the core finding of robust cross-prompt transferability and validate VLM vulnerability to these attacks.

**Claims and evidence — 6.1.2**

- CroPA has non-zero ASR in every evaluated task: Table 1, all CroPA rows; values include Captioning 0.0010 for “not sure” and range through VQAspecific 0.9940 for “metaphor.”
- CroPA improves overall results over baselines: Table 1, Overall column, Single-P/Multi-P/CroPA rows: “unknown” 0.2865/0.6095/0.7708; “I am sorry” 0.3640/0.7318/0.7985; “not sure” 0.2308/0.3930/0.5033; “very good” 0.3100/0.6175/0.7020; “too late” 0.2455/0.5918/0.6920; “metaphor” 0.3973/0.7933/0.8680.
- Classification and Captioning are weaker for abstract targets: Table 1, “not sure”/CroPA, 0.2420 and 0.0010; “too late”/CroPA, 0.7010 and 0.1790. The semantic explanation is the section’s interpretation.

**Claims and evidence — 6.1**

- Claim 1 is reproduced: Table 1, CroPA rows for all six target texts; Overall 0.7708, 0.7985, 0.5033, 0.7020, 0.6920, and 0.8680.

### 6.2 Claim 2: CroPA achieves the best overall performance across different number of prompts [Reproduced]

*CroPA outperforms baselines at equal prompt counts, and the baseline saturates below it.*

Figure 4 compares baseline and CroPA performance across prompt counts. ASR increases with the number of prompts for all methods. This comparison is redundant for Single-P because it uses one prompt, represented by the lone star in Figure 4.

At the same prompt count, CroPA consistently achieves higher ASR than the baselines. This supports its cross-prompt motivation. We use “unknown” as the target to avoid common high-frequency vision-language responses, following Luo et al.’s argument.

As prompt count grows, the baseline saturates below CroPA. Adding prompts therefore does not let the baseline surpass CroPA, consistent with Luo et al. (2024).

**Figure 4:** Targeted ASR tested with different number of prompts on Flamingo. With the same number of attack iterations, our CroPA significantly outperforms baselines. "unknown" is used as the target text.

**Claims and evidence**

- More prompts increase ASR, and CroPA performs better at equal prompt counts: Figure 4; the comparison uses Flamingo, target “unknown,” and the same number of attack iterations. No numerical curve values are stated.
- The baseline saturates below CroPA: Figure 4; no numerical saturation values are stated.
- Single-P has one plotted point because it uses one prompt: Figure 4’s lone star.

### 6.3 Claim 3: CroPA converges towards higher ASR as number of iterations increase [Reproduced]

*Overall convergence is reproduced, but individual tasks show instability.*

Our convergence investigation gives results that contrast with the original convergence metrics in Appendix A.2. We use the same training and evaluation parameters and the authors’ code. The original metrics showed CroPA converging with a higher number of prompts.

Overall ASR across tasks follows that trend, as Figure 6 shows, reproducing the form of the original figures. However, task-by-task ASR is prone to instability, as Figure 5 shows.

The original authors did not specify the model for this evaluation. We therefore report OpenFlamingo results with target text “unknown.”

(a) VQA

(b) VQA-Specific

(c) Classification

(d) Captioning

**Figure 5:** Comparison of Single-P, Multi-P, and CroPA across different categories.

**Figure 6:** Overall Convergence and Comparison of CroPA

**Claims and evidence**

- Overall ASR follows the reported convergence trend: Figure 6; OpenFlamingo, target “unknown.” No numerical curve values are stated.
- Individual task convergence is unstable: Figure 5, panels (a) VQA, (b) VQA-Specific, (c) Classification, and (d) Captioning.
- Overall convergence is reproduced using the authors’ code and matching parameters, while task-level instability is this study’s additional analysis: Figures 5–6 and Appendix A.2; the original evaluation model was unspecified.

### 6.4 Claim 4: CroPA outperforms baselines under few-shot settings [Reproduced]

*CroPA remains stronger than Multi-P when two in-context images are added at evaluation.*

Flamingo can use extra images as in-context learning examples to improve task adaptation. Luo et al. (2024) test whether attacks remain effective when evaluation uses a different number of examples from optimization.

Optimization uses **0-shot**, meaning no extra images. Evaluation uses **2-shot**, meaning two extra in-context images.

| Setting | Method | VQAgeneral | VQAspecific | Classification | Captioning | Overall |
|---|---|---:|---:|---:|---:|---:|
| shot=0 | Multi-P | 0.7240 | 0.8740 | 0.5550 | 0.2850 | 0.6095 |
| shot=0 | CroPA | 0.9680 | 0.9880 | 0.7070 | 0.4200 | 0.7708 |
| shot=2 | Multi-P | 0.7860 | 0.9420 | 0.5630 | 0.0010 | 0.5730 |
| shot=2 | CroPA | 0.9440 | 0.9940 | 0.6860 | 0.0300 | 0.6635 |

**Table 2:** Targeted ASRs with and without visual in-context learning. The shot indicates the number of images added for in-context learning.

In-context examples make attacks inherently harder because they shift generation conditions away from those used in optimization, Luo et al. (2024). They also make the model more aware of the correct prompt space, which differs from the target-text space. Zhao et al. (2023) and Qi et al. (2023) verify that context-specific attacks such as CroPA lose effectiveness when context changes.

CroPA nevertheless performs significantly better than the baseline in most tasks. It is more robust to the expanded context space. This verifies the motivation behind Section 4.3’s min-max objective: robustness to a wider prompt space and consequent context changes.

**Claims and evidence**

- Adding in-context examples reduces overall effectiveness: Table 2, Multi-P Overall 0.6095 at shot=0 and 0.5730 at shot=2; CroPA Overall 0.7708 and 0.6635.
- CroPA outperforms Multi-P under 2-shot evaluation: Table 2, shot=2 rows; VQAgeneral 0.9440 versus 0.7860, VQAspecific 0.9940 versus 0.9420, Classification 0.6860 versus 0.5630, Captioning 0.0300 versus 0.0010, Overall 0.6635 versus 0.5730.
- The result supports robustness to context changes: Table 2’s shot=2 comparison; the context-space explanation is the section’s interpretation.

### 6.5 Results Beyond the Original Paper

*The extensions test initialization, image augmentation, and target value-vector guidance.*

The following experiments extend CroPA with enhancements intended to mitigate its limitations.

#### 6.5.1 Noise Initialization

*Semantically informed initialization improves reported overall targeted ASR and convergence.*

Noise Initialization via Vision Encoding Optimization, described in Section 5.1, significantly improves performance across tasks. Semantically informed perturbation initialization gives superior adversarial robustness and cross-prompt transferability.

As Table 3 shows, **CroPA+Init**, CroPA with this initialization, consistently outperforms CroPA and Multi-P across all metrics, achieving 6-15% gains in overall accuracy.

These results validate semantic guidance over random noise. Vision-encoder-anchored perturbations improve ASR and efficiency by reducing optimization steps. The method offers a robust, adaptable alternative for attack initialization.

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

**Table 3:** Targeted ASRs on Blip2 with different target texts using CroPA Vision Encoder noise initialization. The detailed hyperparameters used are discussed under section D.2.

##### Theoretical Analysis of Initialization in Bi-level Optimization

*The analysis attributes initialization gains to min-max sensitivity, alternating updates, and semantic priors.*

Three factors explain the gains from diffusion-guided initialization.

First, CroPA resembles a **saddle-point problem**, the min-max form whose stability depends on initialization. Zhang et al. (2025) show that min-max problems are more initialization-sensitive than ordinary minimization and often need designed initialization for stable convergence. CroPA’s prompt perturbations continually change its optimization landscape, making this sensitivity especially pronounced.

Second, its alternating schedule updates image perturbations more often than prompt perturbations. This asymmetry increases initialization’s importance. Chen et al. (2019) find that alternating min-max gradient methods can exhibit **local cycling**, repeated local optimization behavior, and that convergence depends strongly on proximity to adversarially robust regions. Semantic initialization deliberately starts closer to meaningful adversarial directions.

Third, our empirical evidence agrees with Wu et al. (2020): perturbations initialized with semantic priors transfer better across models than randomly initialized perturbations, improving ASR. This effect is especially pronounced in complex bi-level optimization such as CroPA.

Our diffusion-based initialization starts in the direction of semantic targets rather than random noise. It enables faster convergence and improves transferability. Table 3 shows significant ASR gains.

| Method | VQAgeneral | VQAspecific | Classification | Captioning | Overall |
|---|---:|---:|---:|---:|---:|
| Multi-P | 0.7053 | 0.8807 | 0.7001 | 0.5553 | 0.7104 |
| CroPA | 0.8723 | 0.9457 | 0.7518 | 0.6138 | 0.7958 |
| CIA | 0.2985 | 0.2281 | 0.4857 | 0.4687 | 0.370 |
| CroPA+Init | 0.9145 | 0.9342 | 0.8982 | 0.9464 | 0.9209 |

**Table 4:** Average Targeted ASRs on BLIP-2 for different methods. Results show the impact on different vision-language tasks: Visual Question Answering (general and specific), Classification, and Captioning. The best performance values for each task are highlighted in bold. Apart from the previously mentioned baselines, we also compare our results with the Contextual Injection Attack (CIA) Yang et al. (2024) as an additional baseline.

Gradient-based attacks often acquire target semantics. In theory, they should not need semantically aligned initialization. However, the min-max complexities described above weaken gradient-produced alignment. Injecting target semantics at initialization is necessary to strengthen it and help convergence. Figure 7 provides experimental evidence.

We compare target attributes in perturbations using the **CLIP score**, a similarity score between the target text and the learned noise. We score perturbations from base CroPA and the **Init** enhancement, the initialization method.

CroPA acquires target context as expected for gradient-based attacks. Initialization produces a modest CLIP-score increase. This suggests a subtle shift toward adversarially vulnerable regions, rather than a large improvement in acquired target semantics.

The convergence comparison also supports the hypothesis that starting closer to meaningful adversarial directions yields faster convergence.

| Method | CLIP Score |
|---|---:|
| CroPA | 21.80 |
| CroPA+Init | 23.09 |

Target Text: "Bomb"

(a) (b)

**Figure 7:** (a) Overall Convergence Comparison of base CroPA and CroPA+Init. (b) Quantifying feature similarity of learned perturbation to target by CLIP score.

**Claims and evidence — Theoretical Analysis of Initialization in Bi-level Optimization**

- Initialization improves reported overall ASR: Table 3, CroPA/CroPA+Init Overall pairs: 0.7708/0.8977, 0.8433/0.9260, 0.7985/0.9458, 0.8190/0.8761, and 0.8535/0.9333.
- Average targeted performance improves: Table 4, Overall 0.9209 for CroPA+Init, 0.7958 for CroPA, 0.7104 for Multi-P, and 0.370 for CIA.
- Initialization modestly increases target-feature similarity: Figure 7(b), target “Bomb,” CLIP scores 21.80 for CroPA and 23.09 for CroPA+Init.
- Initialization speeds convergence: Figure 7(a); numerical convergence steps are not stated.
- Min-max sensitivity, alternating-update asymmetry, and semantic priors explain the gains: the section’s analysis cites Zhang et al. (2025), Chen et al. (2019), and Wu et al. (2020).

**Claims and evidence — 6.5.1**

- The section claims consistent improvement across all metrics and 6-15% gains in overall accuracy: Table 3, all target/method rows.
- Overall and task-average gains are reported: Table 4, CroPA+Init row, 0.9145, 0.9342, 0.8982, 0.9464, and 0.9209.
- Fewer optimization steps and modestly stronger semantic alignment are reported: Figure 7(a–b), with CLIP scores 21.80 and 23.09.

#### 6.5.2 Investigating Cross-Image transferability and Image Augmentation

*Augmentation improves cross-image ASR, but cross-image transfer remains a limitation.*

CroPA retains adversarial impact across prompts, but its signals remain tightly coupled to the source image’s low-level features. This prevents generalization to other images and makes universal cross-image attacks ineffective. The finding weakens the original authors’ claim that learning perturbations across images would provide cross-image transferability.

We integrate SCMix and CutMix, described in Section 5.3, to diversify image features during training. The aim is to move perturbations beyond a single image.

We believe CutMix outperforms SCMix in promoting adversarial transferability because its region-based mixing preserves distinct, non-overlapping areas. This encourages modular, independently manipulable features and more generalizable perturbations.

Table 5 shows measurable gains from both methods, although cross-image transfer remains difficult. The gap between cross-image methods and CroPA applied to one image shows a fundamental limitation: CroPA excels at cross-prompt robustness but does not effectively support transferable attacks across images.

| Method | VQA | VQAspecific | Classification | Captioning | Overall |
|---|---:|---:|---:|---:|---:|
| CroPA (Base) | 0.3500 | 0.4920 | 0.4000 | 0.5000 | 0.4355 |
| CroPA with SCMix | 0.6560 | 0.7160 | 0.5000 | 0.9500 | 0.7055 |
| CroPA with CutMix | 0.4520 | 0.7240 | 0.5000 | 0.5500 | 0.5565 |

**Table 5:** Untargeted ASR of CroPA in cross-image settings with and without SCMix and CutMix. The results represent the best ASRs achieved across all iterations. CroPA without input mixing attained its highest ASRs at 1700 iterations, while CroPA with SCMix and CutMix outperformed it significantly earlier, at just 1500 iterations and 1600 iterations, respectively. Additional details on experimental parameters are reported in Appendix D.3.

**Claims and evidence**

- Base cross-image performance is limited: Table 5, CroPA (Base) row, VQA 0.3500, VQAspecific 0.4920, Classification 0.4000, Captioning 0.5000, Overall 0.4355.
- Both augmentations improve cross-image results: Table 5, SCMix row 0.6560, 0.7160, 0.5000, 0.9500, 0.7055; CutMix row 0.4520, 0.7240, 0.5000, 0.5500, 0.5565.
- Augmentation reaches its best results earlier: Table 5 caption, base 1700 iterations, SCMix 1500, CutMix 1600.
- The section states that CutMix outperforms SCMix and gives a regional-feature explanation: this is the section’s interpretation; the exact comparisons are the SCMix and CutMix rows in Table 5.
- Source-image coupling and the weakness of the cross-image claim are interpretations of the reported cross-image results.

#### 6.5.3 Guiding perturbations via Target Value Vectors

*CroPA+D-UAP improves targeted attack performance on BLIP-2 and Open Flamingo.*

Tables 6 and 7 evaluate the D-UAP-modified framework from Section 5.2. **CroPA+D-UAP** combines CroPA with target value-vector guidance.

The enhancement improves the existing algorithm across the board. Overall ASR reaches 96.85%, a 12.5% increase over base CroPA. VQAgeneral and Captioning improve by 13% and 28%, respectively.

The target text is “Bomb.” This demonstrates improvement even for harmful text that is inherently set to trigger model **guardrails**, the model’s safety mechanisms. Thus, in the worse expected cases of harmful words, the enhancement makes the model perform significantly better in vulnerable scenarios.

| Method | VQAgeneral | VQAspecific | Classification | Captioning | Overall |
|---|---:|---:|---:|---:|---:|
| Multi-P | 0.68 | 0.82 | 0.90 | 0.60 | 0.75 |
| CroPA | 0.8176 | 0.9100 | 0.9498 | 0.6960 | 0.8433 |
| CIA | 0.3431 | 0.3102 | 0.4732 | 0.5534 | 0.4199 |
| CroPA+D-UAP | 0.9420 | 0.9720 | 0.9790 | 0.9810 | 0.9685 |

**Table 6:** Targeted ASRs on Blip2 using D-UAP function and target text being "Bomb", outperforming PGD baselines and CIA, as well as the base CroPA method.

| Method | VQAgeneral | VQAspecific | Classification | Captioning | Overall |
|---|---:|---:|---:|---:|---:|
| Multi-P | 0.6960 | 0.8540 | 0.8990 | 0.6060 | 0.7638 |
| CroPA | 0.7900 | 0.8860 | 0.9470 | 0.6700 | 0.8233 |
| CIA | 0.3027 | 0.4302 | 0.5112 | 0.5080 | 0.4380 |
| CroPA+D-UAP | 0.9000 | 0.9520 | 0.9500 | 0.9060 | 0.9270 |

**Table 7:** Targeted ASRs on Open Flamingo using D-UAP Loss function and target text being "Bomb".

**Claims and evidence**

- BLIP-2 overall ASR reaches 96.85%, described as a 12.5% increase: Table 6, CroPA+D-UAP Overall 0.9685 and CroPA Overall 0.8433.
- VQAgeneral and Captioning improve by the stated 13% and 28%: Table 6, CroPA/CroPA+D-UAP pairs 0.8176/0.9420 and 0.6960/0.9810.
- The enhanced method improves every reported task: Table 6, CroPA+D-UAP row 0.9420, 0.9720, 0.9790, 0.9810; Table 7, corresponding row 0.9000, 0.9520, 0.9500, 0.9060.
- Open Flamingo overall performance improves: Table 7, Overall 0.9270 for CroPA+D-UAP, 0.8233 for CroPA, 0.7638 for Multi-P, and 0.4380 for CIA.
- These results use harmful target text: Tables 6 and 7 captions specify “Bomb.” The guardrail interpretation is stated in the section.

#### 6.5.4 Analyzing Cross Model Transferability by Value Vector guidance

*Value-vector guidance improves cross-model results but remains dependent on the vision encoder.*

Table 8 makes a strong case for increased cross-model transferability. Gains over vanilla CroPA are especially large for Classification and Captioning, 14% and 21%, respectively, with modest VQA increases. We attribute the task difference to VQA prompts spanning a more complex embedding space than Classification or Captioning, whose outputs occupy smaller spans.

The results also verify the limitation hypothesized in Section 5.2. Value vectors depend on the particular vision encoder. Transfer can therefore be limited to models with the same or similar encoders.

In Table 8, BLIP2-trained perturbations transfer passably to InstructBLIP, which uses the same encoder, but fail to transfer to OpenFlamingo. The expected pattern also appears when Flamingo-trained perturbations are applied to other models, where ASR is insignificant. See Table D.4.5 in Appendix D.4.5. Appendix D.4 gives experimental details.

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

**Table 8:** The cross model test under different settings. "BLIP2 to InstructBLIP" means the perturbations optimised on BLIP2 are tested on the InstructBLIP and similarly for "BLIP2 to Flamingo"

Here, **Cls** and **Cap** abbreviate Classification and Captioning.

For a given target text, we also observe approximately consistent vision-encoder value vectors across input images. This has three implications:

1. **Vision Encoder Specificity:** Values are aligned with the encoder’s architecture and pretraining.
2. **Limited Diversity in Perturbations:** Consistent values may constrain perturbation diversity and the ability to produce varied adversarial examples.
3. **Target Text Dependency:** Similar values across images suggest that target-text choice influences attack outcomes more strongly than specific input-image features.

**Claims and evidence**

- Value-vector guidance improves cross-model overall ASR: Table 8, CroPA/CroPA+D-UAP Overall pairs: BLIP2→InstructBLIP 0.5065/0.6850; BLIP2→Flamingo 0.1435/0.2335; BLIP2→LLaVA 0.3040/0.5325; InstructBLIP→BLIP2 0.2288/0.4900; InstructBLIP→LLaVA 0.1345/0.3558; InstructBLIP→Flamingo 0.0598/0.1375; Flamingo→BLIP2 0.0335/0.1013; Flamingo→InstructBLIP 0.0290/0.0938; Flamingo→LLaVA 0.0370/0.1025.
- The section reports especially large Classification and Captioning gains of 14% and 21%: Table 8, Cls and Cap columns across the CroPA and CroPA+D-UAP rows; no particular row is assigned to those percentages in the text.
- Transfer depends on encoder similarity: Table 8, CroPA + D-UAP Overall 0.6850 for BLIP2→InstructBLIP and 0.2335 for BLIP2→Flamingo; the encoder-dependency explanation is the section’s interpretation.
- Flamingo-trained transfer is described as insignificant: Table 8, Flamingo-source CroPA+D-UAP Overall 0.1013, 0.0938, and 0.1025; the text also cites Table D.4.5.
- Value vectors are approximately consistent across images for a target text: a qualitative observation in this section; no numerical consistency measure is reported. Encoder specificity, limited perturbation diversity, and target-text dependency are its stated implications.

**Claims and evidence — 6.5**

- Initialization improves reported overall targeted ASR: Table 3, CroPA+Init Overall 0.8977, 0.9260, 0.9458, 0.8761, 0.9333; Table 4, average Overall 0.9209.
- Image augmentation improves cross-image ASR: Table 5, Overall 0.4355 for base, 0.7055 for SCMix, 0.5565 for CutMix.
- Target value-vector guidance improves targeted ASR: Table 6, Overall 0.9685; Table 7, Overall 0.9270.
- Cross-model gains remain encoder-dependent: Table 8, particularly the BLIP2→InstructBLIP and BLIP2→Flamingo comparisons.

**Claims and evidence — 6**

- The four reproduction claims are evaluated: Table 1 for target-text transfer, Figure 4 for prompt counts, Figures 5–6 for convergence, and Table 2 for few-shot settings.
- The extensions improve reported results while retaining transfer limitations: Tables 3–8, as detailed in Section 6.5.

## 7 Discussion

*The discussion reviews reproduction findings, enhancement results, transferability trade-offs, impacts, and limitations.*

### 7.1 Observations with reproducibility

*The reproduction supports CroPA’s cross-prompt and few-shot claims.*

We substantiate Luo et al.’s main claims about CroPA’s cross-prompt adversarial transferability. CroPA performs strongly across VQA, classification, and captioning and consistently outperforms baselines.

It also remains robust in few-shot settings, Section 6.4. Its advantage over baselines shows resilience to the changed generation conditions introduced by in-context examples.

**Claims and evidence**

- CroPA outperforms baselines overall across target texts: Table 1, CroPA Overall 0.7708, 0.7985, 0.5033, 0.7020, 0.6920, and 0.8680.
- CroPA remains stronger in few-shot evaluation: Table 2, shot=2 Overall 0.6635 versus Multi-P 0.5730, with task values 0.9440/0.7860, 0.9940/0.9420, 0.6860/0.5630, and 0.0300/0.0010.

### 7.2 Noise Initialization findings

*Semantic initialization improves attack efficacy and reduces optimization steps.*

Section 5.1 identifies random initialization as a critical limitation of CroPA and related attacks. It ignores target-prompt semantics. We hypothesize that this causes instability, slower convergence, and reduced cross-prompt and cross-modal transferability.

Our alternative uses a diffusion-synthesized reference image and vision-feature alignment to orient perturbations toward meaningful adversarial directions from the start.

Section 6.5.1 empirically validates this hypothesis. Across all evaluated benchmarks, initialization significantly outperforms CroPA and random-initialized baselines, with up to 15% targeted-ASR gains and notably fewer optimization steps.

Semantic anchoring therefore improves both efficacy and efficiency. It supplies a more principled and robust initialization foundation for VLM attack frameworks.

**Claims and evidence**

- Initialization produces gains described as up to 15%: Table 3, CroPA/CroPA+Init Overall pairs 0.7708/0.8977, 0.8433/0.9260, 0.7985/0.9458, 0.8190/0.8761, 0.8535/0.9333.
- Average overall targeted performance improves: Table 4, CroPA 0.7958 and CroPA+Init 0.9209.
- Optimization requires fewer steps: Figure 7(a); no numerical step reduction is stated.
- Semantic alignment changes modestly: Figure 7(b), CLIP scores 21.80 and 23.09.

### 7.3 Evaluation of Transferability Experiments

*The transfer experiments show augmentation gains and encoder-dependent value-vector transfer.*

#### 7.3.1 Cross-Image Transferability

*SCMix and CutMix improve cross-image transfer, but further methods are needed.*

The original authors suggest that computing perturbations across images can achieve cross-image transfer. Our experiments instead show that CroPA’s strong cross-prompt perturbations remain tightly coupled to source-image low-level features.

We integrate SCMix and CutMix, Section 5.3, to increase cross-image diversity while preserving original semantics.

Section 6.5.2 shows significant SCMix gains, particularly for VQA, VQA-specific, and Captioning. Overall ASR increases by approximately 27 percent. CutMix achieves a net increase of around 12 percent over base cross-image CroPA.

We believe SCMix works because cropping and resizing introduce spatial and scale invariance before soft mixing with another image. Perturbations can then target semantically aligned features less sensitive to exact location or scale, improving generalization.

Both augmentation methods add cross-image features while learning perturbations. Future work should go beyond augmentation to more complex, layered methods. Possibilities include combining augmentation with feature-alignment attacks on latent representations, the images’ internal representations, or perturbing the latent space directly.

**Claims and evidence**

- SCMix improves overall cross-image ASR by approximately 27 percent: Table 5, CroPA (Base) Overall 0.4355 and CroPA with SCMix 0.7055.
- SCMix improves the highlighted tasks: Table 5, base/SCMix VQA 0.3500/0.6560, VQAspecific 0.4920/0.7160, Captioning 0.5000/0.9500.
- CutMix gives an increase of around 12 percent: Table 5, base Overall 0.4355 and CutMix 0.5565.
- Spatial and scale invariance explain SCMix’s gains: the section’s interpretation of the augmentation mechanism and Table 5.

#### 7.3.2 Cross-Model Transferability

*D-UAP improves targeted and cross-model results, but encoder dependence limits generalization.*

The D-UAP-modified framework, Section 5.2, advances CroPA’s attack method. Section 6.5.3 reports overall targeted ASR of 96.85%, a 13.5% improvement over base CroPA. VQAgeneral and Captioning increase by 13% and 28%, respectively.

The improvements also occur for challenging or harmful text such as “Bomb,” which typically triggers model guardrails. This demonstrates D-UAP’s robustness in bypassing safety mechanisms.

Cross-model transfer also improves, especially for similar vision encoders such as those of BLIP2 and InstructBLIP. Dependence on the encoder nevertheless constrains perturbation diversity and transfer to other architectures.

This motivates **encoder-agnostic methods**, attacks not dependent on a particular encoder. Despite the limitation, D-UAP is a step toward universal perturbations effective across prompts and images, supporting further research on generalizability and robustness.

We exclude the Contextual Injection Attack from cross-model comparisons because it is inherently non-transferable and could not produce comparable results.

**Claims and evidence**

- The section reports 96.85% targeted ASR and a 13.5% improvement: Table 6, CroPA+D-UAP Overall 0.9685 and CroPA Overall 0.8433.
- The stated VQAgeneral and Captioning gains are 13% and 28%: Table 6, CroPA/CroPA+D-UAP values 0.8176/0.9420 and 0.6960/0.9810.
- Harmful target text is used: Tables 6 and 7 captions specify “Bomb”; bypassing guardrails is the section’s interpretation.
- Similar-encoder transfer improves: Table 8, BLIP2→InstructBLIP Overall 0.5065 for CroPA and 0.6850 for CroPA + D-UAP.
- Different-architecture transfer remains limited: Table 8, BLIP2→Flamingo Overall 0.1435/0.2335; Flamingo-source CroPA+D-UAP Overall 0.1013, 0.0938, 0.1025.
- CIA is excluded because comparable transfer results could not be obtained: qualitative report in Section 5.4.3; no numerical transfer values are stated.

**Claims and evidence — 7.3**

- Augmentation improves cross-image results: Table 5, Overall 0.4355, 0.7055, and 0.5565 for base, SCMix, and CutMix.
- D-UAP improves attack results and similar-encoder transfer: Table 6, Overall 0.9685; Table 8, BLIP2→InstructBLIP CroPA + D-UAP Overall 0.6850.
- Cross-image and cross-architecture limitations remain: Table 5’s cross-image results and Table 8’s cross-architecture rows; the discussion interprets these limitations.

### 7.4 Trade-offs Between Different Types of Adversarial Transferability

*The analysis identifies competing objectives across prompts, models, and images.*

Our analysis reveals fundamental trade-offs among adversarial transferability dimensions in VLMs. Attack design must account for them.

#### 7.4.1 Cross-Prompt vs. Cross-Model Transferability

*Prompt-focused optimization anchors perturbations to a model, limiting transfer to other architectures.*

CroPA excels across prompts by optimizing against competing text embeddings. However, this anchors perturbations to the target model’s architecture. Appendix A.1 shows that ASR drops significantly when perturbations transfer to another model.

D-UAP, Section 5.2, addresses the trade-off by targeting generalizable vision-encoder components rather than model-specific features. It improves transfer between similar models such as BLIP-2 and InstructBLIP. Gains remain modest between different architectures such as BLIP-2 and Flamingo.

This suggests a tension between simultaneous optimization for prompt diversity and model diversity.

**Claims and evidence**

- Base CroPA loses effectiveness across models: Appendix A.1, as cited here; Table 8 includes base-CroPA Overall values of 0.5065 for BLIP2→InstructBLIP and 0.1435 for BLIP2→Flamingo.
- D-UAP helps similar-architecture transfer more: Table 8, BLIP2→InstructBLIP Overall 0.5065/0.6850 for CroPA/CroPA + D-UAP; BLIP2→Flamingo 0.1435/0.2335.
- Prompt and model diversity are in tension: the section’s interpretation of these transfer results.

#### 7.4.2 Cross-Prompt vs. Cross-Image Transferability

*Image-specific coupling supports cross-prompt attacks but limits generalization across images.*

CroPA’s strong cross-prompt transfer comes from tightly coupling perturbations to image-specific features. This fundamentally limits cross-image generalization, making the two objectives compete.

SCMix and CutMix, Section 5.3, diversify optimization through augmentation. Table 5’s gains are promising. However, augmentation does not fully resolve the tension between the objectives.

**Claims and evidence**

- Augmentation improves cross-image transfer: Table 5, Overall base 0.4355, SCMix 0.7055, CutMix 0.5565.
- Cross-image generalization remains limited despite improvement: the interpretation in Sections 6.5.2 and 7.4.2, based on Table 5.
- Image-specific coupling creates competing objectives: the section’s analysis; no separate numerical trade-off measure is stated.

#### 7.4.3 Role of Initialization in Transferability Balance

*Initialization improves ASR without sacrificing cross-prompt transfer, but does not remove transferability tensions.*

Diffusion-based initialization, Section 5.1, offers another approach. Semantic starting points allow higher ASRs without sacrificing cross-prompt transferability. This suggests that the optimization starting point affects the attainable balance among transferability types.

Unlike methods that change optimization itself, better initialization lets perturbations converge toward more general adversarial directions that remain effective across prompts.

Even so, the fundamental tension between cross-prompt and cross-image or cross-model transfer remains. Optimizing one dimension inevitably sacrifices effectiveness in others.

**Claims and evidence**

- Initialization improves reported overall ASR while retaining cross-prompt effectiveness: Table 3, CroPA+Init Overall 0.8977, 0.9260, 0.9458, 0.8761, 0.9333; Table 4, average Overall 0.9209.
- Initialization improves convergence: Figure 7(a); no numerical convergence values are stated.
- Initialization does not remove transferability tension: the discussion’s interpretation; no separate initialization trade-off measurement is reported here.

#### 7.4.4 Implications for Future Research

*Future attacks should explicitly balance competing transferability objectives.*

No single attack methodology can simultaneously maximize all transferability forms. Future frameworks should provide controlled balance, potentially through **multi-objective optimization**, optimization balancing competing objectives, or architecturally aware attack designs.

Individual techniques can improve particular dimensions. A comprehensive solution would require rethinking how perturbations interact with model architectures and data representations.

Cross-prompt adversarial transferability remains underexplored. The lack of literature leaves a gap in understanding prompt-induced vulnerabilities, with important implications for secure deployment.

**Claims and evidence — 7.4**

- Different enhancements improve different dimensions: Table 3, initialization results; Table 5, cross-image augmentation results; Table 8, cross-model value-vector results.
- These findings motivate balancing rather than simultaneously maximizing transferability objectives: the discussion’s interpretation of those results; no numerical joint optimum is reported.

### 7.5 Broader Impact

*Cross-prompt attacks create security threats but may also support privacy protection.*

For attackers, high cross-prompt transferability poses substantial threats. Adversarial images can induce malicious or misleading outputs even under harmless queries. Such attacks could spread false information or subvert VLM-based generation and decision systems.

For defenders, imperceptible perturbations could protect sensitive information. Embedded perturbations could make VLMs consistently output predetermined non-sensitive text, blocking unauthorized extraction from personal images. This offers a proactive privacy and data-security mechanism.

Our initialization and loss refinements markedly improve ASR and generalization across models and images. They strengthen cross-prompt attacks and may support more resilient defenses.

The reproduction affirms Luo et al.’s foundation and highlights the need for deeper investigation. Understanding cross-prompt transferability is important for robust VLMs, countermeasures, user-data protection, and output integrity.

**Claims and evidence**

- Refinements improve reported ASR: Table 4, CroPA+Init Overall 0.9209 versus CroPA 0.7958; Table 6, CroPA+D-UAP 0.9685 versus CroPA 0.8433.
- Extensions improve cross-image and cross-model results: Table 5, augmentation Overall 0.7055 and 0.5565 versus base 0.4355; Table 8, cross-model comparisons.
- Security threats and privacy uses are broader-impact possibilities stated here, not separately evaluated results.

### 7.6 Limitations and Future Work

*Computational resources limited experiment scale, tuning, architectural coverage, and ensemble evaluation.*

We reproduce CroPA’s key results and conduct extensive ablations, but limited computation repeatedly hinders progress. Large VLM attacks, including Flamingo and BLIP2, require substantial GPU memory and processing. This limits the scale and depth of some experiments.

Resource availability constrains extensive hyperparameter tuning and evaluation across more model architectures. We also aim to investigate an **ensemble**, multiple models used together, for deeper transferability analysis. Its memory and computation demands make it infeasible with our infrastructure.

Future work could develop encoder-agnostic attacks, improve cross-image generalization, and implement optimization with query-based strategies.

### 7.7 Communication With Original Authors

*No direct contact was established, and the reported model-code issue remains unresolved.*

We could not establish direct communication with the original authors during replication. The GitHub issue about BLIP-2 and InstructBLIP remains unresolved at the time of writing.

**Claims and evidence — 7**

- The reproduction supports cross-prompt and few-shot performance: Table 1, CroPA rows; Table 2, shot=2 CroPA Overall 0.6635 versus Multi-P 0.5730.
- Initialization, augmentation, and value-vector guidance improve their reported metrics: Table 4, CroPA+Init Overall 0.9209; Table 5, SCMix 0.7055 and CutMix 0.5565; Table 6, CroPA+D-UAP 0.9685; Table 8, cross-model rows.
- Transferability trade-offs and encoder dependence remain: Table 5 and Table 8, as interpreted in Sections 7.3–7.4.
- Resource and communication limitations are qualitative reports in Sections 7.6–7.7; no additional numerical measurements are stated.
