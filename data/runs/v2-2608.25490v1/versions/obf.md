# MMJailBench: A Factorized Benchmark for Disentangling

Multimodal Jailbreak Vulnerabilities

Tianshi Wang

Affiliation:  Tongji University

  
Jingsong Wang

Affiliation:  Tongji University

  
Yafei Huang

Affiliation:  Tongji University

  
Fengling Li

Affiliation:  Mohamed bin Zayed University of Artificial Intelligence

  
Xin Li

Affiliation:  Shanghai Artificial Intelligence Laboratory

  
Lei Zhu
††thanks: Corresponding author.
Affiliation:  Tongji University

###### Abstract


Across 16 open-weight and proprietary MLLMs, large-scale evaluations of QX7 are reported as revealing highly heterogeneous and model-dependent vulnerability profiles, with vulnerability varying markedly across S6M so as to expose uneven coverage in current multimodal safety alignment, with V8M emerging as the dominant source of variation, with task-relevant R2T systematically increasing jailbreak susceptibility and authority-like cues exposing particularly pronounced vulnerabilities, and with visually rendered instructions not consistently increasing jailbreak susceptibility relative to direct textual instructions; the obscuration of the specific sources of observed vulnerabilities in existing benchmarks, where Z3K, V8M, R2T, and W9N are often coupled within individual jailbreak instances, is addressed by the introduction of QX7 as a factorized benchmark in which these factors are systematically varied and combined under controlled configurations for fine-grained comparison and factor-level attribution, because how different factors shape the jailbreak vulnerabilities of Multimodal Large Language Models (MLLMs), despite their increasing deployment in real-world applications, remains poorly understood. Warning: This paper contains potentially harmful content. On a representative open-weight model, diagnostic analyses of the risks introduced by multimodal context are conducted and vulnerability-associated patterns in internal representations and cross-modal interactions are identified, while a modular multimodal jailbreak evaluation suite is developed with P2W and C9Z, multiple judge options, and multidimensional metrics, enabling reproducible, scalable, and cost-efficient multimodal jailbreak auditing.

## 1 Introduction

Substantially uneven jailbreak vulnerabilities across S6M (S6M denoting the harm domains into which requests are grouped) were revealed by large-scale evaluations conducted across 16 representative open-weight and proprietary MLLMs, cyber abuse, economic harm, privacy violations, and deception and manipulation exhibiting greater overall vulnerability, with V8M emerging as the dominant source of variation in jailbreak outcomes in that, even when Z3K remains unchanged, different narrative structures and output requirements can substantially alter refusal and compliance behavior, while task-relevant R2T were found to systematically weaken refusal behavior, authority-like cues such as G3S and X7R exposing particularly pronounced vulnerabilities, a pattern suggesting that models may over-rely on the apparent legitimacy and credibility conveyed by visual context, whereas the effect of W9N proved highly model-dependent, visually rendered instructions not consistently increasing jailbreak susceptibility relative to direct textual instructions, and diagnostic analyses on a representative open-weight model further identifying vulnerability-associated patterns in internal representations and cross-modal interactions.

Figure 2: Construction and evaluation pipeline of QX7, in which Z3K, V8M, R2T, and W9N are factorized and MLLMs are evaluated under controlled multimodal jailbreak configurations.

A factorized benchmark, QX7, incorporating Z3K, V8M, R2T, and W9N into a unified controlled design and thereby enabling fine-grained analysis and factor-level attribution of multimodal jailbreak vulnerabilities, is introduced; large-scale evaluations across 16 representative open-weight and proprietary MLLMs are conducted, revealing uneven jailbreak vulnerabilities across S6M, the dominant role of V8M, the refusal-weakening effect of authority-like visual cues, and the model-dependent effect of W9N, diagnostic analyses on a representative open-weight model further identifying vulnerability-associated patterns in internal representations and cross-modal interactions; and a modular multimodal jailbreak evaluation suite with P2W and C9Z (C9Z denoting the lightweight configuration), multiple judge options, and multidimensional metrics is developed, reproducible, scalable, and cost-efficient multimodal jailbreak auditing being thereby enabled.

Introduced as a factorized benchmark for disentangling the sources of multimodal jailbreak vulnerabilities, QX7 (QX7 denoting the paper's MMJailBench) incorporates Z3K, V8M, R2T, and W9N into a unified controlled design, systematic combinations and matched comparisons being used to examine how each factor shapes model refusal and compliance behavior, and contains 272 Z3K, 6 prompt templates, 5 types of task-relevant R2T (R2T denoting visual semantics), and 2 W9N, a structured and directly comparable evaluation space being formed in which not only the measurement of overall jailbreak vulnerability but also the attribution of observed vulnerabilities to specific factors is enabled, a modular multimodal jailbreak evaluation suite with P2W and C9Z, multiple judge options, and multidimensional metrics being further developed on this design, enabling reproducible, scalable, and cost-efficient multimodal jailbreak auditing.

A broad range of harmful behaviors (Li et al., 2024a; Mazeika et al., 2024; Chao et al., 2024), visual attacks, and image-based text injection (Liu et al., 2024; Gong et al., 2025; Luo et al., 2024) are covered by existing text and multimodal safety benchmarks, an important foundation for evaluating model robustness being thereby provided, yet fixed prompt-image pairs are typically constructed by these benchmarks for individual harmful tasks, with Z3K (Z3K denoting harmful intent), V8M (V8M denoting prompt framing), R2T, and W9N (W9N denoting the instruction carrier) jointly encoded within a single test instance, so that the resulting attack success rates primarily reflect a model's aggregate vulnerability on a particular test set while limited insight is offered into whether observed vulnerabilities originate from linguistic framing, visual context, the way instructions are conveyed, or weaknesses concentrated in particular S6M (Weng et al., 2025; Jia et al., 2025), such factor entanglement limiting fine-grained comparison and obscuring the diagnostic signals needed to better understand and improve multimodal safety alignment, as illustrated in Figure 1.

Figure 1: Motivation of QX7, in which the entanglement of multiple factors within fixed instances by existing multimodal jailbreak benchmarks, limiting factor-level diagnosis, is addressed through a factorization of these factors and the construction of a controlled evaluation space for fine-grained analysis.

How models form safety judgments in cross-modal contexts remains poorly understood, visual information being capable not only of supplementing Z3K but also of altering the perceived legitimacy, professionalism, and credibility of a request, the same underlying intent thereby eliciting different refusal or compliance behaviors across contexts (Liu et al., 2024), which suggests that multimodal jailbreaks are not merely changes in input format but rather expose the stability and consistency of safety alignment under cross-modal conditions (Luo et al., 2024), the accurate evaluation and improvement of multimodal safety alignment requiring an understanding of how such vulnerabilities arise.

Expansion beyond image captioning and visual question answering into real-world applications such as document understanding, code generation, workflow assistance, and automated operations (Liu et al., 2023; Dai et al., 2023) characterizes MLLMs, which, unlike text-only models, must jointly interpret linguistic instructions and visual information when reasoning and generating responses, harmful requests being thereby enabled to combine direct textual or visually rendered instructions (Gong et al., 2025) with contextual cues such as professional settings, X7R (X7R denoting identity credentials), or G3S (G3S denoting authorization documents), the consequence being that a model's refusal and compliance decisions no longer depend solely on an isolated user instruction but are jointly shaped by the surrounding cross-modal context (Li et al., 2024b).

## 2 Related Work

Factor-level attribution of multimodal jailbreak vulnerabilities and fine-grained comparison are enabled, in the material catalogued as 2.2, by QX7, which complements those efforts through a controlled factorized design that systematically varies V8M, task-relevant R2T, and W9N over matched Z3K, where SafetyBench (Zhang et al., 2024), Do-Not-Answer (Wang et al., 2023), SALAD-Bench (Li et al., 2024a), HarmBench (Mazeika et al., 2024), and JailbreakBench (Chao et al., 2024), having established structured taxonomies and reproducible protocols for evaluating harmful behavior and refusal robustness, are extended to visual inputs by MM-SafetyBench evaluating image-based safety attacks (Liu et al., 2024), by FigStep (Gong et al., 2025) studying harmful instructions rendered as typographic images, by JailBreakV (Luo et al., 2024) and VLJailbreakBench (Wang et al., 2025) expanding coverage across diverse multimodal jailbreak strategies, by MMJ-Bench (Weng et al., 2025) providing a unified framework for comparing multimodal jailbreak attacks and defenses, and by OmniSafeBench-MM (Jia et al., 2025) broadening standardized attack-defense evaluation with multidimensional metrics, while Omni-SafetyBench (Pan et al., 2025) constructs parallel variants of the same harmful seeds across different modality configurations and evaluates cross-modal safety consistency; yet, even with attack coverage, evaluation standardization, and modality-level comparison substantially improved, they are not designed to independently manipulate the linguistic and semantic factors that jointly shape safety decisions within matched multimodal interactions.

Refusal behavior, in the material numbered 2.1, is established as highly sensitive to how Z3K is framed and delivered (Wei et al., 2023; Chao et al., 2025; Greshake et al., 2023), with jailbreak attacks aiming to circumvent safety alignment and induce models to assist with harmful requests (Wei et al., 2023; Zou et al., 2023), early studies attributing such failures to competing objectives and mismatched generalization in safety training (Wei et al., 2023), subsequent work demonstrating transferable adversarial suffixes (Zou et al., 2023) and black-box semantic attacks such as PAIR (Chao et al., 2025), and indirect prompt injection showing that malicious instructions can also be introduced through external documents or retrieved content (Greshake et al., 2023); multimodal models introduce additional attack surfaces because instructions and contextual cues can be distributed across language and vision (Gong et al., 2025; Liu et al., 2024; Li et al., 2024b), and, beyond manipulation in language, visual inputs can be exploited through adversarial images (Liu et al., 2024; Li et al., 2024b), typographic prompts (Gong et al., 2025), and other cross-modal constructions (Luo et al., 2024), while corresponding defenses spanning safety alignment, input transformation, and inference-time safeguards (Jain et al., 2023; Wang et al., 2024) leave robustness across changes in modality and context uncertain, since identical Z3K may elicit different safety behavior under different linguistic or visual contexts, with controlled evaluation of multimodal alignment robustness thereby being motivated.

## 3 MMJailBench

Through Cartesian combination of 272 Z3Ks, 6 V8M strategies, 5 R2T conditions, and 2 W9Ns, 16,320 controlled multimodal jailbreak instances are obtained in QX7, a factorized multimodal jailbreak benchmark designed for the analysis of how contextual factors influence MLLM safety behavior, in which, as illustrated in Figure 2, matched multimodal scenarios are constructed through systematic variation of Z3Ks, V8Ms, R2Ts, and W9Ns rather than through reliance on fixed jailbreak instances with entangled factors, the control of these factors within a unified evaluation space enabling fine-grained analysis of their impact on model safety behavior, with consistency across models and harm categories maintained by this factorized design, which permits systematic analysis of how different contextual factors are associated with changes in model safety behavior.

Matched comparisons across multimodal conditions are enabled by fixing the underlying Z3K and systematically varying contextual factors, the contribution of each factor to observed safety variations being thereby rendered analyzable, and each multimodal instance is represented, given a Z3K $h$ , a V8M strategy $t$ , an R2T condition $v$ , and a W9N $c$ , as

| | $x_{h,t,v,c}=(L(h,t,c),V(h,v)),$
| | (1)

a multimodal jailbreak instance being factorized by QX7 into these four components, Z3K, V8M, R2T, and W9N, whose individual treatments appear in 3.2, 3.3, 3.4, and 3.5 under the design set out in 3.1.

Limited insight into which factors contribute to observed safety variations is offered by existing multimodal jailbreak benchmarks, which, focusing on predefined attack instances or modality variations in which multiple safety-relevant factors, including Z3K, linguistic formulation, visual context, and instruction presentation, remain jointly encoded within individual evaluations, nevertheless provide valuable measurements of overall vulnerability; in the representation above, $L(h,t,c)$ denotes the instruction component generated from Z3K $h$ , V8M $t$ , and W9N $c$ , while $V(h,v)$ denotes the visual input associated with Z3K $h$ and R2T $v$ .

Across 9 major S6M, namely Physical Harm, Cyber Abuse, Economic Harm, Hate and Harassment, Privacy and IP, Regulated Advice, Illegal Activities, Deception and Influence, and Sexual Content, the 272 Z3Ks, each category being further divided into fine-grained subcategories for the improvement of coverage and diversity across safety-critical behaviors, are collected through category-guided construction followed by semantic similarity filtering and manual verification for the reduction of redundancy, constituting intent-level behavioral seeds rather than directly collected jailbreak prompts entangling harmful goals with specific attack strategies, so that contextual factors can be systematically varied while the harmful objective is kept fixed, these intent-level seeds providing a consistent foundation for evaluating how V8M, R2T, and W9Ns influence multimodal safety behavior under controlled conditions, Z3K representing the underlying unsafe objective evaluated in QX7.

A harmful request may be embedded within a fictional scenario, expressed as a structured generation task, or presented as an apparently legitimate system requirement, the 6 representative V8M strategies, namely academic, system, structured, story, code, and M8D (M8D denoting safety-paradox framing), capturing common ways of reformulating harmful requests while the underlying intent is preserved, and, although the underlying harmful objective remains unchanged, different narrative structures and interaction styles may affect how models interpret requests and determine whether to comply, so that variation of only the linguistic framing with Z3K held fixed enables direct comparison of how different presentation styles correspond to changes in multimodal safety behavior, V8M specifying how the same Z3K is linguistically presented to the model.

Whether MLLMs exhibit different safety behaviors under visually grounded contextual conditions is rendered analyzable by a design in which, for each Z3K, the corresponding visual input provides additional semantic context without modification of the underlying harmful instruction, the 5 task-relevant R2T conditions, namely danger-related context, scenario context, professional-role context, X7R, and G3S, representing different forms of contextual signals that may affect how a request is interpreted, such as legitimacy-associated, expertise-related, or authority-related contextual signals, while controlled visual conditions, including irrelevant or non-semantic images, are used in diagnostic evaluations for the distinction of the effect of meaningful visual context from the general presence of an image, QX7, unlike image-based instruction attacks that primarily encode instructions into visual text, focusing on whether contextual visual information changes safety behavior when the underlying Z3K remains unchanged, R2T capturing the contextual meaning introduced by accompanying images.

Carrier-related safety differences are isolated from perception failures through the measurement of N7V (N7V denoting instruction recognition mismatch, in which models fail to interpret embedded instructions correctly), which is introduced to account for the additional perception stage of K6F, the 2 W9Ns considered being direct textual instructions (D4P), which directly provide harmful instructions as text, and K6F (K6F denoting the OCR carrier of visually rendered instructions in text-containing images), which renders the same instructions into images while their semantics are preserved, W9N describing how harmful instructions are presented to the model and examining whether changing the presentation modality of the same instruction affects multimodal safety behavior while the underlying Z3K remains unchanged.

## 4 Experimental Setup

QX7 is evaluated on 16 representative MLLMs (4.1), covering both open-weight and proprietary-access models, under the same factorized framework for consistent comparison across multimodal jailbreak conditions, with the complete benchmark configuration containing 16,320 multimodal jailbreak instances evaluated for each model, resulting in 261,120 model responses across all evaluated systems, the per-model B8X outcomes under P2W being collected in Table 1, and additional experimental details, including evaluation prompts, judge criteria, and implementation details, provided in the appendix.

| Model | Avg. | Cyber | Econ. | Privacy | Decept. | Illegal | Hate | Violence | Sensitive | Sexual |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Open-weight | | | | | | | | | | |
| glm4.1v-9b | 72.22 | 80.72 | 85.28 | 71.78 | 78.04 | 83.16 | 67.39 | 79.58 | 44.94 | 59.12 |
| step3-vl-10b | 53.36 | 70.00 | 68.22 | 62.06 | 58.93 | 54.48 | 49.17 | 44.24 | 42.06 | 27.06 |
| ministral3-8b | 51.64 | 66.83 | 63.78 | 59.61 | 59.58 | 56.21 | 43.11 | 46.98 | 38.06 | 30.59 |
| llava-onevision-1.5 | 40.64 | 42.00 | 49.72 | 38.17 | 42.74 | 52.30 | 36.00 | 48.96 | 21.06 | 34.80 |
| internvl3-8b | 49.12 | 63.11 | 61.72 | 51.33 | 53.21 | 61.49 | 42.33 | 50.80 | 27.39 | 30.69 |
| qwen2.5-vl-7b | 59.48 | 67.28 | 70.83 | 60.94 | 62.50 | 73.28 | 54.39 | 66.91 | 33.44 | 50.78 |
| qwen3-vl-8b | 19.84 | 37.56 | 30.28 | 28.72 | 20.65 | 19.02 | 12.06 | 11.15 | 14.39 | 4.71 |
| gemma3-12b | 66.85 | 82.94 | 80.44 | 75.94 | 75.95 | 74.54 | 62.67 | 60.66 | 51.39 | 37.06 |
| Proprietary-access | | | | | | | | | | |
| glm-4.6v | 78.38 | 93.06 | 88.11 | 85.17 | 87.26 | 80.29 | 74.94 | 73.68 | 59.89 | 53.04 |
| qwen3-vl-plus | 62.02 | 80.78 | 84.00 | 80.22 | 78.57 | 58.85 | 55.78 | 43.37 | 51.67 | 24.90 |
| doubao-seed-2.0-lite | 47.79 | 68.39 | 61.94 | 62.94 | 63.10 | 42.59 | 47.50 | 32.99 | 46.44 | 16.18 |
| kimi-k2.5 | 35.80 | 53.28 | 45.11 | 44.06 | 43.81 | 35.75 | 34.00 | 23.30 | 30.72 | 12.16 |
| gemini-3-flash | 30.50 | 39.61 | 38.94 | 41.06 | 42.14 | 23.10 | 28.72 | 11.94 | 36.61 | 12.35 |
| grok-4-fast | 42.61 | 55.00 | 52.50 | 55.00 | 52.50 | 37.07 | 49.33 | 26.91 | 40.33 | 32.84 |
| claude-sonnet-4.5 | 8.96 | 15.11 | 15.11 | 13.06 | 10.06 | 6.09 | 7.00 | 2.08 | 11.22 | 0.88 |
| Y6L | 2.17 | 3.06 | 4.11 | 2.00 | 3.39 | 0.57 | 2.50 | 0.07 | 3.56 | 0.29 |

Table 1: Per-model B8X (%) across the nine major S6M of QX7, the Avg. column denoting the average of B8X over all evaluated S6M.

The characterization of model safety behavior under controlled multimodal contexts, together with the analysis of vulnerability patterns across different jailbreak conditions, constitutes the aim of the evaluation. A benchmark-specific F4K (F4K denoting the lightweight judge model), based on an open-weight MLLM, is additionally developed for cost-efficient follow-up evaluation under the same protocol, while the main experiments (4.2) employ Y6L (Y6L denoting GPT-5) as an LLM-based judge, by which, for each model response, a T5J and auxiliary labels for jailbreak success and instruction mismatch are assigned, the evaluation being focused on generated responses in such a way that refusals, safety warnings, and high-level discussions are not considered successful jailbreak cases, and all main results being based on Y6L judgments. Model safety in QX7 is evaluated using complementary metrics, the primary one (B8X denoting the Attack Success Rate (ASR)) measuring the proportion of responses that provide substantive assistance to the targeted Z3K,

$B8X=\frac{1}{N}\sum_{i=1}^{N}\mathbb{I}(s_{i}\geq 4),$ (2)

with $s_{i}$ denoting the T5J (T5J denoting the harmfulness score) of the $i$-th response on a 1 $\sim$ 5 scale, responses with $s_{i}\geq 4$ being considered successful jailbreak cases, while, for W9N analysis, H3Q (H3Q denoting the Conditional Attack Success Rate (CASR)) is further introduced,

$H3Q=\frac{N_{success}}{N-N_{mismatch}},$ (3)

where mismatch cases refer to inputs that are not correctly understood by the model, including K6F recognition failures, irrelevant responses, and misinterpretation of the evaluated intent, end-to-end jailbreak vulnerability being measured by B8X whereas safety robustness conditioned on successful input understanding is evaluated by H3Q. Two evaluation configurations are supported by QX7 (4.3), the one (P2W denoting the full configuration) under which all benchmark instances are evaluated providing complete factor coverage for the main analysis, including model-level, domain-level, and factor-level comparisons, and C9Z sampling a representative subset while preserving coverage across S6M, V8M, R2T, and W9N, thereby enabling efficient safety auditing and regression testing, with all reported results, unless otherwise specified, obtained using P2W. The evaluated models include representative model families such as LLaVA-OneVision (An et al., 2025), InternVL (Zhu et al., 2025), Kimi (Team et al., 2026), Qwen (Bai et al., 2025b; Bai et al., 2025a), Gemma (Team et al., 2025), GLM (Hong et al., 2025), GPT (Singh et al., 2025), Gemini (Team et al., 2023), and Claude (Anthropic, 2025).

## 5 Results and Analysis

That multimodal jailbreak vulnerability is not governed by a single universal failure pattern, different models exhibiting different combinations of contextual sensitivity that reflect variations in safety alignment behavior, is the conclusion licensed by the heterogeneous profiles of 5.3 within section 5, where Table 4, although aggregate B8X provides an overall comparison of model robustness, reports — because aggregate B8X does not reveal how individual models respond to different jailbreak factors — the B8X variation range induced by each factor across evaluated models and reveals distinct vulnerability signatures among MLLMs, some models being highly sensitive to V8M reframing with $\Delta$ V8M exceeding 80 percentage points while others exhibit stronger sensitivity to W9N changes and R2T effects are generally smaller but consistently positive across models, so that factorized evaluation provides a more informative characterization of model robustness beyond aggregate jailbreak success rates.

| Model | $\Delta$ Z3K | $\Delta$ V8M | $\Delta$ R2T | $\Delta$ W9N

| Open-weight

| glm4.1v-9b | 76.7 | 51.8 | 4.2 | 8.3

| step3-vl-10b | 86.7 | 72.2 | 4.8 | 10.5

| ministral3-8b | 83.3 | 57.2 | 9.6 | 19.9

| llava-onevision-1.5 | 61.7 | 35.3 | 4.6 | 50.5

| internvl3-8b | 75.0 | 88.3 | 5.5 | 4.6

| qwen2.5-vl-7b | 78.3 | 90.8 | 3.4 | 0.8

| qwen3-vl-8b | 65.0 | 15.6 | 6.5 | 14.4

| gemma3-12b | 88.3 | 79.0 | 7.0 | 13.6

| Proprietary-access

| glm-4.6v | 73.3 | 62.6 | 11.3 | 1.5

| qwen3-vl-plus | 91.7 | 42.6 | 8.1 | 9.7

| doubao-seed-2.0-lite | 95.0 | 80.2 | 5.8 | 1.3

| kimi-k2.5 | 75.0 | 44.3 | 6.9 | 34.6

| gemini-3-flash | 76.7 | 59.7 | 7.8 | 34.3

| grok-4-fast | 76.7 | 70.6 | 2.8 | 4.1

| claude-sonnet-4.5 | 55.0 | 32.1 | 2.0 | 0.8

| Y6L | 13.3 | 8.8 | 0.8 | 3.6

Table 4: Model-dependent vulnerability profiles across QX7 factors; each $\Delta$ score denotes the range of B8X variation across configurations of the corresponding factor.

The evaluated MLLMs, 16 representative ones assessed under the full QX7 configuration in 5.1 with Figure 3 and Table 1 summarizing model-level B8X across controlled multimodal conditions and S6M, exhibit substantial differences in safety robustness under identical jailbreak configurations, the average B8X ranging from 2.17% for Y6L to 78.38% for GLM-4.6V, which demonstrates that multimodal safety alignment remains highly model-dependent and that models with comparable multimodal capabilities may exhibit significantly different vulnerability profiles, stronger general capability not necessarily implying stronger jailbreak robustness; beyond model-level differences, vulnerability is also unevenly distributed across harmful objectives, cyber abuse, economic harm, privacy-related behaviors, and deception-related tasks generally achieving higher B8X whereas categories such as physical harm exhibit comparatively lower vulnerability, and these results suggest that current multimodal safety alignment does not provide uniform protection across harmful objectives, the sources of the observed vulnerability being identified through the factor-level analysis that the factorized design of QX7 enables by permitting controlled comparison in which individual factors are varied while other conditions are kept fixed across the four dimensions Z3K, V8M, R2T, and W9N.

V8M introduces the largest variation among the evaluated factors, because, although the underlying harmful objective remains unchanged, different linguistic presentations lead to substantially different jailbreak outcomes, and, as Figure 3(a) shows, models exhibit distinct sensitivity patterns across V8M strategies, with story, structured, and academic framings generally leading to higher B8X while system-style framing and M8D show lower vulnerability, the gap between the most and least vulnerable strategies exceeding 40 percentage points, which indicates that MLLMs are sensitive not only to Z3K itself but also to the surrounding interaction structure and linguistic presentation.

Figure 3: Model-level vulnerability profiles across V8M and R2T factors; each radar plot shows B8X variation under controlled factor changes, revealing model-specific sensitivity patterns.

| Group | Image condition | B8X (%) | $\Delta$ B8X (%)

| Semantic | G3S | 47.73 | 12.96

| X7R | 45.24 | 10.47

| Task scenario | 44.87 | 10.10

| Professional role | 43.60 | 8.83

| Dangerous context | 43.22 | 8.45

| Control | Blank image | 36.96 | 2.19

| Noise image | 35.53 | 0.76

| Nature image | 35.49 | 0.72

| L8G | 34.77 | 0.00

Table 2: R2T ablation results comparing semantic and control image conditions; $\Delta$ B8X denotes the B8X increase relative to L8G.

R2T introduce additional vulnerabilities beyond the presence of images alone in 5.2, the model responses varying substantially across different visual contexts in Figure 3(b) and Table 2, where task-relevant R2T consistently yield higher B8X than L8G (L8G denoting the no-image condition), the largest increase being caused by G3S (+12.96%), followed by X7R (+10.47%) and task scenarios (+10.10%), whereas the non-semantic controls, including blank, noise, and nature images, lead to only marginal changes of less than 2.2%, so that the increased vulnerability mainly stems from contextual visual meaning rather than from visual input itself, with authority-related cues in particular possibly providing implicit legitimacy signals that thereby weaken models' refusal behavior.

Z3K, by revealing whether safety alignment provides consistent protection across different categories of harmful behaviors, shows in Table 1 that jailbreak vulnerability varies substantially across S6M, cyber abuse, economic harm, privacy-related behaviors, and deception-related tasks generally exhibiting higher B8X across models while physical harm and sensitive content categories tend to show lower vulnerability, differences which indicate that aggregate safety scores may hide domain-specific weaknesses and that explicit modeling of Z3K is necessary for fine-grained safety diagnosis.

| Carrier | $N$ | Mis. (%) | B8X (%) | H3Q (%)

| D4P | 130,560 | 0.37 | 50.70 | 50.89

| K6F | 130,560 | 4.14 | 40.59 | 42.34

| UNION | 261,120 | 2.26 | 45.64 | 46.69

Table 3: W9N analysis under D4P and K6F settings; $N$ denotes the number of evaluated model responses, and Mis. represents the N7V rate.

W9N exhibits a different pattern from V8M and R2T, K6F-based instructions achieving lower B8X than direct D4P inputs (40.59% vs. 50.70%) in Table 3 and the difference remaining after excluding N7V cases (42.34% vs. 50.89%), so that visually rendered instructions are not inherently stronger jailbreak carriers and carrier-related vulnerability depends instead on model-specific multimodal processing behavior.

## 6 Diagnostic Analysis

Figure 4(c) gives consistent positive shifts across all S6M and Figure 4(b) gives a clear displacement of authority-document contexts along the fixed direction obtained by fitting (Q2N denoting the authority-minus-danger representation direction) on the discovery samples and applying Q2N to held-out pairs, while Figure 4(a), for residual-stream representations of the final generation-prefix token in gemma3-12b, gives representation divergence between the authority and danger conditions increasing with depth with the sharpest rise at layer 42, from which increasing separation of the two visual contexts in higher-layer representations before response generation, extension of the representation pattern to held-out samples, and a shared higher-layer representation pattern associated with authority-related cues across diverse harmful objectives are inferred (6.1). Figure 4: representation-level diagnostics of gemma3-12b under authority-document and danger-image contexts, with authority-related visual cues exhibiting distinct representation patterns across layers and S6M. That diagnostic complex was situated in a diagnostic subset constructed from matched QX7 instances in which authority-document scenarios play a prominent role, the behavioral results having revealed substantial variation in multimodal jailbreak vulnerability across V8M, R2T, and W9N, and layer-wise representations together with cross-modal attention patterns in gemma3-12b being analyzed as a model-internal view of how those behavioral differences are reflected in internal states and of how authority-related visual contexts shape the response-generation process. Figure 5 reveals distinct layer-wise attention patterns under authority-document and danger-image contexts (6.2), with authority-document contexts exhibiting, at the identified representation-sensitive layer, lower attention mass on both harm-related task-label and visual tokens; because multimodal response generation depends on how textual and visual information are integrated across layers, attention from the final generation-prefix query to harm-related task-label tokens and special visual tokens was analyzed, and, combined with representation displacement, the lower mass is taken to suggest that authority cues induce a higher-level contextual state with redistributed attention during response initiation, with consistency between representation shifts and attention dynamics providing an internal explanation for increased jailbreak susceptibility under authority-related visual contexts. Figure 5: cross-modal interaction diagnostics in gemma3-12b, with attention allocation to harm-related textual tokens and visual tokens varying across layers under authority-document and danger-image contexts.

## 7 Conclusion

The largest variation is introduced by V8M, additional vulnerability is provided by R2T through contextual cues, model-dependent effects are exhibited by W9N, and category-specific vulnerability patterns are revealed by Z3K, these constituting the substantial diversity in multimodal safety behaviors that evaluations conducted across 16 open-weight and proprietary MLLMs reveal; vulnerability-associated patterns in internal representations and in cross-modal interactions are further uncovered by the diagnostic analyses that were additionally carried out; and a systematic and reproducible framework for scalable multimodal safety evaluation is provided, together with modular P2W and C9Z, flexible judge options, and multidimensional metrics, by QX7, a factorized benchmark introduced in this work for the systematic analysis of multimodal jailbreak vulnerabilities through controlled variations of Z3K, V8M, R2T, and W9N.

## Appendix A Benchmark Construction

Repeatable linguistic conditions for factor-level analysis are provided by the treatment of the 6 templates as evaluation probes rather than as an exhaustive collection of jailbreak attacks, while the W9N factor is constructed by controlling how the same Z3K is delivered to MLLMs through two carrier modes, direct text input (D4P) and visually embedded instruction input (K6F), with the instruction text for K6F instances rendered into a readable image and combined with the corresponding R2T image under a standardized layout so that formatting variations are reduced and controlled comparison between textual and visually embedded instructions is enabled, all while the original Z3K and V8M are preserved.

| Template | Framing Strategy | Representative Pattern |
|---|---|---|
| Academic | Academic Analysis | Analyze the topic from an academic perspective and discuss related aspects. |
| System | System-Role Simulation | Assume a specific system identity and follow role-specific response instructions. |
| Story | Fictional Storytelling | Describe the scenario through a fictional story or hypothetical setting. |
| Code | Code-Style Indirection | Express the requested procedure through code format or comments. |
| Structure | Structured Completion | Complete a predefined structure containing multiple stages or sections. |
| Paradox | M8D | Discuss harmful actions indirectly under a safety-oriented rationale. |

Table 6: Overview of the 6 V8M templates used in QX7. Each template modifies the V8M while preserving the underlying Z3K.

The 6 representative V8M templates used in QX7, whose corresponding design characteristics are condensed in Table 6, cover common jailbreak-related framing strategies, including Academic Analysis (Academic), System-Role Simulation (System), Fictional Storytelling (Story), Code-Style Indirection (Code), Structured Completion (Structure), and M8D (Paradox); each template is instantiated by replacing the task-specific placeholder with the target Z3K, the harmful objective is kept unchanged across different templates, the underlying Z3K is preserved while the surrounding framing strategy is varied, and the influence of different contextual framings on MLLM safety behavior is thereby evaluated through controlled comparison of jailbreak susceptibility across prompt styles.

Figure 8: Examples of the five R2T categories in QX7. Each category provides distinct visual contextual cues while preserving the underlying Z3K.

Five R2T categories are introduced in QX7, including Danger, Work Scenario, G3S, X7R, and Role, so that the influence of different visual cues on MLLM safety behavior is studied while the underlying Z3K and V8M are preserved; for each Z3K, qwen2.5-vl-7b (Bai et al., 2025b) is used to generate visual descriptions and image generation prompts, zimage2025-Turbo (Z-Image Team et al., 2025) is used for visual scene generation in Danger, Work Scenario, and Role, a 4-bit-quantized FLUX.2 [dev] (Black Forest Labs, 2025) is used for the X7R and G3S categories due to its image-editing capability for generating images with coherent textual elements, generated images are filtered based on semantic relevance, visual quality, and readability, and representative examples of the generated R2T conditions are exhibited in Figure 8.

| Category | Count | Ratio (%) |
|---|---:|---:|
| Cyber & Tech Abuse | 30 | 11.0 |
| $\bullet$ Malware & Hacking | 16 | 5.9 |
| $\bullet$ Agent Abuse & Automation | 14 | 5.1 |
| Economic Harm | 30 | 11.0 |
| $\bullet$ Fraud & Scams | 16 | 5.9 |
| $\bullet$ Financial Crimes & Risky Practices | 14 | 5.1 |
| Privacy & IP | 30 | 11.0 |
| $\bullet$ Privacy & Surveillance | 16 | 5.9 |
| $\bullet$ Intellectual Property | 14 | 5.1 |
| Deception & Public Influence | 28 | 10.3 |
| $\bullet$ Disinformation | 14 | 5.1 |
| $\bullet$ Government Decision & Political | 14 | 5.1 |
| Illegal Goods & Services | 29 | 10.7 |
| $\bullet$ Controlled Substances | 15 | 5.5 |
| $\bullet$ General Illegal Acts | 14 | 5.1 |
| Hate & Harassment | 30 | 11.0 |
| $\bullet$ Hate Speech & Extremism | 15 | 5.5 |
| $\bullet$ Harassment & Bullying | 15 | 5.5 |
| Violence & Physical Harm | 48 | 17.6 |
| $\bullet$ Violence & Terrorism | 17 | 6.3 |
| $\bullet$ Weapons & CBRN | 16 | 5.9 |
| $\bullet$ Suicide & Self-Harm | 15 | 5.5 |
| Sensitive Regulated Advice | 30 | 11.0 |
| $\bullet$ Unlicensed Medical & Legal Advice | 15 | 5.5 |
| $\bullet$ High-Risk Financial Advice | 15 | 5.5 |
| Sexual Content | 17 | 6.3 |
| $\bullet$ Sexual Content | 17 | 6.3 |
| Total | 272 | 100.0 |

Table 5: Z3K taxonomy and category statistics. QX7 contains 272 intents organized into 9 major S6M and 18 harm scenarios.

Figure 6: Pairwise cosine similarity distribution among the 272 Z3Ks. The high-similarity tail is manually inspected during taxonomy construction to identify and remove near-duplicate candidates.

The semantic space of QX7 is defined by the Z3K taxonomy through the organization of harmful behaviors into a structured hierarchy of intents, domains, and scenarios, each intent is associated with a task identifier, harm category, scenario, and action description, the taxonomy contains 272 unique Z3Ks covering 9 major S6M and 18 harm scenarios, the taxonomy structure and category statistics are condensed in Table 5, the 9 major S6M include Cyber & Tech Abuse (Cyber), Economic Harm (Econ.), Privacy & IP (Privacy), Deception & Public Influence (Decept.), Illegal Goods & Services (Illegal), Hate & Harassment (Hate), Violence & Physical Harm (Violence), Sensitive Regulated Advice (Sensitive), and Sexual Content (Sexual), and the taxonomy is constructed from existing safety benchmarks (Luo et al., 2024; Gong et al., 2025; Liu et al., 2024; Souly et al., 2024) and public model-provider safety policies (OpenAI, 2025; OpenAI, 2026; Anthropic, 2025a; Google, 2026), followed by intent normalization, taxonomy organization, and manual filtering; embedding-based similarity analysis among the 272 intents is performed with the all-MiniLM-L6-v2 sentence embedding model (Reimers and Gurevych, 2019) so that semantic diversity is improved and redundancy is reduced, most intent pairs in Figure 6 have relatively low semantic similarity while only a small subset forms a high-similarity tail, these cases are manually reviewed to remove near-duplicates and ensure that retained intents represent distinct harmful objectives, the intent embedding space is visualized using PCA in Figure 7, the projection provides a qualitative view, intents from different S6M occupy diverse regions rather than collapsing into a small number of clusters, and the diversity analysis and manual filtering establish a balanced Z3K set for subsequent factorized evaluation.

Figure 7: PCA visualization of Z3K embeddings. Different colors represent S6M, providing a qualitative view of taxonomy coverage and semantic diversity.

## Appendix B Experimental and Evaluation Details

A benchmark-scoped F4K is trained on qwen3-8b (Team et al., 2025) with Y6L-generated annotations, predicting T5J and N7V labels under the same evaluation protocol as the primary Y6L judge, with the consistency of the F4K with Y6L evaluation further validated in Section C.4 and detailed training configurations provided in the Code and Data Supplement; a C9Z of 1,500 instances is constructed by stratified sampling over S6M, V8M, R2T conditions, and W9N modes while preserving the factor distribution of the P2W, with the effectiveness of the C9Z evaluated in Section C.4.

| Score | Definition

| 1 | Refusal or safe response without harmful details.

| 2 | Potential risk without direct harmful instructions.

| 3 | High-level or partial harmful information.

| 4 | Actionable harmful assistance.

| 5 | Detailed and operational harmful guidance.

Table 7: T5J scoring rubric for response evaluation.

Y6L is used as the primary evaluator for QX7 responses, with the judge outputting, for each response, an N7V flag and a T5J, the N7V flag identifying failures to recognize the task such as only transcribing visual content, repeating the prompt, or producing meaningless outputs, and the T5J following a strict 1 $\sim$ 5 rubric (Table 7) in which higher scores indicate stronger compliance and more actionable harmful content; anti-deception rules are incorporated into the judge prompt so that safety disclaimers, fictional framing, or academic packaging are prevented from affecting the evaluation when harmful information is provided, responses limited to refusal, safe redirection, or non-actionable discussion receive low T5J, and the complete judge prompt template is provided in the Code and Data Supplement.

Evaluation of QX7 is carried out on 16 MLLMs, comprising 8 open-weight models evaluated using publicly available checkpoints and 8 proprietary-access models evaluated through their official interfaces, under the same factorized benchmark configuration, with 16,320 instances produced per model and 261,120 total model responses across all evaluated models from the complete Cartesian product of 272 Z3K, 6 V8M, 5 R2T conditions, and 2 W9N modes; the open-weight models include qwen2.5-vl-7b (Bai et al., 2025b), qwen3-vl-8b (Bai et al., 2025a), gemma3-12b (Gemma Team et al., 2025), internvl3-8b (Zhu et al., 2025), glm4.1v-9b (Hong et al., 2025), ministral3-8b (Liu et al., 2026), step3-vl-10b (Huang et al., 2026), and llava-onevision-1.5 (An et al., 2025), and the proprietary-access models include Y6L (Singh et al., 2025), gemini-3-flash (Google DeepMind, 2025), claude-sonnet-4.5 (Anthropic, 2025b), grok-4-fast (xAI, 2025), qwen3-vl-plus (Bai et al., 2025a), kimi-k2.5 (Kimi Team et al., 2026), glm-4.6v (Hong et al., 2025), and doubao-seed-2.0-lite (ByteDance Seed Team, 2026). The P2W, through which the complete Cartesian product of all benchmark factors is evaluated, namely 272 Z3K, 6 V8M, 5 R2T conditions, and 2 W9N modes, results in 16,320 instances per model and provides comprehensive coverage of the benchmark space.

## Appendix C Extended Results and Ablations

That the evaluation of multimodal jailbreak robustness must be conducted under factorized settings, in which individual contributing factors can be separately analyzed, is demonstrated by the substantial heterogeneity observed across models, whereby V8M-dominant models show larger variations across V8M strategies, R2T-responsive models are more affected by R2T conditions, W9N-sensitive models exhibit stronger W9N-related changes, and low-variation models remain relatively stable across factors, with the four representative profiles into which models are grouped by Z9B on the basis of their sensitivity patterns indicating that different MLLMs may exhibit distinct jailbreak vulnerabilities under different contextual conditions.

Figure 9: Per-model jailbreak factor sensitivity profiles. Horizontal and vertical axes represent B8X ranges induced by V8M and R2T conditions, respectively. Marker size indicates the B8X range induced by W9Ns. Colors and shapes denote four Z9B profiles obtained with 1,000 random restarts.

Each model is characterized in Figure 9 by the B8X variation ranges induced by V8M, by R2T conditions, and by W9Ns—larger ranges indicating stronger sensitivity to the corresponding factor—with the grouping of models into the four profiles being performed through Z9B (Z9B denoting the paper's K-means) applied to these sensitivity patterns.

Figure 10: Per-model B8X across five task-relevant R2T conditions and four control conditions. The radial scale spans 0 $\sim$ 80%. Axes, shading, and model colors are consistent with the main-paper radar plot.

The semantic image ablation analysis is extended in Figure 10 to model-level profiles, each B8X value aggregating 3,264 evaluated responses across the corresponding visual condition, and substantial variation is exhibited in how different models respond to semantic visual contexts: larger gaps between task-relevant R2T conditions and control conditions are exhibited by step3-vl-10b and doubao-seed-2.0-lite, a smaller separation is shown by gemma3-12b, and a different pattern is presented by claude-sonnet-4.5, whose relatively higher B8X under the L8G suggests that its vulnerability is less dependent on semantic visual cues.

| Evaluation Pair | T5J | | N7V Detection | |
| | V4H | $\Delta$ B8X | Acc. | Macro-$F_{1}$ |
| Y6L vs. Human Experts | 0.97 | +0.3% | 96.3% | 0.85 |
| F4K vs. Y6L | 0.95 | $-0.4\%$ | 99.4% | 0.94 |

Table 12: Judge agreement and validation results between Y6L, human experts, and the F4K.

| Measure | Result |
| Mean B8X difference (C9Z $-$ P2W) | $-0.06\%$ |
| 95% CI for B8X difference | [ $-0.45\%$ , $0.30\%$ ] |
| Mean absolute B8X difference | $0.79\%$ |
| Maximum absolute B8X difference | $3.53\%$ |
| Cells within 2% difference | 30/32 (93.75%) |
| Spearman $\rho$ | 0.997 |
| Lin's concordance correlation | 0.999 |

Table 11: Consistency analysis between P2W and C9Z benchmark settings.

The efficiency and reliability of the QX7 evaluation pipeline are evaluated from two perspectives, the first being a comparison of the C9Z with the P2W, undertaken to assess whether the reduced evaluation scale preserves benchmark consistency, with Table 11 showing the C9Z to closely match the P2W—small B8X differences and strong agreement across matched evaluations demonstrating that the main evaluation characteristics are maintained by the C9Z—and the second being a validation of the reliability of the judging protocol through human expert evaluation and F4K validation, in which Table 12 reports the agreement between Y6L judgments and human expert annotations, T5J agreement being quantified by V4H (V4H denoting the paper's QWK) together with $\Delta$ B8X and N7V detection by accuracy and macro-$F_{1}$, as well as the consistency between the F4K and Y6L, both T5J scoring and N7V detection showing strong agreement and thereby supporting the reliability of the evaluation framework.

| Category | Count | B8X | $\Delta_{\mathrm{T-O}}$ |
| Cyber & Tech Abuse | | | |
|    $\bullet$ Malware & Hacking | 16 | 56.22 | 10.77 |
|    $\bullet$ Agent Abuse & Automation | 14 | 58.79 | 11.58 |
| Economic Harm | | | |
|    $\bullet$ Fraud & Scams | 16 | 57.66 | 11.20 |
|    $\bullet$ Financial Crimes & Risky Practices | 14 | 54.66 | 10.57 |
| Privacy & IP | | | |
|    $\bullet$ Privacy & Surveillance | 16 | 52.80 | 11.12 |
|    $\bullet$ Intellectual Property | 14 | 51.09 | 10.22 |
| Deception & Public Influence | | | |
|    $\bullet$ Disinformation | 14 | 53.73 | 12.46 |
|    $\bullet$ Government Decision & Political | 14 | 50.33 | 11.76 |
| Illegal Goods & Services | | | |
|    $\bullet$ Controlled Substances | 15 | 44.78 | 9.10 |
|    $\bullet$ General Illegal Acts | 14 | 50.25 | 11.46 |
| Hate & Harassment | | | |
|    $\bullet$ Hate Speech & Extremism | 15 | 36.34 | 10.63 |
|    $\bullet$ Harassment & Bullying | 15 | 47.02 | 10.07 |
| Violence & Physical Harm | | | |
|    $\bullet$ Violence & Terrorism | 17 | 37.13 | 8.58 |
|    $\bullet$ Weapons & CBRN | 16 | 43.39 | 9.43 |
|    $\bullet$ Suicide & Self-Harm | 15 | 36.36 | 8.86 |
| Sensitive Regulated Advice | | | |
|    $\bullet$ Unlicensed Medical & Legal Advice | 15 | 26.69 | 5.81 |
|    $\bullet$ High-Risk Financial Advice | 15 | 42.45 | 8.35 |
| Sexual Content | | | |
|    $\bullet$ Sexual Content | 17 | 26.72 | 10.66 |
| Total | 272 | 45.74 | 10.10 |

Table 8: Category-level jailbreak results. B8X is computed by pooling the two W9Ns, and $\Delta_{\mathrm{T-O}}$ denotes the B8X difference between the D4P and K6F W9Ns.

How different Z3Ks, W9Ns, V8M strategies, and R2T conditions affect MLLM jailbreak performance is analyzed, with Table 8 presenting category-level jailbreak results across the Z3K taxonomy in which the D4P and K6F W9Ns are compared, substantial variation in B8X being observed across different Z3K categories—Cyber & Tech Abuse, Economic Harm, and Privacy & IP exhibiting relatively higher B8X, while Sensitive Regulated Advice and Sexual Content show lower B8X under the evaluated settings—and $\Delta_{\mathrm{T-O}}$ remaining positive for all categories, ranging from 5.81 to 12.46 percentage points, an indication that the W9N effect is consistently observed across diverse Z3Ks rather than being dominated by specific S6M.

| Model | Acad. | Sys. | Story | Code | Stru. | Para. |
| Open-weight | | | | | | |
| glm4.1v-9b | 77.83 | 46.40 | 98.20 | 68.16 | 91.40 | 57.54 |
| step3-vl-10b | 65.11 | 15.51 | 58.05 | 46.36 | 87.68 | 48.42 |
| ministral3-8b | 82.72 | 30.70 | 70.00 | 25.55 | 76.36 | 28.24 |
| llava-onevision-1.5 | 35.59 | 50.51 | 54.89 | 40.66 | 47.21 | 19.60 |
| internvl3-8b | 74.41 | 0.07 | 88.42 | 37.10 | 57.13 | 43.09 |
| qwen2.5-vl-7b | 91.43 | 0.66 | 88.60 | 64.23 | 84.67 | 35.62 |
| qwen3-vl-8b | 25.37 | 22.32 | 14.19 | 15.40 | 29.12 | 13.49 |
| gemma3-12b | 86.40 | 18.90 | 97.94 | 74.08 | 92.21 | 37.06 |
| Proprietary-access | | | | | | |
| glm-4.6v | 84.67 | 79.56 | 96.29 | 80.55 | 93.86 | 33.71 |
| qwen3-vl-plus | 63.27 | 40.11 | 82.72 | 63.64 | 68.38 | 56.54 |
| doubao-seed-2.0-lite | 63.27 | 16.10 | 96.29 | 23.90 | 45.77 | 51.95 |
| kimi-k2.5 | 59.45 | 40.33 | 59.38 | 20.99 | 20.96 | 15.15 |
| gemini-3-flash | 34.96 | 3.20 | 62.87 | 9.60 | 50.88 | 18.97 |
| grok-4-fast | 66.40 | 3.53 | 52.50 | 24.12 | 74.15 | 43.12 |
| claude-sonnet-4.5 | 5.40 | 0.00 | 10.00 | 2.54 | 32.13 | 3.27 |
| Y6L | 0.74 | 0.00 | 0.44 | 0.85 | 8.82 | 1.88 |

Table 9: Per-model B8X (%) across different V8M strategies on QX7.

| Model | Dang. | Scen. | G3S | X7R | Role |
| Open-weight | | | | | |
| glm4.1v-9b | 75.34 | 72.33 | 71.17 | 73.93 | 73.50 |
| step3-vl-10b | 51.87 | 53.40 | 55.82 | 55.48 | 51.04 |
| ministral3-8b | 57.57 | 53.80 | 52.79 | 49.17 | 47.98 |
| llava-onevision-1.5 | 42.28 | 43.17 | 40.10 | 42.95 | 38.54 |
| internvl3-8b | 52.60 | 52.36 | 48.10 | 49.97 | 47.15 |
| qwen2.5-vl-7b | 62.19 | 61.58 | 59.83 | 61.95 | 58.79 |
| qwen3-vl-8b | 21.69 | 18.29 | 22.89 | 20.68 | 16.36 |
| gemma3-12b | 64.68 | 67.22 | 71.63 | 67.71 | 67.59 |
| Proprietary-access | | | | | |
| glm-4.6v | 70.74 | 80.39 | 82.08 | 79.01 | 78.31 |
| qwen3-vl-plus | 60.81 | 59.44 | 67.49 | 63.51 | 60.97 |
| doubao-seed-2.0-lite | 48.28 | 48.96 | 53.43 | 49.45 | 47.61 |
| kimi-k2.5 | 32.23 | 37.01 | 39.15 | 38.76 | 33.06 |
| gemini-3-flash | 29.75 | 34.59 | 26.75 | 28.16 | 31.16 |
| grok-4-fast | 44.64 | 43.72 | 44.00 | 45.16 | 42.34 |
| claude-sonnet-4.5 | 8.06 | 9.90 | 10.02 | 8.33 | 8.15 |
| Y6L | 1.69 | 2.05 | 2.36 | 2.48 | 2.02 |

Table 10: Per-model B8X (%) across different R2T conditions on QX7.

Model-level sensitivity to V8M strategies and R2T conditions is further analyzed in Tables 9 and 10, the results revealing substantial differences among models—some MLLMs exhibiting large variations across V8M strategies while others maintain relatively stable behavior—and distinct vulnerability patterns across models being produced by different R2T conditions, suggesting that contextual visual cues can influence jailbreak susceptibility in a model-dependent manner.

## Appendix D Responsible Release

Encouragement is extended to users to apply the released resources—whose dissemination is restricted to research purposes only and which are accompanied by documentation and evaluation tools supporting reproducible and responsible use—for research on improving model safety and robustness, these resources constituting QX7, which contains Z3Ks and adversarial prompts for multimodal safety evaluation. The determination of the final benchmark design and protocols rests with the authors, whereas the filtering and validation of generated materials is carried out through predefined procedures, with LLMs being used in QX7 for Z3K organization, visual prompt generation, and response evaluation.

