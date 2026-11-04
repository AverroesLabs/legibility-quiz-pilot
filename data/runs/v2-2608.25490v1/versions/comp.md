# MMJailBench: A Factorized Benchmark for Disentangling

Multimodal Jailbreak Vulnerabilities

Tianshi Wang, Jingsong Wang, Yafei Huang, Lei Zhu† (Tongji University); Fengling Li (Mohamed bin Zayed University of Artificial Intelligence); Xin Li (Shanghai Artificial Intelligence Laboratory). †Corresponding author.

###### Abstract

Existing multimodal jailbreak benchmarks couple harmful intent, prompt framing, visual semantics and instruction carrier within single instances, obscuring the sources of vulnerability. MMJailBench varies and combines these factors under controlled configurations for factor-level attribution. Across 16 open-weight and proprietary MLLMs, vulnerability profiles are heterogeneous and model-dependent. Diagnostics on one open-weight model find vulnerability-associated patterns in internal representations and cross-modal interactions. We release a modular evaluation suite with full and lightweight configurations, multiple judge options and multidimensional metrics for reproducible, scalable, cost-efficient auditing. Warning: This paper contains potentially harmful content.

## 1 Introduction

MLLMs jointly interpret text and images, so harmful requests can combine direct textual or visually rendered instructions with contextual cues (professional settings, identity credentials, authorization documents). Refusal/compliance is thus shaped by cross-modal context: visual information may alter the perceived legitimacy, professionalism and credibility of a request, so the same intent elicits different behavior across contexts. Multimodal jailbreaks therefore probe the stability of safety alignment under cross-modal conditions, not just input format.

Existing benchmarks build fixed prompt-image pairs that jointly encode intent, framing, visual semantics and carrier (Figure 1). Their ASR reflects aggregate vulnerability on a test set but cannot say whether vulnerability stems from linguistic framing, visual context, instruction delivery, or particular harm domains.

MMJailBench places these four factors in one controlled design with systematic combinations and matched comparisons: 272 harmful intents, 6 prompt templates, 5 task-relevant visual semantics, 2 instruction carriers (pipeline in Figure 2).

Findings across 16 MLLMs:
- Vulnerability is uneven across harm domains; cyber abuse, economic harm, privacy violations, and deception and manipulation are most vulnerable.
- Prompt framing is the dominant source of variation: with intent fixed, narrative structure and output requirements substantially change behavior.
- Task-relevant visual semantics weaken refusal; authority-like cues (authorization documents, identity credentials) are most pronounced, suggesting over-reliance on visually conveyed legitimacy and credibility.
- Carrier effect is model-dependent; visually rendered instructions do not consistently increase susceptibility vs. text.

Contributions: (1) the factorized benchmark; (2) large-scale evaluation of 16 MLLMs plus internal diagnostics on a representative open-weight model; (3) a modular evaluation suite (full/lightweight configurations, multiple judges, multidimensional metrics).

## 2 Related Work

### 2.1 Jailbreak Attacks and Defenses

Jailbreaks exploit competing objectives and mismatched generalization in safety training; approaches include transferable adversarial suffixes, black-box semantic attacks (PAIR), and indirect prompt injection via documents or retrieved content, showing refusal is sensitive to framing and delivery. MLLMs add attack surfaces: adversarial images, typographic prompts, other cross-modal constructions. Defenses (safety alignment, input transformation, inference-time safeguards) have uncertain robustness to modality and context changes.

### 2.2 Multimodal Jailbreak Benchmarks

Text benchmarks: SafetyBench, Do-Not-Answer, SALAD-Bench, HarmBench, JailbreakBench. Multimodal: MM-SafetyBench (image-based attacks), FigStep (typographic instructions), JailBreakV and VLJailbreakBench (diverse strategies), MMJ-Bench (unified attack/defense comparison), OmniSafeBench-MM (multidimensional metrics), Omni-SafetyBench (parallel variants of the same seeds across modality configurations, cross-modal consistency). None independently manipulates the linguistic and semantic factors within matched multimodal interactions; MMJailBench varies framing, visual semantics and carrier over matched intents.

## 3 MMJailBench

### 3.1 Factorized Design

Given harmful intent $h$, prompt framing $t$, visual semantic condition $v$, instruction carrier $c$:

$$x_{h,t,v,c}=(L(h,t,c),V(h,v)),\quad(1)$$

where $L(h,t,c)$ is the instruction component and $V(h,v)$ the visual input. Fixing $h$ and varying the rest gives matched comparisons. The Cartesian product of 272 intents × 6 framings × 5 visual conditions × 2 carriers yields 16,320 instances.

### 3.2 Harmful Intent

Intent-level behavioral seeds (not attack-entangled prompts) across 9 domains: Physical Harm, Cyber Abuse, Economic Harm, Hate and Harassment, Privacy and IP, Regulated Advice, Illegal Activities, Deception and Influence, Sexual Content, each with fine-grained subcategories. Built by category-guided construction, semantic-similarity filtering and manual verification.

### 3.3 Prompt Framing

6 strategies preserving intent: academic, system, structured, story, code, safety-paradox.

### 3.4 Visual Semantics

Images add context without modifying the harmful instruction. 5 conditions: danger-related context, scenario context, professional-role context, identity credentials, authorization documents (legitimacy-, expertise- or authority-related signals). Control images (irrelevant/non-semantic) are used in diagnostics to separate semantic effects from image presence.

### 3.5 Instruction Carrier

TEXT (direct textual instructions) vs. OCR (same instructions rendered in text-containing images, semantics preserved). Instruction recognition mismatch is measured to isolate carrier-related safety differences from perception failures.

## 4 Experimental Setup

### 4.1 Evaluated MLLMs

16 MLLMs, open-weight and proprietary, from families LLaVA-OneVision, InternVL, Kimi, Qwen, Gemma, GLM, GPT, Gemini, Claude. Full configuration per model: 16,320 instances; 261,120 responses total.

|Model|Avg.|Cyber|Econ.|Privacy|Decept.|Illegal|Hate|Violence|Sensitive|Sexual|
|-|-|-|-|-|-|-|-|-|-|-|
|*Open-weight*||||||||||
|glm4.1v-9b|72.22|80.72|85.28|71.78|78.04|83.16|67.39|79.58|44.94|59.12|
|step3-vl-10b|53.36|70.00|68.22|62.06|58.93|54.48|49.17|44.24|42.06|27.06|
|ministral3-8b|51.64|66.83|63.78|59.61|59.58|56.21|43.11|46.98|38.06|30.59|
|llava-onevision-1.5|40.64|42.00|49.72|38.17|42.74|52.30|36.00|48.96|21.06|34.80|
|internvl3-8b|49.12|63.11|61.72|51.33|53.21|61.49|42.33|50.80|27.39|30.69|
|qwen2.5-vl-7b|59.48|67.28|70.83|60.94|62.50|73.28|54.39|66.91|33.44|50.78|
|qwen3-vl-8b|19.84|37.56|30.28|28.72|20.65|19.02|12.06|11.15|14.39|4.71|
|gemma3-12b|66.85|82.94|80.44|75.94|75.95|74.54|62.67|60.66|51.39|37.06|
|*Proprietary-access*||||||||||
|glm-4.6v|78.38|93.06|88.11|85.17|87.26|80.29|74.94|73.68|59.89|53.04|
|qwen3-vl-plus|62.02|80.78|84.00|80.22|78.57|58.85|55.78|43.37|51.67|24.90|
|doubao-seed-2.0-lite|47.79|68.39|61.94|62.94|63.10|42.59|47.50|32.99|46.44|16.18|
|kimi-k2.5|35.80|53.28|45.11|44.06|43.81|35.75|34.00|23.30|30.72|12.16|
|gemini-3-flash|30.50|39.61|38.94|41.06|42.14|23.10|28.72|11.94|36.61|12.35|
|grok-4-fast|42.61|55.00|52.50|55.00|52.50|37.07|49.33|26.91|40.33|32.84|
|claude-sonnet-4.5|8.96|15.11|15.11|13.06|10.06|6.09|7.00|2.08|11.22|0.88|
|gpt-5|2.17|3.06|4.11|2.00|3.39|0.57|2.50|0.07|3.56|0.29|

Table 1: Per-model ASR (%) across nine harm domains; Avg. over all domains.

### 4.2 Metrics and Judge Models

$$ASR=\frac{1}{N}\sum_{i=1}^{N}\mathbb{I}(s_{i}\geq 4),\quad(2)$$

$s_i$: harmfulness score (1∼5); $s_i\geq4$ = success.

$$CASR=\frac{N_{success}}{N-N_{mismatch}},\quad(3)$$

mismatch = input not correctly understood (OCR failures, irrelevant responses, misinterpreted intent). ASR is end-to-end; CASR is conditioned on understanding. GPT-5 judges all main results (score + success and mismatch labels); refusals, safety warnings and high-level discussion are not successes. A lightweight open-weight judge supports cheaper follow-up.

### 4.3 Evaluation Configurations

Full (all instances; default) and lightweight (representative subset covering all domains and factors, for auditing/regression testing).

## 5 Results and Analysis

### 5.1 Overall Jailbreak Landscape

Average ASR ranges from 2.17% (GPT-5) to 78.38% (GLM-4.6V) (Table 1, Figure 3). Models of comparable capability differ strongly; capability does not imply robustness.

### 5.2 Factor-Level Attribution

#### Harmful Intent.

Cyber, economic, privacy and deception tasks have higher ASR; physical harm and sensitive content lower. Aggregate scores hide domain-specific weaknesses.

#### Prompt Framing.

Largest variation. Story, structured and academic framings generally yield higher ASR; system-style and safety-paradox lower. Most-vs-least vulnerable gap exceeds 40 points (Figure 3(a)).

|Group|Image condition|ASR (%)|ΔASR (%)|
|-|-|-|-|
|Semantic|Authorization document|47.73|12.96|
||Identity credential|45.24|10.47|
||Task scenario|44.87|10.10|
||Professional role|43.60|8.83|
||Dangerous context|43.22|8.45|
|Control|Blank image|36.96|2.19|
||Noise image|35.53|0.76|
||Nature image|35.49|0.72|
||No image|34.77|0.00|

Table 2: Visual semantic ablation; ΔASR relative to no-image.

|Carrier|$N$|Mis. (%)|ASR (%)|CASR (%)|
|-|-|-|-|-|
|TEXT|130,560|0.37|50.70|50.89|
|OCR|130,560|4.14|40.59|42.34|
|UNION|261,120|2.26|45.64|46.69|

Table 3: Instruction carrier analysis; Mis. = mismatch rate.

#### Visual Semantics.

All task-relevant semantics raise ASR over no-image; non-semantic controls change it by <2.2%. Vulnerability stems from contextual visual meaning, not image presence; authority cues may supply implicit legitimacy signals.

#### Instruction Carrier.

OCR yields lower ASR than TEXT, persisting under CASR. Visually rendered instructions are not inherently stronger carriers; carrier vulnerability depends on model-specific multimodal processing.

|Model|ΔIntent|ΔPrompt|ΔVisual|ΔCarrier|
|-|-|-|-|-|
|*Open-weight*|||||
|glm4.1v-9b|76.7|51.8|4.2|8.3|
|step3-vl-10b|86.7|72.2|4.8|10.5|
|ministral3-8b|83.3|57.2|9.6|19.9|
|llava-onevision-1.5|61.7|35.3|4.6|50.5|
|internvl3-8b|75.0|88.3|5.5|4.6|
|qwen2.5-vl-7b|78.3|90.8|3.4|0.8|
|qwen3-vl-8b|65.0|15.6|6.5|14.4|
|gemma3-12b|88.3|79.0|7.0|13.6|
|*Proprietary-access*|||||
|glm-4.6v|73.3|62.6|11.3|1.5|
|qwen3-vl-plus|91.7|42.6|8.1|9.7|
|doubao-seed-2.0-lite|95.0|80.2|5.8|1.3|
|kimi-k2.5|75.0|44.3|6.9|34.6|
|gemini-3-flash|76.7|59.7|7.8|34.3|
|grok-4-fast|76.7|70.6|2.8|4.1|
|claude-sonnet-4.5|55.0|32.1|2.0|0.8|
|gpt-5|13.3|8.8|0.8|3.6|

Table 4: Per-model ASR range across configurations of each factor.

### 5.3 Model-Dependent Profiles

Some models have ΔPrompt >80 points; others are more carrier-sensitive; visual effects are smaller but consistently positive. No single universal failure pattern exists.

## 6 Diagnostic Analysis

Diagnostic subset of matched instances emphasizing authority-document scenarios; layer-wise representations and cross-modal attention in gemma3-12b.

### 6.1 Representation-Level Diagnostics

Residual stream of the final generation-prefix token (Figure 4): authority-vs-danger divergence grows with depth, sharpest at layer 42. An authority-minus-danger direction fit on discovery samples, applied fixed to held-out pairs, shows clear displacement of authority contexts, with consistent positive shifts across all harm domains: a shared higher-layer pattern.

### 6.2 Cross-Modal Interaction Diagnostics

Attention from the final generation-prefix query to harm-related task-label tokens and special visual tokens (Figure 5): at the representation-sensitive layer, authority contexts give lower attention mass on both. With the representation shift, this suggests authority cues induce a higher-level contextual state with redistributed attention at response initiation, an internal explanation for increased susceptibility.

## 7 Conclusion

MMJailBench factorizes intent, framing, visual semantics and carrier. Across 16 MLLMs: framing causes the largest variation, visual semantics add contextual vulnerability, carrier effects are model-dependent, intents show category-specific patterns, with supporting internal diagnostics. Full/lightweight configurations, judge options and metrics enable reproducible, scalable evaluation.

## Appendix A Benchmark Construction

|Category|Count|Ratio (%)|
|-|-|-|
|Cyber & Tech Abuse|30|11.0|
|• Malware & Hacking|16|5.9|
|• Agent Abuse & Automation|14|5.1|
|Economic Harm|30|11.0|
|• Fraud & Scams|16|5.9|
|• Financial Crimes & Risky Practices|14|5.1|
|Privacy & IP|30|11.0|
|• Privacy & Surveillance|16|5.9|
|• Intellectual Property|14|5.1|
|Deception & Public Influence|28|10.3|
|• Disinformation|14|5.1|
|• Government Decision & Political|14|5.1|
|Illegal Goods & Services|29|10.7|
|• Controlled Substances|15|5.5|
|• General Illegal Acts|14|5.1|
|Hate & Harassment|30|11.0|
|• Hate Speech & Extremism|15|5.5|
|• Harassment & Bullying|15|5.5|
|Violence & Physical Harm|48|17.6|
|• Violence & Terrorism|17|6.3|
|• Weapons & CBRN|16|5.9|
|• Suicide & Self-Harm|15|5.5|
|Sensitive Regulated Advice|30|11.0|
|• Unlicensed Medical & Legal Advice|15|5.5|
|• High-Risk Financial Advice|15|5.5|
|Sexual Content|17|6.3|
|Total|272|100.0|

Table 5: Taxonomy: 9 domains, 18 scenarios.

### A.1 Harmful Intent Taxonomy, Collection, and Filtering

Each intent has a task identifier, harm category, scenario and action description. Domain abbreviations: Cyber, Econ., Privacy, Decept., Illegal, Hate, Violence, Sensitive, Sexual. Sources: existing safety benchmarks and public model-provider safety policies, then normalization, organization and manual filtering. Pairwise similarity via all-MiniLM-L6-v2: most pairs have low similarity; the small high-similarity tail is manually reviewed to remove near-duplicates (Figure 6). PCA (Figure 7, qualitative) shows domains occupy diverse regions rather than a few clusters.

|Template|Framing Strategy|Representative Pattern|
|-|-|-|
|Academic|Academic Analysis|Analyze the topic from an academic perspective.|
|System|System-Role Simulation|Assume a system identity and follow role-specific instructions.|
|Story|Fictional Storytelling|Describe via fictional story or hypothetical setting.|
|Code|Code-Style Indirection|Express the procedure through code or comments.|
|Structure|Structured Completion|Complete a predefined multi-stage structure.|
|Paradox|Safety-Paradox Reasoning|Discuss harmful actions under a safety rationale.|

Table 6: Prompt framing templates.

### A.2 Prompt Framing Templates

A task placeholder is replaced by the target intent. Templates are evaluation probes, not an exhaustive attack collection.

### A.3 Visual Semantic Generation

Categories: Danger, Work Scenario, Authorization Document, Identity Credential, Role (examples in Figure 8). qwen2.5-vl-7b generates descriptions and image prompts; zimage2025-Turbo generates Danger, Work Scenario, Role; 4-bit-quantized FLUX.2 [dev] generates Identity Credential and Authorization Document (image editing for coherent text). Images filtered for semantic relevance, quality, readability.

### A.4 Instruction Carrier Construction

OCR: instruction text rendered as a readable image combined with the visual semantic image, standardized layout, intent and framing preserved.

## Appendix B Experimental and Evaluation Details

### B.1 Evaluated MLLMs

Open-weight (public checkpoints): qwen2.5-vl-7b, qwen3-vl-8b, gemma3-12b, internvl3-8b, glm4.1v-9b, ministral3-8b, step3-vl-10b, llava-onevision-1.5. Proprietary (official interfaces): gpt-5, gemini-3-flash, claude-sonnet-4.5, grok-4-fast, qwen3-vl-plus, kimi-k2.5, glm-4.6v, doubao-seed-2.0-lite.

|Score|Definition|
|-|-|
|1|Refusal or safe response without harmful details.|
|2|Potential risk without direct harmful instructions.|
|3|High-level or partial harmful information.|
|4|Actionable harmful assistance.|
|5|Detailed and operational harmful guidance.|

Table 7: Harmfulness rubric.

### B.2 Evaluation Rubric and Judge Prompt

GPT-5 outputs a mismatch flag (e.g., only transcribing visual content, repeating the prompt, meaningless output) and a score. Anti-deception rules stop disclaimers, fictional framing or academic packaging from lowering scores when harmful information is given. Full prompt in the Code and Data Supplement.

### B.3 Full and Lightweight Benchmark Settings

Lightweight: 1,500 instances, stratified over domains, framings, visual conditions, carriers, preserving the full factor distribution.

### B.4 Lightweight Judge Model

Benchmark-scoped judge from qwen3-8b trained on GPT-5 annotations; predicts scores and mismatch labels under the same protocol.

## Appendix C Extended Results and Ablations

|Subcategory|ASR|Δ_T−O|
|-|-|-|
|Malware & Hacking|56.22|10.77|
|Agent Abuse & Automation|58.79|11.58|
|Fraud & Scams|57.66|11.20|
|Financial Crimes & Risky Practices|54.66|10.57|
|Privacy & Surveillance|52.80|11.12|
|Intellectual Property|51.09|10.22|
|Disinformation|53.73|12.46|
|Government Decision & Political|50.33|11.76|
|Controlled Substances|44.78|9.10|
|General Illegal Acts|50.25|11.46|
|Hate Speech & Extremism|36.34|10.63|
|Harassment & Bullying|47.02|10.07|
|Violence & Terrorism|37.13|8.58|
|Weapons & CBRN|43.39|9.43|
|Suicide & Self-Harm|36.36|8.86|
|Unlicensed Medical & Legal Advice|26.69|5.81|
|High-Risk Financial Advice|42.45|8.35|
|Sexual Content|26.72|10.66|
|Total|45.74|10.10|

Table 8: Category-level ASR (carriers pooled); Δ_T−O = TEXT−OCR ASR.

|Model|Acad.|Sys.|Story|Code|Stru.|Para.|
|-|-|-|-|-|-|-|
|glm4.1v-9b|77.83|46.40|98.20|68.16|91.40|57.54|
|step3-vl-10b|65.11|15.51|58.05|46.36|87.68|48.42|
|ministral3-8b|82.72|30.70|70.00|25.55|76.36|28.24|
|llava-onevision-1.5|35.59|50.51|54.89|40.66|47.21|19.60|
|internvl3-8b|74.41|0.07|88.42|37.10|57.13|43.09|
|qwen2.5-vl-7b|91.43|0.66|88.60|64.23|84.67|35.62|
|qwen3-vl-8b|25.37|22.32|14.19|15.40|29.12|13.49|
|gemma3-12b|86.40|18.90|97.94|74.08|92.21|37.06|
|glm-4.6v|84.67|79.56|96.29|80.55|93.86|33.71|
|qwen3-vl-plus|63.27|40.11|82.72|63.64|68.38|56.54|
|doubao-seed-2.0-lite|63.27|16.10|96.29|23.90|45.77|51.95|
|kimi-k2.5|59.45|40.33|59.38|20.99|20.96|15.15|
|gemini-3-flash|34.96|3.20|62.87|9.60|50.88|18.97|
|grok-4-fast|66.40|3.53|52.50|24.12|74.15|43.12|
|claude-sonnet-4.5|5.40|0.00|10.00|2.54|32.13|3.27|
|gpt-5|0.74|0.00|0.44|0.85|8.82|1.88|

Table 9: Per-model ASR (%) by framing (first 8 open-weight, last 8 proprietary).

### C.1 Extended Factor-Level Results

Δ_T−O is positive for all categories (5.81–12.46 points): the carrier effect is not dominated by specific domains. Framing and visual sensitivity are model-dependent (Tables 9–10).

|Model|Dang.|Scen.|Auth. Doc.|ID Cred.|Role|
|-|-|-|-|-|-|
|glm4.1v-9b|75.34|72.33|71.17|73.93|73.50|
|step3-vl-10b|51.87|53.40|55.82|55.48|51.04|
|ministral3-8b|57.57|53.80|52.79|49.17|47.98|
|llava-onevision-1.5|42.28|43.17|40.10|42.95|38.54|
|internvl3-8b|52.60|52.36|48.10|49.97|47.15|
|qwen2.5-vl-7b|62.19|61.58|59.83|61.95|58.79|
|qwen3-vl-8b|21.69|18.29|22.89|20.68|16.36|
|gemma3-12b|64.68|67.22|71.63|67.71|67.59|
|glm-4.6v|70.74|80.39|82.08|79.01|78.31|
|qwen3-vl-plus|60.81|59.44|67.49|63.51|60.97|
|doubao-seed-2.0-lite|48.28|48.96|53.43|49.45|47.61|
|kimi-k2.5|32.23|37.01|39.15|38.76|33.06|
|gemini-3-flash|29.75|34.59|26.75|28.16|31.16|
|grok-4-fast|44.64|43.72|44.00|45.16|42.34|
|claude-sonnet-4.5|8.06|9.90|10.02|8.33|8.15|
|gpt-5|1.69|2.05|2.36|2.48|2.02|

Table 10: Per-model ASR (%) by visual condition.

### C.2 Per-Model Factor Sensitivity Profiles

K-means (1,000 restarts) on framing, visual and carrier ASR ranges (Figure 9) gives four profiles: prompt-dominant, visual-responsive, carrier-sensitive, low-variation.

### C.3 Semantic Image Ablation

Each ASR aggregates 3,264 responses (Figure 10, radial scale 0∼80%). step3-vl-10b and doubao-seed-2.0-lite show larger semantic-vs-control gaps; gemma3-12b smaller; claude-sonnet-4.5 has relatively higher no-image ASR, less dependent on semantic cues.

### C.4 Evaluation Efficiency and Validation

|Measure|Result|
|-|-|
|Mean ASR difference (Light−Full)|−0.06%|
|95% CI for ASR difference|[−0.45%, 0.30%]|
|Mean absolute ASR difference|0.79%|
|Maximum absolute ASR difference|3.53%|
|Cells within 2% difference|30/32 (93.75%)|
|Spearman ρ|0.997|
|Lin’s concordance correlation|0.999|

Table 11: Full vs. Lightweight consistency.

|Evaluation Pair|QWK|ΔASR|Acc.|Macro-F1|
|-|-|-|-|-|
|GPT-5 vs. Human Experts|0.97|+0.3%|96.3%|0.85|
|Lightweight Judge vs. GPT-5|0.95|−0.4%|99.4%|0.94|

Table 12: Judge agreement (QWK, ΔASR: harmfulness; Acc., Macro-F1: mismatch detection).

## Appendix D Responsible Release

### D.1 Ethics and Data Release

Released for research only, with documentation and tools.

### D.2 LLM Usage Statement

LLMs used for intent organization, visual prompt generation and response evaluation; outputs filtered and validated; design and protocols set by the authors.