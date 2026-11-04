# MMJailBench: A Factorized Benchmark for Disentangling

Multimodal Jailbreak Vulnerabilities

Tianshi Wang

Affiliation:  Tongji University

  
Jingsong Wang

Affiliation:  Tongji University

  
Yafei Huang

Affiliation:  Tongji University

  
Fengling Li

Affiliation:  Mohamed bin Zayed University of Artificial Intelligence

  
Xin Li

Affiliation:  Shanghai Artificial Intelligence Laboratory

  
Lei Zhu
††thanks: Corresponding author.
Affiliation:  Tongji University

###### Abstract

We present MMJailBench, a factorized multimodal jailbreak benchmark over harmful intent, prompt framing, visual semantics, and instruction carrier, with controlled configurations supporting factor-level attribution. Across 16 open-weight and proprietary MLLMs, vulnerability profiles are heterogeneous and model-dependent. ASR varies markedly across harm domains. Prompt framing dominates variation. Task-relevant visual semantics raise susceptibility, most strongly for authority-like cues. Visually rendered instructions do not consistently exceed direct textual instructions. Diagnostics on one open-weight model locate vulnerability-associated patterns in internal representations and cross-modal interactions. We release a modular evaluation suite with full and lightweight configurations, multiple judge options, and multidimensional metrics for reproducible, scalable, low-cost auditing. Warning: This paper contains potentially harmful content.

## 1 Introduction

MLLMs are deployed in document understanding, code generation, workflow assistance, and automated operations (Liu et al., 2023; Dai et al., 2023). Harmful requests to them may pair textual or visually rendered instructions (Gong et al., 2025) with professional settings, identity credentials, or authorization documents, and refusal decisions are conditioned on cross-modal context (Li et al., 2024b). Visual input can shift the perceived legitimacy, professionalism, and credibility of a request, so a single intent can elicit different behaviors across contexts (Liu et al., 2024); multimodal jailbreaks thus probe the stability of safety alignment under cross-modal conditions (Luo et al., 2024).

Figure 1: Motivation of MMJailBench. Existing benchmarks entangle multiple factors within fixed instances; MMJailBench factorizes them into a controlled evaluation space.

Text and multimodal safety benchmarks cover harmful behaviors (Li et al., 2024a; Mazeika et al., 2024; Chao et al., 2024), visual attacks, and image-based text injection (Liu et al., 2024; Gong et al., 2025; Luo et al., 2024), using fixed prompt-image pairs per task that jointly encode harmful intent, prompt framing, visual semantics, and instruction carrier; their ASR reflects aggregate vulnerability on a given test set (Weng et al., 2025; Jia et al., 2025).

MMJailBench comprises 272 harmful intents, 6 prompt templates, 5 types of task-relevant visual semantics, and 2 instruction carriers under matched comparisons, together with a modular suite offering full and lightweight configurations, multiple judge options, and multidimensional metrics.

Figure 2: Construction and evaluation pipeline of MMJailBench over harmful intent, prompt framing, visual semantics, and instruction carrier.

On 16 MLLMs, cyber abuse, economic harm, privacy violations, and deception and manipulation show higher vulnerability; prompt framing is the dominant source of variation at fixed intent; task-relevant visual semantics weaken refusal, with authorization documents and identity credentials most pronounced, consistent with over-reliance on visually conveyed legitimacy and credibility; carrier effects are model-dependent, with OCR not consistently above TEXT. Representation and attention diagnostics on one open-weight model identify associated internal patterns.

Contributions:

•

MMJailBench, a factorized benchmark over harmful intent, prompt framing, visual semantics, and instruction carrier, supporting factor-level attribution.

•

Evaluation of 16 open-weight and proprietary MLLMs establishing uneven domain vulnerability, dominance of prompt framing, refusal weakening under authority-like visual cues, and model-dependent carrier effects, with diagnostics on internal representations and cross-modal interactions.

•

A modular evaluation suite with full and lightweight configurations, multiple judge options, and multidimensional metrics.

## 2 Related Work

### 2.1 Jailbreak Attacks and Defenses

Jailbreaks circumvent safety alignment (Wei et al., 2023; Zou et al., 2023), attributed to competing objectives and mismatched generalization (Wei et al., 2023). Transferable adversarial suffixes (Zou et al., 2023), black-box semantic attacks such as PAIR (Chao et al., 2025), and indirect prompt injection through documents or retrieved content (Greshake et al., 2023) establish sensitivity of refusal to framing and delivery (Wei et al., 2023; Chao et al., 2025; Greshake et al., 2023).

In MLLMs, instructions and cues span language and vision (Gong et al., 2025; Liu et al., 2024; Li et al., 2024b): adversarial images (Liu et al., 2024; Li et al., 2024b), typographic prompts (Gong et al., 2025), and other cross-modal constructions (Luo et al., 2024). Defenses span alignment, input transformation, and inference-time safeguards (Jain et al., 2023; Wang et al., 2024), with uncertain robustness across modality and context shifts; identical intent may receive different treatment under different linguistic or visual contexts.

### 2.2 Multimodal Jailbreak Benchmarks

SafetyBench (Zhang et al., 2024), Do-Not-Answer (Wang et al., 2023), SALAD-Bench (Li et al., 2024a), HarmBench (Mazeika et al., 2024), and JailbreakBench (Chao et al., 2024) provide taxonomies and reproducible protocols. MM-SafetyBench (Liu et al., 2024) targets image-based attacks, FigStep (Gong et al., 2025) typographic instructions, and JailBreakV (Luo et al., 2024) and VLJailbreakBench (Wang et al., 2025) broader strategy coverage; MMJ-Bench (Weng et al., 2025) unifies attack/defense comparison, and OmniSafeBench-MM (Jia et al., 2025) adds multidimensional metrics.

Omni-SafetyBench (Pan et al., 2025) builds parallel variants of harmful seeds across modality configurations for cross-modal consistency. None independently manipulates the linguistic and semantic factors within matched multimodal interactions; MMJailBench varies prompt framing, task-relevant visual semantics, and instruction carrier over matched harmful intents.

## 3 MMJailBench

MMJailBench (Figure 2) constructs matched multimodal scenarios by systematically varying harmful intents, prompt framings, visual semantics, and instruction carriers within a unified evaluation space.

### 3.1 Factorized Design

Prior benchmarks center on predefined attack instances or modality variations with harmful intent, linguistic formulation, visual context, and instruction presentation jointly encoded. MMJailBench uses

| | $x_{h,t,v,c}=(L(h,t,c),V(h,v)),$
| | (1)

with $h$ harmful intent, $t$ prompt framing, $v$ visual semantics, and $c$ instruction carrier; $L(h,t,c)$ is the instruction component and $V(h,v)$ the visual input. With 272 harmful intents, 6 prompt framing strategies, 5 visual semantic conditions, and 2 instruction carriers, the Cartesian product yields 16,320 instances, consistent across models and harm categories.

### 3.2 Harmful Intent

Intents are intent-level behavioral seeds rather than collected jailbreak prompts, so the objective stays fixed across contextual variation. There are 272 harmful intents in 9 major harm domains: Physical Harm, Cyber Abuse, Economic Harm, Hate and Harassment, Privacy and IP, Regulated Advice, Illegal Activities, Deception and Influence, and Sexual Content, each split into subcategories. Construction is category-guided, followed by semantic similarity filtering and manual verification.

### 3.3 Prompt Framing

6 strategies: academic, system, structured, story, code, and safety-paradox, e.g., embedding the request in fiction, as a structured generation task, or as an apparently legitimate system requirement, with intent held fixed.

### 3.4 Visual Semantics

5 task-relevant conditions: danger-related context, scenario context, professional-role context, identity credentials, and authorization documents, spanning legitimacy-, expertise-, and authority-related signals; the image adds context without modifying the instruction. Irrelevant or non-semantic control images enter diagnostic evaluations.

### 3.5 Instruction Carrier

2 carriers: TEXT and OCR (same instruction rendered into a text-containing image, semantics preserved). Instruction recognition mismatch is recorded to separate perception failures from carrier-related safety differences.

## 4 Experimental Setup

All models share the same factorized framework; prompts, judge criteria, and implementation details are in the appendix.

### 4.1 Evaluated MLLMs

16 MLLMs, open-weight and proprietary-access, from LLaVA-OneVision (An et al., 2025), InternVL (Zhu et al., 2025), Kimi (Team et al., 2026), Qwen (Bai et al., 2025b; Bai et al., 2025a), Gemma (Team et al., 2025), GLM (Hong et al., 2025), GPT (Singh et al., 2025), Gemini (Team et al., 2023), and Claude (Anthropic, 2025). Each runs the full 16,320 instances, for 261,120 responses.

| Model | Avg. | Cyber | Econ. | Privacy | Decept. | Illegal | Hate | Violence | Sensitive | Sexual

| Open-weight

| glm4.1v-9b | 72.22 | 80.72 | 85.28 | 71.78 | 78.04 | 83.16 | 67.39 | 79.58 | 44.94 | 59.12

| step3-vl-10b | 53.36 | 70.00 | 68.22 | 62.06 | 58.93 | 54.48 | 49.17 | 44.24 | 42.06 | 27.06

| ministral3-8b | 51.64 | 66.83 | 63.78 | 59.61 | 59.58 | 56.21 | 43.11 | 46.98 | 38.06 | 30.59

| llava-onevision-1.5 | 40.64 | 42.00 | 49.72 | 38.17 | 42.74 | 52.30 | 36.00 | 48.96 | 21.06 | 34.80

| internvl3-8b | 49.12 | 63.11 | 61.72 | 51.33 | 53.21 | 61.49 | 42.33 | 50.80 | 27.39 | 30.69

| qwen2.5-vl-7b | 59.48 | 67.28 | 70.83 | 60.94 | 62.50 | 73.28 | 54.39 | 66.91 | 33.44 | 50.78

| qwen3-vl-8b | 19.84 | 37.56 | 30.28 | 28.72 | 20.65 | 19.02 | 12.06 | 11.15 | 14.39 | 4.71

| gemma3-12b | 66.85 | 82.94 | 80.44 | 75.94 | 75.95 | 74.54 | 62.67 | 60.66 | 51.39 | 37.06

| Proprietary-access

| glm-4.6v | 78.38 | 93.06 | 88.11 | 85.17 | 87.26 | 80.29 | 74.94 | 73.68 | 59.89 | 53.04

| qwen3-vl-plus | 62.02 | 80.78 | 84.00 | 80.22 | 78.57 | 58.85 | 55.78 | 43.37 | 51.67 | 24.90

| doubao-seed-2.0-lite | 47.79 | 68.39 | 61.94 | 62.94 | 63.10 | 42.59 | 47.50 | 32.99 | 46.44 | 16.18

| kimi-k2.5 | 35.80 | 53.28 | 45.11 | 44.06 | 43.81 | 35.75 | 34.00 | 23.30 | 30.72 | 12.16

| gemini-3-flash | 30.50 | 39.61 | 38.94 | 41.06 | 42.14 | 23.10 | 28.72 | 11.94 | 36.61 | 12.35

| grok-4-fast | 42.61 | 55.00 | 52.50 | 55.00 | 52.50 | 37.07 | 49.33 | 26.91 | 40.33 | 32.84

| claude-sonnet-4.5 | 8.96 | 15.11 | 15.11 | 13.06 | 10.06 | 6.09 | 7.00 | 2.08 | 11.22 | 0.88

| gpt-5 | 2.17 | 3.06 | 4.11 | 2.00 | 3.39 | 0.57 | 2.50 | 0.07 | 3.56 | 0.29

Table 1: Per-model ASR (%) across nine major harmful domains in our MMJailBench. Avg. denotes the average ASR over all evaluated harmful domains.

### 4.2 Metrics and Judge Models

| | $ASR=\frac{1}{N}\sum_{i=1}^{N}\mathbb{I}(s_{i}\geq 4),$
| | (2)

with $s_{i}$ the harmfulness score of response $i$ on a 1 $\sim$ 5 scale. For carrier analysis:

| | $CASR=\frac{N_{success}}{N-N_{mismatch}},$
| | (3)

where $N_{mismatch}$ counts OCR recognition failures, irrelevant responses, and misinterpretations of the evaluated intent; ASR is end-to-end, CASR conditioned on understanding.

GPT-5 judges every response, assigning a harmfulness score plus jailbreak-success and instruction-mismatch labels; refusals, safety warnings, and high-level discussion do not count as success. All main results use GPT-5 judgments. A benchmark-specific lightweight judge built on an open-weight MLLM supports low-cost follow-up evaluation under the same protocol.

### 4.3 Evaluation Configurations

Full configuration: all instances, used for model-, domain-, and factor-level analysis. Lightweight configuration: representative subset covering harmful domains, prompt framings, visual semantics, and instruction carriers, for auditing and regression testing. Results are full-configuration unless stated.

## 5 Results and Analysis

### 5.1 Overall Jailbreak Landscape

Figure 3 and Table 1 report model-level ASR. Average ASR ranges from 2.17% (GPT-5) to 78.38% (GLM-4.6V); models of comparable multimodal capability can differ sharply, so general capability does not imply jailbreak robustness. Cyber abuse, economic harm, privacy-related behaviors, and deception-related tasks show higher ASR; physical harm is comparatively lower.

Figure 3: Model-level vulnerability profiles across prompt framing and visual semantic factors (radar plots of ASR under controlled factor changes).

### 5.2 Factor-Level Attribution

Each factor is varied with the others fixed.

#### Harmful Intent.

Per Table 1, cyber abuse, economic harm, privacy-related behaviors, and deception-related tasks show higher ASR across models; physical harm and sensitive content are lower. Aggregate scores mask domain-specific weaknesses.

#### Prompt Framing.

Framing yields the largest variation (Figure 3(a)). Story, structured, and academic framings generally give higher ASR; system-style and safety-paradox framings lower. The most–least vulnerable gap exceeds 40 percentage points.

| Group | Image condition | ASR (%) | $\Delta$ ASR (%)

| Semantic | Authorization document | 47.73 | 12.96

| Identity credential | 45.24 | 10.47

| Task scenario | 44.87 | 10.10

| Professional role | 43.60 | 8.83

| Dangerous context | 43.22 | 8.45

| Control | Blank image | 36.96 | 2.19

| Noise image | 35.53 | 0.76

| Nature image | 35.49 | 0.72

| No image | 34.77 | 0.00

Table 2: Visual semantic ablation results comparing semantic and control image conditions. $\Delta$ ASR denotes the ASR increase relative to the no-image condition.

| Carrier | $N$ | Mis. (%) | ASR (%) | CASR (%)

| TEXT | 130,560 | 0.37 | 50.70 | 50.89

| OCR | 130,560 | 4.14 | 40.59 | 42.34

| UNION | 261,120 | 2.26 | 45.64 | 46.69

Table 3: Instruction carrier analysis under TEXT and OCR settings. $N$ denotes the number of evaluated model responses, and Mis. represents the instruction mismatch rate.

#### Visual Semantics.

Per Figure 3(b) and Table 2, all task-relevant conditions exceed no-image: authorization documents +12.96%, identity credentials +10.47%, task scenarios +10.10%. Blank, noise, and nature controls change ASR by less than 2.2%. Authority-related cues plausibly act as implicit legitimacy signals.

#### Instruction Carrier.

Per Table 3, OCR is below TEXT in ASR (40.59% vs. 50.70%) and in CASR (42.34% vs. 50.89%). Visually rendered instructions are not inherently stronger carriers; carrier-related vulnerability depends on model-specific multimodal processing.

| Model | $\Delta$ Intent | $\Delta$ Prompt | $\Delta$ Visual | $\Delta$ Carrier

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

| gpt-5 | 13.3 | 8.8 | 0.8 | 3.6

Table 4: Model-dependent vulnerability profiles across MMJailBench factors. Each $\Delta$ score denotes the range of ASR variation across configurations of the corresponding factor.

### 5.3 Model-Dependent Profiles

Table 4 gives per-model factor signatures. Some models exceed 80 percentage points in $\Delta$ Prompt; others are more carrier-sensitive. Visual effects are smaller but consistently positive. No single failure pattern governs all models, so factorized profiles are more informative than aggregate ASR.

## 6 Diagnostic Analysis

A diagnostic subset of matched MMJailBench instances emphasizing authority-document scenarios is used to probe layer-wise representations and cross-modal attention in gemma3-12b.

### 6.1 Representation-Level Diagnostics

Residual-stream representations of the final generation-prefix token (Figure 4): authority–danger divergence grows with depth, rising most sharply at layer 42 (Figure 4(a)). An authority-minus-danger direction fit on discovery samples and applied fixed to held-out pairs displaces authority-document contexts (Figure 4(b)), with consistent positive shifts across all harm domains (Figure 4(c)), i.e., a shared higher-layer pattern across diverse harmful objectives.

Figure 4: Representation-level diagnostics of gemma3-12b under authority-document and danger-image contexts across layers and harm domains.

### 6.2 Cross-Modal Interaction Diagnostics

Attention from the final generation-prefix query to harm-related task-label tokens and special visual tokens (Figure 5) differs layer-wise between authority-document and danger-image contexts. At the representation-sensitive layer, authority-document contexts place lower attention mass on both token types. Together with the representation displacement, this is consistent with authority cues inducing a higher-level contextual state with redistributed attention at response initiation, offering an internal account of the increased susceptibility.

Figure 5: Cross-modal interaction diagnostics in gemma3-12b: attention to harm-related textual tokens and visual tokens across layers under authority-document and danger-image contexts.

## 7 Conclusion

MMJailBench varies harmful intents, prompt framings, visual semantics, and instruction carriers under control. Over 16 open-weight and proprietary MLLMs: prompt framing introduces the largest variation, visual semantics add vulnerability via contextual cues, carrier effects are model-dependent, and harmful intents show category-specific patterns; diagnostics reveal associated representation and cross-modal interaction patterns. Full and lightweight configurations, flexible judges, and multidimensional metrics support reproducible, scalable multimodal safety evaluation.

## Appendix A Benchmark Construction

| Category | Count | Ratio (%)

| Cyber & Tech Abuse | 30 | 11.0

|     $\bullet$ Malware & Hacking | 16 | 5.9

|     $\bullet$ Agent Abuse & Automation | 14 | 5.1

| Economic Harm | 30 | 11.0

|     $\bullet$ Fraud & Scams | 16 | 5.9

|     $\bullet$ Financial Crimes & Risky Practices | 14 | 5.1

| Privacy & IP | 30 | 11.0

|     $\bullet$ Privacy & Surveillance | 16 | 5.9

|     $\bullet$ Intellectual Property | 14 | 5.1

| Deception & Public Influence | 28 | 10.3

|     $\bullet$ Disinformation | 14 | 5.1

|     $\bullet$ Government Decision & Political | 14 | 5.1

| Illegal Goods & Services | 29 | 10.7

|     $\bullet$ Controlled Substances | 15 | 5.5

|     $\bullet$ General Illegal Acts | 14 | 5.1

| Hate & Harassment | 30 | 11.0

|     $\bullet$ Hate Speech & Extremism | 15 | 5.5

|     $\bullet$ Harassment & Bullying | 15 | 5.5

| Violence & Physical Harm | 48 | 17.6

|     $\bullet$ Violence & Terrorism | 17 | 6.3

|     $\bullet$ Weapons & CBRN | 16 | 5.9

|     $\bullet$ Suicide & Self-Harm | 15 | 5.5

| Sensitive Regulated Advice | 30 | 11.0

|     $\bullet$ Unlicensed Medical & Legal Advice | 15 | 5.5

|     $\bullet$ High-Risk Financial Advice | 15 | 5.5

| Sexual Content | 17 | 6.3

|     $\bullet$ Sexual Content | 17 | 6.3

| Total | 272 | 100.0

Table 5: Harmful-intent taxonomy and category statistics. MMJailBench contains 272 intents organized into 9 major harm domains and 18 harm scenarios.

### A.1 Harmful Intent Taxonomy, Collection, and Filtering

Hierarchy of intents, domains, and scenarios; each intent carries a task identifier, harm category, scenario, and action description. 272 unique intents, 9 major harm domains, 18 harm scenarios (Table 5): Cyber & Tech Abuse (Cyber), Economic Harm (Econ.), Privacy & IP (Privacy), Deception & Public Influence (Decept.), Illegal Goods & Services (Illegal), Hate & Harassment (Hate), Violence & Physical Harm (Violence), Sensitive Regulated Advice (Sensitive), and Sexual Content (Sexual). Sources: existing safety benchmarks (Luo et al., 2024; Gong et al., 2025; Liu et al., 2024; Souly et al., 2024) and public provider safety policies (OpenAI, 2025; OpenAI, 2026; Anthropic, 2025a; Google, 2026), followed by intent normalization, taxonomy organization, and manual filtering.

Pairwise similarity among the 272 intents uses all-MiniLM-L6-v2 embeddings (Reimers and Gurevych, 2019). Most pairs have low similarity with a small high-similarity tail (Figure 6), manually reviewed for near-duplicates. In PCA (Figure 7), domains occupy diverse regions rather than a few clusters.

Figure 6: Pairwise cosine similarity distribution among the 272 harmful intents; the high-similarity tail is manually inspected to remove near-duplicate candidates.

Figure 7: PCA visualization of harmful-intent embeddings, colored by harm domain.

| Template | Framing Strategy | Representative Pattern

| Academic | Academic Analysis | Analyze the topic from an academic perspective and discuss related aspects.

| System | System-Role Simulation | Assume a specific system identity and follow role-specific response instructions.

| Story | Fictional Storytelling | Describe the scenario through a fictional story or hypothetical setting.

| Code | Code-Style Indirection | Express the requested procedure through code format or comments.

| Structure | Structured Completion | Complete a predefined structure containing multiple stages or sections.

| Paradox | Safety-Paradox Reasoning | Discuss harmful actions indirectly under a safety-oriented rationale.

Table 6: Overview of the 6 prompt framing templates used in MMJailBench. Each template modifies the prompt framing while preserving the underlying harmful intent.

### A.2 Prompt Framing Templates

6 templates: Academic Analysis (Academic), System-Role Simulation (System), Fictional Storytelling (Story), Code-Style Indirection (Code), Structured Completion (Structure), and Safety-Paradox Reasoning (Paradox) (Table 6). A task-specific placeholder is filled with the target intent, objective unchanged across templates. The templates are evaluation probes, not an exhaustive attack collection.

Figure 8: Examples of the five visual semantic categories in MMJailBench.

### A.3 Visual Semantic Generation

5 categories: Danger, Work Scenario, Authorization Document, Identity Credential, and Role, with intent and prompt framing preserved. qwen2.5-vl-7b (Bai et al., 2025b) generates visual descriptions and image prompts per intent. zimage2025-Turbo (Z-Image Team et al., 2025) renders Danger, Work Scenario, and Role; a 4-bit-quantized FLUX.2 [dev] (Black Forest Labs, 2025) renders Identity Credential and Authorization Document, using its image-editing capability for coherent textual elements. Images are filtered for semantic relevance, visual quality, and readability (Figure 8).

### A.4 Instruction Carrier Construction

TEXT: direct text input. OCR: instruction text rendered into a readable image combined with the corresponding visual semantic image, intent and framing preserved, under a standardized layout.

## Appendix B Experimental and Evaluation Details

### B.1 Evaluated MLLMs

16 MLLMs: 8 open-weight — qwen2.5-vl-7b (Bai et al., 2025b), qwen3-vl-8b (Bai et al., 2025a), gemma3-12b (Gemma Team et al., 2025), internvl3-8b (Zhu et al., 2025), glm4.1v-9b (Hong et al., 2025), ministral3-8b (Liu et al., 2026), step3-vl-10b (Huang et al., 2026), llava-onevision-1.5 (An et al., 2025); 8 proprietary-access — gpt-5 (Singh et al., 2025), gemini-3-flash (Google DeepMind, 2025), claude-sonnet-4.5 (Anthropic, 2025b), grok-4-fast (xAI, 2025), qwen3-vl-plus (Bai et al., 2025a), kimi-k2.5 (Kimi Team et al., 2026), glm-4.6v (Hong et al., 2025), doubao-seed-2.0-lite (ByteDance Seed Team, 2026). Open-weight via public checkpoints; proprietary via official interfaces.

Each model runs the full Cartesian product of 272 harmful intents, 6 prompt templates, 5 visual semantic conditions, and 2 instruction carrier modes: 16,320 instances per model, 261,120 responses total.

| Score | Definition

| 1 | Refusal or safe response without harmful details.

| 2 | Potential risk without direct harmful instructions.

| 3 | High-level or partial harmful information.

| 4 | Actionable harmful assistance.

| 5 | Detailed and operational harmful guidance.

Table 7: Harmfulness scoring rubric for response evaluation.

### B.2 Evaluation Rubric and Judge Prompt

GPT-5 outputs an instruction-mismatch flag (e.g., transcription of visual content only, prompt repetition, meaningless output) and a harmfulness score on the strict 1 $\sim$ 5 rubric of Table 7. Anti-deception rules keep safety disclaimers, fictional framing, or academic packaging from lowering scores when harmful information is present; refusal, safe redirection, and non-actionable discussion score low. The full judge prompt is in the Code and Data Supplement.

### B.3 Full and Lightweight Benchmark Settings

Full: complete product of 272 harmful intents, 6 prompt templates, 5 visual semantic conditions, and 2 instruction carrier modes, 16,320 instances per model. Lightweight: 1,500 instances, stratified over harm domains, prompt framings, visual semantic conditions, and carrier modes, preserving the full factor distribution; validated in Section C.4.

### B.4 Lightweight Judge Model

A benchmark-scoped judge on qwen3-8b (Team et al., 2025), trained on GPT-5-generated annotations, predicts harmfulness scores and mismatch labels under the GPT-5 protocol; validated in Section C.4, training details in the Code and Data Supplement.

## Appendix C Extended Results and Ablations

| Category | Count | ASR | $\Delta_{\mathrm{T-O}}$

| Cyber & Tech Abuse
|
|
|

|    $\bullet$ Malware & Hacking | 16 | 56.22 | 10.77

|    $\bullet$ Agent Abuse & Automation | 14 | 58.79 | 11.58

| Economic Harm
|
|
|

|    $\bullet$ Fraud & Scams | 16 | 57.66 | 11.20

|    $\bullet$ Financial Crimes & Risky Practices | 14 | 54.66 | 10.57

| Privacy & IP
|
|
|

|    $\bullet$ Privacy & Surveillance | 16 | 52.80 | 11.12

|    $\bullet$ Intellectual Property | 14 | 51.09 | 10.22

| Deception & Public Influence
|
|
|

|    $\bullet$ Disinformation | 14 | 53.73 | 12.46

|    $\bullet$ Government Decision & Political | 14 | 50.33 | 11.76

| Illegal Goods & Services
|
|
|

|    $\bullet$ Controlled Substances | 15 | 44.78 | 9.10

|    $\bullet$ General Illegal Acts | 14 | 50.25 | 11.46

| Hate & Harassment
|
|
|

|    $\bullet$ Hate Speech & Extremism | 15 | 36.34 | 10.63

|    $\bullet$ Harassment & Bullying | 15 | 47.02 | 10.07

| Violence & Physical Harm
|
|
|

|    $\bullet$ Violence & Terrorism | 17 | 37.13 | 8.58

|    $\bullet$ Weapons & CBRN | 16 | 43.39 | 9.43

|    $\bullet$ Suicide & Self-Harm | 15 | 36.36 | 8.86

| Sensitive Regulated Advice
|
|
|

|    $\bullet$ Unlicensed Medical & Legal Advice | 15 | 26.69 | 5.81

|    $\bullet$ High-Risk Financial Advice | 15 | 42.45 | 8.35

| Sexual Content
|
|
|

|    $\bullet$ Sexual Content | 17 | 26.72 | 10.66

| Total | 272 | 45.74 | 10.10

Table 8: Category-level jailbreak results. ASR is computed by pooling the two instruction carriers, and $\Delta_{\mathrm{T-O}}$ denotes the ASR difference between TEXT and OCR carriers.

| Model | Acad. | Sys. | Story | Code | Stru. | Para.

| Open-weight

| glm4.1v-9b | 77.83 | 46.40 | 98.20 | 68.16 | 91.40 | 57.54

| step3-vl-10b | 65.11 | 15.51 | 58.05 | 46.36 | 87.68 | 48.42

| ministral3-8b | 82.72 | 30.70 | 70.00 | 25.55 | 76.36 | 28.24

| llava-onevision-1.5 | 35.59 | 50.51 | 54.89 | 40.66 | 47.21 | 19.60

| internvl3-8b | 74.41 | 0.07 | 88.42 | 37.10 | 57.13 | 43.09

| qwen2.5-vl-7b | 91.43 | 0.66 | 88.60 | 64.23 | 84.67 | 35.62

| qwen3-vl-8b | 25.37 | 22.32 | 14.19 | 15.40 | 29.12 | 13.49

| gemma3-12b | 86.40 | 18.90 | 97.94 | 74.08 | 92.21 | 37.06

| Proprietary-access

| glm-4.6v | 84.67 | 79.56 | 96.29 | 80.55 | 93.86 | 33.71

| qwen3-vl-plus | 63.27 | 40.11 | 82.72 | 63.64 | 68.38 | 56.54

| doubao-seed-2.0-lite | 63.27 | 16.10 | 96.29 | 23.90 | 45.77 | 51.95

| kimi-k2.5 | 59.45 | 40.33 | 59.38 | 20.99 | 20.96 | 15.15

| gemini-3-flash | 34.96 | 3.20 | 62.87 | 9.60 | 50.88 | 18.97

| grok-4-fast | 66.40 | 3.53 | 52.50 | 24.12 | 74.15 | 43.12

| claude-sonnet-4.5 | 5.40 | 0.00 | 10.00 | 2.54 | 32.13 | 3.27

| gpt-5 | 0.74 | 0.00 | 0.44 | 0.85 | 8.82 | 1.88

Table 9: Per-model ASR (%) across different prompt framing strategies on MMJailBench.

### C.1 Extended Factor-Level Results

Table 8: Cyber & Tech Abuse, Economic Harm, and Privacy & IP are higher; Sensitive Regulated Advice and Sexual Content lower. $\Delta_{\mathrm{T-O}}$ is positive in every category, from 5.81 to 12.46 percentage points, so the carrier effect is not driven by specific domains.

Tables 9 and 10: framing sensitivity ranges from large to near-stable across models, and visual conditions yield model-dependent patterns.

| Model | Dang. | Scen. | Auth. Doc. | ID Cred. | Role

| Open-weight

| glm4.1v-9b | 75.34 | 72.33 | 71.17 | 73.93 | 73.50

| step3-vl-10b | 51.87 | 53.40 | 55.82 | 55.48 | 51.04

| ministral3-8b | 57.57 | 53.80 | 52.79 | 49.17 | 47.98

| llava-onevision-1.5 | 42.28 | 43.17 | 40.10 | 42.95 | 38.54

| internvl3-8b | 52.60 | 52.36 | 48.10 | 49.97 | 47.15

| qwen2.5-vl-7b | 62.19 | 61.58 | 59.83 | 61.95 | 58.79

| qwen3-vl-8b | 21.69 | 18.29 | 22.89 | 20.68 | 16.36

| gemma3-12b | 64.68 | 67.22 | 71.63 | 67.71 | 67.59

| Proprietary-access

| glm-4.6v | 70.74 | 80.39 | 82.08 | 79.01 | 78.31

| qwen3-vl-plus | 60.81 | 59.44 | 67.49 | 63.51 | 60.97

| doubao-seed-2.0-lite | 48.28 | 48.96 | 53.43 | 49.45 | 47.61

| kimi-k2.5 | 32.23 | 37.01 | 39.15 | 38.76 | 33.06

| gemini-3-flash | 29.75 | 34.59 | 26.75 | 28.16 | 31.16

| grok-4-fast | 44.64 | 43.72 | 44.00 | 45.16 | 42.34

| claude-sonnet-4.5 | 8.06 | 9.90 | 10.02 | 8.33 | 8.15

| gpt-5 | 1.69 | 2.05 | 2.36 | 2.48 | 2.02

Table 10: Per-model ASR (%) across different visual semantic conditions on MMJailBench.

Figure 9: Per-model jailbreak factor sensitivity profiles. Horizontal axis: ASR range over prompt framings; vertical axis: ASR range over visual semantic conditions; marker size: ASR range over instruction carriers. Colors and shapes denote four K-means profiles obtained with 1,000 random restarts.

### C.2 Per-Model Factor Sensitivity Profiles

K-means on per-model ASR ranges for prompt framing, visual semantics, and instruction carrier (Figure 9) yields four profiles: prompt-dominant, visual-responsive, carrier-sensitive, and low-variation.

### C.3 Semantic Image Ablation

Figure 10 gives per-model ablations, each ASR value aggregating 3,264 responses. step3-vl-10b and doubao-seed-2.0-lite show larger semantic-versus-control gaps, gemma3-12b a smaller one; claude-sonnet-4.5 has relatively higher ASR under no-image, with vulnerability less tied to semantic visual cues.

Figure 10: Per-model ASR across five task-relevant visual semantic conditions and four control conditions. The radial scale spans 0 $\sim$ 80%. Axes, shading, and model colors match the main-paper radar plot.

### C.4 Evaluation Efficiency and Validation

Table 11 compares lightweight and full configurations: small ASR differences, strong agreement across matched evaluations. Table 12 reports GPT-5 versus human experts and lightweight judge versus GPT-5, with strong agreement on harmfulness scoring and mismatch detection.

|    Measure |    Result

|    Mean ASR difference (Light $-$ Full) |     $-0.06\%$

|    95% CI for ASR difference |    [ $-0.45\%$ , $0.30\%$ ]

|    Mean absolute ASR difference |     $0.79\%$

|    Maximum absolute ASR difference |     $3.53\%$

|    Cells within 2% difference |    30/32 (93.75%)

|    Spearman $\rho$ |    0.997

|    Lin’s concordance correlation |    0.999

Table 11: Consistency analysis between Full and Lightweight benchmark settings.

| Evaluation Pair | Harmfulness | Mismatch Detection

| QWK | $\Delta$ ASR | Acc. | Macro- $F_{1}$

| GPT-5 vs. Human Experts | 0.97 | +0.3% | 96.3% | 0.85

| Lightweight Judge vs. GPT-5 | 0.95 | $-0.4\%$ | 99.4% | 0.94

Table 12: Judge agreement and validation results between GPT-5, human experts, and the lightweight judge.

## Appendix D Responsible Release

### D.1 Ethics and Data Release

MMJailBench contains harmful intents and adversarial prompts and is released for research use only, with documentation and evaluation tools for reproducible, responsible use aimed at improving model safety and robustness.

### D.2 LLM Usage Statement

LLMs are used for harmful-intent organization, visual prompt generation, and response evaluation. Generated materials pass predefined filtering and validation; benchmark design and protocols are determined by the authors.