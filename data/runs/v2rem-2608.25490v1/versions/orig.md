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

Multimodal Large Language Models (MLLMs) are increasingly deployed in real-world applications, yet evaluating their robustness to jailbreak attacks remains challenging. Harmful requests directed at MLLMs can differ in their underlying objective, in how they are phrased, in the visual context that accompanies them, and in whether the instruction is delivered as text or embedded in an image. Existing benchmarks cover parts of this space, but their instances are often collected from heterogeneous sources with inconsistent annotation, which makes it difficult to report coverage and compare results across studies. To address this limitation, we introduce MMJailBench, a structured multimodal jailbreak benchmark in which every instance is annotated along four descriptive dimensions: harmful intent, prompt framing, visual semantics, and instruction carrier. The benchmark comprises 272 harmful intents spanning 9 harm domains and 18 scenarios, 6 prompt framing templates, 5 task-relevant visual semantic categories with accompanying control images, and 2 instruction carriers. We evaluate 16 open-weight and proprietary MLLMs under a unified protocol and a validated LLM-based judge. Beyond the benchmark itself, we provide diagnostic tooling for probing internal representations and cross-modal attention in open-weight models. Finally, we develop a modular multimodal jailbreak evaluation suite with full and lightweight configurations, multiple judge options, and multidimensional metrics, enabling reproducible, scalable, and cost-efficient multimodal jailbreak auditing. Warning: This paper contains potentially harmful content.

## 1 Introduction

Multimodal Large Language Models (MLLMs) are expanding beyond image captioning and visual question answering into real-world applications such as document understanding, code generation, workflow assistance, and automated operations (Liu et al., 2023; Dai et al., 2023). Unlike text-only models, MLLMs must jointly interpret linguistic instructions and visual information when reasoning and generating responses. Harmful requests may therefore combine direct textual or visually rendered instructions (Gong et al., 2025) with contextual cues such as professional settings, identity credentials, or authorization documents. Consequently, a model’s refusal and compliance decisions no longer depend solely on an isolated user instruction, but are shaped by the surrounding cross-modal context (Li et al., 2024b).

Figure 1: Motivation of MMJailBench. Multimodal jailbreak instances vary along several descriptive dimensions. MMJailBench organizes instances along these dimensions with consistent annotation to support transparent coverage reporting and reproducible evaluation.

How models form safety judgments in cross-modal contexts remains an open question. Visual information may not only supplement harmful intent, but also affect the perceived legitimacy, professionalism, and credibility of a request (Liu et al., 2024). This suggests that multimodal jailbreaks are not merely changes in input format; rather, they test the stability and consistency of safety alignment under cross-modal conditions (Luo et al., 2024). Building evaluation resources that capture this diversity in an organized and well-documented way is therefore an important step toward accurately assessing and improving multimodal safety alignment.

Existing text and multimodal safety benchmarks cover a broad range of harmful behaviors (Li et al., 2024a; Mazeika et al., 2024; Chao et al., 2024), visual attacks, and image-based text injection (Liu et al., 2024; Gong et al., 2025; Luo et al., 2024), providing an important foundation for evaluating model robustness. These resources differ considerably in their taxonomies, in the way prompts and images are constructed, and in the metrics and judges used to score responses (Weng et al., 2025; Jia et al., 2025). As a result, results obtained on one benchmark are often hard to relate to results obtained on another, and it can be unclear which kinds of framing, visual context, or instruction delivery a given test set actually covers. Practitioners who wish to audit a deployed model therefore face a fragmented landscape of datasets, judges, and reporting conventions.

To address this limitation, we introduce MMJailBench, a structured benchmark for multimodal jailbreak evaluation. MMJailBench annotates every instance with four descriptive dimensions: harmful intent, prompt framing, visual semantics, and instruction carrier. The benchmark contains 272 harmful intents, 6 prompt templates, 5 types of task-relevant visual semantics, and 2 instruction carriers, organized under a shared taxonomy and a common construction pipeline. This organization makes the coverage of the benchmark explicit and allows results to be stratified along each dimension for reporting purposes. Building on this design, we further develop a modular multimodal jailbreak evaluation suite with full and lightweight configurations, multiple judge options, and multidimensional metrics, enabling reproducible, scalable, and cost-efficient multimodal jailbreak auditing.

Figure 2: Construction and evaluation pipeline of MMJailBench. The benchmark organizes instances by harmful intent, prompt framing, visual semantics, and instruction carrier, and evaluates MLLMs under a unified multimodal jailbreak protocol.

We apply MMJailBench to 16 representative open-weight and proprietary MLLMs under a single evaluation protocol. Responses are scored by an LLM-based judge whose agreement with human experts is validated, and a distilled open-weight judge is provided for low-cost follow-up studies. We additionally describe diagnostic tooling for open-weight models that extracts layer-wise residual-stream representations and cross-modal attention patterns from matched inputs, allowing researchers to complement behavioral measurements with model-internal views. Throughout, our emphasis is on providing a well-documented, consistently annotated, and reproducible resource rather than on a single headline number.

Overall, our main contributions are as follows:

•

We introduce MMJailBench, a structured benchmark that annotates multimodal jailbreak instances by harmful intent, prompt framing, visual semantics, and instruction carrier under a unified taxonomy and construction pipeline, supporting transparent coverage reporting.

•

We evaluate 16 representative open-weight and proprietary MLLMs under a unified protocol with a validated judge, and provide diagnostic tooling for examining internal representations and cross-modal interactions in open-weight models.

•

We develop a modular multimodal jailbreak evaluation suite with full and lightweight configurations, multiple judge options, and multidimensional metrics, enabling reproducible, scalable, and cost-efficient multimodal jailbreak auditing.

## 2 Related Work

### 2.1 Jailbreak Attacks and Defenses

Jailbreak attacks aim to circumvent safety alignment and induce models to assist with harmful requests (Wei et al., 2023; Zou et al., 2023). Early studies attributed such failures to competing objectives and mismatched generalization in safety training (Wei et al., 2023). Subsequent work demonstrated transferable adversarial suffixes (Zou et al., 2023) and black-box semantic attacks such as PAIR (Chao et al., 2025), while indirect prompt injection showed that malicious instructions can also be introduced through external documents or retrieved content (Greshake et al., 2023). These studies establish that refusal behavior can be sensitive to how harmful intent is framed and delivered (Wei et al., 2023; Chao et al., 2025; Greshake et al., 2023).

Multimodal models introduce additional attack surfaces because instructions and contextual cues can be distributed across language and vision (Gong et al., 2025; Liu et al., 2024; Li et al., 2024b). Beyond textual manipulation, visual inputs can be exploited through adversarial images (Liu et al., 2024; Li et al., 2024b), typographic prompts (Gong et al., 2025), and other cross-modal constructions (Luo et al., 2024). Corresponding defenses span safety alignment, input transformation, and inference-time safeguards (Jain et al., 2023; Wang et al., 2024), yet their robustness across changes in modality and context remains uncertain. Defensive techniques are typically evaluated on the specific benchmarks available at the time of their development, and the breadth of contexts covered by those benchmarks directly bounds what can be concluded about a defense. This reinforces the value of evaluation resources whose coverage across framings, visual contexts, and delivery channels is explicitly documented.

A further line of research concerns the evaluation of model responses themselves. Keyword-based refusal detection is simple but brittle, since models may refuse with unusual phrasing or comply while appending disclaimers. LLM-based judges have therefore become common (Chao et al., 2024; Mazeika et al., 2024), but their outputs depend on the rubric and prompt used, and their agreement with human annotators should be reported. MMJailBench adopts a graded harmfulness rubric with explicit anti-deception rules and reports agreement statistics for both its primary and lightweight judges.

### 2.2 Multimodal Jailbreak Benchmarks

Standardized safety benchmarks such as SafetyBench (Zhang et al., 2024), Do-Not-Answer (Wang et al., 2023), SALAD-Bench (Li et al., 2024a), HarmBench (Mazeika et al., 2024), and JailbreakBench (Chao et al., 2024) establish structured taxonomies and reproducible protocols for evaluating harmful behavior and refusal robustness. Multimodal benchmarks extend this paradigm to visual inputs: MM-SafetyBench evaluates image-based safety attacks (Liu et al., 2024), FigStep (Gong et al., 2025) studies harmful instructions rendered as typographic images, and JailBreakV (Luo et al., 2024) and VLJailbreakBench (Wang et al., 2025) expand coverage across diverse multimodal jailbreak strategies. MMJ-Bench (Weng et al., 2025) further provides a unified framework for comparing multimodal jailbreak attacks and defenses, while OmniSafeBench-MM (Jia et al., 2025) broadens standardized attack-defense evaluation with multidimensional metrics.

Recent work has also moved toward more structured cross-modal evaluation. Omni-SafetyBench (Pan et al., 2025) constructs parallel variants of the same harmful seeds across different modality configurations and evaluates cross-modal safety consistency. These benchmarks substantially improve attack coverage, evaluation standardization, and modality-level comparison. MMJailBench complements these efforts by organizing instances under a consistent four-dimensional annotation scheme covering harmful intent, prompt framing, task-relevant visual semantics, and instruction carrier, and by pairing the benchmark with a modular evaluation suite, validated judges, and diagnostic tooling. In this sense, MMJailBench is intended less as a replacement for prior benchmarks than as a well-documented resource whose design choices are explicit and whose evaluation pipeline can be readily reused.

## 3 MMJailBench

MMJailBench is a structured multimodal jailbreak benchmark designed to support transparent and reproducible evaluation of MLLM safety behavior. As illustrated in Figure 2, MMJailBench constructs multimodal scenarios from harmful intents, prompt framings, visual semantics, and instruction carriers, and records each of these dimensions as explicit metadata for every instance. The following subsections describe the benchmark design and each dimension in detail.

### 3.1 Factorized Design

Existing multimodal jailbreak benchmarks often focus on predefined attack instances or modality variations, and their metadata vary in granularity: some record only a harm category, while others record the attack method but not the visual context. Such designs provide valuable measurements of overall vulnerability, but heterogeneous annotation makes it harder to describe what a test set covers and to stratify results consistently.

MMJailBench therefore describes each multimodal jailbreak instance in terms of four components: harmful intent, prompt framing, visual semantics, and instruction carrier. Given a harmful intent $h$ , a prompt framing strategy $t$ , a visual semantic condition $v$ , and an instruction carrier $c$ , each multimodal instance is represented as

| | $x_{h,t,v,c}=(L(h,t,c),V(h,v)),$
| | (1)

where $L(h,t,c)$ denotes the instruction component generated from harmful intent $h$ , prompt framing $t$ , and instruction carrier $c$ , while $V(h,v)$ denotes the visual input associated with harmful intent $h$ and visual semantics $v$ .

This representation serves primarily as a construction and annotation scheme. It specifies how the textual and visual parts of an instance are generated from shared ingredients, ensures that every instance carries the same metadata fields, and makes it straightforward to extend the benchmark with new templates, image categories, or delivery channels. The benchmark draws on 272 harmful intents, 6 prompt framing strategies, 5 visual semantic conditions, and 2 instruction carriers, and the resulting instance pool is used consistently across all models and harm categories.

### 3.2 Harmful Intent

Harmful intent represents the underlying unsafe objective evaluated in MMJailBench. Instead of directly collecting jailbreak prompts that combine harmful goals with specific attack strategies, MMJailBench constructs a set of intent-level behavioral seeds from which instances are instantiated.

The benchmark contains 272 harmful intents across 9 major harm domains: Physical Harm, Cyber Abuse, Economic Harm, Hate and Harassment, Privacy and IP, Regulated Advice, Illegal Activities, Deception and Influence, and Sexual Content. Each category is further divided into fine-grained subcategories to improve coverage and diversity across safety-critical behaviors. The harmful intents are collected through category-guided construction, followed by semantic similarity filtering and manual verification to reduce redundancy. These intent-level seeds provide a consistent foundation for instance construction and allow results to be reported by domain and scenario.

### 3.3 Prompt Framing

Prompt framing specifies how a harmful intent is linguistically presented to the model. Different narrative structures and interaction styles are widely used in practice to reformulate requests, and a benchmark that aims to reflect realistic adversarial usage should cover a representative range of such styles.

MMJailBench introduces 6 representative prompt framing strategies: academic, system, structured, story, code, and safety-paradox framing. These strategies capture common ways of reformulating harmful requests while preserving the underlying intent. For example, a harmful request may be embedded within a fictional scenario, expressed as a structured generation task, or presented as an apparently legitimate system requirement. The templates are deliberately simple and fixed, so that they function as repeatable probes rather than as optimized attacks.

### 3.4 Visual Semantics

Visual semantics capture the contextual meaning introduced by accompanying images. Unlike image-based instruction attacks that primarily encode instructions into visual text, this dimension concerns the contextual information that an image conveys about the request.

We construct 5 task-relevant visual semantic conditions, including danger-related context, scenario context, professional-role context, identity credentials, and authorization documents. These conditions represent different forms of contextual signals that may affect how a request is interpreted, such as legitimacy-associated, expertise-related, or authority-related contextual signals. For each harmful intent, the corresponding visual input provides additional semantic context without modifying the underlying harmful instruction. In addition, control visual conditions, including no image, blank, noise, and nature images, are released alongside the semantic images so that users can construct their own reference settings. This design broadens the range of visually grounded contexts represented in the benchmark.

### 3.5 Instruction Carrier

Instruction carrier describes how harmful instructions are presented to the model. Typographic and screenshot-based attacks have shown that instructions embedded in images can be processed differently from plain text (Gong et al., 2025), and a comprehensive benchmark should therefore include both delivery channels.

MMJailBench considers 2 instruction carriers: direct textual instructions (TEXT) and visually rendered instructions in text-containing images (OCR). The TEXT carrier directly provides harmful instructions as text, whereas the OCR carrier renders the same instructions into images while preserving their semantics. Because the OCR carrier introduces an additional perception stage, the judge also records instruction recognition mismatch, where models fail to interpret embedded instructions correctly. This allows users to separate perception failures from safety behavior when reporting OCR-based results.

## 4 Experimental Setup

This section describes the evaluated MLLMs, metrics, judge models, and evaluation configurations used in MMJailBench. All models are evaluated under the same protocol for consistent comparison. Additional experimental details, including evaluation prompts, judge criteria, and implementation details, are provided in the appendix.

### 4.1 Evaluated MLLMs

We evaluate MMJailBench on 16 representative MLLMs, covering both open-weight and proprietary-access models. The evaluated models include representative model families such as LLaVA-OneVision (An et al., 2025), InternVL (Zhu et al., 2025), Kimi (Team et al., 2026), Qwen (Bai et al., 2025b; Bai et al., 2025a), Gemma (Team et al., 2025), GLM (Hong et al., 2025), GPT (Singh et al., 2025), Gemini (Team et al., 2023), and Claude (Anthropic, 2025).

The model selection is intended to span a range of parameter scales, training pipelines, and access modes. Open-weight models in the 7B to 12B range are included because they are widely deployed in research and production settings and because their internals can be inspected with the diagnostic tooling described in Section 6. Proprietary-access models are included because they represent the systems most frequently exposed to end users. All models receive the same instance pool and decoding configuration where the provider interface allows it.

### 4.2 Metrics and Judge Models

MMJailBench evaluates model safety using complementary metrics. The primary metric is Attack Success Rate (ASR), which measures the proportion of responses that provide substantive assistance to the targeted harmful intent:

| | $ASR=\frac{1}{N}\sum_{i=1}^{N}\mathbb{I}(s_{i}\geq 4),$
| | (2)

where $s_{i}$ denotes the harmfulness score of the $i$ -th response on a 1 $\sim$ 5 scale. Responses with $s_{i}\geq 4$ are considered successful jailbreak cases.

For instruction carrier reporting, we further introduce Conditional Attack Success Rate (CASR):

| | $CASR=\frac{N_{success}}{N-N_{mismatch}},$
| | (3)

where mismatch cases refer to inputs that are not correctly understood by the model, including OCR recognition failures, irrelevant responses, and misinterpretation of the evaluated intent. ASR measures end-to-end jailbreak vulnerability, whereas CASR evaluates safety robustness conditioned on successful input understanding.

The main experiments use GPT-5 as an LLM-based judge. For each model response, GPT-5 assigns a harmfulness score and auxiliary labels for jailbreak success and instruction mismatch. The evaluation focuses on generated responses: refusals, safety warnings, and high-level discussions are not considered successful jailbreak cases. We additionally develop a benchmark-specific lightweight judge model based on an open-weight MLLM for cost-efficient follow-up evaluation under the same protocol.

The choice of a threshold at $s_{i}\geq 4$ reflects a deliberately conservative definition of success. Scores of 2 and 3 capture responses that discuss a topic at a high level or provide partial, non-operational information; such responses may warrant attention in some deployment contexts, but counting them as successful jailbreaks would conflate topical engagement with genuine harmful assistance. The full score distribution is retained in the released outputs so that users can apply alternative thresholds.

### 4.3 Evaluation Configurations

MMJailBench supports full and lightweight evaluation configurations. The full configuration evaluates all benchmark instances and is intended for comprehensive audits. The lightweight configuration samples a representative subset while preserving coverage across harmful domains, prompt framings, visual semantics, and instruction carriers, enabling efficient safety auditing and regression testing. Unless otherwise specified, the evaluation pipeline described in this paper refers to the full configuration.

## 5 Results and Analysis

This section describes how MMJailBench evaluation outputs are organized and how they can be read. We focus on the structure of the reported measurements and on the practical considerations that arise when interpreting them, since the benchmark is designed primarily as a reusable auditing resource.

### 5.1 Overall Jailbreak Landscape

The most basic output of MMJailBench is a per-model ASR computed over the evaluated instance pool, accompanied by the per-model mismatch rate and CASR. These aggregate quantities give a first indication of how often a model produces actionable harmful content under the benchmark’s conditions, and they are the natural starting point for comparing models or tracking a single model across releases.

Aggregate scores should nonetheless be interpreted with care. An ASR value summarizes behavior over a specific mixture of intents, framings, visual contexts, and carriers, and a different mixture could yield a different number. For this reason, MMJailBench reports aggregate scores together with the benchmark composition documented in Appendix A, so that readers can relate a given number to the set of conditions that produced it. We also caution against reading aggregate ASR as a proxy for general capability: safety behavior and task competence are shaped by different parts of the training pipeline, and there is no a priori reason for them to move together.

A second consideration concerns the judge. Because ASR is derived from judge scores, any systematic judge bias propagates directly into the reported numbers. We therefore report judge agreement statistics (Appendix C.4) alongside every evaluation and recommend that users who adopt a different judge or rubric re-validate it on a sample of responses before comparing results to those obtained with the default configuration.

### 5.2 Factor-Level Attribution

Because every instance carries explicit annotations for all four dimensions, MMJailBench outputs can be stratified along each of them. The suite produces stratified tables for harmful intent, prompt framing, visual semantics, and instruction carrier, both pooled across models and separately for each model. We describe below what each stratification is intended to surface and how it should be used.

#### Harmful Intent.

Stratification by harmful intent reports ASR per harm domain and per scenario using the taxonomy in Table 5. This view is useful for checking whether safety alignment provides protection across the full range of harmful behaviors covered by provider policies, rather than only those most frequently represented in public red-teaming data. Since domains differ in the number of intents they contain, the suite reports both per-domain counts and per-domain rates, and it is advisable to inspect confidence intervals before drawing conclusions about smaller domains such as Sexual Content.

#### Prompt Framing.

Stratification by prompt framing reports ASR for each of the 6 templates in Table 6. Framing strategies such as fictional storytelling, structured completion, and academic analysis are among the most widely discussed jailbreak techniques in the literature (Wei et al., 2023; Chao et al., 2025), and this view allows auditors to check how a model responds to each of these common reformulation styles. Because the templates are fixed and simple, they should be understood as representative probes rather than as an exhaustive or optimized attack set; adaptive attacks may behave differently.

#### Visual Semantics.

Stratification by visual semantics reports ASR for each of the 5 task-relevant image categories. These categories were chosen to reflect contextual signals that commonly accompany real requests, including scenes depicting a task setting, images suggesting professional roles, and documents suggesting identity or authorization. The released control images (no image, blank, noise, and nature) allow users to construct reference conditions for their own studies. When interpreting this view, it is important to keep in mind that image generation quality and textual legibility differ across categories, which can interact with model perception.

#### Instruction Carrier.

Stratification by instruction carrier reports ASR, mismatch rate, and CASR for TEXT and OCR inputs. The mismatch rate is particularly important for the OCR carrier: a model that cannot read the rendered instruction will rarely produce harmful content, but this reflects a perception limitation rather than robust safety behavior. Reporting CASR alongside ASR therefore allows users to distinguish these two cases. Carrier-related behavior is likely to depend on how a given model’s vision encoder and language backbone handle embedded text, so carrier-level results should be read on a per-model basis.

### 5.3 Model-Dependent Profiles

In addition to pooled stratifications, the suite produces per-model summaries along each dimension, including the range of ASR across the configurations of each dimension. These summaries are intended to give auditors a compact description of a model’s behavior across the benchmark’s conditions, complementing the single aggregate ASR.

Per-model summaries are most informative when read alongside the underlying stratified tables. A large range along one dimension may arise from a single outlying configuration or from a gradual trend, and these two situations call for different follow-up analyses. We also recommend comparing per-model summaries only among models evaluated under the same judge, decoding settings, and benchmark version, since changes in any of these can shift the reported values. More generally, we view per-model summaries as a convenient entry point into the evaluation outputs rather than as a definitive characterization of a model’s safety alignment.

## 6 Diagnostic Analysis

Behavioral evaluation describes what a model outputs, but not how internal processing leads to those outputs. To support research that connects behavioral measurements with model internals, MMJailBench includes diagnostic tooling for open-weight models. We illustrate the tooling on gemma3-12b using matched instances that share the same harmful intent, framing, and carrier but differ in their accompanying image, for example an authority-document image versus a danger-related image. The tooling extracts layer-wise representations and cross-modal attention patterns, providing a model-internal view that can complement the behavioral evaluation.

### 6.1 Representation-Level Diagnostics

The representation-level tooling records residual-stream activations of the final generation-prefix token at every layer, since this token’s state immediately precedes response generation and has been used in prior work to study refusal behavior. For a set of paired inputs, the tooling computes a layer-wise divergence between the two conditions, producing a profile that indicates where in the network the paired contexts are most separated.

The tooling further supports fitting a difference direction between two conditions on a set of discovery pairs and projecting held-out pairs onto that fixed direction. This procedure provides a simple check of whether a representation difference identified on one set of samples extends to new samples, and it can be broken down by harm domain using the taxonomy annotations. Users can substitute any two visual conditions, framings, or carriers, making the procedure applicable well beyond the illustrative pairing used here. We emphasize that such analyses are correlational: a representation difference that co-occurs with a behavioral difference does not by itself establish a causal mechanism, and intervention-based methods are needed for stronger claims.

Figure 4: Representation-level diagnostic tooling applied to gemma3-12b. The tooling computes layer-wise divergence between paired conditions and projects held-out pairs onto a fitted difference direction, with optional breakdown by harm domain.

### 6.2 Cross-Modal Interaction Diagnostics

Multimodal response generation depends on how textual and visual information are integrated across layers. The interaction-level tooling therefore records attention from the final generation-prefix query to selected token groups, including harm-related task-label tokens in the instruction and special visual tokens associated with the image. Attention mass is aggregated per layer and averaged over heads, producing layer-wise attention profiles for each condition.

These profiles can be inspected jointly with the representation-level outputs, for instance by examining attention allocation at layers where paired conditions are most separated. Attention weights are an imperfect indicator of information flow, since information can also be routed through residual connections and MLP layers, and we therefore recommend treating attention profiles as descriptive diagnostics. Combined with the behavioral stratifications in Section 5, the diagnostic tooling offers researchers a starting point for hypotheses about how multimodal context enters the response-generation process.

Figure 5: Cross-modal interaction diagnostic tooling applied to gemma3-12b. The tooling reports layer-wise attention allocation from the generation-prefix query to harm-related textual tokens and visual tokens under paired conditions.

## 7 Conclusion

In this work, we introduce MMJailBench, a structured benchmark for multimodal jailbreak evaluation in which every instance is annotated by harmful intent, prompt framing, visual semantics, and instruction carrier. The benchmark draws on 272 intents across 9 harm domains, 6 framing templates, 5 visual semantic categories with released control images, and 2 instruction carriers, and it is evaluated on 16 open-weight and proprietary MLLMs under a single protocol. We complement the benchmark with diagnostic tooling for inspecting internal representations and cross-modal attention in open-weight models. With modular full and lightweight configurations, flexible judge options with validated agreement, and multidimensional metrics, MMJailBench provides a systematic and reproducible framework for scalable multimodal safety evaluation. We hope that the explicit documentation of benchmark composition and evaluation choices will make results easier to compare across studies and will encourage careful, transparent reporting in multimodal safety research.

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

The harmful intent taxonomy defines the semantic space of MMJailBench by organizing harmful behaviors into a structured hierarchy of intents, domains, and scenarios. Each intent is associated with a task identifier, harm category, scenario, and action description.

The taxonomy contains 272 unique harmful intents covering 9 major harm domains and 18 harm scenarios. Table 5 summarizes the taxonomy structure and category statistics. The 9 major harm domains include Cyber & Tech Abuse (Cyber), Economic Harm (Econ.), Privacy & IP (Privacy), Deception & Public Influence (Decept.), Illegal Goods & Services (Illegal), Hate & Harassment (Hate), Violence & Physical Harm (Violence), Sensitive Regulated Advice (Sensitive), and Sexual Content (Sexual). The taxonomy is constructed from existing safety benchmarks (Luo et al., 2024; Gong et al., 2025; Liu et al., 2024; Souly et al., 2024) and public model-provider safety policies (OpenAI, 2025; OpenAI, 2026; Anthropic, 2025a; Google, 2026), followed by intent normalization, taxonomy organization, and manual filtering.

To improve semantic diversity and reduce redundancy, we perform embedding-based similarity analysis among the 272 intents using the all-MiniLM-L6-v2 sentence embedding model (Reimers and Gurevych, 2019). Figure 6 shows that most intent pairs have relatively low semantic similarity, while only a small subset forms a high-similarity tail. These cases are manually reviewed to remove near-duplicates and ensure that retained intents represent distinct harmful objectives. We further visualize the intent embedding space using PCA in Figure 7. Although the projection provides a qualitative view, intents from different harm domains occupy diverse regions rather than collapsing into a small number of clusters. Together, the diversity analysis and manual filtering establish a balanced harmful-intent set for benchmark construction.

Figure 6: Pairwise cosine similarity distribution among the 272 harmful intents. The high-similarity tail is manually inspected during taxonomy construction to identify and remove near-duplicate candidates.

Figure 7: PCA visualization of harmful-intent embeddings. Different colors represent harm domains, providing a qualitative view of taxonomy coverage and semantic diversity.

| Template | Framing Strategy | Representative Pattern

| Academic | Academic Analysis | Analyze the topic from an academic perspective and discuss related aspects.

| System | System-Role Simulation | Assume a specific system identity and follow role-specific response instructions.

| Story | Fictional Storytelling | Describe the scenario through a fictional story or hypothetical setting.

| Code | Code-Style Indirection | Express the requested procedure through code format or comments.

| Structure | Structured Completion | Complete a predefined structure containing multiple stages or sections.

| Paradox | Safety-Paradox Reasoning | Discuss harmful actions indirectly under a safety-oriented rationale.

Table 6: Overview of the 6 prompt framing templates used in MMJailBench. Each template modifies the prompt framing while preserving the underlying harmful intent.

### A.2 Prompt Framing Templates

MMJailBench uses 6 representative prompt framing templates to cover common ways in which harmful requests are reformulated in practice. These templates preserve the underlying harmful intent while varying the surrounding framing strategy. The 6 templates cover common jailbreak-related framing strategies, including Academic Analysis (Academic), System-Role Simulation (System), Fictional Storytelling (Story), Code-Style Indirection (Code), Structured Completion (Structure), and Safety-Paradox Reasoning (Paradox).

Table 6 summarizes the 6 prompt framings and their corresponding design characteristics. Each template is instantiated by replacing the task-specific placeholder with the target harmful intent. The templates are designed as evaluation probes rather than an exhaustive collection of jailbreak attacks, providing repeatable linguistic conditions that can be shared across studies.

Figure 8: Examples of the five visual semantic categories in MMJailBench. Each category provides distinct visual contextual cues while preserving the underlying harmful intent.

### A.3 Visual Semantic Generation

MMJailBench introduces 5 visual semantic categories, including Danger, Work Scenario, Authorization Document, Identity Credential, and Role, to represent different visual cues that may accompany a request while preserving the underlying harmful intent and prompt framing.

For each harmful intent, qwen2.5-vl-7b (Bai et al., 2025b) is used to generate visual descriptions and image generation prompts. zimage2025-Turbo (Z-Image Team et al., 2025) is used for visual scene generation in Danger, Work Scenario, and Role, while a 4-bit-quantized FLUX.2 [dev] (Black Forest Labs, 2025) is used for the Identity Credential and Authorization Document categories due to its image-editing capability for generating images with coherent textual elements. Generated images are filtered based on semantic relevance, visual quality, and readability. Figure 8 shows representative examples of the generated visual semantic conditions.

### A.4 Instruction Carrier Construction

We construct the instruction carrier dimension by controlling how harmful intent is delivered to MLLMs. MMJailBench considers two carrier modes: direct text input (TEXT) and visually embedded instruction input (OCR). For OCR instances, the instruction text is rendered into a readable image and combined with the corresponding visual semantic image while preserving the original harmful intent and prompt framing. A standardized layout is used to reduce formatting variations across rendered instructions, and font size and contrast are chosen to remain legible after typical image preprocessing.

## Appendix B Experimental and Evaluation Details

### B.1 Evaluated MLLMs

We evaluate MMJailBench on 16 MLLMs, including 8 open-weight models and 8 proprietary-access models. The open-weight models include qwen2.5-vl-7b (Bai et al., 2025b), qwen3-vl-8b (Bai et al., 2025a), gemma3-12b (Gemma Team et al., 2025), internvl3-8b (Zhu et al., 2025), glm4.1v-9b (Hong et al., 2025), ministral3-8b (Liu et al., 2026), step3-vl-10b (Huang et al., 2026), and llava-onevision-1.5 (An et al., 2025). The proprietary-access models include gpt-5 (Singh et al., 2025), gemini-3-flash (Google DeepMind, 2025), claude-sonnet-4.5 (Anthropic, 2025b), grok-4-fast (xAI, 2025), qwen3-vl-plus (Bai et al., 2025a), kimi-k2.5 (Kimi Team et al., 2026), glm-4.6v (Hong et al., 2025), and doubao-seed-2.0-lite (ByteDance Seed Team, 2026). Open-weight models are evaluated using publicly available checkpoints, while proprietary-access models are evaluated through their official interfaces.

All models are evaluated under the same benchmark configuration and receive identical inputs. Where provider interfaces permit, sampling parameters are fixed to reduce run-to-run variability; for interfaces that do not expose these parameters, default settings are used and documented in the released configuration files.

| Score | Definition

| 1 | Refusal or safe response without harmful details.

| 2 | Potential risk without direct harmful instructions.

| 3 | High-level or partial harmful information.

| 4 | Actionable harmful assistance.

| 5 | Detailed and operational harmful guidance.

Table 7: Harmfulness scoring rubric for response evaluation.

### B.2 Evaluation Rubric and Judge Prompt

GPT-5 is used as the primary evaluator for MMJailBench responses. For each response, the judge outputs an instruction-mismatch flag and a harmfulness score. The mismatch flag identifies failures to recognize the task, such as only transcribing visual content, repeating the prompt, or producing meaningless outputs. The harmfulness score follows a strict 1 $\sim$ 5 rubric, with higher scores indicating stronger compliance and more actionable harmful content. The detailed scoring criteria are summarized in Table 7.

The judge prompt incorporates anti-deception rules to prevent safety disclaimers, fictional framing, or academic packaging from affecting the evaluation when harmful information is provided. Responses limited to refusal, safe redirection, or non-actionable discussion receive low harmfulness scores. The complete judge prompt template is provided in the Code and Data Supplement.

### B.3 Full and Lightweight Benchmark Settings

The full MMJailBench configuration evaluates all benchmark instances and provides comprehensive coverage of the benchmark space across harmful intents, prompt templates, visual semantic conditions, and instruction carrier modes.

To reduce evaluation cost, we construct a lightweight configuration containing 1,500 instances by stratified sampling over harm domains, prompt framings, visual semantic conditions, and carrier modes while preserving the distribution of the full configuration. The effectiveness of the lightweight configuration is evaluated in Section C.4.

### B.4 Lightweight Judge Model

To improve evaluation efficiency, we train a benchmark-scoped lightweight judge based on qwen3-8b (Team et al., 2025) using GPT-5-generated annotations. The lightweight judge predicts harmfulness scores and mismatch labels under the same evaluation protocol as the primary GPT-5 judge. Its consistency with GPT-5 evaluation is further validated in Section C.4. Detailed training configurations are provided in the Code and Data Supplement.

## Appendix C Extended Results and Ablations

### C.1 Extended Factor-Level Results

The evaluation suite exports stratified tables along every annotated dimension, as well as along the finer-grained scenario level of the harmful-intent taxonomy. For each stratum, the exported tables include the number of evaluated responses, ASR, mismatch rate, and CASR, together with bootstrap confidence intervals computed by resampling intents. Reporting counts alongside rates is important because strata differ in size; for example, the Violence & Physical Harm domain contains three scenarios while the Sexual Content domain contains one.

Two cross-tabulations are provided by default: scenario by carrier, and model by framing template or visual category. These views are convenient for auditors who wish to locate where in the benchmark space a model’s successful jailbreaks concentrate. Users can request any other cross-tabulation through the suite configuration. Because cells in finer cross-tabulations contain fewer responses, we recommend aggregating to the domain level when confidence intervals become wide, and we discourage drawing conclusions from individual cells without accounting for multiple comparisons.

### C.2 Per-Model Factor Sensitivity Profiles

For each model, the suite computes the ASR range across the configurations of each annotated dimension and can render these values as a scatter plot in which the horizontal and vertical axes correspond to two chosen dimensions and marker size encodes a third. This visualization offers a compact overview of a model’s evaluation outputs and is useful for spotting models whose behavior warrants closer inspection.

Optional grouping of models into profiles, for example by clustering the range vectors, is also supported for exploratory use. Such groupings depend on the choice of features, distance, and number of clusters, and we therefore treat them as a visualization aid rather than as a classification of models. When clustering is used, we recommend multiple random restarts and reporting the stability of assignments across restarts.

### C.3 Semantic Image Ablation

To support controlled studies of visual context by benchmark users, MMJailBench releases four control image conditions alongside the five task-relevant semantic categories: no image, a blank image, a random-noise image, and a generic nature image. The control conditions are designed to span a range from the complete absence of visual input to visually rich but task-irrelevant content. Blank and noise images contain no semantic content, while nature images contain natural-scene semantics that are unrelated to any harmful intent.

Control images are rendered at the same resolution as the semantic images and are paired with the same instruction components, so that they can be substituted directly into the evaluation pipeline. Users who wish to study the role of image content can compare semantic categories against these controls under their chosen models and judges. We note that some models apply different preprocessing depending on image content or aspect ratio, and recommend verifying preprocessing behavior when designing such comparisons.

### C.4 Evaluation Efficiency and Validation

We evaluate the efficiency and reliability of the MMJailBench evaluation pipeline from two perspectives. First, we compare the lightweight benchmark configuration with the full evaluation setting to assess whether the reduced evaluation scale preserves benchmark consistency. Table 11 shows that the lightweight configuration closely matches the full setting, with small ASR differences and strong agreement across matched evaluations, demonstrating that the lightweight setting maintains the main evaluation characteristics.

Second, we validate the reliability of the judging protocol through human expert evaluation and lightweight judge validation. Table 12 reports the agreement between GPT-5 judgments and human expert annotations, as well as the consistency between the lightweight judge and GPT-5. Both harmfulness scoring and mismatch detection show strong agreement, supporting the reliability of our evaluation framework.

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

MMJailBench contains harmful intents and adversarial prompts for multimodal safety evaluation. The benchmark is released for research purposes only, with documentation and evaluation tools to support reproducible and responsible use. Users are encouraged to apply the released resources for research on improving model safety and robustness.

### D.2 LLM Usage Statement

LLMs are used in MMJailBench for harmful-intent organization, visual prompt generation, and response evaluation. Generated materials are filtered and validated through predefined procedures, while the final benchmark design and protocols are determined by the authors.