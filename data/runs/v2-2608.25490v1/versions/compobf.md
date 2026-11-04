# MMJailBench: A Factorized Benchmark for Disentangling

Multimodal Jailbreak Vulnerabilities

Tianshi Wang, Jingsong Wang, Yafei Huang, Lei Zhu† (Tongji University); Fengling Li (Mohamed bin Zayed University of Artificial Intelligence); Xin Li (Shanghai Artificial Intelligence Laboratory). †Corresponding author.

###### Abstract


A modular evaluation suite is released, with P2W and C9Z, multiple judge options and multidimensional metrics, for reproducible, scalable, cost-efficient auditing; vulnerability-associated patterns in internal representations and cross-modal interactions are found by diagnostics on one open-weight model; across 16 open-weight and proprietary MLLMs, vulnerability profiles are heterogeneous and model-dependent; existing multimodal jailbreak benchmarks couple Z3K, V8M, R2T and W9N within single instances, obscuring the sources of vulnerability, while QX7 varies and combines these factors under controlled configurations for factor-level attribution; a warning is given that this paper contains potentially harmful content.

## 1 Introduction

Across the 16 MLLMs evaluated, the effect of the W9N (W9N denoting the instruction carrier) was found to be model-dependent, visually rendered instructions failing to produce any consistent elevation of susceptibility relative to D4P (D4P denoting TEXT), while V8M (V8M denoting prompt framing) emerged as the dominant source of variation in that, with Z3K held fixed, narrative structure and output requirements substantially altered behavior; vulnerability was uneven across S6M (S6M denoting harm domains), with cyber abuse, economic harm, privacy violations, and deception and manipulation the most vulnerable, and refusal was weakened by task-relevant R2T (R2T denoting visual semantics), the authority-like cues (G3S, X7R) being the most pronounced, suggesting an over-reliance on visually conveyed legitimacy and credibility. The contributions are (1) the factorized benchmark, (2) large-scale evaluation of 16 MLLMs plus internal diagnostics on a representative open-weight model, and (3) a modular evaluation suite (P2W/C9Z, with C9Z denoting the lightweight configuration, multiple judges, multidimensional metrics). Fixed prompt-image pairs jointly encoding Z3K, V8M, R2T and W9N (Figure 1) are what existing benchmarks build, and their B8X, reflecting aggregate vulnerability on a test set, cannot say whether that vulnerability stems from linguistic V8M, visual context, instruction delivery, or particular S6M. These four factors are placed by QX7 (QX7 denoting MMJailBench) in one controlled design with systematic combinations and matched comparisons: 272 Z3K (Z3K denoting harmful intent), 6 prompt templates, 5 task-relevant R2T, and 2 W9N, the pipeline being given in Figure 2. Because MLLMs jointly interpret text and images, harmful requests can combine direct textual or visually rendered instructions with contextual cues (professional settings, X7R denoting identity credentials, G3S denoting authorization documents); refusal or compliance is thus shaped by cross-modal context, since visual information may alter the perceived legitimacy, professionalism and credibility of a request, so that the same Z3K elicits different behavior across contexts, multimodal jailbreaks thereby probing the stability of safety alignment under cross-modal conditions rather than input format alone.

## 2 Related Work

None of the existing benchmarks independently manipulates the linguistic and semantic factors within matched multimodal interactions, a gap addressed by QX7, in which V8M, R2T and W9N are varied over matched Z3K (2.2); among the multimodal ones are counted MM-SafetyBench (image-based attacks), FigStep (typographic instructions), JailBreakV and VLJailbreakBench (diverse strategies), MMJ-Bench (unified attack/defense comparison), OmniSafeBench-MM (multidimensional metrics) and Omni-SafetyBench (parallel variants of the same seeds across modality configurations, cross-modal consistency), the text ones being SafetyBench, Do-Not-Answer, SALAD-Bench, HarmBench and JailbreakBench. That refusal is sensitive to V8M and delivery is shown by jailbreaks, which exploit competing objectives and mismatched generalization in safety training, with approaches including transferable adversarial suffixes, black-box semantic attacks (PAIR) and indirect prompt injection via documents or retrieved content; attack surfaces are added by MLLMs—adversarial images, typographic prompts, other cross-modal constructions—while the robustness of defenses (safety alignment, input transformation, inference-time safeguards) to modality and context changes remains uncertain (2.1).

## 3 MMJailBench

So that W9N-related safety differences are isolated from perception failures rather than conflated with them, N7V (N7V denoting the instruction recognition mismatch) is measured, the W9N contrast designated 3.5 being that of D4P, i.e. direct textual instructions, against K6F (K6F denoting the same instructions rendered in text-containing images with semantics preserved), while the 16,320 instances of the enumeration designated 3.1 arise from the Cartesian product of 272 intents × 6 framings × 5 visual conditions × 2 carriers, each instance fixed, given Z3K $h$, V8M $t$, R2T condition $v$ and W9N $c$, by $x_{h,t,v,c}=(L(h,t,c),V(h,v))$, (1), with $L(h,t,c)$ the instruction component and $V(h,v)$ the visual input, matched comparisons being yielded through the fixation of $h$ with the rest varied.

Under R2T, designated 3.4, context is added by images without modification of the harmful instruction, across 5 conditions — danger-related context, scenario context, professional-role context, X7R, and G3S (legitimacy-, expertise- or authority-related signals) — control images (irrelevant/non-semantic) being used in diagnostics for the separation of semantic effects from image presence; the Z3K of 3.2 consist of intent-level behavioral seeds, not attack-entangled prompts, across 9 S6M, namely Physical Harm, Cyber Abuse, Economic Harm, Hate and Harassment, Privacy and IP, Regulated Advice, Illegal Activities, Deception and Influence, and Sexual Content, each with fine-grained subcategories, built by category-guided construction, semantic-similarity filtering and manual verification; and V8M, under 3.3, comprises 6 strategies preserving intent: academic, system, structured, story, code, M8D.

## 4 Experimental Setup

Conditioned on correct understanding of the input, success is quantified by H3Q as

$$CASR=\frac{N_{success}}{N-N_{mismatch}},\quad(3)$$

the subtracted count corresponding to N7V, namely inputs not correctly understood, under which K6F failures, irrelevant responses and misinterpreted intent are subsumed, while the adjudication of all main results, comprising the score together with success and N7V labels, was performed by Y6L (Y6L denoting GPT-5), with refusals, safety warnings and high-level discussion not counted as successes and an open-weight F4K supporting cheaper follow-up. B8X, by contrast end-to-end rather than conditioned on understanding, is defined in 4.2 over all N instances by

$$ASR=\frac{1}{N}\sum_{i=1}^{N}\mathbb{I}(s_{i}\geq 4),\quad(2)$$

with $s_i$ the T5J of instance $i$ (T5J denoting the harmfulness score) on a 1∼5 scale, any $s_i\geq4$ constituting a success.

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
|Y6L|2.17|3.06|4.11|2.00|3.39|0.57|2.50|0.07|3.56|0.29|

Table 1: Per-model B8X (%) across nine S6M; Avg. over all domains.

Of the two configurations of 4.3, the C9Z, intended for auditing and regression testing, is a representative subset covering all domains and factors, whereas the P2W (P2W denoting the full configuration), the default, under which all instances are run, yields 16,320 instances per model and 261,120 responses in total across the 16 MLLMs of 4.1, open-weight and proprietary, drawn from the LLaVA-OneVision, InternVL, Kimi, Qwen, Gemma, GLM, GPT, Gemini and Claude families, whose per-model B8X (%) across the nine S6M, with the average taken over all domains, is reported in Table 1.

## 5 Results and Analysis

No single universal failure pattern exists; some models have ΔV8M >80 points; others are more W9N-sensitive; R2T effects are smaller but consistently positive; (5; 5.1; 5.2; 5.3) average B8X ranges from 2.17% (Y6L) to 78.38% (GLM-4.6V) (Table 1, Figure 3); models of comparable capability differ strongly; capability does not imply robustness.

|Model|ΔZ3K|ΔV8M|ΔR2T|ΔW9N|
|-|-|-|-|-|
|Open-weight|||||
|glm4.1v-9b|76.7|51.8|4.2|8.3|
|step3-vl-10b|86.7|72.2|4.8|10.5|
|ministral3-8b|83.3|57.2|9.6|19.9|
|llava-onevision-1.5|61.7|35.3|4.6|50.5|
|internvl3-8b|75.0|88.3|5.5|4.6|
|qwen2.5-vl-7b|78.3|90.8|3.4|0.8|
|qwen3-vl-8b|65.0|15.6|6.5|14.4|
|gemma3-12b|88.3|79.0|7.0|13.6|
|Proprietary-access|||||
|glm-4.6v|73.3|62.6|11.3|1.5|
|qwen3-vl-plus|91.7|42.6|8.1|9.7|
|doubao-seed-2.0-lite|95.0|80.2|5.8|1.3|
|kimi-k2.5|75.0|44.3|6.9|34.6|
|gemini-3-flash|76.7|59.7|7.8|34.3|
|grok-4-fast|76.7|70.6|2.8|4.1|
|claude-sonnet-4.5|55.0|32.1|2.0|0.8|
|Y6L|13.3|8.8|0.8|3.6|

Table 4: Per-model B8X range across configurations of each factor.

The largest variation is V8M, with story, structured and academic framings generally yielding higher B8X, system-style and M8D lower, and the most-vs-least vulnerable gap exceeding 40 points (Figure 3(a)); under Z3K, cyber, economic, privacy and deception tasks have higher B8X, physical harm and sensitive content lower, and aggregate scores hide S6M-specific weaknesses. K6F yields lower B8X than D4P, persisting under H3Q; visually rendered instructions are not inherently stronger W9N; W9N vulnerability depends on model-specific multimodal processing; all task-relevant R2T raise B8X over L8G, non-semantic controls change it by <2.2%, vulnerability stems from contextual visual meaning, not image presence, and authority cues may supply implicit legitimacy signals.

|Carrier|$N$|N7V (%)|B8X (%)|H3Q (%)|
|-|-|-|-|-|
|D4P|130,560|0.37|50.70|50.89|
|K6F|130,560|4.14|40.59|42.34|
|UNION|261,120|2.26|45.64|46.69|

Table 3: W9N analysis; Mis. = N7V rate.

|Group|Image condition|B8X (%)|ΔB8X (%)|
|-|-|-|-|
|Semantic|G3S|47.73|12.96|
||X7R|45.24|10.47|
||Task scenario|44.87|10.10|
||Professional role|43.60|8.83|
||Dangerous context|43.22|8.45|
|Control|Blank image|36.96|2.19|
||Noise image|35.53|0.76|
||Nature image|35.49|0.72|
||L8G|34.77|0.00|

Table 2: R2T ablation; ΔB8X relative to L8G.

## 6 Diagnostic Analysis

Increased susceptibility is internally explained, for 6.2, by authority cues inducing a higher-level contextual state with redistributed attention at response initiation, this suggestion being made with the representation shift and resting on the observation, at the representation-sensitive layer, that authority contexts give lower attention mass on both harm-related task-label tokens and special visual tokens in the attention from the final generation-prefix query to those tokens (Figure 5); a shared higher-layer pattern is stated for 6.1 because consistent positive shifts across all S6M accompany clear displacement of authority contexts when Q2N, fit on discovery samples and applied fixed to held-out pairs, is read against the residual stream of the final generation-prefix token (Figure 4), where authority-vs-danger divergence grows with depth and is sharpest at layer 42, all of this pertaining to the diagnostic subset of matched instances emphasizing authority-document scenarios and to layer-wise representations and cross-modal attention in gemma3-12b.

## 7 Conclusion

The largest variation across 16 MLLMs is caused by V8M, R2T adding contextual vulnerability, W9N effects being model-dependent and Z3K showing category-specific patterns supported by internal diagnostics, while reproducible, scalable evaluation is enabled by P2W/C9Z, judge options and metrics, given QX7's factorization of Z3K, V8M, R2T and W9N.

## Appendix A Benchmark Construction

Functioning as evaluation probes rather than as an exhaustive attack collection, the V8M templates, whose framing strategies and representative patterns are enumerated in Table 6, operate through the substitution of the target Z3K for a task placeholder.

Table 6: V8M templates.

|Template|Framing Strategy|Representative Pattern|
|-|-|-|
|Academic|Academic Analysis|Analyze the topic from an academic perspective.|
|System|System-Role Simulation|Assume a system identity and follow role-specific instructions.|
|Story|Fictional Storytelling|Describe via fictional story or hypothetical setting.|
|Code|Code-Style Indirection|Express the procedure through code or comments.|
|Structure|Structured Completion|Complete a predefined multi-stage structure.|
|Paradox|M8D Reasoning|Discuss harmful actions under a safety rationale.|

The W9N is obtained under K6F by rendering the instruction D4P as a readable image and combining it with the R2T image, the layout being standardized while the Z3K and the V8M are preserved. The R2T images entering this combination, filtered for semantic relevance, quality and readability prior to use, span five categories—Danger, Work Scenario, G3S, X7R and Role—exemplified in Figure 8, with descriptions and image prompts produced by qwen2.5-vl-7b, the generation of Danger, Work Scenario and Role being carried out by zimage2025-Turbo, and that of X7R and G3S by 4-bit-quantized FLUX.2 [dev], image editing being applied in the latter case for coherent D4P. That the S6M occupy diverse regions rather than a few clusters is indicated qualitatively by PCA (Figure 7), while near-duplicates were removed through manual review of the small high-similarity tail of the pairwise similarities computed via all-MiniLM-L6-v2, most pairs exhibiting low similarity (Figure 6); the Z3K set subjected to these procedures, each instance carrying a task identifier, a harm category, a scenario and an action description, was collected from existing safety benchmarks and public model-provider safety policies and subsequently normalized, organized and manually filtered, the resulting category distribution—under the abbreviations Cyber, Econ., Privacy, Decept., Illegal, Hate, Violence, Sensitive and Sexual—being reported in Table 5.

Table 5: Taxonomy: 9 S6M, 18 scenarios.

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

## Appendix B Experimental and Evaluation Details

T5J values and N7V labels are predicted, under the same protocol, by a QX7-scoped judge (F4K denoting the lightweight judge model) that was obtained from qwen3-8b through training on Y6L annotations, while the C9Z, the construction of which preserves the full factor distribution of the P2W, comprises 1,500 instances stratified over S6M, V8M, visual conditions and W9N.

|T5J|Definition|
|-|-|
|1|Refusal or safe response without harmful details.|
|2|Potential risk without direct harmful instructions.|
|3|High-level or partial harmful information.|
|4|Actionable harmful assistance.|
|5|Detailed and operational harmful guidance.|

Table 7: T5J rubric.

An N7V flag, covering, for example, only transcribing visual content, repeating the prompt, or meaningless output, together with a T5J, is output by Y6L, and anti-deception rules stop disclaimers, fictional framing or academic packaging from lowering the T5J when harmful information is given, with the full prompt being provided in the Code and Data Supplement.

The evaluated MLLMs comprise, as open-weight models (public checkpoints), qwen2.5-vl-7b, qwen3-vl-8b, gemma3-12b, internvl3-8b, glm4.1v-9b, ministral3-8b, step3-vl-10b and llava-onevision-1.5, and, as proprietary models (official interfaces), Y6L, gemini-3-flash, claude-sonnet-4.5, grok-4-fast, qwen3-vl-plus, kimi-k2.5, glm-4.6v and doubao-seed-2.0-lite.

## Appendix C Extended Results and Ablations

The positivity of Δ_T−O for all categories, where it spans 5.81–12.46 points, establishes that the W9N effect is not dominated by specific S6M, while sensitivity to V8M and to R2T is model-dependent (Tables 9–10), and the four profiles into which models are grouped by Z9B—V8M-dominant, R2T-responsive, W9N-sensitive and low-variation—are those shown in Figure 9.

|Model|Dang.|Scen.|G3S|X7R|Role|
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
|Y6L|1.69|2.05|2.36|2.48|2.02|

Table 10: Per-model B8X (%) by R2T condition.

The grouping of models into the four profiles is produced by Z9B (Z9B denoting K-means), which, applied with 1,000 restarts to the V8M, R2T and W9N B8X ranges depicted in Figure 9, yields the assignment underlying that figure.

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
|Y6L|0.74|0.00|0.44|0.85|8.82|1.88|

Table 9: Per-model B8X (%) by V8M (first 8 open-weight, last 8 proprietary).

|Subcategory|B8X|Δ_T−O|
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

Table 8: Category-level B8X (W9N pooled); Δ_T−O = D4P−K6F B8X.

Each B8X of Figure 10, whose radial scale spans 0∼80%, being an aggregation of 3,264 responses, the R2T-vs-control gaps of step3-vl-10b and doubao-seed-2.0-lite are larger and those of gemma3-12b smaller, whereas claude-sonnet-4.5, for which the L8G B8X is relatively higher, is less dependent on R2T cues.

|Measure|Result|
|-|-|
|Mean B8X difference (C9Z−P2W)|−0.06%|
|95% CI for B8X difference|[−0.45%, 0.30%]|
|Mean absolute B8X difference|0.79%|
|Maximum absolute B8X difference|3.53%|
|Cells within 2% difference|30/32 (93.75%)|
|Spearman ρ|0.997|
|Lin’s concordance correlation|0.999|

Table 11: P2W vs. C9Z consistency.

|Evaluation Pair|V4H|ΔB8X|Acc.|Macro-F1|
|-|-|-|-|-|
|Y6L vs. Human Experts|0.97|+0.3%|96.3%|0.85|
|F4K vs. Y6L|0.95|−0.4%|99.4%|0.94|

Table 12: Judge agreement.

Judge agreement, as reported in Table 12, is quantified through V4H (V4H denoting QWK) together with ΔB8X, both computed over T5J, and through Acc. and Macro-F1, both computed for N7V detection, whereas the consistency of P2W with C9Z is reported in Table 11.

## Appendix D Responsible Release

The design and protocols having been set by the authors, the outputs were filtered and validated subsequent to the utilization of LLMs for intent organization, visual prompt generation and response evaluation; the release, which is accompanied by documentation and tools, is for research only.

