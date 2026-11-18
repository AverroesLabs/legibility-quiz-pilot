# From Preferences to Prejudice:

The Role of Alignment Tuning in Shaping Social Bias in Video Diffusion Models

Zefan Cai  
††thanks: Equal contribution, ordered by last name.  
Email: zefncai@gmail.com  
Affiliation: University of Wisconsin–Madison

Haoyi Qiu  
Email: haoyiqiu@cs.ucla.edu

Haozhe Zhao  
Email: haozhez6@illinois.edu

Ke Wan  
Affiliation: University of California, San Diego

Jiachen Li  
Affiliation: University of California, Santa Barbara

Jiuxiang Gu

Wen Xiao  
Affiliation: Microsofthttps://github.com/Zefan-Cai/VideoBiasEval

Nanyun Peng  
††thanks: Equal advising.  
Affiliation: University of California, Los Angeles

Junjie Hu  
Affiliation: University of Illinois Urbana-Champaign

###### Abstract

Alignment-tuned video diffusion yields smoother yet more stereotyped, temporally persistent portrayals; preference-data biases enter reward models, undergo amplification, and propagate into generation. QX2 supplies event-based separation of verbs and contexts from actor gender and ethnicity, with multi-granular measurement of ethnic skew, ethnicity-conditioned gender skew, cross-variant attribute-distribution shifts, and within-video persistence. The first end-to-end analysis connects human comparison data, learned image rewards, and aligned video outputs. Human-preference-trained rewards improve visual quality while unintentionally encoding and amplifying social bias; fair, socially responsible generation requires bias-aware assessment and mitigation throughout alignment.

Reviewed on OpenReview: https://openreview.net/forum?id=C0yxuS6jty

**Figure 1.** QX2’s event prompts and multi-granular ethnicity/gender audit occupy the bottom left; frame-wise VLM inference and aggregation of actor attributes occupy the top. Bottom right: the first comprehensive analysis separating human preference bias, image-reward preference bias, and downstream diffusion-video generation bias, including human-labeled rewards’ effects on social-attribute distributions.

## 1 Introduction

Both human comparison collections favor White representations and exhibit non-neutral gender preferences; learned rewards inherit and amplify these patterns, male- or female-preferred rewards systematically redirect gender portrayals, and ethnic distributions remain uneven. Curating attribute distributions can intentionally control image-reward datasets, learned preferences, and resulting video representations; targeted composition and counter-biased rewards offer mitigation guidance. The analysis covers reward-preference changes over 42 events and corresponding pre-/post-alignment video-bias changes.

High-quality natural-language video generation has advanced substantially (Chen et al., 2024a; Wang et al., 2023a; Yuan et al., 2024; Li et al., 2024), supporting creative media, education, and professional simulation (Cho et al., 2024; Miller et al., 2024). State-of-the-art systems increasingly use alignment to improve fidelity, coherence, and controllability, particularly human-preference learning (Wu et al., 2023; Xu et al., 2024; Li et al., 2024; Yuan et al., 2024; Liu et al., 2024a; Prabhudesai et al., 2024; Black et al., 2023; Ma et al., 2025); human-labeled comparisons commonly train frame- or image-judgment rewards for post-training optimization (Wu et al., 2023; Kirstain et al., 2023a; Xu et al., 2024).

“Aligned models” denotes video diffusion models optimized after likelihood-based training to better match human preferences, commonly through reward-weighted fine-tuning or RLHF-style optimization guided by learned preference/aesthetic rewards trained on human comparisons; the conceptual sequence is
\[
\text{base video diffusion model}\rightarrow
\text{preference or reward model}\rightarrow
\text{post-training optimization}.
\]
Unaligned counterparts use likelihood objectives alone and supply controlled baselines isolating alignment effects.

Subjective preference judgments, often collected without explicit cultural, social, or demographic diversity considerations, can encode systematic bias inherited and reinforced by rewards and downstream generators (Qiu et al., 2023), affecting not only generation quality but who appears, how frequently, and in which roles. The question concerns alignment’s influence on social-attribute distributions.

A probing method, distributional metrics, and a pre-/post-alignment analysis protocol are jointly required; existing frameworks (Huang et al., 2024; Liu et al., 2024b; Sun et al., 2024) inadequately represent diverse identities in prompts, lack comprehensive identity coverage and specific distribution-tracking metrics, and lack a dedicated alignment-shift method. Established social-bias taxonomies ground the controlled separation of actor identity and semantic content (QX2 = VideoBiasEval), with precise verb/identity variation supporting robust, interpretable assessment across contexts (Zhao et al., 2017; Hendricks & Nematzadeh, 2021; Cho et al., 2023; Qiu et al., 2023). Gender and ethnicity have comparatively well-defined evaluation boundaries in prior generative alignment/fairness work (Luccioni et al., 2023; Shen et al., 2023).

Three contributions are the comprehensive event-prompt/multi-granularity audit; the first connected preference-data–reward–video analysis of inheritance, amplification, and stabilization; and systematic evidence that perceptual/temporal improvements also reshape, sometimes reinforce, generated social composition, together with controllable-preference guidance toward equitable behavior.

## 2 Related Work

**T2V evaluation.** VBench (Huang et al., 2024), EvalCrafter (Liu et al., 2024b), and T2V-CompBench (Sun et al., 2024) use Fréchet Video Distance (Unterthiner et al., 2019), CLIP-Score (Hessel et al., 2021), and object consistency without auditing depicted identities or their portrayal. GRiT-based measures (Wu et al., 2025) can verify a doctor’s presence without flagging exclusively white male doctors; Y2F-based alignment rewards textual fidelity rather than demographic balance. Fair, trustworthy T2V evaluation requires explicit output-distribution auditing beyond surface measures; QX2 provides ethnicity-/gender-aware, event-centric quantification throughout generation.

**Generative-model bias evaluation.** Most image/language-generation studies examine static portraits or isolated object scenes. StableBias (Luccioni et al., 2023), DALL-Eval (Cho et al., 2023), and SocialCounterfactuals (Howard et al., 2024) chiefly count identities, seldom their activities. Independent-image demographic audits conceal recurring authoritative-men/supportive-women portrayals. Omitting joint actor–verb–context analysis misses role-specific stereotypes and narrative/temporal bias; event-level disentanglement exposes scenario-dependent representation.

## 3 QX2

QX2 generates structured-event videos with state-of-the-art diffusion models, extracts social representations, and audits identity consistency and fairness across event categories and temporal/representational granularities.

**Figure 2.** Different diffusion models’ videos use varied actor-attribute templates; VLM-based extraction records the main character’s gender and ethnicity. Representative prompts, classification prompts, and outputs illustrate the analysis.

### 3.1 Event Definition

Socially associated verbs have historical identity-stereotype associations; imbalanced gender/ethnic portrayals during their performance reinforce stereotypes and undermine fairness (Bolukbasi et al., 2016; Sun & Peng, 2021; Zajko, 2021). An event is
\[
\langle p,a,c\rangle,
\]
with actor \(p\), performed verb \(a\), and context \(c\). The formulation measures demographic influence on visual depictions and systematic model bias, following socially associated-verb studies (Zhao et al., 2017; Garg et al., 2018; Cho et al., 2023; Qiu et al., 2023).

The bounded attribute space balances coverage against control: four gender categories, seven ethnic groups, and 42 verbs, grounded in established taxonomies and benchmark conventions, enable reproducible, interpretable first end-to-end bias-propagation tracing with clear attribution. Open-ended/intersectional expansion remains promising; isolating disparities precedes unconstrained scaling.

Each actor carries gender and ethnicity attributes. Luccioni et al. (2023) supplies **man**, **woman**, neutral **person**, and **non-binary person**: inclusive but incomplete coverage of gender diversity, with controlled evaluation boundaries. Karkkainen & Joo (2021) and U.S. Census Bureau conventions supply **White, Black, Indian, East Asian, Southeast Asian, Middle Eastern, Latino**: socially constructed categories, neither exhaustive nor universally representative.

The verbs are **bake, bike, call, clean, climb, cook, cough, cry, drink, drive, eat, exercise, fish, hit, jump, kick, kneel, laugh, lift, paint, pick, pitch, pray, read, ride, row, run, shop, shout, sit, skate, sleep, smile, stand, stare, stretch, study, sweep, throw, walk, wash, work**. Prior literature identifies their statistical gender/ethnicity associations (Zhao et al., 2017; Garg et al., 2018; Cho et al., 2023; Qiu et al., 2023). High visual distinctiveness and short-clip depictability additionally support reliable annotation and output interpretation; this curated socially meaningful set permits reproducible extension toward complex, culturally specific, or temporally dynamic events.

### 3.2 Event-Based Prompting Protocol

| Setting | # of prompts | Examples |
|---|---:|---|
| J7K | 42 | A person is baking cookies in a cozy kitchen, with warm light and the aroma of vanilla filling the air. |
| V9R | 294 | An East Asian person is baking cookies in a cozy kitchen, with warm light and the aroma of vanilla filling the air. |

**Table 1.** Video-generation audit prompts explicitly contain ethnicity when specified, verb, and surrounding context.

Underspecified actors receive implicit gender/ethnicity assignments in J7K; V9R tests interactions between explicit ethnicity and inferred attributes including gender. Identical decoding and fixed videos-per-prompt permit systematically comparable diversity; post hoc demographic analysis uses a VLM ensemble on generated frames. Demographic attributes are not introduced, constrained, or inferred during generation.

The fully deterministic template is **“A/An [actor] is [verb]-ing [context].”** Neither actor nor context is LLM-generated at runtime; fixed predefined instantiations maintain reproducibility, control, and situational variation without changing verb semantics. Fixing the actor to “person” yields one prompt per curated activity (J7K = Person-only), retaining the same semantic structure while omitting explicit demographic cues. Deterministically prepending a predefined ethnic descriptor to “person,” including “An East Asian person,” permutes the verbs across ethnic groups (V9R = Ethnicity+Person), preserving contexts and making an additional ethnicity-only setting unnecessary for isolating ethnicity effects. This controlled construction attributes observed disparities to demographic conditioning rather than uncontrolled contextual drift.

### 3.3 Multi-Granular Event-Centric Bias Evaluation

A neutral-person audit uses parity diagnostically, not as empirical demographic truth or universal normative appropriateness; gender proportions can legitimately differ by verb, ethnicity, and sociocultural context, and ethnic proportions by region, culture, and application. Neutral identity/content decoupling makes deviations probes of learned stereotype reproduction or internal favoring/suppression, rather than external statistics. Uniform/parity baselines follow representational-harm and fairness practice.

**Attribute extraction.** Three open-source judges independently classify each uniformly sampled frame: Qwen2-VL-7B (Wang et al., 2024a), Qwen2.5-VL-7B (Yang et al., 2024), InternVL2.5-8B (Chen et al., 2024b). For each video, 16 frames match standard evaluated-model inference defaults, including VideoCrafter; preliminary frame-count reductions significantly destabilized detection. Outputs are
\[
g\in G=\{\text{man,woman}\},\qquad
e\in E=\{\text{White,Black,Indian,East Asian,Southeast Asian,Middle Eastern,Latino}\}.
\]
Within-model frame majority votes produce video labels, subsequently fused across models. Initial manual inspection found single-model instability and hallucinated attributes; distinct-architecture ensembling, not repeated sampling of one model, cross-validates outputs, improves robustness, and mitigates idiosyncratic bias (Qiu et al., 2024b). Frame/video representations also assess alignment to intended prompt demographics.

**Temporal quantity.** The percentage of frames whose classified attribute matches the video’s final majority-voted label defines within-video temporal stability (T8Q = Temporal Attribute Stability, TAS), with high values denoting coherent demographic identity and low values denoting attribute flicker/inconsistent identity, a critical artifact of biased or unstable generation.

**Conditional gender quantity.** For each verb and ethnicity, representation counts define the signed diagnostic (L2V = Proportion Bias Score for Gender, PBSG), with
\[
\mathrm{L2V}=\frac{N_{\text{man}}-N_{\text{woman}}}{N_{\text{total}}}\in[-1,1],
\qquad N_{\text{total}}=N_{\text{man}}+N_{\text{woman}}.
\]
Positive values favor men, negative values women, near-zero values balance. Computing it under the ethnicity+gender condition probes ethnicity-conditioned verb stereotypes, consistency of balance across groups, and uneven group-specific amplification. Gender-only analysis is excluded: absent specified ethnicity, generation predominantly depicts White individuals, reducing that analysis effectively to White-group gender bias.

| Diagnostic illustration | Interpretation |
|---|---|
| \(\mathrm{L2V}=0.5\) | Men occur three times as frequently: \(N_{\text{man}}=3N_{\text{woman}}\). |
| \(\mathrm{L2V}=-0.5\) | Inverse ratio. |
| \(\mathrm{L2V}\approx0\) | Controlled diagnostic target, not real-world ground truth. |
| Seven ethnic groups | Balanced probability \(1/7\approx14.3\%\). |
| \(\mathrm{F6N}_e=0.769\) | \(0.769+0.143\approx91.2\%\) of videos; more than six times expected frequency. |

**Ethnic quantities.** Identifiable outputs define
\[
P_e=\frac{N_e}{N_{\text{total}}},\quad e\in E,
\]
where \(N_e\) counts outputs classified as \(e\) and \(N_{\text{total}}\) counts all outputs with identifiable ethnicity; group-specific deviation from uniform coverage supplies one complementary quantity (F6N = Representation Deviation Score for Ethnicity, \(\mathrm{RDS}_e\)), following Feldman et al. (2015) and Mehrabi et al. (2021),
\[
\mathrm{F6N}_e=P_e-\frac1{|E|}.
\]
Positive/negative values denote over-/underrepresentation. The uniform reference pertains to neutral J7K auditing, whose prompts intentionally remove contextual and identity cues. The probability that two random outputs belong to different groups supplies the other quantity (W3H = Simpson’s Diversity Index, SDI), following Simpson (1949),
\[
\mathrm{W3H}=1-\sum_{e\in E}P_e^2.
\]
Higher values indicate more diverse/balanced distributions; lower values indicate concentration in dominant groups. Group deviations identify who is over-/underrepresented, whereas overall diversity measures landscape balance, jointly exposing homogeneous/skewed rather than globally diverse portrayal.

**Shift quantities.** Alignment-induced directional changes are \(\Delta\mathrm{L2V}\), \(\Delta\mathrm{F6N}_e\), and \(\Delta\mathrm{W3H}\). More balanced gender ratios, less ethnic skew, or higher diversity constitute improvement. Aggregating across videos complements fine frame dynamics and temporal persistence in assessing bias formation/mitigation.

## 4 Social Biases in Video Generative Models

| Models | Average L2V | White L2V | White F6N | Black L2V | Black F6N | Latino L2V | Latino F6N | East Asian L2V | East Asian F6N | Southeast Asian L2V | Southeast Asian F6N | India L2V | India F6N | Middle Eastern L2V | Middle Eastern F6N | Overall W3H |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| D8P (u) | 0.4815 | 0.5683 | 0.7690 | 0.3912 | -0.1952 | 0.6308 | -0.1952 | 0.4406 | -0.1810 | 0.4611 | — | 0.3938 | — | 0.4833 | -0.1976 | 0.0538 |
| K5X (a) | 0.5295 | 0.5584 | 0.7833 | 0.5114 | -0.1976 | 0.6729 | -0.1929 | 0.4282 | -0.1976 | 0.5020 | — | 0.4878 | — | 0.5393 | -0.1952 | 0.0267 |
| \(\Delta\) | +0.0480 | -0.0099 | +0.0143 | +0.1202 | -0.0024 | +0.0421 | +0.0023 | -0.0124 | -0.0166 | +0.0409 | — | +0.0940 | — | +0.0560 | +0.0024 | -0.0271 |
| B7Q (u) | 0.7581 | 0.7485 | 0.6905 | 0.6167 | -0.1905 | 0.8599 | -0.1952 | 0.6976 | -0.1500 | 0.8272 | — | 0.8032 | — | 0.7560 | -0.1548 | 0.1252 |
| H4M (a) | 0.8306 | 0.8713 | 0.6381 | 0.8095 | — | 0.8599 | -0.2476 | 0.7762 | -0.2426 | 0.8929 | — | 0.7762 | — | 0.7664 | -0.1476 | 0.1119 |
| \(\Delta\) | +0.0725 | +0.1228 | -0.0524 | +0.1928 | — | 0.0000 | -0.0524 | +0.0786 | -0.0926 | +0.0657 | — | -0.0270 | — | +0.0104 | +0.0072 | -0.0133 |

**Table 2.** Two unaligned (u)/aligned (a) pairs; entries average across 42 verbs. Nonzero unaligned L2V indicates inherent pre-alignment bias, not bias-free baselines. Metric signs describe absolute model bias, not shifts; \(\Delta\) rows are relative to the corresponding baseline. Positive/negative \(\Delta\mathrm{L2V}\) marks man-/woman-preference shifts; positive/negative \(\Delta\mathrm{F6N}\) marks over-/underrepresentation shifts. Throughout the video-distribution tables, “—” denotes zero recognizable corresponding-ethnicity instances under J7K, severe diversity absence preventing subgroup computation.

Gender portrayal under V9R remains male-biased across all ethnicities; aligned variants increase the average imbalance, suggesting preference fine-tuning may worsen disparity. All models overrepresent White individuals. D8P has the strongest imbalance, worsened in K5X; B7Q is moderately more balanced, and H4M reduces White dominance while lowering overall diversity. Ethnic-skew reduction can accompany suppressed demographic variety.

| Additional narrative quantities | Value |
|---|---:|
| K5X L2V increase | 0.04 |
| K5X White F6N | 0.783 |
| D8P White F6N | 0.769 |

The four state-of-the-art models differ in alignment strategy: a baseline supplies the first architecture (D8P = ModelScope), cited to Wang et al. (2023a), while its A3J-aligned derivative supplies the first optimized counterpart (K5X = InstructVideo), cited to Yuan et al. (2024). The second baseline appears under several spellings (B7Q = VideoCrafter-2, VideoCrafter-V2, Video-Crafter-V2), with citations to Chen et al. (2024a) and, for the distilled-model analysis, Chen et al. (2023); its optimized counterpart uses U8D, InternVid2-S2 (Wang et al., 2024b), and ViCLIP (Wang et al., 2023b) as alignment rewards (H4M = T2V-Turbo-V1), following Li et al. (2024). Comparing each architecture’s pair isolates alignment from base architecture. Each prompt is generated 10 times per model with different seeds; results are averaged to reduce sampling variance. V9R supplies gender assessment, J7K supplies ethnic balance.

**Human evaluation.** The ensemble is reliable, scalable, and human-aligned despite residual ethnicity disagreement. Three independent researchers experienced in social-bias analysis annotated generated videos using standardized written guidelines and task-specific training grounded in prior gender/ethnicity conventions; verification concerned correct reflection of prompted demographics, not subjective/normative judgment.

| Human verification quantity | Gender | Ethnicity |
|---|---:|---:|
| Generated videos in full pool | 400 | 400 |
| Independent annotators | 3 | 3 |
| Pearson correlation with ensemble | 0.89 | 0.73 |
| Cohen’s Kappa, video-level full-pool verification | 0.91 | 0.78 |
| Fleiss’ Kappa | 0.92 | 0.82 |
| Human-majority/VLM consensus rate | 99.0% | 83.3% |
| Full-pool → perfect-human-majority/VLM-agreement subset diagnostic | L2V: 0.846 → 0.863 | W3H: 0.745 → 0.761 |
| Reported significance for L2V comparison | \(p=0.68\) | — |

Majority-voted open-source judges align strongly with humans; Kappa indicates strong to near-perfect inter-annotator consistency. Many ethnicity discrepancies arise from holistic human judgment versus frame-level precision: fast-motion faces visible briefly can be missed or inferred by humans yet explicitly captured by the pipeline, indicating higher recall rather than systematic false positives. Restriction to perfect human-majority/VLM agreement leaves all core bias metrics statistically stable; the nonsignificant difference indicates residual evaluator uncertainty does not distort measurements. Human review grounds and sanity-checks, rather than replaces, large-scale automated assessment; larger review is desirable but labor-intensive, and small high-consensus validation follows prior model-judge practice.

The investigation separates demographic distributions in image-reward data, biases of learned rewards, and rewards’ influence on aligned-video gender/ethnicity representation.

## 5 Social Biases in Image Reward Datasets and Reward Models

Image-based rewards constitute the de facto state-of-the-art video-alignment approach (Wu et al., 2023; Xu et al., 2024; Li et al., 2024; Yuan et al., 2024; Liu et al., 2024a; Prabhudesai et al., 2024; Black et al., 2023; Ma et al., 2025). Static-image-trained rewards guide temporally coherent video learning; inherent bias can transfer and amplify across frames. Human-labeled image preferences and learned rewards must be distinguished when assessing demographic disparities and temporal attribute evolution.

### 5.1 Preference in Image Reward Datasets

| Dataset | Validated (images, caption, preference) tuples | Covered verbs | Valid man–woman-pair verbs, out of 42 | Directionally preferred verbs | White preference |
|---|---:|---:|---:|---|---:|
| R9C | 28,783 | 29 | 26 | Men: 69.23% (18/26) | 43.34% |
| N6T | 14,958 | 19 | 18 | Women: 61.11% (11/18) | 40.08% |

R9C trends man-preferred and N6T relatively woman-preferred; both strongly favor White, followed by East Asian and Indian representations. Verb-specific exceptions include Black preference for “bake” and East Asian preference for “fish.” Collected preferences may implicitly privilege Western-centric aesthetics/representation; their imbalances risk downstream reward-training reinforcement of existing social inequities. Inclusive, demographically representative global-diversity data are urgently needed.

Two widely used collections supply the human comparisons (R9C = HPDv2), following Wu et al. (2023), and the other collection supplies the contrasting preference trend (N6T = Pick-a-Pic), following Kirstain et al. (2023a; 2023b). GPT-4o-mini extracts caption gender, ethnicity, and verbs; Qwen2-VL-7B, Qwen2.5-VL-7B, and InternVL2.5-8B classify images. Caption/image attributes are aggregated, retaining instances containing predefined verbs. Each tuple contains two images and a human choice of the better caption match. Explicit gender preference uses pairs with one gender preferred and the other dispreferred, including man-versus-woman comparisons; valid-pair filtering narrows verb coverage. Ethnicity and gender preferences measure selection frequency for given verbs.

**Figure 3.** Image-reward-dataset gender preference distributions across 42 verbs.

**Figure 4.** Image-reward-dataset ethnicity preference distributions across 42 verbs.

### 5.2 Preference in Image Reward Models

| Models | Average L2V | White L2V | White F6N | Black L2V | Black F6N | Latino L2V | Latino F6N | East Asian L2V | East Asian F6N | Southeast Asian L2V | Southeast Asian F6N | India L2V | India F6N | Middle Eastern L2V | Middle Eastern F6N | Overall W3H |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Y2F | -0.0726 | 0.0343 | 0.0182 | -0.1198 | 0.0002 | -0.0934 | -0.0013 | -0.1315 | 0.0141 | -0.0865 | 0.0094 | -0.0508 | -0.0299 | -0.0607 | -0.0108 | 0.8495 |
| A3J | 0.6039 | 0.6090 | -0.0423 | 0.7341 | -0.0069 | 0.6512 | 0.0237 | 0.4752 | -0.0031 | 0.5192 | -0.0100 | 0.5922 | 0.0070 | 0.6464 | 0.0315 | 0.8492 |
| \(\Delta\) | +0.6765 | +0.5747 | -0.0605 | +0.8539 | -0.0071 | +0.7446 | +0.0250 | +0.6067 | -0.0172 | +0.6057 | -0.0194 | +0.6430 | +0.0369 | +0.7071 | +0.0423 | -0.0003 |
| U8D | -0.0984 | -0.0833 | -0.0189 | 0.0257 | -0.0321 | -0.0031 | 0.0382 | -0.3044 | 0.0091 | -0.2181 | -0.0099 | -0.0006 | -0.0077 | -0.1053 | 0.0214 | 0.8470 |
| \(\Delta\) | -0.0258 | -0.1176 | -0.0371 | +0.1455 | -0.0323 | +0.0903 | +0.0395 | -0.1729 | -0.0050 | -0.1316 | -0.0193 | +0.0502 | +0.0222 | -0.0446 | +0.0322 | -0.0025 |
| E5W | -0.1157 | 0.0321 | 0.0069 | -0.0777 | 0.0279 | -0.3479 | -0.0118 | -0.2257 | 0.0316 | -0.2163 | 0.0115 | 0.1531 | -0.0391 | -0.1277 | -0.0271 | 0.8483 |
| \(\Delta\) | -0.0431 | -0.0022 | -0.0113 | +0.0421 | +0.0277 | -0.2545 | -0.0105 | -0.0942 | +0.0175 | -0.1298 | +0.0021 | +0.2039 | -0.0092 | -0.0670 | -0.0163 | -0.0012 |

**Table 4.** Reward-model preference bias; averages across 42 verbs.

Y2F slightly favors women and White individuals and has the highest diversity, a relatively balanced ethnic distribution. R9C fine-tuning produces strong A3J man preference consistently across ethnicities; U8D and E5W favor women. Shifts accord with training-data preferences and reveal consistent ethnicity-spanning gender preferences, although U8D’s training data remain undisclosed. Fine-tuned ethnic preferences shift toward Middle Eastern in A3J, Latino in U8D, and East Asian in E5W; all lose diversity.

| Additional narrative quantity | Value |
|---|---:|
| E5W East Asian F6N | 0.0352 |

**Benchmark construction.** The controlled T2I benchmark draws inspiration from R9C and ImageRewardDB (Xu et al., 2024), using state-of-the-art FLUX (Labs, 2023) and event templates to systematically vary gender, ethnicity, and verbs. Explicitly specifying both actor attributes supplies the second generation condition (C4Z = Ethnicity+Gender), while V9R specifies ethnicity alone and both settings generate 100 images per prompt for statistical robustness and a large, diverse evaluation set.

| Settings | # of prompts | Examples |
|---|---:|---|
| V9R | 294 | An East Asian person is baking cookies in a cozy kitchen, with warm light and the aroma of vanilla filling the air. |
| C4Z | 1176 | An East Asian woman is baking cookies in a cozy kitchen, with warm light and the aroma of vanilla filling the air. |

**Table 3.** Image-reward-benchmark prompts explicitly contain ethnicity when specified, verbs, and context.

FLUX is not assumed unbiased; controlled, reproducible demographic conditioning isolates reward preferences independently of generator-specific artifacts. Three annotators independently reviewed 100 randomly sampled images: unanimous agreement covered 77%, conservatively; majority confirmation of correct attributes covered 97%, with only three images lacking consensus. Disagreements concerned visually ambiguous prompts, not systematic failure or demographic-specific distortions. Clear gender/ethnicity cues and high signal-to-noise support reliable reward processing; residual noise is unlikely to materially affect preference patterns.

**Figure 5.** Constructed-benchmark images use **“A/An [ethnicity][gender] is baking [context].”** Only \([\mathrm{gender}]\in\{\text{man,woman}\}\) appears.

**Reward identities and scoring.** Four models are evaluated, including the shared pre-fine-tuning base of the three learned rewards (Y2F = CLIP), cited to Radford et al. (2021), and a reward trained on R9C (A3J = HPSv2.0), cited to Wu et al. (2023). The later reward version is also cited to Wu et al. (2023) (U8D = HPSv2.1), and its unreleased training collection has a distinct designation (X7H = HPDv2.1), whereas N6T trains the remaining learned reward (E5W = PickScore), cited to Kirstain et al. (2023b). Predictions are continuous scalar preference scores, including the corresponding versioned human-preference score, not discrete labels; standardization supports relative controlled demographic comparisons.

For gender evaluation, **“A/An [ethnicity] person is [verb]-ing [context]”** omits gender, while generation uses **“A/An [ethnicity] [gender] is [verb]-ing [context]”** and explicitly fixes all three dimensions. Reward scores are standardized using mean and standard deviation, then averaged over images for each generation prompt. At fixed ethnicity/verb, subtracting women’s average standardized score from men’s yields adapted L2V; it can exceed one. Its sign retains the gender-preference interpretation.

For ethnicity evaluation, **“A person is [verb]-ing [context]”** omits ethnicity, while generation uses **“A/An [ethnicity] person is [verb]-ing [context]”** with ethnicity and verb explicit. Mean/standard-deviation standardization and prompt-level averaging precede fixed-verb softmax normalization across ethnicities (Bridle, 1990; Bishop, 2006), yielding ethnicity-preference distributions for F6N and W3H.

## 6 Social Biases in Preference Alignment

| Models | Average L2V | White L2V | White F6N | Black L2V | Black F6N | Latino L2V | Latino F6N | East Asian L2V | East Asian F6N | Southeast Asian L2V | Southeast Asian F6N | India L2V | India F6N | Middle Eastern L2V | Middle Eastern F6N | Overall W3H |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| G4S | 0.8034 | 0.7925 | 0.6405 | 0.7758 | -0.2381 | 0.8090 | — | 0.7115 | -0.2333 | 0.7945 | — | 0.8634 | — | 0.8071 | -0.1690 | 0.1433 |
| + A3J | 0.9116 | 0.9667 | 0.3667 | 0.9214 | — | 0.9667 | — | 0.8500 | — | 0.9214 | — | 0.9000 | — | 0.8548 | -0.3667 | 0.1257 |
| \(\Delta\) | +0.1082 | +0.1742 | -0.2738 | +0.1456 | — | +0.1577 | — | +0.1385 | — | +0.1269 | — | +0.0366 | — | +0.0477 | -0.1977 | -0.0176 |
| + U8D | 0.2267 | 0.1321 | 0.4286 | 0.2381 | — | 0.3738 | — | 0.1571 | — | 0.3619 | — | 0.1452 | — | 0.1786 | -0.4286 | 0.0976 |
| \(\Delta\) | -0.5767 | -0.6604 | -0.2119 | -0.5377 | — | -0.4352 | — | -0.5544 | — | -0.4326 | — | -0.7182 | — | -0.6285 | -0.2596 | -0.0457 |
| + E5W | 0.3714 | 0.3429 | 0.6833 | 0.3357 | -0.1810 | 0.7190 | -0.1929 | 0.1450 | -0.1952 | 0.4548 | — | 0.2500 | — | 0.3515 | -0.1143 | 0.1467 |
| \(\Delta\) | -0.4320 | -0.4496 | +0.0428 | -0.4401 | +0.0571 | -0.0900 | -0.1929 | -0.5665 | +0.0381 | -0.3397 | — | -0.6134 | — | -0.4556 | +0.0547 | +0.0034 |

**Table 5.** Aligned-model social bias; averages across 42 verbs; missingness follows the J7K zero-recognizable-instance convention.

**Gender.** V9R evaluation finds strong base-model man bias across ethnicities, increased by A3J. U8D/E5W significantly reduce it toward balanced or woman-preferred outputs, reflecting their reward-level woman preference and counteracting the baseline’s man dominance.

**Ethnicity.** Under the ethnicity-only condition, the baseline strongly favors White individuals and underrepresents Black, East Asian, and Middle Eastern groups. U8D improves White/Black balance but significantly decreases Latino representation and diversity; E5W gives the highest diversity, more balanced representation across most groups, and the most demographically equitable outputs.

| Additional narrative quantity | Value |
|---|---:|
| U8D-aligned Latino representation F6N | –0.4352 |

A distilled consistency generator supplies the baseline for reward-specific tuning (G4S = Video Consistency Model distilled from VideoCrafter-V2, VCM-VC2), following Li et al. (2024), with A3J, U8D, and E5W supplying the three image-reward conditions. The H4M training protocol incorporates reward feedback into state-of-the-art Latent Consistency Distillation (Luo et al., 2023), using single-step video generation. During student distillation from a pretrained text-to-video teacher, decoded frames are directly optimized to maximize image–text alignment rewards, guiding each frame toward human-preferred representations. Pre-/post-tuning QX2 comparison measures reward-specific identity shifts.

Temporal results use another reward-version label (M1Y = HPSv2), retained distinctly in the temporal table while all reported changes remain relative to G4S.

| Model | Attribute | Mean T8Q | Median T8Q | Std T8Q | Perfect stability |
|---|---|---:|---:|---:|---:|
| G4S | Ethnicity | 0.8904 | 1.0000 | 0.1673 | 0.5670 |
| G4S | Gender | 0.9763 | 1.0000 | 0.0829 | 0.8850 |
| + M1Y | Ethnicity | 0.9305<sub>+0.0401</sub> | 1.0000 | 0.1358<sub>–0.0315</sub> | 0.7050<sub>+0.1380</sub> |
| + M1Y | Gender | 0.9967<sub>+0.0204</sub> | 1.0000 | 0.0310<sub>–0.0519</sub> | 0.9830<sub>+0.0980</sub> |
| + U8D | Ethnicity | 0.9632<sub>+0.0728</sub> | 1.0000 | 0.1019<sub>–0.0654</sub> | 0.8330<sub>+0.2660</sub> |
| + U8D | Gender | 0.9895<sub>+0.0132</sub> | 1.0000 | 0.0519<sub>–0.0310</sub> | 0.9440<sub>+0.0590</sub> |
| + E5W | Ethnicity | 0.9406<sub>+0.0502</sub> | 1.0000 | 0.1302<sub>–0.0371</sub> | 0.7560<sub>+0.1890</sub> |
| + E5W | Gender | 0.9884<sub>+0.0121</sub> | 1.0000 | 0.0553<sub>–0.0276</sub> | 0.9410<sub>+0.0560</sub> |

**Table 6.** Actor-identity temporal consistency; high scores indicate stable representation, low scores flicker/change. Red and blue subscripts indicate relative improvements and degradations against G4S.

**Temporal behavior.** Alignment substantially improves technical coherence; U8D increases ethnicity stability and perfect-stability frequency, gender stability nears saturation, and standard deviations consistently decline, yielding more uniform frame consistency. Biased rewards can entrench rather than mitigate disparity: A3J amplifies man preference while locking it temporally, with nearly all videos perfectly gender-stable. Smoother, more coherent stereotyping makes bias persistent and deeply embedded; QX2 uniquely exposes alignment’s conversion of representational artifacts into temporally resilient systematic distortions.

Reward profiles largely determine aligned bias direction and magnitude. A3J reinforces male- and White-preferred base tendencies; U8D/E5W counteract man dominance without achieving complete demographic parity, leaving ethnic disparities.

## 7 Controllable Preference Modeling for Video Diffusion Models

Gender supplies a case study of deliberately calibrated or more equitable portrayal through social-attribute composition in image preferences (Sheng et al., 2020); varying composition tests reward-bias direction and video-alignment dynamics.

### 7.1 Image Reward Dataset Construction

Two collections each contain 2.94 million preference pairs from C4Z; each pair fixes action and ethnicity while changing gender, including M-1 versus W-1. The image source is identified with the human-preference-dataset analysis. Prompts are **“A/An [ethnicity] person is [action]-ing [context].”** Man-preferred labels assign male images 1 and female images 0; woman-preferred labels reverse this. An additional 537,660 R9C image pairs enhance face-free diversity. Applied to a man-biased base model, the woman-preferred data correct imbalance and promote equitable gender representation.

### 7.2 Image Reward Model Development & Alignment Tuning

| Models | Average | White | Black | Latino | East Asian | Southeast Asian | India | Middle Eastern |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Y2F | -0.0726 | 0.0343 | -0.1198 | -0.0934 | -0.1315 | -0.0865 | -0.0508 | -0.0607 |
| Z6B | 1.5280<sub>+1.60</sub> | 1.6300<sub>+1.60</sub> | 1.5752<sub>+1.70</sub> | 1.5524<sub>+1.65</sub> | 1.4323<sub>+1.56</sub> | 1.4525<sub>+1.54</sub> | 1.5619<sub>+1.61</sub> | 1.4914<sub>+1.55</sub> |
| P3L | -0.7448<sub>-2.27</sub> | -0.6318<sub>-2.26</sub> | -0.7943<sub>-2.37</sub> | -0.8279<sub>-2.38</sub> | -0.6282<sub>-2.06</sub> | -0.6429<sub>-2.10</sub> | -0.8846<sub>-2.45</sub> | -0.8042<sub>-2.30</sub> |

**Table 7.** Reward-model L2V preference bias; averages across 42 verbs.

Z6B consistently raises gender preference scores across demographic groups, while P3L systematically favors women; reward tuning captures and amplifies gender-specific aesthetic/representational preferences.

The two datasets fine-tune pretrained Y2F vision encoders into the corresponding reward systems (Z6B = Man-Preferred Reward Model, RMM), with the other encoder optimized for the opposite dataset’s preferences (P3L = Woman-Preferred Reward Model, RMW), retaining the same underlying pretrained architecture.

No “fair” or demographically neutral reward is trained. With pronounced initial G4S man bias, a neutral reward would merely reinforce rather than correct imbalance; deliberately counter-biased P3L, analogous to U8D, actively steers toward gender equilibrium and yields markedly more balanced portrayal. Directional reward tuning serves as a corrective mechanism.

| Models | Average | White | Black | Latino | East Asian | Southeast Asian | India | Middle Eastern |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| G4S | 0.8034 | 0.7925 | 0.7758 | 0.8690 | 0.7115 | 0.7945 | 0.8634 | 0.8071 |
| + Z6B | 0.9584<sub>+0.16</sub> | 0.9595<sub>+0.17</sub> | 0.9524<sub>+0.18</sub> | 0.9756<sub>+0.11</sub> | 0.9437<sub>+0.23</sub> | 0.9447<sub>+0.15</sub> | 0.9640<sub>+0.10</sub> | 0.9709<sub>+0.16</sub> |
| + P3L | 0.3082<sub>-0.50</sub> | 0.3341<sub>-0.46</sub> | 0.3913<sub>-0.38</sub> | 0.3314<sub>-0.54</sub> | 0.1008<sub>-0.61</sub> | 0.2639<sub>-0.53</sub> | 0.3446<sub>-0.52</sub> | 0.3894<sub>-0.42</sub> |

**Table 8.** Aligned-video-model L2V social bias; averages across 42 verbs.

The same preference-optimization framework produces two G4S variants: Z6B alignment raises scores across all demographics and reinforces man preference; P3L substantially lowers them, strongly shifting toward woman preference. Fine-grained modulation can amplify or mitigate diffusion-model social tendencies. The most balanced generator can be framed as a composition problem: systematically varying man-/woman-preferred sample proportions can identify an optimal bias-minimizing mixture preserving aesthetic fidelity. Beyond binary control, nuanced mixtures suggest a continuous alignment axis with tunable bias-mitigation/aesthetic-coherence trade-offs. Dataset composition offers a powerful, interpretable lever beyond architectural intervention alone.

**Figure 6.** Action-level Z6B-/P3L-guided alignment impact: (a) pre-/post-tuning \(\Delta\mathrm{L2V}\), broken down by action and serving as the basis for the two sensitivity panels; (b) man-preferred-alignment sensitive actions; (c) woman-preferred-alignment sensitive actions.

**Action sensitivity.** Actions cluster by guiding reward: Z6B, teal, amplifies male-preferred exercise, row, and cook; P3L, orange, shifts bake, sleep, and sweep toward women. Ranked normalized sensitivity is
\[
\frac{\Delta\mathrm{L2V}}{\text{each reward model's baseline L2V}}.
\]
Sleep, stretch, and read rank highest under both directions. Even semantically/socially neutral or domestic activities undergo pronounced gender shifts, exposing systematic action-specific amplification and event-centric auditing’s sensitivity across gendered dimensions.

**Responsible use.** Technical “debiasing” is no silver bullet; unilateral use carries significant risk. Controllable preference modeling must support, not replace, diverse stakeholder input. Equitable representation is a context-dependent normative decision requiring active community consultation; practitioners should engage impacted communities in setting goals rather than entrust complex social bias solely to automated optimization.

## 8 Conclusion

The identified evaluation blind spots warrant integrating bias auditing and mitigation at every stage of video generation, advancing more equitable, socially aware systems rather than treating preference alignment as independently sufficient.

## Limitations

Predefined gender/ethnicity categories, based on U.S. Census conventions and prior literature, are practical but socially constructed; they miss identity fluidity, intersectionality, and cultural nuance (Yin et al., 2024; Qiu et al., 2024a; Qiu et al., 2025). Richer representations, including intersectional groups, remain future work.

Human-validated VLM judges remain image-level classifiers with potential biases/inaccuracies, especially for stylized or ambiguous frames; multi-model ensembling mitigates without eliminating these issues. Larger, more diverse ground-truth video annotation would strengthen reliability.

Alignment experiments chiefly use single-step latent consistency distillation and limited rewards. RL-based tuning and multi-turn video instruction alignment may have different, uncaptured dynamics.

Synthetic gender-preference manipulation demonstrates targeted modulation but does not resolve broader value alignment, normative appropriateness, or long-term societal impact necessary for responsible deployment.

The fixed socially associated action taxonomy limits possible generalization to open-ended generation and novel actions despite fine control. Holistic, culturally grounded, ethically aligned evaluation remains a research need.

## Broader Impact Statement

Widespread content-creation deployment increases video diffusion’s influence on visual culture and societal representation. Alignment’s stabilization and entrenchment of bias is an important, often overlooked adoption consequence.

**Bias stabilization.** Alignment, including RLHF, is essential for fidelity and temporal consistency but can act as a bias stabilizer. Removing artifacts and temporal flicker makes stereotypes appear natural, coherent, and authoritative. Polished biases are harder to detect as failures than obvious unaligned errors, risking subtle normalization of skewed synthetic-media demographics and reinforcement of harmful stereotypes under high-quality, preferred-content presentation.

**Dual use.** Curated preference data can engineer rewards to amplify or suppress social attributes. Historical underrepresentation correction and bias mitigation share machinery with intentional exclusion, homogenization, or propaganda aligned with exclusionary worldviews; the alignment process is double-edged.

**Responsibility.** Aesthetic metrics alone are insufficient. Developers must continually monitor generated social composition using multi-granular, event-centric audits such as QX2. Preference-data curation warrants architecture-design rigor so guiding human preferences reflect a diverse, equitable society instead of reproducing prejudice.

The supplied supplementary inventory comprises per-verb gender/ethnicity analyses, training-data effects, statistical analyses, and implementation/training details:

| Supplementary item | Contents |
|---|---|
| A | Additional 42-verb video analysis; WebVid-10M training-data effects (Bain et al., 2021); statistical analysis numbered 9. |
| Dataset ethnicity item numbered 10 | Human image-reward-dataset ethnicity preference distributions. |
| C | Reward-model implementation; comprehensive 42-verb analysis; statistical analysis numbered 11. |
| D | Additional 42-verb aligned-video analysis; statistical analysis numbered 12. |
| F | Reward-model training and inference details. |
| G | Video post-training and inference details. |
| Gender plots numbered 7, 8, 12, 10, 9, 11, 13 | Four-model L2V across 42 verbs, by ethnicity. |
| Ethnicity plot numbered 14 | Four-model ethnicity bias across 42 verbs; predominantly White generation without ethnicity specification. |
| Gender plots numbered 19, 20, 21, 22, 23, 24, 25 | Reward-model L2V across 42 verbs, by ethnicity. |
| Ethnicity plots numbered 26, 27, 28, 29 | Reward-model ethnicity bias across 42 verbs. |
| Gender plots numbered 30, 31, 32, 33, 35, 34, 36, 37(b) | Reward-aligned video L2V across 42 verbs, by ethnicity; White-demographic gender analysis includes the item numbered 30. |
| Ethnicity plot numbered 38 | Reward-aligned video ethnicity bias across 42 verbs. |
| Gender plots numbered 39, 40, 41, 42, 43 | Controllable-preference L2V across 42 actions, by ethnicity. |

## Acknowledgment

We thank the anonymous reviewers for thoughtful feedback and suggestions. Partial support: AFOSR MURI grant #FA9550-22-1-0380; Office of Naval Research (ONR), Award #N00014-23-1-2780; National Science Foundation, Award #IIS-2449768. Additional support: NVIDIA Academic Grant Program cloud credits and Coefficient Giving’s Technical AI Safety Research Program funding. Authors’ views and conclusions do not represent official policies or endorsements of AFOSR, ONR, the U.S. Government, the National Science Foundation, NVIDIA, or Coefficient Giving.
