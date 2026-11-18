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

Alignment tuning optimizes a trained video diffusion model to better match human preferences. It produces smoother videos, but also more stereotyped and temporally persistent portrayals. Biases in human preference data enter learned reward models, which score outputs for preference-based optimization. These biases are amplified and propagate into video generation.

QX2, also called VideoBiasEval, separates event verbs and contexts from actor gender and ethnicity. It measures ethnic representation, gender imbalance within ethnic groups, attribute-distribution changes between model variants, and attribute persistence within videos. This is the first end-to-end analysis connecting human comparison data, learned image rewards, and aligned video outputs. Dataset preferences appear in Section 5.1; reward preferences appear in Table 4; downstream demographic changes and temporal persistence appear in Tables 5 and 6.

Rewards trained on human preferences improve visual quality while unintentionally encoding and amplifying social bias. Fair, socially responsible generation requires bias-aware assessment and mitigation throughout alignment.

Reviewed on OpenReview: https://openreview.net/forum?id=C0yxuS6jty

**Figure 1.** Bottom left: QX2’s event prompts and multi-granular ethnicity/gender audit. Top: frame-wise inference by vision–language models (VLMs), which process images and text, followed by aggregation of actor attributes. Bottom right: the first comprehensive analysis separating bias in human preferences, learned image-reward preferences, and downstream diffusion-video generation. It includes the effects of rewards trained on human labels on social-attribute distributions.

## 1 Introduction

High-quality natural-language video generation has advanced substantially (Chen et al., 2024a; Wang et al., 2023a; Yuan et al., 2024; Li et al., 2024). It supports creative media, education, and professional simulation (Cho et al., 2024; Miller et al., 2024). State-of-the-art systems increasingly use alignment to improve fidelity, coherence, and controllability, especially through human-preference learning (Wu et al., 2023; Xu et al., 2024; Li et al., 2024; Yuan et al., 2024; Liu et al., 2024a; Prabhudesai et al., 2024; Black et al., 2023; Ma et al., 2025). Human-labeled comparisons commonly train rewards that judge individual images or video frames and guide optimization after initial training (Wu et al., 2023; Kirstain et al., 2023a; Xu et al., 2024).

Here, **aligned models** are video diffusion models optimized after likelihood-based training to better match human preferences. Likelihood-based training uses the likelihood of training examples as its objective. Alignment commonly uses reward-weighted fine-tuning or reinforcement learning from human feedback (RLHF), guided by learned preference or aesthetic rewards trained on human comparisons. The conceptual sequence is
\[
\text{base video diffusion model}\rightarrow
\text{preference or reward model}\rightarrow
\text{post-training optimization}.
\]
The arrows indicate the sequence from a base generator to reward-guided optimization. **Unaligned counterparts** use likelihood objectives alone. They provide controlled baselines for isolating alignment effects; they are not assumed to be bias-free.

Subjective preferences are often collected without explicit attention to cultural, social, or demographic diversity. Such judgments can encode systematic biases that rewards and downstream generators inherit and reinforce (Qiu et al., 2023). These biases can affect not only visual quality, but also who appears, how often they appear, and which roles they occupy. We therefore examine how alignment changes distributions of social attributes.

The analysis connects three stages: human comparison data, learned image rewards, and aligned video outputs. The two human comparison collections are **R9C, or HPDv2**, and **N6T, or Pick-a-Pic**. Both favor White representations and show non-neutral gender preferences. R9C prefers men on 69.23% (18/26) of valid man–woman-pair verbs; N6T prefers women on 61.11% (11/18). Their White preference rates are 43.34% and 40.08%, respectively (Section 5.1, dataset table).

Learned rewards inherit and amplify these patterns (Table 4). Man- or woman-preferred rewards systematically redirect generated gender portrayals, while ethnic distributions remain uneven (Tables 5 and 8). Curating social-attribute distributions can intentionally control image-reward datasets, learned preferences, and resulting video representations (Tables 7 and 8). Targeted dataset composition and counter-biased rewards therefore offer mitigation guidance. The analysis tracks reward-preference changes over 42 events and corresponding video-bias changes before and after alignment.

This investigation requires a probing method, distributional metrics, and a protocol for comparing models before and after alignment. Existing frameworks inadequately represent diverse identities in prompts, lack comprehensive identity coverage and specific distribution-tracking metrics, and lack a dedicated method for measuring alignment shifts (Huang et al., 2024; Liu et al., 2024b; Sun et al., 2024).

**QX2, or VideoBiasEval**, uses established social-bias taxonomies to separate actor identity from event content. Controlled changes to verbs and identities support robust, interpretable comparisons across contexts (Zhao et al., 2017; Hendricks & Nematzadeh, 2021; Cho et al., 2023; Qiu et al., 2023). We study gender and ethnicity because prior work on generative alignment and fairness gives them comparatively well-defined evaluation boundaries (Luccioni et al., 2023; Shen et al., 2023).

The paper makes three contributions:

- A comprehensive event-prompt audit at multiple levels, from individual frames to output distributions.
- The first connected preference-data–reward–video analysis of bias inheritance, amplification, and stabilization.
- Systematic evidence that perceptual and temporal improvements also reshape, and sometimes reinforce, generated social composition, together with controllable-preference guidance toward equitable behavior.

## 2 Related Work

**T2V evaluation.** Text-to-video (T2V) evaluation frameworks include VBench (Huang et al., 2024), EvalCrafter (Liu et al., 2024b), and T2V-CompBench (Sun et al., 2024). They use measures such as Fréchet Video Distance, a video-distribution comparison measure (Unterthiner et al., 2019); CLIP-Score, an image–text matching score (Hessel et al., 2021); and object consistency. They do not audit depicted identities or their portrayal.

CLIP means Contrastive Language–Image Pre-training. It is designated **Y2F** in this paper and is the shared pre-fine-tuning base of the learned rewards evaluated here (Radford et al., 2021). Y2F-based alignment rewards textual fidelity rather than demographic balance. Likewise, GRiT-based measures, used here as object-presence measures, can verify that a doctor appears without flagging exclusively white male doctors (Wu et al., 2025). Fair, trustworthy T2V evaluation requires explicit auditing of output distributions beyond these surface measures. QX2 provides ethnicity- and gender-aware, event-centric quantification throughout generation.

**Generative-model bias evaluation.** Most image- and language-generation studies examine static portraits or isolated object scenes. StableBias (Luccioni et al., 2023), DALL-Eval (Cho et al., 2023), and SocialCounterfactuals (Howard et al., 2024) chiefly count identities and seldom examine their activities. Audits of independent images can conceal recurring portrayals of authoritative men and supportive women. Omitting joint actor–verb–context analysis misses role-specific stereotypes and narrative or temporal bias. Separating these event components exposes scenario-dependent representation.

## 3 QX2

QX2 generates videos from structured events using state-of-the-art diffusion models. It extracts depicted social attributes, then audits identity consistency and fairness across event categories, frames, videos, and output distributions.

**Figure 2.** Videos from different diffusion models use varied actor-attribute templates. VLM-based extraction records the main character’s gender and ethnicity. Representative generation prompts, classification prompts, and outputs illustrate the analysis.

### 3.1 Event Definition

Socially associated verbs have historical associations with identity stereotypes. Imbalanced gender or ethnic portrayals of people performing these verbs reinforce stereotypes and undermine fairness (Bolukbasi et al., 2016; Sun & Peng, 2021; Zajko, 2021).

An **event** is the three-part description
\[
\langle p,a,c\rangle,
\]
where \(p\) is the actor, \(a\) is the verb the actor performs, and \(c\) is the surrounding context. This formulation measures demographic influence on visual depictions and systematic model bias. It follows studies of socially associated verbs (Zhao et al., 2017; Garg et al., 2018; Cho et al., 2023; Qiu et al., 2023).

Each actor has gender and ethnicity attributes. The bounded prompt space contains four gender categories, seven ethnic groups, and 42 verbs. Established taxonomies and benchmark conventions ground these choices. This space balances coverage against experimental control. It enables reproducible, interpretable tracing of bias through the full pipeline, with clear attribution. Open-ended and intersectional expansion remains promising; the present approach isolates disparities before unconstrained scaling.

The four gender categories, from Luccioni et al. (2023), are **man**, **woman**, neutral **person**, and **non-binary person**. They provide inclusive but incomplete coverage of gender diversity within controlled evaluation boundaries.

The seven ethnic categories, from Karkkainen & Joo (2021) and United States (U.S.) Census Bureau conventions, are **White, Black, Indian, East Asian, Southeast Asian, Middle Eastern, Latino**. These categories are socially constructed. They are neither exhaustive nor universally representative. Tables retain the label **India** for the category called **Indian** in the prose.

The verbs, also called actions or activities, are **bake, bike, call, clean, climb, cook, cough, cry, drink, drive, eat, exercise, fish, hit, jump, kick, kneel, laugh, lift, paint, pick, pitch, pray, read, ride, row, run, shop, shout, sit, skate, sleep, smile, stand, stare, stretch, study, sweep, throw, walk, wash, work**.

Prior literature identifies statistical gender and ethnicity associations for these verbs (Zhao et al., 2017; Garg et al., 2018; Cho et al., 2023; Qiu et al., 2023). They are also visually distinctive and can be depicted in short clips. These properties support reliable annotation and interpretation. The curated, socially meaningful set permits reproducible extension toward more complex, culturally specific, or temporally dynamic events.

### 3.2 Event-Based Prompting Protocol

The fully deterministic prompt template is:

**“A/An [actor] is [verb]-ing [context].”**

Square brackets mark placeholders. `[actor]` supplies the actor description, `[verb]-ing` supplies the activity in its “-ing” form, and `[context]` supplies its surroundings. “A/An” selects the article used before the actor description.

Neither actors nor contexts are generated at runtime by a large language model (LLM). Both use fixed, predefined instantiations. This preserves reproducibility, experimental control, and situational variation without changing verb meanings.

We use two settings:

- **J7K = Person-only.** The actor is always “person.” One prompt per activity retains the event structure but omits explicit demographic cues.
- **V9R = Ethnicity+Person.** A predefined ethnic descriptor is deterministically placed before “person,” as in “An East Asian person.” Every verb is paired with each ethnic group while its context remains unchanged.

| Setting | # of prompts | Examples |
|---|---:|---|
| J7K | 42 | A person is baking cookies in a cozy kitchen, with warm light and the aroma of vanilla filling the air. |
| V9R | 294 | An East Asian person is baking cookies in a cozy kitchen, with warm light and the aroma of vanilla filling the air. |

**Table 1.** Video-generation audit prompts explicitly contain ethnicity when specified, the verb, and surrounding context. “#” means number.

J7K leaves actors underspecified, so the generator implicitly assigns gender and ethnicity. V9R tests interactions between explicit ethnicity and inferred attributes, including gender. Because V9R changes only the ethnic descriptor while preserving verbs and contexts, another ethnicity-only setting is unnecessary for isolating ethnicity effects. This controlled construction attributes observed disparities to demographic conditioning rather than uncontrolled contextual drift.

Identical decoding settings and a fixed number of videos per prompt support systematic diversity comparisons. Demographic analysis occurs afterward, using a VLM ensemble on generated frames. Demographic attributes are not introduced, constrained, or inferred during generation.

### 3.3 Multi-Granular Event-Centric Bias Evaluation

The neutral-person audit uses **parity**, or equal representation, as a diagnostic reference. It is not treated as empirical demographic truth or a universally appropriate norm. Gender proportions can legitimately differ by verb, ethnicity, and sociocultural context. Ethnic proportions can differ by region, culture, and application.

Separating neutral identity descriptions from event content makes deviations probes of learned stereotype reproduction or internal favoring and suppression, rather than comparisons with external population statistics. Uniform and parity baselines follow representational-harm and fairness practice.

**Attribute extraction.** Three open-source VLM judges independently classify each uniformly sampled frame: Qwen2-VL-7B (Wang et al., 2024a), Qwen2.5-VL-7B (Yang et al., 2024), and InternVL2.5-8B (Chen et al., 2024b). These names identify the three classifier models.

We sample 16 frames per video. This matches standard inference defaults for evaluated models, including VideoCrafter. Preliminary reductions in frame count significantly destabilized detection.

The extracted gender label \(g\) belongs to the set \(G\), and the extracted ethnicity label \(e\) belongs to the set \(E\):
\[
g\in G=\{\text{man,woman}\},\qquad
e\in E=\{\text{White,Black,Indian,East Asian,Southeast Asian,Middle Eastern,Latino}\}.
\]
The symbol \(\in\) means “belongs to.” Thus, although the prompt attribute space has four gender categories, extracted gender labels use man and woman.

Within each judge, a majority vote over frames produces a video label. Video labels are then fused across judges. Initial manual inspection found instability and hallucinated attributes from single models. Ensembling distinct architectures, rather than repeatedly sampling one model, cross-validates their outputs, improves robustness, and mitigates model-specific bias (Qiu et al., 2024b). Frame- and video-level representations also assess whether outputs match intended prompt demographics. Human verification of this ensemble appears in Section 4.

**Temporal quantity.** **T8Q = Temporal Attribute Stability (TAS)** is the percentage of frames whose classified attribute matches the video’s final majority-voted label. High values indicate coherent demographic identity. Low values indicate attribute flicker or inconsistent identity, a critical artifact of biased or unstable generation.

**Conditional gender quantity.** **L2V = Proportion Bias Score for Gender (PBSG)** measures signed gender imbalance for each verb and ethnicity:
\[
\mathrm{L2V}=\frac{N_{\text{man}}-N_{\text{woman}}}{N_{\text{total}}}\in[-1,1],
\qquad N_{\text{total}}=N_{\text{man}}+N_{\text{woman}}.
\]
Here, \(N_{\text{man}}\) and \(N_{\text{woman}}\) count outputs classified as men and women. \(N_{\text{total}}\) is their sum. The interval \([-1,1]\) gives the score’s range.

Positive values favor men; negative values favor women; values near zero indicate balance. Computing this quantity under the ethnicity+gender condition probes ethnicity-conditioned verb stereotypes, consistency of balance across ethnic groups, and uneven group-specific amplification.

Gender-only analysis is excluded. Without specified ethnicity, generation predominantly depicts White individuals, so such an analysis would effectively reduce to White-group gender bias.

**Ethnic quantities.** For identifiable outputs, the proportion assigned to ethnic group \(e\) is
\[
P_e=\frac{N_e}{N_{\text{total}}},\quad e\in E.
\]
Here, \(N_e\) counts outputs classified as group \(e\). In this formula, \(N_{\text{total}}\) counts all outputs with identifiable ethnicity.

**F6N = Representation Deviation Score for Ethnicity**, also written \(\mathrm{RDS}_e\), measures each group’s deviation from uniform coverage (Feldman et al., 2015; Mehrabi et al., 2021):
\[
\mathrm{F6N}_e=P_e-\frac1{|E|}.
\]
The subscript \(e\) identifies the ethnic group, and \(|E|\) is the number of groups in \(E\). Positive values mean overrepresentation; negative values mean underrepresentation. The uniform reference applies to neutral J7K auditing, whose prompts intentionally remove contextual and identity cues.

**W3H = Simpson’s Diversity Index (SDI)** measures the probability that two random outputs belong to different ethnic groups (Simpson, 1949):
\[
\mathrm{W3H}=1-\sum_{e\in E}P_e^2.
\]
The sum adds the squared proportions \(P_e^2\) over all groups. Higher values indicate a more diverse or balanced distribution. Lower values indicate concentration in dominant groups.

F6N and W3H answer complementary questions. F6N identifies which groups are over- or underrepresented. W3H describes overall distributional balance. Together, they distinguish homogeneous or skewed portrayals from globally diverse representation.

| Diagnostic illustration | Interpretation |
|---|---|
| \(\mathrm{L2V}=0.5\) | Men occur three times as frequently: \(N_{\text{man}}=3N_{\text{woman}}\). |
| \(\mathrm{L2V}=-0.5\) | Inverse ratio. |
| \(\mathrm{L2V}\approx0\) | Controlled diagnostic target, not real-world ground truth. |
| Seven ethnic groups | Balanced probability \(1/7\approx14.3\%\). |
| \(\mathrm{F6N}_e=0.769\) | \(0.769+0.143\approx91.2\%\) of videos; more than six times expected frequency. |

In this illustration, \(\approx\) means approximately equal.

**Shift quantities.** The symbol \(\Delta\) denotes an alignment-induced change relative to the corresponding baseline. We track \(\Delta\mathrm{L2V}\), \(\Delta\mathrm{F6N}_e\), and \(\Delta\mathrm{W3H}\). More balanced gender ratios, less ethnic skew, or higher diversity constitute improvement. Aggregation across videos complements frame-level dynamics and temporal persistence when assessing how bias forms or is mitigated.

## 4 Social Biases in Video Generative Models

We compare two unaligned/aligned pairs from four state-of-the-art models:

- **D8P = ModelScope**, the first baseline architecture (Wang et al., 2023a).
- **K5X = InstructVideo**, the D8P derivative aligned using **A3J = HPSv2.0**, an image reward trained on R9C/HPDv2 (Yuan et al., 2024; Wu et al., 2023).
- **B7Q = VideoCrafter-2**, also written **VideoCrafter-V2** and **Video-Crafter-V2** (Chen et al., 2024a). The distilled-model analysis also cites Chen et al. (2023).
- **H4M = T2V-Turbo-V1**, whose alignment rewards are **U8D = HPSv2.1**, InternVid2-S2, and ViCLIP (Li et al., 2024; Wu et al., 2023; Wang et al., 2024b; Wang et al., 2023b). InternVid2-S2 and ViCLIP are the additional named reward systems in this alignment setup.

Comparing each architecture’s pair isolates alignment from base architecture. Each prompt is generated 10 times per model with different random seeds. Results are averaged to reduce sampling variance. V9R supplies gender assessment; J7K supplies ethnic-balance assessment.

| Models | Average L2V | White L2V | White F6N | Black L2V | Black F6N | Latino L2V | Latino F6N | East Asian L2V | East Asian F6N | Southeast Asian L2V | Southeast Asian F6N | India L2V | India F6N | Middle Eastern L2V | Middle Eastern F6N | Overall W3H |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| D8P (u) | 0.4815 | 0.5683 | 0.7690 | 0.3912 | -0.1952 | 0.6308 | -0.1952 | 0.4406 | -0.1810 | 0.4611 | — | 0.3938 | — | 0.4833 | -0.1976 | 0.0538 |
| K5X (a) | 0.5295 | 0.5584 | 0.7833 | 0.5114 | -0.1976 | 0.6729 | -0.1929 | 0.4282 | -0.1976 | 0.5020 | — | 0.4878 | — | 0.5393 | -0.1952 | 0.0267 |
| \(\Delta\) | +0.0480 | -0.0099 | +0.0143 | +0.1202 | -0.0024 | +0.0421 | +0.0023 | -0.0124 | -0.0166 | +0.0409 | — | +0.0940 | — | +0.0560 | +0.0024 | -0.0271 |
| B7Q (u) | 0.7581 | 0.7485 | 0.6905 | 0.6167 | -0.1905 | 0.8599 | -0.1952 | 0.6976 | -0.1500 | 0.8272 | — | 0.8032 | — | 0.7560 | -0.1548 | 0.1252 |
| H4M (a) | 0.8306 | 0.8713 | 0.6381 | 0.8095 | — | 0.8599 | -0.2476 | 0.7762 | -0.2426 | 0.8929 | — | 0.7762 | — | 0.7664 | -0.1476 | 0.1119 |
| \(\Delta\) | +0.0725 | +0.1228 | -0.0524 | +0.1928 | — | 0.0000 | -0.0524 | +0.0786 | -0.0926 | +0.0657 | — | -0.0270 | — | +0.0104 | +0.0072 | -0.0133 |

**Table 2.** Two unaligned (**u**)/aligned (**a**) pairs. Entries average across 42 verbs. Nonzero unaligned L2V indicates inherent pre-alignment bias, not bias-free baselines. Metric signs describe absolute model bias, not shifts; \(\Delta\) rows are relative to the corresponding baseline. Positive/negative \(\Delta\mathrm{L2V}\) marks shifts toward men/women. Positive/negative \(\Delta\mathrm{F6N}\) marks shifts toward over-/underrepresentation. Throughout the video-distribution tables, **“—” means zero recognizable instances of the corresponding ethnicity under J7K**. This severe absence of diversity prevents subgroup computation.

**Gender.** Under V9R, every model remains male-biased across all ethnicities: all group-specific model L2V values in Table 2 are positive. Alignment increases average imbalance. D8P rises from 0.4815 to 0.5295 in K5X, a change of +0.0480. B7Q rises from 0.7581 to 0.8306 in H4M, a change of +0.0725. These results suggest preference fine-tuning may worsen disparity.

**Ethnicity.** All models overrepresent White individuals: their White F6N values are 0.7690, 0.7833, 0.6905, and 0.6381 (Table 2, model rows). D8P has the strongest imbalance, and K5X worsens it. B7Q is moderately more balanced. H4M reduces White dominance relative to B7Q, with White F6N falling by -0.0524. Yet overall W3H also falls, from 0.1252 to 0.1119, a change of -0.0133. Thus, reducing ethnic skew can accompany suppressed demographic variety.

| Additional narrative quantities | Value |
|---|---:|
| K5X L2V increase | 0.04 |
| K5X White F6N | 0.783 |
| D8P White F6N | 0.769 |

**Human evaluation.** Three independent researchers experienced in social-bias analysis annotated generated videos. They used standardized written guidelines and task-specific training grounded in prior gender and ethnicity conventions. Verification concerned whether videos correctly reflected prompted demographics, not subjective or normative judgments.

The checks use several agreement measures. **Pearson correlation** measures association between human and ensemble results. **Cohen’s Kappa** measures agreement between two sets of judgments after accounting for chance agreement. **Fleiss’ Kappa** measures chance-adjusted agreement among multiple annotators. The **consensus rate** is the proportion of matching human-majority and VLM judgments. The reported \(p\) value expresses the significance of the stated L2V comparison.

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

The ensemble is reliable, scalable, and human-aligned despite residual ethnicity disagreement. Majority-voted open-source judges align strongly with humans: correlations are 0.89 for gender and 0.73 for ethnicity, and consensus rates are 99.0% and 83.3%. The Kappa values indicate strong to near-perfect inter-annotator consistency.

Many ethnicity discrepancies arise from holistic human judgment versus frame-level precision. During fast motion, a face may be visible only briefly. Humans can miss it or infer its attributes, while the pipeline explicitly captures it. These discrepancies indicate higher recall rather than systematic false positives.

Restricting evaluation to videos with perfect human-majority/VLM agreement leaves all core bias metrics statistically stable. L2V changes from 0.846 to 0.863, with \(p=0.68\); W3H changes from 0.745 to 0.761. The nonsignificant L2V difference indicates that residual evaluator uncertainty does not distort measurements.

Human review grounds and sanity-checks the automated assessment rather than replacing it. Larger review is desirable but labor-intensive. Small, high-consensus validation follows prior model-judge practice.

The remaining investigation separates demographic distributions in image-reward data, biases of learned rewards, and those rewards’ effects on gender and ethnicity in aligned videos.

## 5 Social Biases in Image Reward Datasets and Reward Models

Image-based rewards are the de facto state-of-the-art approach to video alignment (Wu et al., 2023; Xu et al., 2024; Li et al., 2024; Yuan et al., 2024; Liu et al., 2024a; Prabhudesai et al., 2024; Black et al., 2023; Ma et al., 2025). Rewards trained on static images guide temporally coherent video learning. Their inherent biases can transfer and amplify across frames. Tables 4–6 examine learned preferences, downstream representation, and temporal stability.

Human-labeled image preferences and learned rewards must therefore be distinguished when assessing demographic disparities and changes in attributes over time.

### 5.1 Preference in Image Reward Datasets

We analyze two widely used human-comparison collections: R9C/HPDv2 (Wu et al., 2023) and N6T/Pick-a-Pic (Kirstain et al., 2023a; 2023b).

GPT-4o-mini, the caption-analysis model, extracts gender, ethnicity, and verbs from captions. Qwen2-VL-7B, Qwen2.5-VL-7B, and InternVL2.5-8B classify the images. Caption and image attributes are aggregated, and instances containing predefined verbs are retained.

Each tuple contains two images, a caption, and a human choice of which image better matches the caption. Explicit gender-preference analysis uses pairs in which one gender is preferred and the other is dispreferred, including man-versus-woman comparisons. Filtering for valid pairs reduces verb coverage. Gender and ethnicity preferences measure selection frequency for given verbs.

| Dataset | Validated (images, caption, preference) tuples | Covered verbs | Valid man–woman-pair verbs, out of 42 | Directionally preferred verbs | White preference |
|---|---:|---:|---:|---|---:|
| R9C | 28,783 | 29 | 26 | Men: 69.23% (18/26) | 43.34% |
| N6T | 14,958 | 19 | 18 | Women: 61.11% (11/18) | 40.08% |

R9C trends toward men: 18/26 valid verbs, or 69.23%, are man-preferred. N6T is relatively woman-preferred: 11/18, or 61.11%, are woman-preferred. Both strongly favor White representations, at 43.34% and 40.08%, followed by East Asian and Indian representations.

There are verb-specific exceptions, including Black preference for “bake” and East Asian preference for “fish” (Figure 4). These collected preferences may implicitly privilege Western-centric aesthetics and representation. Their imbalances risk reinforcing existing social inequities during reward training. Inclusive data that represent global demographic diversity are urgently needed.

**Figure 3.** Image-reward-dataset gender preference distributions across 42 verbs.

**Figure 4.** Image-reward-dataset ethnicity preference distributions across 42 verbs.

### 5.2 Preference in Image Reward Models

**Benchmark construction.** We construct a controlled text-to-image (T2I) benchmark inspired by R9C and ImageRewardDB, an image-reward dataset (Xu et al., 2024). The benchmark uses the state-of-the-art FLUX image generator (Labs, 2023) and event templates to vary gender, ethnicity, and verbs systematically.

V9R specifies ethnicity and uses “person.” **C4Z = Ethnicity+Gender** explicitly specifies both actor attributes. Both settings generate 100 images per prompt, supporting statistical robustness and a large, diverse evaluation set.

| Settings | # of prompts | Examples |
|---|---:|---|
| V9R | 294 | An East Asian person is baking cookies in a cozy kitchen, with warm light and the aroma of vanilla filling the air. |
| C4Z | 1176 | An East Asian woman is baking cookies in a cozy kitchen, with warm light and the aroma of vanilla filling the air. |

**Table 3.** Image-reward-benchmark prompts explicitly contain ethnicity when specified, verbs, and context.

FLUX is not assumed to be unbiased. Controlled, reproducible demographic conditioning isolates reward preferences independently of generator-specific artifacts.

Three annotators independently reviewed 100 randomly sampled images. Conservatively, unanimous agreement covered 77%. Majority confirmation that attributes were correct covered 97%, and only three images lacked consensus. Disagreements concerned visually ambiguous prompts rather than systematic failures or demographic-specific distortions. Clear gender and ethnicity cues and high signal-to-noise support reliable reward processing. Residual noise is unlikely to materially affect preference patterns.

**Figure 5.** Constructed-benchmark images use **“A/An [ethnicity][gender] is baking [context].”** Only \([\mathrm{gender}]\in\{\text{man,woman}\}\) appears.

**Reward identities and scoring.** Four models are evaluated:

- **Y2F = CLIP**, the shared pre-fine-tuning base of the three learned rewards (Radford et al., 2021).
- **A3J = HPSv2.0**, trained on R9C/HPDv2 (Wu et al., 2023).
- **U8D = HPSv2.1**, the later reward version, also cited to Wu et al. (2023). Its unreleased training collection is designated **X7H = HPDv2.1**.
- **E5W = PickScore**, trained on N6T/Pick-a-Pic (Kirstain et al., 2023b).

The HPS versions provide versioned human-preference scores. All predictions are continuous scalar preference scores—single numerical ratings—not discrete labels. Standardizing these scores supports controlled comparisons of relative demographic preferences.

For **gender evaluation**, the reward receives:

**“A/An [ethnicity] person is [verb]-ing [context]”**.

This omits gender. Image generation instead uses:

**“A/An [ethnicity] [gender] is [verb]-ing [context]”**.

The generation prompt explicitly fixes ethnicity, gender, and verb. Scores are standardized using their mean and standard deviation, which describe their average and spread. Standardized scores are then averaged over images for each generation prompt. At fixed ethnicity and verb, subtracting women’s average standardized score from men’s gives an **adapted reward-model L2V**. Unlike count-based video L2V, it can exceed one. Its sign still indicates preference for men or women.

For **ethnicity evaluation**, the reward receives:

**“A person is [verb]-ing [context]”**.

This omits ethnicity. Generation instead uses:

**“A/An [ethnicity] person is [verb]-ing [context]”**.

Scores are standardized using their mean and standard deviation, then averaged by generation prompt. At each fixed verb, **softmax normalization** converts the ethnic groups’ scores into a probability distribution by exponentiating scores and normalizing them to sum to one (Bridle, 1990; Bishop, 2006). These distributions supply F6N and W3H.

| Models | Average L2V | White L2V | White F6N | Black L2V | Black F6N | Latino L2V | Latino F6N | East Asian L2V | East Asian F6N | Southeast Asian L2V | Southeast Asian F6N | India L2V | India F6N | Middle Eastern L2V | Middle Eastern F6N | Overall W3H |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Y2F | -0.0726 | 0.0343 | 0.0182 | -0.1198 | 0.0002 | -0.0934 | -0.0013 | -0.1315 | 0.0141 | -0.0865 | 0.0094 | -0.0508 | -0.0299 | -0.0607 | -0.0108 | 0.8495 |
| A3J | 0.6039 | 0.6090 | -0.0423 | 0.7341 | -0.0069 | 0.6512 | 0.0237 | 0.4752 | -0.0031 | 0.5192 | -0.0100 | 0.5922 | 0.0070 | 0.6464 | 0.0315 | 0.8492 |
| \(\Delta\) | +0.6765 | +0.5747 | -0.0605 | +0.8539 | -0.0071 | +0.7446 | +0.0250 | +0.6067 | -0.0172 | +0.6057 | -0.0194 | +0.6430 | +0.0369 | +0.7071 | +0.0423 | -0.0003 |
| U8D | -0.0984 | -0.0833 | -0.0189 | 0.0257 | -0.0321 | -0.0031 | 0.0382 | -0.3044 | 0.0091 | -0.2181 | -0.0099 | -0.0006 | -0.0077 | -0.1053 | 0.0214 | 0.8470 |
| \(\Delta\) | -0.0258 | -0.1176 | -0.0371 | +0.1455 | -0.0323 | +0.0903 | +0.0395 | -0.1729 | -0.0050 | -0.1316 | -0.0193 | +0.0502 | +0.0222 | -0.0446 | +0.0322 | -0.0025 |
| E5W | -0.1157 | 0.0321 | 0.0069 | -0.0777 | 0.0279 | -0.3479 | -0.0118 | -0.2257 | 0.0316 | -0.2163 | 0.0115 | 0.1531 | -0.0391 | -0.1277 | -0.0271 | 0.8483 |
| \(\Delta\) | -0.0431 | -0.0022 | -0.0113 | +0.0421 | +0.0277 | -0.2545 | -0.0105 | -0.0942 | +0.0175 | -0.1298 | +0.0021 | +0.2039 | -0.0092 | -0.0670 | -0.0163 | -0.0012 |

**Table 4.** Reward-model preference bias; averages across 42 verbs. Each \(\Delta\) row compares the preceding learned reward with Y2F.

Y2F slightly favors women on average, with L2V -0.0726, and White individuals, with F6N 0.0182. Its ethnic distribution is relatively balanced. It has the highest W3H, 0.8495 (Table 4, Y2F row).

R9C fine-tuning produces strong A3J preference for men across ethnicities. Average L2V rises to 0.6039, a change of +0.6765, and every A3J group-specific L2V is positive. U8D and E5W favor women, with average L2V values of -0.0984 and -0.1157. The shifts accord with training-data preferences and reveal consistent ethnicity-spanning gender preferences. U8D’s training data remain undisclosed. Individual group values, including positive values, are retained in Table 4.

Fine-tuned ethnic preferences shift toward Middle Eastern representations in A3J, Latino representations in U8D, and East Asian representations in E5W. Their corresponding Table 4 F6N values are 0.0315, 0.0382, and 0.0316. All fine-tuned rewards lose diversity relative to Y2F: their W3H changes are -0.0003, -0.0025, and -0.0012.

| Additional narrative quantity | Value |
|---|---:|
| E5W East Asian F6N | 0.0352 |

## 6 Social Biases in Preference Alignment

The baseline for reward-specific tuning is **G4S = Video Consistency Model distilled from VideoCrafter-V2**, also called **VCM-VC2** (Li et al., 2024). A3J, U8D, and E5W provide three separate image-reward conditions.

The H4M training protocol incorporates reward feedback into state-of-the-art **Latent Consistency Distillation**, a teacher-to-student distillation procedure used here for single-step video generation (Luo et al., 2023). A student is distilled from a pretrained T2V teacher. Its decoded frames are directly optimized to maximize image–text alignment rewards. This guides each frame toward human-preferred representations. QX2 comparisons before and after tuning measure reward-specific identity shifts.

| Models | Average L2V | White L2V | White F6N | Black L2V | Black F6N | Latino L2V | Latino F6N | East Asian L2V | East Asian F6N | Southeast Asian L2V | Southeast Asian F6N | India L2V | India F6N | Middle Eastern L2V | Middle Eastern F6N | Overall W3H |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| G4S | 0.8034 | 0.7925 | 0.6405 | 0.7758 | -0.2381 | 0.8090 | — | 0.7115 | -0.2333 | 0.7945 | — | 0.8634 | — | 0.8071 | -0.1690 | 0.1433 |
| + A3J | 0.9116 | 0.9667 | 0.3667 | 0.9214 | — | 0.9667 | — | 0.8500 | — | 0.9214 | — | 0.9000 | — | 0.8548 | -0.3667 | 0.1257 |
| \(\Delta\) | +0.1082 | +0.1742 | -0.2738 | +0.1456 | — | +0.1577 | — | +0.1385 | — | +0.1269 | — | +0.0366 | — | +0.0477 | -0.1977 | -0.0176 |
| + U8D | 0.2267 | 0.1321 | 0.4286 | 0.2381 | — | 0.3738 | — | 0.1571 | — | 0.3619 | — | 0.1452 | — | 0.1786 | -0.4286 | 0.0976 |
| \(\Delta\) | -0.5767 | -0.6604 | -0.2119 | -0.5377 | — | -0.4352 | — | -0.5544 | — | -0.4326 | — | -0.7182 | — | -0.6285 | -0.2596 | -0.0457 |
| + E5W | 0.3714 | 0.3429 | 0.6833 | 0.3357 | -0.1810 | 0.7190 | -0.1929 | 0.1450 | -0.1952 | 0.4548 | — | 0.2500 | — | 0.3515 | -0.1143 | 0.1467 |
| \(\Delta\) | -0.4320 | -0.4496 | +0.0428 | -0.4401 | +0.0571 | -0.0900 | -0.1929 | -0.5665 | +0.0381 | -0.3397 | — | -0.6134 | — | -0.4556 | +0.0547 | +0.0034 |

**Table 5.** Aligned-model social bias; averages across 42 verbs. “+” identifies the reward used to tune G4S. Changes are relative to G4S. Missingness follows the J7K zero-recognizable-instance convention.

**Gender.** V9R evaluation finds strong base-model preference for men across ethnicities. G4S has average L2V 0.8034. A3J increases it to 0.9116, a change of +0.1082, with positive shifts in every ethnic group (Table 5).

U8D and E5W significantly reduce this bias toward balanced or woman-preferred outputs. Their average L2V values fall to 0.2267 and 0.3714, changes of -0.5767 and -0.4320. This direction reflects their reward-level preference for women and counteracts the baseline’s man dominance. Their final aggregate and group-specific L2V values in Table 5 remain positive.

**Ethnicity.** Under the ethnicity-only condition, the baseline strongly favors White individuals and underrepresents Black, East Asian, and Middle Eastern groups. G4S has F6N values of 0.6405, -0.2381, -0.2333, and -0.1690 for these groups, respectively (Table 5, G4S row).

U8D improves White/Black balance but significantly decreases Latino representation and diversity. White F6N falls from 0.6405 to 0.4286; the U8D row records “—” for Black and Latino F6N. W3H falls from 0.1433 to 0.0976, a change of -0.0457.

E5W provides the highest diversity, more balanced representation across most groups, and the most demographically equitable outputs. Its W3H is 0.1467, compared with 0.1433 for G4S, 0.1257 for A3J, and 0.0976 for U8D. Its Black, East Asian, and Middle Eastern F6N values move toward zero; all values and changes appear in Table 5.

| Additional narrative quantity | Value |
|---|---:|
| U8D-aligned Latino representation F6N | –0.4352 |

Temporal results use another reward-version label, **M1Y = HPSv2**. This label is retained distinctly in the temporal table. All reported temporal changes are relative to G4S.

In Table 6, **mean** is the average T8Q, **median** is its middle value, and **Std** means standard deviation, which measures its spread. **Perfect stability** is the frequency of videos whose attribute remains fully stable.

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

**Table 6.** Actor-identity temporal consistency. High scores indicate stable representation; low scores indicate flicker or change. Red and blue subscripts indicate relative improvements and degradations against G4S.

**Temporal behavior.** Alignment substantially improves technical coherence. Every aligned mean T8Q increases, and every listed standard deviation decreases (Table 6). U8D raises mean ethnicity stability from 0.8904 to 0.9632 and perfect-stability frequency from 0.5670 to 0.8330. Gender stability nears saturation: aligned means are 0.9967 for M1Y, 0.9895 for U8D, and 0.9884 for E5W. Lower standard deviations indicate more uniform frame consistency.

Biased rewards can entrench rather than mitigate disparity. A3J amplifies preference for men while locking it temporally, with nearly all videos perfectly gender-stable. Table 5 records A3J’s average L2V increase of +0.1082. The temporal table’s distinct M1Y row records mean gender T8Q 0.9967 and perfect stability 0.9830.

Smoother, more coherent stereotyping makes bias persistent and deeply embedded. QX2 uniquely exposes alignment’s conversion of representational artifacts into temporally resilient systematic distortions.

Reward profiles largely determine aligned bias direction and magnitude, as shown by the reward preferences in Table 4 and video shifts in Table 5. A3J reinforces male- and White-preferred base tendencies. Its Table 5 White F6N remains positive at 0.3667, with a reported change of -0.2738. U8D and E5W counteract man dominance without achieving complete demographic parity, and ethnic disparities remain.

## 7 Controllable Preference Modeling for Video Diffusion Models

Gender provides a case study of deliberately calibrated or more equitable portrayal through the social-attribute composition of image preferences (Sheng et al., 2020). Varying this composition tests the direction of reward bias and the resulting video-alignment dynamics.

### 7.1 Image Reward Dataset Construction

We construct two collections, each containing 2.94 million preference pairs from C4Z. Each pair fixes action and ethnicity while changing gender. The paired male- and female-image examples include those labeled M-1 and W-1. The image source is identified with the human-preference-dataset analysis.

The reward prompts are:

**“A/An [ethnicity] person is [action]-ing [context].”**

Here, `[action]` is the event verb. In the man-preferred collection, male images receive preference label 1 and female images receive label 0. The woman-preferred collection reverses these labels. An additional 537,660 R9C image pairs enhance face-free diversity.

Applied to a man-biased base model, the woman-preferred data correct imbalance and promote equitable gender representation. Table 8 reports the resulting video-model shifts.

### 7.2 Image Reward Model Development & Alignment Tuning

The two datasets fine-tune pretrained Y2F **vision encoders**, the components that represent images, into two reward systems:

- **Z6B = Man-Preferred Reward Model (RMM)**.
- **P3L = Woman-Preferred Reward Model (RMW)**.

Each encoder is optimized for its dataset’s preferences. Both retain the same underlying pretrained architecture. Table 7 uses the adapted reward-model L2V, which can exceed one.

| Models | Average | White | Black | Latino | East Asian | Southeast Asian | India | Middle Eastern |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Y2F | -0.0726 | 0.0343 | -0.1198 | -0.0934 | -0.1315 | -0.0865 | -0.0508 | -0.0607 |
| Z6B | 1.5280<sub>+1.60</sub> | 1.6300<sub>+1.60</sub> | 1.5752<sub>+1.70</sub> | 1.5524<sub>+1.65</sub> | 1.4323<sub>+1.56</sub> | 1.4525<sub>+1.54</sub> | 1.5619<sub>+1.61</sub> | 1.4914<sub>+1.55</sub> |
| P3L | -0.7448<sub>-2.27</sub> | -0.6318<sub>-2.26</sub> | -0.7943<sub>-2.37</sub> | -0.8279<sub>-2.38</sub> | -0.6282<sub>-2.06</sub> | -0.6429<sub>-2.10</sub> | -0.8846<sub>-2.45</sub> | -0.8042<sub>-2.30</sub> |

**Table 7.** Reward-model L2V preference bias; averages across 42 verbs. Subscripts retain the reported changes.

Z6B consistently raises gender-preference scores across demographic groups. Its average L2V is 1.5280, and every group-specific value is positive. P3L systematically favors women: its average is -0.7448, and every group-specific value is negative. Reward tuning thus captures and amplifies gender-specific aesthetic and representational preferences.

No “fair” or demographically neutral reward is trained. G4S initially has pronounced preference for men, with average L2V 0.8034. A neutral reward would merely reinforce rather than correct this imbalance. Deliberately counter-biased P3L, analogous to U8D, actively steers toward gender equilibrium and yields markedly more balanced portrayals. Directional reward tuning serves as a corrective mechanism.

The same preference-optimization framework produces two G4S variants, one guided by Z6B and the other by P3L.

| Models | Average | White | Black | Latino | East Asian | Southeast Asian | India | Middle Eastern |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| G4S | 0.8034 | 0.7925 | 0.7758 | 0.8690 | 0.7115 | 0.7945 | 0.8634 | 0.8071 |
| + Z6B | 0.9584<sub>+0.16</sub> | 0.9595<sub>+0.17</sub> | 0.9524<sub>+0.18</sub> | 0.9756<sub>+0.11</sub> | 0.9437<sub>+0.23</sub> | 0.9447<sub>+0.15</sub> | 0.9640<sub>+0.10</sub> | 0.9709<sub>+0.16</sub> |
| + P3L | 0.3082<sub>-0.50</sub> | 0.3341<sub>-0.46</sub> | 0.3913<sub>-0.38</sub> | 0.3314<sub>-0.54</sub> | 0.1008<sub>-0.61</sub> | 0.2639<sub>-0.53</sub> | 0.3446<sub>-0.52</sub> | 0.3894<sub>-0.42</sub> |

**Table 8.** Aligned-video-model L2V social bias; averages across 42 verbs. Subscripts report changes relative to G4S.

Z6B alignment raises scores across all demographic groups and reinforces preference for men. Average L2V rises from 0.8034 to 0.9584. P3L substantially lowers scores, strongly shifting toward preference for women. Its average falls to 0.3082, and all group-specific values decrease (Table 8). Its final values remain positive.

These opposing changes show that fine-grained preference modulation can amplify or mitigate diffusion-model social tendencies. The most balanced generator can be framed as a composition problem: systematically varying the proportions of man- and woman-preferred samples can identify a bias-minimizing mixture that preserves aesthetic fidelity. Beyond binary control, nuanced mixtures suggest a continuous alignment axis with tunable trade-offs between bias mitigation and aesthetic coherence. Dataset composition offers a powerful, interpretable lever beyond architectural intervention alone.

**Figure 6.** Action-level effects of Z6B- and P3L-guided alignment: (a) pre-/post-tuning \(\Delta\mathrm{L2V}\), broken down by action and used as the basis for the two sensitivity panels; (b) actions sensitive to man-preferred alignment; (c) actions sensitive to woman-preferred alignment.

**Action sensitivity.** Actions cluster by guiding reward. Z6B, shown in teal, amplifies male-preferred exercise, row, and cook. P3L, shown in orange, shifts bake, sleep, and sweep toward women (Figure 6).

Actions are ranked using normalized sensitivity:
\[
\frac{\Delta\mathrm{L2V}}{\text{each reward model's baseline L2V}}.
\]
This divides the action’s L2V change by the corresponding reward model’s baseline L2V. Sleep, stretch, and read rank highest under both directions. Even semantically or socially neutral activities and domestic activities undergo pronounced gender shifts. These results expose systematic action-specific amplification and demonstrate event-centric auditing’s sensitivity across gendered dimensions.

**Responsible use.** Technical “debiasing” is no silver bullet. Unilateral use carries significant risk. Controllable preference modeling must support, not replace, diverse stakeholder input. Equitable representation is a context-dependent normative decision requiring active community consultation. Practitioners should engage impacted communities in setting goals rather than entrust complex social bias solely to automated optimization.

## 8 Conclusion

The identified evaluation blind spots warrant bias auditing and mitigation at every stage of video generation. This approach advances more equitable, socially aware systems rather than treating preference alignment as independently sufficient. The dataset, reward, video-distribution, and temporal results show why: alignment can redirect representation, leave disparities unresolved, and stabilize biased portrayals.

## Limitations

The predefined gender and ethnicity categories are based on U.S. Census conventions and prior literature. They are practical but socially constructed. They miss identity fluidity, intersectionality, and cultural nuance (Yin et al., 2024; Qiu et al., 2024a; Qiu et al., 2025). Richer representations, including intersectional groups, remain future work.

Human-validated VLM judges remain image-level classifiers with potential biases and inaccuracies, especially for stylized or ambiguous frames. Multi-model ensembling mitigates these issues but does not eliminate them. Larger and more diverse ground-truth video annotations would strengthen reliability.

Alignment experiments chiefly use single-step latent consistency distillation and a limited set of rewards. Reinforcement-learning-based tuning and multi-turn video instruction alignment may have different dynamics that this study does not capture.

Synthetic gender-preference manipulation demonstrates targeted modulation. It does not resolve broader value alignment, normative appropriateness, or long-term societal impact, all of which matter for responsible deployment.

The fixed taxonomy of socially associated actions provides fine control but limits possible generalization to open-ended generation and novel actions. Holistic, culturally grounded, ethically aligned evaluation remains a research need.

## Broader Impact Statement

Widespread deployment for content creation increases video diffusion’s influence on visual culture and societal representation. Alignment’s stabilization and entrenchment of bias is an important, often overlooked consequence of adoption.

**Bias stabilization.** Alignment, including RLHF, is essential for fidelity and temporal consistency, but it can also act as a bias stabilizer. Removing artifacts and temporal flicker makes stereotypes appear natural, coherent, and authoritative. Polished biases are harder to detect as failures than obvious unaligned errors. This risks subtly normalizing skewed synthetic-media demographics and reinforcing harmful stereotypes through high-quality content presented as preferred.

**Dual use.** Curated preference data can engineer rewards to amplify or suppress social attributes. Correcting historical underrepresentation and mitigating bias use the same machinery as intentional exclusion, homogenization, or propaganda aligned with exclusionary worldviews. The alignment process is therefore double-edged.

**Responsibility.** Aesthetic metrics alone are insufficient. Developers must continually monitor generated social composition using multi-granular, event-centric audits such as QX2. Preference-data curation warrants the same rigor as architecture design. Guiding human preferences should reflect a diverse, equitable society rather than reproduce prejudice.

The supplied supplementary inventory comprises per-verb gender and ethnicity analyses, training-data effects, statistical analyses, and implementation and training details. The letters and numbers below identify supplementary items and plots. WebVid-10M is the named training dataset whose effects are analyzed in item A.

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

We thank the anonymous reviewers for thoughtful feedback and suggestions. Partial support came from the Air Force Office of Scientific Research (AFOSR), through the Multidisciplinary University Research Initiative (MURI), grant #FA9550-22-1-0380; the Office of Naval Research (ONR), Award #N00014-23-1-2780; and the National Science Foundation, Award #IIS-2449768. These identifiers are the supporting grant and award numbers. Additional support came from NVIDIA Academic Grant Program cloud credits and Coefficient Giving’s Technical AI Safety Research Program funding, where AI means artificial intelligence. Authors’ views and conclusions do not represent official policies or endorsements of AFOSR, ONR, the U.S. Government, the National Science Foundation, NVIDIA, or Coefficient Giving.
