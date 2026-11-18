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

Alignment tuning adjusts a model to better match human preferences. In video diffusion models, which generate videos through a diffusion process, alignment produces smoother portrayals but also makes them more stereotyped and persistent across frames. Biases in preference data enter reward models, which score outputs according to learned preferences. These models amplify the biases and pass them into video generation.

QX2, also called VideoBiasEval, separates an event’s verb and context from the actor’s gender and ethnicity. It measures ethnic skew, gender skew within ethnic groups, changes in attribute distributions across model variants, and attribute persistence within videos.

This first end-to-end analysis connects human comparison data, learned image rewards, and aligned video outputs. Rewards trained on human preferences improve visual quality while unintentionally encoding and amplifying social bias. Fair, socially responsible generation requires bias-aware assessment and mitigation throughout alignment.

Reviewed on OpenReview: https://openreview.net/forum?id=C0yxuS6jty

**Figure 1.** Bottom left: QX2’s event prompts and ethnicity/gender audit at multiple levels of detail. Top: a vision-language model (VLM), which processes images and text, infers actor attributes frame by frame; the pipeline then combines these results. Bottom right: the first comprehensive analysis that separates human preference bias, image-reward preference bias, and downstream diffusion-video generation bias. The analysis includes the effects of human-labeled rewards on social-attribute distributions.

## 1 Introduction

Both collections of human comparisons favor White representations and show non-neutral gender preferences. Learned rewards inherit and amplify these patterns. Rewards that prefer men or women systematically redirect gender portrayals, while ethnic distributions remain uneven.

Curating attribute distributions can deliberately control image-reward datasets, learned preferences, and the resulting video representations. Targeted dataset composition and counter-biased rewards offer guidance for mitigation. Our analysis covers changes in reward preferences across 42 events and the corresponding changes in video bias before and after alignment.

High-quality video generation from natural-language prompts has advanced substantially (Chen et al., 2024a; Wang et al., 2023a; Yuan et al., 2024; Li et al., 2024). It supports creative media, education, and professional simulation (Cho et al., 2024; Miller et al., 2024). Leading systems increasingly use alignment to improve fidelity, coherence, and controllability. They rely especially on learning from human preferences (Wu et al., 2023; Xu et al., 2024; Li et al., 2024; Yuan et al., 2024; Liu et al., 2024a; Prabhudesai et al., 2024; Black et al., 2023; Ma et al., 2025). Human-labeled comparisons commonly train rewards that judge frames or images and guide optimization after initial training (Wu et al., 2023; Kirstain et al., 2023a; Xu et al., 2024).

We use **aligned models** to mean video diffusion models optimized after likelihood-based training to better match human preferences. Likelihood-based training teaches models to fit the training data. Alignment commonly uses reward-weighted fine-tuning or reinforcement learning from human feedback (RLHF), guided by learned preference or aesthetic rewards trained on human comparisons. The conceptual sequence is
\[
\text{base video diffusion model}\rightarrow
\text{preference or reward model}\rightarrow
\text{post-training optimization}.
\]
Unaligned counterparts use likelihood objectives alone. They provide controlled baselines that isolate alignment’s effects.

Researchers often collect subjective preference judgments without explicitly considering cultural, social, or demographic diversity. These judgments can encode systematic bias that rewards and downstream generators inherit and reinforce (Qiu et al., 2023). This bias can affect not only generation quality, but also who appears, how often they appear, and which roles they fill. We ask how alignment influences social-attribute distributions.

Answering this question requires a probing method, distributional metrics, and a protocol for analysis before and after alignment. Existing frameworks inadequately represent diverse identities in prompts (Huang et al., 2024; Liu et al., 2024b; Sun et al., 2024). They lack comprehensive identity coverage, specific metrics for tracking distributions, and a dedicated method for measuring alignment-induced shifts.

QX2 uses established social-bias taxonomies to separate actor identity from semantic content in a controlled way. Precise changes to verbs and identities support robust, interpretable assessment across contexts (Zhao et al., 2017; Hendricks & Nematzadeh, 2021; Cho et al., 2023; Qiu et al., 2023). Prior work on generative alignment and fairness gives gender and ethnicity comparatively well-defined evaluation boundaries (Luccioni et al., 2023; Shen et al., 2023).

We make three contributions:

- A comprehensive audit that uses event prompts and measures bias at multiple levels of detail.
- The first connected analysis of how bias passes from preference data through rewards into video, where it is inherited, amplified, and stabilized.
- Systematic evidence that improvements in perception and temporal coherence also reshape—and sometimes reinforce—the social composition of generated videos, along with guidance for controlling preferences toward more equitable behavior.

## 2 Related Work

**T2V evaluation.** Text-to-video (T2V) evaluation frameworks include VBench (Huang et al., 2024), EvalCrafter (Liu et al., 2024b), and T2V-CompBench (Sun et al., 2024). They use Fréchet Video Distance, a measure for comparing video distributions (Unterthiner et al., 2019); CLIP-Score, a measure of image–text matching (Hessel et al., 2021); and object consistency. They do not audit depicted identities or how models portray them.

Measures based on GRiT (Wu et al., 2025) can verify that a doctor appears without flagging that all depicted doctors are white men. Alignment based on Y2F rewards fidelity to text rather than demographic balance. Fair, trustworthy T2V evaluation therefore requires explicit audits of output distributions beyond surface measures. QX2 provides event-centered, ethnicity- and gender-aware measurements throughout generation.

**Generative-model bias evaluation.** Most studies of bias in image or language generation examine static portraits or isolated object scenes. StableBias (Luccioni et al., 2023), DALL-Eval (Cho et al., 2023), and SocialCounterfactuals (Howard et al., 2024) mainly count identities and seldom examine their activities.

Demographic audits of independent images hide recurring portrayals of authoritative men and supportive women. Without jointly analyzing actors, verbs, and contexts, audits miss role-specific stereotypes and bias in narratives or over time. Separating these event components reveals how representation depends on the scenario.

## 3 QX2

QX2 generates videos from structured events with leading diffusion models. It then extracts social representations and audits identity consistency and fairness across event categories, over time, and at different levels of representation.

**Figure 2.** Videos from different diffusion models use varied actor-attribute templates. VLM-based extraction records the main character’s gender and ethnicity. Representative prompts, classification prompts, and outputs illustrate the analysis.

### 3.1 Event Definition

Some verbs have historical associations with identity stereotypes. Imbalanced gender or ethnic portrayals of people performing these actions reinforce stereotypes and undermine fairness (Bolukbasi et al., 2016; Sun & Peng, 2021; Zajko, 2021).

We define an event as
\[
\langle p,a,c\rangle,
\]
where \(p\) is the actor, \(a\) is the performed verb, and \(c\) is the context. This formulation measures demographic influences on visual depictions and systematic model bias. It follows studies of socially associated verbs (Zhao et al., 2017; Garg et al., 2018; Cho et al., 2023; Qiu et al., 2023).

We bound the attribute space to balance coverage and experimental control. Four gender categories, seven ethnic groups, and 42 verbs draw on established taxonomies and benchmark conventions. These choices enable reproducible, interpretable tracing of bias propagation from beginning to end, with clear attribution. Expanding to open-ended and intersectional identities remains promising. We first isolate disparities before scaling to unconstrained settings.

Each actor has gender and ethnicity attributes. Following Luccioni et al. (2023), we use **man**, **woman**, neutral **person**, and **non-binary person**. These categories provide inclusive but incomplete coverage of gender diversity within controlled evaluation boundaries.

Following Karkkainen & Joo (2021) and U.S. Census Bureau conventions, we use **White, Black, Indian, East Asian, Southeast Asian, Middle Eastern, Latino**. These socially constructed categories are neither exhaustive nor universally representative.

The verbs are **bake, bike, call, clean, climb, cook, cough, cry, drink, drive, eat, exercise, fish, hit, jump, kick, kneel, laugh, lift, paint, pick, pitch, pray, read, ride, row, run, shop, shout, sit, skate, sleep, smile, stand, stare, stretch, study, sweep, throw, walk, wash, work**.

Prior literature identifies statistical gender and ethnicity associations for these verbs (Zhao et al., 2017; Garg et al., 2018; Cho et al., 2023; Qiu et al., 2023). The actions are also visually distinct and can appear in short clips. These properties support reliable annotation and interpretation. This curated, socially meaningful set allows reproducible extension to complex, culturally specific, or temporally changing events.

### 3.2 Event-Based Prompting Protocol

| Setting | Number of prompts | Examples |
|---|---:|---|
| J7K | 42 | A person is baking cookies in a cozy kitchen, with warm light and the aroma of vanilla filling the air. |
| V9R | 294 | An East Asian person is baking cookies in a cozy kitchen, with warm light and the aroma of vanilla filling the air. |

**Table 1.** Video-generation audit prompts explicitly include ethnicity when specified, a verb, and surrounding context.

In J7K, models implicitly assign gender and ethnicity to underspecified actors. V9R tests interactions between explicit ethnicity and inferred attributes, including gender. We use identical decoding procedures and a fixed number of videos per prompt to compare diversity systematically. After generation, a VLM ensemble—a group of models whose judgments are combined—analyzes demographics in the generated frames. The generation process does not introduce, constrain, or infer demographic attributes.

The fully deterministic template is **“A/An [actor] is [verb]-ing [context].”** Neither the actor nor the context comes from a large language model (LLM) at runtime. Fixed, predefined versions preserve reproducibility, control, and situational variation without changing verb meanings.

Fixing the actor to “person” produces one prompt for each curated activity. This setting, **J7K = Person-only**, keeps the same semantic structure but omits explicit demographic cues.

For **V9R = Ethnicity+Person**, we deterministically add a predefined ethnic descriptor before “person,” as in “An East Asian person.” We combine the verbs with each ethnic group while preserving their contexts. This design makes an additional ethnicity-only setting unnecessary for isolating ethnicity effects. It attributes observed disparities to demographic conditioning rather than uncontrolled changes in context.

### 3.3 Multi-Granular Event-Centric Bias Evaluation

A neutral-person audit uses parity as a diagnostic reference, not as empirical demographic truth or a universally appropriate norm. Gender proportions can legitimately vary by verb, ethnicity, and sociocultural context. Ethnic proportions can vary by region, culture, and application.

Separating neutral identity from content lets us probe whether a model reproduces learned stereotypes or internally favors or suppresses groups. It does not compare outputs with external demographic statistics. Uniform and parity baselines follow practices for assessing representational harm and fairness.

**Attribute extraction.** Three open-source judges independently classify each uniformly sampled frame: Qwen2-VL-7B (Wang et al., 2024a), Qwen2.5-VL-7B (Yang et al., 2024), and InternVL2.5-8B (Chen et al., 2024b).

We use 16 frames per video. This matches standard inference defaults for the evaluated models, including VideoCrafter. Preliminary reductions in frame count significantly destabilized detection. The output categories are
\[
g\in G=\{\text{man,woman}\},\qquad
e\in E=\{\text{White,Black,Indian,East Asian,Southeast Asian,Middle Eastern,Latino}\}.
\]
Here, \(g\) denotes gender, \(G\) is the set of gender labels, \(e\) denotes ethnicity, and \(E\) is the set of ethnicity labels.

Each judge first assigns a video label through a majority vote over frames. We then combine the labels across models. Initial manual inspection found unstable single-model judgments and hallucinated attributes—attributes that the model reported without support. Combining distinct model architectures, rather than repeatedly sampling one model, cross-validates outputs, improves robustness, and reduces model-specific bias (Qiu et al., 2024b). Frame- and video-level representations also assess whether outputs match the demographics intended by the prompt.

**Temporal quantity.** **T8Q = Temporal Attribute Stability (TAS)** measures the percentage of frames whose classified attribute matches the video’s final majority-voted label. High values indicate a coherent demographic identity. Low values indicate attribute flicker or inconsistent identity, a critical artifact of biased or unstable generation.

**Conditional gender quantity.** For each verb and ethnicity, representation counts define **L2V = Proportion Bias Score for Gender (PBSG)**:
\[
\mathrm{L2V}=\frac{N_{\text{man}}-N_{\text{woman}}}{N_{\text{total}}}\in[-1,1],
\qquad N_{\text{total}}=N_{\text{man}}+N_{\text{woman}}.
\]
Here, \(N_{\text{man}}\) and \(N_{\text{woman}}\) count outputs classified as men and women. Positive values favor men, negative values favor women, and values near zero indicate balance.

Computing this score under the ethnicity+gender condition probes verb stereotypes within ethnic groups. It also tests whether balance remains consistent across groups and whether some groups show stronger amplification. We exclude gender-only analysis because generation without specified ethnicity predominantly depicts White individuals. Such analysis therefore effectively reduces to gender bias within the White group.

| Diagnostic illustration | Interpretation |
|---|---|
| \(\mathrm{L2V}=0.5\) | Men occur three times as frequently: \(N_{\text{man}}=3N_{\text{woman}}\). |
| \(\mathrm{L2V}=-0.5\) | Inverse ratio. |
| \(\mathrm{L2V}\approx0\) | Controlled diagnostic target, not real-world ground truth. |
| Seven ethnic groups | Balanced probability \(1/7\approx14.3\%\). |
| \(\mathrm{F6N}_e=0.769\) | \(0.769+0.143\approx91.2\%\) of videos; more than six times expected frequency. |

**Ethnic quantities.** For outputs with identifiable ethnicity, we define
\[
P_e=\frac{N_e}{N_{\text{total}}},\quad e\in E,
\]
where \(N_e\) counts outputs classified as ethnicity \(e\), and \(N_{\text{total}}\) counts all outputs with identifiable ethnicity.

The first complementary quantity measures each group’s deviation from uniform coverage. Following Feldman et al. (2015) and Mehrabi et al. (2021), we define **F6N = Representation Deviation Score for Ethnicity**, also written \(\mathrm{RDS}_e\):
\[
\mathrm{F6N}_e=P_e-\frac1{|E|}.
\]
Here, \(|E|\) is the number of ethnic groups. Positive values indicate overrepresentation, and negative values indicate underrepresentation. The uniform reference applies to neutral J7K auditing, whose prompts deliberately remove contextual and identity cues.

The second quantity measures the probability that two randomly selected outputs belong to different groups. Following Simpson (1949), we define **W3H = Simpson’s Diversity Index (SDI)**:
\[
\mathrm{W3H}=1-\sum_{e\in E}P_e^2.
\]
Higher values indicate more diverse and balanced distributions. Lower values indicate concentration in dominant groups.

Group deviations identify who is overrepresented or underrepresented. Overall diversity measures the balance of the full distribution. Together, they distinguish homogeneous or skewed portrayals from broadly diverse ones.

**Shift quantities.** We measure directional changes caused by alignment as \(\Delta\mathrm{L2V}\), \(\Delta\mathrm{F6N}_e\), and \(\Delta\mathrm{W3H}\), where \(\Delta\) means change. More balanced gender ratios, less ethnic skew, or greater diversity count as improvement. Aggregating across videos complements fine-grained frame dynamics and within-video persistence when assessing how bias forms or decreases.

## 4 Social Biases in Video Generative Models

| Models | Average L2V | White L2V | White F6N | Black L2V | Black F6N | Latino L2V | Latino F6N | East Asian L2V | East Asian F6N | Southeast Asian L2V | Southeast Asian F6N | India L2V | India F6N | Middle Eastern L2V | Middle Eastern F6N | Overall W3H |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| D8P (u) | 0.4815 | 0.5683 | 0.7690 | 0.3912 | -0.1952 | 0.6308 | -0.1952 | 0.4406 | -0.1810 | 0.4611 | — | 0.3938 | — | 0.4833 | -0.1976 | 0.0538 |
| K5X (a) | 0.5295 | 0.5584 | 0.7833 | 0.5114 | -0.1976 | 0.6729 | -0.1929 | 0.4282 | -0.1976 | 0.5020 | — | 0.4878 | — | 0.5393 | -0.1952 | 0.0267 |
| \(\Delta\) | +0.0480 | -0.0099 | +0.0143 | +0.1202 | -0.0024 | +0.0421 | +0.0023 | -0.0124 | -0.0166 | +0.0409 | — | +0.0940 | — | +0.0560 | +0.0024 | -0.0271 |
| B7Q (u) | 0.7581 | 0.7485 | 0.6905 | 0.6167 | -0.1905 | 0.8599 | -0.1952 | 0.6976 | -0.1500 | 0.8272 | — | 0.8032 | — | 0.7560 | -0.1548 | 0.1252 |
| H4M (a) | 0.8306 | 0.8713 | 0.6381 | 0.8095 | — | 0.8599 | -0.2476 | 0.7762 | -0.2426 | 0.8929 | — | 0.7762 | — | 0.7664 | -0.1476 | 0.1119 |
| \(\Delta\) | +0.0725 | +0.1228 | -0.0524 | +0.1928 | — | 0.0000 | -0.0524 | +0.0786 | -0.0926 | +0.0657 | — | -0.0270 | — | +0.0104 | +0.0072 | -0.0133 |

**Table 2.** Two pairs of unaligned (u) and aligned (a) models. Entries average across 42 verbs. Nonzero unaligned L2V values indicate inherent bias before alignment; the baselines are not bias-free. Metric signs describe absolute model bias, not shifts. Each \(\Delta\) row reports changes from its corresponding baseline. Positive and negative \(\Delta\mathrm{L2V}\) values mark shifts toward preferring men and women, respectively. Positive and negative \(\Delta\mathrm{F6N}\) values mark shifts toward overrepresentation and underrepresentation. Throughout the video-distribution tables, “—” means that J7K produced zero recognizable instances of the corresponding ethnicity. This severe absence of diversity prevents subgroup computation.

Under V9R, gender portrayal remains male-biased across all ethnicities. Aligned variants increase the average imbalance, suggesting that preference fine-tuning may worsen disparity.

All models overrepresent White individuals. D8P has the strongest imbalance, and K5X worsens it. B7Q is moderately more balanced. H4M reduces White dominance but also lowers overall diversity. Thus, reduced ethnic skew can accompany reduced demographic variety.

| Additional narrative quantities | Value |
|---|---:|
| K5X L2V increase | 0.04 |
| K5X White F6N | 0.783 |
| D8P White F6N | 0.769 |

The four leading models use different alignment strategies. The first baseline is **D8P = ModelScope** (Wang et al., 2023a). Its optimized counterpart is **K5X = InstructVideo**, which aligns the baseline with A3J (Yuan et al., 2024).

The second baseline appears under several spellings: **B7Q = VideoCrafter-2, VideoCrafter-V2, Video-Crafter-V2**. We cite Chen et al. (2024a) and, for the distilled-model analysis, Chen et al. (2023). Its optimized counterpart is **H4M = T2V-Turbo-V1** (Li et al., 2024). H4M uses U8D, InternVid2-S2 (Wang et al., 2024b), and ViCLIP (Wang et al., 2023b) as alignment rewards.

Comparing the pair within each architecture separates alignment effects from base-architecture effects. Each model generates each prompt 10 times with different random seeds. We average the results to reduce sampling variance. V9R supplies the gender assessment, and J7K supplies the ethnic-balance assessment.

**Human evaluation.** Despite residual disagreement on ethnicity, the ensemble is reliable, scalable, and aligned with human judgments. Three independent researchers with experience in social-bias analysis annotated generated videos. They used standardized written guidelines and task-specific training based on prior gender and ethnicity conventions. Verification checked whether videos correctly reflected the prompted demographics, not subjective or normative judgments.

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

Pearson correlation measures how closely the judgments vary together. Cohen’s Kappa and Fleiss’ Kappa measure agreement while accounting for chance agreement. The majority-voted open-source judges agree strongly with humans. The Kappa results indicate strong to near-perfect consistency among annotators.

Many ethnicity disagreements arise because humans judge videos holistically while the pipeline examines individual frames. Humans may miss or infer faces that appear briefly during fast motion, while the pipeline explicitly captures them. This pattern indicates higher recall rather than systematic false positives.

Restricting the analysis to cases with perfect human-majority/VLM agreement leaves all core bias metrics statistically stable. The nonsignificant difference indicates that residual evaluator uncertainty does not distort the measurements.

Human review grounds and sanity-checks large-scale automated assessment rather than replacing it. A larger review would be desirable but would require substantial labor. Small, high-consensus validation follows prior practice for model-based judges.

The investigation separates three stages: demographic distributions in image-reward data, biases in learned rewards, and the rewards’ influence on gender and ethnicity representation in aligned videos.

## 5 Social Biases in Image Reward Datasets and Reward Models

Image-based rewards are the de facto state-of-the-art approach to video alignment (Wu et al., 2023; Xu et al., 2024; Li et al., 2024; Yuan et al., 2024; Liu et al., 2024a; Prabhudesai et al., 2024; Black et al., 2023; Ma et al., 2025). Rewards trained on static images guide learning of temporally coherent videos. Their inherent biases can transfer and amplify across frames.

Assessing demographic disparities and changes in attributes over time requires distinguishing human-labeled image preferences from learned rewards.

### 5.1 Preference in Image Reward Datasets

| Dataset | Validated (images, caption, preference) tuples | Covered verbs | Valid man–woman-pair verbs, out of 42 | Directionally preferred verbs | White preference |
|---|---:|---:|---:|---|---:|
| R9C | 28,783 | 29 | 26 | Men: 69.23% (18/26) | 43.34% |
| N6T | 14,958 | 19 | 18 | Women: 61.11% (11/18) | 40.08% |

R9C tends to prefer men, while N6T shows a relative preference for women. Both strongly favor White representations, followed by East Asian and Indian representations. Verb-specific exceptions include a preference for Black representations for “bake” and East Asian representations for “fish.”

Collected preferences may implicitly privilege Western-centered aesthetics and representation. These imbalances risk reinforcing existing social inequities during downstream reward training. Inclusive data that demographically represent global diversity are urgently needed.

Two widely used collections provide the human comparisons: **R9C = HPDv2** (Wu et al., 2023) and **N6T = Pick-a-Pic** (Kirstain et al., 2023a; 2023b). The latter provides the contrasting preference trend.

GPT-4o-mini extracts gender, ethnicity, and verbs from captions. Qwen2-VL-7B, Qwen2.5-VL-7B, and InternVL2.5-8B classify the images. We combine caption and image attributes, retaining instances that contain predefined verbs.

Each tuple contains two images and a human choice of which image better matches the caption. To measure explicit gender preference, we use pairs in which one gender is preferred and another is not, including man-versus-woman comparisons. Filtering for valid pairs narrows verb coverage. Ethnicity and gender preferences measure how often people select each group for a given verb.

**Figure 3.** Gender preference distributions in image-reward datasets across 42 verbs.

**Figure 4.** Ethnicity preference distributions in image-reward datasets across 42 verbs.

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

**Table 4.** Reward-model preference bias, averaged across 42 verbs.

Y2F slightly favors women and White individuals. It has the highest diversity and a relatively balanced ethnic distribution. Fine-tuning on R9C produces A3J’s strong preference for men across ethnicities. U8D and E5W favor women.

These shifts match the training-data preferences and reveal consistent gender preferences across ethnicities, although U8D’s training data remain undisclosed. After fine-tuning, ethnic preferences shift toward Middle Eastern representations in A3J, Latino representations in U8D, and East Asian representations in E5W. All lose diversity.

| Additional narrative quantity | Value |
|---|---:|
| E5W East Asian F6N | 0.0352 |

**Benchmark construction.** The controlled text-to-image (T2I) benchmark draws inspiration from R9C and ImageRewardDB (Xu et al., 2024). It uses the leading FLUX model (Labs, 2023) and event templates to vary gender, ethnicity, and verbs systematically.

The second generation condition explicitly specifies both actor attributes: **C4Z = Ethnicity+Gender**. V9R specifies ethnicity alone. Both settings generate 100 images per prompt to support statistical robustness and provide a large, diverse evaluation set.

| Settings | Number of prompts | Examples |
|---|---:|---|
| V9R | 294 | An East Asian person is baking cookies in a cozy kitchen, with warm light and the aroma of vanilla filling the air. |
| C4Z | 1176 | An East Asian woman is baking cookies in a cozy kitchen, with warm light and the aroma of vanilla filling the air. |

**Table 3.** Image-reward-benchmark prompts explicitly include ethnicity when specified, verbs, and context.

We do not assume that FLUX is unbiased. Controlled, reproducible demographic conditioning isolates reward preferences independently of generator-specific artifacts.

Three annotators independently reviewed 100 randomly sampled images. Conservatively, unanimous agreement covered 77%. Majority confirmation of correct attributes covered 97%, with only three images lacking consensus. Disagreements concerned visually ambiguous prompts rather than systematic failure or demographic-specific distortions.

Clear gender and ethnicity cues, along with a high ratio of signal to noise, support reliable reward processing. Residual noise is unlikely to materially affect preference patterns.

**Figure 5.** Images in the constructed benchmark use **“A/An [ethnicity][gender] is baking [context].”** Only \([\mathrm{gender}]\in\{\text{man,woman}\}\) appears.

**Reward identities and scoring.** We evaluate four models. **Y2F = CLIP** is the shared base before fine-tuning for the three learned rewards (Radford et al., 2021). **A3J = HPSv2.0** trains on R9C (Wu et al., 2023). The later version, **U8D = HPSv2.1**, is also cited to Wu et al. (2023). Its unreleased training collection has a distinct label: **X7H = HPDv2.1**. N6T trains the remaining learned reward, **E5W = PickScore** (Kirstain et al., 2023b).

The models predict continuous numerical preference scores, not discrete labels. These include the corresponding versioned human-preference score. Standardizing scores supports controlled, relative demographic comparisons.

For gender evaluation, the scoring prompt is **“A/An [ethnicity] person is [verb]-ing [context]”**, which omits gender. Generation instead uses **“A/An [ethnicity] [gender] is [verb]-ing [context]”**, explicitly fixing all three dimensions.

We standardize reward scores using their mean and standard deviation, a measure of score spread. We then average scores across images for each generation prompt. At a fixed ethnicity and verb, we subtract women’s average standardized score from men’s. This produces an adapted L2V, which can exceed one. Its sign retains the same gender-preference interpretation.

For ethnicity evaluation, the scoring prompt is **“A person is [verb]-ing [context]”**, which omits ethnicity. Generation instead uses **“A/An [ethnicity] person is [verb]-ing [context]”**, explicitly fixing ethnicity and verb.

We first standardize by the mean and standard deviation, then average by prompt. For each fixed verb, we apply softmax normalization across ethnicities to convert scores into a preference distribution (Bridle, 1990; Bishop, 2006). We use these distributions to calculate F6N and W3H.

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

**Table 5.** Social bias in aligned models, averaged across 42 verbs. Missing values follow the J7K convention for zero recognizable instances.

**Gender.** V9R evaluation finds strong base-model bias toward men across ethnicities. A3J increases this bias. U8D and E5W significantly reduce it toward balanced or woman-preferred outputs. These changes reflect the rewards’ preference for women and counteract the baseline’s dominance of men.

**Ethnicity.** Under the ethnicity-only condition, the baseline strongly favors White individuals and underrepresents Black, East Asian, and Middle Eastern groups. U8D improves White/Black balance but significantly decreases Latino representation and diversity. E5W produces the highest diversity, more balanced representation across most groups, and the most demographically equitable outputs.

| Additional narrative quantity | Value |
|---|---:|
| U8D-aligned Latino representation F6N | –0.4352 |

For reward-specific tuning, we use a distilled consistency generator as the baseline: **G4S = Video Consistency Model distilled from VideoCrafter-V2 (VCM-VC2)** (Li et al., 2024). Distillation trains a student model from a pretrained teacher. A3J, U8D, and E5W provide the three image-reward conditions.

The H4M training protocol adds reward feedback to state-of-the-art **Latent Consistency Distillation** (Luo et al., 2023), which supports single-step video generation. During student distillation from a pretrained text-to-video teacher, the protocol directly optimizes decoded frames to maximize image–text alignment rewards. This guides each frame toward human-preferred representations. Comparing QX2 results before and after tuning measures reward-specific identity shifts.

The temporal results use another reward-version label, **M1Y = HPSv2**. We retain this distinct label in the temporal table. All reported changes remain relative to G4S.

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

**Table 6.** Temporal consistency of actor identity. High scores indicate stable representation; low scores indicate flicker or change. “Std” means standard deviation. Red and blue subscripts indicate relative improvements and degradations against G4S.

**Temporal behavior.** Alignment substantially improves technical coherence. U8D increases ethnicity stability and the frequency of perfect stability. Gender stability approaches saturation. Standard deviations consistently decline, indicating more uniform frame consistency.

Biased rewards can entrench disparity rather than reduce it. A3J amplifies the preference for men while locking it in over time, with nearly all videos showing perfectly stable gender. Smoother, more coherent stereotyping makes bias persistent and deeply embedded. QX2 uniquely reveals how alignment converts representational artifacts into systematic distortions that resist change across frames.

Reward profiles largely determine the direction and magnitude of aligned bias. A3J reinforces the base model’s tendencies to prefer men and White individuals. U8D and E5W counteract the dominance of men without reaching complete demographic parity, leaving ethnic disparities.

## 7 Controllable Preference Modeling for Video Diffusion Models

We use gender as a case study of how the social-attribute composition of image preferences can deliberately calibrate portrayals or make them more equitable (Sheng et al., 2020). Varying this composition tests the direction of reward bias and the dynamics of video alignment.

### 7.1 Image Reward Dataset Construction

We construct two collections, each with 2.94 million preference pairs from C4Z. Each pair fixes action and ethnicity while changing gender, including M-1 versus W-1. The image source is identified with the human-preference-dataset analysis.

The prompts are **“A/An [ethnicity] person is [action]-ing [context].”** Man-preferred labels assign male images 1 and female images 0. Woman-preferred labels reverse these assignments. An additional 537,660 R9C image pairs increase diversity in images without faces.

When applied to a man-biased base model, the woman-preferred data correct imbalance and promote equitable gender representation.

### 7.2 Image Reward Model Development & Alignment Tuning

| Models | Average | White | Black | Latino | East Asian | Southeast Asian | India | Middle Eastern |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Y2F | -0.0726 | 0.0343 | -0.1198 | -0.0934 | -0.1315 | -0.0865 | -0.0508 | -0.0607 |
| Z6B | 1.5280<sub>+1.60</sub> | 1.6300<sub>+1.60</sub> | 1.5752<sub>+1.70</sub> | 1.5524<sub>+1.65</sub> | 1.4323<sub>+1.56</sub> | 1.4525<sub>+1.54</sub> | 1.5619<sub>+1.61</sub> | 1.4914<sub>+1.55</sub> |
| P3L | -0.7448<sub>-2.27</sub> | -0.6318<sub>-2.26</sub> | -0.7943<sub>-2.37</sub> | -0.8279<sub>-2.38</sub> | -0.6282<sub>-2.06</sub> | -0.6429<sub>-2.10</sub> | -0.8846<sub>-2.45</sub> | -0.8042<sub>-2.30</sub> |

**Table 7.** Reward-model L2V preference bias, averaged across 42 verbs.

Z6B consistently raises gender preference scores across demographic groups. P3L systematically favors women. Reward tuning captures and amplifies gender-specific aesthetic and representational preferences.

The two datasets fine-tune pretrained Y2F vision encoders, which turn images into model representations, into corresponding reward systems. **Z6B = Man-Preferred Reward Model (RMM)** learns from the man-preferred dataset. **P3L = Woman-Preferred Reward Model (RMW)** learns the opposite dataset’s preferences. Both retain the same underlying pretrained architecture.

We do not train a “fair” or demographically neutral reward. Given G4S’s pronounced initial bias toward men, a neutral reward would merely reinforce the imbalance rather than correct it. Deliberately counter-biased P3L, like U8D, actively steers generation toward gender equilibrium and produces markedly more balanced portrayals. Directional reward tuning thus serves as a corrective mechanism.

| Models | Average | White | Black | Latino | East Asian | Southeast Asian | India | Middle Eastern |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| G4S | 0.8034 | 0.7925 | 0.7758 | 0.8690 | 0.7115 | 0.7945 | 0.8634 | 0.8071 |
| + Z6B | 0.9584<sub>+0.16</sub> | 0.9595<sub>+0.17</sub> | 0.9524<sub>+0.18</sub> | 0.9756<sub>+0.11</sub> | 0.9437<sub>+0.23</sub> | 0.9447<sub>+0.15</sub> | 0.9640<sub>+0.10</sub> | 0.9709<sub>+0.16</sub> |
| + P3L | 0.3082<sub>-0.50</sub> | 0.3341<sub>-0.46</sub> | 0.3913<sub>-0.38</sub> | 0.3314<sub>-0.54</sub> | 0.1008<sub>-0.61</sub> | 0.2639<sub>-0.53</sub> | 0.3446<sub>-0.52</sub> | 0.3894<sub>-0.42</sub> |

**Table 8.** L2V social bias in aligned video models, averaged across 42 verbs.

The same preference-optimization framework produces two G4S variants. Alignment with Z6B raises scores across all demographic groups and reinforces preference for men. Alignment with P3L substantially lowers scores and strongly shifts generation toward preference for women.

Fine-grained control can amplify or reduce social tendencies in diffusion models. Finding the most balanced generator can be framed as a dataset-composition problem. Systematically varying the proportions of man- and woman-preferred samples can identify an optimal mixture that minimizes bias while preserving aesthetic fidelity.

Beyond binary control, more nuanced mixtures suggest a continuous alignment axis. This axis could offer tunable trade-offs between bias mitigation and aesthetic coherence. Dataset composition provides a powerful, interpretable control beyond changes to architecture alone.

**Figure 6.** Effects of Z6B- and P3L-guided alignment by action: (a) \(\Delta\mathrm{L2V}\) before and after tuning, broken down by action and used as the basis for the two sensitivity panels; (b) actions sensitive to man-preferred alignment; (c) actions sensitive to woman-preferred alignment.

**Action sensitivity.** Actions cluster according to the guiding reward. Z6B, shown in teal, amplifies male preference for exercise, row, and cook. P3L, shown in orange, shifts bake, sleep, and sweep toward women.

We rank normalized sensitivity using
\[
\frac{\Delta\mathrm{L2V}}{\text{each reward model's baseline L2V}}.
\]
Sleep, stretch, and read rank highest in both directions. Even semantically or socially neutral activities, and domestic activities, show pronounced gender shifts. These results reveal systematic amplification specific to actions and show how event-centered auditing detects changes across gendered dimensions.

**Responsible use.** Technical “debiasing” is not a complete solution, and using it unilaterally carries significant risk. Controllable preference modeling must support diverse stakeholder input, not replace it.

Equitable representation is a context-dependent normative decision that requires active community consultation. Practitioners should involve affected communities in setting goals rather than leave complex social-bias decisions solely to automated optimization.

## 8 Conclusion

The identified evaluation blind spots warrant bias auditing and mitigation at every stage of video generation. This approach advances more equitable, socially aware systems rather than treating preference alignment as sufficient on its own.

## Limitations

Our predefined gender and ethnicity categories follow U.S. Census conventions and prior literature. They are practical but socially constructed, and they miss identity fluidity, intersectionality, and cultural nuance (Yin et al., 2024; Qiu et al., 2024a; Qiu et al., 2025). Richer representations, including intersectional groups, remain future work.

Human-validated VLM judges remain image-level classifiers with possible biases and inaccuracies, especially for stylized or ambiguous frames. Combining multiple models reduces these problems but does not eliminate them. Larger, more diverse ground-truth video annotation would strengthen reliability.

The alignment experiments mainly use single-step latent consistency distillation and a limited set of rewards. Reinforcement-learning-based tuning and multi-turn video instruction alignment may have different dynamics that these experiments do not capture.

Synthetic manipulation of gender preferences demonstrates targeted control. It does not resolve broader value alignment, normative appropriateness, or the long-term societal impacts that responsible deployment must address.

Despite enabling fine control, the fixed taxonomy of socially associated actions limits possible generalization to open-ended generation and new actions. Research still needs holistic, culturally grounded, ethically aligned evaluation.

## Broader Impact Statement

As video diffusion spreads through content creation, it gains influence over visual culture and societal representation. Alignment’s ability to stabilize and entrench bias is an important but often overlooked consequence of adoption.

**Bias stabilization.** Alignment, including RLHF, is essential for fidelity and temporal consistency, but it can stabilize bias. Removing artifacts and temporal flicker makes stereotypes appear natural, coherent, and authoritative.

Polished biases are harder to recognize as failures than obvious errors in unaligned outputs. High-quality, preferred content therefore risks subtly normalizing skewed demographics in synthetic media and reinforcing harmful stereotypes.

**Dual use.** Curated preference data can shape rewards to amplify or suppress social attributes. Correcting historical underrepresentation and reducing bias use the same machinery as intentional exclusion, homogenization, or propaganda that serves exclusionary worldviews. Alignment is therefore double-edged.

**Responsibility.** Aesthetic metrics alone are insufficient. Developers must continually monitor the social composition of generated outputs through event-centered audits at multiple levels, such as QX2. Preference-data curation deserves the same rigor as architecture design. The human preferences that guide models should reflect a diverse, equitable society rather than reproduce prejudice.

The supplied supplementary inventory includes per-verb gender and ethnicity analyses, training-data effects, statistical analyses, and implementation and training details:

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

We thank the anonymous reviewers for their thoughtful feedback and suggestions. Partial support came from AFOSR MURI grant #FA9550-22-1-0380; Office of Naval Research (ONR), Award #N00014-23-1-2780; and National Science Foundation, Award #IIS-2449768. Additional support came from NVIDIA Academic Grant Program cloud credits and Coefficient Giving’s Technical AI Safety Research Program funding. The authors’ views and conclusions do not represent official policies or endorsements of AFOSR, ONR, the U.S. Government, the National Science Foundation, NVIDIA, or Coefficient Giving.
