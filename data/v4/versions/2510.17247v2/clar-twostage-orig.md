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

Recent video diffusion models have substantially improved text-to-video generation: generating videos from natural-language descriptions. These improvements include alignment tuning, which further optimizes a trained model using reward models trained on human preferences. A reward model assigns a score that guides this optimization. Although alignment improves visual quality, it can also encode and amplify social biases, including imbalanced portrayals of gender and ethnicity.

We introduce VideoBiasEval, a diagnostic framework for tracing these biases through the alignment pipeline. It uses event-based prompts that separate what happens—the verb and context—from who performs it—the actor’s gender and ethnicity. Its metrics assess four aspects: overall ethnicity bias, gender bias within each ethnic group, changes in social-attribute distributions between model variants, and the persistence of those attributes across video frames.

Using this framework, we conduct the first end-to-end analysis connecting biases in human preference datasets, their amplification in reward models, and their propagation into alignment-tuned video diffusion models. Our results show that alignment both strengthens representational bias and makes it temporally stable. For example, alignment with one reward model increases the average gender-bias score from 0.8034 to 0.9116, while gender stability reaches 0.9967 (Tables 5 and 6). The resulting portrayals are smoother but more stereotyped. These findings show the need for bias-aware evaluation and mitigation throughout alignment to support fair and socially responsible video generation.

Reviewed on OpenReview: https://openreview.net/forum?id=C0yxuS6jty

**Figure 1:** Overview of our work. (1) We introduce VideoBiasEval, a social-bias evaluation framework for video generation. It uses event-based prompts and metrics at several levels of detail to assess ethnicity and gender bias (bottom left, Section 3). The framework represents videos through social-attribute annotations (top). Visual-language models, also called vision–language models (VLMs), infer actors’ gender and ethnicity from individual frames. These annotations are aggregated to quantify bias (Section 3.3). (2) We conduct the first comprehensive analysis of how image-based reward models, trained on human-labeled preferences, influence social-attribute distributions in diffusion-generated videos. The analysis separates human preference biases, reward-model preference biases, and their downstream effects on video-generation biases (bottom right, Sections 5.1, 5.2, and 6).

## 1 Introduction

Recent video diffusion models have substantially improved the quality of videos generated from natural-language prompts (Chen et al., 2024a; Wang et al., 2023a; Yuan et al., 2024; Li et al., 2024). These models support applications in creative media, education, and professional simulation (Cho et al., 2024; Miller et al., 2024). Many state-of-the-art systems now use alignment tuning to improve visual fidelity, coherence, and controllability. The most prominent approach learns from human preferences (Wu et al., 2023; Xu et al., 2024; Li et al., 2024; Yuan et al., 2024; Liu et al., 2024a; Prabhudesai et al., 2024; Black et al., 2023; Ma et al., 2025). It typically uses reward functions trained on human-labeled comparisons. These functions often judge individual frames or images and guide optimization after the model’s initial training (Wu et al., 2023; Kirstain et al., 2023a; Xu et al., 2024).

We use **aligned models** to mean video diffusion models that undergo this post-training optimization to better match human preferences beyond likelihood-based training. Alignment commonly uses reward-weighted fine-tuning or reinforcement learning from human feedback (RLHF)-style procedures. A learned preference or aesthetic reward model, trained on human comparisons, guides further optimization of the base video model. The conceptual pipeline is:

**base video diffusion model → preference or reward model → post-training optimization.**

The arrows indicate the stages through which we trace alignment effects. We contrast aligned models with **unaligned models**, which are trained only with likelihood-based objectives. These unaligned counterparts serve as controlled baselines for isolating alignment effects.

Alignment improves perceptual quality and temporal coherence, but its reliance on subjective preferences creates an underexplored risk. Human judgments are often collected without explicit consideration of cultural, social, or demographic diversity. They may therefore encode systematic biases that reward models and downstream generators inherit and reinforce (Qiu et al., 2023). Alignment can shape not only what a model generates well, but also who appears, how often, and in what roles. We ask: **How does alignment tuning influence the distribution of social attributes in video diffusion models?** Here, social attributes are the gender and ethnicity categories used in our evaluation.

Answering this question requires three components: prompts that elicit social attributes, metrics that quantify their distributions, and a protocol that tracks those distributions before and after alignment. Existing frameworks (Huang et al., 2024; Liu et al., 2024b; Sun et al., 2024) have three limitations:

1. Their prompts do not adequately represent diverse social identities, limiting analysis of how models portray or misrepresent those identities.
2. Their identity coverage and metrics are insufficient for tracking alignment’s effects on social-attribute distributions.
3. They lack a dedicated method for measuring pre- and post-alignment shifts in those distributions.

We address these limitations with VideoBiasEval (Section 3). It uses event-based prompting grounded in established social-bias taxonomies (Zhao et al., 2017; Hendricks & Nematzadeh, 2021; Cho et al., 2023; Qiu et al., 2023). An event separates the actor from the verb and surrounding context. This design controls identity attributes independently of semantic content, making comparisons across contexts more interpretable. Following prior work on alignment and fairness in generative systems (Luccioni et al., 2023; Shen et al., 2023), we focus on gender and ethnicity because they have comparatively well-defined evaluative boundaries.

VideoBiasEval measures ethnicity bias, gender bias within each ethnic group, distribution shifts between models, and temporal persistence within videos. We use it to trace social-attribute distributions through the alignment pipeline.

First, we analyze demographic preferences in two human-labeled image preference datasets: **HPDv2** (Wu et al., 2023) and **Pick-a-Pic** (Kirstain et al., 2023a) (Section 5.1). Next, we examine three image reward models: **HPSv2.0**, trained on HPDv2; **HPSv2.1**, trained on an unreleased preference dataset (Wu et al., 2023); and **PickScore**, trained on Pick-a-Pic (Kirstain et al., 2023a) (Section 5.2). Finally, we use these reward models to fine-tune a video consistency model distilled from the VideoCrafter-2 video generator (Chen et al., 2023). We compare its outputs before and after alignment (Section 6).

Both human preference datasets show non-neutral gender preferences and a strong imbalance favoring White representations. Among verbs with valid man–woman comparison pairs, HPDv2 favors men for 69.23% (18/26), while Pick-a-Pic favors women for 61.11% (11/18). White representations receive 43.34% and 40.08% of ethnicity preferences, respectively (Section 5.1). Reward models trained on these datasets inherit and amplify social biases, which then propagate into aligned video models (Tables 4 and 5). Male- or female-preferred reward models systematically shift gender portrayals, while ethnic representation remains uneven.

Our **Temporal Attribute Stability (TAS)** metric measures the percentage of sampled frames whose classified attributes match the video’s final majority-voted label. TAS shows that alignment improves the consistency of identity portrayals but can make biased portrayals more persistent. For example, HPSv2.0 alignment raises average gender bias from 0.8034 to 0.9116 while gender TAS reaches 0.9967 (Tables 5 and 6). Thus, these findings demonstrate that alignment improves quality and coherence while reinforcing or stabilizing existing social bias. Bias-aware evaluation and alignment strategies are needed throughout the pipeline.

We also ask whether deliberately controlling social-attribute distributions in image preference data can produce controllable reward models and video outputs (Section 7). Our gender-focused experiments show that a counter-biased reward model can reduce average video gender bias from 0.8034 to 0.3082 (Table 8). We further analyze reward-model preferences and video-model shifts across 42 events. These findings provide guidance on targeted data composition and counter-biased reward design as potential ways to reduce representational disparities.

We make three contributions:

1. **VideoBiasEval:** an event-based framework with metrics for ethnicity bias, ethnicity-conditioned gender bias, and temporal stability.
2. **An end-to-end analysis:** the first analysis connecting human preference datasets, reward models, and aligned video diffusion models to show how biases are inherited, amplified, and stabilized.
3. **Experiments and controllable preference modeling:** evidence that preference alignment improves perceptual quality and temporal coherence while reshaping—and sometimes reinforcing—the social composition of generated content. Targeted preference-data composition and counter-biased reward design can steer models toward more equitable behavior.

## 2 Related Work

**T2V Evaluation.** Text-to-video (T2V) benchmarks such as VBench (Huang et al., 2024), EvalCrafter (Liu et al., 2024b), and T2V-CompBench (Sun et al., 2024) evaluate generated videos using measures such as Fréchet Video Distance, a measure of video-distribution differences (Unterthiner et al., 2019); CLIP-Score, a text–visual alignment score (Hessel et al., 2021); and object consistency, which assesses whether depicted objects remain consistent. These measures overlook who is depicted and how identities are portrayed.

GRiT-based metrics, which check the depicted objects or roles, may verify that a “doctor” appears without flagging that all doctors are white men (Wu et al., 2025). CLIP-based alignment uses a shared image–text model to reward textual fidelity, but it ignores demographic balance. CLIP also serves as the base model for the reward models examined in Section 5.2. Fair and trustworthy T2V evaluation must therefore audit social-attribute distributions, not only surface-level generation quality. Our event-centric framework measures gender- and ethnicity-aware biases throughout the T2V pipeline.

**Bias Evaluation in Generative Models.** Most studies of social bias in text-to-image or language generation examine static, single-frame outputs, such as portraits or isolated object scenes. StableBias (Luccioni et al., 2023), DALL-Eval (Cho et al., 2023), and SocialCounterfactuals (Howard et al., 2024) are bias-evaluation approaches that mainly count identity frequencies. They seldom examine what those identities are shown doing.

Even recent demographic-representation benchmarks often evaluate each image independently. This can conceal recurring patterns, such as portraying men in authoritative roles and women in supportive roles. Evaluations that do not jointly analyze actors, verbs, and context cannot capture role-specific stereotypes or reveal bias in narrative and temporal settings. We address this gap by auditing events. We separate actor attributes from verbs and environments to examine representation across scenarios.

## 3 VideoBiasEval

VideoBiasEval evaluates social biases in video-generation models. We systematically vary actor gender and ethnicity across real-world events (Section 3.1), generate videos from structured prompts (Section 3.2), and extract social attributes from the outputs. We then evaluate those attributes across frames, videos, and event categories (Section 3.3). Figure 2 illustrates representative prompts and the attribute-extraction process.

**Figure 2:** Videos generated by different diffusion models using prompt templates that specify actor attributes, as detailed in Section 3.2. The main character’s gender and ethnicity are extracted using the VLM-based evaluation method in Section 3.3.

### 3.1 Event Definition

We examine how video models visually represent actors performing different verbs. Social biases can appear as imbalanced portrayals across gender or ethnic groups. Such patterns can reinforce stereotypes and undermine fairness, as documented in prior work (Bolukbasi et al., 2016; Sun & Peng, 2021; Zajko, 2021).

We represent an event as the tuple \(\langle p,a,c\rangle\): actor \(p\) performs verb \(a\) in context \(c\). The angle brackets group these three components into one event. We focus on **socially associated verbs**: verbs historically tied to identity-related stereotypes (Zhao et al., 2017; Garg et al., 2018; Cho et al., 2023; Qiu et al., 2023). This representation lets us compare demographic depictions while controlling the activity and context.

**Controlled Attribute Space.** We balance coverage with experimental control. Our design combines four gender categories, seven ethnic groups, and 42 verbs, following established social-bias taxonomies and benchmark conventions. This bounded setup enables the first end-to-end tracing of bias propagation in video generation while supporting clear attribution, interpretability, and reproducibility. Open-ended and intersectional attributes are promising extensions. A well-defined, theoretically grounded scope is important for isolating disparities before moving to unconstrained scenarios.

**Actors.** Each actor \(p\) has gender and ethnicity attributes. Following Luccioni et al. (2023), the four gender categories are **man**, **woman**, the neutral term **person**, and **non-binary person**. This schema provides clear boundaries for controlled evaluation but cannot capture the full diversity of gender identities.

The seven ethnic groups are **White, Black, Indian, East Asian, Southeast Asian, Middle Eastern, and Latino**, following Karkkainen & Joo (2021) and U.S. Census Bureau conventions. These categories are socially constructed. They are not exhaustive or universally representative. The tables retain the heading “India” for the category described here as “Indian.”

**Verbs.** We examine these 42 verbs \(a\):

bake, bike, call, clean, climb, cook, cough, cry, drink, drive, eat, exercise, fish, hit, jump, kick, kneel, laugh, lift, paint, pick, pitch, pray, read, ride, row, run, shop, shout, sit, skate, sleep, smile, stand, stare, stretch, study, sweep, throw, walk, wash, work.

Prior work identifies statistical associations between these verbs and particular genders or ethnic groups (Zhao et al., 2017; Garg et al., 2018; Cho et al., 2023; Qiu et al., 2023). We also selected them for visual distinctiveness and ease of depiction in short clips, supporting reliable annotation and interpretable outputs. This curated, socially meaningful set provides a reproducible basis for later extensions to more complex, culturally specific, or temporally dynamic events. Later sections also call these verbs “activities” or “actions.”

### 3.2 Event-Based Prompting Protocol

| Settings | # of Prompts | Examples |
|---|---:|---|
| Person Only | 42 | A person is baking cookies in a cozy kitchen, with warm light and the aroma of vanilla filling the air. |
| Ethnicity + Person | 294 | An East Asian person is baking cookies in a cozy kitchen, with warm light and the aroma of vanilla filling the air. |

**Table 1:** Statistics of social-bias evaluation prompts for video generation. Each prompt explicitly identifies the actor’s ethnicity when specified, the verb, and the surrounding context. These components provide a structured basis for analyzing social representations.

We use the deterministic template:

> A/An [actor] is [verb]-ing [context].

Square brackets mark slots filled from fixed, predefined text. The `[actor]` slot identifies the actor. The `[verb]` slot contains one of the 42 activities, expressed in its “-ing” form. The `[context]` slot supplies situational detail without changing the verb’s meaning. “A/An” indicates the article used before the actor description.

Neither the actor nor the context is generated at runtime by a large language model (LLM), a model that could otherwise generate the prompt text. All prompts come from a fixed set to support reproducible, controlled comparisons.

We use two conditions:

1. **Person-only:** `[actor]` is “person.” There is one prompt per verb, for 42 prompts.
2. **Ethnicity+Person:** a predefined ethnic descriptor is prepended to “person,” as in “An East Asian person.” Combining the same 42 verbs with seven ethnic groups yields 294 prompts.

Table 1 gives examples and counts. Because Ethnicity+Person already encodes ethnicity, an additional ethnicity-only setting is unnecessary for isolating ethnicity effects.

We query each target video model directly with each prompt. We generate a fixed number of videos per prompt under identical decoding settings. Section 4 specifies 10 generations per prompt per model, using different random seeds.

The aim is to measure how models represent actors when attributes are underspecified for a fixed verb. Person-only prompts reveal which gender and ethnicity the model implicitly assigns without explicit demographic cues. Ethnicity+Person prompts reveal how a specified ethnicity interacts with inferred attributes such as gender.

Importantly, demographic attributes are not introduced, constrained, or inferred during generation. All demographic analysis occurs afterward: an ensemble of VLMs infers actor attributes from generated frames (Section 3.3). The fixed event prompts control contextual wording, so observed disparities can be attributed to demographic conditioning rather than uncontrolled contextual drift. This control supports reproducible bias auditing.

### 3.3 Multi-Granular Event-Centric Bias Evaluation

Our evaluation covers several levels of detail: individual frames, temporal consistency within a video, and demographic distributions across videos. It measures both who appears and how consistently an identity is maintained over time.

**Social Attribute Representations.** We use three open-source VLMs as automated attribute judges: **Qwen2-VL-7B** (Wang et al., 2024a), **Qwen2.5-VL-7B** (Yang et al., 2024), and **InternVL2.5-8B** (Chen et al., 2024b). These names identify the three models used throughout the evaluation.

We uniformly sample 16 frames from each video. This count matches the standard inference defaults of evaluated models such as VideoCrafter. Preliminary experiments showed that reducing the frame count significantly compromised attribute-detection stability.

Each VLM independently classifies gender and ethnicity in every sampled frame. Its gender label is \(g\), drawn from the set

\[
G=\{\text{man, woman}\}.
\]

Its ethnicity label is \(e\), drawn from

\[
E=\{\text{White, Black, Indian, East Asian, Southeast Asian, Middle Eastern, Latino}\}.
\]

The symbol \(\in\) means “belongs to.” These output gender labels are man and woman, while the actor-category design in Section 3.1 contains four categories.

Within each VLM, we use majority voting over frame predictions to obtain video-level labels. We then combine the models’ labels through an ensemble strategy; the ensemble is described as majority voting in Section 4.

Initial manual inspections found that individual VLMs frequently produced unstable or hallucinated attributes. We therefore use distinct architectures rather than repeatedly sampling one model. This “VLMs-as-judges” ensemble cross-validates predictions to improve robustness and reduce model-specific biases (Qiu et al., 2024b). Figure 2 shows representative classification prompts and outputs. The top of Figure 1 shows the extraction pipeline.

The resulting frame- and video-level labels support comparisons between generated attributes and intended prompt attributes. Section 4 reports human–model correlation analyses to assess reliability and fairness.

**Temporal Attribute Stability.** TAS measures identity consistency within a video. For each video, it is the percentage of sampled frames whose classified attributes match the final majority-voted video label. High TAS means a stable demographic portrayal. Low TAS indicates attribute “flickering”: inconsistent identity depiction across frames. This is a temporal artifact that static evaluation does not capture.

The following video-level metrics assess social balance across events and demographic groups.

**Ethnicity-Aware Gender Bias.** We use the **Proportion Bias Score for Gender**, written **PBSG** or \(\mathrm{PBS}_G\), to measure gender portrayals within each ethnicity.[^1] For each verb and ethnic group,

\[
\mathrm{PBS}_G
=
\frac{N_{\text{man}}-N_{\text{woman}}}{N_{\text{total}}}
\in[-1,1].
\]

Here, \(N_{\text{man}}\) counts representations depicting men, \(N_{\text{woman}}\) counts representations depicting women, and \(N_{\text{total}}\) is their sum. The interval \([-1,1]\) gives the possible range.

A score of \(0.5\) means men appear three times as often as women: \(N_{\text{man}}=3N_{\text{woman}}\). A score of \(-0.5\) means the inverse. Positive scores indicate male-preferred representation; negative scores indicate female-preferred representation. Scores near zero indicate balance under this metric.

We use \(\mathrm{PBS}_G\approx0\), meaning approximately zero, as a diagnostic benchmark—not as factual or normative ground truth. Real-world gender distributions may vary across verbs, ethnicities, and sociocultural contexts. Neutral “person”-based prompts deliberately separate verbs from identity attributes. Under this controlled setup, deviations from zero measure how strongly models reproduce gendered stereotypes associated with verbs. This use of a parity baseline follows representational-bias analysis in generative models; it does not assert that equal proportions are universally appropriate.

By computing PBSG under the ethnicity+gender condition, we measure how gender portrayals vary within ethnic groups. This reveals whether gender disparities are consistent across groups or disproportionately amplified for particular ethnicities. PBSG is therefore a probe of stereotype amplification in ethnicity-conditioned activities, not a universal prescription for gender composition.

[^1]: We exclude gender bias from this analysis because, without explicit ethnicity specifications, generative models predominantly depict White individuals (Figure 14). Analyzing gender alone therefore effectively becomes an analysis of gender bias within the White demographic (Figure 30).

**Ethnicity Bias.** We use two complementary metrics: the **Representation Deviation Score for Ethnicity**, written \(\mathrm{RDS}_e\) or RDSe (Feldman et al., 2015; Mehrabi et al., 2021), and **Simpson’s Diversity Index (SDI)** (Simpson, 1949).

For each ethnic group \(e\) in the set \(E\), its representation proportion is

\[
P_e=\frac{N_e}{N_{\text{total}}}.
\]

Here, \(N_e\) counts outputs identified as ethnicity \(e\). In this ethnicity metric, \(N_{\text{total}}\) counts all outputs with an identifiable ethnicity.

The deviation score is

\[
\mathrm{RDS}_e=P_e-\frac{1}{|E|},
\]

where \(|E|\) is the number of ethnic groups. The reference \(1/|E|\) assigns every group the same proportion. With seven groups, this is \(1/7\approx14.3\%\).

For example, an \(\mathrm{RDS}_e\) of 0.769 means the group appears in \(0.769+0.143\approx91.2\%\) of generated videos—more than six times the expected frequency under the reference distribution. Positive RDS means overrepresentation; negative RDS means underrepresentation. We also use “RDS” as shorthand and \(\mathrm{RDS}_{\text{White}}\) for the White group’s score.

The uniform reference is not intended to reproduce real-world demographics. Appropriate proportions can vary by region, culture, or application. We use uniformity only as a diagnostic benchmark under neutral person-only prompts. Because these prompts remove contextual and identity cues, deviations reveal internal preferences for or against ethnic identities rather than external demographic statistics. As in prior representational-harm analysis, parity exposes systematic skew without asserting empirical correctness.

Overall diversity is measured by

\[
\mathrm{SDI}=1-\sum_{e\in E}P_e^2.
\]

The summation adds the squared proportions over all ethnic groups. SDI captures the probability that two randomly selected outputs belong to different groups. Higher scores indicate greater diversity and balance. Lower scores indicate concentration in a few dominant groups. RDS identifies who is over- or underrepresented; SDI describes the overall distribution.

**Bias Shift.** We compare unaligned and aligned models using \(\Delta\mathrm{PBS}_G\), \(\Delta\mathrm{RDS}_e\), and \(\Delta\mathrm{SDI}\). The symbol \(\Delta\) means the change in a metric between model variants. More balanced gender ratios, less ethnic skew, or greater diversity indicate improvement under these diagnostic metrics. Together, the frame-level labels, TAS, and video-level distribution metrics track bias formation and mitigation.

## 4 Social Biases in Video Generative Models

We evaluate four state-of-the-art video diffusion models with different alignment strategies. As defined in Section 1, aligned models undergo post-training optimization to match human preferences, commonly through reward-weighted fine-tuning or RLHF-style procedures using learned preference or aesthetic rewards:

**base video diffusion model → preference/reward model → post-training optimization.**

The aligned models are:

- **InstructVideo** (Yuan et al., 2024), built on the **ModelScope** video generator (Wang et al., 2023a) and aligned using HPSv2.0.
- **T2V-Turbo-V1** (Li et al., 2024), built on **VideoCrafter-2** (Chen et al., 2024a) and aligned with HPSv2.1, **InternVid2-S2** (Wang et al., 2024b), and **ViCLIP** (Wang et al., 2023b). The latter two are additional alignment components used by this model.

Their unaligned counterparts, ModelScope and VideoCrafter-2, provide controlled comparisons that allow us to isolate alignment effects from those of base-model architecture. The paper also uses the names VideoCrafter-V2 and Video-Crafter-V2 for VideoCrafter-2.

For each model, we generate 10 videos per prompt with different random seeds and average the results to reduce sampling variance. Table 2 reports PBSG for ethnicity-aware gender bias, and RDS and SDI for ethnicity representation. Appendix A provides analysis across 42 verbs, the effects of the WebVid-10M training dataset (Bain et al., 2021), and statistical analysis in Table 9.

In the tables, **u** means unaligned and **a** means aligned. A dash, **—**, marks a missing value as described in the captions.

| Models | Average PBSG | White PBSG | White RDS | Black PBSG | Black RDS | Latino PBSG | Latino RDS | East Asian PBSG | East Asian RDS | Southeast Asian PBSG | Southeast Asian RDS | India PBSG | India RDS | Middle Eastern PBSG | Middle Eastern RDS | Overall SDI |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ModelScope (u) | 0.4815 | 0.5683 | 0.7690 | 0.3912 | -0.1952 | 0.6308 | -0.1952 | 0.4406 | -0.1810 | 0.4611 | — | 0.3938 | — | 0.4833 | -0.1976 | 0.0538 |
| InstructVideo (a) | 0.5295 | 0.5584 | 0.7833 | 0.5114 | -0.1976 | 0.6729 | -0.1929 | 0.4282 | -0.1976 | 0.5020 | — | 0.4878 | — | 0.5393 | -0.1952 | 0.0267 |
| \(\Delta\) | +0.0480 | -0.0099 | +0.0143 | +0.1202 | -0.0024 | +0.0421 | +0.0023 | -0.0124 | -0.0166 | +0.0409 | — | +0.0940 | — | +0.0560 | +0.0024 | -0.0271 |
| Video-Crafter-V2 (u) | 0.7581 | 0.7485 | 0.6905 | 0.6167 | -0.1905 | 0.8599 | -0.1952 | 0.6976 | -0.1500 | 0.8272 | — | 0.8032 | — | 0.7560 | -0.1548 | 0.1252 |
| T2V-Turbo-V1 (a) | 0.8306 | 0.8713 | 0.6381 | 0.8095 | — | 0.8599 | -0.2476 | 0.7762 | -0.2426 | 0.8929 | — | 0.7762 | — | 0.7664 | -0.1476 | 0.1119 |
| \(\Delta\) | +0.0725 | +0.1228 | -0.0524 | +0.1928 | — | 0.0000 | -0.0524 | +0.0786 | -0.0926 | +0.0657 | — | -0.0270 | — | +0.0104 | +0.0072 | -0.0133 |

**Table 2:** Social-attribute distributions in two pairs of unaligned (u) and aligned (a) video diffusion models. Every entry is an average across 42 verbs. For absolute PBSG scores, positive means man-preference, negative means woman-preference, and values closer to zero mean less gender skew. Non-zero scores in the unaligned ModelScope and VideoCrafter-V2 baselines therefore indicate pre-alignment bias, not bias-free behavior. For absolute RDS scores, positive means overrepresentation and negative means underrepresentation. SDI measures overall ethnic balance; higher values mean greater diversity and more uniform representation. “—” means the model generated zero recognizable instances of that ethnicity under person-only prompts, where ethnicity is unspecified. These missing values reflect a severe lack of diversity that prevents metric computation for those subgroups. The \(\Delta\) rows report alignment-induced changes relative to each model’s unaligned counterpart. For \(\Delta\)PBSG, (+) marks a man-preference shift and (–) a woman-preference shift. For \(\Delta\)RDS, (+) and (–) mark shifts toward overrepresentation and underrepresentation, respectively.

**Ethnicity-Aware Gender Bias.** Under Ethnicity+Person prompts, every model has positive average PBSG across the ethnic groups, indicating a consistent male bias (Table 2). Alignment appears to amplify it. The increase is reported as 0.04 for InstructVideo and 0.0725 for T2V-Turbo-V1; the corresponding Table 2 average \(\Delta\) entries are +0.0480 and +0.0725. These results indicate that preference-based fine-tuning may worsen gender disparity rather than alleviate it. Figures 7, 8, 12, 10, 9, 11, and 13 provide PBSG scores across the 42 verbs for each ethnicity.

**Ethnicity Bias.** Under person-only prompts, all models strongly overrepresent White individuals. ModelScope has the strongest imbalance, with \(\mathrm{RDS}_{\text{White}}=0.769\) and SDI = 0.0538. Alignment further amplifies this imbalance in InstructVideo: \(\mathrm{RDS}_{\text{White}}=0.783\) and SDI = 0.0267. Table 2 gives the RDS values at its reported precision, 0.7690 and 0.7833.

VideoCrafter-2 has moderately better balance, with \(\mathrm{RDS}_{\text{White}}=0.6905\) and SDI = 0.1252. T2V-Turbo-V1 reduces White dominance to \(\mathrm{RDS}_{\text{White}}=0.6381\), but diversity also falls to SDI = 0.1119. Thus, alignment can alleviate a particular ethnic skew while suppressing overall diversity. This suggests a trade-off between bias reduction and representational variety. Figure 14 shows ethnicity bias across the 42 verbs.

**Human Evaluation.** We validate the VLM evaluators using 400 generated videos, each annotated for gender and ethnicity by three independent researchers experienced in social-bias analysis. They followed standardized written guidelines and received task-specific training grounded in categorization conventions from prior bias-evaluation work. Their task was to verify whether generated content reflected the prompted demographic attribute, not to make subjective or normative judgments.

The three-VLM ensemble, combined by majority voting, aligns strongly with human judgments. Pearson correlation, which measures correspondence between the human and model judgments, is 0.89 for gender and 0.73 for ethnicity.

Ethnicity judgments retain some disagreement. Further analysis shows that many discrepancies arise from holistic human review versus frame-level VLM precision. In fast-motion clips, faces or other visually valid attributes may appear for only a few frames. Humans sometimes miss or infer these attributes, while the VLM pipeline explicitly captures them. This indicates higher recall—capturing valid instances—rather than systematic false positives.

Agreement measures also show high consistency across all 400 videos. Cohen’s Kappa, an agreement statistic, is 0.91 for gender and 0.78 for ethnicity. Fleiss’ Kappa, which measures agreement among multiple annotators, is 0.92 and 0.82, respectively. These values indicate strong to near-perfect inter-annotator agreement.

We also test whether evaluator disagreement changes the downstream results. We restrict the analysis to a high-agreement subset where human majority votes exactly match VLM predictions. Consensus rates are 99.0% for gender and 83.3% for ethnicity. Core metrics remain statistically stable: PBSG changes from 0.846 to 0.863, with \(p=0.68\), and SDI changes from 0.745 to 0.761. Here, \(p\) is the statistical test’s reported significance value, distinct from the actor symbol \(p\) in Section 3.1. The non-significant difference confirms that residual evaluator uncertainty does not distort the reported measurements.

Human review grounds and sanity-checks the automated pipeline; it does not replace large-scale evaluation. Larger human annotation would be ideal, but substantial labor costs limit its scale. This follows established practice of validating model-based judges with small, high-consensus human reviews. Together, these results support the VLM ensemble as a reliable, scalable, human-aligned method for social-attribute analysis.

These findings motivate our central question about alignment’s effects. We address it by examining image reward datasets (Section 5.1), trained reward models (Section 5.2), and the video outputs produced when those reward models guide alignment (Section 6).

## 5 Social Biases in Image Reward Datasets and Reward Models

Image-based reward models are the de facto and state-of-the-art approach to aligning video diffusion models (Wu et al., 2023; Xu et al., 2024; Li et al., 2024; Yuan et al., 2024; Liu et al., 2024a; Prabhudesai et al., 2024; Black et al., 2023; Ma et al., 2025). They are trained on static images but guide learning in temporally coherent video generation. Their biases can therefore propagate, and even amplify, across frames.

We first examine demographic disparities in human-labeled image preferences. We then test trained reward models to understand how dataset biases transfer into reward preferences and may shape the temporal evolution of video attributes.

### 5.1 Preference in Image Reward Datasets

We analyze HPDv2 (Wu et al., 2023) and Pick-a-Pic (Kirstain et al., 2023b). **GPT-4o-mini**, the language model used for caption analysis, extracts gender, ethnicity, and verb attributes from captions. The three VLMs—Qwen2-VL-7B, Qwen2.5-VL-7B, and InternVL2.5-8B—classify attributes from the images. We aggregate caption and image attributes, retaining only instances containing one of our predefined verbs.

After processing, HPDv2 contains 28,783 validated **(images, caption, preference) tuples**, covering 29 verbs. Pick-a-Pic contains 14,958 such tuples across 19 verbs. Each tuple contains two images and a caption, together with a human choice of the image that best matches that caption. We measure how often annotators prefer particular gender or ethnicity representations for each verb.

**Figure 3:** Image reward datasets gender preference distribution across 42 verbs.

**Figure 4:** Image reward datasets ethnicity preference across 42 verbs.

For gender, we directly analyze pairs where the preferred image depicts one gender and the dispreferred image depicts the other—for example, a preferred man and a dispreferred woman. This captures explicit gender preference within each comparison pair.

After filtering for valid man–woman pairs, 26 out of 42 verbs in HPDv2 qualify. Of these, 69.23% (18/26) favor men, showing a consistent man-preferred tendency. In Pick-a-Pic, 18 out of 42 verbs qualify, and 61.11% (11/18) favor women, showing a relatively woman-preferred trend. Figure 3 presents the gender-preference distributions.

Table 10 and Figure 4 present ethnicity preferences. Both datasets strongly favor White representations: 43.34% in HPDv2 and 40.08% in Pick-a-Pic. East Asian and Indian representations follow. Some verbs differ from the overall trend: “bake” favors Black individuals, while “fish” favors East Asians.

These overall patterns suggest that collected preferences may implicitly favor Western-centric aesthetics and representation. Such imbalances risk propagating through reward-model training and reinforcing inequities in downstream video generation. The results highlight the need for more inclusive preference datasets that represent global demographic diversity.

### 5.2 Preference in Image Reward Models

We next examine whether reward models trained on human preference data inherit gender and ethnicity biases.

**Evaluation Benchmark Construction.** We construct a controlled benchmark using **text-to-image (T2I) generation**, which generates images from text descriptions. The design is inspired by HPDv2 (Wu et al., 2023) and **ImageRewardDB**, an image reward dataset (Xu et al., 2024).

We use the event templates from Section 3.2 and **FLUX**, a state-of-the-art T2I model (Labs, 2023), to generate image sets that systematically vary gender, ethnicity, and verb. There are two settings:

1. **Ethnicity+Person:** the prompt specifies ethnicity but uses the gender-neutral actor “person.”
2. **Ethnicity+Gender:** the prompt explicitly specifies both ethnicity and gender.

The `[ethnicity]` and `[gender]` slots below hold those actor descriptors. We generate 100 images per prompt. Table 3 reports prompt coverage, and Figure 5 shows examples.

| Settings | # of Prompts | Examples |
|---|---:|---|
| Ethnicity + Person | 294 | An East Asian person is baking cookies in a cozy kitchen, with warm light and the aroma of vanilla filling the air. |
| Ethnicity + Gender | 1176 | An East Asian woman is baking cookies in a cozy kitchen, with warm light and the aroma of vanilla filling the air. |

**Table 3:** Statistics of Benchmark Construction Prompts for Image Reward Models. Each prompt explicitly identifies the actor’s ethnicity when specified, the verb, and the surrounding context. These components provide a structured basis for analyzing social representations in image reward models.

We do not assume that FLUX is bias-free. Its role is to provide a controlled, reproducible setting for probing reward preferences. Systematically conditioning on demographic attributes isolates reward-model preferences independently of generator-specific artifacts.

To verify the benchmark, three annotators independently reviewed 100 randomly sampled images for correct attribute representation. Unanimous agreement was 77%. This is intentionally conservative: 97% of samples reached majority agreement, meaning at least 2/3 annotators confirmed the attributes. Only three images lacked consensus. These rare disagreements concerned visually ambiguous prompts rather than systematic generation failures or demographic-specific distortions. The high signal-to-noise ratio indicates that gender and ethnicity cues are sufficiently clear for reward-model evaluation. Residual label noise is unlikely to materially affect the observed preference patterns.

**Figure 5:** Image examples from our benchmark for evaluating image reward-model preferences. Generation prompts have the form “A/An [ethnicity][gender] is baking [context].” We show only images with `[gender]` \(\in\{\text{man, woman}\}\).

**Preference Bias Evaluation.** We evaluate four models:

1. **HPSv2.0** (Wu et al., 2023), trained on HPDv2.
2. **HPSv2.1** (Wu et al., 2023), trained on the unreleased HPDv2.1 dataset.
3. **PickScore** (Kirstain et al., 2023b), trained on Pick-a-Pic.
4. **CLIP** (Radford et al., 2021), the shared base model before the other three models are fine-tuned on their preference datasets.

Each model assigns a continuous scalar preference score—a numerical score rather than a discrete class label—to each generated image. We standardize scores using their mean and standard deviation, so that comparisons use scores relative to their mean and spread.

For reward models, PBSG is adapted from the count-based video metric: it compares standardized scores for male and female images. Ethnicity RDS and SDI are likewise computed from normalized reward preferences rather than video-output counts. The procedures are given below. Table 4 reports the results. Appendix C provides implementation details, analysis across 42 verbs, and statistical analysis in Table 11. Its \(\Delta\) rows compare each fine-tuned reward model with CLIP.

| Models | Average PBSG | White PBSG | White RDS | Black PBSG | Black RDS | Latino PBSG | Latino RDS | East Asian PBSG | East Asian RDS | Southeast Asian PBSG | Southeast Asian RDS | India PBSG | India RDS | Middle Eastern PBSG | Middle Eastern RDS | Overall SDI |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CLIP | -0.0726 | 0.0343 | 0.0182 | -0.1198 | 0.0002 | -0.0934 | -0.0013 | -0.1315 | 0.0141 | -0.0865 | 0.0094 | -0.0508 | -0.0299 | -0.0607 | -0.0108 | 0.8495 |
| HPSv2.0 | 0.6039 | 0.6090 | -0.0423 | 0.7341 | -0.0069 | 0.6512 | 0.0237 | 0.4752 | -0.0031 | 0.5192 | -0.0100 | 0.5922 | 0.0070 | 0.6464 | 0.0315 | 0.8492 |
| \(\Delta\) | +0.6765 | +0.5747 | -0.0605 | +0.8539 | -0.0071 | +0.7446 | +0.0250 | +0.6067 | -0.0172 | +0.6057 | -0.0194 | +0.6430 | +0.0369 | +0.7071 | +0.0423 | -0.0003 |
| HPSv2.1 | -0.0984 | -0.0833 | -0.0189 | 0.0257 | -0.0321 | -0.0031 | 0.0382 | -0.3044 | 0.0091 | -0.2181 | -0.0099 | -0.0006 | -0.0077 | -0.1053 | 0.0214 | 0.8470 |
| \(\Delta\) | -0.0258 | -0.1176 | -0.0371 | +0.1455 | -0.0323 | +0.0903 | +0.0395 | -0.1729 | -0.0050 | -0.1316 | -0.0193 | +0.0502 | +0.0222 | -0.0446 | +0.0322 | -0.0025 |
| PickScore | -0.1157 | 0.0321 | 0.0069 | -0.0777 | 0.0279 | -0.3479 | -0.0118 | -0.2257 | 0.0316 | -0.2163 | 0.0115 | 0.1531 | -0.0391 | -0.1277 | -0.0271 | 0.8483 |
| \(\Delta\) | -0.0431 | -0.0022 | -0.0113 | +0.0421 | +0.0277 | -0.2545 | -0.0105 | -0.0942 | +0.0175 | -0.1298 | +0.0021 | +0.2039 | -0.0092 | -0.0670 | -0.0163 | -0.0012 |

**Table 4:** Preference bias of different reward models. All values are average scores across 42 verbs.

**Ethnicity-Aware Gender Bias.** We distinguish the text used to generate an image from the text used to evaluate it:

- **Evaluation: Ethnicity+Person.** “A/An [ethnicity] person is [verb]-ing [context].”
- **Generation: Ethnicity+Gender.** “A/An [ethnicity] [gender] is [verb]-ing [context].”

The evaluation prompt omits gender, so the reward score reveals the model’s inherent gender preference. The generation prompt specifies gender, allowing controlled comparisons.

We standardize each model’s reward scores using their mean and standard deviation, then average the standardized scores across the 100 images for each generation prompt. For each fixed ethnicity and verb, reward-model PBSG is the average score for men minus the average score for women. This adapted score can exceed one. Positive means preference for men; negative means preference for women.

CLIP has a slight woman preference, with average PBSG = –0.0726. Fine-tuning on HPDv2 shifts HPSv2.0 to a strong man preference, +0.6039, which is consistent across ethnic groups. PickScore and HPSv2.1 instead have average woman preferences of –0.1157 and –0.0984. HPSv2.1’s training data remain undisclosed. These shifts align with the models’ training data and reveal consistent gender preferences across ethnicities. Table 4 gives all group-specific scores; Figures 19, 20, 21, 22, 23, 24, and 25 provide the 42-verb results for each group.

**Ethnicity Bias.** Here, the two prompt roles are:

- **Evaluation: person-only.** “A person is [verb]-ing [context].”
- **Generation: Ethnicity+Person.** “A/An [ethnicity] person is [verb]-ing [context].”

The evaluation prompt omits ethnicity to reveal the reward model’s inherent preference. The generation prompt specifies ethnicity for controlled comparison.

We standardize reward scores using their mean and standard deviation and average the 100 images for each generation prompt. For a fixed verb, we apply **softmax**, a function that normalizes the ethnicity scores into relative preference proportions (Bridle, 1990; Bishop, 2006). We use these proportions to calculate RDS and SDI. Positive RDS means overrepresentation in reward preference; negative means underrepresentation. Higher SDI means more balanced preferences across groups.

CLIP slightly favors White individuals, with RDS = 0.0182, and has the highest SDI, 0.8495. After fine-tuning, HPSv2.0 shifts toward Middle Eastern individuals (RDS = 0.0315), and HPSv2.1 shifts toward Latino individuals (RDS = 0.0382). PickScore shifts toward East Asian individuals, with RDS reported here as 0.0352; its East Asian RDS entry in Table 4 is 0.0316.

All fine-tuned reward models have lower SDI than CLIP: 0.8492 for HPSv2.0, 0.8470 for HPSv2.1, and 0.8483 for PickScore (Table 4). This indicates reduced ethnic diversity after alignment. Figures 26, 27, 28, and 29 show ethnicity preferences across the 42 verbs.

## 6 Social Biases in Preference Alignment

We now test how image reward-model biases affect video generation during alignment. Our base is a **Video Consistency Model distilled from VideoCrafter-V2**, abbreviated **VCM-VC2** (Li et al., 2024). We fine-tune it separately with HPSv2.0, HPSv2.1, and PickScore, then compare social-attribute distributions before and after tuning.

We follow the T2V-Turbo-V1 training protocol (Li et al., 2024). Reward feedback is incorporated into **Latent Consistency Distillation**, the distillation process used for single-step video generation (Luo et al., 2023). A student video model learns from a pretrained teacher text-to-video model. During distillation, we directly optimize the decoded video frames to maximize scores from image–text alignment reward models. Each frame is thereby guided toward representations that better match human preferences. Appendix G provides post-training and inference details.

We evaluate these aligned models with the framework used in Section 4. Table 5 reports PBSG for ethnicity-conditioned gender imbalance, RDS for ethnicity disparities, and SDI for overall diversity. A model label beginning with “+” means VCM-VC2 aligned using the named reward model. Appendix D provides further 42-verb analysis and statistical analysis in Table 12.

| Models | Average PBSG | White PBSG | White RDS | Black PBSG | Black RDS | Latino PBSG | Latino RDS | East Asian PBSG | East Asian RDS | Southeast Asian PBSG | Southeast Asian RDS | India PBSG | India RDS | Middle Eastern PBSG | Middle Eastern RDS | Overall SDI |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| VCM-VC2 | 0.8034 | 0.7925 | 0.6405 | 0.7758 | -0.2381 | 0.8090 | — | 0.7115 | -0.2333 | 0.7945 | — | 0.8634 | — | 0.8071 | -0.1690 | 0.1433 |
| + HPSv2.0 | 0.9116 | 0.9667 | 0.3667 | 0.9214 | — | 0.9667 | — | 0.8500 | — | 0.9214 | — | 0.9000 | — | 0.8548 | -0.3667 | 0.1257 |
| \(\Delta\) | +0.1082 | +0.1742 | -0.2738 | +0.1456 | — | +0.1577 | — | +0.1385 | — | +0.1269 | — | +0.0366 | — | +0.0477 | -0.1977 | -0.0176 |
| + HPSv2.1 | 0.2267 | 0.1321 | 0.4286 | 0.2381 | — | 0.3738 | — | 0.1571 | — | 0.3619 | — | 0.1452 | — | 0.1786 | -0.4286 | 0.0976 |
| \(\Delta\) | -0.5767 | -0.6604 | -0.2119 | -0.5377 | — | -0.4352 | — | -0.5544 | — | -0.4326 | — | -0.7182 | — | -0.6285 | -0.2596 | -0.0457 |
| + PickScore | 0.3714 | 0.3429 | 0.6833 | 0.3357 | -0.1810 | 0.7190 | -0.1929 | 0.1450 | -0.1952 | 0.4548 | — | 0.2500 | — | 0.3515 | -0.1143 | 0.1467 |
| \(\Delta\) | -0.4320 | -0.4496 | +0.0428 | -0.4401 | +0.0571 | -0.0900 | -0.1929 | -0.5665 | +0.0381 | -0.3397 | — | -0.6134 | — | -0.4556 | +0.0547 | +0.0034 |

**Table 5:** Social biases of aligned models. All values are average scores across 42 verbs. “—” marks cases where the model generated zero recognizable instances of the corresponding ethnicity under person-only prompts, where ethnicity is not specified. These missing values reflect a severe lack of diversity that prevents metric computation for those subgroups.

**Ethnicity-Aware Gender Bias.** We evaluate gender under Ethnicity+Person prompts. Positive PBSG means more frequent male portrayals; negative means a preference for female portrayals.

VCM-VC2 has strong man bias across ethnicities, with average PBSG = 0.8034. HPSv2.0 alignment increases it to 0.9116. In contrast, HPSv2.1 reduces it to 0.2267, and PickScore reduces it to 0.3714 (Table 5). These reductions indicate a shift toward more balanced or woman-preferred outputs. They reflect the woman bias in the guiding HPSv2.1 and PickScore reward models, which counteracts the base model’s man-dominant bias. Figures 30, 31, 32, 33, 35, 34, 36, and 37(b) show PBSG across the 42 verbs for each ethnicity.

**Ethnicity Bias.** Under the ethnicity-only condition, we evaluate representation balance using RDS and SDI. Positive RDS means overrepresentation, negative means underrepresentation, and higher SDI means more equitable overall representation.

VCM-VC2 strongly favors White individuals, with RDS = 0.6405. Black, East Asian, and Middle Eastern groups are underrepresented, with RDS values of -0.2381, -0.2333, and -0.1690 (Table 5).

HPSv2.1 alignment reduces some disparities by improving balance for White and Black groups. It significantly decreases Latino representation, reported as RDS = –0.4352, and lowers SDI, indicating less diversity. In Table 5, the HPSv2.1 White RDS is 0.4286, the Black and Latino RDS entries are “—,” and -0.4352 appears in the Latino \(\Delta\)PBSG entry. Its SDI falls from 0.1433 to 0.0976.

PickScore achieves the highest SDI, 0.1467, and produces more balanced representation across most groups, resulting in the most demographically equitable outputs. Table 5 provides the group scores; Figure 38 shows ethnicity bias across the 42 verbs.

Table 6 summarizes TAS. **Mean** and **median** describe its average and middle value across videos. **Std TAS** is its standard deviation, describing variation across videos. **Perfect Stability** is the proportion of videos whose attributes remain perfectly stable.

| Model | Attribute | Mean TAS | Median TAS | Std TAS | Perfect Stability |
|---|---|---:|---:|---:|---:|
| VCM-VC2 | Ethnicity | 0.8904 | 1.0000 | 0.1673 | 0.5670 |
| VCM-VC2 | Gender | 0.9763 | 1.0000 | 0.0829 | 0.8850 |
| + HPSv2 | Ethnicity | 0.9305<sub>+0.0401</sub> | 1.0000 | 0.1358<sub>–0.0315</sub> | 0.7050<sub>+0.1380</sub> |
| + HPSv2 | Gender | 0.9967<sub>+0.0204</sub> | 1.0000 | 0.0310<sub>–0.0519</sub> | 0.9830<sub>+0.0980</sub> |
| + HPSv2.1 | Ethnicity | 0.9632<sub>+0.0728</sub> | 1.0000 | 0.1019<sub>–0.0654</sub> | 0.8330<sub>+0.2660</sub> |
| + HPSv2.1 | Gender | 0.9895<sub>+0.0132</sub> | 1.0000 | 0.0519<sub>–0.0310</sub> | 0.9440<sub>+0.0590</sub> |
| + PickScore | Ethnicity | 0.9406<sub>+0.0502</sub> | 1.0000 | 0.1302<sub>–0.0371</sub> | 0.7560<sub>+0.1890</sub> |
| + PickScore | Gender | 0.9884<sub>+0.0121</sub> | 1.0000 | 0.0553<sub>–0.0276</sub> | 0.9410<sub>+0.0560</sub> |

**Table 6:** Temporal Attribute Stability (TAS) across models and attributes. High TAS means the actor’s identity representation remains stable and consistent throughout the video. Low TAS means that representation “flickers” or changes, a key temporal artifact. Subscripts in red and blue indicate relative improvements and degradations compared with VCM-VC2. The rows labeled “+ HPSv2” contain the stability results discussed below for HPSv2.0 alignment.

**Temporal Attribute Stability.** Alignment substantially improves technical coherence. With HPSv2.1, mean ethnicity TAS rises from 0.8904 to 0.9632. The proportion of perfectly stable videos rises by 0.2660, from 0.5670 to 0.8330. Gender stability also approaches saturation, reaching mean TAS = 0.9895 and perfect stability = 0.9440. TAS standard deviation consistently decreases; for example, the HPSv2.1 ethnicity decrease is –0.0654. These decreases indicate more uniform frame-level consistency across videos (Table 6).

However, stability can entrench bias rather than mitigate it. HPSv2.0 raises average man-preference bias from PBSG = 0.8034 to 0.9116 (Table 5). At the same time, gender TAS reaches 0.9967, and nearly all videos achieve perfect gender stability, 0.9830 (Table 6). Thus, the stronger man-preference bias is also more stable over time.

In this sense, the model becomes better at being biased: its portrayals are smoother and more coherent, but also more stereotyped. Our temporal evaluation framework uniquely reveals that alignment can inadvertently make social bias more persistent and deeply embedded even while improving perceptual and temporal quality. VideoBiasEval detects not only whether a model is biased, but also whether alignment makes that bias temporally resilient, turning representational artifacts into stable, systematic distortions.

**Summary and Implications.** Preference alignment can mitigate or amplify existing biases, depending on the reward model. HPSv2.0 reinforces the base model’s male- and White-preferred tendencies. Its gender PBSG rises from 0.8034 to 0.9116; Table 5 also reports White RDS changing from 0.6405 to 0.3667. HPSv2.1 and PickScore, which have woman-preferred reward patterns, counteract the original man-dominant bias and move portrayals toward balance.

Neither achieves complete demographic parity. Ethnic disparities remain. Together, Tables 4–6 suggest that the direction and magnitude of bias in aligned video models are largely inherited from the guiding reward model’s bias profile.

## 7 Controllable Preference Modeling for Video Diffusion Models

We next ask whether these biases can be controlled. Specifically, can changes to social-attribute distributions in image preference datasets systematically steer reward models, and then video models, toward more equitable or intentionally calibrated portrayals (Sheng et al., 2020)?

We use gender as a case study. We vary the gender composition of preference data and examine the resulting reward preferences and video-alignment behavior.

### 7.1 Image Reward Dataset Construction

We construct a man-preferred dataset and a woman-preferred dataset using images from Section 5.1. Each contains 2.94 million preference pairs from the Ethnicity+Gender set. Within each pair, action and ethnicity are the same, but gender differs. For example, Figure 5’s **M-1** and **W-1** identify a male-image example and a female-image example that can be paired.

The reward prompt is:

> A/An [ethnicity] person is [action]-ing [context].

Here, `[action]` refers to the verb, using the terminology of this section.

In the man-preferred dataset, male images receive label 1 and female images label 0. The woman-preferred dataset reverses these labels. The labels encode the intended preference direction. To enhance face-free diversity, we also include 537,660 additional image pairs from HPDv2.

When applied to a man-biased base model, the woman-preferred dataset helps correct that imbalance and promotes more equitable gender representation, as shown by the lower PBSG values in Table 8.

### 7.2 Image Reward Model Development & Alignment Tuning

We fine-tune two reward models from a pretrained CLIP vision encoder, the component that represents visual inputs:

- **Man-Preferred Reward Model (RMM)**, also written \(\mathrm{RM}_{\mathrm M}\), trained on man-preferred data.
- **Woman-Preferred Reward Model (RMW)**, also written \(\mathrm{RM}_{\mathrm W}\), trained on woman-preferred data.

Each model learns the gender-specific aesthetic and representational preferences in its dataset. Table 7 reports PBSG, using the reward-score adaptation from Section 5.2. RMM has positive scores across groups, whereas RMW has negative scores. This divergence shows that reward tuning captures and amplifies gendered preferences. Appendix F provides training and inference details. The attached signed annotations are retained in Table 7.

| Models | Average | White | Black | Latino | East Asian | Southeast Asian | India | Middle Eastern |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CLIP | -0.0726 | 0.0343 | -0.1198 | -0.0934 | -0.1315 | -0.0865 | -0.0508 | -0.0607 |
| RMM | 1.5280<sub>+1.60</sub> | 1.6300<sub>+1.60</sub> | 1.5752<sub>+1.70</sub> | 1.5524<sub>+1.65</sub> | 1.4323<sub>+1.56</sub> | 1.4525<sub>+1.54</sub> | 1.5619<sub>+1.61</sub> | 1.4914<sub>+1.55</sub> |
| RMW | -0.7448<sub>-2.27</sub> | -0.6318<sub>-2.26</sub> | -0.7943<sub>-2.37</sub> | -0.8279<sub>-2.38</sub> | -0.6282<sub>-2.06</sub> | -0.6429<sub>-2.10</sub> | -0.8846<sub>-2.45</sub> | -0.8042<sub>-2.30</sub> |

**Table 7:** Preference bias of reward models. All values are average scores across 42 verbs.

We intentionally do not train a “fair,” or demographically neutral, reward model. VCM-VC2 already has pronounced man bias, as discussed in Section 6. Under this asymmetric initialization, a neutral reward would merely reinforce the imbalance rather than correct it. We therefore use a deliberately **counter-biased reward model**: the woman-preferred RMW, analogous to HPSv2.1. Its preference opposes the base model’s bias and actively steers generation toward gender equilibrium. Table 8’s reduction from average PBSG = 0.8034 to 0.3082 shows markedly more balanced portrayals. Directional reward tuning can therefore act as a corrective mechanism.

We also align VCM-VC2 with RMM using the same preference-optimization framework. RMM increases PBSG across all groups, reinforcing man preference. RMW substantially reduces every group’s score, indicating a strong shift toward woman-preference bias (Table 8). These results show fine-grained control over gender bias and provide a flexible way to amplify or mitigate social tendencies in diffusion models.

The most balanced aligned generator can also be framed as a data-composition problem. By systematically varying the proportions of man- and woman-preferred samples, one can identify an optimal mixture that minimizes overall bias while preserving aesthetic fidelity. This is a promising direction for dataset-driven bias control. Figures 39, 40, 41, 42, and 43 show PBSG across 42 actions for each ethnicity.

| Models | Average | White | Black | Latino | East Asian | Southeast Asian | India | Middle Eastern |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| VCM-VC2 | 0.8034 | 0.7925 | 0.7758 | 0.8690 | 0.7115 | 0.7945 | 0.8634 | 0.8071 |
| + RMM | 0.9584<sub>+0.16</sub> | 0.9595<sub>+0.17</sub> | 0.9524<sub>+0.18</sub> | 0.9756<sub>+0.11</sub> | 0.9437<sub>+0.23</sub> | 0.9447<sub>+0.15</sub> | 0.9640<sub>+0.10</sub> | 0.9709<sub>+0.16</sub> |
| + RMW | 0.3082<sub>-0.50</sub> | 0.3341<sub>-0.46</sub> | 0.3913<sub>-0.38</sub> | 0.3314<sub>-0.54</sub> | 0.1008<sub>-0.61</sub> | 0.2639<sub>-0.53</sub> | 0.3446<sub>-0.52</sub> | 0.3894<sub>-0.42</sub> |

**Table 8:** Social biases of aligned models. All values are average scores across 42 verbs.

**Summary and Implications.** Social-attribute-aware preference datasets can systematically direct bias in video models. Changing gender composition causes reward models to learn directional preferences, which aligned video models then reproduce. RMM reinforces the base model’s tendency, raising average video PBSG from 0.8034 to 0.9584. Counter-biased RMW lowers it to 0.3082, producing more balanced and socially representative generations (Table 8).

Beyond the two preference directions, these results suggest a continuous alignment axis. Mixtures of man- and woman-preferred data yield tunable trade-offs between bias mitigation and aesthetic coherence. The 42-action sensitivity analysis also shows that semantically neutral activities can undergo large gender shifts, making event-level auditing important. Dataset composition, rather than architectural intervention alone, is therefore a powerful and interpretable lever for steering generative models’ social behavior.

**(a)** \(\Delta\)PBSG of the video-generation model before and after alignment with \(\mathrm{RM}_{\mathrm M}\) and \(\mathrm{RM}_{\mathrm W}\), broken down by action. Figures 6(b) and 6(c) are based on this figure.

**(b)** Sensitive actions in man-preferred alignment tuning.

**(c)** Sensitive actions in woman-preferred alignment tuning.

**Figure 6:** Action-level impact of alignment tuning guided by \(\mathrm{RM}_{\mathrm M}\) and \(\mathrm{RM}_{\mathrm W}\).

**Which Actions Are Most Sensitive During Alignment Tuning?** We measure action-specific changes in PBSG before and after gender-oriented reward tuning. Figure 6(a) shows distinct action clusters for each reward model. RMM-guided alignment, shown in teal, amplifies male-preferred portrayals for activities such as exercise, row, and cook. RMW-guided alignment, shown in orange, shifts portrayals toward women for activities such as bake, sleep, and sweep.

Figures 6(b) and 6(c) rank actions by **normalized sensitivity**: \(\Delta\)PBSG divided by the guiding reward model’s baseline PBSG. Sleep, stretch, and read rank highest under both man- and woman-preferred tuning. Thus, even socially neutral or domestic activities can exhibit pronounced gender bias after alignment. These results show systematic, action-specific bias amplification and demonstrate how event-centric evaluation reveals fine-grained behavioral sensitivity.

**Responsible Use of Controllable Preference Modeling.** Controllable preference modeling can steer social-attribute distributions, but technical “debiasing” is not a silver bullet. Unilateral application carries significant risks. The technique should support, not replace, input from diverse stakeholders.

“Equitable representation” is a context-dependent normative decision. It requires active community consultation. Practitioners should work with affected communities to define representation goals rather than relying solely on automated optimization to resolve complex social biases.

## 8 Conclusion

We identify and investigate blind spots in evaluating social bias in text-to-video generation. VideoBiasEval separates identity attributes from content semantics and systematically tracks representation changes caused by alignment.

Our analyses show that reward-model-based alignment inherits and frequently amplifies biases encoded in human preference data. The dataset results in Section 5.1, reward preferences in Table 4, aligned-video distributions in Table 5, and temporal stability in Table 6 trace this process across the pipeline. These findings highlight the need to audit and mitigate bias at every stage of video generation to support more equitable and socially aware systems.

## Limitations

Several limitations remain.

First, we evaluate only gender and ethnicity using predefined categories drawn from U.S. Census conventions and prior literature. These categories are practical for controlled evaluation but socially constructed. They cannot fully capture identity fluidity, intersectionality, or cultural nuance (Yin et al., 2024; Qiu et al., 2024a; Qiu et al., 2025). Future work should use richer identity representations, including intersectional groups.

Second, our VLM evaluators use image-level classification. Even though we validate them against human judgments, they may have their own biases or inaccuracies, especially for stylized or ambiguous frames. Ensembling reduces this problem, but larger and more diverse ground-truth video annotations would strengthen measurement reliability.

Third, we mainly study one training strategy—single-step latent consistency distillation—and a limited set of reward models. Other protocols, including reinforcement-learning-based (RL-based) tuning or multi-turn video instruction alignment, may produce bias dynamics that this study does not capture.

Fourth, controllable preference modeling is tested only through synthetic manipulation of gender preferences. These experiments establish the feasibility of targeted bias control, but they do not resolve broader questions about value alignment, normative appropriateness, or long-term societal effects. Those questions are essential for responsible deployment.

Lastly, VideoBiasEval uses a fixed set of 42 socially associated actions. This supports fine-grained control but may limit generalization to open-ended generation or novel actions outside the taxonomy. We hope these limitations encourage holistic, culturally grounded, and ethically aligned evaluation pipelines for video generation.

## Broader Impact Statement

Video diffusion models are moving from research prototypes into widely deployed content-creation tools. Their influence on visual culture and social representation is therefore growing. Our work highlights an often-overlooked consequence: alignment can stabilize and entrench social bias.

**Stabilization of Bias.** Alignment tuning, including RLHF, is essential for improving visual fidelity and temporal consistency. Yet our findings show that it can act as a “bias stabilizer.” Removing artifacts and smoothing temporal flickers can make stereotyped portrayals look more natural, coherent, and authoritative. For example, HPSv2.0 alignment strengthens man-preference bias while making gender portrayals almost perfectly stable (Tables 5 and 6).

These polished biases may be harder to recognize as failures than obvious errors in unaligned outputs. This risks normalizing skewed demographic portrayals in synthetic media. Left unchecked, stabilization could reinforce harmful stereotypes through content that appears high-quality and preferred.

**Dual-Use of Controllable Preference Modeling.** Section 7 shows that curated preference datasets can systematically amplify or suppress social attributes. This is a dual-use capability: the same mechanism can correct historical underrepresentation or enable harmful exclusion.

Controllable rewards offer a powerful means to mitigate bias. They can also be misused to exclude groups, enforce homogenization, or generate propaganda aligned with exclusionary worldviews. Alignment is therefore a double-edged tool.

**Mitigation and Responsibility.** Aesthetic metrics alone are insufficient. Developers must adopt multi-granular, event-centric auditing frameworks such as VideoBiasEval to continually monitor who appears in generated content.

Preference-dataset curation deserves the same rigor as architecture design. We urge researchers and practitioners to ensure that the human preferences guiding these systems reflect a diverse and equitable society rather than reinforcing existing prejudices.

## Acknowledgment

We thank the anonymous reviewers for their thoughtful feedback and suggestions. This research was supported in part by the AFOSR MURI grant #FA9550-22-1-0380, the Office of Naval Research (ONR) under Award #N00014-23-1-2780, and the National Science Foundation under Award #IIS-2449768. Additional support was provided through cloud credits from the NVIDIA Academic Grant Program and funding from the Technical AI Safety Research Program at Coefficient Giving. The views and conclusions expressed in this work are those of the authors and should not be interpreted as representing the official policies or endorsements of AFOSR, ONR, the U.S. Government, the National Science Foundation, NVIDIA, or Coefficient Giving.
