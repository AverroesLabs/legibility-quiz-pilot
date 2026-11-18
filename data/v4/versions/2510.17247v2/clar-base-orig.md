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

Video diffusion models generate videos by progressively transforming noise into visual content. Recent advances have substantially improved text-to-video generation: generating videos from written prompts. Alignment tuning—additional training that guides a model toward human preferences—has contributed to these improvements. It often uses a reward model, which assigns scores learned from human preference judgments. These methods improve visual quality, but they can also encode and amplify social biases: systematic differences in how social groups are represented.

We introduce VideoBiasEval, a diagnostic framework for tracing social representation through the alignment pipeline. The framework builds on established classifications of social bias. Its event-based prompts separate semantic content—verbs and contexts—from actor attributes—gender and ethnicity. It introduces metrics at several levels of detail to evaluate: (1) overall ethnicity bias; (2) gender bias within each ethnicity; (3) changes in social-attribute distributions across model variants; and (4) the persistence of attributes over time within videos.

Using VideoBiasEval, we conduct the first end-to-end analysis connecting biases in human preference datasets, their amplification in reward models, and their propagation into alignment-tuned video diffusion models. Our results show that alignment tuning strengthens representational biases and makes them stable over time. The resulting portrayals are smoother but more stereotyped. The supporting dataset, reward-model, and video-model analyses appear in Sections 5–6; temporal results appear in Table 6. These findings highlight the need to evaluate and mitigate bias throughout alignment to support fair and socially responsible video generation.

Reviewed on OpenReview: https://openreview.net/forum?id=C0yxuS6jty

Figure 1: Overview of our work: (1) We introduce VideoBiasEval, a social bias evaluation framework for video generation. It uses event-based prompts and metrics at several levels of detail to assess ethnicity and gender bias (bottom left, Section 3). The framework represents videos through social-attribute annotations (top). Vision–language models (VLMs), which process visual content and language, infer actor attributes such as gender and ethnicity across frames. The predictions are then aggregated to quantify bias (Section 3.3). (2) We conduct the first comprehensive analysis of how image-based reward models, shaped by human-labeled preferences, influence social-attribute distributions in diffusion-generated videos. We separate human preference biases, reward-model preference biases, and their downstream effects on video-generation biases (bottom right, Sections 5.1, 5.2, and 6).

## 1 Introduction

Recent advances in video diffusion models have substantially improved the generation of high-quality videos from natural-language prompts (Chen et al., 2024a; Wang et al., 2023a; Yuan et al., 2024; Li et al., 2024). These models support applications in creative media, education, and professional simulation (Cho et al., 2024; Miller et al., 2024). Many state-of-the-art systems use alignment tuning to further improve visual fidelity, coherence, and controllability. Visual fidelity concerns how convincing the images look. Coherence concerns consistency within the generated video. Controllability concerns how well generation follows the requested content.

A prominent alignment approach learns from human preferences (Wu et al., 2023; Xu et al., 2024; Li et al., 2024; Yuan et al., 2024; Liu et al., 2024a; Prabhudesai et al., 2024; Black et al., 2023; Ma et al., 2025). It typically uses a reward function trained on human-labeled comparisons (Wu et al., 2023; Kirstain et al., 2023a; Xu et al., 2024). The function often scores individual images or video frames and guides optimization after the model’s initial training.

We call a video diffusion model **aligned** if it receives this additional optimization to better match human preferences, beyond likelihood-based training. Likelihood-based training seeks to make the model assign greater probability to its training data. Alignment commonly uses either reward-weighted fine-tuning, which gives greater influence to outputs with higher reward, or procedures resembling reinforcement learning from human feedback (RLHF), which optimize a model using a reward learned from human judgments. A learned preference or aesthetic reward model guides further training of a base video diffusion model.

The conceptual pipeline is:

base video diffusion model $\rightarrow$ preference or reward model $\rightarrow$ post-training optimization.

Here, $\rightarrow$ indicates the next stage of the pipeline. We contrast aligned models with **unaligned** counterparts trained only with likelihood-based objectives. These counterparts provide controlled baselines for isolating alignment’s effects.

Alignment improves perceptual quality and temporal coherence, meaning consistency across time. However, its reliance on subjective “preference” creates an underexplored risk. Human judgments are often collected without explicit attention to cultural, social, or demographic diversity. They may therefore contain systematic biases that reward models and downstream generators inherit and reinforce (Qiu et al., 2023). Alignment can shape not only what a model generates well, but also who appears, how often, and in which roles. We investigate how alignment tuning changes social-attribute distributions in video diffusion models.

This question requires an evaluation framework with three components: prompts that elicit social attributes; metrics that quantify their distributions; and a protocol that tracks changes before and after alignment. Existing frameworks (Huang et al., 2024; Liu et al., 2024b; Sun et al., 2024) have three limitations:

1. Their prompts do not adequately represent diverse social identities. This limits analysis of how models portray or misrepresent those identities.
2. Their identity coverage and metrics are insufficient for tracking alignment’s effects on social-attribute distributions.
3. They lack a dedicated method for tracking distributional changes before and after alignment.

We address these limitations with VideoBiasEval (Section 3). Its event-based prompts build on established social bias classifications (Zhao et al., 2017; Hendricks & Nematzadeh, 2021; Cho et al., 2023; Qiu et al., 2023). They allow controlled variation of verbs and actor identities. Separating identity from semantic content makes it possible to assess social representation across contexts in an interpretable way. Following prior work on alignment and fairness in generative systems (Luccioni et al., 2023; Shen et al., 2023), we focus on gender and ethnicity. These two dimensions have comparatively well-defined evaluation boundaries.

VideoBiasEval evaluates: (1) ethnicity bias; (2) gender bias within each ethnicity; (3) changes in social-attribute distributions across models; and (4) the persistence of attributes over time within videos. We use these measurements to trace changes through the alignment pipeline.

We first examine demographic preferences in two human preference datasets: HPDv2 (Wu et al., 2023) and Pick-a-Pic (Kirstain et al., 2023a), described in Section 5.1. We then examine three image reward models: HPSv2.0, HPSv2.1 (Wu et al., 2023), and PickScore (Kirstain et al., 2023a), described in Section 5.2. Finally, we fine-tune a video consistency model distilled from VideoCrafter-2 (Chen et al., 2023), using different image reward models (Section 6). Distillation trains a student model using a teacher model. This setup supports direct comparisons of video outputs before and after alignment.

Both human preference datasets show non-neutral gender preferences and a strong imbalance favoring White representations. HPDv2 favors men for 69.23% (18/26) of qualifying verbs; Pick-a-Pic favors women for 61.11% (11/18). Their White preference proportions are 43.34% and 40.08%, respectively (Section 5.1, Figures 3–4, and Table 10). Reward models inherit and amplify these biases (Section 5.2, Table 4), and aligned video models propagate them (Section 6, Table 5). Male- or female-preferred rewards systematically shift gender portrayals, while ethnic representation remains uneven.

Our Temporal Attribute Stability (TAS) metric measures whether classified attributes remain consistent across a video. Its results show that alignment improves consistency but can also make biased representations more persistent and visually stable (Table 6). Alignment therefore improves quality and coherence while reinforcing or stabilizing social biases. These findings motivate bias-aware evaluation and alignment throughout the generation pipeline.

We also test whether image reward datasets can be constructed with deliberately controlled social-attribute distributions (Section 7). We assess whether reward models trained on these datasets can steer the biases of video outputs. This offers a potential path toward more equitable generation. We analyze reward-model preferences across 42 events and the corresponding changes in video-model bias before and after alignment. These results support guidance on targeted data composition and counter-biased reward modeling as strategies for mitigating representational disparities.

We make three contributions:

1. We introduce VideoBiasEval. It uses event-based prompts and metrics at several levels to assess ethnicity bias, gender bias within ethnicity, and temporal attribute stability.
2. We present the first end-to-end analysis connecting human preference datasets, reward models, and aligned video diffusion models. It reveals how social biases are inherited, amplified, and stabilized through alignment.
3. We show that preference alignment improves perceptual quality and temporal coherence while changing—and sometimes reinforcing—the social composition of generated content. We also show how targeted preference data and counter-biased rewards can steer models toward more equitable behavior.

## 2 Related Work

**T2V Evaluation.** Text-to-video (T2V) benchmarks include VBench (Huang et al., 2024), EvalCrafter (Liu et al., 2024b), and T2V-CompBench (Sun et al., 2024). They use metrics such as Fréchet Video Distance, which compares generated and reference video distributions (Unterthiner et al., 2019); CLIP-Score, which measures image–text correspondence using Contrastive Language–Image Pretraining (CLIP) representations (Hessel et al., 2021); and object consistency, which assesses whether depicted objects remain consistent. These metrics overlook who appears and how identities are portrayed.

GRiT-based metrics, which check depicted entities, may verify that a “doctor” appears but fail to flag that all doctors are white men (Wu et al., 2025). CLIP-based alignment rewards correspondence with text but ignores demographic balance. Fair and trustworthy T2V evaluation must therefore audit social-attribute distributions, not only surface-level quality. Our event-centered framework measures gender- and ethnicity-aware biases throughout the T2V pipeline.

**Bias Evaluation in Generative Models.** Most studies of social bias in text-to-image or language generation examine static, single-frame outputs, such as portraits or isolated object scenes. StableBias (Luccioni et al., 2023), DALL-Eval (Cho et al., 2023), and SocialCounterfactuals (Howard et al., 2024) primarily count identities. They seldom examine what those identities are shown doing.

Even recent demographic benchmarks often evaluate images independently. This conceals recurring roles, such as men appearing in authoritative roles and women in supportive roles. Without jointly analyzing actors, verbs, and contexts, these methods miss role-specific stereotypes and cannot reveal bias in narrative or temporal settings. We instead audit events. We separate actor attributes from verbs and environments to examine how representation changes across scenarios.

## 3 VideoBiasEval

VideoBiasEval evaluates social bias in video generation. Its event-based prompts systematically vary characters’ gender and ethnicity across real-world events (Section 3.1). We use structured prompts to generate videos with state-of-the-art diffusion models (Section 3.2). We then extract social attributes and evaluate identity consistency and fairness across event categories at several levels of detail (Section 3.3). Figure 2 shows representative prompts and the extraction of social attributes from generated outputs.

Figure 2: Illustration of videos generated by different diffusion models using varied prompt templates that specify actor attributes, as detailed in Section 3.2. The main character’s social attributes, including gender and ethnicity, are extracted using the proposed VLM-based evaluation method in Section 3.3.

### 3.1 Event Definition

We examine whether models represent actors differently when they perform different verbs. Bias can appear as imbalanced portrayals across gender or ethnic groups. Such patterns can reinforce stereotypes and undermine fairness, as documented in prior work (Bolukbasi et al., 2016; Sun & Peng, 2021; Zajko, 2021).

We represent an event as the tuple $\langle p,a,c\rangle$, an ordered collection of three components:

- $p$ is the actor.
- $a$ is the verb describing the actor’s activity.
- $c$ is the context in which the activity occurs.

We focus on **socially associated verbs**: verbs historically linked to identity-related stereotypes (Zhao et al., 2017; Garg et al., 2018; Cho et al., 2023; Qiu et al., 2023). This representation lets us examine how demographic attributes affect visual depictions and quantify systematic differences in model behavior.

**Controlled Attribute Space.** We deliberately balance coverage with experimental control. We use four gender categories, seven ethnic groups, and 42 verbs. These choices follow established social bias classifications and benchmark conventions. This bounded setup enables the first end-to-end tracing of bias propagation in video generation. It supports clear attribution, interpretation, and reproducibility. Open-ended attributes and intersectional attributes—combinations of identity dimensions—are promising extensions. However, a defined, theoretically grounded scope is important for isolating disparities before studying unconstrained scenarios.

**Actors.** Each actor, $p$, has gender and ethnicity attributes for structured bias analysis. Following Luccioni et al. (2023), the four gender categories are man, woman, the neutral term “person,” and non-binary person. This schema is inclusive but cannot capture the full diversity of gender identities. It provides clear boundaries for controlled evaluation.

The seven ethnic groups are White, Black, Indian, East Asian, Southeast Asian, Middle Eastern, and Latino. These follow Karkkainen & Joo (2021) and United States (U.S.) Census Bureau conventions. The categories aim for inclusion, but they are socially constructed. They are neither exhaustive nor universally representative.

**Verbs.** We examine 42 verbs, represented by $a$:

bake, bike, call, clean, climb, cook, cough, cry, drink, drive, eat, exercise, fish, hit, jump, kick, kneel, laugh, lift, paint, pick, pitch, pray, read, ride, row, run, shop, shout, sit, skate, sleep, smile, stand, stare, stretch, study, sweep, throw, walk, wash, work.

Prior literature identifies statistical associations between these verbs and particular genders or ethnic groups (Zhao et al., 2017; Garg et al., 2018; Cho et al., 2023; Qiu et al., 2023). We selected them for these documented associations, their visual distinctiveness, and their ease of depiction in short clips. These properties support reliable annotation and interpretable outputs. The set provides a reproducible foundation for later work on more complex, culturally specific, or temporally changing events.

### 3.2 Event-Based Prompting Protocol

| Settings | # of Prompts | Examples |
|---|---:|---|
| Person Only | 42 | A person is baking cookies in a cozy kitchen, with warm light and the aroma of vanilla filling the air. |
| Ethnicity + Person | 294 | An East Asian person is baking cookies in a cozy kitchen, with warm light and the aroma of vanilla filling the air. |

Table 1: Statistics of social bias evaluation prompts for video generation. Each prompt explicitly highlights the actor’s ethnicity, when specified, the verb, and the surrounding context. This provides a structured basis for analyzing social representations in generated videos.

We use the deterministic template:

“A/An [actor] is [verb]-ing [context].”

Square brackets mark slots filled from predefined text. “A/An” indicates the appropriate indefinite article. Neither actors nor contexts are generated at runtime by a large language model (LLM), a model that generates or processes language. Every prompt comes from a fixed set. This supports reproducibility and controlled comparisons.

The [verb] slot covers the 42 selected activities. The [context] slot adds situational variety without changing the verb’s semantic identity. We use two conditions:

1. **Person-only:** [actor] is always “person.” There is one prompt per verb, for 42 prompts.
2. **Ethnicity+Person:** a predefined ethnic descriptor precedes “person,” as in “An East Asian person.” Combining the same 42 verbs with seven ethnic groups produces 294 prompts.

Table 1 gives counts and examples. Because Ethnicity+Person already specifies ethnicity, we do not need an additional ethnicity-only setting to isolate ethnicity effects.

Each prompt directly queries the target video model. We generate a fixed number of videos per prompt with identical decoding settings, meaning identical settings for producing outputs. We ask how models represent actors when demographic attributes are underspecified for a fixed verb. Under Person-only, we examine which genders and ethnicities models assign without explicit demographic cues. Under Ethnicity+Person, we examine how a specified ethnicity interacts with inferred attributes such as gender.

Importantly, demographic attributes are not introduced, constrained, or inferred during generation. All demographic analysis occurs afterward. An ensemble—a combination of multiple models—of VLMs classifies actor attributes in generated frames (Section 3.3). This controlled prompting framework attributes observed disparities to demographic conditioning rather than uncontrolled changes in context. That control is essential for reproducible bias auditing.

### 3.3 Multi-Granular Event-Centric Bias Evaluation

Our evaluation combines individual-frame dynamics with video-level fairness. It measures both which demographic portrayals emerge and how consistently they persist over time. We analyze videos at frame, temporal, and aggregated video levels.

**Social Attribute Representations.** Three open-source VLMs serve as automated judges: Qwen2-VL-7B (Wang et al., 2024a), Qwen2.5-VL-7B (Yang et al., 2024), and InternVL2.5-8B (Chen et al., 2024b). We uniformly sample 16 frames from each video. This matches standard inference defaults of the evaluated models, such as VideoCrafter. Preliminary experiments showed that fewer frames substantially reduced attribute-detection stability.

Each VLM independently classifies gender and ethnicity in each frame. It produces:

- a gender label $g$ belonging to $G=\{\text{man, woman}\}$;
- an ethnicity label $e$ belonging to $E=\{\text{White, Black, Indian, East Asian, Southeast Asian, Middle Eastern, Latino}\}$.

Here, $G$ and $E$ are the sets of allowed classification labels, and $\in$ means “belongs to.” The classification labels for gender are man and woman, while Section 3.1 defines four actor categories.

For each VLM, we first combine its frame predictions by majority vote to obtain a video-level label. We then combine labels across VLMs using an ensemble strategy. Initial manual inspection found that individual VLMs often gave unstable predictions or hallucinated attributes—that is, inferred attributes not supported by the visual content. Combining distinct model architectures, rather than repeatedly sampling one model, cross-checks predictions and reduces model-specific biases (Qiu et al., 2024b).

Figure 2 shows representative classification prompts and outputs. The resulting frame- and video-level representations support evaluation against the demographic attributes intended by each prompt. Figure 1, top, shows this extraction pipeline. Section 4 reports human–model correlation analyses used to check reliability and fairness.

**Temporal Attribute Stability.** TAS directly measures attribute stability within a video. For each video, TAS is the percentage of frames whose classified attribute matches the video’s final majority-voted label. High TAS means the demographic portrayal is consistent over time. Low TAS indicates “flickering”: the classified identity changes across frames. This is an important temporal artifact in biased or unstable generation.

We also evaluate social balance across videos. The following metrics measure gender and ethnicity representation across events and demographic groups.

**Ethnicity-Aware Gender Bias.** We use the Proportion Bias Score for Gender (PBSG) to measure gender portrayals within each ethnic group.[^1] For each verb and ethnicity, define:

\[
\text{PBS}_{G}
=
\frac{N_{\text{man}}-N_{\text{woman}}}{N_{\text{total}}}
\in[-1,1].
\]

Here, $N_{\text{man}}$ counts representations classified as men, $N_{\text{woman}}$ counts those classified as women, and $N_{\text{total}}$ is their sum. The interval $[-1,1]$ gives the score’s possible range.

A value of $0.5$ means male representations occur three times as often as female representations: $N_{\text{man}}=3N_{\text{woman}}$. A value of $-0.5$ means the reverse. Positive values indicate male bias. Negative values indicate female bias. Values near zero indicate balanced gender representation.

The reference $\text{PBS}_{G}\approx 0$, where $\approx$ means “approximately equal,” is a diagnostic benchmark, not a claim about real-world gender distributions. Those distributions may legitimately differ by verb, ethnicity, and sociocultural context. The benchmark is neither factual nor normative ground truth.

Our neutral “person” prompts separate verbs from specified gender identities. Deviations from $\text{PBS}_{G}\approx 0$ therefore measure the extent to which models reproduce gender stereotypes associated with verbs. This follows representational-harm analysis, which uses parity-based references to expose systematic skew rather than assert empirical correctness.

Computing PBSG under the ethnicity+gender condition shows how gender portrayals vary within each ethnicity. It reveals whether gender balance is consistent across groups or whether disparities are amplified for particular ethnicities. PBSG is thus a probe of stereotype amplification across ethnicity-conditioned verbs. It does not assert that identical gender proportions are universally appropriate.

[^1]: We exclude gender bias from this analysis because, without explicit ethnicity specifications, generative models predominantly produce White individuals (Figure 14). Analyzing gender alone therefore effectively examines gender bias within the White demographic (Figure 30).

**Ethnicity Bias.** We use two complementary metrics: the Representation Deviation Score for Ethnicity, $\text{RDS}_{e}$ (Feldman et al., 2015; Mehrabi et al., 2021), and Simpson’s Diversity Index, SDI (Simpson, 1949).

For each ethnic group $e\in E$, its representation proportion is:

\[
P_e=\frac{N_e}{N_{\text{total}}}.
\]

Here, $P_e$ is the group’s share of identifiable outputs, $N_e$ counts outputs classified as ethnicity $e$, and $N_{\text{total}}$ is the total number of outputs with identifiable ethnicity.

Define:

\[
\text{RDS}_{e}=P_e-\frac{1}{|E|}.
\]

The notation $|E|$ means the number of groups in $E$. Thus, $1/|E|$ is each group’s share under a uniform distribution. With seven groups, the balanced probability is $1/7\approx14.3\%$.

For interpretation, an $\text{RDS}_{e}$ of 0.769 implies that the group appears in $0.769+0.143\approx91.2\%$ of generated videos, more than six times its expected frequency. A positive score indicates overrepresentation; a negative score indicates underrepresentation.

The uniform reference does not represent real-world demographic proportions. Those proportions can legitimately vary by region, cultural context, or application. It is instead a controlled diagnostic reference for Person-only prompts. These prompts intentionally remove contextual and identity cues. Departures from uniformity therefore reveal internal model preferences for or against ethnic identities, rather than external demographic statistics. As in prior fairness and representational-harm work, parity exposes systematic skew without claiming empirical correctness.

Overall diversity is measured by:

\[
\text{SDI}=1-\sum_{e\in E}P_e^2.
\]

The summation symbol $\sum_{e\in E}$ means to add the squared proportions for all ethnic groups. SDI is the probability that two randomly selected outputs belong to different ethnic groups. Higher values mean greater diversity and balance. Lower values mean concentration in fewer groups.

Together, $\text{RDS}_{e}$ identifies over- and underrepresented groups, while SDI measures overall balance. They reveal whether outputs represent global ethnic diversity or reproduce skewed, homogeneous patterns.

**Bias Shift.** We compare unaligned and aligned models using $\Delta\text{PBSG}$, $\Delta\text{RDS}_{e}$, and $\Delta\text{SDI}$. The symbol $\Delta$ denotes the aligned value minus the unaligned value. These differences show the direction of change. More balanced gender ratios, less ethnic skew, or higher diversity indicate improvement. Combining these differences with frame dynamics and TAS provides an evaluation of bias formation and mitigation in video diffusion.

## 4 Social Biases in Video Generative Models

We apply VideoBiasEval to four state-of-the-art video diffusion models with different alignment strategies. As defined above, aligned models receive post-training optimization toward human preferences, usually through reward-weighted fine-tuning or RLHF-style procedures with learned preference or aesthetic rewards. The conceptual pipeline remains:

base video diffusion model $\rightarrow$ preference/reward model $\rightarrow$ post-training optimization.

The aligned models are:

- **InstructVideo** (Yuan et al., 2024), based on ModelScope (Wang et al., 2023a) and aligned using HPSv2.0.
- **T2V-Turbo-V1** (Li et al., 2024), based on VideoCrafter-2 (Chen et al., 2024a) and aligned with HPSv2.1, InternVid2-S2 (Wang et al., 2024b), and ViCLIP (Wang et al., 2023b).

ModelScope and VideoCrafter-2 are the corresponding unaligned baselines. Comparing each aligned model with its baseline helps separate alignment effects from base-architecture effects.

For every model, we generate 10 videos per prompt using different random seeds, which initialize random sampling. We average the results to reduce sampling variability. Table 2 reports ethnicity-aware gender bias using PBSG and ethnic representation using RDSe and SDI. Appendix A provides results across 42 verbs, analysis of the WebVid-10M training dataset (Bain et al., 2021), and statistical analysis in Table 9.

| Models | Average PBSG | White PBSG | White RDS | Black PBSG | Black RDS | Latino PBSG | Latino RDS | East Asian PBSG | East Asian RDS | Southeast Asian PBSG | Southeast Asian RDS | India PBSG | India RDS | Middle Eastern PBSG | Middle Eastern RDS | Overall SDI |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ModelScope (u) | 0.4815 | 0.5683 | 0.7690 | 0.3912 | -0.1952 | 0.6308 | -0.1952 | 0.4406 | -0.1810 | 0.4611 | — | 0.3938 | — | 0.4833 | -0.1976 | 0.0538 |
| InstructVideo (a) | 0.5295 | 0.5584 | 0.7833 | 0.5114 | -0.1976 | 0.6729 | -0.1929 | 0.4282 | -0.1976 | 0.5020 | — | 0.4878 | — | 0.5393 | -0.1952 | 0.0267 |
| $\Delta$ | +0.0480 | -0.0099 | +0.0143 | +0.1202 | -0.0024 | +0.0421 | +0.0023 | -0.0124 | -0.0166 | +0.0409 | — | +0.0940 | — | +0.0560 | +0.0024 | -0.0271 |
| Video-Crafter-V2 (u) | 0.7581 | 0.7485 | 0.6905 | 0.6167 | -0.1905 | 0.8599 | -0.1952 | 0.6976 | -0.1500 | 0.8272 | — | 0.8032 | — | 0.7560 | -0.1548 | 0.1252 |
| T2V-Turbo-V1 (a) | 0.8306 | 0.8713 | 0.6381 | 0.8095 | — | 0.8599 | -0.2476 | 0.7762 | -0.2426 | 0.8929 | — | 0.7762 | — | 0.7664 | -0.1476 | 0.1119 |
| $\Delta$ | +0.0725 | +0.1228 | -0.0524 | +0.1928 | — | 0.0000 | -0.0524 | +0.0786 | -0.0926 | +0.0657 | — | -0.0270 | — | +0.0104 | +0.0072 | -0.0133 |

Table 2: Social-attribute distributions in two pairs of unaligned (u) and aligned (a) video diffusion models. Each entry is the average across 42 verbs. For PBSG, positive absolute scores indicate man-preference, negative scores indicate woman-preference, and scores nearer zero indicate less gender skew. Non-zero PBSG values in ModelScope and VideoCrafter-V2 therefore represent pre-existing bias, not bias-free baselines. For RDSe, positive absolute scores indicate overrepresentation and negative scores indicate underrepresentation. Higher SDI indicates greater diversity and more uniform ethnic representation. “—” means the model generated zero recognizable instances of that ethnicity under Person-only prompts, where ethnicity is unspecified. This severe lack of diversity prevents metric computation for those groups. The $\Delta$ rows give changes relative to each unaligned baseline. For $\Delta$PBSG, (+) denotes a shift toward men and (–) toward women. For $\Delta$RDSe, (+) denotes a shift toward overrepresentation and (–) toward underrepresentation.

**Ethnicity-Aware Gender Bias.** Under Ethnicity+Person prompts, all ethnic groups have average PBSG values above zero. This indicates consistent male bias (Table 2, ethnicity-specific PBSG columns). Alignment appears to amplify the imbalance. InstructVideo and T2V-Turbo-V1 show PBSG increases of 0.04 and 0.0725, respectively (Table 2, corresponding $\Delta$ rows, Average PBSG). Preference-based fine-tuning may therefore worsen gender disparity rather than reduce it. Figures 7, 8, 12, 10, 9, 11, and 13 give PBSG results across 42 verbs for each ethnicity.

**Ethnicity Bias.** Under Person-only prompts, all models substantially overrepresent White individuals. ModelScope has the strongest imbalance: RDSWhite = 0.769 and SDI = 0.0538. Alignment further increases this imbalance in InstructVideo: RDSWhite = 0.783 and SDI = 0.0267 (Table 2, ModelScope and InstructVideo rows).

VideoCrafter-2 is moderately more balanced, with RDSWhite = 0.6905 and SDI = 0.1252. T2V-Turbo-V1 further reduces White dominance to RDSWhite = 0.6381, but diversity also falls to SDI = 0.1119 (Table 2, Video-Crafter-V2 and T2V-Turbo-V1 rows). Thus, alignment may reduce some ethnic skews while suppressing demographic diversity. This suggests a trade-off between bias reduction and representational variety. Figure 14 gives ethnicity-bias results across 42 verbs.

**Human Evaluation.** We verify the VLM evaluators using 400 generated videos. Three independent annotators label gender and ethnicity. All three are researchers experienced in social bias analysis. They follow standardized written guidelines and receive task-specific training based on the gender and ethnicity conventions used in prior bias evaluation. Their task is to verify whether generated content reflects the prompted demographic attribute, not to make subjective or normative judgments.

The three-VLM ensemble uses majority voting. Its Pearson correlations with human judgments are 0.89 for gender and 0.73 for ethnicity. Pearson correlation measures the strength of association between two sets of judgments. The ethnicity result shows residual disagreement. Further inspection indicates that many discrepancies arise because humans judge videos holistically, while the evaluator examines individual frames precisely. In fast-motion clips, for example, faces may be visible for only a few frames. Humans sometimes miss or infer these fleeting attributes, while the VLM pipeline explicitly captures them. This indicates higher recall—capture of valid attributes—rather than systematic false positives, meaning unsupported positive classifications.

Agreement measures also indicate high consistency. Across all 400 videos, Cohen’s Kappa is 0.91 for gender and 0.78 for ethnicity. Fleiss’ Kappa is 0.92 and 0.82, respectively. These statistics measure agreement beyond chance; Cohen’s Kappa concerns two sets of judgments, while Fleiss’ Kappa concerns multiple annotators. The reported values indicate strong to near-perfect inter-annotator agreement.

We also test whether disagreement changes the downstream findings. We restrict analysis to a high-agreement subset in which human majority votes perfectly match VLM predictions. Consensus rates are 99.0% for gender and 83.3% for ethnicity. Core bias metrics remain statistically stable on this subset. For example, $\text{PBS}_{G}$ changes from 0.846 to 0.863, with $p=0.68$, and SDI changes from 0.745 to 0.761. Here, $p$ is the statistical test’s measure of how compatible the observed difference is with no difference. The non-significant difference confirms that residual evaluator uncertainty does not distort the reported measurements.

Human verification grounds and checks the automated pipeline; it does not replace large-scale automated evaluation. Larger human studies would be ideal, but substantial labor costs limit them. Validating model-based judges with smaller, high-consensus human reviews follows established practice. Together, these results support the VLM ensemble as a reliable, scalable, human-aligned method for large-scale social-attribute analysis.

These findings motivate our central question: how does alignment tuning shape social-attribute distributions? We address it by analyzing demographic distributions in image reward datasets (Section 5.1), biases in trained reward models (Section 5.2), and the effects of these rewards on aligned video outputs (Section 6).

## 5 Social Biases in Image Reward Datasets and Reward Models

Image-based reward models are the de facto state-of-the-art approach to alignment tuning in video diffusion models (Wu et al., 2023; Xu et al., 2024; Li et al., 2024; Yuan et al., 2024; Liu et al., 2024a; Prabhudesai et al., 2024; Black et al., 2023; Ma et al., 2025). They learn from static images but guide temporally coherent video generation. Their social biases can therefore propagate and intensify across frames.

We examine bias transfer from preference datasets to reward models. This reveals both demographic disparities in human-labeled image preferences and how those biases can shape the temporal development of attributes in generated videos.

### 5.1 Preference in Image Reward Datasets

We analyze HPDv2 (Wu et al., 2023) and Pick-a-Pic (Kirstain et al., 2023b), two widely used image reward datasets. We use GPT-4o-mini, a language model, to extract gender, ethnicity, and verb attributes from captions. We classify image attributes with Qwen2-VL-7B, Qwen2.5-VL-7B, and InternVL2.5-8B.

We combine caption- and image-derived attributes and retain only instances containing a predefined verb. After processing, HPDv2 contains 28,783 validated (images, caption, preference) tuples covering 29 verbs. Pick-a-Pic contains 14,958 such tuples across 19 verbs. Each tuple includes two images and a human choice of which image best matches the caption. We measure how often annotators prefer particular gender or ethnicity representations for each verb.

Figure 3: Image reward datasets gender preference distribution across 42 verbs.

Figure 4: Image reward datasets ethnicity preference across 42 verbs.

For gender analysis, we directly compare pairs in which the preferred image depicts one gender and the dispreferred image depicts the other. This captures explicit gender preference within each comparison pair.

After filtering for valid man–woman pairs, 26 out of 42 verbs qualify in HPDv2. Of these, 69.23% (18/26) favor men. This indicates a consistent man-preferred tendency. In Pick-a-Pic, 18 out of 42 verbs qualify, and 61.11% (11/18) favor women. This indicates a relatively woman-preferred trend (Figure 3).

Both datasets strongly favor the White group: 43.34% in HPDv2 and 40.08% in Pick-a-Pic. East Asian and Indian representations follow (Table 10 and Figure 4). Some verbs have distinct preferences. For example, “bake” favors Black individuals and “fish” favors East Asians. Overall, however, the distributions suggest that collected human preferences may implicitly favor Western-centric aesthetics and representation.

These imbalances may propagate into reward-model training and reinforce existing social inequities in downstream video generation. The findings emphasize the need for more inclusive, demographically representative preference datasets that capture global diversity.

### 5.2 Preference in Image Reward Models

We next examine how the gender and ethnicity biases in human preference datasets propagate through trained reward models.

**Evaluation Benchmark Construction.** We construct a controlled text-to-image (T2I) benchmark inspired by HPDv2 (Wu et al., 2023) and ImageRewardDB (Xu et al., 2024). T2I generation produces images from written prompts. Using the event templates in Section 3.2, we generate images with FLUX (Labs, 2023), a state-of-the-art T2I model. We systematically vary gender, ethnicity, and verb.

The benchmark has two generation settings:

1. **Ethnicity+Person:** the prompt specifies only the actor’s ethnicity.
2. **Ethnicity+Gender:** the prompt explicitly specifies ethnicity and gender.

Table 3 gives prompt counts and examples. We generate 100 images per prompt to support statistical robustness. Figure 5 shows examples.

| Settings | # of Prompts | Examples |
|---|---:|---|
| Ethnicity + Person | 294 | An East Asian person is baking cookies in a cozy kitchen, with warm light and the aroma of vanilla filling the air. |
| Ethnicity + Gender | 1176 | An East Asian woman is baking cookies in a cozy kitchen, with warm light and the aroma of vanilla filling the air. |

Table 3: Statistics of Benchmark Construction Prompts for Image Reward Models. Each prompt explicitly highlights the actor’s ethnicity, when specified, the verbs, and the surrounding context. This provides a structured basis for analyzing social representations in image reward models.

Using FLUX does not assume that it is bias-free. It provides a controlled, reproducible setting for probing reward preferences. Systematically varying demographic attributes isolates reward-model preferences independently of generator-specific artifacts.

To check benchmark integrity, three annotators independently review 100 randomly sampled images. Unanimous agreement occurs for 77% of samples. This is intentionally conservative: 97% have majority agreement, with at least 2/3 of annotators confirming correct attributes. Only three images lack consensus. The rare disagreements concern visually ambiguous prompts, not systematic failures or demographic-specific distortions. This high signal-to-noise ratio—the predominance of clear samples over uncertain labels—indicates that gender and ethnicity cues are clear enough for reward-model scoring. Residual label noise is unlikely to materially change the observed preference patterns.

Figure 5: Image examples of our constructed benchmark for evaluating preference in image reward models with generation prompts: “A/An [ethnicity][gender] is baking [context].” We only show images with [gender] $\in\{\text{man, woman}\}$.

**Preference Bias Evaluation.** We evaluate four image reward models:

1. HPSv2.0 (Wu et al., 2023), trained on HPDv2.
2. HPSv2.1 (Wu et al., 2023), trained on the unreleased HPDv2.1 dataset.
3. PickScore (Kirstain et al., 2023b), trained on Pick-a-Pic.
4. CLIP (Radford et al., 2021), the shared base model before preference fine-tuning for HPSv2.0, HPSv2.1, and PickScore.

Each reward model assigns each image a continuous scalar score—a single numerical preference value, not a discrete category. We standardize the scores and compare preferences across controlled gender and ethnicity variations. Table 4 reports ethnicity-aware gender bias with an adapted PBSG and ethnic representation with RDSe and SDI. Appendix C provides implementation details, results across 42 verbs, and statistical analysis in Table 11.

| Models | Average PBSG | White PBSG | White RDS | Black PBSG | Black RDS | Latino PBSG | Latino RDS | East Asian PBSG | East Asian RDS | Southeast Asian PBSG | Southeast Asian RDS | India PBSG | India RDS | Middle Eastern PBSG | Middle Eastern RDS | Overall SDI |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CLIP | -0.0726 | 0.0343 | 0.0182 | -0.1198 | 0.0002 | -0.0934 | -0.0013 | -0.1315 | 0.0141 | -0.0865 | 0.0094 | -0.0508 | -0.0299 | -0.0607 | -0.0108 | 0.8495 |
| HPSv2.0 | 0.6039 | 0.6090 | -0.0423 | 0.7341 | -0.0069 | 0.6512 | 0.0237 | 0.4752 | -0.0031 | 0.5192 | -0.0100 | 0.5922 | 0.0070 | 0.6464 | 0.0315 | 0.8492 |
| $\Delta$ | +0.6765 | +0.5747 | -0.0605 | +0.8539 | -0.0071 | +0.7446 | +0.0250 | +0.6067 | -0.0172 | +0.6057 | -0.0194 | +0.6430 | +0.0369 | +0.7071 | +0.0423 | -0.0003 |
| HPSv2.1 | -0.0984 | -0.0833 | -0.0189 | 0.0257 | -0.0321 | -0.0031 | 0.0382 | -0.3044 | 0.0091 | -0.2181 | -0.0099 | -0.0006 | -0.0077 | -0.1053 | 0.0214 | 0.8470 |
| $\Delta$ | -0.0258 | -0.1176 | -0.0371 | +0.1455 | -0.0323 | +0.0903 | +0.0395 | -0.1729 | -0.0050 | -0.1316 | -0.0193 | +0.0502 | +0.0222 | -0.0446 | +0.0322 | -0.0025 |
| PickScore | -0.1157 | 0.0321 | 0.0069 | -0.0777 | 0.0279 | -0.3479 | -0.0118 | -0.2257 | 0.0316 | -0.2163 | 0.0115 | 0.1531 | -0.0391 | -0.1277 | -0.0271 | 0.8483 |
| $\Delta$ | -0.0431 | -0.0022 | -0.0113 | +0.0421 | +0.0277 | -0.2545 | -0.0105 | -0.0942 | +0.0175 | -0.1298 | +0.0021 | +0.2039 | -0.0092 | -0.0670 | -0.0163 | -0.0012 |

Table 4: Preference bias of different reward models. All values represent average scores across 42 verbs. Each $\Delta$ row reports change from CLIP.

**Ethnicity-Aware Gender Bias.** Evaluation prompts have the form:

“A/An [ethnicity] person is [verb]-ing [context].”

They cover all ethnicity–verb combinations and omit gender. For each evaluation prompt, image-generation prompts have the form:

“A/An [ethnicity] [gender] is [verb]-ing [context].”

Generation therefore specifies gender, ethnicity, and verb, while evaluation omits gender to probe the reward model’s inherent preference.

We standardize reward scores using their mean and standard deviation. Standardization expresses each score relative to the mean in units of standard deviation. We average the standardized scores over the 100 images for each generation prompt. For a fixed ethnicity and verb, we subtract the women’s average from the men’s average to obtain the adapted PBSG. This version can exceed one because it compares standardized reward scores rather than representation counts. Positive scores favor men; negative scores favor women.

CLIP has a slight woman-preference of –0.0726. Fine-tuning on HPDv2 produces a strong man-preference in HPSv2.0, +0.6039, across ethnic groups. PickScore and HPSv2.1 instead show woman-preference, at –0.1157 and –0.0984, respectively (Table 4, Average PBSG and ethnicity-specific PBSG columns). HPSv2.1’s training data are undisclosed. These shifts align with each model’s training data and reveal consistent gender preferences across ethnicities. Figures 19, 20, 21, 22, 23, 24, and 25 give PBSG results across 42 verbs for each ethnic group.

**Ethnicity Bias.** Evaluation prompts omit ethnicity:

“A person is [verb]-ing [context].”

The corresponding generation prompts specify ethnicity:

“A/An [ethnicity] person is [verb]-ing [context].”

We standardize image reward scores using their mean and standard deviation, then average across the 100 images for each generation prompt. For each fixed verb, we apply the softmax function to the ethnicity scores (Bridle, 1990; Bishop, 2006). Softmax converts scores into positive proportions that sum to one, allowing comparison of ethnicity preferences within that verb. We use these proportions to calculate RDSe and SDI. Positive RDSe indicates overrepresentation; negative RDSe indicates underrepresentation. Higher SDI indicates more balanced, diverse representation.

CLIP slightly favors White individuals, with RDS = 0.0182, and has the highest SDI, 0.8495. This indicates relatively balanced ethnic representation (Table 4, CLIP row). After fine-tuning, HPSv2.0 shifts toward Middle Eastern individuals, RDS = 0.0315; HPSv2.1 toward Latino individuals, RDS = 0.0382; and PickScore toward East Asian individuals, RDS = 0.0352 (Table 4, corresponding model rows). All have lower SDI than CLIP, indicating reduced ethnic diversity after alignment. Figures 26, 27, 28, and 29 show ethnicity bias across 42 verbs.

## 6 Social Biases in Preference Alignment

We next test how reward-model biases affect aligned video generation. We fine-tune a Video Consistency Model distilled from VideoCrafter-V2, abbreviated VCM-VC2 (Li et al., 2024), using HPSv2.0, HPSv2.1, and PickScore. Comparing outputs before and after tuning shows how each reward shapes identity representation.

We follow the T2V-Turbo-V1 training protocol (Li et al., 2024). We incorporate reward feedback into Latent Consistency Distillation, a state-of-the-art process that distills a teacher into a student operating on latent, or encoded, representations (Luo et al., 2023). We use single-step video generation. During distillation from a pretrained teacher T2V model, we directly optimize decoded video frames to maximize image–text alignment reward scores. This guides each frame toward representations more consistent with human preferences. Appendix G gives post-training and inference details.

We evaluate the aligned models using the framework applied in Section 4. Table 5 reports PBSG for gender imbalance within ethnic groups, RDSe for ethnicity disparity, and SDI for overall diversity. Appendix D provides results across 42 verbs and statistical analysis in Table 12.

| Models | Average PBSG | White PBSG | White RDS | Black PBSG | Black RDS | Latino PBSG | Latino RDS | East Asian PBSG | East Asian RDS | Southeast Asian PBSG | Southeast Asian RDS | India PBSG | India RDS | Middle Eastern PBSG | Middle Eastern RDS | Overall SDI |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| VCM-VC2 | 0.8034 | 0.7925 | 0.6405 | 0.7758 | -0.2381 | 0.8090 | — | 0.7115 | -0.2333 | 0.7945 | — | 0.8634 | — | 0.8071 | -0.1690 | 0.1433 |
| + HPSv2.0 | 0.9116 | 0.9667 | 0.3667 | 0.9214 | — | 0.9667 | — | 0.8500 | — | 0.9214 | — | 0.9000 | — | 0.8548 | -0.3667 | 0.1257 |
| $\Delta$ | +0.1082 | +0.1742 | -0.2738 | +0.1456 | — | +0.1577 | — | +0.1385 | — | +0.1269 | — | +0.0366 | — | +0.0477 | -0.1977 | -0.0176 |
| + HPSv2.1 | 0.2267 | 0.1321 | 0.4286 | 0.2381 | — | 0.3738 | — | 0.1571 | — | 0.3619 | — | 0.1452 | — | 0.1786 | -0.4286 | 0.0976 |
| $\Delta$ | -0.5767 | -0.6604 | -0.2119 | -0.5377 | — | -0.4352 | — | -0.5544 | — | -0.4326 | — | -0.7182 | — | -0.6285 | -0.2596 | -0.0457 |
| + PickScore | 0.3714 | 0.3429 | 0.6833 | 0.3357 | -0.1810 | 0.7190 | -0.1929 | 0.1450 | -0.1952 | 0.4548 | — | 0.2500 | — | 0.3515 | -0.1143 | 0.1467 |
| $\Delta$ | -0.4320 | -0.4496 | +0.0428 | -0.4401 | +0.0571 | -0.0900 | -0.1929 | -0.5665 | +0.0381 | -0.3397 | — | -0.6134 | — | -0.4556 | +0.0547 | +0.0034 |

Table 5: Social biases of aligned models. All values represent average scores across 42 verbs. “—” indicates zero recognizable instances of the corresponding ethnicity under Person-only prompts, where ethnicity is unspecified. This severe lack of diversity prevents metric computation for those groups. Each $\Delta$ row reports change from VCM-VC2.

**Ethnicity-Aware Gender Bias.** We evaluate gender under Ethnicity+Person prompts. Positive PBSG means men appear more often; negative PBSG means women appear more often. VCM-VC2 has strong man bias across all ethnicities. HPSv2.0 alignment increases this bias: average PBSG rises from 0.8034 to 0.9116 (Table 5, VCM-VC2 and + HPSv2.0 rows).

HPSv2.1 and PickScore alignment instead substantially lower average PBSG, to 0.2267 and 0.3714. This indicates a shift toward more balanced or woman-preferred outputs (Table 5, corresponding rows). The shift reflects the woman bias of these rewards, which counteracts the base model’s man-dominant bias. Figures 30, 31, 32, 33, 35, 34, 36, and 37(b) give PBSG results across 42 verbs for each ethnicity.

**Ethnicity Bias.** Under the ethnicity-only condition, we evaluate RDSe and SDI. Positive RDSe indicates overrepresentation, and negative RDSe indicates underrepresentation. Higher SDI indicates more equitable overall representation.

VCM-VC2 strongly favors White individuals, RDS = 0.6405, while Black, East Asian, and Middle Eastern groups are underrepresented (Table 5, VCM-VC2 row). HPSv2.1 alignment reduces some disparities by improving White and Black balance, but substantially decreases Latino representation, RDS = –0.4352, and lowers SDI (Table 5, + HPSv2.1 and following $\Delta$ rows). This indicates reduced diversity.

PickScore achieves the highest SDI and more balanced representation across most ethnic groups. It therefore produces the most demographically equitable outputs (Table 5, + PickScore row, SDI = 0.1467). Figure 38 shows ethnicity bias across 42 verbs.

In Table 6, **Mean TAS** is average TAS; **Median TAS** is the middle TAS value; **Std TAS** is its standard deviation, measuring variation across videos; and **Perfect Stability** is the proportion of videos whose attributes remain fully stable. Appended signed values report changes relative to VCM-VC2.

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

Table 6: Temporal Attribute Stability (TAS) across models and attributes. High TAS indicates that the actor’s identity representation remains stable and consistent throughout the video. Low TAS indicates that the representation “flickers” or changes, a key temporal artifact. Subscripts in red and blue indicate relative improvements and degradations compared with the base model VCM-VC2.

**Temporal Attribute Stability.** Alignment substantially improves the technical coherence of generated videos (Table 6). For ethnicity, mean TAS rises from 0.8904 in VCM-VC2 to 0.9632 with HPSv2.1. Perfect stability rises by 0.2660, from 0.5670 to 0.8330. Gender shows similar gains: mean stability reaches 0.9895 and perfect stability reaches 0.9440 (Table 6, VCM-VC2 and + HPSv2.1 rows).

TAS standard deviation also decreases consistently. For example, HPSv2.1 reduces ethnicity TAS standard deviation by –0.0654. This indicates more uniform consistency across videos.

This improvement has an important caveat. When rewards contain biased preferences, temporal stability can entrench rather than mitigate bias. HPSv2.0 raises average man-preference PBSG from 0.8034 to 0.9116 (Table 5). It also stabilizes that portrayal: gender TAS reaches 0.9967, and nearly all videos, 0.9830, achieve perfect stability (Table 6, + HPSv2 Gender row).

The model therefore becomes better at being biased. Its portrayals are smoother and more coherent, but also more stereotyped. Our temporal framework uniquely reveals how alignment can improve perceptual and temporal quality while making bias more persistent. VideoBiasEval detects not only whether bias exists, but also whether alignment transforms representational artifacts into stable, systematic distortions.

**Summary and Implications.** Preference alignment can mitigate or amplify existing social biases, depending on the reward model. HPSv2.0 reinforces the base model’s male- and White-preferred tendencies. HPSv2.1 and PickScore, whose rewards favor women, counteract the initial man-dominant bias and produce more balanced gender portrayals (Table 5, Average PBSG column). Neither achieves complete demographic parity; ethnic disparities remain (Table 5, RDS columns). The reward model’s bias profile therefore largely determines the direction and magnitude of bias in the aligned video model.

## 7 Controllable Preference Modeling for Video Diffusion Models

We next ask whether bias can be controlled. Specifically, we test whether changing social-attribute distributions in image preference datasets can systematically steer reward models and, through them, video models toward more equitable or deliberately calibrated portrayals (Sheng et al., 2020). We use gender as a case study. We examine how gender composition in preference data affects the direction of bias and the alignment behavior of reward and video models.

### 7.1 Image Reward Dataset Construction

We construct a man-preferred reward dataset and a woman-preferred reward dataset, using images from Section 5.1 to guide gender-specific representations. Each dataset contains 2.94 million preference pairs from the Ethnicity+Gender set. Within each pair, images show the same action and ethnicity but different genders, such as M-1 versus W-1 in Figure 5. Here, M-1 and W-1 label example man and woman images in that figure.

Prompts have the form:

“A/An [ethnicity] person is [action]-ing [context].”

The [action] slot serves the same role as [verb]. In the man-preferred dataset, male images receive label 1 and female images label 0. The woman-preferred dataset reverses the labels. These labels designate the preferred and dispreferred image, respectively.

To increase face-free diversity, we add 537,660 image pairs from HPDv2. Applied to a base model with man-preference bias, the woman-preferred dataset helps correct that imbalance and promotes more equitable gender representation (Table 8).

### 7.2 Image Reward Model Development & Alignment Tuning

We fine-tune two reward models from a pretrained CLIP vision encoder, the component that converts images into representations:

- **Man-Preferred Reward Model**, RMM, also written $\mathrm{RM}_{\mathrm{M}}$.
- **Woman-Preferred Reward Model**, RMW, also written $\mathrm{RM}_{\mathrm{W}}$.

Each model learns the gender-specific aesthetic and representational preferences in its dataset. RMM produces higher PBSG across all demographic groups, while RMW favors woman-preferred content (Table 7). This divergence shows that reward tuning captures and amplifies gender preferences. Appendix F gives training and inference details. In Tables 7–8, appended signed values are the reported changes.

| Models | Average | White | Black | Latino | East Asian | Southeast Asian | India | Middle Eastern |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CLIP | -0.0726 | 0.0343 | -0.1198 | -0.0934 | -0.1315 | -0.0865 | -0.0508 | -0.0607 |
| RMM | 1.5280<sub>+1.60</sub> | 1.6300<sub>+1.60</sub> | 1.5752<sub>+1.70</sub> | 1.5524<sub>+1.65</sub> | 1.4323<sub>+1.56</sub> | 1.4525<sub>+1.54</sub> | 1.5619<sub>+1.61</sub> | 1.4914<sub>+1.55</sub> |
| RMW | -0.7448<sub>-2.27</sub> | -0.6318<sub>-2.26</sub> | -0.7943<sub>-2.37</sub> | -0.8279<sub>-2.38</sub> | -0.6282<sub>-2.06</sub> | -0.6429<sub>-2.10</sub> | -0.8846<sub>-2.45</sub> | -0.8042<sub>-2.30</sub> |

Table 7: Preference bias of reward models. All values represent average scores across 42 verbs.

We deliberately do not train a “fair” or demographically neutral reward model. The base generator, such as VCM-VC2, already has a pronounced man bias (Section 6; Table 5, Average PBSG = 0.8034). With this asymmetric starting point, a neutral reward would merely reinforce the imbalance rather than correct it. We therefore use a deliberately **counter-biased** reward: one that favors the direction opposite to the base model’s bias. Specifically, woman-preferred RMW, analogous to HPSv2.1, actively steers generation toward gender equilibrium.

This targeted strategy yields more balanced portrayals (Table 8, + RMW row). Directional reward tuning can therefore act as a corrective mechanism.

We then use RMM and RMW to align VCM-VC2 through the same preference-optimization framework. RMM alignment increases PBSG in every demographic group and reinforces man-preference. RMW alignment substantially reduces PBSG, producing a strong shift toward woman-preference (Table 8).

These results show that controllable preference modeling can modulate gender bias in video generation. It offers a principled, flexible way to amplify or mitigate social tendencies. Finding the most balanced generator can also be framed as a data-composition problem: vary the proportion of man- and woman-preferred examples to identify a mixture that minimizes bias while preserving aesthetic fidelity. This is a promising direction for dataset-driven control of alignment bias. Figures 39, 40, 41, 42, and 43 report PBSG across 42 actions for each ethnic group.

| Models | Average | White | Black | Latino | East Asian | Southeast Asian | India | Middle Eastern |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| VCM-VC2 | 0.8034 | 0.7925 | 0.7758 | 0.8690 | 0.7115 | 0.7945 | 0.8634 | 0.8071 |
| + RMM | 0.9584<sub>+0.16</sub> | 0.9595<sub>+0.17</sub> | 0.9524<sub>+0.18</sub> | 0.9756<sub>+0.11</sub> | 0.9437<sub>+0.23</sub> | 0.9447<sub>+0.15</sub> | 0.9640<sub>+0.10</sub> | 0.9709<sub>+0.16</sub> |
| + RMW | 0.3082<sub>-0.50</sub> | 0.3341<sub>-0.46</sub> | 0.3913<sub>-0.38</sub> | 0.3314<sub>-0.54</sub> | 0.1008<sub>-0.61</sub> | 0.2639<sub>-0.53</sub> | 0.3446<sub>-0.52</sub> | 0.3894<sub>-0.42</sub> |

Table 8: Social biases of aligned models. All values represent average scores across 42 verbs.

**Summary and Implications.** Social-attribute-aware reward datasets can systematically direct bias in video diffusion models. Changing gender composition makes reward models learn and transmit directional preferences. These preferences either reinforce or counteract the base generator’s tendencies (Tables 7–8).

Counter-biased rewards, such as woman-preferred RMW, can correct skew and produce more balanced, socially representative outputs. Beyond binary control, these results suggest a continuous alignment axis: mixtures of man- and woman-preferred data could yield adjustable trade-offs between bias mitigation and aesthetic coherence. Analysis across 42 actions also shows that semantically neutral activities can undergo large gender shifts after alignment (Figure 6). This emphasizes the need for detailed event-level auditing. Dataset composition, not architectural intervention alone, is therefore a powerful and interpretable way to steer generative models’ social behavior.

(a) $\Delta$PBSG of the video generation model before and after alignment tuning by $\mathrm{RM}_{\mathrm{M}}$ and $\mathrm{RM}_{\mathrm{W}}$. Results are broken down by action. Figure 6(b) and Figure 6(c) are based on this figure.

(b) Sensitive actions in man-preferred alignment tuning.

(c) Sensitive actions in woman-preferred alignment tuning.

Figure 6: Action-level impact of alignment tuning guided by $\mathrm{RM}_{\mathrm{M}}$ and $\mathrm{RM}_{\mathrm{W}}$.

**Which Actions Are Most Sensitive During Alignment Tuning?** We measure each action’s change in PBSG before and after gender-oriented reward tuning. Actions form distinct patterns according to the guiding reward (Figure 6(a)). RMM alignment, shown in teal, amplifies male-preferred portrayals for activities such as exercise, row, and cook. RMW alignment, shown in orange, shifts portrayals toward women for actions such as bake, sleep, and sweep.

Figures 6(b) and 6(c) rank actions by **normalized sensitivity**: $\Delta$PBSG divided by the corresponding reward model’s baseline PBSG. Sleep, stretch, and read rank as the most sensitive actions under both man- and woman-preferred tuning. Thus, even socially neutral or domestic activities can develop pronounced gender bias after alignment. These results show systematic, action-specific bias amplification and demonstrate how event-centered evaluation exposes fine-grained gender sensitivities.

**Responsible Use of Controllable Preference Modeling.** Although controllable preference modeling can steer social-attribute distributions, technical “debiasing” is not a silver bullet. Unilateral use carries significant risks. The technique should support diverse stakeholder input, not replace it.

Defining “equitable representation” is a context-dependent normative decision: a judgment about what representation should be. It requires active community consultation. Practitioners should engage affected communities to establish appropriate goals, rather than rely only on automated optimization to resolve complex social biases.

## 8 Conclusion

Our work identifies and investigates blind spots in evaluating social bias in T2V generation. VideoBiasEval separates identity attributes from content semantics and tracks how alignment changes social representation.

Our analyses show that reward-based alignment inherits and frequently amplifies biases in human preference data (Sections 5–6, Tables 4–5). These findings emphasize the need to audit and mitigate bias at every stage of video generation. Doing so can support more equitable and socially aware generative systems.

## Limitations

Several limitations require further consideration.

First, we study only gender and ethnicity, using categories drawn from U.S. Census conventions and prior literature. These categories make controlled evaluation practical, but they are socially constructed. They cannot fully capture identity’s fluidity, intersectionality, or cultural nuances (Yin et al., 2024; Qiu et al., 2024a; Qiu et al., 2025). Future work should consider richer identity representations, including intersectional groups.

Second, our VLM evaluators classify individual images. Despite validation against human judgments, they may have their own biases or inaccuracies, especially for stylized or ambiguous frames. Combining multiple models helps mitigate this concern. However, reference human annotations for a larger and more diverse video collection would strengthen measurement reliability.

Third, we mainly study one training strategy: single-step latent consistency distillation. We also use a limited set of reward models. Other approaches, including reinforcement-learning-based tuning or multi-turn video instruction alignment—alignment through multiple exchanges of video-related instructions—may produce bias dynamics that our study does not capture.

Fourth, our controllable preference experiments demonstrate targeted bias modulation only through synthetic gender-preference manipulations. They do not resolve broader questions about value alignment, normative appropriateness, or long-term societal effects. Those questions remain essential for responsible deployment.

Finally, VideoBiasEval is benchmarked on a fixed set of 42 socially associated actions. This supports detailed control but may limit generalization to open-ended generation or unfamiliar actions outside the classification. We hope these limitations motivate more holistic, culturally grounded, and ethically aligned evaluation pipelines.

## Broader Impact Statement

As video diffusion models move from research prototypes to widely used content-creation tools, they increasingly influence visual culture and social representation. Our work highlights an often-overlooked consequence: alignment can stabilize and entrench social bias.

**Stabilization of Bias.** Alignment, including RLHF, is essential for improving visual fidelity and temporal consistency. Yet our results show that it can also act as a “bias stabilizer” (Section 6, Tables 5–6). By removing artifacts and smoothing temporal flicker, alignment tuning makes stereotypes look more natural, coherent, and authoritative. These polished biases are harder to recognize as failures than obvious errors in unaligned outputs. They may subtly normalize skewed demographic portrayals in synthetic media. Without intervention, stabilization could reinforce harmful stereotypes under the guise of high-quality, preferred content.

**Dual-Use of Controllable Preference Modeling.** Section 7 demonstrates the dual-use nature of controlled preferences: the same technique can serve beneficial or harmful purposes. Curated preference datasets can make rewards systematically amplify or suppress social attributes (Tables 7–8). This can correct historical underrepresentation and mitigate bias. It also makes alignment a double-edged sword. The same tools could intentionally exclude groups, impose homogeneous representation, or generate propaganda consistent with exclusionary worldviews.

**Mitigation and Responsibility.** These risks require changes in generative-video evaluation. Aesthetic metrics alone are insufficient. Developers must use event-centered auditing at multiple levels, such as VideoBiasEval, to continually monitor outputs’ social composition. Preference-dataset curation deserves the same rigor as architecture design. The human preferences guiding these systems should reflect a diverse, equitable society rather than reinforce existing prejudices.

## Acknowledgment

We thank the anonymous reviewers for their thoughtful feedback and suggestions. This research was supported in part by the Air Force Office of Scientific Research (AFOSR) Multidisciplinary University Research Initiative (MURI) grant #FA9550-22-1-0380, the Office of Naval Research (ONR) under Award #N00014-23-1-2780, and the National Science Foundation under Award #IIS-2449768. Additional support came from cloud credits through the NVIDIA Academic Grant Program and funding from the Technical Artificial Intelligence (AI) Safety Research Program at Coefficient Giving. The views and conclusions are those of the authors. They should not be interpreted as official policies or endorsements of AFOSR, ONR, the U.S. Government, the National Science Foundation, NVIDIA, or Coefficient Giving.
